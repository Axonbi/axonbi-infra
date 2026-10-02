"""
Regressions from the tanasuq-production log of 2026-10-01 12:38-17:06.

  - "ابي اعرف موعدي" -> "هذا هو موعدك الذي تبغى تلغيه؟" -> "صح" -> the
    appointment was CANCELLED. The patient only asked to see it.
  - "نعم" was passed to compare_phone as a phone number.
  - The patient's own name "حاتم العنزي" was matched as the specialty
    "اخصائية اجتماعية" (fuzzy score 0.63).
  - An invented specialty id was searched for the surname "المديفر".
  - A WhatsApp reaction got the English welcome and a clarify question.
  - "same WhatsApp number?" was asked before a time was picked.
  - Specialists copied the out-of-scope refusal for in-scope messages.
"""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import graph
import main
import tools
import understanding
from conftest import send


# ----------------------------------------------------------------------
# No cancellation the patient did not ask for
# ----------------------------------------------------------------------

def _cancel_call_state(previous_agent):
    return {
        "previous_agent": previous_agent, "active_agent": "cancel", "session_id": "log-1001e-cancel",
        "messages": [HumanMessage(content="صح"),
                     AIMessage(content="", tool_calls=[{"name": "cancel_appointment", "args": {}, "id": "c1"}])],
    }


def test_a_yes_to_the_assistants_own_cancel_suggestion_cancels_nothing():
    gated = graph._gated_tool_calls(_cancel_call_state("concierge"))
    assert [payload["status"] for _, payload in gated] == ["cancellation_not_requested"]


@pytest.mark.parametrize("previous", ["cancel", "reschedule"])
def test_a_cancellation_the_patient_asked_for_goes_ahead(previous):
    assert graph._gated_tool_calls(_cancel_call_state(previous)) == []


def test_the_concierge_is_told_not_to_offer_cancelling_an_appointment_it_shows():
    import agents.registry as registry
    job = registry.AGENT_SPECS[registry.CONCIERGE].job
    assert "ابي اعرف موعدي" in job and "Never ask whether they want to cancel" in job


# ----------------------------------------------------------------------
# Phone, names, ids
# ----------------------------------------------------------------------

def test_words_are_not_a_phone_number():
    state = {"channel_phone": "966535230420", "messages": [], "session_id": "log-1001e-phone"}
    assert tools.compare_phone.func(state=state, provided_phone="نعم") == {"status": "no_number_given"}


_SPECIALTIES = {"success": True, "data": {"items": [
    {"id": "a", "name": "اخصائية اجتماعية"}, {"id": "b", "name": "طب نفسي"}, {"id": "c", "name": "علاج نفسي"},
]}}


@pytest.mark.parametrize("text,expected", [
    ("حاتم العنزي", None),
    ("اجتماعية", "a"),
    ("الطب النفسي", "b"),
])
def test_only_a_close_match_names_a_specialty(text, expected):
    state = {"session_id": "log-1001e-spec", "messages": []}
    with patch.object(tools.api, "get_specialties", return_value=_SPECIALTIES):
        found = tools._specialty_named_by(state, "https://example.test", text)
    assert (found or {}).get("id") == expected


def test_an_invented_specialty_id_is_dropped():
    invented = "7f3a3a3a-7f3a-4a3a-8a3a-7f3a3a3a3a3a"
    state = {"session_id": "log-1001e-id", "messages": [HumanMessage(content="المديفر")]}
    with patch.object(tools.api, "get_specialties", return_value=_SPECIALTIES):
        clean, unresolved = tools._sanitize_specialty_ids(state, "https://example.test", [invented])
    assert clean == [] and unresolved == [invented]


def test_a_real_id_from_a_tool_result_is_kept():
    real = "82dd6ccb-5e16-49c1-b96e-2f23c4526d9f"
    state = {"session_id": "log-1001e-id2", "messages": [
        HumanMessage(content="علاج نفسي"),
        ToolMessage(content=f'{{"items": [{{"id": "{real}"}}]}}', name="list_specialties", tool_call_id="t1"),
    ]}
    clean, unresolved = tools._sanitize_specialty_ids(state, "https://example.test", [real])
    assert clean == [real] and unresolved == []


# ----------------------------------------------------------------------
# Reaction, premature number question, scope text
# ----------------------------------------------------------------------

def test_a_reaction_gets_a_short_acknowledgement_and_no_model(session_id, llm, reader):
    result = send(session_id, "[reaction]")
    assert result["reply"] == "🌷" and len(llm.calls) == 0


def test_the_same_number_question_before_a_time_is_replaced_by_the_times():
    sid = "log-1001e-nb6"
    session = tools._get_booking_session(sid)
    session.update({"doctor_id": "d1", "branch_id": "b1"})
    session["last_list"] = {"entity_type": "slot", "items": [
        {"slotStart": "2026-10-05T16:10:00", "time_display": "4:10 مساءً"},
        {"slotStart": "2026-10-05T17:10:00", "time_display": "5:10 مساءً"},
    ]}
    draft = ("أقرب موعد متاح عند عمر المديفر في المنار:\n🗓️ الاثنين 05/10/2026 — من 4:10 مساءً إلى 5:10 مساءً\n"
             "نكمل الحجز على نفس رقم الواتساب ده؟ ✅")
    try:
        fixed = graph._premature_same_number_fix(draft, {"session_id": sid})
        assert "واتساب" not in fixed and fixed.endswith("أي رقم أو وقت تفضل؟")
        session["selected_slot"] = {"slotStart": "2026-10-05T16:10:00"}
        assert graph._premature_same_number_fix(draft, {"session_id": sid}) is None
    finally:
        tools._BOOKING_SESSIONS.pop(sid, None)


def test_an_in_scope_message_is_not_handed_the_refusal_text(session_id, llm, reader):
    reader.table["احجز مع د فرح"] = {"intent": "booking", "doctor_name": "فرح"}
    llm._responses.append(AIMessage(content="تحب تحجز في أي يوم؟"))
    with patch.multiple(graph, _VERIFIERS_SAFETY_STRICT=False, _VERIFIERS_FLOW_STRICT=False):
        send(session_id, "احجز مع د فرح")
    prompt = llm.system_prompts()[-1]
    assert "BEGIN-EXACT-TEXT" not in prompt and "THIS MESSAGE IS ABOUT THE HOSPITAL" in prompt


def test_the_understanding_prompt_knows_a_bare_surname_and_seeing_an_appointment():
    assert '"المديفر"' in understanding.PROMPT and "ابي اعرف موعدي" in understanding.PROMPT
