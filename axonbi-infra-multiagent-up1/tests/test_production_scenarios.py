"""
The conversations that failed in production (Tanasuq, 2026-09-22/23),
replayed end to end through main.send_message_with_signals - the same
call n8n makes. Each test is one real failure; it must stay green before
any deploy.

The understanding LLM is faked here (FakeReader) so these tests check
the WIRING: when the patient means X, does the system actually do X, or
does a guard/verifier/regex get in the way? Whether the real model reads
new phrasings correctly is checked separately by evals/.
"""

from unittest.mock import patch

import pytest

import graph
import tools
from conftest import EXHAUSTED, send, state_of, tool_call
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


OFFER_AR = "هل تود أن أرسل طلب تحويلك إلى أحد ممثلي خدمة العملاء الآن؟"
OFFER_AR_2 = "هل تود أن أساعدك في التواصل مع أحد ممثلي خدمة العملاء؟"
OFFER_EN = ("I understand, and I'm here to support you 🌷 Would you like me to connect "
            "you to one of our human customer service representatives?")


def _offer_turn(session, llm, reader, opening, offer):
    """Turn 1: the patient opens, the assistant ends by offering a person."""
    llm._responses.append(AIMessage(content=offer))
    reader.table[opening] = {"intent": "faq"}
    return send(session, opening)


# ======================================================================
# 1. "هل تود أن أحولك؟" -> "نعم" -> the same question again, forever
#    (966555030365, 966503161309, 966504922400)
# ======================================================================

@pytest.mark.parametrize("yes", ["نعم", "أكيد", "يفضل", "اي", "لامانع", "الان الأسهل", "تمام"])
def test_yes_to_a_handoff_offer_transfers_the_patient(session_id, llm, reader, yes):
    _offer_turn(session_id, llm, reader, "هل يوجد اخصائي نفسي يستطيع الحضور للمنزل", OFFER_AR)
    calls_before = len(llm.calls)

    reader.table[yes] = {"intent": "answer", "wants_human": True}
    result = send(session_id, yes)

    assert result["escalate"] is True, result
    assert "تم تحويلك" in result["reply"]
    assert OFFER_AR not in result["reply"], "asked the same question again"
    # The transfer happens in code - no model call, so no verifier can
    # turn the "yes" back into a question.
    assert len(llm.calls) == calls_before


def test_the_first_yes_is_enough(session_id, llm, reader):
    """966503161309 answered "اي" four times. The first one must work."""
    _offer_turn(session_id, llm, reader, "كم سعر الموعد عند الدكتور المديفر", OFFER_AR_2)
    reader.table["اي"] = {"intent": "answer", "wants_human": True}
    assert send(session_id, "اي")["escalate"] is True


def test_asking_for_a_person_directly_transfers(session_id, llm, reader):
    reader.table["ابي اكلم موظف"] = {"intent": "human", "wants_human": True}
    result = send(session_id, "ابي اكلم موظف")
    assert result["escalate"] is True
    assert len(llm.calls) == 0


def test_saying_no_to_the_offer_does_not_transfer(session_id, llm, reader):
    _offer_turn(session_id, llm, reader, "استفسار عن المواعيد", OFFER_AR)
    llm._responses.append(AIMessage(content="تمام، كيف أقدر أساعدك؟"))
    reader.table["لا شكرا"] = {"intent": "answer", "wants_human": False}
    result = send(session_id, "لا شكرا")
    assert result["escalate"] is False


# --- the same failure with the understanding call DOWN (fallback path) ---

@pytest.mark.parametrize("yes", ["نعم", "yes", "confirm"])
def test_fallback_model_call_after_offer_is_not_blocked(session_id, llm, reader, yes):
    """ROOT CAUSE of the loop: the consent gate read "the latest AI
    message" while the tool was running - which is the tool-call message
    itself (empty) - so it never saw the offer and refused every yes.
    Here the understanding call is unavailable and the MODEL calls the
    tool correctly; the tool must not refuse."""
    _offer_turn(session_id, llm, reader, "استفسار", OFFER_AR)
    reader.fail = True
    llm._responses.extend([
        tool_call("request_human_handoff", {"reason": "patient accepted offer", "patient_agreed": True}),
        AIMessage(content="تم تحويلك إلى أحد ممثلي خدمة العملاء 🌷"),
    ])
    result = send(session_id, yes)
    assert result["escalate"] is True, result


# ======================================================================
# 2. "I feel like I want to hurt myself and end it" - never transferred
#    (966545930176)
# ======================================================================

CRISIS_EN = "I feel like I want to hurt myself and end it"


def test_crisis_message_escalates_immediately(session_id, llm, reader):
    reader.table[CRISIS_EN] = {"intent": "medical", "crisis": True}
    result = send(session_id, CRISIS_EN)

    assert result["escalate"] is True
    assert "emergency" in result["reply"].lower()
    assert "connecting you" in result["reply"].lower()
    assert state_of(session_id).get("crisis_active") is True
    assert len(llm.calls) == 0, "a crisis must not wait on a model reply"


def test_crisis_escalates_even_when_understanding_is_down(session_id, llm, reader):
    reader.fail = True
    result = send(session_id, CRISIS_EN)
    assert result["escalate"] is True


def test_crisis_rules_survive_the_next_message(session_id, llm, reader):
    """The crisis directive used to check only the LATEST message, so a
    follow-up "ok" / "i need help" dropped it."""
    reader.table[CRISIS_EN] = {"crisis": True}
    send(session_id, CRISIS_EN)
    first_turn_calls = len(llm.calls)

    llm._responses.append(AIMessage(content="I'm here with you. Our team has been notified."))
    reader.table["ok"] = {"intent": "answer"}
    send(session_id, "ok")

    followup = llm.calls[first_turn_calls:]
    prompts = [str(m.content) for call in followup for m in call if getattr(m, "type", None) == "system"]
    assert prompts, "the specialist should have been called for the follow-up"
    assert any("THIS PATIENT HAS SAID THEY WANT TO HARM THEMSELVES" in p for p in prompts), "crisis directive was dropped"


def test_in_crisis_the_handoff_tool_is_never_refused(session_id, llm, reader):
    reader.table[CRISIS_EN] = {"crisis": True}
    send(session_id, CRISIS_EN)

    reader.table["come help me i said that many times"] = {"intent": "other", "wants_human": False}
    llm._responses.extend([
        tool_call("request_human_handoff", {"reason": "crisis", "patient_agreed": False}),
        AIMessage(content="I'm connecting you with our team right now."),
    ])
    result = send(session_id, "come help me i said that many times")
    assert result["escalate"] is True


@pytest.mark.parametrize("repeat", ["yes", "i need help", "then come help me"])
def test_crisis_followups_with_wants_human_transfer(session_id, llm, reader, repeat):
    reader.table[CRISIS_EN] = {"crisis": True}
    send(session_id, CRISIS_EN)
    reader.table[repeat] = {"intent": "answer", "wants_human": True}
    assert send(session_id, repeat)["escalate"] is True


# ======================================================================
# 3. The appointment just booked was CANCELLED on "تم تاكيد الموعد مسبقا"
#    (PDF scenario 1, 966505451795)
# ======================================================================

# Tested at the tool: a scripted multi-turn cancel flow gets rewritten
# by the reply verifiers (itself the guard-interplay problem), so the
# exact PDF conversation is built as state and the irreversible call is
# made the way the model made it.

def _pdf_conversation(last_message, reading=None):
    booking = {"id": "GUID-OMAR-1", "ref": "TNS-2026-09-23-12", "statusName": "New"}
    messages = [
        # Days earlier in the same (reused) session.
        HumanMessage(content="ابي الغاء موعدي"),
        ToolMessage(content=json.dumps({"status": "found_one", "appointment": booking}),
                    name="lookup_appointment", tool_call_id="old-lookup"),
        AIMessage(content="تم، كيف أقدر أساعدك؟"),
        # Today: a new booking with Dr. Omar, successfully created.
        HumanMessage(content="اود اخذ موعد مع الدكتور عمر"),
        ToolMessage(content=json.dumps({"status": "success"}), name="create_new_booking", tool_call_id="new-booking"),
        AIMessage(content="تم الحجز بنجاح ✅ الطبيب: عمر المديفر. إذا رغبت في حجز موعد جديد أو تعديل موعد آخر، يسعدنا خدمتك"),
        HumanMessage(content=last_message),
        tool_call("cancel_appointment", {"booking_id": "GUID-OMAR-1"}),
    ]
    state = {"messages": messages, "session_id": "pdf-1", "client_id": "tanasuq-test",
             "templates": {"_base_url": "http://booking.test"}}
    if reading is not None:
        state["understanding"] = reading
    return state


_CANCELLED = {"success": True, "status_code": 200, "data": {"isSuccess": True}, "error": None}


@pytest.mark.parametrize("reading", [None, {"cancel_confirmed": False}], ids=["fallback", "understanding"])
def test_already_confirmed_is_not_a_cancellation(reading):
    with patch("api.cancel_booking_by_guid", return_value=_CANCELLED) as cancel_api:
        result = tools.cancel_appointment.func(
            state=_pdf_conversation("تم تاكيد الموعد مسبقا", reading), booking_id="GUID-OMAR-1",
        )
    assert cancel_api.call_count == 0, "the appointment just booked was cancelled"
    assert result["status"] != "success"


def test_a_clear_yes_to_cancelling_still_cancels():
    state = _pdf_conversation("نعم الغيه", {"cancel_confirmed": True})
    with patch("api.cancel_booking_by_guid", return_value=_CANCELLED) as cancel_api:
        result = tools.cancel_appointment.func(state=state, booking_id="GUID-OMAR-1")
    assert cancel_api.call_count == 1
    assert result["status"] == "success"


# ======================================================================
# 4. "ما لقيت دكتور اسمه نفسي" (966530389229)
# ======================================================================

def test_doctor_nafsi_is_a_specialty_not_a_name(session_id, llm, reader):
    msg = "كم موعد عند الدكتور نفسي"
    reader.table[msg] = {"intent": "booking", "specialty": "نفسي", "doctor_name": None}
    llm._responses.append(AIMessage(content="تبي أي تخصص نفسي؟"))
    send(session_id, msg)

    prompt = "\n".join(llm.system_prompts())
    assert 'user_input="نفسي", entity_type="doctor"' not in prompt
    assert "A SPECIALTY OR SERVICE" in prompt


# ======================================================================
# 5. Egyptian fallback text in a Saudi hospital ("شكلي مش قادرة ...")
# ======================================================================

EGYPTIAN_MARKERS = ("شكلي", "مش ", " ده ", "ما قدرتش", "ما لقيتش", "حابب أحولك", "توضحلي")


def test_fallback_lines_are_not_egyptian():
    for text in (graph._soft_recovery_reply("ar"), graph._SOFT_RECOVERY_TEXT["ar"]):
        for marker in EGYPTIAN_MARKERS:
            assert marker not in text, (marker, text)


# ======================================================================
# 6. Guard interplay: the handoff path costs zero model calls and runs
#    no verifier, whatever the conversation looked like before.
# ======================================================================

def test_handoff_turn_bypasses_every_verifier(session_id, llm, reader):
    _offer_turn(session_id, llm, reader, "مرحبا", OFFER_AR)
    reader.table["نعم"] = {"wants_human": True}
    calls_before = len(llm.calls)
    result = send(session_id, "نعم")
    assert result["escalate"] is True
    assert len(llm.calls) == calls_before
    assert EXHAUSTED not in result["reply"]
