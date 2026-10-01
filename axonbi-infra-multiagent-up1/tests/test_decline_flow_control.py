"""
A refusal stops what was refused.

THE PRODUCTION FAILURE (tanasuq, 2026-09-30 / 2026-10-01, twice):

    ASSISTANT: الدكتور عبدالله الهلال موجود لكن ما عنده مواعيد متاحة حالياً،
               تحب أحاول أساعدك تحجز مع دكتور ثاني في نفس التخصص؟
    PATIENT:   لا
    ASSISTANT: من فضلك أرسل رقم الجوال مع رمز الدولة عشان أقدر أساعدك في الحجز.
    PATIENT:   055401389
    ASSISTANT: أرسلنا لك رمز التحقق ...        <- an OTP, with no slot at all

WHERE IT GOT LOST. The reading said `declines`, and nothing acted on it:
the router saw intent "booking" == flow "booking" and continued the flow;
the booking specialist got a prompt paragraph whose FIRST bullet ordered
"ask for the alternative phone number"; no tool was gated; and the phone
step had no precondition in code, so the next turn's number went straight
to `send_otp`.

WHAT THESE TESTS PIN DOWN
  - `turn_action` is derived in code from the EXISTING reading fields -
    "decline" (no, nothing new asked) vs "decline_and_request" (no, and
    here is what I want instead) vs confirm / switch / answer;
  - routing: a refusal keeps the owner, never starts the declined flow,
    is never "cancel" by itself, and is never sent to clarification or
    out-of-scope because of a confidence number;
  - the decline gate: on a "decline" turn nothing that would advance the
    refused thing runs (selection, identity, review, booking, new search);
  - the booking-readiness invariant: no phone/OTP step for a new booking
    before a doctor and a slot are locked;
  - the output guard: a "no" never turns into "send your phone number";
  - none of it costs a model call.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import graph
import tools
import understanding
from agents.semantic_router import (
    ACTION_ANSWER, ACTION_CONFIRM, ACTION_CONTINUE, ACTION_DECLINE,
    ACTION_DECLINE_AND_REQUEST, ACTION_SWITCH, TurnFacts, turn_action,
)
from conftest import send, state_of, tool_call

H, A = HumanMessage, AIMessage

OFFER_OTHER_DOCTOR = ("الدكتور عبدالله الهلال موجود لكن ما عنده مواعيد متاحة حالياً، "
                      "تحب أحاول أساعدك تحجز مع دكتور ثاني في نفس التخصص؟")
THURSDAY_OFFER = "ممكن أحجز لك يوم الخميس، هل يناسبك؟"
THURSDAY_Q = "الخميس مناسب؟"
NEW_BOOKING_OFFER = "تحب أحجز لك موعد جديد؟"
SWAP_DOCTOR_OFFER = "أقدر أحجز لك مع دكتور محمد بدل عبدالله، تحب؟"
BOOK_THURSDAY_Q = "تحب أحجز لك يوم الخميس؟"
HELP_BOOK_Q = "هل تحب أساعدك في الحجز؟"
CANCEL_Q = "تحب تلغي الموعد؟"
SATURDAY_Q = "تحب الموعد يوم السبت؟"
DOCTOR_LIST = "الدكاترة المتاحين:\n1️⃣ عبدالله\n2️⃣ محمد\n3️⃣ أحمد"
CONFIRM_CANCEL_Q = "هل تريد إلغاء الموعد؟"
BOOK_Q = "تحب أحجز لك؟"
SAME_NUMBER_Q = "تم اختيار الموعد ✅ نكمل الحجز على نفس رقم الواتساب ده؟ ✅"

PHONE_ASK = "من فضلك أرسل رقم الجوال مع رمز الدولة عشان أقدر أساعدك في الحجز."


def R(intent, **fields):
    reading = {"intent": intent, "confidence": 0.9, "is_ambiguous": False,
               "answer_to_previous_question": False, "changes_intent": False}
    reading.update(fields)
    return understanding._parse(json.dumps(reading, ensure_ascii=False))


# ----------------------------------------------------------------------
# 1. turn_action - derived from the existing fields, no new LLM field
# ----------------------------------------------------------------------

BOOKING = TurnFacts(previous="booking")


@pytest.mark.parametrize("reading,facts,expected", [
    # The production reading: "لا" read as a booking answer that declines.
    (R("booking", answer_to_previous_question=True, declines=True), BOOKING, ACTION_DECLINE),
    # What the prompt now asks for: a bare no is intent "answer".
    (R("answer", answer_to_previous_question=True, declines=True), BOOKING, ACTION_DECLINE),
    # The reading missed `declines` on a message that is nothing but "لا".
    (R("booking", answer_to_previous_question=True), TurnFacts(previous="booking", bare_refusal=True),
     ACTION_DECLINE),
    # "لا" answering "any objection?" IS a yes - the reading decides.
    (R("booking", answer_to_previous_question=True, confirms=True),
     TurnFacts(previous="booking", bare_refusal=True), ACTION_CONFIRM),
    # A refusal that says what instead.
    (R("booking", answer_to_previous_question=True, declines=True, entities={"date": "السبت"}),
     BOOKING, ACTION_DECLINE_AND_REQUEST),
    (R("booking", answer_to_previous_question=True, declines=True, doctor_name="عبدالله"),
     BOOKING, ACTION_DECLINE_AND_REQUEST),
    (R("booking", answer_to_previous_question=True, declines=True, wants_options=True),
     BOOKING, ACTION_DECLINE_AND_REQUEST),
    (R("reschedule", answer_to_previous_question=True, declines=True), TurnFacts(previous="cancel"),
     ACTION_DECLINE_AND_REQUEST),
    # A deliberate move to another request beats the refusal.
    (R("complaint", changes_intent=True, declines=True), BOOKING, ACTION_SWITCH),
    (R("booking", answer_to_previous_question=True, confirms=True), BOOKING, ACTION_CONFIRM),
    (R("booking", answer_to_previous_question=True), BOOKING, ACTION_ANSWER),
    (R("faq", asks_price=True), BOOKING, ACTION_CONTINUE),
    # A reading that says yes AND no is not a yes.
    (R("booking", answer_to_previous_question=True, confirms=True, declines=True), BOOKING, ACTION_ANSWER),
])
def test_turn_action_is_derived_from_the_reading(reading, facts, expected):
    assert turn_action(reading, facts) == expected


def test_turn_action_without_a_reading_uses_only_the_data_fact():
    assert turn_action(None, TurnFacts(previous="booking", bare_refusal=True)) == ACTION_DECLINE
    assert turn_action(None, TurnFacts(previous="booking")) is None


# ----------------------------------------------------------------------
# 2. Routing - scenarios A-M, through the real router node
# ----------------------------------------------------------------------

def _route(reader, question, message, reading, flow, session_id="decline-route"):
    reader.table[message] = reading
    messages = [H(content="مرحبا"), A(content=question), H(content=message)]
    return graph.router({"messages": messages, "session_id": session_id, "active_agent": flow})


def _selected(update):
    if update["handoff_now"]:
        return "handoff"
    if update["clarify_now"]:
        return "clarify"
    if update["out_of_scope_now"]:
        return "out_of_scope"
    return update["active_agent"]


ROUTING = [
    # id, question, message, reading, flow, expected owner, expected action
    ("A-decline-other-doctor", OFFER_OTHER_DOCTOR, "لا",
     R("booking", answer_to_previous_question=True, declines=True), "booking", "booking", ACTION_DECLINE),
    ("B-decline-thursday", THURSDAY_OFFER, "لا",
     R("booking", answer_to_previous_question=True, declines=True), "booking", "booking", ACTION_DECLINE),
    ("C-decline-then-saturday", THURSDAY_Q, "لا، السبت",
     R("booking", answer_to_previous_question=True, declines=True, entities={"date": "السبت"}),
     "booking", "booking", ACTION_DECLINE_AND_REQUEST),
    ("D-decline-then-cancel", NEW_BOOKING_OFFER, "لا، خلاص الغيه",
     R("cancel", answer_to_previous_question=True, declines=True, cancel_request=True),
     "booking", "cancel", ACTION_DECLINE_AND_REQUEST),
    ("E-keep-original-doctor", SWAP_DOCTOR_OFFER, "لا، عايز عبدالله",
     R("booking", answer_to_previous_question=True, declines=True, doctor_name="عبدالله"),
     "booking", "booking", ACTION_DECLINE_AND_REQUEST),
    ("F-confirmation", BOOK_THURSDAY_Q, "أيوه",
     R("booking", answer_to_previous_question=True, confirms=True), "booking", "booking", ACTION_CONFIRM),
    ("G-bare-negation-concierge", HELP_BOOK_Q, "لا",
     R("booking", answer_to_previous_question=True, declines=True), "concierge", "concierge", ACTION_DECLINE),
    ("G-bare-negation-no-owner", HELP_BOOK_Q, "لا",
     R("booking", answer_to_previous_question=True, declines=True), None, "concierge", ACTION_DECLINE),
    ("H-reschedule-not-cancel", CANCEL_Q, "لا، خليه الأسبوع الجاي",
     R("reschedule", answer_to_previous_question=True, declines=True, entities={"date": "الأسبوع الجاي"}),
     "cancel", "reschedule", ACTION_DECLINE_AND_REQUEST),
    ("I-answer-inside-reschedule", SATURDAY_Q, "لا، الأحد",
     R("booking", answer_to_previous_question=True, declines=True, entities={"date": "الأحد"}),
     "reschedule", "reschedule", ACTION_DECLINE_AND_REQUEST),
    ("J-list-selection", DOCTOR_LIST, "2",
     R("booking", answer_to_previous_question=True), "booking", "booking", ACTION_ANSWER),
    ("K-short-yes-in-context", CONFIRM_CANCEL_Q, "تمام",
     R("cancel", answer_to_previous_question=True, confirms=True, cancel_confirmed=True),
     "cancel", "cancel", ACTION_CONFIRM),
    ("L-changes-intent", BOOK_Q, "لا، عندي شكوى على الاستقبال",
     R("complaint", changes_intent=True, declines=True), "booking", "complaint", ACTION_SWITCH),
    ("M-price-during-booking", BOOK_Q, "بكام الكشف؟",
     R("faq", asks_price=True), "booking", "booking", ACTION_CONTINUE),
]


@pytest.mark.parametrize("question,message,reading,flow,owner,action",
                         [r[1:] for r in ROUTING], ids=[r[0] for r in ROUTING])
def test_scenarios_route_on_meaning_and_action(reader, question, message, reading, flow, owner, action):
    update = _route(reader, question, message, reading, flow)
    assert _selected(update) == owner, update["routing_reason"]
    assert update["turn_action"] == action


def test_a_bare_no_is_never_a_cancel_whatever_was_declined(reader):
    """"لا" is not automatically cancel - not even inside the cancel flow,
    where it must stay with cancel and cancel nothing."""
    update = _route(reader, CANCEL_Q, "لا", R("cancel", answer_to_previous_question=True, declines=True), "cancel")
    assert update["active_agent"] == "cancel" and update["turn_action"] == ACTION_DECLINE
    update = _route(reader, OFFER_OTHER_DOCTOR, "لا",
                    R("cancel", answer_to_previous_question=True, declines=True), "booking")
    assert update["active_agent"] == "booking", "a no to another doctor became a cancellation"


def test_a_clear_refusal_is_not_clarified_for_low_confidence(reader):
    update = _route(reader, HELP_BOOK_Q, "لا",
                    R("answer", answer_to_previous_question=True, declines=True, confidence=0.3), None)
    assert _selected(update) == "concierge"


def test_a_no_read_as_other_is_not_out_of_scope(reader):
    update = _route(reader, HELP_BOOK_Q, "لا خلاص",
                    R("other", answer_to_previous_question=True, declines=True), None)
    assert _selected(update) == "concierge"


def test_isolated_tamam_is_not_a_confirmation(reader):
    reader.table["تمام"] = R("greeting")
    update = graph.router({"messages": [H(content="تمام")], "session_id": "decline-tamam", "active_agent": None})
    assert update["turn_action"] != ACTION_CONFIRM


def test_the_routing_log_carries_the_turn_action(reader, caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="graph"):
        _route(reader, OFFER_OTHER_DOCTOR, "لا", R("booking", answer_to_previous_question=True, declines=True),
               "booking")
    decisions = [json.loads(r.getMessage()[len("routing_decision "):]) for r in caplog.records
                 if r.getMessage().startswith("routing_decision ")]
    assert decisions[-1]["turn_action"] == ACTION_DECLINE


def test_semantic_decision_is_not_overridden_by_the_fallback_router(reader, monkeypatch):
    """The keyword router runs only when there is no reading at all."""
    called = []
    monkeypatch.setattr(graph.agents, "route_turn", lambda *a, **k: called.append(1) or ("cancel", "x"))
    update = _route(reader, OFFER_OTHER_DOCTOR, "لا",
                    R("booking", answer_to_previous_question=True, declines=True), "booking")
    assert not called and update["active_agent"] == "booking"


# ----------------------------------------------------------------------
# 3. The decline gate and the booking-readiness invariant
# ----------------------------------------------------------------------

def _gate_state(calls, turn_action_value=None, agent="booking", session=None, sid="decline-gate"):
    tools._BOOKING_SESSIONS[sid] = dict(session or {})
    message = AIMessage(content="", tool_calls=[
        {"name": name, "args": {}, "id": f"c{i}", "type": "tool_call"} for i, name in enumerate(calls)])
    return {"messages": [H(content="لا"), message], "session_id": sid,
            "active_agent": agent, "turn_action": turn_action_value}


def _gated_names(state):
    try:
        return sorted(tc["name"] for tc, _ in graph._gated_tool_calls(state))
    finally:
        tools._BOOKING_SESSIONS.pop(state["session_id"], None)


def test_a_bare_no_blocks_every_tool_that_would_advance_the_refused_thing():
    advancing = ["compare_phone", "send_otp", "verify_otp", "get_patient_info", "create_new_booking",
                 "confirm_booking_review", "select_appointment_slot", "find_available_doctors",
                 "match_entity_for_booking", "find_best_doctor_in_specialty", "reset_booking_session"]
    ready = {"doctor_id": "D1", "selected_slot": {"slot_start": "x"}}
    assert _gated_names(_gate_state(advancing, ACTION_DECLINE, session=ready)) == sorted(advancing)


def test_a_refused_day_can_still_be_answered_with_other_days():
    looking = ["list_available_days_for_booking", "get_available_slots_for_booking", "resolve_available_day"]
    assert _gated_names(_gate_state(looking, ACTION_DECLINE, session={"doctor_id": "D1"})) == []


def test_a_no_that_names_what_instead_is_not_gated():
    calls = ["resolve_available_day", "match_entity_for_booking", "find_available_doctors"]
    assert _gated_names(_gate_state(calls, ACTION_DECLINE_AND_REQUEST, session={"doctor_id": "D1"})) == []


def test_no_phone_step_for_a_new_booking_before_a_slot_is_locked():
    identity = ["compare_phone", "send_otp", "get_patient_info"]
    # The screenshot: a doctor, no schedule, no slot - and an OTP went out.
    assert _gated_names(_gate_state(identity, None, session={"doctor_id": "D1"})) == sorted(identity)
    assert _gated_names(_gate_state(identity, None, session={})) == sorted(identity)
    # Locked -> NB6 is exactly where we are.
    assert _gated_names(_gate_state(identity, None, session={"doctor_id": "D1", "selected_slot": {"a": 1}})) == []
    # Picked and confirmed the number in one message.
    assert _gated_names(_gate_state(["select_appointment_slot", "get_patient_info"], None,
                                    session={"doctor_id": "D1"})) == []


def test_existing_booking_flows_keep_their_identity_step():
    identity = ["compare_phone", "send_otp", "verify_otp"]
    assert _gated_names(_gate_state(identity, None, agent="reschedule", session={})) == []
    assert _gated_names(_gate_state(identity, None, agent="cancel", session={})) == []


# ----------------------------------------------------------------------
# 4. End to end - the production conversation through main
# ----------------------------------------------------------------------

def _offer_turn(session_id, llm, reader):
    opening = "عايز احجز مع الدكتور عبدالله الهلال"
    reader.table[opening] = {"intent": "booking", "doctor_name": "عبدالله الهلال", "confidence": 0.95}
    llm._responses.append(AIMessage(content=OFFER_OTHER_DOCTOR))
    # In production this offer followed a real schedule lookup; here the
    # availability verifier has no tool result to ground it on, so it is
    # relaxed for THIS turn only - the turn under test runs strict.
    with patch.object(graph, "_VERIFIERS_SAFETY_STRICT", False):
        send(session_id, opening)
    last_ai = [m for m in state_of(session_id)["messages"] if isinstance(m, AIMessage) and m.content][-1]
    assert OFFER_OTHER_DOCTOR in last_ai.content
    # What `match_entity_for_booking` leaves behind: a doctor, no branch
    # (no schedule anywhere), no slot.
    tools._BOOKING_SESSIONS.setdefault(session_id, {}).update(
        {"doctor_id": "D-ABD", "doctor_display_name": "عبدالله الهلال"})
    assert state_of(session_id)["active_agent"] == "booking"


PRODUCTION_READINGS = {
    "as-seen": {"intent": "booking", "answer_to_previous_question": True, "declines": True, "confidence": 0.8},
    "prompt-compliant": {"intent": "answer", "answer_to_previous_question": True, "declines": True},
    "reading-missed-the-no": {"intent": "booking", "answer_to_previous_question": True, "confidence": 0.6},
}


@pytest.mark.parametrize("reading", PRODUCTION_READINGS.values(), ids=PRODUCTION_READINGS.keys())
def test_no_to_another_doctor_never_asks_for_the_phone(session_id, llm, reader, reading):
    _offer_turn(session_id, llm, reader)
    calls_before = len(llm.calls)

    reader.table["لا"] = reading
    llm._responses.append(AIMessage(content=PHONE_ASK))  # what production sent
    result = send(session_id, "لا")

    reply = result["reply"]
    assert "رقم الجوال" not in reply and "واتساب" not in reply, reply
    assert reply == graph._DECLINED_OFFER_REPLY["ar"]
    state = state_of(session_id)
    assert state["active_agent"] == "booking" and state["turn_action"] == ACTION_DECLINE
    assert result["escalate"] is False
    # One specialist call, no correction call on top of it.
    assert len(llm.calls) == calls_before + 1
    prompt = llm.system_prompts()[-1]
    assert "THEY SAID NO - STOP WHAT YOU OFFERED" in prompt
    assert "alternative phone number" not in prompt
    assert "offer the other available doctors" not in prompt


def test_no_to_another_doctor_runs_no_search_no_otp_no_booking(session_id, llm, reader):
    _offer_turn(session_id, llm, reader)
    reader.table["لا"] = PRODUCTION_READINGS["as-seen"]
    llm._responses.extend([
        AIMessage(content="", tool_calls=[
            {"name": "find_available_doctors", "args": {"specialty_ids": ["S1"]}, "id": "t1", "type": "tool_call"},
            {"name": "compare_phone", "args": {"provided_phone": "0538122705"}, "id": "t2", "type": "tool_call"},
            {"name": "send_otp", "args": {"phone": "+966538122705"}, "id": "t3", "type": "tool_call"},
            {"name": "create_new_booking", "args": {}, "id": "t4", "type": "tool_call"},
        ]),
        AIMessage(content="تمام 🌷 تحب أساعدك بشي ثاني؟"),
    ])
    doctors, otp, booking = MagicMock(), MagicMock(), MagicMock()
    with patch("api.get_doctors", doctors), patch("api.authentica_send_otp", otp), \
            patch("api.create_booking", booking):
        result = send(session_id, "لا")

    assert not doctors.called and not otp.called and not booking.called
    results = {m.name: json.loads(m.content)["status"] for m in state_of(session_id)["messages"]
               if isinstance(m, ToolMessage) and m.tool_call_id in ("t1", "t2", "t3", "t4")}
    assert results == {name: "blocked_patient_declined" for name in
                       ("find_available_doctors", "compare_phone", "send_otp", "create_new_booking")}
    assert "رقم" not in result["reply"]


def test_the_number_typed_after_a_no_does_not_get_an_otp(session_id, llm, reader):
    """Screenshot 5's second half: even if a number arrives, no slot means
    no phone step."""
    _offer_turn(session_id, llm, reader)
    reader.table["0538122705"] = {"intent": "booking", "answer_to_previous_question": True,
                                  "entities": {"phone": "0538122705"}}
    llm._responses.extend([
        tool_call("send_otp", {"phone": "+966538122705"}, "otp1"),
        AIMessage(content="قبل الرقم، خليني أساعدك نختار الموعد أول 🌷"),
    ])
    otp = MagicMock()
    with patch("api.authentica_send_otp", otp):
        send(session_id, "0538122705")
    assert not otp.called
    blocked = [m for m in state_of(session_id)["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == "otp1"]
    assert json.loads(blocked[0].content)["status"] == "not_at_phone_step"


def test_no_to_the_same_number_question_still_asks_for_the_other_number(session_id, llm, reader):
    """The one "no" that legitimately leads to a phone question: NB6, slot
    already locked. The guard must not touch it."""
    _offer_turn(session_id, llm, reader)
    tools._BOOKING_SESSIONS[session_id]["selected_slot"] = {"slot_start": "2026-10-05T09:00:00+00:00"}
    reader.table["2"] = {"intent": "booking", "answer_to_previous_question": True}
    llm._responses.append(AIMessage(content=SAME_NUMBER_Q))
    send(session_id, "2")

    reader.table["لا"] = {"intent": "answer", "answer_to_previous_question": True, "declines": True}
    llm._responses.append(AIMessage(content="من فضلك أرسل رقم الجوال مع رمز الدولة."))
    assert send(session_id, "لا")["reply"] == "من فضلك أرسل رقم الجوال مع رمز الدولة."


# ----------------------------------------------------------------------
# 5. The pieces, one at a time
# ----------------------------------------------------------------------

def test_the_output_guard_is_scoped():
    sid = "decline-guard"
    state = {"session_id": sid, "turn_action": ACTION_DECLINE}
    tools._BOOKING_SESSIONS[sid] = {"doctor_id": "D1"}
    try:
        assert graph._reply_advances_past_a_refusal(PHONE_ASK, state, "booking")
        assert graph._reply_advances_past_a_refusal("نكمل الحجز على نفس رقم الواتساب ده؟", state, "booking")
        assert not graph._reply_advances_past_a_refusal(PHONE_ASK, state, "reschedule")
        assert not graph._reply_advances_past_a_refusal(PHONE_ASK, {**state, "turn_action": ACTION_ANSWER}, "booking")
        assert not graph._reply_advances_past_a_refusal(
            "تقدر تتواصل مع الاستقبال على رقم الهاتف 920012345", state, "booking")
        assert not graph._reply_advances_past_a_refusal("تمام 🌷 أي يوم يناسبك بدال الخميس؟", state, "booking")
        tools._BOOKING_SESSIONS[sid]["selected_slot"] = {"a": 1}
        assert not graph._reply_advances_past_a_refusal(PHONE_ASK, state, "booking")
    finally:
        tools._BOOKING_SESSIONS.pop(sid, None)


def test_the_clinic_can_author_the_acknowledgement():
    assert graph._declined_offer_reply({"msg_declined_offer": "ولا يهمك 🌷"}, "ar") == "ولا يهمك 🌷"
    assert graph._declined_offer_reply({}, "en") == graph._DECLINED_OFFER_REPLY["en"]


def test_the_acknowledgement_is_not_egyptian():
    for marker in ("شكلي", "مش ", " ده ", "حابب أحولك", "توضحلي"):
        assert marker not in graph._DECLINED_OFFER_REPLY["ar"]


def test_the_refusal_directive_has_two_shapes_and_got_smaller():
    msgs = [A(content=OFFER_OTHER_DOCTOR), H(content="لا")]
    reading = R("booking", answer_to_previous_question=True, declines=True)
    bare = graph._build_negation_directive(msgs, reading, ACTION_DECLINE)
    assert "STOP WHAT YOU OFFERED" in bare and "blocked" in bare
    assert "alternative phone number" not in bare and "offer the other available doctors" not in bare

    msgs = [A(content=SWAP_DOCTOR_OFFER), H(content="لا، عايز عبدالله")]
    reading = R("booking", answer_to_previous_question=True, declines=True, doctor_name="عبدالله")
    detailed = graph._build_negation_directive(msgs, reading, ACTION_DECLINE_AND_REQUEST)
    assert "TURNED DOWN" in detailed and "act on THAT now" in detailed
    # The text it replaced was ~3,500 characters on every refusal turn.
    assert len(bare) < 1700 and len(detailed) < 900

    # "لا، السبت" names the replacement day: no "show the other days" text
    # may stand between the patient and Saturday (scenario C).
    msgs = [A(content=THURSDAY_Q), H(content="لا، السبت")]
    reading = R("booking", answer_to_previous_question=True, declines=True, entities={"date": "السبت"})
    saturday = graph._build_negation_directive(msgs, reading, ACTION_DECLINE_AND_REQUEST)
    assert "STOP WHAT YOU OFFERED" not in saturday and "OTHER days" not in saturday


def test_a_refusal_is_never_a_yes_to_the_hooks():
    assert graph._patient_confirms("تمام", {"declines": True, "confirms": False}) is False
    assert graph._patient_confirms("تمام", {"confirms": True}) is True
    assert graph._patient_declines_this_turn({"turn_action": ACTION_DECLINE, "understanding": {"declines": False},
                                              "messages": [H(content="لا")]})


def test_the_understanding_prompt_states_the_refusal_rule_compactly():
    prompt = understanding.PROMPT
    assert "A no moves nothing forward" in prompt
    assert "Never true together with declines" in prompt
    # Still a compact prompt - the refusal rule is a few lines, not a list.
    assert len(prompt) < 9000


# ----------------------------------------------------------------------
# 6. One branch is not a choice (screenshot: فرع المدار only)
# ----------------------------------------------------------------------

def test_a_doctor_at_one_branch_is_not_asked_which_branch(session_id, llm, reader, monkeypatch):
    payload = {"status": "found", "schedules": [
        {"doctorName": "أحمد يوسف", "branchName": "المدار", "branchId": "B1",
         "recurringDaysNames": ["Saturday", "Monday"],
         "fromDateTime": "2026-10-03T07:00:00+00:00", "toDateTime": "2026-10-03T15:00:00+00:00"}]}

    def fake_lookup(state, agent_name):
        tools._BOOKING_SESSIONS[state["session_id"]]["branch_id"] = "B1"
        return list(graph._forge_tool_pair("get_doctor_schedule_for_booking", {}, payload))

    monkeypatch.setattr(graph, "_deterministic_doctor_schedule_lookup", fake_lookup)
    monkeypatch.setattr(graph, "_deterministic_single_doctor_confirmation", lambda s, a: [])
    tools._BOOKING_SESSIONS[session_id] = {"doctor_id": "D-AHMED", "doctor_display_name": "أحمد يوسف"}

    reader.table["احمد"] = {"intent": "booking", "doctor_name": "احمد", "answer_to_previous_question": True}
    llm._responses.append(AIMessage(content="مواعيد الدكتور أحمد يوسف في فرع المدار:\n• السبت\n\nأي يوم يناسبك؟"))
    send(session_id, "احمد")

    prompt = llm.system_prompts()[-1]
    assert "THIS DOCTOR WORKS AT ONE BRANCH" in prompt
    assert "حابب تحجز في أنهي فرع وأنهي يوم؟" not in prompt
