"""
Follow-ups from the tanasuq-production logs of 2026-10-01 (12:04-12:20).

  - "ايه" (Saudi "yes") to "نكمل الحجز على نفس رقم الواتساب ده؟" was read as
    not-yes, and the patient was asked to type her own phone number.
  - "الاحد الجاي" on Thursday 01/10 resolved to Sunday 18/10 - the two
    nearer Sundays had nothing open - and the reply showed 18/10's times
    as if it were the Sunday she asked for.
  - A question about the hospital in the middle of a booking, then back:
    the booking in progress was wiped on the way back.
  - A greeting or "thanks" after the first turn cost a ~31-33k-token
    concierge call for one line.
"""

from datetime import datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import graph
import tools
import understanding
from conftest import send, state_of

H, A = HumanMessage, AIMessage
SAME_NUMBER_Q = "تم اختيار الموعد ✅\nنكمل الحجز على نفس رقم الواتساب ده؟ ✅"


# ----------------------------------------------------------------------
# "ايه" is yes in the clinic's dialect
# ----------------------------------------------------------------------

def test_the_understanding_model_is_told_the_clinics_dialect():
    state = {"templates": {"_dialect_name": "Saudi-Tanasuq"}, "messages": [H(content="ايه")], "active_agent": "booking"}
    assert graph._understanding_context(state)["dialect"] == "Saudi-Tanasuq"
    assert "ايه" in understanding.PROMPT.split('"confirms":', 1)[1].splitlines()[0]


@pytest.mark.parametrize("reading", [
    {"intent": "answer", "answer_to_previous_question": True},                    # as logged: no confirms
    {"intent": "booking", "answer_to_previous_question": True, "confirms": True},  # what the prompt now asks for
])
def test_yes_to_the_same_number_question_uses_the_whatsapp_number(reading):
    msgs = [A(content=SAME_NUMBER_Q), H(content="ايه")]
    directive = graph._build_same_number_yes_directive(msgs, "booking", reading)
    assert "BOOK ON THIS WHATSAPP NUMBER" in directive and "get_patient_info" in directive


def test_no_or_another_question_is_not_a_yes_to_the_number():
    msgs = [A(content=SAME_NUMBER_Q), H(content="لا")]
    assert graph._build_same_number_yes_directive(msgs, "booking", {"declines": True}) == ""
    other = [A(content="تحب أشوف لك مواعيد يوم الاثنين؟"), H(content="ايه")]
    assert graph._build_same_number_yes_directive(other, "booking", {"confirms": True}) == ""
    assert graph._build_same_number_yes_directive(msgs[:1] + [H(content="ايه")], "cancel", {"confirms": True}) == ""


# ----------------------------------------------------------------------
# The nearer ones of the asked weekday had nothing open
# ----------------------------------------------------------------------

def _sundays_from(start, count):
    first = start + timedelta(days=(6 - start.weekday()) % 7)
    return [first + timedelta(days=7 * i) for i in range(count)]


def _resolve_sunday(open_dates):
    sid = "followup-sunday"
    tools._BOOKING_SESSIONS[sid] = {"doctor_id": "D1", "branch_id": "B1"}
    state = {"session_id": sid, "client_id": "tanasuq-test", "messages": [H(content="الاحد الجاي")],
             "templates": {"_doctors_base_url": "http://doctors.test", "_timezone": "Asia/Riyadh"}}
    items = [{"slotStart": f"{d.isoformat()}T17:00:00+03:00", "slotEnd": f"{d.isoformat()}T17:30:00+03:00",
              "isBooked": False} for d in open_dates]
    listed = {"success": True, "status_code": 200, "error": None, "data": {"items": items}}
    try:
        with patch("api.get_doctor_schedule_slots", return_value=listed):
            return tools.resolve_available_day.func(state=state, weekday_name="الاحد الجاي")
    finally:
        tools._BOOKING_SESSIONS.pop(sid, None)


def test_a_later_sunday_says_the_nearer_ones_have_nothing_open():
    lead = (datetime.now(ZoneInfo("Asia/Riyadh")) + timedelta(hours=12)).date()
    sundays = _sundays_from(lead, 3)
    result = _resolve_sunday([sundays[2]])
    assert result["status"] == "found" and result["date"] == sundays[2].isoformat()
    skipped = result["nearer_dates_without_slots"]
    assert len(skipped) == 2 and skipped[0].endswith(sundays[0].strftime("%d/%m/%Y"))
    assert "note" in result


def test_the_nearest_sunday_carries_no_note():
    lead = (datetime.now(ZoneInfo("Asia/Riyadh")) + timedelta(hours=12)).date()
    result = _resolve_sunday(_sundays_from(lead, 2))
    assert result["status"] == "found" and "note" not in result and "nearer_dates_without_slots" not in result


# ----------------------------------------------------------------------
# A short detour does not wipe the booking in progress
# ----------------------------------------------------------------------

def _booking_session(sid, booking_turn):
    tools._BOOKING_SESSIONS[sid] = {"doctor_id": "D1", "branch_id": "B1", "selected_slot": {"a": 1},
                                    "_booking_turn": booking_turn}
    return tools._BOOKING_SESSIONS[sid]


@pytest.mark.parametrize("turn,reading,kept", [
    (7, {"intent": "booking", "entities": {"time": "5"}}, True),         # back two messages later: "الساعة 5"
    (7, {"intent": "answer"}, True),
    (7, {"intent": "booking", "doctor_name": "أحمد"}, False),           # a new doctor is a new booking
    (7, {"intent": "booking", "entities": {"service": "تقييم"}}, False),
    (12, {"intent": "booking", "entities": {"time": "5"}}, False),      # left long ago
])
def test_returning_to_booking_after_a_detour(turn, reading, kept):
    sid = f"followup-detour-{turn}-{sorted(reading)}"
    session = _booking_session(sid, booking_turn=5)
    try:
        graph._clear_abandoned_booking_context("booking", "faq", "semantic: intent changed faq -> booking", sid,
                                               reading=reading, turn=turn)
        assert bool(session.get("doctor_id")) is kept
    finally:
        tools._BOOKING_SESSIONS.pop(sid, None)


def test_the_router_records_when_booking_last_had_the_turn(reader):
    sid = "followup-router-turn"
    reader.table["احجز مع د فرح"] = {"intent": "booking", "doctor_name": "فرح"}
    try:
        graph.router({"messages": [H(content="هلا"), A(content="أهلا"), H(content="احجز مع د فرح")],
                      "session_id": sid, "active_agent": "concierge"})
        assert tools._BOOKING_SESSIONS[sid]["_booking_turn"] == 2
    finally:
        tools._BOOKING_SESSIONS.pop(sid, None)


# ----------------------------------------------------------------------
# A greeting or thanks after the first turn: one line from code
# ----------------------------------------------------------------------

@pytest.mark.parametrize("message,expected", [
    ("شكرا", "في الخدمة دائمًا"),
    ("مساء الخير", "مساء النور"),
    ("السلام عليكم", "وعليكم السلام"),
])
def test_a_later_greeting_costs_no_model_call(session_id, llm, reader, message, expected):
    reader.table["هلا"] = {"intent": "greeting", "confidence": 1.0}
    send(session_id, "هلا")
    reader.table[message] = {"intent": "greeting", "confidence": 1.0}
    reply = send(session_id, message)["reply"]
    assert expected in reply and "لطيفة" not in reply
    assert len(llm.calls) == 0


def test_a_greeting_in_the_middle_of_a_booking_still_reaches_the_model(session_id, llm, reader):
    reader.table["احجز مع د فرح"] = {"intent": "booking", "doctor_name": "فرح"}
    llm._responses.append(AIMessage(content="تحب تحجز في أي يوم؟"))
    with patch.multiple(graph, _VERIFIERS_SAFETY_STRICT=False, _VERIFIERS_FLOW_STRICT=False):
        send(session_id, "احجز مع د فرح")
    calls = len(llm.calls)
    reader.table["شكرا"] = {"intent": "greeting", "confidence": 1.0}
    llm._responses.append(AIMessage(content="العفو 🌷 أي يوم يناسبك؟"))
    with patch.multiple(graph, _VERIFIERS_SAFETY_STRICT=False, _VERIFIERS_FLOW_STRICT=False):
        send(session_id, "شكرا")
    assert state_of(session_id)["active_agent"] == "booking"
    assert len(llm.calls) == calls + 1, "the booking step waiting on an answer is the model's to continue"


def test_the_later_greeting_follows_the_conversations_language():
    assert graph._later_greeting_reply("thanks a lot", "en").startswith("Always happy to help")
    assert graph._later_greeting_reply("good morning", "en").startswith("Good morning")


def test_the_yes_reaches_the_booking_agent_in_a_real_turn(session_id, llm, reader):
    relaxed = patch.multiple(graph, _VERIFIERS_SAFETY_STRICT=False, _VERIFIERS_FLOW_STRICT=False)
    reader.table["احجز مع د نجود"] = {"intent": "booking", "doctor_name": "نجود"}
    llm._responses.append(AIMessage(content=SAME_NUMBER_Q))
    with relaxed:
        send(session_id, "احجز مع د نجود")
    reader.table["ايه"] = {"intent": "answer", "answer_to_previous_question": True}
    llm._responses.append(AIMessage(content="تمام 🌷"))
    with relaxed:
        send(session_id, "ايه")
    assert "THEY SAID YES - BOOK ON THIS WHATSAPP NUMBER" in llm.system_prompts()[-1]
