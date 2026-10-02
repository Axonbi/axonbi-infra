"""
Regressions from the tanasuq-production log of 2026-10-01 17:06 to
2026-10-02 15:22.

  - The SMS provider refused every code ("you've run out of points") and
    the patient was told "أرسلت لك رمز التحقق" each time.
  - "This slot is already booked" -> the same 5:10 was offered, picked and
    refused again, three times.
  - "ارسلي اللوكيشن" after a booking got the out-of-scope refusal; "اي فرع
    أقرب للنظيم" got "no branch named أقرب للنظيم".
  - "السلام عليكم ورحمة الله وبركاته" with booking holding the turn got the
    out-of-scope menu.
  - "+535230420" (a Saudi mobile with no 966) went to the provider as is.
  - "كم مدة الجلسه؟" was answered "50 minutes" with nothing behind it.
"""

import json
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agents.semantic_router as sr
import graph
import tools
import understanding
from agents.semantic_router import TurnFacts
from conftest import send, state_of

SAUDI = {"templates": {"_timezone": "Asia/Riyadh"}, "channel_phone": "966535230420",
         "messages": [], "session_id": "log-1002-otp"}


# ----------------------------------------------------------------------
# A code that was not sent is never reported as sent
# ----------------------------------------------------------------------

@pytest.mark.parametrize("body,status", [
    ('{"success":false,"message":"Unfortunately, it seems you\'ve run out of points."}', "otp_send_failed"),
    ('{"message":"Phone number is not valid, phone should start with + sign"}', "invalid_phone"),
])
def test_send_otp_reports_what_the_provider_did(body, status):
    refused = {"success": False, "status_code": 422, "error": "send_otp_failed", "details": [body], "data": None}
    with patch.object(tools, "OTP_PROVIDER", "authentica"), \
            patch.object(tools.api, "authentica_send_otp", return_value=refused):
        assert tools.send_otp.func(state=SAUDI, phone="0505490801") == {"status": status}


def test_send_otp_still_reports_a_real_send():
    sent = {"success": True, "status_code": 200, "error": None, "details": [], "data": {}}
    with patch.object(tools, "OTP_PROVIDER", "authentica"), \
            patch.object(tools.api, "authentica_send_otp", return_value=sent):
        assert tools.send_otp.func(state=SAUDI, phone="0505490801") == {"status": "otp_sent"}


def _turn_with_send_otp(status):
    return {"messages": [
        HumanMessage(content="اعتمد هذا الرقم 0505490801"),
        AIMessage(content="", tool_calls=[{"name": "send_otp", "args": {"phone": "0505490801"}, "id": "c1"}]),
        ToolMessage(content=json.dumps({"status": status}), name="send_otp", tool_call_id="c1"),
    ]}


@pytest.mark.parametrize("status,expected", [
    ("otp_send_failed", "ما قدرت أرسل رمز التحقق"),
    ("invalid_phone", "الرقم يبدو غير كامل"),
])
def test_a_claim_that_the_code_was_sent_is_replaced(status, expected):
    draft = "أرسلت لك رمز التحقق على الرقم 0505490801. من فضلك أرسل لي رمز التحقق."
    corrected = graph._otp_not_sent_correction(draft, _turn_with_send_otp(status), "ar")
    assert corrected and expected in corrected


def test_a_sent_code_or_an_honest_draft_is_left_alone():
    draft = "أرسلت لك رمز التحقق على الرقم 0505490801."
    assert graph._otp_not_sent_correction(draft, _turn_with_send_otp("otp_sent"), "ar") is None
    honest = "عذرًا، ما قدرت أرسل الرمز. تحب أحولك لخدمة العملاء؟"
    assert graph._otp_not_sent_correction(honest, _turn_with_send_otp("otp_send_failed"), "ar") is None


# ----------------------------------------------------------------------
# A slot the booking system refused is never offered or locked again
# ----------------------------------------------------------------------

def _slot(start, end):
    return {"slotStart": start, "slotEnd": end, "time_display": start[11:16], "date_display": "05/10/2026"}


def test_a_refused_slot_leaves_the_list_and_cannot_be_locked():
    sid = "log-1002-slot"
    session = tools._get_booking_session(sid)
    session.update({"doctor_id": "d1", "branch_id": "b1"})
    session["last_list"] = {"entity_type": "slot", "items": [
        _slot("2026-10-05T16:10:00", "2026-10-05T16:20:00"),
        _slot("2026-10-05T17:10:00", "2026-10-05T17:20:00"),
    ]}
    try:
        tools._remember_failed_slot(session, "d1", "b1", "2026-10-05T17:10:00")
        assert [s["slotStart"] for s in session["last_list"]["items"]] == ["2026-10-05T16:10:00"]

        # Shown again from an older list, it is still refused.
        session["last_list"]["items"].append(_slot("2026-10-05T17:10:00", "2026-10-05T17:20:00"))
        result = tools.select_appointment_slot.func(state={"session_id": sid}, user_input="2")
        assert result == {"status": "slot_unavailable"}
        assert not session.get("selected_slot")
    finally:
        tools._BOOKING_SESSIONS.pop(sid, None)


def test_the_slot_length_comes_from_the_booking_system():
    sid = "log-1002-length"
    tools._get_booking_session(sid)["last_list"] = {"entity_type": "slot", "items": [
        _slot("2026-10-05T17:10:00", "2026-10-05T17:20:00")]}
    try:
        assert "10 minutes" in graph._build_slot_length_directive(sid)
    finally:
        tools._BOOKING_SESSIONS.pop(sid, None)
    assert graph._build_slot_length_directive("log-1002-nothing") == ""


# ----------------------------------------------------------------------
# Location questions are never out of scope
# ----------------------------------------------------------------------

@pytest.mark.parametrize("facts,agent", [
    (TurnFacts(previous="booking", flow_completed=True), "booking"),
    (TurnFacts(previous=None), "faq"),
])
def test_send_me_the_location_is_answered(facts, agent):
    reading = {"intent": "other", "confidence": 1.0, "asks_location": True}
    decision = sr.decide(reading, facts)
    assert not decision.out_of_scope and decision.agent == agent


def test_something_unrelated_is_still_out_of_scope():
    decision = sr.decide({"intent": "other", "confidence": 1.0}, TurnFacts(previous=None))
    assert decision.out_of_scope


def test_a_location_question_carries_the_real_addresses(tmp_path):
    kb = tmp_path / "kb.txt"
    kb.write_text(
        "1. معلومات عامة | General Overview\nمستشفى للطب النفسي.\n"
        "6. معلومات التواصل والفروع | Contact & Branch Information\n"
        "فرع المنار\nحي المنار، الرياض\nفرع النزهة\nالنزهة، الرياض\n",
        encoding="utf-8",
    )
    directive = graph._build_location_question_directive(
        {"asks_location": True}, {"_knowledge_base_file": str(kb)})
    assert "حي المنار" in directive and "share_branch_location" in directive
    assert graph._build_location_question_directive({}, {"_knowledge_base_file": str(kb)}) == ""


# ----------------------------------------------------------------------
# A greeting inside a flow: the greeting, then the open question
# ----------------------------------------------------------------------

def test_a_salam_inside_a_flow_returns_the_open_question():
    reply = graph._greeting_inside_a_flow_reply(
        "السلام عليكم ورحمة الله وبركاته", "تحب تحجز في أي يوم؟", "ar")
    assert reply == "وعليكم السلام 🌷\nتحب تحجز في أي يوم؟"


def test_a_salam_over_a_card_sends_the_whole_card_back():
    # The gates read the reply right before the patient's yes - a lone
    # closing question there held the booking back a turn.
    card = "يرجى مراجعة بيانات الحجز:\n🏥 الفرع: المنار\n✅ هل جميع البيانات صحيحة وتود تأكيد الحجز؟"
    reply = graph._greeting_inside_a_flow_reply("السلام عليكم", card, "ar")
    assert reply == "وعليكم السلام 🌷\n" + card


def test_thanks_with_a_question_open_is_left_to_the_model():
    assert graph._greeting_inside_a_flow_reply("شكرا", "تحب تحجز في أي يوم؟", "ar") is None
    assert "في الخدمة" in graph._greeting_inside_a_flow_reply("شكرا", "تم تأكيد حجزك 🌷", "ar")


def test_a_salam_in_the_middle_of_a_booking_costs_no_model_call(session_id, llm, reader):
    relaxed = patch.multiple(graph, _VERIFIERS_SAFETY_STRICT=False, _VERIFIERS_FLOW_STRICT=False)
    reader.table["احجز مع د فرح"] = {"intent": "booking", "doctor_name": "فرح"}
    llm._responses.append(AIMessage(content="تحب تحجز في أي يوم؟"))
    with relaxed:
        send(session_id, "احجز مع د فرح")
    calls = len(llm.calls)
    reader.table["السلام عليكم"] = {"intent": "greeting", "confidence": 1.0}
    with relaxed:
        reply = send(session_id, "السلام عليكم")["reply"]
    assert reply.startswith("وعليكم السلام") and "تحب تحجز في أي يوم؟" in reply
    assert "عذرًا" not in reply and len(llm.calls) == calls
    assert state_of(session_id)["active_agent"] == "booking"


# ----------------------------------------------------------------------
# Smaller ones
# ----------------------------------------------------------------------

@pytest.mark.parametrize("typed,normalized", [
    ("+535230420", "+966535230420"),        # a Saudi mobile with a "+" and no 966
    ("+966535230420", "+966535230420"),
    ("+201001255864", "+201001255864"),     # a real foreign number is untouched
    ("+447911123456", "+447911123456"),
])
def test_a_plus_before_a_local_mobile(typed, normalized):
    assert tools.normalize_phone_number(typed, SAUDI) == normalized


def test_a_single_open_time_is_not_shown_as_a_range():
    assert graph._time_range_text("2:00 مساءً", "2:00 مساءً") == " — الساعة 2:00 مساءً"
    assert graph._time_range_text("1:00 مساءً", "4:00 مساءً") == " — من 1:00 مساءً إلى 4:00 مساءً"


def test_a_no_to_a_later_question_is_not_rejecting_the_booking_shown_earlier():
    found = {"status": "found_one", "booking": {"doctorName": "ندى جنّادي"}}
    messages = [
        HumanMessage(content="نعم"),
        AIMessage(content="", tool_calls=[{"name": "lookup_appointment", "args": {}, "id": "l1"}]),
        ToolMessage(content=json.dumps(found), name="lookup_appointment", tool_call_id="l1"),
        AIMessage(content="هل هذا هو الموعد الذي تبغى تعدله؟ (نعم/لا)"),
        HumanMessage(content="نعم"),
        AIMessage(content="ما فيه مواعيد متاحة ليوم السبت. تحب نجرب يوم ثاني؟"),
        HumanMessage(content="لا خلص ثبت الموعد"),
    ]
    reply = "الموعد الحالي مثبت عندك مع الدكتورة ندى جنّادي يوم السبت."
    assert not graph._reply_mishandles_rejected_single_booking(reply, {"messages": messages})


def test_the_understanding_prompt_knows_these_cases():
    prompt = understanding.PROMPT
    assert "مواعيد الزيارة" in prompt and "لغيتوا موعدي" in prompt
    assert "ارسلي اللوكيشن" in prompt and "Saying it WAS cancelled is not" in prompt
