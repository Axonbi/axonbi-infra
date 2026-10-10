"""
tanasuq-production, 2026-10-10 - from the clinic's complaint and the
patient's screenshots (no log survived):

  - After "Yes" to the review card the reservation was refused. The refusal
    came back as "invalid_details", the patient was asked for their name and
    email again, and was finally told "it is already confirmed" for a 12:00
    the clinic never held. They turned up at 12:00.
  - The review card read "🏥 الفرع: أي فرع تفضلين؟".
  - "اليوم" / "اليوم السبت ١٠ اكتوبر" was answered with other days and
    Saturday 17/10's times, with no word that today cannot be booked; "مافي
    10 اكتوبر ؟" got "خلني أتأكد ... بدل ما أخمّن" twice.
  - "اليوم عندي موعد ايش" was answered with "which one?" over a list holding
    one appointment today.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import graph
import tools
import understanding

_SID = "log-1010"
_TZ = "Asia/Riyadh"
_PHONE = "966551768024"


def _state(messages=None):
    return {
        "session_id": _SID,
        "channel_phone": _PHONE,
        "templates": {"_timezone": _TZ, "_cms_base_url": "https://cms.x.test",
                      "_base_url": "https://portal.x.test", "_doctors_base_url": "https://portal.x.test"},
        "messages": messages or [],
    }


# ----------------------------------------------------------------------
# A refused reservation is never "invalid details" unless a detail was
# refused, and the patient's real booking that day is shown as it is
# ----------------------------------------------------------------------

_SLOT = "2026-10-17T12:00:00"
_SLOTS = {"success": True, "data": {"items": [{
    "slotStart": "2026-10-17T12:00:00+03:00", "slotEnd": "2026-10-17T12:30:00+03:00",
    "isBooked": False, "doctorId": "dr-samar", "branchId": "b1", "serviceId": "s1",
}]}}
_REFUSED = {"success": False, "status_code": 400, "error": "validation_error",
            "details": [{"field": "", "message": "Patient already has a booking with this doctor on the same day"}],
            "data": None}
_HER_BOOKING = {"success": True, "data": {"items": [{
    "id": "bk-1", "bookingRefNum": "GuestBookingNum-2026-10-17-9", "status": 1,
    "doctorId": "dr-samar", "doctorName": "سمر الخليفي", "branchName": "المنار",
    "bookingTimeFrom": "2026-10-17T15:00:00+03:00", "bookingTimeTo": "2026-10-17T15:30:00+03:00",
}]}}
_NO_BOOKINGS = {"success": True, "data": {"items": []}}


def _book(bookings):
    tools._BOOKING_SESSIONS.pop(_SID, None)
    session = tools._get_booking_session(_SID)
    session.update({"doctor_id": "dr-samar", "branch_id": "b1", "review_shown": True})
    try:
        with patch.object(tools.api, "get_doctor_schedule_slots", return_value=_SLOTS), \
             patch.object(tools.api, "create_booking", return_value=_REFUSED), \
             patch.object(tools.api, "get_bookings_by_phone", return_value=bookings):
            return tools.create_new_booking.func(
                state=_state(), slot_start=_SLOT, slot_end="2026-10-17T12:30:00",
                patient_full_name="Laiba Faruqi", mobile_number=_PHONE)
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)


def test_her_own_booking_that_day_is_returned_with_its_real_time():
    result = _book(_HER_BOOKING)
    assert result["status"] == "patient_has_booking_that_day"
    assert result["appointment"]["ref"] == "GuestBookingNum-2026-10-17-9"
    assert result["appointment"]["bookingTimeFrom"].startswith("2026-10-17T15:00")


def test_a_refusal_not_about_her_details_is_not_invalid_details():
    result = _book(_NO_BOOKINGS)
    assert result["status"] == "booking_refused"
    assert result["reason"] == ["Patient already has a booking with this doctor on the same day"]


# ----------------------------------------------------------------------
# "It is already confirmed" with nothing behind it never reaches the patient
# ----------------------------------------------------------------------

_CLAIM_1351 = ("You already have a booking with Dr. Samar Alkhelfi on Saturday, 10 October 2026 "
               "at 12:00 PM. If you want to keep this appointment, it is already confirmed. "
               "Is there anything else I can assist you with?")


def _after(result):
    return _state([
        HumanMessage(content="Yes"),
        AIMessage(content="", tool_calls=[{"name": "create_new_booking", "args": {}, "id": "c1"}]),
        ToolMessage(content=json.dumps(result), name="create_new_booking", tool_call_id="c1"),
    ])


def test_the_screenshot_claim_is_caught():
    assert graph._reply_claims_an_unverified_booking(_CLAIM_1351, _after({"status": "error"}))


def test_confirmed_is_caught_even_when_the_refusal_gave_a_reason():
    state = _after({"status": "booking_refused", "reason": ["already has a booking same day"]})
    assert graph._reply_claims_an_unverified_booking(_CLAIM_1351, state)


def test_relaying_the_refusal_reason_is_allowed():
    state = _after({"status": "booking_refused", "reason": ["already has a booking same day"]})
    reply = ("Sorry, the booking was not completed: you already have a booking with the same "
             "doctor that day, so a second one isn't possible.")
    assert not graph._reply_claims_an_unverified_booking(reply, state)


def test_a_booking_a_tool_returned_may_be_stated():
    state = _after({"status": "patient_has_booking_that_day", "appointment": {"ref": "X"}})
    assert not graph._reply_claims_an_unverified_booking("عندك حجز مع سمر الخليفي الساعة 3:00 مساءً.", state)


def test_conditionals_and_negations_are_not_claims():
    state = _after({"status": "error"})
    assert not graph._reply_claims_an_unverified_booking("لو عندك موعد قائم تقدر تعدله.", state)
    assert not graph._reply_claims_an_unverified_booking("ما عندك حجز مسجل برقمك.", state)
    assert not graph._reply_claims_an_unverified_booking("هل عندك موعد؟", state)


def test_the_check_is_registered_and_substitutes():
    descriptions = [d for _c, _r, d in graph._REPLY_VERIFIERS]
    assert any("unverified existing booking" in d for d in descriptions)
    assert graph._substitute_despite_disabled_fallback(
        next(d for d in descriptions if "unverified existing booking" in d))


def test_the_substitute_names_the_real_reason():
    cms = "Patient already has a booking with this doctor on the same day"
    refused = _after({"status": "booking_refused", "reason": [cms]})
    # The system's message as it is, then the offer of a staff member - nothing else.
    assert graph._safe_fallback_reply(refused, "ar", "unverified existing booking") == f"{cms}\nتحب أحولك لموظف؟"
    assert graph._safe_fallback_reply(refused, "en", "unverified existing booking") == (
        f"{cms}\nWould you like me to transfer you to a staff member?")
    # "نعم" to it is a yes to a transfer the assistant really offered.
    offer = _state([AIMessage(content=f"{cms}\nتحب أحولك لموظف؟"), HumanMessage(content="نعم")])
    assert graph._assistant_offered_a_transfer(offer["messages"])

    own = _after({"status": "patient_has_booking_that_day", "appointment": {
        "doctorName": "سمر الخليفي", "weekday_display": "السبت", "date_display": "17/10/2026",
        "time_display": "3:00 مساءً"}})
    text = graph._safe_fallback_reply(own, "ar", "unverified existing booking")
    assert "نفس اليوم" in text and "3:00 مساءً" in text and "سمر الخليفي" in text


# ----------------------------------------------------------------------
# The review card's branch line never carries a question
# ----------------------------------------------------------------------

def test_a_question_in_the_branch_line_is_replaced_by_the_branch_on_file():
    card = ("يرجى مراجعة بيانات الحجز:\n🏥 الفرع: أي فرع تفضلين؟\n👨‍⚕️ الطبيب: سمر الخليفي\n"
            "📅 التاريخ: السبت 17-10-2026\n🕐 الوقت: 12:00 ظهرًا\n\n"
            "✅ هل جميع البيانات صحيحة وتود تأكيد الحجز؟")
    tools._BOOKING_SESSIONS.pop(_SID, None)
    session = tools._get_booking_session(_SID)
    session.update({"doctor_id": "dr-samar", "branch_id": "b1", "branch_display_name": "المنار"})
    try:
        fixed = graph._review_card_branch_fix(card, _state(), "ar")
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)
    assert "🏥 الفرع: المنار" in fixed and "أي فرع" not in fixed


# ----------------------------------------------------------------------
# Today is refused first, in code
# ----------------------------------------------------------------------

def _today_state(text):
    tools._BOOKING_SESSIONS.pop(_SID, None)
    tools._get_booking_session(_SID)["doctor_id"] = "dr-ahmad"
    return _state([HumanMessage(content=text)])


def _today():
    return tools._local_now_naive(_TZ).date()


def test_today_gets_the_same_day_notice_first():
    days = "🗓️ مواعيد أحمد يوسف المتاحة في المنار:\n1️⃣ الأربعاء 14/10/2026\nأي يوم يناسبك؟"
    try:
        for text in ("اليوم", f"اليوم السبت {_today().day} اكتوبر"):
            out = graph._same_day_notice(days, _today_state(text), "ar")
            assert out.startswith("للأسف ما نقدر نحجز في نفس اليوم")
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)


def test_todays_date_alone_counts_as_today():
    months = ["يناير", "فبراير", "مارس", "ابريل", "مايو", "يونيو", "يوليو",
              "اغسطس", "سبتمبر", "اكتوبر", "نوفمبر", "ديسمبر"]
    text = f"مافي {_today().day} {months[_today().month - 1]} ؟"
    try:
        assert graph._patient_asked_for_today(_today_state(text))
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)


def test_no_double_notice_and_no_notice_for_another_day():
    try:
        already = "للأسف ما نقدر نحجز في نفس اليوم 🌷\nتحب يوم ثاني؟"
        assert graph._same_day_notice(already, _today_state("اليوم"), "ar") == already
        assert graph._same_day_notice("تمام", _today_state("بكره"), "ar") == "تمام"
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)


def test_nothing_today_is_not_an_unverified_availability_denial():
    try:
        state = _today_state("اليوم")
        assert not graph._reply_denies_availability_without_lookup("ما فيه مواعيد متاحة اليوم", state)
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)


def test_the_guess_sentence_is_gone():
    for language in ("ar", "en"):
        text = graph._unverified_availability_reply(
            "reply said a doctor has no available appointments while no availability tool has run", language)
        assert text and "أخمّن" not in text and "guess" not in text


# ----------------------------------------------------------------------
# "اليوم عندي موعد ايش" - today's appointment is marked in the list
# ----------------------------------------------------------------------

def test_todays_appointment_is_marked():
    now = tools._local_now_naive(_TZ)
    today_at_5 = now.replace(hour=17, minute=0, second=0, microsecond=0).isoformat()
    later = (now + timedelta(days=5)).replace(hour=12, minute=0, second=0, microsecond=0).isoformat()
    shaped_today = tools._shape_appointment({"bookingTimeFrom": today_at_5 + "+03:00"}, _TZ)
    shaped_later = tools._shape_appointment({"bookingTimeFrom": later + "+03:00"}, _TZ)
    assert shaped_today["day_relation"] == "today" and shaped_later["day_relation"] is None

    found = {"status": "found_many", "appointments": [
        dict(shaped_later, doctorName="أحمد يوسف"), dict(shaped_today, doctorName="معاذ الحقيلي")]}
    messages = [HumanMessage(content="اليوم عندي موعد ايش"),
                AIMessage(content="", tool_calls=[{"name": "lookup_appointment", "args": {}, "id": "c1"}]),
                ToolMessage(content=json.dumps(found), name="lookup_appointment", tool_call_id="c1")]
    directive = graph._build_appointment_choice_directive(messages)
    assert "معاذ الحقيلي" in directive and "(اليوم)" in directive
    assert "exactly ONE row is on that day" in directive


def test_a_cancellation_call_question_is_not_a_cancel_request():
    assert "وصلني اتصال بالغاء الموعد" in understanding.PROMPT


# ----------------------------------------------------------------------
# The doctor's days shown are the days that can be booked (أحمد يوسف,
# 12:46: rota Sun/Mon/Tue, bookable Wed 14 / Thu 15 / Sat 17)
# ----------------------------------------------------------------------

def _slot(day, hour, branch="b-manar"):
    return {"slotStart": f"2026-10-{day:02d}T{hour:02d}:00:00+03:00",
            "slotEnd": f"2026-10-{day:02d}T{hour:02d}:30:00+03:00",
            "isBooked": False, "branchId": branch, "serviceName": "جلسة استشاره نفسيه تناسق"}


_ROTA = {"success": True, "data": {"items": [
    {"branchId": "b-manar", "branchName": "المنار", "recurringDaysNames": [d],
     "fromDateTime": f"2026-09-01T{f}:00:00+03:00", "toDateTime": f"2026-12-31T{t}:00:00+03:00",
     "serviceName": "جلسة استشاره نفسيه تناسق"}
    for d, f, t in (("Sunday", "14", "20"), ("Monday", "11", "17"), ("Tuesday", "14", "21"))]}}

_OPEN = {"success": True, "data": {"items": [
    _slot(14, 10), _slot(14, 15), _slot(15, 10), _slot(15, 13), _slot(17, 11), _slot(17, 16),
    _slot(18, 14), _slot(19, 11), _slot(20, 14),
]}}


def _schedule_with(open_slots):
    tools._BOOKING_SESSIONS.pop(_SID, None)
    session = tools._get_booking_session(_SID)
    session.update({"doctor_id": "dr-ahmad", "doctor_display_name": "أحمد يوسف"})
    fixed_now = datetime(2026, 10, 10, 12, 46)
    try:
        with patch.object(tools.api, "get_doctor_schedule", return_value=_ROTA), \
             patch.object(tools.api, "get_doctor_schedule_slots", return_value=open_slots), \
             patch.object(tools.api, "get_branches", return_value={"success": False}), \
             patch.object(tools, "_local_now_naive", return_value=fixed_now):
            return tools.get_doctor_schedule_for_booking.func(state=_state())
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)


def test_bookable_days_off_the_rota_are_shown_with_their_hours():
    result = _schedule_with(_OPEN)
    days = {row["recurringDaysNames"][0]: row for row in result["schedules"]}
    assert set(days) == {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Saturday"}
    assert days["Wednesday"]["fromDateTime"][11:16] == "10:00"
    assert days["Wednesday"]["toDateTime"][11:16] == "15:30"
    assert days["Wednesday"]["branchName"] == "المنار"


def test_a_rota_that_matches_the_slots_gets_nothing_added():
    only_rota_days = {"success": True, "data": {"items": [_slot(18, 14), _slot(19, 11), _slot(20, 14)]}}
    result = _schedule_with(only_rota_days)
    assert sorted(r["recurringDaysNames"][0] for r in result["schedules"]) == ["Monday", "Sunday", "Tuesday"]


def test_slots_at_another_branch_are_dropped():
    mixed = {"success": True, "data": {"items": [_slot(14, 10), _slot(15, 10, branch="b-nuzha")]}}
    kept = tools.api._only_branches(mixed, ["b-manar"])
    assert [i["branchId"] for i in kept["data"]["items"]] == ["b-manar"]
    assert tools.api._only_branches({"success": True, "data": {"items": [_slot(15, 10, "x")]}}, None)["data"]["items"]
