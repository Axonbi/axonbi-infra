"""
tanasuq-production, 2026-10-05 13:49 (session 201025981330) and 21:23
(session 966567700950): the doctor list came from the medical agent and
was shown again by booking. The patient answered "2" / "7"; the model
called get_doctor_schedule_for_booking without confirming the doctor,
got "missing_doctor", and told the patient "ما قدرت أجيب جدول مواعيد
الدكتور". The number is now resolved in code against the line the patient
saw with that number.
"""

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

import tools

_SID = "doctor-pick-shown-list"

_REMEMBERED = [
    {"id": "d1", "name": "اصيلا الحسن"},
    {"id": "d2", "name": "عمر المديفر"},
    {"id": "d3", "name": "عبدالله الهذلول"},
    {"id": "d4", "name": "سعد الماضي"},
    {"id": "d5", "name": "أحمد يوسف"},
    {"id": "d6", "name": "فرح الحميدي"},
    {"id": "d7", "name": "جزيل شاهين"},
    {"id": "d8", "name": "مشاعل الشعلان"},
    {"id": "d9", "name": "العنود الخليفة"},
    {"id": "d10", "name": "مها المحبوب"},
    {"id": "d11", "name": "معاذ الحقيلي"},
]

_SHOWN_1348 = (
    "1️⃣ اصيلا الحسن - علاج نفسي\n2️⃣ عمر المديفر - طب نفسي\n"
    "3️⃣ عبدالله الهذلول - طب نفسي\n4️⃣ سعد الماضي - علاج نفسي\n"
    "5️⃣ أحمد يوسف - علاج نفسي\n6️⃣ فرح الحميدي - طب نفسي\n\n"
    "أي دكتور تحب تحجز عنده؟"
)

_SCHEDULE = {"success": True, "data": {"items": [
    {"branchId": "b1", "branchName": "النزهة", "recurringDaysNames": ["Wednesday"],
     "fromDateTime": "2026-09-01T16:00:00+03:00", "toDateTime": "2026-11-30T18:00:00+03:00"},
    {"branchId": "b2", "branchName": "المنار", "recurringDaysNames": ["Monday"],
     "fromDateTime": "2026-10-01T16:00:00+03:00", "toDateTime": "2027-01-30T18:00:00+03:00"},
]}}


def _state(shown, answer):
    return {
        "session_id": _SID,
        "templates": {"_doctors_base_url": "https://x.test"},
        "messages": [
            AIMessage(content=shown),
            HumanMessage(content=answer),
            AIMessage(content="", tool_calls=[{"name": "get_doctor_schedule_for_booking", "args": {}, "id": "c1"}]),
        ],
    }


def _schedule(shown, answer, remembered=_REMEMBERED, doctor_id=None):
    tools._BOOKING_SESSIONS.pop(_SID, None)
    session = tools._get_booking_session(_SID)
    session["last_list"] = {"entity_type": "doctor", "items": list(remembered)}
    if doctor_id:
        session["doctor_id"] = doctor_id
    try:
        with patch.object(tools.api, "get_doctor_schedule", return_value=_SCHEDULE) as call:
            result = tools.get_doctor_schedule_for_booking.func(state=_state(shown, answer))
        asked = call.call_args.kwargs["doctor_ids"] if call.called else None
        return result, asked, dict(session)
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)


def test_the_number_picks_the_doctor_on_that_line_and_the_schedule_comes_back():
    result, asked, session = _schedule(_SHOWN_1348, "2")
    assert result["status"] == "found"
    assert asked == ["d2"]
    assert session["doctor_id"] == "d2" and session["doctor_display_name"] == "عمر المديفر"


def test_arabic_digit_is_the_same_pick():
    _result, asked, _session = _schedule(_SHOWN_1348, "٢")
    assert asked == ["d2"]


def test_the_shown_line_wins_over_the_remembered_position():
    # The reply showed fewer doctors than the tool returned: "2" is the
    # SECOND LINE (جزيل), not the second remembered doctor (عمر).
    shown = "1️⃣ اصيلا الحسن\n2️⃣ جزيل شاهين\n\nأي دكتور تحب تحجز عنده؟"
    _result, asked, _session = _schedule(shown, "2")
    assert asked == ["d7"]


def test_two_digit_keycaps_with_bidi_marks_are_read():
    shown = "🔟 مها المحبوب — أخصائي نفسي\n⁦1️⃣1️⃣⁩ معاذ الحقيلي — أخصائي نفسي"
    _result, asked, _session = _schedule(shown, "11")
    assert asked == ["d11"]
    _result, asked, _session = _schedule(shown, "10")
    assert asked == ["d10"]


def test_a_number_with_no_such_line_is_still_missing_doctor():
    result, asked, _session = _schedule(_SHOWN_1348, "9")
    assert result == {"status": "missing_doctor"} and asked is None


def test_a_line_naming_no_remembered_doctor_is_still_missing_doctor():
    shown = "1️⃣ فرع النزهة\n2️⃣ فرع المنار"
    result, asked, _session = _schedule(shown, "2")
    assert result == {"status": "missing_doctor"} and asked is None


def test_words_are_not_a_pick():
    result, asked, _session = _schedule(_SHOWN_1348, "نعم")
    assert result == {"status": "missing_doctor"} and asked is None


def test_a_branch_list_is_not_used_for_a_doctor():
    tools._BOOKING_SESSIONS.pop(_SID, None)
    session = tools._get_booking_session(_SID)
    session["last_list"] = {"entity_type": "branch", "items": [{"id": "d2", "name": "عمر المديفر"}]}
    try:
        with patch.object(tools.api, "get_doctor_schedule", return_value=_SCHEDULE) as call:
            result = tools.get_doctor_schedule_for_booking.func(state=_state(_SHOWN_1348, "2"))
        assert result == {"status": "missing_doctor"} and not call.called
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)


def test_an_already_confirmed_doctor_is_kept():
    _result, asked, session = _schedule(_SHOWN_1348, "2", doctor_id="d4")
    assert asked == ["d4"] and session["doctor_id"] == "d4"
