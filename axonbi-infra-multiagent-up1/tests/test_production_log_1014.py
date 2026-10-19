"""
tanasuq-production, 2026-10-14 (from the chats, no log):

  - "حاب اتاكد من الايميل الخاص فيني" -> the appointment and its email,
    then "هل هذا هو الموعد الذي تبغى تلغيه؟" - nobody had asked to cancel.
  - "ابي التقرير الطبي يجهز قبل الجمعة" -> "الله يشافيك، حاول ترتاح وتشرب
    سوائل دافية" twice: a request for a medical report read as a symptom.
"""

import graph
import tools
import understanding

_SID = "log-1014"

_CARD = (
    "👤 الاسم: Rayan Khaled AL Jameli\n"
    "👨‍⚕️ الطبيب: عمر المديفر\n"
    "🗓️ التاريخ: 16/11/2026\n\n"
    "الإيميل المسجل عندنا هو: x@example.com\n\n"
    "هل هذا هو الموعد الذي تبغى تلغيه؟"
)


def _state(cancel_requested=False):
    tools._BOOKING_SESSIONS.pop(_SID, None)
    if cancel_requested:
        tools._get_booking_session(_SID)["_cancel_requested"] = True
    return {"session_id": _SID, "messages": []}


def test_an_unasked_cancel_question_becomes_anything_else():
    try:
        fixed = graph._unasked_cancel_question_fix(_CARD, _state(), "ar")
        assert "تلغيه" not in fixed
        assert "الإيميل المسجل عندنا هو" in fixed
        assert fixed.endswith("في شي ثاني أقدر أساعدك فيه؟")
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)


def test_the_question_stays_when_they_asked_to_cancel():
    try:
        assert graph._unasked_cancel_question_fix(_CARD, _state(cancel_requested=True), "ar") == _CARD
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)


def test_the_capability_menu_is_untouched():
    menu = "يمكنني مساعدتك في:\n✏️ تعديل أو إلغاء موعد قائم\nكيف أقدر أساعدك؟"
    try:
        assert graph._unasked_cancel_question_fix(menu, _state(), "ar") == menu
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)


def test_a_medical_report_is_staff_work_not_a_symptom():
    assert "ابي التقرير الطبي" in understanding.PROMPT
    assert "medical report" in understanding.PROMPT


# ----------------------------------------------------------------------
# 13:35 (966508800894): eleven psychotherapists listed as available, then
# "ما في مواعيد ... مع أي من أخصائيي العلاج النفسي"
# ----------------------------------------------------------------------

from datetime import datetime, timedelta
from unittest.mock import patch

import tool_result_guidance


def test_nothing_for_the_specialty_is_checked_doctor_by_doctor():
    later = (datetime.now() + timedelta(days=3)).replace(hour=16, minute=0, second=0, microsecond=0)
    doctors = {"success": True, "data": {"items": [{"id": "dr-nada"}, {"id": "dr-asila"}]}}

    def slots(base_url, doctor_ids, **kwargs):
        if len(doctor_ids) > 1:
            return {"success": True, "data": {"items": []}}
        items = [] if doctor_ids == ["dr-nada"] else [
            {"slotStart": later.isoformat(), "slotEnd": (later + timedelta(minutes=30)).isoformat(),
             "doctorId": "dr-asila", "isBooked": False}]
        return {"success": True, "data": {"items": items}}

    state = {"session_id": _SID, "messages": [], "understanding": None,
             "templates": {"_timezone": "Asia/Riyadh", "_doctors_base_url": "https://portal.x.test"}}
    with patch.object(tools, "_sanitize_specialty_ids", return_value=(["sp"], [])), \
         patch.object(tools.api, "get_doctors", return_value=doctors), \
         patch.object(tools.api, "get_doctor_schedule_slots", side_effect=slots):
        result = tools.find_best_doctor_in_specialty.func(state=state, specialty_ids=["sp"], criteria="soonest")
    assert result["status"] == "found" and result["doctor"]["id"] == "dr-asila"


def test_nothing_found_is_not_said_of_the_whole_specialty():
    guidance = tool_result_guidance.guidance_for("find_best_doctor_in_specialty", {"status": "not_found"})
    assert guidance and "never contradict" in guidance
