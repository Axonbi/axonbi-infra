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
