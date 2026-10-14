"""The review card is recognised even when the client's own
msg_booking_confirmation template words its question differently
(tanasuq-saudi, 2026-10-14 15:45 - every "نعم" re-printed the card)."""

from langchain_core.messages import AIMessage, HumanMessage

import tools

CARD_AR = (
    "يرجى مراجعة بيانات الحجز:\n🏥 الفرع: فرع أ\n👨‍⚕️ الطبيب: سامح حسن\n"
    "📅 التاريخ: الأربعاء 28/10/2026\n🕐 الوقت: 6:10 مساءً\n👤 الاسم: hanine aymen\n"
    "📱 الجوال: +201003365691\n\n✅ هل جميع البيانات صحيحة وتود تأكيد الحجز؟"
)
CARD_EN = (
    "Please review your booking details:\n🏥 Branch: A\n\n"
    "✅ Are all the details correct and would you like to confirm the booking?"
)


def _state(card, template="تم استلام طلبك، هل تريد المتابعة؟"):
    return {"templates": {"msg_booking_confirmation": template},
            "messages": [AIMessage(card), HumanMessage("نعم")]}


def test_documented_card_counts_even_with_a_different_template():
    assert tools._review_card_was_shown(_state(CARD_AR))


def test_english_card_counts():
    assert tools._review_card_was_shown(_state(CARD_EN))


def test_no_card_still_refused():
    assert not tools._review_card_was_shown(_state("تحب تضيف بريد إلكتروني؟"))
