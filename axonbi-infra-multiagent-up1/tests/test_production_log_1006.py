"""
tanasuq-production, 2026-10-06/07:

  - the email request without a question mark was not an optional question,
    so "لا" ended the booking
  - the one-question trimmer cut at "د. " and dropped the doctor's name
  - the price follow-up "2" / "أصيلا الحسن" was refused as "no price asked"
  - "نعم إسم <other name>" confirmed the review under the name on the card
"""

from langchain_core.messages import AIMessage, HumanMessage

import graph
import tools


def test_email_request_without_question_mark_is_optional():
    msgs = [HumanMessage(content="x"), AIMessage(content="من فضلك أرسل بريدك الإلكتروني لإضافته للحجز."),
            HumanMessage(content="لا")]
    assert graph._assistant_asked_an_optional_question(msgs)


def test_review_card_refusal_is_not_optional():
    card = "يرجى مراجعة بيانات الحجز:\n📧 البريد الإلكتروني: a@b.com\n\n✅ هل جميع البيانات صحيحة وتود تأكيد الحجز؟"
    msgs = [HumanMessage(content="x"), AIMessage(content=card), HumanMessage(content="لا")]
    assert not graph._assistant_asked_an_optional_question(msgs)


def test_doctor_title_is_not_a_sentence_end():
    text = "ممكن تحدد لي وش السؤال بالضبط عن د. جزيل شاهين؟ هل تبي تعرف سعر الجلسة عنده؟"
    trimmed, removed = graph._strip_extra_questions(text, {})
    assert removed == 1
    assert "د." not in trimmed or "جزيل" in trimmed
    assert trimmed.strip().endswith("عنده؟")


def test_price_follow_up_answer_counts_as_price_turn():
    state = {
        "understanding": {"asks_price": False},
        "messages": [
            HumanMessage(content="سعر الجلسة"),
            AIMessage(content="أي دكتور تود تعرف سعر الجلسة عنده؟"),
            HumanMessage(content="2"),
        ],
    }
    assert tools._fees_requested(state)


def test_fees_still_private_when_not_asked():
    state = {
        "understanding": {"asks_price": False},
        "messages": [
            HumanMessage(content="هلا"),
            AIMessage(content="وش الفروع؟"),
            HumanMessage(content="الفروع"),
        ],
    }
    assert not tools._fees_requested(state)


def _review_state(reply):
    return {"messages": [HumanMessage(content=reply)]}


def test_yes_with_a_different_name_is_a_correction():
    assert tools._review_reply_corrects_card(_review_state("نعم إسم خالد النصيان"), "لطيفه الجربوع") == "name"


def test_plain_yes_is_not_a_correction():
    assert tools._review_reply_corrects_card(_review_state("نعم"), "لطيفه الجربوع") == ""
    assert tools._review_reply_corrects_card(_review_state("نعم اسمي لطيفه"), "لطيفه الجربوع") == ""


# ----------------------------------------------------------------------
# 2026-10-06/07, second batch
# ----------------------------------------------------------------------

from datetime import date


def test_a_date_with_a_month_name_is_a_date():
    assert tools._parse_explicit_date("المنار يوم الاثنين 3 نوفمبر الساعه ٥", date(2026, 10, 6)) == date(2026, 11, 3)
    assert tools._parse_explicit_date("3 November", date(2026, 10, 6)) == date(2026, 11, 3)
    assert tools._parse_explicit_date("نوفمبر", date(2026, 10, 6)) is None
    assert tools._month_named("نوفمبر") == 11


def test_the_doctors_name_is_not_the_patients():
    session = {"doctor_display_name": "عمر المديفر"}
    assert tools._name_is_the_doctor(session, "عمر المديفر")
    assert tools._name_is_the_doctor(session, "د. عمر المديفر")
    assert not tools._name_is_the_doctor(session, "رند ناif العنزي".replace("ناif", "نايف"))
    assert not tools._name_is_the_doctor({}, "عمر المديفر")


def test_being_late_offers_a_person_or_moving_never_cancelling():
    text = graph._clarification_question({"alternatives": ["cancel", "reschedule"]}, False)
    assert "خدمة العملاء" in text and "نأجل" in text and "نلغي" not in text
    english = graph._clarification_question({"alternatives": ["cancel", "reschedule"]}, True)
    assert "customer service" in english and "cancel" not in english.lower()


def test_unverified_availability_claims_become_a_question():
    desc = "reply said a doctor has no available appointments while no availability tool has run in this conversation"
    assert "أي تاريخ" in graph._unverified_availability_reply(desc, "ar")
    assert graph._unverified_availability_reply("some other check", "ar") == ""


def test_two_messages_from_one_patient_do_not_run_together():
    import app
    assert app._session_lock("s1") is app._session_lock("s1")
    assert app._session_lock("s1") is not app._session_lock("s2")


from langchain_core.messages import HumanMessage as _H


def test_lost_on_the_way_idioms_are_a_location_question():
    for text in ["ضيعت اللفه", "ضيعت اللفة", "في طريق ضيعت الدخله", "توهت", "تهت", "مو لاقي المستشفى", "ما لقيت الفرع"]:
        assert graph._says_lost_on_the_way([_H(content=text)]), text


def test_ordinary_messages_are_not_lost():
    for text in ["ابي احجز", "ضيعت موعدي", "يبغالي 10 دقائق", "وين الفرع", "تهتم بالاطفال"]:
        assert not graph._says_lost_on_the_way([_H(content=text)]), text


def test_a_link_request_after_the_address_still_sends_the_pin():
    """tanasuq-production 2026-10-07 10:19: 'ارسل رابط' after the branch address was refused."""
    from langchain_core.messages import AIMessage, HumanMessage
    msgs = [
        HumanMessage(content="وين فرع النزهة"),
        AIMessage(content="فرع النزهة عنوانه: 7282 أبي سفيان بن حرب، النزهة، الرياض."),
        HumanMessage(content="ارسل رابط"),
        AIMessage(content="", tool_calls=[{"name": "share_branch_location", "args": {"branch_name": "النزهة"}, "id": "t1"}]),
    ]
    state = {"messages": msgs, "session_id": "log-1007-pin", "client_id": "t", "understanding": {"asks_location": False}}
    assert tools.share_branch_location.func(state=state, branch_name="النزهة")["status"] == "location_requested"


def test_a_link_request_with_no_location_topic_is_still_refused():
    from langchain_core.messages import AIMessage, HumanMessage
    msgs = [
        HumanMessage(content="ابي احجز"),
        AIMessage(content="أي تخصص؟"),
        HumanMessage(content="ارسل رابط"),
        AIMessage(content="", tool_calls=[{"name": "share_branch_location", "args": {"branch_name": "النزهة"}, "id": "t1"}]),
    ]
    state = {"messages": msgs, "session_id": "log-1007-pin2", "client_id": "t", "understanding": {"asks_location": False}}
    assert tools.share_branch_location.func(state=state, branch_name="النزهة")["status"] != "location_requested"


# ----------------------------------------------------------------------
# 2026-10-07 10:52-10:56: reschedule went in a circle between branches
# ----------------------------------------------------------------------

def _reschedule_state(session_id):
    return {"session_id": session_id, "client_id": "t", "messages": [],
            "templates": {"_timezone": "Asia/Riyadh", "doctors_base_url": "https://x.test"}}


def test_a_full_requested_day_moves_to_the_next_same_weekday_with_slots():
    from datetime import datetime, timedelta
    from unittest.mock import patch
    from zoneinfo import ZoneInfo
    today = datetime.now(ZoneInfo("Asia/Riyadh")).date()
    asked = today + timedelta(days=7)            # full
    next_week = asked + timedelta(days=7)         # has slots
    other_day = asked + timedelta(days=3)         # different weekday - never offered

    def slots(day, hour):
        return {"slotStart": f"{day}T{hour}:00:00+03:00", "slotEnd": f"{day}T{hour}:30:00+03:00",
                "isBooked": False, "doctorName": "عمر المديفر", "branchName": "النزهة"}

    calls = []

    def fake(base_url, doctor_ids, from_date, to_date, is_booked=False, **kw):
        calls.append((from_date, to_date))
        if len(calls) == 1:
            items = []
        else:
            items = [slots(other_day, 16), slots(next_week, 16), slots(next_week, 17)]
        return {"success": True, "status_code": 200, "error": None, "data": {"items": items}}

    state = _reschedule_state("log-1007-rs")
    try:
        with patch("tools._resolve_doctor_id", return_value={"status": "found", "doctor_id": "D1"}), \
             patch("tools._doctors_base_url", return_value="https://x.test"), \
             patch("api.get_doctor_schedule_slots", side_effect=fake):
            shown = tools.get_available_reschedule_slots.func(
                state=state, ref_number="TNS-1",
                from_date=f"{asked}T00:00:00+03:00", to_date=f"{asked}T23:59:00+03:00")
        assert shown["status"] == "found"
        assert {s["slotStart"][:10] for s in shown["slots"]} == {next_week.isoformat()}
        assert len(calls) == 2
    finally:
        tools._BOOKING_SESSIONS.pop("log-1007-rs", None)


def test_the_next_weekday_tool_says_it_checks_no_availability():
    assert "does not check availability" in tools.get_next_weekday_date.description


def test_yes_to_our_transfer_question_is_a_handoff():
    from langchain_core.messages import AIMessage, HumanMessage
    offer = ("إذا كان استفسارك عن التوظيف أو التدريب، تقدر تراسل HR@x.sa 🌷\n"
             "ولأي استفسار ثاني، تحب أحوّلك لخدمة العملاء؟")
    msgs = [HumanMessage(content="احتاج شراكة"), AIMessage(content=offer), HumanMessage(content="اي")]
    assert graph._yes_to_our_transfer_question(msgs, {"intent": "answer", "confirms": False, "declines": False})
    # a refusal stays a refusal
    msgs_no = msgs[:-1] + [HumanMessage(content="لا")]
    assert not graph._yes_to_our_transfer_question(msgs_no, {"declines": True})


def test_the_greeting_menu_is_not_a_transfer_offer():
    from langchain_core.messages import AIMessage, HumanMessage
    menu = ("👤 التحدث مع أحد ممثلي خدمة العملاء، فقط اطلب مني ذلك وسأقوم بتحويلك\n\n"
            "كيف أستطيع مساعدتك اليوم؟ 😊")
    msgs = [HumanMessage(content="هلا"), AIMessage(content=menu), HumanMessage(content="اي")]
    assert not graph._yes_to_our_transfer_question(msgs, {"intent": "answer"})


# ----------------------------------------------------------------------
# 2026-10-07 11:29: a pasted reference with a stray space
# ----------------------------------------------------------------------

def test_a_pasted_reference_is_cleaned_before_the_lookup():
    assert tools._clean_booking_ref("Booking Reference: APT- CL01-20260930-594") == "APT-CL01-20260930-594"
    assert tools._clean_booking_ref("APT-CL01-20260930-594") == "APT-CL01-20260930-594"
    assert tools._clean_booking_ref("GuestBookingNum-2026-10-07-336") == "GuestBookingNum-2026-10-07-336"
    assert tools._clean_booking_ref("") == ""


def test_a_pasted_reference_does_not_switch_the_conversation_to_english():
    from langchain_core.messages import AIMessage, HumanMessage
    msgs = [HumanMessage(content="تاجيل موعد"), AIMessage(content="ممكن تعطيني رقم الحجز؟"),
            HumanMessage(content="Booking Reference: APT- CL01-20260930-594")]
    assert graph._detect_target_language(msgs) == "ar"
    # a real switch still switches
    msgs_en = msgs[:-1] + [HumanMessage(content="can you answer in English please")]
    assert graph._detect_target_language(msgs_en) == "en"
