"""
tanasuq-production, 2026-10-02/03:

  - "لا" to the optional email question ended the booking
  - a dropped question took its apology with it, or left "أو ..." /
    "وإذا نعم، ..." / "😊 ..." as the whole reply
  - "تعديل للوصفه" read as a reschedule; "موضف" not read as a person
  - a doctor's only branch, filled in automatically, then hid the next
    doctor the patient named (أمل الدوسري searched at المنار only)
  - a doctor with no rota made a listed branch "not exist"
  - a review card went out with no branch line
"""

from langchain_core.messages import AIMessage, HumanMessage

import agents.semantic_router as sr
import graph
import tools
import understanding
from tool_result_guidance import guidance_for


EMAIL_QUESTION = "تحب تضيف بريدك الإلكتروني؟ (اختياري)"


def _messages(ai_text, human_text):
    return [HumanMessage(content="ابي احجز"), AIMessage(content=ai_text), HumanMessage(content=human_text)]


# ----------------------------------------------------------------------
# "لا" to the optional email is an answer
# ----------------------------------------------------------------------

def test_the_email_question_is_seen_as_optional():
    assert graph._assistant_asked_an_optional_question(_messages(EMAIL_QUESTION, "لا"))
    assert not graph._assistant_asked_an_optional_question(
        _messages("تحب أساعدك تحجز مع دكتور ثاني؟", "لا"))


def test_no_to_the_review_card_is_still_a_refusal_even_though_it_lists_an_email():
    card = ("يرجى مراجعة بيانات الحجز:\n🏥 الفرع: المنار\n📧 البريد الإلكتروني: a@b.com\n\n"
            "✅ هل جميع البيانات صحيحة وتود تأكيد الحجز؟")
    assert not graph._assistant_asked_an_optional_question(_messages(card, "لا"))


def test_no_to_the_optional_email_moves_the_booking_on():
    facts = sr.TurnFacts(previous="booking", optional_question=True, bare_refusal=True)
    reading = {"intent": "answer", "declines": True, "answer_to_previous_question": True}
    assert sr.turn_action(reading, facts) == sr.ACTION_ANSWER
    assert sr.turn_action(None, facts) == sr.ACTION_ANSWER


def test_no_elsewhere_is_still_a_decline():
    facts = sr.TurnFacts(previous="booking", bare_refusal=True)
    reading = {"intent": "answer", "declines": True}
    assert sr.turn_action(reading, facts) == sr.ACTION_DECLINE


def test_no_plus_a_new_request_on_the_email_step_is_still_that_request():
    facts = sr.TurnFacts(previous="booking", optional_question=True)
    reading = {"intent": "human", "declines": True, "wants_human": True}
    assert sr.turn_action(reading, facts) == sr.ACTION_DECLINE_AND_REQUEST


def test_no_negation_directive_for_no_to_the_email():
    messages = _messages(EMAIL_QUESTION, "لا")
    reading = {"intent": "answer", "declines": True}
    assert graph._build_negation_directive(messages, reading, sr.ACTION_ANSWER) == ""


# ----------------------------------------------------------------------
# Trimming a second question leaves a whole reply
# ----------------------------------------------------------------------

def test_the_apology_before_a_dropped_question_stays():
    text = ("آسفة، ما قدرنا نرسل رمز التحقق على الرقم اللي أعطيتني إياه الحين. "
            "ممكن نكمل الحجز على نفس رقم الواتساب اللي تتواصل منه؟ أو تحب أحولك لخدمة العملاء؟")
    trimmed, removed = graph._strip_extra_questions(text, {})
    assert removed == 1
    assert trimmed.startswith("آسفة، ما قدرنا نرسل رمز التحقق")
    assert "أو تحب" not in trimmed and "تحب أحولك لخدمة العملاء؟" in trimmed


def test_no_reply_starts_with_a_leftover_connector_or_emoji():
    for text in (
        "هل تقصد حجز موعد لزيارة مريض في المستشفى؟ وإذا نعم، تفضل أخبرني اسم الطبيب أو التخصص اللي تبيه؟",
        "كيف أقدر أخدمك اليوم؟ 😊 هل تبي تعرف عن خدمات المستشفى أو حجز موعد أو شيء ثاني؟",
    ):
        trimmed, _ = graph._strip_extra_questions(text, {})
        assert not trimmed.startswith(("و", "أو", "😊")), trimmed


def test_the_first_question_kept_is_unchanged():
    text = "الله يحفظها ويطول بعمرها 🌷 كيف حالها؟ وش المشكلة الصحية أو العرض اللي تحس فيه؟"
    assert graph._strip_extra_questions(text, {})[0] == "الله يحفظها ويطول بعمرها 🌷 كيف حالها؟"


# ----------------------------------------------------------------------
# Understanding: prescriptions and "موظف"
# ----------------------------------------------------------------------

def test_the_understanding_prompt_covers_prescriptions_and_staff():
    assert "تعديل للوصفه" in understanding.PROMPT
    assert "موضف" in understanding.PROMPT
    assert "(اختياري)" in understanding.PROMPT


# ----------------------------------------------------------------------
# A branch filled in automatically does not hide the next doctor
# ----------------------------------------------------------------------

def test_the_auto_confirm_path_sets_the_inferred_flag_in_source():
    import inspect
    source = inspect.getsource(tools.get_doctor_schedule_for_booking.func)
    at = source.index('session["branch_id"] = only_branch_id')
    assert 'session["branch_auto_resolved"] = True' in source[at:at + 800]


# ----------------------------------------------------------------------
# A doctor with no rota does not make a branch disappear
# ----------------------------------------------------------------------

def test_doctor_not_scheduled_guidance_never_denies_the_branch():
    text = guidance_for("match_entity_for_booking", {"matched": False, "status": "doctor_not_scheduled"})
    assert text and "Never say the branch does not exist" in text


# ----------------------------------------------------------------------
# The review card names the branch
# ----------------------------------------------------------------------

def _card_state(sid):
    session = tools._get_booking_session(sid)
    session.update(doctor_id="d1", branch_id="b1", branch_display_name="المنار")
    return {"session_id": sid, "templates": {}}


def test_a_review_card_without_a_branch_gets_the_branch_on_file():
    sid = "log1003-card"
    state = _card_state(sid)
    card = ("يرجى مراجعة بيانات الحجز:\n👨‍⚕️ الطبيب: رغد المهيلب\n📅 التاريخ: الاثنين 05/10/2026\n"
            "🕐 الوقت: 3:00 مساءً\n\n✅ هل جميع البيانات صحيحة وتود تأكيد الحجز؟")
    try:
        fixed = graph._review_card_branch_fix(card, state, "ar")
        lines = fixed.split("\n")
        assert lines[1] == "🏥 الفرع: المنار"
        assert lines[2].startswith("👨")
        assert graph._review_card_branch_fix(fixed, state, "ar") == fixed
    finally:
        tools._BOOKING_SESSIONS.pop(sid, None)


def test_a_reply_that_is_not_a_card_is_untouched():
    sid = "log1003-nocard"
    state = _card_state(sid)
    try:
        text = "مواعيد د. رغد المهيلب:\n• الاثنين\nأي يوم يناسبك؟"
        assert graph._review_card_branch_fix(text, state, "ar") == text
    finally:
        tools._BOOKING_SESSIONS.pop(sid, None)
