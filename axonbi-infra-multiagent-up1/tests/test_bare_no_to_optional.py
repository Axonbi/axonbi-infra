"""
2026-10-12 09:30: "لا" to "تحب تضيف بريدك الإلكتروني؟ (اختياري)" came back from
the understanding with entities taken from the conversation (branch, date,
doctor, time). The optional-question rule read that as a new request, the
turn became a refusal, confirm_booking_review was blocked and the patient got
"ما عندي مشكلة. هل تبي تساعدك في شي ثاني". A bare "لا" asks for nothing.
"""

import agents.semantic_router as sr

_READING_0930 = {
    "intent": "booking", "confidence": 1.0, "answer_to_previous_question": True,
    "confirms": False, "declines": True,
    "entities_present": ["branch", "date", "doctor", "time"],
    "entities": {"branch": "المنار", "date": "14/10/2026", "doctor": "د. محمد", "time": "4:30"},
}


def test_bare_no_to_the_email_question_skips_the_email():
    facts = sr.TurnFacts(previous="booking", optional_question=True, bare_refusal=True)
    assert sr.turn_action(_READING_0930, facts) == sr.ACTION_ANSWER


def test_a_no_with_a_real_request_after_the_email_question_is_still_that_request():
    facts = sr.TurnFacts(previous="booking", optional_question=True, bare_refusal=False)
    reading = {**_READING_0930, "changes_intent": True, "intent": "cancel"}
    assert sr.turn_action(reading, facts) != sr.ACTION_ANSWER


def test_bare_no_to_a_real_question_is_still_a_refusal():
    facts = sr.TurnFacts(previous="booking", optional_question=False, bare_refusal=True)
    assert sr.turn_action(_READING_0930, facts) == sr.ACTION_DECLINE
