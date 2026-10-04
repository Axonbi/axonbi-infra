"""
tanasuq-production, 2026-10-04 10:39-10:40:

  - "تعديل موعد" right after a specialty list stayed in booking, which
    asked a booking step ("نكمل تعديل موعدك على نفس رقم الواتساب هذا؟")
  - "لا" to "same WhatsApp number?" was read as a refusal: the patient got
    "تمام 🌷 إذا احتجت أي شيء ثاني" instead of being asked for the number
  - the correct verification code was "wrong": Authentica answers a right
    code with {"status": true}, which was never read
  - "جاري إرسال رمز التحقق" was shown while a code was being CHECKED
"""

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

import agents.semantic_router as sr
import api
import graph
import progress


# ----------------------------------------------------------------------
# "تعديل موعد" is a reschedule
# ----------------------------------------------------------------------

def test_a_bare_reschedule_request_during_an_early_booking_goes_to_reschedule():
    facts = sr.TurnFacts(previous="booking")
    reading = {"intent": "reschedule", "answer_to_previous_question": True}
    assert sr.decide(reading, facts).agent == "reschedule"


def test_a_value_for_the_booking_still_stays_in_booking():
    facts = sr.TurnFacts(previous="booking")
    reading = {"intent": "reschedule", "answer_to_previous_question": True, "entities_present": ["time"]}
    assert sr.decide(reading, facts).agent == "booking"


# ----------------------------------------------------------------------
# "لا" to "same WhatsApp number?"
# ----------------------------------------------------------------------

def _messages(ai_text):
    return [HumanMessage(content="تعديل موعد"), AIMessage(content=ai_text), HumanMessage(content="لا")]


def test_no_to_the_same_number_question_is_an_answer():
    messages = _messages("نكمل تعديل موعدك على نفس رقم الواتساب هذا؟ ✅")
    assert graph._assistant_asked_an_optional_question(messages)
    facts = sr.TurnFacts(previous="booking", optional_question=True, bare_refusal=True)
    reading = {"intent": "answer", "declines": True, "answer_to_previous_question": True}
    assert sr.turn_action(reading, facts) == sr.ACTION_ANSWER


def test_no_to_the_booking_same_number_question_is_an_answer_too():
    assert graph._assistant_asked_an_optional_question(_messages("نكمل الحجز على نفس رقم الواتساب ده؟ ✅"))


def test_no_to_another_doctor_is_still_a_refusal():
    assert not graph._assistant_asked_an_optional_question(_messages("تحب أساعدك تحجز مع دكتور ثاني؟"))


# ----------------------------------------------------------------------
# Authentica: a correct code
# ----------------------------------------------------------------------

class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body


def _verify(status, body):
    with patch.object(api, "_request_with_retry", lambda *a, **k: (_Resp(status, body), False, None)):
        return api.authentica_verify_otp("+201155611045", "4017")


def test_a_correct_code_is_accepted():
    assert _verify(200, {"status": True, "message": "OTP verified"})["success"] is True


def test_a_wrong_code_is_refused():
    assert _verify(422, {"status": False, "message": "Failed to verify OTP"})["success"] is False
    assert _verify(200, {"status": False})["success"] is False


# ----------------------------------------------------------------------
# Checking a code says so
# ----------------------------------------------------------------------

def test_checking_a_code_has_its_own_interim_message():
    assert progress._GROUP_FOR_TOOL["verify_otp"] == "verifying_otp"
    assert progress._GROUP_FOR_TOOL["send_otp"] == "sending_otp"
    assert "verifying_otp" in progress._GROUP_PRIORITY
