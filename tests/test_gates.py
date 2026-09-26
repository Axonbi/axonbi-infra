"""
Unit tests for the decision points the production failures went
through: the understanding parser, the handoff and cancel gates (with
and without an understanding reading), doctor-vs-specialty extraction,
and routing from a reading. No graph run, no network.
"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import graph
import tools
import understanding
from conftest import tool_call


def _handoff(state, agreed=True):
    return tools.request_human_handoff.func(state=state, reason="test", patient_agreed=agreed)


def _cancel(state):
    return tools.cancel_appointment.func(state=state, booking_id="GUID-1")


# ----------------------------------------------------------------------
# understanding.py
# ----------------------------------------------------------------------

def test_last_visible_reply_skips_the_tool_call_message():
    messages = [
        HumanMessage(content="استفسار"),
        AIMessage(content="هل تود أن أحولك لأحد ممثلي خدمة العملاء؟"),
        HumanMessage(content="نعم"),
        tool_call("request_human_handoff", {"reason": "x", "patient_agreed": True}),
    ]
    assert "خدمة العملاء" in understanding.last_ai_text_before_latest_human(messages)


@pytest.mark.parametrize("raw", [
    '{"intent": "human"}',
    '```json\n{"intent": "human", "wants_human": "true"}\n```',
    'sure: {"intent":"human","wants_human":true,"doctor_name":"null"}',
])
def test_parser_is_tolerant(raw):
    result = understanding._parse(raw)
    assert result["intent"] == "human"
    assert result["wants_human"] is True
    assert result["doctor_name"] is None


def test_parser_rejects_garbage_and_unknown_intents():
    assert understanding._parse("no json here") is None
    assert understanding._parse('{"intent": "teleport"}')["intent"] == "other"


def test_understanding_never_raises():
    class Broken:
        def invoke(self, *a, **k):
            raise RuntimeError("boom")
    assert understanding.understand_turn([HumanMessage(content="hi")], Broken()) is None
    assert understanding.understand_turn([HumanMessage(content="hi")], None) is None


def test_prompt_carries_the_previous_question():
    seen = {}

    class Capture:
        def invoke(self, messages, *a, **k):
            seen["prompt"] = messages[-1].content
            return AIMessage(content="{}")

    understanding.understand_turn(
        [AIMessage(content="هل تود أن أحولك؟"), HumanMessage(content="اي")], Capture(),
    )
    assert "ASSISTANT: هل تود أن أحولك؟" in seen["prompt"]
    assert seen["prompt"].rstrip().endswith("اي")


# ----------------------------------------------------------------------
# Handoff gate
# ----------------------------------------------------------------------

OFFER = [HumanMessage(content="استفسار"), AIMessage(content="هل تود أن أرسل طلب تحويلك إلى أحد ممثلي خدمة العملاء الآن؟")]


@pytest.mark.parametrize("yes", ["نعم", "اي", "يفضل", "yes", "confirm"])
def test_fallback_gate_accepts_yes_after_an_offer(yes):
    state = {"messages": OFFER + [HumanMessage(content=yes),
                                  tool_call("request_human_handoff", {"reason": "x", "patient_agreed": True})]}
    assert _handoff(state)["status"] == "handoff_requested"


def test_fallback_gate_still_blocks_yes_with_no_offer():
    state = {"messages": [AIMessage(content="تحب أحجز لك موعد؟"), HumanMessage(content="نعم"),
                          tool_call("request_human_handoff", {"reason": "x", "patient_agreed": True})]}
    assert _handoff(state)["status"] == "not_requested"


def test_fallback_gate_still_blocks_a_bare_complaint_word():
    state = {"messages": [HumanMessage(content="شكوى")]}
    assert _handoff(state)["status"] == "not_requested"


def test_understanding_decides_the_handoff():
    base = {"messages": [HumanMessage(content="ابغى احد يساعدني")]}
    assert _handoff({**base, "understanding": {"wants_human": True}}, agreed=False)["status"] == "handoff_requested"
    assert _handoff({**base, "understanding": {"wants_human": False}})["status"] == "not_requested"


def test_crisis_always_gets_a_person():
    state = {"messages": [HumanMessage(content="ok")], "crisis_active": True,
             "understanding": {"wants_human": False}}
    assert _handoff(state, agreed=False)["status"] == "handoff_requested"


# ----------------------------------------------------------------------
# Cancel gate
# ----------------------------------------------------------------------

def _booked_then(text, old_cancel=True):
    msgs = []
    if old_cancel:
        msgs += [HumanMessage(content="ابي الغاء موعدي"), AIMessage(content="تم")]
    msgs += [
        HumanMessage(content="اود اخذ موعد مع الدكتور عمر"),
        ToolMessage(content=json.dumps({"status": "success"}), name="create_new_booking", tool_call_id="b1"),
        AIMessage(content="تم الحجز بنجاح ✅ الطبيب: عمر المديفر. إذا رغبت في حجز موعد جديد أو تعديل موعد آخر يسعدنا خدمتك"),
        HumanMessage(content=text),
    ]
    return {"messages": msgs, "session_id": "unit"}


def test_fallback_old_cancel_word_does_not_unlock_cancelling_a_new_booking():
    assert tools._patient_asked_to_cancel(_booked_then("تم تاكيد الموعد مسبقا")) is False


def test_fallback_yes_to_a_cancel_question_still_counts():
    state = {"messages": [
        HumanMessage(content="ابي الغي موعدي"),
        AIMessage(content="هل تريد إلغاء موعدك مع د. عمر يوم الأربعاء؟"),
        HumanMessage(content="نعم"),
    ]}
    assert tools._patient_asked_to_cancel(state) is True


def test_fallback_yes_to_a_different_question_does_not_count():
    state = {"messages": [
        HumanMessage(content="ابي الغي موعدي"),
        AIMessage(content="لقيت موعدك. تحب أحجز لك موعد ثاني بعد؟"),
        HumanMessage(content="نعم"),
    ]}
    assert tools._patient_asked_to_cancel(state) is False


def test_understanding_blocks_cancel_without_clear_confirmation():
    state = {**_booked_then("تم تاكيد الموعد مسبقا"), "understanding": {"cancel_confirmed": False}}
    assert _cancel(state)["status"] == "not_confirmed"


# ----------------------------------------------------------------------
# Doctor name vs specialty
# ----------------------------------------------------------------------

@pytest.mark.parametrize("text", ["كم موعد عند الدكتور نفسي", "ابي دكتور نفسي", "دكتور عيون"])
def test_specialty_after_doctor_word_is_not_a_name(text):
    assert graph._doctor_fragment(text) == ""


def test_real_names_still_extracted():
    assert graph._doctor_fragment("ابي موعد مع الدكتور عمر المديفر") == "عمر المديفر"


def test_reading_wins_over_the_regex():
    assert graph._doctor_fragment("اود اخذ موعد مع عمر", {"doctor_name": "عمر"}) == "عمر"
    assert graph._doctor_fragment("عند الدكتور نفسي", {"doctor_name": None}) == ""


# ----------------------------------------------------------------------
# Routing from a reading
# ----------------------------------------------------------------------

def test_answer_stays_with_the_owner():
    assert graph._route_from_reading({"intent": "answer"}, "booking")[0] == "booking"


def test_price_question_goes_to_faq_unless_booking():
    assert graph._route_from_reading({"intent": "faq", "asks_price": True}, None)[0] == "faq"
    assert graph._route_from_reading({"intent": "booking", "asks_price": True}, "booking")[0] == "booking"


def test_no_reading_defers_to_the_deterministic_router():
    # Only a TECHNICAL failure (no reading) hands the turn to the cues.
    assert graph._route_from_reading(None, "booking") is None


def test_a_reading_that_names_no_flow_is_not_second_guessed_by_the_cues():
    # "other" is a real reading: the active flow keeps the turn, and with
    # nothing active it is the concierge's - never a keyword score.
    assert graph._route_from_reading({"intent": "other"}, "booking")[0] == "booking"
    assert graph._route_from_reading({"intent": "other"}, None)[0] == "concierge"
