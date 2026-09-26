"""
Semantic-first routing.

WHAT THESE TESTS PROVE, AND WHAT THEY CANNOT
--------------------------------------------
The understanding model is faked here, exactly as in the other suites:
each phrase is mapped to the reading the real model is expected to give.
So these tests prove the WIRING:

  - the router routes on MEANING (the reading) and on conversation state,
    never on keywords: every test in this file runs with the keyword cue
    lists switched OFF (`no_cues`), so nothing here can pass because a
    phrase happened to be in agents/router._CUES;
  - the understanding model is actually SHOWN what it needs - the
    assistant's previous question and the current flow - so a short reply
    can be read in context;
  - the same word routes differently under different questions;
  - intent changes win over stickiness; answers keep the flow;
  - uncertainty leads to context or one clarification question, never a
    dead end, and is not treated as a technical failure;
  - the hard rules (crisis, consent integrity, tool capability) still
    hold, and are the only overrides.

Whether the REAL model reads these phrasings - and paraphrases it has
never seen - correctly is measured by evals/run_understanding_eval.py
over evals/understanding_cases.json, which carries every phrase below.
"""

import json
import logging
import re

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agents
import graph
import tools
import understanding
from conftest import send, state_of

H, A = HumanMessage, AIMessage


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_cues(monkeypatch):
    """The keyword cue lists, switched off for every test in this file."""
    monkeypatch.setattr(agents.router, "_COMPILED", {})


def _question_of(prompt: str) -> str:
    """The last ASSISTANT line the understanding prompt shows."""
    lines = [l for l in prompt.split("\n") if l.startswith("ASSISTANT: ")]
    return lines[-1][len("ASSISTANT: "):] if lines else ""


class ContextReader:
    """Fake understanding model keyed on (previous question, message) -
    the pair the real model is asked to interpret together. Records every
    prompt so a test can check what the model was shown."""

    def __init__(self):
        self.table = {}
        self.prompts = []
        self.fail = False
        self.garbage = False

    def add(self, message, reading, question=None):
        self.table[(question, message)] = reading

    def invoke(self, messages, *a, **k):
        prompt = str(messages[-1].content)
        self.prompts.append(prompt)
        if self.fail:
            raise TimeoutError("understanding unavailable (simulated)")
        if self.garbage:
            return AIMessage(content="I think the patient wants to cancel.")
        message = prompt.rsplit("THE PATIENT'S NEW MESSAGE:", 1)[-1].strip()
        question = _question_of(prompt)
        reading = self.table.get((question, message), self.table.get((None, message), {}))
        return AIMessage(content=json.dumps(reading, ensure_ascii=False))


@pytest.fixture
def ctx_reader(monkeypatch):
    fake = ContextReader()
    monkeypatch.setattr(graph, "_understanding_llm", fake, raising=False)
    return fake


def R(intent, **fields):
    """A reading as the real model returns it."""
    reading = {"intent": intent, "confidence": 0.9, "is_ambiguous": False,
               "answer_to_previous_question": False, "changes_intent": False}
    reading.update(fields)
    return reading


def _route(messages, previous=None, session_id="semantic-unit", crisis_active=False):
    return graph.router({"messages": messages, "session_id": session_id,
                         "active_agent": previous, "crisis_active": crisis_active})


def _selected(update):
    if update["handoff_now"]:
        return "handoff"
    if update["clarify_now"]:
        return "clarify"
    return update["active_agent"]


# ----------------------------------------------------------------------
# 1. Routing matrix - 10+ phrasings per intent, no conversation before it
#    Egyptian, Gulf, Levantine, MSA, English, Arabizi, typos.
# ----------------------------------------------------------------------

MATRIX = {
    "cancel": [
        "الغيه", "مش هقدر أجي", "الموعد ده مش مناسب", "خلاص بلاش الموعد",
        "مافيش داعي للموعد", "خليه يتلغي", "ما ابي الموعد اللي بكرا",
        "مش عايزة الموعد خلاص", "بدي بطّل الموعد", "cancel my appointment pls",
        "kansel el ma3ad", "ما رح اقدر احضر، شيلوا الحجز",
    ],
    "reschedule": [
        "ممكن نخليه يوم تاني", "عايزه أغير الموعد", "ينفع الأسبوع الجاي؟",
        "الموعد ده مش مناسب خلينا نأجله", "ممكن ننقله؟", "خليه بعدين",
        "ابي اقدم موعدي", "بدي أجّل الموعد لبعد العيد", "can we move it to thursday",
        "momken n2agel el ma3ad", "الوقت ما يناسبني، في وقت ثاني؟",
    ],
    "booking": [
        "عايز أشوف دكتور", "محتاج أكشف", "ممكن تحجزلي", "نفسي أحجز", "عايز موعد",
        "ابغى موعد عند دكتور عيون", "بدي احجز عند الدكتور", "i need an appointment",
        "3ayez a7gez", "فيه مواعيد بكرة؟", "الدكتور ده عنده إمتى؟",
    ],
    "medical": [
        "حاسس بحاجة غريبة في بطني", "رجلي اتخبطت", "عيني فيها حاجة",
        "مش عارف أروح لأي دكتور", "حاسس إن في حاجة مش طبيعية", "عندي صداع من يومين",
        "راسي يعورني", "بطني عم يوجعني", "my knee has been swelling", "3andi 7ararah",
        "عايز أشوف حد بخصوص عيني",
    ],
    "faq": [
        "الفروع فين", "بتفتحوا الساعة كام", "تقبلون تأمين بوبا؟", "عندكم زيارات منزلية؟",
        "وين موقعكم", "شو الخدمات اللي عندكم", "do you have parking", "fe 3ndko ashe3a?",
        "ايش التخصصات المتوفرة", "هل يوجد قسم للأطفال",
    ],
    "complaint": [
        "عايز أشتكي على الخدمة", "الاستقبال عاملوني وحش", "عندي ملاحظة على الدكتور",
        "بدي قدم شكوى", "الخدمة سيئة جدا ومحد رد علي", "ابي ارفع بلاغ عن موظف",
        "I want to file a complaint", "3ayez a3mel shakwa", "انتظرت ساعتين ومحدش سأل فيا",
        "عندي اقتراح لتحسين الخدمة",
    ],
    "human": [
        "عايز أكلم حد", "خليني مع موظف", "ممكن حد من الفريق يساعدني", "عايز إنسان يكلمني",
        "حولني لخدمة العملاء", "ابي اكلم احد", "بدي احكي مع حدا", "I'd like an actual human on this chat",
        "3ayez akalem 7ad", "وصلني بأحد من الموظفين",
    ],
    "greeting": [
        "السلام عليكم", "صباح الخير", "هلا", "مرحبا", "heyy", "good evening",
        "مساء النور", "اهلين", "يعطيكم العافية", "hello there",
    ],
}

EXPECTED_OWNER = {"human": "handoff", "greeting": "concierge"}
MATRIX_CASES = [(intent, phrase) for intent, phrases in MATRIX.items() for phrase in phrases]


def test_every_intent_has_at_least_ten_phrasings():
    for intent, phrases in MATRIX.items():
        assert len(set(phrases)) >= 10, intent


@pytest.mark.parametrize("intent,phrase", MATRIX_CASES, ids=[f"{i}:{p}" for i, p in MATRIX_CASES])
def test_routing_matrix_follows_meaning_not_keywords(ctx_reader, intent, phrase):
    ctx_reader.add(phrase, R(intent))
    update = _route([H(content=phrase)])
    assert _selected(update) == EXPECTED_OWNER.get(intent, intent)


@pytest.mark.parametrize("intent,phrase", MATRIX_CASES, ids=[f"{i}:{p}" for i, p in MATRIX_CASES])
def test_matrix_phrases_are_not_taught_to_the_model_verbatim(intent, phrase):
    """The prompt teaches concepts, not a phrase list: none of the
    matrix phrasings appears in it word for word (the handful of
    illustrative words the prompt does use are all shorter than these)."""
    assert phrase not in understanding.PROMPT


# ----------------------------------------------------------------------
# 2. Unseen paraphrases - not in the prompt, not in any cue list.
#    The reading is what the model is expected to give; the router must
#    act on it, including when it is honestly unsure.
# ----------------------------------------------------------------------

UNSEEN = [
    # (message, previous question, active flow, reading, expected owner)
    ("مش شايف إن الموعد ده هينفعني", None, None,
     R("cancel", is_ambiguous=True, confidence=0.55, alternatives=["cancel", "reschedule"]), "clarify"),
    ("مش شايف إن الموعد ده هينفعني", "لقيت موعدك مع د. عمر يوم الخميس. تحب تلغيه ولا تعدله؟", "cancel",
     R("cancel", is_ambiguous=True, confidence=0.55, alternatives=["cancel", "reschedule"],
       answer_to_previous_question=True), "cancel"),
    ("الظروف اتغيرت ومش هعرف أوصل ساعتها", None, None, R("cancel", cancel_request=True), "cancel"),
    ("لو في حاجة بعد الضهر يبقى أحسن", "موعدك الحالي الأحد 10 الصبح. تحب نغيره؟", "reschedule",
     R("reschedule", answer_to_previous_question=True), "reschedule"),
    ("في حد يقدر يشوف الشامة اللي ظهرت في رقبتي؟", None, None, R("medical"), "medical"),
    ("الموظفة ردت علي بأسلوب مش كويس", None, None, R("complaint"), "complaint"),
    ("كم يكلف تنظيف الأسنان عندكم", None, None, R("faq", asks_price=True), "faq"),
    ("محتاج أتكلم مع بني آدم مش روبوت", None, None, R("human"), "handoff"),
]


@pytest.mark.parametrize("message,question,flow,reading,expected", UNSEEN,
                         ids=[f"{m}|{f}" for m, _q, f, _r, _e in UNSEEN])
def test_unseen_paraphrases_route_on_meaning(ctx_reader, message, question, flow, reading, expected):
    assert message not in understanding.PROMPT
    ctx_reader.add(message, reading, question)
    history = [H(content="مرحبا"), A(content=question)] if question else []
    update = _route(history + [H(content=message)], previous=flow)
    assert _selected(update) == expected


def test_the_unsure_paraphrase_is_offered_exactly_its_candidates(session_id, llm, ctx_reader):
    msg = "مش شايف إن الموعد ده هينفعني"
    ctx_reader.add(msg, R("cancel", is_ambiguous=True, confidence=0.5, alternatives=["cancel", "reschedule"]))
    result = send(session_id, msg)
    # Exactly the two candidates, as one natural question.
    assert "تحب نلغي الموعد، ولا نأجله ليوم تاني؟" in result["reply"]
    assert len(llm.calls) == 0, "a clarification is written in code - no specialist call"


# ----------------------------------------------------------------------
# 3. Short answers read from the question they answer
# ----------------------------------------------------------------------

CANCEL_Q = "لقيت موعدك مع د. عمر يوم الخميس الساعة 4 مساءً. هل تريد إلغاء الموعد؟"
BOOK_OFFER = "هل تريد حجز موعد مع د. عمر؟"
HANDOFF_OFFER = "هل تريد التواصل مع خدمة العملاء؟"
DOCTOR_LIST = "دول الدكاترة المتاحين في العيون: 1️⃣ د. عمر 2️⃣ د. سارة 3️⃣ د. أحمد. مين يناسبك؟"
SLOTS_Q = "المواعيد المتاحة يوم الخميس: 1️⃣ 4:00 مساءً 2️⃣ 5:00 مساءً 3️⃣ 6:00 مساءً. اختار رقم الموعد"
DAY_Q = "أي يوم يناسبك؟"
BRANCH_Q = "أي فرع يناسبك؟"
DOCTOR_Q = "مع أي دكتور تحب تحجز؟"
PHONE_Q = "ممكن رقم الجوال المسجل في الحجز؟"

CONTEXTUAL = [
    # (message, question, flow, reading, expected owner)
    ("أكيد", CANCEL_Q, "cancel", R("cancel", answer_to_previous_question=True, cancel_confirmed=True), "cancel"),
    ("أكيد", BOOK_OFFER, "medical", R("booking", answer_to_previous_question=True), "booking"),
    ("أكيد", HANDOFF_OFFER, "faq", R("human", answer_to_previous_question=True, wants_human=True), "handoff"),
    ("تمام", CANCEL_Q, "cancel", R("cancel", answer_to_previous_question=True, cancel_confirmed=True), "cancel"),
    ("تمام", BOOK_OFFER, "medical", R("booking", answer_to_previous_question=True), "booking"),
    ("تمام", HANDOFF_OFFER, "faq", R("human", answer_to_previous_question=True, wants_human=True), "handoff"),
    ("أيوه", BOOK_OFFER, "medical", R("booking", answer_to_previous_question=True), "booking"),
    ("أيوه", HANDOFF_OFFER, "faq", R("human", answer_to_previous_question=True, wants_human=True), "handoff"),
    ("نعم", CANCEL_Q, "cancel", R("cancel", answer_to_previous_question=True, cancel_confirmed=True), "cancel"),
    ("لا", CANCEL_Q, "cancel", R("cancel", answer_to_previous_question=True), "cancel"),
    ("لا", HANDOFF_OFFER, "faq", R("answer", answer_to_previous_question=True), "faq"),
    ("الثاني", DOCTOR_LIST, "booking", R("booking", answer_to_previous_question=True), "booking"),
    ("3", DOCTOR_LIST, "booking", R("booking", answer_to_previous_question=True), "booking"),
    ("3", SLOTS_Q, "booking", R("booking", answer_to_previous_question=True), "booking"),
    ("الخميس", DAY_Q, "booking", R("booking", answer_to_previous_question=True, entities={"date": "الخميس"}), "booking"),
    ("الخميس", DAY_Q, "reschedule", R("reschedule", answer_to_previous_question=True), "reschedule"),
    ("بكرة", DAY_Q, "booking", R("booking", answer_to_previous_question=True, entities={"date": "بكرة"}), "booking"),
    ("مدينة نصر", BRANCH_Q, "booking", R("booking", answer_to_previous_question=True, entities={"branch": "مدينة نصر"}), "booking"),
    ("عمر", DOCTOR_Q, "booking", R("booking", answer_to_previous_question=True, doctor_name="عمر"), "booking"),
    ("01123456789", PHONE_Q, "cancel", R("answer", entities={"phone": "01123456789"}), "cancel"),
]


@pytest.mark.parametrize("message,question,flow,reading,expected", CONTEXTUAL,
                         ids=[f"{m}|{f}|{q[:18]}" for m, q, f, _r, _e in CONTEXTUAL])
def test_short_answers_take_their_meaning_from_the_question(ctx_reader, message, question, flow, reading, expected):
    ctx_reader.add(message, reading, question)
    messages = [H(content="مرحبا"), A(content=question), H(content=message)]
    update = _route(messages, previous=flow)

    # The model was SHOWN the question and the flow it belongs to...
    prompt = ctx_reader.prompts[-1]
    assert _question_of(prompt) == question
    assert f'"flow":"{flow}"' in prompt
    # ...and the router acted on what the answer means there.
    assert _selected(update) == expected


def test_the_same_word_means_three_different_things(ctx_reader):
    outcomes = {}
    for question, flow, reading in (
        (CANCEL_Q, "cancel", R("cancel", answer_to_previous_question=True, cancel_confirmed=True)),
        (BOOK_OFFER, "medical", R("booking", answer_to_previous_question=True)),
        (HANDOFF_OFFER, "faq", R("human", answer_to_previous_question=True, wants_human=True)),
    ):
        ctx_reader.add("أكيد", reading, question)
        update = _route([H(content="مرحبا"), A(content=question), H(content="أكيد")], previous=flow)
        outcomes[flow] = (_selected(update), update["understanding"].get("cancel_confirmed"))
    assert outcomes == {"cancel": ("cancel", True), "medical": ("booking", False), "faq": ("handoff", False)}


def test_a_list_pick_shown_by_an_agent_that_cannot_book_goes_to_booking(ctx_reader):
    """TOOL-CAPABILITY INVARIANT, from state: medical listed doctors (the
    booking session says a doctor list is on screen) and the reading only
    knows it is an answer."""
    sid = "semantic-medical-pick"
    tools._BOOKING_SESSIONS[sid] = {"last_list": {"entity_type": "doctor", "items": []}}
    try:
        ctx_reader.add("3", R("answer"), DOCTOR_LIST)
        update = _route([H(content="رجلي وجعاني"), A(content=DOCTOR_LIST), H(content="3")],
                        previous="medical", session_id=sid)
        assert update["active_agent"] == "booking"
        assert "no booking tools" in update["routing_reason"]
    finally:
        tools._BOOKING_SESSIONS.pop(sid, None)


def test_the_answer_that_moves_medical_into_booking_keeps_the_doctor_on_file(ctx_reader):
    """A continuation is not an abandoned booking: the doctor medical just
    offered must survive the switch (it was wiped before on "تمام")."""
    sid = "semantic-continuation"
    tools._BOOKING_SESSIONS[sid] = {"doctor_id": "D-OMAR", "specialty_ids": ["S-EYE"]}
    try:
        ctx_reader.add("تمام", R("booking", answer_to_previous_question=True), BOOK_OFFER)
        update = _route([H(content="عيني فيها حاجة"), A(content=BOOK_OFFER), H(content="تمام")],
                        previous="medical", session_id=sid)
        assert update["active_agent"] == "booking"
        assert tools._BOOKING_SESSIONS[sid].get("doctor_id") == "D-OMAR"
    finally:
        tools._BOOKING_SESSIONS.pop(sid, None)


# ----------------------------------------------------------------------
# 4. Change of intent beats stickiness; an answer does not
# ----------------------------------------------------------------------

CHANGES = [
    ("booking", "cancel", "طيب عايز ألغي الموعد اللي عندي"),
    ("booking", "reschedule", "ممكن أأجل الموعد ده؟"),
    ("booking", "faq", "استنى، انتو بتقبلوا تأمين؟"),
    ("medical", "booking", "خلاص احجزلي"),
    ("medical", "faq", "طب الفرع ده بيفتح امتى؟"),
    ("faq", "complaint", "عايز أشتكي على الخدمة"),
    ("complaint", "booking", "سيبك من الشكوى، عايز أحجز"),
    ("reschedule", "cancel", "لا خلاص الغيه خالص"),
    ("cancel", "reschedule", "بلاش إلغاء، خليه يوم تاني"),
]


@pytest.mark.parametrize("flow,new,message", CHANGES, ids=[f"{f}->{n}" for f, n, _m in CHANGES])
def test_a_genuine_change_of_intent_replaces_the_active_flow(ctx_reader, flow, new, message):
    ctx_reader.add(message, R(new, changes_intent=True))
    update = _route([H(content="مرحبا"), A(content="تمام، كمل معايا"), H(content=message)], previous=flow)
    assert update["active_agent"] == new
    assert "intent changed" in update["routing_reason"]


def test_an_answer_to_the_active_flow_does_not_switch(ctx_reader):
    ctx_reader.add("الخميس", R("answer"), DAY_Q)
    update = _route([H(content="عايز احجز"), A(content=DAY_Q), H(content="الخميس")], previous="booking")
    assert update["active_agent"] == "booking"


def test_an_unconfident_different_intent_does_not_tear_a_flow_away(ctx_reader):
    ctx_reader.add("ينفع كده؟", R("faq", confidence=0.6))
    update = _route([H(content="عايز احجز"), A(content=DAY_Q), H(content="ينفع كده؟")], previous="booking")
    assert update["active_agent"] == "booking"


# ----------------------------------------------------------------------
# 5. Confidence: context first, then one clarification - never a dead end
# ----------------------------------------------------------------------

def test_ambiguous_with_no_context_asks_one_useful_question(session_id, llm, ctx_reader):
    ctx_reader.add("الموعد", R("other", is_ambiguous=True, confidence=0.3,
                               alternatives=["booking", "reschedule", "cancel"]))
    result = send(session_id, "الموعد")
    reply = result["reply"]
    assert "هل تقصد حجز موعد جديد، تعديل موعد موجود، أو إلغاء موعد؟" in reply
    assert "لم أتمكن من فهم" not in reply and "توضح" not in reply
    assert len(llm.calls) == 0
    assert result["escalate"] is False


def test_ambiguous_inside_a_flow_continues_that_flow(ctx_reader):
    ctx_reader.add("الموعد", R("other", is_ambiguous=True, confidence=0.3,
                               alternatives=["booking", "reschedule", "cancel"]))
    update = _route([H(content="عايز الغي"), A(content="تحب تلغي الموعد برقم الجوال ولا برقم الحجز؟"),
                     H(content="الموعد")], previous="cancel")
    assert _selected(update) == "cancel"


def test_low_confidence_greeting_is_not_a_clarification(ctx_reader):
    ctx_reader.add("هاي", R("greeting", confidence=0.4))
    assert _selected(_route([H(content="هاي")])) == "concierge"


def test_clarification_in_english_conversation(session_id, llm, ctx_reader):
    ctx_reader.add("the appointment", R("other", is_ambiguous=True, confidence=0.3,
                                        alternatives=["booking", "cancel"]))
    reply = send(session_id, "the appointment")["reply"]
    assert "Do you mean booking a new appointment or cancelling an appointment?" in reply


def test_the_clarified_answer_routes_on_the_next_turn(session_id, llm, ctx_reader):
    ctx_reader.add("الموعد", R("other", is_ambiguous=True, confidence=0.3,
                               alternatives=["booking", "reschedule", "cancel"]))
    send(session_id, "الموعد")
    question = [m for m in state_of(session_id)["messages"] if m.type == "ai"][-1].content
    ctx_reader.add("إلغاء", R("cancel", answer_to_previous_question=True, cancel_request=True),
                   " ".join(question.split())[-600:])
    llm._responses.append(AIMessage(content="تحب تلغي الموعد برقم الجوال ولا برقم الحجز؟"))
    send(session_id, "إلغاء")
    assert state_of(session_id)["active_agent"] == "cancel"


# ----------------------------------------------------------------------
# 6. Technical failure is not uncertainty
# ----------------------------------------------------------------------

def _decision_logs(caplog):
    out = []
    for record in caplog.records:
        message = record.getMessage()
        if message.startswith("routing_decision "):
            out.append(json.loads(message[len("routing_decision "):]))
    return out


def test_understanding_timeout_is_a_logged_deterministic_fallback(ctx_reader, caplog):
    ctx_reader.fail = True
    with caplog.at_level(logging.INFO, logger="graph"):
        _route([H(content="مش هقدر اجي")], previous="booking")
    decision = _decision_logs(caplog)[-1]
    assert decision["routing_mode"] == "deterministic_fallback"
    assert decision["semantic_intent"] is None


def test_malformed_model_output_is_a_logged_deterministic_fallback(ctx_reader, caplog):
    ctx_reader.garbage = True
    with caplog.at_level(logging.INFO, logger="graph"):
        _route([H(content="مش هقدر اجي")])
    assert _decision_logs(caplog)[-1]["routing_mode"] == "deterministic_fallback"


def test_uncertainty_is_semantic_not_fallback(ctx_reader, caplog):
    ctx_reader.add("الموعد", R("other", is_ambiguous=True, confidence=0.3))
    with caplog.at_level(logging.INFO, logger="graph"):
        update = _route([H(content="الموعد")])
    decision = _decision_logs(caplog)[-1]
    assert update["clarify_now"] is True
    assert decision["routing_mode"] == "semantic"
    assert decision["selected_agent"] == "clarify"


def test_routing_log_carries_the_decision_and_not_the_patient(ctx_reader, caplog):
    secret = "اسمي منى ورقمي 0501234567 وعايزة ألغي"
    ctx_reader.add(secret, R("cancel", cancel_request=True, entities={"phone": "0501234567"}))
    with caplog.at_level(logging.DEBUG):
        _route([H(content="مرحبا"), A(content="أهلا"), H(content=secret)], previous="booking")
    decision = _decision_logs(caplog)[-1]
    for key in ("previous_agent", "semantic_intent", "confidence", "answer_to_previous_question",
                "changes_intent", "selected_agent", "routing_mode", "override_reason"):
        assert key in decision
    assert decision["selected_agent"] == "cancel" and decision["previous_agent"] == "booking"
    everything = "\n".join(r.getMessage() for r in caplog.records)
    assert "0501234567" not in everything and "منى" not in everything


# ----------------------------------------------------------------------
# 7. The hard rules still hold - and are the only overrides
# ----------------------------------------------------------------------

def test_yes_please_to_a_transfer_offer_transfers(session_id, llm, ctx_reader):
    """REGRESSION: "أيوه ياريت" was vetoed by the old bare-yes regex, and
    the vetoed reading then made request_human_handoff refuse too."""
    llm._responses.append(AIMessage(content=HANDOFF_OFFER))
    ctx_reader.add("عندي مشكلة في الفاتورة", R("faq"))
    send(session_id, "عندي مشكلة في الفاتورة")
    # Keyed on the message alone: the first reply also carries the clinic
    # greeting, so the question line is longer than HANDOFF_OFFER.
    ctx_reader.add("أيوه ياريت", R("human", answer_to_previous_question=True, wants_human=True))
    result = send(session_id, "أيوه ياريت")
    assert result["escalate"] is True


@pytest.mark.parametrize("yes", ["ياليت والله", "ايوه ياريت بسرعة", "اكيد يعطيكي العافية",
                                 "yes confirm i need help", "co firm"])
def test_acceptances_the_old_yes_regex_refused(ctx_reader, yes):
    ctx_reader.add(yes, R("human", answer_to_previous_question=True, wants_human=True), HANDOFF_OFFER)
    update = _route([H(content="استفسار"), A(content=HANDOFF_OFFER), H(content=yes)], previous="faq")
    assert update["handoff_now"] is True


def test_accepting_a_transfer_that_was_never_offered_is_not_carried_out(ctx_reader):
    ctx_reader.add("اه", R("answer", wants_human=True), BOOK_OFFER)
    update = _route([H(content="عيني"), A(content=BOOK_OFFER), H(content="اه")], previous="medical")
    assert update["handoff_now"] is False
    assert update["understanding"]["wants_human"] is False, "the tool must see the refused reading"


def test_a_list_number_is_never_consent(ctx_reader):
    offer = "عذرًا 🌷 هل تحب أحوّلك لأحد ممثلي خدمة العملاء؟ أو اختار: 1️⃣ حجز 2️⃣ إلغاء 3️⃣ تعديل"
    ctx_reader.add("3", R("answer", wants_human=True), " ".join(offer.split()))
    update = _route([H(content="؟"), A(content=offer), H(content="3")], previous="concierge")
    assert update["handoff_now"] is False


def test_an_unsure_yes_is_not_a_transfer(ctx_reader):
    offer = "هذا القسم غير متوفر حاليًا. أقدر أحولك لأحد ممثلي خدمة العملاء؟"
    ctx_reader.add("دي", R("answer", wants_human=True, confidence=0.4), offer)
    update = _route([H(content="عندكم جلدية؟"), A(content=offer), H(content="دي")], previous="faq")
    assert update["handoff_now"] is False


def test_asking_for_a_person_while_answering_something_else_still_transfers(ctx_reader):
    ctx_reader.add("ابي اكلم موظف", R("human", answer_to_previous_question=True, wants_human=True), DAY_Q)
    update = _route([H(content="احجز"), A(content=DAY_Q), H(content="ابي اكلم موظف")], previous="booking")
    assert update["handoff_now"] is True


def test_crisis_overrides_everything_even_a_confident_booking_reading(ctx_reader, caplog):
    msg = "مابي اعيش خلاص"
    ctx_reader.add(msg, R("booking", crisis=True))
    with caplog.at_level(logging.INFO, logger="graph"):
        update = _route([H(content=msg)], previous="booking")
    assert update["handoff_now"] is True and update["crisis_active"] is True
    assert _decision_logs(caplog)[-1]["routing_mode"] == "safety_override"


def test_the_crisis_safety_pattern_still_fires_when_the_reading_misses(ctx_reader):
    ctx_reader.add("I want to kill myself", R("other"))
    assert _route([H(content="I want to kill myself")])["handoff_now"] is True


# ----------------------------------------------------------------------
# 8. What the understanding model is shown - compact, not the transcript
# ----------------------------------------------------------------------

def test_the_model_sees_compact_state_not_the_whole_conversation(ctx_reader):
    sid = "semantic-context"
    tools._BOOKING_SESSIONS[sid] = {"doctor_display_name": "د. عمر المديفر",
                                    "branch_display_name": "مدينة نصر",
                                    "last_list": {"entity_type": "slot", "items": []},
                                    "booking_phone": "0501234567"}
    try:
        history = []
        for i in range(15):
            history += [H(content=f"رسالة قديمة {i} " + "x" * 500), A(content=f"رد قديم {i} " + "y" * 500)]
        ctx_reader.add("2", R("booking", answer_to_previous_question=True))
        _route(history + [A(content=SLOTS_Q), H(content="2")], previous="booking", session_id=sid)
        prompt = ctx_reader.prompts[-1]
    finally:
        tools._BOOKING_SESSIONS.pop(sid, None)

    conversation = prompt.split("CONVERSATION", 1)[1].split("THE PATIENT'S NEW MESSAGE:")[0]
    assert len([l for l in conversation.split("\n") if l.startswith(("PATIENT: ", "ASSISTANT: "))]) <= 6
    assert "رسالة قديمة 0 " not in prompt, "old turns leak into the prompt"
    assert _question_of(prompt) == SLOTS_Q, "the question being answered must be shown whole"
    state_line = re.search(r"^STATE: (.*)$", prompt, re.M).group(1)
    state = json.loads(state_line)
    assert state["flow"] == "booking"
    assert state["booking"]["doctor"] == "د. عمر المديفر" and state["booking"]["list_on_screen"] == "slot"
    assert "0501234567" not in prompt, "the context must not carry phone numbers"


def test_parser_reads_the_semantic_fields():
    raw = json.dumps({
        "intent": "reschedule", "confidence": "0.82", "is_ambiguous": False,
        "alternatives": ["reschedule", "cancel", "teleport"], "answer_to_previous_question": "true",
        "changes_intent": True, "doctor_name": None, "specialty": "عيون",
        "entities": {"doctor": "عمر", "date": "الأسبوع الجاي", "phone": "null"},
        "reason": "wants another day " * 20,
    }, ensure_ascii=False)
    got = understanding._parse(raw)
    assert got["intent"] == "reschedule" and got["confidence"] == pytest.approx(0.82)
    assert got["alternatives"] == ["reschedule", "cancel"]
    assert got["answer_to_previous_question"] is True and got["changes_intent"] is True
    assert got["doctor_name"] == "عمر" and got["entities"]["doctor"] == "عمر"
    assert got["entities"]["date"] == "الأسبوع الجاي" and got["entities"]["phone"] is None
    assert len(got["reason"]) <= 120


def test_an_older_reading_without_the_new_fields_still_routes():
    got = understanding._parse('{"intent": "cancel"}')
    assert got["confidence"] is None and got["entities"]["branch"] is None
    assert graph._route_from_reading(got, "booking")[0] == "cancel"


def test_reason_is_never_part_of_the_reply(session_id, llm, ctx_reader):
    ctx_reader.add("هلا", R("greeting", reason="INTERNAL-REASON-MARKER"))
    llm._responses.append(AIMessage(content="أهلا 🌷"))
    assert "INTERNAL-REASON-MARKER" not in send(session_id, "هلا")["reply"]


# ----------------------------------------------------------------------
# 9. Directives that used to contradict the reading
# ----------------------------------------------------------------------

def test_fixed_step1_question_uses_the_meaning_not_the_first_verb():
    msgs = [H(content="مش عايز الغي، عايز اعدل")]
    assert graph._cancel_or_reschedule_intent(msgs, "cancel") == "cancel"  # the verb regex alone
    assert graph._cancel_or_reschedule_intent(msgs, "cancel", R("reschedule")) == "reschedule"


def test_fixed_booking_opener_stands_down_for_a_doctor_named_without_a_cue():
    msgs = [H(content="عايز احجز مع عمر المديفر")]
    reading = R("booking", doctor_name="عمر المديفر")
    assert graph._build_booking_entry_directive(msgs, "semantic-entry", "booking", reading=reading) == ""


def test_nafsi_as_i_would_love_to_is_not_the_psychiatry_specialty():
    msgs = [H(content="نفسي احجز موعد")]
    directive = graph._build_multi_intent_directive(msgs, "semantic-nafsi", "booking",
                                                    reading=R("booking", specialty=None))
    assert "A SPECIALTY OR SERVICE" not in directive


def test_asking_which_booking_after_an_indirect_cancellation_is_not_an_invention():
    reply = "تحب تلغي الموعد برقم الجوال ولا برقم الحجز؟"
    state = {"messages": [H(content="مش هقدر اجي بكرة")],
             "understanding": R("cancel", cancel_request=True)}
    assert graph._reply_asks_to_identify_a_booking_that_was_never_mentioned(reply, state) is False
    state_without = {"messages": [H(content="مش هقدر اجي بكرة")]}
    assert graph._reply_asks_to_identify_a_booking_that_was_never_mentioned(reply, state_without) is True


def test_just_booked_directive_follows_an_indirect_cancellation():
    msgs = [
        H(content="عايز احجز"),
        ToolMessage(content=json.dumps({"status": "success", "booking_ref": "TNS-9"}),
                    name="create_new_booking", tool_call_id="b1"),
        A(content="تم الحجز ✅ رقم الحجز TNS-9"),
        H(content="معلش مش هقدر اجي"),
    ]
    assert graph._build_just_booked_directive(msgs) == ""
    assert "TNS-9" in graph._build_just_booked_directive(msgs, R("cancel", cancel_request=True))


# ----------------------------------------------------------------------
# "مش هعرف اجي بكره" - they can't make it. That is NOT a cancellation:
# it is cancel OR reschedule, and only the patient knows which.
# (Production, agent-mu1, 2026-09-26: it was routed straight to cancel.)
# ----------------------------------------------------------------------

def test_cant_come_asks_cancel_or_move_instead_of_guessing(session_id, llm, ctx_reader):
    ctx_reader.add("مش هعرف اجي بكره", R("cancel", is_ambiguous=True, confidence=0.5,
                                          alternatives=["cancel", "reschedule"]))
    result = send(session_id, "مش هعرف اجي بكره")
    assert "تحب نلغي الموعد، ولا نأجله ليوم تاني؟" in result["reply"]
    assert len(llm.calls) == 0
    assert state_of(session_id)["active_agent"] != "cancel", "guessed cancel"


def test_the_answer_to_cancel_or_move_goes_to_the_right_flow(session_id, llm, ctx_reader):
    ctx_reader.add("مش هعرف اجي بكره", R("cancel", is_ambiguous=True, confidence=0.5,
                                          alternatives=["cancel", "reschedule"]))
    send(session_id, "مش هعرف اجي بكره")
    ctx_reader.add("أجله", R("reschedule", answer_to_previous_question=True))
    llm._responses.append(AIMessage(content="تحب تعدل الموعد برقم الجوال ولا برقم الحجز؟"))
    send(session_id, "أجله")
    assert state_of(session_id)["active_agent"] == "reschedule"


def test_the_prompt_no_longer_teaches_cant_come_as_cancel():
    assert "not being able to come to a booked appointment is a cancellation" not in understanding.PROMPT
    assert 'alternatives ["cancel", "reschedule"]' in understanding.PROMPT
