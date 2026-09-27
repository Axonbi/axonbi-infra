"""
Semantic routing - WIRING tests (no network).

WHAT THESE PROVE, AND WHAT THEY CANNOT
--------------------------------------
The router model is replaced by a stand-in that returns the reading a
correct model would give. So these tests prove the ROUTER'S part:

  - the patient's words have no path to the routing decision except
    through the model's reading: paraphrases, keyword-free rewrites and
    messages stuffed with another intent's keywords all route the same
    way under the same reading;
  - the model is SHOWN what a short reply needs: the assistant's last
    question/list, the patient's previous message, the flow in progress
    and the booking facts - and not the rest of the history;
  - the rules applied to a reading (keep an open flow, follow a topic
    change, hand over when the owner lacks the tool, crisis -> medical);
  - a failed reading never moves the conversation to a new specialist;
  - the rest of the graph uses the SAME reading (crisis, health guard,
    booking-context wipe), and a stale reading is never applied.

Whether the REAL model reads these phrasings correctly is what
test_live_router.py measures, over the same scenarios.
"""

import inspect
import re

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agents.router as router
from router_cases import CASES, build_messages


class StandInReader:
    """Plays the router model. `judge(prompt)` returns a reading dict or
    an Exception to raise; every prompt it was shown is kept."""

    def __init__(self, judge):
        self.judge = judge
        self.prompts = []

    def with_structured_output(self, schema):
        reader = self

        class _Bound:
            def invoke(self, messages):
                prompt = messages[0].content
                reader.prompts.append(prompt)
                result = reader.judge(prompt)
                if isinstance(result, Exception):
                    raise result
                return schema(**result)

        return _Bound()


def fixed(reading):
    return StandInReader(lambda prompt: dict(reading))


def route(case, message=None, reader=None, facts=None):
    reader = reader or fixed(case.reading)
    return router.route_turn(
        build_messages(case, message), case.active,
        facts=facts or router.compact_facts(case.session), llm=reader,
    )


# ----------------------------------------------------------------------
# 1. The words themselves cannot decide
# ----------------------------------------------------------------------

def test_router_contains_no_patterns_or_keyword_tables():
    source = inspect.getsource(router)
    assert not re.search(r"^\s*import re\b|^\s*from re\b", source, re.M)
    assert not any(isinstance(value, re.Pattern) for value in vars(router).values())
    for gone in ("score_message", "_CUES", "_START_THRESHOLD", "_SWITCH_THRESHOLD",
                 "CRISIS_RE", "INJURY_RE", "looks_like_health_message", "normalize"):
        assert not hasattr(router, gone), gone


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_every_paraphrase_routes_like_the_original(case):
    """The required phrasings and their keyword-free rewrites, under the
    same reading, land on the same specialist."""

    original = route(case)[0]
    assert original in case.expected
    for paraphrase in case.paraphrases:
        assert route(case, paraphrase)[0] == original, paraphrase


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_another_intents_keywords_cannot_override_the_reading(case):
    """A message crammed with every old cue word still routes by the
    reading - there is no keyword path left to win."""

    decoys = "الغاء الغي حجز احجز تعديل اعدل شكوى وجع صداع cancel book reschedule complaint"
    assert route(case, decoys)[0] == route(case)[0]


def test_same_words_different_question_different_owner():
    by_id = {case.id: case for case in CASES}
    inside_booking = route(by_id["tomorrow_inside_booking"])[0]
    inside_reschedule = route(by_id["tomorrow_inside_reschedule"])[0]
    assert (inside_booking, inside_reschedule) == ("booking", "reschedule")


# ----------------------------------------------------------------------
# 2. The reader is shown what a short reply needs - and not more
# ----------------------------------------------------------------------

def test_reader_sees_last_question_previous_message_flow_and_facts():
    case = next(c for c in CASES if c.id == "monday_then")
    reader = fixed(case.reading)
    route(case, reader=reader)
    prompt = reader.prompts[-1]

    assert "مفيش مواعيد يوم الأحد" in prompt                      # the assistant's question
    assert "PATIENT'S PREVIOUS MESSAGE: الأحد" in prompt           # what the patient said before
    assert "FLOW IN PROGRESS: booking" in prompt
    assert "doctor=د. أحمد سامي" in prompt                          # facts on file
    assert prompt.rstrip().endswith("طب الاثنين؟")


def test_reader_is_not_sent_the_whole_history():
    old = "رسالة قديمة جدا لا علاقة لها"
    history = [HumanMessage(content=old, id="old")]
    for i in range(30):
        history += [AIMessage(content=f"رد {i}"), HumanMessage(content=f"سؤال {i}", id=f"q{i}")]
    history += [AIMessage(content="تحبي أي يوم؟"), HumanMessage(content="بكره", id="now")]

    reader = fixed({"intent": "booking", "topic_changed": False, "health": "none"})
    router.route_turn(history, "booking", llm=reader)
    prompt = reader.prompts[-1]

    assert old not in prompt and "سؤال 3\n" not in prompt
    assert "تحبي أي يوم؟" in prompt and "سؤال 29" in prompt
    assert len(prompt) < 2600


def test_long_list_keeps_its_question_the_tail_is_shown():
    options = "\n".join(f"{n}️⃣ {n}:00 مساءً" for n in range(1, 9))
    reply = ("المواعيد المتاحة:\n" + options + "\n") * 6 + "اختاري رقم الموعد اللي يناسبك"
    history = [HumanMessage(content="الثلاثاء", id="a"), AIMessage(content=reply),
               HumanMessage(content="رقم 2", id="b")]
    reader = fixed({"intent": "booking", "topic_changed": False, "health": "none"})
    router.route_turn(history, "booking", llm=reader)
    assert "اختاري رقم الموعد اللي يناسبك" in reader.prompts[-1]


def test_facts_are_names_not_ids():
    facts = router.compact_facts({
        "doctor_id": "GUID-123", "doctor_display_name": "د. أحمد",
        "branch_display_name": "المعادي",
        "selected_slot": {"date_display": "الاثنين", "time_display": "10:00 صباحًا", "slotStart": "x"},
        "last_list": {"entity_type": "slot", "items": [1, 2, 3]},
    })
    assert "GUID" not in facts
    assert facts == ("doctor=د. أحمد; branch=المعادي; slot chosen=الاثنين 10:00 صباحًا; "
                     "last list shown=slot (3 items)")
    assert router.compact_facts(None) == "none"


# ----------------------------------------------------------------------
# 3. What the router does with a reading
# ----------------------------------------------------------------------

def _two_turns(active_question="تحبي أي يوم؟", message="اه"):
    return [HumanMessage(content="عايزة أحجز", id="1"), AIMessage(content=active_question),
            HumanMessage(content=message, id="2")]


def test_answer_inside_open_flow_keeps_owner_even_if_labelled_otherwise():
    reading = {"intent": "faq", "topic_changed": False, "health": "none"}
    assert router.route_turn(_two_turns(), "booking", llm=fixed(reading))[0] == "booking"


def test_unclear_inside_open_flow_is_an_abstention():
    reading = {"intent": "concierge", "topic_changed": False, "health": "none"}
    assert router.route_turn(_two_turns(), "reschedule", llm=fixed(reading))[0] == "reschedule"


def test_topic_change_moves_the_conversation():
    reading = {"intent": "complaint", "topic_changed": True, "health": "none"}
    assert router.route_turn(_two_turns(), "booking", llm=fixed(reading))[0] == "complaint"


@pytest.mark.parametrize("owner", ["medical", "faq"])
def test_owner_without_the_tool_hands_over_the_accepted_offer(owner):
    reading = {"intent": "booking", "topic_changed": False, "health": "none"}
    chosen, reason, _ = router.route_turn(_two_turns(), owner, llm=fixed(reading))
    assert chosen == "booking" and "cannot" in reason


def test_crisis_goes_to_medical_whatever_else_is_open():
    reading = {"intent": "concierge", "topic_changed": False, "health": "crisis"}
    assert router.route_turn(_two_turns(), "booking", llm=fixed(reading))[0] == "medical"


def test_completed_flow_releases_the_owner():
    history = [
        HumanMessage(content="اه أكدي", id="1"),
        AIMessage(content="", tool_calls=[{"name": "create_new_booking", "args": {}, "id": "t1"}]),
        ToolMessage(content='{"status": "success"}', name="create_new_booking", tool_call_id="t1"),
        AIMessage(content="تم الحجز ✅"),
        HumanMessage(content="شكرا", id="2"),
    ]
    reading = {"intent": "concierge", "topic_changed": False, "health": "none"}
    assert router.route_turn(history, "booking", llm=fixed(reading))[0] == "concierge"


def test_reading_is_stamped_with_the_message_it_is_about():
    _, _, reading = router.route_turn(_two_turns(), "booking",
                                      llm=fixed({"intent": "booking", "topic_changed": False, "health": "none"}))
    assert reading["status"] == "ok" and reading["message_id"] == "2"


# ----------------------------------------------------------------------
# 4. A failed reading never guesses
# ----------------------------------------------------------------------

FAILURES = [
    TimeoutError("router timed out"),
    ValueError("provider returned garbage"),
]


@pytest.mark.parametrize("failure", FAILURES, ids=lambda e: type(e).__name__)
@pytest.mark.parametrize("owner", ["booking", "cancel", "reschedule", "medical", "complaint"])
def test_failure_keeps_an_open_flow_with_its_owner(failure, owner):
    chosen, _, reading = router.route_turn(
        _two_turns(message="عايزة ألغي"), owner, llm=StandInReader(lambda p: failure))
    assert chosen == owner and reading["status"] == "unavailable"


@pytest.mark.parametrize("owner", [None, "concierge"])
def test_failure_with_no_open_flow_goes_to_clarification(owner):
    chosen, reason, reading = router.route_turn(
        _two_turns(message="عايزة ألغي الحجز"), owner, llm=StandInReader(lambda p: TimeoutError()))
    assert chosen == "concierge" and reading["status"] == "unavailable" and "clarify" in reason


def test_out_of_schema_answer_counts_as_failure():
    bad = StandInReader(lambda p: {"intent": "payments", "topic_changed": False, "health": "none"})
    chosen, _, reading = router.route_turn(_two_turns(), "booking", llm=bad)
    assert chosen == "booking" and reading["status"] == "unavailable"


def test_no_reader_configured_counts_as_failure(monkeypatch):
    monkeypatch.setattr(router, "_default_llm", lambda: None)
    chosen, _, reading = router.route_turn(_two_turns(), None)
    assert chosen == "concierge" and reading["status"] == "unavailable"


# ----------------------------------------------------------------------
# 5. The rest of the graph uses the same reading
# ----------------------------------------------------------------------

graph = pytest.importorskip("graph")
import tools  # noqa: E402  (after graph, which configures it)


def _state(messages, **extra):
    return {"messages": messages, "session_id": "s-test", "client_id": "1",
            "templates": {}, "greeted": True, **extra}


def test_router_node_publishes_the_reading_and_shows_session_facts(monkeypatch):
    reader = fixed({"intent": "booking", "topic_changed": False, "health": "none"})
    monkeypatch.setattr(graph, "_router_llm", reader)
    monkeypatch.setitem(tools._BOOKING_SESSIONS, "s-test", {
        "doctor_display_name": "د. أحمد سامي",
        "last_list": {"entity_type": "slot", "items": [1, 2, 3]},
    })

    update = graph.router(_state(_two_turns(message="رقم 2"), active_agent="booking"))

    assert update["active_agent"] == "booking"
    assert update["turn_reading"]["status"] == "ok"
    assert update["turn_reading"]["message_id"] == "2"
    assert "doctor=د. أحمد سامي; last list shown=slot (3 items)" in reader.prompts[-1]


@pytest.mark.parametrize("topic_changed, kept", [(False, True), (True, False)])
def test_booking_context_is_wiped_only_on_a_real_change_of_subject(monkeypatch, topic_changed, kept):
    session = {"doctor_id": "d1", "branch_id": "b1", "selected_slot": {"slotStart": "x"}}
    monkeypatch.setitem(tools._BOOKING_SESSIONS, "s-test", session)
    reading = {"intent": "booking", "topic_changed": topic_changed, "health": "none"}

    graph._clear_abandoned_booking_context("booking", "reschedule", reading, "s-test")

    assert (session.get("doctor_id") == "d1") is kept


def test_crisis_directive_comes_from_the_reading_without_any_crisis_words():
    messages = [HumanMessage(content="خلاص مبقاش ليا نفس لأي حاجة", id="x")]
    assert graph._build_crisis_directive(messages, {}, None) == ""
    assert graph._build_crisis_directive(messages, {}, {"health": "crisis"}) != ""


def test_crisis_backstop_only_adds_never_removes():
    messages = [HumanMessage(content="عايزة أموت", id="x")]
    assert graph._build_crisis_directive(messages, {}, None) != ""
    assert graph._build_crisis_directive(messages, {}, {"health": "none"}) != ""


def test_health_guard_reads_the_reading():
    messages = [HumanMessage(content="رجلي وقعت عليها", id="x")]
    assert graph._message_is_about_health(messages, {"health": "health"})
    assert not graph._message_is_about_health(messages, {"health": "none"})


def test_a_stale_reading_is_never_applied_to_a_newer_message():
    reading = {"status": "ok", "message_id": "old", "health": "crisis"}
    state = _state([HumanMessage(content="old", id="old"), AIMessage(content="?"),
                    HumanMessage(content="new", id="new")], turn_reading=reading)
    assert graph._current_turn_reading(state) == {}
    state["turn_reading"] = {**reading, "message_id": "new"}
    assert graph._current_turn_reading(state)["health"] == "crisis"


def test_clarification_on_failure_makes_no_model_call(monkeypatch):
    def no_model(*_a, **_k):
        raise AssertionError("the main model must not be called")

    monkeypatch.setattr(graph, "_llm_for", no_model)
    messages = _two_turns(message="ممكن؟")
    state = _state(messages, turn_reading={"status": "unavailable", "message_id": "2"})

    update = graph._run_agent(state, "concierge")

    assert update["messages"][-1].content == graph._SOFT_RECOVERY_CLARIFY_TEXT["ar"]


def test_end_to_end_router_failure_on_first_message_sends_only_the_greeting(monkeypatch):
    def no_model(*_a, **_k):
        raise AssertionError("the main model must not be called")

    monkeypatch.setattr(graph, "_router_llm", StandInReader(lambda p: TimeoutError()))
    monkeypatch.setattr(graph, "_llm_for", no_model)

    result = graph.graph.invoke(
        {"client_id": "1", "session_id": "e2e-1", "channel_phone": None, "bsuid": None,
         "raw_client_config": None, "greeted": False, "target_language": None,
         "messages": [HumanMessage(content="السلام عليكم")]},
        config={"configurable": {"thread_id": "e2e-router-failure"}},
    )

    reply = result["messages"][-1].content
    assert result["active_agent"] == "concierge"
    assert result["turn_reading"]["status"] == "unavailable"
    assert reply and reply.strip().endswith(("؟", "?"))
