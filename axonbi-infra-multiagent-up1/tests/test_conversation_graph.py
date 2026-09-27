"""
The conversation graph, end to end, with a SCRIPTED model (no network).

The scripted model plays the conversation LLM: each call pops the next
scripted decision and records the exact messages it was sent. So these
tests prove the architecture's side of the contract:

  - one model call for a turn that needs no tool; two when a tool result
    has to be read; zero for outcomes code writes itself (success
    confirmations, the booking review card);
  - the model sees ONE stable system prompt, the recent text history and
    ONE compact state block - never the old tool payloads of earlier
    turns, never per-turn prose directives;
  - the model's structured decisions (flow, language, what it asked to
    confirm) become state; a confirmation request lives exactly one turn;
  - a draft claiming an action that did not happen gets one correction
    and, failing that, never reaches the patient.

What the REAL model understands is measured by test_live_semantics.py.
"""

import itertools
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import fake_hospital

_ids = itertools.count(1)


class ScriptedLLM:
    def __init__(self, *steps):
        self.steps = list(steps)
        self.calls = []

    def invoke(self, messages, config=None, **kwargs):
        self.calls.append(list(messages))
        if not self.steps:
            raise AssertionError("the model was called more times than scripted")
        step = self.steps.pop(0)
        return step() if callable(step) else step


def say(message, flow="general", language="ar", **extra):
    return AIMessage(content="", tool_calls=[{
        "name": "respond", "id": f"r{next(_ids)}", "type": "tool_call",
        "args": {"message": message, "flow": flow, "language": language, **extra}}])


def call(_tool, **args):
    return AIMessage(content="", tool_calls=[{"name": _tool, "args": args, "id": f"t{next(_ids)}", "type": "tool_call"}])


@pytest.fixture
def hospital(monkeypatch):
    h = fake_hospital.install(monkeypatch)
    h.reset()
    return h


@pytest.fixture
def g(hospital, monkeypatch):
    import graph
    return graph


def run(g, llm, monkeypatch, text, thread, first=False, **seed):
    monkeypatch.setattr(g, "_llm_with_tools", llm)
    state = {"client_id": fake_hospital.TENANT["client_id"], "session_id": f"+966500000001+{thread}",
             "channel_phone": "+966500000001", "bsuid": None, "raw_client_config": fake_hospital.TENANT,
             "messages": [HumanMessage(content=text)], **seed}
    if first:
        state.update({"greeted": False, "target_language": None})
    return g.graph.invoke(state, config={"configurable": {"thread_id": thread}, "recursion_limit": 30})


def system_blocks(call_messages):
    return [m.content for m in call_messages if isinstance(m, SystemMessage)]


# ----------------------------------------------------------------------
# calls per turn
# ----------------------------------------------------------------------

def test_a_turn_without_tools_is_one_call_and_greets_on_the_first_reply(g, monkeypatch):
    llm = ScriptedLLM(say("أوقات الدوام من 9 صباحًا إلى 9 مساءً. تحب أساعدك بشي ثاني؟", flow="faq"))
    result = run(g, llm, monkeypatch, "وش أوقات الدوام؟", "faq-1", first=True)

    assert len(llm.calls) == 1
    reply = result["messages"][-1].content
    assert reply.index("أهلا") < reply.index("أوقات الدوام") if "أهلا" in reply else True
    assert result["flow"] == "faq" and result["greeted"] is True


def test_a_tool_turn_is_two_calls_and_the_second_sees_the_result(g, monkeypatch):
    llm = ScriptedLLM(call("list_specialties"), say("عندنا هالتخصصات", flow="booking", show_options=False))
    run(g, llm, monkeypatch, "ابي احجز", "book-1", greeted=True)

    assert len(llm.calls) == 2
    assert any(getattr(m, "type", None) == "tool" and m.name == "list_specialties" for m in llm.calls[1])


def test_the_prompt_is_one_stable_system_message_plus_one_state_block(g, monkeypatch):
    llm = ScriptedLLM(say("تمام", flow="faq"), say("تمام", flow="faq"))
    run(g, llm, monkeypatch, "سؤال أول", "stable-1", greeted=True)
    run(g, llm, monkeypatch, "سؤال ثاني", "stable-1")

    first, second = llm.calls
    assert first[0].content == second[0].content            # byte-identical prefix -> cacheable
    assert len(system_blocks(first)) == 2 and system_blocks(first)[1].startswith("CURRENT STATE")
    assert "INTERNAL INSTRUCTION" not in "".join(system_blocks(second))


def test_earlier_tool_payloads_are_not_resent(g, monkeypatch):
    llm = ScriptedLLM(call("list_specialties"), say("اختر تخصص", flow="booking"), say("تمام", flow="booking"))
    run(g, llm, monkeypatch, "ابي احجز", "hist-1", greeted=True)
    run(g, llm, monkeypatch, "1", "hist-1")

    later = llm.calls[-1]
    assert not any(getattr(m, "type", None) == "tool" for m in later)
    assert not any(getattr(m, "tool_calls", None) for m in later)
    assert any(getattr(m, "content", "") == "اختر تخصص" for m in later)  # the text of that turn stays


def test_a_reply_written_before_the_tool_results_is_discarded(g, monkeypatch):
    both = AIMessage(content="", tool_calls=[
        {"name": "list_specialties", "args": {}, "id": "x1", "type": "tool_call"},
        {"name": "respond", "args": {"message": "مسودة", "flow": "booking", "language": "ar"}, "id": "x2",
         "type": "tool_call"}])
    llm = ScriptedLLM(both, say("النهائي", flow="booking"))
    result = run(g, llm, monkeypatch, "ابي احجز", "early-1", greeted=True)
    assert result["messages"][-1].content == "النهائي"


# ----------------------------------------------------------------------
# structured decisions become state
# ----------------------------------------------------------------------

def test_awaiting_confirmation_is_recorded_and_lives_one_turn(g, monkeypatch):
    llm = ScriptedLLM(
        say("متأكد تبغى تلغي موعد TNS-10001؟", flow="cancel",
            awaiting_confirmation={"action": "cancel", "target": "TNS-10001"}),
        say("تمام", flow="general"),
        say("كيف أقدر أساعدك؟", flow="general"),
    )
    first = run(g, llm, monkeypatch, "ابي الغي موعدي", "pend-1", greeted=True)
    assert first["pending_confirmation"] == {"action": "cancel", "target": "TNS-10001", "turn": 1}

    second = run(g, llm, monkeypatch, "لا خلاص", "pend-1")
    assert second["pending_confirmation"] is None           # the reply did not ask again

    third = run(g, llm, monkeypatch, "طيب", "pend-1")
    assert third["pending_confirmation"] is None


def test_state_block_tells_the_model_what_it_asked_to_confirm(g, monkeypatch):
    llm = ScriptedLLM(
        say("متأكد؟", flow="cancel", awaiting_confirmation={"action": "cancel", "target": "TNS-10001"}),
        say("تمام", flow="cancel"),
    )
    run(g, llm, monkeypatch, "ابي الغي", "pend-2", greeted=True)
    run(g, llm, monkeypatch, "اه", "pend-2")
    assert "YOUR LAST MESSAGE ASKED TO CONFIRM: cancel TNS-10001" in system_blocks(llm.calls[1])[1]


def test_flow_and_language_are_the_models_decisions(g, monkeypatch):
    llm = ScriptedLLM(say("Sure, which specialty?", flow="booking", language="en"))
    result = run(g, llm, monkeypatch, "I'd like to see a doctor", "lang-1", greeted=True)
    assert result["flow"] == "booking" and result["target_language"] == "en"


def test_show_options_renders_the_stored_list_in_order(g, hospital, monkeypatch):
    llm = ScriptedLLM(call("list_specialties"), say("اختر التخصص:", flow="booking", show_options=True))
    result = run(g, llm, monkeypatch, "وش التخصصات؟", "opts-1", greeted=True)
    reply = result["messages"][-1].content
    assert reply.startswith("اختر التخصص:\n1️⃣")


# ----------------------------------------------------------------------
# outcomes written by code - no model call
# ----------------------------------------------------------------------

def test_success_is_rendered_by_code_without_a_second_call(g, monkeypatch):
    import tools as _tools
    payload = {"status": "success"}
    appointment = {"ref": "TNS-10001", "patientFullName": "محمد عبدالله العمري", "date_display": "30-09-2026",
                   "time_display": "1:00 م", "doctorName": "د. أحمد سامي", "branchName": "المنار"}

    def fake_cancel_tool(state):  # the tool itself is exercised in the tools tests
        return payload

    llm = ScriptedLLM(call("cancel_appointment", booking_id="TNS-10001", patient_confirmed=True))
    monkeypatch.setattr(_tools, "_BOOKING_SESSIONS", {"+966500000001+succ-1": {"selected_appointment": appointment}})
    node = g._base_tool_node
    monkeypatch.setattr(node, "invoke", lambda state, config=None: {"messages": [
        __import__("langchain_core.messages", fromlist=["ToolMessage"]).ToolMessage(
            content=json.dumps(payload), name="cancel_appointment",
            tool_call_id=state["messages"][-1].tool_calls[0]["id"])]})

    result = run(g, llm, monkeypatch, "اه الغيه", "succ-1", greeted=True, target_language="ar")
    assert len(llm.calls) == 1
    assert "تم إلغاء موعدك بنجاح" in result["messages"][-1].content


# ----------------------------------------------------------------------
# safety: one correction at most, a false claim never goes out
# ----------------------------------------------------------------------

def test_a_false_done_claim_gets_one_correction_then_a_safe_reply(g, monkeypatch):
    llm = ScriptedLLM(say("تم حجز موعدك بنجاح ✅", flow="booking"),
                      say("تم تأكيد حجزك ✅", flow="booking"))
    result = run(g, llm, monkeypatch, "احجز", "claim-1", greeted=True)
    assert len(llm.calls) == 2
    reply = result["messages"][-1].content
    assert "تم" not in reply.split("\n")[0] or "حجز" not in reply.split("\n")[0]


def test_a_correction_that_calls_a_tool_is_followed(g, monkeypatch):
    roster = "الدكاترة المتاحين:\n1. د. سمير الوهمي - جلدية\n2. د. هاني الخيالي - جلدية"
    llm = ScriptedLLM(say(roster, flow="booking"), call("list_specialties"),
                      say("تفضل اختر التخصص", flow="booking"))
    result = run(g, llm, monkeypatch, "مين الدكاترة؟", "claim-2", greeted=True)
    assert result["messages"][-1].content == "تفضل اختر التخصص"
    assert len(llm.calls) == 3


def test_a_runaway_turn_is_bounded(g, monkeypatch):
    llm = ScriptedLLM(*[call("list_specialties") for _ in range(20)])
    result = run(g, llm, monkeypatch, "ابي احجز", "loop-1", greeted=True)
    assert len(llm.calls) == g.MAX_LLM_CALLS_PER_TURN
    assert result["messages"][-1].content


def test_crisis_wording_adds_a_safety_line_to_the_state_block(g, monkeypatch):
    llm = ScriptedLLM(say("أنا معك", flow="medical"))
    run(g, llm, monkeypatch, "والله تعبت ومش عايزة أعيش", "crisis-1", greeted=True)
    assert "SAFETY:" in system_blocks(llm.calls[0])[1]


def test_model_failure_is_a_short_apology_not_a_crash(g, monkeypatch):
    def boom():
        raise TimeoutError("timeout")
    llm = ScriptedLLM(boom, boom)
    result = run(g, llm, monkeypatch, "مرحبا", "fail-1", greeted=True)
    assert "تأخير" in result["messages"][-1].content


# ----------------------------------------------------------------------
# confirmation: the model reads the word, the code guards the action
# ----------------------------------------------------------------------
#
# The same patient word ("تمام") after three different questions. The
# scripted model is made to call cancel_appointment(patient_confirmed=True)
# in ALL three - i.e. to misread it - and the real tool + gate must only
# let it through where the assistant actually asked to confirm THIS
# cancellation on the previous turn.

def _seed_lookup(g, monkeypatch, thread):
    """Turn 1: the patient asks about their booking; the (scripted)
    model looks it up and replies."""
    llm = ScriptedLLM(call("lookup_appointment", use_channel_identity=True),
                      say("موعدك مع د. أحمد سامي يوم الأربعاء. وش تحب تسوي؟", flow="general"))
    run(g, llm, monkeypatch, "ابي اعرف موعدي", thread, greeted=True, target_language="ar")


@pytest.mark.parametrize("asked, pending, cancelled", [
    ("متأكد تبغى تلغي موعدك مع د. أحمد سامي؟", {"action": "cancel", "target": "TNS-10001"}, True),
    ("اختار التخصص المناسب", None, False),
    ("نكمل الحجز على نفس رقم الواتساب؟", None, False),
    ("متأكد تبغى تلغي موعدك؟", {"action": "cancel", "target": "TNS-99999"}, False),   # another booking
], ids=["asked-to-cancel-this", "asked-specialty", "asked-same-number", "asked-about-another"])
def test_tamam_can_only_cancel_what_was_asked(g, hospital, monkeypatch, asked, pending, cancelled):
    thread = f"tamam-{asked[:6]}-{bool(pending)}"
    _seed_lookup(g, monkeypatch, thread)
    ask = say(asked, flow="cancel" if pending else "booking",
              **({"awaiting_confirmation": pending} if pending else {}))
    run(g, ScriptedLLM(ask), monkeypatch, "طيب", thread)

    misread = ScriptedLLM(call("cancel_appointment", booking_id="TNS-10001", patient_confirmed=True),
                          say("تمام", flow="cancel"))
    run(g, misread, monkeypatch, "تمام", thread)

    assert (hospital.booking("TNS-10001")["status"] == 6) is cancelled


def test_the_model_saying_not_confirmed_never_cancels(g, hospital, monkeypatch):
    thread = "not-confirmed"
    _seed_lookup(g, monkeypatch, thread)
    run(g, ScriptedLLM(say("متأكد تبغى تلغي؟", flow="cancel",
                           awaiting_confirmation={"action": "cancel", "target": "TNS-10001"})),
        monkeypatch, "ابي الغيه", thread)
    llm = ScriptedLLM(call("cancel_appointment", booking_id="TNS-10001", patient_confirmed=False),
                      say("تمام، ما لغيت الموعد", flow="general"))
    run(g, llm, monkeypatch, "لا استنى", thread)
    assert hospital.booking("TNS-10001")["status"] != 6


def test_a_confirmation_question_expires_after_one_turn(g, hospital, monkeypatch):
    thread = "expired"
    _seed_lookup(g, monkeypatch, thread)
    run(g, ScriptedLLM(say("متأكد تبغى تلغي؟", flow="cancel",
                           awaiting_confirmation={"action": "cancel", "target": "TNS-10001"})),
        monkeypatch, "ابي الغيه", thread)
    run(g, ScriptedLLM(say("أوقات الدوام من 9 إلى 9", flow="faq")), monkeypatch, "وش أوقات الدوام؟", thread)
    llm = ScriptedLLM(call("cancel_appointment", booking_id="TNS-10001", patient_confirmed=True),
                      say("تمام", flow="cancel"))
    run(g, llm, monkeypatch, "اه", thread)
    assert hospital.booking("TNS-10001")["status"] != 6


def test_review_card_is_rendered_by_code_and_arms_the_booking_gate(g, hospital, monkeypatch):
    import tools as _tools
    thread = "review-1"
    llm = ScriptedLLM(
        call("match_entity_for_booking", entity_type="doctor", name="أحمد سامي"),
        call("match_entity_for_booking", entity_type="branch", name="النزهة"),
        call("get_available_slots_for_booking", from_date="2026-09-29", to_date="2026-09-29"),
        say("اختار الموعد:", flow="booking", show_options=True))
    run(g, llm, monkeypatch, "ابي احجز مع د. أحمد سامي في النزهة بكرة", thread, greeted=True, target_language="ar")

    llm = ScriptedLLM(call("select_appointment_slot", option_number=1),
                      call("compare_phone", provided_phone="+966500000001"),
                      call("confirm_booking_review", patient_full_name="محمد عبدالله العمري"))
    result = run(g, llm, monkeypatch, "1 وعلى نفس الرقم، اسمي محمد عبدالله العمري", thread)

    assert len(llm.calls) == 3                       # no model call to write the card
    assert "يرجى مراجعة بيانات الحجز" in result["messages"][-1].content
    session = _tools._BOOKING_SESSIONS[f"+966500000001+{thread}"]
    assert result["pending_confirmation"]["action"] == "book"
    assert result["pending_confirmation"]["target"] == session["review"]["slotStart"]

    llm = ScriptedLLM(call("create_new_booking", patient_confirmed=True))
    result = run(g, llm, monkeypatch, "ايوه أكد", thread)
    assert len(llm.calls) == 1                       # the success message is written by code
    assert "رقم الحجز" in result["messages"][-1].content
