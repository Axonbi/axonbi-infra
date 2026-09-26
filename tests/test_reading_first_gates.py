"""
Round three: the remaining places that read the patient's words with a
regex now follow the turn's reading when one exists, and keep the regex
only as the technical-failure fallback. Plus the channel-identity fix in
compare_phone and per-request token observability.
"""

import json
import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import graph
import llm_usage
import tools
import understanding

H, A = HumanMessage, AIMessage


# ----------------------------------------------------------------------
# compare_phone: the channel identity comes from state, never from the model
# ----------------------------------------------------------------------

def _compare(state, provided, channel_arg=""):
    return tools.compare_phone.func(state=state, provided_phone=provided, channel_phone=channel_arg)


def _state(channel, session="cmp-1"):
    return {"session_id": session, "client_id": "tanasuq-test", "messages": [],
            "channel_phone": channel, "templates": {"timezone": "Asia/Riyadh"}}


def test_the_real_channel_number_matches():
    assert _compare(_state("+966500000001"), "+966500000001")["status"] == "match"


def test_a_model_supplied_channel_number_cannot_verify_anything():
    state = _state("+966500000001", session="cmp-2")
    result = _compare(state, "+966511111111", channel_arg="+966511111111")
    assert result["status"] == "no_match"
    assert not tools._phone_is_verified(state, "+966511111111")


def test_no_channel_identity_never_matches():
    state = _state(None, session="cmp-3")
    assert _compare(state, "+966511111111", channel_arg="+966511111111")["status"] == "no_match"
    assert not tools._phone_is_verified(state, "+966511111111")


# ----------------------------------------------------------------------
# Refusals: the reading decides, the prefix regex is the fallback
# ----------------------------------------------------------------------

OFFER = A(content="أقرب موعد عند د. سارة يوم السبت 10 الصبح. يناسبك؟")


def test_a_new_request_starting_with_la_is_not_a_refusal_of_the_offer():
    msgs = [OFFER, H(content="لا عاوزه اعدل الحجز")]
    assert graph._build_negation_directive(msgs) != ""  # the prefix regex alone
    reading = {"intent": "reschedule", "changes_intent": True, "declines": False}
    assert graph._build_negation_directive(msgs, reading) == ""


def test_a_symptom_starting_with_mesh_is_not_a_refusal():
    msgs = [OFFER, H(content="مش كويس وعندي صداع")]
    assert graph._build_negation_directive(msgs, {"intent": "medical", "declines": False}) == ""


def test_an_indirect_refusal_the_regex_cannot_see_is_one():
    msgs = [OFFER, H(content="السبت ده صعب عليا خالص")]
    assert graph._build_negation_directive(msgs) == ""
    assert graph._build_negation_directive(msgs, {"intent": "booking", "declines": True}) != ""


def test_the_day_hook_never_resolves_a_day_the_patient_rejected(monkeypatch):
    sid = "hook-declines"
    tools._BOOKING_SESSIONS[sid] = {"doctor_id": "D-1"}
    called = []
    monkeypatch.setattr(tools, "resolve_weekday_index", lambda text: called.append(text) or 5)
    try:
        state = {"session_id": sid, "messages": [OFFER, H(content="لا مش مناسب السبت")],
                 "understanding": {"intent": "booking", "declines": True}}
        assert graph._deterministic_day_and_slot_resolution(state, "booking") == []
        assert called == [], "the hook read the rejected day"
        # Without a reading the refusal patterns still stop it.
        state.pop("understanding")
        assert graph._deterministic_day_and_slot_resolution(state, "booking") == []
    finally:
        tools._BOOKING_SESSIONS.pop(sid, None)


def test_the_slot_hook_never_locks_a_rejected_time():
    sid = "slot-declines"
    tools._BOOKING_SESSIONS[sid] = {"last_list": {"entity_type": "slot", "items": [{"x": 1}]}}
    try:
        state = {"session_id": sid, "messages": [A(content="1️⃣ 4:00 2️⃣ 5:00"), H(content="لا مش 2")],
                 "understanding": {"intent": "booking", "declines": True}}
        assert graph._deterministic_slot_lock(state, "booking") is None
    finally:
        tools._BOOKING_SESSIONS.pop(sid, None)


# ----------------------------------------------------------------------
# Service directives
# ----------------------------------------------------------------------

def test_cancelling_a_checkup_is_not_a_request_to_book_it():
    msgs = [H(content="عايز الغي الكشف")]
    reading = {"intent": "cancel", "cancel_request": True}
    assert graph._build_service_named_directive(msgs, "svc-1", reading) == ""


def test_no_to_a_service_offer_is_not_choosing_it():
    msgs = [A(content="حابب تحجز موعد لفحص النظر في فرع الدقي؟"), H(content="لا مش دلوقتي")]
    with_no = graph._build_service_chosen_directive(msgs, "svc-2", {"intent": "other", "declines": True})
    assert with_no == ""


# ----------------------------------------------------------------------
# Medical guards are not silenced by a symptom containing "غير"
# ----------------------------------------------------------------------

def test_abnormal_pain_does_not_silence_the_medical_guards():
    state = {"messages": [A(content="عندنا دكاترة باطنة متاحين، تحب أحجزلك؟"),
                          H(content="عندي ألم غير طبيعي")],
             "understanding": {"intent": "medical", "declines": False}}
    assert graph._current_turn_accepts_a_medical_offer(state) is True
    state["understanding"] = {"intent": "reschedule"}
    assert graph._current_turn_accepts_a_medical_offer(state) is False


# ----------------------------------------------------------------------
# "Yes, use this number": no second model call when the turn has a reading
# ----------------------------------------------------------------------

def test_phone_already_known_uses_the_reading_not_another_model_call(monkeypatch):
    def explode(_text):
        raise AssertionError("a second classification call was made")
    monkeypatch.setattr(graph, "_classify_bare_affirmation_with_llm", explode)

    reply = "ممكن تعطيني رقم جوالك عشان أقدر أجيب بيانات موعدك؟"
    base = {"channel_phone": "+966500000001",
            "messages": [A(content="نكمل تعديل موعدك على نفس رقم الواتساب ده؟"), H(content="يب")]}
    assert graph._reply_asks_for_a_phone_already_known(reply, {**base, "understanding": {"confirms": True}}) is True
    assert graph._reply_asks_for_a_phone_already_known(reply, {**base, "understanding": {"confirms": False}}) is False


# ----------------------------------------------------------------------
# Tool gates
# ----------------------------------------------------------------------

@pytest.mark.parametrize("yes", ["صحيح", "ابعتها", "أرسلها يعطيك العافية"])
def test_complaint_yes_in_any_words_is_a_confirmation(yes):
    for question in ("هل تؤكد إرسال الشكوى؟", "هل تريد إرسال الشكوى؟"):
        state = {"messages": [A(content=question), H(content=yes)], "understanding": {"confirms": True}}
        assert tools._complaint_explicitly_confirmed(state) is True, (question, yes)


def test_complaint_confirmation_still_needs_our_question():
    state = {"messages": [A(content="ممكن تحكيلي تفاصيل أكتر؟"), H(content="صحيح")],
             "understanding": {"confirms": True}}
    assert tools._complaint_explicitly_confirmed(state) is False


def test_complaint_fallback_word_list_without_a_reading():
    state = {"messages": [A(content="هل تؤكد إرسال الشكوى؟"), H(content="نعم")]}
    assert tools._complaint_explicitly_confirmed(state) is True


def test_complaint_no_is_never_a_confirmation():
    state = {"messages": [A(content="هل تريد إرسال الشكوى؟"), H(content="استنى")],
             "understanding": {"confirms": False}}
    assert tools._complaint_explicitly_confirmed(state) is False


def test_location_asked_in_words_the_cue_list_lacks():
    base = {"session_id": "loc-1", "client_id": "tanasuq-test", "templates": {},
            "messages": [H(content="إزاي أوصلكم بالعربية من طريق الملك فهد")]}
    refused = tools.share_branch_location.func(state=base, branch_name="المنار")
    assert refused.get("reason") == "no_explicit_location_request"
    allowed = tools.share_branch_location.func(
        state={**base, "understanding": {"intent": "faq", "asks_location": True}}, branch_name="المنار")
    assert allowed.get("reason") != "no_explicit_location_request"


# ----------------------------------------------------------------------
# Parser: the new meaning fields
# ----------------------------------------------------------------------

def test_parser_reads_confirms_declines_and_location():
    got = understanding._parse('{"intent": "booking", "confirms": "true", "declines": false, "asks_location": 1}')
    assert got["confirms"] is True and got["declines"] is False and got["asks_location"] is True
    assert set(("confirms", "declines", "asks_location")) <= set(understanding.log_view(got))


# ----------------------------------------------------------------------
# Clarification wording can be the clinic's own
# ----------------------------------------------------------------------

def test_clinic_authored_clarification_question():
    reading = {"intent": "other", "alternatives": ["booking", "cancel"]}
    text = graph._clarification_question(reading, False, {"msg_clarify_intent": "وش تقصد؟ {options} 🌷"})
    assert text == "وش تقصد؟ حجز موعد جديد، أو إلغاء موعد 🌷"
    assert graph._clarification_question(reading, False, {"msg_clarify_intent": "no placeholder"}).startswith("أكيد")


# ----------------------------------------------------------------------
# Token observability
# ----------------------------------------------------------------------

def test_turn_usage_sums_every_call_and_logs_no_content(caplog):
    first = AIMessage(content="سر المريض", tool_calls=[{"name": "x", "args": {}, "id": "1", "type": "tool_call"}])
    first.usage_metadata = {"input_tokens": 900, "output_tokens": 60, "total_tokens": 960,
                            "input_token_details": {"cache_read": 512}}
    second = AIMessage(content="رد")
    second.usage_metadata = {"input_tokens": 30000, "output_tokens": 120, "total_tokens": 30120}
    with caplog.at_level(logging.INFO, logger="llm_usage"):
        token = llm_usage.start_turn()
        llm_usage.record("understanding", first)
        llm_usage.record("specialist:booking", second)
        summary = llm_usage.end_turn(token, session_id="usage-1")
    assert summary["llm_calls"] == 2 and summary["input_tokens"] == 30900
    assert summary["output_tokens"] == 180 and summary["cached_input_tokens"] == 512
    assert summary["tool_calls"] == 1
    assert summary["by_role"]["specialist"]["input_tokens"] == 30000
    assert "سر المريض" not in caplog.text


def test_a_real_turn_reports_its_calls(session_id, llm, reader, caplog):
    from conftest import send
    reader.table["هلا"] = {"intent": "greeting"}
    llm._responses.append(AIMessage(content="أهلا 🌷"))
    with caplog.at_level(logging.INFO, logger="llm_usage"):
        send(session_id, "هلا")
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("turn_usage ")]
    summary = json.loads(lines[-1][len("turn_usage "):])
    assert summary["by_role"]["understanding"]["calls"] == 1
    assert summary["by_role"]["specialist"]["calls"] == 1


# ----------------------------------------------------------------------
# PRODUCTION (agent-mu1, 2026-09-26 15:48): "مش هعرف اجي بكره" routed to
# cancel correctly, but STEP 1 was composed by the model as two questions
# and trimmed to "إذا نعم، تحب تلغي الموعد برقم الجوال ولا برقم الحجز؟"
# ----------------------------------------------------------------------

def test_indirect_cancellation_gets_the_fixed_step1_question(session_id, llm, reader):
    from conftest import send
    reader.table["مساء الخير"] = {"intent": "greeting"}
    llm._responses.append(AIMessage(content="أهلا 🌷"))
    send(session_id, "مساء الخير")
    calls_before = len(llm.calls)

    reader.table["مش هعرف اجي بكره"] = {"intent": "cancel", "cancel_request": True,
                                         "answer_to_previous_question": True}
    reply = send(session_id, "مش هعرف اجي بكره")["reply"]

    assert "برقم الجوال" in reply and "برقم الحجز" in reply
    assert "إذا نعم" not in reply
    assert len(llm.calls) == calls_before, "STEP 1 is fixed text - no model call"


def test_trimming_a_question_drops_its_if_yes_lead_in():
    text = "تبغى تلغي موعدك القريب؟ إذا نعم، تحب تلغي الموعد برقم الجوال ولا برقم الحجز؟"
    trimmed, removed = graph._strip_extra_questions(text, {})
    assert removed == 1
    assert trimmed == "تحب تلغي الموعد برقم الجوال ولا برقم الحجز؟"
