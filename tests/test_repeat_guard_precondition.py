"""A call that failed only because a precondition was unmet must not block its retry."""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import graph


def _call(call_id, name="get_doctor_schedule_for_booking", args=None):
    return AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": args or {}}])


def test_missing_doctor_result_does_not_count_as_a_repeat():
    messages = [
        HumanMessage(content="عاوز احجز اسنان"),
        _call("1"),
        ToolMessage(content=json.dumps({"status": "missing_doctor"}), name="get_doctor_schedule_for_booking", tool_call_id="1"),
        _call("2", "match_entity_for_booking", {"user_input": "ليلى", "entity_type": "doctor"}),
        ToolMessage(content=json.dumps({"matched": True}), name="match_entity_for_booking", tool_call_id="2"),
    ]
    retry = AIMessage(content="", tool_calls=[{"id": "3", "name": "get_doctor_schedule_for_booking", "args": {}}])
    assert graph._repeated_tool_calls(retry, messages) == []


def test_a_real_answer_still_counts_as_a_repeat():
    messages = [
        HumanMessage(content="مواعيد الدكتور"),
        _call("1"),
        ToolMessage(content=json.dumps({"status": "found", "schedules": []}), name="get_doctor_schedule_for_booking", tool_call_id="1"),
    ]
    retry = AIMessage(content="", tool_calls=[{"id": "2", "name": "get_doctor_schedule_for_booking", "args": {}}])
    assert len(graph._repeated_tool_calls(retry, messages)) == 1


def test_stale_doctor_guard_accepts_the_same_specialty_looked_up_earlier():
    messages = [
        HumanMessage(content="عندي وجع في سني"),
        _call("1", "find_available_doctors", {"specialty_name": "طب اسنان"}),
        ToolMessage(content=json.dumps({"status": "found", "doctors": [{"name": "ليلى الحربي"}]}),
                    name="find_available_doctors", tool_call_id="1"),
        AIMessage(content="تحب أحجز؟"),
        HumanMessage(content="ايوه"),
    ]
    reply = "الدكاترة المتاحين في تخصص طب اسنان:\n1️⃣ د. ليلى الحربي"
    assert graph._reply_shows_doctor_for_service_with_no_lookup_this_turn(reply, {"messages": messages}) is False


def test_stale_doctor_guard_still_flags_a_name_carried_to_another_specialty():
    messages = [
        HumanMessage(content="عندي وجع في سني"),
        _call("1", "find_available_doctors", {"specialty_name": "طب اسنان"}),
        ToolMessage(content=json.dumps({"status": "found", "doctors": [{"name": "ليلى الحربي"}]}),
                    name="find_available_doctors", tool_call_id="1"),
        HumanMessage(content="وايش عندكم تغذية"),
    ]
    reply = "الدكاترة المتاحين لخدمة أخصائي التغذية:\n1️⃣ د. ليلى الحربي"
    assert graph._reply_shows_doctor_for_service_with_no_lookup_this_turn(reply, {"messages": messages}) is True
