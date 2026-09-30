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
