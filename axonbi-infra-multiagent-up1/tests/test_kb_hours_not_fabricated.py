"""Opening hours quoted from the knowledge base are not a fabricated
appointment (tanasuq-saudi, 2026-10-14 15:11 false positive)."""

import json

from langchain_core.messages import HumanMessage, ToolMessage

import graph

REPLY = (
    "ذا مديكال أفينيو منصة متكاملة للطب الدقيق في جدة. "
    "ساعات العمل من السبت إلى الخميس من 8:00 صباحًا إلى 10:00 مساءً. "
    "تحب تعرف عن الخدمات اللي نقدمها؟"
)


def _state_with_kb():
    kb = json.dumps({"status": "found", "passages": [
        "Working hours: Saturday to Thursday, 8:00 AM - 10:00 PM.",
    ]})
    return {"messages": [
        HumanMessage("معلومات عن مديكال أفينيو"),
        ToolMessage(content=kb, name="answer_hospital_faq", tool_call_id="1"),
    ]}


def test_kb_opening_hours_are_not_an_invented_appointment():
    assert not graph._reply_invents_availability(REPLY, _state_with_kb())


def test_invented_slot_next_to_kb_hours_is_still_caught():
    reply = REPLY + "\nوعندنا موعد يوم الأحد 3:30 مساءً"
    assert graph._reply_invents_availability(reply, _state_with_kb())


def test_hours_without_any_kb_lookup_are_still_caught():
    state = {"messages": [HumanMessage("x")]}
    assert graph._reply_invents_availability(REPLY, state)
