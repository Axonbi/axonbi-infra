"""
"١٠ ونص" is 10:30, and answering the reschedule slot list with it (or with
a bare number) is resolved in code (lab-ezz staging 2026-10-07 12:52:39:
the list showed "5️⃣ 10:30 صباحًا", "١٠ ونص" was answered with "10:30 مش
متاح", with no tool call).
"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import graph
import tools


@pytest.mark.parametrize("text, hour, minute", [
    ("١٠ ونص", 10, 30),
    ("الساعة 10 ونصف", 10, 30),
    ("9 وربع الصبح", 9, 15),
    ("4 وتلت", 4, 20),
    ("11 الا ربع", 10, 45),
    ("11 إلا ربع", 10, 45),
    ("1 الا ربع", 12, 45),
    ("10:30", 10, 30),
])
def test_spoken_quarters_are_minutes(text, hour, minute):
    parsed = tools._parse_clock_time(text)
    assert (parsed["hour"], parsed["minute"]) == (hour, minute)


def test_a_bare_hour_is_still_any_slot_in_that_hour():
    assert tools._parse_clock_time("الساعة 10")["minute"] is None


@pytest.mark.parametrize("text", ["١٠ ونص", "10 وربع", "الساعة 11 الا ربع"])
def test_a_spoken_quarter_counts_as_a_named_time(text):
    assert graph._requested_clock_time_in_latest_human([HumanMessage(content=text)])


def _slot(hour, minute, branch="b1"):
    start = f"2026-10-10T{hour:02d}:{minute:02d}:00"
    return {"slotStart": start, "slotEnd": start, "_localStart": start,
            "time_display": f"{hour}:{minute:02d} صباحًا", "date_display": "10/10/2026",
            "branchId": branch, "branchName": "حدائق الاهرام"}


SHOWN = [_slot(9, 30), _slot(10, 0), _slot(10, 30), _slot(11, 0)]
ARGS = {"ref_number": "GBN-1", "from_date": "2026-10-10T00:00:00", "to_date": "2026-10-10T23:59:00"}


def _state(answer, last_tool="get_available_reschedule_slots"):
    tools._get_booking_session("s")["last_list"] = {"entity_type": "slot", "items": [dict(s) for s in SHOWN]}
    return {"session_id": "s", "messages": [
        AIMessage(content="", tool_calls=[{"name": last_tool, "args": ARGS, "id": "c1"}]),
        ToolMessage(content="{}", name=last_tool, tool_call_id="c1"),
        AIMessage(content="1️⃣ 9:30\n2️⃣ 10:00\n3️⃣ 10:30\n4️⃣ 11:00\nأي رقم؟"),
        HumanMessage(content=answer),
    ]}


def teardown_function():
    tools._BOOKING_SESSIONS.pop("s", None)


def _fake_slots(calls, slots):
    def fetch(state, **kwargs):
        calls.append(kwargs)
        return {"status": "found", "slots": [dict(s) for s in slots]}
    return fetch


@pytest.mark.parametrize("answer", ["١٠ ونص", "3", "٣", "10:30"])
def test_the_pick_is_resolved_and_rechecked_in_code(monkeypatch, answer):
    calls = []
    monkeypatch.setattr(tools.get_available_reschedule_slots, "func", _fake_slots(calls, SHOWN))
    pair = graph._deterministic_reschedule_slot_pick(_state(answer), "reschedule")
    assert pair is not None and calls == [ARGS]
    payload = json.loads(pair[1].content)
    assert payload["patient_picked"]["position_in_list_shown"] == 3
    assert payload["patient_picked"]["slotStart"] == "2026-10-10T10:30:00"


def test_a_slot_taken_since_is_said_so(monkeypatch):
    monkeypatch.setattr(tools.get_available_reschedule_slots, "func",
                        _fake_slots([], [s for s in SHOWN if s["slotStart"] != SHOWN[2]["slotStart"]]))
    payload = json.loads(graph._deterministic_reschedule_slot_pick(_state("3"), "reschedule")[1].content)
    assert "patient_picked" not in payload and payload["patient_picked_taken"]["position_in_list_shown"] == 3


def test_a_new_booking_list_is_not_touched(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.get_available_reschedule_slots, "func", _fake_slots(calls, SHOWN))
    state = _state("3", last_tool="get_available_slots_for_booking")
    assert graph._deterministic_reschedule_slot_pick(state, "reschedule") is None and calls == []


@pytest.mark.parametrize("answer", ["9", "الخميس", "الساعة 2"])
def test_no_single_slot_goes_to_the_model(monkeypatch, answer):
    calls = []
    monkeypatch.setattr(tools.get_available_reschedule_slots, "func", _fake_slots(calls, SHOWN))
    assert graph._deterministic_reschedule_slot_pick(_state(answer), "reschedule") is None and calls == []


def test_only_the_reschedule_agent(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.get_available_reschedule_slots, "func", _fake_slots(calls, SHOWN))
    assert graph._deterministic_reschedule_slot_pick(_state("3"), "booking") is None and calls == []
