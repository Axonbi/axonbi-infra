"""
A bare number answering "which of these places?" finds the nearest branch
to that place in code (elborgdemo staging 2026-10-05 12:26: "1" was answered
with the same Carrefour list, twice).
"""

import json

from langchain_core.messages import AIMessage, HumanMessage

import graph
import tools

OPTIONS = [
    {"option": 1, "label": "كارفور, طريق النصر, مساكن صقر قريش", "latitude": 30.05, "longitude": 31.33},
    {"option": 2, "label": "كارفور, الطريق الدائري, كايرو فستيفال سيتى", "latitude": 30.02, "longitude": 31.40},
]
REPLY = ("1️⃣ طريق النصر، مساكن صقر قريش\n2️⃣ الطريق الدائري، كايرو فستيفال سيتي\nأي رقم منهم؟")


def _state(answer, reply=REPLY):
    tools._PENDING_PLACES["s"] = [dict(o) for o in OPTIONS]
    return {"session_id": "s", "messages": [AIMessage(content=reply), HumanMessage(content=answer)]}


def teardown_function():
    tools._PENDING_PLACES.clear()


def _fake_nearest(calls):
    def nearest(state, latitude=0.0, longitude=0.0, place_option=0):
        calls.append(place_option)
        return {"status": "found", "nearest": {"name": "جولف كورنر"}}
    return nearest


def test_the_agents_that_answer_location_hold_the_tool():
    assert graph._agent_holds_tools("faq", "find_nearest_branch")


def test_a_number_finds_the_nearest_branch_to_that_place(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.find_nearest_branch, "func", _fake_nearest(calls))
    pair = graph._deterministic_place_pick(_state("1"), "faq")
    assert pair is not None and calls == [1]
    assert json.loads(pair[1].content)["status"] == "found"
    assert "s" not in tools._PENDING_PLACES


def test_a_number_answering_another_list_goes_to_the_model(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.find_nearest_branch, "func", _fake_nearest(calls))
    state = _state("1", reply="1️⃣ حدائق الاهرام\n2️⃣ جولف كورنر\nتحب تعرف تفاصيل؟")
    assert graph._deterministic_place_pick(state, "faq") is None and calls == []


def test_out_of_range_goes_to_the_model(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.find_nearest_branch, "func", _fake_nearest(calls))
    assert graph._deterministic_place_pick(_state("7"), "faq") is None and calls == []
