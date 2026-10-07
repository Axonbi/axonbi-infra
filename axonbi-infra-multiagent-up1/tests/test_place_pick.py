"""
A bare number answering "which of these places?" finds the nearest branch
to that place in code (elborgdemo staging 2026-10-05: "1" was answered with
the same Carrefour list, twice), and find_nearest_branch takes the chosen
place's coordinates from the offered list.
"""

import json

from langchain_core.messages import AIMessage, HumanMessage

import graph
import tools

OPTIONS = [
    {"option": 1, "label": "كارفور, طريق النصر, مدينة نصر", "latitude": 30.05, "longitude": 31.33},
    {"option": 2, "label": "كارفور, محور 26 يوليو, الشيخ زايد", "latitude": 30.05, "longitude": 30.97},
]
REPLY = "1️⃣ طريق النصر، مدينة نصر\n2️⃣ محور 26 يوليو، الشيخ زايد\nأي واحد فيهم؟"


def _state(answer, reply=REPLY):
    tools._PENDING_PLACES["s"] = [dict(o) for o in OPTIONS]
    return {"session_id": "s", "messages": [AIMessage(content=reply), HumanMessage(content=answer)]}


def teardown_function():
    tools._PENDING_PLACES.clear()


def _fake_nearest(calls):
    def nearest(state, latitude=0.0, longitude=0.0, place_option=0):
        calls.append(place_option)
        return {"status": "found", "branches": [{"name": "Sheraton - Heliopolis", "distance_km": 5.1}]}
    return nearest


def test_the_agents_that_answer_location_hold_the_tools():
    for agent in ("faq", "booking", "medical", "concierge"):
        assert graph._agent_holds_tools(agent, "geocode_address", "find_nearest_branch"), agent


def test_a_number_finds_the_nearest_branch_to_that_place(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.find_nearest_branch, "func", _fake_nearest(calls))
    pair = graph._deterministic_place_pick(_state("1"), "faq")
    assert pair is not None and calls == [1]
    assert json.loads(pair[1].content)["status"] == "found"
    assert "s" not in tools._PENDING_PLACES


def test_an_arabic_digit_picks_too(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.find_nearest_branch, "func", _fake_nearest(calls))
    assert graph._deterministic_place_pick(_state("٢"), "faq") is not None and calls == [2]


def test_a_number_answering_another_list_goes_to_the_model(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.find_nearest_branch, "func", _fake_nearest(calls))
    state = _state("1", reply="1️⃣ تقويم الأسنان\n2️⃣ زراعة الأسنان\nتحب تعرف تفاصيل؟")
    assert graph._deterministic_place_pick(state, "faq") is None and calls == []


def test_out_of_range_goes_to_the_model(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.find_nearest_branch, "func", _fake_nearest(calls))
    assert graph._deterministic_place_pick(_state("7"), "faq") is None and calls == []


def test_place_option_measures_from_the_chosen_place(monkeypatch):
    tools._PENDING_PLACES["s"] = [dict(o) for o in OPTIONS]
    monkeypatch.setattr(tools, "_doctors_base_url", lambda state: "http://booking")
    monkeypatch.setattr(tools.api, "get_branches", lambda *a, **k: {"success": True, "data": {"items": [
        {"name": "Sheikh Zayed", "address": "Beverly Hills"},
        {"name": "Sheraton - Heliopolis", "address": "Cairo Complex Mall"},
    ]}})
    state = {"session_id": "s", "templates": {}}
    assert tools.find_nearest_branch.func(state, place_option=2)["branches"][0]["name"] == "Sheikh Zayed"
    assert tools.find_nearest_branch.func(state, place_option=1)["branches"][0]["name"] == "Sheraton - Heliopolis"
    # Coordinates that are none of the offered places: ask again, never guess.
    assert tools.find_nearest_branch.func(state, latitude=30.0444, longitude=31.2357)["status"] == "needs_place_choice"


def test_a_place_number_with_no_list_measures_nothing():
    assert tools.find_nearest_branch.func({"session_id": "s", "templates": {}}, place_option=1)["status"] == "not_found"
