"""
"أقرب فرع لـ/من <مكان>" is looked up on the map in code, and a found place
carries its nearest branch (elborgdemo staging 2026-10-05 12:11/12:27/12:28).
"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import graph
import tools


@pytest.mark.parametrize("text,place", [
    ("ايه اقرب فرع للهرم", "الهرم"),
    ("اقرب فرع من مول المرشدي", "مول المرشدي"),
    ("ايه اقرب فرع لمول المرشدي", "مول المرشدي"),
    ("اقرب فرع  للمعمل من كارفور", "كارفور"),
    ("أقرب فرع عند سيتي ستارز؟", "سيتي ستارز"),
    ("ايه اقرب فرع ليه", None),
    ("ايه اقرب فرع ليا", None),
    ("عاوزه احجز", None),
])
def test_the_place_is_read_from_the_question(text, place):
    assert graph._place_in_nearest_question(text) == place


def test_the_question_is_answered_from_the_map(monkeypatch):
    calls = []
    def geocode(state, address):
        calls.append(address)
        return {"status": "found", "latitude": 30.0, "longitude": 31.0, "display_name": "x",
                "nearest_branch": {"status": "found", "nearest": {"name": "فرع اكتوبر"}}}
    monkeypatch.setattr(tools.geocode_address, "func", geocode)
    state = {"session_id": "s", "messages": [AIMessage(content="أهلاً"), HumanMessage(content="ايه اقرب فرع للهرم")]}
    pair = graph._deterministic_nearest_to_place(state, "faq")
    assert calls == ["الهرم"] and json.loads(pair[1].content)["nearest_branch"]["nearest"]["name"] == "فرع اكتوبر"


def test_a_place_the_map_does_not_know_goes_to_the_model(monkeypatch):
    monkeypatch.setattr(tools.geocode_address, "func", lambda state, address: {"status": "not_found"})
    state = {"session_id": "s", "messages": [HumanMessage(content="اقرب فرع من مكان مش موجود")]}
    assert graph._deterministic_nearest_to_place(state, "faq") is None


def test_a_found_place_carries_its_nearest_branch(monkeypatch):
    monkeypatch.setattr(tools, "_geocode_candidates", lambda *a, **k: [
        {"display_name": "المرشدى مول, الجيزة", "lat": "30.01", "lon": "31.0", "class": "shop", "addresstype": "mall"}])
    monkeypatch.setattr(tools, "_client_branches_viewbox", lambda: None)
    monkeypatch.setattr(tools.find_nearest_branch, "func",
                        lambda state, latitude=0.0, longitude=0.0, place_option=0: {"status": "found", "nearest": {"name": "فرع اكتوبر"}})
    result = tools.geocode_address.func({"session_id": "s", "templates": {}}, address="مول المرشدي")
    assert result["status"] == "found" and result["nearest_branch"]["nearest"]["name"] == "فرع اكتوبر"
