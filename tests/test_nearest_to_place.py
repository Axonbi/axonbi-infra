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


def _turn(question, tool_name, payload):
    from langchain_core.messages import ToolMessage
    call = AIMessage(content="", tool_calls=[{"name": tool_name, "args": {}, "id": "c1"}])
    return [HumanMessage(content=question), call,
            ToolMessage(content=json.dumps(payload, ensure_ascii=False), name=tool_name, tool_call_id="c1")]


BRANCH = {"name": "حدائق الاهرام", "address": "122 ج مدخل خوفو الدور الثاني", "phone": "+20 1123212321",
          "working_hours": "يوميًا من 8 صباحًا إلى 10 مساءً", "distance_km": 3.2}


def test_the_card_has_the_branch_details():
    messages = _turn("اقرب فرع للهرم", "geocode_address",
                     {"status": "found", "nearest_branch": {"status": "found", "branches": [BRANCH]}})
    card, name = graph._nearest_branch_card(messages, "ar")
    assert name == "حدائق الاهرام"
    assert card.startswith("أقرب فرع ليك من الهرم هو:")
    for part in ("🏥 الفرع: حدائق الاهرام", "📍 العنوان: 122 ج", "📞 التليفون: +20 1123212321",
                 "⏰ مواعيد العمل:", "📏 المسافة: حوالي 3.2 كم"):
        assert part in card


def test_a_find_nearest_branch_result_makes_a_card_too():
    messages = _turn("1", "find_nearest_branch", {"status": "found", "branches": [BRANCH]})
    assert graph._nearest_branch_card(messages, "ar")[1] == "حدائق الاهرام"


def test_no_card_for_an_unusually_far_branch_or_no_result():
    far = _turn("اقرب فرع للهرم", "find_nearest_branch", {"status": "found", "branches": [BRANCH], "unusually_far": True})
    assert graph._nearest_branch_card(far, "ar") == (None, None)
    assert graph._nearest_branch_card([HumanMessage(content="هاي")], "ar") == (None, None)
