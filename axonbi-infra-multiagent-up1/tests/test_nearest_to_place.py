"""
"أقرب فرع لـ/من <مكان>" is looked up on the map in code, a found place
carries its nearest branch, and the reply says the branch that was found
(ported from elborgdemo, staging 2026-10-05). Al-Alem Dental has two
branches: Sheikh Zayed and Sheraton - Heliopolis.
"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import config
import graph
import tools

API_BRANCHES = [
    {"name": "Sheikh Zayed", "altName": "الشيخ زايد", "address": "Beverly Hills, The Polygon HQ, Bldg 3",
     "mobile": "15180"},
    {"name": "Sheraton - Heliopolis", "altName": "شيراتون - مصر الجديدة",
     "address": "1 Ankara St, Cairo Complex Mall", "mobile": "15180"},
]


@pytest.mark.parametrize("text,place", [
    ("اقرب فرع لسيتي ستارز", "سيتي ستارز"),
    ("ايه اقرب فرع للتجمع", "التجمع"),
    ("أقرب فرع عند مول العرب؟", "مول العرب"),
    ("ايه اقرب فرع ليا من مول العرب", "مول العرب"),
    ("انا قاعد في فندق الماسة ايه اقرب فرع؟", "فندق الماسة"),
    ("انا ساكن في مدينة نصر، فين اقرب فرع", "مدينة نصر"),
    ("nearest branch to Cairo Festival City", "Cairo Festival City"),
    ("Which is the closest branch to Mall of Arabia?", "Mall of Arabia"),
    ("ايه اقرب فرع ليه", None),
    ("ايه اقرب فرع ليا", None),
    ("nearest branch to me", None),
    ("انا جاي من عند مول العرب", None),   # no nearest-branch question in sight
    ("عاوزه احجز", None),
])
def test_the_place_is_read_from_the_question(text, place):
    assert graph._place_in_nearest_question(text) == place


def test_where_i_am_answers_a_nearest_branch_exchange():
    asked = "تمام، قولي انت في انهي منطقة عشان أقولك أقرب فرع ليك؟"
    assert graph._place_in_nearest_question("انا جاي من عند مول العرب", asked) == "مول العرب"


def test_the_coordinates_file_has_both_branches():
    geo = config.load_branches_geo()
    for name in ("Sheikh Zayed", "Sheraton - Heliopolis", "الشيخ زايد", "شيراتون - مصر الجديدة"):
        assert name in geo


def _real_branches(monkeypatch, items=API_BRANCHES):
    monkeypatch.setattr(tools, "_doctors_base_url", lambda state: "http://booking")
    monkeypatch.setattr(tools.api, "get_branches", lambda *a, **k: {"success": True, "data": {"items": items}})


@pytest.mark.parametrize("lat,lon,nearest", [
    (30.0287, 31.4085, "شيراتون - مصر الجديدة"),   # Cairo Festival City
    (30.0728, 31.3459, "شيراتون - مصر الجديدة"),   # City Stars
    (30.0838, 31.3426, "شيراتون - مصر الجديدة"),   # Al Masa Hotel, Nasr City
    (30.0067, 30.9737, "الشيخ زايد"),               # Mall of Arabia
])
def test_the_nearest_of_the_two_branches(monkeypatch, lat, lon, nearest):
    _real_branches(monkeypatch)
    result = tools.find_nearest_branch.func({"session_id": "geo", "templates": {}}, latitude=lat, longitude=lon)
    assert result["status"] == "found" and not result.get("unusually_far")
    assert [b["name"] for b in result["branches"]][0] == nearest
    assert len(result["branches"]) == 2 and result["branches"][0]["distance_km"] < 15


def test_english_api_names_without_arabic_still_match(monkeypatch):
    _real_branches(monkeypatch, [{"name": "Sheikh Zayed Branch"}, {"name": "Sheraton-Heliopolis"}])
    result = tools.find_nearest_branch.func({"session_id": "geo", "templates": {}}, latitude=30.0287, longitude=31.4085)
    assert [b["name"] for b in result["branches"]] == ["Sheraton-Heliopolis", "Sheikh Zayed Branch"]


def test_a_place_far_from_cairo_is_flagged(monkeypatch):
    _real_branches(monkeypatch)
    result = tools.find_nearest_branch.func({"session_id": "geo", "templates": {}}, latitude=31.2001, longitude=29.9187)
    assert result["unusually_far"] is True


def test_the_question_is_answered_from_the_map(monkeypatch):
    calls = []

    def geocode(state, address):
        calls.append(address)
        return {"status": "found", "latitude": 30.0, "longitude": 31.0, "display_name": "x",
                "nearest_branch": {"status": "found", "branches": [{"name": "الشيخ زايد"}]}}

    monkeypatch.setattr(tools.geocode_address, "func", geocode)
    state = {"session_id": "s", "messages": [AIMessage(content="أهلاً"), HumanMessage(content="اقرب فرع لسيتي ستارز")]}
    pair = graph._deterministic_nearest_to_place(state, "faq")
    assert calls == ["سيتي ستارز"]
    assert json.loads(pair[1].content)["nearest_branch"]["branches"][0]["name"] == "الشيخ زايد"


def test_the_place_is_looked_up_once_per_turn(monkeypatch):
    monkeypatch.setattr(tools.geocode_address, "func", lambda state, address: pytest.fail("looked up twice"))
    messages = _turn("اقرب فرع لسيتي ستارز", "geocode_address", {"status": "found"})
    assert graph._deterministic_nearest_to_place({"session_id": "s", "messages": messages}, "faq") is None


def test_a_place_the_map_does_not_know_goes_to_the_model(monkeypatch):
    monkeypatch.setattr(tools.geocode_address, "func", lambda state, address: {"status": "not_found"})
    state = {"session_id": "s", "messages": [HumanMessage(content="اقرب فرع من مكان مش موجود")]}
    assert graph._deterministic_nearest_to_place(state, "faq") is None


def test_an_agent_without_the_tools_is_left_alone(monkeypatch):
    monkeypatch.setattr(tools.geocode_address, "func", lambda state, address: pytest.fail("called"))
    state = {"session_id": "s", "messages": [HumanMessage(content="اقرب فرع لسيتي ستارز")]}
    assert graph._deterministic_nearest_to_place(state, "complaint") is None


def test_a_found_place_carries_its_nearest_branch(monkeypatch):
    monkeypatch.setattr(tools, "_geocode_candidates", lambda *a, **k: [
        {"display_name": "Cairo Festival City Mall, New Cairo", "lat": "30.0287", "lon": "31.4085",
         "class": "shop", "addresstype": "mall"}])
    _real_branches(monkeypatch)
    result = tools.geocode_address.func({"session_id": "s", "templates": {}}, address="Cairo Festival City")
    assert result["status"] == "found"
    assert result["nearest_branch"]["branches"][0]["name"] == "شيراتون - مصر الجديدة"


def _turn(question, tool_name, payload, extra=()):
    call = AIMessage(content="", tool_calls=[{"name": tool_name, "args": {}, "id": "c1"}])
    return [HumanMessage(content=question), call,
            ToolMessage(content=json.dumps(payload, ensure_ascii=False), name=tool_name, tool_call_id="c1"),
            *extra]


NEAREST = {"name": "شيراتون - مصر الجديدة", "address": "1 ش أنقرة، كايرو كومبليكس مول", "phone": "15180",
           "distance_km": 3.2}
FARTHER = {"name": "الشيخ زايد", "address": "بيفرلي هيلز", "phone": "15180", "distance_km": 41.7}


def test_the_card_has_the_branch_details_and_offers_the_pin():
    messages = _turn("اقرب فرع لسيتي ستارز", "geocode_address",
                     {"status": "found", "nearest_branch": {"status": "found", "branches": [NEAREST, FARTHER]}})
    card, name = graph._nearest_branch_card(messages, "ar")
    assert name == "شيراتون - مصر الجديدة"
    assert card.startswith("أقرب فرع ليك من سيتي ستارز هو:")
    for part in ("🏥 الفرع: شيراتون - مصر الجديدة", "📍 العنوان: 1 ش أنقرة", "📞 التليفون: 15180",
                 "📏 المسافة: حوالي 3.2 كم", "لوكيشن"):
        assert part in card


def test_once_the_pin_went_out_the_card_offers_booking():
    pin = ToolMessage(content=json.dumps({"status": "location_requested", "branch_name": NEAREST["name"]},
                                         ensure_ascii=False), name="share_branch_location", tool_call_id="c2")
    messages = _turn("nearest branch to Cairo Festival City", "find_nearest_branch",
                     {"status": "found", "branches": [NEAREST, FARTHER]}, extra=[pin])
    card, _ = graph._nearest_branch_card(messages, "en")
    assert card.startswith("The nearest branch to you from Cairo Festival City is:")
    assert "sent you its location" in card and "book" in card


def test_no_card_for_an_unusually_far_branch_or_no_result():
    far = _turn("اقرب فرع لاسكندرية", "find_nearest_branch",
                {"status": "found", "branches": [FARTHER], "unusually_far": True})
    assert graph._nearest_branch_card(far, "ar") == (None, None)
    assert graph._nearest_branch_card([HumanMessage(content="هاي")], "ar") == (None, None)


def test_the_reply_must_lead_with_the_branch_found():
    branches = [NEAREST, FARTHER]
    assert graph._reply_names_nearest_first("أقرب فرع ليك هو فرع شيراتون - مصر الجديدة، على بعد 3 كم", branches)
    assert graph._reply_names_nearest_first(
        "أقرب فرع: شيراتون - مصر الجديدة (3.2 كم)، وفرع الشيخ زايد على بعد 41 كم", branches)
    assert not graph._reply_names_nearest_first("أقرب فرع ليك هو فرع الشيخ زايد", branches)
    assert not graph._reply_names_nearest_first("أهلاً بيك في مركز العالم! أقدر أساعدك إزاي؟", branches)


def test_a_whole_turn_answers_the_nearest_branch_and_then_sends_the_pin(session_id, llm, reader, monkeypatch):
    from unittest.mock import patch

    import main
    from conftest import TANASUQ

    client = {**TANASUQ, "timezone": "Africa/Cairo", "Dialect": "Egyptian"}
    monkeypatch.setattr(tools, "_geocode_candidates", lambda *a, **k: [
        {"display_name": "فندق الماسة, مدينة نصر, القاهرة", "lat": "30.0838", "lon": "31.3426",
         "class": "tourism", "addresstype": "hotel"}])
    _real_branches(monkeypatch)

    def say(text):
        with patch.multiple(graph, _VERIFIERS_SAFETY_STRICT=False, _VERIFIERS_FLOW_STRICT=False):
            return main.send_message_with_signals(client["client_id"], session_id, text,
                                                  channel_phone="201000000001", client_config=client)

    question = "انا قاعد في فندق الماسة ايه اقرب فرع؟"
    reader.table[question] = {"intent": "faq", "confidence": 1.0, "asks_location": True}
    # The draft that went out on elborgdemo: no branch at all.
    llm._responses.append(AIMessage(content="أهلاً بيك! أقدر أساعدك إزاي؟"))
    first = say(question)
    assert "🏥 الفرع: شيراتون - مصر الجديدة" in first["reply"] and "📏 المسافة: حوالي" in first["reply"]
    assert "geocode_address" in str([getattr(m, "name", None) for m in llm.calls[0]])

    reader.table["اه"] = {"intent": "answer", "confidence": 1.0, "confirms": True}
    llm._responses.extend([
        AIMessage(content="", tool_calls=[{"name": "share_branch_location",
                                           "args": {"branch_name": "شيراتون - مصر الجديدة"},
                                           "id": "pin1", "type": "tool_call"}]),
        AIMessage(content="بعتلك لوكيشن فرع شيراتون - مصر الجديدة 📍"),
    ])
    second = say("اه")
    assert second["location"] is True and second["branch_name"] == "شيراتون - مصر الجديدة"


def test_a_draft_that_names_the_branch_in_its_own_words_still_becomes_the_card(session_id, llm, reader, monkeypatch):
    # The card is the one shape for this answer - a free-text draft that
    # does lead with the right branch is replaced too.
    from unittest.mock import patch

    import main
    from conftest import TANASUQ

    client = {**TANASUQ, "timezone": "Africa/Cairo", "Dialect": "Egyptian"}
    monkeypatch.setattr(tools, "_geocode_candidates", lambda *a, **k: [
        {"display_name": "فندق الماسة, مدينة نصر, القاهرة", "lat": "30.0838", "lon": "31.3426",
         "class": "tourism", "addresstype": "hotel"}])
    _real_branches(monkeypatch)

    question = "انا قاعد في فندق الماسة ايه اقرب فرع؟"
    reader.table[question] = {"intent": "faq", "confidence": 1.0, "asks_location": True}
    llm._responses.append(AIMessage(content="أقرب فرع لك هو فرع شيراتون - مصر الجديدة، ويبعد حوالي 4 كم."))
    with patch.multiple(graph, _VERIFIERS_SAFETY_STRICT=False, _VERIFIERS_FLOW_STRICT=False):
        reply = main.send_message_with_signals(client["client_id"], session_id, question,
                                               channel_phone="201000000001", client_config=client)["reply"]
    assert "🏥 الفرع: شيراتون - مصر الجديدة" in reply and "📏 المسافة: حوالي" in reply
    assert "أقرب فرع لك هو فرع" not in reply


def test_the_card_is_not_read_as_an_invented_branch():
    messages = _turn("انا قاعد في فندق الماسة ايه اقرب فرع؟", "geocode_address",
                     {"status": "found", "display_name": "فندق الماسة, مدينة نصر",
                      "nearest_branch": {"status": "found", "branches": [NEAREST, FARTHER]}})
    card, _ = graph._nearest_branch_card(messages, "ar")
    assert graph._find_invented_branches(card, {"session_id": "card", "messages": messages, "templates": {}}) == []
