"""
A bare number is resolved against the remembered test list only when the
reply the patient answered shows it (elborgdemo staging 2026-10-05: "1" to
a branch list booked the first TEST of an older list).
"""

from langchain_core.messages import AIMessage, HumanMessage

import graph
import tools

TESTS = [{"id": "d1", "name": "تحليل البول الكامل"}, {"id": "d2", "name": "تحليل سكر صائم"}]


def _state(session_id, reply, answer="1"):
    session = tools._get_booking_session(session_id)
    session["last_list"] = {"entity_type": "lab_test_doctor", "items": list(TESTS)}
    return {"session_id": session_id,
            "messages": [AIMessage(content=reply), HumanMessage(content=answer)]}


def teardown_function():
    tools._BOOKING_SESSIONS.clear()


def _fake_match(calls):
    def match(state, user_input, entity_type):
        calls.append(user_input)
        return {"matched": True, "doctor_id": "d1"}
    return match


def test_a_number_answering_another_list_goes_to_the_model(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.match_entity_for_booking, "func", _fake_match(calls))
    state = _state("s1", "دي الفروع المتاحة عندنا:\n1️⃣ مصر الجديدة\n\nحابب تحجز في فرع معين منهم؟")
    assert graph._deterministic_service_pick(state, "booking") is None
    assert calls == []


def test_a_number_answering_the_test_list_is_resolved(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.match_entity_for_booking, "func", _fake_match(calls))
    state = _state("s2", "التحاليل المتاحة:\n1️⃣ تحليل البول الكامل\n2️⃣ تحليل سكر صائم\nتحب أي واحد؟")
    assert graph._deterministic_service_pick(state, "booking") is not None
    assert calls == ["1"]


def test_spelling_variants_still_count(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.match_entity_for_booking, "func", _fake_match(calls))
    state = _state("s3", "1️⃣ تحليل البول الكامل\n2️⃣ تحليل سكر صايم", answer="٢")
    assert graph._deterministic_service_pick(state, "booking") is not None


def test_out_of_range_goes_to_the_model(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.match_entity_for_booking, "func", _fake_match(calls))
    state = _state("s4", "1️⃣ تحليل البول الكامل\n2️⃣ تحليل سكر صائم", answer="5")
    assert graph._deterministic_service_pick(state, "booking") is None


def test_an_article_added_by_the_model_still_counts(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.match_entity_for_booking, "func", _fake_match(calls))
    state = _state("s5", "1️⃣ تحليل البول الكامل\n2️⃣ تحليل السكر الصائم", answer="2")
    assert graph._deterministic_service_pick(state, "booking") is not None
