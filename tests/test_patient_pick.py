"""
A bare number answering the registered-names list picks that name in code
(elborgdemo staging 2026-10-05: "7" was answered with "give me your full
name" twice).
"""

import json

from langchain_core.messages import AIMessage, HumanMessage

import graph
import tools

NAMES = ["Fatma Abdullah", "QA Fixture Patient", "Fatma Nasser Ellithy"]
LIST_REPLY = ("فيه كذا اسم مسجلين على الرقم ده:\n"
              "1️⃣ Fatma Abdullah\n2️⃣ QA Fixture Patient\n3️⃣ Fatma Nasser Ellithy\n\n"
              "تحب تختاري أي اسم منهم للحجز؟")


def _state(session_id, answer, reply=LIST_REPLY):
    session = tools._get_booking_session(session_id)
    session["patient_choices"] = [{"patientFullName": n, "email": None} for n in NAMES]
    session["booking_phone"] = "+201000000001"
    return {"session_id": session_id,
            "messages": [AIMessage(content=reply), HumanMessage(content=answer)]}


def teardown_function():
    tools._BOOKING_SESSIONS.clear()


def test_the_booking_agent_holds_the_lookup():
    assert graph._agent_holds_tools("booking", "get_patient_info")


def test_a_number_picks_that_name():
    pair = graph._deterministic_patient_pick(_state("s1", "3"), "booking")
    assert pair is not None
    payload = json.loads(pair[1].content)
    assert payload["status"] == "found"
    assert payload["patientFullName"] == "Fatma Nasser Ellithy"
    assert payload["mobileNumber"] == "+201000000001"
    assert "patient_choices" not in tools._get_booking_session("s1")


def test_an_arabic_digit_picks_too():
    pair = graph._deterministic_patient_pick(_state("s2", "٢"), "booking")
    assert json.loads(pair[1].content)["patientFullName"] == "QA Fixture Patient"


def test_out_of_range_goes_to_the_model():
    assert graph._deterministic_patient_pick(_state("s3", "7"), "booking") is None


def test_a_typed_name_goes_to_the_model():
    assert graph._deterministic_patient_pick(_state("s4", "Fatma Nasser Ellithy"), "booking") is None


def test_a_list_shown_in_another_order_goes_to_the_model():
    reordered = "1️⃣ Fatma Nasser Ellithy\n2️⃣ Fatma Abdullah\n3️⃣ QA Fixture Patient"
    assert graph._deterministic_patient_pick(_state("s5", "1", reordered), "booking") is None


def test_no_list_remembered_does_nothing():
    state = {"session_id": "s6", "messages": [AIMessage(content=LIST_REPLY), HumanMessage(content="1")]}
    assert graph._deterministic_patient_pick(state, "booking") is None


def test_the_lookup_remembers_several_names_and_forgets_on_one(monkeypatch):
    state = {"session_id": "s7", "channel_phone": "201000000001", "templates": {"_doctors_base_url": "https://p"}}
    session = tools._get_booking_session("s7")
    session["slots_shown"] = True
    monkeypatch.setattr(tools, "_phone_is_verified", lambda *a, **k: True)
    rows = [{"patientFullName": n, "email": None} for n in NAMES]
    monkeypatch.setattr(tools.api, "get_patient_info",
                        lambda *a, **k: {"success": True, "data": {"items": rows, "totalCount": 3}})
    assert tools.get_patient_info.func(state, mobile_number="+201000000001")["status"] == "found_multiple"
    assert [c["patientFullName"] for c in session["patient_choices"]] == NAMES

    monkeypatch.setattr(tools.api, "get_patient_info",
                        lambda *a, **k: {"success": True, "data": {"items": rows[:1], "totalCount": 1}})
    assert tools.get_patient_info.func(state, mobile_number="+201000000001")["status"] == "found"
    assert "patient_choices" not in session
