"""
tanasuq-production 2026-10-18 14:55 and 18:45: two reschedules refused at
the patient's final "نعم". The lookup was three turns back, its booking id
had been shortened out of the model's view, and the model sent
`booking_id='⟦ID_1⟧'` - no id this conversation knew.
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import graph
import pii_vault
import tools

GUID = "5b0f2c1e-9a7d-4c3e-8f11-2a6b9d0c7e44"
REF = "GuestBookingNum-2026-10-12-208"
APPOINTMENT = {"id": GUID, "ref": REF, "doctorName": "العنود الخليفة",
               "branchName": "المنار", "date_display": "18/10/2026", "time_display": "7:00 مساءً"}


def _lookup_state(*appointments):
    payload = {"status": "found", "appointments": list(appointments)}
    return {"session_id": "966555370295-TEST", "messages": [
        HumanMessage(content="ايه"),
        AIMessage(content="", tool_calls=[{"name": "lookup_appointment", "args": {}, "id": "t1"}]),
        ToolMessage(content=json.dumps(payload, ensure_ascii=False), name="lookup_appointment", tool_call_id="t1"),
    ]}


def test_one_booking_on_file_resolves_a_placeholder_id():
    out = tools._resolve_booking_guid(_lookup_state(APPOINTMENT), "⟦ID_1⟧")
    assert out == {"status": "resolved", "booking_id": GUID}


def test_two_bookings_on_file_still_refuse_an_unknown_id():
    other = {**APPOINTMENT, "id": "11111111-2222-3333-4444-555555555555", "ref": "GuestBookingNum-2026-10-12-209"}
    out = tools._resolve_booking_guid(_lookup_state(APPOINTMENT, other), "⟦ID_1⟧")
    assert out == {"status": "not_looked_up"}


def test_compaction_keeps_a_bookings_id_and_drops_others():
    payload = {"status": "found", "appointments": [APPOINTMENT],
               "note": "x" * 400,
               "doctors": [{"id": "doc-guid-0001", "name": "عمر المديفر"}]}
    out = graph._compact_tool_payload(json.dumps(payload, ensure_ascii=False))
    assert GUID in out
    assert "doc-guid-0001" not in out


def test_the_vault_tokenises_a_booking_id_and_gives_it_back():
    vault = {"values": {}, "tokens": {}, "counts": {}}
    pii_vault._walk_fields(vault, {"appointments": [APPOINTMENT], "doctorId": "doc-guid-0001"})
    token = vault["values"].get(GUID)
    assert token and token.startswith("⟦ID_")
    assert "doc-guid-0001" not in vault["values"]
    assert pii_vault.unmask_text(vault, token, {}) == GUID
