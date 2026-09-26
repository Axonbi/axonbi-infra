"""
reschedule_appointment never moves an appointment onto a time it could
not re-check as still free.

Before: when the looked-up booking record carried no doctorId, the live
re-check was skipped with a warning and the appointment was moved anyway
- onto a slot that may have been taken since it was shown. Now:
  1. the doctor is taken from the slot the patient actually picked
     (get_available_reschedule_slots remembers it, session-only), and
  2. when neither knows the doctor - or the time cannot be read - the
     tool refuses with `cannot_verify_slot` instead of writing blind.
"""

import json
from unittest.mock import patch

from langchain_core.messages import HumanMessage, ToolMessage

import tools

SLOT = {"slotStart": "2026-09-30T13:00:00", "slotEnd": "2026-09-30T13:12:00",
        "date_display": "30/09/2026", "time_display": "4:00 مساءً"}
LIVE_FREE = {"success": True, "status_code": 200, "error": None,
             "data": {"items": [{"slotStart": "2026-09-30T13:00:00Z", "isBooked": False}]}}
LIVE_TAKEN = {"success": True, "status_code": 200, "error": None, "data": {"items": []}}
MOVED = {"success": True, "status_code": 200, "error": None, "data": {"isSuccess": True}}


def _state(session, doctor_on_record=True):
    appointment = {"id": "GUID-1", "ref": "TNS-1", "statusName": "New"}
    if doctor_on_record:
        appointment["doctorId"] = "D-RECORD"
    return {
        "session_id": session, "client_id": "tanasuq-test",
        "templates": {"_doctors_base_url": "http://doctors.test", "_timezone": "Asia/Riyadh"},
        "messages": [
            HumanMessage(content="ابي اعدل موعدي"),
            ToolMessage(content=json.dumps({"status": "found_one", "appointment": appointment}),
                        name="lookup_appointment", tool_call_id="l1"),
        ],
    }


def _lock(session, doctor_id=None):
    slot = dict(SLOT)
    if doctor_id:
        slot["_doctor_id"] = doctor_id
    tools._BOOKING_SESSIONS[session] = {"selected_reschedule_slot": slot,
                                        "last_list": {"entity_type": "slot", "items": [slot]}}


def _move(state):
    return tools.reschedule_appointment.func(
        state=state, booking_id="GUID-1",
        new_time_from=SLOT["slotStart"], new_time_to=SLOT["slotEnd"],
    )


def test_doctor_on_the_record_is_checked_live():
    _lock("rs-1")
    try:
        with patch("api.get_doctor_schedule_slots", return_value=LIVE_FREE) as live, \
             patch("api.reschedule_booking", return_value=MOVED) as write:
            assert _move(_state("rs-1"))["status"] == "success"
        assert live.call_args.kwargs["doctor_ids"] == ["D-RECORD"]
        assert write.call_count == 1
    finally:
        tools._BOOKING_SESSIONS.pop("rs-1", None)


def test_no_doctor_on_the_record_uses_the_doctor_of_the_picked_slot():
    _lock("rs-2", doctor_id="D-SLOT")
    try:
        with patch("api.get_doctor_schedule_slots", return_value=LIVE_FREE) as live, \
             patch("api.reschedule_booking", return_value=MOVED) as write:
            assert _move(_state("rs-2", doctor_on_record=False))["status"] == "success"
        assert live.call_args.kwargs["doctor_ids"] == ["D-SLOT"]
        assert write.call_count == 1
    finally:
        tools._BOOKING_SESSIONS.pop("rs-2", None)


def test_a_slot_taken_since_it_was_shown_is_not_written():
    _lock("rs-3", doctor_id="D-SLOT")
    try:
        with patch("api.get_doctor_schedule_slots", return_value=LIVE_TAKEN), \
             patch("api.reschedule_booking", return_value=MOVED) as write:
            assert _move(_state("rs-3", doctor_on_record=False))["status"] == "slot_unavailable"
        assert write.call_count == 0
    finally:
        tools._BOOKING_SESSIONS.pop("rs-3", None)


def test_nobody_knows_the_doctor_so_nothing_is_written():
    """REGRESSION: this used to log a warning and move the appointment."""
    _lock("rs-4")
    try:
        with patch("api.get_doctor_schedule_slots", return_value=LIVE_FREE) as live, \
             patch("api.reschedule_booking", return_value=MOVED) as write:
            assert _move(_state("rs-4", doctor_on_record=False))["status"] == "cannot_verify_slot"
        assert live.call_count == 0 and write.call_count == 0
    finally:
        tools._BOOKING_SESSIONS.pop("rs-4", None)


def test_an_unreadable_time_is_not_written():
    tools._BOOKING_SESSIONS["rs-5"] = {"selected_reschedule_slot": {**SLOT, "slotStart": "next thursday"}}
    try:
        with patch("api.reschedule_booking", return_value=MOVED) as write:
            result = tools.reschedule_appointment.func(
                state=_state("rs-5"), booking_id="GUID-1",
                new_time_from="next thursday", new_time_to="")
        assert result["status"] == "cannot_verify_slot"
        assert write.call_count == 0
    finally:
        tools._BOOKING_SESSIONS.pop("rs-5", None)


def test_the_listing_remembers_the_doctor_but_never_shows_it():
    state = _state("rs-6")
    listed = {"success": True, "status_code": 200, "error": None,
              "data": {"items": [{"slotStart": "2026-09-30T13:00:00Z", "slotEnd": "2026-09-30T13:12:00Z",
                                  "isBooked": False, "doctorName": "د. عمر"}]}}
    try:
        with patch("tools._resolve_doctor_id", return_value={"status": "found", "doctor_id": "D-LIST"}), \
             patch("api.get_doctor_schedule_slots", return_value=listed):
            shown = tools.get_available_reschedule_slots.func(
                state=state, ref_number="TNS-1",
                from_date="2026-09-30T00:00:00+03:00", to_date="2026-09-30T23:59:00+03:00")
        assert shown["status"] == "found"
        assert all("_doctor_id" not in slot for slot in shown["slots"]), "internal id shown to the model"
        remembered = tools._BOOKING_SESSIONS["rs-6"]["last_list"]["items"]
        assert remembered and all(item["_doctor_id"] == "D-LIST" for item in remembered)

        picked = tools.select_reschedule_slot.func(state=state, user_input="1")
        assert picked["status"] == "selected" and "_doctor_id" not in picked["slot"]
        assert tools._BOOKING_SESSIONS["rs-6"]["selected_reschedule_slot"]["_doctor_id"] == "D-LIST"
    finally:
        tools._BOOKING_SESSIONS.pop("rs-6", None)
