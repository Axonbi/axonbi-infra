"""A slot pick still resolves when the shown list is no longer remembered
(tanasuq-saudi, 2026-10-14 16:03 - "3" looped back to the same list)."""

from types import SimpleNamespace

import tools

SLOTS = [
    {"slotStart": f"2026-10-28T18:{m:02d}:00+03:00", "slotEnd": "", "date_display": "28/10/2026",
     "weekday_display": "الأربعاء", "time_display": f"6:{m:02d} مساءً", "serviceName": "x"}
    for m in (0, 5, 10)
]


def _state(sid):
    return {"session_id": sid, "messages": [], "templates": {}}


def test_pick_resolves_after_list_was_forgotten(monkeypatch):
    sid = "refetch-1"
    session = tools._get_booking_session(sid)
    session.update({"doctor_id": "d", "branch_id": "b",
                    "last_slot_range": ("2026-10-28T00:00:00+03:00", "2026-10-28T23:59:00+03:00")})
    session.pop("last_list", None)

    def fake_fetch(state, from_date, to_date):
        tools._remember_list(state, "slot", SLOTS)
        return {"status": "found", "slots": SLOTS}

    monkeypatch.setattr(tools, "get_available_slots_for_booking", SimpleNamespace(func=fake_fetch))
    result = tools.select_appointment_slot.func(_state(sid), user_input="3")
    assert result["status"] == "selected"
    assert result["slot"]["time_display"] == "6:10 مساءً"


def test_no_range_on_file_tells_the_model_to_fetch():
    sid = "refetch-2"
    tools._get_booking_session(sid).pop("last_list", None)
    result = tools.select_appointment_slot.func(_state(sid), user_input="3")
    assert result["status"] == "no_list_shown"
    assert "get_available_slots_for_booking" in result["_guidance"]
