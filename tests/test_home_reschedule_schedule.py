"""
Rescheduling a home-collection booking shows the home-service rows, not a
walk-in branch (elborgdemo staging 2026-10-05 11:36: offered فرع حدائق
الاهرام for a home visit).
"""

import tools

HOME = "Home Branch"
ROWS = [
    {"branchId": "home", "branchName": HOME, "recurringDaysNames": ["Sunday"]},
    {"branchId": "hadayek", "branchName": "حدائق الاهرام", "recurringDaysNames": ["Saturday"]},
]


def _run(monkeypatch, booking_branch):
    state = {"session_id": "s", "templates": {
        "_lab_uses_per_test_doctors": True, "_lab_home_service_branch_name": HOME,
        "_doctors_base_url": "https://portal", "_cms_base_url": "https://cms"}}
    monkeypatch.setattr(tools.api, "get_bookings_by_ref", lambda *a, **k: {
        "success": True, "data": {"items": [{"doctorId": "d", "branchName": booking_branch}]}})
    monkeypatch.setattr(tools.api, "get_doctor_schedule", lambda *a, **k: {
        "success": True, "data": {"items": [dict(r) for r in ROWS]}})
    result = tools.get_doctor_schedule.func(state, ref_number="GBN-2026-10-05-459")
    return [row.get("branchName") for row in str_rows(result)]


def str_rows(result):
    for key in ("schedule", "items", "rows", "schedules"):
        if isinstance(result.get(key), list):
            return result[key]
    raise AssertionError(f"no rows in {result}")


def test_a_home_booking_keeps_the_home_rows(monkeypatch):
    assert _run(monkeypatch, HOME) == [HOME]


def test_a_walk_in_booking_drops_the_home_rows(monkeypatch):
    assert _run(monkeypatch, "حدائق الاهرام") == ["حدائق الاهرام"]
