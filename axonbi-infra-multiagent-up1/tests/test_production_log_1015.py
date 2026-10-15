"""
tanasuq-production, 2026-10-15: "ما لقيت أي حجز" for 0593719856, whose
booking had been cancelled. The phone search asks the CMS for New and
Confirmed bookings only, so a cancelled one never came back.
"""

from datetime import timedelta
from unittest.mock import patch

import tools

_SID = "log-1015"
_PHONE = "966593719856"


def _state():
    return {"session_id": _SID, "channel_phone": _PHONE, "messages": [],
            "templates": {"_timezone": "Asia/Riyadh", "_cms_base_url": "https://cms.x.test"}}


def _booking(status, days_ahead):
    when = (tools._local_now_naive("Asia/Riyadh") + timedelta(days=days_ahead)).replace(
        hour=16, minute=0, second=0, microsecond=0)
    return {"id": f"bk-{status}-{days_ahead}", "status": status, "doctorName": "عمر المديفر",
            "branchName": "المنار", "bookingTimeFrom": when.isoformat() + "+03:00",
            "bookingTimeTo": (when + timedelta(minutes=30)).isoformat() + "+03:00"}


def _lookup(all_bookings):
    calls = []

    def by_phone(base_url, phone, **kwargs):
        calls.append((phone, kwargs.get("status_list")))
        if kwargs.get("status_list"):
            items = [b for b in all_bookings if b["status"] in kwargs["status_list"]]
        else:
            items = list(all_bookings)
        return {"success": True, "data": {"items": items}}

    with patch.object(tools.api, "get_bookings_by_phone", side_effect=by_phone):
        result = tools.lookup_appointment.func(state=_state(), use_channel_identity=True)
    return result, calls


def test_a_cancelled_booking_is_said_to_be_cancelled_not_missing():
    result, calls = _lookup([_booking(6, 2), _booking(6, -20)])
    assert result["status"] == "found_but_inactive"
    assert len(result["cancelled_appointments"]) == 1   # the past one is not brought up
    assert all(phone.startswith("+966") for phone, _ in calls)


def test_no_booking_at_all_is_still_not_found():
    result, _ = _lookup([])
    assert result == {"status": "not_found"}


def test_an_active_booking_is_found_as_before():
    result, calls = _lookup([_booking(1, 3)])
    assert result["status"] == "found_one"
    assert len(calls) == 1


def test_the_patient_lookup_gets_the_full_number():
    seen = []

    def patient_info(base_url, mobile_number, **kwargs):
        seen.append(mobile_number)
        return {"success": True, "data": {"items": []}}

    state = {**_state(), "templates": {**_state()["templates"], "_doctors_base_url": "https://portal.x.test"}}
    tools._BOOKING_SESSIONS.pop(_SID, None)
    tools._get_booking_session(_SID)["slots_shown"] = True
    with patch.object(tools, "_phone_is_verified", return_value=True), \
         patch.object(tools.api, "get_patient_info", side_effect=patient_info):
        tools.get_patient_info.func(state=state, mobile_number="0593719856")
    tools._BOOKING_SESSIONS.pop(_SID, None)
    assert seen == ["+966593719856"]


def test_tanasuq_branches_have_coordinates_and_nothing_else_does():
    import config
    config._all_branches_geo.cache_clear()
    geo = config.load_branches_geo()
    assert {"المنار", "النزهة"} <= set(geo)
    assert {row["country_code"] for row in geo.values()} == {"sa"}
    assert 24 < geo["المنار"]["latitude"] < 25 and 46 < geo["المنار"]["longitude"] < 47
