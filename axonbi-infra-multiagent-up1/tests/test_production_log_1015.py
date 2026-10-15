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


# ----------------------------------------------------------------------
# A degree asked for where a specialty was expected ("ابي اخصائي نفسي"):
# the doctors with that degree, and "not available" only when neither
# a specialty nor a degree matches.
# ----------------------------------------------------------------------

_ROSTER = {"success": True, "data": {"items": [
    {"id": "dr-1", "formatedName": "أصيلا الحسن", "altName": "أصيلا الحسن", "degreeName": "أخصائي نفسي",
     "specialtyName": "علاج نفسي", "hasSlots": True},
    {"id": "dr-2", "formatedName": "سعد الماضي", "altName": "سعد الماضي", "degreeName": "استشاري طب نفسي",
     "specialtyName": "طب نفسي", "hasSlots": True},
]}}


def _doctor_state():
    return {"session_id": _SID, "messages": [], "understanding": None,
            "templates": {"_timezone": "Asia/Riyadh", "_doctors_base_url": "https://portal.x.test"}}


def test_a_degree_finds_the_doctors_with_it():
    with patch.object(tools.api, "get_doctors", return_value=_ROSTER):
        assert [d["id"] for d in tools._doctors_with_degree(_doctor_state(), "https://p", "اخصائي نفسي")] == ["dr-1"]
        assert [d["id"] for d in tools._doctors_with_degree(_doctor_state(), "https://p", "استشاري")] == ["dr-2"]
        assert tools._doctors_with_degree(_doctor_state(), "https://p", "دكتور") == []
        assert tools._doctors_with_degree(_doctor_state(), "https://p", "جراحة قلب") == []


def test_find_available_doctors_falls_back_to_the_degree():
    tools._BOOKING_SESSIONS.pop(_SID, None)
    try:
        with patch.object(tools, "_sanitize_specialty_ids", return_value=([], ["اخصائي نفسي"])), \
             patch.object(tools.api, "get_doctors", return_value=_ROSTER):
            result = tools.find_available_doctors.func(state=_doctor_state(), specialty_ids=["اخصائي نفسي"])
        assert result["status"] == "found" and result["matched_by"] == "degree"
        assert [d["id"] for d in result["doctors"]] == ["dr-1"]
        with patch.object(tools, "_sanitize_specialty_ids", return_value=([], ["جراحة قلب"])), \
             patch.object(tools.api, "get_doctors", return_value=_ROSTER):
            missing = tools.find_available_doctors.func(state=_doctor_state(), specialty_ids=["جراحة قلب"])
        assert missing["status"] == "specialty_not_resolved"
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)
