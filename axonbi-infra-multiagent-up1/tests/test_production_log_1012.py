"""
2026-10-12 (studio, tanasuq-saudi on the CURE knowledge base):

  - 11:42 "الاتنين الي بعده" with Monday 12/10 on screen came back as 12/10.
  - 11:43 a Monday-only doctor with seven bookable Mondays showed one; "more
    days?" was told "ما فيش مواعيد متاحة".
  - 11:43 "أقرب فرع للجيزة": none of CURE's eight branches had coordinates.
"""

from datetime import datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

import tools

_SID = "log-1012"
_TZ = "Africa/Cairo"


def _mondays(count):
    today = datetime.now(ZoneInfo(_TZ)).date()
    first = today + timedelta(days=(7 - today.weekday()) % 7 or 7)  # the next Monday, never today
    return [first + timedelta(weeks=i) for i in range(count)]


def _slots(days):
    items = []
    for day in days:
        for hour in (16, 17):
            items.append({"slotStart": f"{day.isoformat()}T{hour}:00:00+03:00",
                          "slotEnd": f"{day.isoformat()}T{hour}:30:00+03:00", "isBooked": False,
                          "branchId": "b-oct", "serviceName": "زراعة الاسنان"})
    return items


def _api(days):
    items = _slots(days)

    def get_slots(base_url, doctor_ids, from_date, to_date, **kwargs):
        lo, hi = str(from_date)[:10], str(to_date)[:10]
        return {"success": True, "data": {"items": [i for i in items if lo <= i["slotStart"][:10] <= hi]}}

    return get_slots


@pytest.fixture
def session():
    tools._BOOKING_SESSIONS.pop(_SID, None)
    s = tools._get_booking_session(_SID)
    s.update({"doctor_id": "dr-mansour", "branch_id": "b-oct"})
    yield s
    tools._BOOKING_SESSIONS.pop(_SID, None)


def _state():
    return {"session_id": _SID, "messages": [],
            "templates": {"_timezone": _TZ, "_doctors_base_url": "https://x.test", "_base_url": "https://x.test"}}


# ----------------------------------------------------------------------
# "الاتنين الي بعده" is the Monday after the one on screen
# ----------------------------------------------------------------------

def test_the_monday_after_the_one_on_screen(session):
    mondays = _mondays(3)
    session["last_list"] = {"entity_type": "day", "items": [{"date": mondays[0].isoformat()}]}
    with patch.object(tools.api, "get_doctor_schedule_slots", side_effect=_api(mondays)):
        result = tools.resolve_available_day.func(state=_state(), weekday_name="الاتنين الي بعده")
    assert result["status"] == "found"
    assert str(result.get("date") or result.get("from_date"))[:10] == mondays[1].isoformat()


def test_a_bare_monday_is_still_the_one_on_screen(session):
    mondays = _mondays(3)
    session["last_list"] = {"entity_type": "day", "items": [{"date": mondays[1].isoformat()}]}
    with patch.object(tools.api, "get_doctor_schedule_slots", side_effect=_api(mondays)):
        result = tools.resolve_available_day.func(state=_state(), weekday_name="الاتنين")
    assert str(result.get("date") or result.get("from_date"))[:10] == mondays[1].isoformat()


# ----------------------------------------------------------------------
# A one-weekday doctor: the list is filled with the following weeks, and
# "more" continues after the days on screen
# ----------------------------------------------------------------------

def _days(session, offset=0, limit=3, mondays=None):
    mondays = mondays or _mondays(7)
    with patch.object(tools.api, "get_doctor_schedule_slots", side_effect=_api(mondays)):
        return tools.list_available_days_for_booking.func(state=_state(), limit=limit, offset=offset)


def test_a_monday_only_doctor_shows_three_mondays(session):
    mondays = _mondays(7)
    result = _days(session, mondays=mondays)
    assert result["status"] == "found"
    assert [d["date"] for d in result["days"]] == [m.isoformat() for m in mondays[:3]]
    assert result["has_more"] is True and result["next_offset"] == 3


def test_more_days_continue_after_the_one_on_screen_whatever_offset_the_model_passed(session):
    mondays = _mondays(7)
    session["last_list"] = {"entity_type": "day", "items": [{"date": mondays[0].isoformat()}]}
    result = _days(session, offset=7, mondays=mondays)
    assert result["status"] == "found"
    assert result["days"][0]["date"] == mondays[1].isoformat()


def test_the_last_monday_shown_means_no_more(session):
    mondays = _mondays(2)
    session["last_list"] = {"entity_type": "day", "items": [{"date": m.isoformat()} for m in mondays]}
    assert _days(session, offset=2, mondays=mondays)["status"] == "no_more_days"


# ----------------------------------------------------------------------
# CURE's branches, as the CMS names them, have coordinates
# ----------------------------------------------------------------------

@pytest.mark.parametrize("api_name", [
    "Nasr City", "6th of October", "Fifth Settlement", "Maadi", "Heliopolis (Masr El Gedida)",
    "Mohandessin", "Alexandria", "Mokattam",
])
def test_every_cure_branch_has_coordinates(api_name):
    rows = tools.load_branches_geo()
    index = {tools._geo_name_key(name): geo for name, geo in rows.items()}
    assert rows.get(api_name) or index.get(tools._geo_name_key(api_name))


def test_giza_is_nearest_to_mohandessin():
    branches = {"success": True, "data": {"items": [
        {"id": str(i), "name": n} for i, n in enumerate(
            ["Nasr City", "6th of October", "Fifth Settlement", "Maadi", "Heliopolis (Masr El Gedida)",
             "Mohandessin", "Alexandria", "Mokattam"])]}}
    with patch.object(tools.api, "get_branches", return_value=branches):
        result = tools.find_nearest_branch.func(state={**_state(), "target_language": "en"},
                                                latitude=29.9870753, longitude=31.2118063)
    assert result["status"] == "found"
    nearest = result.get("nearest") or (result.get("branches") or [{}])[0]
    assert nearest["name"] in ("Mohandessin", "Maadi")


# ----------------------------------------------------------------------
# 11:46 "الساعه ٥" -> "5:00 or 5:30?" -> "1" booked 4:00
# ----------------------------------------------------------------------

def _day_slots():
    out = []
    for hhmm, shown in (("16:00", "4:00 مساءً"), ("16:30", "4:30 مساءً"), ("17:00", "5:00 مساءً"),
                        ("17:30", "5:30 مساءً"), ("18:00", "6:00 مساءً")):
        out.append({"slotStart": f"2030-10-17T{hhmm}:00", "slotEnd": f"2030-10-17T{hhmm}:00",
                    "time_display": shown, "date_display": "17/10/2030"})
    return out


def _pick(session, text):
    return tools.select_appointment_slot.func(state=_state(), user_input=text)


def test_one_after_five_or_five_thirty_is_five(session):
    session["last_list"] = {"entity_type": "slot", "items": _day_slots()}
    assert _pick(session, "الساعه ٥")["status"] == "ambiguous_time"
    result = _pick(session, "1")
    assert result["status"] == "selected" and result["slot"]["slotStart"].endswith("17:00:00")


def test_a_time_off_the_short_list_still_resolves_from_the_whole_day(session):
    session["last_list"] = {"entity_type": "slot", "items": _day_slots()}
    _pick(session, "الساعه ٥")
    result = _pick(session, "4:30")
    assert result["status"] == "selected" and result["slot"]["slotStart"].endswith("16:30:00")
