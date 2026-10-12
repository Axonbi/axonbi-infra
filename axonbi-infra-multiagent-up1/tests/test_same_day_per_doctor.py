"""
Same-day booking is decided per doctor by the CMS (US_5.4.5.3 Booking
Rules: a booking cutoff of X hours before a slot, a maximum number of cases
per day). The API returns only what may be booked, so today's open slots
are offered like any other day's - nothing here refuses today on its own.
"""

from datetime import timedelta

from langchain_core.messages import HumanMessage

import config
import graph
import tools

_SID = "same-day-per-doctor"
_TZ = "Asia/Riyadh"


def _state(text="اليوم"):
    return {"session_id": _SID, "channel_phone": "966500000000",
            "templates": {"_timezone": _TZ}, "messages": [HumanMessage(content=text)]}


def _now():
    return tools._local_now_naive(_TZ)


def test_same_day_is_open_by_default():
    assert config.ALLOW_SAME_DAY_BOOKING is True


def test_todays_open_slots_are_kept():
    later = (_now() + timedelta(hours=3)).replace(second=0, microsecond=0).isoformat()
    slots = [{"_localStart": later}]
    assert tools._drop_same_day_slots(list(slots), _TZ) == (slots, 0)


def test_today_resolves_to_today():
    result = tools.get_next_weekday_date.func(weekday_name="اليوم", timezone_name=_TZ)
    assert result["status"] == "found" and result["date"] == _now().date().isoformat()


def test_todays_weekday_is_today():
    name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][_now().weekday()]
    result = tools.get_next_weekday_date.func(weekday_name=name, timezone_name=_TZ)
    assert result["date"] == _now().date().isoformat()


def test_no_same_day_notice():
    tools._BOOKING_SESSIONS.pop(_SID, None)
    tools._get_booking_session(_SID)["doctor_id"] = "dr"
    try:
        reply = "📅 المواعيد المتاحة اليوم:\n1️⃣ 8:00 مساءً\nأي رقم أو وقت تفضل؟"
        assert not graph._patient_asked_for_today(_state())
        assert graph._same_day_notice(reply, _state(), "ar") == reply
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)


def test_the_old_rule_can_still_be_switched_back(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_SAME_DAY_BOOKING", False)
    result = tools.get_next_weekday_date.func(weekday_name="اليوم", timezone_name=_TZ)
    assert result["status"] == "same_day_not_allowed"


# ----------------------------------------------------------------------
# Rovan, 2026-10-12 11:09-11:10: today's 8:30 pm slot was dropped by a
# fixed 12-hour lead, and "انهارده" was read as a yes to next Monday.
# ----------------------------------------------------------------------

def test_no_fixed_lead_when_the_cms_decides():
    now = _now()
    assert tools._booking_lead_time(now) == now.replace(tzinfo=None)


def test_the_old_lead_stands_when_same_day_is_off(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_SAME_DAY_BOOKING", False)
    now = _now()
    assert tools._booking_lead_time(now) == now.replace(tzinfo=None) + timedelta(hours=12)


def test_egyptian_today_is_today():
    assert tools.resolve_relative_date("انهارده", _TZ)["offset_days"] == 0


def _offered_next_week():
    day = _now().date() + timedelta(days=7)
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return {"date": day.isoformat(), "weekday_name": names[day.weekday()],
            "from_date": f"{day.isoformat()}T00:00:00", "to_date": f"{day.isoformat()}T23:59:59"}


def test_today_is_not_a_yes_to_next_week(monkeypatch):
    import graph
    monkeypatch.setattr(config, "DETERMINISTIC_DAY_RESOLUTION", True)
    tools._BOOKING_SESSIONS.pop(_SID, None)
    tools._get_booking_session(_SID).update({
        "doctor_id": "dr", "branch_id": "b", "last_list": {"entity_type": "day", "items": [_offered_next_week()]}})
    try:
        state = {**_state("انهارده"), "understanding": {"confirms": True, "answer_to_previous_question": True}}
        assert graph._deterministic_slots_for_accepted_day(state, "booking") == []
        assert graph._names_another_day("انهارده", _offered_next_week(), state)
        assert not graph._names_another_day("اه", _offered_next_week(), state)
        same_weekday = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"][
            (_now().date() + timedelta(days=7)).weekday()]
        assert not graph._names_another_day(f"تمام {same_weekday}", _offered_next_week(), state)
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)
