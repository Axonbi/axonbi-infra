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
