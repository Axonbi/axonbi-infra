"""
2026-10-12 afternoon (tanasuq-production):

  - 12:34 session 966599991508: the booking was cancelled, reschedule asked
    "تبغى أساعدك بحجز موعد جديد؟", and "حجز موعد" stayed in reschedule. It
    built the whole booking with no tool to finish it; at "تاكيد" the claim
    gate caught "✅ تم تأكيد حجز موعدك" and nothing was booked.
  - 12:48 "اقرب فرع للرياض" searched a branch named "للرياض" and the reply
    opened with 'ما لقيت فرع اسمه "للرياض"'.
"""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import agents
import graph
import tools
from agents.semantic_router import TurnFacts, owner

_DEAD_BOOKING = ("هذا الحجز اللي رقم GuestBookingNum-2026-10-11-349 إما ملغي، مكتمل، أو التاريخ "
                 "والوقت حقه مرّ، فما نقدر نعدله أو نلغيّه الحين. تبغى أساعدك بحجز موعد جديد؟")
_GREETING_MENU = ("أهلاً وسهلاً بك 👋\nيمكنني مساعدتك في:\n🗓️ حجز موعد جديد\n✏️ تعديل أو إلغاء موعد قائم\n\n"
                  "تحب تعدل الموعد برقم الجوال ولا برقم الحجز؟")


def _reading(intent, **flags):
    return {"intent": intent, "confidence": 1.0, "answer_to_previous_question": True, **flags}


# ----------------------------------------------------------------------
# A new booking reschedule offered goes to booking
# ----------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    _DEAD_BOOKING,
    "الحجز الأصلي اللي تحاول تعدله ملغي. تبغى أساعدك تحجز موعد جديد مع د. أحمد يوسف في نفس الوقت؟",
])
def test_reschedule_offering_a_new_booking_is_detected(question):
    assert graph._assistant_offered_a_new_booking([AIMessage(content=question), HumanMessage(content="حجز موعد")])


@pytest.mark.parametrize("question", [
    _GREETING_MENU,
    "هل هذا هو الموعد الجديد اللي تود تأكيده؟ (نعم/لا)",
])
def test_the_menu_and_a_new_time_are_not_an_offer(question):
    assert not graph._assistant_offered_a_new_booking([AIMessage(content=question), HumanMessage(content="نعم")])


@pytest.mark.parametrize("flow", ["reschedule", "cancel"])
@pytest.mark.parametrize("reading", [
    _reading("booking"),                                  # "حجز موعد"
    _reading("booking", confirms=True),                   # "نعم"
    _reading("reschedule", confirms=True),                # "نعم" read as the flow's own intent
    _reading("answer", confirms=True),
])
def test_accepting_the_offered_new_booking_moves_to_booking(flow, reading):
    agent, reason, _ = owner(reading, TurnFacts(previous=flow, new_booking_offered=True))
    assert agent == "booking"
    # Keeps the doctor on file - see graph._CONTINUATION_REASONS.
    assert any(phrase in reason for phrase in graph._CONTINUATION_REASONS)


def test_a_booking_shaped_answer_with_no_offer_still_stays_in_reschedule():
    agent, _, _ = owner(_reading("booking"), TurnFacts(previous="reschedule"))
    assert agent == "reschedule"


def test_declining_the_offered_new_booking_does_not_start_one():
    agent, _, _ = owner(_reading("answer", declines=True),
                        TurnFacts(previous="reschedule", new_booking_offered=True, bare_refusal=True))
    assert agent == "reschedule"


# ----------------------------------------------------------------------
# "اقرب فرع للرياض" is a city
# ----------------------------------------------------------------------

_BRANCHES = [
    {"id": "b-manar", "name": "Al Manar", "altName": "المنار", "cityName": "Riyadh",
     "address": "2955 شارع الشيخ عبدالرحمن، حي المنار، الرياض 14221، المملكة العربية السعودية"},
    {"id": "b-nozha", "name": "Al Nozha", "altName": "النزهة", "cityName": "Riyadh",
     "address": "7282 أبي سفيان بن حرب، النزهة، الرياض 12473، المملكة العربية السعودية"},
    {"id": "b-er", "name": "Emergency", "altName": "الطوارئ", "cityName": "Jeddah",
     "address": "جدة، المملكة العربية السعودية"},
]


@pytest.mark.parametrize("text", ["للرياض", "اقرب فرع للرياض", "بالرياض", "في الرياض", "Riyadh"])
def test_a_city_returns_its_branches(text):
    assert [b["id"] for b in tools._branches_in_city(_BRANCHES, text)] == ["b-manar", "b-nozha"]


@pytest.mark.parametrize("text", ["فرع", "اقرب فرع", "الشفا"])
def test_no_city_no_branches(text):
    assert tools._branches_in_city(_BRANCHES, text) == []


def test_match_entity_info_answers_a_city_with_its_branches():
    sid = "log-1012-city"
    state = {"session_id": sid, "messages": [],
             "templates": {"_doctors_base_url": "https://x.test", "_base_url": "https://x.test"}}
    try:
        with patch.object(tools.api, "get_branches",
                          return_value={"success": True, "data": {"items": _BRANCHES}}), \
             patch.object(tools, "_branch_ids_with_available_doctors", return_value={"b-manar", "b-nozha"}):
            result = tools.match_entity_info.func(state=state, user_input="للرياض", entity_type="branch")
    finally:
        tools._BOOKING_SESSIONS.pop(sid, None)
    assert result["status"] == "list"
    assert [b["id"] for b in result["items"]] == ["b-manar", "b-nozha"]


# ----------------------------------------------------------------------
# "كمل الحجز هنا" as the opening message asks to book
# ----------------------------------------------------------------------
# 12:22 session 966533803462: "كمل" read as a confirmation, the entry
# question stood down, and the model asked STEP NB6's same-number question
# with no doctor or time.

@pytest.mark.parametrize("text", ["كمل الحجز هنا", "كمل الحجز", "اكمل الحجز من هنا", "تابع الحجز"])
def test_opening_with_continue_the_booking_gets_the_entry_question(text):
    directive = graph._build_booking_entry_directive(
        [HumanMessage(content=text)], "log-1012-continue", "booking", {"intent": "booking"})
    assert directive is graph._BOOKING_ENTRY_ASK_DIRECTIVE


def test_continue_mid_conversation_is_still_a_confirmation():
    messages = [AIMessage(content="تم اختيار موعد 5:00 مساءً مع د. أحمد يوسف. نكمل؟"),
                HumanMessage(content="كمل الحجز")]
    assert graph._build_booking_entry_directive(messages, "log-1012-continue", "booking",
                                                {"intent": "booking"}) == ""


# ----------------------------------------------------------------------
# 17:05-17:08 session 201000625084 (Dr Ahmed Youssef, today's slots over)
# ----------------------------------------------------------------------
# - "اليوم": today was over; the reply showed 17-19/10 (the days AFTER the
#   13-15/10 on screen) and said nothing about today.
# - "الاتنين اليوم" was resolved to the Monday on screen, 19/10.
# - "12/10/2026": "no appointments that day" was flagged as unchecked although
#   today had come back empty a minute earlier, replaced with "أي تاريخ تحب؟",
#   and the patient typed the same date again.

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

_TZ = "Asia/Riyadh"
_SID2 = "log-1012-today"


def _today():
    return datetime.now(ZoneInfo(_TZ)).date()


@pytest.fixture
def today_session():
    tools._BOOKING_SESSIONS.pop(_SID2, None)
    s = tools._get_booking_session(_SID2)
    s.update({"doctor_id": "dr-ahmed", "branch_id": "b-manar"})
    yield s
    tools._BOOKING_SESSIONS.pop(_SID2, None)


def _state_with(text):
    return {"session_id": _SID2, "messages": [HumanMessage(content=text)],
            "templates": {"_timezone": _TZ, "_doctors_base_url": "https://x.test", "_base_url": "https://x.test"}}


@pytest.mark.parametrize("text,expected", [
    ("اليوم", True), ("الاتنين اليوم", True), ("النهارده", True), ("today please", True),
    ("الاتنين", False), ("اي يوم", False),
])
def test_the_patients_own_word_for_today(text, expected):
    assert tools._patient_says_today(_state_with(text)) is expected


def test_weekday_with_today_is_today_not_the_one_on_screen(today_session):
    today = _today()
    next_week = today + timedelta(days=7)
    today_session["last_list"] = {"entity_type": "day", "items": [{"date": next_week.isoformat()}]}
    weekday = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][today.weekday()]
    seen = {}

    def get_slots(base_url, doctor_ids, from_date, to_date, **kwargs):
        seen.setdefault("from", str(from_date)[:10])
        return {"success": True, "data": {"items": []}}

    with patch.object(tools.api, "get_doctor_schedule_slots", side_effect=get_slots):
        tools.resolve_available_day.func(state=_state_with(f"{weekday} اليوم"), weekday_name=weekday)
    assert seen.get("from") == today.isoformat()


def test_a_day_found_empty_backs_a_later_no_appointments_reply(today_session):
    today = _today()
    tools._remember_empty_day(today_session, "dr-ahmed", today.isoformat())
    state = _state_with(today.strftime("%d/%m/%Y"))
    assert graph._named_day_already_found_empty(state)
    other = _state_with((today + timedelta(days=2)).strftime("%d/%m/%Y"))
    assert not graph._named_day_already_found_empty(other)


def test_asking_for_today_shows_the_nearest_days_not_the_ones_after_the_list(today_session):
    today = _today()
    days = [today + timedelta(days=i) for i in range(1, 9)]
    items = [{"slotStart": f"{d.isoformat()}T{h}:00:00+03:00", "slotEnd": f"{d.isoformat()}T{h}:30:00+03:00",
              "isBooked": False, "branchId": "b-manar", "serviceName": "جلسة"} for d in days for h in (16, 17)]

    def get_slots(base_url, doctor_ids, from_date, to_date, **kwargs):
        lo, hi = str(from_date)[:10], str(to_date)[:10]
        return {"success": True, "data": {"items": [i for i in items if lo <= i["slotStart"][:10] <= hi]}}

    today_session["last_list"] = {"entity_type": "day", "items": [{"date": d.isoformat()} for d in days[:3]]}
    with patch.object(tools.api, "get_doctor_schedule_slots", side_effect=get_slots):
        result = tools.list_available_days_for_booking.func(state=_state_with("اليوم"), offset=3)
    assert result["days"][0]["date"] == days[0].isoformat()


# ----------------------------------------------------------------------
# 17:12 "موقع العيادة بحي النزهه ابيه": the address went out, the pin did not
# ----------------------------------------------------------------------

import json

from langchain_core.messages import ToolMessage

import main


def _location_turn(text, tool_name="match_entity_info", payload=None):
    payload = payload or {"status": "matched", "item": {"id": "b-nozha", "name": "النزهة",
                                                        "address": "7282 أبي سفيان بن حرب، النزهة، الرياض"}}
    return {"session_id": "log-1012-pin", "templates": {}, "understanding": {"intent": "faq", "asks_location": True},
            "messages": [HumanMessage(content=text),
                         AIMessage(content="", tool_calls=[{"id": "c1", "name": tool_name, "args": {}}]),
                         ToolMessage(content=json.dumps(payload, ensure_ascii=False), name=tool_name,
                                     tool_call_id="c1")]}


_ADDRESS_REPLY = AIMessage(content="عنوان فرع النزهة هو: 7282 أبي سفيان بن حرب، النزهة، الرياض")


def test_a_location_question_answered_without_the_pin_sends_it_from_code():
    pair = graph._missed_location_pin_pair(_location_turn("موقع العيادة بحي النزهه ابيه"), _ADDRESS_REPLY, "ar")
    signals = main._turn_signals(pair)
    assert signals["location"] is True and signals["branch_name"] == "النزهة"


def test_no_second_pin_when_the_model_already_sent_it():
    state = _location_turn("موقع فرع النزهة", tool_name="share_branch_location",
                           payload={"status": "location_requested", "branch_name": "النزهة"})
    assert graph._missed_location_pin_pair(state, _ADDRESS_REPLY, "ar") == []


def test_no_pin_when_they_did_not_ask_where():
    state = _location_turn("ابي احجز في النزهة")
    state["understanding"] = {"intent": "booking"}
    assert graph._missed_location_pin_pair(state, _ADDRESS_REPLY, "ar") == []


def test_no_pin_for_a_doctor_match():
    state = _location_turn("وين د. احمد يوسف", payload={"status": "matched",
                                                       "item": {"id": "d1", "formatedName": "أحمد يوسف"}})
    state["understanding"] = {"intent": "faq"}
    assert graph._missed_location_pin_pair(state, AIMessage(content="د. أحمد يوسف"), "ar") == []
