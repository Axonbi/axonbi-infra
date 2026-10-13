"""
tanasuq-production, 2026-10-10 afternoon/evening log:

  - "تمام ٣" over 3:00 / 5:00 / 6:00 locked option 3 (6:00); the patient
    then agreed to 3:00 and the reschedule was sent at 6:00.
  - A time locked at المنار survived a switch to النزهة.
  - "ابي الافضل والموعد الاقرب" got the cheapest doctor and his fee.
  - "ثمن الكشف" was not read as a price question.
  - "نوره متى متاحه" after a reply naming نورة العتيبي: "no schedule".
  - The review card showed the doctor's name as the patient's.
  - Three doctors matched in one response; the session kept the last.
  - "هنا مو مشكله" was answered "أرسل رقم الجوال مع رمز الدولة" while the
    same-number question had been left behind.
  - "السبت الساعة ٢" on a Saturday was offered that same Saturday.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, HumanMessage

import graph
import tool_result_guidance
import tools
import understanding

_SID = "log-1010-evening"
_TZ = "Asia/Riyadh"


def _state(messages=None, **extra):
    return {
        "session_id": _SID,
        "channel_phone": "966534089903",
        "templates": {"_timezone": _TZ, "_cms_base_url": "https://cms.x.test",
                      "_base_url": "https://portal.x.test", "_doctors_base_url": "https://portal.x.test"},
        "messages": messages or [],
        **extra,
    }


def _fresh_session(**values):
    tools._BOOKING_SESSIONS.pop(_SID, None)
    session = tools._get_booking_session(_SID)
    session.update(values)
    return session


def _slot(hour, minute=0, day=12):
    shown = f"{hour - 12 if hour > 12 else hour}:{minute:02d} مساءً"
    return {"slotStart": f"2026-10-{day:02d}T{hour:02d}:{minute:02d}:00",
            "slotEnd": f"2026-10-{day:02d}T{hour:02d}:{minute + 30 if minute < 30 else 59:02d}:00",
            "time_display": shown, "date_display": f"{day:02d}/10/2026"}


_THREE = [_slot(15), _slot(17), _slot(18)]   # 3:00, 5:00, 6:00


# ----------------------------------------------------------------------
# A number that is both a list position and an hour is asked about
# ----------------------------------------------------------------------

def test_a_number_that_is_also_another_hour_is_asked_about():
    _fresh_session(doctor_id="dr", last_list={"entity_type": "slot", "items": list(_THREE)})
    try:
        result = tools.select_appointment_slot.func(state=_state(), user_input="3")
        assert result["status"] == "ambiguous_pick"
        assert [c["time_display"] for c in result["candidates"]] == ["6:00 مساءً", "3:00 مساءً"]
        assert not tools._get_booking_session(_SID).get("selected_slot")
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)


def test_a_reschedule_pick_is_asked_about_too():
    _fresh_session(last_list={"entity_type": "slot", "items": list(_THREE)})
    try:
        result = tools.select_reschedule_slot.func(state=_state(), user_input="3")
        assert result["status"] == "ambiguous_pick"
        assert tool_result_guidance.guidance_for("select_reschedule_slot", result)
        assert tool_result_guidance.guidance_for("select_appointment_slot", result)
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)


def test_half_hourly_list_names_the_hour_itself():
    half_hourly = [_slot(16), _slot(16, 30), _slot(17), _slot(17, 30), _slot(18), _slot(18, 30)]
    both = tools._position_or_hour(half_hourly, 5, {"hour": 5, "minute": None, "period": None})
    assert [s["time_display"] for s in both] == ["6:00 مساءً", "5:00 مساءً"]


def test_a_number_that_is_the_same_slot_either_way_is_just_picked():
    hourly = [_slot(13), _slot(14), _slot(15)]   # 1:00, 2:00, 3:00
    assert tools._position_or_hour(hourly, 2, {"hour": 2, "minute": None, "period": None}) == []
    # no slot at that hour - a plain position
    assert tools._position_or_hour(_THREE, 2, {"hour": 2, "minute": None, "period": None}) == []


def test_the_time_the_patient_agreed_to_is_the_one_sent():
    session = _fresh_session(last_list={"entity_type": "slot", "items": list(_THREE)})
    locked = dict(_THREE[2])   # 6:00, from "تمام ٣"
    messages = [AIMessage(content="نعم، فيه موعد الساعة 3:00 مساءً يوم 12/10/2026. تحب أأكد التعديل؟"),
                HumanMessage(content="تمام")]
    try:
        chosen = tools._slot_the_patient_agreed_to(_state(messages), locked, "selected_reschedule_slot")
        assert chosen["time_display"] == "3:00 مساءً"
        assert session["selected_reschedule_slot"]["slotStart"] == _THREE[0]["slotStart"]
        # the confirmation names the locked time -> unchanged
        same = [AIMessage(content="موعدك الساعة 6:00 مساءً، تحب أأكد؟"), HumanMessage(content="اه")]
        assert tools._slot_the_patient_agreed_to(_state(same), locked, "selected_reschedule_slot") is locked
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)


# ----------------------------------------------------------------------
# A branch change drops the time chosen at the old branch
# ----------------------------------------------------------------------

def test_another_branch_drops_the_locked_time():
    session = _fresh_session(doctor_id="dr", branch_id="b-manar", selected_slot=dict(_THREE[0]),
                             slots_shown=True, review_shown=True)
    tools._retire_previous_doctors_branch(session, "branch", "b-nuzha")
    assert not session["selected_slot"] and not session["review_shown"]
    assert session["doctor_id"] == "dr"


def test_the_same_branch_keeps_the_locked_time():
    session = _fresh_session(doctor_id="dr", branch_id="b-manar", selected_slot=dict(_THREE[0]))
    tools._retire_previous_doctors_branch(session, "branch", "b-manar")
    assert session["selected_slot"]


# ----------------------------------------------------------------------
# Prices only when asked; "ثمن" asks
# ----------------------------------------------------------------------

def test_cheapest_is_refused_when_no_price_was_asked():
    state = _state([HumanMessage(content="ابي الافضل والموعد الاقرب")],
                   understanding={"asks_price": False})
    result = tools.find_best_doctor_in_specialty.func(state=state, specialty_ids=["sp-1"], criteria="cheapest")
    assert result["status"] == "not_requested"


def test_thaman_is_a_price_word():
    state = _state([HumanMessage(content="كم ثمن الكشف عند الدكتورة")], understanding={"asks_price": False})
    assert tools._fees_requested(state)


def test_fees_guidance_keeps_each_service_with_its_price():
    guidance = tool_result_guidance.guidance_for(
        "get_doctor_fees", {"status": "found", "fees": [{"service": "استشارة", "price": 600}]})
    assert guidance and "follow-up" in guidance


# ----------------------------------------------------------------------
# A doctor named from the reply on screen is that doctor
# ----------------------------------------------------------------------

_DOCTORS = {"entity_type": "doctor", "items": [
    {"id": "dr-noura", "name": "نورة العتيبي"},
    {"id": "dr-saad", "name": "سعد الماضي"},
]}


def test_a_first_name_from_the_reply_on_screen_confirms_the_doctor():
    session = _fresh_session(last_list=dict(_DOCTORS))
    messages = [AIMessage(content="أقرب موعد عند نورة العتيبي يوم الخميس."),
                HumanMessage(content="نوره متى متاحه")]
    try:
        assert tools._confirm_doctor_picked_from_shown_list(_state(messages)) == "dr-noura"
        assert session["doctor_id"] == "dr-noura"
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)


def test_a_name_not_in_the_reply_on_screen_confirms_nothing():
    session = _fresh_session(last_list=dict(_DOCTORS))
    messages = [AIMessage(content="أقرب موعد عند سعد الماضي يوم الخميس."),
                HumanMessage(content="نوره متى متاحه")]
    try:
        assert tools._confirm_doctor_picked_from_shown_list(_state(messages)) is None
        assert not session.get("doctor_id")
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)


# ----------------------------------------------------------------------
# The patient's name is never the doctor's
# ----------------------------------------------------------------------

def test_a_booking_named_after_its_doctor_has_no_patient_name():
    shaped = tools._shape_appointment({"id": "b1", "patientName": "مشاعل الشعلان",
                                       "doctorName": "مشاعل الشعلان"}, _TZ)
    assert not shaped.get("patientFullName")


def test_the_card_line_naming_the_doctor_as_the_patient_is_dropped():
    card = "هذا هو موعدك:\n👤 الاسم: مشاعل الشعلان\n👨‍⚕️ الطبيب: مشاعل الشعلان\n🏥 الفرع: المنار"
    fixed = graph._card_name_is_the_doctor_fix(card)
    assert "👤" not in fixed and "الطبيب: مشاعل الشعلان" in fixed
    real = "👤 الاسم: سارة أحمد\n👨‍⚕️ الطبيب: مشاعل الشعلان"
    assert graph._card_name_is_the_doctor_fix(real) == real


# ----------------------------------------------------------------------
# One doctor at a time
# ----------------------------------------------------------------------

def _calls(*names):
    return AIMessage(content="", tool_calls=[
        {"name": "match_entity_for_booking", "id": f"c{i}",
         "args": {"user_input": n, "entity_type": "doctor"}} for i, n in enumerate(names)])


def test_several_doctor_matches_in_one_response_run_none():
    _fresh_session()
    try:
        state = _state([HumanMessage(content="اي دكتور"), _calls("نورة العتيبي", "سعد الماضي", "سمر الخليفي")])
        gated = graph._gated_tool_calls(state)
        assert len(gated) == 3
        assert {p["status"] for _, p in gated} == {"one_doctor_at_a_time"}
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)


def test_one_doctor_match_runs():
    _fresh_session()
    try:
        state = _state([HumanMessage(content="نورة"), _calls("نورة العتيبي")])
        assert graph._gated_tool_calls(state) == []
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)


# ----------------------------------------------------------------------
# A same-number question left behind is not an answered one
# ----------------------------------------------------------------------

def _phone_state(*ai_texts):
    messages = []
    for text in ai_texts:
        messages += [HumanMessage(content="."), AIMessage(content=text)]
    messages.append(HumanMessage(content="هنا مو مشكله"))
    return _state(messages)


def test_a_phone_demand_after_the_question_was_left_behind_is_caught():
    _fresh_session(doctor_id="dr", selected_slot=dict(_THREE[0]))
    try:
        state = _phone_state("نكمل الحجز على نفس رقم الواتساب هذا؟ ✅",
                             "أقرب موعد في المنار يوم 21/10. تحب نكمل الحجز على موعد 21/10 في المنار؟")
        assert graph._reply_skips_same_number_question_when_ready(
            "من فضلك أرسل رقم الجوال مع رمز الدولة.", state)
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)


def test_a_phone_demand_right_after_a_no_is_allowed():
    _fresh_session(doctor_id="dr", selected_slot=dict(_THREE[0]))
    try:
        state = _phone_state("نكمل الحجز على نفس رقم الواتساب هذا؟ ✅",
                             "تمام، من فضلك أرسل رقم الجوال مع رمز الدولة.")
        assert not graph._reply_skips_same_number_question_when_ready(
            "من فضلك أرسل رقم الجوال مع رمز الدولة.", state)
    finally:
        tools._BOOKING_SESSIONS.pop(_SID, None)


# ----------------------------------------------------------------------
# Today is never offered
# ----------------------------------------------------------------------

def test_todays_weekday_is_next_week(monkeypatch):
    monkeypatch.setattr(graph.config, "ALLOW_SAME_DAY_BOOKING", False)  # the old blanket rule
    today = datetime.now(ZoneInfo(_TZ)).date()
    name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][today.weekday()]
    result = tools.get_next_weekday_date.func(weekday_name=name, timezone_name=_TZ)
    assert result["status"] == "found"
    assert (datetime.strptime(result["date"], "%Y-%m-%d").date() - today).days == 7


def test_today_itself_is_refused(monkeypatch):
    monkeypatch.setattr(graph.config, "ALLOW_SAME_DAY_BOOKING", False)  # the old blanket rule
    result = tools.get_next_weekday_date.func(weekday_name="اليوم", timezone_name=_TZ)
    assert result["status"] == "same_day_not_allowed"
    assert tool_result_guidance.guidance_for("get_next_weekday_date", result)


# ----------------------------------------------------------------------
# "ثامي" is "ثاني"
# ----------------------------------------------------------------------

def test_the_thami_typo_is_in_the_reading():
    assert "ثامي" in understanding.PROMPT
