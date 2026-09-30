"""A call that failed only because a precondition was unmet must not block its retry."""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import graph


def _call(call_id, name="get_doctor_schedule_for_booking", args=None):
    return AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": args or {}}])


def test_missing_doctor_result_does_not_count_as_a_repeat():
    messages = [
        HumanMessage(content="عاوز احجز اسنان"),
        _call("1"),
        ToolMessage(content=json.dumps({"status": "missing_doctor"}), name="get_doctor_schedule_for_booking", tool_call_id="1"),
        _call("2", "match_entity_for_booking", {"user_input": "ليلى", "entity_type": "doctor"}),
        ToolMessage(content=json.dumps({"matched": True}), name="match_entity_for_booking", tool_call_id="2"),
    ]
    retry = AIMessage(content="", tool_calls=[{"id": "3", "name": "get_doctor_schedule_for_booking", "args": {}}])
    assert graph._repeated_tool_calls(retry, messages) == []


def test_a_real_answer_still_counts_as_a_repeat():
    messages = [
        HumanMessage(content="مواعيد الدكتور"),
        _call("1"),
        ToolMessage(content=json.dumps({"status": "found", "schedules": []}), name="get_doctor_schedule_for_booking", tool_call_id="1"),
    ]
    retry = AIMessage(content="", tool_calls=[{"id": "2", "name": "get_doctor_schedule_for_booking", "args": {}}])
    assert len(graph._repeated_tool_calls(retry, messages)) == 1


def test_stale_doctor_guard_accepts_the_same_specialty_looked_up_earlier():
    messages = [
        HumanMessage(content="عندي وجع في سني"),
        _call("1", "find_available_doctors", {"specialty_name": "طب اسنان"}),
        ToolMessage(content=json.dumps({"status": "found", "doctors": [{"name": "ليلى الحربي"}]}),
                    name="find_available_doctors", tool_call_id="1"),
        AIMessage(content="تحب أحجز؟"),
        HumanMessage(content="ايوه"),
    ]
    reply = "الدكاترة المتاحين في تخصص طب اسنان:\n1️⃣ د. ليلى الحربي"
    assert graph._reply_shows_doctor_for_service_with_no_lookup_this_turn(reply, {"messages": messages}) is False


def test_stale_doctor_guard_still_flags_a_name_carried_to_another_specialty():
    messages = [
        HumanMessage(content="عندي وجع في سني"),
        _call("1", "find_available_doctors", {"specialty_name": "طب اسنان"}),
        ToolMessage(content=json.dumps({"status": "found", "doctors": [{"name": "ليلى الحربي"}]}),
                    name="find_available_doctors", tool_call_id="1"),
        HumanMessage(content="وايش عندكم تغذية"),
    ]
    reply = "الدكاترة المتاحين لخدمة أخصائي التغذية:\n1️⃣ د. ليلى الحربي"
    assert graph._reply_shows_doctor_for_service_with_no_lookup_this_turn(reply, {"messages": messages}) is True


def test_booking_success_message_is_not_a_phone_request():
    reply = ("عزيزتي نهي محمود\n✅ تم تأكيد موعدك بنجاح.\n📅 التاريخ: 06/10/2026\n"
             "رقم الحجز راح يصلك قريبًا على جوالك برسالة نصية.")
    state = {"channel_phone": "201158877175", "understanding": {"confirms": True}, "messages": []}
    assert graph._reply_asks_for_a_phone_already_known(reply, state) is False


def test_a_real_phone_request_after_a_yes_is_still_flagged():
    state = {"channel_phone": "201158877175", "understanding": {"confirms": True}, "messages": []}
    assert graph._reply_asks_for_a_phone_already_known("من فضلك أرسل رقم الجوال مع رمز الدولة.", state) is True


def test_nearest_weekday_suggestion_is_calendar_arithmetic_not_a_fabrication(monkeypatch):
    from datetime import date as real_date, datetime as real_datetime

    class FrozenDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime(2026, 9, 29, 12, 0, tzinfo=tz)

    monkeypatch.setattr(graph, "datetime", FrozenDatetime)
    state = {"messages": [ToolMessage(content=json.dumps({"status": "found", "schedules": []}),
                                      name="lookup_appointment", tool_call_id="x")]}
    # 29/09/2026 is a Tuesday and 04/10/2026 the next Sunday
    assert graph._reply_invents_availability("أقرب يوم أحد متاح هو 04/10/2026 - يناسبك؟", state) is False
    assert graph._reply_invents_availability("أقرب يوم ثلاثاء متاح هو 29/09/2026 - يناسبك؟", state) is False
    # a date that is not any occurrence of the named weekday is still rejected
    assert graph._reply_invents_availability("أقرب يوم أحد متاح هو 07/10/2026 - يناسبك؟", state) is True
    # and times are never accepted this way
    assert graph._reply_invents_availability("أقرب يوم أحد 04/10/2026 الساعة 5:40 مساءً", state) is True
