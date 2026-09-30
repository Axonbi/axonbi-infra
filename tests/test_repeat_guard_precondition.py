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


def _fees_conversation(reply, patient):
    return [
        HumanMessage(content="بكم د احمد"),
        _call("1", "get_doctor_fees", {"doctor_name": "احمد"}),
        ToolMessage(content=json.dumps({"status": "found"}), name="get_doctor_fees", tool_call_id="1"),
        AIMessage(content=reply),
        HumanMessage(content=patient),
    ]


def test_yes_after_a_price_answer_keeps_the_doctor():
    messages = _fees_conversation("د. أحمد يوسف جلسة الاستشارة النفسية سعرها ٢٥٠ ريال. تحب أحجز لك موعد عنده؟",
                                  "نعم بكرا ان شاء الله")
    directive = graph._build_priced_doctor_affirmation_directive(messages, "s-none", "booking")
    assert "match_entity_for_booking" in directive and "احمد" in directive


def test_no_directive_when_the_reply_did_not_offer_booking():
    messages = _fees_conversation("د. أحمد يوسف جلسة الاستشارة النفسية سعرها ٢٥٠ ريال.", "نعم")
    assert graph._build_priced_doctor_affirmation_directive(messages, "s-none", "booking") == ""


def test_ambiguous_name_shows_only_the_candidates_and_remembers_them():
    import tools
    session_id = "sess-ambiguous-1"
    tools._BOOKING_SESSIONS.pop(session_id, None)
    candidates = [{"name": "العنود الخليفة"}, {"name": "نجود الخليفي"}]
    roster = {"status": "found", "doctors": [{"name": f"دكتور {i}"} for i in range(14)]}
    messages = [
        HumanMessage(content="نجود"),
        _call("1", "match_entity_for_booking", {"user_input": "نجود", "entity_type": "doctor"}),
        ToolMessage(content=json.dumps({"matched": False, "ambiguous": True, "candidates": candidates}),
                    name="match_entity_for_booking", tool_call_id="1"),
        _call("2", "find_available_doctors", {}),
        ToolMessage(content=json.dumps(roster), name="find_available_doctors", tool_call_id="2"),
    ]
    directive = graph._build_entity_list_directive(messages, session_id)
    assert "نجود الخليفي" in directive and "دكتور 5" not in directive
    assert "Print NO other list" in directive
    remembered = tools._get_booking_session(session_id)["last_list"]["items"]
    assert [i["name"] for i in remembered] == ["العنود الخليفة", "نجود الخليفي"]


def test_affirmation_is_read_by_meaning_when_the_turn_reading_says_so():
    assert graph._patient_confirms("ياريت يا دكتور بكرا الصبح", {"confirms": True}) is True
    assert graph._patient_confirms("ياريت يا دكتور بكرا الصبح", {"confirms": False}) is False
    assert graph._patient_confirms("اه", None) is True
    assert graph._patient_confirms("بكرا", None) is False


def test_single_doctor_yes_in_any_wording():
    session_id = "sess-single-1"
    import tools
    tools._BOOKING_SESSIONS.pop(session_id, None)
    tools._get_booking_session(session_id)["last_list"] = {"entity_type": "doctor", "items": [{"name": "ليلى الحربي"}]}
    messages = [
        HumanMessage(content="عاوز اسنان"),
        AIMessage(content="الدكتورة المتاحة هي د. ليلى الحربي - تحب أحجز لك عندها؟"),
        HumanMessage(content="تمام خلاص احجزي لي بكرا"),
    ]
    assert "ليلى الحربي" in graph._build_single_doctor_affirmation_directive(
        messages, session_id, "booking", reading={"confirms": True})
    assert graph._build_single_doctor_affirmation_directive(messages, session_id, "booking") == ""


def test_a_named_doctor_is_matched_even_when_a_neighbour_shares_the_family_name():
    import tools
    doctors = [{"name": "العنود الخليفة"}, {"name": "نجود الخليفي"}, {"name": "اسيلا الحسن"}]
    for typed in ("نجود الخليفة", "أ. نجود الخليفة"):
        result = tools._fuzzy_match(typed, doctors, ["name"])
        assert result["result"] == "matched" and result["item"]["name"] == "نجود الخليفي"
    # a bare given name is still a genuine choice
    people = [{"name": "احمد عبدالرحمن"}, {"name": "احمد عقيل"}]
    assert tools._fuzzy_match("احمد", people, ["name"])["result"] == "ambiguous"


def test_a_file_sent_by_the_patient_is_handed_to_staff():
    import agents.semantic_router as sr
    from langchain_core.messages import HumanMessage
    cv = "[Client sent a document: CV.pdf] https://x.example/y?id=1 [media attached]"
    assert graph._message_is_media([HumanMessage(content=cv)]) is True
    assert graph._message_is_media([HumanMessage(content="عايز احجز")]) is False
    facts = sr.TurnFacts(media_received=True)
    decision = sr.decide(None, facts)
    assert decision.handoff is True and decision.mode == sr.ROUTING_SAFETY
    # with a reading too
    assert sr.decide({"intent": "booking", "confidence": 1.0}, facts).handoff is True
    # a normal message is untouched
    assert sr.decide({"intent": "booking", "confidence": 1.0}, sr.TurnFacts()).handoff is False


def test_a_day_named_on_its_own_after_an_offer_is_not_treated_as_a_refusal():
    from langchain_core.messages import AIMessage, HumanMessage
    offer = AIMessage(content="المواعيد المتاحة ليوم الأربعاء 07/10/2026: 1 4:00 مساءً. أي رقم أو وقت تفضل؟")
    declines = {"declines": True, "changes_intent": False}
    for said in ("اليوم", "بكره", "الخميس"):
        assert graph._build_negation_directive([offer, HumanMessage(content=said)], declines) == ""
    # a real refusal still fires, even when it names a day
    assert graph._build_negation_directive([offer, HumanMessage(content="لا مش مناسب")], declines) != ""


def test_same_day_slots_are_dropped_unless_allowed(monkeypatch):
    import config, tools
    from datetime import datetime
    today = tools._local_now_naive("Asia/Riyadh").date().isoformat()
    slots = [{"_localStart": today + "T23:30:00"}, {"_localStart": "2099-01-01T10:00:00"}]
    monkeypatch.setattr(config, "ALLOW_SAME_DAY_BOOKING", False)
    kept, removed = tools._drop_same_day_slots(list(slots), "Asia/Riyadh")
    assert removed == 1 and kept == [slots[1]]
    monkeypatch.setattr(config, "ALLOW_SAME_DAY_BOOKING", True)
    assert tools._drop_same_day_slots(list(slots), "Asia/Riyadh") == (slots, 0)


def test_asking_for_today_gets_the_same_day_answer_not_todays_times():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    import json
    msgs = [HumanMessage(content="عايز اعدل موعدي"),
            AIMessage(content="", tool_calls=[{"id": "1", "name": "lookup_appointment", "args": {}}]),
            ToolMessage(content=json.dumps({"status": "found_one", "appointment": {"ref": "X"}}),
                        name="lookup_appointment", tool_call_id="1"),
            AIMessage(content="أي يوم تفضل؟"), HumanMessage(content="ابي اليوم")]
    directive = graph._build_relative_date_directive(msgs, "s-sameday", {"_timezone": "Asia/Riyadh"})
    assert "SAME-DAY BOOKING IS NOT AVAILABLE" in directive
    assert "get_available_slots_for_booking(" not in directive
    # tomorrow is still an ordinary date request
    msgs[-1] = HumanMessage(content="بكره")
    assert "SAME-DAY" not in graph._build_relative_date_directive(msgs, "s-sameday", {"_timezone": "Asia/Riyadh"})


def test_typing_the_word_the_assistant_asked_for_is_a_request_for_a_person():
    from langchain_core.messages import AIMessage, HumanMessage
    canned = ("وصلني الملف اللي أرسلته 📎\nحالياً ما أقدر أفتح الصور أو الملفات مباشرة، لكن أقدر أحوّلك "
              "لموظف خدمة العملاء يطّلع عليه ويساعدك — بس اكتب لي «موظف» وأحوّلك فوراً 👤")
    assert graph._types_the_word_we_asked_for([AIMessage(content=canned), HumanMessage(content="موظف")]) is True
    assert graph._types_the_word_we_asked_for([AIMessage(content=canned), HumanMessage(content="موظف.")]) is True
    # not the invited word / no invitation -> untouched
    assert graph._types_the_word_we_asked_for([AIMessage(content=canned), HumanMessage(content="عايز وظيفة")]) is False
    assert graph._types_the_word_we_asked_for([AIMessage(content="أهلا"), HumanMessage(content="موظف")]) is False


def test_schedule_requested_with_the_doctor_match_waits_for_the_match():
    from langchain_core.messages import AIMessage
    both = AIMessage(content="", tool_calls=[
        {"id": "1", "name": "match_entity_for_booking", "args": {"user_input": "نوره الماضي", "entity_type": "doctor"}},
        {"id": "2", "name": "get_doctor_schedule_for_booking", "args": {}},
    ])
    kept = graph._defer_calls_that_need_the_confirmation(both)
    assert [c["name"] for c in kept.tool_calls] == ["match_entity_for_booking"]
    alone = AIMessage(content="", tool_calls=[{"id": "3", "name": "get_doctor_schedule_for_booking", "args": {}}])
    assert graph._defer_calls_that_need_the_confirmation(alone) is alone


def test_removing_an_extra_question_removes_the_clause_that_hung_off_it():
    text = ("الله يشافيك 🌷\nلو سمحت، وش المشكلة؟ عشان أقدر أساعدك وأوجهك للتخصص المناسب.\n"
            "حاول ترتاح.\nعندنا دكاترة في الطب النفسي، تحب أحجز لك موعد؟")
    out, removed = graph._strip_extra_questions(text, {})
    assert removed == 1 and "عشان" not in out and "تحب أحجز لك موعد" in out


def test_a_generic_offer_naming_no_doctor_is_not_a_stale_doctor():
    offer = "عندنا دكاترة في تخصص الطب النفسي، تحب أحجز لك موعد مع واحد منهم؟"
    assert graph._reply_shows_doctor_for_service_with_no_lookup_this_turn(offer, {"messages": []}) is False
    listed = "الدكاترة المتاحين في تخصص الطب النفسي:\n1️⃣ د. ليلى الحربي"
    assert graph._reply_shows_doctor_for_service_with_no_lookup_this_turn(listed, {"messages": []}) is True
