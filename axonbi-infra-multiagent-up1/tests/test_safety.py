"""Tests for safety.py - deterministic reply fact checks. No network, no LLM."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import safety


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

_ids = iter(range(10_000))


def human(text="مرحبا"):
    return HumanMessage(content=text)


def ai(text=""):
    return AIMessage(content=text)


def tool(name, payload, ascii_escaped=False):
    return ToolMessage(
        content=json.dumps(payload, ensure_ascii=ascii_escaped),
        name=name,
        tool_call_id=f"call_{next(_ids)}",
    )


def state_of(*messages, **extra):
    state = {"messages": list(messages), "templates": {"_timezone": "Asia/Riyadh"}}
    state.update(extra)
    return state


def checks(reply, state, session=None):
    return [v["check"] for v in safety.check_reply(reply, state, session)]


SLOTS_FOUND = {
    "status": "found",
    "slots": [
        {"slotStart": "2026-09-10T10:30:00", "slotEnd": "2026-09-10T10:45:00",
         "date_display": "10/09/2026", "weekday_display": "الخميس",
         "time_display": "10:30 صباحًا", "doctorName": "د. طه مبروك"},
        {"slotStart": "2026-09-10T10:45:00", "slotEnd": "2026-09-10T11:00:00",
         "date_display": "10/09/2026", "weekday_display": "الخميس",
         "time_display": "10:45 صباحًا", "doctorName": "د. طه مبروك"},
        {"slotStart": "2026-09-10T11:00:00", "slotEnd": "2026-09-10T11:15:00",
         "date_display": "10/09/2026", "weekday_display": "الخميس",
         "time_display": "11:00 صباحًا", "doctorName": "د. طه مبروك"},
    ],
}

DOCTORS_FOUND = {
    "status": "found",
    "doctors": [
        {"id": 11, "name": "د. طه مبروك", "formatedName": "Dr. Taha Mabrouk",
         "altName": "د. طه مبروك", "specialtyName": "الباطنة"},
        {"id": 12, "name": "استشاري أحمد سمير الجندي", "formatedName": "Ahmed Samir El Gendy",
         "altName": "أحمد سمير الجندي", "specialtyName": "الباطنة"},
    ],
}

BRANCHES_FOUND = {
    "status": "found",
    "branches": [
        {"id": 1, "name": "المنار", "doctorCount": 3, "doctors": []},
        {"id": 2, "name": "النزهة", "doctorCount": 2, "doctors": []},
    ],
}

TIME_LIST = "المواعيد المتاحة:\n1️⃣ 10:30 صباحًا\n2️⃣ 10:45 صباحًا\n3️⃣ 11:00 صباحًا"


# ----------------------------------------------------------------------
# API shape
# ----------------------------------------------------------------------

def test_clean_reply_has_no_violations():
    assert safety.check_reply("أهلًا بك! كيف أقدر أساعدك اليوم؟", state_of(human())) == []


def test_empty_reply_and_empty_state():
    assert safety.check_reply("", {}) == []
    assert safety.check_reply("أهلًا", {}) == []


def test_violation_shape_and_note_length():
    reply = "تم الحجز بنجاح ✅ خذ بنادول، ولو استمر الألم أكثر من 3 أيام راجعنا.\n" + TIME_LIST
    violations = safety.check_reply(reply, state_of(human()))
    assert violations
    for v in violations:
        assert set(v) >= {"check", "claim", "note"}
        assert v["claim"]
        assert 0 < len(v["note"]) <= 160


def test_order_of_checks():
    reply = (
        "تم الحجز بنجاح ✅\n"
        "خذ بنادول، ولو استمر الألم أكثر من 3 أيام راجعنا.\n"
        "الدكاترة المتاحين:\n1️⃣ د. خالد الوهمي\n"
        + TIME_LIST + "\nيوم 12/09/2026 في فرع الدقي"
    )
    state = state_of(tool("list_branches_for_specialty", BRANCHES_FOUND), human("احجز"))
    assert checks(reply, state) == [
        safety.CLAIM_CHECK,
        safety.MEDICATION_CHECK,
        safety.CLINICAL_THRESHOLD_CHECK,
        safety.INVENTED_AVAILABILITY_CHECK,
        safety.TIMES_WITHOUT_LOOKUP_CHECK,
        safety.INVENTED_DOCTOR_CHECK,
        safety.INVENTED_BRANCH_CHECK,
    ]


def test_accepts_plain_dict_messages():
    state = {"messages": [
        {"role": "user", "content": "احجز"},
        {"type": "tool", "name": "create_new_booking",
         "content": json.dumps({"status": "success", "booking_ref": "GBN-2026-06-20-151"})},
    ]}
    assert safety.claimed_action("تم الحجز بنجاح ✅", state) is None


# ----------------------------------------------------------------------
# 1. claim gate
# ----------------------------------------------------------------------

def test_claim_booking_without_tool_fires():
    state = state_of(human("أكد الحجز"))
    violations = safety.check_reply("تم الحجز بنجاح ✅ نشوفك يوم الخميس", state)
    assert violations[0]["check"] == safety.CLAIM_CHECK == "unsupported_action_claim"
    assert "create_new_booking" in violations[0]["note"]
    assert safety.claimed_action("تم الحجز بنجاح ✅", state) == "create_new_booking"


def test_claim_booking_with_success_this_turn_is_fine():
    state = state_of(human("أكد"), tool("create_new_booking", {"status": "success", "booking_ref": "GBN-1"}))
    assert safety.claimed_action("تم الحجز بنجاح ✅", state) is None
    state = state_of(human("أكد"), tool("create_new_booking", {"status": "success_ref_pending", "booking_id": "9"}))
    assert safety.claimed_action("تم تأكيد موعدك", state) is None


def test_claim_booking_success_only_in_an_earlier_turn_fires():
    state = state_of(
        human("أكد"), tool("create_new_booking", {"status": "success"}), ai("تم الحجز"),
        human("واحجز لأخوي كمان"),
    )
    assert safety.claimed_action("تم الحجز بنجاح ✅", state) == "create_new_booking"


def test_claim_booking_failed_status_fires():
    state = state_of(human("أكد"), tool("create_new_booking", {"status": "slot_taken"}))
    assert safety.claimed_action("تم الحجز بنجاح", state) == "create_new_booking"


def test_question_is_not_a_claim():
    state = state_of(human("ابغى موعد"))
    assert safety.claimed_action("هل تحب أحجز؟", state) is None
    assert safety.claimed_action("هل تريد أن يتم الحجز الآن؟", state) is None
    assert safety.check_reply("أأكد الحجز؟", state) == []


def test_claim_followed_by_a_question_on_next_line_still_fires():
    state = state_of(human("أكد"))
    assert safety.claimed_action("تم الحجز بنجاح ✅\nتحتاج أي شيء ثاني؟", state) == "create_new_booking"


def test_english_booking_claim_fires():
    assert safety.claimed_action("Your appointment has been confirmed.", state_of(human("ok"))) == "create_new_booking"


def test_reschedule_success_covers_old_slot_cancelled_line():
    state = state_of(human("أكد التعديل"), tool("reschedule_appointment", {"status": "success"}))
    reply = "تم تعديل موعدك بنجاح ✅\n📌 تم إلغاء الموعد السابق المحدد بتاريخ 08/09/2026"
    assert safety.claimed_action(reply, state) is None


def test_cancel_claim_without_tool_fires():
    assert safety.claimed_action("تم إلغاء موعدك بنجاح", state_of(human("الغي"))) == "cancel_appointment"


def test_complaint_claim():
    reply = "تم تسجيل الشكوى وسيتواصل معك فريق الجودة"
    assert safety.claimed_action(reply, state_of(human("اشتكي"))) == "send_complaint_email"
    state = state_of(human("اشتكي"), tool("send_complaint_email", {"status": "sent", "via": "smtp"}))
    assert safety.claimed_action(reply, state) is None


def test_handoff_claim():
    reply = "جاري تحويلك لموظف خدمة العملاء"
    assert safety.claimed_action(reply, state_of(human("موظف"))) == "request_human_handoff"
    state = state_of(human("موظف"), tool("request_human_handoff", {"status": "handoff_requested"}))
    assert safety.claimed_action(reply, state) is None


def test_python_repr_tool_content_is_read():
    msg = ToolMessage(content="{'status': 'success', 'booking_ref': 'GBN-1'}",
                      name="create_new_booking", tool_call_id="c1")
    assert safety.claimed_action("تم الحجز بنجاح", state_of(human("أكد"), msg)) is None


# ----------------------------------------------------------------------
# 2. medication
# ----------------------------------------------------------------------

@pytest.mark.parametrize("reply", [
    "ممكن تاخذ بنادول لحد ما تشوف الدكتور",
    "البارستامول يخفف الألم",
    "إيبوبروفين ممكن يساعد",
    "تحتاج مضاد حيوي غالبًا",
    "You can take ibuprofen for now.",
])
def test_medication_fires(reply):
    assert safety.MEDICATION_CHECK in checks(reply, state_of(human("عندي صداع")))


def test_bare_word_medicine_in_a_complaint_does_not_fire():
    reply = "آسفين جدًا إن الدواء اللي اتوصفلك كان غلط. ممكن توضح لي تفاصيل أكثر عن الشكوى؟"
    assert safety.MEDICATION_CHECK not in checks(reply, state_of(human("الدواء اتوصفلي غلط")))


# ----------------------------------------------------------------------
# 3. clinical threshold
# ----------------------------------------------------------------------

@pytest.mark.parametrize("reply", [
    "لو استمر الألم أكثر من ٣ أيام لازم تشوف دكتور",
    "لو الحرارة فوق 38 روح الطوارئ",
    "اشرب سوائل كل 6 ساعات",
    "If the pain lasts more than 3 days, see a doctor.",
    "الجرعة 500 mg",
])
def test_clinical_threshold_fires(reply):
    assert safety.CLINICAL_THRESHOLD_CHECK in checks(reply, state_of(human("عندي ألم")))


@pytest.mark.parametrize("reply", [
    "⚕️ هذه معلومات عامة وليست تشخيصًا طبيًا مباشرة.",
    "لو الألم استمر أو زاد، الأفضل تشوف دكتور باطنة. ⚕️ هذه معلومات عامة وليست تشخيصًا.",
    "موعدك يوم 10/09/2026 الساعة 10:30 صباحًا",
    "الدكتور متواجد مرتين في الأسبوع في فرع المنار",
    "عندنا 3 فروع و 12 دكتور",
])
def test_clinical_threshold_false_positives(reply):
    assert safety.reply_states_clinical_threshold(reply) is None


# ----------------------------------------------------------------------
# 4. invented availability
# ----------------------------------------------------------------------

def test_availability_with_no_tool_fires():
    reply = "متاح يوم الثلاثاء 30-05-2024 الساعة 10:30"
    found = safety.find_invented_availability(reply, state_of(human("الفرع الأول")))
    assert "30-05-2024" in found and "10:30" in found


def test_availability_backed_by_slot_result_is_fine():
    state = state_of(human("الخميس"), tool("get_available_slots_for_booking", SLOTS_FOUND))
    reply = "أقرب موعد: الخميس 10/09/2026 — 10:30 صباحًا. هل يناسبك؟"
    assert safety.find_invented_availability(reply, state) == []


def test_availability_backed_by_earlier_turn_is_fine():
    state = state_of(human("الخميس"), tool("get_available_slots_for_booking", SLOTS_FOUND),
                     ai("..."), human("طيب"))
    assert safety.find_invented_availability("موعدك 10/9/2026 الساعة 10:45", state) == []


def test_availability_wrong_minutes_fires():
    state = state_of(human("الخميس"), tool("get_available_slots_for_booking", SLOTS_FOUND))
    assert safety.find_invented_availability("الخميس 10/09/2026 الساعة 10:50", state) == ["10:50"]


def test_booking_reference_is_not_a_date():
    state = state_of(human("الغي"), tool("cancel_appointment", {"status": "success"}))
    assert safety.find_invented_availability("رقم الحجز GBN-2026-06-20-151", state) == []


@pytest.mark.parametrize("reply", [
    "الدكتور امنية مغربي ما عنده عيادة يوم الأحد في أي فرع",
    "يوم 30 ما هو يوم من أيام الأسبوع، ممكن تقول لي اسم اليوم مثل الثلاثاء أو الأربعاء؟",
])
def test_weekday_denial_or_example_is_not_a_claim(reply):
    assert safety.find_invented_availability(reply, state_of(human("الأحد"))) == []


def test_weekday_offer_with_no_tool_fires():
    reply = "الأيام المتاحة:\n1️⃣ الخميس\n2️⃣ السبت\n3️⃣ الاثنين"
    assert set(safety.find_invented_availability(reply, state_of(human("متى")))) == {"الخميس", "السبت", "الاثنين"}


def test_weekday_of_a_returned_date_is_grounded():
    payload = {"status": "found", "days": [{"date": "2026-09-13", "date_display": "13/09/2026"}]}
    state = state_of(human("متى"), tool("list_available_days_for_booking", payload))
    assert safety.find_invented_availability("أقرب يوم متاح الأحد 13/09/2026", state) == []


def test_weekday_in_ascii_escaped_payload_is_grounded():
    state = state_of(human("متى"), tool("get_available_slots_for_booking", SLOTS_FOUND, ascii_escaped=True))
    assert safety.find_invented_availability("متاح الخميس", state) == []


def test_utc_schedule_shown_in_local_time_is_grounded():
    payload = {"status": "found", "schedule": [
        {"fromDateTime": "2026-09-10T12:00:00+00:00", "toDateTime": "2026-09-10T16:00:00+00:00"},
    ]}
    state = state_of(human("جدول الدكتور"), tool("get_doctor_schedule_for_booking", payload))
    assert safety.find_invented_availability("الدكتور موجود من 3:00 مساءً إلى 7:00 مساءً", state) == []


# ----------------------------------------------------------------------
# 5. times listed with no lookup this turn
# ----------------------------------------------------------------------

def test_time_list_with_lookup_only_in_earlier_turn_fires():
    state = state_of(human("الخميس"), tool("get_available_slots_for_booking", SLOTS_FOUND),
                     ai(TIME_LIST), human("اعرضها مرة ثانية"))
    assert safety.reply_lists_times_with_no_lookup_this_turn(TIME_LIST, state)
    assert safety.TIMES_WITHOUT_LOOKUP_CHECK in checks(TIME_LIST, state)


def test_time_list_with_slots_this_turn_is_fine():
    state = state_of(human("الخميس"), tool("get_available_slots_for_booking", SLOTS_FOUND))
    assert not safety.reply_lists_times_with_no_lookup_this_turn(TIME_LIST, state)
    assert safety.check_reply(TIME_LIST, state) == []


def test_time_list_with_schedule_tool_this_turn_is_fine():
    state = state_of(human("الخميس"), tool("resolve_available_day", {"status": "found", "date": "2026-09-10"}))
    assert not safety.reply_lists_times_with_no_lookup_this_turn(TIME_LIST, state)


def test_two_times_in_prose_is_not_a_list():
    reply = "1️⃣ 10:30 صباحًا\n2️⃣ 10:45 صباحًا"
    assert not safety.reply_lists_times_with_no_lookup_this_turn(reply, state_of(human("x")))


def test_time_list_with_no_human_message_is_silent():
    assert not safety.reply_lists_times_with_no_lookup_this_turn(TIME_LIST, {"messages": []})


# ----------------------------------------------------------------------
# 6. invented doctors
# ----------------------------------------------------------------------

ROSTER = "الدكاترة المتاحين في تخصص الباطنة:\n1️⃣ د. طه مبروك\n2️⃣ د. سارة عبد الله"


def test_invented_roster_with_no_tool_fires():
    found = safety.find_invented_doctors(ROSTER, state_of(human("معرفوش")))
    assert found == ["طه مبروك", "سارة عبد الله"]


def test_roster_partly_invented_fires_only_for_the_invented_name():
    state = state_of(human("باطنة"), tool("find_available_doctors", DOCTORS_FOUND))
    assert safety.find_invented_doctors(ROSTER, state) == ["سارة عبد الله"]


def test_doctor_from_an_earlier_tool_result_is_known():
    state = state_of(human("باطنة"), tool("find_available_doctors", DOCTORS_FOUND),
                     ai("..."), human("مين عندكم؟"))
    reply = "الدكاترة:\n1️⃣ د. طه مبروك\n2️⃣ استشاري أحمد سمير الجندي"
    assert safety.find_invented_doctors(reply, state) == []


def test_doctor_shortened_name_is_known():
    state = state_of(human("باطنة"), tool("find_available_doctors", DOCTORS_FOUND))
    assert safety.find_invented_doctors("الدكاترة:\n1️⃣ د. احمد الجندي", state) == []


def test_doctor_with_an_invented_extra_word_fires():
    state = state_of(human("باطنة"), tool("find_available_doctors", DOCTORS_FOUND))
    assert safety.find_invented_doctors("الدكاترة:\n1️⃣ د. طه مبروك الشافعي", state) == ["طه مبروك الشافعي"]


def test_doctor_known_from_session_memory():
    session = {"known_doctor_names": {"د. سارة عبد الله"}}
    state = state_of(human("باطنة"), tool("find_available_doctors", DOCTORS_FOUND))
    assert safety.find_invented_doctors(ROSTER, state, session) == []


def test_time_list_is_not_a_doctor_roster():
    reply = "مواعيد الدكتور طه مبروك:\n" + TIME_LIST
    state = state_of(human("الخميس"), tool("get_available_slots_for_booking", SLOTS_FOUND))
    assert safety.find_invented_doctors(reply, state) == []


def test_specialty_list_without_doctor_wording_is_ignored():
    reply = "التخصصات المتاحة:\n1️⃣ جراحة العظام\n2️⃣ طب الأطفال"
    assert safety.find_invented_doctors(reply, state_of(human("ايش عندكم"))) == []


def test_branch_list_with_address_is_not_a_doctor_roster():
    reply = ("الدكتور طه مبروك متاح في:\n1️⃣ الشاطئ\nالعنوان: شارع الملك فهد\n"
             "2️⃣ الربوة - مركز تناسق، RQMA3217، 7131 ابن النفيس، الرياض 14222")
    assert safety.find_invented_doctors(reply, state_of(human("وين"))) == []


def test_known_branch_list_is_not_a_doctor_roster():
    state = state_of(human("الفروع"), tool("list_branches_for_specialty", BRANCHES_FOUND))
    reply = "الدكتور طه مبروك يداوم في:\n1️⃣ المنار\n2️⃣ النزهة"
    assert safety.find_invented_doctors(reply, state) == []


def test_which_branch_question_is_left_to_the_branch_check():
    reply = "في أي فرع تفضل تحجز موعدك عند د. طه مبروك؟\n1️⃣ الشيخ زايد\n2️⃣ فرع آخر"
    assert safety.find_invented_doctors(reply, state_of(human("ياريت"))) == []


def test_doctor_in_prose_is_not_checked():
    assert safety.find_invented_doctors("الدكتور خالد الوهمي ممتاز", state_of(human("x"))) == []


# ----------------------------------------------------------------------
# 7. invented branches
# ----------------------------------------------------------------------

def _branch_state(*extra_msgs, **kw):
    return state_of(human("الفروع"), tool("list_branches_for_specialty", BRANCHES_FOUND), *extra_msgs, **kw)


def test_invented_branch_fires():
    reply = "الدكتور متاح في:\nفرع الدقي\nفرع المنار"
    assert safety.find_invented_branches(reply, _branch_state()) == ["الدقي"]


def test_real_branch_with_different_alef_or_ta_marbuta_is_fine():
    assert safety.find_invented_branches("متاح في فرع النزهه", _branch_state()) == []
    state = state_of(human("x"), tool("list_branches_for_specialty",
                                      {"status": "found", "branches": [{"name": "الأندلس"}]}))
    assert safety.find_invented_branches("متاح في فرع الاندلس", state) == []


def test_partial_branch_name_is_fine():
    state = state_of(human("x"), tool("list_branches_for_specialty",
                                      {"status": "found", "branches": [{"name": "الشيخ زايد"}]}))
    assert safety.find_invented_branches("متاح في فرع زايد", state) == []


def test_english_transliteration_is_fine():
    state = state_of(human("x"), tool("match_entity_info",
                                      {"status": "matched", "type": "branch", "name": "Al Nozha", "id": 2}))
    assert safety.find_invented_branches("تم اختيار فرع النزهة", state) == []
    assert safety.find_invented_branches("متاح في فرع الدقي", state) == ["الدقي"]


def test_generic_english_branch_translated_is_fine():
    state = state_of(human("x"), tool("list_branches_for_specialty",
                                      {"status": "found", "branches": [{"name": "Emergency"}]}))
    assert safety.find_invented_branches("متاح في فرع الطوارئ", state) == []


def test_ascii_escaped_branch_payload_is_fine():
    state = state_of(human("x"), tool("list_branches_for_specialty", BRANCHES_FOUND, ascii_escaped=True))
    assert safety.find_invented_branches("متاح في فرع المنار", state) == []


@pytest.mark.parametrize("reply", [
    "أبغى أتأكد من اختيار الدكتور والفرع أولاً",
    "أي فرع تفضل؟",
    "تحب تعرف عن خدمات أي فرع منهم؟",
    "الدكتور طه مبروك ما تم تأكيد فرع له بعد، ممكن تحدد لي الفرع اللي تفضل تحجز فيه؟",
    "حابب تحجز في أنهي فرع وانهي يوم؟",
    "متوفر في فرع واحد بس",
    "ما قدرنا نلاقي فرع بالرقم 1، تختار من: 1️⃣ المنار 2️⃣ النزهة",
    "فرع النزهه عنوانه: شارع الأمير سلطان",
    'ما لقيت فرع اسمه "فرع النيل"، الفروع المتاحة: فرع المنار وفرع النزهة',
])
def test_branch_false_positives(reply):
    assert safety.find_invented_branches(reply, _branch_state()) == []


def test_config_alias_counts_as_known():
    templates = {"_timezone": "Asia/Riyadh",
                 "_branch_aliases": [{"name": "Dokki", "aliases": ["Dokki", "الدقي"]}]}
    state = state_of(human("x"), templates=templates)
    assert safety.find_invented_branches("متاح في فرع الدقي", state) == []
    assert safety.find_invented_branches("متاح في فرع المعادي", state) == ["المعادي"]


def test_session_known_branch_counts_as_known():
    session = {"known_branch_names": {"الطوارئ"}}
    state = _branch_state(ai("..."), human("وفرع الطوارئ؟"))
    assert safety.find_invented_branches("متاح في فرع الطوارئ", state, session) == []
    assert safety.find_invented_branches("متاح في فرع الطوارئ", state) == ["الطوارئ"]


def test_nothing_known_stays_silent():
    assert safety.find_invented_branches("متاح في فرع الدقي", state_of(human("x"))) == []


def test_branch_check_reports_through_check_reply():
    violations = safety.check_reply("متاح في فرع الدقي", _branch_state())
    assert [v["check"] for v in violations] == [safety.INVENTED_BRANCH_CHECK]
    assert violations[0]["claim"] == "الدقي"


def test_module_is_standalone():
    import ast as _ast
    import pathlib
    tree = _ast.parse(pathlib.Path(safety.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, _ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"graph", "tools", "config", "langchain_core", "langchain_openai", "openai"}
