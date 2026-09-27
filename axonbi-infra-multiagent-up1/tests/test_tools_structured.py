"""
Structured tool arguments, the confirmation gate, the prepared booking
review, the psychiatry specialty bug and session persistence - driven
through the REAL tools against the fake hospital (no LLM, no network).
"""

import inspect
import json
import socket

import pytest
import requests
from langchain_core.messages import AIMessage, HumanMessage

import fake_hospital as fh


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError(f"real network attempted: args={args!r}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(requests.Session, "request", refuse)


@pytest.fixture
def hospital(monkeypatch):
    return fh.install(monkeypatch)


@pytest.fixture
def tools():
    import tools as tools_module
    return tools_module


class Conversation:
    """A state dict plus a ToolNode-style tool invoker."""

    def __init__(self, hospital, session_id):
        self.state = hospital.make_state(session_id=session_id)
        self.session_id = session_id
        self._n = 0

    def say(self, text):
        self.state["messages"].append(HumanMessage(content=text))

    def asked_to_confirm(self, action, target=None):
        self.state["pending_confirmation"] = {"action": action, "target": target}

    def session(self):
        import tools
        return tools._BOOKING_SESSIONS[self.session_id]

    def call(self, tool, **args):
        self._n += 1
        call_id = f"call_{self._n}"
        self.state["messages"].append(AIMessage(
            content="", tool_calls=[{"name": tool.name, "args": dict(args), "id": call_id}],
        ))
        message = tool.invoke({
            "type": "tool_call", "name": tool.name, "id": call_id,
            "args": {**args, "state": self.state},
        })
        self.state["messages"].append(message)
        payload = json.loads(message.content)
        assert isinstance(payload, dict), message.content
        return payload


def _specialty_id(c, tools, name):
    r = c.call(tools.list_specialties)
    assert r["status"] == "found", r
    return next(s["id"] for s in r["specialties"] if s["name"] == name)


def _to_slots(c, tools):
    """Dr. Ahmed Sami at Al Nuzha, Tuesday 29/09 - slot list shown."""
    derm = _specialty_id(c, tools, "طب الجلدية")
    assert c.call(tools.find_available_doctors, specialty_ids=[derm])["status"] == "found"
    assert c.call(tools.match_entity_for_booking, entity_type="doctor", option_number=1)["matched"]
    assert c.call(tools.match_entity_for_booking, entity_type="branch", name="النزهة")["matched"]
    day = c.call(tools.resolve_available_day, date="2026-09-29")
    assert day["status"] == "found", day
    slots = c.call(tools.get_available_slots_for_booking,
                   from_date=day["from_date"], to_date=day["to_date"])
    assert slots["status"] == "found", slots
    return slots["slots"]


def _to_review(c, tools):
    slots = _to_slots(c, tools)
    picked = c.call(tools.select_appointment_slot, option_number=2)
    assert picked["status"] == "selected"
    assert c.call(tools.compare_phone, provided_phone="0500000001",
                  channel_phone=fh.CHANNEL_PHONE) == {"status": "match"}
    review = c.call(tools.confirm_booking_review, patient_full_name="محمد عبدالله العمري")
    assert review["status"] == "review_ready", review
    return picked["slot"], review["review"], slots


# ==========================================================
# Slot picks by option_number
# ==========================================================

def test_slot_pick_by_option_number(hospital, tools):
    c = Conversation(hospital, "966500000001+slots")
    slots = _to_slots(c, tools)

    r = c.call(tools.select_appointment_slot, option_number=2)
    assert r["status"] == "selected"
    assert r["slot"]["slotStart"] == slots[1]["slotStart"]
    assert c.session()["selected_slot"]["slotStart"] == slots[1]["slotStart"]
    assert c.session()["selected_date"] == "2026-09-29"
    assert "29/09/2026" in c.session()["selected_date_display"]


@pytest.mark.parametrize("option", [0, 99, -1])
def test_slot_pick_out_of_range(hospital, tools, option):
    c = Conversation(hospital, f"966500000001+slots-oor{option}")
    slots = _to_slots(c, tools)
    r = c.call(tools.select_appointment_slot, option_number=option)
    assert r["status"] == "out_of_range" and r["list_size"] == len(slots), r
    assert r["hint"] and len(r["hint"]) <= 90
    assert not c.session().get("selected_slot")


def test_slot_pick_without_a_list(hospital, tools):
    c = Conversation(hospital, "966500000001+slots-none")
    assert c.call(tools.select_appointment_slot, option_number=1)["status"] == "no_list_shown"
    assert c.call(tools.select_reschedule_slot, option_number=1)["status"] == "no_list_shown"


def test_slot_options_carry_labels(hospital, tools):
    c = Conversation(hospital, "966500000001+labels")
    _to_slots(c, tools)
    items = c.session()["last_list"]["items"]
    assert items and all(isinstance(i.get("label"), str) and i["label"] for i in items)
    assert items[0]["label"] == "الثلاثاء 29/09/2026 · 4:20 مساءً"


# ==========================================================
# Entity picks: option_number and name
# ==========================================================

def test_doctor_pick_by_option_number(hospital, tools):
    c = Conversation(hospital, "966500000001+entity-opt")
    derm = _specialty_id(c, tools, "طب الجلدية")
    c.call(tools.find_available_doctors, specialty_ids=[derm])

    r = c.call(tools.match_entity_for_booking, entity_type="doctor", option_number=2)
    assert r["matched"] is True and r["item"]["id"] == fh.DR_REEM, r
    assert c.session()["doctor_id"] == fh.DR_REEM

    r = c.call(tools.match_entity_for_booking, entity_type="doctor", option_number=9)
    assert r["status"] == "out_of_range" and r["list_size"] == 2


def test_entity_option_without_a_list(hospital, tools):
    c = Conversation(hospital, "966500000001+entity-nolist")
    r = c.call(tools.match_entity_for_booking, entity_type="branch", option_number=1)
    assert r["status"] == "no_list_shown", r
    r = c.call(tools.match_entity_info, entity_type="branch", option_number=1)
    assert r["status"] == "no_list_shown", r


def test_madi_is_ambiguous_across_specialties(hospital, tools):
    c = Conversation(hospital, "966500000001+madi")
    r = c.call(tools.match_entity_for_booking, entity_type="doctor", name="ماضي")
    assert r.get("ambiguous") is True, r
    assert {d["id"] for d in r["candidates"]} == {fh.DR_SAAD_MADI, fh.NOURA_MADI}
    # The candidates become the options: option 1 then picks by position.
    first = r["candidates"][0]["id"]
    r = c.call(tools.match_entity_for_booking, entity_type="doctor", option_number=1)
    assert r["matched"] is True and r["item"]["id"] == first

    info = c.call(tools.match_entity_info, entity_type="doctor", name="ماضي")
    assert info["status"] == "ambiguous", info
    assert {d["id"] for d in info["candidates"]} == {fh.DR_SAAD_MADI, fh.NOURA_MADI}


def test_madi_resolves_within_the_specialty_passed(hospital, tools):
    c = Conversation(hospital, "966500000001+madi-psy")
    psy = _specialty_id(c, tools, "طب نفسي")
    assert c.call(tools.find_available_doctors, specialty_ids=[psy])["status"] == "found"
    r = c.call(tools.match_entity_for_booking, entity_type="doctor", name="ماضي")
    assert r["matched"] is True and r["needsConfirmation"] is False, r
    assert r["item"]["id"] == fh.DR_SAAD_MADI


def test_specialty_pick_by_option_number(hospital, tools):
    c = Conversation(hospital, "966500000001+spec-opt")
    r = c.call(tools.list_specialties)
    names = [s["name"] for s in r["specialties"]]
    position = names.index("طب نفسي") + 1

    r = c.call(tools.match_entity_for_booking, entity_type="specialty", option_number=position)
    assert r["matched"] is True and r["item"]["id"] == fh.SPEC_PSY, r
    assert c.session()["specialty_ids"] == [fh.SPEC_PSY]
    assert c.session()["specialty_display_name"] == "طب نفسي"

    # A bare find_available_doctors now reuses the settled specialty.
    r = c.call(tools.find_available_doctors)
    assert {d["id"] for d in r["doctors"]} == set(fh.PSYCHIATRISTS)


def test_match_entity_info_list_then_option(hospital, tools):
    c = Conversation(hospital, "966500000001+info-opt")
    r = c.call(tools.match_entity_info, entity_type="branch")
    assert r["status"] == "list"
    names = [b["name"] for b in r["items"]]
    r = c.call(tools.match_entity_info, entity_type="branch", option_number=names.index("المنار") + 1)
    assert r["status"] == "matched" and r["item"]["id"] == fh.MANAR, r
    r = c.call(tools.match_entity_info, entity_type="branch", option_number=7)
    assert r["status"] == "out_of_range"


# ==========================================================
# The psychiatry bug
# ==========================================================

def test_psychiatry_returns_exactly_the_five_psychiatrists(hospital, tools):
    c = Conversation(hospital, "966500000001+psy")
    psy = _specialty_id(c, tools, "طب نفسي")

    r = c.call(tools.find_available_doctors, specialty_ids=[psy], allow_broader_search=False)
    assert r["status"] == "found", r
    ids = [d["id"] for d in r["doctors"]]
    assert len(ids) == 5 and set(ids) == set(fh.PSYCHIATRISTS)
    assert not set(ids) & set(fh.PSYCHOLOGISTS)
    assert "د. سعد الماضي" in [d["name"] for d in r["doctors"]]
    assert "نورة الماضي" not in [d["name"] for d in r["doctors"]]

    assert c.session()["specialty_display_name"] == "طب نفسي"
    labels = [i["label"] for i in c.session()["last_list"]["items"]]
    assert "د. سعد الماضي - طب نفسي" in labels

    # The psychologists are a separate specialty and stay reachable by it.
    psyc = _specialty_id(c, tools, "أخصائي نفسي")
    r = c.call(tools.find_available_doctors, specialty_ids=[psyc])
    assert {d["id"] for d in r["doctors"]} == set(fh.PSYCHOLOGISTS)


def test_specialty_stem_expansion_is_gone(tools):
    assert not hasattr(tools, "_expand_specialty_ids")
    assert not hasattr(tools, "_SPECIALTY_STOPWORDS")


# ==========================================================
# resolve_available_day(date)
# ==========================================================

def _doctor_ahmed(c, tools):
    derm = _specialty_id(c, tools, "طب الجلدية")
    c.call(tools.find_available_doctors, specialty_ids=[derm])
    c.call(tools.match_entity_for_booking, entity_type="doctor", option_number=1)


def test_resolve_available_day_iso_date(hospital, tools):
    c = Conversation(hospital, "966500000001+day")
    _doctor_ahmed(c, tools)
    c.call(tools.match_entity_for_booking, entity_type="branch", name="النزهة")

    r = c.call(tools.resolve_available_day, date="2026-09-29")
    assert r["status"] == "found", r
    assert r["date"] == "2026-09-29" and r["date_display"] == "29/09/2026"
    assert r["first_time_display"] == "4:20 مساءً"
    assert c.session()["selected_date"] == "2026-09-29"

    # Wednesday: Dr. Ahmed does not work at Al Nuzha - nearest open dates ride back.
    r = c.call(tools.resolve_available_day, date="2026-09-30")
    assert r["status"] == "not_found", r
    assert r["nearest_dates"] and all(d["date"] != "2026-09-30" for d in r["nearest_dates"])

    assert c.call(tools.resolve_available_day, date="بكره")["status"] == "invalid_date"
    assert c.call(tools.resolve_available_day, date="2026-09-01")["status"] == "past_date"


def test_resolve_available_day_infers_branch_from_the_date(hospital, tools):
    c = Conversation(hospital, "966500000001+day-branch")
    _doctor_ahmed(c, tools)
    # Wednesday is an Al Manar day only for Dr. Ahmed Sami.
    r = c.call(tools.resolve_available_day, date="2026-09-30")
    assert r["status"] == "found", r
    assert c.session()["branch_id"] == fh.MANAR
    assert c.session()["branch_auto_resolved"] is True


def test_resolve_available_day_needs_a_doctor(hospital, tools):
    c = Conversation(hospital, "966500000001+day-nodoc")
    assert c.call(tools.resolve_available_day, date="2026-09-29")["status"] == "missing_doctor"


# ==========================================================
# Booking review (prepare) + create
# ==========================================================

def test_review_failure_statuses(hospital, tools):
    c = Conversation(hospital, "966500000001+review-fail")
    name = "محمد عبدالله العمري"
    assert c.call(tools.confirm_booking_review, patient_full_name=name)["status"] == "missing_doctor"

    _doctor_ahmed(c, tools)   # two branches -> none settled yet
    assert c.call(tools.confirm_booking_review, patient_full_name=name)["status"] == "missing_branch"

    c.call(tools.match_entity_for_booking, entity_type="branch", name="النزهة")
    assert c.call(tools.confirm_booking_review, patient_full_name=name)["status"] == "missing_slot"

    day = c.call(tools.resolve_available_day, date="2026-09-29")
    c.call(tools.get_available_slots_for_booking, from_date=day["from_date"], to_date=day["to_date"])
    c.call(tools.select_appointment_slot, option_number=1)
    r = c.call(tools.confirm_booking_review, patient_full_name=name)
    assert r["status"] == "phone_not_verified" and r["hint"]

    c.call(tools.compare_phone, provided_phone="0500000001", channel_phone=fh.CHANNEL_PHONE)
    assert c.call(tools.confirm_booking_review, patient_full_name="محمد")["status"] == "name_incomplete"
    assert not c.session()["review"]

    r = c.call(tools.confirm_booking_review, patient_full_name=name, email="m@example.com")
    assert r["status"] == "review_ready"
    assert set(r["review"]) == {"patient_full_name", "email", "mobile_number", "branch", "doctor",
                                "weekday", "date_display", "time_display", "slotStart"}
    assert r["review"]["weekday"] == "الثلاثاء" and r["review"]["email"] == "m@example.com"
    assert c.session()["review_shown"] is True
    json.dumps(c.session()["review"])


def test_create_refuses_without_review(hospital, tools):
    c = Conversation(hospital, "966500000001+no-review")
    _to_slots(c, tools)
    slot = c.call(tools.select_appointment_slot, option_number=1)["slot"]
    c.call(tools.compare_phone, provided_phone="0500000001", channel_phone=fh.CHANNEL_PHONE)
    c.asked_to_confirm("book", slot["slotStart"])
    r = c.call(tools.create_new_booking, patient_confirmed=True)
    assert r["status"] == "needs_review", r
    assert not hospital.calls_to("create_booking")


def test_selecting_another_slot_clears_the_review(hospital, tools):
    c = Conversation(hospital, "966500000001+review-slot")
    slot, review, _slots = _to_review(c, tools)
    other = c.call(tools.select_appointment_slot, option_number=3)["slot"]
    assert c.session()["review"] is None and c.session()["review_shown"] is False
    c.asked_to_confirm("book", other["slotStart"])
    assert c.call(tools.create_new_booking, patient_confirmed=True)["status"] == "needs_review"


def test_changing_doctor_clears_the_review(hospital, tools):
    c = Conversation(hospital, "966500000001+review-doc")
    _to_review(c, tools)
    # Picking another doctor (by option, from a roster shown clinic-wide).
    derm = _specialty_id(c, tools, "طب الجلدية")
    c.call(tools.find_available_doctors, specialty_ids=[derm], all_branches=True)
    c.session()["review"] = {"slotStart": "x"}      # pretend a review survived the roster
    c.session()["review_shown"] = True
    r = c.call(tools.match_entity_for_booking, entity_type="doctor", option_number=2)
    assert r["matched"] is True and r["item"]["id"] == fh.DR_REEM, r
    assert c.session()["review"] is None and c.session()["review_shown"] is False


def test_changing_branch_clears_the_review(hospital, tools):
    c = Conversation(hospital, "966500000001+review-branch")
    _to_review(c, tools)
    c.call(tools.match_entity_for_booking, entity_type="branch", name="المنار")
    assert c.session()["branch_id"] == fh.MANAR
    assert c.session()["review"] is None


def test_create_refuses_a_review_for_another_slot(hospital, tools):
    c = Conversation(hospital, "966500000001+review-mismatch")
    slot, review, slots = _to_review(c, tools)
    c.session()["review"]["slotStart"] = slots[0]["slotStart"]   # stale review
    c.asked_to_confirm("book", slot["slotStart"])
    r = c.call(tools.create_new_booking, patient_confirmed=True)
    assert r["status"] == "review_mismatch", r
    assert not hospital.calls_to("create_booking")


def test_create_books_from_the_review(hospital, tools):
    c = Conversation(hospital, "966500000001+review-book")
    slot, review, _slots = _to_review(c, tools)
    c.asked_to_confirm("book", slot["slotStart"])
    # Whatever the model types, the review's values are what get booked.
    r = c.call(tools.create_new_booking, patient_confirmed=True,
               patient_full_name="اسم آخر تماما", mobile_number="+966599999999")
    assert r["status"] == "success", r
    args = hospital.calls_to("create_booking")[0]
    assert args["patient_full_name"] == "محمد عبدالله العمري"
    assert args["mobile_number"] == fh.CHANNEL_PHONE
    assert args["booking_time_from"] == slot["slotStart"]


# ==========================================================
# The confirmation gate on every irreversible tool
# ==========================================================

def _cancel_setup(c, tools):
    appt = c.call(tools.lookup_appointment, use_channel_identity=True)["appointment"]
    assert c.session()["selected_appointment"]["ref"] == "TNS-10001"
    return lambda: c.call(tools.cancel_appointment, booking_id=appt["id"],
                          patient_confirmed=c._confirmed), "cancel", "TNS-10001"


def _reschedule_setup(c, tools):
    appt = c.call(tools.lookup_appointment, use_channel_identity=True)["appointment"]
    c.call(tools.get_available_reschedule_slots, ref_number="TNS-10001",
           from_date="2026-10-05T00:00:00+03:00", to_date="2026-10-05T23:59:59+03:00")
    slot = c.call(tools.select_reschedule_slot, option_number=1)["slot"]
    return lambda: c.call(tools.reschedule_appointment, booking_id=appt["id"],
                          patient_confirmed=c._confirmed,
                          new_time_from=slot["slotStart"], new_time_to=slot["slotEnd"]), \
        "reschedule", appt["id"]


def _book_setup(c, tools):
    slot, _review, _slots = _to_review(c, tools)
    return lambda: c.call(tools.create_new_booking, patient_confirmed=c._confirmed), \
        "book", slot["slotStart"]


def _complaint_setup(c, tools):
    return lambda: c.call(
        tools.send_complaint_email, patient_name="محمد العتيبي", phone=fh.CHANNEL_PHONE,
        branch="النزهة", category="الاستقبال",
        details="الاستقبال في فرع النزهة خلوني انتظر ساعة كاملة",
        patient_confirmed=c._confirmed), "complaint", None


SETUPS = {"cancel": _cancel_setup, "reschedule": _reschedule_setup,
          "book": _book_setup, "complaint": _complaint_setup}
SIDE_EFFECT = {"cancel": "cancel_booking_by_guid", "reschedule": "reschedule_booking",
               "book": "create_booking"}


def _acted(hospital, action):
    if action == "complaint":
        return bool(hospital.outbound)
    return bool(hospital.calls_to(SIDE_EFFECT[action]))


@pytest.mark.parametrize("action", sorted(SETUPS))
def test_gate_refuses_without_pending_confirmation(hospital, tools, action):
    c = Conversation(hospital, f"966500000001+gate-none-{action}")
    run, _action, _target = SETUPS[action](c, tools)
    c._confirmed = True
    r = run()
    assert r["status"] == "needs_confirmation", r
    assert not _acted(hospital, action)


@pytest.mark.parametrize("action", sorted(SETUPS))
def test_gate_refuses_when_not_confirmed(hospital, tools, action):
    c = Conversation(hospital, f"966500000001+gate-no-{action}")
    run, gate_action, target = SETUPS[action](c, tools)
    c.asked_to_confirm(gate_action, target)
    c._confirmed = False
    r = run()
    assert r["status"] == "not_confirmed", r
    assert not _acted(hospital, action)


@pytest.mark.parametrize("action", ["cancel", "reschedule", "book"])
def test_gate_refuses_on_target_mismatch(hospital, tools, action):
    c = Conversation(hospital, f"966500000001+gate-mismatch-{action}")
    run, gate_action, _target = SETUPS[action](c, tools)
    c.asked_to_confirm(gate_action, "SOMETHING-ELSE")
    c._confirmed = True
    r = run()
    assert r["status"] == "confirmation_mismatch", r
    assert not _acted(hospital, action)


def test_gate_refuses_a_confirmation_for_another_action(hospital, tools):
    c = Conversation(hospital, "966500000001+gate-other")
    run, _a, _t = _complaint_setup(c, tools)
    c.asked_to_confirm("cancel", "TNS-10001")
    c._confirmed = True
    assert run()["status"] == "needs_confirmation"
    assert not hospital.outbound


@pytest.mark.parametrize("action", sorted(SETUPS))
def test_gate_runs_when_everything_holds(hospital, tools, action):
    c = Conversation(hospital, f"966500000001+gate-ok-{action}")
    run, gate_action, target = SETUPS[action](c, tools)
    c.asked_to_confirm(gate_action, target)
    c._confirmed = True
    r = run()
    assert r["status"] in ("success", "sent"), r
    assert _acted(hospital, action)


def test_cancel_confirmation_by_reference_or_guid(hospital, tools):
    c = Conversation(hospital, "966500000001+gate-guid")
    appt = c.call(tools.lookup_appointment, use_channel_identity=True)["appointment"]
    c.asked_to_confirm("cancel", appt["id"])     # the GUID is a valid target too
    r = c.call(tools.cancel_appointment, booking_id="TNS-10001", patient_confirmed=True)
    assert r == {"status": "success"}
    assert c.session()["selected_appointment"]["status"] == tools.CANCELLED_STATUS_NAME


def test_handoff_needs_only_the_patients_agreement(hospital, tools):
    c = Conversation(hospital, "966500000001+handoff")
    r = c.call(tools.request_human_handoff, reason="asked for staff", patient_agreed=False)
    assert r["status"] == "not_requested" and r["hint"]
    r = c.call(tools.request_human_handoff, reason="asked for staff", patient_agreed=True)
    assert r == {"status": "handoff_requested"}


def test_share_location_trusts_the_call(hospital, tools):
    c = Conversation(hospital, "966500000001+location")
    c.say("فرع المنار")   # no location wording - the model decided anyway
    r = c.call(tools.share_branch_location, branch_name="المنار")
    assert r == {"status": "location_requested", "branch_name": "المنار"}


# ==========================================================
# Appointments on the session
# ==========================================================

def test_appointments_recorded_and_picked_by_option(hospital, tools):
    # A second active booking for the channel phone -> found_many.
    hospital.bookings.append(hospital._booking_row(
        ref="TNS-10050", booking_id=fh._guid("booking:TNS-10050"), patient="محمد عبدالله العمري",
        phone=fh.CHANNEL_PHONE, email="", slot=hospital._slot_template(fh.DR_KHALID, fh.NUZHA,
                                                                      "2026-10-03T14:00:00+00:00"),
        start="2026-10-03T14:00:00+00:00", end="2026-10-03T14:30:00+00:00", status=1,
    ))
    c = Conversation(hospital, "966500000001+appts")
    r = c.call(tools.lookup_appointment, use_channel_identity=True)
    assert r["status"] == "found_many", r
    session = c.session()
    assert [a["ref"] for a in session["appointments"]] == [a["ref"] for a in r["appointments"]]
    assert set(session["appointments"][0]) == {"ref", "guid", "doctor", "specialty", "branch",
                                               "date_display", "time_display", "status"}
    assert all(i["label"] for i in session["last_list"]["items"])

    r2 = c.call(tools.check_booking_status, option_number=2)
    assert r2["status"] == "active" and r2["appointment"]["ref"] == r["appointments"][1]["ref"]
    assert session["selected_appointment"]["ref"] == r["appointments"][1]["ref"]

    assert c.call(tools.check_booking_status, option_number=5)["status"] == "out_of_range"
    fresh = Conversation(hospital, "966500000001+appts-fresh")
    assert fresh.call(tools.check_booking_status, option_number=1)["status"] == "no_list_shown"
    assert fresh.call(tools.check_booking_status, ref_number="TNS-10001")["status"] == "active"


# ==========================================================
# Persistence
# ==========================================================

def test_export_restore_round_trip(hospital, tools):
    c = Conversation(hospital, "966500000001+persist")
    slot, review, _slots = _to_review(c, tools)
    c.call(tools.lookup_appointment, use_channel_identity=True)

    snapshot = tools.export_session(c.session_id)
    text = json.dumps(snapshot, ensure_ascii=False)       # JSON-serializable
    assert isinstance(snapshot["verified_phones"], list)
    assert snapshot["verified_phones"] == sorted(snapshot["verified_phones"])
    assert "_touched_at" not in snapshot

    original = tools._BOOKING_SESSIONS.pop(c.session_id)
    assert tools.export_session(c.session_id) is None
    assert tools.restore_session(c.session_id, json.loads(text)) is True
    restored = tools._BOOKING_SESSIONS[c.session_id]

    for key in tools._SESSION_SET_KEYS:
        assert isinstance(restored[key], set)
        assert restored[key] == set(original[key])
    for key in ("doctor_id", "branch_id", "selected_slot", "review", "selected_date",
                "appointments", "selected_appointment", "specialty_display_name", "last_list"):
        assert json.dumps(restored[key], sort_keys=True) == json.dumps(
            tools._json_safe(original[key]), sort_keys=True), key

    # A live session is never overwritten.
    assert tools.restore_session(c.session_id, {"doctor_id": "other"}) is False
    assert tools._BOOKING_SESSIONS[c.session_id]["doctor_id"] == original["doctor_id"]

    # And the restored session still books.
    c.asked_to_confirm("book", slot["slotStart"])
    assert c.call(tools.create_new_booking, patient_confirmed=True)["status"] == "success"


# ==========================================================
# Removed heuristics / tool surface
# ==========================================================

REMOVED = (
    "_parse_clock_time", "_slot_matches_clock_time", "_slots_at_clock_time",
    "_extract_selection_number", "_CLOCK_TIME_RE", "_PERIOD_WORDS_AM", "_PERIOD_WORDS_PM",
    "_PERIOD_WORDS_NOON", "_ARABIC_DIGIT_MAP", "_is_generic_entity_word",
    "_is_entity_list_request", "_GENERIC_ENTITY_WORDS", "_LIST_REQUEST_CUES",
    "_LIST_REQUEST_FILLER", "_expand_specialty_ids", "_SPECIALTY_STOPWORDS",
    "_LOCATION_REQUEST_CUE_RE", "_BARE_PICK_RE", "_bare_disambiguation_reply",
    "_LOCATION_INTENT_LOOKBACK", "resolve_weekday_index", "resolve_relative_date",
    "_WEEKDAY_NAMES", "_WEEKDAY_FOLD_MAP", "_AMBIGUOUS_WEEKDAY_KEYS", "_WEEKDAY_CUE_WORDS",
    "_RELATIVE_DATE_OFFSETS", "_fold_weekday_token", "get_next_weekday_date",
    "_review_card_was_shown", "_REVIEW_CARD_QUESTION_FALLBACKS",
    "_complaint_explicitly_confirmed", "_handoff_recently_raised",
    "_EXPLICIT_HUMAN_REQUEST_ROOTS", "_booking_summary_for_confirmation",
)


def _code_only(source: str) -> str:
    """`source` without comments and string literals - the code that runs."""
    import io
    import tokenize
    kept = [tok.string for tok in tokenize.generate_tokens(io.StringIO(source).readline)
            if tok.type not in (tokenize.COMMENT, tokenize.STRING)]
    return " ".join(kept)


def test_removed_heuristics_are_gone(tools):
    for name in REMOVED:
        assert not hasattr(tools, name), name
    assert "intent" not in vars(tools), "tools.py must not import intent"

    names = {t.name for t in tools.ALL_TOOLS}
    assert "get_next_weekday_date" not in names
    for t in tools.ALL_TOOLS:
        source = _code_only(inspect.getsource(t.func))
        for name in REMOVED + ("intent.",):
            assert name not in source, (t.name, name)


def test_tool_signatures_are_structured(tools):
    def args(tool):
        return set(tool.args)

    assert args(tools.select_appointment_slot) == {"option_number"}
    assert args(tools.select_reschedule_slot) == {"option_number"}
    assert args(tools.match_entity_for_booking) == {"entity_type", "name", "option_number"}
    assert args(tools.match_entity_info) == {"entity_type", "name", "option_number"}
    assert args(tools.check_booking_status) == {"ref_number", "option_number"}
    assert args(tools.resolve_available_day) == {"date", "branch_name"}
    for tool in (tools.lookup_appointment, tools.check_booking_status,
                 tools.get_doctor_schedule, tools.get_available_reschedule_slots):
        assert "language" not in args(tool), tool.name
    for tool in (tools.cancel_appointment, tools.reschedule_appointment,
                 tools.create_new_booking, tools.send_complaint_email):
        assert "patient_confirmed" in args(tool), tool.name
    assert "patient_agreed" in args(tools.request_human_handoff)
    assert args(tools.confirm_booking_review) == {"patient_full_name", "email"}


def test_tool_descriptions_are_short(tools):
    for t in tools.ALL_TOOLS:
        doc = inspect.getdoc(t.func) or ""
        assert len(doc) <= 350, (t.name, len(doc))
        assert "_guidance" not in doc and "STEP" not in doc, t.name


# ==========================================================
# Production regressions (origin/tanasuq-production 820c5ed + 51f90d4)
# ==========================================================

def test_named_branch_wins_and_is_never_swapped(hospital, tools):
    # Session 201000625084: "المنار الاثنين" was booked at النزهة. Dr. Ahmed
    # Sami works Tuesdays ONLY at النزهة; asking for المنار on a Tuesday is
    # not_found at المنار - never a النزهة day.
    c = Conversation(hospital, "966500000001+named-branch")
    _doctor_ahmed(c, tools)
    r = c.call(tools.resolve_available_day, date="2026-09-29", branch_name="المنار")
    assert r["status"] in ("not_found", "fully_booked"), r
    assert r["branch"] == "المنار"
    assert c.session()["branch_id"] == fh.MANAR
    assert c.session()["branch_auto_resolved"] is False
    import datetime as real
    for day in r["nearest_dates"]:
        # Al Manar is Monday/Wednesday for this doctor.
        assert real.date.fromisoformat(day["date"]).weekday() in (0, 2), day
    assert not any(call.get("branch_ids") == [fh.NUZHA] for call in hospital.calls_to("get_doctor_schedule_slots"))


def test_named_branch_overrides_the_session_branch(hospital, tools):
    c = Conversation(hospital, "966500000001+named-over")
    _doctor_ahmed(c, tools)
    c.call(tools.match_entity_for_booking, entity_type="branch", name="النزهة")
    r = c.call(tools.resolve_available_day, date="2026-09-30", branch_name="المنار")
    assert r["status"] == "found" and r["branch"] == "المنار", r
    assert c.session()["branch_id"] == fh.MANAR
    slots = c.call(tools.get_available_slots_for_booking, from_date=r["from_date"], to_date=r["to_date"])
    assert slots["status"] == "found"
    assert hospital.calls_to("get_doctor_schedule_slots")[-1]["branch_ids"] == [fh.MANAR]


def test_auto_resolved_branch_is_named_in_the_result(hospital, tools):
    c = Conversation(hospital, "966500000001+auto-branch")
    _doctor_ahmed(c, tools)
    r = c.call(tools.resolve_available_day, date="2026-09-30")
    assert r["status"] == "found", r
    assert r["branch"] == "المنار" and r["branch_auto_resolved"] is True


def test_unknown_branch_name_is_not_guessed(hospital, tools):
    c = Conversation(hospital, "966500000001+branch-unknown")
    _doctor_ahmed(c, tools)
    r = c.call(tools.resolve_available_day, date="2026-09-29", branch_name="فرع الشرقية الجديد")
    assert r["status"] == "branch_not_matched", r
    assert not c.session()["branch_id"]


def test_reschedule_slots_stay_at_the_bookings_branch(hospital, tools):
    c = Conversation(hospital, "966500000001+resched-branch")
    appt = c.call(tools.lookup_appointment, use_channel_identity=True)["appointment"]
    assert appt["branchId"] == fh.MANAR
    # A whole week: Al Nuzha (Tue/Thu/Sun) must never appear.
    r = c.call(tools.get_available_reschedule_slots, ref_number="TNS-10001",
               from_date="2026-09-29T00:00:00+03:00", to_date="2026-10-05T23:59:59+03:00")
    assert r["status"] == "found", r
    assert {s["branchId"] for s in r["slots"]} == {fh.MANAR}
    assert hospital.calls_to("get_doctor_schedule_slots")[-1]["branch_ids"] == [fh.MANAR]
    # Tuesday alone is an Al Nuzha day -> nothing to offer.
    r = c.call(tools.get_available_reschedule_slots, ref_number="TNS-10001",
               from_date="2026-09-29T00:00:00+03:00", to_date="2026-09-29T23:59:59+03:00")
    assert r["status"] == "not_found", r


def test_reschedule_refuses_a_slot_at_another_branch(hospital, tools):
    c = Conversation(hospital, "966500000001+resched-other")
    appt = c.call(tools.lookup_appointment, use_channel_identity=True)["appointment"]
    c.call(tools.get_available_reschedule_slots, ref_number="TNS-10001",
           from_date="2026-10-05T00:00:00+03:00", to_date="2026-10-05T23:59:59+03:00")
    c.call(tools.select_reschedule_slot, option_number=1)
    # An Al Nuzha Tuesday slot sneaks into the lock.
    nuzha = {"slotStart": "2026-09-29T13:20:00", "slotEnd": "2026-09-29T13:40:00",
             "branchId": fh.NUZHA}
    c.session()["selected_reschedule_slot"] = dict(nuzha)
    c.asked_to_confirm("reschedule", appt["id"])
    r = c.call(tools.reschedule_appointment, booking_id=appt["id"], patient_confirmed=True,
               new_time_from=nuzha["slotStart"], new_time_to=nuzha["slotEnd"])
    assert r["status"] == "different_branch", r
    assert not hospital.calls_to("reschedule_booking")

    # Even without a branch on the slot, the live re-check is branch-locked.
    c.session()["selected_reschedule_slot"] = {"slotStart": nuzha["slotStart"], "slotEnd": nuzha["slotEnd"]}
    r = c.call(tools.reschedule_appointment, booking_id=appt["id"], patient_confirmed=True,
               new_time_from=nuzha["slotStart"], new_time_to=nuzha["slotEnd"])
    assert r["status"] == "slot_unavailable", r
    assert not hospital.calls_to("reschedule_booking")
    assert hospital.booking("TNS-10001")["branchId"] == fh.MANAR
