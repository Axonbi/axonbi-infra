"""
The fake hospital backend, driven through the REAL tool functions -
no LLM anywhere. Each tool is invoked the way ToolNode invokes it: a
ToolCall whose args carry the injected `state`, and the resulting
ToolMessage appended to state["messages"] so tools that read earlier
tool results (cancel/reschedule gates) see exactly what they would in
the graph.

Real network is refused for the whole module (socket connect and
requests.Session.request both raise).
"""

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
        self._n = 0

    def say(self, text):
        self.state["messages"].append(HumanMessage(content=text))

    def assistant(self, text):
        self.state["messages"].append(AIMessage(content=text))

    def asked_to_confirm(self, action, target=None):
        """What the graph records when its reply asks to confirm `action`."""
        self.state["pending_confirmation"] = {"action": action, "target": target}

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


# ==========================================================
# Wiring
# ==========================================================

def test_tenant_config_points_only_at_fake_hosts(hospital):
    state = hospital.make_state()
    templates = state["templates"]
    assert templates["_base_url"] == fh.FAKE_BOOKINGS_BASE_URL
    assert templates["_doctors_base_url"] == fh.FAKE_DOCTORS_BASE_URL
    assert templates["_timezone"] == "Asia/Riyadh"
    assert templates["_clinic_name_ar"] == "مستشفى تناسق الطبية"
    assert templates["_dialect_name"] == "Saudi"
    assert templates.get("msg_booking_confirmation"), "Saudi dialect row not merged"
    assert {"name": "Al Nuzha", "aliases": ["Al Nuzha", "النزهة"]} in templates["_branch_aliases"]

    import config
    # CSV-less: the tenant resolves by client_id too, with no client_config.csv.
    assert config.get_client_config(fh.TENANT["client_id"]) is fh.TENANT


def test_clock_is_frozen(hospital, tools):
    assert tools.date.today() == fh.FAKE_TODAY
    assert tools._local_now_naive("Asia/Riyadh").isoformat() == "2026-09-28T08:00:00"
    assert tools.datetime.utcnow().isoformat() == "2026-09-28T05:00:00"
    # Real datetimes still count as datetimes for the patched name.
    import datetime as real
    assert isinstance(real.datetime(2026, 1, 1), tools.datetime)


def test_every_faked_api_function_is_patched(hospital):
    import api
    for name in fh.FAKED_API_FUNCTIONS:
        assert getattr(getattr(api, name), "__self__", None) is hospital, name
    # The transport backstop answers with a structured error, not the network.
    assert api._post_json("https://x.test", {})["error"] == "fake_hospital_unfaked_endpoint"
    assert hospital.blocked


# ==========================================================
# New booking - full happy path
# ==========================================================

def test_booking_happy_path(hospital, tools):
    c = Conversation(hospital, "966500000001+booking")
    c.say("ابي احجز موعد جلدية")

    r = c.call(tools.list_specialties)
    assert r["status"] == "found", r
    names = [s["name"] for s in r["specialties"]]
    assert "طب الجلدية" in names and "جراحة العظام" not in names
    assert "جراحة العظام" in r["unstaffed_specialties"]
    derm_id = next(s["id"] for s in r["specialties"] if s["name"] == "طب الجلدية")

    r = c.call(tools.find_available_doctors, specialty_ids=[derm_id])
    assert r["status"] == "found", r
    assert [d["name"] for d in r["doctors"]] == ["د. أحمد سامي", "د. ريم الحربي"]

    c.say("1")
    r = c.call(tools.match_entity_for_booking, entity_type="doctor", option_number=1)
    assert r["matched"] is True and r["item"]["id"] == fh.DR_AHMED_SAMI, r

    # Dr. Ahmed Sami works at BOTH branches -> the tool asks which one.
    r = c.call(tools.list_available_days_for_booking)
    assert r["status"] == "missing_branch", r
    assert {b["name"] for b in r["branches"]} == {"النزهة", "المنار"}

    c.say("النزهة")
    r = c.call(tools.match_entity_for_booking, entity_type="branch", name="النزهة")
    assert r["matched"] is True and r["item"]["id"] == fh.NUZHA, r

    r = c.call(tools.list_available_days_for_booking)
    assert r["status"] == "found", r
    first_day = r["days"][0]
    # Monday 08:00 + 12h lead -> the first Al Nuzha day is Tuesday 29/09.
    assert first_day["date"] == "2026-09-29" and first_day["weekday_name"] == "Tuesday"

    r = c.call(tools.get_available_slots_for_booking,
               from_date=first_day["from_date"], to_date=first_day["to_date"])
    assert r["status"] == "found", r
    # 16:00 is held by TNS-10002 (another patient), so 16:20 comes first.
    assert r["slots"][0]["time_display"] == "4:20 مساءً"
    assert r["slots"][0]["slotStart"] == "2026-09-29T13:20:00"   # wire value: UTC wall clock

    c.say("2")
    r = c.call(tools.select_appointment_slot, option_number=2)
    assert r["status"] == "selected", r
    slot = r["slot"]
    assert slot["time_display"] == "4:40 مساءً"

    c.say("0500000001")
    r = c.call(tools.compare_phone, provided_phone="0500000001", channel_phone=fh.CHANNEL_PHONE)
    assert r == {"status": "match"}

    r = c.call(tools.get_patient_info, mobile_number=fh.CHANNEL_PHONE)
    assert r["status"] == "found", r
    name = r["patientFullName"]
    assert name == "محمد عبدالله العمري"

    # The review is PREPARED by the tool; code renders the card from it.
    r = c.call(tools.confirm_booking_review, patient_full_name=name)
    assert r["status"] == "review_ready", r
    assert r["review"]["slotStart"] == slot["slotStart"]
    assert r["review"]["mobile_number"] == fh.CHANNEL_PHONE
    assert r["review"]["doctor"] == "د. أحمد سامي" and r["review"]["branch"] == "النزهة"

    # The graph shows the card and records the pending confirmation; then yes.
    c.asked_to_confirm("book", slot["slotStart"])
    c.say("ايوه أكد")
    r = c.call(tools.create_new_booking, patient_confirmed=True)
    assert r == {"status": "success", "booking_ref": "TNS-10101"}

    booking = hospital.booking("TNS-10101")
    assert booking["status"] == 1
    assert booking["doctorId"] == fh.DR_AHMED_SAMI and booking["branchId"] == fh.NUZHA
    assert booking["bookingTimeFrom"] == "2026-09-29T13:40:00+00:00"
    assert booking["mobileNumber"] == fh.CHANNEL_PHONE

    create_args = hospital.calls_to("create_booking")[0]
    assert create_args["base_url"] == fh.FAKE_DOCTORS_BASE_URL
    assert create_args["service_id"] == fh.SVC_DERM and create_args["service_price"] == 300

    # The slot is now taken: re-verification no longer offers it.
    import api
    again = api.get_doctor_schedule_slots(
        fh.FAKE_DOCTORS_BASE_URL, doctor_ids=[fh.DR_AHMED_SAMI], branch_ids=[fh.NUZHA],
        from_date="2026-09-29T00:00:00", to_date="2026-09-29T23:59:59", is_booked=False,
    )
    starts = [i["slotStart"] for i in again["data"]["items"]]
    assert "2026-09-29T13:40:00+00:00" not in starts
    assert "2026-09-29T13:20:00+00:00" in starts

    # Booking the same slot again is refused by the fake API.
    dup = api.create_booking(
        fh.FAKE_DOCTORS_BASE_URL, "سعد علي", "+966500000009", fh.NUZHA, fh.DR_AHMED_SAMI,
        fh.SVC_DERM, 300, slot["slotStart"], slot["slotEnd"], "", "", "",
    )
    assert dup["success"] is False and dup["error"] == "api_reported_failure"

    # Every recorded call went to a fake host.
    for _fn, args in hospital.calls:
        if "base_url" in args:
            assert args["base_url"] in (fh.FAKE_BOOKINGS_BASE_URL, fh.FAKE_DOCTORS_BASE_URL), args


def test_doctor_ahmed_is_ambiguous(hospital, tools):
    c = Conversation(hospital, "966500000001+ambiguous")
    c.say("دكتور أحمد")
    r = c.call(tools.match_entity_for_booking, entity_type="doctor", name="دكتور أحمد")
    assert r.get("ambiguous") is True, r
    assert {d["id"] for d in r["candidates"]} == {fh.DR_AHMED_SAMI, fh.DR_AHMED_OTAIBI}


# ==========================================================
# Cancel
# ==========================================================

def test_cancel_path(hospital, tools):
    c = Conversation(hospital, "966500000001+cancel")
    c.say("ابي الغي موعدي")

    r = c.call(tools.lookup_appointment, use_channel_identity=True)
    assert r["status"] == "found_one", r      # the past TNS-09874 is filtered out
    appointment = r["appointment"]
    assert appointment["ref"] == "TNS-10001"
    assert appointment["doctorName"] == "د. أحمد سامي"
    assert appointment["branchName"] == "المنار"
    assert appointment["time_display"] == "10:00 صباحًا"
    assert appointment["date_display"] == "30/09/2026"

    # No confirmation asked yet -> the gate refuses, nothing is cancelled.
    r = c.call(tools.cancel_appointment, booking_id=appointment["id"], patient_confirmed=True)
    assert r["status"] == "needs_confirmation", r
    assert hospital.booking("TNS-10001")["status"] == 1
    assert not hospital.calls_to("cancel_booking_by_guid")

    c.assistant("متأكد تبي تلغي موعدك مع د. أحمد سامي يوم الأربعاء 30/09/2026؟")
    c.asked_to_confirm("cancel", "TNS-10001")
    c.say("ايوه الغيه")
    # The human-readable reference resolves to the GUID too.
    r = c.call(tools.cancel_appointment, booking_id="TNS-10001", patient_confirmed=True)
    assert r == {"status": "success"}
    assert hospital.booking("TNS-10001")["status"] == 6
    assert hospital.calls_to("cancel_booking_by_guid")[0]["booking_guid"] == appointment["id"]

    # Cancelled -> no longer found as active, and its slot is free again.
    r = c.call(tools.lookup_appointment, use_channel_identity=True)
    assert r["status"] == "not_found", r


def test_cancel_refuses_unlooked_up_booking(hospital, tools):
    c = Conversation(hospital, "966500000001+cancel-guard")
    r = c.call(tools.cancel_appointment, booking_id=hospital.booking("TNS-10002")["id"],
               patient_confirmed=True)
    assert r == {"status": "not_looked_up"}


# ==========================================================
# Reschedule
# ==========================================================

def test_reschedule_path(hospital, tools):
    c = Conversation(hospital, "966500000001+reschedule")
    c.say("ابي اغير موعدي")

    r = c.call(tools.lookup_appointment, use_channel_identity=True)
    assert r["status"] == "found_one", r
    appointment = r["appointment"]

    # Tuesday: Dr. Ahmed Sami works only at Al Nuzha - and TNS-10001 is at
    # Al Manar. The Update endpoint cannot change a branch, so no Al Nuzha
    # slot may be offered (production fix, session 201000625084).
    c.say("بكرة")
    r = c.call(tools.get_available_reschedule_slots, ref_number="TNS-10001",
               from_date="2026-09-29T00:00:00+03:00", to_date="2026-09-29T23:59:59+03:00")
    assert r["status"] == "not_found", r

    # Monday 05/10 at Al Manar, 09:00-13:00.
    c.say("الاثنين الجاي")
    r = c.call(tools.get_available_reschedule_slots, ref_number="TNS-10001",
               from_date="2026-10-05T00:00:00+03:00", to_date="2026-10-05T23:59:59+03:00")
    assert r["status"] == "found", r
    assert r["slots"][0]["time_display"] == "9:00 صباحًا"
    assert {s["branchId"] for s in r["slots"]} == {fh.MANAR}

    c.say("1")
    r = c.call(tools.select_reschedule_slot, option_number=1)
    assert r["status"] == "selected", r
    slot = r["slot"]

    c.asked_to_confirm("reschedule", appointment["id"])
    c.say("ايوه")
    r = c.call(tools.reschedule_appointment, booking_id=appointment["id"], patient_confirmed=True,
               new_time_from=slot["slotStart"], new_time_to=slot["slotEnd"])
    assert r["status"] == "success", r
    assert r["new_date_display"] == "05/10/2026" and r["new_time_display"] == "9:00 صباحًا"

    booking = hospital.booking("TNS-10001")
    assert booking["bookingTimeFrom"] == "2026-10-05T06:00:00+00:00"
    assert booking["branchId"] == fh.MANAR
    assert hospital.calls_to("reschedule_booking")[0]["booking_id"] == appointment["id"]

    # The old Wednesday 10:00 slot at Al Manar is free again.
    import api
    wed = api.get_doctor_schedule_slots(
        fh.FAKE_DOCTORS_BASE_URL, doctor_ids=[fh.DR_AHMED_SAMI], branch_ids=[fh.MANAR],
        from_date="2026-09-30T00:00:00", to_date="2026-09-30T23:59:59",
    )
    assert "2026-09-30T07:00:00+00:00" in [i["slotStart"] for i in wed["data"]["items"]]


# ==========================================================
# OTP, complaint, handoff, location, FAQ
# ==========================================================

def test_otp(hospital, tools):
    c = Conversation(hospital, "966500000001+otp")
    r = c.call(tools.send_otp, phone="0555123456")
    assert r == {"status": "otp_sent"}
    assert hospital.otp_sent == ["+966555123456"]
    assert c.call(tools.verify_otp, phone="0555123456", otp="000000") == {"status": "otp_invalid"}
    assert c.call(tools.verify_otp, phone="0555123456", otp="123456") == {"status": "otp_valid"}
    # Verified -> that number's bookings can now be looked up.
    r = c.call(tools.lookup_appointment, phone="0555123456")
    assert r["status"] == "found_one" and r["appointment"]["ref"] == "TNS-10002"


def test_complaint_handoff_location_are_recorded(hospital, tools):
    c = Conversation(hospital, "966500000001+signals")
    c.asked_to_confirm("complaint")
    c.say("ايوه ارسلها")
    r = c.call(tools.send_complaint_email, patient_name="محمد العتيبي", phone=fh.CHANNEL_PHONE,
               branch="النزهة", category="الاستقبال",
               details="الاستقبال في فرع النزهة خلوني انتظر ساعة كاملة",
               patient_confirmed=True)
    assert r == {"status": "sent", "via": "webhook"}
    delivery = hospital.outbound[-1]
    assert delivery["url"] == fh.FAKE_COMPLAINT_WEBHOOK_URL
    assert delivery["json"]["to"] == ["quality@fake-hospital.test"]
    assert delivery["json"]["branch"] == "النزهة"

    c.say("ابي اكلم موظف")
    r = c.call(tools.request_human_handoff, reason="patient asked for staff", patient_agreed=True)
    assert r == {"status": "handoff_requested"}

    c.say("وين موقع فرع النزهة؟")
    r = c.call(tools.share_branch_location, branch_name="النزهة")
    assert r == {"status": "location_requested", "branch_name": "النزهة"}

    assert [s["tool"] for s in hospital.signals] == [
        "send_complaint_email", "request_human_handoff", "share_branch_location"]
    assert hospital.signals[1]["args"]["reason"] == "patient asked for staff"


def test_faq_without_embeddings(hospital, tools):
    c = Conversation(hospital, "966500000001+faq")
    r = c.call(tools.answer_hospital_faq, question="وش ساعات العمل للطوارئ؟")
    assert r["status"] == "found" and r["passages"], r
    assert any("24/7" in p for p in r["passages"])

    r = c.call(tools.list_hospital_services)
    assert r["status"] == "found", r
    assert "العيادة الافتراضية" in r["services"]


# ==========================================================
# reset() / manual mode
# ==========================================================

def test_reset_restores_seed_and_clears_sessions(hospital, tools):
    c = Conversation(hospital, "966500000001+reset")
    appointment = c.call(tools.lookup_appointment, use_channel_identity=True)["appointment"]
    c.asked_to_confirm("cancel", appointment["id"])
    assert c.call(tools.cancel_appointment, booking_id=appointment["id"],
                  patient_confirmed=True) == {"status": "success"}
    c.call(tools.send_otp, phone="0555123456")
    c.call(tools.compare_phone, provided_phone="0500000001", channel_phone=fh.CHANNEL_PHONE)
    tools._otp_storage["+966555123456"] = {"otp": "123456", "created_at": 0}
    assert tools._BOOKING_SESSIONS and tools._otp_storage

    hospital.reset()
    assert hospital.booking("TNS-10001")["status"] == 1
    assert hospital.calls == [] and hospital.otp_sent == []
    assert tools._BOOKING_SESSIONS == {}
    assert tools._otp_storage == {}


def test_manual_install_and_uninstall():
    import api
    import tools
    original = api.get_doctors
    with fh.install(None) as hospital:
        assert api.get_doctors.__self__ is hospital
        assert tools.date.today() == fh.FAKE_TODAY
    assert api.get_doctors is original
    import datetime as real
    assert tools.datetime is real.datetime and tools.date is real.date
