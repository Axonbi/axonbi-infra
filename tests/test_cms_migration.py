"""Lookup -> reschedule -> cancel against a mocked cms-api (SSO bearer),
plus the +03:00 time handling. No network."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("OPENAI_API_KEY", "")

import api  # noqa: E402
import tools  # noqa: E402

CMS = "https://cms.test:1302"
SSO = {"login_url": "https://sso.test/api/Auth/Login", "email": "svc@x", "password": "pw",
       "organization_id": "org-1"}
BOOKING = {
    "id": "guid-1", "bookingRefNum": "BK-123", "status": 1, "statusName": "New",
    "doctorName": "CBC test", "branchName": "Old Branch", "serviceName": "CBC",
    "bookingTimeFrom": "2099-01-01T09:00:00+03:00", "bookingTimeTo": "2099-01-01T09:15:00+03:00",
    "patientName": "Sara", "patientMobile": "+966500000000", "patientEmail": "s@x",
    "rowVersion": "rv1", "patientId": None, "guestPatientId": "gp-1", "branchId": "b-old",
    "doctorId": "d-1",
}


class Resp:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body
        self.text = str(body)
        self.headers = {"Content-Type": "application/json"}

    def json(self):
        return self._body


class FakeHttp:
    def __init__(self):
        self.calls = []
        self.logins = 0

    def post(self, url, json=None, headers=None, timeout=None):  # SSO login
        self.logins += 1
        self.calls.append(("LOGIN", url, json))
        return Resp(200, {"data": {"access_token": f"tok{self.logins}", "expires_in": 3600}})

    def request(self, method, url, timeout=None, **kw):
        self.calls.append((method.upper(), url, kw.get("json") or kw.get("params"), kw["headers"].get("Authorization")))
        if url.endswith("/api/Bookings/GetList"):
            items = [BOOKING, {**BOOKING, "id": "guid-2", "bookingRefNum": "BK-1234"}]
            return Resp(200, {"isSuccess": True, "data": {"items": items, "totalCount": 2}})
        if url.endswith("/api/Bookings/GetById"):
            return Resp(200, {"isSuccess": True, "data": BOOKING})
        if url.endswith("/api/Bookings/Update") or url.endswith("/api/Bookings/UpdateStatus"):
            return Resp(200, {"isSuccess": True, "data": {}})
        return Resp(404, {})


def _install(monkeypatch):
    http = FakeHttp()
    monkeypatch.setattr(api.requests, "post", http.post)
    monkeypatch.setattr(api.requests, "request", http.request)
    api._SSO_TOKENS.clear()
    return http


def test_lookup_exact_ref_and_bearer(monkeypatch):
    http = _install(monkeypatch)
    res = api.get_bookings_by_ref(CMS, "BK-123", sso=SSO)
    assert [i["id"] for i in res["data"]["items"]] == ["guid-1"]
    method, url, body, auth = http.calls[1]
    assert url == f"{CMS}/api/Bookings/GetList" and body["bookingRefNum"] == "BK-123"
    assert auth == "Bearer tok1" and http.logins == 1
    assert http.calls[0][2]["organizationId"] == "org-1"


def test_phone_lookup_uses_patient_mobile(monkeypatch):
    http = _install(monkeypatch)
    api.get_bookings_by_phone(CMS, "+966500000000", status_list=[1, 2], sso=SSO)
    body = http.calls[1][2]
    assert body["patientMobile"] == "+966500000000" and "mobileNumber" not in body


def test_field_map_reads_cms_names():
    out = tools._format_booking(BOOKING) if hasattr(tools, "_format_booking") else None
    if out is not None:
        assert out["patientFullName"] == "Sara" and out["mobileNumber"] == "+966500000000"


def test_reschedule_uses_getbyid_and_slot_ids(monkeypatch):
    http = _install(monkeypatch)
    slot = {"branchId": "b-new", "doctorId": "d-1", "serviceId": "s-1", "spaceId": "sp-1", "scheduleId": "sch-1"}
    res = api.reschedule_booking(CMS, "guid-1", slot, "2099-01-02T10:00:00+03:00",
                                 "2099-01-02T10:15:00+03:00", sso=SSO)
    assert res["success"]
    put = [c for c in http.calls if c[0] == "PUT"][0]
    assert put[1] == f"{CMS}/api/Bookings/Update"
    body = put[2]
    assert body["rowVersion"] == "rv1" and body["branchId"] == "b-new"
    assert body["doctorScheduleId"] == "sch-1" and body["bookingTimeFrom"].endswith("+03:00")
    assert body["guestPatientId"] == "gp-1" and body["patientId"] is None


def test_cancel_status_6(monkeypatch):
    http = _install(monkeypatch)
    assert api.cancel_booking_by_guid(CMS, "guid-1", sso=SSO)["success"]
    put = http.calls[-1]
    assert put[1] == f"{CMS}/api/Bookings/UpdateStatus" and put[2]["status"] == 6


def test_401_relogin(monkeypatch):
    http = _install(monkeypatch)
    real = http.request
    state = {"n": 0}

    def flaky(method, url, timeout=None, **kw):
        state["n"] += 1
        if state["n"] == 1:
            return Resp(401, {})
        return real(method, url, timeout=timeout, **kw)

    monkeypatch.setattr(api.requests, "request", flaky)
    assert api.get_booking_by_id(CMS, "guid-1", sso=SSO)["success"]
    assert http.logins == 2


def test_no_cms_url_is_not_configured():
    assert api.get_bookings_by_ref(None, "BK-1")["error"] == "not_configured"


def test_time_helpers():
    assert tools.to_api_time("2099-01-02T10:00:00", "Asia/Riyadh") == "2099-01-02T10:00:00+03:00"
    assert tools.to_api_time("2099-01-02T10:00:00+03:00") == "2099-01-02T10:00:00+03:00"
    assert tools.to_wire_utc("2099-01-02T10:00:00+03:00") == "2099-01-02T10:00:00+03:00"
    assert tools.to_clinic_local("2099-01-02T10:00:00+03:00") == "2099-01-02T10:00:00"


def test_get_messages_reads_client_row_keys():
    import config
    row = {"client_id": "lab", "CMS_API_BASE_URL": "https://demo.catalystsystems.io:1302",
           "SSO_EMAIL": "e", "SSO_PASSWORD": "p", "SSO_LOGIN_URL": "https://sso/x", "SSO_ORGANIZATION_ID": "o"}
    merged = config.get_messages("lab", None, row)
    assert merged["_cms_base_url"] == "https://demo.catalystsystems.io:1302"
    assert merged["_sso"]["email"] == "e" and merged["_sso"]["organization_id"] == "o"


def test_tool_flow_lookup_reschedule_cancel(monkeypatch):
    http = _install(monkeypatch)
    state = {
        "session_id": "s-cms-1", "client_id": "lab",
        "messages": [],
        "templates": {"_cms_base_url": CMS, "_sso": SSO, "_doctors_base_url": "http://doctors.test",
                      "_timezone": "Asia/Riyadh"},
    }
    monkeypatch.setattr(tools, "conversation_language", lambda s: "en")

    found = tools.lookup_appointment.func(state=state, ref_number="BK-123", language="en")
    assert found["status"] == "found_one", found
    assert found["appointment"]["patientFullName"] == "Sara"

    import json as _json
    from langchain_core.messages import HumanMessage, ToolMessage
    state["messages"] = [
        HumanMessage(content="reschedule BK-123"),
        ToolMessage(content=_json.dumps(found), name="lookup_appointment", tool_call_id="c1"),
        HumanMessage(content="yes"),
    ]

    slot_item = {"slotStart": "2099-01-02T10:00:00+03:00", "slotEnd": "2099-01-02T10:15:00+03:00",
                 "branchId": "b-new", "branchName": "New Branch", "doctorId": "d-1", "serviceId": "s-1",
                 "spaceId": "sp-1", "scheduleId": "sch-1", "doctorName": "CBC test", "isBooked": False}
    monkeypatch.setattr(api, "get_doctor_schedule_slots",
                        lambda *a, **k: {"success": True, "status_code": 200, "data": {"items": [slot_item]}})
    slots = tools.get_available_reschedule_slots.func(
        state=state, ref_number="BK-123", from_date="2099-01-02T00:00:00+03:00",
        to_date="2099-01-02T23:59:00+03:00")
    assert slots["status"] == "found" and slots["slots"][0]["slotStart"] == "2099-01-02T10:00:00"
    assert slots["slots"][0]["scheduleId"] == "sch-1"

    res = tools.reschedule_appointment.func(
        state=state, booking_id="guid-1", new_time_from=slots["slots"][0]["slotStart"],
        new_time_to=slots["slots"][0]["slotEnd"])
    assert res["status"] == "success", res
    assert res["new_branch_name"] == "New Branch"
    put = [c for c in http.calls if c[0] == "PUT" and c[1].endswith("/Bookings/Update")][0]
    assert put[2]["bookingTimeFrom"] == "2099-01-02T10:00:00+03:00"
    assert put[2]["branchId"] == "b-new"

    http.calls.clear()
    state["active_agent"] = "cancel"
    cancelled = tools.cancel_appointment.func(state=state, booking_id="guid-1", confirmed_by_patient=True)
    assert cancelled["status"] == "success", cancelled
    put = [c for c in http.calls if c[0] == "PUT"][0]
    assert put[1].endswith("/api/Bookings/UpdateStatus") and put[2]["status"] == 6
