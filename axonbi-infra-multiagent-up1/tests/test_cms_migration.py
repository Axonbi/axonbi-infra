"""
Every call on cms-api (Catalyst CMS API - AI Booking Integration Guide v1.2):

  - a clinic whose config names its cms host books, browses and reads
    slots on cms-api with its token; other clients stay on portal-api
  - the guide's rules: pageSize <= 100 with paging, hasPublishedService,
    GetBookableScheduleSlots (pageSize 0, daily-capacity slots dropped),
    DoctorAssignedServices, active services only, Reservation sent once,
    GuestBookings/Get, GuestBookings/Update for a guest booking
  - a call cms-api refuses for access (missing permission) is served by
    portal-api, so a missing permission cannot stop booking
"""

from unittest.mock import patch

import pytest
import requests

import api
import config
import tools

CMS = "https://cms.test"
PORTAL = "https://portal.test"


class _Resp:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body if body is not None else {"isSuccess": True, "data": {"items": []}}
        self.text = str(self._body)
        self.headers = {}

    def json(self):
        return self._body


class _Wire:
    """Records every outbound call; answers from a queue (default: empty list)."""

    def __init__(self, answers=()):
        self.answers = list(answers)
        self.calls = []

    def _next(self):
        answer = self.answers.pop(0) if self.answers else _Resp()
        if isinstance(answer, Exception):
            raise answer
        return answer

    def request(self, method, url, **kwargs):
        self.calls.append((method.upper(), url, kwargs))
        return self._next()

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self._next()


@pytest.fixture
def wire():
    w = _Wire()
    api._CMS_HOSTS.clear()
    api._CMS_PORTAL_FALLBACK.clear()
    api.register_cms_host(CMS, {"email": "bot@clinic"}, portal_url=PORTAL)
    with patch.object(api.requests, "request", w.request), \
         patch.object(api.requests, "post", w.post), \
         patch.object(api, "_get_sso_token", lambda force_refresh=False, sso=None: "tok"), \
         patch.object(api.time, "sleep", lambda s: None):
        yield w
    api._CMS_HOSTS.clear()
    api._CMS_PORTAL_FALLBACK.clear()


def _page(items, more=False):
    return _Resp(body={"isSuccess": True, "data": {"items": items, "hasNextPage": more}})


# ----------------------------------------------------------------------
# Catalogue
# ----------------------------------------------------------------------

def test_specialties_are_paged_on_cms_with_the_token(wire):
    wire.answers = [_page([{"id": "a"}], more=True), _page([{"id": "b"}])]
    result = api.get_specialties(CMS, page_size=200)
    assert [i["id"] for i in result["data"]["items"]] == ["a", "b"]
    (m1, u1, k1), (m2, u2, k2) = wire.calls
    assert u1 == CMS + "/api/Specialties/GetList" and k1["headers"]["Authorization"] == "Bearer tok"
    assert k1["json"]["pageSize"] == 100 and k1["json"]["pageNumber"] == 1 and k2["json"]["pageNumber"] == 2
    assert config.CLIENT_ID_HEADER not in k1["headers"]


def test_doctors_always_ask_for_a_published_service(wire):
    api.get_doctors(CMS, has_published_service=None, has_service_schedule=None)
    _, url, kwargs = wire.calls[0]
    assert url == CMS + "/api/Doctors/GetList" and kwargs["json"]["hasPublishedService"] is True


def test_bookable_slots_use_the_patient_route_and_drop_full_days(wire):
    wire.answers = [_page([
        {"slotStart": "2026-10-05T16:00:00+03:00", "isAtDailyCapacity": False},
        {"slotStart": "2026-10-05T15:00:00+03:00", "isAtDailyCapacity": False},
        {"slotStart": "2026-10-05T17:00:00+03:00", "isAtDailyCapacity": True},
    ])]
    result = api.get_doctor_schedule_slots(CMS, ["d"], "2026-10-05", "2026-10-06", page_size=1000)
    _, url, kwargs = wire.calls[0]
    assert url == CMS + "/api/Doctors/GetBookableScheduleSlots"
    assert kwargs["json"]["pageSize"] == 0 and kwargs["json"]["isBooked"] is False
    assert [i["slotStart"][11:16] for i in result["data"]["items"]] == ["15:00", "16:00"]


def test_fees_and_services_use_the_cms_paths_and_filters(wire):
    api.get_doctor_fees(CMS, ["d"])
    api.get_services(CMS)
    (_, fees_url, fees), (_, services_url, services) = wire.calls
    assert fees_url == CMS + "/api/DoctorAssignedServices/GetList" and fees["json"]["isPublished"] is True
    assert services_url == CMS + "/api/Services/GetList"
    assert services["json"]["status"] == 1 and services["json"]["isPublished"] is True


def test_schedules_and_branches_move_host_only(wire):
    wire.answers = [_page([{"id": "s"}]), _page([{"id": "b"}])]
    api.get_doctor_schedule(CMS, ["d"])
    api.get_branches(CMS)
    assert [c[1] for c in wire.calls] == [CMS + "/api/DoctorSchedules/GetList", CMS + "/api/Branches/GetList"]


def test_a_portal_base_url_is_untouched(wire):
    api.get_specialties(PORTAL)
    method, url, kwargs = wire.calls[0]
    assert url == PORTAL + "/api/Specialties/GetList" and "Authorization" not in kwargs["headers"]


# ----------------------------------------------------------------------
# Missing permission -> portal-api for that call
# ----------------------------------------------------------------------

def test_a_forbidden_catalogue_call_is_served_by_portal(wire):
    wire.answers = [_Resp(403, {"isSuccess": False}), _page([{"id": "a"}])]
    result = api.get_specialties(CMS)
    assert result["success"] and result["data"]["items"] == [{"id": "a"}]
    assert [c[1] for c in wire.calls] == [CMS + "/api/Specialties/GetList", PORTAL + "/api/Specialties/GetList"]


def test_a_validation_error_is_the_answer_not_a_fallback(wire):
    wire.answers = [_Resp(400, {"isSuccess": False, "messages": [{"prop": "x", "message": "bad"}]})]
    result = api.get_specialties(CMS)
    assert not result["success"] and len(wire.calls) == 1


# ----------------------------------------------------------------------
# Bookings
# ----------------------------------------------------------------------

def _reserve():
    return api.create_booking(CMS, "سارة علي", "+966500000001", "b", "d", "s", 200,
                              "2026-10-05T13:00:00Z", "2026-10-05T14:00:00Z", "sp", "sch", "space")


def test_the_reservation_goes_to_cms(wire):
    wire.answers = [_Resp(body={"isSuccess": True, "data": "new-id"})]
    result = _reserve()
    method, url, kwargs = wire.calls[0]
    assert (method, url) == ("POST", CMS + "/api/GuestBookings/Reservation")
    assert result["data"] == "new-id" and kwargs["json"]["doctorScheduleId"] == "sch"


@pytest.mark.parametrize("failure", [requests.Timeout(), _Resp(503, {})])
def test_the_reservation_is_never_retried(wire, failure):
    wire.answers = [failure, _Resp(body={"isSuccess": True, "data": "x"})]
    result = _reserve()
    assert not result["success"] and len(wire.calls) == 1


def test_a_reservation_without_the_permission_falls_back_but_a_404_does_not(wire):
    wire.answers = [_Resp(403, {}), _Resp(body={"isSuccess": True, "data": "portal-id"})]
    assert _reserve()["data"] == "portal-id"
    assert wire.calls[1][1] == PORTAL + "/api/GuestBookings/Reservation"

    wire.calls.clear()
    wire.answers = [_Resp(404, {})]
    assert not _reserve()["success"] and len(wire.calls) == 1


def test_reading_a_booking_back_uses_guest_bookings_get(wire):
    wire.answers = [_Resp(body={"isSuccess": True, "data": {"bookingRefNum": "BK-1"}})]
    result = api.get_booking_by_id(CMS, "id-1")
    method, url, kwargs = wire.calls[0]
    assert (method, url, kwargs["json"]) == ("POST", CMS + "/api/GuestBookings/Get", {"id": "id-1"})
    assert result["data"]["bookingRefNum"] == "BK-1"


def test_a_guest_booking_is_moved_with_guest_bookings_update(wire):
    wire.answers = [_Resp(body={"isSuccess": True, "data": {"guestPatientId": "g1", "patientId": None}}),
                    _Resp(body={"isSuccess": True, "data": True})]
    result = api.reschedule_booking(CMS, "id-1", {"doctorId": "d"}, "2026-10-06T13:00:00Z", "2026-10-06T14:00:00Z")
    method, url, kwargs = wire.calls[1]
    assert (method, url) == ("PUT", CMS + "/api/GuestBookings/Update")
    assert kwargs["json"] == {"id": "id-1", "fromBookingTime": "2026-10-06T13:00:00Z",
                              "toBookingTime": "2026-10-06T14:00:00Z"}
    assert result["success"]


def test_a_registered_patients_booking_keeps_the_full_update(wire):
    wire.answers = [_Resp(body={"isSuccess": True, "data": {"guestPatientId": None, "patientId": "p1"}}),
                    _Resp(body={"isSuccess": True, "data": {"rowVersion": "rv", "patientId": "p1"}}),
                    _Resp(body={"isSuccess": True, "data": True})]
    api.reschedule_booking(CMS, "id-1", {"doctorId": "d", "scheduleId": "s"}, "a", "b")
    assert [(c[0], c[1]) for c in wire.calls[1:]] == [
        ("GET", CMS + "/api/Bookings/GetById"), ("PUT", CMS + "/api/Bookings/Update")]
    assert wire.calls[2][2]["json"]["rowVersion"] == "rv"


def test_booking_lookup_by_phone_keeps_page_size_within_100(wire):
    api.get_bookings_by_phone(CMS, "+966500000001", sso={"email": "x"})
    assert wire.calls[0][2]["json"]["pageSize"] == 100


# ----------------------------------------------------------------------
# Which clinics move
# ----------------------------------------------------------------------

def test_a_clinic_with_its_cms_host_in_config_books_on_cms():
    state = {"templates": {"_booking_on_cms": True, "_cms_base_url": CMS, "_sso": {"email": "bot"},
                           "_doctors_base_url": PORTAL, "_base_url": PORTAL}}
    try:
        assert tools._doctors_base_url(state) == CMS and tools._base_url(state) == CMS
        assert api._cms_sso_for(CMS) == {"email": "bot"} and api._CMS_PORTAL_FALLBACK[CMS] == PORTAL
    finally:
        api._CMS_HOSTS.clear()
        api._CMS_PORTAL_FALLBACK.clear()


def test_any_other_clinic_stays_on_portal():
    state = {"templates": {"_cms_base_url": CMS, "_doctors_base_url": PORTAL, "_base_url": PORTAL}}
    assert tools._doctors_base_url(state) == PORTAL and tools._base_url(state) == PORTAL


@pytest.mark.parametrize("row,env,expected", [
    ({"CMS_API_BASE_URL": CMS}, None, True),
    ({}, None, False),
    ({"CMS_API_BASE_URL": CMS}, "0", False),
])
def test_the_config_flag(row, env, expected, monkeypatch):
    if env is None:
        monkeypatch.delenv("BOOKING_ON_CMS", raising=False)
    else:
        monkeypatch.setenv("BOOKING_ON_CMS", env)
    merged = config.get_messages("cms-flag-test", client_row_override={"base_url": PORTAL, **row})
    assert merged["_booking_on_cms"] is expected


# ----------------------------------------------------------------------
# Bookable slots: at most 31 days per unpaginated request
# (tanasuq-production 2026-10-04 10:14: "The requested date range cannot
# exceed 31 days when requesting unpaginated results")
# ----------------------------------------------------------------------

def test_a_six_week_range_is_asked_in_windows_and_merged(wire):
    wire.answers = [
        _page([{"slotStart": "2026-10-07T16:00:00+03:00", "doctorId": "d"},
               {"slotStart": "2026-11-03T16:00:00+03:00", "doctorId": "d"}]),
        _page([{"slotStart": "2026-11-03T16:00:00+03:00", "doctorId": "d"},   # boundary repeat
               {"slotStart": "2026-11-11T16:00:00+03:00", "doctorId": "d"}]),
    ]
    result = api.get_doctor_schedule_slots(
        CMS, ["d"], "2026-10-04T10:14:16+03:00", "2026-11-15T10:14:16+03:00")
    windows = [(c[2]["json"]["fromDate"], c[2]["json"]["toDate"]) for c in wire.calls]
    assert len(windows) == 2
    assert windows[0][0] == "2026-10-04T10:14:16+03:00" and windows[-1][1] == "2026-11-15T10:14:16+03:00"
    assert windows[0][1] == windows[1][0]
    from datetime import datetime
    for start, end in windows:
        assert (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days <= 31
    assert [i["slotStart"][:10] for i in result["data"]["items"]] == ["2026-10-07", "2026-11-03", "2026-11-11"]


def test_a_short_range_is_one_request_as_given(wire):
    api.get_doctor_schedule_slots(CMS, ["d"], "2026-10-11T00:00:00", "2026-10-11T23:59:59")
    assert len(wire.calls) == 1
    assert wire.calls[0][2]["json"]["fromDate"] == "2026-10-11T00:00:00"


def test_a_failing_window_returns_that_failure(wire):
    wire.answers = [_page([{"slotStart": "2026-10-07T16:00:00+03:00"}]),
                    _Resp(400, {"isSuccess": False, "messages": [{"prop": "ToDate", "message": "x"}]})]
    result = api.get_doctor_schedule_slots(CMS, ["d"], "2026-10-04T00:00:00", "2026-11-15T00:00:00")
    assert not result["success"]


# ----------------------------------------------------------------------
# An empty specialties/branches list from cms-api is the account, not the
# clinic (tanasuq-production 2026-10-04 10:13: "0 specialties returned")
# ----------------------------------------------------------------------

def test_an_empty_specialty_list_is_served_by_portal(wire):
    wire.answers = [_page([]), _page([{"id": "a"}])]
    result = api.get_specialties(CMS)
    assert result["success"] and result["data"]["items"] == [{"id": "a"}]
    assert wire.calls[1][1] == PORTAL + "/api/Specialties/GetList"


def test_an_empty_branch_list_is_served_by_portal_but_a_search_may_be_empty(wire):
    wire.answers = [_page([]), _page([{"id": "b"}])]
    assert api.get_branches(CMS)["data"]["items"] == [{"id": "b"}]
    wire.calls.clear()
    wire.answers = [_page([])]
    result = api.get_branches(CMS, search_query="xyz")
    assert result["success"] and result["data"]["items"] == [] and len(wire.calls) == 1


def test_an_empty_doctor_list_is_an_answer(wire):
    wire.answers = [_page([])]
    result = api.get_doctors(CMS, specialty_ids=["s"])
    assert result["success"] and len(wire.calls) == 1


def test_with_no_portal_an_empty_list_stays_a_success(wire):
    api._CMS_PORTAL_FALLBACK.clear()
    wire.answers = [_page([])]
    result = api.get_specialties(CMS)
    assert result["success"] and result["data"]["items"] == []


# ----------------------------------------------------------------------
# The same rota published for two periods is one row
# (tanasuq-production 2026-10-04 10:13: فرع النزهة listed twice)
# ----------------------------------------------------------------------

def test_two_periods_of_one_rota_become_one_row_and_branches_stay_together():
    import tools as _tools
    rows = [
        {"branchId": "nozha", "branchName": "النزهة", "recurringDaysNames": ["Wednesday"],
         "fromDateTime": "2026-12-01T16:00:00", "toDateTime": "2027-03-31T18:00:00", "serviceName": "جلسة"},
        {"branchId": "manar", "branchName": "المنار", "recurringDaysNames": ["Monday"],
         "fromDateTime": "2026-10-01T16:00:00", "toDateTime": "2027-01-30T18:00:00", "serviceName": "جلسة"},
        {"branchId": "nozha", "branchName": "النزهة", "recurringDaysNames": ["Wednesday"],
         "fromDateTime": "2026-09-01T16:00:00", "toDateTime": "2026-11-30T18:00:00", "serviceName": "جلسة"},
        {"branchId": "nozha", "branchName": "النزهة", "recurringDaysNames": ["Saturday"],
         "fromDateTime": "2026-09-01T10:00:00", "toDateTime": "2026-11-30T12:00:00", "serviceName": "جلسة"},
    ]
    merged = _tools._merge_schedule_periods(rows)
    assert [(r["branchId"], r["recurringDaysNames"][0]) for r in merged] == [
        ("nozha", "Wednesday"), ("nozha", "Saturday"), ("manar", "Monday")]
    assert merged[0]["fromDateTime"] == "2026-09-01T16:00:00" and merged[0]["toDateTime"] == "2027-03-31T18:00:00"


def test_rows_with_different_hours_are_not_merged():
    import tools as _tools
    rows = [
        {"branchId": "b", "recurringDaysNames": ["Monday"], "fromDateTime": "2026-10-01T16:00:00",
         "toDateTime": "2026-11-30T18:00:00"},
        {"branchId": "b", "recurringDaysNames": ["Monday"], "fromDateTime": "2026-12-01T17:00:00",
         "toDateTime": "2027-01-30T20:00:00"},
    ]
    assert len(_tools._merge_schedule_periods(rows)) == 2
