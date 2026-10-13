"""
System-failure alerts (system_alerts.py): every create/cancel/reschedule
the CMS does not accept is reported with the request that failed, tagged
"system" or "cms_rejected", with the patient's details masked - and
reporting can never break or slow the tool.
"""

from unittest.mock import MagicMock, patch

import pytest

import api
import system_alerts
import tools
from test_production_scenarios import _pdf_conversation


class _Response:
    def __init__(self, status_code, text="", body=None):
        self.status_code = status_code
        self.text = text
        self._body = body
        self.headers = {}

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class _RunNow:
    """threading.Thread stand-in that runs the target at start()."""

    def __init__(self, target, args=(), **_):
        self._target, self._args = target, args

    def start(self):
        self._target(*self._args)


@pytest.fixture
def webhook(monkeypatch):
    sent = []
    monkeypatch.setattr(system_alerts, "SYSTEM_FAILURE_WEBHOOK_URL", "https://n8n.example/webhook/system-failure")
    monkeypatch.setattr(system_alerts, "SYSTEM_FAILURE_WEBHOOK_SECRET", "s3cret")
    monkeypatch.setattr(system_alerts.threading, "Thread", _RunNow)

    def post(url, json, headers, timeout):
        sent.append({"url": url, "event": json, "headers": headers})
        return MagicMock(raise_for_status=lambda: None)

    monkeypatch.setattr(system_alerts.requests, "post", post)
    return sent


def _cms_call(response):
    """Run a real cancel_booking_by_guid through _cms_request with the HTTP layer faked."""
    with patch("api._get_sso_token", return_value="token"), \
         patch("api._request_with_retry", return_value=(response, False, None)):
        return api.cancel_booking_by_guid("https://cms.example", "GUID-1", sso={"login_url": "x"})


def test_a_failed_cms_call_keeps_its_request_off_the_result():
    result = _cms_call(_Response(500, "boom"))

    assert result["error"] == "server_error"
    assert "request" not in result, "the request must not reach tool results or the model"
    ctx = api.failure_context(result)
    assert ctx["request"] == {"method": "PUT", "url": "https://cms.example/api/Bookings/UpdateStatus",
                              "body": {"id": "GUID-1", "isConfirmed": True, "status": 6}}
    assert ctx["response"]["body"] == "boom"


def test_an_older_failure_is_never_attached_to_a_new_result():
    _cms_call(_Response(500, "old"))
    assert api.failure_context({"success": False, "error": "timeout"}) is None


def test_a_successful_call_records_nothing():
    ok = _cms_call(_Response(200, body={"isSuccess": True, "data": {}}))
    assert ok["success"] and api.failure_context(ok) is None


@pytest.mark.parametrize("error,expected", [
    ("server_error", "system"), ("timeout", "system"), ("authentication_error", "system"),
    ("endpoint_not_found", "system"), ("empty_response", "system"), ("invalid_json_response", "system"),
    ("not_configured", "system"), ("Connection refused", "system"),
    ("validation_error", "cms_rejected"), ("api_reported_failure", "cms_rejected"),
])
def test_category(error, expected):
    assert system_alerts.category({"error": error}) == expected


def test_a_failed_cancel_is_reported_and_the_patient_still_gets_the_reason(webhook):
    state = _pdf_conversation("نعم الغيه", {"cancel_confirmed": True})
    failed = {"success": False, "status_code": 503, "data": None, "error": "server_error", "details": []}
    with patch("api.cancel_booking_by_guid", return_value=failed):
        result = tools.cancel_appointment.func(state=state, booking_id="GUID-OMAR-1")

    assert result == {"status": "error", "reason": "server_error", "status_code": 503}
    assert len(webhook) == 1
    event = webhook[0]["event"]
    assert event["operation"] == "cancel"
    assert event["category"] == "system"
    assert event["session_id"] == state["session_id"]
    assert webhook[0]["headers"] == {"X-Webhook-Secret": "s3cret"}


def test_patient_details_are_masked_in_the_alert():
    payload = {"patientFullName": "محمد أحمد علي", "mobileNumber": "+966501234567",
               "email": "omar.k@example.com", "doctorId": "3fa85f64-5717-4562-b3fc-2c963f66afa6"}
    result = api._note_failure(
        {"success": False, "status_code": 400, "error": "validation_error", "details": []},
        "post", "https://cms.example/api/GuestBookings/Reservation", payload,
        _Response(400, '{"messages":[{"message":"Mobile 966501234567 not valid"}]}'),
    )
    event = system_alerts.build_event({"client_id": "Egypt-demo", "session_id": "s1"}, "create_booking", result)

    body = event["request"]["body"]
    assert body["patientFullName"] == "محمد أ. ع."
    assert body["mobileNumber"] == "•••567"
    assert body["email"] == "o***@example.com"
    assert body["doctorId"] == payload["doctorId"], "ids must stay readable"
    assert "966501234567" not in event["response"]["body"]
    assert event["category"] == "cms_rejected"


def test_a_dead_webhook_never_breaks_the_tool(monkeypatch):
    monkeypatch.setattr(system_alerts, "SYSTEM_FAILURE_WEBHOOK_URL", "https://n8n.example/hook")
    monkeypatch.setattr(system_alerts.threading, "Thread", _RunNow)
    monkeypatch.setattr(system_alerts.requests, "post", MagicMock(side_effect=OSError("down")))

    event = system_alerts.report({"session_id": "s1"}, "reschedule",
                                 {"success": False, "status_code": None, "error": "timeout"})
    assert event["operation"] == "reschedule"


def test_no_webhook_configured_only_logs(monkeypatch):
    monkeypatch.setattr(system_alerts, "SYSTEM_FAILURE_WEBHOOK_URL", "")
    post = MagicMock()
    monkeypatch.setattr(system_alerts.requests, "post", post)
    system_alerts.report({"session_id": "s1"}, "cancel", {"success": False, "error": "timeout"})
    post.assert_not_called()


def test_client_config_overrides_the_webhook(webhook):
    state = {"session_id": "s1", "raw_client_config": {"webhook_system_failure": "https://n8n.example/rovan"}}
    system_alerts.report(state, "cancel", {"success": False, "error": "timeout"})
    assert webhook[0]["url"] == "https://n8n.example/rovan"
