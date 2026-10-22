"""
OTP by OurSMS (bridge, in place of Authentica): we generate and verify the
code; OurSMS only delivers it. OTP_PROVIDER=oursms, OURSMS_TOKEN,
OURSMS_SENDER in the server's .env.
"""

from unittest.mock import patch

import pytest

import api
import tools

_PHONE = "966512345678"
_STATE = {"session_id": "oursms-test", "channel_phone": "966500000000", "messages": [],
          "templates": {"_timezone": "Asia/Riyadh"}}


class _Response:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body


_ACCEPTED = [{"jobId": "J01", "total": 1, "rejected": 0, "accepted": 1, "cost": 1, "enNotes": ""}]
_REJECTED = [{"jobId": "J02", "total": 1, "rejected": 1, "accepted": 0, "enNotes": "Invalid destination"}]


@pytest.fixture
def oursms(monkeypatch):
    monkeypatch.setattr(tools, "OTP_PROVIDER", "oursms")
    monkeypatch.setattr(api, "OURSMS_TOKEN", "test-token")
    monkeypatch.setattr(api, "OURSMS_SENDER", "BRIDGE")
    tools._otp_storage.clear()
    yield
    tools._otp_storage.clear()


def test_the_code_is_sent_and_then_verified(oursms):
    sent = {}

    def fake(method, url, **kwargs):
        sent.update(kwargs)
        return _Response(200, _ACCEPTED)

    with patch.object(api.requests, "request", side_effect=fake):
        assert tools.send_otp.func(state=_STATE, phone=_PHONE) == {"status": "otp_sent"}

    body = sent["json"]
    assert body["src"] == "BRIDGE" and body["dests"] == [_PHONE] and body["msgClass"] == "transactional"
    assert sent["headers"]["Authorization"] == "Bearer test-token"
    code = next(iter(tools._otp_storage.values()))["otp"]
    assert len(code) == 4 and code in body["body"]

    assert tools.verify_otp.func(state=_STATE, phone=_PHONE, otp="0000" if code != "0000" else "1111")["status"] == "otp_invalid"
    assert tools.verify_otp.func(state=_STATE, phone=_PHONE, otp=code)["status"] != "otp_invalid"


def test_a_rejected_send_leaves_no_code(oursms):
    with patch.object(api.requests, "request", return_value=_Response(200, _REJECTED)):
        result = tools.send_otp.func(state=_STATE, phone=_PHONE)
    assert result["status"] in ("otp_send_failed", "invalid_phone")
    assert not tools._otp_storage


def test_no_token_is_not_configured(oursms, monkeypatch):
    monkeypatch.setattr(api, "OURSMS_TOKEN", "")
    assert api.oursms_send_otp(_PHONE, "1234")["error"] == "not_configured"
