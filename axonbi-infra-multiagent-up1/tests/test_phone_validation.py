"""
A code is only ever sent to a complete mobile number.

CONFIRMED (tanasuq, 2026-10-01): the patient typed "055401389" - a Saudi
mobile missing a digit. The generic "7-15 digits" check passed it, it was
normalized to "+96655401389", and an OTP went to that number. send_otp
itself never checked anything.
"""

from unittest.mock import patch

import pytest

import tool_result_guidance
import tools

SAUDI = {"templates": {"_timezone": "Asia/Riyadh"}, "channel_phone": "966554075524",
         "messages": [], "session_id": "phone-sa"}
EGYPT = {"templates": {"_timezone": "Africa/Cairo"}, "channel_phone": "201001234567",
         "messages": [], "session_id": "phone-eg"}


def _validate(phone, state=SAUDI):
    return tools.validate_phone_format.func(state=state, phone=phone)


@pytest.mark.parametrize("phone,normalized", [
    ("0554013890", "+966554013890"),
    ("0538122705", "+966538122705"),
    ("+966 50 599 2148", "+966505992148"),
    ("966505992148", "+966505992148"),
    ("٠٥٠٥٩٩٢١٤٨", "+966505992148"),
    # A foreign number is fine - patients register with them.
    ("+201001234567", "+201001234567"),
    ("+447911123456", "+447911123456"),
    ("+971501234567", "+971501234567"),
])
def test_complete_mobile_numbers_are_valid(phone, normalized):
    assert _validate(phone) == {"status": "valid", "normalized": normalized}


@pytest.mark.parametrize("phone", [
    "055401389",        # the screenshot: one digit short
    "05540138901",      # one digit long
    "+96655401389",
    "0112345678",       # a Riyadh landline - no SMS
    "+20100123456",     # Egyptian mobile, one digit short
    "+97150123456",     # UAE, one digit short
    "12345",
])
def test_incomplete_numbers_and_landlines_are_invalid(phone):
    assert _validate(phone) == {"status": "invalid"}


def test_a_bare_local_number_follows_the_patients_country():
    assert _validate("01001234567", EGYPT) == {"status": "valid", "normalized": "+201001234567"}
    assert _validate("0100123456", EGYPT) == {"status": "invalid"}


def test_send_otp_never_sends_to_an_incomplete_number():
    with patch.object(tools, "OTP_PROVIDER", "authentica"), \
            patch("api.authentica_send_otp") as sms:
        result = tools.send_otp.func(state=SAUDI, phone="055401389")
    assert result == {"status": "invalid_phone"}
    assert not sms.called


def test_send_otp_still_sends_to_a_complete_number():
    with patch.object(tools, "OTP_PROVIDER", "authentica"), \
            patch("api.authentica_send_otp") as sms:
        result = tools.send_otp.func(state=SAUDI, phone="0538122705")
    assert result == {"status": "otp_sent"}
    sms.assert_called_once_with("+966538122705")


def test_compare_phone_reports_an_incomplete_number_instead_of_no_match():
    assert tools.compare_phone.func(state=SAUDI, provided_phone="055401389") == {"status": "invalid_phone"}
    assert tools.compare_phone.func(state=SAUDI, provided_phone="0538122705") == {"status": "no_match"}
    assert tools.compare_phone.func(state=SAUDI, provided_phone="0554075524") == {"status": "match"}


@pytest.mark.parametrize("tool_name,status", [
    ("validate_phone_format", "invalid"), ("compare_phone", "invalid_phone"), ("send_otp", "invalid_phone"),
])
def test_the_model_is_told_what_to_do_with_an_incomplete_number(tool_name, status):
    guidance = tool_result_guidance.guidance_for(tool_name, {"status": status})
    assert guidance and "no code was sent" in guidance
