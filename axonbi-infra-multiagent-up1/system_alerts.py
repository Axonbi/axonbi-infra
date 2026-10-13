"""
System-failure alerts.

When the CMS does not accept a booking, a cancellation or a reschedule,
two things need to know: the team, with the exact request that failed,
and the analytics dashboard, which marks the conversation as failed
because of the system rather than the assistant. `report()` sends one
event per failed write to SYSTEM_FAILURE_WEBHOOK_URL; n8n stores it and
emails it.

Every failure is reported, tagged by `category`:
  "system"       - the CMS or the network broke: 5xx, timeout, connection
                   error, auth, wrong path, an empty or unreadable body.
  "cms_rejected" - the CMS answered and refused this request (a field it
                   would not accept, isSuccess=false).

Delivery is fire-and-forget on a daemon thread, like progress.py: it
never raises into the tool and never delays the patient's reply.

The alert goes out by email, so the patient's name, number and email in
the request and response are partially masked before sending.
"""

import logging
import re
import threading
from datetime import datetime, timezone
from typing import Optional

import requests

import api
from config import (
    SYSTEM_FAILURE_TIMEOUT_SECONDS,
    SYSTEM_FAILURE_WEBHOOK_SECRET,
    SYSTEM_FAILURE_WEBHOOK_URL,
)

logger = logging.getLogger(__name__)

# The CMS answered and said no. Every other error is the system failing.
_REJECTION_ERRORS = {"validation_error", "api_reported_failure"}

_NAME_KEYS = {"patientfullname", "fullname", "name", "patientname"}
_CONTACT_KEYS = {"mobilenumber", "mobile", "phone", "phonenumber", "email", "nationalid"}

_PHONE = re.compile(r"(?<![\w-])\+?\d{7,15}(?![\w-])")
_EMAIL = re.compile(r"([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+)")

_BODY_LIMIT = 2000


def category(result: dict) -> str:
    return "cms_rejected" if str(result.get("error") or "") in _REJECTION_ERRORS else "system"


def _mask_text(text: str) -> str:
    text = _EMAIL.sub(lambda m: f"{m.group(1)}***@{m.group(2)}", text)
    return _PHONE.sub(lambda m: "•••" + m.group(0)[-3:], text)


def _mask_name(name: str) -> str:
    parts = name.split()
    if not parts:
        return name
    return " ".join([parts[0]] + [p[0] + "." for p in parts[1:]])


def _mask(value, key: str = ""):
    if isinstance(value, dict):
        return {k: _mask(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask(v, key) for v in value]
    if not isinstance(value, str):
        return value
    k = key.lower()
    if k in _NAME_KEYS:
        return _mask_name(value)
    if k in _CONTACT_KEYS and value:
        masked = _mask_text(value)
        return masked if masked != value else "•••" + value[-3:]
    return _mask_text(value)


def _webhook_url(state: dict) -> str:
    override = (state.get("raw_client_config") or {}).get("webhook_system_failure")
    return str(override or SYSTEM_FAILURE_WEBHOOK_URL or "").strip()


def build_event(state: dict, operation: str, result: dict) -> dict:
    ctx = api.failure_context(result) or {}
    response = ctx.get("response") or {}
    body = response.get("body")
    return {
        "type": "system_failure",
        "category": category(result),
        "operation": operation,
        "client_id": state.get("client_id"),
        "session_id": state.get("session_id"),
        "bsuid": state.get("bsuid"),
        "error": result.get("error"),
        "status_code": result.get("status_code"),
        "details": _mask(result.get("details") or []),
        "request": _mask(ctx.get("request")) if ctx.get("request") else None,
        "response": {
            "status_code": response.get("status_code", result.get("status_code")),
            "body": _mask_text(body[:_BODY_LIMIT]) if isinstance(body, str) else None,
        },
        "occurred_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _deliver(url: str, event: dict) -> None:
    headers = {"X-Webhook-Secret": SYSTEM_FAILURE_WEBHOOK_SECRET} if SYSTEM_FAILURE_WEBHOOK_SECRET else {}
    try:
        requests.post(url, json=event, headers=headers, timeout=SYSTEM_FAILURE_TIMEOUT_SECONDS).raise_for_status()
    except Exception as exc:  # an alert must never take anything down with it
        logger.warning("system failure alert not delivered (operation=%s session_id=%s): %s",
                       event.get("operation"), event.get("session_id"), exc)


def report(state: dict, operation: str, result: dict) -> Optional[dict]:
    """Record one failed CMS write. Returns the event (for tests/logs), or
    None if building it failed. Never raises."""

    try:
        event = build_event(state, operation, result)
        logger.warning(
            "system failure: operation=%s category=%s error=%s status_code=%s session_id=%s",
            operation, event["category"], event["error"], event["status_code"], event["session_id"],
        )
        url = _webhook_url(state)
        if url:
            threading.Thread(target=_deliver, args=(url, event), daemon=True,
                             name="system-failure-alert").start()
        return event
    except Exception:
        logger.exception("system failure alert could not be built (operation=%s)", operation)
        return None
