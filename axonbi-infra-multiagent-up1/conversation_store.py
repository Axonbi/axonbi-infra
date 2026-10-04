"""
One structured summary row per conversation, in PostgreSQL.

After every turn main.py hands this module what the turn already knows -
the routing reading, the booking session, the tool results, the token
usage - and the row for that conversation is created or updated
(INSERT ... ON CONFLICT DO UPDATE). No model call, and no message text:
the conversations are mental-health care, so the row holds what happened
(the outcome, the doctor, the branch, the booking reference), never what
the patient wrote.

OFF unless DATABASE_URL is set, and off (with one warning) if psycopg is
not installed - a client without it behaves exactly as before. Writes run
on one background thread, in order, so a slow or unreachable database
never delays a reply; a failed write is logged and the turn goes on.
"""

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id  text PRIMARY KEY,
    client_id        text NOT NULL,
    session_id       text NOT NULL,
    phone            text,
    started_at       timestamptz NOT NULL,
    last_message_at  timestamptz NOT NULL,
    turns            integer NOT NULL DEFAULT 0,
    intents          text[] NOT NULL DEFAULT '{}',
    last_agent       text,
    outcome          text,
    doctor_name      text,
    branch_name      text,
    appointment_at   text,
    booking_ref      text,
    escalated        boolean NOT NULL DEFAULT false,
    llm_calls        integer NOT NULL DEFAULT 0,
    tokens           integer NOT NULL DEFAULT 0,
    details          jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS conversations_client_last ON conversations (client_id, last_message_at DESC);
CREATE INDEX IF NOT EXISTS conversations_phone ON conversations (phone);
CREATE INDEX IF NOT EXISTS conversations_outcome ON conversations (client_id, outcome);
"""

UPSERT_SQL = """
INSERT INTO conversations (
    conversation_id, client_id, session_id, phone, started_at, last_message_at,
    turns, intents, last_agent, outcome, doctor_name, branch_name, appointment_at,
    booking_ref, escalated, llm_calls, tokens, details
) VALUES (
    %(conversation_id)s, %(client_id)s, %(session_id)s, %(phone)s, %(at)s, %(at)s,
    1, %(intents)s, %(last_agent)s, %(outcome)s, %(doctor_name)s, %(branch_name)s,
    %(appointment_at)s, %(booking_ref)s, %(escalated)s, %(llm_calls)s, %(tokens)s,
    %(details)s::jsonb
)
ON CONFLICT (conversation_id) DO UPDATE SET
    last_message_at = EXCLUDED.last_message_at,
    turns           = conversations.turns + 1,
    intents         = ARRAY(SELECT DISTINCT unnest(conversations.intents || EXCLUDED.intents)),
    last_agent      = COALESCE(EXCLUDED.last_agent, conversations.last_agent),
    outcome         = COALESCE(EXCLUDED.outcome, conversations.outcome),
    doctor_name     = COALESCE(EXCLUDED.doctor_name, conversations.doctor_name),
    branch_name     = COALESCE(EXCLUDED.branch_name, conversations.branch_name),
    appointment_at  = COALESCE(EXCLUDED.appointment_at, conversations.appointment_at),
    booking_ref     = COALESCE(EXCLUDED.booking_ref, conversations.booking_ref),
    escalated       = conversations.escalated OR EXCLUDED.escalated,
    llm_calls       = conversations.llm_calls + EXCLUDED.llm_calls,
    tokens          = conversations.tokens + EXCLUDED.tokens,
    details         = conversations.details || EXCLUDED.details
"""

# A terminal tool that succeeded this turn says how the conversation went.
_OUTCOME_TOOLS = (
    ("create_new_booking", ("success",), "booked"),
    ("reschedule_appointment", ("success",), "rescheduled"),
    ("cancel_appointment", ("success",), "cancelled"),
    ("send_complaint_email", ("sent",), "complaint"),
)

_executor: Optional[ThreadPoolExecutor] = None
_lock = threading.Lock()
_conn = None
_schema_ready = False
_disabled_reason: Optional[str] = None


def enabled() -> bool:
    return bool(DATABASE_URL) and _disabled_reason is None


def _tool_payload(message) -> dict:
    content = getattr(message, "content", "")
    if isinstance(content, dict):
        return content
    try:
        parsed = json.loads(content) if isinstance(content, str) else {}
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def build_summary(*, client_id: str, session_id: str, conversation_id: str, phone: Optional[str],
                  result: dict, new_messages: list, booking_session: Optional[dict],
                  usage: Optional[dict], escalated: bool, at: Optional[datetime] = None) -> dict:
    """The row's values for this turn. Pure - no I/O - so it is tested alone."""

    reading = (result or {}).get("understanding") or {}
    session = booking_session or {}

    outcome, booking_ref, failed = None, None, []
    for message in new_messages or []:
        name = getattr(message, "name", None)
        if not name or getattr(message, "type", None) != "tool":
            continue
        payload = _tool_payload(message)
        status = str(payload.get("status") or "")
        for tool, statuses, label in _OUTCOME_TOOLS:
            if name == tool and status in statuses:
                outcome = label
                if tool == "create_new_booking":
                    booking_ref = payload.get("booking_ref") or booking_ref
        if status in ("error", "api_error", "otp_send_failed", "send_otp_failed"):
            failed.append(f"{name}:{status}")
    if outcome is None and escalated:
        outcome = "handoff"

    slot = session.get("selected_slot") or {}
    details = {"last_intent": reading.get("intent")}
    if failed:
        details["failed_tools"] = failed

    usage = usage or {}
    intent = reading.get("intent")
    return {
        "conversation_id": conversation_id,
        "client_id": client_id,
        "session_id": session_id,
        "phone": phone,
        "at": at or datetime.now(timezone.utc),
        "intents": [intent] if intent else [],
        "last_agent": (result or {}).get("active_agent"),
        "outcome": outcome,
        "doctor_name": session.get("doctor_display_name"),
        "branch_name": session.get("branch_display_name"),
        "appointment_at": slot.get("slotStart") if isinstance(slot, dict) else None,
        "booking_ref": booking_ref,
        "escalated": bool(escalated),
        "llm_calls": int(usage.get("llm_calls") or 0),
        "tokens": int(usage.get("total_tokens") or 0),
        "details": json.dumps(details, ensure_ascii=False, default=str),
    }


def _connection():
    global _conn, _schema_ready, _disabled_reason
    if _conn is not None and not getattr(_conn, "closed", False):
        return _conn
    try:
        import psycopg  # noqa: WPS433 - optional dependency, imported only when enabled
    except ImportError:
        _disabled_reason = "psycopg is not installed (pip install 'psycopg[binary]')"
        logger.warning("conversation_store: DATABASE_URL is set but %s - summaries are OFF", _disabled_reason)
        return None
    _conn = psycopg.connect(DATABASE_URL, autocommit=True, connect_timeout=5)
    if not _schema_ready:
        with _conn.cursor() as cursor:
            cursor.execute(SCHEMA_SQL)
        _schema_ready = True
    return _conn


def _write(row: dict) -> None:
    global _conn
    try:
        conn = _connection()
        if conn is None:
            return
        with conn.cursor() as cursor:
            cursor.execute(UPSERT_SQL, row)
    except Exception:  # noqa: BLE001 - a summary must never break a conversation
        logger.exception("conversation_store: could not save the summary for %s", row.get("conversation_id"))
        try:
            if _conn is not None:
                _conn.close()
        except Exception:  # noqa: BLE001
            pass
        _conn = None


def record_turn(**kwargs) -> None:
    """Build this turn's row and queue its write. Never raises."""

    global _executor
    if not enabled():
        return
    try:
        row = build_summary(**kwargs)
    except Exception:  # noqa: BLE001
        logger.exception("conversation_store: could not build the summary")
        return
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="conversation-store")
    _executor.submit(_write, row)
