"""
One structured summary row per conversation (conversation_store.py):
built from what the turn already knows, no model call, no message text,
off unless DATABASE_URL is set, and never able to break a reply.
"""

from datetime import datetime, timezone
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

import conversation_store as store
import main
from conftest import send


def _summary(new_messages=(), session=None, escalated=False, reading=None):
    return store.build_summary(
        client_id="tanasuq-production", session_id="9665", conversation_id="t:9665",
        phone="9665", result={"understanding": reading or {"intent": "booking"}, "active_agent": "booking"},
        new_messages=list(new_messages), booking_session=session,
        usage={"llm_calls": 3, "total_tokens": 1200}, escalated=escalated,
        at=datetime(2026, 10, 4, 18, 0, tzinfo=timezone.utc),
    )


def test_a_booking_turn_carries_the_outcome_and_the_reference():
    booked = ToolMessage(content='{"status": "success", "booking_ref": "BK-77"}',
                         name="create_new_booking", tool_call_id="1")
    row = _summary([booked], session={"doctor_display_name": "عمر المديفر", "branch_display_name": "النزهة",
                                      "selected_slot": {"slotStart": "2026-10-07T16:00:00"}})
    assert row["outcome"] == "booked" and row["booking_ref"] == "BK-77"
    assert (row["doctor_name"], row["branch_name"], row["appointment_at"]) == (
        "عمر المديفر", "النزهة", "2026-10-07T16:00:00")
    assert row["intents"] == ["booking"] and row["llm_calls"] == 3 and row["tokens"] == 1200


def test_cancel_reschedule_complaint_and_handoff_outcomes():
    def tool(name, status):
        return ToolMessage(content=f'{{"status": "{status}"}}', name=name, tool_call_id=name)
    assert _summary([tool("cancel_appointment", "success")])["outcome"] == "cancelled"
    assert _summary([tool("reschedule_appointment", "success")])["outcome"] == "rescheduled"
    assert _summary([tool("send_complaint_email", "sent")])["outcome"] == "complaint"
    assert _summary(escalated=True)["outcome"] == "handoff"
    assert _summary([tool("cancel_appointment", "error")])["outcome"] is None


def test_no_message_text_is_stored():
    row = _summary([AIMessage(content="الله يشافيها، صعوبة النوم ...")], reading={"intent": "medical"})
    assert "صعوبة" not in str(row)


def test_failed_tools_are_noted():
    failed = ToolMessage(content='{"status": "otp_send_failed"}', name="send_otp", tool_call_id="1")
    assert "send_otp:otp_send_failed" in _summary([failed])["details"]


def test_off_without_a_database_url():
    with patch.object(store, "DATABASE_URL", ""), patch.object(store, "_write") as write:
        store.record_turn(client_id="c", session_id="s", conversation_id="t", phone=None, result={},
                          new_messages=[], booking_session=None, usage=None, escalated=False)
    write.assert_not_called()


class _Cursor:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.log.append((sql, params))


class _Conn:
    closed = False

    def __init__(self):
        self.log = []

    def cursor(self):
        return _Cursor(self.log)

    def close(self):
        self.closed = True


def test_the_write_creates_the_table_once_then_upserts():
    conn = _Conn()
    with patch.object(store, "_conn", None), patch.object(store, "_schema_ready", False), \
         patch.dict("sys.modules", {"psycopg": type("P", (), {"connect": staticmethod(lambda *a, **k: conn)})}):
        store._write(_summary())
        store._write(_summary())
    sqls = [sql for sql, _ in conn.log]
    assert sum("CREATE TABLE" in s for s in sqls) == 1
    assert sum("ON CONFLICT (conversation_id) DO UPDATE" in s for s in sqls) == 2


def test_a_failing_database_never_raises():
    with patch.object(store, "_connection", side_effect=RuntimeError("db down")):
        store._write(_summary())   # logged, not raised


def test_every_turn_is_recorded_with_the_conversation_thread(session_id, llm, reader):
    reader.table["ابي احجز"] = {"intent": "booking"}
    llm._responses.append(AIMessage(content="عندك دكتور أو تخصص معيّن في بالك؟"))
    with patch.object(main.conversation_store, "record_turn") as record:
        send(session_id, "ابي احجز")
    kwargs = record.call_args.kwargs
    assert kwargs["session_id"] == session_id and kwargs["conversation_id"].endswith(session_id)
    assert kwargs["result"]["understanding"]["intent"] == "booking" and "usage" in kwargs
