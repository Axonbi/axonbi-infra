"""
Run a turn THROUGH a LangGraph server instead of invoking the compiled
graph in-process.

WHY THIS EXISTS
---------------
LangGraph Studio can only display runs that the LangGraph SERVER
executed. When app.py invokes the graph in-process (the default), the
real WhatsApp/Messenger conversations never reach that server, so Studio
only ever shows whatever was typed into it by hand - useless for
answering "which agent handled this patient, and which tools did it
call?" about actual traffic.

Pointing config.LANGGRAPH_SERVER_URL at a running server makes app.py
forward each turn here instead. n8n keeps calling the same /chat
endpoint with the same payload and gets the same response shape back;
the difference is that the run now happens on the server, so it appears
in Studio (and in that server's own thread history) like any other run.

WHAT STAYS THE SAME
-------------------
The graph itself, the tools, the prompts, and the response contract
{reply, escalate, location, branch_name}. This module only changes WHERE
the graph runs, never what it does.

WHAT DIFFERS FROM IN-PROCESS MODE
---------------------------------
- Persistence: the SERVER owns it (graph.py already detects this and
  skips its own MemorySaver - see _RUNNING_UNDER_LANGGRAPH_API there).
  The session-reset timers in main.py are an in-process concept and do
  NOT apply here; the server keeps a thread until it's deleted.
- Interim "please wait" messages (progress.py) run inside the graph
  process, which is the server - so they still work, provided the
  server process has the same PROGRESS_* environment variables set.
"""

import logging
import threading
from typing import Dict, Optional

import config

logger = logging.getLogger("langgraph_client")

# session_id -> server-side thread_id. The server addresses conversations
# by its own thread ids, so this is what keeps a patient's follow-up
# messages landing in the same thread rather than starting fresh each
# turn.
#
# In-memory on purpose: it's a cache, not a source of truth. If this
# process restarts, the mapping is rebuilt by looking the thread up on
# the SERVER by its session_id metadata (see _thread_id_for) - the
# conversation history itself lives on the server and is never lost
# here.
_thread_ids: Dict[str, str] = {}
_lock = threading.Lock()

_client = None


def is_enabled() -> bool:
    return bool(config.LANGGRAPH_SERVER_URL)


def _get_client():
    """Lazily build the SDK client, so nothing about this module is
    required (or imported) when forwarding is switched off."""

    global _client

    if _client is None:
        from langgraph_sdk import get_sync_client

        kwargs = {"url": config.LANGGRAPH_SERVER_URL}
        if config.LANGGRAPH_API_KEY:
            kwargs["api_key"] = config.LANGGRAPH_API_KEY

        _client = get_sync_client(**kwargs)
        logger.info(
            "LangGraph server forwarding ENABLED: url=%s graph_id=%s",
            config.LANGGRAPH_SERVER_URL, config.LANGGRAPH_GRAPH_ID,
        )

    return _client


def _thread_id_for(session_id: str) -> str:
    """The server-side thread for this session, reused across turns.

    Looked up on the server by metadata before creating a new one, so a
    restart of THIS process continues existing conversations instead of
    silently starting each patient over with an empty history.
    """

    with _lock:
        cached = _thread_ids.get(session_id)
    if cached:
        return cached

    client = _get_client()

    try:
        existing = client.threads.search(metadata={"session_id": session_id}, limit=1)
        if existing:
            thread_id = existing[0]["thread_id"]
            with _lock:
                _thread_ids[session_id] = thread_id
            logger.info("session_id=%s: reusing existing server thread %s", session_id, thread_id)
            return thread_id
    except Exception:
        # A search failure shouldn't block the turn - worst case we
        # create a new thread and the patient loses earlier context.
        logger.warning("session_id=%s: thread search failed, creating a new thread", session_id, exc_info=True)

    thread = client.threads.create(metadata={"session_id": session_id})
    thread_id = thread["thread_id"]
    with _lock:
        _thread_ids[session_id] = thread_id

    logger.info("session_id=%s: created server thread %s", session_id, thread_id)
    return thread_id


def _message_field(message, field: str):
    """Read a field from a message that may be a dict (what the server
    returns over JSON) or a LangChain object (what in-process code
    returns), so the extraction below works with either."""

    if isinstance(message, dict):
        return message.get(field)
    return getattr(message, field, None)


def send_message(
    client_id: str,
    session_id: str,
    message: str,
    channel_phone: Optional[str] = None,
    client_config: Optional[dict] = None,
) -> Dict:
    """Run one turn on the LangGraph server.

    Returns the same shape as main.send_message_with_signals():
    {"reply": str, "escalate": bool, "location": bool, "branch_name": str|None}
    """

    client = _get_client()
    thread_id = _thread_id_for(session_id)

    state = {
        "client_id": client_id,
        "session_id": session_id,
        "channel_phone": channel_phone,
        "raw_client_config": client_config,
        "messages": [{"role": "human", "content": message}],
    }

    before = 0
    try:
        current = client.threads.get_state(thread_id)
        before = len(((current or {}).get("values") or {}).get("messages", []))
    except Exception:
        # A brand-new thread has no state yet - that's not an error.
        logger.debug("session_id=%s: no prior state on thread %s", session_id, thread_id)

    if before == 0:
        # Same reasoning as main.py: seed these only on a genuinely new
        # thread, since re-sending them every turn would reset them.
        state["greeted"] = False
        state["target_language"] = None

    result = client.runs.wait(
        thread_id,
        config.LANGGRAPH_GRAPH_ID,
        input=state,
    )

    messages = (result or {}).get("messages", []) or []
    if not messages:
        logger.error("session_id=%s: LangGraph server returned no messages", session_id)
        return {"reply": "", "escalate": False, "location": False, "branch_name": None}

    reply = _message_field(messages[-1], "content") or ""

    # Same signal extraction as main._turn_signals, over this turn's new
    # messages only - see that function for why each condition is what
    # it is (in particular, why a handoff the patient hasn't agreed to
    # must NOT count).
    escalate = False
    location = False
    branch_name = None

    for msg in messages[before:]:
        name = _message_field(msg, "name")
        content = str(_message_field(msg, "content") or "")
        if name == "request_human_handoff":
            if "handoff_requested" in content:
                escalate = True
        elif name == "share_branch_location":
            import re

            match = re.search(r"'branch_name':\s*'([^']*)'|\"branch_name\":\s*\"([^\"]*)\"", content)
            if match:
                location = True
                branch_name = match.group(1) or match.group(2)

    if escalate or location:
        logger.info(
            "session_id=%s: turn signals={'escalate': %s, 'location': %s, 'branch_name': %r}",
            session_id, escalate, location, branch_name,
        )

    return {"reply": reply, "escalate": escalate, "location": location, "branch_name": branch_name}
