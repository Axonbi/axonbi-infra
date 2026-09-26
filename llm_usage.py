"""
Token / call observability - per LLM call and per patient request.

Every model call site records the provider's usage metadata here under a
short ROLE ("understanding", "router", "specialist:booking",
"verifier:booking", "claim_gate:cancel", ...). One `llm_usage` log line
per call, and one `turn_usage` summary per request (main.send_message),
so a prompt change can be compared on tokens/request and calls/request
before and after.

Never logs content - only counts, the model name and the role.
"""

import contextvars
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_TURN = contextvars.ContextVar("llm_usage_turn", default=None)


def start_turn():
    """Begin collecting for one request. Returns the token for end_turn."""
    return _TURN.set([])


def _usage_of(response) -> dict:
    usage = getattr(response, "usage_metadata", None) or {}
    details = usage.get("input_token_details") or {}
    meta = getattr(response, "response_metadata", None) or {}
    return {
        "model": meta.get("model_name") or meta.get("model"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cached_input_tokens": details.get("cache_read"),
    }


def record(role: str, response, model: Optional[str] = None) -> None:
    """Record one model call. Never raises."""
    try:
        entry = {"role": role, **_usage_of(response)}
        if model and not entry.get("model"):
            entry["model"] = model
        tool_calls = getattr(response, "tool_calls", None) or []
        entry["tool_calls"] = len(tool_calls)
        collected = _TURN.get()
        if collected is not None:
            collected.append(entry)
        logger.info("llm_usage %s", json.dumps(entry, default=str))
    except Exception:  # pragma: no cover - observability must never break a turn
        logger.debug("llm_usage: could not record", exc_info=True)


def end_turn(token, session_id: Optional[str] = None) -> dict:
    """Log and return the request's totals, and stop collecting."""
    collected = _TURN.get() or []
    try:
        _TURN.reset(token)
    except (ValueError, LookupError):  # pragma: no cover - reset from another context
        pass

    def _sum(key, rows):
        return sum(row.get(key) or 0 for row in rows)

    by_role: dict = {}
    for row in collected:
        family = str(row.get("role") or "?").split(":", 1)[0]
        bucket = by_role.setdefault(family, [])
        bucket.append(row)

    summary = {
        "session_id": session_id,
        "llm_calls": len(collected),
        "input_tokens": _sum("input_tokens", collected),
        "output_tokens": _sum("output_tokens", collected),
        "cached_input_tokens": _sum("cached_input_tokens", collected),
        "tool_calls": _sum("tool_calls", collected),
        "by_role": {
            family: {"calls": len(rows), "input_tokens": _sum("input_tokens", rows),
                     "output_tokens": _sum("output_tokens", rows)}
            for family, rows in sorted(by_role.items())
        },
    }
    summary["total_tokens"] = summary["input_tokens"] + summary["output_tokens"]
    logger.info("turn_usage %s", json.dumps(summary, default=str))
    return summary
