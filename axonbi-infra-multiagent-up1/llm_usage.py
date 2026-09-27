"""
Token and call accounting - one `llm_call` log line per model call, one
`turn_usage` summary per patient message.

A LangChain callback handler, so it sees EVERY model call made while a
graph run is in progress - including calls made from inside nodes and
tools without passing a config along (LangChain propagates the run
config through context variables) - without any call site having to
remember to report itself.

Never logs content: only counts, the model, the node and the purpose.

Purpose is taken from the call's metadata (`{"purpose": ...}`) when the
call site sets it. When it does not, it is inferred from the call's
shape - which is what lets this meter measure older code that was never
written to report anything.
"""

import json
import logging
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


def _system_text(messages) -> str:
    for message in messages or []:
        if getattr(message, "type", None) == "system":
            content = message.content
            return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return ""


def _first_text(messages) -> str:
    for message in messages or []:
        content = getattr(message, "content", "")
        return content if isinstance(content, str) else str(content)
    return ""


# Markers of the older architecture's auxiliary calls (graph.py / intent.py).
_MARKERS = (
    ("claim_gate", "YOU CLAIMED SOMETHING THAT HAS NOT HAPPENED"),
    ("repeat_loop", "YOU HAVE ALREADY MADE THIS EXACT CALL THIS TURN"),
)
_HUMAN_ONLY_MARKERS = (
    ("router", "patient message sent to a hospital's WhatsApp assistant"),
    ("router", "You decide which specialist owns ONE patient message"),
    ("affirmation_classifier", "asked the patient a plain yes/no question"),
)


class LLMUsageMeter(BaseCallbackHandler):
    """Collects usage per call; `start_turn`/`end_turn` bracket one
    patient message."""

    raise_error = False

    def __init__(self, emit_logs: bool = True):
        self.emit_logs = emit_logs
        self._lock = threading.Lock()
        self._pending: Dict[Any, dict] = {}
        self._main_system: Dict[str, str] = {}
        self.turns: List[dict] = []
        self._turn: Optional[dict] = None

    # ------------------------------------------------------------ turns
    def start_turn(self, label: str = "") -> None:
        with self._lock:
            self._turn = {"label": label, "calls": []}
            self._main_system.clear()

    def end_turn(self) -> dict:
        with self._lock:
            turn, self._turn = self._turn or {"label": "", "calls": []}, None
        turn["summary"] = summarize(turn["calls"])
        self.turns.append(turn)
        if self.emit_logs:
            logger.info("turn_usage %s", json.dumps({"label": turn["label"], **turn["summary"]},
                                                     ensure_ascii=False))
        return turn

    # -------------------------------------------------------- callbacks
    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        batch = messages[0] if messages else []
        metadata = metadata or {}
        node = metadata.get("langgraph_node") or ""
        task = f"{metadata.get('langgraph_checkpoint_ns', '')}|{metadata.get('langgraph_step', '')}"
        purpose = metadata.get("purpose") or self._infer_purpose(node, task, batch)
        system = _system_text(batch)
        with self._lock:
            self._pending[run_id] = {
                "agent": metadata.get("agent") or node or "-",
                "purpose": purpose,
                "system_chars": len(system),
                "system_tokens": _count_tokens(system),
            }

    def _infer_purpose(self, node: str, task: str, batch) -> str:
        system = _system_text(batch)
        if not system:
            text = _first_text(batch)
            if "You judge a WhatsApp chat" in text:
                return "confirmation"
            for purpose, marker in _HUMAN_ONLY_MARKERS:
                if marker in text:
                    return purpose
            return "aux"
        if system.startswith("You judge a WhatsApp chat"):
            return "confirmation"
        for purpose, marker in _MARKERS:
            if marker in system:
                return purpose
        with self._lock:
            main = self._main_system.get(task)
            if main is None:
                self._main_system[task] = system
                return "reason"
        if system.startswith(main) and len(system) > len(main):
            return "verifier_correction"
        return "reason"

    def on_llm_end(self, response, *, run_id, **kwargs):
        with self._lock:
            entry = self._pending.pop(run_id, None)
        if entry is None:
            return
        entry.update(_usage_from(response))
        with self._lock:
            if self._turn is not None:
                self._turn["calls"].append(entry)
        if self.emit_logs:
            logger.info(
                "llm_call agent=%s purpose=%s model=%s input_tokens=%s cached_tokens=%s "
                "output_tokens=%s system_tokens=%s",
                entry["agent"], entry["purpose"], entry.get("model"), entry.get("input_tokens"),
                entry.get("cached_tokens"), entry.get("output_tokens"), entry.get("system_tokens"),
            )

    def on_llm_error(self, error, *, run_id, **kwargs):
        with self._lock:
            entry = self._pending.pop(run_id, None)
            if entry is not None and self._turn is not None:
                self._turn["calls"].append({**entry, "error": type(error).__name__})


def _usage_from(response) -> dict:
    usage, model = {}, None
    try:
        generation = response.generations[0][0]
        message = getattr(generation, "message", None)
        usage = dict(getattr(message, "usage_metadata", None) or {})
        meta = getattr(message, "response_metadata", None) or {}
        model = meta.get("model_name") or meta.get("model")
    except Exception:
        pass
    if not usage:
        token_usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
        usage = {
            "input_tokens": token_usage.get("prompt_tokens"),
            "output_tokens": token_usage.get("completion_tokens"),
            "input_token_details": {
                "cache_read": (token_usage.get("prompt_tokens_details") or {}).get("cached_tokens")},
        }
    details = usage.get("input_token_details") or {}
    return {
        "model": model,
        "input_tokens": usage.get("input_tokens") or 0,
        "cached_tokens": details.get("cache_read") or 0,
        "output_tokens": usage.get("output_tokens") or 0,
    }


_ENCODER = None


def _count_tokens(text: str) -> int:
    global _ENCODER
    if not text:
        return 0
    try:
        if _ENCODER is None:
            import tiktoken
            _ENCODER = tiktoken.get_encoding("o200k_base")
        return len(_ENCODER.encode(text))
    except Exception:
        return len(text) // 4


def summarize(calls: List[dict]) -> dict:
    by_purpose: Dict[str, int] = defaultdict(int)
    for call in calls:
        by_purpose[call.get("purpose", "?")] += 1
    total_in = sum(c.get("input_tokens") or 0 for c in calls)
    total_cached = sum(c.get("cached_tokens") or 0 for c in calls)
    total_out = sum(c.get("output_tokens") or 0 for c in calls)
    return {
        "llm_calls": len(calls),
        "input_tokens": total_in,
        "cached_tokens": total_cached,
        "uncached_input_tokens": total_in - total_cached,
        "output_tokens": total_out,
        "total_tokens": total_in + total_out,
        "max_system_tokens": max((c.get("system_tokens") or 0 for c in calls), default=0),
        "calls_by_purpose": dict(by_purpose),
    }
