"""
Shared fixtures for the regression suite.

Nothing here touches the network:
  - the specialist LLM is a scripted FakeLLM (graph._llm_with_tools),
  - the turn-understanding LLM is a FakeReader (graph._understanding_llm)
    that answers from a per-message table, or fails on purpose to
    exercise the deterministic fallback,
  - the optional LLM router is switched off (ROUTER_MODE=deterministic),
  - every booking API call is mocked per test.

Run:  python -m pytest tests -q
"""

import itertools
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("OPENROUTER_API_KEY", "")
os.environ.setdefault("LOG_LEVEL", "WARNING")

from langchain_core.messages import AIMessage  # noqa: E402

import config  # noqa: E402
import graph  # noqa: E402
import main  # noqa: E402


# The Tanasuq tenant as n8n sends it (only the fields the graph reads).
TANASUQ = {
    "client_id": "tanasuq-test",
    "Dialect": "Saudi-Tanasuq",
    "clinic_name": "Tanasuq Medical Center",
    "clinic_name_ar": "مستشفى تناسق الطبية",
    "agent_name": "Latifa",
    "agent_name_ar": "لطيفة",
    "timezone": "Asia/Riyadh",
    "base_url": "http://booking.test",
    "doctors_base_url": "http://doctors.test",
    "phone_example": "+966500000000",
}

EXHAUSTED = "__FAKE_LLM_EXHAUSTED__"


class FakeLLM:
    """Pops one scripted AIMessage per .invoke(). Records every call so a
    test can assert how many model calls a turn really cost - a verifier
    or retry firing unexpectedly shows up here as an extra call."""

    def __init__(self, responses=()):
        self._responses = list(responses)
        self.calls = []

    def invoke(self, messages, *args, **kwargs):
        self.calls.append(messages)
        if not self._responses:
            return AIMessage(content=EXHAUSTED)
        return self._responses.pop(0)

    def bind_tools(self, *args, **kwargs):
        return self

    def remaining(self):
        return len(self._responses)

    def system_prompts(self):
        out = []
        for messages in self.calls:
            for m in messages:
                if getattr(m, "type", None) == "system":
                    out.append(str(m.content))
        return out


class FakeReader:
    """Stands in for the turn-understanding LLM.

    `table` maps the patient's message text to the fields the real model
    would return; anything not listed reads as intent "other". Pass
    fail=True to simulate the understanding call being unavailable."""

    DEFAULT = {
        "intent": "other", "wants_human": False, "cancel_request": False,
        "cancel_confirmed": False, "crisis": False, "doctor_name": None,
        "specialty": None, "asks_price": False,
    }

    def __init__(self, table=None, fail=False):
        self.table = table or {}
        self.fail = fail
        self.seen = []

    def invoke(self, messages, *args, **kwargs):
        prompt = str(messages[-1].content)
        message = prompt.rsplit("THE PATIENT'S NEW MESSAGE:", 1)[-1].strip()
        self.seen.append(message)
        if self.fail:
            raise TimeoutError("understanding unavailable (simulated)")
        return AIMessage(content=json.dumps({**self.DEFAULT, **self.table.get(message, {})}, ensure_ascii=False))


def tool_call(name, args, call_id=None):
    call_id = call_id or f"call_{name}_{next(_ids)}"
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


_ids = itertools.count(1)
_sessions = itertools.count(1)


@pytest.fixture
def session_id():
    return f"test-session-{next(_sessions)}"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(config, "ROUTER_MODE", "deterministic")
    monkeypatch.setattr(config, "UNDERSTANDING_ENABLED", True, raising=False)
    original = graph._llm_with_tools
    yield
    graph._llm_with_tools = original


@pytest.fixture
def llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(graph, "_llm_with_tools", fake)
    return fake


@pytest.fixture
def reader(monkeypatch):
    fake = FakeReader()
    monkeypatch.setattr(graph, "_understanding_llm", fake, raising=False)
    return fake


def send(session, text, phone="966500000001"):
    """One patient message through the real public entry point (the same
    call app.py's /chat makes), returning {"reply", "escalate", ...}."""

    return main.send_message_with_signals(
        TANASUQ["client_id"], session, text, channel_phone=phone, client_config=TANASUQ,
    )


def state_of(session):
    return graph.graph.get_state(main._config_for(session)).values
