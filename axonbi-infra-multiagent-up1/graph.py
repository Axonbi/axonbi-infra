"""
The conversation graph - specialists that hand over to each other.

    START -> load_config -> agent_<active specialist> <-> tools -> END
                                     |  transfer_to_<other>
                                     v
                             agent_<other specialist>  (same message, same turn)

The conversation is owned by one specialist at a time (specialists.py:
coordinator, booking, reschedule, cancel, medical, info). The owner reads
the patient's message in context and answers, calls its own tools, or
hands over by calling another specialist's transfer tool. The hand-over
is the model's decision from meaning; there is no router, no keyword
classifier, no per-turn prose directives and no second "understanding"
call. A normal turn is one model call; a change of specialist costs one
more.

What the model decides is returned as STRUCTURED data through the
`respond` tool (the reply, the flow the patient is in, the language, and
whether the reply asks to confirm an irreversible action). Code turns
those decisions into state; tools enforce the business invariants; code
renders the texts that must never vary (greeting, review card, success
confirmations, option lists).

Safety is deterministic: tool gates (gates.py and the guards inside
tools.py) and one output check on the final reply (safety.py). A failed
check costs at most one correction call, never a loop.
"""

import logging
import re
import sys
from typing import Literal, Optional

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, ValidationError, create_model

import agent_prompt
import config
import flow_context
import replies
import safety
import specialists
import tools
from replies import parse_tool_content, soft_recovery_reply, upstream_api_failed  # noqa: F401 (public API)
from state import AgentState

try:  # the provider's own timeout types, when available
    from openai import APIConnectionError as _APIConnectionError, APITimeoutError as _APITimeoutError
except Exception:  # pragma: no cover
    _APIConnectionError = _APITimeoutError = TimeoutError

logger = logging.getLogger(__name__)

FLOWS = ("booking", "cancel", "reschedule", "medical", "faq", "complaint", "general")
MAX_LLM_CALLS_PER_TURN = 8
MAX_HANDOFFS_PER_TURN = 2
COORDINATOR = specialists.COORDINATOR
HISTORY_MESSAGES = config.MAX_HISTORY_MESSAGES   # earlier-turn text messages kept for context


# ======================================================================
# The model
# ======================================================================

def _make_llm(model: str, **kwargs):
    if config.LLM_PROVIDER == "azure":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=model,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_version=config.AZURE_OPENAI_API_VERSION,
            api_key=config.OPENAI_API_KEY or "sk-not-configured",
            **kwargs,
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, api_key=config.OPENAI_API_KEY or "sk-not-configured", **kwargs)


class Confirmation(BaseModel):
    action: Literal["book", "cancel", "reschedule", "complaint"]
    target: Optional[str] = Field(None, description="The booking reference or slot the patient is asked to confirm.")


class respond(BaseModel):
    """Send your reply to the patient and end your turn. Call it alone, after any other tools."""

    message: str = Field(description="Your WhatsApp reply to the patient.")
    flow: Literal[FLOWS] = Field(description="What the patient is doing now.")
    language: Literal["ar", "en"] = Field(description="The language of your reply.")
    awaiting_confirmation: Optional[Confirmation] = Field(
        None, description="Set when your message asks the patient to confirm an irreversible action.")
    show_options: bool = Field(False, description="Append the OPTIONS SHOWN list, numbered, under your message.")
    salutation: Optional[str] = Field(
        None, description="First reply only: a reply in kind to a time-of-day greeting, e.g. 'صباح النور! 😊'.")


def _transfer_tool(name: str):
    spec = specialists.SPECIALISTS[name]
    return create_model(
        f"transfer_to_{name}",
        __doc__=f"Hand this message over to the part of the assistant that handles: {spec.handles}",
        reason=(Optional[str], Field(None, description="One line: what the patient wants.")),
    )


TRANSFERS = {name: _transfer_tool(name) for name in specialists.NAMES}
_TRANSFER_TARGETS = {f"transfer_to_{name}": name for name in specialists.NAMES}

specialists.check_registry(tools.ALL_TOOLS)

_llm = _make_llm(config.OPENAI_MODEL, timeout=config.OPENAI_TIMEOUT_SECONDS,
                 temperature=config.OPENAI_TEMPERATURE)

# One binding per specialist: its own tools, respond, and a transfer to
# every other specialist. Built once; binding makes no network call.
_BOUND = {
    name: _llm.bind_tools(
        specialists.tools_for(name, tools.ALL_TOOLS) + [respond]
        + [TRANSFERS[other] for other in specialists.NAMES if other != name],
        tool_choice="required",
    )
    for name in specialists.NAMES
}

# Tests set this to a scripted model that then plays every specialist.
_llm_with_tools = None


def _model_for(agent: str):
    return _llm_with_tools if _llm_with_tools is not None else _BOUND[agent]

_LLM_FAILURE_TEXT = {
    "ar": "عذرًا، حصل تأخير مؤقت في الرد. ممكن تبعت رسالتك تاني؟ 🌷",
    "en": "Sorry, there was a temporary delay on our end. Could you please resend your last message?",
}


def _invoke(agent: str, messages, purpose: str):
    """One model call; a timeout is retried once, then reported as None."""

    for attempt in (1, 2):
        try:
            return _model_for(agent).invoke(
                messages, config={"metadata": {"purpose": purpose, "agent": agent}})
        except (_APITimeoutError, _APIConnectionError, TimeoutError) as exc:
            logger.warning("agent[%s]: %s call failed (attempt %d): %s", agent, purpose, attempt, exc)
    return None


# ======================================================================
# Crisis backstop - additive only
# ======================================================================
#
# The model recognises a crisis from meaning. This pattern can only ADD a
# safety line to the context when the message is also phrased in one of
# the common explicit ways; it never routes and never removes anything.
_CRISIS_RE = re.compile(
    r"(?:ه|ح|سا|سأ)?انتحر|(?:ه|ح)نتحر|الانتحار|"
    r"(?:عايز|عاوز|بدي|ابي|ابغى|نفسي)\s*(?:\w+\s+){0,2}(?:اموت|انهي\s*حياتي|اقتل\s*نفسي)|"
    r"(?:مش|ما|مو)\s*(?:عايز|عاوز|بدي|ابي)\s*(?:\w+\s+){0,2}(?:اعيش|اكمل)|"
    r"(?:اذي|أذي|اؤذي|أؤذي|اجرح|أجرح)\s*نفسي|(?:انهي|أنهي|اخلص\s*من)\s*حيات|"
    r"(?:تعبت|زهقت|مليت)\s*من\s*(?:ال)?حياه|"
    r"\bkill\s+my\s?self\b|\bsuicid\w*|\bend\s+my\s+life\b|\bwant\s+to\s+die\b|"
    r"\bhurt\s+my\s?self\b|\bself[\s-]?harm\b|\bdon'?t\s+want\s+to\s+(?:live|be\s+here)\b",
    re.IGNORECASE,
)


def _fold(text: str) -> str:
    text = re.sub(r"[ً-ْٰـ]", "", text or "")
    text = re.sub(r"[أإآٱ]", "ا", text)
    return text.replace("ى", "ي").replace("ة", "ه")


# ======================================================================
# Nodes
# ======================================================================

def load_config(state: AgentState) -> dict:
    """Tenant config for this turn, and the booking session restored from
    the checkpoint if this process does not hold it (a restart, another
    worker)."""

    templates = config.get_messages(state["client_id"], client_row_override=state.get("raw_client_config"))
    session_id = state.get("session_id")
    if session_id and state.get("booking"):
        tools.restore_session(session_id, state["booking"])
    updates = {"templates": templates}
    messages = state.get("messages") or []
    if messages and getattr(messages[-1], "type", None) == "human":
        updates.update(_turn_start_updates(state))
    return updates


def _session(state) -> dict:
    return tools._BOOKING_SESSIONS.get(state.get("session_id")) or {}


def _recent_history(messages: list) -> list:
    """Earlier turns as plain text only (their tool payloads are already
    represented by the state block), plus everything from this turn."""

    index = replies.latest_human_index(messages)
    if index < 0:
        return list(messages[-HISTORY_MESSAGES:])
    earlier = [
        m for m in messages[:index]
        if getattr(m, "type", None) in ("human", "ai")
        and not getattr(m, "tool_calls", None) and str(m.content or "").strip()
    ]
    return earlier[-HISTORY_MESSAGES:] + list(messages[index:])


def _turn_start_updates(state) -> dict:
    """Bookkeeping when a new patient message has just arrived."""

    turn = int(state.get("turn") or 0) + 1
    pending = state.get("pending_confirmation")
    # A confirmation question only counts for the reply that follows it.
    if not (isinstance(pending, dict) and pending.get("turn") == turn - 1):
        pending = None
    agent = state.get("active_agent") if state.get("active_agent") in specialists.NAMES else COORDINATOR
    return {"turn": turn, "pending_confirmation": pending, "turn_calls": 0, "turn_corrections": 0,
            "turn_handoffs": 0, "handoff_to": None, "active_agent": agent}


def _specialist_node(agent: str):
    def node(state: AgentState) -> dict:
        return _run_specialist(state, agent)
    node.__name__ = f"agent_{agent}"
    return node


def _run_specialist(state: AgentState, agent: str) -> dict:
    messages = state.get("messages") or []
    updates: dict = {"handoff_to": None, "active_agent": agent}
    view = {**state, **updates}
    session = _session(view)
    language = view.get("target_language") or _script_language(messages_latest_human_text(messages))
    updates.setdefault("target_language", language)

    # ---- outcomes that are written in code, with no model call -------
    if messages and getattr(messages[-1], "type", None) == "tool":
        rendered = _rendered_outcome(view, session, language)
        if rendered is not None:
            text, pending, release = rendered
            done = _finish(view, text, language, pending, flow=view.get("flow"))
            if release:   # the flow is over; the next message starts at the front desk
                done["active_agent"] = COORDINATOR
            return {**updates, **done}

    calls = int(view.get("turn_calls") or 0)
    if calls >= MAX_LLM_CALLS_PER_TURN:
        logger.error("agent[%s]: %d calls this turn - ending with a recovery line", agent, calls)
        return {**updates, **_finish(view, soft_recovery_reply(language, messages), language, None)}

    prompt = [SystemMessage(content=agent_prompt.build(view.get("templates") or {}, agent))]
    prompt += _recent_history(messages)
    prompt.append(SystemMessage(content=_state_block(view, session)))

    response = _invoke(agent, prompt, "reason")
    updates["turn_calls"] = calls + 1
    if response is None:
        return {**updates, **_finish(view, _LLM_FAILURE_TEXT["en" if language == "en" else "ar"],
                                     language, None)}

    step = _next_step(response, view, agent)
    if step["kind"] == "handoff":
        handoffs = int(view.get("turn_handoffs") or 0)
        if handoffs < MAX_HANDOFFS_PER_TURN:
            logger.info("handoff: %s -> %s (%s)", agent, step["target"], step.get("reason") or "")
            return {**updates, "active_agent": step["target"], "handoff_to": step["target"],
                    "turn_handoffs": handoffs + 1}
        logger.error("agent[%s]: handoff limit reached this turn - answering with a recovery line", agent)
        return {**updates, **_finish(view, soft_recovery_reply(language, messages), language, None)}
    if step["kind"] == "tools":
        return {**updates, "messages": [step["message"]]}

    decision = step["decision"]
    text = _compose(decision, session)
    text, decision, extra_calls = _check_and_correct(agent, text, decision, view, session, prompt)
    updates["turn_calls"] = updates["turn_calls"] + extra_calls.get("calls", 0)
    if extra_calls.get("tools_message") is not None:
        return {**updates, "turn_corrections": 1, "messages": [extra_calls["tools_message"]]}

    pending = None
    if decision.awaiting_confirmation is not None:
        pending = {"action": decision.awaiting_confirmation.action,
                   "target": decision.awaiting_confirmation.target}
    return {**updates, **_finish(view, text, decision.language, pending, flow=decision.flow,
                                 salutation=decision.salutation)}


def _state_block(view, session) -> str:
    block = flow_context.build_context(view, session)
    latest = messages_latest_human_text(view.get("messages") or [])
    if latest and _CRISIS_RE.search(_fold(latest)):
        block += "\nSAFETY: the patient's latest message may express self-harm - follow the crisis rule."
    return block


def _script_language(text: str) -> str:
    """Until the model has declared the conversation's language, the
    script it is written in - a formatting default for tool output."""

    if replies.looks_arabic(text):
        return "ar"
    return "en" if re.search(r"[A-Za-z]{2,}", text or "") else "ar"


def messages_latest_human_text(messages: list) -> str:
    index = replies.latest_human_index(messages)
    return str(messages[index].content or "") if index >= 0 else ""


def _next_step(response, view, agent: str = COORDINATOR) -> dict:
    """A hand-over, tool calls to run, or the final structured decision."""

    tool_calls = list(getattr(response, "tool_calls", None) or [])
    handoff = next((c for c in tool_calls if c.get("name") in _TRANSFER_TARGETS
                    and _TRANSFER_TARGETS[c["name"]] != agent), None)
    if handoff is not None:
        return {"kind": "handoff", "target": _TRANSFER_TARGETS[handoff["name"]],
                "reason": (handoff.get("args") or {}).get("reason")}
    work = [c for c in tool_calls if c.get("name") != "respond" and c.get("name") not in _TRANSFER_TARGETS]
    if work:
        # A reply written before seeing these results is discarded.
        return {"kind": "tools", "message": AIMessage(content="", tool_calls=work)}

    final = next((c for c in reversed(tool_calls) if c.get("name") == "respond"), None)
    args = dict((final or {}).get("args") or {})
    if final is None:
        args = {"message": str(getattr(response, "content", "") or "")}
    args.setdefault("flow", view.get("flow") or "general")
    args.setdefault("language", view.get("target_language") or "ar")
    try:
        decision = respond(**args)
    except ValidationError:
        logger.warning("conversation_agent: respond arguments failed validation: %r", args)
        decision = respond(message=str(args.get("message") or ""),
                           flow=view.get("flow") if view.get("flow") in FLOWS else "general",
                           language="en" if args.get("language") == "en" else "ar")
    return {"kind": "final", "decision": decision}


def _compose(decision: "respond", session: dict) -> str:
    text = (decision.message or "").strip()
    if decision.show_options:
        options = replies.render_options(session)
        if options:
            text = f"{text}\n{options}".strip() if text else options
    return replies.emojify_numbered_lines(text)


def _check_and_correct(agent, text, decision, view, session, prompt):
    """Deterministic output check; at most ONE correction call."""

    extra = {"calls": 0, "tools_message": None}
    violations = safety.check_reply(text, view, session)
    if not violations:
        return text, decision, extra

    logger.error("safety: %s", [(v["check"], v.get("claim")) for v in violations])
    already = int(view.get("turn_corrections") or 0)
    calls = int(view.get("turn_calls") or 0) + 1
    if already or calls >= MAX_LLM_CALLS_PER_TURN:
        return _after_failed_correction(text, decision, view, violations), decision, extra

    notes = "\n".join(f"- {v['note']}" for v in violations)
    correction = SystemMessage(content=(
        f"Your draft reply was:\n{text}\n\nIt cannot be sent as is:\n{notes}\n"
        "Call the tool you need, or send a corrected reply."))
    response = _invoke(agent, prompt + [correction], "correction")
    extra["calls"] = 1
    if response is None:
        return _after_failed_correction(text, decision, view, violations), decision, extra

    step = _next_step(response, view, agent)
    if step["kind"] == "handoff":
        return _after_failed_correction(text, decision, view, violations), decision, extra
    if step["kind"] == "tools":
        extra["tools_message"] = step["message"]
        return text, decision, extra

    new_decision = step["decision"]
    new_text = _compose(new_decision, session)
    remaining = safety.check_reply(new_text, view, session)
    if remaining:
        logger.error("safety: correction still fails %s", [v["check"] for v in remaining])
        return _after_failed_correction(new_text, new_decision, view, remaining), new_decision, extra
    return new_text, new_decision, extra


def _after_failed_correction(text, decision, view, violations) -> str:
    """A claim that an action happened when it did not is never sent.
    Other findings are logged and the reply goes out - the owner's
    standing decision after false positives replaced correct answers."""

    if any(v["check"] == safety.CLAIM_CHECK for v in violations):
        return soft_recovery_reply(decision.language, view.get("messages"), force_handoff=True)
    return text


def _rendered_outcome(view, session, language) -> Optional[tuple]:
    """(text, pending_confirmation, flow_is_over) when this step's tool
    result has a fixed, code-written reply - or None."""

    messages = view.get("messages") or []
    templates = view.get("templates") or {}

    text = replies.success_reply(messages, templates, language, session)
    if text:
        return text, None, True

    last = messages[-1]
    data = parse_tool_content(last) or {}
    if getattr(last, "name", None) == "confirm_booking_review" and data.get("status") == "review_ready":
        review = data.get("review") or {}
        card = replies.render_review_card(templates, review) if language == "ar" else None
        if card:
            return card, {"action": "book", "target": review.get("slotStart")}, False
    return None


def _finish(view, text: str, language: Optional[str], pending: Optional[dict],
            flow: Optional[str] = None, salutation: Optional[str] = None) -> dict:
    """The end of a turn: greeting on the first reply, state written."""

    text = (text or "").strip()
    if not view.get("greeted"):
        greeting = replies.build_greeting(view.get("templates") or {}, language, salutation)
        if greeting and text:
            head = replies.greeting_without_closing_question(greeting) if ("؟" in text or "?" in text) else greeting
            text = f"{head}\n\n{text}"
        elif greeting:
            text = greeting
    if not text:
        text = soft_recovery_reply(language, view.get("messages"))

    if pending is not None:
        pending = {**pending, "turn": int(view.get("turn") or 0)}

    flow = flow if flow in FLOWS else (view.get("flow") or "general")
    session = _session(view)
    return {
        "messages": [AIMessage(content=text)],
        "flow": flow,
        "step": flow_context.derive_step(flow, session),
        "target_language": "en" if language == "en" else "ar",
        "pending_confirmation": pending,
        "greeted": True,
        "booking": tools.export_session(view.get("session_id")) if view.get("session_id") else None,
    }


_base_tool_node = ToolNode(list(tools.ALL_TOOLS))


def tools_node(state: AgentState, config=None) -> dict:
    """Runs the owner's tool calls. A call to a tool the active specialist
    does not own is refused here as well as at binding time."""

    import json
    from langchain_core.messages import ToolMessage

    agent = state.get("active_agent") if state.get("active_agent") in specialists.NAMES else COORDINATOR
    allowed = set(specialists.SPECIALISTS[agent].tools)
    last = (state.get("messages") or [None])[-1]
    calls = list(getattr(last, "tool_calls", None) or [])
    refused = [c for c in calls if c.get("name") not in allowed]
    refusals = [ToolMessage(
        content=json.dumps({"status": "not_available",
                            "hint": "Not your job: transfer to the part of the assistant that handles it."}),
        name=c.get("name"), tool_call_id=c.get("id")) for c in refused]

    updates: dict = {"messages": refusals}
    if len(refused) < len(calls):
        run_state = state
        if refused:
            allowed_calls = [c for c in calls if c.get("name") in allowed]
            run_state = {**state, "messages": list(state["messages"][:-1])
                         + [last.model_copy(update={"tool_calls": allowed_calls})]}
        result = _base_tool_node.invoke(run_state, config)
        ran = result.get("messages", []) if isinstance(result, dict) else list(result)
        updates["messages"] = refusals + list(ran)
    if state.get("session_id"):
        updates["booking"] = tools.export_session(state["session_id"])
    return updates


def _node_of(agent: Optional[str]) -> str:
    return f"agent_{agent if agent in specialists.NAMES else COORDINATOR}"


def route_to_owner(state: AgentState) -> str:
    return _node_of(state.get("active_agent"))


def route_after_agent(state: AgentState) -> str:
    if state.get("handoff_to"):
        return _node_of(state["handoff_to"])
    last = (state.get("messages") or [None])[-1]
    return "tools" if getattr(last, "tool_calls", None) else END


# ======================================================================
# Graph
# ======================================================================

builder = StateGraph(AgentState)
builder.add_node("load_config", load_config)
builder.add_node("tools", tools_node)
_AGENT_NODES = {_node_of(name): _node_of(name) for name in specialists.NAMES}
for _name in specialists.NAMES:
    builder.add_node(_node_of(_name), _specialist_node(_name))
    builder.add_conditional_edges(_node_of(_name), route_after_agent,
                                  {"tools": "tools", END: END, **_AGENT_NODES})
builder.set_entry_point("load_config")
builder.add_conditional_edges("load_config", route_to_owner, _AGENT_NODES)
builder.add_conditional_edges("tools", route_to_owner, _AGENT_NODES)

if "langgraph_api" in sys.modules:   # the LangGraph server brings its own persistence
    graph = builder.compile()
else:
    graph = builder.compile(checkpointer=MemorySaver())
