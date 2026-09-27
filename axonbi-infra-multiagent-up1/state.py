"""
Graph state for the conversation agent.

`messages` is the natural-language history (LangGraph's `add_messages`
reducer appends each turn; the checkpointer persists it per thread).
Everything else is STRUCTURED state, each fact with one owner:

  - tenant identity and config   -> set by the caller / load_config
  - flow, language, confirmation -> the model's own structured decisions
                                    (the `respond` tool), written by code
  - step                         -> derived by code from the booking facts
  - booking                      -> the business session the tools write
                                    (doctor, branch, slot, verified phone,
                                    appointments, ...), checkpointed here
                                    so a restart or another worker can
                                    restore it
  - turn counters                -> bound the model calls of one turn
"""

from typing import Annotated, NotRequired, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    client_id: str
    session_id: str
    channel_phone: Optional[str]  # verified channel identity (e.g. WhatsApp sender), used by compare_phone
    bsuid: Optional[str]          # sender id on channels that use a username instead of a number
    raw_client_config: NotRequired[Optional[dict]]
    templates: dict
    messages: Annotated[list, add_messages]
    target_language: NotRequired[Optional[str]]
    greeted: bool

    flow: NotRequired[Optional[str]]
    step: NotRequired[Optional[str]]
    # {action, target, turn}: the irreversible action the assistant's last
    # reply asked the patient to confirm. Read by gates.py.
    pending_confirmation: NotRequired[Optional[dict]]
    booking: NotRequired[Optional[dict]]

    turn: NotRequired[int]
    turn_calls: NotRequired[int]
    turn_corrections: NotRequired[int]
