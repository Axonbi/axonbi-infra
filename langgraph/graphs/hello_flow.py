"""A no-LLM demo flow.

Exists so Studio has something to draw without needing any API key:
branching, a retry loop and a fan-in are all visible in the graph view.
"""

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    text: str
    words: int
    attempts: int
    verdict: str
    log: Annotated[list[str], operator.add]


def intake(state: State) -> dict:
    text = state.get("text") or ""
    return {"text": text, "attempts": 0, "log": [f"intake: {len(text)} chars"]}


def count_words(state: State) -> dict:
    n = len(state["text"].split())
    return {"words": n, "log": [f"count_words: {n} words"]}


def route(state: State) -> Literal["expand", "summarise", "reject"]:
    if state["words"] == 0:
        return "reject"
    if state["words"] < 5:
        return "expand"
    return "summarise"


def expand(state: State) -> dict:
    """Pads the text and loops back until it is long enough (a retry cycle)."""
    attempts = state["attempts"] + 1
    return {
        "text": state["text"] + " lorem ipsum",
        "attempts": attempts,
        "log": [f"expand: attempt {attempts}"],
    }


def expand_done(state: State) -> Literal["count_words", "reject"]:
    return "reject" if state["attempts"] >= 3 else "count_words"


def summarise(state: State) -> dict:
    words = state["text"].split()
    head = " ".join(words[:8])
    return {"verdict": f"ok: {head}...", "log": ["summarise: built summary"]}


def reject(state: State) -> dict:
    return {"verdict": "rejected: not enough content", "log": ["reject"]}


builder = StateGraph(State)
builder.add_node("intake", intake)
builder.add_node("count_words", count_words)
builder.add_node("expand", expand)
builder.add_node("summarise", summarise)
builder.add_node("reject", reject)

builder.add_edge(START, "intake")
builder.add_edge("intake", "count_words")
builder.add_conditional_edges("count_words", route)
builder.add_conditional_edges("expand", expand_done)
builder.add_edge("summarise", END)
builder.add_edge("reject", END)

graph = builder.compile()
graph.name = "hello_flow"
