"""A tool-calling agent on Claude, so Studio shows a real agent loop.

Needs ANTHROPIC_API_KEY in .env. The graph still renders without a key --
only running it requires one.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    table = {
        "cairo": "37 C, clear",
        "riyadh": "41 C, dusty",
        "london": "18 C, rain",
    }
    return table.get(city.strip().lower(), f"no data for {city}")


@tool
def convert_currency(amount: float, frm: str, to: str) -> str:
    """Convert an amount between EGP, SAR and USD."""
    per_usd = {"usd": 1.0, "egp": 48.5, "sar": 3.75}
    f, t = frm.strip().lower(), to.strip().lower()
    if f not in per_usd or t not in per_usd:
        return "unsupported currency"
    return f"{amount * per_usd[t] / per_usd[f]:.2f} {t.upper()}"


llm = ChatAnthropic(
    model="claude-opus-5",
    max_tokens=16000,
    thinking={"type": "adaptive"},
)

graph = create_react_agent(
    llm,
    tools=[get_weather, convert_currency],
    prompt="You are a concise assistant. Use the tools when they apply.",
)
graph.name = "support_agent"
