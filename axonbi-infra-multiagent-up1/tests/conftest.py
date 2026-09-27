import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # the project root (graph.py, agents/, ...)
sys.path.insert(0, HERE)                    # router_cases

# Tests never send traces anywhere.
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")
