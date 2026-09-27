"""
Measure LLM calls and tokens per patient turn, end to end, against the
fake hospital backend (tests/fake_hospital.py - nothing real is booked,
cancelled or sent).

    python evals/measure_turns.py --code-dir <project dir> --label baseline
    python evals/measure_turns.py --code-dir . --label after

`--code-dir` is the project whose graph is measured (e.g. a checkout of
origin/tanasuq-production for the baseline). The meter, the fake backend
and the scenarios always come from THIS project, so both runs use the
same instruments. Model credentials come from THIS project's .env.

Writes evals/results/<label>.json and prints a per-flow table.
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--flows", default="")
    args = parser.parse_args()

    code_dir = os.path.abspath(args.code_dir)
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
    os.environ.setdefault("LANGSMITH_TRACING", "false")
    os.environ.setdefault("PROGRESS_ENABLED", "false")

    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))          # credentials: this project's .env

    sys.path.insert(0, code_dir)                     # the graph being measured
    meter_mod = _load(os.path.join(ROOT, "llm_usage.py"), "_meter_llm_usage")
    fake = _load(os.path.join(ROOT, "tests", "fake_hospital.py"), "_meter_fake_hospital")
    scenarios = _load(os.path.join(HERE, "scenarios.py"), "_meter_scenarios")

    logging.basicConfig(level=logging.WARNING)
    hospital = fake.install(None)

    import config
    if not config.OPENAI_API_KEY:
        sys.exit("No model credentials in .env - nothing to measure.")

    import graph
    from langchain_core.messages import HumanMessage

    chosen = [f for f in args.flows.split(",") if f] or list(scenarios.SCENARIOS)
    meter = meter_mod.LLMUsageMeter(emit_logs=False)
    report = {"label": args.label, "code_dir": code_dir, "flows": {}}

    for flow in chosen:
        thread = f"{args.label}-{flow}-{uuid.uuid4().hex[:6]}"
        session_id = f"{scenarios.CHANNEL_PHONE}+{thread}"
        hospital.reset()
        turns = []
        for index, text in enumerate(scenarios.SCENARIOS[flow]):
            state = {
                "client_id": fake.TENANT["client_id"],
                "session_id": session_id,
                "channel_phone": scenarios.CHANNEL_PHONE,
                "bsuid": None,
                "raw_client_config": fake.TENANT,
                "messages": [HumanMessage(content=text)],
            }
            if index == 0:
                state.update({"greeted": False, "target_language": None})

            meter.start_turn(f"{flow}[{index}]")
            started = time.time()
            error = None
            try:
                result = graph.graph.invoke(state, config={
                    "configurable": {"thread_id": thread},
                    "callbacks": [meter],
                    "recursion_limit": getattr(config, "GRAPH_RECURSION_LIMIT", 30),
                })
                reply = result["messages"][-1].content
                owner = result.get("active_agent") or result.get("flow")
            except Exception as exc:  # measured, not hidden
                reply, owner, error = "", None, f"{type(exc).__name__}: {exc}"
            turn = meter.end_turn()
            turns.append({
                "user": text, "reply": reply, "owner": owner, "error": error,
                "seconds": round(time.time() - started, 1),
                **turn["summary"], "calls": turn["calls"],
            })
            print(f"  {flow}[{index}] calls={turn['summary']['llm_calls']} "
                  f"in={turn['summary']['input_tokens']} out={turn['summary']['output_tokens']} "
                  f"{turn['summary']['calls_by_purpose']}", flush=True)
        report["flows"][flow] = turns

    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.label}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)

    print(table(report))
    print(f"\nwritten: {out_path}")


def table(report) -> str:
    rows = ["| flow | turns | calls/turn | input/turn | cached/turn | output/turn | total/turn "
            "| router | verifier+gate | confirmation | max system tok |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for flow, turns in report["flows"].items():
        n = len(turns) or 1
        purposes = {}
        for t in turns:
            for k, v in t["calls_by_purpose"].items():
                purposes[k] = purposes.get(k, 0) + v
        rows.append("| {} | {} | {:.1f} | {:,.0f} | {:,.0f} | {:,.0f} | {:,.0f} | {} | {} | {} | {:,} |".format(
            flow, len(turns),
            sum(t["llm_calls"] for t in turns) / n,
            sum(t["input_tokens"] for t in turns) / n,
            sum(t["cached_tokens"] for t in turns) / n,
            sum(t["output_tokens"] for t in turns) / n,
            sum(t["total_tokens"] for t in turns) / n,
            purposes.get("router", 0),
            purposes.get("verifier_correction", 0) + purposes.get("claim_gate", 0) + purposes.get("repeat_loop", 0),
            purposes.get("confirmation", 0),
            max((t["max_system_tokens"] for t in turns), default=0),
        ))
    return "\n".join(rows)


if __name__ == "__main__":
    main()
