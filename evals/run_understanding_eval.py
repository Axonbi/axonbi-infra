"""
Live eval: does the REAL model read patients correctly - including
phrasings no regex in this project has ever seen?

Uses the exact client the graph uses (graph._understanding_llm, so the
same model, prompt, timeout and JSON mode), against evals/understanding_cases.json.

    python evals/run_understanding_eval.py              # one pass
    python evals/run_understanding_eval.py --repeat 3   # stability: every run must pass
    python evals/run_understanding_eval.py --only crisis

Exit code 0 = every CRITICAL case passed on every repeat, 1 = a critical
case failed (do not deploy), 2 = not runnable (no OPENAI_API_KEY).
A JSON report is written to evals/last_report.json.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("LOG_LEVEL", "ERROR")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

import config  # noqa: E402
import graph  # noqa: E402
import understanding  # noqa: E402


def _norm(text):
    return graph._norm_ar(str(text or "")).lower()


def _check(expect, got):
    problems = []
    for field, want in expect.items():
        have = got.get(field)
        if field in ("doctor_name", "specialty"):
            if want is None:
                if have:
                    problems.append(f"{field}: expected null, got {have!r}")
            elif not have or _norm(want) not in _norm(have):
                problems.append(f"{field}: expected ~{want!r}, got {have!r}")
        elif have != want:
            problems.append(f"{field}: expected {want!r}, got {have!r}")
    return problems


def _messages(case, offers):
    out = []
    for role, text in case["history"]:
        text = offers.get(text[1:], text) if text.startswith("@") else text
        out.append(HumanMessage(content=text) if role == "human" else AIMessage(content=text))
    out.append(HumanMessage(content=case["message"]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--only", default="", help="run cases whose id contains this")
    args = parser.parse_args()

    if not config.llm_api_key():
        print(f"No API key for LLM_PROVIDER={config.LLM_PROVIDER} (OPENROUTER_API_KEY / OPENAI_API_KEY) - "
              "this eval calls the real model. Not runnable.")
        return 2

    data = json.load(open(os.path.join(HERE, "understanding_cases.json"), encoding="utf-8"))
    cases = [c for c in data["cases"] if args.only in c["id"]]
    offers = data["offers"]

    print(f"model={config.OPENAI_MODEL_UNDERSTANDING}  cases={len(cases)}  repeat={args.repeat}\n")

    report, critical_failures, total_failures, latencies = [], 0, 0, []

    for case in cases:
        failures = []
        for attempt in range(args.repeat):
            started = time.time()
            got = understanding.understand_turn(_messages(case, offers), graph._understanding_llm)
            latencies.append(time.time() - started)
            problems = ["no reading (call failed / unparseable)"] if got is None else _check(case["expect"], got)
            if problems:
                failures.append({"attempt": attempt + 1, "problems": problems, "got": got})

        ok = not failures
        mark = "PASS" if ok else ("FAIL*" if case["critical"] else "fail ")
        print(f"{mark}  {case['id']:<38} {case['message'][:45]}")
        for f in failures[:1]:
            for p in f["problems"]:
                print(f"         - {p}")

        if not ok:
            total_failures += 1
            critical_failures += int(case["critical"])
        report.append({"id": case["id"], "critical": case["critical"], "passed": ok, "failures": failures})

    passed = len(cases) - total_failures
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95) - 1] if latencies else 0
    print(f"\n{passed}/{len(cases)} passed  |  critical failures: {critical_failures}  |  latency p50={p50:.2f}s p95={p95:.2f}s")

    json.dump(
        {"model": config.OPENAI_MODEL_UNDERSTANDING, "repeat": args.repeat, "passed": passed,
         "total": len(cases), "critical_failures": critical_failures, "p50_s": p50, "p95_s": p95,
         "cases": report},
        open(os.path.join(HERE, "last_report.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=2,
    )
    return 1 if critical_failures else 0


if __name__ == "__main__":
    sys.exit(main())
