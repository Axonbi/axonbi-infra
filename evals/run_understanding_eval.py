"""
Live eval: does the REAL model read patients correctly - including
phrasings no regex in this project has ever seen?

Uses the exact client the graph uses (graph._understanding_llm, so the
same model, prompt, timeout and JSON mode), against evals/understanding_cases.json.

    python evals/run_understanding_eval.py              # one pass
    python evals/run_understanding_eval.py --repeat 3   # stability: every run must pass
    python evals/run_understanding_eval.py --only crisis

Exit code 0 = every CRITICAL case passed on every repeat, 1 = a critical
case failed (do not deploy), 2 = not runnable (no API key).
A JSON report is written to evals/last_report.json: per-case results,
per-field accuracy, latency, and input/output tokens per call (from the
provider's usage metadata) so prompt changes can be compared on cost as
well as on accuracy.

CASE FORMAT (evals/understanding_cases.json):
  history  [[role, text], ...] oldest first; "@KEY" expands from `offers`
  state    optional compact conversation state, passed exactly as the
           graph passes it (graph._understanding_context)
  expect   fields the reading MUST have:
             plain field           exact match (intent, booleans, ...)
             doctor_name/specialty normalised substring; null = must be null
             "entities.<key>"      normalised substring; null = must be null
             "intent_in": [...]    intent must be one of these
             "alternatives_include": [...]  all must appear in alternatives
             "confidence_below" / "confidence_at_least": number
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


def _substring(field, want, have, problems):
    if want is None:
        if have:
            problems.append(f"{field}: expected null, got {have!r}")
    elif not have or _norm(want) not in _norm(have):
        problems.append(f"{field}: expected ~{want!r}, got {have!r}")


def _check(expect, got):
    problems = []
    for field, want in expect.items():
        if field in ("doctor_name", "specialty"):
            _substring(field, want, got.get(field), problems)
        elif field.startswith("entities."):
            _substring(field, want, (got.get("entities") or {}).get(field.split(".", 1)[1]), problems)
        elif field == "intent_in":
            if got.get("intent") not in want:
                problems.append(f"intent: expected one of {want}, got {got.get('intent')!r}")
        elif field == "alternatives_include":
            missing = [name for name in want if name not in (got.get("alternatives") or [])]
            if missing:
                problems.append(f"alternatives: missing {missing}, got {got.get('alternatives')}")
        elif field == "confidence_below":
            if got.get("confidence") is None or got["confidence"] >= want:
                problems.append(f"confidence: expected < {want}, got {got.get('confidence')}")
        elif field == "confidence_at_least":
            if got.get("confidence") is None or got["confidence"] < want:
                problems.append(f"confidence: expected >= {want}, got {got.get('confidence')}")
        elif got.get(field) != want:
            problems.append(f"{field}: expected {want!r}, got {got.get(field)!r}")
    return problems


def _field_names(expect):
    names = []
    for field in expect:
        names.append("intent" if field == "intent_in" else
                     "alternatives" if field == "alternatives_include" else
                     "confidence" if field.startswith("confidence_") else field)
    return names


def _messages(case, offers):
    out = []
    for role, text in case["history"]:
        text = offers.get(text[1:], text) if text.startswith("@") else text
        out.append(HumanMessage(content=text) if role == "human" else AIMessage(content=text))
    out.append(HumanMessage(content=case["message"]))
    return out


class _UsageRecorder:
    """Wraps the real client and keeps each response's token usage."""

    def __init__(self, llm):
        self.llm = llm
        self.last = {}

    def invoke(self, *args, **kwargs):
        answer = self.llm.invoke(*args, **kwargs)
        self.last = dict(getattr(answer, "usage_metadata", None) or {})
        return answer


def main():
    # Arabic case text on a Windows console (cp1252) used to crash the run.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
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
    llm = _UsageRecorder(graph._understanding_llm)

    print(f"model={config.OPENAI_MODEL_UNDERSTANDING}  cases={len(cases)}  repeat={args.repeat}\n")

    report, critical_failures, total_failures, latencies = [], 0, 0, []
    tokens_in, tokens_out = [], []
    field_totals, field_passes = {}, {}

    for case in cases:
        failures = []
        for attempt in range(args.repeat):
            started = time.time()
            got = understanding.understand_turn(_messages(case, offers), llm, case.get("state"))
            latencies.append(time.time() - started)
            if llm.last:
                tokens_in.append(llm.last.get("input_tokens") or 0)
                tokens_out.append(llm.last.get("output_tokens") or 0)
            problems = ["no reading (call failed / unparseable)"] if got is None else _check(case["expect"], got)

            failed_fields = {p.split(":", 1)[0] for p in problems}
            for name in _field_names(case["expect"]):
                field_totals[name] = field_totals.get(name, 0) + 1
                if got is not None and name not in failed_fields:
                    field_passes[name] = field_passes.get(name, 0) + 1

            if problems:
                failures.append({"attempt": attempt + 1, "problems": problems, "got": got})

        ok = not failures
        mark = "PASS" if ok else ("FAIL*" if case["critical"] else "fail ")
        print(f"{mark}  {case['id']:<44} {case['message'][:40]}")
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
    p95 = latencies[max(0, int(len(latencies) * 0.95 + 0.5) - 1)] if latencies else 0
    avg_in = sum(tokens_in) / len(tokens_in) if tokens_in else None
    avg_out = sum(tokens_out) / len(tokens_out) if tokens_out else None
    field_accuracy = {name: round(field_passes.get(name, 0) / total, 3)
                      for name, total in sorted(field_totals.items())}

    print(f"\n{passed}/{len(cases)} passed  |  critical failures: {critical_failures}  |  "
          f"latency p50={p50:.2f}s p95={p95:.2f}s")
    if avg_in is not None:
        print(f"tokens/call: input avg={avg_in:.0f}  output avg={avg_out:.0f}")
    print("field accuracy: " + ", ".join(f"{k}={v:.0%}" for k, v in field_accuracy.items()))

    json.dump(
        {"model": config.OPENAI_MODEL_UNDERSTANDING, "repeat": args.repeat, "passed": passed,
         "total": len(cases), "critical_failures": critical_failures, "p50_s": p50, "p95_s": p95,
         "avg_input_tokens": avg_in, "avg_output_tokens": avg_out,
         "field_accuracy": field_accuracy, "cases": report},
        open(os.path.join(HERE, "last_report.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=2,
    )
    return 1 if critical_failures else 0


if __name__ == "__main__":
    sys.exit(main())
