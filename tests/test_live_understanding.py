"""
The live eval (evals/run_understanding_eval.py) as a pytest test.
Skipped unless OPENROUTER_API_KEY is set - it calls the real model.

    OPENROUTER_API_KEY=sk-or-... python -m pytest tests/test_live_understanding.py -q
"""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")),
                    reason="live eval needs OPENROUTER_API_KEY")
def test_real_model_reads_patients_correctly():
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "evals", "run_understanding_eval.py")],
        capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
    )
    print(result.stdout)
    assert result.returncode == 0, "critical understanding cases failed - see evals/last_report.json"
