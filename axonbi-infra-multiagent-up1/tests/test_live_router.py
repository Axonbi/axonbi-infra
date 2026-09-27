"""
Semantic routing - LIVE tests against the real router model.

These are the tests that measure understanding: every scenario in
router_cases.py, in its original wording AND every paraphrase, is read
by the actual configured router model (config.OPENAI_MODEL_ROUTER) and
must land on an acceptable specialist. Many paraphrases carry none of
the words a keyword list would look for, so passing them is evidence of
reading by meaning, not by vocabulary.

Skipped unless OPENAI_API_KEY is configured. Costs one small structured
call per phrasing (about 50 in total). Run with:

    pytest tests/test_live_router.py -v
"""

import pytest

import config
from router_cases import CASES, build_messages

pytestmark = pytest.mark.skipif(
    not config.OPENAI_API_KEY, reason="OPENAI_API_KEY not configured - live router tests skipped",
)

PHRASINGS = [
    pytest.param(case, text, id=f"{case.id}[{i}]")
    for case in CASES
    for i, text in enumerate([case.message] + case.paraphrases)
]


@pytest.mark.parametrize("case, text", PHRASINGS)
def test_real_model_routes_by_meaning(case, text):
    import agents.router as router

    chosen, reason, reading = router.route_turn(
        build_messages(case, text), case.active, facts=router.compact_facts(case.session),
    )

    assert reading["status"] == "ok", f"router model unavailable: {reason}"
    assert chosen in case.expected, (
        f"{text!r} routed to {chosen} ({reason}); reading={reading}; expected {case.expected}"
    )
    if case.reading["health"] == "crisis":
        assert reading["health"] == "crisis"
