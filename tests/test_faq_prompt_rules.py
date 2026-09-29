"""The FAQ agent must actually receive the insurance and doctor-name rules.

A rule written into the wrong section of prompts.py silently never reaches
the specialist that needs it: the fee and insurance rules first landed in the
booking section, so the FAQ agent - the one that answers "do you accept
Tawuniya?" and "كم كشف الطبيب؟" - never saw them. It also proves the edit did
not break how the prompt is split into sections.
"""

import inspect

import agents.registry as registry
import agents.sections as sections
import prompts


def _sections():
    signature = inspect.signature(prompts.build_system_prompt)
    kwargs = {
        name: ({} if name == "templates" else ("Test" if "name" in name else None))
        for name, param in signature.parameters.items()
        if param.default is inspect._empty
    }
    return sections.split_sections(prompts.build_system_prompt(**kwargs))


def test_every_required_section_is_still_found():
    found = _sections()

    assert [key for key in sections.REQUIRED_KEYS if key not in found] == []


def test_the_faq_agent_gets_the_insurance_rule():
    prompt = registry.build_agent_prompt(_sections(), "faq")

    assert "INSURANCE QUESTIONS - NEVER CONFIRM OR DENY AN INSURER" in prompt


def test_the_faq_agent_gets_the_doctor_name_rule():
    prompt = registry.build_agent_prompt(_sections(), "faq")

    assert "A DESCRIPTION IS NOT A DOCTOR'S NAME" in prompt


def test_the_agents_that_can_be_asked_who_the_best_doctor_is_get_the_rule():
    """Asked "مين احسن دكتور" the assistant listed doctors without saying
    it does not rank them, so the first name read as the recommended one.
    The rule tells it to say all the doctors are competent, then list."""

    found = _sections()

    for agent in ("faq", "concierge", "booking", "medical"):
        prompt = registry.build_agent_prompt(found, agent)
        assert "WHO IS THE BEST DOCTOR - NEVER RANK, ALWAYS REASSURE" in prompt, agent
