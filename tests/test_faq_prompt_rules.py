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
