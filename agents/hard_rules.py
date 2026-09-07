"""
Routes the GLOBAL HARD RULES block to the specialists each rule is for.

WHY THIS EXISTS
---------------
`GLOBAL HARD RULES` is the single most expensive thing in the prompt
that EVERY specialist pays for: ~7.7k tokens, sixty-four rules, sent in
full on every LLM call of every turn - including the verifier layer's
correction calls, which re-send the whole system prompt.

Most of those rules are not global at all. "NEVER reschedule without
calling `reschedule_appointment`" is meaningless to the complaint
agent, which is not bound to that tool and cannot reschedule anything.
"NEVER show the booking review card while any of its fields is still
unknown" is meaningless to `cancel`. Sending them anyway costs tokens
on every call and, worse, dilutes the rules that DO apply - the
distractor effect this project has already been bitten by (see the
comment on `channel_identity_directive`'s placement in graph.py, where
an unrelated but more emphatic instruction won out over the correct
one).

HOW IT KEEPS THE RULES SAFE
---------------------------
Three properties, in order of importance:

1. NOTHING IS EVER DELETED. prompts.py is untouched and stays
   authoritative. This module only decides which of its rules a given
   specialist is SHOWN, exactly as `agents/sections.py` already does
   for whole flow sections.

2. THE DEFAULT IS THE CURRENT BEHAVIOUR. A rule whose opening text
   matches no entry in `_RULE_SCOPE` below is treated as global and
   goes to every specialist. So an edit to prompts.py that rewords a
   rule does not silently drop it - it just stops being narrowed. New
   rules are likewise global until somebody classifies them.

3. IT FAILS SAFE, LOUDLY. If the block cannot be split into a
   plausible number of rules (someone changed the bullet format), the
   whole block is returned unchanged and a warning is logged.

`concierge` is excluded on purpose: it is the router's fallback and
deliberately carries the entire legacy prompt, so an unclassifiable
message still behaves exactly as it did before any of this existed.
See agents/registry.py's design notes.
"""

import logging
import re
from typing import Dict, FrozenSet, Tuple

logger = logging.getLogger(__name__)


ALL = frozenset({"cancel", "reschedule", "booking", "medical", "faq", "complaint"})

# Convenience groups, named for what they mean rather than who is in
# them, so a change of membership is a one-line edit here.
_ANY_BOOKING = frozenset({"booking", "cancel", "reschedule"})
# `medical` is in every booking-side group on purpose: the medical flow
# hands over INTO booking without changing agent (symptom -> specialty
# -> doctor list -> day -> slot all run inside `medical`), which is the
# same reason graph.py's day and doctor verifiers check
# `_in_medical_guidance_handoff` as well as the booking agents.
_NEW_BOOKING = frozenset({"booking", "medical"})
_SCHEDULING = frozenset({"booking", "reschedule", "medical"})
_SPECIALTY = frozenset({"medical", "booking", "faq"})


# (stable opening text of the rule, the specialists that need it)
#
# Matched against the rule's opening text with whitespace collapsed and
# the bullet stripped, so a reflow in prompts.py does not break the
# match. Keep the prefixes long enough to be unambiguous and short
# enough to survive a small wording tweak.
_RULE_SCOPE: Tuple[Tuple[str, FrozenSet[str]], ...] = (
    # --- medical guidance -------------------------------------------
    ("In the MEDICAL GUIDANCE flow, ALWAYS call", frozenset({"medical"})),
    ("In the medical guidance flow, once the user has actually named a symptom",
     frozenset({"medical"})),
    ("In the MEDICAL GUIDANCE flow, recommending a specialty", frozenset({"medical"})),
    ("NEVER present medical guidance as a diagnosis", frozenset({"medical"})),
    ("NEVER present `find_available_doctors`", _NEW_BOOKING),
    ("NEVER offer to show the patient doctors in a specialty before", _NEW_BOOKING),
    ("NEVER suggest a doctor or specialty that doesn't genuinely relate", _NEW_BOOKING),
    ("NEVER claim this clinic offers a specialty that `list_specialties`", _SPECIALTY),

    # --- cancellation ------------------------------------------------
    ("NEVER cancel a booking without an explicit", frozenset({"cancel"})),
    ("NEVER call `cancel_appointment` without calling `check_booking_status`",
     frozenset({"cancel"})),

    # --- reschedule --------------------------------------------------
    ("NEVER reschedule without calling `reschedule_appointment`",
     frozenset({"reschedule"})),
    ("NEVER modify, recompute, or reformat a slotStart/slotEnd",
     frozenset({"reschedule"})),

    # --- identity / OTP ----------------------------------------------
    ("The message immediately following your own", _ANY_BOOKING),
    ("NEVER skip OTP when required", _ANY_BOOKING),
    ("NEVER do phone-number comparison yourself", _ANY_BOOKING | {"complaint"}),

    # --- booking references ------------------------------------------
    ("NEVER fabricate a booking reference, booking id, or time slot", _ANY_BOOKING),
    ("NEVER invent, guess, retype-from-memory, or reconstruct a booking reference",
     _ANY_BOOKING),
    ("NEVER call `lookup_appointment` or `check_booking_status` while inside",
     _ANY_BOOKING),

    # --- days and schedules ------------------------------------------
    ("NEVER work out which calendar date a weekday name", _SCHEDULING),
    ("NEVER state whether a doctor works, or does not work, on a given WEEKDAY",
     _SCHEDULING),
    ("NEVER answer a question about ONE specific day with a different day",
     _SCHEDULING),
    ("NEVER show more upcoming days than", _SCHEDULING),

    # --- doctor selection --------------------------------------------
    ("NEVER say a doctor or branch is", _SCHEDULING),
    ("NEVER accept, confirm, or proceed with a doctor name the user typed",
     _NEW_BOOKING | {"faq"}),
    ("Once a doctor has been chosen, NEVER offer to list doctors again",
     _NEW_BOOKING),

    # --- new-booking mechanics ---------------------------------------
    ("NEVER say or imply a booking has been made",
     frozenset({"booking", "reschedule"})),
    ("NEVER skip the", frozenset({"booking"})),
    ("NEVER show the booking review card while any of its fields",
     frozenset({"booking"})),
    ("NEVER ask for a phone number or name before a specific TIME SLOT",
     frozenset({"booking"})),
    ("A tool result saying a specific detail was REJECTED", frozenset({"booking"})),

    # --- fees and the service catalogue ------------------------------
    ("NEVER state a price/fee unless the user explicitly asked", _SPECIALTY),
    ("NEVER answer a", _SPECIALTY),
)


_BULLET_RE = re.compile(r"\n(?=- )")

# Below this, the block clearly did not split into rules and something
# about the prompt's formatting has changed - fail safe instead of
# guessing.
_MIN_PLAUSIBLE_RULES = 20

_CACHE: Dict[Tuple[int, str], str] = {}
_CACHE_MAX_ENTRIES = 64


def _normalize(rule: str) -> str:
    """The rule's opening text, whitespace collapsed, bullet stripped."""

    flat = " ".join(rule.strip().split())
    return flat[2:] if flat.startswith("- ") else flat


def _scope_for(rule: str) -> FrozenSet[str]:
    """Which specialists this rule is for. `ALL` when unclassified -
    see property 2 in the module docstring."""

    text = _normalize(rule)
    for prefix, agents in _RULE_SCOPE:
        if text.startswith(prefix):
            return agents
    return ALL


def scope_hard_rules(hard_rules: str, agent_name: str) -> str:
    """`hard_rules` with the rules that do not apply to `agent_name`
    removed. Returns it unchanged for `concierge`, for an unknown agent,
    or whenever the block cannot be parsed."""

    if not hard_rules or agent_name not in ALL:
        return hard_rules

    key = (hash(hard_rules), agent_name)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    parts = _BULLET_RE.split(hard_rules)

    if len(parts) < _MIN_PLAUSIBLE_RULES:
        logger.warning(
            "agents.hard_rules: the GLOBAL HARD RULES block split into only %d "
            "part(s) - the bullet format must have changed. Sending the whole "
            "block to %s rather than risk dropping a rule.",
            len(parts), agent_name,
        )
        return hard_rules

    # parts[0] is the `====` banner plus whatever preamble sits above
    # the first bullet - always kept, it is not a rule.
    kept = [parts[0]]
    dropped = 0
    for rule in parts[1:]:
        if agent_name in _scope_for(rule):
            kept.append(rule)
        else:
            dropped += 1

    result = "\n".join(kept)

    logger.info(
        "agents.hard_rules: %s keeps %d/%d rules (%d not applicable)",
        agent_name, len(kept) - 1, len(parts) - 1, dropped,
    )

    if len(_CACHE) >= _CACHE_MAX_ENTRIES:
        _CACHE.clear()
    _CACHE[key] = result

    return result


def coverage_report(hard_rules: str) -> Dict[str, object]:
    """How many rules each specialist keeps. Diagnostics/tests only."""

    parts = _BULLET_RE.split(hard_rules)
    rules = parts[1:]
    report: Dict[str, object] = {"_total": len(rules)}
    for name in sorted(ALL):
        report[name] = sum(1 for r in rules if name in _scope_for(r))
    report["_unclassified"] = sum(1 for r in rules if _scope_for(r) == ALL)
    report["_narrowed"] = sum(1 for r in rules if _scope_for(r) != ALL)
    return report
