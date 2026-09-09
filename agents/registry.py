"""
The specialist roster.

Each entry says three things and nothing more:
  - which of prompts.py's existing sections that specialist is given,
  - which of tools.py's existing tools it may call,
  - the one-paragraph "YOUR JOB" that replaces the generic six-item one.

Design notes worth knowing before changing anything here
--------------------------------------------------------
1. `concierge` is the fallback the router uses whenever it cannot
   confidently classify a message. It keeps EVERY TOOL (see design note
   3 - a fallback that cannot act is the one thing worse than a
   fallback that answers imperfectly), and every shared section
   including the complete GLOBAL HARD RULES.

   Its PROMPT is narrowed to the sections it can actually act on:
   `medical`, `faq`, `entity_info`. It was previously given all seven
   flows - ~47k tokens on every call, by far the most expensive agent
   in the system - and the reason it can be narrowed safely is
   structural rather than a judgement call: the openings of the booking
   and cancel/reschedule flows are AUTHORED IN CODE
   (`_build_booking_entry_directive`,
   `_build_identifier_choice_directive`), and both already list
   `concierge` in their own agent gates. A vague "عايز أحجز" or "عايز
   ألغي" arriving at the fallback is therefore answered with fixed text
   and ZERO model calls, with or without the flow's prose in the
   prompt. The router also re-runs before every turn (note 4), so the
   owning specialist has the conversation by the next message.

   `medical` is kept deliberately: a symptom message carrying no strong
   cue routes here, and that is the one case where a poor first reply
   is dangerous rather than merely clumsy - see the comment on the
   unrelated-specialty verifier in graph.py, which names this exact
   path.

   Set CONCIERGE_FULL_PROMPT=true to restore the old full-prompt
   fallback without a code change.

2. `reschedule` is given the CANCEL section as well, on purpose - the
   reschedule flow explicitly reuses cancellation's STEP 1-2 for
   identifying the booking and verifying identity ("STEP R1/R2 -
   Identify the booking and verify identity ... reuses STEPs 1-2").
   Dropping it would have broken rescheduling on the first turn.

3. Tool subsets are generous rather than minimal. A specialist that is
   missing a tool it needs cannot recover - the flow simply stalls with
   the model unable to act. A specialist that has one extra tool it
   never calls costs a few tokens of schema. The asymmetry is obvious,
   so every borderline tool is included.

4. Every specialist also receives the SERVICE INDEX block, so it can
   always recognise a request that belongs to a teammate instead of
   telling the patient "I can't help with that". The supervisor re-runs
   before every turn, so a patient who switches topics is already being
   handled by the right specialist by the time the next reply is
   written - no lost turn, no visible handover.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import config
import tools as tools_module
from agents.hard_rules import scope_hard_rules
from agents.response_contract import RESPONSE_FORMAT_CONTRACT
from agents.sections import (PREAMBLE_KEY, extra_keys, flow_is_sliceable,
                             has_all_required, split_flow_steps, steps_to_send)

logger = logging.getLogger(__name__)


CONCIERGE = "concierge"


# ==========================================================
# Tool lookup by name (tools.ALL_TOOLS is left untouched)
# ==========================================================

_TOOLS_BY_NAME = {}
for _tool in tools_module.ALL_TOOLS:
    _TOOLS_BY_NAME[getattr(_tool, "name", getattr(_tool, "__name__", ""))] = _tool


_IDENTITY_TOOLS = (
    "validate_phone_format",
    "compare_phone",
    "send_otp",
    "verify_otp",
    "get_patient_info",
)


# ==========================================================
# The shared index of everything this assistant can do
# ==========================================================

SERVICE_INDEX = """\
============================================================
EVERYTHING THIS ASSISTANT CAN DO
============================================================
You are one voice, and this assistant as a whole handles all of the
following. The section(s) below describe the part you are answering
right now, but you must never tell a patient that something on this list
is outside what you can do:
  🗓️  Booking a brand new appointment
  ✏️  Rescheduling an existing appointment
  ❌  Cancelling an existing appointment
  🩺  Medical guidance - symptom to the right specialty and doctor
  ℹ️  Questions about the hospital, its branches, doctors and services
  📝  Recording a complaint or a suggestion
  👤  Handing over to a human member of staff on request

If the patient's message turns to one of these that isn't covered by
your own section below, answer whatever you safely can this turn and ask
the single most useful next question - the conversation continues
seamlessly, and the patient must never be told they are being
transferred, routed, or handed to a different agent. There is only ever
one assistant from their side.
"""


# ==========================================================
# Specialist definitions
# ==========================================================

@dataclass(frozen=True)
class AgentSpec:
    """One specialist: what it knows, what it can do, what it is for."""

    name: str
    title: str
    job: str
    section_keys: Tuple[str, ...] = ()
    tool_names: Tuple[str, ...] = ()
    # True -> ignore section_keys/tool_names and use the entire prompt
    # and every tool (the legacy single-agent behaviour).
    full_access: bool = False
    # True -> use `section_keys` for the prompt, but bind EVERY tool.
    #
    # Only `concierge` sets this. Its prompt can be narrowed safely
    # (see config.CONCIERGE_FULL_PROMPT for exactly why), but its TOOLS
    # cannot: it is the fallback, and design note 3 below applies with
    # full force - a fallback that is missing a tool it turns out to
    # need cannot recover, it simply stalls.
    full_tools: bool = False

    def tools(self) -> List:
        """Resolves this specialist's tools out of tools.ALL_TOOLS.

        Falls back to the full set when tool scoping is disabled, and
        logs (rather than raises) if a name ever stops existing - a typo
        here must not be able to take the whole agent down at import
        time.
        """

        if self.full_access or self.full_tools or not config.AGENT_TOOL_SCOPING:
            return list(tools_module.ALL_TOOLS)

        resolved = []
        for name in self.tool_names:
            tool = _TOOLS_BY_NAME.get(name)
            if tool is None:
                logger.warning(
                    "agents.registry: %s lists unknown tool %r - skipped. "
                    "Check it still exists in tools.ALL_TOOLS.",
                    self.name, name,
                )
                continue
            resolved.append(tool)

        if not resolved:
            logger.warning(
                "agents.registry: %s resolved to zero tools - falling back "
                "to the full tool set rather than leaving it unable to act.",
                self.name,
            )
            return list(tools_module.ALL_TOOLS)

        return resolved


_SPECS: Tuple[AgentSpec, ...] = (

    AgentSpec(
        name=CONCIERGE,
        title="Concierge / fallback",
        # PROMPT NARROWED, TOOLS NOT. See config.CONCIERGE_FULL_PROMPT
        # for the full reasoning and the kill switch; in short, the
        # openings of the booking and cancel/reschedule flows are
        # authored in code and already gated to include `concierge`, so
        # the fallback can start either of them with fixed text and no
        # model call at all - it does not need their prose. What it does
        # still need is the medical section (a symptom carrying no
        # strong cue lands here, and answering that one wrongly is the
        # dangerous case), plus hospital info and entity lookup for the
        # questions it genuinely answers itself.
        #
        # `full_access=True` is what CONCIERGE_FULL_PROMPT restores.
        full_access=config.CONCIERGE_FULL_PROMPT,
        full_tools=True,
        section_keys=("medical", "faq", "entity_info"),
        job="""\
============================================================
YOUR JOB
============================================================
You are the first point of contact and the fallback for anything that
hasn't clearly become one specific request yet. Greet the patient, find
out what they actually need, and take the next step with them.

If their message states no intent yet (just "مرحبا", "hi", "صباح
الخير"), do not guess and do not start asking for a booking reference or
a phone number. Let the greeting's own closing question stand and wait.

If they ask for something this hospital genuinely doesn't do, say so
warmly in one sentence and offer what you can help with instead.""",
    ),

    AgentSpec(
        name="cancel",
        title="Cancellation",
        section_keys=("cancel",),
        tool_names=_IDENTITY_TOOLS + (
            "lookup_appointment",
            "check_booking_status",
            "cancel_appointment",
            "match_entity_info",
            "reset_booking_session",
            "request_human_handoff",
            "share_branch_location",
        ),
        job="""\
============================================================
YOUR JOB
============================================================
This patient wants to CANCEL an existing appointment. Take them through
the cancellation flow below - identify the booking, verify identity,
confirm the exact appointment back to them, and only then cancel it.

Two things you must not do here: never cancel without an explicit "yes"
in the same turn, and never start creating, moving, or suggesting a new
appointment. If, after the cancellation is done, they ask to book or
move something instead, just take the next natural step with them.""",
    ),

    AgentSpec(
        name="reschedule",
        title="Reschedule",
        # The reschedule flow explicitly reuses cancellation's STEP 1-2
        # to identify the booking and verify identity - it is incomplete
        # without that section.
        section_keys=("reschedule", "cancel"),
        tool_names=_IDENTITY_TOOLS + (
            "lookup_appointment",
            "check_booking_status",
            "get_doctor_schedule",
            "get_available_reschedule_slots",
            "reschedule_appointment",
            "get_next_weekday_date",
            "resolve_available_day",
            "match_entity_info",
            "reset_booking_session",
            "request_human_handoff",
            "share_branch_location",
        ),
        job="""\
============================================================
YOUR JOB
============================================================
This patient wants to MOVE an existing appointment to a different time.
Follow the RESCHEDULE FLOW below. Identifying the booking and verifying
identity work exactly like cancellation's STEP 1-2, which is included
below for that reason - use it for those two steps only, and never
actually cancel anything.

Never reschedule without a fresh `lookup_appointment` in the same turn,
and never alter a slot value returned by
`get_available_reschedule_slots` before passing it on.""",
    ),

    AgentSpec(
        name="booking",
        title="New booking",
        section_keys=("booking", "entity_info"),
        tool_names=_IDENTITY_TOOLS + (
            "list_specialties",
            "find_available_doctors",
            "find_best_doctor_in_specialty",
            "list_branches_for_specialty",
            "list_hospital_services",
            "list_branch_services",
            "match_entity_for_booking",
            "list_available_days_for_booking",
            "get_doctor_schedule_for_booking",
            "get_available_slots_for_booking",
            "select_appointment_slot",
            "create_new_booking",
            "get_doctor_fees",
            "resolve_available_day",
            "get_next_weekday_date",
            "match_entity_info",
            "reset_booking_session",
            "request_human_handoff",
            "share_branch_location",
        ),
        job="""\
============================================================
YOUR JOB
============================================================
This patient wants a BRAND NEW appointment that does not exist yet.
Follow the NEW BOOKING FLOW below, one question per message, all the way
to a real `create_new_booking` result.

You are deliberately not given the tools that look up existing bookings.
That is not an oversight - reaching for an existing booking mid-way
through creating a new one has surfaced an unrelated patient's
appointment before. Everything you need to identify this patient is in
the flow below.

If the patient asks what SERVICES the hospital or a branch offers (not
which medical specialties/doctors it has), that is a different question
from this flow - call `list_hospital_services` (hospital-wide) or
`list_branch_services` (one branch) for it, never `list_specialties` or
`list_branches_for_specialty`, which are for medical specialties and
booking availability, not the clinic's service catalogue.""",
    ),

    AgentSpec(
        name="medical",
        title="Medical guidance",
        section_keys=("medical", "entity_info"),
        tool_names=(
            "list_specialties",
            "find_available_doctors",
            "find_best_doctor_in_specialty",
            "list_branches_for_specialty",
            "list_hospital_services",
            "list_branch_services",
            "find_branches_offering_service",
            "match_entity_info",
            "get_doctor_fees",
            "request_human_handoff",
            "share_branch_location",
        ),
        job="""\
============================================================
YOUR JOB
============================================================
This patient has described a symptom or a health concern. Follow the
MEDICAL GUIDANCE FLOW below: safety first, understand the symptom
before naming any specialty, and never present anything you say as a
diagnosis.

Once they have actually named a symptom, every reply must carry both
comfort/self-care AND your single question - never a bare question. If
they have named no symptom yet, ask what it is first and invent no
comfort advice out of nothing.

Only ever name a specialty `list_specialties` actually returned, and
only ever name a doctor a tool returned in this conversation.""",
    ),

    AgentSpec(
        name="faq",
        title="Hospital information",
        section_keys=("faq", "entity_info"),
        tool_names=(
            "answer_hospital_faq",
            "list_hospital_services",
            "list_branch_services",
            "find_branches_offering_service",
            "list_specialties",
            "list_branches_for_specialty",
            "find_available_doctors",
            "match_entity_info",
            "get_doctor_fees",
            "request_human_handoff",
            "share_branch_location",
        ),
        job="""\
============================================================
YOUR JOB
============================================================
This patient is asking about the hospital itself - its services, vision
and values, branches and addresses, contact details, policies, partners,
or a specific doctor or branch by name. Answer from the sections below.

Never answer a "what services do you offer" question from
`list_specialties` or from `answer_hospital_faq` similarity results -
call `list_hospital_services` and show the complete list it returns,
unchanged. Never state a fee unless they asked about cost and
`get_doctor_fees` returned it.""",
    ),

    AgentSpec(
        name="complaint",
        title="Complaints and suggestions",
        section_keys=("complaint",),
        tool_names=(
            "send_complaint_email",
            "validate_phone_format",
            "compare_phone",
            "match_entity_info",
            "request_human_handoff",
            "share_branch_location",
        ),
        job="""\
============================================================
YOUR JOB
============================================================
This patient wants to raise a COMPLAINT or offer a SUGGESTION. Follow
the COMPLAINT FLOW below and collect what it asks for, one question per
message.

Do not answer a complaint with an FAQ answer or a booking offer, and do
not make them explain twice that they want to complain. Only the status
"sent" means it actually reached the quality team - never tell them it
was sent when it wasn't.""",
    ),
)


AGENT_SPECS: Dict[str, AgentSpec] = {spec.name: spec for spec in _SPECS}
AGENT_NAMES: Tuple[str, ...] = tuple(AGENT_SPECS)


def get_spec(name: Optional[str]) -> AgentSpec:
    """Never raises - an unknown name falls back to the full-access
    concierge, i.e. the old single-agent behaviour."""

    spec = AGENT_SPECS.get(name or "")
    if spec is None:
        if name:
            logger.warning(
                "agents.registry: unknown agent %r - falling back to %s",
                name, CONCIERGE,
            )
        return AGENT_SPECS[CONCIERGE]
    return spec


def tools_for(name: Optional[str]) -> List:
    return get_spec(name).tools()


# ==========================================================
# Prompt composition
# ==========================================================

# Sections every specialist receives, in the order the original prompt
# used them, so the model sees the structure it was tuned against.
_SHARED_HEAD_KEYS = (PREAMBLE_KEY, "language", "dialect")
_SHARED_MID_KEYS = ("reference_phrases", "fixed_templates")
_SHARED_TAIL_KEYS = ("hard_rules",)


def build_agent_prompt(sections: Dict[str, str], agent_name: str,
                       step: str = None) -> str:
    """
    Composes one specialist's system prompt out of the already-split
    sections of the tenant's real, fully-substituted prompt.

    `step` (e.g. "NB3") asks for JUST-IN-TIME delivery of that flow: the
    flow's step blocks are dropped from their usual position and only
    the current step and the next one are appended, at the very end.

    WHY THE END, AND NOT IN PLACE. Sending only the current step is the
    obvious version of this and it is the expensive one. OpenAI's cache
    matches a leading PREFIX, so a step block sitting in its usual
    position - a third of the way in - makes everything after it
    uncacheable too, including the 7.7k of hard rules. Measured on the
    booking prompt, per call:

        whole flow, one prefix      28,741 tokens   $0.0144 cached
        step in place               16,173 tokens   $0.0258 cached
        step at the end             16,173 tokens   $0.0113 cached

    Slicing in place sends 44% fewer tokens and costs 79% MORE once the
    cache is warm. Moving the slice past the hard rules keeps 14,015
    tokens in the stable prefix and is cheaper at every hit rate.

    FAIL-SAFE, twice: if the split didn't produce everything expected
    (someone renamed a heading in prompts.py) this returns the full
    prompt, and if `step` is unknown or the flow has no step structure
    the flow is sent whole - both of which are the unsliced behaviour.
    """

    spec = get_spec(agent_name)

    if not has_all_required(sections):
        logger.warning(
            "agents.registry: prompt sections incomplete - %s falls back "
            "to the full prompt.", spec.name,
        )
        return _rejoin_everything(sections)

    if spec.full_access:
        flow_keys = [
            key for key in sections
            if key not in _SHARED_HEAD_KEYS + _SHARED_MID_KEYS + _SHARED_TAIL_KEYS
            and key not in ("your_job", "__order__")
        ]
    else:
        flow_keys = list(spec.section_keys) + extra_keys(sections)

    parts: List[str] = []

    for key in _SHARED_HEAD_KEYS:
        parts.append(sections[key])

    # The output contract sits directly after the language rules and
    # before everything flow-specific: high enough to govern the whole
    # reply, low enough that it cannot be read as overriding the
    # language/dialect mirroring above it.
    parts.append(RESPONSE_FORMAT_CONTRACT)

    for key in _SHARED_MID_KEYS:
        parts.append(sections[key])

    parts.append(spec.job)
    parts.append(SERVICE_INDEX)

    # Step blocks held back to the very end - see this function's
    # docstring for why the position, not the size, is what decides
    # whether this saves anything.
    deferred: List[str] = []

    seen = set(_SHARED_HEAD_KEYS + _SHARED_MID_KEYS + _SHARED_TAIL_KEYS)
    for key in flow_keys:
        if key in seen or key not in sections:
            continue
        seen.add(key)

        wanted = steps_to_send(key, step) if step else None
        if wanted and not flow_is_sliceable(key, sections[key]):
            logger.warning(
                "agents.registry: %s's step headings no longer split cleanly - "
                "sending the whole flow rather than one with a step missing", key,
            )
            wanted = None
        if not wanted:
            parts.append(sections[key])
            continue

        head, steps = split_flow_steps(sections[key])
        if not steps:
            parts.append(sections[key])
            continue

        # The flow's banner and preamble stay where they were, so the
        # prompt still reads as having that flow in it.
        if head.strip():
            parts.append(head)

        chosen = [text for step_key, text in steps if step_key in wanted]
        if not chosen:
            # Asked for steps that this flow does not contain - send it
            # whole rather than a flow with its body missing.
            parts.append("\n\n".join(text for _, text in steps))
            continue

        deferred.append(
            "============================================================\n"
            f"WHERE THIS BOOKING HAS GOT TO - STEP {wanted[0]}\n"
            "============================================================\n"
            "The step below is the one this conversation is actually on, "
            "followed by the one it moves to next. The earlier steps of "
            "this flow are behind you and are not repeated here.\n\n"
            "This does NOT override the GLOBAL HARD RULES above - it "
            "comes after them only so the rules stay identical on every "
            "turn of every conversation.\n\n"
            + "\n\n".join(chosen)
        )

    # GLOBAL HARD RULES, NARROWED TO THIS SPECIALIST.
    #
    # The block is ~7.7k tokens and used to go to every specialist in
    # full, on every call - the largest single shared cost in the
    # prompt, and roughly half of it is rules the specialist cannot
    # act on (see agents/hard_rules.py). `scope_hard_rules` drops only
    # rules explicitly classified as belonging to another flow;
    # anything it does not recognise stays, so the default is exactly
    # the old behaviour and `concierge` is returned untouched.
    for key in _SHARED_TAIL_KEYS:
        section = sections[key]
        if key == "hard_rules":
            section = scope_hard_rules(section, spec.name)
        parts.append(section)

    parts.extend(deferred)

    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def _rejoin_everything(sections: Dict[str, str]) -> str:
    order = [k for k in sections.get("__order__", "").split("\n") if k]
    if not order:
        order = [k for k in sections if k != "__order__"]
    return "\n\n".join(sections[k].strip() for k in order if k in sections)
