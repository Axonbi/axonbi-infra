"""
The supervisor.

Decides which specialist owns the current turn. It runs ONCE per user
turn, at the top of the graph - never inside the agent<->tools loop - so
a turn that makes six tool calls still routes exactly once.

WHY IT IS DETERMINISTIC BY DEFAULT
----------------------------------
Three reasons, in order of importance:

  1. CONSISTENCY, which is the whole point of this refactor. The same
     sentence must always reach the same specialist and therefore always
     produce the same shape of reply. An LLM classifier re-deciding on
     every turn is exactly the "it answers me one way and answers him
     another way" problem, moved one layer down.

  2. COST AND LATENCY. An LLM router adds a full model call to every
     single turn, before the turn's real work has even started.

  3. TESTABILITY. The existing test suite scripts the LLM's replies one
     per `.invoke()`; a router that quietly consumed one of those calls
     would desynchronise every scripted conversation in the repo.

`ROUTER_MODE=llm` is available for anyone who wants the flexible
version - it only fires on genuinely ambiguous turns, never on clear
ones - but it is off by default.

STICKINESS IS THE OTHER HALF
----------------------------
Most messages inside a flow carry no intent words at all: "نعم", "١",
"123456", "+201001234567", "الخميس". Re-classifying those from scratch
would scatter one conversation across several specialists. So the rule
is: a message with no clear cue KEEPS the current specialist. Only a
clear, deliberate change of subject moves the conversation, and only a
completed flow releases it back to the concierge.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import config
from agents.registry import AGENT_NAMES, CONCIERGE

logger = logging.getLogger(__name__)


# ==========================================================
# Score thresholds
# ==========================================================

# A cue this strong switches the conversation even mid-flow.
_SWITCH_THRESHOLD = 8

# Enough to pick a specialist when nothing is active yet, but not enough
# to interrupt a flow already in progress.
_START_THRESHOLD = 4


# ==========================================================
# Intent cues
#
# Written as (weight, pattern) pairs. Patterns are matched against a
# normalised copy of the message (Arabic diacritics stripped, alef/ya/ta
# marbuta unified, Arabic-Indic digits converted), so "أبغى" and "ابغي"
# and "ابغى" all hit the same rule without listing every spelling.
#
# Weights: 10 = an unambiguous verb+object phrase; 6 = a strong single
# keyword; 3 = a topical hint that only decides an otherwise-empty turn.
# ==========================================================

# ==========================================================
# CRISIS - ONE PATTERN, USED EVERYWHERE
# ==========================================================
#
# THE SINGLE DEFINITION. graph.py builds the crisis directive from this
# same object (graph._CRISIS_RE is assigned from it), and the routing
# cue below uses it too, so a phrasing added here is recognised by every
# part of the system at once.
#
# It had drifted before: the router's own crisis cue covered only
# "\u0627\u0646\u062A\u062D\u0631"/"suicide"/"kill myself", while graph.py's was far broader -
# so "\u0639\u0627\u064A\u0632\u0629 \u0623\u0645\u0648\u062A", one of the commonest ways this is actually said in
# Arabic, raised the crisis DIRECTIVE but never moved the conversation
# to the medical specialist, which is the only one carrying the crisis
# rules. Two patterns for one concept, and the narrower one sat on the
# safety-critical path.
CRISIS_RE = re.compile(
    # Arabic, including the colloquial future prefix ("\u0647\u0646\u062A\u062D\u0631").
    r"(?:\u0647|\u062D|\u0633\u0627|\u0633\u0623)?\u0627\u0646\u062A\u062D\u0631|(?:\u0647|\u062D)\u0646\u062A\u062D\u0631|\u0627\u0644\u0627\u0646\u062A\u062D\u0627\u0631|"
    r"(?:\u0639\u0627\u064A\u0632|\u0639\u0627\u0648\u0632|\u0628\u062F\u064A|\u0627\u0628\u064A|\u0627\u0628\u063A\u0649|\u0646\u0641\u0633\u064A)\s*(?:\w+\s+){0,2}(?:\u0627\u0645\u0648\u062A|\u0627\u0646\u0647\u064A\s*\u062D\u064A\u0627\u062A\u064A|\u0627\u0642\u062A\u0644\s*\u0646\u0641\u0633\u064A)|"
    r"(?:\u0645\u0634|\u0645\u0627|\u0645\u0648)\s*(?:\u0639\u0627\u064A\u0632|\u0639\u0627\u0648\u0632|\u0628\u062F\u064A|\u0627\u0628\u064A)\s*(?:\w+\s+){0,2}(?:\u0627\u0639\u064A\u0634|\u0627\u0643\u0645\u0644)|"
    r"(?:\u0627\u0630\u064A|\u0623\u0630\u064A|\u0627\u0624\u0630\u064A|\u0623\u0624\u0630\u064A|\u0627\u062C\u0631\u062D|\u0623\u062C\u0631\u062D)\s*\u0646\u0641\u0633\u064A|"
    r"(?:\u0627\u0646\u0647\u064A|\u0623\u0646\u0647\u064A|\u0627\u062E\u0644\u0635\s*\u0645\u0646)\s*\u062D\u064A\u0627\u062A|"
    r"(?:\u062A\u0639\u0628\u062A|\u0632\u0647\u0642\u062A|\u0645\u0644\u064A\u062A)\s*\u0645\u0646\s*(?:\u0627\u0644)?\u062D\u064A\u0627\u0647|"
    # English.
    r"\bkill\s+my\s?self\b|\bsuicid\w*|\bend\s+(?:my|it\s+all)\b[^.\n]{0,12}\blife\b|"
    r"\bend\s+my\s+life\b|\bwant\s+to\s+die\b|\bhurt\s+my\s?self\b|"
    r"\bself[\s-]?harm\b|\bdon'?t\s+want\s+to\s+(?:live|be\s+here)\b|"
    r"\bno\s+reason\s+to\s+live\b",
    re.IGNORECASE,
)


# ==========================================================
# INJURIES - A HEALTH MESSAGE WITH NO PAIN WORD IN IT
# ==========================================================
#
# Every medical cue below asks for a SYMPTOM: a pain word ("\u0648\u062C\u0639",
# "\u0635\u062F\u0627\u0639"), or a body part followed by a hurting verb ("\u0628\u0637\u0646\u064A \u0628\u062A\u0648\u062C\u0639\u0646\u064A").
# An injury is described the other way round - by WHAT HAPPENED. "\u0631\u062C\u0644\u064A
# \u0648\u0642\u0639\u062A \u0639\u0644\u064A\u0647\u0627", "\u0627\u062A\u062E\u0628\u0637\u062A \u0641\u064A \u0627\u064A\u062F\u064A", "\u0631\u062C\u0644\u064A \u0627\u062A\u0643\u0633\u0631\u062A", "\u062D\u0631\u0642\u062A \u0627\u064A\u062F\u064A" contain no
# symptom word at all, so they scored NOTHING and stayed with whichever
# specialist happened to be active.
#
# CONFIRMED IN A REAL CONVERSATION: "\u0631\u062C\u0644\u064A \u0648\u0642\u0639\u062A \u0639\u0644\u064A\u0647\u0627" was answered with
# the out-of-scope service menu - "\u0623\u0646\u0627 \u0644\u0637\u064A\u0641\u0629\u060C \u0627\u0644\u0645\u0633\u0627\u0639\u062F\u0629 \u0627\u0644\u0627\u0641\u062A\u0631\u0627\u0636\u064A\u0629...
# \u0648\u0645\u062E\u062A\u0635\u0629 \u0628\u0645\u0633\u0627\u0639\u062F\u062A\u0643 \u0641\u064A \u062E\u062F\u0645\u0627\u062A \u0627\u0644\u0645\u0633\u062A\u0634\u0641\u0649" - to somebody describing an
# injury. They sent the identical message again and only then got
# "\u0627\u0644\u0644\u0647 \u064A\u0634\u0627\u0641\u064A\u0643 \u0648\u064A\u0639\u0627\u0641\u064A\u0643 \uD83C\uDF37 \u0645\u0646 \u0648\u064A\u0646 \u062A\u062D\u0633 \u0628\u0627\u0644\u0623\u0644\u0645 \u0628\u0627\u0644\u0636\u0628\u0637\u061F".
#
# The patterns are built around a BODY PART so that ordinary uses of
# these verbs cannot trip them: "\u0648\u0642\u0639\u062A \u0627\u0644\u0639\u0642\u062F" (I signed the contract)
# has no body part and matches nothing, while "\u0631\u062C\u0644\u064A \u0648\u0642\u0639\u062A \u0639\u0644\u064A\u0647\u0627" does.
_INJURY_BODY_PART = (
    r"(?:\u0631\u062C\u0644|\u0631\u062C\u0644\u064A|\u0631\u062C\u0644\u064A\u0627|\u0627\u064A\u062F|\u0627\u064A\u062F\u064A|\u064A\u062F\u064A|\u0631\u0627\u0633|\u0631\u0627\u0633\u064A|\u0636\u0647\u0631|\u0636\u0647\u0631\u064A|\u0638\u0647\u0631|\u0638\u0647\u0631\u064A|\u0643\u062A\u0641|\u0643\u062A\u0641\u064A|"
    r"\u0631\u0643\u0628\u0647|\u0631\u0643\u0628\u062A\u064A|\u0627\u0635\u0628\u0639|\u0635\u0628\u0627\u0639|\u0635\u0648\u0627\u0628\u0639|\u0642\u062F\u0645|\u0642\u062F\u0645\u064A|\u0628\u0637\u0646|\u0628\u0637\u0646\u064A|\u0635\u062F\u0631|\u0635\u062F\u0631\u064A|\u0631\u0642\u0628\u0647|\u0631\u0642\u0628\u062A\u064A|"
    r"\u0639\u064A\u0646|\u0639\u064A\u0646\u064A|\u0633\u0646|\u0633\u0646\u0627\u0646\u064A|\u0636\u0631\u0633|\u0636\u0631\u0633\u064A|\u0643\u0648\u0639|\u0643\u0648\u0639\u064A|\u0645\u0639\u0635\u0645|\u0645\u0639\u0635\u0645\u064A|\u0641\u062E\u062F|\u0641\u062E\u062F\u064A|\u0643\u0627\u062D\u0644)"
)

_INJURY_VERB = (
    r"(?:\u0648\u0642\u0639|\u0627\u062A\u062E\u0628\u0637|\u062E\u0628\u0637|\u0627\u0646\u062E\u0628\u0637|\u0627\u062A\u0643\u0633\u0631|\u0643\u0633\u0631|\u0627\u0646\u0643\u0633\u0631|\u0627\u062A\u062D\u0631\u0642|\u062D\u0631\u0642|\u0627\u0646\u062D\u0631\u0642|\u0627\u062A\u062C\u0631\u062D|\u062C\u0631\u062D|"
    r"\u0627\u0646\u062C\u0631\u062D|\u0627\u062A\u0639\u0648\u0631|\u062A\u0639\u0648\u0631|\u0627\u0644\u062A\u0648|\u0627\u062A\u0644\u0648|\u0644\u0648\u064A|\u0627\u0646\u062A\u0641\u062E|\u0648\u0631\u0645|\u0646\u0632\u0641|\u0627\u062A\u062F\u0639\u0633|\u0627\u0646\u062F\u0639\u0633)"
)

INJURY_RE = re.compile(
    # body part first: "\u0631\u062C\u0644\u064A \u0648\u0642\u0639\u062A \u0639\u0644\u064A\u0647\u0627", "\u0627\u064A\u062F\u064A \u0627\u062A\u0643\u0633\u0631\u062A"
    _INJURY_BODY_PART + r"\w*\s*[^.\n]{0,12}" + _INJURY_VERB + r"|"
    # verb first: "\u0648\u0642\u0639\u062A \u0639\u0644\u0649 \u0631\u062C\u0644\u064A", "\u0627\u062A\u062E\u0628\u0637\u062A \u0641\u064A \u0627\u064A\u062F\u064A", "\u0643\u0633\u0631\u062A \u0627\u064A\u062F\u064A"
    r"(?:\u0648\u0642\u0639\u062A|\u0627\u062A\u062E\u0628\u0637\u062A|\u062E\u0628\u0637\u062A|\u0643\u0633\u0631\u062A|\u0627\u062A\u0643\u0633\u0631\u062A|\u062D\u0631\u0642\u062A|\u0627\u062A\u062D\u0631\u0642\u062A|\u062C\u0631\u062D\u062A|\u0627\u062A\u062C\u0631\u062D\u062A|\u0644\u0648\u064A\u062A|\u0627\u0644\u062A\u0648\u064A\u062A|\u0627\u062A\u062F\u0639\u0633\u062A)"
    r"\s*(?:\u0639\u0644\u0649|\u0641\u064A|\u0645\u0646|\u0628)?\s*[^.\n]{0,10}" + _INJURY_BODY_PART + r"|"
    # unambiguous on their own - these words have no non-injury reading
    r"(?:^|\s)(?:\u0627\u062A\u0639\u0648\u0631\u062A|\u0627\u062A\u0639\u0648\u0631|\u0627\u062A\u0643\u0633\u0631\u062A|\u0627\u0646\u0643\u0633\u0631\u062A|\u0627\u062A\u062D\u0631\u0642\u062A|\u0627\u0646\u062D\u0631\u0642\u062A|\u0627\u062A\u062C\u0631\u062D\u062A|\u0627\u0646\u062C\u0631\u062D\u062A|"
    r"\u0627\u062A\u062E\u0628\u0637\u062A|\u0627\u0646\u062E\u0628\u0637\u062A|\u0627\u0644\u062A\u0648\u064A\u062A|\u0627\u062A\u062F\u0639\u0633\u062A)(?:\s|$)|"
    # an accident
    r"(?:\u0639\u0645\u0644\u062A|\u062D\u0635\u0644\s*\u0644\u064A|\u062C\u0627\u0644\u064A|\u062A\u0639\u0631\u0636\u062A\s*\u0644)\s*(?:\u062D\u0627\u062F\u062B|\u062D\u0627\u062F\u062B\u0647|\u0635\u062F\u0645\u0647)|"
    r"(?:^|\s)\u062D\u0627\u062F\u062B\s*(?:\u0639\u0631\u0628\u064A\u0647|\u0633\u064A\u0631|\u0645\u0631\u0648\u0631\u064A)|"
    # a bite or a sting
    r"(?:\u0639\u0636\u0646\u064A|\u0639\u0636\u062A\u0646\u064A|\u0644\u062F\u063A\u0646\u064A|\u0644\u062F\u063A\u062A\u0646\u064A|\u0642\u0631\u0635\u0646\u064A)\s*(?:\u0643\u0644\u0628|\u0642\u0637\u0647|\u0646\u062D\u0644\u0647|\u0639\u0642\u0631\u0628|\u062B\u0639\u0628\u0627\u0646|\u062D\u0634\u0631\u0647)?|"
    # English
    r"\b(?:i\s+)?(?:fell|broke|sprained|twisted|burn(?:ed|t)|cut|injured|"
    r"hurt|banged|sprain)\b[^.\n]{0,20}"
    r"\b(?:my|leg|arm|hand|foot|ankle|wrist|knee|back|head|finger|shoulder|toe|rib)\b|"
    r"\b(?:broken|fractured|sprained|dislocated)\s+"
    r"(?:leg|arm|hand|foot|ankle|wrist|knee|finger|shoulder|rib|bone)\b"
)


def looks_like_health_message(text: str) -> bool:
    """True when the message is about the patient's own body - a
    symptom, an injury, a medication question or a crisis.

    ONE definition, so the router and graph.py's scope-refusal guard
    cannot disagree about what counts. It reads the medical cue scores
    rather than keeping a second list, so anything added to `_CUES`
    below is covered here automatically - which is exactly the drift
    that let an injury reach the service menu in the first place."""

    return "medical" in score_message(text)


_DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670\u0640]")
_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_ALEF_RE = re.compile(r"[أإآٱ]")
_WHITESPACE_RE = re.compile(r"\s+")


_CUES: Dict[str, List[Tuple[int, str]]] = {

    "cancel": [
        (10, r"(?:الغ|إلغ|ابطال|ابغى\s*الغ|عايز\s*الغ|عاوز\s*الغ|ابي\s*الغ|حاب\s*الغ)\w*\s*"
             r"(?:ال)?(?:حجز|موعد|معاد|ميعاد)"),
        (10, r"\bcancel\b[^.\n]{0,20}\b(?:booking|appointment|reservation|it)\b"),
        (10, r"\b(?:booking|appointment|reservation)\b[^.\n]{0,20}\bcancel"),
        # "ألغيه"/"ألغيها"/"ابطلها" - verb + attached object pronoun, i.e.
        # "cancel it". Unambiguous, so it is strong enough to interrupt
        # another flow ("خلاص ألغيه بقى" said mid-reschedule).
        (10, r"(?:^|\s)(?:الغي|ابطل)\w*(?:ه|ها)(?:\s|$)"),
        (10, r"\bcancel\s+it\b"),
        # `\w*` on purpose: Arabic attaches the object pronoun to the
        # verb, so "ألغيه"/"ألغيها" ("cancel it") are one word. Anchoring
        # with (?:\s|$) missed every one of them.
        (6, r"(?:^|\s)(?:الغاء|الغي|ابطل)\w*(?:\s|$)"),
        (6, r"(?:^|\s)اريد\s*الالغاء"),
        (6, r"\bcancel(?:lation|ling)?\b"),
        (6, r"\bcall\s*off\b"),
        (3, r"(?:مش\s*هقدر\s*(?:اجي|احضر)|ما\s*اقدر\s*اجي|لن\s*احضر)"),
        (3, r"\b(?:can(?:'|’)?t|cannot|won(?:'|’)?t)\s+(?:make|come|attend)\b"),
    ],

    "reschedule": [
        (10, r"(?:تاجيل|تأجيل|اجل|أجل|تغيير|غير|اغير|أغير|تعديل|عدل|اعدل|نقل|انقل|قدم|تقديم)\w*\s*"
             r"(?:ال)?(?:حجز|موعد|معاد|ميعاد|الوقت|التاريخ)"),
        (10, r"(?:ال)?(?:حجز|موعد|معاد|ميعاد)\w*\s*"
             r"(?:الى|إلى|ل)\s*(?:يوم|وقت|تاريخ|ميعاد|معاد)\s*(?:تاني|اخر|آخر|ثاني|جديد)"),
        (10, r"\b(?:reschedul|postpon|posptone)\w*"),
        (10, r"\b(?:change|move|shift|push|switch)\b[^.\n]{0,25}\b(?:booking|appointment|slot|time|date)\b"),
        (10, r"\bearlier\s+(?:slot|time|appointment)\b"),
        # Verb + attached object pronoun: "أأجله", "أجلها", "تأجيله",
        # "أغيره", "ننقله" - all one word, all meaning "move it".
        (6, r"(?:^|\s)\w{0,2}(?:اجل|تاجيل|غير|نقل)\w*(?:ه|ها)(?:\s|$)"),
        (6, r"\bnew\s+(?:time|date|slot)\s+for\s+(?:my|the)\s+(?:booking|appointment)\b"),
        # THE BARE WORD, WITH NO OBJECT AFTER IT.
        #
        # `cancel` has had this since the beginning
        # ("(?:الغاء|الغي|ابطل)\w*") and `reschedule` did not, so a bare
        # "تعديل" scored ZERO while a bare "الغاء" scored 6 - even
        # though the clinic's own service menu offers the two side by
        # side, in these exact words: "✏️ تعديل أو إلغاء موعد قائم".
        #
        # CONFIRMED REAL PRODUCTION FAILURE (medtown, session
        # 201003365691+medtown2, 2026-09-08 08:47): the patient typed
        # "تعديل". With no cue at all the conversation stayed on the
        # concierge - which holds every tool but is given none of the
        # cancel/reschedule flow text - and it asked "هذا هو موعدك
        # الذي تبغى تلغيه؟", took the patient's "اه" as consent, and
        # CANCELLED the appointment. The next message was "قولتلك
        # تعديل".
        #
        # Deliberately limited to the words that mean only this:
        # "تعديل"/"تأجيل"/"اعدل"/"عدل"/"أجل". The bare "غير", "قدم" and
        # "نقل" are left out on purpose - each is an ordinary Arabic
        # word ("شيء غير كده", "قدم لي", "نقل الكلام") and would fire on
        # messages that are not about an appointment at all.
        (9, r"(?:^|\s)(?:تعديل|التعديل|تاجيل|التاجيل|اعدل|عدل|اجل)\w*(?:\s|$)"),
    ],

    "booking": [
        (10, r"(?:احجز|أحجز|اححز|حجز|ابغى\s*حجز|عايز\s*احجز|عاوز\s*احجز|ابي\s*احجز|"
             r"حاب\s*احجز|ودي\s*احجز|اريد\s*حجز|نفسي\s*احجز)\w*\s*"
             r"(?:موعد|معاد|ميعاد|كشف|عند|مع|جديد)"),
        (10, r"(?:موعد|معاد|ميعاد|كشف)\s*جديد"),
        (10, r"\bbook\b[^.\n]{0,25}\b(?:appointment|slot|consultation|visit|doctor)\b"),
        (10, r"\bnew\s+(?:appointment|booking)\b"),
        (10, r"\bmake\s+(?:an?\s+)?appointment\b"),
        # A bare "احجز" ("book!") with no object attached is still an
        # unambiguous imperative - not a hint, an instruction. Confirmed
        # real production failure: a patient mid-MEDICAL-flow typed just
        # "احجز" and stayed on MEDICAL anyway because this pattern's old
        # weight (6) sat below _SWITCH_THRESHOLD (8), so nothing about
        # this obviously-clear message could interrupt the active flow.
        # Everything downstream then ran without match_entity_for_booking/
        # list_available_days_for_booking - the exact chain of failures
        # (invented-sounding branch prompts, "pick a day yourself"
        # questions) already fixed once for OTHER booking-intent phrases.
        # Weighted to clear the mid-flow switch threshold on its own.
        (9, r"(?:^|\s)(?:احجز|أحجز|احجزلي|احجز\s*لي|اريد\s*الحجز|عايز\s*حجز|عاوز\s*حجز|ابغى\s*احجز)(?:\s|$)"),
        (9, r"\b(?:booking|reserve|schedule)\s+(?:an?\s+)?(?:appointment|visit|consultation)\b"),
        # Asking WHEN a specific doctor/specialty has an opening ("ايه
        # مواعيد", "اقرب معاد") is a request for a real bookable slot,
        # not small talk - it needs `list_available_days_for_booking`,
        # which only the booking specialist has. Confirmed real
        # production failure: this kept the MEDICAL specialist active
        # (it only has `find_available_doctors`'s coarse hasSlots flag),
        # so the reply could only say "no specific appointment showed
        # up" instead of ever fetching an actual next date - leaving the
        # patient with a branch confirmed and no way to find out when.
        # Weighted to switch even mid-flow (>= _SWITCH_THRESHOLD): a
        # patient who has just been told about a doctor and now asks
        # about appointments has unambiguously moved on to booking.
        # Excludes "مواعيد العمل/الدوام" (opening hours) and "مواعيد
        # الزيارة" (visiting hours), which are FAQ questions, not a
        # request for a bookable slot.
        # (?:ال)? IS LOAD-BEARING: confirmed real production miss -
        # "ايه المواعيد" (with the definite article glued onto the noun,
        # as patients very commonly phrase it) did NOT match this
        # pattern at all when it only accounted for the bare noun
        # "مواعيد", so the router silently kept MEDICAL active on
        # exactly the kind of message this rule exists to catch.
        (8, r"(?:ايه|إيه|في|فيه|عنده|عندها|فاضي|فاضيه)\s*"
            r"(?:ال)?(?:مواعيد|معاد|ميعاد)(?!\s*(?:العمل|الدوام|الزياره))"),
        # THE QUESTION WORD COMES FIRST ONLY SOMETIMES. The pattern
        # above needs "ايه"/"في"/"عنده" in FRONT of "مواعيد", and the
        # commonest phrasing of all puts the doctor there instead:
        # "مواعيد دكتور أمنية" - "Dr Omnia's times". That scored ZERO.
        #
        # CONFIRMED REAL PRODUCTION FAILURE (medtown, session
        # 201099530009+medtown2, 2026-09-08 09:05-09:07): "مواعيد
        # دكتوره أمنيه", asked three times in three different
        # spellings, scored nothing each time, so the turn stayed on
        # the concierge - every tool, none of the booking flow text -
        # and it looped on `get_doctor_schedule` (which takes a BOOKING
        # REFERENCE, not a doctor) until the graph hit its recursion
        # limit and the patient got nothing. Three times.
        #
        # Only `booking` holds `get_doctor_schedule_for_booking`, which
        # is the tool that actually answers this, so this has to reach
        # `booking`. "مواعيد العمل/الدوام/الزيارة" cannot match: a
        # doctor cue word has to follow.
        (8, r"(?:ال)?(?:مواعيد|معاد|ميعاد|جدول|اوقات|أوقات)\s*(?:ال)?"
            # `\w*` on the doctor words on purpose: this fires on a
            # typed WhatsApp message, and the one that started the
            # incident was "مواعيد دكتووة امتية ايه افهم" - two
            # typos in four words. A prefix match costs nothing here:
            # the noun in front of it already fixes the meaning.
            r"(?:د\.|دكتو\w*|طبيب\w*|استشاري\w*|dr\.?|doctor)"),
        (8, r"\b(?:times|schedule|hours|availability)\s+(?:for\s+|of\s+)?"
            r"(?:dr\.?|doctor)\b"),
        (8, r"(?:اقرب|أقرب)\s*(?:معاد|موعد|ميعاد)"),
        (8, r"\b(?:nearest|soonest|earliest)\s+(?:appointment|slot|opening)\b"),
        (8, r"\bwhat\s+(?:appointments|times|slots)\s+(?:are\s+)?(?:available|open)\b"),
        # Same class of miss as the "مواعيد" fix above, different word:
        # asking about the doctor's available DAYS is exactly as much a
        # request for `list_available_days_for_booking` as asking about
        # "مواعيد" is - it's the same tool either way. Confirmed real
        # production failure: "ايه الايام المتاحه" only ever matched the
        # generic weak "متاح" hint below (score 3, nowhere near the
        # mid-flow switch threshold), so MEDICAL stayed active with no
        # way to answer it honestly - it re-ran `list_specialties` and
        # `list_branches_for_specialty` pointlessly (data it already
        # had) and still ended on "هل تحب تحدد موعد في يوم معين؟", the
        # exact "pick a day yourself" anti-pattern the booking flow
        # exists to avoid. Excludes "أيام العمل" (working days), which
        # is an FAQ question, not a request for a bookable slot.
        (8, r"(?:ال)?ايام\s*(?:ال)?متاح[ةه](?!\s*(?:العمل|الدوام))"),
        (3, r"(?:متاح|فاضي|مواعيد\s*متاحه|في\s*مواعيد)"),
        (3, r"\bavailable\s+(?:slots?|times?|appointments?)\b"),
    ],

    "medical": [
        (10, r"(?:عندي|بعاني|اعاني|بحس|احس|حاسس|حاسه|بشتكي|فيني|يوجعني|بيوجعني|بيوجعوني)\s*"
             r"\w*\s*(?:الم|وجع|صداع|حراره|سخونه|مغص|دوخه|دوار|كحه|كحة|سعال|ترجيع|قيء|"
             r"غثيان|اسهال|امساك|حكه|طفح|تعب|ارهاق|ضيق|تنميل|حساسيه)"),
        (10, r"\b(?:i\s+have|i(?:'|’)?ve\s+got|i\s+feel|suffering\s+from|experiencing)\b"
             r"[^.\n]{0,30}\b(?:pain|ache|fever|headache|dizziness|nausea|cough|rash|swelling|"
             r"bleeding|burning|numbness|blurred)\b"),
        (10, r"(?:اي|أي|انهي|أنهي|مين)\s*(?:دكتور|طبيب|تخصص|قسم)\s*(?:يناسب|مناسب|اروح|أروح|اشوف|ازور)"),
        (10, r"\bwhich\s+(?:doctor|specialty|speciality|department)\b"),
        (6, r"(?:^|\s)(?:وجع|الم|ألم|صداع|مغص|دوخه|دوخة|سخونيه|سخونية|حراره|حرارة|كحه|كحة|"
             r"غثيان|قيء|ترجيع|اسهال|امساك|طفح|حكه|حكة|تنميل|ضيق\s*تنفس|نزيف)(?:\s|$)"),
        (6, r"\b(?:pain|ache|fever|headache|migraine|nausea|dizzy|dizziness|cough|rash|"
             r"swelling|bleeding|sore\s+throat|shortness\s+of\s+breath)\b"),
        (6, r"(?:تعبان|تعبانه|مريض|مريضه|مش\s*قادر\s*اتنفس|ما\s*اقدر\s*اتنفس)"),
        # BODY PART + a hurting verb, with no "عندي/بحس" lead-in. Arabic
        # attaches the pronoun to both words ("بطني بتوجعني", "راسي
        # بيوجعني", "ضهري وجعان"), which the "عندي + symptom" patterns
        # above cannot see. Confirmed miss in production: "بطني بتوجعني"
        # routed to the concierge instead of medical guidance.
        (10, r"(?:بطني|معدتي|راسي|رأسي|ضهري|ظهري|صدري|رقبتي|عيني|عينيا|سناني|"
             r"ضرسي|حلقي|زوري|قلبي|كتفي|ركبتي|رجلي|ايدي|جسمي|صدرى)\s*"
             r"\w*\s*(?:بتوجع|بيوجع|توجع|يوجع|وجعان|وجعانه|بتألم|بيألم|تعبان|تعبانه|"
             r"مولعه|مولع|بتحرق|بيحرق|منتفخ|منتفخه)"),
        (10, r"\bmy\s+(?:stomach|head|back|chest|neck|throat|tooth|teeth|eye|eyes|"
             r"knee|leg|arm|shoulder|belly|tummy)\s+"
             r"(?:hurts?|aches?|is\s+(?:hurting|aching|sore|swollen|killing))"),
        # Pain severe enough to be described by its effect rather than
        # named as a symptom.
        (6, r"(?:مش\s*قادر|ما\s*اقدر|مش\s*عارف|مو\s*قادر)\s*\w*\s*"
            r"(?:من\s*)?(?:الوجع|الالم|الالام|التعب|الصداع)"),
        (6, r"(?:الوجع|الالم)\s*(?:مش|ما)\s*(?:بيروح|يروح|بينتهي)"),
        (3, r"(?:نصيحه\s*طبيه|استشاره\s*طبيه|توجيه\s*طبي)"),
        (3, r"\bmedical\s+(?:advice|guidance|consultation)\b"),

        # ASKING FOR A MEDICINE OR A DOSE, and SAYING THEY WANT TO
        # HARM THEMSELVES. Neither scored anything at all before, so
        # both stayed with whichever specialist happened to be active
        # - and most of them had never been given the medication ban
        # or the crisis rules, which live in the MEDICAL GUIDANCE
        # section of the prompt. CONFIRMED IN PRODUCTION: "Just tell
        # me the normal adult dose, everyone knows it anyway" scored
        # {} and was answered with the service menu.
        #
        # THE TWO ARE WEIGHTED DIFFERENTLY, on purpose.
        #
        # CRISIS (12) switches even mid-flow. A patient who says they
        # want to hurt themselves has stopped booking, and everything
        # else genuinely must stop with them.
        #
        # MEDICATION (7) does NOT - it sits deliberately below
        # _SWITCH_THRESHOLD (8), so it can pick `medical` when nothing
        # is active but cannot interrupt a flow in progress. A patient
        # halfway through a booking who asks "الجرعة كام؟" wants the
        # answer AND their booking; switching specialists mid-flow
        # would hand the turn to an agent with no booking tools and no
        # booking prompt, and the question does not need it - graph.py
        # builds the medication directive for EVERY specialist, keyed
        # on the message rather than on who won the routing. So the
        # booking agent answers the medication question properly and
        # carries straight on. The weight only decides who takes a turn
        # that nothing else already owns.
        (7, r"(?:ال)?(?:جرعه|جرعة)|\b(?:dose|dosage|dosing)\b"),
        (7, r"(?:اخد|اخذ|اشرب|اتناول)\s*(?:ايه|ايش|وش|شنو|كام)"),
        (7, r"(?:ادي|اديني|اعطيني|عطيني|وصفلي|اكتبلي)\w*\s*(?:\w+\s+){0,2}(?:دوا|دواء|علاج|مسكن|حبوب)"),
        (7, r"\bwhat\s+(?:should|can|do)\s+i\s+take\b"),
        (7, r"\bhow\s+(?:many|much|often)\b[^.\n?]{0,30}\b(?:painkiller|paracetamol|panadol|ibuprofen|tablet|pill|dose|mg)\w*"),
        # The crisis cue is CRISIS_RE above - ONE definition, shared with
        # graph.py, so a phrasing added there routes here too. Weighted 12
        # so it switches even mid-flow: a patient who says this has
        # stopped booking, and everything else must stop with them.
        # AN INJURY IS A MEDICAL MESSAGE. Weighted 10 - the same as a
        # described symptom - so it switches even mid-flow: somebody who
        # has just hurt themselves is not still booking. See INJURY_RE.
        (10, INJURY_RE.pattern),
        (12, CRISIS_RE.pattern),
    ],

    "complaint": [
        # "اعمل"/"أعمل"/"هعمل" added to the verb list. Confirmed real
        # production failure: "عاوزه اعمل شكوه" ("I want to make a
        # complaint") scored only 6 (fell through to the bare-keyword
        # rule below) because "اعمل" wasn't one of the recognised verbs -
        # 6 sits under _SWITCH_THRESHOLD (8), so this message could not
        # interrupt an active booking flow the session was already in,
        # and the patient's stated complaint intent was silently ignored
        # in favour of "still owns this flow".
        (10, r"(?:عندي|اقدم|أقدم|ابغى\s*اقدم|عايز\s*اقدم|حاب\s*اقدم|اريد\s*تقديم|بدي\s*قدم|"
             r"اعمل|أعمل|هعمل|عاوز(?:ه)?\s*اعمل|عايز(?:ه)?\s*اعمل)\s*"
             r"\w*\s*(?:شكوى|شكويه|شكوه|شكاوي|اقتراح|مقترح|ملاحظه|ملاحظة)"),
        (10, r"\b(?:file|submit|make|raise|lodge|register)\s+(?:an?\s+)?"
             r"(?:complaint|grievance|suggestion|feedback)\b"),
        # Describing a doctor having made a medical ERROR is a complaint
        # even with no "شكوى"/"اشتكي" wording at all - "وصفلي دواء غلط"
        # ("prescribed me the wrong medicine") is unambiguously about
        # something a doctor did wrong, not a booking or medical-
        # guidance request. Confirmed real production failure: a
        # complaint that opened this way never scored high enough to
        # switch away from whichever specialist was already active, so
        # a specialist with no `send_complaint_email` tool improvised
        # the entire complaint flow itself and eventually told the
        # patient their complaint was filed when nothing was ever sent.
        (9, r"(?:وصف|كتب|اداني|اعطاني)\w*\s*(?:لي|لى)?\s*دواء\s*(?:غلط|خطأ|خطا)"),
        (9, r"(?:غلط|خطأ|خطا)\s*(?:طبي|في\s*العلاج|في\s*التشخيص|في\s*الدواء|في\s*الوصفه)"),
        (9, r"(?:الدكتور|الطبيب|دكتور|طبيب)\w{0,10}\s*غلط\w*"),
        # BARE "شكوى"/"شكوي"/"شكوه" etc. RAISED FROM 6 TO 8.
        #
        # Same reasoning as the bare "احجز" imperative fix in the
        # "booking" cue list above: a patient who types the word
        # "complaint" on its own, with no other verb attached, has
        # still stated an unambiguous topic - there is no other plausible
        # reading of "شكوي" sent by itself. Leaving this at 6 (below
        # _SWITCH_THRESHOLD=8) meant it could never interrupt an active
        # flow on its own. CONFIRMED REAL PRODUCTION FAILURE: this
        # message, sent while "booking" already owned the conversation,
        # scored 6, stayed on "booking" ("booking still owns this
        # flow"), and the specialist that then ran had no dedicated
        # complaint-flow prompt shaping its reply. Weighted to 8 so a
        # standalone complaint word can now switch specialists on its
        # own, exactly like a standalone "احجز" can for booking.
        (8, r"(?:^|\s)(?:شكوى|شكويه|شكوه|شكاوي|اشتكي|أشتكي|هشتكي|هاشتكي|حاشتكي|"
             r"بشتكي\s*من|اقتراح|مقترح)(?:\s|$)"),
        (8, r"\b(?:complaint|complain|grievance|suggestion)\b"),
        (3, r"(?:خدمه\s*سيئه|معامله\s*سيئه|مستاء|مستاءه|زعلان\s*من|غير\s*راضي)"),
        (3, r"\b(?:poor|bad|terrible|awful)\s+(?:service|treatment|experience)\b"),
        (3, r"\bunhappy\s+with\b"),
    ],

    "faq": [
        (10, r"(?:فين|وين|اين|أين|ايه\s*عنوان|ما\s*هو\s*عنوان)\s*\w*\s*(?:الفرع|المستشفى|العياده|العيادة)"),
        (10, r"\b(?:where\s+is|what(?:'|’)?s\s+the\s+address\s+of)\b[^.\n]{0,25}"
             r"\b(?:branch|hospital|clinic)\b"),
        (10, r"(?:ايه|إيه|ما\s*هي|شنو|وش)\s*(?:هي\s*)?(?:ال)?(?:خدمات|تخصصات)(?:كم|\s+\S+)?"),
        (10, r"\bwhat\s+(?:services|specialt(?:y|ies)|departments)\b"),
        # Unambiguous enough to interrupt another flow: nobody asks
        # about opening hours as part of confirming a cancellation.
        (10, r"(?:مواعيد\s*العمل|ساعات\s*العمل|متى\s*تفتحون|امتى\s*بتفتحوا)"),
        (10, r"\b(?:opening|working)\s+hours\b"),
        (6, r"(?:بتفتحوا|بتقفلوا|رقم\s*التواصل|رقم\s*الهاتف\s*للمستشفى|"
             r"الرؤيه|الرساله|القيم|التامين)"),
        (6, r"\bcontact\s+number\b|\bvision\s+and\s+mission\b|"
            r"\binsurance\b|\bpolic(?:y|ies)\b"),
        (3, r"(?:عن\s*المستشفى|معلومات\s*عن|فروعكم|كام\s*فرع)"),
        (3, r"\babout\s+the\s+hospital\b|\byour\s+branches\b"),
    ],

    CONCIERGE: [
        # A human handoff request always comes back to the concierge -
        # it is not a flow, it is an exit.
        (10, r"(?:^|\s)(?:موظف|موظفه|بشري|انسان|إنسان|احد\s*من\s*الموظفين|خدمه\s*العملاء|"
             r"خدمة\s*العملاء|ممثل|حولني|حوليني|كلمني\s*مع\s*حد)(?:\s|$)"),
        (10, r"\b(?:human|agent|representative|customer\s+service|real\s+person|"
             r"speak\s+to\s+(?:someone|staff))\b"),
    ],
}


def _fold_arabic(text: str) -> str:
    """The letter-folding half of `normalize()`.

    Applied to the CUE PATTERNS as well as to messages, so a cue can be
    written the natural way ("شكوى", "أبغى", "دوخة") and still match a
    normalised message where those became "شكوي", "ابغي", "دوخه".
    Getting this wrong is silent - the cue simply never fires - so it is
    done mechanically rather than by remembering to spell every cue in
    its folded form.

    Deliberately does NOT lower-case, collapse whitespace or convert
    digits: those steps are safe on a message but would corrupt regex
    metacharacters in a pattern.
    """

    result = _DIACRITICS_RE.sub("", text)
    result = _ALEF_RE.sub("ا", result)
    return (result.replace("ى", "ي").replace("ة", "ه")
                  .replace("ؤ", "و").replace("ئ", "ي"))


_COMPILED: Dict[str, List[Tuple[int, "re.Pattern"]]] = {
    agent: [(weight, re.compile(_fold_arabic(pattern))) for weight, pattern in cues]
    for agent, cues in _CUES.items()
}


# ==========================================================
# Normalisation
# ==========================================================

def normalize(text: str) -> str:
    """Folds the spelling variations Arabic users actually type, so one
    cue pattern covers "أبغى"/"ابغي"/"ابغى" instead of three."""

    if not text:
        return ""

    result = _fold_arabic(text.translate(_DIGIT_MAP))
    result = _WHITESPACE_RE.sub(" ", result)
    return result.strip().lower()


# ==========================================================
# Scoring
# ==========================================================

def score_message(text: str) -> Dict[str, int]:
    """Returns {agent: score} for one message. Exposed for the tests and
    for anyone tuning the cue lists."""

    normalized = normalize(text)
    if not normalized:
        return {}

    scores: Dict[str, int] = {}

    for agent, cues in _COMPILED.items():
        matched = [weight for weight, pattern in cues if pattern.search(normalized)]
        if matched:
            # The strongest cue decides, with a small bonus for each
            # corroborating one - so "عايز ألغي الحجز" (phrase + keyword)
            # outranks a bare "إلغاء" appearing in passing.
            scores[agent] = max(matched) + (len(matched) - 1)

    return scores


def _best(scores: Dict[str, int]) -> Tuple[Optional[str], int]:
    if not scores:
        return None, 0
    agent = max(scores, key=lambda key: (scores[key], key))
    return agent, scores[agent]


# ==========================================================
# Flow completion (releases stickiness)
# ==========================================================

# When one of these has just succeeded, the flow that owned the
# conversation is over. The next message that carries no cue of its own
# goes back to the concierge instead of being glued to a finished flow.
_TERMINAL_TOOLS = {
    "cancel_appointment": ("success",),
    "reschedule_appointment": ("success",),
    "create_new_booking": ("success",),
    "send_complaint_email": ("sent",),
}


def _flow_just_completed(messages: List) -> bool:
    """Did the PREVIOUS turn finish its flow?

    The message being routed is itself the newest human message, so the
    scan starts just above it and stops at the human turn before that -
    i.e. it looks at exactly one completed turn, never further back. A
    cancellation that succeeded ten turns ago must not keep releasing
    the conversation forever.
    """

    history = list(messages or [])

    last_human = None
    for index in range(len(history) - 1, -1, -1):
        if getattr(history[index], "type", None) == "human":
            last_human = index
            break

    if last_human is None:
        return False

    for message in reversed(history[:last_human]):
        if getattr(message, "type", None) == "human":
            return False

        if getattr(message, "type", None) != "tool":
            continue

        name = getattr(message, "name", "") or ""
        statuses = _TERMINAL_TOOLS.get(name)
        if not statuses:
            continue

        content = str(getattr(message, "content", "") or "")
        if any(f'"status": "{status}"' in content or f"'status': '{status}'" in content
               for status in statuses):
            return True

    return False


def _latest_human_text(messages: List) -> str:
    for message in reversed(messages or []):
        if getattr(message, "type", None) == "human":
            content = getattr(message, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""


# ==========================================================
# "Yes, book it" - a bare affirmation replying to a booking offer
# ==========================================================
#
# CONFIRMED REAL PRODUCTION FAILURE: the MEDICAL specialist asked
# "تبغى أحجز لك عند د. طه مبروك؟" (do you want me to book you with
# Dr. X?) and the patient replied "اه" (yeah). "اه" alone carries no
# cue at all, so the normal stickiness rule kept MEDICAL active - which
# does not have match_entity_for_booking/list_available_days_for_booking,
# only the coarse find_available_doctors/list_branches_for_specialty.
# With no tool to get the doctor's REAL branches or REAL soonest day,
# the model started inventing branch names from memory and asking "pick
# a day yourself" instead of showing one - two separate confirmed
# failures, both downstream of staying in the wrong specialist.
#
# A bare "yes" is not itself a cue for anything - "اه" said mid-medical-
# guidance about a symptom is not a booking signal. It only means
# "start booking" when the ASSISTANT'S OWN PREVIOUS MESSAGE just offered
# to book something. That is a narrow, safe trigger: it only fires on
# the exact turn right after the assistant asked a booking question.

_BARE_AFFIRMATION_RE = re.compile(
    r"^(?:اه+|ايوه|ايوا|ايه|نعم|تمام|ماشي|اكيد|أكيد|تمام\s*كده|"
    r"يلا|يلا\s*بينا|كده\s*تمام|حاضر|okay|ok|yes|yep|yeah|sure)[\s!.،,؟?]*$"
)

# \w*[^.\n؟?]{0,20}?عند IS LOAD-BEARING: the earlier version required
# "لك" to be immediately followed by "عند" ("أحجز لك عند"), so a
# perfectly natural rephrasing with a word in between - "تحب أحجز لك
# موعد عنده؟" ("would you like me to book you an appointment with
# him?") - never matched at all. Confirmed real production failure:
# the patient said "اه" to exactly that offer, the affirmation-override
# above never fired because THIS regex missed it, and the conversation
# stayed on MEDICAL - which then invented branch names again for the
# same reason as before. The bounded gap (up to ~20 chars, no sentence/
# question-mark boundary crossed) catches "لك موعد عند"/"لك كشف عند"
# and similar short insertions without matching across unrelated
# sentences.
_PREVIOUS_REPLY_OFFERED_BOOKING_RE = re.compile(
    r"(?:تحجز|أحجز|احجز|نحجز)\w*[^.\n؟?]{0,20}?عند|"
    r"نكمل\s*الحجز|تحب\w*\s*تحجز|حاب\w*\s*تحجز|"
    r"تبغى\s*أحجز|تبي\s*أحجز|ابدأ\s*الحجز|أبدأ\s*بالحجز"
)


def _last_ai_text(messages: List) -> str:
    """The assistant's own most recent reply, searched from just before
    the newest human message backward - i.e. "what did the bot just
    say" from the patient's point of view this turn."""

    history = list(messages or [])
    last_human = None
    for index in range(len(history) - 1, -1, -1):
        if getattr(history[index], "type", None) == "human":
            last_human = index
            break

    search_from = history[:last_human] if last_human is not None else history
    for message in reversed(search_from):
        if getattr(message, "type", None) == "ai":
            content = getattr(message, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""


_POSITIONAL_PICK_RE = re.compile(r"^\s*(?:رقم\s*)?([1-9]\d?|[١-٩]\d?)\s*[.!؟?،,]*\s*$")


# WHICH AGENTS CAN SHOW A BOOKABLE LIST BUT NOT ACT ON A PICK.
#
# Computed from the registry rather than written out, so it stays true
# when an agent's tool set changes. An agent that cannot reach the slot
# and booking tools physically cannot take the step after "1" - leaving
# the patient there strands the flow (see the routing rule that uses
# this).
def _agents_without_booking_tools() -> frozenset:
    from agents.registry import AGENT_SPECS, CONCIERGE

    # The two tools nothing can finish a booking without.
    required = {"get_available_slots_for_booking", "create_new_booking"}

    # ...and the tools that PRINT a bookable list in the first place.
    # Both halves matter: `cancel` and `reschedule` also lack the
    # booking tools, but they never show a specialty or doctor roster -
    # a number from them is an appointment or a slot, and handing that
    # to `booking` would break the flow it belongs to.
    shows_lists = {"list_specialties", "find_available_doctors"}

    stranded = set()
    for name, spec in AGENT_SPECS.items():
        if name == CONCIERGE or spec.full_access or spec.full_tools:
            continue  # has everything
        held = {getattr(t, "name", "") for t in spec.tools()}
        if required.issubset(held):
            continue                      # it can finish the booking itself
        if not (held & shows_lists):
            continue                      # it never shows such a list
        stranded.add(name)

    return frozenset(stranded)


def _picks_from_a_specialty_list(messages: List, text: str) -> bool:
    """True when the patient is choosing a SPECIALTY from a numbered
    list the assistant just showed.

    The specialty counterpart of `_picks_from_a_doctor_list`, and it
    matters for the same reason: choosing a specialty is the first step
    of a BOOKING, and the agents that can print the specialty list
    (`medical`, `faq`) own none of the tools that come after it.

    CONFIRMED REAL PRODUCTION FAILURE (medtown, 2026-09-07 13:03): the
    specialty list was shown by `faq`, the patient replied "1", and the
    flow never left `faq` - so instead of the dentists it showed the
    clinic's branches, then that branch's service catalogue.
    """

    if not _POSITIONAL_PICK_RE.match(text.strip()):
        return False

    for msg in reversed(messages or []):
        content = getattr(msg, "content", "")
        content = content if isinstance(content, str) else str(content)
        if getattr(msg, "type", None) == "human" or not content.strip():
            continue
        if getattr(msg, "type", None) != "ai":
            continue
        looks_like_specialty_list = (
            ("تخصص" in content or "التخصصات" in content or "specialt" in content.lower())
            and re.search(r"[1-9]️?⃣", content) is not None
        )
        return bool(looks_like_specialty_list)

    return False


def _picks_from_a_doctor_list(messages: List, text: str) -> bool:
    """True when the patient is choosing a doctor from a numbered list
    the assistant just showed.

    WHY THIS MATTERS FOR ROUTING: the `medical` agent can SHOW doctors
    (it has `find_available_doctors`) but has no booking tools at all -
    no `match_entity_for_booking`, no schedule, no slots. So the moment
    the patient picks one, that agent physically cannot take the next
    step.

    CONFIRMED REAL PRODUCTION FAILURE: medical guidance listed two
    doctors, the patient replied "1", the router's default rule kept
    them in `medical`, and the turn span the agent->tools cycle until it
    hit the step ceiling - the patient got the technical-failure message
    after picking a doctor that existed and was available.
    """

    if not _POSITIONAL_PICK_RE.match(text.strip()):
        return False

    # Only when the previous assistant message actually presented a
    # numbered list of doctors - a number answering something else
    # (a day, a time, a branch) is not a doctor pick.
    for msg in reversed(messages or []):
        content = getattr(msg, "content", "")
        content = content if isinstance(content, str) else str(content)
        if getattr(msg, "type", None) == "human" or not content.strip():
            continue
        if getattr(msg, "type", None) != "ai":
            continue
        lowered = content
        looks_like_doctor_list = (
            ("د." in lowered or "دكتور" in lowered or "الأطباء" in lowered
             or "الاطباء" in lowered or "الدكاتره" in lowered or "الدكاترة" in lowered)
            and re.search(r"[1-9]\uFE0F?\u20E3", lowered)
        )
        return bool(looks_like_doctor_list)

    return False


# ==========================================================
# ANSWERING THE BOOKING FLOW'S OWN OPENING QUESTION
# ==========================================================
#
# The booking flow opens by asking whether to start from a doctor or
# from a specialty. Patients very often answer it with neither - they
# answer with WHAT IS WRONG ("بطني وجعاني وعندي ألم شديد"), because
# that is the thing they actually know. STEP NB1 covers this exactly:
# match the symptom to the closest specialty yourself and carry on to
# the doctors, with no clarifying questions and no comfort tips,
# BECAUSE THIS IS THE BOOKING FLOW.
#
# But that message scores 12 on the medical cues (body part + hurting
# verb), which is above _SWITCH_THRESHOLD, so the supervisor moved the
# conversation to `medical` - and `medical` correctly ran ITS flow:
# comfort measures, red flags, the ⚕️ disclaimer, then an offer to
# book. The patient had already asked to book, one message earlier.
#
# CONFIRMED IN A REAL CONVERSATION: "عاوزه احجز معاد" -> "تحب تبدأ
# بالتخصص ولا بالدكتور؟" -> "بطني وجعاني وعندي الم شديد" -> a full
# medical-guidance reply, and only after another "اه" did the doctor
# list finally appear. Three turns to do what NB1 does in one.
#
# So: a message that ANSWERS the booking flow's own opening question
# belongs to booking, whatever it scores. Narrow by construction - it
# only fires on the single turn immediately after that exact question.

_ASKED_SPECIALTY_OR_DOCTOR_RE = re.compile(
    # NOTE: these alternations are in FOLDED form ("ام", not "أم").
    # The text is normalised before matching - alef forms are unified -
    # but this pattern is compiled raw, so a hamza written here would
    # never match anything. That is a silent failure: the question
    # simply stops being recognised.
    r"بالتخصص\s*ولا\s*بالدكتور|بالدكتور\s*ولا\s*بالتخصص|"
    # "دكتور أو تخصص معيّن في بالك؟" - the current wording, which
    # puts the two options side by side instead of contrasting them
    # with "ولا". Added when the message changed; without it the
    # question stops being recognised and the routing rule that
    # keeps a symptom with `booking` silently stops firing.
    r"(?:دكتور|طبيب)\s*(?:او|ولا|ام)\s*تخصص|"
    r"تخصص\s*(?:او|ولا|ام)\s*(?:دكتور|طبيب)|"
    r"تبدا\s*بالتخصص|تبدا\s*بالدكتور|"
    r"طبيب\s*(?:معين|محدد)[^.\n؟?]{0,40}(?:ولا|او|ام)[^.\n؟?]{0,20}تخصص|"
    r"تخصص[^.\n؟?]{0,40}(?:ولا|او|ام)[^.\n؟?]{0,20}(?:طبيب|دكتور)|"
    r"specific\s+doctor[^.\n?]{0,40}(?:or|prefer)[^.\n?]{0,25}specialt|"
    r"by\s+specialty\s+or\s+by\s+doctor"
)

_CRISIS_OVERRIDE_RE = CRISIS_RE


# ASKING TO SEE THE OPTIONS IS AN ANSWER, NOT A CHANGE OF SUBJECT.
#
# The booking flow's own opening question is "عندك دكتور أو تخصص معيّن
# في بالك؟". A patient who replies "طيب إيه التخصصات؟" is answering it -
# they are saying "show me, then I will pick". It is the single most
# natural reply to that question.
#
# But "ايه التخصصات الموجوده" scores faq:10 - identically to "فين عنوان
# الفرع؟" - and the change-of-subject rule below reads any non-booking
# score over the threshold as a deliberate switch. So the most natural
# answer to the booking flow's own question was routed out of booking.
#
# CONFIRMED REAL PRODUCTION FAILURE (medtown, session
# 201158877175+medtown2, 2026-09-07 13:02-13:09): "عاوزه احجز" -> the
# entry question -> "ايه التخصصات؟" went to `faq`, and stayed there for
# the rest of the flow. `faq` owns no booking tool, so the patient was
# shown branches, then a branch's service catalogue, and finally a dead
# end - never a dentist.
#
# `booking` needs no help answering this: it holds `list_specialties`
# and `find_available_doctors` itself, shows the list, and carries
# straight on to the doctors.
_ASKS_TO_SEE_THE_OPTIONS_RE = re.compile(
    r"(?:ايه|ايش|وش|ما\s*هي|ماهي|انهي|اي)\s*(?:ال)?(?:تخصصات|تخصص|دكاتره|اطباء)|"
    r"(?:اعرض|وريني|ورينى|شوفني|عايز\s*اشوف|عاوز\s*اشوف|ابغى\s*اشوف|تقدر\s*تعرض)"
    r"[^\n]{0,15}(?:ال)?(?:تخصصات|دكاتره|اطباء)|"
    r"(?:مين|من)\s*(?:هم\s*)?(?:ال)?(?:دكاتره|اطباء)|"
    r"(?:ال)?(?:تخصصات|دكاتره|اطباء)\s*(?:ال)?(?:متاح|موجود)|"
    r"(?:what|which|show\s*me\s*the)\s*(?:specialt|doctor)"
)


def _answers_booking_entry_question(messages: List, text: str) -> bool:
    """True when the assistant's own previous reply was the booking
    flow's opening doctor-or-specialty question - AND the answer is one
    the booking specialist can actually act on.

    A SYMPTOM OR AN INJURY IS EXCLUDED, deliberately.

    The entry question invites three kinds of answer: a doctor's name, a
    specialty, or a description of what is wrong. The first two belong
    to `booking` and this rule keeps them there. The third does not:
    `booking` is given the NEW BOOKING FLOW and DOCTOR/BRANCH INFO
    sections only (see agents/registry.py) - it has no medical-guidance
    prompt at all, so it has nothing to say to somebody describing an
    injury, and what it actually produced was a refusal.

    CONFIRMED: with this rule holding the turn, "عايز أحجز موعد" ->
    the entry question -> "رجلي وقعت عليها" came back "عذرًا، ما
    عندي...". Routed to `medical` instead, the same message produces
    the comfort line, the red flags, the ⚕️ notice and an offer of
    orthopedics doctors - and the booking then resumes at the doctor
    list by itself, because `_build_established_specialty_directive`
    picks the specialty up from the tool result.

    So a health message falls through to normal scoring, where the
    medical cues carry it to the specialist that owns it."""

    if _CRISIS_OVERRIDE_RE.search(normalize(text)):
        return False

    if looks_like_health_message(text):
        return False

    # A DELIBERATE REQUEST FOR A DIFFERENT FLOW IS NOT AN ANSWER TO THIS
    # QUESTION.
    #
    # "عاوزه اعدل معاد" scores 10 for reschedule - well over
    # _SWITCH_THRESHOLD - and is nobody's idea of an answer to
    # "doctor or specialty?". This rule was written to keep a specialty,
    # a doctor's name or a symptom with `booking`; it was never meant to
    # outrank an explicit change of subject, and doing so is
    # unrecoverable rather than merely clumsy: `booking` is deliberately
    # not given `lookup_appointment` (see agents/registry.py), so once a
    # reschedule lands there the one action the patient needs cannot be
    # taken at all.
    #
    # CONFIRMED REAL PRODUCTION FAILURE (medtown, session
    # 201158877175+medtown2, 2026-09-06 14:08-14:09): "عاوزه اعدل معاد"
    # stayed with `booking`, which then asked for the booking reference,
    # was told "نفس رقم واتساب", asked again, reached for
    # `get_patient_info` (the NEW-booking tool, which refused), and
    # finally asked "أي دكتور أو تخصص حابة تعدلي موعدك عنده؟" - the
    # booking entry question - to somebody who had asked three times to
    # change an appointment that already exists.
    # See `_ASKS_TO_SEE_THE_OPTIONS_RE`: this one phrasing scores as an
    # information question while being the most natural ANSWER to the
    # question that was just asked.
    asks_to_see_options = bool(_ASKS_TO_SEE_THE_OPTIONS_RE.search(normalize(text)))

    scores = score_message(text)
    for flow, score in scores.items():
        if flow != "booking" and score >= _SWITCH_THRESHOLD:
            # NARROW ON PURPOSE - only the info flow, and only for this
            # phrasing. An explicit cancel/reschedule/complaint request
            # is still a real change of subject and still bails out.
            if flow == "faq" and asks_to_see_options:
                continue
            return False

    last_ai = _last_ai_text(messages)
    if not last_ai:
        return False

    if not _ASKED_SPECIALTY_OR_DOCTOR_RE.search(normalize(last_ai)):
        return False

    # THE MATCH HAS TO BE A QUESTION, NOT THE WELCOME MENU.
    #
    # The clinic's opening greeting lists what the assistant can do, and
    # one of those lines is "🩺 التوجيه الطبي لاختيار التخصص أو الطبيب
    # المناسب". Normalised, that contains "تخصص او طبيب" and matches
    # `_ASKED_SPECIALTY_OR_DOCTOR_RE` exactly - so for the whole of any
    # turn whose reply carried the greeting, every following message was
    # read as an answer to a question that was never asked, and pinned
    # to `booking`.
    #
    # The real entry question is interrogative; a menu bullet describing
    # a service is not. Requiring the matched LINE to carry a question
    # mark separates them cleanly and costs nothing - the authored entry
    # question ends in "؟".
    for line in last_ai.replace("\r", "\n").split("\n"):
        if not _ASKED_SPECIALTY_OR_DOCTOR_RE.search(normalize(line)):
            continue
        if "؟" in line or "?" in line:
            return True

    return False


def _affirms_previous_booking_offer(messages: List, text: str) -> bool:
    if not _BARE_AFFIRMATION_RE.match(normalize(text)):
        return False
    last_ai = _last_ai_text(messages)
    return bool(last_ai) and bool(_PREVIOUS_REPLY_OFFERED_BOOKING_RE.search(last_ai))


# ==========================================================
# The routing decision
# ==========================================================

_CANNOT_COMPLETE_A_BOOKING_CACHE = None


class _StrandedAgents:
    """`in` support with the registry consulted lazily, once."""

    def __contains__(self, name):
        global _CANNOT_COMPLETE_A_BOOKING_CACHE
        if _CANNOT_COMPLETE_A_BOOKING_CACHE is None:
            try:
                _CANNOT_COMPLETE_A_BOOKING_CACHE = _agents_without_booking_tools()
            except Exception:  # pragma: no cover - never break routing
                logger.warning(
                    "router: could not work out which agents lack booking tools - "
                    "falling back to `medical` only", exc_info=True,
                )
                _CANNOT_COMPLETE_A_BOOKING_CACHE = frozenset({"medical"})
            logger.info(
                "router: agents that cannot complete a booking: %s",
                sorted(_CANNOT_COMPLETE_A_BOOKING_CACHE),
            )
        return name in _CANNOT_COMPLETE_A_BOOKING_CACHE


_CANNOT_COMPLETE_A_BOOKING = _StrandedAgents()


def route_turn(messages: List, active_agent: Optional[str] = None) -> Tuple[str, str]:
    """
    Returns `(agent_name, reason)`. The reason is logged, never shown to
    the patient.

    The rules, in the order they are applied:

      1. No message to read      -> keep the active specialist.
      2. A bare "yes" answering
         the assistant's own
         previous booking offer  -> switch straight to booking, even
                                    with zero textual cue of its own.
      3. Strong cue for someone
         other than the active
         specialist              -> switch (a deliberate change of
                                    subject, e.g. "خلاص ألغيه بقى" while
                                    booking).
      4. Nothing active yet      -> the best cue above the start
                                    threshold, otherwise the concierge.
      5. The active flow just
         completed AND this
         message has no cue      -> back to the concierge.
      6. Anything else           -> keep the active specialist. This is
                                    the case that covers "نعم", an OTP,
                                    a phone number, a menu number, a
                                    weekday - i.e. most of a real
                                    conversation.
    """

    if active_agent not in AGENT_NAMES:
        active_agent = None

    text = _latest_human_text(messages)

    if not text.strip():
        return (active_agent or CONCIERGE), "no user message - kept current specialist"

    if active_agent != "booking" and _affirms_previous_booking_offer(messages, text):
        return "booking", "bare affirmation answering the assistant's own booking offer"

    # Picking a DOCTOR or a SPECIALTY out of a list is a BOOKING action,
    # wherever the list was shown.
    #
    # THE GATE IS DERIVED, NOT HARDCODED. This used to read
    # `active_agent == "medical"`, while its own comment said "wherever
    # the list was shown" - and `faq` shows exactly the same lists
    # (`list_specialties`, `find_available_doctors`,
    # `list_branches_for_specialty`) while owning no booking tool at
    # all. `_CANNOT_COMPLETE_A_BOOKING` is computed from the registry,
    # so an agent that gains or loses booking tools cannot silently
    # reintroduce this.
    #
    # CONFIRMED REAL PRODUCTION FAILURE (medtown, session
    # 201158877175+medtown2, 2026-09-07 13:03-13:09): "ايه التخصصات؟"
    # routed to `faq`, the specialty list was shown, and the patient's
    # "1" kept them on `faq` for the entire rest of the flow. Lacking
    # `find_available_doctors`'s booking siblings it showed BRANCHES,
    # then branch SERVICES, and finally dead-ended on "لازم تختار أولاً
    # اليوم" with no way to list the days.
    if (active_agent in _CANNOT_COMPLETE_A_BOOKING
            and (_picks_from_a_doctor_list(messages, text)
                 or _picks_from_a_specialty_list(messages, text))):
        return "booking", (
            "picked from a doctor/specialty list - booking owns the next step "
            f"({active_agent} has no booking tools)"
        )

    # A symptom, a specialty or a doctor name given in answer to the
    # booking flow's own opening question is a BOOKING answer - see
    # _answers_booking_entry_question. Checked before the scores,
    # because the whole point is that the score is misleading here.
    if active_agent in ("booking", "concierge") and _answers_booking_entry_question(messages, text):
        return "booking", "answered the booking flow's own doctor-or-specialty question"

    scores = score_message(text)
    candidate, score = _best(scores)

    if config.ROUTER_MODE == "llm" and score < _START_THRESHOLD:
        llm_choice = _classify_with_llm(text, active_agent)
        if llm_choice:
            return llm_choice, "llm router (message had no deterministic cue)"

    if candidate and score >= _SWITCH_THRESHOLD and candidate != active_agent:
        return candidate, f"strong cue for {candidate} (score {score})"

    # THE CONCIERGE IS NOT A FLOW, SO THERE IS NOTHING TO INTERRUPT.
    #
    # `None` and `CONCIERGE` are the same situation as far as this
    # decision goes: no specialist has taken the conversation yet. The
    # weak-cue rule further down exists to stop "متاح إمتى؟" hijacking
    # a booking already in progress - it was never meant to keep a
    # message with a real cue sitting on the fallback.
    #
    # CONFIRMED REAL PRODUCTION FAILURE (medtown, session
    # 201003365691+medtown2, 2026-09-08 08:47): "تعديل" after the
    # greeting stayed on `concierge`, which has every tool and none of
    # the cancel/reschedule flow text, and it cancelled the
    # appointment. A bare "الغاء" (score 6) had exactly the same hole:
    # strong enough to name the right specialist, not strong enough to
    # get off the concierge, because the concierge counted as an active
    # flow.
    if active_agent is None or active_agent == CONCIERGE:
        if candidate and score >= _START_THRESHOLD:
            return candidate, (
                f"opening cue for {candidate} (score {score})"
                if active_agent is None else
                f"concierge holds no flow - {candidate} takes it (score {score})"
            )
        if active_agent is None:
            return CONCIERGE, "no clear intent yet - concierge opens the conversation"

    if _flow_just_completed(messages):
        # The previous flow finished, so nothing is being interrupted -
        # a WEAK cue is enough to start the next one. Booking a slot and
        # then saying "طيب ممكن أأجله؟" is the obvious case: mid-flow
        # that hint would rightly be ignored, but right after a
        # completed booking it is plainly a new request.
        if candidate and score >= _START_THRESHOLD:
            return candidate, f"{active_agent} completed - {candidate} takes over (score {score})"
        if not scores:
            return CONCIERGE, f"{active_agent} flow completed - released to concierge"

    if candidate and candidate != active_agent and score >= _START_THRESHOLD:
        # A weak hint mid-flow is usually part of the flow itself
        # ("متاح إمتى؟" while booking is a booking question, not a new
        # intent), so it does NOT move the conversation.
        logger.debug(
            "router: weak cue for %s (score %d) ignored - %s still owns this flow",
            candidate, score, active_agent,
        )

    return active_agent, f"{active_agent} still owns this flow"


# ==========================================================
# Optional LLM routing (ROUTER_MODE=llm)
# ==========================================================

_LLM_ROUTER_PROMPT = """You classify one patient message for a hospital assistant.

Reply with EXACTLY ONE of these words and nothing else:
cancel      - wants to cancel an existing appointment
reschedule  - wants to move an existing appointment to another time
booking     - wants a brand new appointment
medical     - describes a symptom or asks which doctor/specialty they need
faq         - asks about the hospital: services, branches, hours, policies
complaint   - wants to file a complaint or a suggestion
concierge   - anything else, a greeting, or unclear

Currently active flow: {active}
Patient message: {message}"""


def _classify_with_llm(text: str, active_agent: Optional[str]) -> Optional[str]:
    """Only reached when ROUTER_MODE=llm AND the deterministic cues found
    nothing. Any failure returns None and the deterministic path
    continues - routing must never be able to break a conversation."""

    try:
        from langchain_core.messages import HumanMessage
        import graph  # imported lazily: graph imports this package

        # The router's OWN binding, which fails fast - see
        # graph._router_llm. Falls back to the main model only if that
        # attribute is missing (an older graph.py), never preferring it.
        llm = getattr(graph, "_router_llm", None) or getattr(graph, "_llm", None)
        if llm is None:
            return None

        prompt = _LLM_ROUTER_PROMPT.format(
            active=active_agent or "none", message=text[:500],
        )
        answer = llm.invoke([HumanMessage(content=prompt)])
        choice = str(getattr(answer, "content", "")).strip().lower()

        for name in AGENT_NAMES:
            if choice.startswith(name):
                logger.info("router: llm classified %r as %s", text[:60], name)
                return name

        # It answered, but not with one of the seven words. Worth seeing:
        # a router that silently returns None looks identical in the logs
        # to one that was never consulted, which is exactly the ambiguity
        # that made "is ROUTER_MODE=llm even on?" unanswerable from the
        # log file.
        logger.warning(
            "router: llm answered %r, which is not an agent name - "
            "falling back to the deterministic result", choice[:60],
        )

    except Exception as exc:
        logger.warning(
            "router: llm classification failed (%s: %s) - using the "
            "deterministic result", type(exc).__name__, exc,
        )

    return None
