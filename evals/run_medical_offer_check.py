"""
Live check: how the medical guidance reply offers doctors, by how many
the clinic has for the fitting specialty. Uses the REAL model (understanding
+ specialist); only the two catalogue tools are faked, so each case gets
exactly the doctors it needs.

    python evals/run_medical_offer_check.py

Expected shape (product decision, 2026-09-26):
  one doctor   -> the specialty and that doctor, offered in the same reply
  several      -> the specialty and "تحب أشوف لك الدكاترة ...؟", no names yet
  none         -> the clinic doesn't have that specialty; never elsewhere

Exit 0 = all cases as expected, 1 = a case failed, 2 = no API key.
"""

import itertools
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("LOG_LEVEL", "ERROR")

import config  # noqa: E402
import main  # noqa: E402
import tools  # noqa: E402

CLIENT = {
    "client_id": "medical-offer-check", "Dialect": "Egyptian",
    "clinic_name": "Demo Hospital", "clinic_name_ar": "مستشفى التجربة",
    "agent_name": "Latifa", "agent_name_ar": "لطيفة", "timezone": "Africa/Cairo",
    "base_url": "http://booking.test", "doctors_base_url": "http://doctors.test",
    "phone_example": "+201000000000",
}

SPECIALTIES = {"status": "found", "specialties": [
    {"id": "S-INT", "name": "طب الباطنة"}, {"id": "S-EYE", "name": "طب العيون"},
    {"id": "S-ORTHO", "name": "جراحة العظام"},
]}
ONE = [{"id": "D-1", "name": "د. عمر المديفر", "specialtyName": "طب الباطنة", "degreeName": "استشاري"}]
MANY = ONE + [
    {"id": "D-2", "name": "د. سارة العتيبي", "specialtyName": "طب الباطنة", "degreeName": "أخصائي"},
    {"id": "D-3", "name": "د. أحمد سالم", "specialtyName": "طب الباطنة", "degreeName": "استشاري"},
]
NAMES = ["عمر المديفر", "سارة العتيبي", "أحمد سالم"]

CASES = {
    "one_doctor": {"status": "found", "doctors": ONE,
                   "about": "استشاري طب الباطنة وأمراض الجهاز الهضمي"},
    "several_doctors": {"status": "found", "doctors": MANY},
    "no_doctor": {"status": "not_found_in_specialty"},
}

OPENING = "بطني وجعاني جدا من امبارح وعندي ترجيع"
FOLLOW_UPS = ["من امبارح بالليل، ومفيش حرارة", "الوجع في نص البطن وبيزيد بعد الأكل", "لا مفيش حاجة تانية"]

ASKS_TO_SHOW = re.compile(r"أشوف\s*لك\s*الدكاترة|أعرض\s*لك\s*الدكاترة|show you the doctors")
NOT_AVAILABLE = re.compile(r"معندناش|ما عندناش|مش متوفر|غير متوفر|مش موجود|ما فيه|لا يوجد|not available")
ELSEWHERE = re.compile(r"مستشفى تاني|مستشفى آخر|مكان تاني|برا المستشفى|خارج المستشفى|elsewhere|another hospital")


def _fake(payload, entity):
    def run(state=None, **kwargs):
        if entity and payload.get("doctors"):
            tools._remember_list(state, entity, payload["doctors"])
        return payload
    return run


def _check(case, reply, searched):
    names_shown = [n for n in NAMES if n in reply]
    if not searched:
        return ["find_available_doctors was never called"]
    if case == "one_doctor":
        return [] if "عمر المديفر" in reply and not ASKS_TO_SHOW.search(reply) else \
            ["expected the single doctor offered directly"]
    if case == "several_doctors":
        problems = []
        if not ASKS_TO_SHOW.search(reply):
            problems.append('expected "تحب أشوف لك الدكاترة ...؟"')
        if names_shown:
            problems.append(f"listed names before the patient said yes: {names_shown}")
        return problems
    problems = []
    if names_shown:
        problems.append(f"named a doctor: {names_shown}")
    if not NOT_AVAILABLE.search(reply):
        problems.append("did not say the specialty is not available here")
    if ELSEWHERE.search(reply):
        problems.append("pointed the patient elsewhere")
    return problems


def main_():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    if not config.llm_api_key():
        print("No API key (OPENROUTER_API_KEY / OPENAI_API_KEY) - this check calls the real model. Not runnable.")
        return 2

    original = (tools.list_specialties.func, tools.find_available_doctors.func)
    sessions = itertools.count(1)
    failures = 0
    try:
        for case, doctors_payload in CASES.items():
            tools.list_specialties.func = _fake(SPECIALTIES, None)
            calls = []

            def find(state=None, _p=doctors_payload, **kwargs):
                calls.append(kwargs)
                return _fake(_p, "doctor")(state=state, **kwargs)

            tools.find_available_doctors.func = find
            session = f"medical-offer-{case}-{next(sessions)}"
            transcript, reply = [], ""
            for text in [OPENING] + FOLLOW_UPS:
                result = main.send_message_with_signals(CLIENT["client_id"], session, text,
                                                        channel_phone="+201000000001", client_config=CLIENT)
                reply = result["reply"]
                transcript += [f"PATIENT: {text}", f"LATIFA:  {reply}"]
                if calls:
                    break
            problems = _check(case, reply, bool(calls))
            failures += bool(problems)
            print(("PASS " if not problems else "FAIL ") + case)
            for line in transcript:
                print("   " + line.replace("\n", "\n            "))
            for p in problems:
                print("   - " + p)
            print()
    finally:
        tools.list_specialties.func, tools.find_available_doctors.func = original

    print(f"{len(CASES) - failures}/{len(CASES)} cases as expected")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main_())
