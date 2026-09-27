"""
Routing scenarios shared by the wiring tests (test_semantic_router.py,
stand-in model) and the live tests (test_live_router.py, real model).

Each scenario is one conversation moment: who owns the flow, what the
assistant just said, and the patient's message - plus PARAPHRASES of
that message. Several paraphrases deliberately carry none of the words
an intent keyword list would look for ("الغاء", "حجز", "موعد", "تعديل"),
and one scenario is written in English, so a pass cannot come from
spotting a word.

`reading` is what a correct reader returns for the moment. The wiring
tests feed it to the router through a stand-in model; the live tests
ignore it and check the real model's routing against `expected`.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

GREETING = (
    "أهلاً بيك 👋 أنا لطيفة، المساعدة الافتراضية للمستشفى.\n"
    "أقدر أساعدك في حجز أو تعديل أو إلغاء موعد، والتوجيه الطبي، والاستفسارات.\n"
    "تحب أساعدك في إيه النهارده؟"
)

BOOKING_ON_SCREEN = (
    "حجزك الحالي:\n👨‍⚕️ د. سارة علي - جلدية\n🏥 فرع المعادي\n"
    "📅 الأحد 12-10-2026\n⏰ 10:00 صباحًا\nأقدر أساعدك في حاجة تانية؟"
)


@dataclass
class Case:
    id: str
    active: Optional[str]
    history: List[Tuple[str, str]]       # ("human"|"ai", text), oldest first
    message: str
    paraphrases: List[str]
    expected: Tuple[str, ...]            # acceptable owners of the turn
    reading: Dict                        # what a correct reader returns
    session: Dict = field(default_factory=dict)


CASES = [
    # 1 --------------------------------------------------------------
    Case(
        id="cancel_without_the_word",
        active=None,
        history=[("human", "عايزة أعرف حجزي"), ("ai", BOOKING_ON_SCREEN)],
        message="مش عايزة الموعد ده خلاص",
        paraphrases=[
            "خلاص مش محتاجاه ده",
            "شيليه خالص مش هاجي",
            "I don't need that one anymore, drop it",
        ],
        expected=("cancel",),
        reading={"intent": "cancel", "topic_changed": True, "health": "none"},
    ),
    # 2 --------------------------------------------------------------
    Case(
        id="reschedule_without_the_word",
        active=None,
        history=[("human", "عايزة أعرف حجزي"), ("ai", BOOKING_ON_SCREEN)],
        message="المعاد ده مش نافعني ممكن وقت تاني؟",
        paraphrases=[
            "مش هلحق أجي الأحد، ينفع نخليه يوم تاني؟",
            "الساعة دي صعبة عليا، فيه بعدها؟",
            "can we push it to another day?",
        ],
        expected=("reschedule",),
        reading={"intent": "reschedule", "topic_changed": True, "health": "none"},
    ),
    # 3 --------------------------------------------------------------
    Case(
        id="medical_vague_feeling",
        active=None,
        history=[("human", "السلام عليكم"), ("ai", GREETING)],
        message="حاسه بحاجة غريبة من امبارح",
        paraphrases=[
            "من امبارح وأنا مش على بعضي",
            "جسمي مش طبيعي بقاله يومين",
            "I've been feeling off since yesterday",
        ],
        expected=("medical",),
        reading={"intent": "medical", "topic_changed": False, "health": "health"},
    ),
    # 4a -------------------------------------------------------------
    Case(
        id="tomorrow_inside_booking",
        active="booking",
        history=[
            ("human", "عايزة أحجز مع د. أحمد"),
            ("ai", "الأيام المتاحة مع د. أحمد سامي:\n1️⃣ الأحد\n2️⃣ الثلاثاء\nتحبي أي يوم؟"),
        ],
        message="بكره ينفع؟",
        paraphrases=["طب ولو بكرة؟", "ينفع اللي بعد النهارده على طول؟"],
        expected=("booking",),
        reading={"intent": "booking", "topic_changed": False, "health": "none"},
        session={"doctor_display_name": "د. أحمد سامي", "last_list": {"entity_type": "day", "items": [1, 2]}},
    ),
    # 4b - the SAME words, read against a reschedule ----------------
    Case(
        id="tomorrow_inside_reschedule",
        active="reschedule",
        history=[
            ("human", "عايزة أغير معاد الأحد"),
            ("ai", "تمام، تحبي تنقلي موعدك مع د. سارة لأي يوم؟"),
        ],
        message="بكره ينفع؟",
        paraphrases=["طب ولو بكرة؟", "ينفع اللي بعد النهارده على طول؟"],
        expected=("reschedule",),
        reading={"intent": "reschedule", "topic_changed": False, "health": "none"},
    ),
    # 5a - "اه" to an offer the owner cannot carry out --------------
    Case(
        id="yes_to_medical_booking_offer",
        active="medical",
        history=[
            ("human", "عندي حكة جامدة في إيدي"),
            ("ai", "ألف سلامة عليكي 🌷 الأنسب تخصص الجلدية.\n⚕️ ده توجيه عام وليس تشخيصًا.\n"
                   "تحبي أحجزلك عند د. طه مبروك - جلدية؟"),
        ],
        message="اه",
        paraphrases=["ايوه ياريت", "ماشي يلا", "okay please"],
        expected=("booking",),
        reading={"intent": "booking", "topic_changed": False, "health": "none"},
    ),
    # 5b - "اه" to the owner's own confirmation ----------------------
    Case(
        id="yes_to_cancel_confirmation",
        active="cancel",
        history=[
            ("human", "عايزة ألغي حجز الأحد"),
            ("ai", BOOKING_ON_SCREEN.replace("أقدر أساعدك في حاجة تانية؟", "متأكدة إنك عايزة تلغي الموعد ده؟")),
        ],
        message="اه",
        paraphrases=["ايوه أكيد", "yes"],
        expected=("cancel",),
        reading={"intent": "cancel", "topic_changed": False, "health": "none"},
    ),
    # 6 --------------------------------------------------------------
    Case(
        id="no_thats_not_what_i_meant",
        active="booking",
        history=[
            ("human", "عايزة دكتور أحمد"),
            ("ai", "تقصدي د. أحمد سامي - جلدية؟"),
        ],
        message="لا مش ده قصدي",
        paraphrases=["لأ مش هو", "مش الدكتور ده، واحد تاني"],
        expected=("booking",),
        reading={"intent": "booking", "topic_changed": False, "health": "none"},
    ),
    # 7 --------------------------------------------------------------
    Case(
        id="doctor_schedule_opening",
        active=None,
        history=[("human", "صباح الخير"), ("ai", GREETING)],
        message="مواعيد دكتور أحمد",
        paraphrases=["دكتور أحمد بيبقى موجود امتى؟", "when is Dr Ahmed available?"],
        expected=("booking",),
        reading={"intent": "booking", "topic_changed": False, "health": "none"},
    ),
    # 8 --------------------------------------------------------------
    Case(
        id="monday_then",
        active="booking",
        history=[
            ("human", "الأحد"),
            ("ai", "للأسف مفيش مواعيد يوم الأحد مع د. أحمد سامي. تحبي يوم تاني؟"),
        ],
        message="طب الاثنين؟",
        paraphrases=["طيب الاتنين", "والإثنين فاضي؟"],
        expected=("booking",),
        reading={"intent": "booking", "topic_changed": False, "health": "none"},
        session={"doctor_display_name": "د. أحمد سامي"},
    ),
    # topic changes mid-flow -----------------------------------------
    Case(
        id="switch_booking_to_cancel",
        active="booking",
        history=[
            ("human", "الساعة 10"),
            ("ai", "تمام ✅ نكمل الحجز على نفس رقم الواتساب ده؟"),
        ],
        message="استني الأول، عايزة أشيل الحجز القديم اللي عندي",
        paraphrases=["لا خلاص مش عايزة ده، ومعاد الأسبوع اللي فات كمان مش هروحه"],
        expected=("cancel",),
        reading={"intent": "cancel", "topic_changed": True, "health": "none"},
        session={"doctor_display_name": "د. أحمد سامي", "selected_slot": {"date_display": "الاثنين", "time_display": "10:00 صباحًا"}},
    ),
    Case(
        id="unrelated_question_mid_booking",
        active="booking",
        history=[
            ("human", "الساعة 10"),
            ("ai", "تمام ✅ نكمل الحجز على نفس رقم الواتساب ده؟"),
        ],
        message="هو أنا كنت مقدمة على تدريب عندكم، في رد؟",
        paraphrases=["بالمناسبة أنا بعت السي في بتاعي عندكم من أسبوع"],
        expected=("concierge",),
        reading={"intent": "concierge", "topic_changed": True, "health": "none"},
    ),
    # safety ---------------------------------------------------------
    Case(
        id="crisis_mid_booking",
        active="booking",
        history=[
            ("human", "الساعة 10"),
            ("ai", "تمام ✅ نكمل الحجز على نفس رقم الواتساب ده؟"),
        ],
        message="مش فارقة، أنا تعبت ومش عايزة أكمل خلاص",
        paraphrases=["مبقتش عايزة أصحى تاني"],
        expected=("medical",),
        reading={"intent": "medical", "topic_changed": True, "health": "crisis"},
    ),
]


def build_messages(case: Case, message: Optional[str] = None):
    """The conversation as LangChain messages, ending with the patient's
    message (or the given paraphrase of it)."""

    from langchain_core.messages import AIMessage, HumanMessage

    built = []
    for index, (role, text) in enumerate(case.history):
        cls = HumanMessage if role == "human" else AIMessage
        built.append(cls(content=text, id=f"{case.id}-{index}"))
    built.append(HumanMessage(content=message or case.message, id=f"{case.id}-now"))
    return built
