"""
Semantic regression - the REAL conversation model, end to end.

Every case is a conversation moment: what the assistant just said, the
facts on file, and the patient's next message. The message is run
through the real graph (real model, fake hospital backend - nothing real
is booked, cancelled or sent) and the test checks what the system
UNDERSTOOD: the flow it put the patient in, the tool it chose, the
action that did or did not happen.

The phrasings are the product owner's list. Most carry no intent
keyword at all ("خلاص مش هروح", "أنا غيرت رأيي", "نفسي أكشف"), several
contextual ones are single words whose meaning depends only on the
question before them ("اه", "لا", "تمام", "2", "بكرة", "مش ده"), and the
same word is run under different questions. There is no keyword code
left in the graph for any of this to pass by accident.

Skipped unless model credentials are configured (.env). Run:
    python -m pytest tests/test_live_semantics.py -v
"""

import json
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import config
import fake_hospital

pytestmark = pytest.mark.skipif(not config.OPENAI_API_KEY, reason="no model credentials - live tests skipped")

TOMORROW = "2026-09-29"   # fake_hospital freezes today at Monday 2026-09-28


@pytest.fixture
def hospital(monkeypatch):
    h = fake_hospital.install(monkeypatch)
    h.reset()
    return h


@pytest.fixture
def g(hospital):
    import graph
    return graph


class Moment:
    """A conversation built by running the real tools against the fake
    backend, so the session and tool history are exactly what the graph
    would have produced."""

    def __init__(self, hospital, flow="general"):
        import tools
        self.tools = {t.name: t for t in tools.ALL_TOOLS}
        self.hospital = hospital
        self.session_id = f"+966500000001+live-{uuid.uuid4().hex[:8]}"
        self.state = hospital.make_state(session_id=self.session_id)
        self.state.update({"flow": flow, "turn": 1, "greeted": True, "pending_confirmation": None})
        self._n = 0

    def patient(self, text):
        self.state["messages"].append(HumanMessage(content=text, id=uuid.uuid4().hex))
        return self

    def assistant(self, text, confirm=None):
        self.state["messages"].append(AIMessage(content=text))
        if confirm:
            action, target = confirm
            self.state["pending_confirmation"] = {"action": action, "target": target, "turn": 1}
        return self

    def tool(self, name, **args):
        self._n += 1
        call_id = f"seed_{self._n}"
        self.state["messages"].append(AIMessage(content="", tool_calls=[
            {"name": name, "args": dict(args), "id": call_id, "type": "tool_call"}]))
        message = self.tools[name].invoke({"type": "tool_call", "name": name, "id": call_id,
                                            "args": {**args, "state": self.state}})
        self.state["messages"].append(message)
        return json.loads(message.content)

    def run(self, g, text):
        import tools
        self.patient(text)
        state = {k: v for k, v in self.state.items() if k not in ("templates", "system_prompt")}
        state["booking"] = tools.export_session(self.session_id)
        before = len(self.state["messages"])
        result = g.graph.invoke(state, config={"configurable": {"thread_id": self.session_id},
                                               "recursion_limit": 30})
        added = result["messages"][before:]
        calls = [c for m in added for c in (getattr(m, "tool_calls", None) or [])]
        return Outcome(result, calls, self.hospital)


class Outcome:
    def __init__(self, result, calls, hospital):
        self.result, self.calls, self.hospital = result, calls, hospital
        self.flow = result.get("flow")
        self.reply = result["messages"][-1].content

    def called(self, name):
        return [c["args"] for c in self.calls if c["name"] == name]

    def booking_status(self, ref="TNS-10001"):
        return (self.hospital.booking(ref) or {}).get("status")


# ----------------------------------------------------------------------
# contexts
# ----------------------------------------------------------------------

def fresh(hospital):
    return Moment(hospital).patient("السلام عليكم").assistant(
        "أهلًا وسهلًا 👋 كيف أقدر أساعدك اليوم؟")


def appointment_shown(hospital):
    m = Moment(hospital, flow="general").patient("ابي اعرف موعدي")
    m.tool("lookup_appointment", use_channel_identity=True)
    return m.assistant("موعدك الحالي: د. أحمد سامي · المنار · الأربعاء 30-09 · 1:00 م. أقدر أساعدك بشي ثاني؟")


def cancel_confirmation_asked(hospital):
    m = Moment(hospital, flow="cancel").patient("ابي الغي موعدي")
    m.tool("lookup_appointment", use_channel_identity=True)
    return m.assistant("موعدك مع د. أحمد سامي · المنار · الأربعاء 30-09 · 1:00 م.\nمتأكد تبغى تلغي هذا الموعد؟",
                       confirm=("cancel", "TNS-10001"))


def specialty_list_shown(hospital):
    m = Moment(hospital, flow="booking").patient("ابي احجز")
    m.tool("list_specialties")
    return m.assistant("اختار التخصص المناسب:\n1️⃣ طب الجلدية\n2️⃣ طب الباطنة\n3️⃣ طب الأسنان")


def same_number_asked(hospital):
    m = Moment(hospital, flow="booking").patient("ابي احجز مع د. أحمد سامي")
    m.tool("match_entity_for_booking", entity_type="doctor", name="أحمد سامي")
    m.tool("match_entity_for_booking", entity_type="branch", name="النزهة")
    m.tool("get_available_slots_for_booking", from_date=TOMORROW, to_date=TOMORROW)
    m.tool("select_appointment_slot", option_number=1)
    return m.assistant("تمام ✅ نكمل الحجز على نفس رقم الواتساب؟")


def slots_shown(hospital):
    m = Moment(hospital, flow="booking").patient("ابي احجز مع د. أحمد سامي في النزهة")
    m.tool("match_entity_for_booking", entity_type="doctor", name="أحمد سامي")
    m.tool("match_entity_for_booking", entity_type="branch", name="النزهة")
    m.tool("get_available_slots_for_booking", from_date=TOMORROW, to_date=TOMORROW)
    return m.assistant("المواعيد المتاحة يوم الثلاثاء:\n1️⃣ 4:30 م\n2️⃣ 5:00 م\n3️⃣ 5:30 م\nأي موعد يناسبك؟")


def day_asked(hospital):
    m = Moment(hospital, flow="booking").patient("ابي احجز مع د. أحمد سامي في النزهة")
    m.tool("match_entity_for_booking", entity_type="doctor", name="أحمد سامي")
    m.tool("match_entity_for_booking", entity_type="branch", name="النزهة")
    return m.assistant("د. أحمد سامي يداوم في النزهة الأحد والثلاثاء والخميس. أي يوم يناسبك؟")


def reschedule_doctor_asked(hospital):
    m = Moment(hospital, flow="reschedule").patient("ابي اغير موعدي")
    m.tool("lookup_appointment", use_channel_identity=True)
    return m.assistant("موعدك مع د. أحمد سامي · المنار · الأربعاء 30-09 · 1:00 م.\n"
                       "تبغى تعدله مع نفس الدكتور ولا دكتور ثاني؟")


def doctor_suggested(hospital):
    m = Moment(hospital, flow="booking").patient("ابي دكتور أحمد")
    m.tool("match_entity_for_booking", entity_type="doctor", name="أحمد")
    return m.assistant("تقصد د. أحمد سامي - جلدية؟")


# ----------------------------------------------------------------------
# intent from meaning, not keywords
# ----------------------------------------------------------------------

INTENT_CASES = [
    # (context, message, acceptable flows)
    (fresh, "عايز أحجز", {"booking"}),
    (fresh, "ممكن أشوف دكتور", {"booking", "medical"}),
    (fresh, "نفسي أكشف", {"booking", "medical"}),
    (day_asked, "بكرة ينفع؟", {"booking"}),
    (fresh, "عايز موعد جديد", {"booking"}),
    (appointment_shown, "خلاص مش هروح", {"cancel"}),
    (appointment_shown, "الموعد ده مش عايزاه", {"cancel"}),
    (appointment_shown, "ممكن تلغي اللي حجزته", {"cancel"}),
    (appointment_shown, "أنا غيرت رأيي", {"cancel", "reschedule"}),
    (appointment_shown, "الوقت ده مش مناسب", {"reschedule"}),
    (appointment_shown, "ممكن نخليه يوم تاني", {"reschedule"}),
    (appointment_shown, "عايز أغير الموعد", {"reschedule"}),
    (appointment_shown, "ينفع الأسبوع الجاي؟", {"reschedule"}),
    (fresh, "حاسس بحاجة غريبة", {"medical"}),
    (fresh, "رجلي وقعت عليها", {"medical"}),
    (fresh, "بطني بقالها يومين بتوجعني", {"medical"}),
    (fresh, "مش عارف ده تبع أنهي دكتور", {"medical", "booking"}),
]


@pytest.mark.parametrize("context, message, flows", INTENT_CASES,
                         ids=[f"{c.__name__}:{m}" for c, m, _ in INTENT_CASES])
def test_intent_is_understood_from_meaning(hospital, g, context, message, flows):
    outcome = context(hospital).run(g, message)
    assert outcome.flow in flows, (message, outcome.flow, outcome.reply)
    # Nothing irreversible happens on the first mention of a change.
    assert outcome.booking_status() != 6, "cancelled without a confirmation question"


# ----------------------------------------------------------------------
# the same short word, read against the question before it
# ----------------------------------------------------------------------

def test_yes_to_the_cancel_question_cancels(hospital, g):
    outcome = cancel_confirmation_asked(hospital).run(g, "اه")
    assert outcome.called("cancel_appointment") and outcome.booking_status() == 6
    assert "إلغاء" in outcome.reply


def test_no_to_the_cancel_question_keeps_the_booking(hospital, g):
    outcome = cancel_confirmation_asked(hospital).run(g, "لا")
    assert outcome.booking_status() != 6
    assert not outcome.result.get("pending_confirmation") or \
        outcome.result["pending_confirmation"].get("action") != "cancel" or "؟" in outcome.reply


def test_tamam_after_choose_a_specialty_is_not_a_cancellation(hospital, g):
    outcome = specialty_list_shown(hospital).run(g, "تمام")
    assert not outcome.called("cancel_appointment")
    assert outcome.flow == "booking" and outcome.booking_status() != 6


def test_tamam_after_same_number_is_a_phone_choice_not_a_cancellation(hospital, g):
    outcome = same_number_asked(hospital).run(g, "تمام")
    assert not outcome.called("cancel_appointment")
    assert outcome.flow == "booking" and outcome.booking_status() != 6
    assert not outcome.called("send_otp")   # the WhatsApp number needs no code


def test_two_picks_the_second_option_shown(hospital, g):
    import tools
    m = slots_shown(hospital)
    second = tools._BOOKING_SESSIONS[m.session_id]["last_list"]["items"][1]
    outcome = m.run(g, "2")
    assert {"option_number": 2} in [{"option_number": a.get("option_number")}
                                    for a in outcome.called("select_appointment_slot")]
    chosen = tools._BOOKING_SESSIONS[m.session_id].get("selected_slot") or {}
    assert chosen.get("slotStart") == second.get("slotStart")


def test_tomorrow_is_the_calendar_date(hospital, g):
    outcome = day_asked(hospital).run(g, "بكرة")
    dates = [a.get("date") or a.get("from_date") for c in ("resolve_available_day", "get_available_slots_for_booking")
             for a in outcome.called(c)]
    assert TOMORROW in dates, outcome.calls


def test_same_doctor_keeps_the_doctor_in_a_reschedule(hospital, g):
    outcome = reschedule_doctor_asked(hospital).run(g, "نفس الدكتور")
    assert outcome.flow == "reschedule"
    assert not [a for a in outcome.called("match_entity_for_booking") if "سامي" not in str(a.get("name", ""))]


def test_not_that_one_rejects_the_suggested_doctor(hospital, g):
    import tools
    m = doctor_suggested(hospital)
    outcome = m.run(g, "مش ده")
    assert outcome.flow == "booking"
    assert "سامي" not in str((tools._BOOKING_SESSIONS.get(m.session_id) or {}).get("doctor_display_name") or "")


# ----------------------------------------------------------------------
# the screenshots and the new scope policy
# ----------------------------------------------------------------------

def test_remote_sessions_go_to_customer_service_not_a_name_search(hospital, g):
    outcome = fresh(hospital).run(g, "فضلاً كنت حابه أعرف إذا كان عند د. ماضي جلسات عن بعد ؟")
    assert "ماضي جلسات" not in outcome.reply
    assert "9200 16388" in outcome.reply or "خدمة العملاء" in outcome.reply


def test_psychiatry_lists_only_psychiatrists(hospital, g):
    outcome = fresh(hospital).run(g, "تخصص طب نفسي؟")
    psychologists = [d["ar"].replace("د. ", "") for d in fake_hospital.DOCTORS if d["specialty"] == "PSYC"]
    assert psychologists and not any(name in outcome.reply for name in psychologists)


def test_off_topic_gets_the_scope_sentence_and_no_tools(hospital, g):
    outcome = fresh(hospital).run(g, "مين فاز بالمباراة امبارح؟")
    assert not outcome.calls and "عذر" in outcome.reply


def test_unknown_hospital_detail_offers_customer_service(hospital, g):
    outcome = fresh(hospital).run(g, "عندكم مواقف سيارات للمراجعين؟")
    assert "9200 16388" in outcome.reply or "خدمة العملاء" in outcome.reply
