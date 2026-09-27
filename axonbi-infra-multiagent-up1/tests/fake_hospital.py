"""
An in-memory FAKE HOSPITAL BACKEND for measuring the WhatsApp booking
agent end to end without touching any real service.

    import fake_hospital
    hospital = fake_hospital.install(monkeypatch)   # pytest
    hospital = fake_hospital.install(None)          # script: call hospital.uninstall() when done
    with fake_hospital.install(None) as hospital:   # script, scoped
        ...

Everything is patched at RUNTIME, on module attributes (api.*, tools.*,
rag.*, config.*, progress.*), never by editing a source file, so the same
fake works against any checkout whose api.py/tools.py have this shape
(this worktree and origin/tanasuq-production alike). The project's own
modules are imported from `sys.path` inside install() - so a script that
puts ANOTHER checkout first on sys.path gets that checkout patched.

What is replaced:
  - every api.py function tools.py calls (booking API, doctors API,
    Authentica OTP) by a stateful fake that speaks the SAME envelope
    (api._result) and the same raw field names the real API returns;
  - api.py's low-level HTTP helpers and `api.requests`, as a backstop, so
    an api function nobody faked yet can never reach the network;
  - tools.requests / tools.smtplib (complaint webhook + SMTP fallback) and
    progress.requests ("please wait" webhook): recorded, 200 OK;
  - rag.search_with_scores / rag._get_embeddings_model: keyword ranking
    over the knowledge-base chunks, no embeddings API;
  - tools.datetime / tools.date (and graph's, when graph is imported
    already): a frozen clock at FAKE_NOW, so "today", "tomorrow" and the
    slot windows are reproducible;
  - config's env base-url overrides and client_config.csv lookup, so the
    TENANT row below is the only tenant and no code path can fall back
    to a real URL.
"""

import copy
import datetime as _dt
import functools
import inspect
import json
import os
import re
import sys
import uuid
from zoneinfo import ZoneInfo

# ==========================================================
# Clock
# ==========================================================

CLINIC_TZ = ZoneInfo("Asia/Riyadh")
UTC = _dt.timezone.utc

FAKE_TODAY = _dt.date(2026, 9, 28)                      # a Monday
FAKE_NOW = _dt.datetime(2026, 9, 28, 8, 0, tzinfo=CLINIC_TZ)  # 08:00 clinic time
SLOT_WINDOW_DAYS = 14                                   # slots exist for FAKE_TODAY .. +13

_REAL_DATETIME = _dt.datetime
_REAL_DATE = _dt.date

# The instant "now" is, while the fake is installed. A dict so FakeHospital
# can move it (set_now) without rebinding the frozen classes.
_CLOCK = {"now": FAKE_NOW.astimezone(UTC)}


class _FrozenDatetimeMeta(type):
    # isinstance(x, tools.datetime) must stay true for REAL datetimes too.
    def __instancecheck__(cls, obj):
        return isinstance(obj, _REAL_DATETIME)

    def __subclasscheck__(cls, sub):
        return issubclass(sub, _REAL_DATETIME)


class FrozenDatetime(_REAL_DATETIME, metaclass=_FrozenDatetimeMeta):
    """datetime with now()/utcnow()/today() pinned to the fake clock."""

    @classmethod
    def now(cls, tz=None):
        instant = _CLOCK["now"]
        if tz is None:
            # Naive "now" is the clinic's wall clock - the deployment runs
            # for this tenant, and it keeps runs independent of the host TZ.
            local = instant.astimezone(CLINIC_TZ).replace(tzinfo=None)
            return cls.fromisoformat(local.isoformat())
        return cls.fromisoformat(instant.astimezone(tz).isoformat())

    @classmethod
    def utcnow(cls):
        return cls.fromisoformat(_CLOCK["now"].replace(tzinfo=None).isoformat())

    @classmethod
    def today(cls):
        return cls.now()


class _FrozenDateMeta(type):
    def __instancecheck__(cls, obj):
        return isinstance(obj, _REAL_DATE)

    def __subclasscheck__(cls, sub):
        return issubclass(sub, _REAL_DATE)


class FrozenDate(_REAL_DATE, metaclass=_FrozenDateMeta):
    """date with today() pinned to the fake clock (clinic-local date)."""

    @classmethod
    def today(cls):
        local = _CLOCK["now"].astimezone(CLINIC_TZ).date()
        return cls(local.year, local.month, local.day)


# ==========================================================
# The tenant row n8n would send as raw_client_config
# ==========================================================

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAKE_BOOKINGS_BASE_URL = "https://bookings.fake-hospital.test"
FAKE_DOCTORS_BASE_URL = "https://doctors.fake-hospital.test"
FAKE_AUTHENTICA_BASE_URL = "https://otp.fake-hospital.test/api/v2"
FAKE_COMPLAINT_WEBHOOK_URL = "https://n8n.fake-hospital.test/webhook/complaint"
FAKE_PROGRESS_WEBHOOK_URL = "https://n8n.fake-hospital.test/webhook/progress"

CHANNEL_PHONE = "+966500000001"
TEST_OTP = "123456"

TENANT = {
    "client_id": "tanasuq-test",
    "clinic_name": "Tanasuq Medical Hospital",
    "clinic_name_ar": "مستشفى تناسق الطبية",
    "agent_name": "Sara",
    "agent_name_ar": "سارة",
    "Dialect": "Saudi",
    "timezone": "Asia/Riyadh",
    "base_url": FAKE_BOOKINGS_BASE_URL,
    "doctors_base_url": FAKE_DOCTORS_BASE_URL,
    "phone_example": "0501234567",
    "country_codes_hint": "966",
    "knowledge_base_file": os.path.join(_PROJECT_ROOT, "knowledge_base", "tanasuq-saudi.txt"),
    "complaint_email_to": "quality@fake-hospital.test",
    "branch1_name": "Al Nuzha",
    "branch1_name_ar": "النزهة",
    "branch2_name": "Al Manar",
    "branch2_name_ar": "المنار",
}


# ==========================================================
# Seed data
# ==========================================================

_NS = uuid.UUID("6b1c3f9e-2a4d-4e8b-9c1a-7f00d5e0a001")


def _guid(key: str) -> str:
    return str(uuid.uuid5(_NS, key))


BRANCHES = [
    {
        "id": _guid("branch:nuzha"), "name": "Al Nuzha", "altName": "النزهة",
        "formatedName": "Al Nuzha Branch",
        "address": "7282 أبي سفيان بن حرب، النزهة، الرياض 12473",
        "cityName": "Riyadh", "stateName": "Riyadh Region", "countryName": "Saudi Arabia",
        "phoneNumber": "+966112000101", "latitude": 24.7574, "longitude": 46.7162,
        "isActive": True,
    },
    {
        "id": _guid("branch:manar"), "name": "Al Manar", "altName": "المنار",
        "formatedName": "Al Manar Branch",
        "address": "2955 شارع الشيخ عبدالرحمن بن إسحاق، حي المنار، الرياض 14221",
        "cityName": "Riyadh", "stateName": "Riyadh Region", "countryName": "Saudi Arabia",
        "phoneNumber": "+966112000202", "latitude": 24.7496, "longitude": 46.8120,
        "isActive": True,
    },
]
_B = {b["name"]: b for b in BRANCHES}
NUZHA, MANAR = _B["Al Nuzha"]["id"], _B["Al Manar"]["id"]

SPECIALTIES = [
    {"id": _guid("spec:derm"), "name": "Dermatology", "altName": "طب الجلدية", "code": "DERM"},
    {"id": _guid("spec:im"), "name": "Internal Medicine", "altName": "طب الباطنة", "code": "IM"},
    {"id": _guid("spec:ortho"), "name": "Orthopedics", "altName": "جراحة العظام", "code": "ORTH"},  # nobody staffed
    {"id": _guid("spec:dent"), "name": "Dentistry", "altName": "طب الأسنان", "code": "DENT"},
    # The psychiatry production bug: two DIFFERENT fields sharing the stem
    # "نفسي". A search for طب نفسي must return its 5 psychiatrists only,
    # never the psychologists registered under أخصائي نفسي.
    {"id": _guid("spec:psy"), "name": "Psychiatry", "altName": "طب نفسي", "code": "PSY"},
    {"id": _guid("spec:psyc"), "name": "Psychology", "altName": "أخصائي نفسي", "code": "PSYC"},
]
_S = {s["code"]: s for s in SPECIALTIES}

SERVICES = [
    {"id": _guid("svc:derm"), "en": "Dermatology Consultation", "ar": "استشارة جلدية",
     "specialty": "DERM", "price": 300,
     "description": "Skin, hair and nail consultation.", "altDescription": "كشف واستشارة لأمراض الجلد والشعر والأظافر."},
    {"id": _guid("svc:im"), "en": "Internal Medicine Consultation", "ar": "استشارة باطنة",
     "specialty": "IM", "price": 250,
     "description": "Adult internal medicine consultation.", "altDescription": "كشف باطنة للبالغين."},
    {"id": _guid("svc:dentcheck"), "en": "Dental Check-up", "ar": "كشف أسنان",
     "specialty": "DENT", "price": 200,
     "description": "Dental examination.", "altDescription": "فحص الأسنان واللثة."},
    {"id": _guid("svc:cleaning"), "en": "Teeth Cleaning", "ar": "تنظيف أسنان",
     "specialty": "DENT", "price": 350,
     "description": "Scaling and polishing.", "altDescription": "تنظيف وتلميع الأسنان."},
    {"id": _guid("svc:ortho"), "en": "Orthopedic Consultation", "ar": "استشارة عظام",
     "specialty": "ORTH", "price": 300,
     "description": "Bones and joints consultation.", "altDescription": "استشارة العظام والمفاصل."},
    {"id": _guid("svc:psy"), "en": "Psychiatry Consultation", "ar": "استشارة طب نفسي",
     "specialty": "PSY", "price": 450,
     "description": "Psychiatric assessment.", "altDescription": "تقييم وعلاج نفسي دوائي."},
    {"id": _guid("svc:psyc"), "en": "Psychotherapy Session", "ar": "جلسة علاج نفسي",
     "specialty": "PSYC", "price": 350,
     "description": "Talk therapy session.", "altDescription": "جلسة علاج نفسي بالكلام."},
]
_SV = {s["id"]: s for s in SERVICES}
SVC_DERM, SVC_IM, SVC_DENT, SVC_CLEAN = (_guid("svc:derm"), _guid("svc:im"),
                                         _guid("svc:dentcheck"), _guid("svc:cleaning"))
SVC_PSY, SVC_PSYC = _guid("svc:psy"), _guid("svc:psyc")

DOCTORS = [
    {"id": _guid("doc:ahmed-sami"), "en": "Dr. Ahmed Sami", "ar": "د. أحمد سامي",
     "specialty": "DERM", "degree": ("Consultant", "استشاري"),
     "about": "استشاري الأمراض الجلدية والتجميل، خبرة 15 سنة في علاج الأكزيما وحب الشباب.",
     "services": [SVC_DERM]},
    {"id": _guid("doc:reem-harbi"), "en": "Dr. Reem Al-Harbi", "ar": "د. ريم الحربي",
     "specialty": "DERM", "degree": ("Specialist", "أخصائي"),
     "about": "أخصائية جلدية، مهتمة بأمراض الجلد عند الأطفال.",
     "services": [SVC_DERM]},
    {"id": _guid("doc:ahmed-otaibi"), "en": "Dr. Ahmed Al-Otaibi", "ar": "د. أحمد العتيبي",
     "specialty": "IM", "degree": ("Consultant", "استشاري"),
     "about": "استشاري الباطنة والسكري والضغط.",
     "services": [SVC_IM]},
    {"id": _guid("doc:khalid-shehri"), "en": "Dr. Khalid Al-Shehri", "ar": "د. خالد الشهري",
     "specialty": "DENT", "degree": ("Specialist", "أخصائي"),
     "about": "أخصائي طب الأسنان العام وتنظيف الأسنان.",
     "services": [SVC_DENT, SVC_CLEAN]},
    {"id": _guid("doc:noura-dosari"), "en": "Dr. Noura Al-Dosari", "ar": "د. نورة الدوسري",
     "specialty": "DENT", "degree": ("Consultant", "استشاري"),
     "about": "استشارية تقويم وطب أسنان.",
     "services": [SVC_DENT]},
    # طب نفسي - exactly FIVE psychiatrists.
    {"id": _guid("doc:saad-madi"), "en": "Dr. Saad Al-Madi", "ar": "د. سعد الماضي",
     "specialty": "PSY", "degree": ("Consultant", "استشاري"),
     "about": "استشاري الطب النفسي للبالغين.", "services": [SVC_PSY]},
    {"id": _guid("doc:khalid-anazi"), "en": "Dr. Khalid Al-Anazi", "ar": "د. خالد العنزي",
     "specialty": "PSY", "degree": ("Specialist", "أخصائي"),
     "about": "أخصائي الطب النفسي.", "services": [SVC_PSY]},
    {"id": _guid("doc:mona-qahtani"), "en": "Dr. Mona Al-Qahtani", "ar": "د. منى القحطاني",
     "specialty": "PSY", "degree": ("Consultant", "استشاري"),
     "about": "استشارية الطب النفسي للأطفال والمراهقين.", "services": [SVC_PSY]},
    {"id": _guid("doc:faisal-shammari"), "en": "Dr. Faisal Al-Shammari", "ar": "د. فيصل الشمري",
     "specialty": "PSY", "degree": ("Specialist", "أخصائي"),
     "about": "أخصائي اضطرابات القلق والنوم.", "services": [SVC_PSY]},
    {"id": _guid("doc:hind-ghamdi"), "en": "Dr. Hind Al-Ghamdi", "ar": "د. هند الغامدي",
     "specialty": "PSY", "degree": ("Consultant", "استشاري"),
     "about": "استشارية الطب النفسي.", "services": [SVC_PSY]},
    # أخصائي نفسي - psychologists (NOT psychiatrists).
    {"id": _guid("doc:noura-madi"), "en": "Noura Al-Madi", "ar": "نورة الماضي",
     "specialty": "PSYC", "degree": ("Psychologist", "أخصائية نفسية"),
     "about": "أخصائية علاج نفسي معرفي سلوكي.", "services": [SVC_PSYC]},
    {"id": _guid("doc:sara-malki"), "en": "Sara Al-Malki", "ar": "سارة المالكي",
     "specialty": "PSYC", "degree": ("Psychologist", "أخصائية نفسية"),
     "about": "أخصائية إرشاد أسري.", "services": [SVC_PSYC]},
    {"id": _guid("doc:abdulrahman-yami"), "en": "Abdulrahman Al-Yami", "ar": "عبدالرحمن اليامي",
     "specialty": "PSYC", "degree": ("Psychologist", "أخصائي نفسي"),
     "about": "أخصائي علاج الإدمان.", "services": [SVC_PSYC]},
    {"id": _guid("doc:lama-zahrani"), "en": "Lama Al-Zahrani", "ar": "لمى الزهراني",
     "specialty": "PSYC", "degree": ("Psychologist", "أخصائية نفسية"),
     "about": "أخصائية علاج نفسي للأطفال.", "services": [SVC_PSYC]},
]
_D = {d["id"]: d for d in DOCTORS}
DR_AHMED_SAMI, DR_REEM, DR_AHMED_OTAIBI, DR_KHALID, DR_NOURA = (d["id"] for d in DOCTORS[:5])
PSYCHIATRISTS = [d["id"] for d in DOCTORS if d["specialty"] == "PSY"]
PSYCHOLOGISTS = [d["id"] for d in DOCTORS if d["specialty"] == "PSYC"]
DR_SAAD_MADI, NOURA_MADI = _guid("doc:saad-madi"), _guid("doc:noura-madi")
SPEC_PSY, SPEC_PSYC = _guid("spec:psy"), _guid("spec:psyc")

# Weekly rota. Times are CLINIC-LOCAL wall clock; the fake API speaks UTC
# ("+00:00") on the wire exactly like the real one (see tools.to_clinic_local).
SCHEDULES = [
    # doctor,          branch, weekdays,                                        start,  end,    minutes, service
    (DR_AHMED_SAMI,    NUZHA,  ("Sunday", "Tuesday", "Thursday"),               "16:00", "20:00", 20, SVC_DERM),
    (DR_AHMED_SAMI,    MANAR,  ("Monday", "Wednesday"),                         "09:00", "13:00", 20, SVC_DERM),
    (DR_REEM,          MANAR,  ("Sunday", "Tuesday", "Wednesday"),              "10:00", "14:00", 30, SVC_DERM),
    (DR_AHMED_OTAIBI,  NUZHA,  ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"), "09:00", "12:00", 15, SVC_IM),
    (DR_KHALID,        NUZHA,  ("Saturday", "Monday", "Wednesday"),             "17:00", "21:00", 30, SVC_DENT),
    (DR_NOURA,         MANAR,  ("Sunday", "Tuesday", "Thursday"),               "09:00", "13:00", 30, SVC_DENT),
    (_guid("doc:saad-madi"),       NUZHA, ("Sunday", "Tuesday"),               "13:00", "16:00", 30, SVC_PSY),
    (_guid("doc:khalid-anazi"),    MANAR, ("Monday", "Wednesday"),             "14:00", "18:00", 30, SVC_PSY),
    (_guid("doc:mona-qahtani"),    NUZHA, ("Wednesday", "Thursday"),           "10:00", "13:00", 30, SVC_PSY),
    (_guid("doc:faisal-shammari"), MANAR, ("Sunday", "Thursday"),              "16:00", "20:00", 30, SVC_PSY),
    (_guid("doc:hind-ghamdi"),     NUZHA, ("Saturday", "Tuesday"),             "09:00", "12:00", 30, SVC_PSY),
    (_guid("doc:noura-madi"),      NUZHA, ("Sunday", "Monday", "Wednesday"),   "10:00", "14:00", 45, SVC_PSYC),
    (_guid("doc:sara-malki"),      MANAR, ("Tuesday", "Thursday"),             "15:00", "19:00", 45, SVC_PSYC),
    (_guid("doc:abdulrahman-yami"), NUZHA, ("Monday", "Thursday"),             "17:00", "21:00", 45, SVC_PSYC),
    (_guid("doc:lama-zahrani"),    MANAR, ("Sunday", "Wednesday"),             "09:00", "12:00", 45, SVC_PSYC),
]
SCHEDULE_VALID_FROM = _dt.date(2026, 9, 1)
SCHEDULE_VALID_TO = _dt.date(2027, 2, 28)

_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

STATUS_NAMES = {
    1: ("New", "جديد"), 2: ("Confirmed", "مؤكد"), 3: ("Arrived", "وصل"),
    4: ("NoShow", "لم يحضر"), 5: ("Completed", "مكتمل"), 6: ("Cancelled", "ملغي"),
}
_ACTIVE = (1, 2)

SEED_PATIENTS = [
    {"id": _guid("pat:omari"), "patientFullName": "محمد عبدالله العمري",
     "mobileNumber": CHANNEL_PHONE, "email": "m.alomari@example.com"},
    {"id": _guid("pat:zahrani"), "patientFullName": "فهد سعد الزهراني",
     "mobileNumber": "+966555123456", "email": ""},
    {"id": _guid("pat:mutairi"), "patientFullName": "هيفاء ناصر المطيري",
     "mobileNumber": "+966541112233", "email": "haifa@example.com"},
]

# (ref, phone, patient, doctor, branch, local date, local time, status)
SEED_BOOKINGS = [
    # THE channel phone's own upcoming appointment - what cancel/reschedule find.
    ("TNS-10001", CHANNEL_PHONE, "محمد عبدالله العمري", DR_AHMED_SAMI, MANAR,
     _dt.date(2026, 9, 30), "10:00", 1),
    # A past, completed visit for the same phone (filtered out as inactive).
    ("TNS-09874", CHANNEL_PHONE, "محمد عبدالله العمري", DR_AHMED_OTAIBI, NUZHA,
     _dt.date(2026, 8, 10), "09:30", 5),
    # Other patients holding slots, so "isBooked" filtering is exercised.
    ("TNS-10002", "+966555123456", "فهد سعد الزهراني", DR_AHMED_SAMI, NUZHA,
     _dt.date(2026, 9, 29), "16:00", 2),
    ("TNS-10003", "+966541112233", "هيفاء ناصر المطيري", DR_AHMED_OTAIBI, NUZHA,
     _dt.date(2026, 9, 29), "09:00", 1),
]
FIRST_NEW_REF = 10101


# ==========================================================
# Helpers
# ==========================================================

def _utc_iso(value: _dt.datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat()


def _local_to_utc_iso(day: _dt.date, hhmm: str) -> str:
    hour, minute = (int(p) for p in hhmm.split(":"))
    return _utc_iso(_REAL_DATETIME(day.year, day.month, day.day, hour, minute, tzinfo=CLINIC_TZ))


def _parse_instant(value):
    """API convention: every timestamp is UTC; a naive one IS UTC."""
    if value is None or value == "":
        return None
    if isinstance(value, _REAL_DATETIME):
        parsed = value
    else:
        try:
            parsed = _REAL_DATETIME.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return _REAL_DATETIME.fromtimestamp(parsed.timestamp(), UTC)


def _parse_day(value):
    parsed = _parse_instant(value)
    return parsed.date() if parsed else None


def _digits(phone) -> str:
    return re.sub(r"\D", "", str(phone or ""))


def _same_phone(a, b) -> bool:
    da, db = _digits(a), _digits(b)
    return bool(da and db) and da[-9:] == db[-9:]


def _is_ar(language) -> bool:
    return bool(language) and not str(language).lower().startswith("en")


class FakeResponse:
    """Just enough of requests.Response for the webhook callers."""

    def __init__(self, status_code=200, body=None, url=""):
        self.status_code = status_code
        self._body = body if body is not None else {"ok": True}
        self.url = url
        self.headers = {"Content-Type": "application/json"}
        self.text = json.dumps(self._body, ensure_ascii=False)
        self.ok = status_code < 400

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code} from fake {self.url}", response=self)


class _RequestsProxy:
    """Stands in for the `requests` module inside one project module.

    Exceptions and everything else resolve to the real module, so
    `except requests.Timeout` keeps working; the verbs go to `handler`."""

    def __init__(self, real, handler, channel):
        self._real = real
        self._handler = handler
        self._channel = channel

    def __getattr__(self, name):
        return getattr(self._real, name)

    def request(self, method, url, **kwargs):
        return self._handler(self._channel, method.upper(), url, kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def put(self, url, **kwargs):
        return self.request("PUT", url, **kwargs)

    def patch(self, url, **kwargs):
        return self.request("PATCH", url, **kwargs)

    def delete(self, url, **kwargs):
        return self.request("DELETE", url, **kwargs)

    def head(self, url, **kwargs):
        return self.request("HEAD", url, **kwargs)


class _FakeSMTP:
    def __init__(self, hospital, host, port=0, timeout=None, **kwargs):
        self._hospital = hospital
        self.host, self.port = host, port

    def starttls(self, *a, **k):
        return (220, b"ok")

    def login(self, *a, **k):
        return (235, b"ok")

    def sendmail(self, from_addr, to_addrs, msg, *a, **k):
        self._hospital.outbound.append({
            "channel": "smtp", "host": self.host, "from": from_addr,
            "to": list(to_addrs) if not isinstance(to_addrs, str) else [to_addrs],
            "message": msg,
        })
        return {}

    send_message = sendmail

    def quit(self):
        return (221, b"bye")


class _FakeSmtplib:
    def __init__(self, hospital, real):
        self._hospital = hospital
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def SMTP(self, host="", port=0, *a, **k):
        return _FakeSMTP(self._hospital, host, port)

    def SMTP_SSL(self, host="", port=0, *a, **k):
        return _FakeSMTP(self._hospital, host, port)


class _FakeEmbeddings:
    """Never used by the patched rag.search_with_scores; installed so any
    other path that asks rag for its embeddings model gets no network."""

    @staticmethod
    def _vec(text):
        vec = [0.0] * 64
        for token in _kw_tokens(text):
            vec[hash(token) % 64] += 1.0
        return vec

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


_AR_DIACRITICS = re.compile(r"[ً-ٰٟـ]")
_KW_STOPWORDS = {
    "وش", "ايش", "شو", "عندكم", "عندك", "كيف", "هل", "ما", "في", "من", "على", "عن", "الى", "إلى",
    "ابي", "ابغى", "اريد", "لو", "يا", "مع", "هذا", "هذه", "the", "and", "what", "is", "are",
    "do", "you", "your", "for", "how", "can", "طيب", "بس", "لكم", "لي",
}


def _kw_tokens(text) -> set:
    text = _AR_DIACRITICS.sub("", str(text or "").lower())
    text = (text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
            .replace("ة", "ه").replace("ى", "ي"))
    tokens = set()
    for raw in re.findall(r"[\w]+", text):
        token = raw
        for prefix in ("وال", "بال", "فال", "كال", "لل", "ال"):
            if token.startswith(prefix) and len(token) - len(prefix) >= 3:
                token = token[len(prefix):]
                break
        if token.startswith("و") and len(token) >= 5:
            token = token[1:]
        if len(token) >= 3 and token not in _KW_STOPWORDS:
            tokens.add(token)
    return tokens


def _endpoint(fn):
    """Record every call to a faked api function as (name, arguments)."""

    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            arguments = {k: copy.deepcopy(v) for k, v in bound.arguments.items() if k != "self"}
        except TypeError:
            arguments = {"args": args, "kwargs": kwargs}
        self.calls.append((fn.__name__, arguments))
        return fn(self, *args, **kwargs)

    wrapper._fake_endpoint = True
    return wrapper


class _Patcher:
    """monkeypatch when given one, otherwise a manual undo stack."""

    def __init__(self, monkeypatch=None):
        self._mp = monkeypatch
        self._undo = []

    def setattr(self, target, name, value):
        if self._mp is not None:
            self._mp.setattr(target, name, value, raising=False)
            return
        had = name in getattr(target, "__dict__", {}) or hasattr(target, name)
        old = getattr(target, name, None)
        setattr(target, name, value)
        self._undo.append((target, name, had, old))

    def undo(self):
        while self._undo:
            target, name, had, old = self._undo.pop()
            if had:
                setattr(target, name, old)
            else:
                try:
                    delattr(target, name)
                except AttributeError:
                    pass


# The api.py functions this fake implements - every one tools.py calls.
FAKED_API_FUNCTIONS = (
    "get_bookings_by_ref", "get_bookings_by_phone", "cancel_booking_by_guid",
    "authentica_send_otp", "authentica_verify_otp",
    "get_specialties", "get_doctors", "get_branches",
    "get_doctor_schedule", "get_doctor_schedule_slots", "get_doctor_fees",
    "get_services", "get_patient_info", "reschedule_booking",
    "create_booking", "get_booking_by_id",
)

# api.py's transport layer - blocked, so an api function added later and
# not faked here fails loudly (structured error, recorded) instead of
# going to the network.
BLOCKED_API_HELPERS = ("_request_with_retry", "_post_json", "_put_json", "_post_bookings")

# Tools whose effect is a signal to n8n rather than an HTTP call of their
# own: their payload and result are recorded in FakeHospital.signals.
SIGNAL_TOOLS = ("request_human_handoff", "share_branch_location", "send_complaint_email")


# ==========================================================
# The fake
# ==========================================================

class FakeHospital:
    """Stateful fake of the clinic's booking/doctors API.

    Public state (all reset by reset()):
      calls     - [(api_function_name, arguments), ...] every faked api call
      bookings  - the live booking rows (raw API shape)
      patients  - the registered patients
      outbound  - webhook/SMTP deliveries (complaint, progress)
      signals   - [{"tool", "args", "result"}, ...] handoff/location/complaint tool calls
      otp_sent  - phones an OTP was "sent" to
      blocked   - api transport helpers that something tried to use
    """

    def __init__(self):
        self._patcher = None
        self._api = None
        self._tools = None
        self.installed = False
        self.reset(_clear_project_state=False)

    # ---------------------------------------------------------- lifecycle

    def reset(self, _clear_project_state: bool = True):
        """Back to the seed: bookings, patients, recorded calls, the clock -
        and the project's own per-session stores (booking sessions, OTPs,
        doctor-list cache, graph retry counters), so each scripted
        conversation starts clean."""

        _CLOCK["now"] = FAKE_NOW.astimezone(UTC)
        self.calls = []
        self.outbound = []
        self.signals = []
        self.otp_sent = []
        self.blocked = []
        self._next_ref = FIRST_NEW_REF
        self.patients = copy.deepcopy(SEED_PATIENTS)
        self.bookings = []
        for ref, phone, patient, doctor_id, branch_id, day, hhmm, status in SEED_BOOKINGS:
            start = _local_to_utc_iso(day, hhmm)
            slot = self._slot_template(doctor_id, branch_id, start)
            self.bookings.append(self._booking_row(
                ref=ref, booking_id=_guid(f"booking:{ref}"), patient=patient, phone=phone,
                email=next((p["email"] for p in SEED_PATIENTS if _same_phone(p["mobileNumber"], phone)), ""),
                slot=slot, start=start, end=slot["slotEnd"] if slot else start, status=status,
            ))

        if _clear_project_state:
            self._clear_project_state()
        return self

    def _clear_project_state(self):
        tools = self._tools or sys.modules.get("tools")
        if tools is not None:
            for name in ("_BOOKING_SESSIONS", "_otp_storage", "_DOCTOR_LIST_CACHE"):
                store = getattr(tools, name, None)
                if isinstance(store, dict):
                    store.clear()
        graph = sys.modules.get("graph")
        if graph is not None:
            for name in ("_VERIFIER_TOOL_RETRIES", "_CLAIM_GATE_RETRIES"):
                store = getattr(graph, name, None)
                if isinstance(store, dict):
                    store.clear()

    def set_now(self, value):
        """Move the fake clock (aware datetime, or naive = clinic local)."""
        if value.tzinfo is None:
            value = value.replace(tzinfo=CLINIC_TZ)
        _CLOCK["now"] = value.astimezone(UTC)

    @property
    def now(self):
        return _CLOCK["now"].astimezone(CLINIC_TZ)

    def uninstall(self):
        if self._patcher is not None:
            self._patcher.undo()
        self.installed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.uninstall()
        return False

    # ---------------------------------------------------------- conveniences

    def make_state(self, session_id: str = None, messages=None, channel_phone: str = CHANNEL_PHONE,
                   target_language: str = "ar", greeted: bool = True) -> dict:
        """A state dict shaped like the graph's AgentState after load_config,
        for invoking tools directly (tests) - templates built through
        config.get_messages with the TENANT override, as graph.load_config does."""

        import config
        templates = config.get_messages(TENANT["client_id"], client_row_override=TENANT)
        return {
            "client_id": TENANT["client_id"],
            "session_id": session_id or f"{_digits(channel_phone)}+{TENANT['client_id']}",
            "channel_phone": channel_phone,
            "bsuid": None,
            "raw_client_config": TENANT,
            "templates": templates,
            "system_prompt": None,
            "messages": list(messages or []),
            "greeted": greeted,
            "target_language": target_language,
        }

    def booking(self, ref: str) -> dict:
        return next((b for b in self.bookings if b["bookingRefNum"].lower() == str(ref).lower()), None)

    def calls_to(self, name: str) -> list:
        return [args for fn, args in self.calls if fn == name]

    # ---------------------------------------------------------- data model

    def _schedule_rows(self):
        rows = []
        for doctor_id, branch_id, days, start, end, minutes, service_id in SCHEDULES:
            rows.append({
                "id": _guid(f"schedule:{doctor_id}:{branch_id}"),
                "doctor_id": doctor_id, "branch_id": branch_id, "days": days,
                "start": start, "end": end, "minutes": minutes, "service_id": service_id,
                "space_id": _guid(f"space:{branch_id}:{doctor_id}"),
            })
        return rows

    def _all_slots(self):
        """Every generated slot in the window, booked or not (raw template)."""
        slots = []
        for row in self._schedule_rows():
            for offset in range(SLOT_WINDOW_DAYS):
                day = FAKE_TODAY + _dt.timedelta(days=offset)
                if _WEEKDAYS[day.weekday()] not in row["days"]:
                    continue
                sh, sm = (int(p) for p in row["start"].split(":"))
                eh, em = (int(p) for p in row["end"].split(":"))
                cursor = _REAL_DATETIME(day.year, day.month, day.day, sh, sm, tzinfo=CLINIC_TZ)
                stop = _REAL_DATETIME(day.year, day.month, day.day, eh, em, tzinfo=CLINIC_TZ)
                step = _dt.timedelta(minutes=row["minutes"])
                while cursor + step <= stop:
                    slots.append({"row": row, "start": _utc_iso(cursor), "end": _utc_iso(cursor + step)})
                    cursor += step
        # Seeded bookings outside the window (past visits) still need a template.
        return slots

    def _slot_template(self, doctor_id, branch_id, start_iso):
        """The raw slot fields for one (doctor, branch, start), even outside
        the generated window (used to shape seeded past bookings)."""
        row = next((r for r in self._schedule_rows()
                    if r["doctor_id"] == doctor_id and r["branch_id"] == branch_id), None)
        if row is None:
            return None
        start = _parse_instant(start_iso)
        end = start + _dt.timedelta(minutes=row["minutes"])
        return {"row": row, "slotStart": _utc_iso(start), "slotEnd": _utc_iso(end)}

    def _booked_instants(self, exclude_booking_id=None):
        taken = set()
        for b in self.bookings:
            if b["status"] in _ACTIVE and b["id"] != exclude_booking_id:
                instant = _parse_instant(b["bookingTimeFrom"])
                if instant:
                    taken.add((b["doctorId"], instant.timestamp()))
        return taken

    def _slot_item(self, slot, taken, language):
        row = slot["row"]
        doctor, service, branch = _D[row["doctor_id"]], _SV[row["service_id"]], self._branch(row["branch_id"])
        specialty = _S[doctor["specialty"]]
        booked = (row["doctor_id"], _parse_instant(slot["start"]).timestamp()) in taken
        ar = _is_ar(language)
        return {
            "slotStart": slot["start"],
            "slotEnd": slot["end"],
            "isBooked": booked,
            "isAtDailyCapacity": False,
            "doctorId": row["doctor_id"],
            "doctorName": doctor["ar"] if ar else doctor["en"],
            "branchId": row["branch_id"],
            "branchName": branch["altName"] if ar else branch["name"],
            "serviceId": service["id"],
            "serviceName": service["ar"] if ar else service["en"],
            "serviceAltName": service["ar"],
            "servicePrice": service["price"],
            "specialtyId": specialty["id"],
            "scheduleId": row["id"],
            "spaceId": row["space_id"],
        }

    def _open_slots(self, doctor_id, branch_ids=None, start=None, end=None):
        taken = self._booked_instants()
        out = []
        for slot in self._all_slots():
            row = slot["row"]
            if row["doctor_id"] != doctor_id:
                continue
            if branch_ids and row["branch_id"] not in branch_ids:
                continue
            instant = _parse_instant(slot["start"])
            if (row["doctor_id"], instant.timestamp()) in taken:
                continue
            if start and instant < start:
                continue
            if end and instant > end:
                continue
            out.append(slot)
        return out

    @staticmethod
    def _branch(branch_id):
        return next(b for b in BRANCHES if b["id"] == branch_id)

    def _booking_row(self, ref, booking_id, patient, phone, email, slot, start, end, status):
        row = (slot or {}).get("row") or {}
        doctor = _D.get(row.get("doctor_id"), {})
        service = _SV.get(row.get("service_id"), {})
        specialty = _S.get(doctor.get("specialty"), {})
        return {
            "id": booking_id,
            "bookingRefNum": ref,
            "patientFullName": patient,
            "mobileNumber": phone,
            "email": email or "",
            "status": status,
            "doctorId": row.get("doctor_id"),
            "branchId": row.get("branch_id"),
            "serviceId": row.get("service_id"),
            "specialtyId": specialty.get("id"),
            "servicePrice": service.get("price"),
            "doctorScheduleId": row.get("id"),
            "spaceId": row.get("space_id"),
            "bookingTimeFrom": _utc_iso(_parse_instant(start)),
            "bookingTimeTo": _utc_iso(_parse_instant(end)),
            "createdAt": _utc_iso(_CLOCK["now"]),
        }

    def _localized_booking(self, b, language):
        ar = _is_ar(language)
        doctor = _D.get(b["doctorId"], {})
        service = _SV.get(b["serviceId"], {})
        specialty = next((s for s in SPECIALTIES if s["id"] == b["specialtyId"]), {})
        branch = next((x for x in BRANCHES if x["id"] == b["branchId"]), {})
        en_status, ar_status = STATUS_NAMES.get(b["status"], ("", ""))
        item = dict(b)
        item.update({
            "statusName": ar_status if ar else en_status,
            "doctorName": doctor.get("ar" if ar else "en"),
            "branchName": branch.get("altName" if ar else "name"),
            "serviceName": service.get("ar" if ar else "en"),
            "specialtyName": specialty.get("altName" if ar else "name"),
        })
        return item

    def _result(self, success, status_code=None, data=None, error=None, details=None):
        api = self._api
        if api is not None and hasattr(api, "_result"):
            try:
                return api._result(success, status_code, copy.deepcopy(data), error, details=details)
            except TypeError:
                return api._result(success, status_code, copy.deepcopy(data), error)
        return {"success": success, "status_code": status_code, "data": copy.deepcopy(data),
                "error": error, "details": details or []}

    def _page(self, items, page_size):
        items = list(items)
        size = int(page_size or 200)
        return {"items": items[:size], "totalCount": len(items), "pageNumber": 1, "pageSize": size}

    def _refused(self, message, prop=""):
        # A 200 with isSuccess=false, as api._post_json reports it.
        body = {"data": None, "statusCode": 400, "isSuccess": False,
                "messages": [{"prop": prop, "message": message}]}
        return self._result(False, 200, data=body, error="api_reported_failure")

    def _invalid(self, prop, message):
        return self._result(False, 400, error="validation_error",
                            details=[{"field": prop, "message": message}])

    # ---------------------------------------------------------- GuestBookings

    @_endpoint
    def get_bookings_by_ref(self, base_url, ref_number, language=None, client_id=None):
        wanted = str(ref_number or "").replace(" ", "").lower()
        items = [self._localized_booking(b, language) for b in self.bookings
                 if b["bookingRefNum"].lower() == wanted]
        return self._result(True, 200, data=self._page(items, 1000))

    @_endpoint
    def get_bookings_by_phone(self, base_url, phone, language=None, client_id=None,
                              page_size=1000, status_list=None):
        items = [self._localized_booking(b, language) for b in self.bookings
                 if _same_phone(b["mobileNumber"], phone)
                 and (not status_list or b["status"] in status_list)]
        items.sort(key=lambda b: b["bookingTimeFrom"])
        return self._result(True, 200, data=self._page(items, page_size))

    @_endpoint
    def cancel_booking_by_guid(self, base_url, booking_guid, client_id=None):
        booking = next((b for b in self.bookings if b["id"] == str(booking_guid)), None)
        if booking is None:
            return self._result(False, 400, error="validation_error")
        if booking["status"] not in _ACTIVE:
            body = {"data": None, "statusCode": 400, "isSuccess": False,
                    "messages": [{"prop": "", "message": "Booking can not be cancelled"}]}
            return self._result(False, 200, data=body, error="api_reported_failure")
        booking["status"] = 6
        body = {"data": booking["id"], "statusCode": 200, "isSuccess": True, "messages": []}
        return self._result(True, 200, data=body)

    @_endpoint
    def get_booking_by_id(self, base_url, booking_id, client_id=None):
        booking = next((b for b in self.bookings if b["id"] == str(booking_id)), None)
        if booking is None:
            return self._refused("Booking not found")
        return self._result(True, 200, data=self._localized_booking(booking, None))

    @_endpoint
    def create_booking(self, base_url, patient_full_name, mobile_number, branch_id, doctor_id,
                       service_id, service_price, booking_time_from, booking_time_to,
                       specialty_id, doctor_schedule_id, space_id, email="", client_id=None):
        if not (patient_full_name or "").strip():
            return self._invalid("PatientFullName", "PatientFullName Required")
        if not re.fullmatch(r"\+\d{8,15}", str(mobile_number or "")):
            return self._invalid("MobileNumber", "Mobile Number Not Valid")
        start, end = _parse_instant(booking_time_from), _parse_instant(booking_time_to)
        if start is None or end is None:
            return self._invalid("BookingTimeFrom", "Booking Time Not Valid")
        if end <= start:
            return self._invalid("BookingTimeTo", "Booking Time To Must Be Greater Than Time From")
        slot = next((s for s in self._all_slots()
                     if s["row"]["doctor_id"] == doctor_id and s["row"]["branch_id"] == branch_id
                     and _parse_instant(s["start"]) == start), None)
        if slot is None or (doctor_id, start.timestamp()) in self._booked_instants():
            return self._refused("This slot is already booked Please select another time")

        ref = f"TNS-{self._next_ref}"
        self._next_ref += 1
        booking_id = _guid(f"booking:{ref}")
        template = {"row": slot["row"]}
        self.bookings.append(self._booking_row(
            ref=ref, booking_id=booking_id, patient=patient_full_name.strip(), phone=mobile_number,
            email=email, slot=template, start=start, end=end, status=1,
        ))
        if not any(_same_phone(p["mobileNumber"], mobile_number)
                   and p["patientFullName"] == patient_full_name.strip() for p in self.patients):
            self.patients.append({"id": _guid(f"pat:{ref}"), "patientFullName": patient_full_name.strip(),
                                  "mobileNumber": mobile_number, "email": email or ""})
        return self._result(True, 200, data=booking_id)

    @_endpoint
    def reschedule_booking(self, base_url, booking_id, new_from, new_to, client_id=None):
        booking = next((b for b in self.bookings if b["id"] == str(booking_id)), None)
        if booking is None:
            return self._result(False, 404, error="endpoint_not_found")
        if booking["status"] not in _ACTIVE:
            return self._refused("Booking can not be updated")
        start, end = _parse_instant(new_from), _parse_instant(new_to)
        if start is None or end is None or end <= start:
            return self._invalid("ToBookingTime", "Booking Time To Must Be Greater Than Time From")
        slot = next((s for s in self._all_slots()
                     if s["row"]["doctor_id"] == booking["doctorId"]
                     and _parse_instant(s["start"]) == start), None)
        if slot is None:
            return self._refused("No doctor schedule at the requested time")
        if (booking["doctorId"], start.timestamp()) in self._booked_instants(exclude_booking_id=booking["id"]):
            return self._refused("This slot is already booked Please select another time")
        row = slot["row"]
        booking.update({
            "bookingTimeFrom": _utc_iso(start), "bookingTimeTo": _utc_iso(end),
            "branchId": row["branch_id"], "doctorScheduleId": row["id"], "spaceId": row["space_id"],
        })
        return self._result(True, 200, data={"id": booking["id"]})

    # ---------------------------------------------------------- OTP

    @_endpoint
    def authentica_send_otp(self, phone):
        self.otp_sent.append(phone)
        return self._result(True, 200, data={"success": True, "message": "OTP sent successfully"})

    @_endpoint
    def authentica_verify_otp(self, phone, otp, email=""):
        verified = str(otp or "").strip() == TEST_OTP
        body = {"success": verified, "message": "OTP verified" if verified else "Invalid OTP"}
        return self._result(verified, 200, data=body)

    # ---------------------------------------------------------- Doctors API

    @_endpoint
    def get_specialties(self, base_url, page_size=200, client_id=None, language=None):
        items = [dict(s, formatedName=s["name"], isActive=True) for s in SPECIALTIES]
        return self._result(True, 200, data=self._page(items, page_size))

    @_endpoint
    def get_doctors(self, base_url, specialty_ids=None, branch_ids=None, service_ids=None,
                    has_published_service=True, has_service_schedule=True,
                    intersection_start=None, intersection_end=None, page_size=200,
                    client_id=None, language=None):
        rows = self._schedule_rows()
        start = _parse_instant(intersection_start)
        end = _parse_instant(intersection_end)
        items = []
        for doctor in DOCTORS:
            specialty = _S[doctor["specialty"]]
            if specialty_ids and specialty["id"] not in specialty_ids:
                continue
            doctor_branches = {r["branch_id"] for r in rows if r["doctor_id"] == doctor["id"]}
            if branch_ids and not (doctor_branches & set(branch_ids)):
                continue
            if service_ids and not (set(doctor["services"]) & set(service_ids)):
                continue
            # Literal equality filters, like the real API (see api.get_doctors).
            if has_published_service is not None and bool(doctor["services"]) != has_published_service:
                continue
            if has_service_schedule is not None and bool(doctor_branches) != has_service_schedule:
                continue
            open_slots = self._open_slots(doctor["id"], branch_ids, start, end)
            items.append({
                "id": doctor["id"],
                "name": doctor["en"],
                "formatedName": doctor["en"],
                "altName": doctor["ar"],
                "specialtyId": specialty["id"],
                "specialtyName": specialty["name"],
                "specialtyAltName": specialty["altName"],
                "degreeName": doctor["degree"][0],
                "degreeAltName": doctor["degree"][1],
                "about": doctor["about"],
                "hasPublishedService": bool(doctor["services"]),
                "hasServiceSchedule": bool(doctor_branches),
                "hasSlots": bool(open_slots),
                "isActive": True,
            })
        return self._result(True, 200, data=self._page(items, page_size))

    @_endpoint
    def get_branches(self, base_url, search_query=None, page_size=200, client_id=None, language=None):
        items = [dict(b) for b in BRANCHES]
        if search_query:
            q = str(search_query).strip().lower()
            items = [b for b in items if q in b["name"].lower() or q in b["altName"]]
        return self._result(True, 200, data=self._page(items, page_size))

    @_endpoint
    def get_doctor_schedule(self, base_url, doctor_ids, branch_ids=None, effective_date=None,
                            page_size=50, client_id=None, language=None, include_future=False):
        ar = _is_ar(language)
        effective = _dt.date.fromisoformat(str(effective_date)[:10]) if effective_date else None
        items = []
        for row in self._schedule_rows():
            if doctor_ids and row["doctor_id"] not in doctor_ids:
                continue
            if branch_ids and row["branch_id"] not in branch_ids:
                continue
            if effective and SCHEDULE_VALID_TO < effective:
                continue
            if effective and not include_future and SCHEDULE_VALID_FROM > effective:
                continue
            doctor, branch, service = _D[row["doctor_id"]], self._branch(row["branch_id"]), _SV[row["service_id"]]
            items.append({
                "id": row["id"],
                "doctorId": row["doctor_id"],
                "doctorName": doctor["ar"] if ar else doctor["en"],
                "branchId": row["branch_id"],
                "branchName": branch["altName"] if ar else branch["name"],
                "branchAltName": branch["altName"],
                "serviceId": service["id"],
                "serviceName": service["ar"] if ar else service["en"],
                "serviceAltName": service["ar"],
                "spaceId": row["space_id"],
                "recurringDaysNames": list(row["days"]),
                # Date part = validity range, time part = working hours (UTC).
                "fromDateTime": _local_to_utc_iso(SCHEDULE_VALID_FROM, row["start"]),
                "toDateTime": _local_to_utc_iso(SCHEDULE_VALID_TO, row["end"]),
                "slotDurationInMinutes": row["minutes"],
                "maxNoOfCases": 20,
            })
        return self._result(True, 200, data=self._page(items, page_size))

    @_endpoint
    def get_doctor_schedule_slots(self, base_url, doctor_ids, from_date, to_date, is_booked=False,
                                  branch_ids=None, page_size=200, client_id=None, language=None):
        start, end = _parse_instant(from_date), _parse_instant(to_date)
        taken = self._booked_instants()
        items = []
        for slot in self._all_slots():
            row = slot["row"]
            if doctor_ids and row["doctor_id"] not in doctor_ids:
                continue
            if branch_ids and row["branch_id"] not in branch_ids:
                continue
            instant = _parse_instant(slot["start"])
            if start and instant < start:
                continue
            if end and instant > end:
                continue
            item = self._slot_item(slot, taken, language)
            if is_booked is not None and item["isBooked"] != bool(is_booked):
                continue
            items.append(item)
        items.sort(key=lambda i: i["slotStart"])
        return self._result(True, 200, data=self._page(items, page_size))

    @_endpoint
    def get_doctor_fees(self, base_url, doctor_ids, is_published=True, page_size=1000,
                        client_id=None, language=None):
        ar = _is_ar(language)
        items = []
        for doctor in DOCTORS:
            if doctor_ids and doctor["id"] not in doctor_ids:
                continue
            for service_id in doctor["services"]:
                service = _SV[service_id]
                items.append({
                    "id": _guid(f"docsvc:{doctor['id']}:{service_id}"),
                    "doctorId": doctor["id"], "serviceId": service_id,
                    "serviceName": service["ar"] if ar else service["en"],
                    "price": service["price"], "isPublished": True,
                })
        return self._result(True, 200, data=self._page(items, page_size))

    @_endpoint
    def get_services(self, base_url, branch_ids=None, is_published=True, page_size=500,
                     client_id=None, language=None):
        ar = _is_ar(language)
        rows = self._schedule_rows()
        items = []
        for service in SERVICES:
            providers = [d for d in DOCTORS if service["id"] in d["services"]]
            at_branches = {r["branch_id"] for r in rows for d in providers if r["doctor_id"] == d["id"]}
            if branch_ids and not (at_branches & set(branch_ids)):
                continue
            items.append({
                "id": service["id"],
                "name": service["ar"] if ar else service["en"],
                "description": service["description"],
                "altDescription": service["altDescription"],
                "price": service["price"],
                "specialtyId": _S[service["specialty"]]["id"],
                "branchIds": sorted(at_branches),
                "isPublished": True,
            })
        return self._result(True, 200, data=self._page(items, page_size))

    @_endpoint
    def get_patient_info(self, base_url, mobile_number, page_size=1000, client_id=None):
        items = [dict(p) for p in self.patients if _same_phone(p["mobileNumber"], mobile_number)]
        return self._result(True, 200, data=self._page(items, page_size))

    # ---------------------------------------------------------- blocked transport

    def _blocked_helper(self, name):
        def blocked(*args, **kwargs):
            self.blocked.append((name, args, kwargs))
            self.calls.append((name, {"args": args, "kwargs": kwargs, "blocked": True}))
            if name == "_request_with_retry":
                import requests
                return None, False, requests.ConnectionError("fake_hospital: network blocked")
            return self._result(False, error="fake_hospital_unfaked_endpoint")
        blocked.__name__ = f"fake_blocked{name}"
        return blocked

    def _http(self, channel, method, url, kwargs):
        """Handler for the `requests` stand-ins in tools/progress."""
        payload = kwargs.get("json")
        self.outbound.append({"channel": channel, "method": method, "url": url,
                              "json": copy.deepcopy(payload)})
        if channel == "api":
            import requests
            raise requests.ConnectionError(f"fake_hospital: api.requests blocked ({method} {url})")
        return FakeResponse(200, {"ok": True, "received": True}, url)

    # ---------------------------------------------------------- RAG

    def _rag_search_with_scores(self, file_path, query, top_k=4):
        self.calls.append(("rag.search_with_scores", {"file_path": file_path, "query": query, "top_k": top_k}))
        rag = sys.modules.get("rag")
        floor = getattr(rag, "RELEVANCE_FLOOR", 0.32)
        if not file_path or not os.path.exists(file_path):
            return []
        with open(file_path, encoding="utf-8") as handle:
            text = handle.read()
        chunks = rag._chunk_text(text) if rag is not None and hasattr(rag, "_chunk_text") else \
            [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
        wanted = _kw_tokens(query)
        scored = []
        for index, chunk in enumerate(chunks):
            have = _kw_tokens(chunk)
            hits = len(wanted & have)
            if hits:
                scored.append((hits / max(1, len(wanted)), -index, chunk))
        if scored:
            scored.sort(reverse=True)
            # Reported at/above the floor: these are the "relevant" passages.
            return [(chunk, max(floor, round(score, 3))) for score, _i, chunk in scored[:top_k]]
        # Nothing shares a word with the question: behave like rag's own
        # embeddings-unavailable fallback (first chunks, unranked).
        return [(chunk, floor) for chunk in chunks[:top_k]]

    # ---------------------------------------------------------- signal tools

    def _wrap_signal_tool(self, name, original):
        @functools.wraps(original)
        def recorded(*args, **kwargs):
            result = original(*args, **kwargs)
            args_seen = {k: v for k, v in kwargs.items() if k != "state"}
            self.signals.append({"tool": name, "args": copy.deepcopy(args_seen),
                                 "result": copy.deepcopy(result)})
            return result
        return recorded


# ==========================================================
# install()
# ==========================================================

def install(monkeypatch=None, patch_graph: bool = True) -> FakeHospital:
    """Apply every patch and return the FakeHospital.

    `monkeypatch`: pytest's fixture (patches undone automatically after
    the test), or None for a script - then call `hospital.uninstall()`,
    or use the returned object as a context manager.

    The project modules are imported from sys.path here, not at module
    import time, so whichever checkout is first on sys.path is the one
    patched."""

    import api
    import config
    import rag
    import requests as _requests
    import smtplib as _smtplib
    import tools

    hospital = FakeHospital()
    hospital._api, hospital._tools = api, tools
    patcher = _Patcher(monkeypatch)
    hospital._patcher = patcher

    # 1. Every api.py function tools.py calls -> stateful fake.
    for name in FAKED_API_FUNCTIONS:
        patcher.setattr(api, name, getattr(hospital, name))

    # 2. api.py's transport layer -> blocked (backstop for anything unfaked).
    for name in BLOCKED_API_HELPERS:
        if hasattr(api, name):
            patcher.setattr(api, name, hospital._blocked_helper(name))
    patcher.setattr(api, "requests", _RequestsProxy(_requests, hospital._http, "api"))
    patcher.setattr(api, "AUTHENTICA_BASE_URL", FAKE_AUTHENTICA_BASE_URL)

    # 3. Direct HTTP/SMTP in tools.py (complaint webhook + SMTP fallback).
    patcher.setattr(tools, "requests", _RequestsProxy(_requests, hospital._http, "tools"))
    patcher.setattr(tools, "smtplib", _FakeSmtplib(hospital, _smtplib))
    patcher.setattr(tools, "COMPLAINT_WEBHOOK_URL", FAKE_COMPLAINT_WEBHOOK_URL)
    patcher.setattr(config, "COMPLAINT_WEBHOOK_URL", FAKE_COMPLAINT_WEBHOOK_URL)

    # 4. OTP through the (faked) Authentica path, so sends are recorded;
    #    the dummy path's code is pinned to the same value regardless.
    patcher.setattr(tools, "OTP_PROVIDER", "authentica")
    patcher.setattr(tools, "TEST_OTP", TEST_OTP)

    # 5. Frozen clock.
    patcher.setattr(tools, "datetime", FrozenDatetime)
    patcher.setattr(tools, "date", FrozenDate)
    graph = sys.modules.get("graph")
    if patch_graph and graph is not None:
        if getattr(graph, "datetime", None) is _REAL_DATETIME:
            patcher.setattr(graph, "datetime", FrozenDatetime)
        if getattr(graph, "date", None) is _REAL_DATE:
            patcher.setattr(graph, "date", FrozenDate)

    # 6. RAG without embeddings.
    patcher.setattr(rag, "search_with_scores", hospital._rag_search_with_scores)
    patcher.setattr(rag, "_get_embeddings_model", lambda: _FakeEmbeddings())
    if isinstance(getattr(rag, "_CACHE", None), dict):
        rag._CACHE.clear()

    # 7. Tenant config: TENANT is the only tenant, no env URL override wins.
    patcher.setattr(config, "_ENV_BASE_URL_OVERRIDE", None)
    patcher.setattr(config, "_ENV_DOCTORS_BASE_URL_OVERRIDE", None)
    patcher.setattr(config, "BASE_URL", FAKE_BOOKINGS_BASE_URL)
    patcher.setattr(config, "_all_client_configs", lambda: {TENANT["client_id"]: TENANT})

    # 8. The "please wait" webhook (progress.py), if the checkout has it.
    try:
        import progress
    except Exception:
        progress = None
    if progress is not None and hasattr(progress, "requests"):
        patcher.setattr(progress, "requests", _RequestsProxy(_requests, hospital._http, "progress"))
        if hasattr(progress, "config") and getattr(progress.config, "PROGRESS_WEBHOOK_URL", None):
            patcher.setattr(progress.config, "PROGRESS_WEBHOOK_URL", FAKE_PROGRESS_WEBHOOK_URL)

    # 9. Signal tools (handoff / location / complaint): record payload + result.
    for name in SIGNAL_TOOLS:
        tool_obj = getattr(tools, name, None)
        func = getattr(tool_obj, "func", None)
        if func is not None:
            patcher.setattr(tool_obj, "func", hospital._wrap_signal_tool(name, func))

    hospital.installed = True
    hospital.reset()
    return hospital
