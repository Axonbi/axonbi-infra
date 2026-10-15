"""
PATIENT DATA REACHES THE MODEL AS PLACEHOLDERS (per client, `pii_tokens`).

The patient's name, phone numbers, email, booking references and OTP codes
are replaced by tokens - ⟦NAME_1⟧, ⟦PHONE_1⟧, ⟦EMAIL_1⟧, ⟦REF_1⟧, ⟦OTP_1⟧ -
in the COPY of the conversation sent to the model, and put back in what the
model returns (its reply text and its tool-call arguments) the moment it
returns. Everything else - the graph state, the tools, the verifiers, the
clinic's system, the patient - only ever sees the real values.

Masking only the outgoing copy, never the stored state, is deliberate:
tools and verifiers read `state["messages"]` directly (the review card,
the name-parts check, the phone comparison), and would break on tokens.

The real values live here, in memory, per conversation, and are dropped
after `_TTL_SECONDS` without a message.

How a value becomes known:
  - the WhatsApp number, at the start of every turn (`begin_turn`);
  - tool results and tool-call arguments, by field name (`_FIELD_KINDS`);
  - the patient's own messages: phone/email/reference by shape; a code
    after `send_otp`; the whole message when the assistant had just asked
    for their name.
Once known, a value is replaced wherever it appears: system prompt,
history, tool results, the understanding prompt.

With `pii_debug` on in the client's n8n row (a test flow), every model
call also logs exactly what the model received - tokens, no real values.

One log line per model call - counts only, never a value:
  pii[<session>]: masked=4 (NAME×1, PHONE×2, REF×1) unmasked=3 unknown=0 leaks=0
`leaks` counts a phone/email found in a tool result that no field name
registered - masked anyway, and logged as an ERROR so the gap is visible.
"""

import contextvars
import json
import logging
import re
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_TTL_SECONDS = 2 * 3600

_FIELD_KINDS = {
    "patientfullname": "NAME", "patient_full_name": "NAME", "patientname": "NAME",
    "patient_name": "NAME",
    "mobilenumber": "PHONE", "mobile_number": "PHONE", "patientmobile": "PHONE",
    "phone": "PHONE", "provided_phone": "PHONE", "channel_phone": "PHONE",
    "booking_phone": "PHONE",
    "email": "EMAIL", "patientemail": "EMAIL",
    "ref": "REF", "booking_ref": "REF", "bookingrefnum": "REF", "ref_number": "REF",
    "otp": "OTP",
}

_TOKEN_RE = re.compile(r"⟦(NAME|PHONE|EMAIL|REF|OTP)_(\d+)⟧")
# The same token with its brackets dropped by the model.
_BARE_TOKEN_RE = re.compile(r"(?<![\w⟦])(NAME|PHONE|EMAIL|REF|OTP)_(\d+)(?![\w⟧])")

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_REF_RE = re.compile(r"(?<![\w-])[A-Za-z]{2,}(?:-[A-Za-z0-9]+)*-\d[A-Za-z0-9-]*\d(?![\w])")
_PHONE_RE = re.compile(r"(?<![\w+])(?:\+|00)?\d[\d \-]{7,16}\d(?!\w)")
_OTP_RE = re.compile(r"^\s*(\d{4,6})\s*$")

# The assistant asking for the patient's name - our own wording, not the
# patient's: "من فضلك أعطني اسمك الكامل", "full name".
_NAME_QUESTION_MARKERS = ("اسمك", "الاسم الكامل", "اسم المريض", "full name", "your name")

# A reply to "your name?" that is a request, not a name ("لا ابغى اغير
# الموعد") must stay readable - masking it would hide the request itself.
_NOT_NAME_WORDS = {
    "لا", "نعم", "اي", "ايه", "أيوه", "ايوه", "تمام", "ابي", "ابغى", "ابغي", "أبغى", "عايز",
    "عاوز", "ابى", "بدي", "اريد", "أريد", "ممكن", "الغي", "الغاء", "إلغاء", "اغير", "تغيير",
    "تعديل", "موعد", "الموعد", "حجز", "الحجز", "دكتور", "الدكتور", "فرع", "الفرع", "شكرا",
    "no", "yes", "cancel", "change", "booking", "appointment", "doctor", "please", "thanks",
    "what", "why", "how", "my", "name", "is",
}

# A NAME THE PATIENT GIVES UNASKED, after the word that introduces it:
# "الساعة ٤ باسم فاطمة العمري" (tanasuq-production, 2026-10-12 14:37) went
# to the model as written. The words after one of these, up to the first
# word that is not part of a name, are the name.
_NAME_CUE_RE = re.compile(
    r"(?:^|\s)(?:باسم|بإسم|بأسم|على اسم|علي اسم|عاسم|اسمي|إسمي|أسمي|"
    r"اسم المريض|اسم المريضة|اسم المريضه|"
    r"my name is|name is|under the name|in the name of)\s+",
    re.IGNORECASE,
)
_NAME_STOP_WORDS = _NOT_NAME_WORDS | {
    "و", "او", "أو", "يوم", "الساعة", "الساعه", "بكرة", "بكره", "اليوم", "مع", "عند", "في",
    "من", "الى", "إلى", "على", "علي", "رقم", "رقمي", "جوالي", "and", "at", "on", "with", "for",
    "ابني", "ابنى", "بنتي", "بنتى", "ابنتي", "زوجتي", "زوجي", "امي", "أمي", "ابوي", "أبوي", "والدي",
    "والدتي", "اخوي", "أخوي", "اخي", "أخي", "اختي", "أختي", "الطفل", "المريض", "المريضة",
}


def _names_after_cue(text: str) -> list:
    """Each name the patient introduced with a cue word, as written."""

    names = []
    for cue in _NAME_CUE_RE.finditer(text or ""):
        # "باسم" is also a first name: "دكتور باسم يوسف" names a doctor.
        before = text[:cue.start()].split()
        if before and before[-1].strip(".").lower() in ("دكتور", "الدكتور", "د", "دكتوره", "دكتورة", "dr", "doctor"):
            continue
        words = []
        for word in text[cue.end():].split():
            bare = word.strip("،,.!؟?:؛")
            if not bare or not bare.isalpha() or bare.lower() in _NAME_STOP_WORDS:
                break
            words.append(bare)
            if len(words) == 4 or bare != word:
                break
        if words:
            names.append(" ".join(words))
    return names


_current = contextvars.ContextVar("pii_vault_session", default=None)
_lock = threading.Lock()
_VAULTS: dict = {}


def _digits(text: str) -> int:
    return sum(ch.isdigit() for ch in text)


def _phone_variants(phone: str) -> list:
    """The WhatsApp number as it may appear: bare, with +, local 0-form."""

    digits = re.sub(r"\D", "", phone or "")
    if len(digits) < 9:
        return []
    variants = [digits, "+" + digits]
    for prefix in ("966", "971", "20"):
        if digits.startswith(prefix) and len(digits) > len(prefix) + 7:
            local = digits[len(prefix):]
            variants += ["0" + local, local]
    return variants


# ----------------------------------------------------------------------
# The per-conversation store
# ----------------------------------------------------------------------

def _vault(session_id: str) -> dict:
    with _lock:
        now = time.time()
        for sid in [s for s, v in _VAULTS.items() if now - v["touched"] > _TTL_SECONDS]:
            _VAULTS.pop(sid, None)
        vault = _VAULTS.get(session_id)
        if vault is None:
            vault = {"values": {}, "tokens": {}, "counts": {}, "enabled": False,
                     "debug": False, "touched": now}
            _VAULTS[session_id] = vault
        vault["touched"] = now
        return vault


def _register(vault: dict, kind: str, value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if len(value) < 3 or _TOKEN_RE.fullmatch(value) or _TOKEN_RE.search(value):
        return None
    token = vault["values"].get(value)
    if token:
        return token
    with _lock:
        vault["counts"][kind] = vault["counts"].get(kind, 0) + 1
        token = f"⟦{kind}_{vault['counts'][kind]}⟧"
        vault["values"][value] = token
        vault["tokens"][token] = value
    return token


def clear(session_id: str) -> None:
    with _lock:
        _VAULTS.pop(session_id, None)


# ----------------------------------------------------------------------
# Turn bracketing
# ----------------------------------------------------------------------

def begin_turn(session_id: str, templates: Optional[dict], channel_phone: Optional[str],
               history: Optional[list], new_message: Optional[str]) -> None:
    """Called by main.py before the graph runs. Records whether this
    client has the feature on, registers the WhatsApp number and every
    value already in the conversation, and binds this session to the
    model calls of this turn."""

    templates = templates or {}
    enabled = bool(templates.get("_pii_tokens"))
    _current.set(session_id if enabled else None)
    if not enabled or not session_id:
        return
    vault = _vault(session_id)
    vault["enabled"] = True
    vault["debug"] = bool(templates.get("_pii_debug"))
    for variant in _phone_variants(channel_phone or ""):
        _register(vault, "PHONE", variant)
    messages = list(history or [])
    observe(vault, messages)
    if new_message:
        _observe_patient_text(vault, new_message, messages)


def attach(state: Optional[dict]) -> None:
    """Bind this session to the model calls that follow - for a node that
    may run outside the context main.py set."""

    session_id = (state or {}).get("session_id")
    vault = _VAULTS.get(session_id) if session_id else None
    _current.set(session_id if vault and vault.get("enabled") else None)


def active_vault() -> Optional[dict]:
    session_id = _current.get()
    if not session_id:
        return None
    vault = _VAULTS.get(session_id)
    return vault if vault and vault.get("enabled") else None


# ----------------------------------------------------------------------
# Learning values
# ----------------------------------------------------------------------

def _kind(message) -> str:
    return getattr(message, "type", "") or ""


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content or "")


def _walk_fields(vault: dict, data) -> None:
    if isinstance(data, dict):
        for key, value in data.items():
            kind = _FIELD_KINDS.get(str(key).lower())
            if kind and isinstance(value, str):
                if kind == "PHONE":
                    for variant in _phone_variants(value) or [value]:
                        _register(vault, "PHONE", variant)
                    _register(vault, "PHONE", value)
                else:
                    _register(vault, kind, value)
            else:
                _walk_fields(vault, value)
    elif isinstance(data, list):
        for item in data:
            _walk_fields(vault, item)


def _observe_patient_text(vault: dict, text: str, history: list) -> None:
    for match in _EMAIL_RE.finditer(text):
        _register(vault, "EMAIL", match.group(0))
    for match in _REF_RE.finditer(text):
        _register(vault, "REF", match.group(0))
    for match in _PHONE_RE.finditer(text):
        if 9 <= _digits(match.group(0)) <= 15:
            _register(vault, "PHONE", match.group(0))
    otp = _OTP_RE.match(text)
    if otp and any(getattr(m, "name", None) == "send_otp" for m in history):
        _register(vault, "OTP", otp.group(1))
    last_ai = next((_text(m.content) for m in reversed(history)
                    if _kind(m) == "ai" and _text(getattr(m, "content", "")).strip()), "")
    asked_for_name = any(marker in last_ai.lower() for marker in _NAME_QUESTION_MARKERS)
    words = text.split()
    if (asked_for_name and 2 <= len(words) <= 6 and all(w.isalpha() for w in words)
            and not any(w.lower() in _NOT_NAME_WORDS for w in words)):
        _register(vault, "NAME", text.strip())
    for name in _names_after_cue(text):
        _register(vault, "NAME", name)


def observe(vault: dict, messages: list) -> None:
    """Learn every value the conversation so far carries."""

    for index, message in enumerate(messages):
        kind = _kind(message)
        if kind == "tool":
            content = _text(getattr(message, "content", ""))
            try:
                _walk_fields(vault, json.loads(content))
            except (ValueError, TypeError):
                pass
        elif kind == "ai":
            for call in getattr(message, "tool_calls", None) or []:
                _walk_fields(vault, (call or {}).get("args") or {})
        elif kind == "human":
            _observe_patient_text(vault, _text(getattr(message, "content", "")), messages[:index])


# ----------------------------------------------------------------------
# Masking and unmasking
# ----------------------------------------------------------------------

def _value_pattern(vault: dict):
    values = sorted(vault["values"], key=len, reverse=True)
    if not values:
        return None
    key = tuple(values)
    if vault.get("_pattern_key") != key:
        vault["_pattern"] = re.compile(
            "|".join(r"(?<![\w+])" + re.escape(v) + r"(?!\w)" for v in values)
        )
        vault["_pattern_key"] = key
    return vault["_pattern"]


def mask_text(vault: dict, text: str, stats: dict) -> str:
    if not isinstance(text, str) or not text:
        return text
    pattern = _value_pattern(vault)
    if pattern is None:
        return text

    def _swap(match):
        stats["masked"] = stats.get("masked", 0) + 1
        token = vault["values"][match.group(0)]
        kind = token[1:].split("_")[0]
        stats.setdefault("kinds", {})[kind] = stats.get("kinds", {}).get(kind, 0) + 1
        return token

    return pattern.sub(_swap, text)


def _mask_deep(vault: dict, data, stats: dict):
    if isinstance(data, str):
        return mask_text(vault, data, stats)
    if isinstance(data, dict):
        return {k: _mask_deep(vault, v, stats) for k, v in data.items()}
    if isinstance(data, list):
        return [_mask_deep(vault, v, stats) for v in data]
    return data


def _check_leaks(vault: dict, text: str, stats: dict) -> str:
    """A phone or email still readable in a TOOL RESULT after masking: a
    field no name in `_FIELD_KINDS` covers. Masked here, counted, logged."""

    found = [m.group(0) for m in _EMAIL_RE.finditer(text)]
    found += [m.group(0) for m in _PHONE_RE.finditer(text) if 9 <= _digits(m.group(0)) <= 15]
    for value in found:
        if _register(vault, "EMAIL" if "@" in value else "PHONE", value):
            stats["leaks"] = stats.get("leaks", 0) + 1
            logger.error("pii leak: a %s in a tool result was not registered by field name - masked",
                         "EMAIL" if "@" in value else "PHONE")
    return mask_text(vault, text, stats) if found else text


def mask_input(vault: dict, model_input, stats: dict):
    """The copy of the model's input with every known value replaced."""

    if isinstance(model_input, str):
        return mask_text(vault, model_input, stats)
    if not isinstance(model_input, (list, tuple)):
        return model_input
    observe(vault, list(model_input))
    masked = []
    for message in model_input:
        if not hasattr(message, "model_copy"):
            masked.append(mask_text(vault, message, stats) if isinstance(message, str) else message)
            continue
        update = {}
        content = getattr(message, "content", None)
        if isinstance(content, str):
            update["content"] = mask_text(vault, content, stats)
            if _kind(message) == "tool":
                update["content"] = _check_leaks(vault, update["content"], stats)
        elif isinstance(content, list):
            update["content"] = _mask_deep(vault, content, stats)
        if getattr(message, "tool_calls", None):
            update["tool_calls"] = [
                {**call, "args": _mask_deep(vault, call.get("args") or {}, stats)}
                for call in message.tool_calls
            ]
        if getattr(message, "additional_kwargs", None):
            update["additional_kwargs"] = _mask_deep(vault, dict(message.additional_kwargs), stats)
        masked.append(message.model_copy(update=update) if update else message)
    return masked


def unmask_text(vault: dict, text: str, stats: dict) -> str:
    if not isinstance(text, str) or ("⟦" not in text and "_" not in text):
        return text

    def _restore(match):
        token = f"⟦{match.group(1)}_{match.group(2)}⟧"
        value = vault["tokens"].get(token)
        if value is None:
            stats["unknown"] = stats.get("unknown", 0) + 1
            logger.error("pii: the model wrote %s, which is no token of this conversation - removed", token)
            return ""
        stats["unmasked"] = stats.get("unmasked", 0) + 1
        return value

    text = _TOKEN_RE.sub(_restore, text)
    return _BARE_TOKEN_RE.sub(
        lambda m: _restore(m) if vault["tokens"].get(f"⟦{m.group(1)}_{m.group(2)}⟧") else m.group(0),
        text,
    )


def _unmask_deep(vault: dict, data, stats: dict):
    if isinstance(data, str):
        return unmask_text(vault, data, stats)
    if isinstance(data, dict):
        return {k: _unmask_deep(vault, v, stats) for k, v in data.items()}
    if isinstance(data, list):
        return [_unmask_deep(vault, v, stats) for v in data]
    return data


def unmask_output(vault: dict, message, stats: dict):
    """The model's answer with every token put back - text and tool calls."""

    if not hasattr(message, "model_copy"):
        return message
    update = {}
    content = getattr(message, "content", None)
    if isinstance(content, (str, list)):
        update["content"] = _unmask_deep(vault, content, stats)
    if getattr(message, "tool_calls", None):
        update["tool_calls"] = [
            {**call, "args": _unmask_deep(vault, call.get("args") or {}, stats)}
            for call in message.tool_calls
        ]
    if getattr(message, "additional_kwargs", None):
        update["additional_kwargs"] = _unmask_deep(vault, dict(message.additional_kwargs), stats)
    return message.model_copy(update=update) if update else message


def log_call(session_id: str, stats: dict, masked_input=None, debug: bool = False) -> None:
    kinds = ", ".join(f"{k}×{n}" for k, n in sorted((stats.get("kinds") or {}).items()))
    logger.info(
        "pii[%s]: masked=%d (%s) unmasked=%d unknown=%d leaks=%d",
        session_id, stats.get("masked", 0), kinds or "-", stats.get("unmasked", 0),
        stats.get("unknown", 0), stats.get("leaks", 0),
    )
    if debug and masked_input is not None:
        if isinstance(masked_input, str):
            shown = masked_input
        else:
            shown = "\n---\n".join(
                f"[{_kind(m) or type(m).__name__}] {_text(getattr(m, 'content', m))}"
                + (f" tool_calls={getattr(m, 'tool_calls')}" if getattr(m, "tool_calls", None) else "")
                for m in masked_input
            )
        logger.info("pii[%s]: DEBUG - what the model received:\n%s", session_id, shown[-6000:])
