"""
Raw HTTP client layer.

Every external HTTP call in the system lives here, one function per call,
mirroring the n8n HTTP Request nodes 1:1:

  - GuestBookings/GetList (by ref)    <- f_lookup_appointment.json "HTTP Request"
                                         f_cancel_appointment.json "HTTP Request"
  - GuestBookings/GetList (by phone)  <- f_lookup_appointment.json "HTTP Request2"
                                         f_cancel_appointment.json "HTTP Request2"
  - GuestBookings/Cancel/{id}         <- f_cancel_appointment.json "HTTP Request1"/"HTTP Request3"/"HTTP Request4"
  - Authentica send-otp / verify-otp  <- langchain_cancellation.json "send_otp5"/"verify_otp5"

No business logic (filtering, selection, formatting) lives here - that's
tools.py's job. Every function catches network failures itself and
returns a structured result rather than raising, so graph nodes never
need a try/except around a tool call.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

import requests

from config import (
    AUTHENTICA_API_KEY,
    AUTHENTICA_BASE_URL,
    AUTHENTICA_FALLBACK_EMAIL,
    AUTHENTICA_TEMPLATE_ID,
    CLIENT_ID_HEADER,
    DOCTORS_API_MAX_RETRIES,
    DOCTORS_API_RETRY_BACKOFF_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    SSO_EMAIL,
    SSO_LOGIN_URL,
    SSO_ORGANIZATION_ID,
    SSO_PASSWORD,
    SSO_TOKEN_TTL_SECONDS,
)

logger = logging.getLogger(__name__)


# ==========================================================
# Result helper
# ==========================================================

def _result(success: bool, status_code: Optional[int] = None, data=None, error: Optional[str] = None,
            details: Optional[list] = None) -> dict:
    return {"success": success, "status_code": status_code, "data": data, "error": error,
            "details": details or []}


def _validation_details(response) -> list:
    """Pull field-level validation complaints out of a 4xx body.

    The API reports these as {"messages": [{"prop": "MobileNumber",
    "message": "Mobile Number Not Valid"}], ...}. Without surfacing them,
    every 400 collapses into one opaque "validation_error" and the
    patient is told there's a "technical problem" they should retry later
    - when in fact something specific and fixable was rejected (a phone
    number in the wrong format, a missing email) that retrying will
    never resolve. Confirmed real production failure: a booking was
    refused because the patient's mobile number wasn't accepted, and the
    reply blamed a technical fault instead of mentioning the number.

    Returns [{"field": ..., "message": ...}, ...], or [] if the body
    isn't in that shape."""

    try:
        body = response.json()
    except ValueError:
        return []

    if not isinstance(body, dict):
        return []

    details = []
    for entry in body.get("messages") or []:
        if not isinstance(entry, dict):
            continue
        message = entry.get("message") or entry.get("Message")
        if not message:
            continue
        details.append({
            "field": entry.get("prop") or entry.get("Prop") or "",
            "message": str(message),
        })

    return details


def _headers(client_id: Optional[str] = None, language: Optional[str] = None) -> dict:
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    if client_id:
        headers[CLIENT_ID_HEADER] = client_id
    if language:
        headers["accept-language"] = language
    return headers


# ==========================================================
# Guest Bookings API
# ==========================================================

def _request_with_retry(method: str, url: str, *, idempotent: bool = True, **kwargs):
    """Shared retry loop for every outbound call in this module (POST/PUT
    alike), so a slow-but-eventually-alive upstream gets the same patience
    everywhere - not just on the Doctors/Specialties endpoint that
    originally motivated this.

    Retries ONLY cover failure modes that are plausibly transient
    (timeout, connection error, 5xx) - see the DOCTORS_API_MAX_RETRIES
    comment in config.py. A 4xx is a real, reproducible problem with THIS
    request and retrying it would just get the same 4xx back slower, so
    those are returned immediately with no retry loop involved.

    `idempotent=False` IS FOR REQUESTS THAT CHANGE SOMETHING UPSTREAM
    (create/cancel/update a booking, send or consume an OTP). A read
    timeout or a 5xx does not mean the server did nothing - it may have
    created the booking and then failed to answer - so repeating the
    request can apply it twice. Those are retried ONLY when the
    connection itself never opened (`requests.ConnectTimeout`), which is
    the one failure where the request provably never reached the server.

    Returns (response_or_None, last_was_timeout, last_exception).
    Callers treat `response is None` as "never got a response at all"
    and fall back to the existing timeout/exception handling; a non-None
    response (even a 4xx/5xx) is handled by the caller's normal
    status-code branches exactly as before.
    """

    max_attempts = max(1, DOCTORS_API_MAX_RETRIES + 1)
    response = None
    last_timeout = False
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        last_timeout = False
        last_exc = None
        never_sent = False
        try:
            response = requests.request(method, url, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
        except requests.ConnectTimeout:
            # Must precede `requests.Timeout`: ConnectTimeout subclasses it.
            last_timeout = True
            never_sent = True
            response = None
        except requests.Timeout:
            last_timeout = True
            response = None
        except requests.RequestException as exc:
            last_exc = exc
            response = None

        if response is not None and response.status_code < 500:
            break

        is_last_attempt = attempt == max_attempts
        will_retry = not is_last_attempt and (idempotent or never_sent)
        if response is not None:
            logger.error(
                "%s %s server error status=%s (attempt %d/%d%s)",
                method.upper(), url, response.status_code, attempt, max_attempts,
                ", retrying" if will_retry else "",
            )
        elif last_timeout:
            logger.warning(
                "Request timed out: %s %s (attempt %d/%d%s)",
                method.upper(), url, attempt, max_attempts,
                ", retrying" if will_retry else "",
            )
        else:
            logger.warning(
                "Request failed: %s %s error=%s (attempt %d/%d%s)",
                method.upper(), url, last_exc, attempt, max_attempts,
                ", retrying" if will_retry else "",
            )

        if not will_retry:
            if not is_last_attempt:
                logger.warning(
                    "%s %s NOT retried: it changes state upstream and may already "
                    "have been applied - a second attempt could duplicate it",
                    method.upper(), url,
                )
            break

        time.sleep(DOCTORS_API_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))

    return response, last_timeout, last_exc


# ==========================================================
# SSO token for cms-api
# ==========================================================
# One token is shared by the whole process and reused until shortly
# before it expires, so the login isn't called on every message.

# One token per (login URL, account, organization), so each client can log in
# with its own account when the n8n client config supplies one.
_SSO_TOKENS: dict = {}
_SSO_LOCK = threading.Lock()


def _sso_settings(sso: Optional[dict]) -> dict:
    """The SSO account to log in with: the client config's values
    (`sso`, from templates["_sso"]) win, config.py / the environment fill
    in whatever the client config leaves out."""

    sso = sso or {}
    return {
        "login_url": (sso.get("login_url") or SSO_LOGIN_URL or "").strip(),
        "email": (sso.get("email") or SSO_EMAIL or "").strip(),
        "password": sso.get("password") or SSO_PASSWORD or "",
        "organization_id": (sso.get("organization_id") or SSO_ORGANIZATION_ID or "").strip(),
    }


def _get_sso_token(force_refresh: bool = False, sso: Optional[dict] = None) -> Optional[str]:
    """Bearer token for cms-api, logging in only when there is no cached
    token or it is about to expire. Returns None if the login failed."""

    cfg = _sso_settings(sso)
    key = (cfg["login_url"], cfg["email"], cfg["organization_id"])

    with _SSO_LOCK:
        cached = _SSO_TOKENS.get(key)
        if not force_refresh and cached and time.time() < cached[1]:
            return cached[0]

        if not cfg["email"] or not cfg["password"]:
            logger.error("SSO email / password are not set (client config SSO_EMAIL / SSO_PASSWORD or the environment) - cannot call cms-api")
            return None

        login_body = {"email": cfg["email"], "password": cfg["password"]}
        if cfg["organization_id"]:
            login_body["organizationId"] = cfg["organization_id"]

        try:
            response = requests.post(
                cfg["login_url"],
                json=login_body,
                headers={"accept": "*/*", "Content-Type": "application/json"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            logger.error("SSO login request failed (%s): %s", cfg["login_url"], exc)
            return None

        if response.status_code != 200:
            logger.error("SSO login failed url=%s status=%s body=%s", cfg["login_url"], response.status_code, response.text[:300])
            return None

        try:
            body = response.json()
        except ValueError:
            logger.error(
                "SSO login returned a non-JSON body: url=%s content_type=%s body=%r "
                "(is SSO_LOGIN_URL the .../api/Auth/Login endpoint itself?)",
                cfg["login_url"], response.headers.get("Content-Type"), response.text[:150],
            )
            return None

        # The token may be top-level or inside the usual "data" envelope.
        data = body.get("data") if isinstance(body, dict) and isinstance(body.get("data"), dict) else body
        if not isinstance(data, dict):
            logger.error("SSO login returned an unexpected body shape")
            return None

        token = data.get("access_token") or data.get("accessToken") or data.get("token")
        orgs = data.get("organizations") if isinstance(data.get("organizations"), list) else []
        if not token and data.get("requiresOrganizationSelection") and not cfg["organization_id"]:
            # An account that belongs to ONE organization can be logged in
            # without configuring it. With several the choice is the
            # clinic's, so it is never guessed.
            only = orgs[0] if len(orgs) == 1 and isinstance(orgs[0], dict) else None
            org_id = (only or {}).get("id") or (only or {}).get("organizationId")
            if org_id:
                try:
                    retry = requests.post(
                        cfg["login_url"], json={**login_body, "organizationId": org_id},
                        headers={"accept": "*/*", "Content-Type": "application/json"},
                        timeout=REQUEST_TIMEOUT_SECONDS,
                    )
                    retry_body = retry.json() if retry.status_code == 200 else {}
                    retry_data = retry_body.get("data") if isinstance(retry_body.get("data"), dict) else retry_body
                    token = retry_data.get("access_token") or retry_data.get("accessToken") or retry_data.get("token")
                    if token:
                        data = retry_data
                except (requests.RequestException, ValueError, AttributeError) as exc:
                    logger.error("SSO login retry with the only organization failed: %s", exc)
            elif len(orgs) > 1:
                logger.error(
                    "SSO account belongs to %d organizations - set SSO_ORGANIZATION_ID in the client config. Options: %s",
                    len(orgs), [(o.get("id") or o.get("organizationId"), o.get("name") or o.get("displayName")) for o in orgs if isinstance(o, dict)],
                )
        if not token:
            logger.error("SSO login response has no access_token, keys=%s", sorted(data.keys()))
            return None

        expires_in = data.get("expires_in") or data.get("expiresIn")
        ttl = int(expires_in) if expires_in else SSO_TOKEN_TTL_SECONDS
        _SSO_TOKENS[key] = (token, time.time() + max(60, ttl - 60))
        return token


def get_bookings_by_ref(base_url: str, ref_number: str, language: Optional[str] = None, client_id: Optional[str] = None, sso: Optional[dict] = None) -> dict:
    """POST {cms base_url}/api/Bookings/GetList with bookingRefNum.
    (Moved from portal-api /api/GuestBookings/GetList.)"""

    result = _cms_request("post", base_url, "/api/Bookings/GetList", language=language, sso=sso,
                          json={"pageNumber": 1, "pageSize": 10, "bookingRefNum": ref_number})

    # bookingRefNum is a "contains" match on cms-api, so BK-123 also
    # returns BK-1234. Keep only the booking whose reference is exactly
    # the one asked for.
    if result["success"] and isinstance(result["data"], dict):
        wanted = str(ref_number).strip().lower()
        items = result["data"].get("items") or []
        exact = [i for i in items if str(i.get("bookingRefNum") or "").strip().lower() == wanted]
        result["data"] = {**result["data"], "items": exact, "totalCount": len(exact)}

    return result


def get_bookings_by_phone(
    base_url: str,
    phone: str,
    language: Optional[str] = None,
    client_id: Optional[str] = None,
    page_size: int = 1000,
    status_list: Optional[list] = None,
    sso: Optional[dict] = None,
) -> dict:
    """POST {cms base_url}/api/Bookings/GetList with patientMobile + pageSize.
    (Moved from portal-api /api/GuestBookings/GetList. cms-api renamed the
    filter mobileNumber -> patientMobile and rejects unknown fields with
    400 "Data Was Not Valid".)

    `status_list`, when given, is sent as the API's "statusList" filter
    (e.g. [1, 2] for New+Confirmed). tools.py's _filter_active still runs
    afterwards as a second layer."""

    # cms-api: pageSize 100 or less (the guide, conventions).
    payload = {"pageNumber": 1, "pageSize": min(page_size, _CMS_MAX_PAGE_SIZE), "patientMobile": phone}
    if status_list:
        payload["statusList"] = status_list

    return _cms_request("post", base_url, "/api/Bookings/GetList", language=language, sso=sso, json=payload)


def _post_bookings(url: str, payload: dict, language: Optional[str], client_id: Optional[str]) -> dict:
    logger.debug("POST %s payload=%s", url, payload)

    response, last_timeout, last_exc = _request_with_retry(
        "post", url, json=payload, headers=_headers(client_id=client_id, language=language),
    )

    if response is None:
        if last_timeout:
            logger.warning("Booking lookup timed out: %s", url)
            return _result(False, error="timeout")
        logger.exception("Booking lookup request failed: %s", url)
        return _result(False, error=str(last_exc) if last_exc else "request_failed")

    if response.status_code >= 500:
        logger.error("GuestBookings API server error: %s status=%s body=%s", url, response.status_code, response.text[:500])
        return _result(False, response.status_code, error="server_error")

    if response.status_code in (401, 403):
        logger.error(
            "GuestBookings API AUTHENTICATION/AUTHORIZATION error (%s) - this is a credentials/access "
            "problem on the API server itself, not a request-content problem: %s body=%s",
            response.status_code, url, response.text[:500],
        )
        return _result(False, response.status_code, error="authentication_error")

    if response.status_code >= 400:
        details = _validation_details(response)
        logger.error(
            "GuestBookings API validation error: %s status=%s body=%s rejected_fields=%s",
            url, response.status_code, response.text[:500],
            [d["field"] for d in details] or "unknown",
        )
        return _result(False, response.status_code, error="validation_error", details=details)

    try:
        body = response.json()
    except ValueError:
        return _result(False, response.status_code, error="invalid_json_response")

    if not body:
        return _result(False, response.status_code, error="empty_response")

    if not body.get("isSuccess"):
        return _result(False, response.status_code, data=body, error="api_reported_failure")

    return _result(True, response.status_code, data=body.get("data", {}))


# ==========================================================
# cms-api for EVERYTHING (Catalyst CMS API - AI Booking Integration
# Guide v1.2, 2026-10-03)
# ==========================================================
#
# portal-api routes for the assistant are being retired: the catalogue,
# bookable slots and the reservation move to cms-api with the clinic's
# token. A clinic is moved when its config names its cms host (config
# `_booking_on_cms`); tools register that host here with the clinic's SSO
# account, and every catalogue function below routes a registered host to
# cms-api. Any other base URL keeps the old portal-api path untouched.
_CMS_HOSTS: dict = {}
_CMS_MAX_PAGE_SIZE = 100   # the guide: pageSize 100 or less, follow hasNextPage
_CMS_MAX_PAGES = 20


_CMS_PORTAL_FALLBACK: dict = {}


def register_cms_host(base_url: Optional[str], sso: Optional[dict], portal_url: Optional[str] = None) -> None:
    if base_url:
        _CMS_HOSTS[base_url.rstrip("/")] = dict(sso or {})
        if portal_url and portal_url.rstrip("/") != base_url.rstrip("/"):
            _CMS_PORTAL_FALLBACK[base_url.rstrip("/")] = portal_url.rstrip("/")


# A CALL cms-api REFUSES FOR ACCESS GOES TO portal-api - while portal-api
# still serves it. The integration account needs one permission per
# endpoint (the guide, section 2); a missing one is a 403 on that endpoint
# alone, and without this the whole catalogue - and every booking - would
# stop on it. Only access/route errors fall back: a validation error, a
# refused slot or a timeout is the answer, not a reason to ask elsewhere.
_FALLBACK_ERRORS = ("authentication_error", "endpoint_not_found", "not_configured")


def _cms_first(base_url: str, cms_call, portal_call, what: str, fallback_errors=_FALLBACK_ERRORS,
               empty_is_refusal: bool = False) -> dict:
    """`empty_is_refusal`: for a list a clinic cannot have empty (its
    specialties, its branches). CONFIRMED (tanasuq-production 2026-10-04
    10:13): cms-api answered Specialties/GetList with success and no
    items - the account's role, not the clinic, is what is missing."""
    original = result = cms_call()
    empty = (empty_is_refusal and result["success"] and isinstance(result.get("data"), dict)
             and not (result["data"].get("items") or []))
    if empty:
        result = {**result, "success": False, "error": "empty_list"}
        fallback_errors = tuple(fallback_errors) + ("empty_list",)
    if result["success"] or result.get("error") not in fallback_errors:
        return result
    portal = _CMS_PORTAL_FALLBACK.get((base_url or "").rstrip("/"))
    if not portal:
        return original
    logger.error(
        "cms-api refused %s (status=%s error=%s) - ASK CATALYST for this endpoint's permission; "
        "served from portal-api %s for now", what, result.get("status_code"), result.get("error"), portal,
    )
    return portal_call(portal)


def _cms_sso_for(base_url: Optional[str]) -> Optional[dict]:
    """The SSO account for a registered cms host, or None for portal-api."""
    return _CMS_HOSTS.get((base_url or "").rstrip("/"))


def _cms_list(base_url: str, path: str, payload: dict, language: Optional[str] = None) -> dict:
    """A paged cms-api list, every page merged into one `items` list - the
    shape the portal-api calls returned with one big page."""

    sso = _cms_sso_for(base_url)
    size = min(int(payload.get("pageSize") or _CMS_MAX_PAGE_SIZE), _CMS_MAX_PAGE_SIZE)
    items: list = []
    last = None
    for page in range(1, _CMS_MAX_PAGES + 1):
        result = _cms_request("post", base_url, path, language=language, sso=sso,
                              json={**payload, "pageNumber": page, "pageSize": size})
        if not result["success"] or not isinstance(result.get("data"), dict):
            return result
        data = result["data"]
        items.extend(data.get("items") or [])
        last = result
        if not data.get("hasNextPage"):
            break
    last["data"] = {**last["data"], "items": items, "totalCount": len(items), "hasNextPage": False}
    return last


def _request_once(method: str, url: str, **kwargs):
    """One attempt, same return shape as `_request_with_retry`."""
    try:
        return requests.request(method, url, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs), False, None
    except requests.Timeout:
        return None, True, None
    except requests.RequestException as exc:
        return None, False, exc


def _cms_request(method: str, base_url: Optional[str], path: str, language: Optional[str] = None, sso: Optional[dict] = None, retry: bool = True, **kwargs) -> dict:
    """GET/POST to cms-api with the SSO bearer token. A 401 means the
    token expired early or was revoked: log in again once and retry.

    `retry=False` sends the request once - no timeout/5xx retries. The
    Reservation must never be retried automatically (the guide, 4.3): a
    timed-out reservation may have been made."""

    if not base_url:
        logger.error("cms-api base URL is not configured (CMS_API_BASE_URL / cms_base_url) - cannot call %s", path)
        return _result(False, error="not_configured")

    url = f"{base_url}{path}"
    response, last_timeout, last_exc = None, False, None

    for attempt in (1, 2):
        token = _get_sso_token(force_refresh=(attempt == 2), sso=sso)
        if not token:
            return _result(False, error="authentication_error")

        headers = _headers(language=language)
        headers["Authorization"] = f"Bearer {token}"

        logger.debug("%s %s %s", method.upper(), url, kwargs)
        # A write (PUT) must not be replayed after a read timeout / 5xx: it may
        # already have been applied. Reads are retried freely.
        if retry:
            response, last_timeout, last_exc = _request_with_retry(
                method, url, idempotent=(method.lower() != "put"), headers=headers, **kwargs)
        else:
            response, last_timeout, last_exc = _request_once(method, url, headers=headers, **kwargs)
        if response is None or response.status_code != 401:
            break

    if response is None:
        if last_timeout:
            logger.warning("cms-api request timed out: %s", url)
            return _result(False, error="timeout")
        logger.error("cms-api request failed: %s error=%s", url, last_exc)
        return _result(False, error=str(last_exc) if last_exc else "request_failed")

    if response.status_code >= 500:
        logger.error("cms-api server error: %s status=%s body=%s", url, response.status_code, response.text[:500])
        return _result(False, response.status_code, error="server_error")

    if response.status_code in (401, 403):
        logger.error("cms-api AUTHENTICATION/AUTHORIZATION error (%s) even after a fresh login: %s body=%s",
                     response.status_code, url, response.text[:500])
        return _result(False, response.status_code, error="authentication_error")

    if response.status_code == 404:
        logger.error("cms-api endpoint NOT FOUND (404): %s body=%s", url, response.text[:300])
        return _result(False, response.status_code, error="endpoint_not_found")

    if response.status_code >= 400:
        details = _validation_details(response)
        logger.error("cms-api validation error: %s status=%s body=%s rejected_fields=%s",
                     url, response.status_code, response.text[:500],
                     [d["field"] for d in details] or "unknown")
        return _result(False, response.status_code, error="validation_error", details=details)

    try:
        body = response.json()
    except ValueError:
        return _result(False, response.status_code, error="invalid_json_response")

    if not body:
        return _result(False, response.status_code, error="empty_response")

    if not body.get("isSuccess"):
        return _result(False, response.status_code, data=body, error="api_reported_failure")

    return _result(True, response.status_code, data=body.get("data", {}))


BOOKING_STATUS_CANCELLED = 6


def cancel_booking_by_guid(base_url: str, booking_guid: str, client_id: Optional[str] = None, sso: Optional[dict] = None) -> dict:
    """PUT {cms base_url}/api/Bookings/UpdateStatus with status 6 (Cancelled).
    (Moved from portal-api PUT /api/GuestBookings/Cancel/{id}: cancel now
    lives inside cms-api and needs the SSO bearer token.)"""

    return _cms_request(
        "put", base_url, "/api/Bookings/UpdateStatus", sso=sso,
        json={"id": booking_guid, "isConfirmed": True, "status": BOOKING_STATUS_CANCELLED},
    )


# ==========================================================
# Authentica OTP API (real provider - langchain_cancellation.json
# "send_otp5" / "verify_otp5"). Only used when config.OTP_PROVIDER ==
# "authentica"; see services in tools.py for the dummy alternative.
# ==========================================================

def authentica_send_otp(phone: str) -> dict:
    url = f"{AUTHENTICA_BASE_URL}/send-otp"

    payload = {
        "method": "sms",
        "template_id": AUTHENTICA_TEMPLATE_ID,
        "fallback_email": AUTHENTICA_FALLBACK_EMAIL,
        "phone": phone,
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Authorization": AUTHENTICA_API_KEY,
    }

    response, last_timeout, last_exc = _request_with_retry("post", url, idempotent=False, headers=headers, json=payload)

    if response is None:
        if last_timeout:
            return _result(False, error="timeout")
        logger.exception("Authentica send_otp failed")
        return _result(False, error=str(last_exc) if last_exc else "request_failed")

    if response.status_code >= 400:
        logger.error(
            "Authentica send_otp rejected phone=%s status=%s body=%s",
            phone, response.status_code, response.text[:500],
        )
        return _result(False, response.status_code, error="send_otp_failed")

    try:
        body = response.json()
    except ValueError:
        body = {}

    return _result(True, response.status_code, data=body)


def authentica_verify_otp(phone: str, otp: str, email: str = "") -> dict:
    url = f"{AUTHENTICA_BASE_URL}/verify-otp"

    # `email` mirrors the fallback_email sent with send-otp above - the
    # confirmed-working curl example verifies with the SAME email used
    # to send, so callers should pass AUTHENTICA_FALLBACK_EMAIL through
    # (see tools.py verify_otp) rather than leaving this blank.
    payload = {"otp": otp, "email": email, "phone": phone}
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Authorization": AUTHENTICA_API_KEY,
    }

    response, last_timeout, last_exc = _request_with_retry("post", url, idempotent=False, headers=headers, json=payload)

    if response is None:
        if last_timeout:
            return _result(False, error="timeout")
        logger.exception("Authentica verify_otp failed")
        return _result(False, error=str(last_exc) if last_exc else "request_failed")

    if response.status_code >= 400:
        logger.error(
            "Authentica verify_otp rejected phone=%s status=%s body=%s",
            phone, response.status_code, response.text[:500],
        )
        return _result(False, response.status_code, error="verify_otp_failed")

    try:
        body = response.json()
    except ValueError:
        body = {}

    verified = bool(body.get("isSuccess") or body.get("success") or body.get("verified"))

    return _result(verified, response.status_code, data=body)


# ==========================================================
# Doctors / Specialties API (Medical Concierge feature)
# ==========================================================
#
# Separate service from GuestBookings, confirmed on a different port
# (1102 vs 1101). Response shape (confirmed directly from the API's own
# Swagger "Execute" output): {"data": {"items": [...], ...},
# "statusCode": 200, "isSuccess": true, "messages": [...]} - handled the
# same way _post_bookings already handles GuestBookings' identical
# response envelope.

def _post_json(url: str, payload: dict, client_id: Optional[str] = None, language: Optional[str] = None,
               idempotent: bool = True) -> dict:
    """Generic POST + envelope handling, shared by get_specialties/
    get_doctors. Mirrors _post_bookings' error handling exactly
    (timeout/5xx/4xx/empty/invalid JSON/isSuccess check), kept as a
    separate function so GuestBookings' own _post_bookings is untouched.

    `language` ("ar"/"en") is sent as the accept-language header, exactly
    as _post_bookings already does for the GuestBookings endpoints. The
    Doctors/Specialties/Branches endpoints honour it too and return
    their doctor, branch, specialty and SERVICE names already localized -
    which is the only reliable way to get e.g. "فحص نظر" instead of
    "Eye Vision Check" for a service, since unlike doctors and branches
    a service has no altName field to fall back on."""

    logger.debug("POST %s payload=%s", url, payload)

    # Retries ONLY cover failure modes that are plausibly transient on
    # this specific endpoint (timeout, connection error, 5xx) - see the
    # DOCTORS_API_MAX_RETRIES comment in config.py. A 4xx is a real,
    # reproducible problem with THIS request (bad payload, bad auth,
    # wrong path) and retrying it would just get the same 4xx back
    # slower, so those still fall straight through to the existing
    # handling below with no retry loop involved.
    #
    # `idempotent=False` (the Reservation POST): same rule as
    # `_request_with_retry` - only a connection that never opened is
    # retried, because a read timeout or 5xx may mean the booking WAS
    # created and a second POST would create it again.
    max_attempts = max(1, DOCTORS_API_MAX_RETRIES + 1)
    response = None
    last_timeout = False
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        last_timeout = False
        last_exc = None
        never_sent = False
        try:
            response = requests.post(
                url,
                json=payload,
                headers=_headers(client_id=client_id, language=language),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.ConnectTimeout:
            # Must precede `requests.Timeout`: ConnectTimeout subclasses it.
            last_timeout = True
            never_sent = True
            response = None
        except requests.Timeout:
            last_timeout = True
            response = None
        except requests.RequestException as exc:
            last_exc = exc
            response = None

        if response is not None and response.status_code < 500:
            # Success or a non-retryable 4xx - stop retrying either way.
            break

        is_last_attempt = attempt == max_attempts
        will_retry = not is_last_attempt and (idempotent or never_sent)
        if response is not None:
            logger.error(
                "Doctors/Specialties API server error: %s status=%s body=%s (attempt %d/%d%s)",
                url, response.status_code, response.text[:1000], attempt, max_attempts,
                ", retrying" if will_retry else "",
            )
        elif last_timeout:
            logger.warning(
                "Request timed out: %s (attempt %d/%d%s)",
                url, attempt, max_attempts, ", retrying" if will_retry else "",
            )
        else:
            logger.warning(
                "Request failed: %s error=%s (attempt %d/%d%s)",
                url, last_exc, attempt, max_attempts, ", retrying" if will_retry else "",
            )

        if not will_retry:
            if not is_last_attempt:
                logger.warning(
                    "POST %s NOT retried: it changes state upstream and may already "
                    "have been applied - a second attempt could duplicate it", url,
                )
            break

        # Exponential backoff (0.5s, 1s, 2s, ...) - a short pause is
        # enough to ride out the sub-second blip this endpoint is known
        # to have, without holding up the patient for long if it's a
        # real outage.
        time.sleep(DOCTORS_API_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))

    if response is None:
        if last_timeout:
            return _result(False, error="timeout")
        return _result(False, error=str(last_exc) if last_exc else "request_failed")

    if response.status_code >= 500:
        logger.error("Doctors/Specialties API server error: %s status=%s body=%s - giving up after %d attempt(s)", url, response.status_code, response.text[:1000], max_attempts)
        return _result(False, response.status_code, error="server_error")

    if response.status_code == 404:
        # A wrong endpoint PATH, not a bad request - this is a bug in our
        # own URL construction (or a changed API), never something the
        # user can fix by "trying again later". Called out separately so
        # it can't hide behind a generic "validation_error" again.
        logger.error(
            "Doctors/Specialties API endpoint NOT FOUND (404) - check the URL path is correct: %s body=%s",
            url, response.text[:500],
        )
        return _result(False, response.status_code, error="endpoint_not_found")

    if response.status_code >= 400:
        details = _validation_details(response)
        logger.error(
            "Doctors/Specialties API validation error: %s status=%s body=%s rejected_fields=%s payload=%s",
            url, response.status_code, response.text[:1000],
            [d["field"] for d in details] or "unknown", payload,
        )
        return _result(False, response.status_code, error="validation_error", details=details)

    try:
        body = response.json()
    except ValueError:
        return _result(False, response.status_code, error="invalid_json_response")

    if not body:
        return _result(False, response.status_code, error="empty_response")

    if not body.get("isSuccess"):
        return _result(False, response.status_code, data=body, error="api_reported_failure")

    return _result(True, response.status_code, data=body.get("data", {}))


def get_specialties(base_url: str, page_size: int = 200, client_id: Optional[str] = None, language: Optional[str] = None) -> dict:
    """POST {base_url}/api/Specialties/GetList.

    NOTE ON THE PATH: the Swagger UI labels this operation
    "GetSpecialtiesPagedList", but that's the operation ID, NOT the HTTP
    path - the actual path is /api/Specialties/GetList. This was
    originally coded as /api/Specialties/GetSpecialtiesPagedList, which
    returned 404 and surfaced to the user as a vague "technical problem"
    (any 4xx was being reported as "validation_error"). Confirmed by the
    same pattern on the Doctors endpoint, whose Swagger operation ID is
    "GetDoctorsPagedList" but whose real path is /api/Doctors/GetList.

    Returns every specialty this clinic offers (scoped by base_url alone,
    confirmed directly - no separate organizationId/branchId needed)."""

    url = f"{base_url}/api/Specialties/GetList"
    # NOTE: pageNumber must be 1 or above, NOT 0 - confirmed directly
    # from the API's own error response ("PageNumber should be above
    # one", thrown by PagingOptions.set_PageNumber). The Swagger UI's
    # example body shows "pageNumber": 0, but that's just a placeholder
    # default and is rejected at runtime with a 500.
    payload = {"pageNumber": 1, "pageSize": page_size}

    if _cms_sso_for(base_url) is not None:
        return _cms_first(base_url, lambda: _cms_list(base_url, "/api/Specialties/GetList", payload, language),
                          lambda portal: get_specialties(portal, page_size, client_id, language), "Specialties/GetList",
                          empty_is_refusal=True)
    return _post_json(url, payload, client_id=client_id, language=language)


def get_doctors(
    base_url: str,
    specialty_ids: Optional[list] = None,
    branch_ids: Optional[list] = None,
    service_ids: Optional[list] = None,
    has_published_service: bool = True,
    has_service_schedule: bool = True,
    intersection_start: Optional[str] = None,
    intersection_end: Optional[str] = None,
    page_size: int = 200,
    client_id: Optional[str] = None,
    language: Optional[str] = None,
) -> dict:
    """POST {base_url}/api/Doctors/GetList.

    `has_published_service`/`has_service_schedule`/`intersection_start`/
    `intersection_end` are REQUEST filter fields (confirmed directly from
    the API's own request schema) - they narrow results to doctors who
    are actually bookable with an available schedule intersecting the
    given time window. The response itself then includes `hasSlots` per
    doctor reflecting that same filter.

    `branch_ids` filters to doctors who work at any of the given
    branches - confirmed as a real request field, used by the New
    Booking flow's branch-first selection path."""

    url = f"{base_url}/api/Doctors/GetList"
    payload = {
        # Must be 1 or above, not 0 - see the note in get_specialties()
        "pageNumber": 1,
        "pageSize": page_size,
        "hasPublishedService": has_published_service,
        "hasServiceSchedule": has_service_schedule,
    }

    if specialty_ids:
        payload["specialtyIds"] = specialty_ids
    if branch_ids:
        payload["branchIds"] = branch_ids
    if service_ids:
        # Confirmed request field: narrows to doctors who actually
        # provide the given service(s). Used when the patient picked a
        # SERVICE first ("فحص النظر") - the doctors for that service are
        # the answer, and re-asking "specialty or doctor?" throws the
        # choice they already made away.
        payload["serviceIds"] = service_ids
    if intersection_start:
        payload["intersectionStart"] = intersection_start
    if intersection_end:
        payload["intersectionEnd"] = intersection_end

    if _cms_sso_for(base_url) is not None:
        # cms-api also serves clinic staff: without this it returns
        # doctors with no published service (the guide, 6.2).
        cms_payload = {**payload, "hasPublishedService": True}
        return _cms_first(
            base_url, lambda: _cms_list(base_url, "/api/Doctors/GetList", cms_payload, language),
            lambda portal: _post_json(f"{portal}/api/Doctors/GetList", payload, client_id=client_id, language=language),
            "Doctors/GetList")
    return _post_json(url, payload, client_id=client_id, language=language)


# ==========================================================
# Doctor Schedule / Reschedule (Reschedule Appointment feature)
# ==========================================================
#
# All three endpoints confirmed directly from the API's own Swagger
# "Execute" output, same demo server/port as Doctors/Specialties.

def get_branches(
    base_url: str,
    search_query: Optional[str] = None,
    page_size: int = 200,
    client_id: Optional[str] = None,
    language: Optional[str] = None,
) -> dict:
    """POST {base_url}/api/Branches/GetList.

    Returns this clinic's branch list - name/altName/address/city/
    country/contact info per branch (confirmed directly from the API's
    real response). `search_query` is optional server-side filtering;
    the caller may also just fetch all and match client-side."""

    url = f"{base_url}/api/Branches/GetList"
    payload = {"pageNumber": 1, "pageSize": page_size}

    if search_query:
        payload["searchQuery"] = search_query

    if _cms_sso_for(base_url) is not None:
        return _cms_first(base_url, lambda: _cms_list(base_url, "/api/Branches/GetList", payload, language),
                          lambda portal: _post_json(f"{portal}/api/Branches/GetList", payload, client_id=client_id, language=language),
                          "Branches/GetList", empty_is_refusal=not search_query)
    return _post_json(url, payload, client_id=client_id, language=language)


def get_doctor_schedule(
    base_url: str,
    doctor_ids: list,
    branch_ids: Optional[list] = None,
    effective_date: Optional[str] = None,
    page_size: int = 50,
    client_id: Optional[str] = None,
    language: Optional[str] = None,
    include_future: bool = False,
) -> dict:
    """POST {base_url}/api/DoctorSchedules/GetList.

    Returns the doctor's GENERAL RECURRING schedule (which weekdays they
    work, and their daily start/end times, and the date range this
    schedule is valid for) - NOT specific available time slots. Each
    item has recurringDaysNames/fromDateTime/toDateTime among other
    fields (confirmed directly from the API's real response).

    `branch_ids`, when given, narrows to that specific branch's schedule
    only - used by the New Booking flow once a branch is confirmed
    (otherwise the schedule spans every branch the doctor works at).

    `effective_date` (e.g. "2026-07-30"), when given, excludes EXPIRED
    schedule rows - the row's own validity END must be on or after this
    date (`toDateTimeFrom`).

    `include_future=False` (the default) ALSO requires the row to have
    already started (`fromDateTimeTo`), i.e. it must be valid on exactly
    that date. Correct when asking about one specific day.

    `include_future=True` drops that second condition, so a rota the
    clinic has published for a LATER period is returned too. Use it for
    any general "when does this doctor work?" question - clinics publish
    the next season's rota in advance so patients can book into it, and
    hiding it makes the doctor look less available than they are."""

    url = f"{base_url}/api/DoctorSchedules/GetList"
    payload = {"pageNumber": 1, "pageSize": page_size, "doctorIds": doctor_ids}

    if branch_ids:
        payload["branchIds"] = branch_ids

    if effective_date:
        # `toDateTimeFrom` excludes EXPIRED rows: the schedule's validity
        # must end on or after this date. That is the part worth
        # filtering - a lapsed schedule is not bookable.
        payload["toDateTimeFrom"] = effective_date

        # `fromDateTimeTo` would additionally require the schedule to
        # have ALREADY STARTED, and that is only correct when asking
        # about one specific date.
        #
        # For a general "when does this doctor work?" it is wrong: a
        # clinic publishes next season's rota in advance precisely so
        # patients can book into it. Confirmed in production - a doctor
        # had Thursdays (valid until 01/09) and Mondays (valid from
        # 01/10), and the Mondays were invisible, so the assistant
        # reported the doctor works only Thursdays while the clinic had
        # deliberately opened Monday bookings.
        if not include_future:
            payload["fromDateTimeTo"] = effective_date

    if _cms_sso_for(base_url) is not None:
        return _cms_first(base_url, lambda: _cms_list(base_url, "/api/DoctorSchedules/GetList", payload, language),
                          lambda portal: _post_json(f"{portal}/api/DoctorSchedules/GetList", payload, client_id=client_id, language=language),
                          "DoctorSchedules/GetList")
    return _post_json(url, payload, client_id=client_id, language=language)


def get_doctor_schedule_slots(
    base_url: str,
    doctor_ids: list,
    from_date: str,
    to_date: str,
    is_booked: bool = False,
    branch_ids: Optional[list] = None,
    page_size: int = 200,
    client_id: Optional[str] = None,
    language: Optional[str] = None,
) -> dict:
    """POST {base_url}/api/Doctors/GetDoctorScheduleSlots.

    Returns SPECIFIC time slots within [from_date, to_date] - the actual
    bookable times, not just working days. `is_booked=False` (default)
    filters to only slots that are NOT already taken - i.e. genuinely
    available ones. `branch_ids` additionally narrows to a specific
    branch (confirmed real request field) - needed for the New Booking
    flow once both a doctor AND branch are confirmed. Each item has
    slotStart/slotEnd/isBooked among other fields (confirmed directly
    from the API's real response)."""

    url = f"{base_url}/api/Doctors/GetDoctorScheduleSlots"
    payload = {
        "pageNumber": 1,
        "pageSize": page_size,
        "fromDate": from_date,
        "toDate": to_date,
        "isBooked": is_booked,
        "doctorIds": doctor_ids,
    }

    if branch_ids:
        payload["branchIds"] = branch_ids

    if _cms_sso_for(base_url) is not None:
        return _cms_first(base_url, lambda: _cms_bookable_slots(base_url, payload, language),
                          lambda portal: _post_json(f"{portal}/api/Doctors/GetDoctorScheduleSlots", payload, client_id=client_id, language=language),
                          "Doctors/GetBookableScheduleSlots")
    return _post_json(url, payload, client_id=client_id, language=language)


def _cms_bookable_slots(base_url: str, payload: dict, language: Optional[str]) -> dict:
    """POST {cms}/api/Doctors/GetBookableScheduleSlots (the guide, 6.3).

    Not Doctors/GetDoctorScheduleSlots: on cms-api that is the STAFF route
    and lists slots the clinic can book but a patient cannot. This one
    hides slots inside the clinic's lead-time cutoff, like the website.
    Asked with pageSize 0 (the date range bounds it), sorted by slotStart
    here, and slots at the doctor's daily capacity are dropped - a patient
    cannot book them.

    AT MOST 31 DAYS PER REQUEST. cms-api refuses an unpaginated range
    longer than that ("The requested date range cannot exceed 31 days when
    requesting unpaginated results", tanasuq-production 2026-10-04 10:14),
    and the day lookups ask for ~6 weeks - so a longer range is asked in
    pieces and merged."""

    items: list = []
    seen: set = set()
    result = None
    for start, end in _date_windows(payload.get("fromDate"), payload.get("toDate")):
        result = _cms_request(
            "post", base_url, "/api/Doctors/GetBookableScheduleSlots", language=language,
            sso=_cms_sso_for(base_url),
            json={**payload, "fromDate": start, "toDate": end, "pageNumber": 1, "pageSize": 0},
        )
        if not result["success"] or not isinstance(result.get("data"), dict):
            return result
        for item in result["data"].get("items") or []:
            key = (item.get("slotStart"), item.get("doctorId"), item.get("branchId"), item.get("scheduleId"))
            if key in seen or item.get("isAtDailyCapacity"):
                continue
            seen.add(key)
            items.append(item)
    items.sort(key=lambda i: str(i.get("slotStart") or ""))
    result["data"] = {**result["data"], "items": items, "totalCount": len(items), "hasNextPage": False}
    return result


_SLOT_WINDOW = timedelta(days=30)


def _date_windows(from_date, to_date) -> list:
    """[from, to] split into consecutive windows of at most 30 days, in the
    same ISO style it came in. Anything unparsable goes as one window -
    the API then says what is wrong with it."""

    try:
        start = datetime.fromisoformat(str(from_date).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(to_date).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return [(from_date, to_date)]
    if end - start <= _SLOT_WINDOW:
        return [(from_date, to_date)]
    windows = []
    while start < end:
        stop = min(start + _SLOT_WINDOW, end)
        windows.append((start.isoformat(), stop.isoformat()))
        start = stop
    return windows


def get_doctor_fees(
    base_url: str,
    doctor_ids: list,
    is_published: bool = True,
    page_size: int = 1000,
    client_id: Optional[str] = None,
    language: Optional[str] = None,
) -> dict:
    """POST {base_url}/api/DoctorServices/GetList.

    Returns a doctor's published services and prices - confirmed
    directly from a real production n8n workflow's request/response
    handling (extracts serviceName/price per item)."""

    url = f"{base_url}/api/DoctorServices/GetList"
    payload = {
        "pageNumber": 1,
        "pageSize": page_size,
        "isPublished": is_published,
        "doctorIds": doctor_ids,
    }

    if _cms_sso_for(base_url) is not None:
        # Renamed on cms-api (the guide, 6.5); the fee is still `price`.
        return _cms_first(base_url, lambda: _cms_list(base_url, "/api/DoctorAssignedServices/GetList", payload, language),
                          lambda portal: _post_json(f"{portal}/api/DoctorServices/GetList", payload, client_id=client_id, language=language),
                          "DoctorAssignedServices/GetList")
    return _post_json(url, payload, client_id=client_id, language=language)


def get_services(
    base_url: str,
    branch_ids: Optional[list] = None,
    service_type_ids: Optional[list] = None,
    is_published: bool = True,
    page_size: int = 500,
    client_id: Optional[str] = None,
    language: Optional[str] = None,
) -> dict:
    """POST {base_url}/api/Services/GetList.

    The clinic's real SERVICE CATALOGUE, straight from the system -
    optionally narrowed to the branches that actually provide each
    service via `branch_ids`, to a service type (lab test / radiology
    test / ...) via `service_type_ids`, and to published services only
    via `is_published`.

    NOT the same thing as the services section of the knowledge base
    file: that one is marketing copy describing the hospital's service
    lines as a whole, with no per-branch information at all. When the
    question is "what services does THIS BRANCH provide?", only this
    endpoint can answer it - see tools.list_branch_services.

    `service_type_ids`: real serviceTypeId GUIDs from this tenant's own
    Services/GetList request schema - NOT the same thing as a
    specialtyId (see tools._lab_test_specialty_id, which filters
    DOCTORS by specialty; this filters SERVICES by type directly, at
    this endpoint). Confirmed per-tenant values live in
    tools._lab_service_type_id / config.CLIENT_LAB_ENTITY_NAMES - never
    hardcode a value here, and never guess one for a client where it
    isn't configured.

    `pageNumber` must be 1 or above, not 0 - same as every other
    paged endpoint here (see get_specialties()'s note)."""

    url = f"{base_url}/api/Services/GetList"
    payload = {
        "pageNumber": 1,
        "pageSize": page_size,
        "isPublished": is_published,
    }

    if branch_ids:
        payload["branchIds"] = branch_ids
    if service_type_ids:
        payload["serviceTypeIds"] = service_type_ids

    if _cms_sso_for(base_url) is not None:
        # Active services only (the guide, 6.6) - staff see disabled ones too.
        return _cms_first(base_url, lambda: _cms_list(base_url, "/api/Services/GetList", {**payload, "status": 1}, language),
                          lambda portal: _post_json(f"{portal}/api/Services/GetList", payload, client_id=client_id, language=language),
                          "Services/GetList")
    return _post_json(url, payload, client_id=client_id, language=language)


def get_patient_info(
    base_url: str,
    mobile_number: str,
    page_size: int = 50,
    client_id: Optional[str] = None,
    sso: Optional[dict] = None,
) -> dict:
    """POST {cms base_url}/api/GuestPatients/GetList.
    (Moved from portal-api: it answers 404 there since the 2026-09-28 API
    change. cms-api takes the same body and returns the same items -
    patientFullName / mobileNumber / email - but needs the SSO token.)

    Looks up whether a patient is already registered by phone number, so a
    returning patient is not asked for their name again. An empty result
    (totalCount=0) means this number is not registered yet."""

    payload = {
        "pageNumber": 1,
        "pageSize": page_size,
        "mobileNumber": mobile_number,
    }

    return _cms_request("post", base_url, "/api/GuestPatients/GetList", sso=sso, json=payload)


def _put_json(url: str, payload: dict, client_id: Optional[str] = None) -> dict:
    """Generic PUT + envelope handling, mirroring _post_json exactly but
    for the one confirmed PUT endpoint (GuestBookings/Update)."""

    logger.info("PUT %s payload=%s", url, payload)

    response, last_timeout, last_exc = _request_with_retry(
        "put", url, idempotent=False, json=payload, headers=_headers(client_id=client_id),
    )

    if response is None:
        if last_timeout:
            logger.warning("Request timed out: %s", url)
            return _result(False, error="timeout")
        logger.exception("Request failed: %s", url)
        return _result(False, error=str(last_exc) if last_exc else "request_failed")

    if response.status_code >= 500:
        logger.error("GuestBookings/Update server error: %s status=%s payload=%s body=%s", url, response.status_code, payload, response.text[:1000])
        return _result(False, response.status_code, error="server_error")

    if response.status_code == 404:
        logger.error("GuestBookings/Update endpoint NOT FOUND (404): %s payload=%s body=%s", url, payload, response.text[:500])
        return _result(False, response.status_code, error="endpoint_not_found")

    if response.status_code >= 400:
        details = _validation_details(response)
        logger.error(
            "GuestBookings/Update validation error: %s status=%s payload=%s body=%r headers=%s rejected_fields=%s",
            url, response.status_code, payload, response.text[:1000], dict(response.headers),
            [d["field"] for d in details] or "unknown",
        )
        return _result(False, response.status_code, error="validation_error", details=details)

    try:
        body = response.json()
    except ValueError:
        return _result(False, response.status_code, error="invalid_json_response")

    if not body:
        return _result(False, response.status_code, error="empty_response")

    if not body.get("isSuccess"):
        return _result(False, response.status_code, data=body, error="api_reported_failure")

    return _result(True, response.status_code, data=body.get("data", {}))


def reschedule_booking(
    base_url: str,
    booking_id: str,
    slot: dict,
    new_from: str,
    new_to: str,
    language: Optional[str] = None,
    sso: Optional[dict] = None,
) -> dict:
    """PUT {cms base_url}/api/Bookings/Update.
    (Moved from portal-api PUT /api/GuestBookings/Update.)

    cms-api updates the WHOLE booking, so the body is built from the
    booking's current state (GetById: rowVersion, bookingRefNum, patientId,
    guestPatientId - exactly one of the two patient ids is null and both
    are sent back as returned) plus the slot the patient picked
    (`slot`: branchId, doctorId, serviceId, spaceId, scheduleId).

    A stale rowVersion (the booking changed in between) is retried once
    with a fresh GetById."""

    result = _result(False, error="request_failed")

    # A GUEST booking (made through the assistant or the website) moves
    # with GuestBookings/Update and only the new times (the guide, 4.4);
    # a registered patient's booking keeps the full Bookings/Update below.
    if _cms_sso_for(base_url) is not None:
        guest = _cms_request("post", base_url, "/api/GuestBookings/Get", sso=sso or _cms_sso_for(base_url),
                             json={"id": booking_id})
        if guest["success"] and isinstance(guest.get("data"), dict) and guest["data"].get("guestPatientId"):
            return _cms_request("put", base_url, "/api/GuestBookings/Update", language=language,
                                sso=sso or _cms_sso_for(base_url),
                                json={"id": booking_id, "fromBookingTime": new_from, "toBookingTime": new_to})

    for attempt in (1, 2):
        current = _get_booking_full(base_url, booking_id, sso=sso)
        if not current["success"]:
            return current

        booking = current["data"] or {}
        payload = {
            "id": booking_id,
            "rowVersion": booking.get("rowVersion"),
            "bookingRefNum": booking.get("bookingRefNum"),
            "patientId": booking.get("patientId"),
            "guestPatientId": booking.get("guestPatientId"),
            "branchId": slot.get("branchId") or booking.get("branchId"),
            "spaceId": slot.get("spaceId"),
            "doctorId": slot.get("doctorId") or booking.get("doctorId"),
            "doctorScheduleId": slot.get("scheduleId"),
            "serviceId": slot.get("serviceId"),
            "bookingTimeFrom": new_from,
            "bookingTimeTo": new_to,
        }

        result = _cms_request("put", base_url, "/api/Bookings/Update", language=language, sso=sso, json=payload)

        # 4xx here is either a stale rowVersion (worth one more read) or a
        # real rejection (a second try gets the same answer, harmlessly).
        if result["success"] or result.get("status_code") is None or result["status_code"] < 400 or result["status_code"] >= 500:
            break

    return result


def create_booking(
    base_url: str,
    patient_full_name: str,
    mobile_number: str,
    branch_id: str,
    doctor_id: str,
    service_id: str,
    service_price,
    booking_time_from: str,
    booking_time_to: str,
    specialty_id: str,
    doctor_schedule_id: str,
    space_id: str,
    email: str = "",
    client_id: Optional[str] = None,
) -> dict:
    """POST {base_url}/api/GuestBookings/Reservation.

    Creates a brand new booking - confirmed directly from a real
    production n8n workflow's exact field list. ALL the id fields
    (branchId, doctorId, serviceId, servicePrice, specialtyId,
    doctorScheduleId, spaceId) must come from a slot the caller just
    re-verified is still available (via get_doctor_schedule_slots) -
    never invented or reused from an earlier, potentially-stale lookup.
    Returns the raw API response - `data` is the new booking's own GUID
    id (pass this to get_booking_by_id to retrieve its bookingRefNum)."""

    url = f"{base_url}/api/GuestBookings/Reservation"
    payload = {
        "patientFullName": patient_full_name,
        "mobileNumber": mobile_number,
        "email": email,
        "branchId": branch_id,
        "doctorId": doctor_id,
        "serviceId": service_id,
        "servicePrice": service_price,
        "bookingTimeFrom": booking_time_from,
        "bookingTimeTo": booking_time_to,
        "specialtyId": specialty_id,
        "doctorScheduleId": doctor_schedule_id,
        "spaceId": space_id,
    }

    if _cms_sso_for(base_url) is not None:
        # Sent ONCE (the guide, 4.3): a timed-out reservation may have been
        # made, so it is never retried automatically.
        # Falls back ONLY on 401/403 (missing Bookings.Create): nothing was
        # made. A 404 is an unknown branch - an answer, not a route problem.
        return _cms_first(
            base_url,
            lambda: _cms_request("post", base_url, "/api/GuestBookings/Reservation",
                                 sso=_cms_sso_for(base_url), retry=False, json=payload),
            lambda portal: _post_json(f"{portal}/api/GuestBookings/Reservation", payload, client_id=client_id, idempotent=False),
            "GuestBookings/Reservation", fallback_errors=("authentication_error", "not_configured"))
    return _post_json(url, payload, client_id=client_id, idempotent=False)


def _get_booking_full(base_url: str, booking_id: str, sso: Optional[dict] = None) -> dict:
    """The whole booking with its rowVersion, for the registered-patient
    Bookings/Update (GuestBookings/Get carries no rowVersion)."""
    return _cms_request("get", base_url, "/api/Bookings/GetById", sso=sso or _cms_sso_for(base_url),
                        params={"Id": booking_id})


def get_booking_by_id(base_url: str, booking_id: str, client_id: Optional[str] = None, sso: Optional[dict] = None) -> dict:
    """GET {cms base_url}/api/Bookings/GetById?Id={id}.
    (Moved from portal-api POST /api/GuestBookings/Get.)

    Fetches one booking by its GUID id. Used right after create_booking
    succeeds, to read back the new booking's bookingRefNum."""

    if _cms_sso_for(base_url) is not None:
        # The guide, 4.2: POST GuestBookings/Get {id}.
        return _cms_request("post", base_url, "/api/GuestBookings/Get", sso=sso or _cms_sso_for(base_url),
                            json={"id": booking_id})
    return _cms_request("get", base_url, "/api/Bookings/GetById", sso=sso, params={"Id": booking_id})
