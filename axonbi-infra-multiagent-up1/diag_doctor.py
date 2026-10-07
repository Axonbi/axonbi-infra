"""Why is a doctor missing from the bot's lists? READ-ONLY - calls Doctors/GetList only.

Run inside the app's container (it uses the container's own SSO env vars):

    CMS_API_BASE_URL=https://<rovan cms host> python diag_doctor.py "العالم"

It asks the API four ways - with the bot's filters (published service +
service schedule) and without them - and prints every doctor whose name
contains the search text, with their specialty and flags.
"""

import os
import sys

import api

name = sys.argv[1] if len(sys.argv) > 1 else "العالم"
base_url = (os.getenv("CMS_API_BASE_URL") or "").rstrip("/")
if not base_url:
    sys.exit("set CMS_API_BASE_URL to the clinic's cms-api host")

# Empty sso dict -> api._sso_settings falls back to the SSO_* env vars.
api.register_cms_host(base_url, {}, portal_url=os.getenv("PORTAL_BASE_URL") or None)

MODES = (
    ("bot filters (published service + schedule)", True, True),
    ("published service only", True, None),
    ("schedule only", None, True),
    ("no filters", None, None),
)

for label, published, schedule in MODES:
    result = api.get_doctors(base_url, has_published_service=published,
                             has_service_schedule=schedule, language="ar")
    if not result.get("success"):
        print(f"[{label}] API error: status={result.get('status_code')} error={result.get('error')}")
        continue
    items = (result.get("data") or {}).get("items") or []
    hits = [d for d in items if name in " ".join(str(d.get(k) or "") for k in
                                                ("name", "altName", "formatedName", "fullName"))]
    print(f"[{label}] {len(items)} doctor(s) returned, {len(hits)} match {name!r}")
    for d in hits:
        print("   ", d.get("name") or d.get("formatedName"), "|",
              "specialties:", d.get("specialtyName") or d.get("specialties"), "|",
              "hasSlots:", d.get("hasSlots"), "|",
              "hasPublishedService:", d.get("hasPublishedService"), "|",
              "hasServiceSchedule:", d.get("hasServiceSchedule"))
