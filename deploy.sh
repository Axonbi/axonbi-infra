#!/usr/bin/env bash
#
# Deploy the agent from the GitHub branch to /opt/langgraph.
#
# WHY THIS EXISTS. /opt/langgraph is not a git checkout and nothing
# pulls into it, so every deploy was a manual file copy. In one
# afternoon that cost: an scp that aborted silently on a host-key
# prompt, "C:\..." parsed by scp as a host named "c", a stale file left
# in /tmp and installed twice, one file simply forgotten, and a GitHub
# raw response served from cache. Each one looked identical from the
# outside - the bot kept answering with the old wording - so each one
# was diagnosed from scratch.
#
# Everything below is that afternoon turned into a script:
#   - fetches from the branch with the cache busted
#   - refuses to install anything that does not parse
#   - backs up exactly what it is about to overwrite
#   - restarts, then PROVES the service answers
#   - restores the backup automatically if it does not
#
# Usage:
#   ./deploy.sh                 deploy the default branch below
#   BRANCH=other ./deploy.sh    deploy a different branch
#   ./deploy.sh --check         download and verify, change nothing
#   ./deploy.sh --rollback      restore the most recent backup
#
set -euo pipefail

REPO="${REPO:-Axonbi/axonbi-infra}"
BRANCH="${BRANCH:-multiagent-up1}"
APP_DIR="${APP_DIR:-/opt/langgraph}"
BACKUP_ROOT="${BACKUP_ROOT:-/opt/langgraph-backups}"
SERVICES="${SERVICES:-cancel-agent-api.service langgraph-dev.service}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/chat}"
HEALTH_CLIENT="${HEALTH_CLIENT:-medtown2}"
KEEP_BACKUPS="${KEEP_BACKUPS:-10}"

# The files this script owns. Deliberately ONLY source code: the CSVs
# and knowledge_base/ are per-clinic data that may legitimately have
# been edited on the server, and silently replacing them from a branch
# is how a clinic loses its own configuration. Add them here if you
# ever decide the branch is authoritative for those too.
FILES=(
  graph.py tools.py prompts.py rag.py main.py app.py config.py
  state.py api.py progress.py start.py
  agents/__init__.py agents/router.py agents/registry.py
  agents/sections.py agents/response_contract.py agents/hard_rules.py
)

RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; DIM=$'\033[2m'; OFF=$'\033[0m'
say()  { printf '%s\n' "$*"; }
ok()   { printf '%s  ok %s %s\n' "$GREEN" "$OFF" "$*"; }
warn() { printf '%s warn%s %s\n' "$YELLOW" "$OFF" "$*"; }
die()  { printf '%s FAIL%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }
step() { printf '\n%s== %s ==%s\n' "$DIM" "$*" "$OFF"; }

[[ $EUID -eq 0 ]] || die "run as root (it writes to $APP_DIR and restarts services)"

MODE="deploy"
case "${1:-}" in
  --check)    MODE="check" ;;
  --rollback) MODE="rollback" ;;
  "")         ;;
  *)          die "unknown option '$1' (use --check or --rollback)" ;;
esac

# ----------------------------------------------------------
# Rollback
# ----------------------------------------------------------
if [[ "$MODE" == "rollback" ]]; then
  latest="$(ls -1d "$BACKUP_ROOT"/* 2>/dev/null | sort | tail -1 || true)"
  [[ -n "$latest" ]] || die "no backups in $BACKUP_ROOT"
  step "Restoring $latest"
  (cd "$latest" && find . -type f -print0 | while IFS= read -r -d '' f; do
      install -D -m 644 "$f" "$APP_DIR/${f#./}"
      printf '  restored %s\n' "${f#./}"
  done)
  systemctl restart $SERVICES
  ok "restored and restarted"
  exit 0
fi

# ----------------------------------------------------------
# 1. Download
# ----------------------------------------------------------
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

step "Downloading $REPO@$BRANCH"
for f in "${FILES[@]}"; do
  # The cache buster is not optional: raw.githubusercontent served a
  # several-minute-old copy of a file that had already been pushed, and
  # the deploy silently installed the previous version.
  url="https://raw.githubusercontent.com/$REPO/$BRANCH/$f?cb=$(date +%s%N)"
  mkdir -p "$STAGE/$(dirname "$f")"
  if ! curl -fsSL -H 'Cache-Control: no-cache' -H 'Pragma: no-cache' -o "$STAGE/$f" "$url"; then
    die "could not download $f - is it on branch $BRANCH?"
  fi
  [[ -s "$STAGE/$f" ]] || die "$f downloaded empty"
  printf '  %-34s %8s bytes\n' "$f" "$(stat -c%s "$STAGE/$f")"
done
ok "${#FILES[@]} file(s) downloaded"

# ----------------------------------------------------------
# 2. Verify before touching anything
# ----------------------------------------------------------
step "Verifying"
PY="$APP_DIR/.venv/bin/python3"; [[ -x "$PY" ]] || PY="$(command -v python3)"
for f in "${FILES[@]}"; do
  [[ "$f" == *.py ]] || continue
  "$PY" -c "import ast,sys; ast.parse(open(sys.argv[1],encoding='utf-8').read())" "$STAGE/$f" \
    || die "$f does not parse - nothing has been changed"
done
ok "every file parses"

CHANGED=(); for f in "${FILES[@]}"; do
  if [[ ! -f "$APP_DIR/$f" ]] || ! cmp -s "$STAGE/$f" "$APP_DIR/$f"; then CHANGED+=("$f"); fi
done

if [[ ${#CHANGED[@]} -eq 0 ]]; then
  ok "already up to date - nothing to deploy"
  exit 0
fi
say "  ${#CHANGED[@]} file(s) differ from what is deployed:"
printf '    %s\n' "${CHANGED[@]}"

if [[ "$MODE" == "check" ]]; then
  ok "--check: verified, nothing changed"
  exit 0
fi

# ----------------------------------------------------------
# 3. Back up, then install
# ----------------------------------------------------------
BACKUP="$BACKUP_ROOT/$(date +%Y-%m-%d-%H%M%S)"
step "Backing up to $BACKUP"
for f in "${CHANGED[@]}"; do
  [[ -f "$APP_DIR/$f" ]] && install -D -m 644 "$APP_DIR/$f" "$BACKUP/$f"
done
ok "backup written"

step "Installing"
for f in "${CHANGED[@]}"; do
  install -D -m 644 "$STAGE/$f" "$APP_DIR/$f"
  printf '  %s\n' "$f"
done
ok "installed"

restore_backup() {
  warn "restoring $BACKUP"
  (cd "$BACKUP" && find . -type f -print0 | while IFS= read -r -d '' f; do
      install -D -m 644 "$f" "$APP_DIR/${f#./}"
  done)
  systemctl restart $SERVICES || true
  sleep 8
}

# ----------------------------------------------------------
# 4. Restart and prove it answers
# ----------------------------------------------------------
step "Restarting"
systemctl restart $SERVICES || { restore_backup; die "restart failed - rolled back"; }
for s in $SERVICES; do
  systemctl is-active --quiet "$s" || { restore_backup; die "$s is not active - rolled back"; }
  ok "$s active"
done

step "Health check"
say "  waiting for the service to come up..."
reply=""
for _ in $(seq 1 12); do
  sleep 5
  reply="$(curl -sS -m 60 -X POST "$HEALTH_URL" \
      -H 'Content-Type: application/json' \
      -d "{\"session_id\":\"deploy-$(date +%s)\",\"client_id\":\"$HEALTH_CLIENT\",\"message\":\"عايز أحجز موعد\"}" \
      2>/dev/null || true)"
  [[ -n "$reply" ]] && break
  say "  not answering yet, retrying..."
done

[[ -n "$reply" ]] || { restore_backup; die "no response from $HEALTH_URL - rolled back"; }

text="$("$PY" -c 'import sys,json
try:
    print(json.loads(sys.stdin.read()).get("reply",""))
except Exception:
    pass' <<<"$reply" || true)"

[[ -n "${text// /}" ]] || { restore_backup; die "the service answered with an empty reply - rolled back"; }

ok "the service answered"
printf '%s\n' "$DIM"; printf '    %s\n' "$text"; printf '%s' "$OFF"

# ----------------------------------------------------------
# 5. Tidy old backups
# ----------------------------------------------------------
mapfile -t old < <(ls -1d "$BACKUP_ROOT"/* 2>/dev/null | sort | head -n -"$KEEP_BACKUPS" || true)
if [[ ${#old[@]} -gt 0 ]]; then
  rm -rf "${old[@]}"
  say "  pruned ${#old[@]} old backup(s), keeping the last $KEEP_BACKUPS"
fi

step "Done"
ok "$REPO@$BRANCH is live in $APP_DIR"
say "  roll back with:  $0 --rollback"
