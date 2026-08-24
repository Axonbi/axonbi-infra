#!/usr/bin/env bash
# Syncs the agent application into the runtime directory and restarts the
# server, only when the tracked branch actually moved.
#
# Two clones, on purpose:
#   /opt/axonbi-infra  tracks main -- this script, nginx configs, systemd units
#   /opt/agent-src     tracks the application branch -- the graph itself
# They are separate because the tooling and the app live on different
# branches; a single clone would lose this script the moment it checked the
# app branch out.
#
# Deliberately does NOT touch nginx or the certificate: a bad commit must not
# be able to take down TLS or lock anyone out. Run setup-proxy.sh by hand
# after changing anything under proxy/.
set -euo pipefail

SRC="${SRC:-/opt/agent-src}"
RUNTIME="${RUNTIME:-/opt/langgraph}"
BRANCH="${BRANCH:-cancel-agent-main}"
VENV="${VENV:-$RUNTIME/.venv}"

cd "$SRC"
git fetch --quiet origin "$BRANCH"

local_rev=$(git rev-parse HEAD)
remote_rev=$(git rev-parse "origin/$BRANCH")
if [ "$local_rev" = "$remote_rev" ]; then
    exit 0
fi

echo "$(date -Is) deploying ${local_rev:0:8} -> ${remote_rev:0:8}"
git reset --hard --quiet "origin/$BRANCH"

# Checked AFTER the reset so it reflects what is about to be synced. Without
# this, a commit that removed langgraph.json would make the --delete below
# empty the runtime directory and take the virtualenv with it.
if [ ! -f "$SRC/langgraph.json" ]; then
    echo "refusing to deploy $(git rev-parse --short HEAD): langgraph.json missing" >&2
    echo "a sync with --delete would wipe $RUNTIME" >&2
    exit 1
fi

# .venv, .env and .langgraph_api live in the runtime directory and must
# survive; the venv in particular has absolute paths baked in and cannot be
# recreated by a sync. .git is excluded because the runtime dir is not a clone.
rsync -a --delete \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '.env' \
    --exclude '.langgraph_api/' \
    --exclude '__pycache__/' \
    "$SRC/" "$RUNTIME/"

# Cheap when nothing changed, and the alternative is a deploy that silently
# fails at import time on a newly added dependency.
if [ -f "$RUNTIME/requirements.txt" ]; then
    VIRTUAL_ENV="$VENV" /root/.local/bin/uv pip install -q -r "$RUNTIME/requirements.txt"
fi

systemctl restart langgraph-dev
echo "$(date -Is) deployed $(git rev-parse --short HEAD)"
