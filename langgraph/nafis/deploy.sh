#!/usr/bin/env bash
# Pulls the agent project and restarts it, only when the tracked branch moved.
#
# Does nothing until /etc/langgraph/deploy.env names a REPO, so the timer can be
# enabled before the AI engineer's project exists.
#
# Deliberately does NOT touch Caddy or certificates: this box also serves
# production n8n, and a bad agent commit must not be able to take TLS down.
set -euo pipefail

CONF=/etc/langgraph/deploy.env
[ -r "$CONF" ] && . "$CONF"

REPO="${REPO:-}"
BRANCH="${BRANCH:-main}"
SRC="${SRC:-/opt/agent-src}"
RUNTIME="${RUNTIME:-/opt/langgraph}"
VENV="${VENV:-$RUNTIME/.venv}"
UV=/root/.local/bin/uv

if [ -z "$REPO" ]; then
    exit 0   # nothing configured yet
fi

if [ ! -d "$SRC/.git" ]; then
    echo "$(date -Is) cloning $REPO ($BRANCH)"
    rm -rf "${SRC:?}"/* "${SRC:?}"/.[!.]* 2>/dev/null || true
    git clone -q -b "$BRANCH" "$REPO" "$SRC"
fi

cd "$SRC"
git remote set-url origin "$REPO"
git fetch --quiet origin "$BRANCH"

local_rev=$(git rev-parse HEAD)
remote_rev=$(git rev-parse "origin/$BRANCH")
if [ "$local_rev" = "$remote_rev" ]; then
    exit 0
fi

echo "$(date -Is) deploying ${local_rev:0:8} -> ${remote_rev:0:8}"
git checkout -q "$BRANCH" 2>/dev/null || true
git reset --hard --quiet "origin/$BRANCH"

# Checked AFTER the reset so it reflects what is about to be synced. Without
# this, a commit that removed langgraph.json would make the --delete below
# empty the runtime directory and take the virtualenv with it.
if [ ! -f "$SRC/langgraph.json" ]; then
    echo "refusing to deploy $(git rev-parse --short HEAD): langgraph.json missing" >&2
    echo "a sync with --delete would wipe $RUNTIME" >&2
    exit 1
fi

# .venv, .env and .langgraph_api live in the runtime directory and must survive;
# the venv in particular has absolute paths baked in and cannot be recreated by
# a sync. .git is excluded because the runtime dir is not a clone.
rsync -a --delete \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '.env' \
    --exclude '.langgraph_api/' \
    --exclude '__pycache__/' \
    "$SRC/" "$RUNTIME/"

# Cheap when unchanged, and the alternative is a deploy that looks clean and
# then fails at import on a newly added dependency.
if [ -f "$RUNTIME/requirements.txt" ]; then
    VIRTUAL_ENV="$VENV" "$UV" pip install -q -r "$RUNTIME/requirements.txt" || {
        echo "dependency install failed - not restarting services" >&2
        exit 1
    }
fi

# Restart only what the project actually provides.
[ -f "$RUNTIME/langgraph.json" ] && systemctl restart langgraph-dev || true
[ -f "$RUNTIME/app.py" ]         && systemctl restart langgraph-api || true

echo "$(date -Is) deployed $(git rev-parse --short HEAD)"
