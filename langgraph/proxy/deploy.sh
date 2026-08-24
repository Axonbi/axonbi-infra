#!/usr/bin/env bash
# Pulls the repo and syncs graph code into the runtime directory, restarting
# the server only when the remote actually moved. Driven by langgraph-deploy.timer.
#
# Deliberately does NOT touch nginx or the certificate: a bad commit should not
# be able to take down TLS or lock you out. Run setup-proxy.sh by hand after
# changing anything under proxy/.
set -euo pipefail

REPO="${REPO:-/opt/axonbi-infra}"
RUNTIME="${RUNTIME:-/opt/langgraph}"
BRANCH="${BRANCH:-main}"

cd "$REPO"
git fetch --quiet origin "$BRANCH"

local_rev=$(git rev-parse HEAD)
remote_rev=$(git rev-parse "origin/$BRANCH")
if [ "$local_rev" = "$remote_rev" ]; then
    exit 0
fi

echo "deploying ${local_rev:0:8} -> ${remote_rev:0:8}"
git reset --hard --quiet "origin/$BRANCH"

# Checked AFTER the reset, so it reflects what is about to be synced. Without
# this, a commit that removed langgraph/ would make the --delete below empty
# the runtime directory.
if [ ! -f "$REPO/langgraph/langgraph.json" ]; then
    echo "refusing to deploy $(git rev-parse --short HEAD): langgraph/langgraph.json missing" >&2
    echo "a sync with --delete would wipe $RUNTIME" >&2
    exit 1
fi

# .venv, .env and .langgraph_api live in the runtime dir and must survive; the
# venv in particular has absolute paths baked in and cannot be recreated by a
# sync.
rsync -a --delete \
    --exclude '.venv/' \
    --exclude '.env' \
    --exclude '.langgraph_api/' \
    --exclude '__pycache__/' \
    "$REPO/langgraph/" "$RUNTIME/"

systemctl restart langgraph-dev
echo "deployed $(git rev-parse --short HEAD)"
