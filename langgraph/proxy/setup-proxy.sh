#!/usr/bin/env bash
# Stands up TLS + basic auth in front of the LangGraph dev server.
# Idempotent: safe to re-run. Never overwrites a real certificate with the
# self-signed placeholder.
#
# Secrets are NOT in this repo (it is public). Copy proxy.env.example to
# /etc/langgraph/proxy.env and fill it in first.
set -euo pipefail

CONF_FILE="${PROXY_ENV:-/etc/langgraph/proxy.env}"
if [ ! -r "$CONF_FILE" ]; then
    echo "missing $CONF_FILE -- copy proxy.env.example there and fill it in" >&2
    exit 1
fi
# shellcheck source=/dev/null
. "$CONF_FILE"
: "${SECRET_PATH:?set in $CONF_FILE}"
: "${STUDIO_USER:?set in $CONF_FILE}"
: "${STUDIO_PASS:?set in $CONF_FILE}"
: "${DOMAIN:?set in $CONF_FILE}"

SRC="$(cd "$(dirname "$0")" && pwd)"
CERT_DIR="/etc/ssl/langgraph"

echo "==> 1. certificate"
mkdir -p "$CERT_DIR"
if [ -s "$CERT_DIR/fullchain.pem" ]; then
    echo "    existing cert kept: $(openssl x509 -noout -issuer -in "$CERT_DIR/fullchain.pem")"
else
    # Placeholder only. issue-cert.sh replaces this with Let's Encrypt and
    # writes to the same two paths, so nginx needs no further edits.
    openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
        -keyout "$CERT_DIR/privkey.pem" -out "$CERT_DIR/fullchain.pem" \
        -subj "/CN=$DOMAIN" -addext "subjectAltName=DNS:$DOMAIN" 2>/dev/null
    echo "    self-signed placeholder created (browser will warn until swapped)"
fi
chmod 600 "$CERT_DIR/privkey.pem"

echo "==> 2. basic auth user"
htpasswd -cbB /etc/nginx/langgraph.htpasswd "$STUDIO_USER" "$STUDIO_PASS" 2>/dev/null
chmod 640 /etc/nginx/langgraph.htpasswd
chown root:www-data /etc/nginx/langgraph.htpasswd
echo "    user '$STUDIO_USER' set"

echo "==> 3. nginx config"
install -m 644 "$SRC/langgraph-auth.conf"  /etc/nginx/langgraph-auth.conf
install -m 644 "$SRC/langgraph-proxy.conf" /etc/nginx/langgraph-proxy.conf
sed "s|SECRET_PATH|$SECRET_PATH|g" "$SRC/langgraph-studio.conf" \
    > /etc/nginx/sites-available/langgraph-studio.conf
ln -sf /etc/nginx/sites-available/langgraph-studio.conf \
       /etc/nginx/sites-enabled/langgraph-studio.conf
nginx -t
systemctl reload nginx || systemctl start nginx
echo "    nginx listening on 2024/tls"

echo "==> 4. langgraph service"
install -m 644 "$SRC/langgraph-dev.service"     /etc/systemd/system/langgraph-dev.service
install -m 644 "$SRC/cancel-agent-api.service" /etc/systemd/system/cancel-agent-api.service
systemctl daemon-reload
systemctl enable --now langgraph-dev cancel-agent-api
sleep 18

echo "==> 5. verify"
printf '    langgraph on 127.0.0.1:2025 : '
curl -s -o /dev/null -w '%{http_code}\n' --max-time 10 http://127.0.0.1:2025/ok
printf '    probe /info, no password    : '
curl -sk -o /dev/null -w '%{http_code} (expect 200)\n' --max-time 10 \
    "https://$DOMAIN:2024/info"
printf '    /assistants, no password    : '
curl -sk -o /dev/null -w '%{http_code} (expect 401)\n' --max-time 10 \
    -X POST "https://$DOMAIN:2024/assistants/search" \
    -H 'Content-Type: application/json' -d '{"limit":1}'
printf '    /assistants + password      : '
curl -sk -o /dev/null -w '%{http_code} (expect 200)\n' --max-time 10 \
    -u "$STUDIO_USER:$STUDIO_PASS" -X POST "https://$DOMAIN:2024/assistants/search" \
    -H 'Content-Type: application/json' -d '{"limit":1}'

echo
echo "baseUrl for Studio:  https://$DOMAIN:2024"
