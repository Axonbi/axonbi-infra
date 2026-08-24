#!/usr/bin/env bash
# Swaps the self-signed placeholder for a real Let's Encrypt certificate.
#
# axonbi.com is on supremedns.com nameservers (SOA -> cloudlogin.co), and ports
# 80 and 443 of 41.38.65.53 both forward to other hosts, so http-01 and
# tls-alpn-01 are unavailable. DNS-01 is the only automatable challenge.
#
# Export the DirectAdmin credentials, then run:
#   export DA_Api="https://your-panel.host:2222"
#   export DA_Api_Insecure=1
#   bash /opt/langgraph/proxy/issue-cert.sh
#
# For cPanel instead, swap --dns dns_da for --dns dns_cpanel and export
# cPanel_Username / cPanel_Apitoken / cPanel_Hostname.
set -euo pipefail

DOMAIN="langchain.axonbi.com"
CERT_DIR="/etc/ssl/langgraph"
ACME="/root/.acme.sh/acme.sh"

: "${DA_Api:?export DA_Api first, e.g. https://panel.host:2222 (with user:pass in the URL)}"

"$ACME" --issue --dns dns_da -d "$DOMAIN" --server letsencrypt

# --install-cert writes to the paths nginx already points at and reloads it, so
# renewals need no further action.
"$ACME" --install-cert -d "$DOMAIN" \
    --key-file       "$CERT_DIR/privkey.pem" \
    --fullchain-file "$CERT_DIR/fullchain.pem" \
    --reloadcmd      "systemctl reload nginx"

echo
echo "installed:"
openssl x509 -noout -subject -issuer -dates -in "$CERT_DIR/fullchain.pem"
