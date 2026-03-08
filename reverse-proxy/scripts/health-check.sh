#!/usr/bin/env bash
# health-check.sh – Basic health-check for reverse-proxy routes
# =============================================================================
# Usage:
#   ./health-check.sh [--host <hostname>] [--port <port>] [--skip-tls-verify]
#
# Options:
#   --host <hostname>     Proxy hostname (default: mac.local)
#   --port <port>         Proxy HTTPS port (default: 443)
#   --skip-tls-verify     Pass -k to curl (for self-signed certs)
#   --user <user:pass>    Basic-auth credentials for protected routes
#
# The script sends an HTTPS request to each configured path prefix and reports
# HTTP status codes. A 200, 301, 302, or 401 indicates the proxy is routing
# correctly (even if the upstream returns an auth challenge or redirect).
# A 502/503 means Traefik/Caddy is running but the upstream is not reachable.
# A connection error means the proxy itself is not running.
# =============================================================================
set -euo pipefail

HOST="mac.local"
PORT="443"
CURL_EXTRA=()
AUTH_HEADER=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)           HOST="$2";              shift ;;
        --port)           PORT="$2";              shift ;;
        --skip-tls-verify) CURL_EXTRA+=("-k")    ;;
        --user)           AUTH_HEADER=("--user" "$2"); shift ;;
        -h|--help)
            sed -n '/^# Usage/,/^# The script/p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

BASE_URL="https://${HOST}:${PORT}"

# Route table: path_prefix -> expected_ok_codes (space-separated)
# 401 is acceptable for protected routes when no auth is provided.
declare -A ROUTES=(
    ["/ollama/api/tags"]="200 401 404"
    ["/llama-swap/"]="200 401 404"
    ["/openwebui/"]="200 301 302"
    ["/n8n/"]="200 301 302 401"
    ["/terminal/"]="200 401"
    ["/code/"]="200 301 302 401"
    ["/searxng/"]="200 301 302"
    ["/llamacpp-rpc/"]="200 401 404"
    ["/vnc/"]="200 401"
)

PASS=0
FAIL=0
WARN=0

echo "==> Health-check: ${BASE_URL}"
echo ""
printf "%-30s  %-6s  %s\n" "PATH" "STATUS" "RESULT"
printf "%-30s  %-6s  %s\n" "----" "------" "------"

for path in "${!ROUTES[@]}"; do
    url="${BASE_URL}${path}"
    ok_codes="${ROUTES[$path]}"

    http_code=$(curl --silent --output /dev/null --write-out "%{http_code}" \
        --max-time 5 \
        --connect-timeout 3 \
        "${CURL_EXTRA[@]}" \
        "${AUTH_HEADER[@]}" \
        "$url" 2>/dev/null || echo "ERR")

    result="FAIL"
    if [[ "$http_code" == "ERR" ]]; then
        result="CONN_ERR"
        FAIL=$((FAIL + 1))
    elif echo "$ok_codes" | grep -qw "$http_code"; then
        result="OK"
        PASS=$((PASS + 1))
    elif [[ "$http_code" == "502" || "$http_code" == "503" ]]; then
        result="UPSTREAM_DOWN"
        WARN=$((WARN + 1))
    else
        FAIL=$((FAIL + 1))
    fi

    printf "%-30s  %-6s  %s\n" "$path" "$http_code" "$result"
done

echo ""
echo "==> Results: ${PASS} OK  |  ${WARN} upstream-down  |  ${FAIL} failed"

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
