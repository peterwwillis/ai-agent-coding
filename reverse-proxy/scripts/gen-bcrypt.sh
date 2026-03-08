#!/usr/bin/env bash
# gen-bcrypt.sh – Generate a bcrypt password hash for basic-auth placeholders
# =============================================================================
# Usage:
#   ./gen-bcrypt.sh [password]
#
#   If no password is given as an argument, the script will prompt for one
#   interactively (hidden input).
#
# Output:
#   Prints a bcrypt hash string suitable for:
#     - Traefik basicAuth users list  (dollar signs are already doubled: $$2y$$...)
#     - Caddy basicauth directive      (single dollar signs: $2y$...)
#
# Requirements: htpasswd (apache2-utils / httpd-tools) OR python3 with bcrypt,
#               OR docker (falls back to caddy image)
# =============================================================================
set -euo pipefail

# --- Read password -----------------------------------------------------------
if [[ $# -ge 1 ]]; then
    PASSWORD="$1"
else
    read -r -s -p "Enter password: " PASSWORD
    echo ""
    read -r -s -p "Confirm password: " PASSWORD2
    echo ""
    if [[ "$PASSWORD" != "$PASSWORD2" ]]; then
        echo "Passwords do not match." >&2
        exit 1
    fi
fi

if [[ -z "$PASSWORD" ]]; then
    echo "Password must not be empty." >&2
    exit 1
fi

# --- Generate hash -----------------------------------------------------------
generate_hash() {
    local pass="$1"

    if command -v htpasswd &>/dev/null; then
        # htpasswd from apache2-utils (recommended, fast)
        htpasswd -bnBC 10 "" "$pass" | tr -d ':\n'
    elif command -v python3 &>/dev/null && python3 -c "import bcrypt" 2>/dev/null; then
        # Python bcrypt fallback
        python3 -c "
import bcrypt, sys
pw = sys.argv[1].encode()
print(bcrypt.hashpw(pw, bcrypt.gensalt(rounds=10)).decode(), end='')
" "$pass"
    elif command -v docker &>/dev/null; then
        # Caddy Docker image fallback (produces Caddy-compatible hash)
        docker run --rm caddy:2.9-alpine caddy hash-password --plaintext "$pass"
    else
        echo "ERROR: No supported tool found to generate bcrypt hash." >&2
        echo "Install one of: apache2-utils (htpasswd), python3-bcrypt, or docker." >&2
        exit 1
    fi
}

RAW_HASH="$(generate_hash "$PASSWORD")"

# Strip a leading ":" if htpasswd included username field
RAW_HASH="${RAW_HASH#:}"

# Ensure hash starts with $2y$ (normalize $2a$ and $2b$ variants)
CADDY_HASH="${RAW_HASH/\$2a\$/\$2y\$}"
CADDY_HASH="${CADDY_HASH/\$2b\$/\$2y\$}"

# Traefik YAML requires dollar signs doubled inside quoted strings
TRAEFIK_HASH="${CADDY_HASH//\$/\$\$}"

echo ""
echo "=== Caddy basicauth hash (single \$ signs) ==="
echo "    admin ${CADDY_HASH}"
echo ""
echo "=== Traefik basicAuth users hash (doubled \$\$ signs for YAML) ==="
echo "    \"admin:${TRAEFIK_HASH}\""
echo ""
echo "Usage:"
echo "  Caddy   – paste the 'admin <hash>' line into the basicauth block in Caddyfile"
echo "  Traefik – paste the quoted string into the users list in dynamic/middlewares.yml"
