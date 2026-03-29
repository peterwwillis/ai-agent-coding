#!/usr/bin/env bash
# gen-certs.sh – Generate local TLS certificates for the reverse-proxy
# =============================================================================
# Usage:
#   ./gen-certs.sh [--self-signed] [--ca] [--domain <name>] [--out <dir>]
#
# Options:
#   --self-signed    Generate a self-signed certificate (quick, no CA needed)
#   --ca             Generate a local CA + a CA-signed server certificate (recommended)
#   --domain <name>  Common name / SAN for the certificate (default: mac.local)
#   --out <dir>      Output directory (default: ./certs)
#
# After running this script:
#   1. Copy the output files to reverse-proxy/traefik/certs/ and/or
#      reverse-proxy/caddy/certs/ (or adjust paths in docker-compose.yml).
#   2. Trust the CA certificate in your OS/browser (see docs/README.md).
#
# Requirements: openssl (available on macOS and most Linux distros)
# =============================================================================
set -euo pipefail

# --- Defaults ----------------------------------------------------------------
MODE="--ca"       # default to CA-signed (recommended)
DOMAIN="mac.local"
OUT_DIR="./certs"
DAYS=825          # max accepted by most browsers (~2 years)

# --- Argument parsing --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --self-signed) MODE="--self-signed" ;;
        --ca)          MODE="--ca" ;;
        --domain)      DOMAIN="$2"; shift ;;
        --out)         OUT_DIR="$2"; shift ;;
        -h|--help)
            sed -n '/^# Usage/,/^# Requirements/p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

mkdir -p "$OUT_DIR"

echo "==> Output directory: $OUT_DIR"
echo "==> Domain:           $DOMAIN"
echo "==> Mode:             $MODE"
echo ""

# --- Common SAN extension ----------------------------------------------------
# Adds the domain as both a DNS SAN and (if it looks like an IP) an IP SAN.
san_value="DNS:${DOMAIN},DNS:*.${DOMAIN}"
if [[ "$DOMAIN" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    san_value="IP:${DOMAIN}"
fi

# =============================================================================
# Self-signed certificate
# =============================================================================
generate_self_signed() {
    echo ">>> Generating self-signed certificate for: $DOMAIN"
    openssl req -x509 -newkey rsa:4096 \
        -keyout "${OUT_DIR}/local.key" \
        -out    "${OUT_DIR}/local.crt" \
        -days   "$DAYS" \
        -nodes \
        -subj   "/CN=${DOMAIN}/O=Local Dev" \
        -addext "subjectAltName=${san_value}" \
        -addext "keyUsage=digitalSignature,keyEncipherment" \
        -addext "extendedKeyUsage=serverAuth"

    echo ""
    echo "==> Files created:"
    echo "    ${OUT_DIR}/local.key  – private key"
    echo "    ${OUT_DIR}/local.crt  – self-signed certificate"
    echo ""
    echo "NOTE: Self-signed certs will show browser warnings."
    echo "      Trust this cert in your OS keychain or use --ca mode."
}

# =============================================================================
# Local CA + CA-signed server certificate (recommended)
# =============================================================================
generate_ca_signed() {
    echo ">>> Generating local CA and CA-signed certificate for: $DOMAIN"

    # --- 1. Generate CA key and self-signed CA cert --------------------------
    openssl genrsa -out "${OUT_DIR}/ca.key" 4096
    openssl req -x509 -new -nodes \
        -key  "${OUT_DIR}/ca.key" \
        -sha256 \
        -days 3650 \
        -out  "${OUT_DIR}/ca.crt" \
        -subj "/CN=Local Dev CA/O=Local Dev CA"

    # --- 2. Generate server key and CSR --------------------------------------
    openssl genrsa -out "${OUT_DIR}/local.key" 4096
    openssl req -new \
        -key  "${OUT_DIR}/local.key" \
        -out  "${OUT_DIR}/local.csr" \
        -subj "/CN=${DOMAIN}/O=Local Dev"

    # --- 3. Create an extension file for the server cert --------------------
    ext_file=$(mktemp /tmp/san_ext_XXXXXX.cnf)
    cat > "$ext_file" <<EOF
[req]
distinguished_name = req_distinguished_name
[req_distinguished_name]
[v3_req]
subjectAltName = ${san_value}
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
basicConstraints = CA:FALSE
EOF

    # --- 4. Sign the server cert with the local CA ---------------------------
    openssl x509 -req \
        -in     "${OUT_DIR}/local.csr" \
        -CA     "${OUT_DIR}/ca.crt" \
        -CAkey  "${OUT_DIR}/ca.key" \
        -CAcreateserial \
        -out    "${OUT_DIR}/local.crt" \
        -days   "$DAYS" \
        -sha256 \
        -extfile "$ext_file" \
        -extensions v3_req

    rm -f "$ext_file" "${OUT_DIR}/local.csr"

    echo ""
    echo "==> Files created:"
    echo "    ${OUT_DIR}/ca.key    – CA private key     (KEEP SAFE, do not share)"
    echo "    ${OUT_DIR}/ca.crt    – CA certificate     (add to OS/browser trust store)"
    echo "    ${OUT_DIR}/local.key – server private key"
    echo "    ${OUT_DIR}/local.crt – server certificate (signed by local CA)"
    echo ""
    echo "Next steps – trust the CA certificate:"
    echo ""
    echo "  macOS:"
    echo "    sudo security add-trusted-cert -d -r trustRoot \\"
    echo "      -k /Library/Keychains/System.keychain ${OUT_DIR}/ca.crt"
    echo ""
    echo "  Linux (Debian/Ubuntu):"
    echo "    sudo cp ${OUT_DIR}/ca.crt /usr/local/share/ca-certificates/local-dev-ca.crt"
    echo "    sudo update-ca-certificates"
    echo ""
    echo "  Linux (RHEL/Fedora):"
    echo "    sudo cp ${OUT_DIR}/ca.crt /etc/pki/ca-trust/source/anchors/local-dev-ca.crt"
    echo "    sudo update-ca-trust"
    echo ""
    echo "  Firefox: Settings > Privacy > View Certificates > Authorities > Import"
}

# --- Run selected mode -------------------------------------------------------
case "$MODE" in
    --self-signed) generate_self_signed ;;
    --ca)          generate_ca_signed ;;
esac

echo "Done."
