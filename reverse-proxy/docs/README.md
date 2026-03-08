# Local Reverse-Proxy for AI Services

A ready-to-use local reverse-proxy setup using **Traefik** or **Caddy** that
routes path-prefixed HTTPS requests to locally-running AI services. No cloud
services, no automatic certificate provisioning – everything works offline on
your LAN.

---

## Table of Contents

1. [Architecture overview](#architecture-overview)
2. [URL scheme](#url-scheme)
3. [Service port assumptions](#service-port-assumptions)
4. [Directory layout](#directory-layout)
5. [macOS setup](#macos-setup)
6. [Linux setup](#linux-setup)
7. [Running the Traefik stack](#running-the-traefik-stack)
8. [Running the Caddy stack](#running-the-caddy-stack)
9. [Route table](#route-table)
10. [Authentication setup](#authentication-setup)
11. [TLS setup](#tls-setup)
12. [Troubleshooting](#troubleshooting)
13. [Security notes](#security-notes)

---

## Architecture overview

```
Browser / curl
      │  HTTPS :443
      ▼
┌─────────────────────┐
│  Traefik  or  Caddy │  ← runs in Docker
│  (reverse proxy)    │
└──────────┬──────────┘
           │  plain HTTP  (host-to-host, no TLS between proxy and services)
           ▼
┌──────────────────────────────────────────────────────┐
│  Services running on the Docker host                 │
│  ollama :11434 │ openwebui :3000 │ n8n :5678 │ ...  │
└──────────────────────────────────────────────────────┘
```

The proxy listens on **HTTPS port 443** and routes each incoming path prefix to
the corresponding local service. Services themselves do **not** need TLS –
they listen on plain HTTP on localhost.

---

## URL scheme

| URL | Service |
|-----|---------|
| `https://<host>/ollama/` | Ollama LLM inference |
| `https://<host>/llama-swap/` | llama-swap model-switcher |
| `https://<host>/openwebui/` | Open WebUI chat front-end |
| `https://<host>/n8n/` | n8n workflow automation |
| `https://<host>/terminal/` | ttyd web terminal |
| `https://<host>/code/` | code-server (VS Code in browser) |
| `https://<host>/searxng/` | SearXNG meta-search |
| `https://<host>/llamacpp-rpc/` | llama.cpp RPC aggregator |
| `https://<host>/vnc/` | noVNC web VNC client |

Replace `<host>` with your machine's hostname or LAN IP (e.g. `mac.local`,
`mybox.local`, or `192.168.1.50`).

---

## Service port assumptions

| Service | Default port | Notes |
|---------|-------------|-------|
| Ollama | 11434 | `ollama serve` |
| llama-swap | 8080 | `llama-swap --port 8080` |
| Open WebUI | 3000 | Docker or bare-metal |
| n8n | 5678 | Docker default |
| ttyd (terminal) | 7681 | ttyd default |
| code-server | 8443 | code-server default |
| SearXNG | 8888 | adjust in SearXNG config |
| llama.cpp RPC | 8081 | `llama-server --port 8081` |
| noVNC | 6080 | noVNC default |

To change a port, edit the upstream `url:` (Traefik) or the `reverse_proxy`
address (Caddy).

---

## Directory layout

```
reverse-proxy/
├── traefik/
│   ├── docker-compose.yml   # Traefik container definition
│   ├── certs/               # TLS cert and key (git-ignored, create manually)
│   │   ├── local.crt
│   │   └── local.key
│   └── dynamic/             # File-provider config (hot-reloaded)
│       ├── tls.yml          # TLS certificate paths
│       ├── middlewares.yml  # basicAuth, stripPrefix, chains
│       └── routes.yml       # Routers and service backends
├── caddy/
│   ├── docker-compose.yml   # Caddy container definition
│   ├── Caddyfile            # All-in-one Caddy config
│   └── certs/               # TLS cert and key (git-ignored, create manually)
│       ├── local.crt
│       └── local.key
├── scripts/
│   ├── gen-certs.sh         # Generate self-signed or CA-signed TLS certs
│   ├── gen-bcrypt.sh        # Generate bcrypt hashes for basic-auth
│   └── health-check.sh      # Validate all proxy routes are reachable
└── docs/
    └── README.md            # This file
```

> **Note:** The `certs/` directories are not committed to version control.
> Generate certificates with the provided script before starting the proxy.

---

## macOS setup

1. **Install Docker Desktop** (includes `host.docker.internal` support automatically).

2. **Generate TLS certificates:**
   ```bash
   cd reverse-proxy/scripts
   chmod +x gen-certs.sh
   ./gen-certs.sh --ca --domain mac.local --out ../traefik/certs
   # Copy to Caddy certs if using Caddy too:
   cp ../traefik/certs/local.{crt,key} ../caddy/certs/
   cp ../traefik/certs/ca.crt ../caddy/certs/
   ```

3. **Trust the CA certificate:**
   ```bash
   sudo security add-trusted-cert -d -r trustRoot \
     -k /Library/Keychains/System.keychain reverse-proxy/traefik/certs/ca.crt
   ```
   Restart your browser after adding the certificate.

4. **Set a real basic-auth password** (see [Authentication setup](#authentication-setup)).

5. **Start the proxy** (see [Running the Traefik stack](#running-the-traefik-stack) or Caddy equivalent).

6. Open `https://mac.local/ollama/api/tags` in your browser.

---

## Linux setup

### host.docker.internal resolution

On Linux, Docker does **not** automatically set `host.docker.internal`. Add the
following to the proxy's `docker-compose.yml` before starting:

```yaml
services:
  traefik:   # or caddy
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

Alternatively, replace `host.docker.internal` with your host's LAN IP in all
upstream URLs (`dynamic/routes.yml` or `Caddyfile`).

### mDNS / Avahi hostname

To use a `.local` hostname on Linux, install Avahi:
```bash
# Debian/Ubuntu
sudo apt-get install avahi-daemon avahi-utils

# RHEL/Fedora
sudo dnf install avahi avahi-tools
sudo systemctl enable --now avahi-daemon
```

The machine will be reachable at `$(hostname).local`.

### Trusting the CA certificate

```bash
# Debian/Ubuntu
sudo cp reverse-proxy/traefik/certs/ca.crt /usr/local/share/ca-certificates/local-dev-ca.crt
sudo update-ca-certificates

# RHEL/Fedora
sudo cp reverse-proxy/traefik/certs/ca.crt /etc/pki/ca-trust/source/anchors/local-dev-ca.crt
sudo update-ca-trust

# Firefox (all Linux): Settings > Privacy > View Certificates > Authorities > Import
```

---

## Running the Traefik stack

```bash
cd reverse-proxy/traefik

# 1. Generate certs (first time only)
../scripts/gen-certs.sh --ca --domain mac.local --out certs/

# 2. Edit dynamic/middlewares.yml and replace the placeholder bcrypt hash
#    (see Authentication setup below)

# 3. Start
docker compose up -d

# 4. View logs
docker compose logs -f

# 5. Access the Traefik dashboard (local only)
open http://localhost:8080
```

The Traefik dashboard is bound to `127.0.0.1:8080` only. Do not expose it to
the network.

To stop: `docker compose down`

---

## Running the Caddy stack

```bash
cd reverse-proxy/caddy

# 1. Copy certs (generate with ../scripts/gen-certs.sh if not done yet)
mkdir -p certs
cp ../traefik/certs/local.{crt,key} certs/

# 2. Edit Caddyfile:
#    a. Change 'mac.local' to your machine's hostname or IP
#    b. Replace the placeholder bcrypt hashes with real ones

# 3. Start
docker compose up -d

# 4. View logs
docker compose logs -f
```

To stop: `docker compose down`

---

## Route table

| Path prefix | Upstream | Auth required | WebSocket |
|------------|----------|--------------|-----------|
| `/ollama` | `host.docker.internal:11434` | No | No |
| `/llama-swap` | `host.docker.internal:8080` | No | No |
| `/openwebui` | `host.docker.internal:3000` | No | No |
| `/n8n` | `host.docker.internal:5678` | **Yes** | No |
| `/terminal` | `host.docker.internal:7681` | **Yes** | **Yes** |
| `/code` | `host.docker.internal:8443` | **Yes** | **Yes** |
| `/searxng` | `host.docker.internal:8888` | No | No |
| `/llamacpp-rpc` | `host.docker.internal:8081` | No | No |
| `/vnc` | `host.docker.internal:6080` | **Yes** | **Yes** |

---

## Authentication setup

### Basic auth (username + password)

#### Step 1 – Generate a bcrypt hash

```bash
cd reverse-proxy/scripts
chmod +x gen-bcrypt.sh
./gen-bcrypt.sh mypassword
```

The script prints two variants:
- **Caddy** format (single `$` signs) – paste into `Caddyfile`
- **Traefik** format (doubled `$$` signs for YAML) – paste into `dynamic/middlewares.yml`

#### Step 2 – Update Traefik config

In `reverse-proxy/traefik/dynamic/middlewares.yml`, replace:
```yaml
- "admin:$$2y$$10$$placeholder.replace.me.with.real.bcrypt.hash.XYZ"
```
with the Traefik-format hash from the script output.

#### Step 3 – Update Caddy config

In `reverse-proxy/caddy/Caddyfile`, replace each:
```
admin $2y$10$placeholder.replace.me.with.real.bcrypt.hash.ABC
```
with the Caddy-format hash from the script output.

Traefik hot-reloads its dynamic config automatically. For Caddy, run:
```bash
docker compose exec caddy caddy reload --config /etc/caddy/Caddyfile
```

---

### API token / header-based auth

Traefik and Caddy do not natively match arbitrary header values, so API token
enforcement requires a **forward-auth** sidecar.

#### Optional: forward-auth pattern

Deploy a small HTTP auth service that:
1. Receives a request with all headers forwarded from the proxy.
2. Validates the `Authorization: Bearer <token>` or `X-Api-Key: <token>` header.
3. Returns `200 OK` if valid, `401 Unauthorized` otherwise.

**Traefik forward-auth example** (add to `middlewares.yml`):
```yaml
http:
  middlewares:
    api-token-auth:
      forwardAuth:
        address: "http://host.docker.internal:9000/auth"
        authRequestHeaders:
          - "Authorization"
          - "X-Api-Key"
```

Then add `api-token-auth` to the middleware list in `routes.yml` for any route
that should require a token.

**Lightweight sidecar options:**
- [traefik-forward-auth](https://github.com/thomseddon/traefik-forward-auth) – OAuth2/OIDC
- A simple Python/Go HTTP server that checks a static token
- Any HTTP service that returns 200/401 based on request headers

---

## TLS setup

Two approaches are supported. Both require placing `local.crt` and `local.key`
in the `certs/` directory of the chosen proxy.

### Option A: Self-signed certificate (quickest)

```bash
./gen-certs.sh --self-signed --domain mac.local --out ../traefik/certs
```

**Downside:** Every browser will show a security warning. You must manually
accept the exception, and API clients (curl, Python) need `--insecure` / `verify=False`.

### Option B: Local CA-signed certificate (recommended)

```bash
./gen-certs.sh --ca --domain mac.local --out ../traefik/certs
```

This creates:
- `ca.key` / `ca.crt` – your local CA (trust this once)
- `local.key` / `local.crt` – server certificate signed by your CA

After trusting `ca.crt` in your OS/browser (see platform-specific steps above),
all browsers and curl will accept the certificate without warnings.

#### Re-using the same CA for multiple services

Generate additional server certs with the same CA:
```bash
./gen-certs.sh --ca --domain 192.168.1.50 --out /tmp/extra-certs
# Manually sign with existing CA key instead of creating a new one:
openssl x509 -req -in /tmp/extra-certs/local.csr \
  -CA ../traefik/certs/ca.crt -CAkey ../traefik/certs/ca.key \
  -CAcreateserial -out /tmp/extra-certs/local.crt -days 825 -sha256
```

#### Certificate renewal

Certificates expire after 825 days (~2.25 years). Re-run `gen-certs.sh` and
reload/restart the proxy to apply the new certificate.

---

## Troubleshooting

### Browser shows "connection refused" on port 443

The proxy container is not running. Check:
```bash
docker compose ps
docker compose logs
```

### Browser shows "502 Bad Gateway" / "upstream not reachable"

The upstream service is not running or not listening on the expected port.
Verify:
```bash
curl http://localhost:11434/api/tags    # test Ollama directly
curl http://localhost:3000              # test Open WebUI directly
```
Also check that `host.docker.internal` resolves (Linux: add `extra_hosts`).

### Path-prefix stripping issues (upstream gets wrong path)

**Symptom:** The upstream receives `/ollama/api/tags` instead of `/api/tags`.

For Traefik: ensure the `strip-<name>` middleware is listed in the router's
`middlewares:` array in `routes.yml`.

For Caddy: `handle_path /prefix/*` automatically strips the prefix. If you use
`handle /prefix/*` instead, you must add `uri strip_prefix /prefix` manually.

### WebSocket connections fail

**Symptom:** Real-time features break in terminal, code-server, or noVNC.

Traefik handles WebSocket upgrades automatically. No additional configuration
is needed.

For Caddy, the `header_up Connection` and `header_up Upgrade` lines in the
`Caddyfile` enable WebSocket proxying. Verify they are present for terminal,
code, and vnc routes.

Check that your reverse-proxy is not stripping the `Upgrade: websocket` header.

### Self-signed certificate errors in curl

```bash
curl --insecure https://mac.local/ollama/api/tags
# or trust the CA at the OS level (see TLS setup)
```

### Traefik dashboard not accessible

The dashboard is bound to `127.0.0.1:8080` inside the container. Access it
from the Docker host machine only:
```
http://localhost:8080
```

### "Invalid plaintext password" / auth always fails

Ensure the bcrypt hash in `middlewares.yml` uses **doubled dollar signs** (`$$`)
as required by YAML. For Caddy's `Caddyfile`, use **single dollar signs** (`$`).

Use `gen-bcrypt.sh` to generate correctly-formatted hashes for both tools.

---

## Security notes

### General

- The proxy is intended for **LAN use only**. Do not expose port 443 to the
  internet without additional hardening (rate-limiting, IP allowlisting, mTLS).
- Secrets (bcrypt hashes, CA private keys) must never be committed to
  version control.
- Rotate the basic-auth password regularly.

### n8n

n8n can execute arbitrary code via its Function nodes and has access to
environment variables. Protect it with basic auth **and** consider running it
in a network-isolated container.

### code-server

code-server provides full shell access to the host user's environment.
**Always** protect it with basic auth. Consider binding it to `127.0.0.1` and
only exposing it through the reverse-proxy.

### VNC

VNC sessions give graphical access to the desktop. Treat it as equivalent to
SSH access. Protect with basic auth and, if possible, restrict by source IP
using firewall rules.

### Terminal (ttyd)

ttyd provides a full interactive shell. Apply the same precautions as for
code-server.

### TLS private key protection

The CA private key (`ca.key`) should be stored securely (e.g., encrypted at
rest or in a secrets manager) and never uploaded to cloud storage or version
control.

---

*Generated as part of the `reverse-proxy/` tooling in this repository.*
