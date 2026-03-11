# Quickstart – Ubuntu 24.04

This guide walks you through setting up and running the local reverse-proxy on
a fresh Ubuntu 24.04 system using either **Traefik** or **Caddy**.

---

## Prerequisites

### 1. Install Docker Engine

```bash
# Remove any old Docker packages
sudo apt-get remove -y docker.io docker-doc docker-compose docker-compose-v2 \
    podman-docker containerd runc 2>/dev/null || true

# Add Docker's official GPG key and repository
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin

# Allow your user to run Docker without sudo
sudo usermod -aG docker "$USER"
newgrp docker   # apply group membership in the current shell
```

Verify:
```bash
docker run --rm hello-world
docker compose version
```

### 2. Install additional tools

```bash
# apache2-utils provides htpasswd for bcrypt hash generation
# avahi-daemon enables <hostname>.local mDNS resolution (optional)
sudo apt-get install -y \
    make \
    openssl \
    apache2-utils \
    avahi-daemon \
    avahi-utils
```

Enable mDNS resolution (lets you use `myhost.local` instead of a raw IP):
```bash
sudo systemctl enable --now avahi-daemon
```

Your machine is now reachable at `$(hostname).local` from other LAN devices.

### 3. host.docker.internal on Linux

Docker on Linux does not set `host.docker.internal` automatically. You must add
it to the proxy's docker-compose file. Open
`reverse-proxy/traefik/docker-compose.yml` (or `caddy/docker-compose.yml`) and
uncomment the `extra_hosts` block:

```yaml
services:
  traefik:   # or caddy
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

---

## Option A – Traefik quickstart

```bash
cd reverse-proxy/traefik
```

### Step 1 – Create your `.env` file

```bash
cp .env.example .env
```

Edit `.env`:
```bash
PROXY_AUTH_USER=admin
PROXY_AUTH_HASH=          # fill in after Step 2
PROXY_DOMAIN=myhost.local # or your LAN IP, e.g. 192.168.1.50
```

### Step 2 – Generate a bcrypt password hash

```bash
make gen-hash
```

The script prints two hash formats. Copy the **Caddy format** (single `$` signs)
and paste it as the value of `PROXY_AUTH_HASH` in `.env`.

### Step 3 – Generate TLS certificates

```bash
make gen-certs
```

This creates `certs/ca.crt`, `certs/local.crt`, and `certs/local.key`.

### Step 4 – Generate the Traefik basic-auth users file

```bash
make gen-users-file
```

This reads `PROXY_AUTH_USER` and `PROXY_AUTH_HASH` from `.env` and writes them
to `secrets/users` (a file that is never committed to git).

### Step 5 – Trust the local CA certificate

```bash
make trust-ca
```

This adds `certs/ca.crt` to the Ubuntu system trust store. After running this,
browsers and curl on this machine will accept your local TLS certificate without
warnings.

> **Firefox note:** Firefox uses its own trust store. Go to
> Settings → Privacy & Security → View Certificates → Authorities → Import
> and select `certs/ca.crt`.

### Step 6 – Start Traefik

```bash
make up
```

Check that it started:
```bash
make logs      # follow logs (Ctrl-C to exit)
make health    # check all routes
```

Visit `https://myhost.local/ollama/api/tags` in your browser (replace with
your actual domain/IP).

The Traefik dashboard is available at `http://localhost:8080` (localhost only).

### Stopping Traefik

```bash
make down
```

---

## Option B – Caddy quickstart

```bash
cd reverse-proxy/caddy
```

### Step 1 – Create your `.env` file

```bash
cp .env.example .env
```

Edit `.env`:
```bash
PROXY_AUTH_USER=admin
PROXY_AUTH_HASH=          # fill in after Step 2
PROXY_DOMAIN=myhost.local # or your LAN IP
```

### Step 2 – Generate a bcrypt password hash

```bash
make gen-hash
```

Copy the **Caddy format** hash (single `$` signs) and paste it as the value of
`PROXY_AUTH_HASH` in `.env`.

### Step 3 – Generate TLS certificates

```bash
make gen-certs
```

### Step 4 – Trust the local CA certificate

```bash
make trust-ca
```

### Step 5 – Start Caddy

```bash
make up
```

```bash
make logs
make health
```

### Reloading Caddy config without downtime

After editing the Caddyfile:
```bash
make reload
```

### Stopping Caddy

```bash
make down
```

---

## Loading credentials from an external secrets manager

Instead of storing the hash in `.env`, you can export it from a secrets vault
before running `make`:

```bash
# HashiCorp Vault
export PROXY_AUTH_HASH="$(vault kv get -field=hash secret/proxy-auth)"
export PROXY_AUTH_USER="$(vault kv get -field=user  secret/proxy-auth)"

# Traefik: regenerate the users file, then start
cd reverse-proxy/traefik
make gen-users-file
make up

# Caddy: the env var is passed directly via env_file is bypassed, so use:
cd reverse-proxy/caddy
PROXY_AUTH_USER="$PROXY_AUTH_USER" PROXY_AUTH_HASH="$PROXY_AUTH_HASH" \
    docker compose up -d
```

For other secret managers (1Password CLI `op`, AWS Secrets Manager, etc.),
replace the `vault kv get` command with the appropriate CLI call.

---

## Changing the password later

1. Run `make gen-hash` and update `PROXY_AUTH_HASH` in `.env`.
2. For Traefik: run `make gen-users-file` then `make restart`.
3. For Caddy: run `make restart` (Caddy re-reads env vars on startup).

---

## Firewall notes

Ubuntu 24.04 uses `ufw` by default. If you want to restrict access to the
proxy to LAN only:

```bash
# Allow HTTPS from your LAN subnet only (adjust to your network)
sudo ufw allow from 192.168.1.0/24 to any port 443

# Deny HTTPS from everywhere else
sudo ufw deny 443

sudo ufw reload
```

---

## Troubleshooting on Ubuntu

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `curl: (7) Failed to connect` | Proxy not running | `make up && make logs` |
| `502 Bad Gateway` | Upstream service not running | Start the service, check its port |
| `curl: (60) SSL certificate problem` | CA not trusted | `make trust-ca` |
| `host.docker.internal: no such host` | Linux Docker limitation | Uncomment `extra_hosts` in docker-compose.yml |
| `avahi-daemon` not found | mDNS not installed | `sudo apt-get install avahi-daemon` |
| `.local` hostname not resolving | NSS not configured | `sudo apt-get install libnss-mdns` |

If `.local` hostnames still don't resolve, verify `/etc/nsswitch.conf` contains
`mdns4_minimal` in the `hosts:` line:
```
hosts: files mdns4_minimal [NOTFOUND=return] dns
```
