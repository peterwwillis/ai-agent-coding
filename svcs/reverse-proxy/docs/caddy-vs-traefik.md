# Caddy vs Traefik – Pros and Cons

Both Caddy and Traefik are excellent reverse-proxy choices for local AI service
routing. The table below captures the key differences to help you decide which
fits your workflow.

---

## Quick summary

| Concern | Caddy | Traefik |
|---------|-------|---------|
| Configuration style | Single `Caddyfile` | Static CLI flags + dynamic YAML files |
| Learning curve | Low | Medium |
| Dynamic config reload | Built-in (`caddy reload`) | File provider watches `dynamic/` dir |
| Dashboard / UI | None built-in | Web dashboard at `localhost:8080` |
| Automatic HTTPS (ACME) | Yes, first-class | Yes, first-class |
| Local / offline TLS | Yes (manual cert) | Yes (manual cert) |
| WebSocket support | Transparent | Transparent |
| Docker integration | Manual (no auto-discovery) | Optional (Docker provider) |
| Basic auth | Native directive | Native middleware |
| Secrets in config | Environment variable placeholders | External `usersFile` or forward-auth |
| Path routing | `handle_path` (strips prefix) | `PathPrefix` rule + `stripPrefix` middleware |
| Middleware chaining | Via `route` blocks | Named middleware chains |
| Plugin / module system | Yes (Go plugins, xcaddy) | Yes (Go plugins) |
| Performance | Very high | Very high |
| Binary size / resource use | Lightweight | Lightweight |
| Community / ecosystem | Growing, strong docs | Large, mature ecosystem |
| Primary use case | Simple sites, developer-friendly | Microservices, Kubernetes ingress |

---

## Caddy – Pros

- **Simple, readable configuration.** The Caddyfile is terse and human-friendly.
  A full reverse-proxy for nine services fits in ~150 lines.

- **Built-in environment variable substitution.** `{$VAR}` placeholders in the
  Caddyfile let you inject secrets from env vars without hard-coding them.

- **Automatic HTTPS with zero config** (when a public domain is available). For
  local use, pointing it at your own cert is equally simple.

- **`handle_path` auto-strips the prefix** before forwarding to the upstream.
  No separate "strip prefix" middleware is needed.

- **`caddy reload` for zero-downtime config updates.** You never need to restart
  the container to pick up a Caddyfile change.

- **Self-contained binary.** One binary handles TLS, HTTP/2, HTTP/3, logging,
  and metrics.

- **Opinionated defaults** (secure headers, HTTP→HTTPS redirect, TLS 1.2+) are
  applied without extra configuration.

## Caddy – Cons

- **No built-in dashboard.** You must use logs or external tooling (Grafana,
  Prometheus) to observe traffic.

- **No auto-discovery of Docker containers.** You must manually define each
  backend in the Caddyfile; there is no Docker label-based routing.

- **Plugin system requires recompiling** (`xcaddy build`). The official Docker
  image does not include community plugins.

- **Less ecosystem support for Kubernetes** compared to Traefik. If you later
  want a single proxy across local Docker and a remote k8s cluster, Traefik has
  better stories for that.

- **Caddyfile can get verbose** for complex middleware chains (e.g., composing
  multiple auth layers for many routes). Named matchers help but are not as
  reusable as Traefik's named middleware chains.

- **Environment variable substitution is global,** not scoped per-directive.
  All `{$VAR}` tokens are replaced before parsing, which means you can't have
  different values for the same variable name in different parts of the file.

---

## Traefik – Pros

- **Separation of static and dynamic configuration.** Static CLI flags configure
  the proxy itself; the `dynamic/` directory (hot-reloaded) defines routes and
  middlewares. This makes it easy to change routing without restarting the proxy.

- **Named middleware chains** (`protected`, `secure-headers`) can be reused
  across many routes, keeping the routing config DRY.

- **First-class Docker / Kubernetes / Consul integration.** Traefik can
  auto-discover containers by reading Docker labels, which is invaluable in
  larger setups.

- **Built-in web dashboard** at `localhost:8080` showing active routers,
  services, and middleware in real time.

- **Rich middleware ecosystem** (rate limiting, circuit breakers, retry,
  compress, headers, etc.) all available without plugins.

- **Forward-auth middleware** is a first-class concept, making it easy to plug
  in an OAuth2 sidecar, OIDC, or any custom auth service.

- **`usersFile` for basic auth** keeps credentials in a separate file that can
  be generated from secrets management tooling without touching config files.

## Traefik – Cons

- **More moving parts.** You need to understand the distinction between static
  config (CLI flags / `traefik.yml`), dynamic config (file provider), routers,
  services, and middlewares before you can do anything non-trivial.

- **Path prefix stripping is a separate middleware.** Every prefixed route needs
  its own `stripPrefix` middleware, which adds boilerplate compared to Caddy's
  `handle_path`.

- **YAML dollar-sign escaping.** When writing basic-auth hashes in YAML files,
  every `$` must be doubled (`$$`). This is a sharp edge that causes subtle
  auth failures.

- **No native env-var substitution in dynamic YAML files.** To avoid hard-coded
  secrets you must use `usersFile` (external file) or a forward-auth sidecar.
  Caddy's `{$VAR}` approach is simpler for local use.

- **Dashboard should be locked down.** The dashboard is enabled with
  `--api.insecure=true` by default in this setup. For anything beyond a
  completely local laptop, you must restrict access.

- **Heavier default configuration.** Routing rules, middlewares, and services
  are defined separately and linked by name, which adds verbosity compared to
  Caddy's all-in-one `handle_path` block.

---

## Recommendation

| Scenario | Recommended choice |
|----------|--------------------|
| Simple local setup, single developer | **Caddy** |
| Large Docker Compose project with many containers | **Traefik** |
| Kubernetes / container orchestration | **Traefik** |
| Want zero-downtime config reload | **Caddy** (or Traefik with file provider) |
| Need rich real-time observability | **Traefik** (dashboard) |
| Want minimal config files | **Caddy** |
| Need forward-auth / OIDC | **Traefik** |
| Loading auth credentials from a secrets vault | **Both** – Traefik via `usersFile`, Caddy via env var |

For the AI services setup in this repository, **Caddy is the simpler choice**
for a single machine, while **Traefik is better** if you have many Docker
containers you want to route without editing config files each time.
