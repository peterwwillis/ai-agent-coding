# caddy

## About

Run a Caddy docker container, with reverse-proxy rules to route URLs to specific backend services.

Binds to port 1443 on host because Colima uses a userspace program to proxy network connections,
and it can't bind to ports lower than 1024. You must run the **simpleproxy** service forwarding
TCP connections from port 443 to port 1443 in order to use this host with a regular HTTPS URL.


---

## How to use subdomains locally

1. Automatic (no configuration needed):
 - If you use localhost as your PROXY_DOMAIN (the default in .env.example), modern browsers like Chrome and Firefox automatically resolve all subdomains of localhost (e.g.,
  ollama.localhost) to 127.0.0.1 without needing any /etc/hosts entries.

2. Using /etc/hosts (manual):
 - If you use a custom domain (e.g., thinkpaddy.local) or if your browser doesn't automatically resolve *.localhost, add the following to your /etc/hosts:
   ```
   127.0.0.1  ollama.thinkpaddy.local
   127.0.0.1  openwebui.thinkpaddy.local
   127.0.0.1  llama-swap.thinkpaddy.local
   127.0.0.1  terminal.thinkpaddy.local
   127.0.0.1  code.thinkpaddy.local
   127.0.0.1  searxng.thinkpaddy.local
   127.0.0.1  llamacpp-rpc.thinkpaddy.local
   127.0.0.1  vnc.thinkpaddy.local
   127.0.0.1  localai.thinkpaddy.local
   ```

3. Using dnsmasq (wildcard):
 - If you want a catch-all solution for any subdomain, add this to your dnsmasq configuration:
   ```
   address=/.localhost/127.0.0.1
   ```

4. `foo.local`
 - Since .local domains are handled by mDNS (Avahi/Bonjour), they don't natively support wildcards like `*.thinkpaddy.local`. To make this work locally on Linux:
   - The "Clean" Way (dnsmasq): If you want a true wildcard (`*.thinkpaddy.local`) so you never have to edit hosts again, use dnsmasq.

     1. Install dnsmasq: `sudo apt update && sudo apt install dnsmasq`
     2. Configure the wildcard:
        Create a file at /etc/dnsmasq.d/thinkpaddy.conf:
        ```
        # Point the base domain and all subdomains to 127.0.0.1
        address=/thinkpaddy.local/127.0.0.1
        ```
     3. Restart dnsmasq: `sudo systemctl restart dnsmasq`
     4. Update your DNS settings:
        - Ensure your system looks at 127.0.0.1 for DNS. If you are using NetworkManager, it often handles this automatically, or you can add nameserver 127.0.0.1 to the top of /etc/resolv.conf.

 - Pro Tip: Use .localhost instead
   If you change your PROXY_DOMAIN to localhost in your .env file, you can avoid all of this. Modern browsers (Chrome/Firefox) and many OS-level resolvers now automatically resolve any subdomain ending in .localhost to 127.0.0.1 by default. Example: ollama.localhost just works out of the box.


Summary:
   - Caddyfile: Completely restructured to use site blocks like ollama.{$PROXY_DOMAIN}. This avoids path-stripping issues and is much cleaner.
   - Landing Page: Updated the central dashboard to link to the new subdomains.
   - .env.example: Removed the path-based URL variables and added DNS configuration instructions.

Note on TLS: The current configuration uses your existing certificate files. If these were generated only for localhost, your browser may show a security warning for subdomains. You may
  need to regenerate your local certificate to include a wildcard (e.g., *.localhost) or add the subdomains as Subject Alternative Names (SANs).
