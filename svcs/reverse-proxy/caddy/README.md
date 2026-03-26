# caddy

Run a Caddy docker container, with reverse-proxy rules to route URLs to specific backend services.

Binds to port 1443 on host because Colima uses a userspace program to proxy network connections,
and it can't bind to ports lower than 1024. You must run the **simpleproxy** service forwarding
TCP connections from port 443 to port 1443 in order to use this host with a regular HTTPS URL.

