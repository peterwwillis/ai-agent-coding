#!/usr/bin/env sh
set -eu

mkdir -p ~/Remote/macdaddy

sshfs \
    macdaddy: ~/Remote/macdaddy \
    -o reconnect \
    -o ServerAliveInterval=15 \
    -o idmap=user \
    -o uid=1001 \
    -o gid=1001 \
    -o default_permissions
#    -o allow_other \
