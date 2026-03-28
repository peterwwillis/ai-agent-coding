#!/usr/bin/env sh
set -eu
[ "${DEBUG:-0}" = "1" ] && set -x

echo "UID: ${UID:-}"
echo "GID: ${GID:-}"
echo "USER: ${USER:-}"

if [ "$(id -u)" = "0" ] ; then 
    groupadd -g ${GID} ${USER} || true
    useradd -l -u ${UID} -g ${GID} -m ${USER} -d "/home/${USER}.linux" || true
    printf "%s\t%s\n" "${USER}" "ALL=(ALL:ALL) ALL" >> /etc/sudoers

    if [ -n "${ADD_GROUPS:-}" ] ; then
        for pair in $ADD_GROUPS ; do
            NEWGID="$(echo "$pair" | cut -d : -f 1)"
            NEWGNAME="$(echo "$pair" | cut -d : -f 2)"
            groupadd -g "$NEWGID" "$NEWGNAME" || true
            usermod -G "$NEWGNAME" "$USER" || true
        done
    fi

    # Handle permissions
    for file in \
        /data /models /configuration /backends
    do
        echo "Changing $file to owner ${UID}:${GID} ..."
        chown -R "${UID}:${GID}" "$file"
    done
    echo "Done changing file ownership"

    cd /
fi

exec sudo -H -E -u "${USER}" "/entrypoint.sh" "$@"
