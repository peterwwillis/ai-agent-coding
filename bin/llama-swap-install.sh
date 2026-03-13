#!/usr/bin/env sh
# llama-swap-install.sh - install llama-swap on linux, MacOS

set -eu
[ "${DEBUG:-0}" = "1" ] && set -x

########################################################################################

if [ -z "${LLAMASWAP_VERSION:-}" ] ; then
    LLAMASWAP_VERSION="$(curl --silent "https://api.github.com/repos/mostlygeek/llama-swap/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')"
fi

mkdir -p ~/.local/bin

arch="$(uname -m)"
os="$( uname -s | tr '[:upper:]' '[:lower:]' )"

tmpf="$(mktemp)"
curl -o "$tmpf" -fSL "https://github.com/mostlygeek/llama-swap/releases/download/v${LLAMASWAP_VERSION}/llama-swap_${LLAMASWAP_VERSION}_${os}_${arch}.tar.gz"

tmpd="$(mktemp -d)"
tar -C "$tmpd" -xvf "$tmpf"

#if [ "$os" = "darwin" ] ; then
#    xattr -d com.apple.quarantine "$tmpd"/llama-swap
#fi

mv -v "$tmpd/llama-swap" ~/.local/bin/

chmod 755 ~/.local/bin/llama-swap

rm -rf "$tmpf" "$tmpd"
