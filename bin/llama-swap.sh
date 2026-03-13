#!/usr/bin/env sh
# llama-swap.sh - wrapper around running llama-swap
set -eu
[ "${DEBUG:-0}" = "1" ] && set -x

LLAMA_SWAP_ADDR="127.0.0.1"
LLAMA_SWAP_PORT="11400"


mkdir -p $HOME/.config/llama-swap

llama-swap \
    -config "$HOME/.config/llama-swap/config.yaml" \
    -watch-config \
    -listen "$LLAMA_SWAP_ADDR:$LLAMA_SWAP_PORT"
