#!/usr/bin/env sh
# sgpt-install.sh - install sgpt on linux, for current user only

set -eu
[ "${DEBUG:-0}" = "1" ] && set -x

SGPT_VERSION="${SGPT_VERSION:-2.19.0}"

########################################################################################

_main () {
    mkdir -p ~/.local/bin ~/.local/share/man/man1 ~/.bash_completion.d ~/.fish_completion.d ~/.zsh_completion.d

    arch="$(uname -m)"

    tmpf="$(mktemp)"
    curl -o "$tmpf" -fSL "https://github.com/tbckr/sgpt/releases/download/v${SGPT_VERSION}/sgpt-${SGPT_VERSION}-1-${arch}.pkg.tar.zst"

    tmpd="$(mktemp -d)"
    tar -C "$tmpd" -xvf "$tmpf"

    mv -v "$tmpd/usr/bin/sgpt" ~/.local/bin/
    mv -v "$tmpd/usr/share/man/man1/sgpt.1.gz" ~/.local/share/man/man1/
    mv -v "$tmpd/usr/share/bash-completion/completions/sgpt" ~/.bash_completion.d/
    mv -v "$tmpd/usr/share/fish/vendor_completions.d/sgpt.fish" ~/.fish_completion.d/
    mv -v "$tmpd/usr/share/zsh/vendor-completions/_sgpt" ~/.zsh_completion.d/

    chmod 755 ~/.local/bin/sgpt

    rm -rf "$tmpf" "$tmpd"

    echo ""

    echo "SGPT installed!"
    echo ""
    echo "Now run:"
    echo "    sgpt config init"
    echo ""
    echo "Then edit ~/.config/sgpt/config.yaml with the model you want to use."
    echo ""
    echo "Finally, to use a local Ollama, add these to your shell startup script:"
    echo "    export OPENAI_BASE_URL=http://localhost:11434/v1"
    echo "    export OPENAI_API_KEY=\"(none)\""
}

if ! command -v sgpt ; then
    _main
fi
