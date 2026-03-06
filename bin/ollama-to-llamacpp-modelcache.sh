#!/usr/bin/env bash
# ollama-to-llamacpp.sh
#
# Creates a directory of .gguf symlinks pointing at Ollama's content-addressed
# blob files, so that llama-server's --models-dir (or its default cache
# directory) can find them without duplicating disk space.
#
# Usage:
#   ollama-to-llamacpp.sh [OPTIONS]
#
# Options:
#   -o, --ollama-dir DIR    Ollama models root dir (default: ~/.ollama/models,
#                           or $OLLAMA_MODELS if set)
#   -d, --dest-dir DIR      Destination directory for .gguf symlinks
#                           (default: platform llama.cpp cache dir, or
#                            $LLAMA_CACHE if set)
#   -f, --force             Overwrite existing symlinks
#   -n, --dry-run           Print actions without making changes
#   -v, --verbose           Show extra detail
#   -h, --help              Show this help
#
# The script reads every Ollama manifest it finds, locates the blob with
# mediaType "application/vnd.ollama.image.model", and creates a symlink:
#
#   <dest-dir>/<namespace>_<model>_<tag>.gguf -> <ollama-blobs>/<sha256-digest>
#
# The resulting names work perfectly with llama-server's router:
# the model id will be "<namespace>_<model>_<tag>".
#
# llama.cpp cache directory logic (mirrors common/common.cpp):
#   Linux/BSDs : ${XDG_CACHE_HOME:-~/.cache}/llama.cpp/
#   macOS      : ~/Library/Caches/llama.cpp/
#   Override   : $LLAMA_CACHE
#
# References:
#   https://github.com/ggml-org/llama.cpp/blob/a96a1120b/common/common.cpp#L839-L870
#   https://github.com/ggml-org/llama.cpp/blob/a96a1120b/common/preset.cpp#L383-L450
#   https://github.com/ggml-org/llama.cpp/blob/a96a1120b/tools/server/README.md#L1441-L1490

set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "INFO:  $*"; }
dbg()  { [[ "${VERBOSE:-0}" == "1" ]] && echo "DEBUG: $*" || true; }

usage() {
    sed -n '/^# Usage:/,/^[^#]/{ /^[^#]/d; s/^# \{0,3\}//; p }' "$0"
    exit 0
}

# ---------------------------------------------------------------------------
# Platform-aware default llama.cpp cache directory
# (mirrors fs_get_cache_directory() in common/common.cpp)
# ---------------------------------------------------------------------------
default_llama_cache_dir() {
    if [[ -n "${LLAMA_CACHE:-}" ]]; then
        echo "${LLAMA_CACHE%/}/"
        return
    fi

    local os
    os="$(uname -s)"
    case "$os" in
        Linux|*BSD|AIX)
            local base="${XDG_CACHE_HOME:-${HOME}/.cache}"
            echo "${base%/}/llama.cpp/"
            ;;
        Darwin)
            echo "${HOME}/Library/Caches/llama.cpp/"
            ;;
        *)
            # Reasonable fallback
            echo "${HOME}/.cache/llama.cpp/"
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
OLLAMA_DIR="${OLLAMA_MODELS:-${HOME}/.ollama/models}"
DEST_DIR=""
FORCE=0
DRY_RUN=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--ollama-dir)   OLLAMA_DIR="$2";  shift 2 ;;
        -d|--dest-dir)     DEST_DIR="$2";    shift 2 ;;
        -f|--force)        FORCE=1;          shift   ;;
        -n|--dry-run)      DRY_RUN=1;        shift   ;;
        -v|--verbose)      VERBOSE=1;        shift   ;;
        -h|--help)         usage ;;
        *) die "Unknown option: $1" ;;
    esac
done

# Default destination: llama.cpp cache directory
if [[ -z "$DEST_DIR" ]]; then
    DEST_DIR="$(default_llama_cache_dir)"
fi

BLOBS_DIR="${OLLAMA_DIR}/blobs"
MANIFESTS_DIR="${OLLAMA_DIR}/manifests"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
[[ -d "$OLLAMA_DIR" ]]    || die "Ollama models directory not found: $OLLAMA_DIR"
[[ -d "$BLOBS_DIR" ]]     || die "Ollama blobs directory not found: $BLOBS_DIR"
[[ -d "$MANIFESTS_DIR" ]] || die "Ollama manifests directory not found: $MANIFESTS_DIR"

command -v python3 &>/dev/null || \
    command -v python  &>/dev/null || \
    die "python3 (or python) is required to parse JSON manifests"

PYTHON=$(command -v python3 2>/dev/null || command -v python)

# ---------------------------------------------------------------------------
# Create destination directory
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$DEST_DIR"
fi

info "Ollama models dir : $OLLAMA_DIR"
info "Destination dir   : $DEST_DIR"
[[ "$DRY_RUN" -eq 1 ]] && info "(dry-run mode – no changes will be made)"
echo ""

# ---------------------------------------------------------------------------
# Walk manifests
# ---------------------------------------------------------------------------
created=0
skipped=0
errors=0


# First remove any symbolic links to missing files
for file in "$DEST_DIR"/* ; do
    [ "${DEST_DIR##*/}" = '*' ] && continue
    if [ ! -e "$file" ] ; then
        info "Found symlink '$file' whose destination does not exist; removing symlink"
        rm -f "$file"
    fi
done

# Manifests live at: manifests/<registry>/<namespace>/<model>/<tag>
# e.g.:  manifests/registry.ollama.ai/library/llama3/latest
#         manifests/registry.ollama.ai/username/mymodel/q4_k_m

while IFS= read -r -d '' manifest_file; do
    # Extract path components relative to MANIFESTS_DIR
    rel="${manifest_file#"${MANIFESTS_DIR}/"}"     # e.g. registry.ollama.ai/library/llama3/latest
    IFS='/' read -ra parts <<< "$rel"

    if [[ ${#parts[@]} -lt 4 ]]; then
        dbg "Skipping unexpected manifest path: $rel"
        continue
    fi

    # registry="${parts[0]}"  # (not used in the symlink name)
    namespace="${parts[1]}"
    model="${parts[2]}"
    tag="${parts[3]}"

    dbg "Processing manifest: $rel"

    # Find the GGUF blob: layer with mediaType ending in ".model"
    # Ollama uses: "application/vnd.ollama.image.model"
    blob_digest="$($PYTHON - "$manifest_file" <<'EOF'
import sys, json
try:
    with open(sys.argv[1]) as f:
        manifest = json.load(f)
    for layer in manifest.get("layers", []):
        mt = layer.get("mediaType", "")
        if mt == "application/vnd.ollama.image.model":
            # digest format: "sha256:abc123..."  -> file is "sha256-abc123..."
            print(layer["digest"].replace(":", "-"))
            sys.exit(0)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(2)
EOF
    )" || {
        echo "WARN:  No model layer found in $rel – skipping" >&2
        (( errors++ )) || true
        continue
    }

    blob_path="${BLOBS_DIR}/${blob_digest}"

    if [[ ! -f "$blob_path" ]]; then
        echo "WARN:  Blob not found for $rel: $blob_path – skipping" >&2
        (( errors++ )) || true
        continue
    fi

    # Build a friendly symlink name for llama.cpp:
    #   <namespace>_<model>_<tag>.gguf
    # This becomes the model "id" in llama-server's router.
    # Sanitise characters that might confuse shells or APIs.
    safe_namespace="${namespace//[^a-zA-Z0-9._-]/_}"
    safe_model="${model//[^a-zA-Z0-9._-]/_}"
    safe_tag="${tag//[^a-zA-Z0-9._-]/_}"

    link_name="${safe_namespace}_${safe_model}_${safe_tag}.gguf"
    link_path="${DEST_DIR}/${link_name}"

    # Use absolute path for the symlink target so it works from anywhere
    abs_blob="$(cd "$(dirname "$blob_path")" && pwd)/$(basename "$blob_path")"

    if [[ -L "$link_path" ]]; then
        if [[ "$FORCE" -eq 1 ]]; then
            info "Replacing existing symlink: $link_name"
            if [[ "$DRY_RUN" -eq 0 ]]; then
                ln -sf "$abs_blob" "$link_path"
            fi
            (( created++ )) || true
        else
            dbg "Symlink already exists (use -f to replace): $link_path"
            (( skipped++ )) || true
        fi
    elif [[ -e "$link_path" ]]; then
        echo "WARN:  $link_path exists and is not a symlink – skipping (use -f to overwrite)" >&2
        (( skipped++ )) || true
    else
        info "Creating symlink: $link_name"
        dbg "  -> $abs_blob"
        if [[ "$DRY_RUN" -eq 0 ]]; then
            ln -s "$abs_blob" "$link_path"
        fi
        (( created++ )) || true
    fi

done < <(find "$MANIFESTS_DIR" -type f -not -name '.*' -print0 | sort -z)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
info "Done."
info "  Symlinks created : $created"
info "  Skipped          : $skipped"
info "  Warnings/errors  : $errors"
echo ""

if [[ "$DRY_RUN" -eq 0 && "$created" -gt 0 ]]; then
    echo "You can now start llama-server in router mode with:"
    echo ""
    echo "  llama-server --models-dir \"${DEST_DIR}\""
    echo ""
    echo "Or, since '${DEST_DIR}' is the default llama.cpp cache directory"
    echo "on this platform, you can simply run:"
    echo ""
    echo "  llama-server"
    echo ""
    echo "Model IDs will be of the form: library_llama3_latest"
fi
