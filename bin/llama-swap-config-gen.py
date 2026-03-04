#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import shlex
import struct
import sys
from pathlib import Path
from typing import List, Optional


def default_llama_cache_dir() -> Path:
    override = os.environ.get("LLAMA_CACHE")
    if override:
        return Path(override).expanduser()

    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Caches" / "llama.cpp"
    if system in {"Linux", "FreeBSD", "OpenBSD", "NetBSD", "AIX"}:
        base = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        return Path(base) / "llama.cpp"
    return Path.home() / ".cache" / "llama.cpp"


def default_llama_swap_config_path() -> Path:
    return Path.home() / ".config" / "llama-swap" / "config.yaml"


def read_gguf_chat_template(path: Path) -> Optional[str]:
    gguf_types = {
        0: 1,  # UINT8
        1: 1,  # INT8
        2: 2,  # UINT16
        3: 2,  # INT16
        4: 4,  # UINT32
        5: 4,  # INT32
        6: 4,  # FLOAT32
        7: 1,  # BOOL
        10: 8,  # UINT64
        11: 8,  # INT64
        12: 8,  # FLOAT64
    }

    def read_u32(f) -> int:
        data = f.read(4)
        if len(data) != 4:
            raise EOFError
        return struct.unpack("<I", data)[0]

    def read_u64(f) -> int:
        data = f.read(8)
        if len(data) != 8:
            raise EOFError
        return struct.unpack("<Q", data)[0]

    def read_str(f) -> str:
        length = read_u64(f)
        data = f.read(length)
        if len(data) != length:
            raise EOFError
        return data.decode("utf-8", errors="replace")

    def skip_value(f, vtype: int):
        if vtype in gguf_types:
            f.seek(gguf_types[vtype], os.SEEK_CUR)
            return
        if vtype == 8:  # STRING
            length = read_u64(f)
            f.seek(length, os.SEEK_CUR)
            return
        if vtype == 9:  # ARRAY
            elem_type = read_u32(f)
            count = read_u64(f)
            for _ in range(count):
                skip_value(f, elem_type)
            return
        raise ValueError(f"Unsupported GGUF value type: {vtype}")

    try:
        if not path.is_file():
            return None
        with path.open("rb") as f:
            if f.read(4) != b"GGUF":
                return None
            _version = read_u32(f)
            _tensor_count = read_u64(f)
            kv_count = read_u64(f)
            for _ in range(kv_count):
                key = read_str(f)
                vtype = read_u32(f)
                if key == "tokenizer.chat_template":
                    if vtype == 8:
                        return read_str(f)
                    skip_value(f, vtype)
                    return None
                skip_value(f, vtype)
    except Exception:
        return None

    return None


def template_supports_thinking(path: Path) -> bool:
    template = read_gguf_chat_template(path)
    if not template:
        return False
    return "enable_thinking" in template


def yaml_key(name: str) -> str:
    safe = all(ch.isalnum() or ch in "._-" for ch in name)
    return name if safe else "'" + name.replace("'", "''") + "'"


def normalize_yaml_key_text(key: str) -> str:
    key = key.strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in ("'", '"'):
        inner = key[1:-1]
        if key[0] == "'":
            return inner.replace("''", "'")
        return inner.replace('\\"', '"')
    return key


def normalize_cache_type(value: str) -> str:
    if value == "q8":
        return "q8_0"
    return value


def parse_n_gpu_layers(value: str) -> str:
    v = value.strip().lower()
    if v == "auto":
        return v
    if v.isdigit():
        return str(int(v))
    raise argparse.ArgumentTypeError("n-gpu-layers must be an integer or 'auto'")


def find_models_block(lines: List[str]) -> tuple[Optional[int], int]:
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("models:"):
            start = i
            break
    if start is None:
        return None, len(lines)
    end = len(lines)
    for j in range(start + 1, len(lines)):
        stripped = lines[j].lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        if not lines[j].startswith(" "):
            end = j
            break
    return start, end


def existing_model_keys(lines: List[str], start: int, end: int) -> set[str]:
    keys: set[str] = set()
    for line in lines[start + 1 : end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("  ") and not line.startswith("    "):
            stripped = line.strip()
            if stripped.endswith(":") and not stripped.startswith("-"):
                key = stripped[:-1].strip()
                keys.add(normalize_yaml_key_text(key))
    return keys


def parse_models_entries(lines: List[str], start: int, end: int) -> List[tuple[str, int, int]]:
    entries: List[tuple[str, int, int]] = []
    i = start + 1
    while i < end:
        line = lines[i]
        if (
            line.startswith("  ")
            and not line.startswith("    ")
            and line.strip()
            and not line.lstrip().startswith("#")
        ):
            stripped = line.strip()
            if stripped.endswith(":") and not stripped.startswith("-"):
                key = normalize_yaml_key_text(stripped[:-1].strip())
                entry_start = i
                i += 1
                while i < end:
                    next_line = lines[i]
                    if (
                        next_line.startswith("  ")
                        and not next_line.startswith("    ")
                        and next_line.strip()
                        and not next_line.lstrip().startswith("#")
                    ):
                        if next_line.strip().endswith(":") and not next_line.strip().startswith("-"):
                            break
                    i += 1
                entry_end = i
                entries.append((key, entry_start, entry_end))
                continue
        i += 1
    return entries


def read_header(template_path: Optional[Path]) -> List[str]:
    if not template_path or not template_path.is_file():
        return ["models:"]

    lines = template_path.read_text(encoding="utf-8").splitlines()
    header: List[str] = []
    for line in lines:
        header.append(line)
        if line.strip().startswith("models:"):
            return header
    header.append("models:")
    return header


def build_cmd(
    llama_server: str,
    model_path: Path,
    thinking: Optional[bool],
    ctx_size: int,
    flash_attn: bool,
    cache_type_k: str,
    cache_type_v: str,
    n_gpu_layers: str,
    mmap: bool,
    batch_size: int,
) -> str:
    cache_type_k = normalize_cache_type(cache_type_k)
    cache_type_v = normalize_cache_type(cache_type_v)
    cmd_parts = [
        shlex.quote(llama_server),
        "--offline",
        "-m",
        shlex.quote(str(model_path)),
        "--ctx-size",
        str(ctx_size),
        "--batch-size",
        str(batch_size),
        "--cache-type-k",
        shlex.quote(cache_type_k),
        "--cache-type-v",
        shlex.quote(cache_type_v),
        "--n-gpu-layers",
        str(n_gpu_layers),
    ]
    if flash_attn:
        cmd_parts.append("--flash-attn")
    cmd_parts.append("--mmap" if mmap else "--no-mmap")
    if thinking is not None:
        val = "true" if thinking else "false"
        cmd_parts.extend(["--chat-template-kwargs", f"'{{\"enable_thinking\":{val}}}'"])
    cmd_parts.extend(["--port", "${PORT}"])
    return " ".join(cmd_parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a llama-swap config from llama.cpp cached GGUF models."
    )
    parser.add_argument(
        "--llama-server",
        default=os.environ.get("LLAMA_SERVER", "llama-server"),
        help="Path to llama-server binary (default: llama-server or $LLAMA_SERVER).",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("LLAMA_MODELS_DIR", str(default_llama_cache_dir())),
        help="Directory containing GGUF models (default: llama.cpp cache or $LLAMA_MODELS_DIR).",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("LLAMA_SWAP_CONFIG", str(default_llama_swap_config_path())),
        help="Write output YAML to this path (use '-' for stdout).",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Optional header template file (default: llama-swap.yaml.example next to this script).",
    )
    parser.add_argument(
        "--ctx-size",
        type=int,
        default=2048,
        help="Context size (default: 2048).",
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        help="Enable flash attention (default: disabled).",
    )
    parser.add_argument(
        "--cache-type-k",
        default="q8",
        help="KV cache K quantization type (default: q8).",
    )
    parser.add_argument(
        "--cache-type-v",
        default="q8",
        help="KV cache V quantization type (default: q8).",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=parse_n_gpu_layers,
        default="auto",
        help="Number of layers to offload to GPU (default: auto).",
    )
    parser.add_argument(
        "--mmap",
        dest="mmap",
        action="store_true",
        default=True,
        help="Enable memory-mapped model loading (default).",
    )
    parser.add_argument(
        "--no-mmap",
        dest="mmap",
        action="store_false",
        help="Disable memory-mapped model loading.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (default: 256).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress to stderr.",
    )
    parser.add_argument(
        "--prune-missing",
        action="store_true",
        help="Remove model entries not present in the llama.cpp cache.",
    )

    args = parser.parse_args()
    log = (lambda msg: print(msg, file=sys.stderr)) if args.verbose else (lambda _msg: None)

    models_dir = Path(args.models_dir).expanduser()
    if not models_dir.is_dir():
        print(f"ERROR: models directory not found: {models_dir}", file=sys.stderr)
        return 2

    log(f"Scanning for .gguf models under: {models_dir}")
    ggufs = sorted(p for p in models_dir.rglob("*.gguf") if p.is_file())
    if not ggufs:
        print(f"ERROR: no .gguf files found under: {models_dir}", file=sys.stderr)
        return 3
    log(f"Found {len(ggufs)} .gguf model(s).")

    template_path = (
        Path(args.template).expanduser()
        if args.template
        else Path(__file__).with_name("llama-swap.yml")
    )
    header_lines = read_header(template_path)
    if template_path.is_file():
        log(f"Using template header from: {template_path}")
    else:
        log("Template header not found; using minimal header.")

    used_names = {}
    entry_blocks: List[tuple[str, List[str]]] = []
    for model_path in ggufs:
        base_name = model_path.stem
        count = used_names.get(base_name, 0) + 1
        used_names[base_name] = count
        name = base_name if count == 1 else f"{base_name}-{count}"

        thinking_optional = template_supports_thinking(model_path)
        log(f"Adding model '{name}' (thinking optional: {'yes' if thinking_optional else 'no'}).")
        if thinking_optional:
            entry_blocks.append(
                (
                    name,
                    [
                        f"  {yaml_key(name)}:",
                        f"    cmd: {build_cmd(args.llama_server, model_path, False, args.ctx_size, args.flash_attn, args.cache_type_k, args.cache_type_v, args.n_gpu_layers, args.mmap, args.batch_size)}",
                    ],
                )
            )
            entry_blocks.append(
                (
                    f"{name}-thinking",
                    [
                        f"  {yaml_key(name)}-thinking:",
                        f"    cmd: {build_cmd(args.llama_server, model_path, True, args.ctx_size, args.flash_attn, args.cache_type_k, args.cache_type_v, args.n_gpu_layers, args.mmap, args.batch_size)}",
                    ],
                )
            )
        else:
            entry_blocks.append(
                (
                    name,
                    [
                        f"  {yaml_key(name)}:",
                        f"    cmd: {build_cmd(args.llama_server, model_path, None, args.ctx_size, args.flash_attn, args.cache_type_k, args.cache_type_v, args.n_gpu_layers, args.mmap, args.batch_size)}",
                    ],
                )
            )

    entries: List[str] = []
    for _name, block in entry_blocks:
        entries.extend(block)
        entries.append("")

    output_lines = header_lines + [""] + entries
    text = "\n".join(output_lines).rstrip() + "\n"

    if args.output == "-":
        sys.stdout.write(text)
        log("Wrote config to stdout.")
        return 0

    out_path = Path(args.output).expanduser()
    if out_path.exists():
        existing_lines = out_path.read_text(encoding="utf-8").splitlines()
        start, end = find_models_block(existing_lines)
        if start is None:
            if existing_lines and existing_lines[-1].strip():
                existing_lines.append("")
            existing_lines.append("models:")
            start = len(existing_lines) - 1
            end = len(existing_lines)

        if start is not None and args.prune_missing:
            entries_meta = parse_models_entries(existing_lines, start, end)
            desired = {name for name, _block in entry_blocks}
            keep = [True] * len(existing_lines)
            removed = 0
            for key, entry_start, entry_end in entries_meta:
                if key in desired:
                    continue
                for idx in range(entry_start, entry_end):
                    keep[idx] = False
                removed += 1
            if removed:
                existing_lines = [line for idx, line in enumerate(existing_lines) if keep[idx]]
                start, end = find_models_block(existing_lines)
                log(f"Pruned {removed} missing model entr{'y' if removed == 1 else 'ies'}.")

        start, end = find_models_block(existing_lines)
        if start is None:
            if existing_lines and existing_lines[-1].strip():
                existing_lines.append("")
            existing_lines.append("models:")
            start = len(existing_lines) - 1
            end = len(existing_lines)

        existing = existing_model_keys(existing_lines, start, end)
        lines_to_add: List[str] = []
        added = 0
        for entry_name, block in entry_blocks:
            if entry_name in existing:
                continue
            if lines_to_add:
                lines_to_add.append("")
            lines_to_add.extend(block)
            added += 1

        if not lines_to_add:
            log(f"No new entries to add; leaving config unchanged: {out_path}")
            return 0

        if end > start + 1 and existing_lines[end - 1].strip():
            lines_to_add = [""] + lines_to_add
        existing_lines[end:end] = lines_to_add
        new_text = "\n".join(existing_lines).rstrip() + "\n"
        out_path.write_text(new_text, encoding="utf-8")
        log(f"Updated config at: {out_path} (added {added} entr{'y' if added == 1 else 'ies'}).")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        log(f"Wrote new config to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
