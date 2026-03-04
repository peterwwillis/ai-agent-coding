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


def build_cmd(llama_server: str, model_path: Path, thinking: Optional[bool]) -> str:
    cmd_parts = [
        shlex.quote(llama_server),
        "--offline",
        "-m",
        shlex.quote(str(model_path)),
    ]
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
        default=os.environ.get("LLAMA_SWAP_CONFIG"),
        help="Write output YAML to this path (default: stdout).",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Optional header template file (default: llama-swap.yaml.example next to this script).",
    )

    args = parser.parse_args()

    models_dir = Path(args.models_dir).expanduser()
    if not models_dir.is_dir():
        print(f"ERROR: models directory not found: {models_dir}", file=sys.stderr)
        return 2

    ggufs = sorted(p for p in models_dir.rglob("*.gguf") if p.is_file())
    if not ggufs:
        print(f"ERROR: no .gguf files found under: {models_dir}", file=sys.stderr)
        return 3

    template_path = (
        Path(args.template).expanduser()
        if args.template
        else Path(__file__).with_name("llama-swap.yaml.example")
    )
    header_lines = read_header(template_path)

    used_names = {}
    entries: List[str] = []
    for model_path in ggufs:
        base_name = model_path.stem
        count = used_names.get(base_name, 0) + 1
        used_names[base_name] = count
        name = base_name if count == 1 else f"{base_name}-{count}"

        thinking_optional = template_supports_thinking(model_path)
        if thinking_optional:
            entries.append(f"  {yaml_key(name)}:")
            entries.append(f"    cmd: {build_cmd(args.llama_server, model_path, False)}")
            entries.append("")
            entries.append(f"  {yaml_key(name)}-thinking:")
            entries.append(f"    cmd: {build_cmd(args.llama_server, model_path, True)}")
        else:
            entries.append(f"  {yaml_key(name)}:")
            entries.append(f"    cmd: {build_cmd(args.llama_server, model_path, None)}")
        entries.append("")

    output_lines = header_lines + [""] + entries
    text = "\n".join(output_lines).rstrip() + "\n"

    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
