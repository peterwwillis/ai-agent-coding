#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import shlex
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

DEFAULT_N_GPU_LAYERS = "40"
LOG_NAME_MAX = 64
BATCH_AUTO = "auto"

HELP_EPILOG = """Batch/ubatch guidance

Hardware Type                            Recommended -b   Recommended -ub   Why?
High-End GPU (e.g., RTX 4090)            4096              1024 - 2048        Fully utilizes many CUDA cores; 2048 can reduce prompt processing time by ~25%.
Mid-Range GPU (8GB-12GB VRAM)           2048              512                Prevents OOM (Out of Memory) while maintaining decent speeds.
CPU / Mixed Inference                   2048              1024               Can provide up to a 3x speed gain for MoE models (like Mixtral).
Apple Silicon (M2/M3 Max)               4096              1024               Efficiently uses high unified memory bandwidth.

Optimization Strategy
  For Speed (Prompt Processing): Increase -ub. Larger values allow the GPU to process more prompt tokens in parallel, though the benefit often plateaus at 2048.
  For Memory (VRAM Constraints): Decrease -b and -ub. High values allocate more memory for the logits/embeddings buffer. If you hit an OOM error, lower both values to 512 or 256.
  Special Case (Embeddings/Reranking): You must set -b and -ub to the same value, or the server may fail.

Start with -b 2048 -ub 512. If your GPU memory (VRAM) is less than 50% full during processing, try doubling -ub to 1024 and check if your "tokens per second" (TPS) for prompt processing increases.
"""


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


def read_gguf_block_count(path: Path) -> Optional[int]:
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
    int_types = {
        0: "<B",
        1: "<b",
        2: "<H",
        3: "<h",
        4: "<I",
        5: "<i",
        10: "<Q",
        11: "<q",
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

    def read_int_value(f, vtype: int) -> Optional[int]:
        fmt = int_types.get(vtype)
        if not fmt:
            return None
        size = struct.calcsize(fmt)
        data = f.read(size)
        if len(data) != size:
            raise EOFError
        return int(struct.unpack(fmt, data)[0])

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
                if key.endswith(".block_count") or key in {"block_count", "n_layer"}:
                    value = read_int_value(f, vtype)
                    if value is None:
                        skip_value(f, vtype)
                        return None
                    return value
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


def sanitize_log_stem(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    cleaned = cleaned.strip("._-")
    if not cleaned:
        return "model"
    return cleaned[:LOG_NAME_MAX]


def parse_n_gpu_layers(value: str) -> str:
    v = value.strip().lower()
    if v == "auto":
        return v
    if v.isdigit():
        return str(int(v))
    raise argparse.ArgumentTypeError("n-gpu-layers must be an integer or 'auto'")


def parse_batch_setting(value: str) -> str:
    v = value.strip().lower()
    if v == BATCH_AUTO:
        return v
    if v.isdigit() and int(v) > 0:
        return str(int(v))
    raise argparse.ArgumentTypeError("batch size must be a positive integer or 'auto'")


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    batch: int
    ubatch: int
    vram_gb: Optional[float] = None


def read_int_file(path: Path) -> Optional[int]:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def read_nvidia_vram_bytes() -> Optional[int]:
    base = Path("/proc/driver/nvidia/gpus")
    if not base.is_dir():
        return None
    for info in base.glob("*/information"):
        try:
            for line in info.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "Video Memory" in line:
                    _, value = line.split(":", 1)
                    parts = value.strip().split()
                    if len(parts) >= 2 and parts[0].isdigit():
                        size = int(parts[0])
                        unit = parts[1].lower()
                        if unit.startswith("mb"):
                            return size * 1024 * 1024
                        if unit.startswith("gb"):
                            return size * 1024 * 1024 * 1024
        except OSError:
            continue
    return None


def detect_linux_gpu(log) -> tuple[Optional[str], Optional[int]]:
    drm = Path("/sys/class/drm")
    if not drm.is_dir():
        log("No /sys/class/drm detected; skipping GPU vendor detection.")
        return None, None
    for vendor_path in drm.glob("card*/device/vendor"):
        try:
            vendor = vendor_path.read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        log(f"Detected DRM vendor {vendor} at {vendor_path.parent}")
        device_dir = vendor_path.parent
        if vendor == "0x1002":
            vram_bytes = read_int_file(device_dir / "mem_info_vram_total")
            if vram_bytes:
                log(f"AMD VRAM total: {vram_bytes / (1024**3):.1f} GB")
            else:
                log("AMD VRAM total not available from mem_info_vram_total.")
            return "amd", vram_bytes
        if vendor == "0x10de":
            vram_bytes = read_nvidia_vram_bytes()
            if vram_bytes:
                log(f"NVIDIA VRAM total: {vram_bytes / (1024**3):.1f} GB")
            else:
                log("NVIDIA VRAM total not available from /proc/driver/nvidia/gpus.")
            return "nvidia", vram_bytes
        if vendor == "0x8086":
            log("Intel GPU detected; treating as mixed inference profile.")
            return "intel", None
    return None, None


def detect_hardware_profile(log) -> HardwareProfile:
    system = platform.system()
    machine = platform.machine().lower()
    log(f"Platform: {system} ({machine})")
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        log("Apple Silicon detected; using Apple Silicon batch profile.")
        return HardwareProfile("apple-silicon", 4096, 1024)
    if system == "Linux":
        vendor, vram_bytes = detect_linux_gpu(log)
        vram_gb = vram_bytes / (1024**3) if vram_bytes else None
        if vendor in {"amd", "nvidia"}:
            if vram_gb is not None and vram_gb >= 20:
                log("High-end GPU detected; using 4096/1024 batch profile.")
                return HardwareProfile("high-end-gpu", 4096, 1024, vram_gb)
            if vram_gb is not None and vram_gb >= 8:
                log("Mid-range GPU detected; using 2048/512 batch profile.")
                return HardwareProfile("mid-range-gpu", 2048, 512, vram_gb)
            if vram_gb is not None and vram_gb < 8:
                log("Low VRAM GPU detected; using 1024/256 batch profile.")
                return HardwareProfile("low-vram-gpu", 1024, 256, vram_gb)
            log("GPU detected without VRAM info; using mid-range GPU profile.")
            return HardwareProfile("mid-range-gpu", 2048, 512, vram_gb)
        if vendor == "intel":
            log("Intel GPU detected; using CPU/mixed batch profile.")
            return HardwareProfile("cpu/mixed", 2048, 1024)
        log("No discrete GPU detected; using CPU/mixed batch profile.")
    return HardwareProfile("cpu/mixed", 2048, 1024)


def auto_batch_settings(profile: HardwareProfile, model_size_gb: float, log) -> tuple[int, int]:
    batch = profile.batch
    ubatch = profile.ubatch
    log(f"Auto batch baseline from profile '{profile.name}': -b {batch} -ub {ubatch}")
    log(f"Model size: {model_size_gb:.1f} GB")
    if model_size_gb >= 20:
        batch = min(batch, 2048)
        ubatch = min(ubatch, 512)
        log("Model >= 20 GB; capping batch/ubatch to 2048/512.")
    elif model_size_gb >= 12:
        batch = min(batch, 2048)
        ubatch = min(ubatch, 512)
        log("Model >= 12 GB; capping batch/ubatch to 2048/512.")
    elif model_size_gb <= 4:
        if batch >= 4096:
            ubatch = max(ubatch, 2048)
            log("Small model; increasing ubatch to at least 2048.")
        elif batch >= 2048:
            ubatch = max(ubatch, 1024)
            log("Small model; increasing ubatch to at least 1024.")
    else:
        log("No size-based batch adjustments needed.")
    if batch <= ubatch:
        batch = max(ubatch + 1, ubatch * 2)
        log(f"Adjusted batch to keep batch > ubatch: -b {batch} -ub {ubatch}")
    return batch, ubatch


def resolve_batch_settings(
    batch_arg: str,
    ubatch_arg: str,
    allow_equal: bool,
    profile: Optional[HardwareProfile],
    model_path: Path,
    log,
) -> tuple[int, int]:
    model_size_gb = model_path.stat().st_size / (1024**3)
    auto_batch = None
    auto_ubatch = None
    if batch_arg == BATCH_AUTO or ubatch_arg == BATCH_AUTO:
        if profile is None:
            raise ValueError("auto batch sizing requires a detected hardware profile")
        auto_batch, auto_ubatch = auto_batch_settings(profile, model_size_gb, log)
    batch = auto_batch if batch_arg == BATCH_AUTO else int(batch_arg)
    ubatch = auto_ubatch if ubatch_arg == BATCH_AUTO else int(ubatch_arg)
    if batch_arg != BATCH_AUTO:
        log(f"Using explicit batch size: -b {batch}")
    if ubatch_arg != BATCH_AUTO:
        log(f"Using explicit ubatch size: -ub {ubatch}")
    if allow_equal:
        log("Allowing batch == ubatch for embeddings/reranking mode.")
        if batch < ubatch:
            raise ValueError("batch size must be greater than or equal to ubatch size")
    else:
        if batch <= ubatch:
            raise ValueError("batch size must be greater than ubatch size")
    if batch_arg == BATCH_AUTO or ubatch_arg == BATCH_AUTO:
        log(
            f"Auto batch settings for '{model_path.name}': -b {batch} -ub {ubatch} (profile: {profile.name}, model {model_size_gb:.1f} GB)"
        )
    return batch, ubatch


def parse_flash_attn(value: str) -> str:
    v = value.strip().lower()
    if v in {"on", "off", "auto"}:
        return v
    raise argparse.ArgumentTypeError("flash-attn must be one of: on, off, auto")


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
    flash_attn: str,
    cache_type_k: str,
    cache_type_v: str,
    n_gpu_layers: Optional[str],
    mmap: bool,
    batch_size: Optional[int],
    ubatch_size: Optional[int],
    log_file: Path,
) -> str:
    cmd_parts = [
        shlex.quote(llama_server),
        "--offline",
        "--log-file",
        shlex.quote(str(log_file)),
        "--log-colors",
        "off",
        "--log-prefix",
        "--log-timestamps",
        "-m",
        shlex.quote(str(model_path)),
        "--ctx-size",
        str(ctx_size),
        "--cache-type-k",
        shlex.quote(cache_type_k),
        "--cache-type-v",
        shlex.quote(cache_type_v),
        "--flash-attn",
        shlex.quote(flash_attn),
    ]
    if batch_size is not None and ubatch_size is not None:
        cmd_parts.extend(["--batch-size", str(batch_size), "--ubatch-size", str(ubatch_size)])
    if n_gpu_layers is not None:
        cmd_parts.extend(["--n-gpu-layers", str(n_gpu_layers)])
    cmd_parts.append("--mmap" if mmap else "--no-mmap")
    if thinking is not None:
        val = "true" if thinking else "false"
        cmd_parts.extend(["--chat-template-kwargs", f"'{{\"enable_thinking\":{val}}}'"])
    cmd_parts.extend(["--port", "${PORT}"])
    return " ".join(cmd_parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a llama-swap config from llama.cpp cached GGUF models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_EPILOG,
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
        type=parse_flash_attn,
        default="on",
        help="Flash attention mode: on, off, auto (default: on).",
    )
    parser.add_argument(
        "--cache-type-k",
        default="q8_0",
        help="KV cache K quantization type (default: q8_0).",
    )
    parser.add_argument(
        "--cache-type-v",
        default="q8_0",
        help="KV cache V quantization type (default: q8_0).",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=parse_n_gpu_layers,
        default=None,
        help="Number of layers to offload to GPU (default: not passed unless set).",
    )
    parser.add_argument(
        "--gpu-layer-autodetect",
        action="store_true",
        help="Auto-detect GPU layers from model metadata.",
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
        type=parse_batch_setting,
        default=None,
        help="Batch size (default: not passed unless set).",
    )
    parser.add_argument(
        "--ubatch-size",
        type=parse_batch_setting,
        default=None,
        help="Ubatch size (default: not passed unless set).",
    )
    parser.add_argument(
        "--batch-size-autodetect",
        action="store_true",
        help="Auto-detect batch/ubatch size based on hardware and model.",
    )
    parser.add_argument(
        "--allow-equal-batch",
        action="store_true",
        help="Allow batch size to equal ubatch size (for embeddings/reranking).",
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

    if (args.batch_size == BATCH_AUTO or args.ubatch_size == BATCH_AUTO) and not args.batch_size_autodetect:
        print(
            "ERROR: batch size 'auto' requires --batch-size-autodetect.",
            file=sys.stderr,
        )
        return 2
    if (args.batch_size is None) != (args.ubatch_size is None):
        print("ERROR: --batch-size and --ubatch-size must be provided together.", file=sys.stderr)
        return 2

    log_dir = default_llama_cache_dir()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"ERROR: unable to create log directory: {log_dir} ({exc})", file=sys.stderr)
        return 2

    hardware_profile = None
    batch_arg = args.batch_size
    ubatch_arg = args.ubatch_size
    if args.batch_size_autodetect:
        if batch_arg is None and ubatch_arg is None:
            batch_arg = BATCH_AUTO
            ubatch_arg = BATCH_AUTO
            log("Batch size autodetect enabled; using auto batch/ubatch.")
        else:
            log("Batch size autodetect enabled, but explicit batch/ubatch provided; skipping auto.")
    else:
        if batch_arg is None and ubatch_arg is None:
            log("Batch/ubatch not provided; leaving llama-server defaults.")
        else:
            log("Using explicit batch/ubatch values; autodetect disabled.")

    n_gpu_layers_arg = args.n_gpu_layers
    if args.gpu_layer_autodetect:
        if n_gpu_layers_arg is None:
            log("GPU layer autodetect enabled; using GGUF block_count.")
        else:
            log("GPU layer autodetect enabled, but explicit n-gpu-layers provided; skipping auto.")
    else:
        if n_gpu_layers_arg is None:
            log("n-gpu-layers not provided; leaving llama-server defaults.")
        else:
            log("Using explicit n-gpu-layers; autodetect disabled.")

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
        else Path(__file__).with_name("llama-swap.yaml.example")
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
        log_file = log_dir / f"llama-swap-{sanitize_log_stem(name)}.log"
        log(f"Log file for '{name}': {log_file}")

        n_gpu_layers = n_gpu_layers_arg
        if n_gpu_layers is None and args.gpu_layer_autodetect:
            block_count = read_gguf_block_count(model_path)
            if block_count:
                n_gpu_layers = str(block_count + 1)
                log(f"Auto n-gpu-layers for '{name}': {n_gpu_layers} (block_count + 1)")
            else:
                n_gpu_layers = DEFAULT_N_GPU_LAYERS
                log(
                    f"Could not read block_count for '{name}'; using n-gpu-layers={n_gpu_layers}."
                )
        if n_gpu_layers is not None:
            log(f"Using n-gpu-layers for '{name}': {n_gpu_layers}")

        batch_size = None
        ubatch_size = None
        if batch_arg is not None and ubatch_arg is not None:
            if batch_arg == BATCH_AUTO or ubatch_arg == BATCH_AUTO:
                if hardware_profile is None:
                    hardware_profile = detect_hardware_profile(log)
                    vram_info = (
                        f", {hardware_profile.vram_gb:.1f} GB VRAM"
                        if hardware_profile.vram_gb
                        else ""
                    )
                    log(f"Auto batch profile: {hardware_profile.name}{vram_info}")
            try:
                batch_size, ubatch_size = resolve_batch_settings(
                    batch_arg,
                    ubatch_arg,
                    args.allow_equal_batch,
                    hardware_profile,
                    model_path,
                    log,
                )
            except ValueError as exc:
                print(f"ERROR: {exc} for model '{name}'", file=sys.stderr)
                return 2

        thinking_optional = template_supports_thinking(model_path)
        log(f"Adding model '{name}' (thinking optional: {'yes' if thinking_optional else 'no'}).")
        if thinking_optional:
            entry_blocks.append(
                (
                    name,
                    [
                        f"  {yaml_key(name)}:",
                        f"    cmd: {build_cmd(args.llama_server, model_path, False, args.ctx_size, args.flash_attn, args.cache_type_k, args.cache_type_v, n_gpu_layers, args.mmap, batch_size, ubatch_size, log_file)}",
                    ],
                )
            )
            entry_blocks.append(
                (
                    f"{name}-thinking",
                    [
                        f"  {yaml_key(name)}-thinking:",
                        f"    cmd: {build_cmd(args.llama_server, model_path, True, args.ctx_size, args.flash_attn, args.cache_type_k, args.cache_type_v, n_gpu_layers, args.mmap, batch_size, ubatch_size, log_file)}",
                    ],
                )
            )
        else:
            entry_blocks.append(
                (
                    name,
                    [
                        f"  {yaml_key(name)}:",
                        f"    cmd: {build_cmd(args.llama_server, model_path, None, args.ctx_size, args.flash_attn, args.cache_type_k, args.cache_type_v, n_gpu_layers, args.mmap, batch_size, ubatch_size, log_file)}",
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
