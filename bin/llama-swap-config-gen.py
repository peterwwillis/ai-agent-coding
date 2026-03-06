#!/usr/bin/env python3
"""Generate a llama-swap config.yaml from GGUF models in the llama.cpp cache.

Uses the ruamel.yaml library for YAML loading and saving so that
comments and whitespace are preserved in existing config files.
"""
from __future__ import annotations

import argparse
import os
import platform
import shlex
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

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

For llama-swap config file, see: https://github.com/mostlygeek/llama-swap/blob/main/config.example.yaml
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


def parse_flash_attn(value: str) -> str:
    v = value.strip().lower()
    if v in {"on", "off", "auto"}:
        return v
    raise argparse.ArgumentTypeError("flash-attn must be one of: on, off, auto")


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
) -> LiteralScalarString:
    # Build logical groups of flag + value(s), one per line.
    lines = [
        shlex.quote(llama_server),
        "  --offline",
        f"  --log-file {shlex.quote(str(log_file))}",
        "  --log-colors off",
        "  --log-prefix",
        "  --log-timestamps",
        f"  -m {shlex.quote(str(model_path))}",
        f"  --ctx-size {ctx_size}",
        f"  --cache-type-k {shlex.quote(cache_type_k)}",
        f"  --cache-type-v {shlex.quote(cache_type_v)}",
        f"  --flash-attn {shlex.quote(flash_attn)}",
    ]
    if batch_size is not None and ubatch_size is not None:
        lines.append(f"  --batch-size {batch_size} --ubatch-size {ubatch_size}")
    if n_gpu_layers is not None:
        lines.append(f"  --n-gpu-layers {n_gpu_layers}")
    lines.append("  --mmap" if mmap else "  --no-mmap")
    if thinking is not None:
        val = "true" if thinking else "false"
        lines.append(f"  --chat-template-kwargs '{{\"enable_thinking\":{val}}}'")
    lines.append("  --port ${PORT}")
    # Join with backslash-newline so the shell treats it as one command.
    cmd = " \\\n".join(lines) + "\n"
    return LiteralScalarString(cmd)


def _make_yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.width = 10000
    return y


def load_yaml_file(path: Path):
    if not path.is_file():
        return {}
    y = _make_yaml()
    data = y.load(path)
    return data if isinstance(data, dict) else {}


def save_yaml_file(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    y = _make_yaml()
    with path.open("w", encoding="utf-8") as f:
        y.dump(data, f)


def set_parser_args(parser):
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


class MyApp:

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Generate a llama-swap config from llama.cpp cached GGUF models.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=HELP_EPILOG,
        )

        set_parser_args(parser)
        self.args = parser.parse_args()

        self.log_dir = default_llama_cache_dir()
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(f"ERROR: unable to create log directory: {self.log_dir} ({exc})", file=sys.stderr)
            return

        self.log = (lambda msg: print(msg, file=sys.stderr)) if self.args.verbose else (lambda _msg: None)

        self.hardware_profile = None

        self.batch_arg = self.args.batch_size
        self.ubatch_arg = self.args.ubatch_size
        if self.args.batch_size_autodetect:
            if self.batch_arg is None and self.ubatch_arg is None:
                self.batch_arg = BATCH_AUTO
                self.ubatch_arg = BATCH_AUTO
                self.log("Batch size autodetect enabled; using auto batch/ubatch.")
            else:
                self.log("Batch size autodetect enabled, but explicit batch/ubatch provided; skipping auto.")
        else:
            if self.batch_arg is None and self.ubatch_arg is None:
                self.log("Batch/ubatch not provided; leaving llama-server defaults.")
            else:
                self.log("Using explicit batch/ubatch values; autodetect disabled.")

    def resolve_batch_settings(
        self,
        allow_equal: bool,
        profile: Optional[HardwareProfile],
        model_path: Path,
        log,
    ) -> tuple[int, int]:
        model_size_gb = model_path.stat().st_size / (1024**3)
        auto_batch = None
        auto_ubatch = None

        if self.batch_arg == BATCH_AUTO or self.ubatch_arg == BATCH_AUTO:
            if profile is None:
                raise ValueError("auto batch sizing requires a detected hardware profile")
            auto_batch, auto_ubatch = auto_batch_settings(profile, model_size_gb, log)

        batch = auto_batch if self.batch_arg == BATCH_AUTO else int(self.batch_arg)
        ubatch = auto_ubatch if self.ubatch_arg == BATCH_AUTO else int(self.ubatch_arg)
        if self.batch_arg != BATCH_AUTO:
            log(f"Using explicit batch size: -b {batch}")
        if self.ubatch_arg != BATCH_AUTO:
            log(f"Using explicit ubatch size: -ub {ubatch}")
        if allow_equal:
            log("Allowing batch == ubatch for embeddings/reranking mode.")
            if batch < ubatch:
                raise ValueError("batch size must be greater than or equal to ubatch size")
        else:
            if batch <= ubatch:
                raise ValueError("batch size must be greater than ubatch size")

        if self.batch_arg == BATCH_AUTO or self.ubatch_arg == BATCH_AUTO:
            log(f"Auto batch settings for '{model_path.name}': -b {batch} -ub {ubatch} (profile: {profile.name}, model {model_size_gb:.1f} GB)")
        return batch, ubatch

    def make_model_entries(self, ggufs: List[Path]) -> Dict[str, dict]:
        args = self.args
        log = self.log
        used_names: Dict[str, int] = {}
        entries: Dict[str, dict] = {}

        for model_path in ggufs:
            base_name = model_path.stem
            count = used_names.get(base_name, 0) + 1
            used_names[base_name] = count
            name = base_name if count == 1 else f"{base_name}-{count}"
            log_file = self.log_dir / f"llama-swap-{sanitize_log_stem(name)}.log"
            log(f"Log file for '{name}': {log_file}")

            n_gpu_layers = args.n_gpu_layers
            if n_gpu_layers is None and args.gpu_layer_autodetect:
                block_count = read_gguf_block_count(model_path)
                if block_count:
                    n_gpu_layers = str(block_count + 1)
                    log(f"Auto n-gpu-layers for '{name}': {n_gpu_layers} (block_count + 1)")
                else:
                    n_gpu_layers = DEFAULT_N_GPU_LAYERS
                    log(f"Could not read block_count for '{name}'; using n-gpu-layers={n_gpu_layers}.")
            if n_gpu_layers is not None:
                log(f"Using n-gpu-layers for '{name}': {n_gpu_layers}")

            batch_size = None
            ubatch_size = None
            if self.batch_arg is not None and self.ubatch_arg is not None:
                if self.batch_arg == BATCH_AUTO or self.ubatch_arg == BATCH_AUTO:
                    if self.hardware_profile is None:
                        self.hardware_profile = detect_hardware_profile(log)
                        vram_info = f", {self.hardware_profile.vram_gb:.1f} GB VRAM" if self.hardware_profile.vram_gb else ""
                        log(f"Auto batch profile: {self.hardware_profile.name}{vram_info}")
                try:
                    batch_size, ubatch_size = self.resolve_batch_settings(
                        args.allow_equal_batch,
                        self.hardware_profile,
                        model_path,
                        log,
                    )
                except ValueError as exc:
                    print(f"ERROR: {exc} for model '{name}'", file=sys.stderr)
                    return {}

            thinking_optional = template_supports_thinking(model_path)
            log(f"Adding model '{name}' (thinking optional: {'yes' if thinking_optional else 'no'}).")

            cmd_kwargs = dict(
                llama_server=args.llama_server,
                model_path=model_path,
                ctx_size=args.ctx_size,
                flash_attn=args.flash_attn,
                cache_type_k=args.cache_type_k,
                cache_type_v=args.cache_type_v,
                n_gpu_layers=n_gpu_layers,
                mmap=args.mmap,
                batch_size=batch_size,
                ubatch_size=ubatch_size,
                log_file=log_file,
            )

            if thinking_optional:
                entries[name] = {"cmd": build_cmd(thinking=False, **cmd_kwargs)}
                entries[f"{name}-thinking"] = {"cmd": build_cmd(thinking=True, **cmd_kwargs)}
            else:
                entries[name] = {"cmd": build_cmd(thinking=None, **cmd_kwargs)}

        return entries

    def main(self) -> int:
        args = self.args
        log = self.log

        if (args.batch_size == BATCH_AUTO or args.ubatch_size == BATCH_AUTO) and not args.batch_size_autodetect:
            print(
                "ERROR: batch size 'auto' requires --batch-size-autodetect.",
                file=sys.stderr,
            )
            return 2
        if (args.batch_size is None) != (args.ubatch_size is None):
            print("ERROR: --batch-size and --ubatch-size must be provided together.", file=sys.stderr)
            return 2

        if args.gpu_layer_autodetect:
            if args.n_gpu_layers is None:
                log("GPU layer autodetect enabled; using GGUF block_count.")
            else:
                log("GPU layer autodetect enabled, but explicit n-gpu-layers provided; skipping auto.")
        else:
            if args.n_gpu_layers is None:
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

        template_path = Path(args.template).expanduser() if args.template else Path(__file__).with_name("llama-swap.yaml.example")
        base_config = load_yaml_file(template_path)
        if template_path.is_file():
            log(f"Using template from: {template_path}")
        else:
            log("Template not found; using minimal config.")
        # Don't carry over template's models section
        base_config.pop("models", None)

        new_entries = self.make_model_entries(ggufs)
        if not new_entries and ggufs:
            return 2

        if args.output == "-":
            config = dict(base_config)
            config["models"] = new_entries
            y = _make_yaml()
            y.dump(config, sys.stdout)
            log("Wrote config to stdout.")
            return 0

        out_path = Path(args.output).expanduser()
        if not out_path.exists():
            config = dict(base_config)
            config["models"] = new_entries
            save_yaml_file(out_path, config)
            log(f"Wrote new config to: {out_path}")
            return 0

        existing_config = load_yaml_file(out_path)
        existing_models = existing_config.get("models") or {}

        pruned = 0
        if args.prune_missing:
            desired_keys = set(new_entries.keys())
            to_remove = [k for k in existing_models if k not in desired_keys]
            for k in to_remove:
                log(f"Pruned model {k}")
                del existing_models[k]
                pruned += 1

        added = 0
        for name, entry in new_entries.items():
            if name not in existing_models:
                existing_models[name] = entry
                added += 1

        existing_config["models"] = existing_models
        save_yaml_file(out_path, existing_config)
        log(f"Updated config at: {out_path} (added {added} entr{'y' if added == 1 else 'ies'}).")
        return 0


if __name__ == "__main__":
    app = MyApp()
    raise SystemExit(app.main())
