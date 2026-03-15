#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

CACHE_TYPE_BITS = {
    "f32": 32,
    "f16": 16,
    "bf16": 16,
    "q8_0": 8,
    "q8_1": 8,
    "q8_k": 8,
    "q6_k": 6,
    "q5_0": 5,
    "q5_1": 5,
    "q4_0": 4,
    "q4_1": 4,
    "q4_k": 4,
    "iq4_nl": 4,
    "iq4_xs": 4,
    "iq3_xxs": 3,
    "iq3_s": 3,
    "iq2_xxs": 2,
}

ACTIVATION_TYPE_BYTES = {
    "f32": 4,
    "f16": 2,
    "bf16": 2,
}

DEFAULT_CACHE_TYPE = "q8_0"
DEFAULT_CTX_SIZE = 4096
DEFAULT_BATCH_SIZE = 2048
DEFAULT_UBATCH_SIZE = 512
DEFAULT_ACTIVATION_TYPE = "f16"
DEFAULT_FLASH_ATTN = "auto"


@dataclass
class ModelMetadata:
    n_layers: Optional[int] = None
    n_embd: Optional[int] = None
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None
    n_experts: Optional[int] = None
    n_experts_used: Optional[int] = None
    context_length: Optional[int] = None


@dataclass
class ModelInfo:
    path: Optional[Path]
    size_bytes: int
    metadata: ModelMetadata


@dataclass
class EstimateResult:
    available_vram_bytes: int
    per_layer_weights_bytes: float
    kv_cache_per_layer_bytes: float
    batch_bytes: float
    per_layer_total_bytes: float
    max_gpu_layers: int
    recommended_gpu_layers: int
    estimated_gpu_bytes: float


def human_size(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for unit in units:
        if f < 1024.0 or unit == "TB":
            return f"{f:.2f} {unit}" if unit != "B" else f"{int(f)} B"
        f /= 1024.0
    return f"{f:.2f} TB"


def parse_bytes(value: str) -> int:
    raw = value.strip().lower()
    if raw.isdigit():
        return int(raw)
    match = re.match(r"^([0-9]*\.?[0-9]+)\s*([a-z]+)$", raw)
    if not match:
        raise argparse.ArgumentTypeError("size must be like '12GB' or '2048'")
    number = float(match.group(1))
    unit = match.group(2)
    unit_map = {
        "b": 1,
        "kb": 1024,
        "kib": 1024,
        "mb": 1024**2,
        "mib": 1024**2,
        "gb": 1024**3,
        "gib": 1024**3,
        "tb": 1024**4,
        "tib": 1024**4,
    }
    if unit not in unit_map:
        raise argparse.ArgumentTypeError(f"unknown size unit '{unit}'")
    return int(number * unit_map[unit])


def cache_type_bytes(cache_type: str, fallback_bytes: float = 2.0) -> tuple[float, bool]:
    key = cache_type.strip().lower()
    bits = CACHE_TYPE_BITS.get(key)
    if bits is None:
        return fallback_bytes, False
    return bits / 8.0, True


def activation_type_bytes(activation_type: str, fallback_bytes: float = 2.0) -> tuple[float, bool]:
    key = activation_type.strip().lower()
    size = ACTIVATION_TYPE_BYTES.get(key)
    if size is None:
        return fallback_bytes, False
    return float(size), True


def read_gguf_metadata(path: Path) -> ModelMetadata:
    gguf_types = {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        4: 4,
        5: 4,
        6: 4,
        7: 1,
        10: 8,
        11: 8,
        12: 8,
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
    float_types = {
        6: "<f",
        12: "<d",
    }
    wanted = {
        "n_layers": {"block_count", "n_layer"},
        "n_embd": {"embedding_length", "n_embd"},
        "n_heads": {"n_head"},
        "n_kv_heads": {"n_head_kv", "n_kv_head"},
        "n_experts": {"n_expert"},
        "n_experts_used": {"n_expert_used"},
        "context_length": {"context_length", "n_ctx"},
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

    def read_number(f, vtype: int) -> Optional[float]:
        fmt = int_types.get(vtype)
        if fmt:
            size = struct.calcsize(fmt)
            data = f.read(size)
            if len(data) != size:
                raise EOFError
            return float(struct.unpack(fmt, data)[0])
        fmt = float_types.get(vtype)
        if fmt:
            size = struct.calcsize(fmt)
            data = f.read(size)
            if len(data) != size:
                raise EOFError
            return float(struct.unpack(fmt, data)[0])
        return None

    def skip_value(f, vtype: int):
        if vtype in gguf_types:
            f.seek(gguf_types[vtype], os.SEEK_CUR)
            return
        if vtype == 8:
            length = read_u64(f)
            f.seek(length, os.SEEK_CUR)
            return
        if vtype == 9:
            elem_type = read_u32(f)
            count = read_u64(f)
            for _ in range(count):
                skip_value(f, elem_type)
            return
        raise ValueError(f"Unsupported GGUF value type: {vtype}")

    def match_key(key: str) -> Optional[str]:
        for canonical, aliases in wanted.items():
            for alias in aliases:
                if key == alias or key.endswith(f".{alias}"):
                    return canonical
        return None

    meta = ModelMetadata()
    if not path.is_file():
        return meta
    try:
        with path.open("rb") as f:
            if f.read(4) != b"GGUF":
                return meta
            _version = read_u32(f)
            _tensor_count = read_u64(f)
            kv_count = read_u64(f)
            for _ in range(kv_count):
                key = read_str(f)
                vtype = read_u32(f)
                canonical = match_key(key)
                if canonical:
                    value = read_number(f, vtype)
                    if value is None:
                        skip_value(f, vtype)
                        continue
                    if canonical == "context_length":
                        setattr(meta, "context_length", int(value))
                    elif canonical == "n_layers":
                        setattr(meta, "n_layers", int(value))
                    elif canonical == "n_embd":
                        setattr(meta, "n_embd", int(value))
                    elif canonical == "n_heads":
                        setattr(meta, "n_heads", int(value))
                    elif canonical == "n_kv_heads":
                        setattr(meta, "n_kv_heads", int(value))
                    elif canonical == "n_experts":
                        setattr(meta, "n_experts", int(value))
                    elif canonical == "n_experts_used":
                        setattr(meta, "n_experts_used", int(value))
                    else:
                        setattr(meta, canonical, int(value))
                else:
                    skip_value(f, vtype)
    except Exception:
        return meta
    return meta


def estimate_kv_cache_per_layer(
    ctx_size: int,
    n_embd: Optional[int],
    n_heads: Optional[int],
    n_kv_heads: Optional[int],
    cache_type_k: str,
    cache_type_v: str,
) -> tuple[float, list[str]]:
    warnings: list[str] = []
    if not n_embd or not n_heads:
        warnings.append("Missing n_embd or n_heads; KV cache estimate skipped.")
        return 0.0, warnings
    kv_heads = n_kv_heads or n_heads
    if n_heads == 0:
        warnings.append("n_heads was 0; KV cache estimate skipped.")
        return 0.0, warnings
    head_dim = n_embd / n_heads
    if n_embd % n_heads != 0:
        warnings.append("n_embd is not divisible by n_heads; using fractional head_dim.")
    bytes_k, known_k = cache_type_bytes(cache_type_k)
    bytes_v, known_v = cache_type_bytes(cache_type_v)
    if not known_k:
        warnings.append(f"Unknown cache type '{cache_type_k}', assuming {bytes_k} bytes/element.")
    if not known_v:
        warnings.append(f"Unknown cache type '{cache_type_v}', assuming {bytes_v} bytes/element.")
    bytes_per_token = kv_heads * head_dim * (bytes_k + bytes_v)
    return float(ctx_size) * bytes_per_token, warnings


def estimate_batch_bytes(
    n_embd: Optional[int],
    batch_size: int,
    ubatch_size: int,
    activation_type: str,
) -> tuple[float, list[str]]:
    warnings: list[str] = []
    if not n_embd:
        warnings.append("Missing n_embd; batch buffer estimate skipped.")
        return 0.0, warnings
    bytes_per, known = activation_type_bytes(activation_type)
    if not known:
        warnings.append(
            f"Unknown activation type '{activation_type}', assuming {bytes_per} bytes/element."
        )
    tokens = max(batch_size, ubatch_size)
    return float(tokens) * float(n_embd) * bytes_per, warnings


def estimate_max_gpu_layers(
    model_size_bytes: int,
    n_layers: int,
    vram_bytes: int,
    kv_cache_per_layer_bytes: float,
    batch_bytes: float,
    vram_overhead_bytes: int,
    moe_cpu_offset: int,
    is_moe: bool,
) -> EstimateResult:
    per_layer_weights = float(model_size_bytes) / float(n_layers)
    per_layer_total = per_layer_weights + kv_cache_per_layer_bytes
    available = vram_bytes - vram_overhead_bytes
    available -= int(batch_bytes)
    if available <= 0 or per_layer_total <= 0:
        max_layers = 0
    else:
        max_layers = int(math.floor(available / per_layer_total))
    max_layers = max(0, min(n_layers, max_layers))
    recommended = max_layers
    if is_moe and moe_cpu_offset > 0:
        recommended = max(0, recommended - moe_cpu_offset)
    estimated_gpu_bytes = recommended * per_layer_total + batch_bytes + vram_overhead_bytes
    return EstimateResult(
        available_vram_bytes=max(0, available),
        per_layer_weights_bytes=per_layer_weights,
        kv_cache_per_layer_bytes=kv_cache_per_layer_bytes,
        batch_bytes=batch_bytes,
        per_layer_total_bytes=per_layer_total,
        max_gpu_layers=max_layers,
        recommended_gpu_layers=recommended,
        estimated_gpu_bytes=estimated_gpu_bytes,
    )


def format_flags(args, model_path: Optional[Path], n_gpu_layers: int) -> list[str]:
    flags = []
    if model_path:
        flags.extend(["-m", str(model_path)])
    flags.extend(["--ctx-size", str(args.ctx_size)])
    flags.extend(["--batch-size", str(args.batch_size)])
    flags.extend(["--ubatch-size", str(args.ubatch_size)])
    flags.extend(["--cache-type-k", args.cache_type_k])
    flags.extend(["--cache-type-v", args.cache_type_v])
    flags.extend(["--n-gpu-layers", str(n_gpu_layers)])
    flags.extend(["--threads", str(args.threads)])
    flags.extend(["--threads-batch", str(args.threads_batch)])
    flags.extend(["--flash-attn", args.flash_attn])
    flags.append("--mmap" if args.mmap else "--no-mmap")
    return flags


def print_report(
    model: ModelInfo,
    args: argparse.Namespace,
    estimate: EstimateResult,
    is_moe: bool,
    warnings: list[str],
):
    lines = [
        "Model",
        f"  path: {model.path if model.path else 'n/a'}",
        f"  size: {human_size(model.size_bytes)}",
        f"  layers: {model.metadata.n_layers if model.metadata.n_layers else 'n/a'}",
        f"  embedding: {model.metadata.n_embd if model.metadata.n_embd else 'n/a'}",
        f"  heads: {model.metadata.n_heads if model.metadata.n_heads else 'n/a'}",
        f"  kv_heads: {model.metadata.n_kv_heads if model.metadata.n_kv_heads else 'n/a'}",
        f"  moe: {'yes' if is_moe else 'no'}",
        "",
        "System",
        f"  vram: {human_size(args.vram_bytes)}",
        f"  vram_reserved: {human_size(args.vram_overhead_bytes)}",
        f"  vram_available: {human_size(estimate.available_vram_bytes)}",
        "",
        "Parameters",
        f"  ctx_size: {args.ctx_size}",
        f"  batch_size: {args.batch_size}",
        f"  ubatch_size: {args.ubatch_size}",
        f"  cache_type_k: {args.cache_type_k}",
        f"  cache_type_v: {args.cache_type_v}",
        f"  activation_type: {args.activation_type}",
        "",
        "Estimates",
        f"  weights_per_layer: {human_size(estimate.per_layer_weights_bytes)}",
        f"  kv_cache_per_layer: {human_size(estimate.kv_cache_per_layer_bytes)}",
        f"  batch_buffer: {human_size(estimate.batch_bytes)}",
        f"  per_layer_total: {human_size(estimate.per_layer_total_bytes)}",
        f"  max_gpu_layers: {estimate.max_gpu_layers}",
        f"  recommended_gpu_layers: {estimate.recommended_gpu_layers}",
        f"  estimated_gpu_usage: {human_size(estimate.estimated_gpu_bytes)}",
        "",
        "Suggested flags",
        "  " + " ".join(format_flags(args, model.path, estimate.recommended_gpu_layers)),
    ]
    if is_moe and args.moe_cpu_offset:
        lines.append(f"  (moe cpu offset: {args.moe_cpu_offset} layers)")
    if warnings:
        lines.append("")
        lines.append("Notes")
        for message in warnings:
            lines.append(f"  - {message}")
    print("\n".join(lines))


def model_is_moe(metadata: ModelMetadata, force_moe: bool) -> bool:
    if force_moe:
        return True
    if metadata.n_experts and metadata.n_experts > 1:
        return True
    if metadata.n_experts_used and metadata.n_experts_used > 1:
        return True
    return False


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate llama.cpp parameters and GPU layer fit for a model."
    )
    parser.add_argument("--model-path", type=Path, help="Path to GGUF model file.")
    parser.add_argument("--model-size", type=parse_bytes, help="Model size if no file (e.g., 7GB).")
    parser.add_argument("--n-layers", type=int, help="Override layer count.")
    parser.add_argument("--n-embd", type=int, help="Override embedding size.")
    parser.add_argument("--n-heads", type=int, help="Override attention head count.")
    parser.add_argument("--n-kv-heads", type=int, help="Override KV head count.")
    parser.add_argument("--n-experts", type=int, help="Override expert count.")
    parser.add_argument("--n-experts-used", type=int, help="Override experts-used count.")
    parser.add_argument("--moe", action="store_true", help="Force MoE handling.")
    parser.add_argument("--moe-cpu-offset", type=int, default=0, help="Layers to keep on CPU.")
    parser.add_argument("--vram", type=parse_bytes, required=True, help="Total GPU VRAM (e.g. 16GB).")
    parser.add_argument(
        "--vram-reserve",
        type=parse_bytes,
        default=0,
        help="VRAM to reserve for overhead (e.g. 1GB).",
    )
    parser.add_argument("--ctx-size", type=int, default=DEFAULT_CTX_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--ubatch-size", type=int, default=DEFAULT_UBATCH_SIZE)
    parser.add_argument("--cache-type-k", default=DEFAULT_CACHE_TYPE)
    parser.add_argument("--cache-type-v", default=DEFAULT_CACHE_TYPE)
    parser.add_argument("--activation-type", default=DEFAULT_ACTIVATION_TYPE)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--threads-batch", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--flash-attn", choices=["auto", "on", "off"], default=DEFAULT_FLASH_ATTN)
    parser.add_argument("--mmap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    return parser.parse_args(argv)


def resolve_model_info(args: argparse.Namespace) -> ModelInfo:
    metadata = ModelMetadata()
    model_path = args.model_path
    if model_path:
        metadata = read_gguf_metadata(model_path)
    if args.n_layers is not None:
        metadata.n_layers = args.n_layers
    if args.n_embd is not None:
        metadata.n_embd = args.n_embd
    if args.n_heads is not None:
        metadata.n_heads = args.n_heads
    if args.n_kv_heads is not None:
        metadata.n_kv_heads = args.n_kv_heads
    if args.n_experts is not None:
        metadata.n_experts = args.n_experts
    if args.n_experts_used is not None:
        metadata.n_experts_used = args.n_experts_used

    if args.model_size is not None:
        size_bytes = args.model_size
    elif model_path and model_path.is_file():
        size_bytes = model_path.stat().st_size
    else:
        raise SystemExit("Model size not found. Provide --model-path or --model-size.")

    if metadata.n_layers is None or metadata.n_layers <= 0:
        raise SystemExit("Layer count not found. Provide --n-layers or a GGUF with block_count.")

    return ModelInfo(path=model_path, size_bytes=size_bytes, metadata=metadata)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    args.vram_bytes = args.vram
    args.vram_overhead_bytes = args.vram_reserve
    model = resolve_model_info(args)
    is_moe = model_is_moe(model.metadata, args.moe)

    kv_cache_per_layer, kv_warnings = estimate_kv_cache_per_layer(
        ctx_size=args.ctx_size,
        n_embd=model.metadata.n_embd,
        n_heads=model.metadata.n_heads,
        n_kv_heads=model.metadata.n_kv_heads,
        cache_type_k=args.cache_type_k,
        cache_type_v=args.cache_type_v,
    )
    batch_bytes, batch_warnings = estimate_batch_bytes(
        n_embd=model.metadata.n_embd,
        batch_size=args.batch_size,
        ubatch_size=args.ubatch_size,
        activation_type=args.activation_type,
    )
    warnings = kv_warnings + batch_warnings
    estimate = estimate_max_gpu_layers(
        model_size_bytes=model.size_bytes,
        n_layers=model.metadata.n_layers,
        vram_bytes=args.vram_bytes,
        kv_cache_per_layer_bytes=kv_cache_per_layer,
        batch_bytes=batch_bytes,
        vram_overhead_bytes=args.vram_overhead_bytes,
        moe_cpu_offset=args.moe_cpu_offset,
        is_moe=is_moe,
    )

    if args.json:
        payload: Dict[str, Any] = {
            "model": {
                "path": str(model.path) if model.path else None,
                "size_bytes": model.size_bytes,
                "n_layers": model.metadata.n_layers,
                "n_embd": model.metadata.n_embd,
                "n_heads": model.metadata.n_heads,
                "n_kv_heads": model.metadata.n_kv_heads,
                "n_experts": model.metadata.n_experts,
                "n_experts_used": model.metadata.n_experts_used,
                "moe": is_moe,
            },
            "system": {
                "vram_bytes": args.vram_bytes,
                "vram_reserved_bytes": args.vram_overhead_bytes,
            },
            "parameters": {
                "ctx_size": args.ctx_size,
                "batch_size": args.batch_size,
                "ubatch_size": args.ubatch_size,
                "cache_type_k": args.cache_type_k,
                "cache_type_v": args.cache_type_v,
                "activation_type": args.activation_type,
                "threads": args.threads,
                "threads_batch": args.threads_batch,
                "flash_attn": args.flash_attn,
                "mmap": args.mmap,
                "moe_cpu_offset": args.moe_cpu_offset,
            },
            "estimate": {
                "available_vram_bytes": estimate.available_vram_bytes,
                "per_layer_weights_bytes": estimate.per_layer_weights_bytes,
                "kv_cache_per_layer_bytes": estimate.kv_cache_per_layer_bytes,
                "batch_bytes": estimate.batch_bytes,
                "per_layer_total_bytes": estimate.per_layer_total_bytes,
                "max_gpu_layers": estimate.max_gpu_layers,
                "recommended_gpu_layers": estimate.recommended_gpu_layers,
                "estimated_gpu_bytes": estimate.estimated_gpu_bytes,
                "suggested_flags": format_flags(args, model.path, estimate.recommended_gpu_layers),
                "notes": warnings,
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print_report(model, args, estimate, is_moe, warnings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
