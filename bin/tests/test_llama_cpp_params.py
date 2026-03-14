"""Tests for llama-cpp-params.py."""
from __future__ import annotations

import sys
from pathlib import Path

import importlib
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
lcpp = importlib.import_module("llama-cpp-params")

from tests.conftest import gguf_u32_value, make_minimal_gguf


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1024", 1024),
        ("1KB", 1024),
        ("2MB", 2 * 1024**2),
        ("1.5GB", int(1.5 * 1024**3)),
    ],
)
def test_parse_bytes(raw, expected):
    assert lcpp.parse_bytes(raw) == expected


def test_cache_type_bytes_known():
    size, known = lcpp.cache_type_bytes("q8_0")
    assert size == pytest.approx(1.0)
    assert known is True


def test_cache_type_bytes_unknown():
    size, known = lcpp.cache_type_bytes("weird", fallback_bytes=2.0)
    assert size == pytest.approx(2.0)
    assert known is False


def test_read_gguf_metadata(tmp_path):
    model_path = tmp_path / "model.gguf"
    make_minimal_gguf(
        model_path,
        kv={
            "block_count": gguf_u32_value(32),
            "n_embd": gguf_u32_value(4096),
            "n_head": gguf_u32_value(32),
            "n_head_kv": gguf_u32_value(8),
            "n_expert": gguf_u32_value(8),
            "n_expert_used": gguf_u32_value(2),
        },
    )
    meta = lcpp.read_gguf_metadata(model_path)
    assert meta.n_layers == 32
    assert meta.n_embd == 4096
    assert meta.n_heads == 32
    assert meta.n_kv_heads == 8
    assert meta.n_experts == 8
    assert meta.n_experts_used == 2


def test_estimate_kv_cache_per_layer():
    kv_bytes, warnings = lcpp.estimate_kv_cache_per_layer(
        ctx_size=2048,
        n_embd=4096,
        n_heads=32,
        n_kv_heads=8,
        cache_type_k="q8_0",
        cache_type_v="q8_0",
    )
    assert warnings == []
    assert kv_bytes == pytest.approx(2048 * 2048)


def test_estimate_gpu_layers_with_moe_offset():
    estimate = lcpp.estimate_max_gpu_layers(
        model_size_bytes=16 * 1024**3,
        n_layers=32,
        vram_bytes=8 * 1024**3,
        kv_cache_per_layer_bytes=4_194_304,
        batch_bytes=0,
        vram_overhead_bytes=0,
        moe_cpu_offset=2,
        is_moe=True,
    )
    assert estimate.max_gpu_layers == 15
    assert estimate.recommended_gpu_layers == 13
