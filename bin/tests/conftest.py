"""Shared fixtures for llama-swap-config-gen tests."""
from __future__ import annotations

import struct
from pathlib import Path

import pytest


def make_minimal_gguf(path: Path, kv: dict[str, tuple[int, bytes]] | None = None):
    """Write a minimal valid GGUF file.

    kv: optional dict of key -> (vtype, raw_value_bytes).
    vtype 8 = STRING (prefixed with u64 length).
    """
    kv = kv or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))       # version
        f.write(struct.pack("<Q", 0))       # tensor_count
        f.write(struct.pack("<Q", len(kv))) # kv_count
        for key, (vtype, raw_val) in kv.items():
            key_bytes = key.encode("utf-8")
            f.write(struct.pack("<Q", len(key_bytes)))
            f.write(key_bytes)
            f.write(struct.pack("<I", vtype))
            f.write(raw_val)


def gguf_string_value(s: str) -> tuple[int, bytes]:
    """Return (vtype, raw_bytes) for a GGUF STRING value."""
    encoded = s.encode("utf-8")
    return 8, struct.pack("<Q", len(encoded)) + encoded


def gguf_u32_value(n: int) -> tuple[int, bytes]:
    """Return (vtype, raw_bytes) for a GGUF UINT32 value."""
    return 4, struct.pack("<I", n)


@pytest.fixture
def tmp_models(tmp_path):
    """Create a temp dir with one minimal GGUF model."""
    models = tmp_path / "models"
    models.mkdir()
    make_minimal_gguf(models / "test-model.gguf")
    return models


@pytest.fixture
def tmp_models_thinking(tmp_path):
    """Create a temp dir with a GGUF that has enable_thinking in its chat template."""
    models = tmp_path / "models"
    models.mkdir()
    template = "{% if enable_thinking %}think{% endif %}"
    make_minimal_gguf(
        models / "thinker.gguf",
        kv={"tokenizer.chat_template": gguf_string_value(template)},
    )
    return models
