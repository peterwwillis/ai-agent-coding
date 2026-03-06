"""Tests for llama-swap-config-gen.py."""
from __future__ import annotations

import struct
import subprocess
import sys
import textwrap
from io import StringIO
from pathlib import Path

import pytest
from ruamel.yaml import YAML

# Allow importing the script as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import importlib
lcg = importlib.import_module("llama-swap-config-gen")

from tests.conftest import (
    gguf_string_value,
    gguf_u32_value,
    make_minimal_gguf,
)


# ---------------------------------------------------------------------------
# Unit tests: pure functions
# ---------------------------------------------------------------------------

class TestSanitizeLogStem:
    def test_alphanumeric(self):
        assert lcg.sanitize_log_stem("hello123") == "hello123"

    def test_special_chars_replaced(self):
        assert lcg.sanitize_log_stem("my model/v2!") == "my_model_v2"

    def test_empty_string(self):
        assert lcg.sanitize_log_stem("") == "model"

    def test_only_special_chars(self):
        assert lcg.sanitize_log_stem("...") == "model"

    def test_truncation(self):
        long = "a" * 200
        assert len(lcg.sanitize_log_stem(long)) == lcg.LOG_NAME_MAX


class TestParseNGpuLayers:
    def test_integer(self):
        assert lcg.parse_n_gpu_layers("40") == "40"

    def test_auto(self):
        assert lcg.parse_n_gpu_layers("auto") == "auto"

    def test_auto_uppercase(self):
        assert lcg.parse_n_gpu_layers(" Auto ") == "auto"

    def test_invalid(self):
        with pytest.raises(Exception):
            lcg.parse_n_gpu_layers("foo")


class TestParseBatchSetting:
    def test_positive_int(self):
        assert lcg.parse_batch_setting("2048") == "2048"

    def test_auto(self):
        assert lcg.parse_batch_setting("auto") == "auto"

    def test_zero_rejected(self):
        with pytest.raises(Exception):
            lcg.parse_batch_setting("0")

    def test_negative_rejected(self):
        with pytest.raises(Exception):
            lcg.parse_batch_setting("-1")


class TestParseFlashAttn:
    def test_on(self):
        assert lcg.parse_flash_attn("on") == "on"

    def test_off(self):
        assert lcg.parse_flash_attn("off") == "off"

    def test_auto(self):
        assert lcg.parse_flash_attn("auto") == "auto"

    def test_invalid(self):
        with pytest.raises(Exception):
            lcg.parse_flash_attn("maybe")


# ---------------------------------------------------------------------------
# Unit tests: build_cmd
# ---------------------------------------------------------------------------

class TestBuildCmd:
    def _call(self, **overrides):
        defaults = dict(
            llama_server="llama-server",
            model_path=Path("/models/test.gguf"),
            thinking=None,
            ctx_size=2048,
            flash_attn="on",
            cache_type_k="q8_0",
            cache_type_v="q8_0",
            n_gpu_layers=None,
            mmap=True,
            batch_size=None,
            ubatch_size=None,
            log_file=Path("/tmp/test.log"),
        )
        defaults.update(overrides)
        return lcg.build_cmd(**defaults)

    def test_returns_literal_scalar(self):
        from ruamel.yaml.scalarstring import LiteralScalarString
        result = self._call()
        assert isinstance(result, LiteralScalarString)

    def test_multiline_with_backslashes(self):
        result = self._call()
        lines = str(result).rstrip("\n").split("\n")
        # Every line except the last should end with a backslash
        for line in lines[:-1]:
            assert line.rstrip().endswith("\\"), f"Missing backslash: {line!r}"
        # Last line should NOT end with a backslash
        assert not lines[-1].rstrip().endswith("\\")

    def test_contains_port_variable(self):
        result = self._call()
        assert "${PORT}" in str(result)

    def test_contains_model_path(self):
        result = self._call(model_path=Path("/my/model.gguf"))
        assert "/my/model.gguf" in str(result)

    def test_mmap_flag(self):
        assert "--mmap" in str(self._call(mmap=True))
        assert "--no-mmap" in str(self._call(mmap=False))

    def test_batch_flags(self):
        result = self._call(batch_size=4096, ubatch_size=1024)
        assert "--batch-size 4096" in str(result)
        assert "--ubatch-size 1024" in str(result)

    def test_no_batch_flags_when_none(self):
        result = self._call(batch_size=None, ubatch_size=None)
        assert "--batch-size" not in str(result)

    def test_gpu_layers(self):
        result = self._call(n_gpu_layers="33")
        assert "--n-gpu-layers 33" in str(result)

    def test_no_gpu_layers_when_none(self):
        result = self._call(n_gpu_layers=None)
        assert "--n-gpu-layers" not in str(result)

    def test_thinking_false(self):
        result = self._call(thinking=False)
        assert "enable_thinking" in str(result)
        assert "false" in str(result)

    def test_thinking_true(self):
        result = self._call(thinking=True)
        assert "enable_thinking" in str(result)
        assert "true" in str(result)

    def test_thinking_none_no_flag(self):
        result = self._call(thinking=None)
        assert "enable_thinking" not in str(result)

    def test_shell_valid(self):
        """The multiline cmd should be valid as a single shell command."""
        result = self._call(
            batch_size=2048,
            ubatch_size=512,
            n_gpu_layers="40",
            thinking=True,
        )
        # Bash -n checks syntax without executing
        proc = subprocess.run(
            ["bash", "-n", "-c", str(result)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, f"Shell parse error: {proc.stderr}"


# ---------------------------------------------------------------------------
# Unit tests: YAML round-trip preserves comments
# ---------------------------------------------------------------------------

class TestYamlRoundTrip:
    def test_comments_preserved(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(textwrap.dedent("""\
            # Top-level comment
            listen: ":8080"  # inline comment

            # Model section
            models:
              old-model:
                cmd: echo hi  # keep this
        """))

        data = lcg.load_yaml_file(config)
        data["models"]["new-model"] = {"cmd": "echo new"}
        lcg.save_yaml_file(config, data)

        text = config.read_text()
        assert "# Top-level comment" in text
        assert "# inline comment" in text
        assert "# keep this" in text
        assert "# Model section" in text
        assert "new-model" in text

    def test_empty_file_returns_dict(self, tmp_path):
        config = tmp_path / "empty.yaml"
        config.write_text("")
        assert lcg.load_yaml_file(config) == {}

    def test_missing_file_returns_dict(self, tmp_path):
        assert lcg.load_yaml_file(tmp_path / "nope.yaml") == {}


# ---------------------------------------------------------------------------
# Unit tests: GGUF readers
# ---------------------------------------------------------------------------

class TestReadGgufChatTemplate:
    def test_no_template(self, tmp_path):
        p = tmp_path / "no-template.gguf"
        make_minimal_gguf(p)
        assert lcg.read_gguf_chat_template(p) is None

    def test_with_template(self, tmp_path):
        p = tmp_path / "with-template.gguf"
        make_minimal_gguf(p, kv={
            "tokenizer.chat_template": gguf_string_value("hello {{ name }}")
        })
        assert lcg.read_gguf_chat_template(p) == "hello {{ name }}"

    def test_missing_file(self, tmp_path):
        assert lcg.read_gguf_chat_template(tmp_path / "missing.gguf") is None


class TestReadGgufBlockCount:
    def test_no_block_count(self, tmp_path):
        p = tmp_path / "no-blocks.gguf"
        make_minimal_gguf(p)
        assert lcg.read_gguf_block_count(p) is None

    def test_with_block_count(self, tmp_path):
        p = tmp_path / "blocks.gguf"
        make_minimal_gguf(p, kv={
            "llama.block_count": gguf_u32_value(32)
        })
        assert lcg.read_gguf_block_count(p) == 32


class TestTemplateSupportsThinking:
    def test_no_thinking(self, tmp_path):
        p = tmp_path / "plain.gguf"
        make_minimal_gguf(p)
        assert lcg.template_supports_thinking(p) is False

    def test_with_thinking(self, tmp_path):
        p = tmp_path / "think.gguf"
        make_minimal_gguf(p, kv={
            "tokenizer.chat_template": gguf_string_value("{% if enable_thinking %}yes{% endif %}")
        })
        assert lcg.template_supports_thinking(p) is True


# ---------------------------------------------------------------------------
# Integration tests: CLI end-to-end
# ---------------------------------------------------------------------------

class TestCLIStdout:
    """Run the script as a subprocess writing to stdout."""

    def _run(self, tmp_path, models_dir, extra_args=None):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "llama-swap-config-gen.py"),
            "--output", "-",
            "--models-dir", str(models_dir),
        ] + (extra_args or [])
        return subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmp_path))

    def test_basic_output(self, tmp_models):
        proc = self._run(tmp_models.parent, tmp_models)
        assert proc.returncode == 0
        y = YAML()
        data = y.load(proc.stdout)
        assert "models" in data
        assert "test-model" in data["models"]

    def test_cmd_is_block_scalar(self, tmp_models):
        proc = self._run(tmp_models.parent, tmp_models)
        assert proc.returncode == 0
        # YAML literal block scalars start with |
        assert "cmd: |" in proc.stdout or "cmd: |\n" in proc.stdout

    def test_cmd_has_backslashes(self, tmp_models):
        proc = self._run(tmp_models.parent, tmp_models)
        assert proc.returncode == 0
        y = YAML()
        data = y.load(proc.stdout)
        cmd = str(data["models"]["test-model"]["cmd"])
        assert " \\\n" in cmd

    def test_thinking_model_gets_two_entries(self, tmp_models_thinking):
        proc = self._run(tmp_models_thinking.parent, tmp_models_thinking)
        assert proc.returncode == 0
        y = YAML()
        data = y.load(proc.stdout)
        models = data["models"]
        assert "thinker" in models
        assert "thinker-thinking" in models

    def test_no_models_dir_error(self, tmp_path):
        proc = self._run(tmp_path, tmp_path / "nonexistent")
        assert proc.returncode == 2

    def test_no_gguf_files_error(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        proc = self._run(tmp_path, empty)
        assert proc.returncode == 3


class TestCLIFileOutput:
    """Test writing/updating config files."""

    def _run(self, tmp_path, models_dir, output, extra_args=None):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "llama-swap-config-gen.py"),
            "--output", str(output),
            "--models-dir", str(models_dir),
        ] + (extra_args or [])
        return subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmp_path))

    def test_creates_new_file(self, tmp_path, tmp_models):
        out = tmp_path / "new-config.yaml"
        proc = self._run(tmp_path, tmp_models, out)
        assert proc.returncode == 0
        assert out.exists()
        y = YAML()
        data = y.load(out)
        assert "test-model" in data["models"]

    def test_preserves_existing_comments(self, tmp_path, tmp_models):
        out = tmp_path / "existing.yaml"
        out.write_text(textwrap.dedent("""\
            # Important comment
            listen: ":8080"
            models: {}
        """))
        proc = self._run(tmp_path, tmp_models, out)
        assert proc.returncode == 0
        text = out.read_text()
        assert "# Important comment" in text
        assert "test-model" in text

    def test_does_not_duplicate_existing_model(self, tmp_path, tmp_models):
        out = tmp_path / "dup.yaml"
        # First run creates the file
        self._run(tmp_path, tmp_models, out)
        # Second run should not duplicate
        self._run(tmp_path, tmp_models, out)
        y = YAML()
        data = y.load(out)
        count = list(data["models"].keys()).count("test-model")
        assert count == 1

    def test_prune_missing(self, tmp_path, tmp_models):
        out = tmp_path / "prune.yaml"
        out.write_text(textwrap.dedent("""\
            models:
              stale-model:
                cmd: echo old
        """))
        proc = self._run(tmp_path, tmp_models, out, ["--prune-missing"])
        assert proc.returncode == 0
        y = YAML()
        data = y.load(out)
        assert "stale-model" not in data["models"]
        assert "test-model" in data["models"]

    def test_gpu_layers_flag(self, tmp_path, tmp_models):
        out = tmp_path / "gpu.yaml"
        proc = self._run(tmp_path, tmp_models, out, ["--n-gpu-layers", "99"])
        assert proc.returncode == 0
        y = YAML()
        data = y.load(out)
        assert "--n-gpu-layers 99" in str(data["models"]["test-model"]["cmd"])

    def test_batch_flags(self, tmp_path, tmp_models):
        out = tmp_path / "batch.yaml"
        proc = self._run(tmp_path, tmp_models, out, [
            "--batch-size", "4096", "--ubatch-size", "512",
        ])
        assert proc.returncode == 0
        y = YAML()
        data = y.load(out)
        cmd = str(data["models"]["test-model"]["cmd"])
        assert "--batch-size 4096" in cmd
        assert "--ubatch-size 512" in cmd
