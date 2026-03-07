"""Tests for hf-model-settings-scraper.py.

This script scrapes LLM model settings from HuggingFace and writes a YAML file
compatible with the model-settings format used in this project.

----------------------------------------------------------------------
USAGE EXAMPLES
----------------------------------------------------------------------

1. Scrape a single model using the short "owner/model" form:

       bin/hf-model-settings-scraper.py unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF

2. Scrape a single model using a full HuggingFace URL:

       bin/hf-model-settings-scraper.py https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct

3. Scrape multiple models listed in a file and write output to a YAML file:

       # models.txt (one spec per line, lines starting with # are comments):
       #   unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF
       #   https://huggingface.co/microsoft/Phi-4-reasoning-plus

       bin/hf-model-settings-scraper.py -f models.txt -o model-settings.yml

----------------------------------------------------------------------
EXAMPLE OUTPUT  (written to stdout or to the file specified by -o)
----------------------------------------------------------------------

    models:
      hf.co/Qwen/Qwen2.5-Coder-7B-Instruct:
        settings:
          temperature: 0.7
          top_p: 0.8
          top_k: 20
          repetition_penalty: 1.1
          do_sample: true
          max_ctx: 32768
        sources:
          - https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
          - https://huggingface.co/Qwen/Qwen2.5-Coder-7B
          - https://qwen.readthedocs.io/en/latest/
        notes: |
          Here provides a code snippet with `apply_chat_template` to show you
          how to load the tokenizer and model and how to generate contents.

      hf.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF:
        modes:
          thinking:
            temperature: 0.6
            top_p: 0.95
            top_k: 20
            do_sample: true
        sources:
          - https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF
          - https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507
          - https://qwen.readthedocs.io/en/latest/
        notes: |
          The code of Qwen3-MoE has been in the latest Hugging Face `transformers`
          and we advise you to use the latest version of `transformers`.

----------------------------------------------------------------------
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
import textwrap
from io import StringIO
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from ruamel.yaml import YAML

# ---------------------------------------------------------------------------
# Import the scraper as a module (same technique used in the other test file)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
hfss = importlib.import_module("hf-model-settings-scraper")


# ===========================================================================
# Unit tests: parse_model_spec
# ===========================================================================

class TestParseModelSpec:
    """parse_model_spec converts any accepted spec string into (hf_id, output_key)."""

    def test_short_form(self):
        hf_id, key = hfss.parse_model_spec("owner/model")
        assert hf_id == "owner/model"
        assert key == "hf.co/owner/model"

    def test_full_https_url(self):
        hf_id, key = hfss.parse_model_spec("https://huggingface.co/owner/model")
        assert hf_id == "owner/model"
        assert key == "hf.co/owner/model"

    def test_http_url(self):
        hf_id, key = hfss.parse_model_spec("http://huggingface.co/owner/model")
        assert hf_id == "owner/model"

    def test_huggingface_co_prefix(self):
        hf_id, key = hfss.parse_model_spec("huggingface.co/owner/model")
        assert hf_id == "owner/model"

    def test_hf_co_prefix(self):
        hf_id, key = hfss.parse_model_spec("hf.co/owner/model")
        assert hf_id == "owner/model"

    def test_extra_path_segments_are_stripped(self):
        # "owner/model/sub-tree/..." should reduce to owner/model
        hf_id, key = hfss.parse_model_spec("https://huggingface.co/owner/model/tree/main")
        assert hf_id == "owner/model"

    def test_whitespace_stripped(self):
        hf_id, _ = hfss.parse_model_spec("  owner/model  ")
        assert hf_id == "owner/model"

    def test_invalid_spec_raises(self):
        with pytest.raises(ValueError, match="Invalid model spec"):
            hfss.parse_model_spec("justonepart")


# ===========================================================================
# Unit tests: _is_thinking_model
# ===========================================================================

class TestIsThinkingModel:
    """_is_thinking_model returns True when the model name or tags contain
    thinking-related keywords."""

    def test_thinking_in_name(self):
        assert hfss._is_thinking_model("owner/Qwen3-Thinking", []) is True

    def test_reasoning_in_name(self):
        assert hfss._is_thinking_model("owner/phi-4-reasoning-plus", []) is True

    def test_cot_in_name(self):
        assert hfss._is_thinking_model("owner/my-cot-model", []) is True

    def test_thinking_in_tag(self):
        assert hfss._is_thinking_model("owner/plain-model", ["thinking"]) is True

    def test_non_thinking_model(self):
        assert hfss._is_thinking_model("owner/Qwen2.5-Coder-7B-Instruct", []) is False

    def test_empty_tags(self):
        assert hfss._is_thinking_model("owner/Llama-3.2", []) is False


# ===========================================================================
# Unit tests: _extract_sampling_from_gen_config
# ===========================================================================

class TestExtractSamplingFromGenConfig:
    """_extract_sampling_from_gen_config pulls known sampling keys from a
    generation_config.json-style dict, ignoring unrelated keys."""

    # Example: typical generation_config.json returned by HuggingFace API
    _EXAMPLE_GEN_CONFIG = {
        "bos_token_id": 151643,
        "do_sample": True,
        "eos_token_id": [151645, 151643],
        "pad_token_id": 151643,
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.95,
        "transformers_version": "4.51.0",
    }

    def test_extracts_known_keys(self):
        result = hfss._extract_sampling_from_gen_config(self._EXAMPLE_GEN_CONFIG)
        assert result["temperature"] == 0.6
        assert result["top_k"] == 20
        assert result["top_p"] == 0.95
        assert result["do_sample"] is True

    def test_ignores_unknown_keys(self):
        result = hfss._extract_sampling_from_gen_config(self._EXAMPLE_GEN_CONFIG)
        assert "bos_token_id" not in result
        assert "eos_token_id" not in result
        assert "transformers_version" not in result

    def test_empty_config_returns_empty(self):
        assert hfss._extract_sampling_from_gen_config({}) == {}

    def test_all_sampling_keys(self):
        cfg = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "min_p": 0.05,
            "repetition_penalty": 1.1,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.1,
            "do_sample": True,
        }
        result = hfss._extract_sampling_from_gen_config(cfg)
        assert result == cfg


# ===========================================================================
# Unit tests: _extract_context_from_config
# ===========================================================================

class TestExtractContextFromConfig:
    """_extract_context_from_config reads the context-length integer from a
    config.json-style dict."""

    def test_max_position_embeddings(self):
        assert hfss._extract_context_from_config({"max_position_embeddings": 32768}) == 32768

    def test_n_ctx(self):
        assert hfss._extract_context_from_config({"n_ctx": 4096}) == 4096

    def test_rope_scaling_fallback(self):
        cfg = {
            "rope_scaling": {
                "original_max_position_embeddings": 8192,
                "factor": 4.0,
            }
        }
        assert hfss._extract_context_from_config(cfg) == 32768

    def test_empty_config_returns_none(self):
        assert hfss._extract_context_from_config({}) is None

    def test_priority_order(self):
        # max_position_embeddings is first in _CONTEXT_KEYS so it wins
        cfg = {
            "max_position_embeddings": 8192,
            "max_length": 2048,
        }
        assert hfss._extract_context_from_config(cfg) == 8192


# ===========================================================================
# Unit tests: _extract_sampling_from_card
# ===========================================================================

class TestExtractSamplingFromCard:
    """_extract_sampling_from_card finds sampling parameters inside fenced
    code blocks in a model card README."""

    # Minimal model card that contains a typical YAML-style recommendation block
    _CARD_YAML_STYLE = textwrap.dedent("""\
        # My Model

        ## Recommended settings

        Use these parameters:

        ```yaml
        temperature: 0.7
        top_p: 0.8
        top_k: 20
        repetition_penalty: 1.1
        do_sample: true
        max_new_tokens: 4096
        ```
    """)

    # Model card with Python kwargs style
    _CARD_PY_STYLE = textwrap.dedent("""\
        # My Model

        ## Usage

        ```python
        output = model.generate(
            input_ids,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            do_sample=True,
            max_new_tokens=8192,
        )
        ```
    """)

    # Model card with JSON style
    _CARD_JSON_STYLE = textwrap.dedent("""\
        # My Model

        ## Parameters

        ```json
        {
          "temperature": 0.6,
          "top_p": 0.95,
          "top_k": 20
        }
        ```
    """)

    def test_yaml_style_code_block(self):
        result = hfss._extract_sampling_from_card(self._CARD_YAML_STYLE)
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.8
        assert result["top_k"] == 20
        assert result["repetition_penalty"] == 1.1
        # max_new_tokens=4096 should be captured as suggested_out_t
        assert result.get("suggested_out_t") == 4096

    def test_python_kwargs_style(self):
        result = hfss._extract_sampling_from_card(self._CARD_PY_STYLE)
        assert result["temperature"] == 0.8
        assert result["top_p"] == 0.95
        # max_new_tokens=8192 → suggested_out_t
        assert result.get("suggested_out_t") == 8192

    def test_json_style_code_block(self):
        result = hfss._extract_sampling_from_card(self._CARD_JSON_STYLE)
        assert result["temperature"] == 0.6
        assert result["top_k"] == 20

    def test_small_max_new_tokens_ignored(self):
        """max_new_tokens below 1024 (trivial code examples) must not be recorded."""
        card = textwrap.dedent("""\
            ## Usage
            ```python
            output = model.generate(input_ids, temperature=0.7, max_new_tokens=200)
            ```
        """)
        result = hfss._extract_sampling_from_card(card)
        assert "suggested_out_t" not in result

    def test_no_code_blocks_returns_empty(self):
        assert hfss._extract_sampling_from_card("No code blocks here.") == {}

    def test_top_k_preserved_as_int(self):
        card = textwrap.dedent("""\
            ## Params
            ```yaml
            temperature: 0.5
            top_k: 40
            ```
        """)
        result = hfss._extract_sampling_from_card(card)
        assert isinstance(result["top_k"], int)
        assert result["top_k"] == 40


# ===========================================================================
# Unit tests: _extract_sources_from_card
# ===========================================================================

class TestExtractSourcesFromCard:
    """_extract_sources_from_card collects meaningful documentation/model links
    from a model card, filtering out asset URLs, datasets, and org pages."""

    _PRIMARY = "https://huggingface.co/owner/model"

    def test_primary_url_always_first(self):
        sources = hfss._extract_sources_from_card("No links here.", self._PRIMARY)
        assert sources[0] == self._PRIMARY

    def test_picks_up_hf_model_url(self):
        card = "See the base model at https://huggingface.co/Qwen/Qwen3-30B for details."
        sources = hfss._extract_sources_from_card(card, self._PRIMARY)
        assert "https://huggingface.co/Qwen/Qwen3-30B" in sources

    def test_picks_up_readthedocs(self):
        card = "Docs: https://qwen.readthedocs.io/en/latest/"
        sources = hfss._extract_sources_from_card(card, self._PRIMARY)
        assert "https://qwen.readthedocs.io/en/latest/" in sources

    def test_picks_up_unsloth_docs(self):
        card = "Guide: https://unsloth.ai/docs/models/qwen3.5"
        sources = hfss._extract_sources_from_card(card, self._PRIMARY)
        assert "https://unsloth.ai/docs/models/qwen3.5" in sources

    def test_skips_blob_resolve_urls(self):
        card = "License: https://huggingface.co/owner/model/blob/main/LICENSE"
        sources = hfss._extract_sources_from_card(card, self._PRIMARY)
        assert all("/blob/" not in s for s in sources)

    def test_skips_dataset_urls(self):
        card = "Eval: https://huggingface.co/datasets/ai2_arc"
        sources = hfss._extract_sources_from_card(card, self._PRIMARY)
        assert all("datasets" not in s for s in sources[1:])

    def test_skips_papers_urls(self):
        card = "Paper: https://huggingface.co/papers/2504.21318"
        sources = hfss._extract_sources_from_card(card, self._PRIMARY)
        assert all("papers" not in s for s in sources[1:])

    def test_skips_fragment_anchor_urls(self):
        card = "Ref: https://huggingface.co/owner/model#usage"
        sources = hfss._extract_sources_from_card(card, self._PRIMARY)
        # Fragment URLs should be skipped
        assert all("#" not in s for s in sources)

    def test_skips_org_only_urls(self):
        card = "Org: https://huggingface.co/nvidia"
        sources = hfss._extract_sources_from_card(card, self._PRIMARY)
        assert all(s == self._PRIMARY or "/" in s.split("//", 1)[-1].split("/", 1)[-1]
                   for s in sources)

    def test_skips_image_urls(self):
        card = "Logo: https://huggingface.co/owner/model/logo.png"
        sources = hfss._extract_sources_from_card(card, self._PRIMARY)
        assert all(not s.endswith(".png") for s in sources)

    def test_caps_at_six_sources(self):
        # Fill the card with many valid model URLs
        links = " ".join(
            f"https://huggingface.co/owner/model{i}" for i in range(20)
        )
        sources = hfss._extract_sources_from_card(links, self._PRIMARY)
        assert len(sources) <= 6

    def test_no_duplicates(self):
        card = "https://huggingface.co/Qwen/Qwen3 https://huggingface.co/Qwen/Qwen3"
        sources = hfss._extract_sources_from_card(card, self._PRIMARY)
        assert len(sources) == len(set(sources))


# ===========================================================================
# Unit tests: _clean_note_text / _extract_notes_from_card
# ===========================================================================

class TestCleanNoteText:
    """_clean_note_text strips markdown blockquote markers and GH callout tags."""

    def test_plain_text_unchanged(self):
        assert hfss._clean_note_text("Use temperature=0.7 for best results.") == \
            "Use temperature=0.7 for best results."

    def test_single_blockquote_stripped(self):
        assert hfss._clean_note_text("> Some advice.") == "Some advice."

    def test_nested_blockquote_stripped(self):
        assert hfss._clean_note_text("> > Deep quote.") == "Deep quote."

    def test_gfm_important_callout_stripped(self):
        assert hfss._clean_note_text("> [!IMPORTANT] Do this.") == "Do this."

    def test_gfm_note_callout_stripped(self):
        assert hfss._clean_note_text("> [!NOTE] Remember this.") == "Remember this."

    def test_empty_line_returns_empty(self):
        assert hfss._clean_note_text("") == ""


class TestExtractNotesFromCard:
    """_extract_notes_from_card finds and cleans the first useful paragraph in
    a recommendation/usage/settings section of the model card."""

    _CARD_WITH_NOTE = textwrap.dedent("""\
        # My Model

        ## Recommended settings

        > [!IMPORTANT]
        > Use temperature=0.8 for best results.
        > For longer outputs set max_new_tokens=32768.

        ## Other section

        Unrelated text.
    """)

    _CARD_WITH_PLAIN_NOTE = textwrap.dedent("""\
        # My Model

        ## Usage

        Here is how to use the model with the apply_chat_template helper.

        ```python
        model.generate(...)
        ```
    """)

    def test_extracts_note_from_recommendation_section(self):
        note = hfss._extract_notes_from_card(self._CARD_WITH_NOTE)
        assert "temperature=0.8" in note
        # Blockquote markers must be stripped
        assert ">" not in note
        assert "[!IMPORTANT]" not in note

    def test_plain_paragraph_before_code_block(self):
        note = hfss._extract_notes_from_card(self._CARD_WITH_PLAIN_NOTE)
        assert "apply_chat_template" in note
        # Code block content must not appear in the note
        assert "model.generate" not in note

    def test_no_relevant_section_returns_empty(self):
        card = "# Model\n\n## Introduction\n\nSome text.\n"
        assert hfss._extract_notes_from_card(card) == ""


# ===========================================================================
# Unit tests: _build_yaml_entry (full assembly from ScrapedModel)
# ===========================================================================

class TestBuildYamlEntry:
    """_build_yaml_entry turns a ScrapedModel dataclass into the dict that is
    written as a YAML entry.  This is the central assembly step."""

    # -----------------------------------------------------------------------
    # Helper factories
    # -----------------------------------------------------------------------
    @staticmethod
    def _make_model(**kwargs):
        """Return a ScrapedModel with sensible defaults, overridden by kwargs."""
        defaults = dict(
            hf_id="owner/my-model",
            output_key="hf.co/owner/my-model",
            api_meta={},
            generation_config={},
            model_config={},
            model_card="",
            base_hf_id=None,
            base_generation_config={},
        )
        defaults.update(kwargs)
        return hfss.ScrapedModel(**defaults)

    # -----------------------------------------------------------------------

    def test_standard_model_uses_settings_key(self):
        model = self._make_model(
            generation_config={"temperature": 0.7, "top_p": 0.9, "do_sample": True},
        )
        entry = hfss._build_yaml_entry(model)
        assert "settings" in entry
        assert "modes" not in entry
        assert entry["settings"]["temperature"] == 0.7

    def test_thinking_model_uses_modes_key(self):
        model = self._make_model(
            hf_id="owner/my-thinking-model",
            output_key="hf.co/owner/my-thinking-model",
            generation_config={"temperature": 0.6, "top_k": 20},
        )
        entry = hfss._build_yaml_entry(model)
        assert "modes" in entry
        assert "thinking" in entry["modes"]
        assert entry["modes"]["thinking"]["temperature"] == 0.6

    def test_sources_always_present(self):
        model = self._make_model()
        entry = hfss._build_yaml_entry(model)
        assert "sources" in entry
        assert entry["sources"][0] == "https://huggingface.co/owner/my-model"

    def test_max_ctx_included_in_settings(self):
        model = self._make_model(
            generation_config={"temperature": 0.7},
            model_config={"max_position_embeddings": 32768},
        )
        entry = hfss._build_yaml_entry(model)
        assert entry["settings"]["max_ctx"] == 32768

    def test_base_model_url_inserted_in_sources(self):
        model = self._make_model(
            base_hf_id="owner/base-model",
            model_card="",
        )
        entry = hfss._build_yaml_entry(model)
        assert "https://huggingface.co/owner/base-model" in entry["sources"]

    def test_no_settings_and_thinking_model_emits_empty_thinking_dict(self):
        model = self._make_model(
            hf_id="owner/my-reasoning-model",
            output_key="hf.co/owner/my-reasoning-model",
        )
        entry = hfss._build_yaml_entry(model)
        assert entry["modes"] == {"thinking": {}}

    def test_notes_included_when_card_has_relevant_section(self):
        card = textwrap.dedent("""\
            # Model

            ## Recommended settings

            Use temperature=0.8 for best results.
        """)
        model = self._make_model(model_card=card)
        entry = hfss._build_yaml_entry(model)
        assert "notes" in entry
        assert "temperature=0.8" in entry["notes"]

    def test_no_notes_key_when_card_has_no_relevant_section(self):
        card = "# Model\n\n## Introduction\n\nSome text.\n"
        model = self._make_model(model_card=card)
        entry = hfss._build_yaml_entry(model)
        assert "notes" not in entry


# ===========================================================================
# Integration test: scrape_models end-to-end with mocked network
#
# This shows the full pipeline:
#   input  → list of model spec strings (URLs or "owner/model")
#   output → YAML file on disk (or stdout)
# ===========================================================================

class TestScrapeModelsIntegration:
    """End-to-end test that mocks all network calls and verifies the complete
    pipeline from spec string → YAML dict → written file.

    Input:
        specs = ["Qwen/Qwen2.5-Coder-7B-Instruct",
                 "https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF"]

    Output (written by _make_yaml().dump):
        models:
          hf.co/Qwen/Qwen2.5-Coder-7B-Instruct:
            settings:
              temperature: 0.7
              top_p: 0.8
              top_k: 20
              do_sample: true
              max_ctx: 32768
            sources:
              - https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
          hf.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF:
            modes:
              thinking:
                temperature: 0.6
                top_k: 20
                top_p: 0.95
                do_sample: true
            sources:
              - https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF
              - https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507
    """

    # ------------------------------------------------------------------
    # Synthetic data returned by the mocked network calls
    # ------------------------------------------------------------------

    # generation_config.json for Qwen2.5-Coder-7B-Instruct
    _CODER_GEN_CFG: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "do_sample": True,
    }

    # config.json for Qwen2.5-Coder-7B-Instruct (supplies context length)
    _CODER_CFG: Dict[str, Any] = {
        "max_position_embeddings": 32768,
        "model_type": "qwen2",
    }

    # HF API metadata for Qwen2.5-Coder-7B-Instruct
    _CODER_API: Dict[str, Any] = {
        "id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "tags": ["text-generation"],
    }

    # generation_config.json for the GGUF (absent → None)
    # The base model generation_config.json (Qwen3-30B-A3B-Thinking-2507)
    _THINKING_BASE_GEN_CFG: Dict[str, Any] = {
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.95,
        "do_sample": True,
    }

    # HF API metadata for the GGUF
    _GGUF_API: Dict[str, Any] = {
        "id": "unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF",
        "tags": ["text-generation"],
        "cardData": {
            "base_model": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        },
    }

    # ------------------------------------------------------------------

    def _make_fetch_json(self):
        """Return a side_effect function that maps known URLs to synthetic data."""
        api_base = hfss.HF_API_BASE
        hf_base = hfss.HF_BASE_URL

        url_map = {
            f"{api_base}/Qwen/Qwen2.5-Coder-7B-Instruct": self._CODER_API,
            f"{hf_base}/Qwen/Qwen2.5-Coder-7B-Instruct/resolve/main/generation_config.json": self._CODER_GEN_CFG,
            f"{hf_base}/Qwen/Qwen2.5-Coder-7B-Instruct/resolve/main/config.json": self._CODER_CFG,
            f"{api_base}/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF": self._GGUF_API,
            # GGUF itself has no generation_config; returns None
            f"{hf_base}/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF/resolve/main/generation_config.json": None,
            f"{hf_base}/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF/resolve/main/config.json": None,
            # Base model supplies the generation config
            f"{hf_base}/Qwen/Qwen3-30B-A3B-Thinking-2507/resolve/main/generation_config.json": self._THINKING_BASE_GEN_CFG,
        }
        return lambda url: url_map.get(url)

    def test_end_to_end_yaml_structure(self, tmp_path):
        """Full pipeline: specs → mocked network → YAML dict with correct structure."""
        specs = [
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF",
        ]

        with patch.object(hfss, "_fetch_json", side_effect=self._make_fetch_json()), \
             patch.object(hfss, "_fetch_text", return_value=None):
            result = hfss.scrape_models(specs)

        assert "models" in result
        models = result["models"]

        # ---- Standard (non-thinking) model ----
        coder_key = "hf.co/Qwen/Qwen2.5-Coder-7B-Instruct"
        assert coder_key in models
        coder = models[coder_key]
        assert "settings" in coder
        assert coder["settings"]["temperature"] == 0.7
        assert coder["settings"]["max_ctx"] == 32768
        assert coder["sources"][0] == "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct"

        # ---- Thinking model (GGUF with base model fallback) ----
        thinking_key = "hf.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF"
        assert thinking_key in models
        thinking = models[thinking_key]
        assert "modes" in thinking
        assert "thinking" in thinking["modes"]
        assert thinking["modes"]["thinking"]["temperature"] == 0.6
        # Base model page should appear in sources
        assert "https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507" in thinking["sources"]

    def test_end_to_end_yaml_file_output(self, tmp_path):
        """Verify that the YAML output written to a file is valid and readable."""
        specs = ["Qwen/Qwen2.5-Coder-7B-Instruct"]
        out_file = tmp_path / "model-settings.yml"

        with patch.object(hfss, "_fetch_json", side_effect=self._make_fetch_json()), \
             patch.object(hfss, "_fetch_text", return_value=None):
            result = hfss.scrape_models(specs)

        yaml = hfss._make_yaml()
        with open(out_file, "w", encoding="utf-8") as fh:
            yaml.dump(result, fh)

        # File must exist and be readable YAML
        assert out_file.exists()
        yaml2 = YAML()
        data = yaml2.load(out_file)
        assert "models" in data
        assert "hf.co/Qwen/Qwen2.5-Coder-7B-Instruct" in data["models"]


# ===========================================================================
# CLI integration test: subprocess invocation
# ===========================================================================

class TestCLI:
    """Run the script as a subprocess to verify end-to-end CLI behaviour."""

    _SCRIPT = str(Path(__file__).resolve().parent.parent / "hf-model-settings-scraper.py")

    def test_help(self):
        proc = subprocess.run(
            [sys.executable, self._SCRIPT, "--help"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0
        assert "owner/model" in proc.stdout or "owner/model" in proc.stderr

    def test_no_args_shows_help_and_nonzero_exit(self):
        proc = subprocess.run(
            [sys.executable, self._SCRIPT],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0

    def test_invalid_spec_skipped_gracefully(self):
        """A single-token spec (no slash) must not crash the script."""
        proc = subprocess.run(
            [sys.executable, self._SCRIPT, "notavalidspec"],
            capture_output=True,
            text=True,
        )
        # Script exits 0 (errors are warnings); invalid spec is skipped
        assert proc.returncode == 0
        assert "[error]" in proc.stderr
