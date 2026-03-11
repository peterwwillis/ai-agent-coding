#!/usr/bin/env python3
"""Scrape LLM model settings from Hugging Face and output as YAML.

Given a list of HuggingFace model identifiers, fetches generation config,
model config, and model card data, then outputs a YAML file compatible
with the model-settings format used in this project.

Usage examples:
  hf-model-settings-scraper.py unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF
  hf-model-settings-scraper.py https://huggingface.co/microsoft/Phi-4-reasoning-plus
  hf-model-settings-scraper.py -f models.txt -o model-settings.yml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

HF_BASE_URL = "https://huggingface.co"
HF_API_BASE = "https://huggingface.co/api/models"
REQUEST_TIMEOUT = 30
USER_AGENT = "hf-model-settings-scraper/1.0"
REQUEST_DELAY = 0.5  # seconds between requests to avoid rate limiting

# Keywords indicating a thinking/reasoning model (matched against model name and tags)
_THINKING_RE = re.compile(r"\b(think(?:ing)?|reason(?:ing)?|cot)\b", re.IGNORECASE)

# Keys to extract from generation_config.json
_SAMPLING_KEYS = [
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "do_sample",
]

# Keys to look for context window size in config.json
_CONTEXT_KEYS = [
    "max_position_embeddings",
    "max_seq_len",
    "n_ctx",
    "max_length",
    "n_positions",
]

# Param names we care about when scanning model card text
_CARD_PARAM_NAMES = frozenset(
    [
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "max_tokens",
        "max_new_tokens",
    ]
)

# Patterns for key: value in YAML/config style (inside code blocks)
_KV_YAML_RE = re.compile(
    r"""^[ \t]*["']?(\w+)["']?[ \t]*:[ \t]*([\d.]+)[ \t]*$""",
    re.MULTILINE,
)
# Patterns for key=value in Python kwargs style (inside code blocks)
_KV_PY_RE = re.compile(
    r"""\b(\w+)\s*=\s*([\d.]+)""",
)
# Patterns for "key": value in JSON style (inside code blocks)
_KV_JSON_RE = re.compile(
    r""""(\w+)"\s*:\s*([\d.]+)""",
)


@dataclass
class ScrapedModel:
    """Holds all scraped information for a single model."""

    hf_id: str  # e.g. "unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF"
    output_key: str  # e.g. "hf.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF"
    api_meta: Dict[str, Any] = field(default_factory=dict)
    generation_config: Dict[str, Any] = field(default_factory=dict)
    model_config: Dict[str, Any] = field(default_factory=dict)
    model_card: str = ""
    base_hf_id: Optional[str] = None  # base/source model ID if this is a derived model
    base_generation_config: Dict[str, Any] = field(default_factory=dict)


def parse_model_spec(spec: str) -> Tuple[str, str]:
    """Parse a model spec string into (hf_id, output_key).

    Accepts:
        - https://huggingface.co/owner/model[/...]
        - huggingface.co/owner/model[/...]
        - hf.co/owner/model[/...]
        - owner/model

    Returns:
        (hf_id, output_key) where hf_id is "owner/model" and
        output_key is "hf.co/owner/model".
    """
    spec = spec.strip()

    for prefix in (
        "https://huggingface.co/",
        "http://huggingface.co/",
        "huggingface.co/",
        "hf.co/",
    ):
        if spec.startswith(prefix):
            spec = spec[len(prefix) :]
            break

    parts = spec.split("/")
    if len(parts) < 2:
        raise ValueError(
            f"Invalid model spec {spec!r}: expected 'owner/model' format"
        )

    hf_id = f"{parts[0]}/{parts[1]}"
    output_key = f"hf.co/{hf_id}"
    return hf_id, output_key


def _fetch_raw(url: str) -> Optional[bytes]:
    """Fetch raw bytes from a URL; returns None on any error."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code not in (404, 403):
            print(f"  [warn] HTTP {e.code} fetching {url}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  [warn] Could not fetch {url}: {e}", file=sys.stderr)
        return None


def _fetch_json(url: str) -> Optional[Dict[str, Any]]:
    """Fetch and parse JSON from a URL; returns None on error."""
    time.sleep(REQUEST_DELAY)
    data = _fetch_raw(url)
    if data is None:
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        print(f"  [warn] Could not parse JSON from {url}: {e}", file=sys.stderr)
        return None


def _fetch_text(url: str) -> Optional[str]:
    """Fetch text content from a URL; returns None on error."""
    time.sleep(REQUEST_DELAY)
    data = _fetch_raw(url)
    if data is None:
        return None
    return data.decode("utf-8", errors="replace")


def _resolve_url(hf_id: str, filename: str) -> str:
    """Build a raw-file URL for a HuggingFace repo."""
    return f"{HF_BASE_URL}/{hf_id}/resolve/main/{filename}"


def _is_thinking_model(hf_id: str, tags: List[str]) -> bool:
    """Return True if name or tags suggest this is a thinking/reasoning model."""
    if _THINKING_RE.search(hf_id):
        return True
    for tag in tags:
        if _THINKING_RE.search(tag):
            return True
    return False


def _extract_sampling_from_gen_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Pull sampling parameters from a generation_config.json dict."""
    out: Dict[str, Any] = {}
    for key in _SAMPLING_KEYS:
        val = cfg.get(key)
        if val is not None:
            out[key] = val
    return out


def _extract_context_from_config(cfg: Dict[str, Any]) -> Optional[int]:
    """Extract the maximum context length from a config.json dict."""
    for key in _CONTEXT_KEYS:
        val = cfg.get(key)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass

    # Some models define context via rope_scaling
    rope = cfg.get("rope_scaling") or {}
    original = rope.get("original_max_position_embeddings")
    factor = rope.get("factor")
    if original and factor:
        try:
            return int(float(original) * float(factor))
        except (ValueError, TypeError):
            pass

    return None


def _parse_code_blocks(text: str) -> List[str]:
    """Extract fenced code block contents from markdown text."""
    blocks: List[str] = []
    for m in re.finditer(r"```[^\n]*\n(.*?)```", text, re.DOTALL):
        blocks.append(m.group(1))
    return blocks


def _extract_params_from_text(text: str) -> Dict[str, float]:
    """Scan text for sampling parameter key/value pairs.

    Returns a dict of parameter names → float values, keeping only
    recognised sampling parameters.
    """
    found: Dict[str, float] = {}
    for pattern in (_KV_YAML_RE, _KV_PY_RE, _KV_JSON_RE):
        for m in pattern.finditer(text):
            name = m.group(1).lower()
            if name in _CARD_PARAM_NAMES:
                try:
                    found[name] = float(m.group(2))
                except ValueError:
                    pass
    return found


def _extract_sampling_from_card(model_card: str) -> Dict[str, Any]:
    """Try to extract sampling parameters from model card code blocks.

    Prefers code blocks that contain at least 'temperature' or 'top_p'.
    Returns a dict of sampling parameters (may be empty).
    """
    blocks = _parse_code_blocks(model_card)

    best: Dict[str, float] = {}
    best_score = 0

    for block in blocks:
        params = _extract_params_from_text(block)
        # Score: prefer blocks with more recognised params and at least temp/top_p
        score = len(params)
        if "temperature" in params or "top_p" in params:
            score += 10
        if score > best_score:
            best = params
            best_score = score

    if best_score == 0:
        return {}

    # Map max_tokens / max_new_tokens to suggested_out_t
    # Only record if large enough to be a real recommendation (not a trivial example)
    _MIN_SUGGESTED_OUT_T = 1024
    out: Dict[str, Any] = {}
    for key in _SAMPLING_KEYS:
        if key in best:
            val = best[key]
            # Preserve integer appearance for top_k and similar
            if key in ("top_k",) and val == int(val):
                out[key] = int(val)
            else:
                out[key] = val

    for tok_key in ("max_tokens", "max_new_tokens"):
        if tok_key in best and "suggested_out_t" not in out:
            val = int(best[tok_key])
            if val >= _MIN_SUGGESTED_OUT_T:
                out["suggested_out_t"] = val

    return out


def _parse_front_matter(model_card: str) -> Dict[str, Any]:
    """Parse YAML front matter (between --- delimiters) from a model card."""
    m = re.match(r"^---\s*\n(.*?)\n---", model_card, re.DOTALL)
    if not m:
        return {}
    try:
        yaml = YAML()
        result = yaml.load(m.group(1))
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


def _extract_base_model(front_matter: Dict[str, Any]) -> Optional[str]:
    """Return the base_model HF id from model card front matter if present."""
    bm = front_matter.get("base_model")
    if not bm:
        return None
    if isinstance(bm, list):
        bm = bm[0] if bm else None
    if isinstance(bm, str) and "/" in bm:
        parts = bm.replace("https://huggingface.co/", "").split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    return None


def _extract_sources_from_card(
    model_card: str, primary_url: str
) -> List[str]:
    """Extract HuggingFace and documentation links from the model card body.

    Keeps only meaningful model or documentation pages; skips CDN/image/file/
    dataset/collection/paper sub-paths that clutter the sources list.
    """
    seen = {primary_url}
    sources = [primary_url]

    # Patterns that indicate a URL is a raw file, image, or non-doc asset
    _SKIP_PATH_RE = re.compile(
        r"/(?:resolve|blob|tree|raw)/|"
        r"[#]|"  # strip URLs with fragment anchors
        r"\.(?:png|jpg|jpeg|gif|svg|webp|ico|pdf|zip|tar|gz|gguf|bin|safetensors)(?:[?#]|$)",
        re.IGNORECASE,
    )

    # Keywords that are never valid as the first or second path segment on huggingface.co
    _HF_SKIP_SEGMENTS = frozenset(
        ["datasets", "collections", "papers", "spaces", "organizations", "joins", "join"]
    )

    # Documentation host patterns (exact match)
    _DOC_HOSTS = re.compile(
        r"^(?:huggingface\.co|hf\.co|"
        r"qwen\.readthedocs\.io|"
        r"docs\.mistral\.ai|"
        r"unsloth\.ai|"
        r"[a-z0-9-]+\.readthedocs\.io|"
        r"platform\.openai\.com)$"
    )

    _HF_HOSTS = re.compile(r"^(?:huggingface\.co|hf\.co)$")

    for url in re.findall(
        r"https?://[^\s\"'<>)\]]+",
        model_card,
    ):
        url = url.rstrip(".,;:)")
        # Parse host and path
        m = re.match(r"https?://([^/]+)(.*)", url)
        if not m:
            continue
        host, path = m.group(1).lower(), m.group(2)
        if not _DOC_HOSTS.match(host):
            continue
        if _SKIP_PATH_RE.search(path):
            continue

        # For HuggingFace, enforce that path has exactly owner/model (2 segments)
        # or is under /docs/. Single-segment paths are organization pages; skip them.
        if _HF_HOSTS.match(host):
            segments = [s for s in path.strip("/").split("/") if s]
            if len(segments) < 2:
                continue  # org-only page like huggingface.co/nvidia
            # Skip if either the first or second segment is a non-model keyword
            if segments[0].lower() in _HF_SKIP_SEGMENTS:
                continue  # e.g. huggingface.co/datasets/foo, huggingface.co/papers/123
            if segments[1].lower() in _HF_SKIP_SEGMENTS:
                continue  # e.g. huggingface.co/hotpot_qa/datasets
            if len(segments) > 2 and segments[0] != "docs":
                continue  # too deep and not a /docs/ path

        if url not in seen:
            seen.add(url)
            sources.append(url)
        if len(sources) >= 6:
            break

    return sources


def _clean_note_text(text: str) -> str:
    """Strip markdown blockquote and GitHub-flavored callout syntax from a single line."""
    # Remove leading `> ` blockquote markers (possibly multiple levels)
    text = re.sub(r"^(\s*>\s*)+", "", text)
    # Remove GitHub-flavored callout tags like [!IMPORTANT], [!NOTE], [!TIP]
    text = re.sub(r"\[!\w+\]\s*", "", text)
    return text.strip()


def _extract_notes_from_card(model_card: str) -> str:
    """Extract the first meaningful recommendation paragraph from the card."""
    lines: List[str] = []

    # Look for sections whose heading mentions usage / recommendation / params
    section_re = re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE)
    headings = list(section_re.finditer(model_card))

    for i, h in enumerate(headings):
        title = h.group(1).strip()
        if not re.search(
            r"(recommend|suggest|best|param|setting|usage|note|tip|quickstart)",
            title,
            re.IGNORECASE,
        ):
            continue

        start = h.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(model_card)
        section_text = model_card[start:end].strip()

        # Take first non-empty, non-code paragraph; clean blockquotes per line
        para_lines: List[str] = []
        for line in section_text.split("\n"):
            cleaned = _clean_note_text(line)
            if cleaned.startswith("```"):
                break
            if cleaned:
                para_lines.append(cleaned)
            elif para_lines:
                break

        if para_lines:
            lines = para_lines
            break

    return " ".join(lines) if lines else ""


def _fetch_model_data(hf_id: str) -> ScrapedModel:
    """Fetch all available data for a single HuggingFace model."""
    _owner, _name = hf_id.split("/", 1)
    output_key = f"hf.co/{hf_id}"
    model = ScrapedModel(hf_id=hf_id, output_key=output_key)

    print(f"  Fetching API metadata …", file=sys.stderr)
    api_meta = _fetch_json(f"{HF_API_BASE}/{hf_id}")
    if api_meta:
        model.api_meta = api_meta

    print(f"  Fetching generation_config.json …", file=sys.stderr)
    gen_cfg = _fetch_json(_resolve_url(hf_id, "generation_config.json"))
    if gen_cfg:
        model.generation_config = gen_cfg

    print(f"  Fetching config.json …", file=sys.stderr)
    m_cfg = _fetch_json(_resolve_url(hf_id, "config.json"))
    if m_cfg:
        model.model_config = m_cfg

    print(f"  Fetching README.md …", file=sys.stderr)
    card = _fetch_text(_resolve_url(hf_id, "README.md"))
    if card:
        model.model_card = card

    # Determine base model from front matter or API metadata
    base_id: Optional[str] = None
    if model.model_card:
        fm = _parse_front_matter(model.model_card)
        base_id = _extract_base_model(fm)

    if not base_id and api_meta:
        card_data = api_meta.get("cardData") or {}
        base_id = _extract_base_model(card_data)

    # If we have no sampling params yet, try the base model
    if base_id and base_id != hf_id:
        model.base_hf_id = base_id
        if not _extract_sampling_from_gen_config(model.generation_config):
            print(
                f"  No generation_config here; trying base model {base_id} …",
                file=sys.stderr,
            )
            base_gen = _fetch_json(_resolve_url(base_id, "generation_config.json"))
            if base_gen:
                model.base_generation_config = base_gen

    return model


def _build_yaml_entry(model: ScrapedModel) -> Dict[str, Any]:
    """Convert a ScrapedModel into the YAML dict entry."""
    tags: List[str] = model.api_meta.get("tags") or []
    thinking = _is_thinking_model(model.hf_id, tags)

    # --- Sampling parameters ---
    # Priority: generation_config.json > base generation_config > model card code blocks
    sampling = _extract_sampling_from_gen_config(model.generation_config)
    if not sampling and model.base_generation_config:
        sampling = _extract_sampling_from_gen_config(model.base_generation_config)
    if not sampling and model.model_card:
        sampling = _extract_sampling_from_card(model.model_card)

    # --- Context length ---
    max_ctx: Optional[int] = None
    if model.model_config:
        max_ctx = _extract_context_from_config(model.model_config)

    # Add max_ctx to sampling params if found
    if max_ctx:
        sampling["max_ctx"] = max_ctx

    # --- Sources ---
    hf_page = f"{HF_BASE_URL}/{model.hf_id}"
    if model.model_card:
        sources = _extract_sources_from_card(model.model_card, hf_page)
    else:
        sources = [hf_page]

    # Include the base model page as a source if it was used
    if model.base_hf_id:
        base_url = f"{HF_BASE_URL}/{model.base_hf_id}"
        if base_url not in sources:
            sources.insert(1, base_url)

    # --- Notes ---
    notes: Optional[str] = None
    if model.model_card:
        note_text = _extract_notes_from_card(model.model_card)
        if note_text:
            notes = note_text

    # --- Assemble the entry ---
    entry: Dict[str, Any] = {}

    if sampling:
        if thinking:
            entry["modes"] = {"thinking": sampling}
        else:
            entry["settings"] = sampling
    elif thinking:
        # No settings found but we know it's a thinking model; note that
        entry["modes"] = {"thinking": {}}

    entry["sources"] = sources

    if notes:
        entry["notes"] = LiteralScalarString(notes + "\n")

    return entry


def scrape_models(specs: List[str]) -> Dict[str, Any]:
    """Scrape a list of model specs and return the assembled YAML dict.

    Returns a dict of the form ``{"models": {"hf.co/owner/model": {...}, ...}}``.
    """
    models: Dict[str, Any] = {}

    for spec in specs:
        print(f"\nScraping {spec} …", file=sys.stderr)
        try:
            hf_id, output_key = parse_model_spec(spec)
        except ValueError as e:
            print(f"  [error] {e}", file=sys.stderr)
            continue

        scraped = _fetch_model_data(hf_id)
        entry = _build_yaml_entry(scraped)
        models[output_key] = entry
        print(f"  → {output_key}: {list(entry.keys())}", file=sys.stderr)

    return {"models": models}


def _make_yaml() -> YAML:
    """Return a ruamel.yaml YAML instance configured for model-settings style."""
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 120
    return yaml


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape LLM model settings from Hugging Face and output a YAML file "
            "in the model-settings format."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Model spec formats accepted:
  owner/model                          e.g.  unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF
  https://huggingface.co/owner/model   full URL
  hf.co/owner/model                    short URL form

Examples:
  %(prog)s unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF
  %(prog)s https://huggingface.co/microsoft/Phi-4-reasoning-plus \\
           unsloth/gemma-3-27b-it-qat-GGUF
  %(prog)s -f models.txt -o my-model-settings.yml
""",
    )
    parser.add_argument(
        "models",
        nargs="*",
        metavar="MODEL",
        help="Model identifier(s) to scrape.",
    )
    parser.add_argument(
        "-f",
        "--file",
        metavar="FILE",
        help="File containing one model spec per line (lines starting with # are ignored).",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Write output to FILE instead of stdout.",
    )
    args = parser.parse_args()

    specs: List[str] = list(args.models)

    if args.file:
        try:
            with open(args.file) as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        specs.append(line)
        except OSError as e:
            print(f"Error reading {args.file}: {e}", file=sys.stderr)
            return 1

    if not specs:
        parser.print_help(sys.stderr)
        print("\nError: at least one model spec is required.", file=sys.stderr)
        return 1

    result = scrape_models(specs)

    yaml = _make_yaml()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            yaml.dump(result, fh)
        print(f"\nWrote {args.output}", file=sys.stderr)
    else:
        yaml.dump(result, sys.stdout)

    return 0


if __name__ == "__main__":
    sys.exit(main())
