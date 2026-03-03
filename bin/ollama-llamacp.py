#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ----------------------------
# Data model
# ----------------------------

@dataclass
class LayerRef:
    media_type: str
    digest: str
    size: int
    blob_path: Path
    exists: bool


@dataclass
class OllamaModelRecord:
    model_tag: str
    manifest_path: Path
    layers: List[LayerRef]
    # Best-effort extras (often stored as layers with specific media types)
    template_text: Optional[str] = None
    system_text: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


# ----------------------------
# Helpers
# ----------------------------

def digest_to_blob_filename(digest: str) -> str:
    # "sha256:abcd" -> "sha256-abcd"
    return digest.replace(":", "-")


def human_size(n: int) -> str:
    # simple human formatter
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024.0 or u == "TB":
            if u == "B":
                return f"{int(f)} B"
            return f"{f:.2f} {u}"
        f /= 1024.0
    return f"{f:.2f} TB"


def iter_manifest_files(manifests_dir: Path) -> Iterable[Path]:
    for p in sorted(manifests_dir.rglob("*")):
        if p.is_file() and not p.name.endswith(".lock") and p.name != ".lock":
            yield p


def model_tag_from_manifest_path(manifests_dir: Path, manifest_path: Path) -> str:
    rel = manifest_path.relative_to(manifests_dir)
    parts = rel.parts
    # Typical: registry.ollama.ai/library/<model...>/<tag>
    if len(parts) >= 4:
        model = "/".join(parts[2:-1])
        tag = parts[-1]
        return f"{model}:{tag}"
    if len(parts) >= 2:
        return f"{'/'.join(parts[:-1])}:{parts[-1]}"
    return manifest_path.name


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_blob_text(path: Path, max_bytes: int = 512 * 1024) -> Optional[str]:
    """
    Best-effort: some metadata blobs are small JSON or text.
    We read a limited amount to avoid slurping huge files by accident.
    """
    try:
        with path.open("rb") as f:
            data = f.read(max_bytes + 1)
        if len(data) > max_bytes:
            return None
        # try utf-8
        return data.decode("utf-8", errors="strict")
    except Exception:
        return None


def maybe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else None
    except Exception:
        return None


# ----------------------------
# Core functionality (mapping)
# ----------------------------

def build_model_record(
    model_tag: str,
    manifest_path: Path,
    manifest: Dict[str, Any],
    blobs_dir: Path,
    weights_only: bool,
) -> OllamaModelRecord:
    layers_raw: List[Dict[str, Any]] = []

    # Include config only when not weights-only (usually irrelevant for llama.cpp)
    if not weights_only:
        cfg = manifest.get("config") or {}
        if isinstance(cfg, dict) and cfg.get("digest"):
            layers_raw.append(cfg)

    layers_raw.extend(manifest.get("layers") or [])

    layers: List[LayerRef] = []
    for lr in layers_raw:
        if not isinstance(lr, dict):
            continue
        media = lr.get("mediaType", "unknown") or "unknown"
        digest = lr.get("digest", "") or ""
        size = int(lr.get("size", 0) or 0)
        if not digest:
            continue
        if weights_only and ".model" not in media:
            continue

        blob_path = blobs_dir / digest_to_blob_filename(digest)
        layers.append(
            LayerRef(
                media_type=media,
                digest=digest,
                size=size,
                blob_path=blob_path,
                exists=blob_path.exists(),
            )
        )

    rec = OllamaModelRecord(model_tag=model_tag, manifest_path=manifest_path, layers=layers)

    # If not weights_only, try to locate template/system/params blobs if present
    if not weights_only:
        # Heuristics: Ollama uses mediaTypes like:
        # - application/vnd.ollama.image.template
        # - application/vnd.ollama.image.system
        # - application/vnd.ollama.image.params
        for layer in rec.layers:
            mt = layer.media_type
            if not layer.exists:
                continue
            if "template" in mt:
                txt = read_blob_text(layer.blob_path)
                if txt:
                    rec.template_text = txt
            elif "system" in mt:
                txt = read_blob_text(layer.blob_path)
                if txt:
                    rec.system_text = txt
            elif "params" in mt:
                txt = read_blob_text(layer.blob_path)
                if txt:
                    js = maybe_parse_json(txt)
                    if js:
                        rec.params = js

    return rec


def map_models(
    manifests_dir: Path,
    blobs_dir: Path,
    model_filter: Optional[str],
    weights_only: bool,
    include_meta: bool,
) -> List[OllamaModelRecord]:
    records: List[OllamaModelRecord] = []

    for mf in iter_manifest_files(manifests_dir):
        model_tag = model_tag_from_manifest_path(manifests_dir, mf)
        if model_filter and model_filter not in model_tag:
            continue

        manifest = load_json(mf)
        if not manifest:
            continue

        rec = build_model_record(
            model_tag=model_tag,
            manifest_path=mf,
            manifest=manifest,
            blobs_dir=blobs_dir,
            weights_only=weights_only,
        )

        # If user wants meta, rebuild without weights_only so we can try to read params/template/system
        if include_meta and weights_only:
            rec_full = build_model_record(
                model_tag=model_tag,
                manifest_path=mf,
                manifest=manifest,
                blobs_dir=blobs_dir,
                weights_only=False,
            )
            # Keep weights layers from rec, but copy meta from rec_full
            rec.template_text = rec_full.template_text
            rec.system_text = rec_full.system_text
            rec.params = rec_full.params

        records.append(rec)

    return records


# ----------------------------
# llama.cpp command generation
# ----------------------------

def pick_weights_blob(rec: OllamaModelRecord) -> Optional[LayerRef]:
    """
    Prefer the .model layer with the largest size, in case there are multiple.
    """
    candidates = [l for l in rec.layers if ".model" in l.media_type]
    if not candidates:
        # if user ran without weights-only, still try to find it
        candidates = [l for l in rec.layers if "model" in l.media_type]
    if not candidates:
        return None
    return max(candidates, key=lambda l: l.size)


def params_to_llamacpp_flags(params: Dict[str, Any]) -> List[str]:
    """
    Best-effort mapping. Ollama params vary; only map common ones.
    Unknown keys are ignored.
    """
    flags: List[str] = []

    def add(flag: str, value: Any):
        flags.extend([flag, str(value)])

    # Common sampling
    if "temperature" in params:
        add("--temp", params["temperature"])
    if "top_p" in params:
        add("--top-p", params["top_p"])
    if "top_k" in params:
        add("--top-k", params["top_k"])
    if "repeat_penalty" in params:
        add("--repeat-penalty", params["repeat_penalty"])
    if "seed" in params:
        add("--seed", params["seed"])

    # Common generation limits (names differ across tools)
    if "num_predict" in params:
        add("-n", params["num_predict"])

    # Context size if present (not always safe to auto-apply; but useful)
    if "num_ctx" in params:
        add("-c", params["num_ctx"])

    return flags


def build_llamacpp_command(
    rec: OllamaModelRecord,
    llama_cli_path: str,
    prompt: Optional[str],
    include_system: bool,
    include_params: bool,
) -> Tuple[List[str], List[str]]:
    """
    Returns (argv, notes). argv is shell-quoteable.
    """
    notes: List[str] = []

    weights = pick_weights_blob(rec)
    if not weights:
        raise ValueError(f"No weights (.model) layer found for {rec.model_tag}")

    if not weights.exists:
        notes.append("WARNING: weights blob file does not exist on disk (check blobs dir).")

    argv: List[str] = [llama_cli_path, "-m", str(weights.blob_path)]

    # Provide a prompt
    if prompt is not None:
        argv += ["-p", prompt]
    else:
        argv += ["-p", "Hello!"]

    # Include system prompt if present (llama.cpp supports -sys in recent builds; if not, user can prepend manually)
    if include_system and rec.system_text:
        argv += ["-sys", rec.system_text]
    elif include_system and not rec.system_text:
        notes.append("No system text found in Ollama metadata; skipping -sys.")

    # Include params best-effort
    if include_params and rec.params:
        argv += params_to_llamacpp_flags(rec.params)
    elif include_params and not rec.params:
        notes.append("No params JSON found in Ollama metadata; skipping sampling flags.")

    # Template note (llama.cpp doesn't directly consume Ollama templates)
    if rec.template_text:
        notes.append("Ollama template found. llama.cpp does not consume Ollama templates directly; use it to format your prompt/chat wrapper.")

    return argv, notes


def shell_join(argv: List[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)


# ----------------------------
# CLI
# ----------------------------

def resolve_dirs(models_dir: Optional[Path], manifests_dir: Optional[Path], blobs_dir: Optional[Path]) -> Tuple[Path, Path]:
    if models_dir:
        md = models_dir.expanduser().resolve()
        return ( (manifests_dir or (md / "manifests")).expanduser().resolve(),
                 (blobs_dir or (md / "blobs")).expanduser().resolve() )

    if manifests_dir and blobs_dir:
        return (manifests_dir.expanduser().resolve(), blobs_dir.expanduser().resolve())

    raise SystemExit("ERROR: Provide --models-dir OR both --manifests-dir and --blobs-dir.")


def main():
    ap = argparse.ArgumentParser(description="Map Ollama blobs to models and generate llama.cpp run commands.")
    ap.add_argument("--models-dir", type=Path, help="Ollama models dir containing manifests/ and blobs/")
    ap.add_argument("--manifests-dir", type=Path, help="Manifests directory")
    ap.add_argument("--blobs-dir", type=Path, help="Blobs directory (sha256-* files)")

    ap.add_argument("--model", help="Filter by substring match on model:tag (e.g. 'llama3' or 'mistral:7b')")
    ap.add_argument("--weights-only", action="store_true", help="Only show the weights layer(s) (mediaType containing '.model')")

    ap.add_argument("--mode", choices=["map", "cmd"], default="map", help="map=print mapping, cmd=print llama.cpp command")
    ap.add_argument("--llama-cli", default="llama-cli", help="Path to llama.cpp CLI binary (default: llama-cli)")
    ap.add_argument("--prompt", help="Prompt to use for cmd mode (-p). If omitted, uses 'Hello!'")

    args = ap.parse_args()

    manifests_dir, blobs_dir = resolve_dirs(args.models_dir, args.manifests_dir, args.blobs_dir)

    if not manifests_dir.is_dir():
        raise SystemExit(f"ERROR: manifests dir not found: {manifests_dir}")
    if not blobs_dir.is_dir():
        raise SystemExit(f"ERROR: blobs dir not found: {blobs_dir}")

    include_meta = (args.mode == "cmd")

    records = map_models(
        manifests_dir=manifests_dir,
        blobs_dir=blobs_dir,
        model_filter=args.model,
        weights_only=args.weights_only if args.mode == "map" else False,  # cmd needs metadata scan
        include_meta=include_meta,
    )

    if args.mode == "map":
        print(f"manifests_dir: {manifests_dir}")
        print(f"blobs_dir    : {blobs_dir}")
        print("")
        print(f"{'MODEL:TAG':45} {'MEDIA_TYPE':45} {'SIZE':10} {'EXISTS':6} BLOB_PATH")
        print("-" * 130)
        for rec in records:
            for layer in rec.layers:
                print(f"{rec.model_tag:45} {layer.media_type:45} {human_size(layer.size):10} "
                      f"{'yes' if layer.exists else 'no':6} {layer.blob_path}")
        return

    # cmd mode
    if args.model:
        if not records:
            raise SystemExit(f"ERROR: no models match filter: {args.model}")
        selected = next((rec for rec in records if rec.model_tag == args.model), records[0])
        records = [selected]

    # If multiple match (no filter), print all commands
    for rec in records:
        argv, notes = build_llamacpp_command(
            rec=rec,
            llama_cli_path=args.llama_cli,
            prompt=args.prompt,
            include_system=True,
            include_params=True,
        )
        print(f"# {rec.model_tag}")
        for n in notes:
            print(f"# {n}")
        print(shell_join(argv))
        print("")


if __name__ == "__main__":
    main()
