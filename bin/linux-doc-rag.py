#!/usr/bin/env python3
"""linux-doc-rag.py – RAG agent for Linux/Ubuntu documentation.

Ingest Ubuntu 24.04 man pages into a local ChromaDB vector store, then
answer questions about Linux tools using a local LLM (via Ollama).
No internet access or API keys required.

Prerequisites
─────────────
  pip install chromadb            # vector database
  ollama serve                    # start Ollama daemon
  ollama pull nomic-embed-text    # (or another embed model)
  ollama pull llama3.2            # (or any chat model you prefer)

Usage examples
──────────────
  # Index all installed man pages (first run, may take several minutes):
  python linux-doc-rag.py ingest

  # Index only specific man-page sections (1=user cmds, 8=admin cmds):
  python linux-doc-rag.py ingest --sections 1 8

  # Ask a one-shot question:
  python linux-doc-rag.py query "How do I find files modified in the last 24 hours?"

  # Interactive session:
  python linux-doc-rag.py query --interactive

  # Show retrieved source chunks alongside the answer:
  python linux-doc-rag.py query --show-sources "How do I list open ports?"
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "linux-doc-rag" / "chroma"
DEFAULT_MAN_DIRS: List[str] = ["/usr/share/man"]
DEFAULT_DOC_DIRS: List[str] = ["/usr/share/doc"]
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_CHAT_MODEL = "llama3.2"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_COLLECTION = "linux-man-pages"

CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between consecutive chunks
DEFAULT_TOP_K = 6      # number of context chunks to retrieve
MAX_CONTEXT_CHARS = 8000  # truncate total context if very large

# Man-page sections to include by default (all standard sections)
ALL_SECTIONS: List[str] = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "n", "l"]

SYSTEM_PROMPT = (
    "You are an expert Linux system administrator and developer. "
    "You have access to Linux man pages and official documentation. "
    "When answering questions, use the provided documentation excerpts to give "
    "accurate, concrete answers with command examples where appropriate. "
    "If the documentation does not cover a specific point, say so clearly. "
    "Always prefer showing actual commands over abstract descriptions."
)

# ─────────────────────────────────────────────────────────────────────────────
# Ollama helpers
# ─────────────────────────────────────────────────────────────────────────────


def _ollama_post(url: str, payload: dict, timeout: int = 120) -> dict:
    """POST JSON payload to an Ollama endpoint and return the parsed response."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"ERROR: Cannot reach Ollama at {url}\n"
            f"  Make sure 'ollama serve' is running.  ({exc})"
        ) from exc


def get_embedding(text: str, model: str, ollama_base: str) -> List[float]:
    """Return an embedding vector for *text* using the Ollama embeddings API."""
    url = f"{ollama_base}/api/embeddings"
    resp = _ollama_post(url, {"model": model, "prompt": text})
    embedding = resp.get("embedding")
    if not embedding:
        raise ValueError(f"Ollama returned no embedding for model '{model}'")
    return embedding


def llm_generate(
    prompt: str,
    model: str,
    ollama_base: str,
    system: str = "",
    timeout: int = 180,
) -> str:
    """Generate a completion from a local LLM via Ollama."""
    url = f"{ollama_base}/api/generate"
    payload: dict = {"model": model, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    resp = _ollama_post(url, payload, timeout=timeout)
    return resp.get("response", "")


# ─────────────────────────────────────────────────────────────────────────────
# Man-page helpers
# ─────────────────────────────────────────────────────────────────────────────

# Strip ANSI colour codes and backspace-based bold/underline sequences
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_BS_RE = re.compile(r".\x08")  # any char followed by backspace


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes and groff backspace-bold artifacts from *text*."""
    text = _ANSI_RE.sub("", text)
    text = _BS_RE.sub("", text)
    return text


def render_man_page(path: Path) -> Optional[str]:
    """Render a man page file (.gz or plain troff) to plain ASCII text.

    Returns ``None`` if the file cannot be rendered.
    """
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rb") as fh:
                raw = fh.read()
        else:
            raw = path.read_bytes()
    except OSError:
        return None

    # Prefer groff for accurate rendering (strips all troff markup)
    try:
        result = subprocess.run(
            ["groff", "-man", "-T", "ascii", "-"],
            input=raw,
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0:
            text = result.stdout.decode("utf-8", errors="replace")
            return _strip_ansi(text).strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: strip troff macros with a simple regex (less accurate but works
    # on systems where groff is unavailable)
    text = raw.decode("utf-8", errors="replace")
    # Remove troff macro lines (.TH, .SH, .PP, etc.)
    text = re.sub(r"^\.[A-Za-z]{1,6}.*$", "", text, flags=re.MULTILINE)
    # Remove inline font/size escapes like \fB, \fR, \s+1, etc.
    text = re.sub(r"\\[a-zA-Z](\[.*?\])?", " ", text)
    text = re.sub(r"\\[*\"nN&(]+", "", text)
    return _strip_ansi(text).strip() or None


# All-caps section headings produced by groff -T ascii
_SECTION_HEADING_RE = re.compile(r"^([A-Z][A-Z0-9 _/():-]+)\s*$", re.MULTILINE)


def split_man_sections(text: str) -> List[Tuple[str, str]]:
    """Split a rendered man page into ``(section_name, content)`` pairs.

    Section names are the all-caps headings such as NAME, SYNOPSIS,
    DESCRIPTION, OPTIONS, EXAMPLES, FILES, etc.  If no headings are
    detected the entire text is returned as a single "FULL" section.
    """
    matches = list(_SECTION_HEADING_RE.finditer(text))
    if not matches:
        return [("FULL", text.strip())]

    sections: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections.append((title, content))
    return sections


def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split *text* into overlapping chunks of at most *size* characters."""
    if len(text) <= size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + size]
        chunks.append(chunk)
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def iter_man_pages(
    man_dirs: List[str],
    sections: Optional[List[str]] = None,
) -> Iterator[Tuple[Path, str, str]]:
    """Yield ``(path, page_name, section)`` for every man page under *man_dirs*.

    *sections* is an optional allowlist of man-page section numbers
    (e.g. ``["1", "8"]``).  When ``None`` all sections are included.
    """
    allowed = set(sections) if sections else None
    for man_dir in man_dirs:
        base = Path(man_dir)
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            name = path.name
            stem = name[:-3] if name.endswith(".gz") else name
            # man pages: <name>.<section>[ext]   e.g. find.1.gz, ls.1, groff.1.gz
            parts = stem.rsplit(".", 1)
            if len(parts) != 2:
                continue
            page_name, section = parts
            # Section field must start with a digit (1-9) or be "n" / "l"
            if not re.fullmatch(r"[0-9nl][a-z]*", section):
                continue
            sec_num = section[0]
            if allowed and sec_num not in allowed:
                continue
            yield path, page_name, section


def iter_doc_files(doc_dirs: List[str]) -> Iterator[Tuple[Path, str]]:
    """Yield ``(path, package_name)`` for plain-text doc files under *doc_dirs*.

    Includes README*, CHANGES*, *.md, *.txt files that are not too large
    (< 256 KB) and skips binary files.
    """
    text_patterns = re.compile(
        r"(README|CHANGES|CHANGELOG|NEWS|TODO|HACKING|AUTHORS|INSTALL|COPYING)"
        r"|(\.(md|txt|rst))$",
        re.IGNORECASE,
    )
    for doc_dir in doc_dirs:
        base = Path(doc_dir)
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if path.stat().st_size > 256 * 1024:
                continue
            if not text_patterns.search(path.name):
                continue
            pkg = path.relative_to(base).parts[0] if path.relative_to(base).parts else "unknown"
            yield path, pkg


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB helpers
# ─────────────────────────────────────────────────────────────────────────────


def _require_chromadb():
    """Import chromadb or raise a helpful SystemExit."""
    try:
        import chromadb  # noqa: F401
        return chromadb
    except ImportError:
        raise SystemExit(
            "ERROR: chromadb is not installed.\n"
            "  Install it with:  pip install chromadb"
        )


def get_chroma_client(db_path: Path):
    """Return a persistent ChromaDB client rooted at *db_path*."""
    chromadb = _require_chromadb()
    db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_path))


def get_or_create_collection(client, collection: str):
    """Get or create a ChromaDB collection using cosine distance."""
    return client.get_or_create_collection(
        name=collection,
        metadata={"hnsw:space": "cosine"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# ingest command
# ─────────────────────────────────────────────────────────────────────────────


def _flush_batch(
    coll,
    ids: List[str],
    docs: List[str],
    metas: List[dict],
    embeds: List[List[float]],
) -> None:
    """Write a batch to ChromaDB and clear the lists in-place."""
    if ids:
        coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)
        ids.clear()
        docs.clear()
        metas.clear()
        embeds.clear()


def cmd_ingest(args) -> None:
    """Index man pages (and optionally /usr/share/doc) into ChromaDB."""
    db_path = Path(args.db_path)
    man_dirs: List[str] = args.man_dirs or DEFAULT_MAN_DIRS
    doc_dirs: List[str] = args.doc_dirs or []
    sections: Optional[List[str]] = args.sections or None
    embed_model: str = args.embed_model
    ollama_base: str = args.ollama_url.rstrip("/")
    collection_name: str = args.collection
    batch_size: int = args.batch_size
    verbose: bool = args.verbose

    print(f"[ingest] DB path      : {db_path}", file=sys.stderr)
    print(f"[ingest] Man dirs     : {man_dirs}", file=sys.stderr)
    if doc_dirs:
        print(f"[ingest] Doc dirs     : {doc_dirs}", file=sys.stderr)
    if sections:
        print(f"[ingest] Sections     : {sections}", file=sys.stderr)
    print(f"[ingest] Embed model  : {embed_model}", file=sys.stderr)
    print(f"[ingest] Ollama URL   : {ollama_base}", file=sys.stderr)

    client = get_chroma_client(db_path)
    coll = get_or_create_collection(client, collection_name)

    existing_ids: set = set(coll.get(include=[])["ids"])
    print(f"[ingest] Existing docs: {len(existing_ids)}", file=sys.stderr)

    pages = list(iter_man_pages(man_dirs, sections))
    print(f"[ingest] Found {len(pages)} man page files", file=sys.stderr)

    ids_b: List[str] = []
    docs_b: List[str] = []
    metas_b: List[dict] = []
    embeds_b: List[List[float]] = []

    processed = skipped = errors = 0

    # ── Man pages ────────────────────────────────────────────────────────────
    for idx, (path, page_name, section) in enumerate(pages):
        if verbose:
            print(f"[ingest] {idx + 1}/{len(pages)} {path}", file=sys.stderr)
        elif (idx + 1) % 200 == 0:
            print(
                f"[ingest] Progress: {idx + 1}/{len(pages)}"
                f"  (processed={processed}, skipped={skipped}, errors={errors})",
                file=sys.stderr,
            )

        text = render_man_page(path)
        if not text or len(text) < 10:
            skipped += 1
            continue

        sections_parsed = split_man_sections(text)
        for sec_name, sec_content in sections_parsed:
            for ci, chunk in enumerate(chunk_text(sec_content)):
                chunk = chunk.strip()
                if len(chunk) < 20:
                    continue
                doc_id = f"man:{page_name}.{section}/{sec_name}/{ci}"
                if doc_id in existing_ids and not args.force:
                    continue
                try:
                    embedding = get_embedding(chunk, embed_model, ollama_base)
                except Exception as exc:
                    if verbose:
                        print(f"[ingest]   embed error {doc_id}: {exc}", file=sys.stderr)
                    errors += 1
                    continue
                ids_b.append(doc_id)
                docs_b.append(chunk)
                metas_b.append(
                    {
                        "source": "man",
                        "page": page_name,
                        "section": section,
                        "section_name": sec_name,
                        "path": str(path),
                    }
                )
                embeds_b.append(embedding)
                processed += 1
                if len(ids_b) >= batch_size:
                    _flush_batch(coll, ids_b, docs_b, metas_b, embeds_b)

    _flush_batch(coll, ids_b, docs_b, metas_b, embeds_b)

    # ── Extra doc files (/usr/share/doc) ─────────────────────────────────────
    if doc_dirs:
        doc_files = list(iter_doc_files(doc_dirs))
        print(f"[ingest] Found {len(doc_files)} doc files", file=sys.stderr)
        for idx, (path, pkg) in enumerate(doc_files):
            if verbose:
                print(f"[ingest] doc {idx + 1}/{len(doc_files)} {path}", file=sys.stderr)
            try:
                text = path.read_text(errors="replace")
            except OSError:
                skipped += 1
                continue
            for ci, chunk in enumerate(chunk_text(text)):
                chunk = chunk.strip()
                if len(chunk) < 20:
                    continue
                doc_id = f"doc:{pkg}/{path.name}/{ci}"
                if doc_id in existing_ids and not args.force:
                    continue
                try:
                    embedding = get_embedding(chunk, embed_model, ollama_base)
                except Exception as exc:
                    if verbose:
                        print(f"[ingest]   embed error {doc_id}: {exc}", file=sys.stderr)
                    errors += 1
                    continue
                ids_b.append(doc_id)
                docs_b.append(chunk)
                metas_b.append(
                    {
                        "source": "doc",
                        "page": pkg,
                        "section": "",
                        "section_name": path.name,
                        "path": str(path),
                    }
                )
                embeds_b.append(embedding)
                processed += 1
                if len(ids_b) >= batch_size:
                    _flush_batch(coll, ids_b, docs_b, metas_b, embeds_b)

        _flush_batch(coll, ids_b, docs_b, metas_b, embeds_b)

    print(
        f"\n[ingest] Done.  processed={processed}  skipped={skipped}  errors={errors}",
        file=sys.stderr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# query command
# ─────────────────────────────────────────────────────────────────────────────


def build_rag_prompt(question: str, chunks: List[Dict]) -> str:
    """Build a RAG prompt from *question* and retrieved *chunks*.

    Each chunk dict must have keys ``"document"`` and ``"metadata"``.
    """
    context_parts: List[str] = []
    total_chars = 0
    for c in chunks:
        doc = c["document"]
        meta = c["metadata"]
        source = meta.get("source", "man")
        if source == "man":
            header = f"[man {meta.get('page', '?')}({meta.get('section', '?')}) – {meta.get('section_name', '?')}]"
        else:
            header = f"[doc {meta.get('page', '?')} / {meta.get('section_name', '?')}]"
        entry = f"{header}\n{doc}"
        if total_chars + len(entry) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(entry)
        total_chars += len(entry)

    context = "\n\n---\n\n".join(context_parts) if context_parts else "(no relevant documentation found)"
    return (
        f"Documentation excerpts:\n\n{context}\n\n"
        f"---\n\nQuestion: {question}\n\nAnswer:"
    )


def cmd_query(args) -> None:
    """Query the indexed man pages using a local LLM."""
    db_path = Path(args.db_path)
    embed_model: str = args.embed_model
    chat_model: str = args.chat_model
    ollama_base: str = args.ollama_url.rstrip("/")
    collection_name: str = args.collection
    top_k: int = args.top_k
    verbose: bool = args.verbose
    show_sources: bool = args.show_sources

    client = get_chroma_client(db_path)
    try:
        coll = client.get_collection(name=collection_name)
    except Exception:
        raise SystemExit(
            f"ERROR: Collection '{collection_name}' not found in {db_path}.\n"
            "  Run 'linux-doc-rag.py ingest' first."
        )

    def answer(question: str) -> None:
        if verbose:
            print("[query] Embedding question …", file=sys.stderr)
        q_embed = get_embedding(question, embed_model, ollama_base)

        if verbose:
            print(f"[query] Searching vector DB (top-{top_k}) …", file=sys.stderr)
        results = coll.query(
            query_embeddings=[q_embed],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: List[Dict] = [
            {"document": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

        if show_sources or verbose:
            print("\nSources:", file=sys.stderr)
            for c in chunks:
                meta = c["metadata"]
                dist = c["distance"]
                src = meta.get("source", "man")
                if src == "man":
                    label = f"  man {meta.get('page')}({meta.get('section')}) – {meta.get('section_name')}"
                else:
                    label = f"  doc {meta.get('page')} / {meta.get('section_name')}"
                print(f"  [{dist:.3f}] {label}", file=sys.stderr)
            print(file=sys.stderr)

        prompt = build_rag_prompt(question, chunks)
        if verbose:
            print(f"[query] Generating answer with {chat_model} …", file=sys.stderr)
        answer_text = llm_generate(prompt, chat_model, ollama_base, system=SYSTEM_PROMPT)
        print(answer_text)

    if args.question:
        answer(" ".join(args.question))
    elif args.interactive:
        print("Linux Doc RAG – type your question, or 'exit' / Ctrl-D to quit.\n")
        while True:
            try:
                question = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.lower() in {"exit", "quit", "q"}:
                break
            if question:
                answer(question)
    else:
        raise SystemExit(
            "ERROR: Provide a question (positional args) or use --interactive."
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI parser
# ─────────────────────────────────────────────────────────────────────────────


def _shared_parser() -> argparse.ArgumentParser:
    """Return an ArgumentParser with flags shared by both subcommands."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        metavar="DIR",
        help=f"ChromaDB directory (default: {DEFAULT_DB_PATH})",
    )
    p.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        metavar="MODEL",
        help=f"Ollama embedding model (default: {DEFAULT_EMBED_MODEL})",
    )
    p.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        metavar="URL",
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})",
    )
    p.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        metavar="NAME",
        help=f"ChromaDB collection name (default: {DEFAULT_COLLECTION})",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    return p


def make_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="linux-doc-rag.py",
        description="RAG agent for Linux/Ubuntu documentation (man pages).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    shared = _shared_parser()
    subs = parser.add_subparsers(dest="command", required=True)

    # ── ingest ───────────────────────────────────────────────────────────────
    ingest_p = subs.add_parser(
        "ingest",
        parents=[shared],
        help="Index man pages into the vector store",
        description="Parse, chunk, embed, and index man pages into ChromaDB.",
    )
    ingest_p.add_argument(
        "--man-dirs",
        nargs="+",
        default=None,
        metavar="DIR",
        help=f"Man page directories to scan (default: {DEFAULT_MAN_DIRS})",
    )
    ingest_p.add_argument(
        "--doc-dirs",
        nargs="+",
        default=None,
        metavar="DIR",
        help=(
            "Additional doc directories to index, e.g. /usr/share/doc "
            "(README, CHANGELOG, *.md, *.txt files; secondary to man pages)"
        ),
    )
    ingest_p.add_argument(
        "--sections",
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "Man-page sections to include, e.g. --sections 1 8 "
            "(default: all sections)"
        ),
    )
    ingest_p.add_argument(
        "--force",
        action="store_true",
        help="Re-embed and overwrite already-indexed entries",
    )
    ingest_p.add_argument(
        "--batch-size",
        type=int,
        default=50,
        metavar="N",
        help="ChromaDB write batch size (default: 50)",
    )
    ingest_p.set_defaults(func=cmd_ingest)

    # ── query ────────────────────────────────────────────────────────────────
    query_p = subs.add_parser(
        "query",
        parents=[shared],
        help="Ask a question about Linux tools",
        description="Retrieve relevant man-page excerpts and answer with a local LLM.",
    )
    query_p.add_argument(
        "question",
        nargs="*",
        help="Question to ask (all remaining arguments are joined into one question)",
    )
    query_p.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Start an interactive question-answer session",
    )
    query_p.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        metavar="N",
        help=f"Number of context chunks to retrieve (default: {DEFAULT_TOP_K})",
    )
    query_p.add_argument(
        "--chat-model",
        default=DEFAULT_CHAT_MODEL,
        metavar="MODEL",
        help=f"Ollama chat/completion model (default: {DEFAULT_CHAT_MODEL})",
    )
    query_p.add_argument(
        "--show-sources",
        action="store_true",
        help="Print the retrieved source chunks used to build the answer",
    )
    query_p.set_defaults(func=cmd_query)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
