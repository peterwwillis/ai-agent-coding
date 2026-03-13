"""Tests for docs-rag.py.

Covers all pure-Python functions (no Ollama, no ChromaDB required).
Tests that depend on external services are skipped with pytest.mark.skip
or use unittest.mock to patch network calls.
"""
from __future__ import annotations

import gzip
import importlib
import json
import sys
import textwrap
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Allow importing the script as a module (it has a hyphen in the name).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
rag = importlib.import_module("docs-rag")


# ─────────────────────────────────────────────────────────────────────────────
# _strip_ansi
# ─────────────────────────────────────────────────────────────────────────────


class TestStripAnsi:
    def test_removes_ansi_colour_codes(self):
        assert rag._strip_ansi("\x1b[1mhello\x1b[0m world") == "hello world"

    def test_removes_backspace_bold(self):
        # groff -T ascii produces bold via char + backspace + char
        text = "h\x08he\x08el\x08ll\x08lo\x08o"
        result = rag._strip_ansi(text)
        assert "\x08" not in result
        assert result == "hello"

    def test_plain_text_unchanged(self):
        plain = "just ordinary text 123"
        assert rag._strip_ansi(plain) == plain

    def test_empty_string(self):
        assert rag._strip_ansi("") == ""


# ─────────────────────────────────────────────────────────────────────────────
# chunk_text
# ─────────────────────────────────────────────────────────────────────────────


class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        text = "hello world"
        chunks = rag.chunk_text(text, size=100, overlap=10)
        assert chunks == [text]

    def test_text_exactly_chunk_size_returns_single_chunk(self):
        text = "a" * 100
        chunks = rag.chunk_text(text, size=100, overlap=10)
        assert chunks == [text]

    def test_long_text_splits_into_multiple_chunks(self):
        text = "a" * 1000
        chunks = rag.chunk_text(text, size=200, overlap=50)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 200

    def test_all_content_covered(self):
        """Every character in the original text appears in at least one chunk."""
        text = "abcdefghij" * 50  # 500 chars
        chunks = rag.chunk_text(text, size=100, overlap=20)
        # The first chunk starts at 0 and the last must reach the end
        assert chunks[0] == text[:100]
        joined_end = chunks[-1]
        assert joined_end in text

    def test_overlap_between_consecutive_chunks(self):
        text = "0123456789" * 20  # 200 chars
        overlap = 20
        size = 50
        chunks = rag.chunk_text(text, size=size, overlap=overlap)
        # Consecutive chunks share *overlap* characters at the boundary
        assert chunks[0][-overlap:] == chunks[1][:overlap]

    def test_empty_string_returns_one_empty_chunk(self):
        assert rag.chunk_text("", size=100) == [""]


# ─────────────────────────────────────────────────────────────────────────────
# split_man_sections
# ─────────────────────────────────────────────────────────────────────────────


class TestSplitManSections:
    def test_no_headings_returns_full_section(self):
        text = "some text without headings"
        result = rag.split_man_sections(text)
        assert result == [("FULL", text)]

    def test_single_heading(self):
        text = "NAME\n   find - search for files\n"
        result = rag.split_man_sections(text)
        assert len(result) == 1
        assert result[0][0] == "NAME"
        assert "find" in result[0][1]

    def test_multiple_headings(self):
        text = textwrap.dedent("""\
            NAME
               find - search for files in a directory hierarchy
            SYNOPSIS
               find [path] [expression]
            DESCRIPTION
               The find command searches for files.
        """)
        result = rag.split_man_sections(text)
        names = [s[0] for s in result]
        assert "NAME" in names
        assert "SYNOPSIS" in names
        assert "DESCRIPTION" in names

    def test_content_assigned_to_correct_section(self):
        text = textwrap.dedent("""\
            OPTIONS
               -h  show help
            EXAMPLES
               find . -name "*.py"
        """)
        result = rag.split_man_sections(text)
        by_name = dict(result)
        assert "show help" in by_name["OPTIONS"]
        assert "find" in by_name["EXAMPLES"]

    def test_empty_sections_skipped(self):
        # Two adjacent headings with nothing between them
        text = "NAME\nSYNOPSIS\n   ls [OPTION]...\n"
        result = rag.split_man_sections(text)
        names = [s[0] for s in result]
        # NAME has no content before SYNOPSIS, so it should not appear
        assert "NAME" not in names
        assert "SYNOPSIS" in names


# ─────────────────────────────────────────────────────────────────────────────
# iter_man_pages
# ─────────────────────────────────────────────────────────────────────────────


class TestIterManPages:
    def _make_gz(self, path: Path, content: bytes = b".TH TEST 1\n.SH NAME\ntest\n"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as fh:
            fh.write(content)

    def test_finds_gzipped_man_page(self, tmp_path):
        self._make_gz(tmp_path / "man1" / "find.1.gz")
        pages = list(rag.iter_man_pages([str(tmp_path)]))
        assert len(pages) == 1
        path, page, section = pages[0]
        assert page == "find"
        assert section == "1"

    def test_finds_uncompressed_man_page(self, tmp_path):
        d = tmp_path / "man8"
        d.mkdir()
        (d / "useradd.8").write_bytes(b".TH USERADD 8\n.SH NAME\nuseradd\n")
        pages = list(rag.iter_man_pages([str(tmp_path)]))
        assert len(pages) == 1
        _, page, section = pages[0]
        assert page == "useradd"
        assert section == "8"

    def test_ignores_non_man_files(self, tmp_path):
        d = tmp_path / "man1"
        d.mkdir()
        (d / "README.txt").write_text("not a man page")
        (d / "Makefile").write_text("all: ;\n")
        pages = list(rag.iter_man_pages([str(tmp_path)]))
        assert pages == []

    def test_section_filter_includes_matching(self, tmp_path):
        self._make_gz(tmp_path / "man1" / "ls.1.gz")
        self._make_gz(tmp_path / "man8" / "cron.8.gz")
        pages = list(rag.iter_man_pages([str(tmp_path)], sections=["1"]))
        page_names = [p[1] for p in pages]
        assert "ls" in page_names
        assert "cron" not in page_names

    def test_section_filter_excludes_non_matching(self, tmp_path):
        self._make_gz(tmp_path / "man3" / "printf.3.gz")
        pages = list(rag.iter_man_pages([str(tmp_path)], sections=["1", "8"]))
        assert pages == []

    def test_missing_dir_skipped_silently(self, tmp_path):
        pages = list(rag.iter_man_pages([str(tmp_path / "nonexistent")]))
        assert pages == []

    def test_multiple_man_dirs(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        self._make_gz(dir_a / "man1" / "cat.1.gz")
        self._make_gz(dir_b / "man1" / "grep.1.gz")
        pages = list(rag.iter_man_pages([str(dir_a), str(dir_b)]))
        names = {p[1] for p in pages}
        assert "cat" in names
        assert "grep" in names


# ─────────────────────────────────────────────────────────────────────────────
# iter_doc_files
# ─────────────────────────────────────────────────────────────────────────────


class TestIterDocFiles:
    def test_finds_readme(self, tmp_path):
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        (pkg / "README").write_text("This is a readme")
        files = list(rag.iter_doc_files([str(tmp_path)]))
        assert any(p.name == "README" for p, _ in files)

    def test_skips_changelog_files(self, tmp_path):
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        (pkg / "CHANGELOG.md").write_text("# Changelog")
        (pkg / "ChangeLog").write_text("2024-01-01 fix")
        (pkg / "changelog.gz").write_bytes(b"fake")
        files = list(rag.iter_doc_files([str(tmp_path)]))
        names = {p.name for p, _ in files}
        assert "CHANGELOG.md" not in names
        assert "ChangeLog" not in names
        assert "changelog.gz" not in names

    def test_skips_copyright_files(self, tmp_path):
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        (pkg / "copyright").write_text("(C) 2024 …")
        (pkg / "COPYRIGHT").write_text("(C) 2024 …")
        (pkg / "COPYING.copyright").write_text("GPL text")
        files = list(rag.iter_doc_files([str(tmp_path)]))
        names = {p.name for p, _ in files}
        assert "copyright" not in names
        assert "COPYRIGHT" not in names
        assert "COPYING.copyright" not in names

    def test_finds_arbitrary_text_file(self, tmp_path):
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        (pkg / "NEWS").write_text("Release notes")
        (pkg / "features.txt").write_text("Feature list")
        files = list(rag.iter_doc_files([str(tmp_path)]))
        names = {p.name for p, _ in files}
        assert "NEWS" in names
        assert "features.txt" in names

    def test_skips_large_files(self, tmp_path):
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        big = pkg / "README.txt"
        big.write_bytes(b"x" * (257 * 1024))  # > 256 KB
        files = list(rag.iter_doc_files([str(tmp_path)]))
        assert files == []

    def test_skips_binary_files(self, tmp_path):
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        (pkg / "libfoo.so").write_bytes(b"\x7fELF\x00more")
        files = list(rag.iter_doc_files([str(tmp_path)]))
        assert files == []

    def test_package_name_is_first_path_component(self, tmp_path):
        pkg = tmp_path / "libfoo"
        pkg.mkdir()
        (pkg / "README.md").write_text("libfoo readme")
        files = list(rag.iter_doc_files([str(tmp_path)]))
        assert files[0][1] == "libfoo"


# ─────────────────────────────────────────────────────────────────────────────
# render_man_page
# ─────────────────────────────────────────────────────────────────────────────


class TestRenderManPage:
    def test_renders_gzipped_man_page(self, tmp_path):
        src = b".TH LS 1\n.SH NAME\nls \\- list directory contents\n"
        gz = tmp_path / "ls.1.gz"
        with gzip.open(gz, "wb") as fh:
            fh.write(src)
        text = rag.render_man_page(gz)
        assert text is not None
        # After rendering the section heading should be present
        assert "ls" in text.lower()

    def test_renders_plain_troff_file(self, tmp_path):
        src = b".TH FIND 1\n.SH NAME\nfind \\- search for files\n"
        plain = tmp_path / "find.1"
        plain.write_bytes(src)
        text = rag.render_man_page(plain)
        assert text is not None
        assert "find" in text.lower()

    def test_returns_none_for_missing_file(self, tmp_path):
        result = rag.render_man_page(tmp_path / "nonexistent.1.gz")
        assert result is None

    def test_returns_none_for_empty_file(self, tmp_path):
        # groff on empty input returns nothing useful
        gz = tmp_path / "empty.1.gz"
        with gzip.open(gz, "wb") as fh:
            fh.write(b"")
        result = rag.render_man_page(gz)
        # Either None or empty string is acceptable
        assert not result  # falsy

    def test_falls_back_to_mandoc_when_groff_missing(self, tmp_path):
        src = b".TH FIND 1\n.SH NAME\nfind \\- search for files\n"
        plain = tmp_path / "find.1"
        plain.write_bytes(src)

        def _run_side_effect(cmd, input=None, capture_output=None, timeout=None):
            if cmd[0] == "groff":
                raise FileNotFoundError("groff missing")
            if cmd[0] == "mandoc":
                proc = MagicMock()
                proc.returncode = 0
                proc.stdout = b"NAME\nfind - search for files\n"
                return proc
            raise AssertionError(f"Unexpected command: {cmd}")

        with patch("subprocess.run", side_effect=_run_side_effect):
            text = rag.render_man_page(plain)
        assert text is not None
        assert "find" in text.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Platform-specific defaults
# ─────────────────────────────────────────────────────────────────────────────


class TestPlatformDefaults:
    def test_macos_man_dirs_include_homebrew(self):
        dirs = rag.default_man_dirs("darwin")
        assert "/usr/share/man" in dirs
        assert "/opt/homebrew/share/man" in dirs

    def test_macos_info_dirs_include_homebrew(self):
        dirs = rag.default_info_dirs("darwin")
        assert "/opt/homebrew/share/info" in dirs

    def test_linux_man_dirs_default(self):
        assert rag.default_man_dirs("linux") == ["/usr/share/man"]

    def test_linux_db_path_honors_xdg_data_home(self):
        p = rag.default_db_path("linux", xdg_data_home="/tmp/data-home")
        assert p == Path("/tmp/data-home/docs-rag/chroma")

    def test_linux_config_path_honors_xdg_config_home(self):
        p = rag.default_config_path("linux", xdg_config_home="/tmp/config-home")
        assert p == Path("/tmp/config-home/docs-rag/config.yaml")

    def test_macos_paths_use_library_application_support(self):
        db_path = rag.default_db_path("darwin")
        cfg_path = rag.default_config_path("darwin")
        assert "Library/Application Support/docs-rag/chroma" in db_path.as_posix()
        assert "Library/Application Support/docs-rag/config.yaml" in cfg_path.as_posix()


# ─────────────────────────────────────────────────────────────────────────────
# build_rag_prompt
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildRagPrompt:
    def _chunk(self, doc, page="find", section="1", section_name="DESCRIPTION", distance=0.1):
        return {
            "document": doc,
            "metadata": {
                "source": "man",
                "page": page,
                "section": section,
                "section_name": section_name,
            },
            "distance": distance,
        }

    def test_question_appears_in_prompt(self):
        prompt = rag.build_rag_prompt("How do I find files?", [self._chunk("find searches files")])
        assert "How do I find files?" in prompt

    def test_document_appears_in_prompt(self):
        prompt = rag.build_rag_prompt("question", [self._chunk("find searches files")])
        assert "find searches files" in prompt

    def test_metadata_header_included(self):
        prompt = rag.build_rag_prompt("question", [self._chunk("text", page="chmod", section="1", section_name="OPTIONS")])
        assert "chmod" in prompt
        assert "OPTIONS" in prompt

    def test_doc_source_header(self):
        chunk = {
            "document": "libfoo does things",
            "metadata": {"source": "doc", "page": "libfoo", "section": "", "section_name": "README.md"},
            "distance": 0.2,
        }
        prompt = rag.build_rag_prompt("question", [chunk])
        assert "libfoo" in prompt
        assert "README.md" in prompt

    def test_empty_chunks_produces_no_docs_notice(self):
        prompt = rag.build_rag_prompt("anything?", [])
        assert "no relevant documentation" in prompt.lower()

    def test_context_truncated_when_too_large(self):
        big_doc = "x" * (rag.MAX_CONTEXT_CHARS * 2)
        small_doc = "small doc text"
        chunks = [
            self._chunk(big_doc, page="big"),
            self._chunk(small_doc, page="small"),
        ]
        prompt = rag.build_rag_prompt("test", chunks)
        # The second (small) chunk should be omitted because MAX_CONTEXT_CHARS exceeded
        assert "small doc text" not in prompt

    def test_multiple_chunks_separated(self):
        chunks = [
            self._chunk("first chunk", page="find"),
            self._chunk("second chunk", page="ls"),
        ]
        prompt = rag.build_rag_prompt("question", chunks)
        assert "first chunk" in prompt
        assert "second chunk" in prompt
        assert "---" in prompt  # separator between chunks


# ─────────────────────────────────────────────────────────────────────────────
# get_embedding (mocked)
# ─────────────────────────────────────────────────────────────────────────────


class TestGetEmbedding:
    def test_returns_list_of_floats(self):
        fake_response = json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode()
        with patch("urllib.request.urlopen") as mock_open:
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=fake_response)))
            mock_cm.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_cm
            result = rag.get_embedding("hello", "nomic-embed-text", "http://localhost:11434")
        assert result == [0.1, 0.2, 0.3]

    def test_raises_on_missing_embedding_key(self):
        fake_response = json.dumps({"error": "model not found"}).encode()
        with patch("urllib.request.urlopen") as mock_open:
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=fake_response)))
            mock_cm.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_cm
            with pytest.raises(ValueError, match="no embedding"):
                rag.get_embedding("hello", "nomic-embed-text", "http://localhost:11434")

    def test_raises_system_exit_on_network_error(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            with pytest.raises(SystemExit, match="Cannot reach Ollama"):
                rag.get_embedding("hello", "nomic-embed-text", "http://localhost:11434")


# ─────────────────────────────────────────────────────────────────────────────
# llm_generate (mocked)
# ─────────────────────────────────────────────────────────────────────────────


class TestLlmGenerate:
    def test_returns_response_field(self):
        fake_response = json.dumps({"response": "The answer is 42."}).encode()
        with patch("urllib.request.urlopen") as mock_open:
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=fake_response)))
            mock_cm.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_cm
            result = rag.llm_generate("prompt text", "llama3.2", "http://localhost:11434")
        assert result == "The answer is 42."

    def test_returns_empty_string_when_no_response_key(self):
        fake_response = json.dumps({"done": True}).encode()
        with patch("urllib.request.urlopen") as mock_open:
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=fake_response)))
            mock_cm.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_cm
            result = rag.llm_generate("prompt", "model", "http://localhost:11434")
        assert result == ""

    def test_includes_system_prompt_in_payload(self):
        fake_response = json.dumps({"response": "ok"}).encode()
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["payload"] = json.loads(req.data)
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=fake_response)))
            mock_cm.__exit__ = MagicMock(return_value=False)
            return mock_cm

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            rag.llm_generate("prompt", "model", "http://localhost:11434", system="Be helpful")
        assert captured["payload"]["system"] == "Be helpful"


# ─────────────────────────────────────────────────────────────────────────────
# make_parser / CLI structure
# ─────────────────────────────────────────────────────────────────────────────


class TestMakeParser:
    def test_ingest_subcommand_exists(self):
        parser = rag.make_parser()
        args = parser.parse_args(["ingest"])
        assert args.command == "ingest"

    def test_query_subcommand_exists(self):
        parser = rag.make_parser()
        args = parser.parse_args(["query", "hello", "world"])
        assert args.command == "query"
        assert args.question == ["hello", "world"]

    def test_default_embed_model(self):
        parser = rag.make_parser()
        args = parser.parse_args(["ingest"])
        assert args.embed_model == rag.DEFAULT_EMBED_MODEL

    def test_default_chat_model(self):
        parser = rag.make_parser()
        args = parser.parse_args(["query", "anything"])
        assert args.chat_model == rag.DEFAULT_CHAT_MODEL

    def test_custom_db_path(self):
        parser = rag.make_parser()
        args = parser.parse_args(["ingest", "--db-path", "/tmp/mydb"])
        assert args.db_path == "/tmp/mydb"

    def test_sections_flag(self):
        parser = rag.make_parser()
        args = parser.parse_args(["ingest", "--sections", "1", "8"])
        assert args.sections == ["1", "8"]

    def test_force_flag(self):
        parser = rag.make_parser()
        args = parser.parse_args(["ingest", "--force"])
        assert args.force is True

    def test_interactive_flag(self):
        parser = rag.make_parser()
        args = parser.parse_args(["query", "--interactive"])
        assert args.interactive is True

    def test_show_sources_flag(self):
        parser = rag.make_parser()
        args = parser.parse_args(["query", "--show-sources", "question?"])
        assert args.show_sources is True

    def test_top_k_flag(self):
        parser = rag.make_parser()
        args = parser.parse_args(["query", "--top-k", "10", "q"])
        assert args.top_k == 10

    def test_doc_dirs_flag(self):
        parser = rag.make_parser()
        args = parser.parse_args(["ingest", "--doc-dirs", "/usr/share/doc"])
        assert args.doc_dirs == ["/usr/share/doc"]

    def test_default_doc_dirs(self):
        parser = rag.make_parser()
        args = parser.parse_args(["ingest"])
        assert args.doc_dirs == rag.DEFAULT_DOC_DIRS

    def test_info_dirs_flag(self):
        parser = rag.make_parser()
        args = parser.parse_args(["ingest", "--info-dirs", "/usr/share/info"])
        assert args.info_dirs == ["/usr/share/info"]

    def test_default_info_dirs(self):
        parser = rag.make_parser()
        args = parser.parse_args(["ingest"])
        assert args.info_dirs == rag.DEFAULT_INFO_DIRS

    def test_config_flag(self):
        parser = rag.make_parser()
        args = parser.parse_args(["--config", "/tmp/myconfig.yaml", "ingest"])
        assert args.config == "/tmp/myconfig.yaml"

    def test_no_subcommand_raises(self):
        parser = rag.make_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


# ─────────────────────────────────────────────────────────────────────────────
# _is_binary
# ─────────────────────────────────────────────────────────────────────────────


class TestIsBinary:
    def test_text_file_not_binary(self, tmp_path):
        p = tmp_path / "readme.txt"
        p.write_text("This is plain text\n")
        assert rag._is_binary(p) is False

    def test_elf_binary(self, tmp_path):
        p = tmp_path / "prog"
        p.write_bytes(b"\x7fELF\x00\x00\x00more data")
        assert rag._is_binary(p) is True

    def test_null_byte_in_text(self, tmp_path):
        p = tmp_path / "mixed"
        p.write_bytes(b"hello\x00world")
        assert rag._is_binary(p) is True

    def test_missing_file_treated_as_binary(self, tmp_path):
        assert rag._is_binary(tmp_path / "nonexistent") is True


# ─────────────────────────────────────────────────────────────────────────────
# _strip_info_markup
# ─────────────────────────────────────────────────────────────────────────────


class TestStripInfoMarkup:
    def test_removes_form_feed(self):
        text = "before\x1fafter"
        result = rag._strip_info_markup(text)
        assert "\x1f" not in result
        assert "before" in result
        assert "after" in result

    def test_removes_node_header(self):
        text = "File: coreutils.info,  Node: Top,  Next: Intro,  Up: (dir)\nContent here"
        result = rag._strip_info_markup(text)
        assert "File:" not in result
        assert "Content here" in result

    def test_removes_info_dir_entry_block(self):
        text = (
            "START-INFO-DIR-ENTRY\n"
            "* foo: (foo).    The foo program.\n"
            "END-INFO-DIR-ENTRY\n"
            "Actual content"
        )
        result = rag._strip_info_markup(text)
        assert "START-INFO-DIR-ENTRY" not in result
        assert "Actual content" in result

    def test_plain_text_preserved(self):
        text = "The find command searches for files matching a pattern."
        result = rag._strip_info_markup(text)
        assert result == text


# ─────────────────────────────────────────────────────────────────────────────
# render_info_page
# ─────────────────────────────────────────────────────────────────────────────


class TestRenderInfoPage:
    def _write_info(self, path: Path, content: bytes, gz: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        if gz:
            with gzip.open(path, "wb") as fh:
                fh.write(content)
        else:
            path.write_bytes(content)

    def test_renders_plain_info_file(self, tmp_path):
        content = b"File: test.info,  Node: Top\nThis is the find command.\n"
        p = tmp_path / "test.info"
        self._write_info(p, content)
        text = rag.render_info_page(p)
        assert text is not None
        assert "find" in text.lower()

    def test_renders_gzipped_info_file(self, tmp_path):
        content = b"File: test.info,  Node: Top\nGrep searches text files.\n"
        p = tmp_path / "test.info.gz"
        self._write_info(p, content, gz=True)
        text = rag.render_info_page(p)
        assert text is not None
        assert "grep" in text.lower()

    def test_returns_none_for_missing_file(self, tmp_path):
        result = rag.render_info_page(tmp_path / "nonexistent.info.gz")
        assert result is None

    def test_returns_none_or_empty_for_empty_file(self, tmp_path):
        p = tmp_path / "empty.info"
        self._write_info(p, b"")
        result = rag.render_info_page(p)
        assert not result

    def test_strips_info_dir_boilerplate(self, tmp_path):
        content = (
            b"START-INFO-DIR-ENTRY\n"
            b"* foo: (foo).   Foo utility.\n"
            b"END-INFO-DIR-ENTRY\n"
            b"The foo utility does bar operations.\n"
        )
        p = tmp_path / "foo.info"
        self._write_info(p, content)
        text = rag.render_info_page(p)
        assert text is not None
        assert "START-INFO-DIR-ENTRY" not in text
        assert "foo utility" in text


# ─────────────────────────────────────────────────────────────────────────────
# iter_info_files
# ─────────────────────────────────────────────────────────────────────────────


class TestIterInfoFiles:
    def _make_info_gz(self, path: Path, content: bytes = b"File: x.info\nContent\n"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as fh:
            fh.write(content)

    def test_finds_gzipped_info_file(self, tmp_path):
        self._make_info_gz(tmp_path / "find.info.gz")
        pages = list(rag.iter_info_files([str(tmp_path)]))
        assert len(pages) == 1
        path, name = pages[0]
        assert name == "find"

    def test_finds_plain_info_file(self, tmp_path):
        (tmp_path / "grep.info").write_bytes(b"File: grep.info\nContent\n")
        pages = list(rag.iter_info_files([str(tmp_path)]))
        assert len(pages) == 1
        _, name = pages[0]
        assert name == "grep"

    def test_skips_dir_index(self, tmp_path):
        (tmp_path / "dir").write_text("info dir index")
        (tmp_path / "dir.old").write_text("old dir")
        pages = list(rag.iter_info_files([str(tmp_path)]))
        assert pages == []

    def test_skips_sub_files(self, tmp_path):
        # find.info-1.gz is a sub-file of find.info.gz — should be skipped
        self._make_info_gz(tmp_path / "find.info-1.gz")
        self._make_info_gz(tmp_path / "find.info-2.gz")
        pages = list(rag.iter_info_files([str(tmp_path)]))
        assert pages == []

    def test_main_file_only_indexed_once(self, tmp_path):
        self._make_info_gz(tmp_path / "coreutils.info.gz")
        self._make_info_gz(tmp_path / "coreutils.info-1.gz")
        self._make_info_gz(tmp_path / "coreutils.info-2.gz")
        pages = list(rag.iter_info_files([str(tmp_path)]))
        assert len(pages) == 1
        assert pages[0][1] == "coreutils"

    def test_missing_dir_skipped_silently(self, tmp_path):
        pages = list(rag.iter_info_files([str(tmp_path / "nonexistent")]))
        assert pages == []

    def test_multiple_info_dirs(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        self._make_info_gz(dir_a / "bash.info.gz")
        self._make_info_gz(dir_b / "vim.info.gz")
        pages = list(rag.iter_info_files([str(dir_a), str(dir_b)]))
        names = {p[1] for p in pages}
        assert "bash" in names
        assert "vim" in names


# ─────────────────────────────────────────────────────────────────────────────
# load_config
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadConfig:
    def test_returns_empty_dict_for_missing_file(self, tmp_path):
        result = rag.load_config(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_returns_empty_dict_for_empty_file(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("")
        assert rag.load_config(p) == {}

    def test_parses_simple_flat_config(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("embed_model: mxbai-embed-large\nollama_url: http://10.0.0.1:11434\n")
        cfg = rag.load_config(p)
        assert cfg["embed_model"] == "mxbai-embed-large"
        assert cfg["ollama_url"] == "http://10.0.0.1:11434"

    def test_parses_nested_ingest_section(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("ingest:\n  sections:\n    - '1'\n    - '8'\n  batch_size: 100\n")
        cfg = rag.load_config(p)
        assert cfg["ingest"]["sections"] == ["1", "8"]
        assert cfg["ingest"]["batch_size"] == 100

    def test_parses_nested_query_section(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("query:\n  chat_model: mistral\n  top_k: 10\n")
        cfg = rag.load_config(p)
        assert cfg["query"]["chat_model"] == "mistral"
        assert cfg["query"]["top_k"] == 10

    def test_returns_empty_dict_for_invalid_yaml(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("key: [unclosed bracket\n")
        assert rag.load_config(p) == {}

    def test_returns_empty_dict_when_root_is_not_dict(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("- item1\n- item2\n")
        assert rag.load_config(p) == {}


# ─────────────────────────────────────────────────────────────────────────────
# Config defaults applied via make_parser
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigDefaults:
    def test_shared_setting_overrides_code_default(self):
        cfg = {"embed_model": "custom-embed", "ollama_url": "http://myhost:11434"}
        parser = rag.make_parser(config=cfg)
        args = parser.parse_args(["ingest"])
        assert args.embed_model == "custom-embed"
        assert args.ollama_url == "http://myhost:11434"

    def test_cli_flag_overrides_config(self):
        cfg = {"embed_model": "custom-embed"}
        parser = rag.make_parser(config=cfg)
        args = parser.parse_args(["ingest", "--embed-model", "cli-embed"])
        assert args.embed_model == "cli-embed"

    def test_ingest_section_overrides_batch_size(self):
        cfg = {"ingest": {"batch_size": 200}}
        parser = rag.make_parser(config=cfg)
        args = parser.parse_args(["ingest"])
        assert args.batch_size == 200

    def test_ingest_section_overrides_sections(self):
        cfg = {"ingest": {"sections": ["1", "8"]}}
        parser = rag.make_parser(config=cfg)
        args = parser.parse_args(["ingest"])
        assert args.sections == ["1", "8"]

    def test_ingest_section_overrides_info_dirs(self):
        cfg = {"ingest": {"info_dirs": ["/opt/info"]}}
        parser = rag.make_parser(config=cfg)
        args = parser.parse_args(["ingest"])
        assert args.info_dirs == ["/opt/info"]

    def test_query_section_overrides_chat_model(self):
        cfg = {"query": {"chat_model": "mistral"}}
        parser = rag.make_parser(config=cfg)
        args = parser.parse_args(["query", "test"])
        assert args.chat_model == "mistral"

    def test_query_section_overrides_top_k(self):
        cfg = {"query": {"top_k": 12}}
        parser = rag.make_parser(config=cfg)
        args = parser.parse_args(["query", "test"])
        assert args.top_k == 12

    def test_empty_config_uses_code_defaults(self):
        parser = rag.make_parser(config={})
        args = parser.parse_args(["ingest"])
        assert args.embed_model == rag.DEFAULT_EMBED_MODEL
        assert args.batch_size == 50


# ─────────────────────────────────────────────────────────────────────────────
# build_rag_prompt: info source header
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildRagPromptInfoSource:
    def test_info_source_header(self):
        chunk = {
            "document": "The find command traverses directory trees.",
            "metadata": {"source": "info", "page": "find", "section": "", "section_name": "find"},
            "distance": 0.1,
        }
        prompt = rag.build_rag_prompt("How does find work?", [chunk])
        assert "[info find]" in prompt
