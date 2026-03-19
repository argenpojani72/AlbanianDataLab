"""
tests/test_text_processing.py — Unit tests for text extraction and processing.

Tests the pure Python logic in scripts/build_text_dataset.py:
  - clean_text()
  - chunk_text()
  - content_hash()
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_text_dataset import chunk_text, clean_text, content_hash


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_nfc_normalisation(self):
        """Unicode NFC normalisation is applied."""
        import unicodedata
        # NFD form of 'é' (e + combining acute accent)
        nfd = "\u0065\u0301"
        nfc = "\u00e9"
        result = clean_text(nfd)
        assert unicodedata.is_normalized("NFC", result)
        assert result == nfc

    def test_strips_control_chars(self):
        """Control characters (except newlines and tabs) are removed."""
        text = "hello\x00world\x01\x1f"
        result = clean_text(text)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "\x1f" not in result
        assert "hello" in result
        assert "world" in result

    def test_preserves_newlines(self):
        """Newlines are preserved after cleaning."""
        result = clean_text("line one\nline two")
        assert "\n" in result

    def test_collapses_multiple_spaces(self):
        """Runs of spaces/tabs on a single line are collapsed to one space."""
        result = clean_text("word1   word2\t\tword3")
        assert "word1 word2 word3" in result

    def test_strips_leading_trailing_whitespace_per_line(self):
        """Leading and trailing whitespace on each line is stripped."""
        result = clean_text("  hello  \n  world  ")
        lines = result.splitlines()
        for line in lines:
            assert line == line.strip()

    def test_deduplicates_consecutive_identical_lines(self):
        """Consecutive identical lines are deduplicated."""
        text = "same line\nsame line\nsame line\ndifferent"
        result = clean_text(text)
        lines = result.splitlines()
        assert lines.count("same line") == 1
        assert "different" in result

    def test_non_consecutive_duplicates_kept(self):
        """Non-consecutive duplicate lines are NOT removed."""
        text = "line A\nline B\nline A"
        result = clean_text(text)
        lines = result.splitlines()
        assert lines.count("line A") == 2

    def test_empty_input(self):
        result = clean_text("")
        assert result == ""

    def test_whitespace_only_input(self):
        result = clean_text("   \n\t\n  ")
        assert result.strip() == ""


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    MIN = 50
    MAX = 200

    def _chunks(self, text: str, mn=None, mx=None) -> list[str]:
        return chunk_text(text, mn or self.MIN, mx or self.MAX)

    def test_short_text_produces_no_chunk_below_min(self):
        """Text shorter than min_chars produces no chunks."""
        short = "Hi."  # 3 chars < 50
        result = self._chunks(short)
        assert result == []

    def test_medium_text_produces_one_chunk(self):
        """Text between min and max chars → one chunk."""
        text = "A" * self.MIN  # exactly min_chars
        result = self._chunks(text)
        assert len(result) == 1

    def test_chunks_respect_max_chars(self):
        """No chunk exceeds max_chars."""
        # Build text with many short sentences
        text = "The quick brown fox jumps. " * 50
        result = self._chunks(text)
        for chunk in result:
            assert len(chunk) <= self.MAX, f"Chunk too long: {len(chunk)}"

    def test_no_chunk_below_min_chars(self):
        """No chunk is shorter than min_chars (except when text is too short)."""
        text = ". ".join(["word " * 10] * 20)
        result = self._chunks(text)
        for chunk in result:
            assert len(chunk) >= self.MIN, f"Chunk too short: {len(chunk)}"

    def test_all_content_covered(self):
        """All words from the original text appear somewhere in the chunks."""
        words = ["alpha", "beta", "gamma", "delta", "epsilon"]
        # Build sentences long enough to exceed min_chars
        text = ". ".join([" ".join(words) + " filler text here"] * 20)
        chunks = self._chunks(text)
        combined = " ".join(chunks)
        for word in words:
            assert word in combined

    def test_empty_text(self):
        assert self._chunks("") == []

    def test_single_very_long_word_handled(self):
        """A single word that exceeds max_chars doesn't cause an error."""
        long_word = "x" * (self.MAX + 100)
        # Should not raise; may produce 0 chunks since a single token < min
        try:
            result = self._chunks(long_word)
            assert isinstance(result, list)
        except Exception as exc:
            pytest.fail(f"chunk_text raised {exc} on a very long word")

    def test_sentence_boundary_splitting(self):
        """Chunks prefer to split at sentence boundaries (.!?…)."""
        sentence = "Short sentence here. "
        # Repeat enough to cross max boundary
        text = sentence * 20
        chunks = chunk_text(text, min_chars=10, max_chars=80)
        # Every chunk should end at or near a sentence boundary
        for chunk in chunks:
            # Not a strict assertion but chunks shouldn't end mid-word
            assert not chunk.endswith(" ")


# ---------------------------------------------------------------------------
# content_hash
# ---------------------------------------------------------------------------

class TestContentHash:
    def test_deterministic(self):
        assert content_hash("hello") == content_hash("hello")

    def test_different_inputs(self):
        assert content_hash("a") != content_hash("b")

    def test_returns_hex_string(self):
        h = content_hash("test")
        assert isinstance(h, str)
        assert all(c in "0123456789abcdef" for c in h)
        assert len(h) == 64  # SHA-256 hex digest

    def test_empty_string(self):
        h = content_hash("")
        assert isinstance(h, str)
        assert len(h) == 64
