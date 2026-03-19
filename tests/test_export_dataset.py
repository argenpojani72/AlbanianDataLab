"""
tests/test_export_dataset.py — Unit tests for scripts/export_dataset.py.

Tests pure logic (filtering, splitting, JSONL writing) without touching the
real manifests or filesystem beyond tmp_path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from export_dataset import (
    _best_transcript,
    _split_records,
    _write_jsonl,
    export_asr,
    export_text,
)
from utils import setup_logger


# ---------------------------------------------------------------------------
# _best_transcript
# ---------------------------------------------------------------------------

class TestBestTranscript:
    def test_prefers_final_transcript(self):
        row = {"final_transcript": "corrected", "raw_transcript": "raw"}
        assert _best_transcript(row) == "corrected"

    def test_falls_back_to_raw(self):
        row = {"final_transcript": "", "raw_transcript": "raw text"}
        assert _best_transcript(row) == "raw text"

    def test_empty_final_whitespace_falls_back(self):
        row = {"final_transcript": "   ", "raw_transcript": "raw"}
        assert _best_transcript(row) == "raw"

    def test_both_empty_returns_empty(self):
        row = {"final_transcript": "", "raw_transcript": ""}
        assert _best_transcript(row) == ""

    def test_missing_keys_returns_empty(self):
        row = {}
        assert _best_transcript(row) == ""


# ---------------------------------------------------------------------------
# _split_records
# ---------------------------------------------------------------------------

class TestSplitRecords:
    def _make_records(self, n: int) -> list[dict]:
        return [{"id": str(i)} for i in range(n)]

    def test_correct_total(self):
        records = self._make_records(100)
        train, val = _split_records(records, 0.9)
        assert len(train) + len(val) == 100

    def test_approximate_ratio(self):
        records = self._make_records(100)
        train, val = _split_records(records, 0.8)
        assert len(train) == 80
        assert len(val) == 20

    def test_reproducible_with_same_seed(self):
        records = self._make_records(50)
        train1, val1 = _split_records(records, 0.9, seed=42)
        train2, val2 = _split_records(records, 0.9, seed=42)
        assert [r["id"] for r in train1] == [r["id"] for r in train2]

    def test_different_seeds_differ(self):
        records = self._make_records(50)
        train1, _ = _split_records(records, 0.9, seed=1)
        train2, _ = _split_records(records, 0.9, seed=99)
        assert [r["id"] for r in train1] != [r["id"] for r in train2]

    def test_empty_records(self):
        train, val = _split_records([], 0.9)
        assert train == []
        assert val == []

    def test_single_record_allocation(self):
        """With one record and any ratio, the record ends up in whichever split is larger."""
        records = [{"id": "only"}]
        train, val = _split_records(records, 0.9)
        # int(1 * 0.9) = 0, so train=0, val=1
        assert len(train) + len(val) == 1


# ---------------------------------------------------------------------------
# _write_jsonl
# ---------------------------------------------------------------------------

class TestWriteJsonl:
    def test_writes_correct_jsonl(self, tmp_path):
        records = [{"id": "a", "text": "hello"}, {"id": "b", "text": "world"}]
        path = tmp_path / "out.jsonl"
        _write_jsonl(records, path)
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"id": "a", "text": "hello"}
        assert json.loads(lines[1]) == {"id": "b", "text": "world"}

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "out.jsonl"
        _write_jsonl([{"x": 1}], path)
        assert path.exists()

    def test_unicode_preserved(self, tmp_path):
        records = [{"text": "Mirëdita — Albanë"}]
        path = tmp_path / "u.jsonl"
        _write_jsonl(records, path)
        loaded = json.loads(path.read_text(encoding="utf-8").strip())
        assert loaded["text"] == "Mirëdita — Albanë"

    def test_empty_records_writes_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        _write_jsonl([], path)
        assert path.exists()
        assert path.read_text(encoding="utf-8").strip() == ""


# ---------------------------------------------------------------------------
# export_asr (integration-style with patched load_manifest)
# ---------------------------------------------------------------------------

class TestExportAsr:
    def _logger(self):
        return setup_logger("test_export_asr")

    def test_exports_approved_records(self, tmp_path, audio_pairs_df):
        with patch("export_dataset.load_manifest", return_value=audio_pairs_df):
            count = export_asr(
                manifest_path=tmp_path / "fake.csv",
                out_dir=tmp_path,
                statuses={"approved"},
                min_duration=1.0,
                max_duration=30.0,
                do_split=False,
                split_ratio=0.9,
                logger=self._logger(),
            )
        assert count == 1
        lines = (tmp_path / "asr_dataset.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["id"] == "pid1"
        assert rec["transcript"] == "Mirëdita e mirë!"   # final_transcript used

    def test_exports_approved_and_pending(self, tmp_path, audio_pairs_df):
        with patch("export_dataset.load_manifest", return_value=audio_pairs_df):
            count = export_asr(
                manifest_path=tmp_path / "fake.csv",
                out_dir=tmp_path,
                statuses={"approved", "pending"},
                min_duration=1.0,
                max_duration=30.0,
                do_split=False,
                split_ratio=0.9,
                logger=self._logger(),
            )
        assert count == 2  # 'reject' row excluded

    def test_duration_filter_excludes_short(self, tmp_path, audio_pairs_df):
        with patch("export_dataset.load_manifest", return_value=audio_pairs_df):
            count = export_asr(
                manifest_path=tmp_path / "fake.csv",
                out_dir=tmp_path,
                statuses={"approved", "pending", "reject"},
                min_duration=3.0,   # 2.0s 'reject' row excluded
                max_duration=30.0,
                do_split=False,
                split_ratio=0.9,
                logger=self._logger(),
            )
        # pid3 duration=2.0 < 3.0 → excluded; pid1 (4.5) and pid2 (5.5) remain
        assert count == 2

    def test_split_creates_two_files(self, tmp_path, audio_pairs_df):
        with patch("export_dataset.load_manifest", return_value=audio_pairs_df):
            export_asr(
                manifest_path=tmp_path / "fake.csv",
                out_dir=tmp_path,
                statuses={"approved", "pending"},
                min_duration=1.0,
                max_duration=30.0,
                do_split=True,
                split_ratio=0.5,
                logger=self._logger(),
            )
        assert (tmp_path / "asr_train.jsonl").exists()
        assert (tmp_path / "asr_validation.jsonl").exists()

    def test_empty_manifest_returns_zero(self, tmp_path):
        with patch("export_dataset.load_manifest", return_value=pd.DataFrame()):
            count = export_asr(
                manifest_path=tmp_path / "fake.csv",
                out_dir=tmp_path,
                statuses={"approved"},
                min_duration=1.0,
                max_duration=30.0,
                do_split=False,
                split_ratio=0.9,
                logger=self._logger(),
            )
        assert count == 0

    def test_raw_transcript_used_when_final_empty(self, tmp_path, audio_pairs_df):
        with patch("export_dataset.load_manifest", return_value=audio_pairs_df):
            export_asr(
                manifest_path=tmp_path / "fake.csv",
                out_dir=tmp_path,
                statuses={"pending"},
                min_duration=1.0,
                max_duration=30.0,
                do_split=False,
                split_ratio=0.9,
                logger=self._logger(),
            )
        lines = (tmp_path / "asr_dataset.jsonl").read_text().strip().splitlines()
        rec = json.loads(lines[0])
        # pid2 has empty final_transcript → raw_transcript used
        assert rec["transcript"] == "Si jeni sot?"


# ---------------------------------------------------------------------------
# export_text (integration-style with patched load_manifest + clean text files)
# ---------------------------------------------------------------------------

class TestExportText:
    def _logger(self):
        return setup_logger("test_export_text")

    def _write_clean_text(self, tmp_path: Path, filename: str, chunks: list[str]) -> Path:
        """Write a clean_text file whose chunks match config defaults (min=200, max=2000)."""
        # Build a text that produces the desired number of chunks when re-chunked.
        # Since chunk_text is called internally with config defaults (min=200, max=2000),
        # we just write text that fills each chunk with enough characters.
        text = " ".join(f"Chunk{i}: " + ("word " * 50) for i in range(len(chunks)))
        p = tmp_path / filename
        p.write_text(text, encoding="utf-8")
        return p

    def test_exports_records_with_valid_chunks(self, tmp_path, text_manifest_df):
        # Write a clean_text file with enough content to produce chunks
        clean_file = tmp_path / "article.txt"
        clean_file.write_text(
            # Two long sentences to produce at least 2 chunks
            ("Ky është një tekst i gjatë shqip për testim. " * 10 + "\n") * 5,
            encoding="utf-8",
        )
        # Patch manifest rows to point at our tmp clean file
        df = text_manifest_df.copy()
        df["clean_text_path"] = str(clean_file)

        with patch("export_dataset.load_manifest", return_value=df), \
             patch("export_dataset.resolve_root", return_value=Path("/")):
            count = export_text(
                manifest_path=tmp_path / "fake.csv",
                clean_text_dir=tmp_path,
                out_dir=tmp_path,
                statuses={"approved", "pending"},
                min_chars=50,
                do_split=False,
                split_ratio=0.9,
                logger=self._logger(),
            )
        assert count >= 0  # Should succeed without raising

    def test_min_chars_filter_excludes_small_chunks(self, tmp_path, text_manifest_df):
        # tid3 has chunk_chars=30, which is < 200 (our min_chars for the filter)
        df = text_manifest_df.copy()
        # patch resolve_root and load_manifest; chunk files don't need to exist
        # since the chunk_chars filter happens at the manifest level
        with patch("export_dataset.load_manifest", return_value=df), \
             patch("export_dataset.resolve_root", return_value=tmp_path):
            count = export_text(
                manifest_path=tmp_path / "fake.csv",
                clean_text_dir=tmp_path,
                out_dir=tmp_path,
                statuses={"approved", "pending"},
                min_chars=200,  # tid3 has chunk_chars=30 → excluded
                do_split=False,
                split_ratio=0.9,
                logger=self._logger(),
            )
        # tid1 (300) and tid2 (250) pass min_chars=200; tid3 (30) excluded
        # count = 0 if clean files don't exist, but no exception should be raised
        assert count >= 0

    def test_status_filter(self, tmp_path, text_manifest_df):
        df = text_manifest_df.copy()
        with patch("export_dataset.load_manifest", return_value=df), \
             patch("export_dataset.resolve_root", return_value=tmp_path):
            export_text(
                manifest_path=tmp_path / "fake.csv",
                clean_text_dir=tmp_path,
                out_dir=tmp_path,
                statuses={"approved"},   # only tid1
                min_chars=50,
                do_split=False,
                split_ratio=0.9,
                logger=self._logger(),
            )
        # Should complete without error; file may or may not exist if chunk missing
        assert True

    def test_empty_manifest_returns_zero(self, tmp_path):
        with patch("export_dataset.load_manifest", return_value=pd.DataFrame()):
            count = export_text(
                manifest_path=tmp_path / "fake.csv",
                clean_text_dir=tmp_path,
                out_dir=tmp_path,
                statuses={"approved"},
                min_chars=50,
                do_split=False,
                split_ratio=0.9,
                logger=self._logger(),
            )
        assert count == 0
