"""
tests/test_utils.py — Unit tests for scripts/utils.py helpers.

Tests pure utility functions that have no external I/O dependencies
(stable_id, load_manifest on missing/existing files, upsert_manifest).
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import sys

# Make scripts/ importable without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from utils import load_manifest, stable_id, upsert_manifest


# ---------------------------------------------------------------------------
# stable_id
# ---------------------------------------------------------------------------

class TestStableId:
    def test_deterministic(self):
        """Same input always produces the same ID."""
        assert stable_id("hello") == stable_id("hello")

    def test_different_inputs_differ(self):
        """Different inputs produce different IDs."""
        assert stable_id("foo") != stable_id("bar")

    def test_default_length_12(self):
        """Default output length is 12 hex characters."""
        result = stable_id("test")
        assert len(result) == 12
        assert all(c in "0123456789abcdef" for c in result)

    def test_custom_length(self):
        """Custom *length* argument is respected."""
        assert len(stable_id("x", length=8)) == 8
        assert len(stable_id("x", length=20)) == 20

    def test_hex_only(self):
        """Output contains only hex characters."""
        sid = stable_id("Albanian 2024 test!")
        assert all(c in "0123456789abcdef" for c in sid)

    def test_empty_string(self):
        """Empty string is handled without error and gives consistent output."""
        result = stable_id("")
        assert isinstance(result, str)
        assert len(result) == 12


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------

class TestLoadManifest:
    def test_missing_file_returns_empty_df(self, tmp_path):
        """Missing CSV returns an empty DataFrame with no columns."""
        df = load_manifest(tmp_path / "does_not_exist.csv")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_existing_csv_loaded(self, tmp_path):
        """Existing CSV is loaded correctly."""
        csv_path = tmp_path / "manifest.csv"
        csv_path.write_text("id,name\n1,foo\n2,bar\n", encoding="utf-8")

        df = load_manifest(csv_path)
        assert list(df.columns) == ["id", "name"]
        assert len(df) == 2
        assert df.iloc[0]["id"] == "1"
        assert df.iloc[1]["name"] == "bar"

    def test_all_columns_are_strings(self, tmp_path):
        """Columns are loaded as a string-like dtype (no numeric coercion)."""
        csv_path = tmp_path / "m.csv"
        csv_path.write_text("id,value\n001,42\n", encoding="utf-8")
        df = load_manifest(csv_path)
        # pandas ≥ 2.0 may return StringDtype; older versions return object.
        # Either way the values must be Python strings.
        assert str(df.iloc[0]["id"]) == "001"   # leading zero preserved
        assert str(df.iloc[0]["value"]) == "42"


# ---------------------------------------------------------------------------
# upsert_manifest
# ---------------------------------------------------------------------------

class TestUpsertManifest:
    def test_creates_new_file(self, tmp_path):
        """upsert_manifest creates the CSV when it doesn't exist."""
        path = tmp_path / "new.csv"
        df = pd.DataFrame([{"id": "a1", "value": "hello"}])
        result = upsert_manifest(df, path, key_col="id")
        assert path.exists()
        assert len(result) == 1
        assert result.iloc[0]["value"] == "hello"

    def test_appends_new_rows(self, tmp_path):
        """New rows with unseen keys are appended."""
        path = tmp_path / "m.csv"
        initial = pd.DataFrame([{"id": "a1", "value": "first"}])
        upsert_manifest(initial, path, key_col="id")

        addition = pd.DataFrame([{"id": "a2", "value": "second"}])
        result = upsert_manifest(addition, path, key_col="id")

        assert len(result) == 2
        values = set(result["value"])
        assert values == {"first", "second"}

    def test_updates_existing_row(self, tmp_path):
        """A row with an existing key is replaced by the new version."""
        path = tmp_path / "m.csv"
        initial = pd.DataFrame([{"id": "k1", "value": "old"}])
        upsert_manifest(initial, path, key_col="id")

        update = pd.DataFrame([{"id": "k1", "value": "new"}])
        result = upsert_manifest(update, path, key_col="id")

        assert len(result) == 1
        assert result.iloc[0]["value"] == "new"

    def test_idempotent(self, tmp_path):
        """Upserting the same rows twice yields the same manifest."""
        path = tmp_path / "m.csv"
        rows = pd.DataFrame([{"id": "x", "value": "v"}])
        upsert_manifest(rows, path, key_col="id")
        result = upsert_manifest(rows, path, key_col="id")
        assert len(result) == 1

    def test_new_columns_added_to_existing(self, tmp_path):
        """If new_rows has extra columns not in the existing manifest, they are added."""
        path = tmp_path / "m.csv"
        initial = pd.DataFrame([{"id": "a", "col1": "v1"}])
        upsert_manifest(initial, path, key_col="id")

        extra = pd.DataFrame([{"id": "b", "col1": "v2", "col2": "extra"}])
        result = upsert_manifest(extra, path, key_col="id")
        assert "col2" in result.columns

    def test_creates_parent_directories(self, tmp_path):
        """Parent directories are created if they do not exist."""
        deep_path = tmp_path / "a" / "b" / "c" / "manifest.csv"
        df = pd.DataFrame([{"id": "1", "x": "y"}])
        upsert_manifest(df, deep_path, key_col="id")
        assert deep_path.exists()
