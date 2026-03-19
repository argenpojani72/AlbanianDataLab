"""
tests/conftest.py — Shared pytest fixtures for AlbanianDataFactory tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make scripts/ importable in every test module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


# ---------------------------------------------------------------------------
# Manifest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def audio_pairs_df():
    """A small in-memory audio_text_pairs DataFrame for export tests."""
    return pd.DataFrame(
        [
            {
                "pair_id": "pid1",
                "segment_id": "sid1",
                "segment_path": "audio_segments/src/seg_0000.wav",
                "start_sec": "0.0",
                "end_sec": "4.5",
                "duration_sec": "4.5",
                "raw_transcript": "Mirëdita e mirë.",
                "final_transcript": "Mirëdita e mirë!",
                "review_status": "approved",
                "review_notes": "",
                "transcription_error": "",
                "model_name": "primusAI/whisper-large-v3-albanian",
                "created_at": "2024-01-01T00:00:00+00:00",
            },
            {
                "pair_id": "pid2",
                "segment_id": "sid2",
                "segment_path": "audio_segments/src/seg_0001.wav",
                "start_sec": "4.5",
                "end_sec": "10.0",
                "duration_sec": "5.5",
                "raw_transcript": "Si jeni sot?",
                "final_transcript": "",
                "review_status": "pending",
                "review_notes": "",
                "transcription_error": "",
                "model_name": "primusAI/whisper-large-v3-albanian",
                "created_at": "2024-01-01T00:00:00+00:00",
            },
            {
                "pair_id": "pid3",
                "segment_id": "sid3",
                "segment_path": "audio_segments/src/seg_0002.wav",
                "start_sec": "10.0",
                "end_sec": "12.0",
                "duration_sec": "2.0",
                "raw_transcript": "Ok.",
                "final_transcript": "",
                "review_status": "reject",
                "review_notes": "too_short",
                "transcription_error": "",
                "model_name": "primusAI/whisper-large-v3-albanian",
                "created_at": "2024-01-01T00:00:00+00:00",
            },
        ]
    )


@pytest.fixture()
def text_manifest_df():
    """A small in-memory text_manifest DataFrame for export tests."""
    return pd.DataFrame(
        [
            {
                "text_id": "tid1",
                "source_path": "raw_sources/text/article.txt",
                "raw_text_path": "raw_text/article.txt",
                "clean_text_path": "clean_text/article.txt",
                "chunk_index": "0",
                "chunk_chars": "300",
                "content_hash": "abc123",
                "domain": "news",
                "review_status": "approved",
                "review_notes": "",
                "created_at": "2024-01-01T00:00:00+00:00",
            },
            {
                "text_id": "tid2",
                "source_path": "raw_sources/text/article.txt",
                "raw_text_path": "raw_text/article.txt",
                "clean_text_path": "clean_text/article.txt",
                "chunk_index": "1",
                "chunk_chars": "250",
                "content_hash": "def456",
                "domain": "news",
                "review_status": "pending",
                "review_notes": "",
                "created_at": "2024-01-01T00:00:00+00:00",
            },
            {
                "text_id": "tid3",
                "source_path": "raw_sources/text/article.txt",
                "raw_text_path": "raw_text/article.txt",
                "clean_text_path": "clean_text/article.txt",
                "chunk_index": "2",
                "chunk_chars": "30",
                "content_hash": "ghi789",
                "domain": "news",
                "review_status": "pending",
                "review_notes": "too_short",
                "created_at": "2024-01-01T00:00:00+00:00",
            },
        ]
    )
