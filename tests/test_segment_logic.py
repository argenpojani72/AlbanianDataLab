"""
tests/test_segment_logic.py — Unit tests for audio segmentation helpers.

Tests the pure Python logic in scripts/segment_audio.py:
  - merge_segments()
  - split_long_segments()

No audio I/O or model loading required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from segment_audio import merge_segments, split_long_segments


# ---------------------------------------------------------------------------
# merge_segments
# ---------------------------------------------------------------------------

class TestMergeSegments:
    def test_empty(self):
        assert merge_segments([], merge_gap_sec=0.3) == []

    def test_single_segment_unchanged(self):
        segs = [(0.0, 1.0)]
        result = merge_segments(segs, merge_gap_sec=0.3)
        assert result == [(0.0, 1.0)]

    def test_segments_within_gap_merged(self):
        """Two segments separated by ≤ merge_gap are merged into one."""
        segs = [(0.0, 1.0), (1.2, 2.5)]  # gap = 0.2 < 0.3
        result = merge_segments(segs, merge_gap_sec=0.3)
        assert len(result) == 1
        assert result[0] == (0.0, 2.5)

    def test_segments_outside_gap_not_merged(self):
        """Two segments separated by > merge_gap stay separate."""
        segs = [(0.0, 1.0), (1.5, 2.5)]  # gap = 0.5 > 0.3
        result = merge_segments(segs, merge_gap_sec=0.3)
        assert len(result) == 2
        assert result[0] == (0.0, 1.0)
        assert result[1] == (1.5, 2.5)

    def test_exact_gap_boundary_merged(self):
        """Segments separated by a gap clearly within merge_gap_sec are merged."""
        segs = [(0.0, 1.0), (1.25, 2.0)]  # gap = 0.25 < 0.3 → should merge
        result = merge_segments(segs, merge_gap_sec=0.3)
        assert len(result) == 1

    def test_chain_merging(self):
        """Multiple consecutive close segments are all merged into one."""
        segs = [(0.0, 1.0), (1.1, 2.0), (2.1, 3.0)]  # all gaps 0.1 < 0.3
        result = merge_segments(segs, merge_gap_sec=0.3)
        assert len(result) == 1
        assert result[0] == (0.0, 3.0)

    def test_mixed_merge_and_keep(self):
        """Some segments merge, others remain separate."""
        segs = [
            (0.0, 1.0),   # gap 0.1 → merge with next
            (1.1, 2.0),   # merged with prev; gap 1.0 → keep separate
            (3.0, 4.0),   # gap 0.1 → merge with next
            (4.1, 5.0),
        ]
        result = merge_segments(segs, merge_gap_sec=0.3)
        assert len(result) == 2
        assert result[0] == (0.0, 2.0)
        assert result[1] == (3.0, 5.0)

    def test_zero_gap_threshold(self):
        """With merge_gap=0 only adjacent (touching) segments merge."""
        segs = [(0.0, 1.0), (1.0, 2.0), (2.5, 3.0)]
        result = merge_segments(segs, merge_gap_sec=0.0)
        # First two touch (gap = 0 ≤ 0), last has gap 0.5 > 0
        assert len(result) == 2
        assert result[0] == (0.0, 2.0)
        assert result[1] == (2.5, 3.0)

    def test_preserves_order(self):
        """Merged output maintains temporal order."""
        segs = [(i, i + 0.5) for i in range(10)]
        result = merge_segments(segs, merge_gap_sec=1.0)
        for i in range(1, len(result)):
            assert result[i][0] >= result[i - 1][1]


# ---------------------------------------------------------------------------
# split_long_segments
# ---------------------------------------------------------------------------

class TestSplitLongSegments:
    def test_empty(self):
        assert split_long_segments([], max_duration_sec=20.0) == []

    def test_short_segment_unchanged(self):
        segs = [(0.0, 5.0)]
        result = split_long_segments(segs, max_duration_sec=20.0)
        assert result == [(0.0, 5.0)]

    def test_exactly_max_not_split(self):
        """A segment of exactly max_duration_sec is NOT split."""
        segs = [(0.0, 20.0)]
        result = split_long_segments(segs, max_duration_sec=20.0)
        assert result == [(0.0, 20.0)]

    def test_long_segment_split_in_two(self):
        """A segment longer than max is split into two halves."""
        segs = [(0.0, 30.0)]  # > 20
        result = split_long_segments(segs, max_duration_sec=20.0)
        assert len(result) == 2
        assert result[0] == (0.0, 15.0)
        assert result[1] == (15.0, 30.0)

    def test_very_long_segment_split_recursively(self):
        """A very long segment is recursively split until all pieces ≤ max."""
        segs = [(0.0, 100.0)]
        result = split_long_segments(segs, max_duration_sec=20.0)
        for start, end in result:
            assert (end - start) <= 20.0

    def test_total_duration_preserved(self):
        """Total duration is the same before and after splitting."""
        segs = [(0.0, 90.0), (100.0, 160.0)]
        original_total = sum(e - s for s, e in segs)
        result = split_long_segments(segs, max_duration_sec=20.0)
        result_total = sum(e - s for s, e in result)
        assert abs(result_total - original_total) < 1e-9

    def test_multiple_segments_some_long(self):
        """Only long segments are split; short ones stay intact."""
        segs = [(0.0, 5.0), (10.0, 50.0), (60.0, 65.0)]
        result = split_long_segments(segs, max_duration_sec=20.0)
        for start, end in result:
            assert (end - start) <= 20.0
        # The short segments (5s, 5s) are unchanged somewhere in result
        short_durations = {round(e - s, 6) for s, e in result if (e - s) <= 5.0}
        assert 5.0 in short_durations

    def test_split_count_for_exact_double(self):
        """A segment of exactly 2× max splits into exactly 2 pieces."""
        max_dur = 10.0
        segs = [(0.0, 20.0)]  # exactly 2× max
        result = split_long_segments(segs, max_duration_sec=max_dur)
        # 20 > 10 → splits into (0, 10) and (10, 20)
        assert len(result) == 2

    def test_midpoints_are_contiguous(self):
        """After splitting, each piece ends where the next begins (no gaps)."""
        segs = [(0.0, 100.0)]
        result = split_long_segments(segs, max_duration_sec=20.0)
        # Sort by start time (should already be sorted)
        result = sorted(result, key=lambda x: x[0])
        for i in range(1, len(result)):
            assert abs(result[i][0] - result[i - 1][1]) < 1e-9
