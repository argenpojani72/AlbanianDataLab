"""
segment_audio.py — Step 3 of the AlbanianDataFactory pipeline.

For every clean WAV in clean_audio/ whose segment folder doesn't yet exist (or
is empty), run Silero VAD to detect speech regions, merge nearby regions, split
long regions, and write individual segment WAVs into
audio_segments/<source_stem>/.

New rows are appended to manifests/audio_segments_manifest.csv.

Usage:
    python scripts/segment_audio.py [--force]

Options:
    --force   Re-segment even if the segment folder already contains WAVs.

Dependencies:
    torch, torchaudio  (for Silero VAD model and WAV I/O)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    ensure_dirs,
    load_config,
    relative_to_root,
    resolve_root,
    setup_logger,
    stable_id,
    upsert_manifest,
)


def load_silero_vad():
    """Load the Silero VAD model from torchhub (cached after first download)."""
    import torch
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    return model, utils


def get_speech_timestamps(
    wav_path: Path,
    model,
    vad_utils,
    threshold: float,
    window_size_samples: int,
    sample_rate: int,
) -> List[Tuple[float, float]]:
    """
    Run Silero VAD on *wav_path* and return a list of (start_sec, end_sec) tuples.
    """
    import torch
    import torchaudio

    waveform, sr = torchaudio.load(str(wav_path))
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    waveform = waveform.mean(dim=0)  # to mono

    get_ts = vad_utils[0]  # get_speech_timestamps function
    speech_timestamps = get_ts(
        waveform,
        model,
        threshold=threshold,
        sampling_rate=sample_rate,
        window_size_samples=window_size_samples,
        return_seconds=True,
    )
    return [(ts["start"], ts["end"]) for ts in speech_timestamps]


def merge_segments(
    timestamps: List[Tuple[float, float]],
    merge_gap_sec: float,
) -> List[Tuple[float, float]]:
    """Merge consecutive segments whose gap is smaller than *merge_gap_sec*."""
    if not timestamps:
        return []
    merged = [timestamps[0]]
    for start, end in timestamps[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= merge_gap_sec:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def split_long_segments(
    timestamps: List[Tuple[float, float]],
    max_duration_sec: float,
) -> List[Tuple[float, float]]:
    """Split any segment longer than *max_duration_sec* into equal halves recursively."""
    result = []
    for start, end in timestamps:
        duration = end - start
        if duration <= max_duration_sec:
            result.append((start, end))
        else:
            mid = start + duration / 2
            result.extend(
                split_long_segments([(start, mid), (mid, end)], max_duration_sec)
            )
    return result


def save_segment(
    wav_path: Path,
    dst: Path,
    start_sec: float,
    end_sec: float,
    sample_rate: int,
) -> None:
    """Write a slice of *wav_path* between *start_sec* and *end_sec* to *dst*."""
    import torch
    import torchaudio

    waveform, sr = torchaudio.load(str(wav_path))
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    waveform = waveform.mean(dim=0, keepdim=True)  # mono

    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)
    end_sample = min(end_sample, waveform.shape[-1])

    segment = waveform[:, start_sample:end_sample]
    dst.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(dst), segment, sample_rate)


def segment_file(
    src: Path,
    seg_dir: Path,
    cfg: dict,
    model,
    vad_utils,
    logger,
) -> List[dict]:
    """
    Segment a single clean WAV file.  Returns list of manifest row dicts.
    """
    sample_rate: int = cfg["audio"]["sample_rate"]
    seg_cfg = cfg["segmentation"]
    threshold: float = seg_cfg["vad_threshold"]
    window_size: int = seg_cfg["vad_window_size_samples"]
    min_dur: float = seg_cfg["min_duration_sec"]
    max_dur: float = seg_cfg["max_duration_sec"]
    merge_gap: float = seg_cfg["merge_gap_sec"]

    logger.info("Running VAD on: %s", src.name)
    timestamps = get_speech_timestamps(src, model, vad_utils, threshold, window_size, sample_rate)
    timestamps = merge_segments(timestamps, merge_gap)
    timestamps = split_long_segments(timestamps, max_dur)

    rows = []
    source_rel = relative_to_root(src)

    for idx, (start, end) in enumerate(timestamps):
        duration = end - start
        if duration < min_dur:
            logger.debug("Skipping short segment %.2fs at %.2f", duration, start)
            continue

        seg_name = f"{src.stem}_seg{idx:04d}.wav"
        seg_path = seg_dir / seg_name
        seg_id = stable_id(f"{source_rel}:{start:.4f}:{end:.4f}")

        save_segment(src, seg_path, start, end, sample_rate)

        rows.append(
            {
                "segment_id": seg_id,
                "segment_path": relative_to_root(seg_path),
                "source_clean_audio": source_rel,
                "start_sec": f"{start:.4f}",
                "end_sec": f"{end:.4f}",
                "duration_sec": f"{duration:.4f}",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment clean audio files using Silero VAD.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-segment even if segment folder already has WAVs.",
    )
    args = parser.parse_args()

    cfg = load_config()
    root = resolve_root()
    ensure_dirs(cfg)
    logger = setup_logger("segment_audio")

    clean_dir = root / cfg["paths"]["clean_audio"]
    seg_base = root / cfg["paths"]["audio_segments"]
    manifest_path = root / cfg["paths"]["manifests"] / "audio_segments_manifest.csv"

    clean_wavs = sorted(clean_dir.glob("*_clean.wav"))
    if not clean_wavs:
        logger.info("No clean WAVs found in %s — nothing to do.", clean_dir)
        return

    logger.info("Found %d clean WAV(s) to segment.", len(clean_wavs))
    logger.info("Loading Silero VAD model…")
    model, vad_utils = load_silero_vad()

    all_new_rows: list[dict] = []
    skipped = 0

    for src in clean_wavs:
        stem = src.stem.replace("_clean", "")
        seg_dir = seg_base / stem

        if not args.force and seg_dir.exists() and any(seg_dir.glob("*.wav")):
            logger.info("Skipping (already segmented): %s", src.name)
            skipped += 1
            continue

        rows = segment_file(src, seg_dir, cfg, model, vad_utils, logger)
        logger.info("  → %d segment(s) written for %s.", len(rows), src.name)
        all_new_rows.extend(rows)

    if all_new_rows:
        import pandas as pd
        df = pd.DataFrame(all_new_rows)
        upsert_manifest(df, manifest_path, key_col="segment_id")
        logger.info(
            "audio_segments_manifest.csv updated with %d new row(s).", len(all_new_rows)
        )
    else:
        logger.info("No new segments produced (skipped: %d).", skipped)


if __name__ == "__main__":
    main()
