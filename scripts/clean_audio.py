"""
clean_audio.py — Step 2 of the AlbanianDataFactory pipeline.

For every WAV file in raw_audio/ that does not yet have a cleaned counterpart
in clean_audio/, apply an ffmpeg filter chain (highpass, lowpass, afftdn,
loudnorm) and write the result as a WAV at the same sample rate.

Usage:
    python scripts/clean_audio.py [--force]

Options:
    --force   Re-clean even if a cleaned WAV already exists.

Skip logic:
    A file is skipped when  <stem>_clean.wav  already exists in clean_audio/,
    unless --force is given.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    ensure_dirs,
    load_config,
    load_manifest,
    relative_to_root,
    resolve_root,
    setup_logger,
    stable_id,
    upsert_manifest,
)


def clean_audio_file(
    src: Path,
    dst: Path,
    audio_filter: str,
    sample_rate: int,
    channels: int,
    logger,
) -> bool:
    """Apply ffmpeg filter chain to *src* and write result to *dst*."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-af", audio_filter,
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-f", "wav",
        str(dst),
    ]
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("ffmpeg failed for %s:\n%s", src, result.stderr[-2000:])
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean extracted audio files.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-clean even if cleaned WAV already exists.",
    )
    args = parser.parse_args()

    cfg = load_config()
    root = resolve_root()
    ensure_dirs(cfg)
    logger = setup_logger("clean_audio")

    raw_dir = root / cfg["paths"]["raw_audio"]
    out_dir = root / cfg["paths"]["clean_audio"]
    manifest_path = root / cfg["paths"]["manifests"] / "audio_manifest.csv"

    audio_filter: str = cfg["audio"]["clean_filter"]
    sample_rate: int = cfg["audio"]["sample_rate"]
    channels: int = cfg["audio"]["channels"]

    raw_wavs = sorted(raw_dir.glob("*.wav"))
    if not raw_wavs:
        logger.info("No WAV files found in %s — nothing to do.", raw_dir)
        return

    logger.info("Found %d WAV file(s) to clean.", len(raw_wavs))

    # Load manifest to be able to update clean_audio_path column
    existing_manifest = load_manifest(manifest_path)

    new_rows: list[dict] = []
    skipped = processed = failed = 0

    for src in raw_wavs:
        dst = out_dir / (src.stem + "_clean.wav")

        if not args.force and dst.exists():
            logger.debug("Skipping (already cleaned): %s", src.name)
            skipped += 1
            continue

        logger.info("Cleaning: %s → %s", relative_to_root(src), relative_to_root(dst))
        success = clean_audio_file(src, dst, audio_filter, sample_rate, channels, logger)

        sid = _find_sample_id(existing_manifest, relative_to_root(src))
        status = "clean_ok" if success else "clean_error"
        if success:
            processed += 1
        else:
            failed += 1

        new_rows.append(
            {
                "sample_id": sid,
                "source_path": _find_source_path(existing_manifest, sid),
                "raw_audio_path": relative_to_root(src),
                "clean_audio_path": relative_to_root(dst) if success else "",
                "sample_rate": sample_rate,
                "channels": channels,
                "status": status,
                "cleaned_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    if new_rows:
        df = pd.DataFrame(new_rows)
        upsert_manifest(df, manifest_path, key_col="sample_id")
        logger.info(
            "audio_manifest.csv updated — cleaned: %d, failed: %d, skipped: %d.",
            processed,
            failed,
            skipped,
        )
    else:
        logger.info("Nothing new to clean (skipped: %d).", skipped)


def _find_source_path(manifest: pd.DataFrame, sid: str) -> str:
    """Look up source_path for *sid* in the manifest; return empty string if not found."""
    if manifest.empty or "sample_id" not in manifest.columns:
        return ""
    row = manifest[manifest["sample_id"].astype(str) == sid]
    if row.empty:
        return ""
    return str(row.iloc[0].get("source_path", ""))


def _find_sample_id(manifest: pd.DataFrame, raw_audio_path_rel: str) -> str:
    """
    Look up sample_id for the given raw_audio_path in the manifest.

    Falls back to computing a stable_id from the raw_audio_path if not found,
    so the column is always populated.
    """
    if not manifest.empty and "raw_audio_path" in manifest.columns and "sample_id" in manifest.columns:
        row = manifest[manifest["raw_audio_path"].astype(str) == raw_audio_path_rel]
        if not row.empty:
            return str(row.iloc[0]["sample_id"])
    # Fallback: derive from raw_audio_path (may not match extract_audio's id)
    return stable_id(raw_audio_path_rel)


if __name__ == "__main__":
    main()
