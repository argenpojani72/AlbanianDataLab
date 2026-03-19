"""
extract_audio.py — Step 1 of the AlbanianDataFactory pipeline.

For every media file in raw_sources/audio_video/ that has not yet been
processed, extract a mono 16 kHz WAV file into raw_audio/ and register a row
in manifests/audio_manifest.csv.

Usage:
    python scripts/extract_audio.py [--force]

Options:
    --force   Re-extract even if the output WAV already exists.

Skip logic:
    A source file is skipped when its sample_id already appears in
    audio_manifest.csv AND the target WAV file exists, unless --force is given.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Allow running as a script from the project root
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

SUPPORTED_EXTENSIONS = {
    ".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv",
    ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".wav", ".opus",
}


def extract_audio_file(
    src: Path,
    dst: Path,
    sample_rate: int,
    channels: int,
    logger,
) -> bool:
    """
    Use ffmpeg to extract audio from *src* into a WAV file at *dst*.

    Returns True on success, False on failure.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vn",
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
    parser = argparse.ArgumentParser(description="Extract audio from media sources.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if output WAV already exists.",
    )
    args = parser.parse_args()

    cfg = load_config()
    root = resolve_root()
    ensure_dirs(cfg)
    logger = setup_logger("extract_audio")

    src_dir = root / cfg["paths"]["raw_sources_audio_video"]
    out_dir = root / cfg["paths"]["raw_audio"]
    manifest_path = root / cfg["paths"]["manifests"] / "audio_manifest.csv"

    sample_rate: int = cfg["audio"]["sample_rate"]
    channels: int = cfg["audio"]["channels"]

    # Collect source media files
    media_files = [
        p for p in src_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not media_files:
        logger.info("No media files found in %s — nothing to do.", src_dir)
        return

    logger.info("Found %d media file(s) in %s.", len(media_files), src_dir)

    existing_manifest = load_manifest(manifest_path)
    existing_ids: set[str] = (
        set(existing_manifest["sample_id"].astype(str))
        if not existing_manifest.empty and "sample_id" in existing_manifest.columns
        else set()
    )

    new_rows: list[dict] = []
    skipped = processed = failed = 0

    for src in sorted(media_files):
        rel = relative_to_root(src)
        sid = stable_id(rel)
        dst = out_dir / (src.stem + ".wav")

        # Skip logic
        if not args.force and sid in existing_ids and dst.exists():
            logger.debug("Skipping (already processed): %s", rel)
            skipped += 1
            continue

        logger.info("Extracting: %s → %s", rel, relative_to_root(dst))
        success = extract_audio_file(src, dst, sample_rate, channels, logger)

        status = "ok" if success else "error"
        if success:
            processed += 1
        else:
            failed += 1

        new_rows.append(
            {
                "sample_id": sid,
                "source_path": rel,
                "raw_audio_path": relative_to_root(dst),
                "sample_rate": sample_rate,
                "channels": channels,
                "status": status,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    if new_rows:
        df = pd.DataFrame(new_rows)
        upsert_manifest(df, manifest_path, key_col="sample_id")
        logger.info(
            "audio_manifest.csv updated — processed: %d, failed: %d, skipped: %d.",
            processed,
            failed,
            skipped,
        )
    else:
        logger.info("Nothing new to extract (skipped: %d).", skipped)


if __name__ == "__main__":
    main()
