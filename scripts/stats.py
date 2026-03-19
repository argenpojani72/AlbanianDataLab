"""
stats.py — Dataset statistics report for AlbanianDataFactory.

Reads the manifests and prints a human-readable summary of the current
state of the dataset: how many audio sources, segments, transcribed pairs,
text chunks, total audio duration, and review coverage.

Usage:
    python scripts/stats.py [--json] [--out PATH]

Options:
    --json       Output statistics as a JSON file instead of a text table.
    --out PATH   File path to write the report to (defaults to stdout).

Examples:
    python scripts/stats.py
    python scripts/stats.py --json
    python scripts/stats.py --out reports/dataset_stats.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    ensure_dirs,
    load_config,
    load_manifest,
    resolve_root,
    setup_logger,
)


# ---------------------------------------------------------------------------
# Stat collectors
# ---------------------------------------------------------------------------

def _safe_float(val) -> float:
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return 0.0


def audio_source_stats(manifests_dir: Path) -> dict:
    df = load_manifest(manifests_dir / "audio_manifest.csv")
    total = len(df)
    with_clean = int(
        df["clean_audio_path"].astype(str).str.strip().ne("").sum()
    ) if "clean_audio_path" in df.columns else 0
    return {"total_sources": total, "sources_with_clean_audio": with_clean}


def segment_stats(manifests_dir: Path) -> dict:
    df = load_manifest(manifests_dir / "audio_segments_manifest.csv")
    total = len(df)
    total_duration = 0.0
    if "duration_sec" in df.columns:
        total_duration = df["duration_sec"].apply(_safe_float).sum()
    return {
        "total_segments": total,
        "total_duration_sec": round(total_duration, 2),
        "total_duration_hours": round(total_duration / 3600, 4),
    }


def transcription_stats(manifests_dir: Path) -> dict:
    df = load_manifest(manifests_dir / "audio_text_pairs.csv")
    total = len(df)
    if total == 0:
        return {
            "total_pairs": 0,
            "transcribed": 0,
            "errors": 0,
            "by_status": {},
        }
    transcribed = int(
        df["raw_transcript"].astype(str).str.strip().ne("").sum()
    ) if "raw_transcript" in df.columns else 0
    errors = int(
        df["transcription_error"].astype(str).str.strip().ne("").sum()
    ) if "transcription_error" in df.columns else 0
    by_status: dict[str, int] = {}
    if "review_status" in df.columns:
        by_status = df["review_status"].value_counts().to_dict()
        by_status = {str(k): int(v) for k, v in by_status.items()}
    return {
        "total_pairs": total,
        "transcribed": transcribed,
        "errors": errors,
        "by_status": by_status,
    }


def text_stats(manifests_dir: Path) -> dict:
    df = load_manifest(manifests_dir / "text_manifest.csv")
    total = len(df)
    if total == 0:
        return {
            "total_chunks": 0,
            "total_chars": 0,
            "total_words_approx": 0,
            "unique_sources": 0,
            "by_status": {},
        }
    total_chars = int(
        df["chunk_chars"].apply(_safe_float).sum()
    ) if "chunk_chars" in df.columns else 0
    unique_sources = int(
        df["source_path"].nunique()
    ) if "source_path" in df.columns else 0
    by_status: dict[str, int] = {}
    if "review_status" in df.columns:
        by_status = df["review_status"].value_counts().to_dict()
        by_status = {str(k): int(v) for k, v in by_status.items()}
    return {
        "total_chunks": total,
        "total_chars": total_chars,
        "total_words_approx": total_chars // 5,
        "unique_sources": unique_sources,
        "by_status": by_status,
    }


def collect_all_stats(manifests_dir: Path) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audio_sources": audio_source_stats(manifests_dir),
        "audio_segments": segment_stats(manifests_dir),
        "asr_pairs": transcription_stats(manifests_dir),
        "text_chunks": text_stats(manifests_dir),
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_text_report(stats: dict) -> str:
    lines = []
    lines.append("=" * 56)
    lines.append("  AlbanianDataFactory — Dataset Statistics")
    lines.append(f"  Generated: {stats['generated_at']}")
    lines.append("=" * 56)

    # Audio sources
    a = stats["audio_sources"]
    lines.append("\n── Audio Sources ──────────────────────────────────────")
    lines.append(f"  Total media files processed:  {a['total_sources']}")
    lines.append(f"  With clean audio:             {a['sources_with_clean_audio']}")

    # Segments
    s = stats["audio_segments"]
    hours = s["total_duration_hours"]
    minutes = s["total_duration_sec"] / 60
    lines.append("\n── Audio Segments ─────────────────────────────────────")
    lines.append(f"  Total segments:               {s['total_segments']}")
    lines.append(f"  Total duration:               {hours:.2f} h  ({minutes:.1f} min)")

    # ASR pairs
    p = stats["asr_pairs"]
    lines.append("\n── ASR Transcription Pairs ────────────────────────────")
    lines.append(f"  Total pairs:                  {p['total_pairs']}")
    lines.append(f"  Transcribed:                  {p['transcribed']}")
    lines.append(f"  Transcription errors:         {p['errors']}")
    if p["by_status"]:
        lines.append("  Review status breakdown:")
        for status, count in sorted(p["by_status"].items()):
            pct = 100 * count / p["total_pairs"] if p["total_pairs"] else 0
            lines.append(f"    {status:12s}  {count:6d}  ({pct:.1f}%)")

    # Text
    t = stats["text_chunks"]
    lines.append("\n── Text Chunks ─────────────────────────────────────────")
    lines.append(f"  Total chunks:                 {t['total_chunks']}")
    lines.append(f"  Total characters:             {t['total_chars']:,}")
    lines.append(f"  Approx. words:                {t['total_words_approx']:,}")
    lines.append(f"  Unique source files:          {t['unique_sources']}")
    if t["by_status"]:
        lines.append("  Review status breakdown:")
        for status, count in sorted(t["by_status"].items()):
            pct = 100 * count / t["total_chunks"] if t["total_chunks"] else 0
            lines.append(f"    {status:12s}  {count:6d}  ({pct:.1f}%)")

    lines.append("\n" + "=" * 56)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print dataset statistics for AlbanianDataFactory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output statistics as JSON.",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Write the report to a file instead of stdout.",
    )
    args = parser.parse_args()

    cfg = load_config()
    root = resolve_root()
    ensure_dirs(cfg)
    logger = setup_logger("stats")

    manifests_dir = root / cfg["paths"]["manifests"]
    stats = collect_all_stats(manifests_dir)

    if args.json:
        output = json.dumps(stats, indent=2, ensure_ascii=False)
    else:
        output = format_text_report(stats)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        logger.info("Stats report written to %s", out_path)
    else:
        print(output)


if __name__ == "__main__":
    main()
