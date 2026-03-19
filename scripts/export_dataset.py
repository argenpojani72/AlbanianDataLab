"""
export_dataset.py — Step 6 of the AlbanianDataFactory pipeline.

Exports training-ready data from the manifests into two JSONL files:

  exports/asr_dataset.jsonl   — audio-text pairs (speech recognition)
  exports/text_dataset.jsonl  — text-only chunks (language modelling)

For each manifest the script filters rows by review_status, applies a
minimum-quality gate, and writes one JSON object per line.

Usage:
    python scripts/export_dataset.py [options]

Options:
    --status STATUS      review_status value(s) to include (default: approved pending).
                         Use "approved" to include only human-reviewed pairs.
    --min-duration SECS  Minimum segment duration for ASR export (default: 1.0).
    --max-duration SECS  Maximum segment duration for ASR export (default: 30.0).
    --min-chars N        Minimum characters per text chunk (default: 50).
    --out-dir PATH       Output directory (default: exports/).
    --split              Create train/validation splits (90/10 by default).
    --split-ratio RATIO  Fraction of data to use for training (default: 0.9).

Output schema — asr_dataset.jsonl:
    {
      "id": "<pair_id>",
      "audio_path": "<relative path to segment WAV>",
      "transcript": "<final_transcript or raw_transcript>",
      "duration_sec": <float>,
      "model_name": "<ASR model used>",
      "review_status": "<status>"
    }

Output schema — text_dataset.jsonl:
    {
      "id": "<text_id>",
      "text": "<chunk text>",
      "source_path": "<relative path to source>",
      "chunk_chars": <int>,
      "review_status": "<status>"
    }
"""

from __future__ import annotations

import argparse
import json
import random
import sys
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
# Helpers
# ---------------------------------------------------------------------------

def _best_transcript(row) -> str:
    """
    Return the best available transcript: prefer `final_transcript` (human-
    corrected) if non-empty, otherwise fall back to `raw_transcript`.
    """
    final = str(row.get("final_transcript", "")).strip()
    if final:
        return final
    return str(row.get("raw_transcript", "")).strip()


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _split_records(
    records: list[dict], train_ratio: float, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Split *records* into (train, validation) with reproducible shuffling."""
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * train_ratio)
    return shuffled[:cut], shuffled[cut:]


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_asr(
    manifest_path: Path,
    out_dir: Path,
    statuses: set[str],
    min_duration: float,
    max_duration: float,
    do_split: bool,
    split_ratio: float,
    logger,
) -> int:
    """
    Export audio-text pairs from *manifest_path* to JSONL.

    Returns the number of records exported.
    """
    df = load_manifest(manifest_path)
    if df.empty:
        logger.info("ASR manifest not found or empty — skipping ASR export.")
        return 0

    required_cols = {"pair_id", "segment_path", "raw_transcript", "review_status"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.warning("ASR manifest missing columns %s — skipping.", missing)
        return 0

    # Filter by review status
    df = df[df["review_status"].astype(str).isin(statuses)]

    # Filter by duration
    if "duration_sec" in df.columns:
        df = df.copy()
        df["_dur"] = df["duration_sec"].apply(
            lambda v: float(v) if str(v).strip() not in ("", "nan") else 0.0
        )
        df = df[(df["_dur"] >= min_duration) & (df["_dur"] <= max_duration)]

    # Drop rows with no transcript
    df = df[df.apply(lambda r: bool(_best_transcript(r)), axis=1)]

    records = []
    for _, row in df.iterrows():
        transcript = _best_transcript(row)
        rec = {
            "id": str(row.get("pair_id", "")),
            "audio_path": str(row.get("segment_path", "")),
            "transcript": transcript,
            "duration_sec": (
                round(float(row["_dur"]), 4) if "_dur" in row.index else None
            ),
            "model_name": str(row.get("model_name", "")),
            "review_status": str(row.get("review_status", "")),
        }
        records.append(rec)

    if not records:
        logger.info("No ASR records matched filters — nothing to export.")
        return 0

    if do_split:
        train, val = _split_records(records, split_ratio)
        _write_jsonl(train, out_dir / "asr_train.jsonl")
        _write_jsonl(val, out_dir / "asr_validation.jsonl")
        logger.info(
            "ASR export: %d train + %d validation records.", len(train), len(val)
        )
    else:
        _write_jsonl(records, out_dir / "asr_dataset.jsonl")
        logger.info("ASR export: %d records.", len(records))

    return len(records)


def export_text(
    manifest_path: Path,
    clean_text_dir: Path,
    out_dir: Path,
    statuses: set[str],
    min_chars: int,
    do_split: bool,
    split_ratio: float,
    logger,
) -> int:
    """
    Export text chunks from *manifest_path* to JSONL.

    Returns the number of records exported.
    """
    df = load_manifest(manifest_path)
    if df.empty:
        logger.info("Text manifest not found or empty — skipping text export.")
        return 0

    required_cols = {"text_id", "clean_text_path", "review_status"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.warning("Text manifest missing columns %s — skipping.", missing)
        return 0

    # Filter by review status
    df = df[df["review_status"].astype(str).isin(statuses)]

    # Filter by minimum chars
    if "chunk_chars" in df.columns:
        df = df[
            df["chunk_chars"].apply(
                lambda v: int(v) >= min_chars
                if str(v).strip() not in ("", "nan")
                else False
            )
        ]

    root = resolve_root()
    records = []

    for _, row in df.iterrows():
        clean_path = root / str(row.get("clean_text_path", ""))
        # Read the specific chunk from the clean text file if it exists
        chunk_text = _read_chunk(
            clean_path,
            row.get("chunk_index", ""),
            int(row.get("chunk_chars", 0)) if str(row.get("chunk_chars", "")).strip() not in ("", "nan") else 0,
        )
        if not chunk_text:
            continue

        rec = {
            "id": str(row.get("text_id", "")),
            "text": chunk_text,
            "source_path": str(row.get("source_path", "")),
            "chunk_chars": len(chunk_text),
            "review_status": str(row.get("review_status", "")),
        }
        records.append(rec)

    if not records:
        logger.info("No text records matched filters — nothing to export.")
        return 0

    if do_split:
        train, val = _split_records(records, split_ratio)
        _write_jsonl(train, out_dir / "text_train.jsonl")
        _write_jsonl(val, out_dir / "text_validation.jsonl")
        logger.info(
            "Text export: %d train + %d validation records.", len(train), len(val)
        )
    else:
        _write_jsonl(records, out_dir / "text_dataset.jsonl")
        logger.info("Text export: %d records.", len(records))

    return len(records)


def _read_chunk(clean_path: Path, chunk_index: str, chunk_chars: int) -> str:
    """
    Read the text of a specific chunk from its clean text file.

    Since chunks are produced by splitting sentences, we re-run the same
    chunking logic.  However to keep the export script self-contained and
    fast, we do a simpler approach: read the full file and extract the chunk
    by its index.
    """
    if not clean_path.exists():
        return ""
    try:
        # Import chunk_text from build_text_dataset to stay DRY
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from build_text_dataset import chunk_text

        cfg = load_config()
        text_cfg = cfg.get("text", {})
        min_c = int(text_cfg.get("min_chunk_chars", 50))
        max_c = int(text_cfg.get("max_chunk_chars", 2000))

        full_text = clean_path.read_text(encoding="utf-8")
        chunks = chunk_text(full_text, min_c, max_c)
        try:
            idx = int(chunk_index)
            if 0 <= idx < len(chunks):
                return chunks[idx]
        except (ValueError, TypeError):
            pass
        return ""
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export training-ready datasets from AlbanianDataFactory manifests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--status",
        nargs="+",
        default=["approved", "pending"],
        metavar="STATUS",
        help="review_status values to include (default: approved pending).",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        metavar="SECS",
        help="Minimum segment duration in seconds for ASR export (default: 1.0).",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        metavar="SECS",
        help="Maximum segment duration in seconds for ASR export (default: 30.0).",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=50,
        metavar="N",
        help="Minimum characters per text chunk (default: 50).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        metavar="PATH",
        help="Output directory for JSONL files (default: exports/ in project root).",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split output into train/validation JSONL files.",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.9,
        metavar="RATIO",
        help="Fraction of data for training when --split is used (default: 0.9).",
    )
    args = parser.parse_args()

    cfg = load_config()
    root = resolve_root()
    ensure_dirs(cfg)
    logger = setup_logger("export_dataset")

    out_dir = Path(args.out_dir) if args.out_dir else root / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    statuses: set[str] = set(args.status)
    manifests_dir = root / cfg["paths"]["manifests"]
    clean_text_dir = root / cfg["paths"]["clean_text"]

    logger.info(
        "Exporting data — statuses: %s | split: %s",
        sorted(statuses),
        args.split,
    )

    asr_count = export_asr(
        manifest_path=manifests_dir / "audio_text_pairs.csv",
        out_dir=out_dir,
        statuses=statuses,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        do_split=args.split,
        split_ratio=args.split_ratio,
        logger=logger,
    )

    text_count = export_text(
        manifest_path=manifests_dir / "text_manifest.csv",
        clean_text_dir=clean_text_dir,
        out_dir=out_dir,
        statuses=statuses,
        min_chars=args.min_chars,
        do_split=args.split,
        split_ratio=args.split_ratio,
        logger=logger,
    )

    logger.info(
        "Export complete — ASR: %d record(s), Text: %d record(s). Output: %s",
        asr_count,
        text_count,
        out_dir,
    )


if __name__ == "__main__":
    main()
