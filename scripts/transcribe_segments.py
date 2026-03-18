"""
transcribe_segments.py — Step 4 of the AlbanianDataFactory pipeline.

For every segment in audio_segments_manifest.csv that does not yet have a
raw_transcript (or --force is given), run Albanian ASR (Whisper via
Hugging Face transformers or faster-whisper) and write results to
manifests/audio_text_pairs.csv.

Usage:
    python scripts/transcribe_segments.py [--force] [--model MODEL_NAME]

Options:
    --force              Re-transcribe segments that already have a transcript.
    --model MODEL_NAME   Override the model name from config.

Schema of audio_text_pairs.csv:
    pair_id, segment_id, segment_path,
    start_sec, end_sec, duration_sec,
    raw_transcript, final_transcript, review_status, review_notes,
    transcription_error, model_name, created_at
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

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


# ---------------------------------------------------------------------------
# ASR back-ends
# ---------------------------------------------------------------------------

def _transcribe_with_transformers(segment_path: Path, model_name: str, device: str) -> str:
    """
    Transcribe *segment_path* using the Hugging Face transformers pipeline
    with language forced to Albanian (sq).
    """
    from transformers import pipeline as hf_pipeline

    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=0 if device == "cuda" else -1,
        generate_kwargs={"language": "sq", "task": "transcribe"},
    )
    result = pipe(str(segment_path))
    return result["text"].strip()


def _transcribe_with_faster_whisper(
    segment_path: Path, model_name: str, device: str
) -> str:
    """
    Transcribe using faster-whisper (CTranslate2 backend).
    Model name should be a faster-whisper compatible name such as
    'tiny', 'base', 'small', etc.
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device=device, compute_type="int8")
    segments, _ = model.transcribe(str(segment_path), language="sq")
    return " ".join(seg.text for seg in segments).strip()


def transcribe_segment(segment_path: Path, model_name: str, device: str) -> str:
    """
    Transcribe *segment_path* with the best available back-end.

    Preference order:
      1. faster-whisper   (if installed)
      2. transformers     (Hugging Face pipeline)
    """
    try:
        import faster_whisper  # noqa: F401
        return _transcribe_with_faster_whisper(segment_path, model_name, device)
    except ImportError:
        pass

    return _transcribe_with_transformers(segment_path, model_name, device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe audio segments with Albanian ASR."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe segments that already have a raw_transcript.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model name from config.",
    )
    args = parser.parse_args()

    cfg = load_config()
    root = resolve_root()
    ensure_dirs(cfg)
    logger = setup_logger("transcribe_segments")

    seg_manifest_path = root / cfg["paths"]["manifests"] / "audio_segments_manifest.csv"
    pairs_manifest_path = root / cfg["paths"]["manifests"] / "audio_text_pairs.csv"

    tscfg = cfg["transcription"]
    model_name: str = args.model or tscfg["model_name"]
    device: str = tscfg["device"]

    seg_manifest = load_manifest(seg_manifest_path)
    if seg_manifest.empty:
        logger.info("No segments manifest found at %s — run segment_audio.py first.", seg_manifest_path)
        return

    pairs_manifest = load_manifest(pairs_manifest_path)

    # Build set of already-transcribed segment IDs (those with non-empty raw_transcript)
    already_transcribed: set[str] = set()
    if not pairs_manifest.empty and "segment_id" in pairs_manifest.columns and "raw_transcript" in pairs_manifest.columns:
        mask = pairs_manifest["raw_transcript"].astype(str).str.strip() != ""
        already_transcribed = set(pairs_manifest.loc[mask, "segment_id"].astype(str))

    segments_to_process = seg_manifest.copy()
    if not args.force:
        segments_to_process = segments_to_process[
            ~segments_to_process["segment_id"].astype(str).isin(already_transcribed)
        ]

    total = len(segments_to_process)
    if total == 0:
        logger.info("All segments already transcribed (use --force to redo).")
        return

    logger.info(
        "Transcribing %d segment(s) with model '%s' on %s…",
        total,
        model_name,
        device,
    )

    new_rows: list[dict] = []
    failed = 0

    for _, row in segments_to_process.iterrows():
        seg_id = str(row["segment_id"])
        seg_path_rel = str(row["segment_path"])
        seg_path = root / seg_path_rel

        if not seg_path.exists():
            logger.warning("Segment file not found: %s — skipping.", seg_path)
            failed += 1
            continue

        try:
            logger.info("  Transcribing: %s", seg_path.name)
            transcript = transcribe_segment(seg_path, model_name, device)
            error_message = ""
        except Exception as exc:  # noqa: BLE001
            logger.error("Transcription failed for %s: %s", seg_path.name, exc)
            transcript = ""
            error_message = str(exc)
            failed += 1

        pair_id = stable_id(f"pair:{seg_id}")

        new_rows.append(
            {
                "pair_id": pair_id,
                "segment_id": seg_id,
                "segment_path": seg_path_rel,
                "start_sec": str(row.get("start_sec", "")),
                "end_sec": str(row.get("end_sec", "")),
                "duration_sec": str(row.get("duration_sec", "")),
                "raw_transcript": transcript,
                "final_transcript": "",
                "review_status": "pending",
                "review_notes": "",
                "transcription_error": error_message,
                "model_name": model_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    if new_rows:
        import pandas as pd
        df = pd.DataFrame(new_rows)
        upsert_manifest(df, pairs_manifest_path, key_col="pair_id")
        logger.info(
            "audio_text_pairs.csv updated — transcribed: %d, failed: %d.",
            len(new_rows) - failed,
            failed,
        )
    else:
        logger.info("No new transcriptions produced.")


if __name__ == "__main__":
    main()
