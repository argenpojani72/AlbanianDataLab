"""
run_pipeline.py — End-to-end runner for the AlbanianDataFactory pipeline.

Runs all pipeline steps in sequence from a single command:

    Step 1 — extract_audio.py      media → raw WAV
    Step 2 — clean_audio.py        raw WAV → clean WAV
    Step 3 — segment_audio.py      clean WAV → speech segments (Silero VAD)
    Step 4 — transcribe_segments.py segments → transcripts (Albanian ASR)
    Step 5 — build_text_dataset.py text sources → cleaned/chunked text

Usage:
    python scripts/run_pipeline.py [options]

Options:
    --force              Pass --force to every step (re-process all inputs).
    --from-step N        Start from step N (1–5); previous steps are skipped.
    --to-step N          Stop after step N (1–5); later steps are skipped.
    --model MODEL_NAME   Override the ASR model for Step 4.
    --audio-only         Run only audio steps (1–4).
    --text-only          Run only the text step (5).

Examples:
    # Full run (incremental — only new files processed)
    python scripts/run_pipeline.py

    # Force re-process everything from scratch
    python scripts/run_pipeline.py --force

    # Re-run only transcription and text steps
    python scripts/run_pipeline.py --from-step 4

    # Use a specific Albanian ASR model
    python scripts/run_pipeline.py --model ard-ali/whisper-medium-albanian
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Resolve the scripts directory so we can call each step as a subprocess
_SCRIPTS_DIR = Path(__file__).resolve().parent


def _step_script(name: str) -> str:
    """Return the absolute path to a pipeline step script."""
    return str(_SCRIPTS_DIR / name)


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

STEPS = [
    {
        "number": 1,
        "name": "extract_audio",
        "script": "extract_audio.py",
        "description": "Extract audio from media sources",
        "extra_args": [],
    },
    {
        "number": 2,
        "name": "clean_audio",
        "script": "clean_audio.py",
        "description": "Clean and normalise audio",
        "extra_args": [],
    },
    {
        "number": 3,
        "name": "segment_audio",
        "script": "segment_audio.py",
        "description": "Segment audio using Silero VAD",
        "extra_args": [],
    },
    {
        "number": 4,
        "name": "transcribe_segments",
        "script": "transcribe_segments.py",
        "description": "Transcribe segments with Albanian ASR",
        "extra_args": [],          # --model injected dynamically
    },
    {
        "number": 5,
        "name": "build_text_dataset",
        "script": "build_text_dataset.py",
        "description": "Build text dataset from text sources",
        "extra_args": [],
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_step(step: dict, extra_args: list[str]) -> bool:
    """
    Run a single pipeline step as a subprocess.

    Returns True on success (exit code 0), False on failure.
    """
    cmd = [sys.executable, _step_script(step["script"])] + step["extra_args"] + extra_args
    print(
        f"\n{'='*60}\n"
        f"  Step {step['number']}: {step['description']}\n"
        f"  Command: {' '.join(cmd)}\n"
        f"{'='*60}"
    )
    t0 = time.monotonic()
    result = subprocess.run(cmd)
    elapsed = time.monotonic() - t0
    if result.returncode == 0:
        print(f"  ✓ Step {step['number']} completed in {elapsed:.1f}s")
        return True
    else:
        print(
            f"  ✗ Step {step['number']} FAILED (exit code {result.returncode}) "
            f"after {elapsed:.1f}s"
        )
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full AlbanianDataFactory pipeline end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process all inputs even if already done (passes --force to each step).",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        metavar="N",
        help="Start from step N (1–5). Default: 1.",
    )
    parser.add_argument(
        "--to-step",
        type=int,
        default=5,
        metavar="N",
        help="Stop after step N (1–5). Default: 5.",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL_NAME",
        help="Override the Albanian ASR model for Step 4 (HuggingFace model ID).",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Run only audio steps (1–4).",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Run only the text step (5).",
    )
    args = parser.parse_args()

    # Resolve step range
    from_step = args.from_step
    to_step = args.to_step

    if args.audio_only and args.text_only:
        parser.error("--audio-only and --text-only are mutually exclusive.")
    if args.audio_only:
        to_step = min(to_step, 4)
    if args.text_only:
        from_step = max(from_step, 5)
        to_step = max(to_step, 5)

    if not (1 <= from_step <= 5 and 1 <= to_step <= 5 and from_step <= to_step):
        parser.error(
            f"--from-step and --to-step must be between 1 and 5 with from ≤ to "
            f"(got {from_step}–{to_step})."
        )

    # Build common extra args passed to every step
    common_args: list[str] = []
    if args.force:
        common_args.append("--force")

    # Inject --model into Step 4 if provided
    for step in STEPS:
        if step["number"] == 4 and args.model:
            step["extra_args"] = ["--model", args.model]

    # Run selected steps
    t_total = time.monotonic()
    results: list[tuple[int, bool]] = []

    selected = [s for s in STEPS if from_step <= s["number"] <= to_step]

    print(
        f"\nAlbanianDataFactory Pipeline\n"
        f"Running steps {from_step}–{to_step} "
        f"({'forced' if args.force else 'incremental'})\n"
    )

    for step in selected:
        ok = run_step(step, common_args)
        results.append((step["number"], ok))
        if not ok:
            print(
                f"\n⚠ Pipeline halted at Step {step['number']}. "
                "Fix the error and re-run (use --from-step to resume).\n"
            )
            sys.exit(1)

    elapsed_total = time.monotonic() - t_total
    print(
        f"\n{'='*60}\n"
        f"  Pipeline complete — {len(results)} step(s) in {elapsed_total:.1f}s\n"
        f"{'='*60}\n"
    )


if __name__ == "__main__":
    main()
