"""
review_manifest.py — Human-review helper for AlbanianDataFactory manifests.

Provides a command-line interface for bulk-updating review_status and
review_notes in the manifests without manual CSV editing.

Usage:
    python scripts/review_manifest.py <command> [options]

Commands:
    set-status    Bulk-set review_status for matched rows.
    set-notes     Append or overwrite review_notes for matched rows.
    list          List rows matching a filter.
    summary       Print a review-status summary for all manifests.

Examples:
    # Approve all pending audio pairs
    python scripts/review_manifest.py set-status \\
        --manifest audio --status approved --filter-status pending

    # Reject a specific pair by ID
    python scripts/review_manifest.py set-status \\
        --manifest audio --status reject --ids pid1 pid2

    # Tag a text chunk with noise note
    python scripts/review_manifest.py set-notes \\
        --manifest text --notes "encoding_issue" --ids tid5

    # List all pending audio pairs
    python scripts/review_manifest.py list \\
        --manifest audio --filter-status pending

    # Print full review summary
    python scripts/review_manifest.py summary
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
    resolve_root,
    setup_logger,
    upsert_manifest,
)

_MANIFEST_ALIASES = {
    "audio": "audio_text_pairs.csv",
    "text": "text_manifest.csv",
    "segments": "audio_segments_manifest.csv",
    "sources": "audio_manifest.csv",
}

_STATUS_VALUES = {"pending", "approved", "reject", "needs_fix"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_manifest(alias: str, manifests_dir: Path) -> Path:
    filename = _MANIFEST_ALIASES.get(alias, alias)
    return manifests_dir / filename


def _key_col(manifest_path: Path) -> str:
    """Return the primary key column for a given manifest file."""
    name = manifest_path.name
    if name == "audio_text_pairs.csv":
        return "pair_id"
    if name == "text_manifest.csv":
        return "text_id"
    if name == "audio_segments_manifest.csv":
        return "segment_id"
    if name == "audio_manifest.csv":
        return "sample_id"
    return "id"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_set_status(args, manifests_dir: Path, logger) -> None:
    path = _resolve_manifest(args.manifest, manifests_dir)
    df = load_manifest(path)
    if df.empty:
        logger.warning("Manifest not found or empty: %s", path)
        return

    if args.status not in _STATUS_VALUES:
        logger.error("Invalid status '%s'. Must be one of: %s", args.status, _STATUS_VALUES)
        sys.exit(1)

    key = _key_col(path)
    mask = _build_mask(df, key, args.ids, args.filter_status)
    n = mask.sum()
    if n == 0:
        logger.info("No rows matched the filter — nothing updated.")
        return

    df = df.copy()
    df.loc[mask, "review_status"] = args.status

    if args.notes:
        df.loc[mask, "review_notes"] = args.notes

    df.loc[mask, "reviewed_at"] = datetime.now(timezone.utc).isoformat()

    upsert_manifest(df, path, key_col=key)
    logger.info("Updated %d row(s) to status='%s' in %s.", n, args.status, path.name)


def cmd_set_notes(args, manifests_dir: Path, logger) -> None:
    path = _resolve_manifest(args.manifest, manifests_dir)
    df = load_manifest(path)
    if df.empty:
        logger.warning("Manifest not found or empty: %s", path)
        return

    key = _key_col(path)
    mask = _build_mask(df, key, args.ids, args.filter_status)
    n = mask.sum()
    if n == 0:
        logger.info("No rows matched the filter — nothing updated.")
        return

    df = df.copy()
    if args.append and "review_notes" in df.columns:
        # Append new note to existing notes (comma-separated)
        df.loc[mask, "review_notes"] = df.loc[mask, "review_notes"].apply(
            lambda existing: (
                (str(existing).strip() + ", " + args.notes).lstrip(", ")
                if str(existing).strip()
                else args.notes
            )
        )
    else:
        df.loc[mask, "review_notes"] = args.notes

    df.loc[mask, "reviewed_at"] = datetime.now(timezone.utc).isoformat()

    upsert_manifest(df, path, key_col=key)
    logger.info("Updated notes for %d row(s) in %s.", n, path.name)


def cmd_list(args, manifests_dir: Path, logger) -> None:
    path = _resolve_manifest(args.manifest, manifests_dir)
    df = load_manifest(path)
    if df.empty:
        print(f"Manifest not found or empty: {path}")
        return

    key = _key_col(path)
    mask = _build_mask(df, key, args.ids, args.filter_status)
    subset = df[mask]

    if subset.empty:
        print("No rows matched the filter.")
        return

    cols = [key, "review_status"]
    if "review_notes" in subset.columns:
        cols.append("review_notes")
    if "duration_sec" in subset.columns:
        cols.append("duration_sec")
    if "segment_path" in subset.columns:
        cols.append("segment_path")
    if "source_path" in subset.columns:
        cols.append("source_path")
    if "chunk_chars" in subset.columns:
        cols.append("chunk_chars")

    cols = [c for c in cols if c in subset.columns]
    print(subset[cols].to_string(index=False))
    print(f"\n{len(subset)} row(s) found.")


def cmd_summary(args, manifests_dir: Path, logger) -> None:
    print("\nAlbanianDataFactory — Review Status Summary")
    print("=" * 52)
    for alias, filename in _MANIFEST_ALIASES.items():
        path = manifests_dir / filename
        df = load_manifest(path)
        if df.empty:
            print(f"\n  {alias:12s}  (no data)")
            continue
        total = len(df)
        print(f"\n  {alias:12s}  ({total} rows)")
        if "review_status" in df.columns:
            counts = df["review_status"].value_counts()
            for status, count in counts.items():
                pct = 100 * count / total
                print(f"    {str(status):12s}  {count:6d}  ({pct:.1f}%)")
        else:
            print("    (no review_status column)")
    print()


# ---------------------------------------------------------------------------
# Mask builder
# ---------------------------------------------------------------------------

def _build_mask(df, key_col: str, ids: list[str] | None, filter_status: str | None):
    """
    Build a boolean mask for rows that match either a list of IDs or a
    review_status filter (or both together as AND).
    """
    import pandas as pd
    mask = pd.Series([True] * len(df), index=df.index)

    if ids:
        if key_col in df.columns:
            mask &= df[key_col].astype(str).isin(set(ids))
        else:
            mask &= pd.Series([False] * len(df), index=df.index)

    if filter_status:
        if "review_status" in df.columns:
            mask &= df["review_status"].astype(str) == filter_status
        else:
            mask &= pd.Series([False] * len(df), index=df.index)

    return mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Human-review helper for AlbanianDataFactory manifests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- set-status ----
    p_status = sub.add_parser("set-status", help="Bulk-set review_status for matched rows.")
    p_status.add_argument(
        "--manifest", required=True,
        choices=list(_MANIFEST_ALIASES.keys()),
        help="Which manifest to update.",
    )
    p_status.add_argument(
        "--status", required=True,
        choices=sorted(_STATUS_VALUES),
        help="New review_status value.",
    )
    p_status.add_argument("--ids", nargs="+", default=None, help="Specific row IDs to update.")
    p_status.add_argument(
        "--filter-status", default=None,
        help="Only update rows that currently have this review_status.",
    )
    p_status.add_argument("--notes", default=None, help="Optionally set review_notes as well.")

    # ---- set-notes ----
    p_notes = sub.add_parser("set-notes", help="Set or append review_notes for matched rows.")
    p_notes.add_argument(
        "--manifest", required=True,
        choices=list(_MANIFEST_ALIASES.keys()),
        help="Which manifest to update.",
    )
    p_notes.add_argument("--notes", required=True, help="Review notes to set or append.")
    p_notes.add_argument("--ids", nargs="+", default=None, help="Specific row IDs to update.")
    p_notes.add_argument(
        "--filter-status", default=None,
        help="Only update rows that currently have this review_status.",
    )
    p_notes.add_argument(
        "--append", action="store_true",
        help="Append the note to existing review_notes instead of replacing.",
    )

    # ---- list ----
    p_list = sub.add_parser("list", help="List rows matching a filter.")
    p_list.add_argument(
        "--manifest", required=True,
        choices=list(_MANIFEST_ALIASES.keys()),
        help="Which manifest to inspect.",
    )
    p_list.add_argument("--ids", nargs="+", default=None, help="Specific row IDs to show.")
    p_list.add_argument(
        "--filter-status", default=None,
        help="Show only rows with this review_status.",
    )

    # ---- summary ----
    sub.add_parser("summary", help="Print a review-status summary for all manifests.")

    args = parser.parse_args()

    cfg = load_config()
    root = resolve_root()
    ensure_dirs(cfg)
    logger = setup_logger("review_manifest")

    manifests_dir = root / cfg["paths"]["manifests"]

    if args.command == "set-status":
        cmd_set_status(args, manifests_dir, logger)
    elif args.command == "set-notes":
        cmd_set_notes(args, manifests_dir, logger)
    elif args.command == "list":
        cmd_list(args, manifests_dir, logger)
    elif args.command == "summary":
        cmd_summary(args, manifests_dir, logger)


if __name__ == "__main__":
    main()
