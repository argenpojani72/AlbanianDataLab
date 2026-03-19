"""
utils.py — Shared utilities for AlbanianDataFactory pipeline scripts.

Provides:
  - load_config()            Load config/config.yaml relative to project root.
  - stable_id(text)          Deterministic short hex ID from a string (SHA-256 prefix).
  - setup_logger(name)       Consistent file + console logging.
  - ensure_dirs(cfg)         Create all output directories defined in config.
  - load_manifest(path)      Load a CSV manifest into a DataFrame; return empty DF if missing.
  - upsert_manifest(df, path, key_col)
                             Merge new rows into an existing manifest CSV by key column.
  - resolve_root()           Return the project-root Path (parent of scripts/).
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import List

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

def resolve_root() -> Path:
    """Return the repository / project root (two levels up from this file)."""
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load config/config.yaml relative to the project root."""
    cfg_path = resolve_root() / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Stable IDs
# ---------------------------------------------------------------------------

def stable_id(text: str, length: int = 12) -> str:
    """Return a deterministic hex ID of *length* chars derived from *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    """
    Return a logger that writes to both the console (INFO) and a file (DEBUG).

    The log file is placed in *log_dir* (or config logs/ if None) as
    ``<name>.log``.
    """
    if log_dir is None:
        cfg = load_config()
        log_dir = resolve_root() / cfg["paths"]["logs"]
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured (e.g. called twice)

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # file handler
    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

def ensure_dirs(cfg: dict) -> None:
    """Create every directory listed in cfg['paths'] if it doesn't exist."""
    root = resolve_root()
    for rel_path in cfg["paths"].values():
        (root / rel_path).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> pd.DataFrame:
    """
    Load a CSV manifest.  Returns an empty DataFrame (no columns) when the
    file does not exist yet.
    """
    if path.exists():
        return pd.read_csv(path, dtype=str)
    return pd.DataFrame()


def upsert_manifest(
    new_rows: pd.DataFrame,
    path: Path,
    key_col: str,
) -> pd.DataFrame:
    """
    Merge *new_rows* into the manifest at *path* keyed on *key_col*.

    - Existing rows whose key already appears in *new_rows* are updated.
    - Rows not present in *new_rows* are kept untouched.
    - The merged result is written back to *path* and returned.

    If *path* does not exist yet it is created from *new_rows* alone.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_manifest(path)

    if existing.empty:
        merged = new_rows.copy()
    else:
        # Align columns: add any columns that exist in new_rows but not in existing
        for col in new_rows.columns:
            if col not in existing.columns:
                existing[col] = ""
        # Remove rows from existing whose key appears in new_rows (will be replaced)
        keys_to_replace = set(new_rows[key_col].astype(str))
        existing_filtered = existing[~existing[key_col].astype(str).isin(keys_to_replace)]
        merged = pd.concat([existing_filtered, new_rows], ignore_index=True)

    merged.to_csv(path, index=False)
    return merged


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def relative_to_root(path: Path) -> str:
    """Return *path* as a POSIX string relative to the project root."""
    root = resolve_root()
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def find_files(directory: Path, extensions: List[str]) -> List[Path]:
    """
    Return a sorted list of files under *directory* whose suffix (lower-case)
    is in *extensions*.
    """
    results = []
    for ext in extensions:
        results.extend(directory.rglob(f"*{ext}"))
    return sorted(set(results))
