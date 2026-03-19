"""
build_text_dataset.py — Step 5 of the AlbanianDataFactory pipeline.

For every text-like file in raw_sources/text/ that has not yet been processed:

1. **Extract** — Read body text from .txt, .html/.htm, .json, .csv, .pdf, .epub
                  and write to raw_text/<stem>.txt.
2. **Clean**   — Normalize whitespace, strip control characters, remove duplicate
                  consecutive lines, write to clean_text/<stem>.txt.
3. **Chunk**   — Split cleaned text into chunks within [min_chunk_chars,
                  max_chunk_chars] (splitting at sentence boundaries where possible).
4. **Dedup**   — Drop any chunk whose SHA-256 content hash has been seen before
                  (across all sources in this run + existing manifest).
5. **Register** — Append new rows to manifests/text_manifest.csv.

Usage:
    python scripts/build_text_dataset.py [--force]

Options:
    --force   Reprocess even if clean_text/<stem>.txt already exists.

text_manifest.csv schema:
    text_id, source_path, raw_text_path, clean_text_path,
    chunk_index, chunk_chars, content_hash,
    domain, review_status, review_notes, created_at
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    ensure_dirs,
    find_files,
    load_config,
    load_manifest,
    relative_to_root,
    resolve_root,
    setup_logger,
    stable_id,
    upsert_manifest,
)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_txt(path: Path) -> str:
    """Read a plain-text file."""
    return path.read_text(encoding="utf-8", errors="replace")


def extract_text_html(path: Path) -> str:
    """Extract visible text from an HTML file (uses trafilatura if available)."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        import trafilatura
        extracted = trafilatura.extract(raw)
        if extracted:
            return extracted
    except ImportError:
        pass
    # Fallback: strip HTML tags with regex
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(text)
    return text


def extract_text_json(path: Path) -> str:
    """Extract all string values from a JSON file (concatenated)."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    parts: List[str] = []

    def _collect(obj):
        if isinstance(obj, str):
            parts.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _collect(v)
        elif isinstance(obj, list):
            for item in obj:
                _collect(item)

    _collect(data)
    return "\n".join(parts)


def extract_text_csv(path: Path) -> str:
    """Concatenate all string cells from a CSV file."""
    import csv

    parts: List[str] = []
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            parts.extend(cell for cell in row if cell.strip())
    return "\n".join(parts)


def extract_text_pdf(path: Path) -> str:
    """Extract text from a PDF using pdfminer.six (if available)."""
    try:
        from pdfminer.high_level import extract_text as pdf_extract
        return pdf_extract(str(path))
    except ImportError:
        pass
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    except ImportError:
        pass
    return ""


def extract_text_epub(path: Path) -> str:
    """Extract text from an EPUB file using ebooklib (if available)."""
    try:
        import ebooklib
        from ebooklib import epub

        book = epub.read_epub(str(path))
        parts: List[str] = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            raw = item.get_content().decode("utf-8", errors="replace")
            text = re.sub(r"<[^>]+>", " ", raw)
            text = html.unescape(text)
            parts.append(text)
        return "\n".join(parts)
    except ImportError:
        return ""


EXTRACTORS = {
    ".txt": extract_text_txt,
    ".html": extract_text_html,
    ".htm": extract_text_html,
    ".json": extract_text_json,
    ".csv": extract_text_csv,
    ".pdf": extract_text_pdf,
    ".epub": extract_text_epub,
}


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    extractor = EXTRACTORS.get(ext)
    if extractor is None:
        return path.read_text(encoding="utf-8", errors="replace")
    return extractor(path)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(raw: str) -> str:
    """
    Normalise and clean raw text:
    - Unicode NFC normalisation
    - Remove control characters (except newlines and tabs)
    - Normalise whitespace: collapse runs of spaces/tabs to a single space
    - Remove duplicate consecutive non-empty lines
    - Strip leading/trailing whitespace per line and globally
    """
    text = unicodedata.normalize("NFC", raw)
    # Remove control characters except \n and \t
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    # Normalise spaces/tabs on each line
    lines = text.splitlines()
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    # Remove duplicate consecutive lines
    deduped: List[str] = []
    prev = None
    for line in lines:
        if line != prev:
            deduped.append(line)
            prev = line
    return "\n".join(deduped).strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_SENTENCE_ENDS = re.compile(r"(?<=[.!?…])\s+")


def chunk_text(text: str, min_chars: int, max_chars: int) -> List[str]:
    """
    Split *text* into chunks between *min_chars* and *max_chars* characters.

    Tries to split at sentence boundaries (`.`, `!`, `?`, `…`).
    When a sentence itself exceeds *max_chars*, falls back to splitting at
    the nearest whitespace before the limit to avoid mid-word cuts.
    """
    # First split into sentences
    sentences = _SENTENCE_ENDS.split(text)
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current and len(current) >= min_chars:
                chunks.append(current)
            # If the sentence itself is longer than max_chars, split at word
            # boundaries to avoid cutting mid-word.
            if len(sentence) > max_chars:
                words = sentence.split()
                piece = ""
                for word in words:
                    candidate_piece = (piece + " " + word).strip() if piece else word
                    if len(candidate_piece) <= max_chars:
                        piece = candidate_piece
                    else:
                        if piece and len(piece) >= min_chars:
                            chunks.append(piece)
                        piece = word
                if piece and len(piece) >= min_chars:
                    chunks.append(piece)
                current = ""
            else:
                current = sentence

    if current and len(current) >= min_chars:
        chunks.append(current)

    return chunks


# ---------------------------------------------------------------------------
# Content hash
# ---------------------------------------------------------------------------

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build text dataset from raw text sources.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess even if clean_text file already exists.",
    )
    args = parser.parse_args()

    cfg = load_config()
    root = resolve_root()
    ensure_dirs(cfg)
    logger = setup_logger("build_text_dataset")

    src_dir = root / cfg["paths"]["raw_sources_text"]
    raw_text_dir = root / cfg["paths"]["raw_text"]
    clean_text_dir = root / cfg["paths"]["clean_text"]
    manifest_path = root / cfg["paths"]["manifests"] / "text_manifest.csv"

    text_cfg = cfg["text"]
    min_chars: int = text_cfg["min_chunk_chars"]
    max_chars: int = text_cfg["max_chunk_chars"]
    dedup_by_hash: bool = text_cfg["dedup_by_hash"]
    extensions: List[str] = text_cfg["text_extensions"]

    sources = find_files(src_dir, extensions)
    if not sources:
        logger.info("No text files found in %s — nothing to do.", src_dir)
        return

    logger.info("Found %d text source(s) in %s.", len(sources), src_dir)

    # Load existing hashes to skip globally duplicate chunks
    existing_manifest = load_manifest(manifest_path)
    seen_hashes: set[str] = set()
    if dedup_by_hash and not existing_manifest.empty and "content_hash" in existing_manifest.columns:
        seen_hashes = set(existing_manifest["content_hash"].dropna().astype(str))

    all_new_rows: list[dict] = []
    skipped_sources = new_chunks = deduped_chunks = 0

    for src in sorted(sources):
        rel = relative_to_root(src)
        raw_out = raw_text_dir / (src.stem + ".txt")
        clean_out = clean_text_dir / (src.stem + ".txt")

        if not args.force and clean_out.exists():
            logger.debug("Skipping (already processed): %s", rel)
            skipped_sources += 1
            continue

        logger.info("Processing: %s", rel)

        # 1. Extract
        raw_text = extract_text(src)
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        raw_out.write_text(raw_text, encoding="utf-8")

        # 2. Clean
        cleaned = clean_text(raw_text)
        clean_out.parent.mkdir(parents=True, exist_ok=True)
        clean_out.write_text(cleaned, encoding="utf-8")

        # 3. Chunk
        chunks = chunk_text(cleaned, min_chars, max_chars)
        logger.info("  → %d chunk(s) from %s.", len(chunks), src.name)

        # 4. Dedup + register
        source_id = stable_id(rel)
        for idx, chunk in enumerate(chunks):
            ch = content_hash(chunk)
            if dedup_by_hash and ch in seen_hashes:
                deduped_chunks += 1
                continue
            seen_hashes.add(ch)

            text_id = stable_id(f"{source_id}:{idx}:{ch}")
            all_new_rows.append(
                {
                    "text_id": text_id,
                    "source_path": rel,
                    "raw_text_path": relative_to_root(raw_out),
                    "clean_text_path": relative_to_root(clean_out),
                    "chunk_index": str(idx),
                    "chunk_chars": str(len(chunk)),
                    "content_hash": ch,
                    "domain": "",
                    "review_status": "pending",
                    "review_notes": "",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            new_chunks += 1

    if all_new_rows:
        import pandas as pd
        df = pd.DataFrame(all_new_rows)
        upsert_manifest(df, manifest_path, key_col="text_id")
        logger.info(
            "text_manifest.csv updated — new chunks: %d, deduped: %d, skipped sources: %d.",
            new_chunks,
            deduped_chunks,
            skipped_sources,
        )
    else:
        logger.info(
            "No new text chunks produced (skipped sources: %d, deduped: %d).",
            skipped_sources,
            deduped_chunks,
        )


if __name__ == "__main__":
    main()
