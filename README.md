# AlbanianDataFactory

An **autonomous pipeline** that turns raw inputs — YouTube videos, podcasts, speeches, books, articles, and other documents — into fully professional, engineered **Albania-Albanian (sq-AL) datasets** ready for model training and research.

The system runs **end-to-end and incrementally**: you supply source files; it extracts, cleans, segments, transcribes, deduplicates, and registers everything in structured manifests. Outputs are **traceable**, **reviewable**, and **training-ready**. Re-running only processes new inputs; nothing already processed is touched.

---

## Project Structure

```
AlbanianDataFactory/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── config.yaml               ← all tunable parameters
├── raw_sources/
│   ├── audio_video/              ← drop media here (mp4, mp3, wav, m4a, …)
│   └── text/                     ← drop text sources here (txt, html, pdf, …)
├── raw_audio/                    ← extracted mono 16 kHz WAVs
├── clean_audio/                  ← denoised/normalised WAVs
├── audio_segments/               ← VAD speech chunks per source
├── manifests/
│   ├── audio_manifest.csv        ← source → raw/clean audio
│   ├── audio_segments_manifest.csv
│   ├── audio_text_pairs.csv      ← segment ↔ transcript + review status
│   └── text_manifest.csv         ← text chunks + review status
├── raw_text/                     ← extracted body text
├── clean_text/                   ← normalised/deduped text
├── review/
│   ├── audio/                    ← human review notes for audio
│   └── text/                     ← human review notes for text
├── logs/                         ← one log file per script
├── scripts/
│   ├── utils.py                  ← shared helpers (stable_id, upsert_manifest, …)
│   ├── run_pipeline.py           ← end-to-end runner (all steps in one command)
│   ├── extract_audio.py          ← Step 1: media → WAV
│   ├── clean_audio.py            ← Step 2: WAV → clean WAV
│   ├── segment_audio.py          ← Step 3: clean WAV → speech chunks (Silero VAD)
│   ├── transcribe_segments.py    ← Step 4: chunks → transcripts (Albanian fine-tuned ASR)
│   ├── build_text_dataset.py     ← Step 5: text → chunks → manifest
│   ├── export_dataset.py         ← Step 6: manifests → training-ready JSONL
│   ├── review_manifest.py        ← human-review CLI (update review_status / notes)
│   └── stats.py                  ← dataset statistics report
├── tests/
│   ├── conftest.py               ← shared pytest fixtures
│   ├── test_utils.py             ← unit tests for utils.py
│   ├── test_text_processing.py   ← unit tests for text clean/chunk logic
│   ├── test_segment_logic.py     ← unit tests for VAD merge/split logic
│   ├── test_export_dataset.py    ← unit tests for export logic
│   └── test_run_pipeline.py      ← unit tests for pipeline step selection
├── .github/
│   └── workflows/ci.yml          ← GitHub Actions CI (runs tests on every push/PR)
├── exports/                      ← JSONL training files (export_dataset.py output)
└── notebooks/                    ← exploratory notebooks
```

---

## Quick Start

### 1. Prerequisites

- Python 3.9+
- **ffmpeg** on PATH (`ffmpeg -version` should work)

### 2. Install Python dependencies

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 3. Add your inputs

| Type | Where to drop files |
|------|---------------------|
| Audio / Video | `raw_sources/audio_video/` |
| Text (books, articles, …) | `raw_sources/text/` |

Supported audio/video formats: `.mp4 .mkv .webm .avi .mov .flv .mp3 .m4a .aac .ogg .flac .wav .opus`

Supported text formats: `.txt .html .htm .json .csv .pdf .epub`

### 4. Run the pipeline

**Option A — single command (recommended)**

```bash
# Run all 5 steps (incremental — only new files processed)
python scripts/run_pipeline.py

# Force re-process everything from scratch
python scripts/run_pipeline.py --force

# Run only audio steps (1–4)
python scripts/run_pipeline.py --audio-only

# Resume from step 3 onwards
python scripts/run_pipeline.py --from-step 3

# Use a specific Albanian ASR model for Step 4
python scripts/run_pipeline.py --model ard-ali/whisper-medium-albanian
```

**Option B — individual steps**

Run each step in order from the **project root**:

```bash
python scripts/extract_audio.py
python scripts/clean_audio.py
python scripts/segment_audio.py
python scripts/transcribe_segments.py
python scripts/build_text_dataset.py
```

Every script is **idempotent**: re-running skips already-processed items.  
Use `--force` to reprocess everything:

```bash
python scripts/extract_audio.py --force
```

### 5. Add more inputs → extend the dataset

Drop new files into `raw_sources/audio_video/` or `raw_sources/text/` and run the pipeline again. Only the new files are processed; existing outputs and manifest rows are untouched.

---

## Pipeline Steps

### Step 1 — `extract_audio.py`

Reads every media file from `raw_sources/audio_video/` and uses **ffmpeg** to produce a mono 16 kHz WAV in `raw_audio/`. A stable SHA-256-derived `sample_id` is computed from the source path to ensure idempotency.  
Registers each source in **`manifests/audio_manifest.csv`**.

Skip logic: source is skipped if its `sample_id` already appears in the manifest **and** the output WAV exists.

### Step 2 — `clean_audio.py`

Applies an ffmpeg filter chain to each `raw_audio/*.wav`:
- `highpass=f=80` — remove sub-80 Hz rumble
- `lowpass=f=8000` — remove above 8 kHz hiss
- `afftdn=nf=-25` — broadband noise reduction
- `loudnorm=I=-16:LRA=11:TP=-1.5` — EBU R128 loudness normalisation

Output: `clean_audio/<stem>_clean.wav`.  
Updates **`manifests/audio_manifest.csv`** with the `clean_audio_path` column.

Skip logic: source is skipped if `<stem>_clean.wav` already exists.

### Step 3 — `segment_audio.py`

Runs **Silero VAD** on each clean WAV to detect speech regions, then:
- merges consecutive regions closer than `merge_gap_sec` (default 0.3 s)
- splits regions longer than `max_duration_sec` (default 20 s)
- discards segments shorter than `min_duration_sec` (default 3 s)

Writes individual segment WAVs to `audio_segments/<source_stem>/`.  
Appends new rows to **`manifests/audio_segments_manifest.csv`** with columns:
`segment_id`, `segment_path`, `source_clean_audio`, `start_sec`, `end_sec`, `duration_sec`, `created_at`.

Skip logic: source is skipped if its segment folder already contains WAV files.

### Step 4 — `transcribe_segments.py`

For each segment in `audio_segments_manifest.csv` without a `raw_transcript`, runs an **Albanian fine-tuned ASR model** via the Hugging Face `transformers` pipeline and writes results to **`manifests/audio_text_pairs.csv`**:

| Column | Description |
|--------|-------------|
| `pair_id` | Stable ID |
| `segment_id` | Links to segments manifest |
| `segment_path` | Relative path to WAV |
| `start_sec` / `end_sec` / `duration_sec` | Timing |
| `raw_transcript` | First-pass ASR output |
| `final_transcript` | Human-corrected (empty until review) |
| `review_status` | `pending` / `approved` / `reject` / `needs_fix` |
| `review_notes` | Free-text tags (see Review Conventions) |
| `transcription_error` | Error message if ASR failed (empty on success) |
| `model_name` | ASR model used |
| `created_at` | ISO 8601 timestamp |

Albanian fine-tuned models (set in `config/config.yaml` or via `--model`):

| Model | Size | Notes |
|-------|------|-------|
| `primusAI/whisper-large-v3-albanian` | large | Highest accuracy; ~4 GB VRAM |
| `ard-ali/whisper-medium-albanian` | medium | Good balance of speed and accuracy |
| `ard-ali/whisper-small-albanian` | small | Lightweight; lower accuracy |

Override model at runtime with `--model MODEL_NAME`.

Skip logic: segments with an existing `raw_transcript` are skipped unless `--force`.

### Step 5 — `build_text_dataset.py`

For each text file in `raw_sources/text/`:

1. **Extract** body text → `raw_text/<stem>.txt`
2. **Clean** (NFC normalise, strip control chars, collapse whitespace, dedup consecutive lines) → `clean_text/<stem>.txt`
3. **Chunk** into segments between `min_chunk_chars` (200) and `max_chunk_chars` (2000) at sentence boundaries
4. **Dedup** by SHA-256 content hash (global across all sources)
5. **Register** in **`manifests/text_manifest.csv`**

Text manifest columns: `text_id`, `source_path`, `raw_text_path`, `clean_text_path`, `chunk_index`, `chunk_chars`, `content_hash`, `domain`, `review_status`, `review_notes`, `created_at`.

Skip logic: source is skipped if `clean_text/<stem>.txt` already exists.

### Step 6 — `export_dataset.py`

Reads the manifests and exports training-ready JSONL files to `exports/`:

```bash
# Export all approved + pending records (default)
python scripts/export_dataset.py

# Export only human-reviewed approved records
python scripts/export_dataset.py --status approved

# Export with train/validation split (90/10)
python scripts/export_dataset.py --split

# Custom output directory
python scripts/export_dataset.py --out-dir /path/to/output
```

Output files:

| File | Content |
|------|---------|
| `exports/asr_dataset.jsonl` | Audio-text pairs (ASR training) |
| `exports/text_dataset.jsonl` | Text chunks (language modelling) |
| `exports/asr_train.jsonl` + `exports/asr_validation.jsonl` | Split variant |
| `exports/text_train.jsonl` + `exports/text_validation.jsonl` | Split variant |

ASR record schema:
```json
{
  "id": "<pair_id>",
  "audio_path": "<relative path to segment WAV>",
  "transcript": "<final or raw transcript>",
  "duration_sec": 4.2,
  "model_name": "primusAI/whisper-large-v3-albanian",
  "review_status": "approved"
}
```

Text record schema:
```json
{
  "id": "<text_id>",
  "text": "<chunk text>",
  "source_path": "<relative path to source>",
  "chunk_chars": 320,
  "review_status": "pending"
}
```

---

## Running Tests

Unit tests cover the pure logic of `utils.py`, `build_text_dataset.py`, and `segment_audio.py` — no external tools required.

```bash
# Install test dependency (pytest)
pip install pytest

# Run all tests
python -m pytest tests/ -v
```

Tests are organised by module:
- `tests/test_utils.py` — `stable_id`, `load_manifest`, `upsert_manifest`
- `tests/test_text_processing.py` — `clean_text`, `chunk_text`, `content_hash`
- `tests/test_segment_logic.py` — `merge_segments`, `split_long_segments`

---

## Configuration

All parameters live in **`config/config.yaml`**. Key settings:

```yaml
audio:
  sample_rate: 16000
  clean_filter: "highpass=f=80,lowpass=f=8000,afftdn=nf=-25,loudnorm=I=-16:LRA=11:TP=-1.5"

segmentation:
  min_duration_sec: 3.0
  max_duration_sec: 20.0
  merge_gap_sec: 0.3
  vad_threshold: 0.5

transcription:
  model_name: "primusAI/whisper-large-v3-albanian"
  # Alternatives: ard-ali/whisper-medium-albanian | ard-ali/whisper-small-albanian
  language: "sq"
  device: "cpu"   # or "cuda"

text:
  min_chunk_chars: 200
  max_chunk_chars: 2000
  dedup_by_hash: true

export:
  out_dir: exports
  include_statuses: ["approved", "pending"]
  min_duration_sec: 1.0
  max_duration_sec: 30.0
  split: false
  split_ratio: 0.9
```

---

## Review Workflow

Human review works directly on the manifests and review notes — it never overwrites raw or processed files.

### Option A — CLI tool (recommended)

Use `review_manifest.py` to update statuses without editing CSVs by hand:

```bash
# See a summary of review coverage across all manifests
python scripts/review_manifest.py summary

# List all pending audio pairs
python scripts/review_manifest.py list --manifest audio --filter-status pending

# Approve all pending audio pairs in bulk
python scripts/review_manifest.py set-status \
    --manifest audio --status approved --filter-status pending

# Reject specific pairs by ID
python scripts/review_manifest.py set-status \
    --manifest audio --status reject --ids pid1 pid7 --notes "non_albanian"

# Append a note to a text chunk
python scripts/review_manifest.py set-notes \
    --manifest text --notes "encoding_issue" --ids tid12 --append
```

### Option B — Direct CSV editing

### Audio pairs review (`manifests/audio_text_pairs.csv`)

- Listen to each segment in `audio_segments/`.
- Edit `final_transcript` with the corrected transcript.
- Set `review_status` to `approved`, `reject`, or `needs_fix`.
- Add tags to `review_notes`.

### Text review (`manifests/text_manifest.csv`)

- Read chunks in `clean_text/`.
- Set `review_status` and `review_notes` as needed.

### Review conventions

**Status values:** `pending` | `approved` | `reject` | `needs_fix`

**Audio tags:** `noise` `music` `overlap` `bad_cut` `unclear_speech` `non_albanian` `too_short` `too_long` `distortion`

**Text tags:** `duplicate` `junk` `html_noise` `encoding_issue` `non_albanian` `too_short` `bad_extraction` `ocr_noise`

---

## Dataset Statistics

Print a quick summary of how much data has been collected and reviewed:

```bash
# Text table (human-readable)
python scripts/stats.py

# JSON (machine-readable)
python scripts/stats.py --json

# Write to a file
python scripts/stats.py --out reports/stats.txt
```

Example output:
```
════════════════════════════════════════════════════════
  AlbanianDataFactory — Dataset Statistics
  Generated: 2024-01-15T12:00:00+00:00
════════════════════════════════════════════════════════

── Audio Sources ──────────────────────────────────────
  Total media files processed:  47
  With clean audio:             47

── Audio Segments ─────────────────────────────────────
  Total segments:               3 218
  Total duration:               8.92 h  (535.1 min)

── ASR Transcription Pairs ────────────────────────────
  Total pairs:                  3 218
  Transcribed:                  3 215
  Transcription errors:         3
  Review status breakdown:
    approved      1 200  (37.3%)
    pending       2 015  (62.6%)
    reject            3  (0.1%)

── Text Chunks ─────────────────────────────────────────
  Total chunks:                 8 542
  Total characters:         1 923 400
  Approx. words:              384 680
  Unique source files:             23
  Review status breakdown:
    approved      5 100  (59.7%)
    pending       3 440  (40.3%)
════════════════════════════════════════════════════════
```

---

## Manifest Standards

| File | Key column | Updated by |
|------|-----------|------------|
| `audio_manifest.csv` | `sample_id` | `extract_audio.py`, `clean_audio.py` |
| `audio_segments_manifest.csv` | `segment_id` | `segment_audio.py` |
| `audio_text_pairs.csv` | `pair_id` | `transcribe_segments.py` |
| `text_manifest.csv` | `text_id` | `build_text_dataset.py` |

All IDs are deterministic (SHA-256 of the source path / content). Manifests are **upserted**, never fully overwritten, so running scripts in any order and any number of times is safe.

---

## Dataset Philosophy

- **Traceable** — Every segment and chunk can be traced back to its source.
- **Reviewable** — Clear `pending / approved / reject / needs_fix` status with free-text notes.
- **Consistent** — One schema, one workflow, one language target (Albania-Albanian).
- **Reproducible** — Same inputs + scripts + config → same outputs.
- **Auditable** — Source files are never modified; logs in `logs/` record every run.

---

## Language Target

**Albania-Albanian (sq-AL)** — All transcription, text cleaning, and review target this language variant. Keep the project focused unless you explicitly add other variants later.

---

## Logs

Each script writes a log file to `logs/<script_name>.log` (DEBUG level) and prints INFO-level messages to the console.

---

## Extending the Pipeline

- **Add more sources**: drop files into `raw_sources/` and re-run.
- **Change ASR model**: edit `transcription.model_name` in `config.yaml` or pass `--model`.
- **Change segmentation parameters**: edit `segmentation.*` in `config.yaml`.
- **Add word-level alignment**: post-process `audio_text_pairs.csv` with a forced-alignment tool (e.g. Montreal Forced Aligner).
- **Export training set**: filter `audio_text_pairs.csv` for `review_status == approved` and use `segment_path` + `final_transcript` (or `raw_transcript` if not reviewed).
