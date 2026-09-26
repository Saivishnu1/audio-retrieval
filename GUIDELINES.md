# Running This Project

Step-by-step setup and the commands to actually run the system, in order.
See `README.md` for a shorter overview and `ARCHITECTURE.md`/`DESIGN_REPORT.md`
for the design.

## 1. Prerequisites

- **Python 3.9+**
- **[`ffmpeg`](https://ffmpeg.org/download.html)** on `PATH` (only needed if
  re-extracting/re-trimming audio clips via `youtube_audio`).
- **A local PostgreSQL instance** (tested against Postgres 16) with two
  extensions available to install: `vector` (pgvector) and `pg_trgm`. You
  don't need to enable them yourself; the Alembic migrations do that.
  - Confirm they're available: connect with `psql` (or any client) and run
    `SELECT * FROM pg_available_extensions WHERE name IN ('vector','pg_trgm');`
    both rows should come back.
  - Create an empty database for this project, e.g. `audio_retrieval`.
- **API keys**:
  - `DEEPGRAM_API_KEY`, only needed if re-transcribing audio (get one free
    at [console.deepgram.com](https://console.deepgram.com/)).
  - `OPENAI_API_KEY`, needed for the primary embedder and for search.

## 2. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
uv sync --extra dev
```

This installs the project's dependencies *and* the project itself
(editable), so `retrieval`/`transcription`/`youtube_audio` import from
anywhere without a `PYTHONPATH=` hint, and installs `pytest` (the `dev`
extra). If you don't have `uv`, `pip install -e ".[dev]"` does the same
thing, though on some Windows setups plain `pip`'s build step can fail
with a `WinError 32` temp-file lock (a local environment issue, not
specific to this project); `uv` sidesteps it reliably.

Copy `.env.example` to `.env` and fill in your own values:

```bash
cp .env.example .env
```

```
DEEPGRAM_API_KEY=your-deepgram-api-key-here
OPENAI_API_KEY=your-openai-api-key-here
POSTGRES_USER=your-postgres-username-here
POSTGRES_PASSWORD=your-postgres-password-here
POSTGRES_DB=audio_retrieval
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

## 3. Apply the database schema

```bash
alembic upgrade head
```

This creates the `files`, `speakers`, and `chunks` tables, enables the
`vector`/`pg_trgm` extensions, and builds all indexes (GIN for keyword
search, HNSW for semantic search). Safe to re-run; it's a no-op if already
at the latest revision.

Check it worked:

```bash
alembic current
```

should print the latest revision id, e.g. `8435639de18d (head)`.

## 4. Ingest the golden dataset

The six audio clips are already transcribed (`transcripts/*.txt`) and
described in `manifest.yaml`; you don't need to re-run transcription to use
the existing dataset. Ingest them into Postgres (chunks, keyword index,
embeddings):

```bash
python -m ingestion.ingest --manifest manifest.yaml --embedder openai
```

This is idempotent (safe to re-run; it upserts, never duplicates). It should
print a per-file chunking summary and end with:

```
Ingested 259 chunks total.
```

To ingest with the local (offline, `sentence-transformers`/`bge-small`)
embedder instead of OpenAI, into a separate column so it doesn't disturb
the primary vectors:

```bash
python -m ingestion.ingest --manifest manifest.yaml --embedder local
```

## 5. Search

**CLI**, one query at a time:

```bash
python -m retrieval.retrieve "your query here" --top-k 5
```

Add `--embedder local` to search against the local-model vectors instead of
OpenAI's (only works after step 4's local-embedder ingestion has been run).

**Python usage** (calling `search()` directly, in-process, instead of the CLI):

```python
from dotenv import load_dotenv
load_dotenv()

from core.embedder import OpenAIEmbedder
from retrieval.retrieve import search

results = search("your query here", OpenAIEmbedder(), top_k=5)
for r in results:
    print(r.file, r.start_sec, r.end_sec, r.speaker, r.legs)
    print(r.text)
```

Each result reports: `file`, `start_sec`/`end_sec`, `speaker`, `text`
(snippet), `legs` (which of keyword/semantic/fuzzy found it), `rrf_score`.

## 6. Re-transcribing audio (optional)

Only needed if you add new audio clips. Extract audio from a YouTube URL:

```bash
python -m youtube_audio.cli "https://www.youtube.com/watch?v=VIDEO_ID" -o downloads -f mp3
```

Transcribe + diarize a directory of clips:

```bash
python -m transcription.cli clips --output-dir transcripts --format txt
```

Then add the new clip's entry to `manifest.yaml` (file path, title, source
URL, `start_sec`/`end_sec` offsets into the original video, speaker names)
before ingesting it (step 4).

## 7. Where to check evaluation results

The evaluation compares three retrieval configurations
(keyword-only / semantic-only / hybrid) against a 41-query golden set, and
writes a markdown report.

**To (re-)run it yourself:**

```bash
python eval/validate_queries.py       # sanity-checks golden_queries.yaml first
python eval/run_eval.py                 # against the primary OpenAI vectors
python eval/run_eval.py --embedder local # against the local bge-small vectors
```

**Where the results land:**

| File | What it is |
|---|---|
| `eval/report.md` | Current ablation results (primary/OpenAI embedder): recall@1/5/10 and MRR per bucket, per config, plus the ground-truth audit trail. **Start here.** |
| `eval/report_local.md` | Same ablation, run against the local (`bge-small`) embedder instead. |
| `eval/golden_queries.yaml` | The 41 labeled queries themselves (query text, bucket, and the `(file, start_sec, end_sec)` span(s) that count as a correct answer). |

`eval/report.md`'s headline numbers (hybrid, all buckets):

```
recall@1 = 0.80   recall@5 = 1.00   recall@10 = 1.00   MRR = 0.878
```

See `DESIGN_REPORT.md` section 4 for the full breakdown per bucket
(exact_keyword / paraphrase / named_entity / multi_span) and the reasoning
behind the ground-truth audit.

## 8. Running the tests

```bash
pytest
```

Covers audio extraction, transcription, and chunking logic (mocked
external APIs, no live Deepgram/OpenAI/Postgres calls needed for these).

## Troubleshooting

- **`ModuleNotFoundError` for anything in `youtube_audio`/`transcription`/
  `retrieval`**: the project isn't installed in this environment. Run
  `uv sync --extra dev` (step 2) once per virtualenv; after that, no
  `PYTHONPATH=` prefix is needed for any command in this guide.
- **`No solution found when resolving dependencies` mentioning
  `langchain-postgres` and a Python version**: your virtualenv's Python is
  older than `requires-python` in `pyproject.toml` allows for that
  dependency. Recreate the virtualenv with Python 3.10+.
- **`EmbedderMismatchError` when searching**: the embedder you're querying
  with doesn't match what the stored vectors were embedded with (e.g. you
  ingested with `--embedder openai` but searched with `--embedder local`,
  or vice versa). Use the same `--embedder` flag for ingestion and search,
  or re-ingest with the one you want to search with.
- **Alembic connection errors involving `psycopg` (not `psycopg2`)**: this
  project deliberately pins `postgresql+psycopg2://` in
  `alembic/env.py`/`core/db.py`; SQLAlchemy's `psycopg` (v3) dialect
  was found to hang indefinitely on connect in this environment. If you've
  changed the driver, switch it back to `psycopg2`.
- **`alembic revision --autogenerate` proposes dropping `tsv` or the custom
  indexes**: this shouldn't happen anymore (they're declared in
  `core/models.py`); if it recurs, something in the live database has
  drifted from what a migration created. Run `alembic current` to confirm
  you're at `head` first.
