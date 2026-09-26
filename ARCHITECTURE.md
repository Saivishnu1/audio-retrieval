# System Architecture

Two independent pipelines and a deliberately disconnected evaluation harness.

**Visual diagrams**: https://claude.ai/artifact/KmHmebemS5CyTBEU5zdWyc

- **Offline ingestion** (`ingestion/ingest.py`, `ingestion/chunking.py`,
  `core/embedder.py`): runs once per audio file. Audio goes in;
  speaker-labeled, embedded, indexed chunks come out in Postgres.
- **Online query** (`retrieval/retrieve.py`): runs once per search request.
  A four-node LangGraph that checks embedder consistency, runs two
  independent search legs, and fuses their ranked results.
- **Evaluation harness** (`eval/golden_queries.yaml`, `eval/validate_queries.py`,
  `eval/run_eval.py`): reads the same raw transcripts ingestion reads, never
  ingestion's output, so the same 41 labeled queries can grade any
  configuration of the system fairly.

## Why it's split this way

Ingestion and query run at completely different frequencies (once per file
vs. once per search), so keeping them as separate pipelines is what lets the
`keyword_leg` and `semantic_leg` nodes be individually disabled for the
ablation evaluation. The same query-time graph produces the `keyword_only`,
`semantic_only`, and `hybrid_rrf` rows in the results table just by toggling
which nodes run, rather than needing three separate implementations.

The evaluation harness is deliberately disconnected from the ingestion
pipeline's output. `golden_queries.yaml`'s labels come from parsing
`transcripts/*.txt` directly, the same raw source the chunking step reads,
never from the `chunks` table or any chunk id. This is a structural choice,
not a convention: `validate_queries.py` has no import of the chunking module
or the database client at all, so a label cannot accidentally encode a
particular chunking run's boundaries. It's what lets the same 41 labeled
queries fairly re-score the system after a chunking change, an embedder
swap, or an RRF weighting change, none of which the ground truth needs to
know about.

## Offline ingestion, in order

1. **Transcribe + diarize** (Deepgram, hosted): audio in, speaker-labeled
   turns with segment-level timestamps out.
2. **Chunk** (`chunking.chunk_transcript`): one diarized turn becomes one
   chunk by default; a turn over ~75 seconds splits at sentence boundaries
   into 20-60 second pieces; a turn under ~5 words merges into an adjacent
   same-speaker chunk, or is dropped if isolated. Every chunk carries
   exactly one speaker, plus file id and absolute start/end seconds.
3. **Split into two text fields**: `text` (the chunk's own words only) and
   `embedding_input` (the last 1-2 sentences of the immediately preceding
   turn, prepended to the chunk's own text). `text` feeds keyword indexing
   and display; `embedding_input` feeds the embedder only. The two are kept
   structurally separate all the way through: the database's `tsv` column
   is `GENERATED ALWAYS AS (to_tsvector('english', text))`, generated from
   `text` alone, and the ingestion call site passes `embedding_input`, never
   `text`, to the embedder.
4. **Embed** (`OpenAIEmbedder`, `text-embedding-3-small`, 1536-dim; a local
   `sentence-transformers`/`bge-small-en-v1.5` fallback exists behind the
   same interface): the resulting vector, plus which model produced it, is
   written to the same chunk row.

Result on this dataset: 259 chunks, duration range 1-63 seconds, mean 14.2s.

## Online query, in order

A LangGraph with four nodes (`retrieve.py`):

1. **`check_embedder`**: reads the `embedding_model` value stamped on
   `chunks` at ingest time and compares it to the model about to embed the
   incoming query. Raises `EmbedderMismatchError` and stops before running
   semantic search if they differ, rather than silently returning results
   computed in an inconsistent vector space. (Verified directly: forcing a
   mismatched model name causes the error to fire, not silently pass.)
2. **`keyword_leg`**: `websearch_to_tsquery` against the `tsv` generated
   column, ranked by `ts_rank_cd` (rewards query terms appearing close
   together, not just present anywhere). Falls back to a `pg_trgm` fuzzy
   match when full-text search returns few or no results, to catch
   ASR-misspelled rare terms.
3. **`semantic_leg`**: cosine similarity between the query's embedding and
   each chunk's `embedding`, via the HNSW index.
4. **`fuse`**: combines both legs' ranked lists with Reciprocal Rank Fusion
   (k=60). RRF needs only each leg's rank position, not its raw score, since
   `ts_rank_cd` and cosine similarity live on incompatible scales and would
   otherwise need manual normalization/weighting.

Every result reports file, start/end seconds, speaker, a text snippet, and
which leg(s) contributed to it. Ingestion is idempotent: re-running against
the same 259 chunks produces the same row count with no duplicates
(verified directly).

## Data model

```
files      (id, file_path, title, podcast, source_url, start_sec, end_sec, sha256)
speakers   (id, file_id, label, real_name)
chunks     (id, file_id, speaker_id, chunk_index,
            start_sec, end_sec,
            text,                          -- own words only: display, tsv, timestamps
            embedding_input,               -- prior-turn context + text: embedding only
            source,                        -- base | split | merged
            tsv,                           -- GENERATED from text, GIN-indexed
            embedding vector(1536),        -- primary (OpenAI), HNSW-indexed, cosine
            embedding_model,               -- provenance, checked at query time
            embedding_local vector(384),   -- local (bge-small), HNSW-indexed, cosine
            embedding_local_model)         -- provenance for the local column
```

The two embedding column pairs are independent: `ingest --embedder openai`
writes `embedding`/`embedding_model`, `ingest --embedder local` writes
`embedding_local`/`embedding_local_model`, and `search(target=...)` picks
which pair the semantic leg reads. The embedder check runs against
whichever pair is selected.

The schema is defined in `src/core/models.py` and applied with the
migrations in `alembic/versions/`. Every table, column, and index
(including `tsv` and the GIN/HNSW indexes) is declared in the model, so
`alembic revision --autogenerate` produces an empty diff against the live
database.

## What's deliberately not in the diagrams

Raw audio files are not committed to the repository, only `manifest.yaml`
(source URLs, offsets into the original video, checksums). Clips were
extracted with `src/youtube_audio` and trimmed to those offsets with
`ffmpeg`. The diagrams cover the processing pipeline, not dataset
provenance, which is documented in `DESIGN_REPORT.md`.

## Related documents

- `README.md`: project overview and a map of these documents.
- `GUIDELINES.md`: setup, run commands, and where to find eval output.
- `DESIGN_REPORT.md`: design rationale, evaluation, and limitations.
- `RESULTS.md`: OpenAI vs. local embedder comparison.
- `AI_DISCLOSURE.md`: how AI tools were used to build this project, and
  which AI/ML components run inside the shipped system.
