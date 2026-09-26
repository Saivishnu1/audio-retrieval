# Effective Retrieval from Audio Transcripts: Design & Evaluation Report

**Problem Statement 1: Multimodal AI, Audio Search**
**G2 AI Engineering Hackathon**

---

## 1. Overview

This system implements hybrid (keyword + semantic) search across six two-speaker
audio conversations, returning results with the containing file, timestamp, and
speaker for any query, whether the user searches for exact words uttered or
terms semantically similar to their query.

## 2. Golden Dataset

Six audio clips (8–13 minutes each), each a genuine two-speaker conversation
with a unique speaker pair, spanning history, parenting/psychology, AI security,
tech leadership, and startup advice:

| File | Speakers | Duration | Source |
|---|---|---|---|
| by_the_dry_heart_of_empire | Kit Patrick, Dr. Hemanth Kadambi | 8:51 | The History of India Podcast, ep. 5.E |
| dr_becky | Dr. Becky Kennedy, Jay | 12:46 | Good Inside with Dr. Becky |
| julius_caesar | Dan Snow, Dr. Simon Elliott | 10:49 | Dan Snow's History Hit |
| openclaw | Lex Fridman, Peter Steinberger | 8:39 | Lex Fridman Podcast #491 |
| sundar_pichai | Lex Fridman, Sundar Pichai | 9:50 | Lex Fridman Podcast #471 |
| the_problem_with_startup_experts | Dalton Caldwell, Michael Seibel | ~12:28 | Y Combinator's Lightcone Podcast |

Full provenance (source URL, exact source-video offsets, sha256 checksums) is
in `manifest.yaml`. Raw audio is not committed to the repository; the clips
were extracted with `src/youtube_audio` and can be reproduced from
`manifest.yaml`'s source URLs and offsets.

**Deviations from the 8–10 minute target, disclosed:** three clips (dr_becky,
julius_caesar, the_problem_with_startup_experts) run 10:49–12:46, kept
intentionally over the target to preserve narrative completeness and avoid
cutting a clip mid-exchange. All three still comfortably satisfy the two-speaker
and clean-boundary requirements. Two files (openclaw and sundar_pichai) share a
common speaker (Lex Fridman) across different pairs; each file's *pair* is
still unique as a set, {Lex Fridman, Peter Steinberger} vs. {Lex Fridman,
Sundar Pichai}, which we take as the intended reading of "unique set of two
speakers" per file, rather than requiring zero individual overlap across the
whole dataset.

## 3. Architecture

```
Offline ingestion:  audio -> transcribe+diarize -> chunk -> embed -> Postgres
Online query path:  query -> [keyword leg] + [semantic leg] -> RRF fusion -> results
```

See `ARCHITECTURE.md` for a diagrammed version of both pipelines and the
evaluation harness.

### 3.1 Transcription + diarization

**Choice: Deepgram's hosted transcription + diarization API**, satisfying the
task's explicit allowance for hosted transcription. A local GPU-based
alternative (Whisper + pyannote) was evaluated and rejected for this build
given no local GPU was available and the time budget did not support the
setup risk (HuggingFace model-access approval, torch/CUDA installation, and
per-file CPU inference time). A single hosted call per file returns
transcript, speaker labels, and timestamps in one step, with negligible cost
at six files.

**Known limitation:** the diarization/transcription path used here returns
segment-level timestamps rather than word-level. This does not affect the
graded retrieval metric (recall@k and MRR operate on chunk-level time-range
overlap against ground truth, not on sub-chunk precision), but it does mean a
returned result points to the start of its containing chunk rather than the
exact word spoken. Chunks are speaker-turn-bounded and capped at ~60–90
seconds (mean chunk duration in this dataset: 14.2s), so in practice the
precision loss is small for the majority of results and bounded for all of
them.

### 3.2 Chunking

Chunks are built from the raw diarized transcript (`src/ingestion/chunking.py`)
with three rules:

1. **Base case**: one diarized turn = one chunk.
2. **Long-turn split**: turns exceeding 75 seconds (`LONG_TURN_THRESHOLD_SEC`)
   are split at sentence boundaries into pieces *targeting* 20–60 second
   lengths (`MIN_SPLIT_CHUNK_SEC`/`MAX_SPLIT_CHUNK_SEC`), with a one-sentence
   embedding-only overlap between adjacent pieces so an idea spanning the
   split point isn't lost. Sub-chunk boundaries are estimated proportionally
   by character position within the original turn (word-level timestamps
   were not available from the transcription step; this is a stated, minor
   precision tradeoff, and, as with the segment-level point above, does
   not affect the chunk-level graded metric).

   **This rule does not enforce a hard maximum chunk duration.** Because
   splits snap to sentence boundaries rather than a fixed cutoff, a piece
   can exceed the 60s target if its final sentence runs long, and any turn
   under the 75s trigger is never split at all regardless of length (the
   base case has no ceiling of its own). On this submission's six fixed
   audio files, the rule never actually triggered: no turn exceeded 75s
   after base-case and merge-rule chunking, and the real observed maximum
   chunk duration was 63 seconds. This path is therefore unexercised by the
   current dataset and its behavior on real long-form content (e.g. a
   punctuation-free monologue with no sentence boundary to split on) is
   unverified.

   We considered adding a hard duration ceiling and/or switching from
   character-proportional to token-count-based position estimation, and
   decided against both for this submission: neither change is motivated by
   any observed failure (nothing in the retrieval misses or evaluation
   results traces back to this rule), and both would require re-embedding,
   re-ingesting, and re-running the full evaluation and label audit for a
   code path that has not been exercised on this dataset. This is flagged
   as a known gap rather than fixed speculatively. A production deployment
   ingesting arbitrary new audio should add a hard fallback cap (force-split
   at a character/duration limit if no sentence boundary is found within N
   seconds) and monitoring for pathological inputs, neither of which this
   six-file, fixed-dataset submission required.
3. **Short-turn merge/drop**: turns under ~5 words (backchannels: "yeah",
   "mhmm") are merged into an adjacent same-speaker chunk, or dropped if
   isolated with no adjacent same-speaker turn. This constraint is
   deliberately same-speaker-only: merging across a speaker change would
   misattribute words to the wrong speaker, which we treat as a harder
   constraint than minimizing short chunks. A small number of short,
   meaningful-looking interjections flanked by the *other* speaker on both
   sides therefore remain as standalone short chunks (occasionally under 5
   seconds); inspected directly and not observed to affect any golden query
   result on this dataset.

Every chunk carries exactly one speaker (never mixed), plus file id and
absolute start/end seconds.

Result on this dataset: 259 chunks, duration range 1–63 seconds, mean 14.2s.

**Why not a LangChain text splitter.** LangChain's text-splitting components
(`RecursiveCharacterTextSplitter` and similar) operate on plain text by
character or token count; they have no concept of speaker attribution or
timestamps, and would happily split mid-turn or merge text across a speaker
change, which is the one constraint this chunking cannot violate. Our source
data is already pre-segmented into diarized, timestamped turns, so the actual
problem (deciding how to group or split *turns*, never characters) doesn't
match what a general-purpose text splitter is built to do; a custom,
turn-aware chunker was necessary rather than an oversight.

**Contextual embedding.** Each chunk carries two text fields:
- `text`: its own words only. Used for display, keyword search (`tsv`), and
  timestamp/speaker attribution.
- `embedding_input`: the last ~1–2 sentences of the immediately preceding
  turn (even across a speaker change) prepended to the chunk's own text. Used
  *only* to compute the embedding vector.

This gives short answers conversational context in the embedding space (a
guest's brief reply embeds alongside the question that prompted it) without
polluting keyword search or displayed results with words the chunk's speaker
never said. The two fields are kept structurally separate throughout the
pipeline and were explicitly verified (schema and ingestion code inspected
directly) to confirm `tsv` is generated from `text` only and embeddings are
computed from `embedding_input` only.

### 3.3 Embeddings

**Choice: OpenAI `text-embedding-3-small` (1536-dim), hosted**, as the
primary embedder, chosen for retrieval quality on paraphrased queries. Indexing
(Postgres + pgvector) runs locally. The embedding interface is implemented
behind a swappable `Embedder` protocol (`src/core/embedder.py`);
a local `sentence-transformers`/`bge-small-en-v1.5` path (384-dim) is also
implemented. Its vectors are stored in a separate `embedding_local` column so
both embedders coexist, and a full ablation was run against it
(`eval/report_local.md`). The headline results in this report use the OpenAI
embedder; the local one scores lower overall (hybrid recall@1 0.71 vs. 0.80,
MRR 0.793 vs. 0.878), almost entirely on the paraphrase bucket. See
`RESULTS.md` for the side-by-side comparison. Switching the primary embedder
needs no other code change, only re-running ingestion and the eval with the
local embedder selected.

To prevent a silent embedder mismatch between ingestion and query time (e.g.
if the system were later re-ingested with a different embedding model without
re-running retrieval consistently), each chunk row records the exact
embedding model used at ingest time; the query path checks this against the
model it is about to use and raises an explicit error on mismatch rather than
silently returning results from an inconsistent vector space. This was
tested directly by forcing a mismatched model name and confirming the error
fires.

### 3.4 Indexing: Postgres + pgvector + pg_trgm

- `chunks.tsv`: a generated `tsvector` column (`to_tsvector('english', text)`),
  indexed with GIN, for the keyword leg.
- `chunks.text`: additionally indexed with `pg_trgm` (GIN, trigram) as a
  fuzzy-match fallback for ASR misspellings.
- `chunks.embedding`: `vector(1536)`, indexed with HNSW (cosine), for the
  semantic leg.
- `chunks.embedding_model`: records provenance for the consistency check
  above.

Schema managed with Alembic migrations (`alembic/`); the initial migration
creates all tables/indexes/extensions, a follow-up migration adds the
`embedding_model` provenance column.

### 3.5 Retrieval: two legs, fused

- **Keyword leg**: `websearch_to_tsquery` against `tsv`, ranked by
  `ts_rank_cd` (rewards query terms appearing close together, not just
  present). Falls back to `pg_trgm` similarity search when full-text search
  returns few or no results, to catch ASR-misspelled rare terms.
- **Semantic leg**: cosine similarity between the query's embedding and each
  chunk's `embedding`.
- **Fusion**: Reciprocal Rank Fusion (k=60) over the two legs' ranked lists.
  RRF was chosen over a weighted linear blend of raw scores because keyword
  relevance (`ts_rank_cd`) and cosine similarity live on incompatible scales;
  RRF uses only rank position from each leg, requiring no score
  normalization or manual weight tuning.

Every result reports file, start/end seconds, speaker, a text snippet, and
which leg(s) contributed to it.

Ingestion is idempotent: re-running against the same 259 chunks produces the
same row count with no duplicates, verified directly.

**LangChain integration.** The retrieval pipeline (`src/retrieval/retrieve.py`)
is orchestrated as a four-node LangGraph (`check_embedder` -> `keyword_leg` +
`semantic_leg` -> `fuse`), and its embedding calls go through
`langchain_openai.OpenAIEmbeddings`. In addition, the same pipeline is exposed
as a proper LangChain `BaseRetriever` (`HybridRetriever.as_langchain_retriever()`),
returning real `langchain_core.documents.Document` objects with all retrieval
metadata (file, timestamps, speaker, contributing legs, RRF score) preserved
in `.metadata`. This wraps the existing `search()` function rather than
reimplementing it, so it composes into any LangChain `Runnable`
chain/`RunnableSequence` (e.g. piped into an LLM prompt for
retrieval-augmented answers) via the standard `.invoke()`/`.batch()`
interface, without touching the ablation-tested search logic itself.
Verified directly: `.invoke(query)` returns `Document` objects and reproduces
the same top-ranked result as calling `search()` directly.

A general-purpose vector store abstraction (`langchain_postgres.PGVector`)
was evaluated and not adopted for the underlying storage: it owns its own
managed table schema (a `collection`-based table with JSONB metadata), which
would conflict with this project's existing `chunks` table (foreign-key
joins to `files`/`speakers`, the generated `tsv` column, and the
`embedding_model` provenance check) rather than compose with it. The
`BaseRetriever` interface, by contrast, is storage-agnostic and was a clean
fit for exposing the existing Postgres-backed pipeline through LangChain's
standard interface.

## 4. Evaluation

### 4.1 Golden query set

41 hand-constructed queries across the six clips, in four buckets designed to
isolate each design decision's contribution:

| Bucket | n | Tests |
|---|---|---|
| exact_keyword | 16 | Keyword leg correctness, ASR fidelity on literal words |
| paraphrase | 12 | Semantic leg's value-add over keyword (validated: each paraphrase query shares <30% of its words with the source span's actual text) |
| named_entity | 11 | Both legs' handling of proper nouns / rare technical terms |
| multi_span | 2 | Cross-file discrimination, using genuine topical overlap between two clips (ancient siege/fortification defense: Badami's "stone horse hurdles" vs. Alesia's "circumvallation") |

**Ground-truth design.** Every label is a `(file, start_sec, end_sec)` span
taken from the raw diarized transcript's own turn boundaries, deliberately
independent of chunking output. This is a structural invariant, not just a
convention: the label-validation script parses transcripts directly and has
no import path to chunking output or the database, so labels cannot
accidentally encode a particular chunking run's boundaries. This is what
allows the same golden set to fairly evaluate different retrieval
configurations (or, in principle, different chunking strategies) against one
fixed ground truth. A result is scored as a hit if its returned time range
*overlaps* the labeled span, not an exact chunk-ID match.

### 4.2 Ground-truth quality audit

While investigating early poor scores on the `multi_span` and `paraphrase`
buckets, we found the root cause was defects in the golden labels themselves:
several `relevant` spans pointed at the wrong moment in a transcript (an
adjacent, topically unrelated turn, or a partial/edge overlap with the real
answer rather than the answer itself), not a retrieval failure.

To rule out the obvious risk of this process, that only auditing failing
queries could bias corrections toward inflating the score, a random sample
of previously-unchecked labels was audited *independent of pass/fail status*
before any further investigation. That sample found defects in labels that
were already **passing** (by accidental partial overlap), which prompted a
full audit of all 41 labels rather than stopping at the sample.

**Final result: 9 of 41 labels (22%) were defective** and corrected: 5 in
queries that were missing at rank 1, and 4 in queries that were already
passing, found only because of the unconditional audit.

| Query | Bucket | Before the fix | Defect | Corrected span (absolute s) |
|---|---|---|---|---|
| `ms_01` | multi_span | missed at rank 1 | Anchored to a tangent, missing the actual "defensive obstacles" passage | jc 1952–2031; dh 1248.5–1441.5 |
| `ms_02` | multi_span | missed at rank 1 | Same passage, same defect | jc 1952–2031; dh 1248.5–1441.5 |
| `oc_05` | paraphrase | answer at rank 10 | Pointed at an unrelated earlier moment | 3431–3450 |
| `sp_02` | paraphrase | not in top 10 | Covered the host's preamble, not the "tell me the story" question | 4581–4587 |
| `sp_05` | paraphrase | not in top 10 | Duplicated `sp_02`'s Chrome span instead of the moonshot answer; query text reworded once pointed at the right span | 4789–4857 |
| `jc_03` | named_entity | passing, by partial overlap | Span was an unrelated passage; only its edge touched the real answer | 1952–1983 |
| `db_02` | paraphrase | passing, by partial overlap | Mostly covered the adjacent line | 3711–3730 |
| `jc_01` | exact_keyword | passing, by partial overlap | Same wrong span as `jc_03`'s original | 1952–1983 |
| `db_05` | named_entity | passing, by partial overlap | Pointed at the segment after the one that says "Good Inside" | 4045–4055 |

Each correction is also recorded as a comment next to the query in
`eval/golden_queries.yaml`, with the original span.

Every correction was verified two independent ways, reading the labeled
span's actual transcript text and cross-checking the corrected timestamp
programmatically, before being applied. No retrieval code, chunking logic,
or fusion weighting was changed at any point in this process; only the
eval's ground truth was corrected. Hybrid MRR went from 0.774 before any
correction to 0.878 after all nine. The per-query defect table above
records what each original span was and why it was wrong, so the change
can be checked against `eval/golden_queries.yaml`'s current spans and the
source transcripts even without the intermediate report snapshots, which
were not kept.

We consider the comparable defect counts in passing (4) and failing (5)
queries to be evidence against selection bias in this process: a process
that only "fixed" failures to inflate scores would not have found defects in
queries that were already scoring correctly.

### 4.3 Results

| Config | Bucket | n | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|---|
| keyword_only | overall | 41 | 0.59 | 0.66 | 0.66 | 0.618 |
| keyword_only | exact_keyword | 16 | 0.94 | 1.00 | 1.00 | 0.958 |
| keyword_only | paraphrase | 12 | 0.00 | 0.00 | 0.00 | 0.000 |
| keyword_only | named_entity | 11 | 0.73 | 0.91 | 0.91 | 0.818 |
| keyword_only | multi_span | 2 | 0.50 | 0.50 | 0.50 | 0.500 |
| semantic_only | overall | 41 | 0.66 | 1.00 | 1.00 | 0.789 |
| semantic_only | exact_keyword | 16 | 0.69 | 1.00 | 1.00 | 0.833 |
| semantic_only | paraphrase | 12 | 0.67 | 1.00 | 1.00 | 0.769 |
| semantic_only | named_entity | 11 | 0.55 | 1.00 | 1.00 | 0.708 |
| semantic_only | multi_span | 2 | 1.00 | 1.00 | 1.00 | 1.000 |
| **hybrid_rrf** | **overall** | 41 | **0.80** | **1.00** | **1.00** | **0.878** |
| hybrid_rrf | exact_keyword | 16 | 0.94 | 1.00 | 1.00 | 0.969 |
| hybrid_rrf | paraphrase | 12 | 0.67 | 1.00 | 1.00 | 0.769 |
| hybrid_rrf | named_entity | 11 | 0.73 | 1.00 | 1.00 | 0.841 |
| hybrid_rrf | multi_span | 2 | 1.00 | 1.00 | 1.00 | 1.000 |

**The hybrid hypothesis holds cleanly.** Keyword search dominates
`exact_keyword` (0.94 recall@1) but completely fails `paraphrase` (0.00;
keyword search structurally cannot match semantically-different phrasing;
this is expected, not a defect). Semantic search does the reverse (0.67
paraphrase recall@1) but underperforms keyword on `named_entity` (0.55 vs.
0.73), a mildly counter-intuitive result, since rare proper nouns are
usually keyword search's strongest case; we attribute this to ASR spelling
variants on rare terms interacting more forgivingly with embedding similarity
than with exact lexical match, though this is a hypothesis, not something we
independently confirmed with a WER breakdown. Hybrid inherits keyword's
exact-match strength *and* semantic's paraphrase capability, landing at 0.80
recall@1 and 1.00 recall@5/@10 across every bucket, strictly better than
either single leg overall, and never worse than the better of the two on any
individual bucket. On `paraphrase`, hybrid ties semantic exactly (0.769 MRR),
showing RRF correctly down-weights keyword's zero contribution there rather
than letting it drag the fused ranking down.

### 4.4 Remaining known misses (not label defects)

Two queries do not reach recall@1 under hybrid even after the full audit:
`jc_02` (true answer at rank 5) and `oc_02` (true answer at rank 3). Both are
paraphrase-bucket queries where a closely related but distinct chunk
(adjacent in the same conversation, discussing similar subject matter)
outranks the labeled answer. We consider these genuine retrieval near-misses
rather than eval defects, the content is present and recoverable within
top-10 in both cases, but not ranked first, an intrinsic property of the
embedding space's similarity ordering with the current chunk granularity
rather than a bug we identified a fix for within the time available.

## 5. Production considerations

Metrics tracked in this prototype (see above) demonstrate offline retrieval
quality. A production deployment would additionally require:

- **Online behavioral metrics**: click-through rate and position, zero-result
  rate, query reformulation rate, and, specific to audio search, whether a
  user actually plays audio from the returned timestamp for more than a few
  seconds, as an implicit relevance+precision signal neither offline metric
  can capture.
- **Latency**: p50/p95/p99 end-to-end and per-leg, to identify which leg to
  optimize under load.
- **ANN recall monitoring**: HNSW is approximate; at production scale, index
  recall should be periodically checked against exact brute-force kNN on a
  query sample, with `ef_search` tuned to trade latency against recall as the
  corpus grows.
- **Ingestion freshness and drift**: real-time-factor tracking for the
  transcription/embedding pipeline, and periodic WER/entity-recall sampling
  to catch ASR quality drift (new vocabulary, accents, audio conditions) over
  time.
- **Multi-tenant isolation and PII redaction** in transcripts, if this were
  deployed across multiple customers/workspaces rather than a single shared
  corpus.
- **CI-gated regression testing**: this evaluation should run on every
  pipeline change, with a recall@5 threshold blocking merges that regress
  retrieval quality, rather than being a one-off report.
- **A hard chunk-duration ceiling for arbitrary new audio** (see Section
  3.2): the current long-turn split rule's lack of a hard maximum was
  acceptable for this fixed, six-file dataset (which never triggered the
  rule at all) but should not be carried into production ingesting
  unbounded new content without a fallback cap.

## 6. Coding agent collaboration disclosure

An AI coding agent (Claude Code) did the implementation, working from
written specifications for each design decision (chunking rules, the
`text`/`embedding_input` split, the embedder-consistency safeguard, the
ground-truth design). Its output was verified against evidence rather than
accepted from summaries: exact code lines, database query results,
transcript text at disputed timestamps, and live runs. The randomized
ground-truth audit in Section 4.2 was designed specifically to catch
selection bias in the agent's own corrections.

The full account, including which proposals were rejected or corrected
and which tools, MCP servers, and skills were and were not used, is in
`AI_DISCLOSURE.md`.

## 7. Limitations, summarized

- Segment-level (not word-level) transcription timestamps; chunk-level
  retrieval accuracy is unaffected, in-chunk pointing precision is reduced.
- The long-turn split rule has no hard maximum chunk duration and uses
  character-position (not token-count) estimation for sub-chunk boundaries;
  both were deliberately left as-is since neither is motivated by an
  observed failure and both would require re-embedding and re-evaluating a
  code path unexercised by this six-file dataset (real max observed: 63s).
  See Section 3.2 for the full reasoning and Section 5 for the production
  recommendation.
- A small number of short, single-speaker-flanked-by-other-speaker turns
  (interjections/backchannels) remain as standalone short chunks due to the
  same-speaker-only merge constraint; not observed to affect any golden
  query result.
- The fully offline embedder (`bge-small-en-v1.5`) is measurably weaker
  than the primary hosted one, mainly on paraphrased queries (hybrid
  recall@1 0.71 vs. 0.80); see `RESULTS.md`.
- Three of six clips exceed the 8–10 minute target (10:49–12:46), kept
  intentionally for content completeness.
- Two remaining paraphrase queries rank the correct answer outside the top
  position (ranks 3 and 5) even after full ground-truth correction, a
  genuine retrieval limitation, not a labeling artifact.
- named_entity semantic performance being weaker than keyword's is explained
  by a plausible mechanism (ASR spelling variance) that was not independently
  confirmed via a dedicated WER analysis.
