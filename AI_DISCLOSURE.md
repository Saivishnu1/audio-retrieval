# AI Disclosure

This document discloses how AI tools were used to build this project, per the
hackathon's disclosure requirement. It covers two distinct things: (1) AI
tools used *during development* to design, implement, and verify the system,
and (2) AI/ML models used *inside the shipped system itself*. These are kept
separate below since they answer different questions: the first is about
how the code was built, the second is about how the running application
works.

---

## 1. AI tools used during development

### 1.1 Overview of the flow

Two AI surfaces were used together, in a specific division of labor:

- **Claude (chat interface)**: used for architecture decisions, design
  tradeoffs, reviewing proposed changes, verifying claims before accepting
  them, and drafting this documentation. Did not write or execute code
  directly against the project.
- **Claude Code (agentic coding tool)**: used for all actual implementation,
  writing source files, running the project's own scripts and tests,
  reading/parsing transcripts, calling the transcription and embedding APIs,
  managing the database, and running the evaluation harness.

The general pattern throughout the build was: a design decision or
implementation task was discussed and specified (in the chat interface),
Claude Code implemented it, and the result was reviewed, often by asking
Claude Code to show the exact code/output backing a claim, before moving to
the next step, rather than accepting summarized descriptions of what had been
done.

### 1.2 What each tool was directed to do

**Architecture and design decisions** made in the chat interface, then
handed to Claude Code as explicit specifications, included:

- Choosing hybrid keyword + semantic retrieval over either alone, and
  Reciprocal Rank Fusion over a weighted-score blend, with the reasoning
  written out before implementation began.
- The chunking strategy: base case (one turn = one chunk), a long-turn split
  rule (threshold, sentence-boundary splitting, proportional timing
  estimation, embedding-only overlap), and a short-turn merge/drop rule
  (word-count threshold, same-speaker-only merge constraint), specified as
  an explicit, numbered rule set before any chunking code was written.
- The `text` vs. `embedding_input` field separation (own-words-only for
  display/keyword-search/timestamps vs. prior-turn-context-prepended for the
  embedding call only), specified with an explicit instruction that the two
  must never be conflated.
- Choice of transcription/diarization provider and embedding provider,
  including reversing an initial choice (a diarization-capable model that
  turned out to require account access we didn't have) mid-build and
  redirecting to an available alternative, and the tradeoff between
  local vs. hosted embeddings (see Section 2.2).
- The embedder-consistency safeguard design: recording the embedding model
  used per chunk at ingest time and checking it against the query-time
  embedder, rather than relying on convention alone.
- The golden query set's ground-truth design: labels anchored to raw
  transcript turn timestamps, deliberately independent of chunking output,
  specified as a structural invariant (verified by confirming the validation
  script has no code path that reads chunking output at all, not merely by
  convention).
- The audit methodology used to check the golden set for labeling errors,
  including a specific instruction to randomly sample previously-unchecked
  labels *independent of pass/fail status*, to test for selection bias in
  the correction process itself.

**Implementation, verification, and iteration**, done in Claude Code,
included:

- Writing and testing the chunking module, embedder interfaces (OpenAI and a
  local `sentence-transformers` fallback), ingestion pipeline, Postgres
  schema/migrations, and the keyword/semantic/RRF retrieval pipeline.
- Building the golden query set (41 queries across four buckets) by reading
  the raw transcripts directly.
- Running the evaluation harness and ablation comparisons (keyword-only vs.
  semantic-only vs. hybrid), across multiple correction passes as labeling
  defects were found and fixed.
- Investigating and reporting specific discrepancies when asked, for example
  tracing exactly why two "multi-span" queries scored near-zero, confirming
  via direct transcript inspection (and cross-checking with a second,
  programmatic method) that the golden labels themselves pointed at the
  wrong sub-span, not that retrieval had failed.
- Simulating a proposed change (raising a chunking threshold) and reporting
  the effect on the chunk distribution *before* committing to re-chunk and
  re-ingest, so the tradeoff could be evaluated before the cost was paid.

### 1.3 Points where proposed AI output was corrected or rejected

Several points in the build involved directly correcting or overriding what
Claude Code proposed, rather than accepting it as-is:

- Rejected a proposal to re-run transcription on all six clips solely to
  recover word-level timestamps; verified first that the graded evaluation
  metric operates at chunk-level time-range overlap and is unaffected by
  sub-chunk timing precision, then accepted a cheaper proportional-estimate
  fallback instead.
- Rejected a proposal to widen two golden-label spans purely because a
  retrieval result had landed just outside them, until the underlying
  transcript content was independently verified line-by-line to confirm the
  original labels were genuinely wrong (not just narrow), and the corrected
  spans excluded off-topic material rather than being expanded to cover it.
- Caught and corrected an arithmetic error in one such correction (an offset
  calculation was off by 20 seconds) by requesting a second, independent,
  programmatic verification of the same numbers before they were written to
  the golden file.
- When directly asked "are you overfitting [the eval]?", required Claude Code
  to distinguish between changing the system under test (which did not
  happen at any point) and changing ground-truth labels (which did, with
  disclosure), and to address the real underlying risk, that only auditing
  failing queries could bias corrections toward inflating the score, by
  running an unconditional random audit of passing queries as well, which
  surfaced additional genuine defects.
- Declined a proposed re-chunking change (raising a short-turn merge
  threshold) after a requested simulation showed the benefit was cosmetic
  (no observed retrieval failure traced to it) and the cost was substantial
  (re-embedding, re-ingesting, re-running the full evaluation, and
  invalidating numbers already finalized in the design document).
- Challenged how much of LangChain the project actually used, after an
  earlier instruction to use it wherever possible. Inspection showed only
  the embedding call went through LangChain. This led to exposing the
  retrieval pipeline as a LangChain `BaseRetriever`, and to documenting why
  two other LangChain components were not adopted: `PGVector` owns its own
  table schema, which conflicts with this project's `chunks` table, and
  LangChain's text splitters are character/token based with no notion of
  speaker turns or timestamps (see `DESIGN_REPORT.md` sections 3.2 and 3.5).

### 1.4 Verification practices used throughout

A consistent pattern was applied to any claim of correctness: rather than
accepting a natural-language summary ("this works," "this is correct"),
verification was requested at the level of actual evidence: the exact code
line making a call, the actual database query result, the actual transcript
text at a disputed timestamp, or an independent second calculation method.
Examples: requesting the exact ingestion code line calling the embedder to
confirm which text field was passed to it; requesting a printout of the
shortest chunks by duration to confirm short chunks contained real words
rather than leaked backchannel noise; requesting programmatic (not just
hand-calculated) verification of timestamp offset arithmetic before writing
corrections to the golden query file; and running the LangChain retriever
wrapper live to confirm it returns real `Document` objects and the same
top result as calling `search()` directly.

### 1.5 Tools and skills used

- **`project-conventions` skill** (`.claude/skills/`): a custom Claude
  Code skill encoding this repo's own conventions (code style, doc tone,
  package boundaries, the doc set), so later work stays consistent
  without re-explaining the rules each time.
- **Claude Code subagents** (built-in `Agent` tool): parallelized
  implementation across independent modules by dispatching a separate
  subagent per module (e.g. chunking, embedding, retrieval) against an
  explicit spec, rather than building each one sequentially. Results were
  reviewed and integrated afterward.
- **[Graphify](https://github.com/Graphify-Labs/graphify)**: a third-party
  `/graphify` skill for Claude Code that parses a codebase (source, docs,
  SQL schemas, configs) into a local, queryable knowledge graph via
  deterministic AST parsing, without a vector store. Used once against this
  project's codebase to build a knowledge-graph representation for
  codebase exploration/onboarding purposes.
- **Built-in agentic tools**: file read/write/edit, directory
  listing/search (glob/grep), and shell command execution (for running
  Python scripts, `psql`/`psycopg` queries, `pytest`, `alembic` migrations,
  `ffmpeg`/`yt-dlp` for audio extraction, and package installation).
- **Web search**: used at several points during development to identify and
  verify source material for the audio dataset (podcast episode titles,
  guest/host names, source URLs) and to confirm current documentation of
  external APIs (e.g. verifying a transcription model's actual
  capabilities, such as which timestamp granularities it supports, and the
  correct request/response schema for the Deepgram SDK version installed,
  rather than relying on assumptions or possibly-outdated training
  knowledge).
- **PDF reading**: used once, to read the hackathon's own problem-statement
  PDF directly from disk.
- **A clarifying-question tool**: used repeatedly throughout the build to
  ask the human collaborator to make an explicit choice at decision points
  (e.g. which embedding backend to use, how to handle a labeling
  discrepancy, whether to apply a simulated chunking change) rather than
  silently choosing a default.

---

## 2. AI/ML components inside the shipped system

Distinct from the above (which describes how the code was built), the
running application itself uses the following AI/ML components as part of
its actual functionality:

### 2.1 Transcription and diarization

The audio-to-text and speaker-labeling step uses a hosted transcription and
diarization API (Deepgram), chosen for a combination of setup simplicity, no
local GPU being available in the development environment, and native
single-call speaker diarization support. The task's rules explicitly permit
hosted transcription. A local GPU-based alternative (Whisper + pyannote) was
evaluated and not used in the final build for the reasons above.

### 2.2 Embeddings

Semantic search uses OpenAI's `text-embedding-3-small` (1536-dimensional)
as the primary embedder, called at chunk-ingestion time and at query time.
A fully local embedder (`sentence-transformers`, `bge-small-en-v1.5`,
384-dimensional) is implemented behind the same interface, stored in its
own column, and evaluated separately (`eval/report_local.md`,
`RESULTS.md`).

### 2.3 What is not AI

Indexing (Postgres + pgvector HNSW, `pg_trgm`), keyword search
(`websearch_to_tsquery`/`ts_rank_cd`), and result fusion (Reciprocal Rank
Fusion) are conventional, non-learned algorithms. No model is involved in
ranking or combining results once embeddings and transcripts exist. This
distinction matters for understanding what actually drives result quality:
the two AI components (transcription accuracy, embedding quality) determine
what content is available to search over and how well paraphrased queries
match; everything downstream of that is deterministic.

---

## 3. Summary

| Stage | AI/tool used | Human-directed decision |
|---|---|---|
| Design & architecture | Claude (chat) | All major design choices specified before implementation |
| Implementation | Claude Code (built-in file/bash tools) | Executed against explicit specifications; output verified, not assumed |
| Module implementation | Claude Code subagents | Independent modules built in parallel by separate subagents, each against an explicit spec |
| Codebase knowledge graph | Graphify skill | Used once for codebase exploration/onboarding |
| Transcription/diarization | Deepgram hosted API | Provider chosen after an initial choice required account access that wasn't available |
| Embeddings | OpenAI `text-embedding-3-small` (primary); `bge-small-en-v1.5` (local) | Primary chosen for paraphrase quality; local embedder implemented and evaluated side by side |
| Retrieval fusion | Reciprocal Rank Fusion (non-learned) | Chosen over weighted-score blending for scale-incompatibility reasons |
| Ground-truth construction & audit | Claude Code, human-directed methodology | Audit methodology (unconditional random sampling) specifically designed to detect and disclose the AI collaborator's own error patterns |

Claude Code subagents were used to parallelize module implementation, and
Graphify (codebase knowledge-graph skill) was used once for codebase
exploration. See Section 1.5 for the full tool list.
