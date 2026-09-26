# Audio Retrieval

Hybrid (keyword + semantic) search over diarized, two-speaker audio
conversations, built for the G2 AI Engineering Hackathon's Problem Statement
1, "Effective retrieval from audio transcripts". A search returns the
containing file, timestamp range, and speaker for each hit, and works both
for exact words spoken and for semantically similar phrasing.

Headline result (hybrid keyword + semantic with Reciprocal Rank Fusion,
41-query golden set): **recall@1 0.80, recall@5 1.00, MRR 0.878**.

## Documents

| File | Read it for |
|---|---|
| `GUIDELINES.md` | Setup (Postgres, API keys), every command to run the system, and where the eval output lands. **Start here to run it.** |
| `ARCHITECTURE.md` | How the ingestion and query pipelines work, the data model, and a link to the diagrams. |
| `DESIGN_REPORT.md` | Why it's built this way, the evaluation method and results, the ground-truth audit, and limitations. |
| `RESULTS.md` | OpenAI vs. local (`bge-small`) embedder comparison. |
| `AI_DISCLOSURE.md` | How AI tools were used to build the project, and which AI/ML components run inside it. |
| `eval/report.md` | Generated ablation report for the primary embedder. |
| `eval/report_local.md` | Generated ablation report for the local embedder. |

## Project layout

```
Audio Retrieval/
├── manifest.yaml           # golden dataset: clips, offsets, speakers, checksums
├── transcripts/            # diarized transcripts, one .txt per clip
├── alembic/                # Postgres schema migrations
├── src/
│   ├── youtube_audio/      # extract audio from a YouTube URL
│   ├── transcription/      # Deepgram transcription + diarization
│   ├── core/
│   │   ├── embedder.py     # swappable OpenAI / local embedding backends
│   │   ├── models.py       # SQLAlchemy schema
│   │   └── db.py           # Postgres connection
│   ├── ingestion/
│   │   ├── chunking.py     # speaker-turn chunking rules
│   │   └── ingest.py       # LangGraph ingestion pipeline
│   └── retrieval/
│       └── retrieve.py     # LangGraph hybrid search pipeline + LangChain retriever
├── eval/
│   ├── golden_queries.yaml # 41 labeled queries in 4 buckets
│   ├── validate_queries.py # sanity-checks the labels
│   ├── run_eval.py         # ablation: keyword-only / semantic-only / hybrid
│   └── report*.md          # generated reports
└── tests/
```

## Quick start

Requires a running PostgreSQL instance (with `pgvector` and `pg_trgm`
available to install) reachable with the credentials you put in `.env`.
See `GUIDELINES.md` section 1 for setup details.

```bash
uv sync --extra dev
cp .env.example .env        # then fill in API keys and Postgres credentials
alembic upgrade head
python -m ingestion.ingest --embedder openai
python -m retrieval.retrieve "your query here" --top-k 5
```

## Switching embedders (OpenAI vs. local)

Two embedding backends are supported behind the same interface, each writing
to its own column so both can coexist: `--embedder openai` (primary,
`text-embedding-3-small`, needs `OPENAI_API_KEY`) and `--embedder local`
(offline, `bge-small-en-v1.5`, no API key). Ingest and search with the same
flag:

```bash
python -m ingestion.ingest --manifest manifest.yaml --embedder local
python -m retrieval.retrieve "your query here" --embedder local
```

Both were evaluated on the same 41-query golden set. OpenAI wins overall,
with the gap concentrated entirely in paraphrased queries:

| Embedder | recall@1 | recall@5/10 | MRR |
|---|---|---|---|
| OpenAI `text-embedding-3-small` (primary) | **0.80** | 1.00 | **0.878** |
| Local `bge-small-en-v1.5` (offline) | 0.71 | 1.00 | 0.793 |

See `RESULTS.md` for the full per-bucket breakdown and analysis, and
`GUIDELINES.md` section 4-5 for the full walkthrough.

Full details, including prerequisites and troubleshooting, are in
`GUIDELINES.md`.
