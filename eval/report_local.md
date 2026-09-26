# Retrieval Evaluation Report (local embedder)

Golden query set: `eval/golden_queries.yaml` (41 queries, buckets: exact_keyword, paraphrase, named_entity, multi_span)

Embedder: sentence-transformers bge-small-en-v1.5 (384-dim), local-only fallback embedder, run against the separate embedding_local column so results here don't affect or depend on the primary report.

Hit criterion: a result counts as relevant if its `(file, start_sec, end_sec)` **overlaps** any of the query's labeled spans (ground truth from `transcripts/*.txt` turn boundaries, independent of chunk boundaries) -- not an exact chunk-ID match.

## Ablation: keyword-only vs semantic-only vs hybrid (RRF)

| Config | Bucket | n | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|---|
| keyword_only | **overall** | 41 | 0.59 | 0.66 | 0.66 | 0.618 |
| keyword_only | exact_keyword | 16 | 0.94 | 1.00 | 1.00 | 0.958 |
| keyword_only | paraphrase | 12 | 0.00 | 0.00 | 0.00 | 0.000 |
| keyword_only | named_entity | 11 | 0.73 | 0.91 | 0.91 | 0.818 |
| keyword_only | multi_span | 2 | 0.50 | 0.50 | 0.50 | 0.500 |
| semantic_only | **overall** | 41 | 0.51 | 0.88 | 0.93 | 0.672 |
| semantic_only | exact_keyword | 16 | 0.69 | 1.00 | 1.00 | 0.825 |
| semantic_only | paraphrase | 12 | 0.25 | 0.67 | 0.75 | 0.419 |
| semantic_only | named_entity | 11 | 0.55 | 0.91 | 1.00 | 0.712 |
| semantic_only | multi_span | 2 | 0.50 | 1.00 | 1.00 | 0.750 |
| hybrid_rrf | **overall** | 41 | 0.71 | 0.90 | 0.93 | 0.793 |
| hybrid_rrf | exact_keyword | 16 | 1.00 | 1.00 | 1.00 | 1.000 |
| hybrid_rrf | paraphrase | 12 | 0.25 | 0.67 | 0.75 | 0.419 |
| hybrid_rrf | named_entity | 11 | 0.73 | 1.00 | 1.00 | 0.864 |
| hybrid_rrf | multi_span | 2 | 1.00 | 1.00 | 1.00 | 1.000 |

## Configs

- **keyword_only**: `websearch_to_tsquery` + `ts_rank_cd` on `tsv` (pg_trgm fuzzy fallback if no full-text hits), semantic leg disabled.
- **semantic_only**: cosine similarity on `embedding_local` (sentence-transformers bge-small-en-v1.5 (384-dim)), keyword leg disabled.
- **hybrid_rrf**: both legs, fused with Reciprocal Rank Fusion (k=60).
