# Retrieval Evaluation Report

Golden query set: `eval/golden_queries.yaml` (41 queries, buckets: exact_keyword, paraphrase, named_entity, multi_span)

Embedder: OpenAI text-embedding-3-small (1536-dim), primary/production embedder.

Hit criterion: a result counts as relevant if its `(file, start_sec, end_sec)` **overlaps** any of the query's labeled spans (ground truth from `transcripts/*.txt` turn boundaries, independent of chunk boundaries) -- not an exact chunk-ID match.

## Ablation: keyword-only vs semantic-only vs hybrid (RRF)

| Config | Bucket | n | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|---|
| keyword_only | **overall** | 41 | 0.59 | 0.66 | 0.66 | 0.618 |
| keyword_only | exact_keyword | 16 | 0.94 | 1.00 | 1.00 | 0.958 |
| keyword_only | paraphrase | 12 | 0.00 | 0.00 | 0.00 | 0.000 |
| keyword_only | named_entity | 11 | 0.73 | 0.91 | 0.91 | 0.818 |
| keyword_only | multi_span | 2 | 0.50 | 0.50 | 0.50 | 0.500 |
| semantic_only | **overall** | 41 | 0.66 | 1.00 | 1.00 | 0.789 |
| semantic_only | exact_keyword | 16 | 0.69 | 1.00 | 1.00 | 0.833 |
| semantic_only | paraphrase | 12 | 0.67 | 1.00 | 1.00 | 0.769 |
| semantic_only | named_entity | 11 | 0.55 | 1.00 | 1.00 | 0.708 |
| semantic_only | multi_span | 2 | 1.00 | 1.00 | 1.00 | 1.000 |
| hybrid_rrf | **overall** | 41 | 0.80 | 1.00 | 1.00 | 0.878 |
| hybrid_rrf | exact_keyword | 16 | 0.94 | 1.00 | 1.00 | 0.969 |
| hybrid_rrf | paraphrase | 12 | 0.67 | 1.00 | 1.00 | 0.769 |
| hybrid_rrf | named_entity | 11 | 0.73 | 1.00 | 1.00 | 0.841 |
| hybrid_rrf | multi_span | 2 | 1.00 | 1.00 | 1.00 | 1.000 |

## Configs

- **keyword_only**: `websearch_to_tsquery` + `ts_rank_cd` on `tsv` (pg_trgm fuzzy fallback if no full-text hits), semantic leg disabled.
- **semantic_only**: cosine similarity on `embedding` (OpenAI text-embedding-3-small (1536-dim)), keyword leg disabled.
- **hybrid_rrf**: both legs, fused with Reciprocal Rank Fusion (k=60).
