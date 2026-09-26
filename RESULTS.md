# Embedder Comparison: OpenAI vs. Local

Two full ablation evaluations were run against the same 41-query golden
set and the same 259 ingested chunks, differing only in which model
produced the semantic-leg embeddings:

- **Primary**: OpenAI `text-embedding-3-small` (1536-dim), hosted. Full
  report: `eval/report.md`.
- **Local**: `sentence-transformers`/`bge-small-en-v1.5` (384-dim), fully
  offline. Full report: `eval/report_local.md`.

Both were ingested into the same `chunks` table using separate columns
(`embedding`/`embedding_model` for primary, `embedding_local`/
`embedding_local_model` for local), so neither run disturbed the other,
and `keyword_only` results are identical between the two reports (keyword
search never touches embeddings), confirming the comparison isolates only
the embedder.

## Results

| Config | Bucket | n | OpenAI recall@1 | Local recall@1 | OpenAI MRR | Local MRR |
|---|---|---|---|---|---|---|
| keyword_only | overall | 41 | 0.59 | 0.59 | 0.618 | 0.618 |
| keyword_only | exact_keyword | 16 | 0.94 | 0.94 | 0.958 | 0.958 |
| keyword_only | paraphrase | 12 | 0.00 | 0.00 | 0.000 | 0.000 |
| keyword_only | named_entity | 11 | 0.73 | 0.73 | 0.818 | 0.818 |
| keyword_only | multi_span | 2 | 0.50 | 0.50 | 0.500 | 0.500 |
| semantic_only | overall | 41 | 0.66 | 0.51 | 0.789 | 0.672 |
| semantic_only | exact_keyword | 16 | 0.69 | 0.69 | 0.833 | 0.825 |
| semantic_only | paraphrase | 12 | 0.67 | 0.25 | 0.769 | 0.419 |
| semantic_only | named_entity | 11 | 0.55 | 0.55 | 0.708 | 0.712 |
| semantic_only | multi_span | 2 | 1.00 | 0.50 | 1.000 | 0.750 |
| **hybrid_rrf** | **overall** | 41 | **0.80** | 0.71 | **0.878** | 0.793 |
| hybrid_rrf | exact_keyword | 16 | 0.94 | **1.00** | 0.969 | **1.000** |
| hybrid_rrf | paraphrase | 12 | **0.67** | 0.25 | **0.769** | 0.419 |
| hybrid_rrf | named_entity | 11 | 0.73 | 0.73 | 0.841 | 0.864 |
| hybrid_rrf | multi_span | 2 | **1.00** | **1.00** | **1.000** | **1.000** |

## What this shows

**OpenAI's embedder wins overall, and the gap is concentrated in exactly
the bucket semantic search exists to help with.** Hybrid overall recall@1
drops from 0.80 to 0.71 and MRR from 0.878 to 0.793 when swapping to the
local model, entirely because `paraphrase` recall@1 falls from 0.67 to
0.25, a direct consequence of semantic_only's paraphrase recall@1
dropping from 0.67 to 0.25 in the same swap. `bge-small-en-v1.5` is a much
smaller, general-purpose 33M-parameter model versus OpenAI's larger,
purpose-tuned embedding model; a real capability gap on this bucket is the
expected outcome, not a bug in either configuration.

**Keyword-driven buckets are close to identical or slightly favor local in
this run.** `exact_keyword` and `multi_span` are the same or marginally
better under the local embedder in `hybrid_rrf` (1.00 vs. 0.94, 1.00 vs.
1.00), which makes sense: these buckets are largely carried by the
keyword leg and RRF fusion rather than semantic-leg quality, so a weaker
embedder has less room to hurt them, and can occasionally edge ahead due
to a bucket this small (n=16 and n=2) being noisy at the single-query
level. `named_entity` is essentially a wash (0.73 either way).

**Hybrid still meaningfully outperforms either single leg under the local
embedder too**, not only under OpenAI's: local hybrid_rrf (0.71 recall@1,
0.793 MRR) beats local semantic_only (0.51, 0.672) and roughly matches or
exceeds local keyword_only (0.59, 0.618) while adding real paraphrase
capability keyword_only has none of. The hybrid architecture's value
doesn't depend on which embedder is plugged in behind it.

## Conclusion

For a fully offline deployment, `bge-small-en-v1.5` is a workable but
measurably weaker option, costing roughly 9 points of overall hybrid
recall@1 and most of the paraphrase bucket's semantic advantage. The
swappable `Embedder` interface (`src/core/embedder.py`) means
adopting a stronger local model later (e.g. a larger `bge`/`e5` variant)
requires no other code change, just re-running ingestion and this
comparison again.
