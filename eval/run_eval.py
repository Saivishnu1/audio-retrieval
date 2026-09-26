"""Run the golden query set against the retrieval system and report
recall@1/5/10 and MRR, broken down per bucket, across 3 ablation
configurations: keyword-only, semantic-only, and hybrid (keyword +
semantic, fused with RRF).

Orchestrated with LangGraph, matching the rest of this repo's pipelines:

    load  ->  run_configs  ->  score  ->  write_report

    - load: reads eval/golden_queries.yaml.
    - run_configs: for each of the 3 configs, runs every query through
      retrieval.retrieve.search() with that config's enabled_legs.
    - score: for each (config, query) pair, checks whether each ranked
      result's (file, start_sec, end_sec) *overlaps* (not exact-matches)
      any of the query's labeled `relevant` spans -- consistent with the
      project-wide decision that ground truth is chunking-agnostic
      (labels come from transcripts/*.txt turns, not chunk boundaries;
      see golden_queries.yaml's header comment). Computes recall@1/5/10
      and MRR per bucket and overall, per config.
    - write_report: writes eval/report.md with the full ablation table.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --queries eval/golden_queries.yaml --report eval/report.md
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, TypedDict

import yaml

# Make the package importable when run as a script (python eval/run_eval.py)
# without needing PYTHONPATH=src set.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.embedder import Embedder  # noqa: E402
from retrieval.retrieve import SearchResult, search  # noqa: E402

TOP_K_VALUES = (1, 5, 10)
MAX_TOP_K = max(TOP_K_VALUES)
BUCKETS = ("exact_keyword", "paraphrase", "named_entity", "multi_span")

CONFIGS: dict[str, Optional[set[str]]] = {
    "keyword_only": {"keyword"},
    "semantic_only": {"semantic"},
    "hybrid_rrf": {"keyword", "semantic"},
}


# --------------------------------------------------------------------------
# Ground-truth overlap check
# --------------------------------------------------------------------------
def overlaps(
    result: SearchResult, relevant_spans: list[dict], slack_sec: float = 0.0
) -> bool:
    """True if `result`'s (file, start_sec, end_sec) overlaps ANY labeled
    relevant span for this query (same file, and the time ranges
    intersect). `slack_sec` widens each labeled span symmetrically, in
    case a chunk boundary sits just outside a tightly-drawn label.
    """
    for span in relevant_spans:
        if span["file"] != result.file:
            continue
        lo = span["start_sec"] - slack_sec
        hi = span["end_sec"] + slack_sec
        if result.start_sec < hi and result.end_sec > lo:
            return True
    return False


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def reciprocal_rank(results: list[SearchResult], relevant_spans: list[dict]) -> float:
    for rank, r in enumerate(results, start=1):
        if overlaps(r, relevant_spans):
            return 1.0 / rank
    return 0.0


def recall_at_k(results: list[SearchResult], relevant_spans: list[dict], k: int) -> float:
    return 1.0 if any(overlaps(r, relevant_spans) for r in results[:k]) else 0.0


# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------
class QueryOutcome(TypedDict):
    id: str
    bucket: str
    recall: dict[int, float]  # k -> 0/1
    rr: float
    error: Optional[str]


class GraphState(TypedDict, total=False):
    queries_path: str
    embedder: Embedder
    # "primary" (OpenAI, default) or "local" (bge-small ablation report)
    target: str
    queries: list[dict]
    # config name -> query id -> QueryOutcome
    outcomes: dict[str, dict[str, QueryOutcome]]
    # config name -> bucket (or "overall") -> {metric: value}
    summary: dict[str, dict[str, dict[str, float]]]
    report_path: str
    error: Optional[str]


# --------------------------------------------------------------------------
# Node: load
# --------------------------------------------------------------------------
def load_node(state: GraphState) -> GraphState:
    queries_path = Path(state["queries_path"])
    if not queries_path.exists():
        return {**state, "error": f"queries file not found: {queries_path}"}
    with open(queries_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    queries = (data or {}).get("queries", [])
    if not queries:
        return {**state, "error": f"no queries found in {queries_path}"}
    print(f"Loaded {len(queries)} queries from {queries_path}")
    return {**state, "queries": queries}


# --------------------------------------------------------------------------
# Node: run_configs
# --------------------------------------------------------------------------
def run_configs_node(state: GraphState) -> GraphState:
    if state.get("error"):
        return state

    embedder = state["embedder"]
    outcomes: dict[str, dict[str, QueryOutcome]] = {name: {} for name in CONFIGS}

    for config_name, enabled_legs in CONFIGS.items():
        print(f"\nRunning config: {config_name} (legs={enabled_legs})")
        for q in state["queries"]:
            qid = q["id"]
            query_text = q["query"]
            bucket = q["bucket"]
            relevant_spans = q["relevant"]

            try:
                results = search(
                    query_text,
                    embedder,
                    top_k=MAX_TOP_K,
                    enabled_legs=enabled_legs,
                    target=state.get("target", "primary"),
                )
                recall = {k: recall_at_k(results, relevant_spans, k) for k in TOP_K_VALUES}
                rr = reciprocal_rank(results, relevant_spans)
                outcomes[config_name][qid] = {
                    "id": qid,
                    "bucket": bucket,
                    "recall": recall,
                    "rr": rr,
                    "error": None,
                }
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR on {qid} ({config_name}): {exc}")
                outcomes[config_name][qid] = {
                    "id": qid,
                    "bucket": bucket,
                    "recall": {k: 0.0 for k in TOP_K_VALUES},
                    "rr": 0.0,
                    "error": str(exc),
                }

    return {**state, "outcomes": outcomes}


# --------------------------------------------------------------------------
# Node: score
# --------------------------------------------------------------------------
def _aggregate(outcomes: list[QueryOutcome]) -> dict[str, float]:
    n = len(outcomes)
    if n == 0:
        return {f"recall@{k}": 0.0 for k in TOP_K_VALUES} | {"mrr": 0.0, "n": 0}
    agg = {f"recall@{k}": sum(o["recall"][k] for o in outcomes) / n for k in TOP_K_VALUES}
    agg["mrr"] = sum(o["rr"] for o in outcomes) / n
    agg["n"] = n
    return agg


def score_node(state: GraphState) -> GraphState:
    if state.get("error"):
        return state

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for config_name, per_query in state["outcomes"].items():
        all_outcomes = list(per_query.values())
        by_bucket: dict[str, list[QueryOutcome]] = {b: [] for b in BUCKETS}
        for o in all_outcomes:
            by_bucket.setdefault(o["bucket"], []).append(o)

        config_summary = {"overall": _aggregate(all_outcomes)}
        for bucket in BUCKETS:
            config_summary[bucket] = _aggregate(by_bucket.get(bucket, []))
        summary[config_name] = config_summary

    return {**state, "summary": summary}


# --------------------------------------------------------------------------
# Node: write_report
# --------------------------------------------------------------------------
def _format_table(summary: dict[str, dict[str, dict[str, float]]]) -> str:
    rows = ["overall", *BUCKETS]
    header = (
        "| Config | Bucket | n | recall@1 | recall@5 | recall@10 | MRR |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    lines = [header]
    for config_name, config_summary in summary.items():
        for row in rows:
            m = config_summary[row]
            label = "**overall**" if row == "overall" else row
            lines.append(
                f"| {config_name} | {label} | {int(m['n'])} | "
                f"{m['recall@1']:.2f} | {m['recall@5']:.2f} | {m['recall@10']:.2f} | "
                f"{m['mrr']:.3f} |\n"
            )
    return "".join(lines)


def write_report_node(state: GraphState) -> GraphState:
    if state.get("error"):
        print(f"FATAL: {state['error']}")
        return state

    report_path = Path(state["report_path"])
    table = _format_table(state["summary"])
    target = state.get("target", "primary")
    embedder_desc = (
        "OpenAI text-embedding-3-small (1536-dim), primary/production embedder"
        if target == "primary"
        else "sentence-transformers bge-small-en-v1.5 (384-dim), local-only "
        "fallback embedder, run against the separate embedding_local column "
        "so results here don't affect or depend on the primary report"
    )

    lines = [
        "# Retrieval Evaluation Report",
        " (local embedder)\n\n" if target == "local" else "\n\n",
        f"Golden query set: `{state['queries_path']}` "
        f"({len(state['queries'])} queries, buckets: {', '.join(BUCKETS)})\n\n",
        f"Embedder: {embedder_desc}.\n\n",
        "Hit criterion: a result counts as relevant if its "
        "`(file, start_sec, end_sec)` **overlaps** any of the query's labeled "
        "spans (ground truth from `transcripts/*.txt` turn boundaries, "
        "independent of chunk boundaries) -- not an exact chunk-ID match.\n\n",
        "## Ablation: keyword-only vs semantic-only vs hybrid (RRF)\n\n",
        table,
        "\n",
        "## Configs\n\n",
        "- **keyword_only**: `websearch_to_tsquery` + `ts_rank_cd` on `tsv` "
        "(pg_trgm fuzzy fallback if no full-text hits), semantic leg disabled.\n",
        f"- **semantic_only**: cosine similarity on "
        f"`{'embedding' if target == 'primary' else 'embedding_local'}` "
        f"({embedder_desc.split(',')[0]}), keyword leg disabled.\n",
        "- **hybrid_rrf**: both legs, fused with Reciprocal Rank Fusion (k=60).\n",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"\nWrote report to {report_path}")
    print("\n" + table)

    return state


# --------------------------------------------------------------------------
# Build the graph
# --------------------------------------------------------------------------
def build_graph():
    from langgraph.graph import END, StateGraph

    graph = StateGraph(GraphState)
    graph.add_node("load", load_node)
    graph.add_node("run_configs", run_configs_node)
    graph.add_node("score", score_node)
    graph.add_node("write_report", write_report_node)

    graph.set_entry_point("load")
    graph.add_edge("load", "run_configs")
    graph.add_edge("run_configs", "score")
    graph.add_edge("score", "write_report")
    graph.add_edge("write_report", END)

    return graph.compile()


def main() -> int:
    import argparse

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default="eval/golden_queries.yaml")
    parser.add_argument("--report", default=None)
    parser.add_argument("--embedder", choices=["openai", "local"], default="openai")
    args = parser.parse_args()

    if args.embedder == "openai":
        from core.embedder import OpenAIEmbedder

        embedder: Embedder = OpenAIEmbedder()
        target = "primary"
        default_report = "eval/report.md"
    else:
        from core.embedder import LocalEmbedder

        embedder = LocalEmbedder()
        target = "local"
        default_report = "eval/report_local.md"

    report_path = args.report or default_report

    app = build_graph()
    final_state = app.invoke(
        {
            "queries_path": args.queries,
            "embedder": embedder,
            "target": target,
            "report_path": report_path,
        }
    )

    return 1 if final_state.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
