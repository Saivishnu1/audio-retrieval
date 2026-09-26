"""Hybrid retrieval over ingested chunks: keyword + semantic, fused with RRF.

LangGraph pipeline: check_embedder -> keyword_leg -> semantic_leg -> fuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TypedDict

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from core.embedder import Embedder
from core.models import Chunk, File, Speaker

RRF_K = 60
DEFAULT_CANDIDATE_K = 50
DEFAULT_TOP_K = 10
FUZZY_MIN_SIMILARITY = 0.3


class EmbedderMismatchError(RuntimeError):
    pass


@dataclass
class SearchResult:
    chunk_id: int
    file: str
    start_sec: float
    end_sec: float
    speaker: str
    text: str
    legs: list[str] = field(default_factory=list)
    leg_ranks: dict[str, int] = field(default_factory=dict)
    rrf_score: float = 0.0


def check_embedder_compat(
    stored_models: set[str], stored_dims: set[int], embedder: Embedder
) -> None:
    if not stored_models:
        raise EmbedderMismatchError(
            "No embedded chunks found for this target. Run ingestion before "
            "semantic search (use --embedder local to populate embedding_local)."
        )
    if stored_models != {embedder.model_name}:
        raise EmbedderMismatchError(
            f"Query embedder is '{embedder.model_name}', but stored chunks were "
            f"embedded with {sorted(stored_models)}. Vectors from different models "
            f"aren't comparable -- use the same embedder, or re-ingest with this one."
        )
    if stored_dims != {embedder.dimension}:
        raise EmbedderMismatchError(
            f"Query embedder produces {embedder.dimension}-dim vectors, but stored "
            f"vectors have dimension(s) {sorted(stored_dims)}."
        )


def get_stored_embedding_info(
    session: Session, target: str = "primary"
) -> tuple[set[str], set[int]]:
    embedding_col, model_col = (
        ("embedding", "embedding_model") if target == "primary"
        else ("embedding_local", "embedding_local_model")
    )
    rows = session.execute(
        text(
            f"SELECT DISTINCT {model_col}, vector_dims({embedding_col}) "
            f"FROM chunks WHERE {embedding_col} IS NOT NULL"
        )
    ).all()
    models = {r[0] for r in rows if r[0] is not None}
    dims = {r[1] for r in rows if r[1] is not None}
    if any(r[0] is None for r in rows):
        models.add("<unstamped>")
    return models, dims


def keyword_search(
    session: Session, query: str, k: int = DEFAULT_CANDIDATE_K
) -> tuple[list[int], str]:
    rows = session.execute(
        text(
            """
            SELECT id
            FROM chunks, websearch_to_tsquery('english', :q) AS tsq
            WHERE tsv @@ tsq
            ORDER BY ts_rank_cd(tsv, tsq) DESC, id
            LIMIT :k
            """
        ),
        {"q": query, "k": k},
    ).all()
    if rows:
        return [r[0] for r in rows], "keyword"

    rows = session.execute(
        text(
            """
            SELECT id
            FROM chunks
            WHERE word_similarity(:q, text) >= :min_sim
            ORDER BY word_similarity(:q, text) DESC, id
            LIMIT :k
            """
        ),
        {"q": query, "k": k, "min_sim": FUZZY_MIN_SIMILARITY},
    ).all()
    return [r[0] for r in rows], "fuzzy"


def semantic_search(
    session: Session,
    embedder: Embedder,
    query: str,
    k: int = DEFAULT_CANDIDATE_K,
    target: str = "primary",
) -> list[int]:
    query_vec = embedder.embed_query(query)
    column = Chunk.embedding if target == "primary" else Chunk.embedding_local
    stmt = (
        select(Chunk.id)
        .where(column.is_not(None))
        .order_by(column.cosine_distance(query_vec), Chunk.id)
        .limit(k)
    )
    return list(session.execute(stmt).scalars())


def rrf_fuse(
    ranked_lists: dict[str, list[int]], k: int = RRF_K
) -> list[tuple[int, float, dict[str, int]]]:
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for leg, ids in ranked_lists.items():
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(chunk_id, {})[leg] = rank
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [(chunk_id, score, ranks[chunk_id]) for chunk_id, score in ordered]


def load_results(
    session: Session, fused: list[tuple[int, float, dict[str, int]]]
) -> list[SearchResult]:
    if not fused:
        return []
    ids = [chunk_id for chunk_id, _, _ in fused]
    rows = session.execute(
        select(Chunk, File.file_path, Speaker.label, Speaker.real_name)
        .join(File, Chunk.file_id == File.id)
        .join(Speaker, Chunk.speaker_id == Speaker.id)
        .where(Chunk.id.in_(ids))
    ).all()
    by_id = {row[0].id: row for row in rows}

    results = []
    for chunk_id, score, leg_ranks in fused:
        chunk, file_path, label, real_name = by_id[chunk_id]
        results.append(
            SearchResult(
                chunk_id=chunk_id,
                file=file_path,
                start_sec=chunk.start_sec,
                end_sec=chunk.end_sec,
                speaker=real_name or label,
                text=chunk.text,
                legs=sorted(leg_ranks),
                leg_ranks=leg_ranks,
                rrf_score=score,
            )
        )
    return results


class GraphState(TypedDict, total=False):
    query: str
    embedder: Embedder
    top_k: int
    candidate_k: int
    enabled_legs: set[str]
    target: str
    ranked_lists: dict[str, list[int]]
    results: list[SearchResult]
    error: Optional[str]


def _session_node(fn):
    def node(state: GraphState) -> GraphState:
        if state.get("error"):
            return state
        from core.db import get_session

        session = get_session()
        try:
            return fn(state, session)
        finally:
            session.close()

    node.__name__ = fn.__name__
    return node


@_session_node
def check_embedder_node(state: GraphState, session: Session) -> GraphState:
    if "semantic" not in state.get("enabled_legs", {"keyword", "semantic"}):
        return state
    models, dims = get_stored_embedding_info(session, state.get("target", "primary"))
    try:
        check_embedder_compat(models, dims, state["embedder"])
    except EmbedderMismatchError as exc:
        return {**state, "error": str(exc)}
    return state


@_session_node
def keyword_leg_node(state: GraphState, session: Session) -> GraphState:
    if "keyword" not in state.get("enabled_legs", {"keyword", "semantic"}):
        return state
    ids, leg = keyword_search(
        session, state["query"], state.get("candidate_k", DEFAULT_CANDIDATE_K)
    )
    ranked = dict(state.get("ranked_lists", {}))
    if ids:
        ranked[leg] = ids
    return {**state, "ranked_lists": ranked}


@_session_node
def semantic_leg_node(state: GraphState, session: Session) -> GraphState:
    if "semantic" not in state.get("enabled_legs", {"keyword", "semantic"}):
        return state
    ids = semantic_search(
        session,
        state["embedder"],
        state["query"],
        state.get("candidate_k", DEFAULT_CANDIDATE_K),
        state.get("target", "primary"),
    )
    ranked = dict(state.get("ranked_lists", {}))
    ranked["semantic"] = ids
    return {**state, "ranked_lists": ranked}


@_session_node
def fuse_node(state: GraphState, session: Session) -> GraphState:
    fused = rrf_fuse(state.get("ranked_lists", {}))[: state.get("top_k", DEFAULT_TOP_K)]
    return {**state, "results": load_results(session, fused)}


def build_graph():
    from langgraph.graph import END, StateGraph

    graph = StateGraph(GraphState)
    graph.add_node("check_embedder", check_embedder_node)
    graph.add_node("keyword_leg", keyword_leg_node)
    graph.add_node("semantic_leg", semantic_leg_node)
    graph.add_node("fuse", fuse_node)

    graph.set_entry_point("check_embedder")
    graph.add_edge("check_embedder", "keyword_leg")
    graph.add_edge("keyword_leg", "semantic_leg")
    graph.add_edge("semantic_leg", "fuse")
    graph.add_edge("fuse", END)

    return graph.compile()


def search(
    query: str,
    embedder: Embedder,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    enabled_legs: Optional[set[str]] = None,
    target: str = "primary",
) -> list[SearchResult]:
    state = build_graph().invoke(
        {
            "query": query,
            "embedder": embedder,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "enabled_legs": enabled_legs or {"keyword", "semantic"},
            "target": target,
        }
    )
    if state.get("error"):
        if "embedded with" in state["error"] or "dim vectors" in state["error"] \
                or "No embedded chunks" in state["error"]:
            raise EmbedderMismatchError(state["error"])
        raise RuntimeError(state["error"])
    return state["results"]


def _result_to_document(result: SearchResult):
    from langchain_core.documents import Document

    return Document(
        page_content=result.text,
        metadata={
            "chunk_id": result.chunk_id,
            "file": result.file,
            "start_sec": result.start_sec,
            "end_sec": result.end_sec,
            "speaker": result.speaker,
            "legs": result.legs,
            "leg_ranks": result.leg_ranks,
            "rrf_score": result.rrf_score,
        },
    )


class HybridRetriever:
    """Exposes `search()` as a `langchain_core.retrievers.BaseRetriever`."""

    def __init__(
        self,
        embedder: Embedder,
        top_k: int = DEFAULT_TOP_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        enabled_legs: Optional[set[str]] = None,
    ):
        self.embedder = embedder
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.enabled_legs = enabled_legs

    def as_langchain_retriever(self):
        from langchain_core.retrievers import BaseRetriever

        outer = self

        class _Retriever(BaseRetriever):
            def _get_relevant_documents(self, query: str, *, run_manager=None):
                results = search(
                    query,
                    outer.embedder,
                    top_k=outer.top_k,
                    candidate_k=outer.candidate_k,
                    enabled_legs=outer.enabled_legs,
                )
                return [_result_to_document(r) for r in results]

        return _Retriever()


def main() -> None:
    import argparse

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Hybrid search over ingested chunks.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedder", choices=["openai", "local"], default="openai")
    args = parser.parse_args()

    if args.embedder == "openai":
        from core.embedder import OpenAIEmbedder

        embedder: Embedder = OpenAIEmbedder()
    else:
        from core.embedder import LocalEmbedder

        embedder = LocalEmbedder()

    try:
        results = search(args.query, embedder, top_k=args.top_k)
    except EmbedderMismatchError as exc:
        raise SystemExit(f"Embedder mismatch: {exc}")

    for i, r in enumerate(results, start=1):
        print(
            f"{i}. [{r.rrf_score:.4f}] {r.file} {r.start_sec:.1f}-{r.end_sec:.1f}s "
            f"{r.speaker} legs={r.leg_ranks}"
        )
        print(f"   {r.text[:160]}")


if __name__ == "__main__":
    main()
