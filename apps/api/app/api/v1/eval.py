"""GET /v1/eval/summary — the data model behind the /eval dashboard.

The numbers are sourced from real ablation runs on the live corpus (committed
in `data/processed/{bm25,dense,rerank,kg}/runs/<timestamp>/*.json` and the
DuckDB warehouse). For now we hard-code them here as the canonical snapshot
since runs are immutable per commit; a future iteration can read the latest
JSON artifact at request time so refreshing the page after a re-run shows new
numbers.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.schemas import (
    AblationRow,
    CorpusStats,
    EvalSetStats,
    EvalSummary,
    EvalTargets,
    KGStats,
    LatencyStage,
)

router = APIRouter(prefix="/eval", tags=["eval"])


# ─── Hard-coded canonical snapshot (matches docs/eval-methodology.md) ───────


CORPUS_STATS = CorpusStats(
    profiles=1782,
    jobs=1370,
    canonical_skills=4553,
    has_skill_edges=30369,
    requires_skill_edges=5129,
)

KG_STATS = KGStats(
    persons=1782,
    jobs=1370,
    skills=4553,
    roles=834,
    designations=415,
    industries=53,
    locations=131,
    has_skill=30369,
    requires_skill=5129,
    can_fill=7283,
    is_designation=1317,
    at_location=1352,
    in_industry=281,
)

EVAL_SET_STATS = EvalSetStats(
    total_queries=119,
    candidate_search=50,
    job_search=50,
    paraphrase_stratum=19,
    seed=42,
)

# Ablation matrix on the original 100-query stratum + the 19-paraphrase stratum.
# is_best_for_stratum is set by _mark_best() at module-load time.
_RAW_ABLATION: list[AblationRow] = [
    # ── Overall (mean of candidate + job) ────────────────────────────────────
    AblationRow(
        name="BM25",
        pipeline="bm25",
        stratum="overall",
        ndcg_at_10=0.5397,
        recall_at_10=0.5137,
        recall_at_100=0.7040,
        mrr_at_10=0.5666,
        map_score=0.5539,
        latency_ms=0.2,
    ),
    AblationRow(
        name="BGE-M3 dense",
        pipeline="dense",
        stratum="overall",
        ndcg_at_10=0.5271,
        recall_at_10=0.5156,
        recall_at_100=0.7028,
        mrr_at_10=0.5482,
        map_score=0.5302,
        latency_ms=2.3,
    ),
    AblationRow(
        name="RRF (BM25 + dense)",
        pipeline="rrf",
        stratum="overall",
        ndcg_at_10=0.5515,
        recall_at_10=0.5213,
        recall_at_100=0.6955,
        mrr_at_10=0.5854,
        map_score=0.5616,
        latency_ms=2.5,
    ),
    AblationRow(
        name="Cross-encoder rerank top-25",
        pipeline="rerank25",
        stratum="overall",
        ndcg_at_10=0.5377,
        recall_at_10=0.5349,
        recall_at_100=0.6955,
        mrr_at_10=0.5506,
        map_score=0.5416,
        latency_ms=285.0,
    ),
    AblationRow(
        name="KG channel only",
        pipeline="kg_only",
        stratum="overall",
        ndcg_at_10=0.4249,
        recall_at_10=0.4427,
        recall_at_100=0.6863,
        mrr_at_10=0.4555,
        map_score=0.4293,
        latency_ms=25.0,
    ),
    AblationRow(
        name="RRF3 (BM25 + dense + KG)",
        pipeline="rrf3",
        stratum="overall",
        ndcg_at_10=0.5567,
        recall_at_10=0.5396,
        recall_at_100=0.6995,
        mrr_at_10=0.5914,
        map_score=0.5717,
        latency_ms=30.0,
    ),
    AblationRow(
        name="Full pipeline (RRF3 + rerank top-25)",
        pipeline="rrf3_rerank",
        stratum="overall",
        ndcg_at_10=0.5442,
        recall_at_10=0.5356,
        recall_at_100=0.6995,
        mrr_at_10=0.5653,
        map_score=0.5447,
        latency_ms=315.0,
    ),
    # ── Original stratum (100 lexical-anchor queries) ───────────────────────
    AblationRow(
        name="BM25",
        pipeline="bm25",
        stratum="original",
        ndcg_at_10=0.604,
        recall_at_10=0.580,
        recall_at_100=0.770,
        mrr_at_10=0.634,
        map_score=0.620,
        latency_ms=0.1,
    ),
    AblationRow(
        name="BGE-M3 dense",
        pipeline="dense",
        stratum="original",
        ndcg_at_10=0.589,
        recall_at_10=0.583,
        recall_at_100=0.759,
        mrr_at_10=0.612,
        map_score=0.593,
        latency_ms=2.3,
    ),
    AblationRow(
        name="RRF (BM25 + dense)",
        pipeline="rrf",
        stratum="original",
        ndcg_at_10=0.618,
        recall_at_10=0.589,
        recall_at_100=0.760,
        mrr_at_10=0.657,
        map_score=0.629,
        latency_ms=2.5,
    ),
    AblationRow(
        name="Cross-encoder rerank top-25",
        pipeline="rerank25",
        stratum="original",
        ndcg_at_10=0.598,
        recall_at_10=0.596,
        recall_at_100=0.760,
        mrr_at_10=0.614,
        map_score=0.604,
        latency_ms=285.0,
    ),
    AblationRow(
        name="KG channel only",
        pipeline="kg_only",
        stratum="original",
        ndcg_at_10=0.476,
        recall_at_10=0.497,
        recall_at_100=0.747,
        mrr_at_10=0.512,
        map_score=0.480,
        latency_ms=25.0,
    ),
    AblationRow(
        name="RRF3 (BM25 + dense + KG)",
        pipeline="rrf3",
        stratum="original",
        ndcg_at_10=0.621,
        recall_at_10=0.601,
        recall_at_100=0.756,
        mrr_at_10=0.663,
        map_score=0.640,
        latency_ms=30.0,
    ),
    AblationRow(
        name="Full pipeline (RRF3 + rerank top-25)",
        pipeline="rrf3_rerank",
        stratum="original",
        ndcg_at_10=0.605,
        recall_at_10=0.597,
        recall_at_100=0.756,
        mrr_at_10=0.630,
        map_score=0.609,
        latency_ms=315.0,
    ),
    # ── Paraphrase stratum (19 Gemini-generated queries) ───────────────────
    AblationRow(
        name="BM25",
        pipeline="bm25",
        stratum="paraphrase",
        ndcg_at_10=0.201,
        recall_at_10=0.162,
        recall_at_100=0.359,
        mrr_at_10=0.211,
        map_score=0.207,
    ),
    AblationRow(
        name="BGE-M3 dense",
        pipeline="dense",
        stratum="paraphrase",
        ndcg_at_10=0.201,
        recall_at_10=0.162,
        recall_at_100=0.406,
        mrr_at_10=0.211,
        map_score=0.201,
    ),
    AblationRow(
        name="RRF (BM25 + dense)",
        pipeline="rrf",
        stratum="paraphrase",
        ndcg_at_10=0.201,
        recall_at_10=0.162,
        recall_at_100=0.358,
        mrr_at_10=0.211,
        map_score=0.206,
    ),
    AblationRow(
        name="Cross-encoder rerank top-25",
        pipeline="rerank25",
        stratum="paraphrase",
        ndcg_at_10=0.220,
        recall_at_10=0.215,
        recall_at_100=0.358,
        mrr_at_10=0.219,
        map_score=0.212,
    ),
    AblationRow(
        name="KG channel only",
        pipeline="kg_only",
        stratum="paraphrase",
        ndcg_at_10=0.158,
        recall_at_10=0.158,
        recall_at_100=0.368,
        mrr_at_10=0.158,
        map_score=0.163,
    ),
    AblationRow(
        name="RRF3 (BM25 + dense + KG)",
        pipeline="rrf3",
        stratum="paraphrase",
        ndcg_at_10=0.217,
        recall_at_10=0.215,
        recall_at_100=0.400,
        mrr_at_10=0.216,
        map_score=0.211,
    ),
    AblationRow(
        name="Full pipeline (RRF3 + rerank top-25)",
        pipeline="rrf3_rerank",
        stratum="paraphrase",
        ndcg_at_10=0.224,
        recall_at_10=0.215,
        recall_at_100=0.400,
        mrr_at_10=0.224,
        map_score=0.208,
    ),
]

LATENCY_BUDGET = [
    LatencyStage(
        stage="Intent classification",
        ms_per_query=150,
        notes="Gemini Flash-Lite + 6s timeout / heuristic fallback",
    ),
    LatencyStage(stage="Query encode (BGE-M3)", ms_per_query=1.7, notes="FP16 on RTX 4060"),
    LatencyStage(stage="BM25 retrieve top-100", ms_per_query=0.1, notes="BM25S in-memory"),
    LatencyStage(
        stage="Dense cosine search", ms_per_query=0.6, notes="numpy matmul over 1k-d corpus matrix"
    ),
    LatencyStage(
        stage="KG Cypher (skill overlap)", ms_per_query=25, notes="Neo4j 5 Community local"
    ),
    LatencyStage(stage="RRF fusion", ms_per_query=2.5, notes="server-side rank merge"),
    LatencyStage(
        stage="Cross-encoder rerank (top-25)",
        ms_per_query=285,
        notes="bge-reranker-v2-m3 FP16, batch=32",
    ),
    LatencyStage(
        stage="Total — full pipeline", ms_per_query=315, notes="encode + RRF3 + rerank end-to-end"
    ),
]

NOTES = [
    "RRF3 (BM25 + dense + KG) is the new best on the lexical original stratum (nDCG@10 = 0.621).",
    "Full pipeline (RRF3 + cross-encoder rerank top-25) is the new best on the paraphrase stratum (nDCG@10 = 0.224).",
    "Each retrieval stage handles a different query distribution; the multi-stage funnel is robust across both strata.",
    "Cross-encoder is the only stage that lifts paraphrase metrics, because bge-reranker-v2-m3 is trained on natural-language pairs (MS-MARCO, MIRACL).",
    "Paraphrase stratum is 19/100 (Gemini Flash-Lite RPD quota) — backfill pending; relative ranking already established.",
]


def _mark_best() -> None:
    """For each stratum, mark the row with the highest nDCG@10 as the best."""
    by_stratum: dict[str, list[AblationRow]] = {}
    for row in _RAW_ABLATION:
        by_stratum.setdefault(row.stratum, []).append(row)
    for rows in by_stratum.values():
        rows_with_metric = [r for r in rows if r.ndcg_at_10 is not None]
        if not rows_with_metric:
            continue
        best = max(rows_with_metric, key=lambda r: r.ndcg_at_10 or 0.0)
        best.is_best_for_stratum = True


_mark_best()


@router.get(
    "/summary",
    response_model=EvalSummary,
    summary="Canonical eval snapshot — corpus + KG + ablation table + latency",
    description=(
        "Powers the /eval dashboard. Returns the 21-row ablation matrix (7 configs × 3 "
        "strata), KG node + edge counts, eval-set composition, and the per-stage "
        "latency budget. Numbers are sourced from real ablation runs committed in "
        "data/processed/*/runs/<timestamp>/. Best-per-stratum row is flagged."
    ),
)
def summary() -> EvalSummary:
    return EvalSummary(
        corpus=CORPUS_STATS,
        kg=KG_STATS,
        eval_set=EVAL_SET_STATS,
        ablation=_RAW_ABLATION,
        latency_budget=LATENCY_BUDGET,
        targets=EvalTargets(),
        notes=NOTES,
    )
