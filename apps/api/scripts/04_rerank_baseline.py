"""Cross-encoder rerank baseline — RRF top-N → bge-reranker-v2-m3 → top-10.

Idempotent. Run with:
    uv run python scripts/04_rerank_baseline.py            # default: rerank top-50
    uv run python scripts/04_rerank_baseline.py --top-n 25 # tighter funnel
    uv run python scripts/04_rerank_baseline.py --top-n 100

Pipeline:
    Stage 1: BM25 top-100 + BGE-M3 dense top-100
    Stage 2: RRF fusion (k=60) → top-N (default N=50)
    Stage 3: Cross-encoder rerank (bge-reranker-v2-m3 FP16) → top-100 (re-ordered)
    Stage 4: ranx metrics over the reranked list

Outputs (under data/processed/rerank/runs/<timestamp>/):
    rerank_run.json       params + per-task metrics + latency
    rerank_run.trec       reranked top-100 in TREC format

We re-use the BM25 indexes (data/processed/bm25/) and dense embeddings
(data/processed/dense/) built by scripts 02 + 03; this script never re-encodes
the corpus. Only queries get encoded fresh (cheap, ~1.7 ms/q on FP16).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import bm25s
import duckdb
import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from app.eval import ir_metrics, testset_gen
from app.services.dense import encode_dense
from app.services.rerank import rerank_pairs


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pnpm-workspace.yaml").exists():
            return parent
    raise RuntimeError("repo root marker not found")


REPO_ROOT = _find_repo_root()
DUCKDB_PATH = REPO_ROOT / "data" / "processed" / "pathfinder.duckdb"
BM25_DIR = REPO_ROOT / "data" / "processed" / "bm25"
DENSE_DIR = REPO_ROOT / "data" / "processed" / "dense"
RERANK_RUNS_DIR = REPO_ROOT / "data" / "processed" / "rerank" / "runs"

CANDIDATE_BUDGET = 100  # stage-1 top-K per channel
RRF_K = 60  # RRF constant (Cormack 2009)
DEFAULT_RERANK_TOP_N = 50  # how many candidates the cross-encoder sees per query

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)


# ─────────────────────────────────────────────────────────────────────────────
# Index loaders
# ─────────────────────────────────────────────────────────────────────────────


def _load_bm25() -> tuple[bm25s.BM25, bm25s.BM25]:
    p, j = BM25_DIR / "profiles_index", BM25_DIR / "jobs_index"
    if not (p.exists() and j.exists()):
        raise FileNotFoundError("BM25 indexes missing — run scripts/02_bm25_baseline.py first.")
    return bm25s.BM25.load(str(p), load_corpus=True), bm25s.BM25.load(str(j), load_corpus=True)


class _DenseIdx:
    def __init__(self, npy_path: Path, ids_path: Path):
        m = np.load(npy_path).astype(np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
        self.matrix = m / norms
        self.doc_ids: list[str] = json.loads(ids_path.read_text(encoding="utf-8"))

    def search(self, q: np.ndarray, k: int = CANDIDATE_BUDGET) -> dict[str, float]:
        q = q.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = self.matrix @ q
        if k >= len(sims):
            top = np.argsort(-sims)
        else:
            top = np.argpartition(-sims, k)[:k]
            top = top[np.argsort(-sims[top])]
        return {self.doc_ids[i]: float(sims[i]) for i in top}


def _load_dense() -> tuple[_DenseIdx, _DenseIdx]:
    p = DENSE_DIR / "profiles.npy"
    j = DENSE_DIR / "jobs.npy"
    if not (p.exists() and j.exists()):
        raise FileNotFoundError(
            "Dense embeddings missing — run scripts/03_dense_baseline.py first."
        )
    return (
        _DenseIdx(p, DENSE_DIR / "profiles_ids.json"),
        _DenseIdx(j, DENSE_DIR / "jobs_ids.json"),
    )


def _load_canonical_text() -> dict[str, str]:
    """{doc_id: canonical_text} for both profiles + jobs (used for rerank pairs)."""
    if not DUCKDB_PATH.exists():
        raise FileNotFoundError(f"{DUCKDB_PATH} missing — run scripts/01_etl.py first.")
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    out: dict[str, str] = {}
    for did, txt in con.execute("SELECT id, canonical_text FROM profiles").fetchall():
        out[did] = txt
    for did, txt in con.execute("SELECT id, canonical_text FROM jobs").fetchall():
        out[did] = txt
    con.close()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Fusion + search helpers
# ─────────────────────────────────────────────────────────────────────────────


def _bm25_search(retriever: bm25s.BM25, query: str, k: int = CANDIDATE_BUDGET) -> dict[str, float]:
    tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
    docs, scores = retriever.retrieve(tokens, k=k, show_progress=False)
    out: dict[str, float] = {}
    for i in range(docs.shape[1]):
        d = docs[0, i]
        doc_id = str(d.get("text") or d.get("id")) if isinstance(d, dict) else str(d)
        out[doc_id] = float(scores[0, i])
    return out


def _rrf_fuse(runs: list[dict[str, float]], k: int = RRF_K) -> dict[str, float]:
    rrf: dict[str, float] = {}
    for run in runs:
        for rank, (doc_id, _) in enumerate(
            sorted(run.items(), key=lambda kv: kv[1], reverse=True), start=1
        ):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (k + rank)
    return rrf


# ─────────────────────────────────────────────────────────────────────────────
# Main eval
# ─────────────────────────────────────────────────────────────────────────────


def run_eval(rerank_top_n: int = DEFAULT_RERANK_TOP_N, regen_eval_set: bool = False) -> dict:
    testset_gen.generate(force=regen_eval_set)
    queries_df = testset_gen.load_queries()
    qrels = testset_gen.load_qrels()

    canonical = _load_canonical_text()
    bm25_p, bm25_j = _load_bm25()
    dense_p, dense_j = _load_dense()

    console.print(f"[cyan]rerank[/] encoding {len(queries_df)} queries (dense)…")
    t0 = time.perf_counter()
    qvecs = encode_dense(
        queries_df["text"].tolist(),
        batch_size=32,
        max_length=512,
        show_progress=False,
    )
    encode_elapsed = time.perf_counter() - t0

    # Stage 1+2: per-query RRF candidate list of length CANDIDATE_BUDGET (100).
    # The cross-encoder reranks the top-N within that list; below-N stays at RRF order.
    console.print(
        f"[cyan]rerank[/] stage 1+2: BM25 + dense + RRF "
        f"(candidate_budget={CANDIDATE_BUDGET}, rerank_top_n={rerank_top_n})…"
    )
    t1 = time.perf_counter()
    rrf_full: dict[str, list[tuple[str, float]]] = {}
    for i, row in enumerate(queries_df.itertuples()):
        if row.task == "candidate_search":
            bm25_run = _bm25_search(bm25_p, row.text)
            dense_run = dense_p.search(qvecs[i])
        else:
            bm25_run = _bm25_search(bm25_j, row.text)
            dense_run = dense_j.search(qvecs[i])
        fused = _rrf_fuse([bm25_run, dense_run])
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:CANDIDATE_BUDGET]
        rrf_full[row.qid] = ranked
    stage12_elapsed = time.perf_counter() - t1

    # Stage 3: build (query, doc) pairs ONLY for the top-N to rerank.
    pairs: list[tuple[str, str]] = []
    pair_meta: list[tuple[str, str]] = []  # (qid, doc_id) parallel to pairs
    for row in queries_df.itertuples():
        for doc_id, _ in rrf_full[row.qid][:rerank_top_n]:
            text = canonical.get(doc_id, "")
            pairs.append((row.text, text))
            pair_meta.append((row.qid, doc_id))

    console.print(
        f"[cyan]rerank[/] stage 3: cross-encoder over {len(pairs):,} pairs "
        f"({rerank_top_n} per query x {len(queries_df)} queries)..."
    )
    t2 = time.perf_counter()
    scores = rerank_pairs(pairs, batch_size=32, normalize=True)
    rerank_elapsed = time.perf_counter() - t2
    avg_pair_us = (rerank_elapsed / max(len(pairs), 1)) * 1e6
    avg_query_ms = (rerank_elapsed / max(len(queries_df), 1)) * 1000

    # Re-bucket pair scores back into per-query runs.
    # Cross-encoder scores are in [0, 1]; we use them directly for ranks 1..rerank_top_n.
    # For ranks rerank_top_n+1..CANDIDATE_BUDGET we keep the RRF order, mapped to
    # negative scores so they sort strictly BELOW any reranked candidate.
    runs: dict[str, dict[str, float]] = {qid: {} for qid in rrf_full}
    for (qid, doc_id), s in zip(pair_meta, scores, strict=True):
        runs[qid][doc_id] = float(s)
    for row in queries_df.itertuples():
        for rank, (doc_id, _rrf_score) in enumerate(
            rrf_full[row.qid][rerank_top_n:], start=rerank_top_n + 1
        ):
            # Strict ordering below the rerank floor: -rank guarantees sort stability
            # AND that any reranked doc (≥ 0 score) outranks any padded doc.
            runs[row.qid].setdefault(doc_id, -float(rank))

    metrics = ir_metrics.per_task_scores(qrels, runs, queries_df, name="cross_encoder_rerank")

    total_elapsed = encode_elapsed + stage12_elapsed + rerank_elapsed
    console.print(
        f"[dim]  encode {encode_elapsed:.2f}s | stage 1+2 {stage12_elapsed:.2f}s | "
        f"rerank {rerank_elapsed:.2f}s | total {total_elapsed:.2f}s "
        f"({avg_query_ms:.1f} ms/query, {avg_pair_us:.0f} µs/pair)[/]"
    )

    t = Table(
        title=f"Cross-encoder rerank — bge-reranker-v2-m3 (top-{rerank_top_n})", show_lines=False
    )
    t.add_column("split", style="cyan")
    for m in ir_metrics.METRICS:
        t.add_column(m, justify="right")
    for split, vals in metrics.items():
        t.add_row(split, *[f"{vals.get(m, 0):.4f}" for m in ir_metrics.METRICS])
    console.print(t)

    # Persist artifacts
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RERANK_RUNS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    ir_metrics.write_trec_run(runs, run_dir / "rerank_run.trec", run_name="rerank")
    run_doc = {
        "run_id": ts,
        "name": "cross_encoder_rerank",
        "params": {
            "stage1_method": "bm25_lucene_k1=1.5_b=0.75",
            "stage1_dense_model": "BAAI/bge-m3",
            "fuse_method": f"rrf_k={RRF_K}",
            "rerank_model": "BAAI/bge-reranker-v2-m3",
            "rerank_fp16": True,
            "rerank_top_n": rerank_top_n,
            "candidate_budget": CANDIDATE_BUDGET,
            "search_normalize": True,
        },
        "n_queries": len(queries_df),
        "n_pairs": len(pairs),
        "latency": {
            "encode_total_s": round(encode_elapsed, 3),
            "stage12_total_s": round(stage12_elapsed, 3),
            "rerank_total_s": round(rerank_elapsed, 3),
            "total_s": round(total_elapsed, 3),
            "avg_ms_per_query": round(avg_query_ms, 2),
            "avg_us_per_pair": round(avg_pair_us, 1),
        },
        "metrics": metrics,
    }
    (run_dir / "rerank_run.json").write_text(json.dumps(run_doc, indent=2), encoding="utf-8")
    console.print(f"[green]✓[/] artifacts → {run_dir}")
    return run_doc


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


@app.command(name="eval")
def eval_cmd(
    top_n: int = typer.Option(
        DEFAULT_RERANK_TOP_N,
        "--top-n",
        help="Number of RRF candidates the cross-encoder sees per query.",
    ),
    regen_eval_set: bool = typer.Option(False, "--regen-eval-set"),
) -> None:
    run_eval(rerank_top_n=top_n, regen_eval_set=regen_eval_set)


@app.command(name="all")
def all_cmd(
    top_n: int = typer.Option(DEFAULT_RERANK_TOP_N, "--top-n"),
) -> None:
    """Run rerank eval (default top-N=50)."""
    console.rule("[bold magenta]Cross-encoder rerank baseline[/]")
    run_eval(rerank_top_n=top_n)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("all")
    app()
