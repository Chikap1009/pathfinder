"""KG retrieval baseline — RRF(BM25, dense, KG) ablation row.

Idempotent. Run with:
    uv run python scripts/08_kg_baseline.py            # all 3 KG-channel ablation modes
    uv run python scripts/08_kg_baseline.py kg-only    # KG channel alone vs gold
    uv run python scripts/08_kg_baseline.py rrf3       # RRF over BM25 + dense + KG
    uv run python scripts/08_kg_baseline.py rrf3-rerank # full pipeline incl. cross-encoder

Pre-requisite: Neo4j running and ingested via scripts/07_kg_build.py.

The KG retrieval channel:
  - Maps each query's `query_skills` field to canonical skill IDs via the
    canonical.parquet produced by 06_skill_canonicalize.py.
  - Runs `PERSON_BY_SKILL_OVERLAP` (candidate_search task) or
    `JOB_BY_SKILL_OVERLAP` (job_search task) against Neo4j.
  - Returns top-K candidates ranked by proficiency-weighted skill-overlap score.

Outputs (data/processed/kg/runs/<timestamp>/):
    kg_only.json          metrics + params for KG-channel-only ablation
    rrf3.json             metrics + params for RRF(BM25, dense, KG)
    rrf3_rerank.json      metrics + params for full-pipeline (RRF3 + cross-encoder top-25)
    *.trec                TREC run files
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
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from app.eval import ir_metrics, testset_gen
from app.services.dense import encode_dense
from app.services.kg import search_jobs_by_skills, search_persons_by_skills
from app.services.rerank import rerank_pairs
from app.services.skills import _slug


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pnpm-workspace.yaml").exists():
            return parent
    raise RuntimeError("repo root marker not found")


REPO_ROOT = _find_repo_root()
DUCKDB_PATH = REPO_ROOT / "data" / "processed" / "pathfinder.duckdb"
SKILLS_PARQUET = REPO_ROOT / "data" / "processed" / "skills" / "canonical.parquet"
BM25_DIR = REPO_ROOT / "data" / "processed" / "bm25"
DENSE_DIR = REPO_ROOT / "data" / "processed" / "dense"
KG_RUNS_DIR = REPO_ROOT / "data" / "processed" / "kg" / "runs"

CANDIDATE_BUDGET = 100
RRF_K = 60
RERANK_TOP_N = 25

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)


# ─── Loaders shared with previous baselines ──────────────────────────────────


def _load_bm25() -> tuple[bm25s.BM25, bm25s.BM25]:
    p = BM25_DIR / "profiles_index"
    j = BM25_DIR / "jobs_index"
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
        top = np.argpartition(-sims, k)[:k] if k < len(sims) else np.argsort(-sims)
        top = top[np.argsort(-sims[top])]
        return {self.doc_ids[i]: float(sims[i]) for i in top}


def _load_dense() -> tuple[_DenseIdx, _DenseIdx]:
    p, j = DENSE_DIR / "profiles.npy", DENSE_DIR / "jobs.npy"
    if not (p.exists() and j.exists()):
        raise FileNotFoundError(
            "Dense embeddings missing — run scripts/03_dense_baseline.py first."
        )
    return (
        _DenseIdx(p, DENSE_DIR / "profiles_ids.json"),
        _DenseIdx(j, DENSE_DIR / "jobs_ids.json"),
    )


def _load_canonical_skill_map() -> dict[str, str]:
    """legacy_id (slug of raw_name) → canonical_id."""
    if not SKILLS_PARQUET.exists():
        raise FileNotFoundError(
            f"{SKILLS_PARQUET} missing — run scripts/06_skill_canonicalize.py first."
        )
    df = pd.read_parquet(SKILLS_PARQUET)
    return dict(zip(df["legacy_id"], df["canonical_id"], strict=True))


def _load_canonical_text() -> dict[str, str]:
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    out: dict[str, str] = {}
    for did, txt in con.execute("SELECT id, canonical_text FROM profiles").fetchall():
        out[did] = txt
    for did, txt in con.execute("SELECT id, canonical_text FROM jobs").fetchall():
        out[did] = txt
    con.close()
    return out


# ─── KG channel ──────────────────────────────────────────────────────────────


def _query_canonical_skill_ids(query_skills: list[str], skill_map: dict[str, str]) -> list[str]:
    """Map query_skills (raw names) → canonical skill IDs."""
    out: list[str] = []
    for name in query_skills or []:
        legacy = _slug(name)
        canonical = skill_map.get(legacy)
        if canonical is not None:
            out.append(canonical)
    return out


def _kg_search(task: str, skill_ids: list[str], top_k: int = CANDIDATE_BUDGET) -> dict[str, float]:
    """Run the appropriate Cypher template; return doc_id → score."""
    if not skill_ids:
        return {}
    if task == "candidate_search":
        rows = search_persons_by_skills(skill_ids, top_k=top_k)
    elif task == "job_search":
        rows = search_jobs_by_skills(skill_ids, top_k=top_k)
    else:
        return {}
    return {r["doc_id"]: float(r["score"]) for r in rows}


# ─── BM25 + dense retrieval helpers ─────────────────────────────────────────


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


# ─── Eval modes ──────────────────────────────────────────────────────────────


def _persist(name: str, run_doc: dict, runs: dict[str, dict[str, float]]) -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = KG_RUNS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    run_doc["run_id"] = ts
    ir_metrics.write_trec_run(runs, run_dir / f"{name}.trec", run_name=name)
    (run_dir / f"{name}.json").write_text(json.dumps(run_doc, indent=2), encoding="utf-8")
    console.print(f"[green]✓[/] {name} → {run_dir}")
    return run_dir


def _print_metrics(name: str, metrics: dict[str, dict[str, float]]) -> None:
    t = Table(title=name, show_lines=False)
    t.add_column("split", style="cyan")
    for m in ir_metrics.METRICS:
        t.add_column(m, justify="right")
    for split, vals in metrics.items():
        t.add_row(split, *[f"{vals.get(m, 0):.4f}" for m in ir_metrics.METRICS])
    console.print(t)


def run_kg_only() -> dict:
    """KG channel alone vs gold."""
    queries_df = testset_gen.load_queries()
    qrels = testset_gen.load_qrels()
    skill_map = _load_canonical_skill_map()

    runs: dict[str, dict[str, float]] = {}
    n_no_skills = 0
    t0 = time.perf_counter()
    for row in queries_df.itertuples():
        sids = _query_canonical_skill_ids(row.query_skills, skill_map)
        if not sids:
            n_no_skills += 1
            runs[row.qid] = {}
            continue
        runs[row.qid] = _kg_search(row.task, sids)
    elapsed = time.perf_counter() - t0
    metrics = ir_metrics.per_task_scores(qrels, runs, queries_df, name="kg_only")
    _print_metrics("KG channel only", metrics)

    run_doc = {
        "name": "kg_only",
        "params": {"top_k": CANDIDATE_BUDGET, "weighting": "proficiency × category-bucket"},
        "n_queries": len(queries_df),
        "n_queries_no_skills": n_no_skills,
        "search_total_s": round(elapsed, 3),
        "search_avg_ms_per_query": round(elapsed / max(len(queries_df), 1) * 1000, 2),
        "metrics": metrics,
    }
    _persist("kg_only", run_doc, runs)
    return run_doc


def run_rrf3() -> dict:
    """RRF over BM25 + dense + KG."""
    queries_df = testset_gen.load_queries()
    qrels = testset_gen.load_qrels()
    skill_map = _load_canonical_skill_map()
    bm25_p, bm25_j = _load_bm25()
    dense_p, dense_j = _load_dense()

    qvecs = encode_dense(
        queries_df["text"].tolist(), batch_size=32, max_length=512, show_progress=False
    )

    runs: dict[str, dict[str, float]] = {}
    t0 = time.perf_counter()
    for i, row in enumerate(queries_df.itertuples()):
        if row.task == "candidate_search":
            bm25_run = _bm25_search(bm25_p, row.text)
            dense_run = dense_p.search(qvecs[i])
        else:
            bm25_run = _bm25_search(bm25_j, row.text)
            dense_run = dense_j.search(qvecs[i])
        sids = _query_canonical_skill_ids(row.query_skills, skill_map)
        kg_run = _kg_search(row.task, sids) if sids else {}
        channels = [bm25_run, dense_run]
        if kg_run:
            channels.append(kg_run)
        runs[row.qid] = _rrf_fuse(channels)
    elapsed = time.perf_counter() - t0

    metrics = ir_metrics.per_task_scores(qrels, runs, queries_df, name="rrf_bm25_dense_kg")
    _print_metrics(f"RRF (k={RRF_K}) — BM25 + dense + KG", metrics)
    run_doc = {
        "name": "rrf3",
        "params": {"rrf_k": RRF_K, "channels": ["bm25", "bge_m3_dense", "kg_skill_overlap"]},
        "n_queries": len(queries_df),
        "search_total_s": round(elapsed, 3),
        "metrics": metrics,
    }
    _persist("rrf3", run_doc, runs)
    return run_doc


def run_rrf3_rerank() -> dict:
    """Full pipeline: RRF3 → cross-encoder rerank top-N."""
    queries_df = testset_gen.load_queries()
    qrels = testset_gen.load_qrels()
    skill_map = _load_canonical_skill_map()
    canonical = _load_canonical_text()
    bm25_p, bm25_j = _load_bm25()
    dense_p, dense_j = _load_dense()

    qvecs = encode_dense(
        queries_df["text"].tolist(), batch_size=32, max_length=512, show_progress=False
    )

    rrf_full: dict[str, list[tuple[str, float]]] = {}
    for i, row in enumerate(queries_df.itertuples()):
        if row.task == "candidate_search":
            bm25_run = _bm25_search(bm25_p, row.text)
            dense_run = dense_p.search(qvecs[i])
        else:
            bm25_run = _bm25_search(bm25_j, row.text)
            dense_run = dense_j.search(qvecs[i])
        sids = _query_canonical_skill_ids(row.query_skills, skill_map)
        kg_run = _kg_search(row.task, sids) if sids else {}
        channels = [bm25_run, dense_run]
        if kg_run:
            channels.append(kg_run)
        fused = _rrf_fuse(channels)
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:CANDIDATE_BUDGET]
        rrf_full[row.qid] = ranked

    pairs: list[tuple[str, str]] = []
    pair_meta: list[tuple[str, str]] = []
    for row in queries_df.itertuples():
        for doc_id, _ in rrf_full[row.qid][:RERANK_TOP_N]:
            pairs.append((row.text, canonical.get(doc_id, "")))
            pair_meta.append((row.qid, doc_id))

    console.print(f"[cyan]rerank[/] cross-encoder over {len(pairs):,} pairs…")
    t0 = time.perf_counter()
    scores = rerank_pairs(pairs, batch_size=32, normalize=True)
    rerank_elapsed = time.perf_counter() - t0

    runs: dict[str, dict[str, float]] = {qid: {} for qid in rrf_full}
    for (qid, doc_id), s in zip(pair_meta, scores, strict=True):
        runs[qid][doc_id] = float(s)
    for row in queries_df.itertuples():
        for rank, (doc_id, _) in enumerate(
            rrf_full[row.qid][RERANK_TOP_N:], start=RERANK_TOP_N + 1
        ):
            runs[row.qid].setdefault(doc_id, -float(rank))

    metrics = ir_metrics.per_task_scores(qrels, runs, queries_df, name="rrf3_rerank")
    _print_metrics("Full pipeline — RRF3 + cross-encoder rerank top-25", metrics)
    run_doc = {
        "name": "rrf3_rerank",
        "params": {
            "rrf_k": RRF_K,
            "channels": ["bm25", "bge_m3_dense", "kg_skill_overlap"],
            "rerank_model": "BAAI/bge-reranker-v2-m3",
            "rerank_top_n": RERANK_TOP_N,
        },
        "n_queries": len(queries_df),
        "rerank_total_s": round(rerank_elapsed, 3),
        "rerank_avg_ms_per_query": round(rerank_elapsed / max(len(queries_df), 1) * 1000, 2),
        "metrics": metrics,
    }
    _persist("rrf3_rerank", run_doc, runs)
    return run_doc


# ─── CLI ─────────────────────────────────────────────────────────────────────


@app.command(name="kg-only")
def kg_only_cmd() -> None:
    run_kg_only()


@app.command(name="rrf3")
def rrf3_cmd() -> None:
    run_rrf3()


@app.command(name="rrf3-rerank")
def rrf3_rerank_cmd() -> None:
    run_rrf3_rerank()


@app.command(name="all")
def all_cmd() -> None:
    """Run all three KG-channel ablation modes."""
    console.rule("[bold magenta]KG retrieval baseline[/]")
    run_kg_only()
    run_rrf3()
    run_rrf3_rerank()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("all")
    app()
