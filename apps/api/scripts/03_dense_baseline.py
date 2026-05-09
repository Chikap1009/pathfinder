"""BGE-M3 dense + RRF fusion baselines.

Idempotent. Run with:
    uv run python scripts/03_dense_baseline.py            # encode + dense eval + RRF eval
    uv run python scripts/03_dense_baseline.py encode     # only encode the corpus
    uv run python scripts/03_dense_baseline.py eval-dense
    uv run python scripts/03_dense_baseline.py eval-rrf
    uv run python scripts/03_dense_baseline.py all --regen-eval-set

Outputs (under data/processed/dense/):
    profiles.npy          (n_profiles, 1024) float32 BGE-M3 dense vectors
    profiles_ids.json     parallel list of doc_ids
    jobs.npy              (n_jobs, 1024) float32
    jobs_ids.json
    runs/<timestamp>/dense_run.json    metrics + params for the dense-only ablation row
    runs/<timestamp>/rrf_run.json      metrics + params for the BM25+dense RRF row

Why in-memory cosine instead of Qdrant on Day 2: we have ~3.1k docs total. A
1,024-d FP32 corpus matrix is ~12 MB. NumPy matmul against a single query
embedding takes <1 ms — faster than Qdrant's network round-trip. We swap in
Qdrant when Docker comes up; the search interface stays the same.
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
from app.services.dense import EMBEDDING_DIM, encode_dense, get_dense_encoder


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pnpm-workspace.yaml").exists():
            return parent
    raise RuntimeError("repo root marker not found")


REPO_ROOT = _find_repo_root()
DUCKDB_PATH = REPO_ROOT / "data" / "processed" / "pathfinder.duckdb"
DENSE_DIR = REPO_ROOT / "data" / "processed" / "dense"
BM25_DIR = REPO_ROOT / "data" / "processed" / "bm25"
DENSE_RUNS_DIR = DENSE_DIR / "runs"

TOP_K = 100
RRF_K = 60  # Cormack 2009 default; per ADR-0001

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)


# ─────────────────────────────────────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────────────────────────────────────


def _encode_collection(
    rows: list[tuple[str, str]],
    out_npy: Path,
    out_ids: Path,
    label: str,
) -> None:
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    if out_npy.exists() and out_ids.exists():
        console.print(f"[dim]dense[/] {label}: cached → {out_npy} ({out_npy.stat().st_size:,} B)")
        return

    doc_ids = [r[0] for r in rows]
    docs = [r[1] for r in rows]

    console.print(f"[cyan]dense[/] {label}: encoding {len(docs):,} docs (BGE-M3, FP16, batch=16)…")
    t0 = time.perf_counter()
    vecs = encode_dense(docs, batch_size=16, max_length=2048, show_progress=False)
    elapsed = time.perf_counter() - t0

    arr = np.asarray(vecs, dtype=np.float32)
    np.save(out_npy, arr)
    out_ids.write_text(json.dumps(doc_ids), encoding="utf-8")
    console.print(
        f"  → {out_npy} ({arr.shape}) + {out_ids.name} "
        f"in {elapsed:.1f}s ({elapsed / max(len(docs), 1) * 1000:.1f} ms/doc)"
    )


def encode() -> None:
    if not DUCKDB_PATH.exists():
        raise FileNotFoundError(f"{DUCKDB_PATH} missing — run scripts/01_etl.py first.")
    # Warm the model first so timing in _encode_collection is encode-only.
    get_dense_encoder()

    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    profiles = con.execute("SELECT id, canonical_text FROM profiles").fetchall()
    jobs = con.execute("SELECT id, canonical_text FROM jobs").fetchall()
    con.close()

    _encode_collection(
        profiles, DENSE_DIR / "profiles.npy", DENSE_DIR / "profiles_ids.json", "profiles"
    )
    _encode_collection(jobs, DENSE_DIR / "jobs.npy", DENSE_DIR / "jobs_ids.json", "jobs")


# ─────────────────────────────────────────────────────────────────────────────
# In-memory dense retrieval (cosine over FP32 matrix)
# ─────────────────────────────────────────────────────────────────────────────


class DenseIndex:
    def __init__(self, vectors: np.ndarray, doc_ids: list[str]):
        if vectors.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"expected {EMBEDDING_DIM}-d vectors, got {vectors.shape}")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        self.matrix = vectors / norms  # row-normalised for cosine via dot-product
        self.doc_ids = doc_ids

    @classmethod
    def load(cls, npy_path: Path, ids_path: Path) -> DenseIndex:
        vectors = np.load(npy_path).astype(np.float32)
        doc_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        return cls(vectors, doc_ids)

    def search(self, query_vec: np.ndarray, k: int = TOP_K) -> dict[str, float]:
        q = query_vec.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = self.matrix @ q  # (n,)
        if k >= len(sims):
            top_idx = np.argsort(-sims)
        else:
            top_idx = np.argpartition(-sims, k)[:k]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
        return {self.doc_ids[i]: float(sims[i]) for i in top_idx}


# ─────────────────────────────────────────────────────────────────────────────
# Eval helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_dense_indexes() -> tuple[DenseIndex, DenseIndex]:
    p = DENSE_DIR / "profiles.npy"
    j = DENSE_DIR / "jobs.npy"
    if not (p.exists() and j.exists()):
        raise FileNotFoundError("dense embeddings not built — run `… encode` first.")
    return (
        DenseIndex.load(p, DENSE_DIR / "profiles_ids.json"),
        DenseIndex.load(j, DENSE_DIR / "jobs_ids.json"),
    )


def _load_bm25_indexes() -> tuple[bm25s.BM25, bm25s.BM25]:
    p = BM25_DIR / "profiles_index"
    j = BM25_DIR / "jobs_index"
    if not (p.exists() and j.exists()):
        raise FileNotFoundError(
            "BM25 indexes missing — run scripts/02_bm25_baseline.py index first."
        )
    return bm25s.BM25.load(str(p), load_corpus=True), bm25s.BM25.load(str(j), load_corpus=True)


def _bm25_search(retriever: bm25s.BM25, query: str, k: int = TOP_K) -> dict[str, float]:
    tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
    docs, scores = retriever.retrieve(tokens, k=k, show_progress=False)
    out: dict[str, float] = {}
    for i in range(docs.shape[1]):
        d = docs[0, i]
        doc_id = str(d.get("text") or d.get("id")) if isinstance(d, dict) else str(d)
        out[doc_id] = float(scores[0, i])
    return out


def _rrf_fuse(
    runs: list[dict[str, float]],
    k: int = RRF_K,
    top_k: int = TOP_K,
) -> dict[str, float]:
    """Reciprocal Rank Fusion (Cormack 2009): score = sum 1/(k + rank_i)."""
    rrf_scores: dict[str, float] = {}
    for run in runs:
        ranked = sorted(run.items(), key=lambda kv: kv[1], reverse=True)
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    # truncate to top_k
    return dict(sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k])


# ─────────────────────────────────────────────────────────────────────────────
# Run modes
# ─────────────────────────────────────────────────────────────────────────────


def _persist_run(name: str, run_doc: dict, runs_dict: dict[str, dict[str, float]]) -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = DENSE_RUNS_DIR / ts / name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_doc["run_id"] = ts
    ir_metrics.write_trec_run(runs_dict, run_dir / f"{name}.trec", run_name=name)
    ir_metrics.write_metrics_json(run_doc, run_dir / f"{name}.json")
    console.print(f"[green]✓[/] {name} → {run_dir}")
    return run_dir


def _print_metrics_table(name: str, metrics: dict[str, dict[str, float]]) -> None:
    t = Table(title=name, show_lines=False)
    t.add_column("split", style="cyan")
    for m in ir_metrics.METRICS:
        t.add_column(m, justify="right")
    for split, vals in metrics.items():
        t.add_row(split, *[f"{vals.get(m, 0):.4f}" for m in ir_metrics.METRICS])
    console.print(t)


def eval_dense(regen_eval_set: bool = False) -> dict:
    """Ablation row 2: BGE-M3 dense alone."""
    testset_gen.generate(force=regen_eval_set)
    queries_df = testset_gen.load_queries()
    qrels = testset_gen.load_qrels()

    profiles_idx, jobs_idx = _load_dense_indexes()

    console.print(f"[cyan]dense[/] encoding {len(queries_df)} queries…")
    t0 = time.perf_counter()
    query_vecs = encode_dense(
        queries_df["text"].tolist(),
        batch_size=32,
        max_length=512,
        show_progress=False,
    )
    encode_elapsed = time.perf_counter() - t0

    runs: dict[str, dict[str, float]] = {}
    t1 = time.perf_counter()
    for i, row in enumerate(queries_df.itertuples()):
        idx = profiles_idx if row.task == "candidate_search" else jobs_idx
        runs[row.qid] = idx.search(query_vecs[i])
    search_elapsed = time.perf_counter() - t1
    avg_ms = (search_elapsed / max(len(queries_df), 1)) * 1000

    console.print(
        f"  encode {encode_elapsed:.2f}s ({encode_elapsed / max(len(queries_df), 1) * 1000:.1f} ms/q) · "
        f"search {search_elapsed:.2f}s ({avg_ms:.2f} ms/q)"
    )

    metrics = ir_metrics.per_task_scores(qrels, runs, queries_df, name="bge_m3_dense")
    _print_metrics_table("BGE-M3 dense", metrics)

    run_doc = {
        "name": "bge_m3_dense",
        "params": {
            "model": "BAAI/bge-m3",
            "fp16": True,
            "embedding_dim": EMBEDDING_DIM,
            "top_k": TOP_K,
            "max_length_doc": 2048,
            "max_length_query": 512,
            "search": "in-memory cosine",
        },
        "n_queries": len(queries_df),
        "encode_avg_ms_per_query": round(encode_elapsed / max(len(queries_df), 1) * 1000, 2),
        "search_avg_ms_per_query": round(avg_ms, 2),
        "metrics": metrics,
    }
    _persist_run("dense_run", run_doc, runs)
    return run_doc


def eval_rrf(regen_eval_set: bool = False) -> dict:
    """Ablation row 3: BM25 + BGE-M3 dense fused with RRF (k=60)."""
    testset_gen.generate(force=regen_eval_set)
    queries_df = testset_gen.load_queries()
    qrels = testset_gen.load_qrels()

    bm25_profiles, bm25_jobs = _load_bm25_indexes()
    dense_profiles, dense_jobs = _load_dense_indexes()

    console.print(f"[cyan]rrf[/] encoding {len(queries_df)} queries (dense)…")
    t0 = time.perf_counter()
    query_vecs = encode_dense(
        queries_df["text"].tolist(),
        batch_size=32,
        max_length=512,
        show_progress=False,
    )
    encode_elapsed = time.perf_counter() - t0

    runs: dict[str, dict[str, float]] = {}
    t1 = time.perf_counter()
    for i, row in enumerate(queries_df.itertuples()):
        if row.task == "candidate_search":
            bm25_run = _bm25_search(bm25_profiles, row.text)
            dense_run = dense_profiles.search(query_vecs[i])
        else:
            bm25_run = _bm25_search(bm25_jobs, row.text)
            dense_run = dense_jobs.search(query_vecs[i])
        runs[row.qid] = _rrf_fuse([bm25_run, dense_run], k=RRF_K)
    search_elapsed = time.perf_counter() - t1
    avg_ms = (search_elapsed / max(len(queries_df), 1)) * 1000

    console.print(
        f"  encode {encode_elapsed:.2f}s · search {search_elapsed:.2f}s ({avg_ms:.2f} ms/q)"
    )

    metrics = ir_metrics.per_task_scores(qrels, runs, queries_df, name="rrf_bm25_dense")
    _print_metrics_table(f"RRF (k={RRF_K}) — BM25 + BGE-M3 dense", metrics)

    run_doc = {
        "name": "rrf_bm25_dense",
        "params": {
            "rrf_k": RRF_K,
            "channels": ["bm25_lucene_k1=1.5_b=0.75", "bge_m3_dense_fp16"],
            "top_k": TOP_K,
        },
        "n_queries": len(queries_df),
        "encode_avg_ms_per_query": round(encode_elapsed / max(len(queries_df), 1) * 1000, 2),
        "search_avg_ms_per_query": round(avg_ms, 2),
        "metrics": metrics,
    }
    _persist_run("rrf_run", run_doc, runs)
    return run_doc


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def encode_cmd() -> None:
    """Encode the corpus with BGE-M3 dense (idempotent / cached)."""
    encode()


# Typer can't bind a command literally named `encode` if there's a collision; using `encode` here.
app.command(name="encode")(encode_cmd)


@app.command(name="eval-dense")
def eval_dense_cmd(
    regen_eval_set: bool = typer.Option(False, "--regen-eval-set"),
) -> None:
    """Run dense-only eval (row 2)."""
    eval_dense(regen_eval_set=regen_eval_set)


@app.command(name="eval-rrf")
def eval_rrf_cmd(
    regen_eval_set: bool = typer.Option(False, "--regen-eval-set"),
) -> None:
    """Run RRF(BM25, dense) eval (row 3)."""
    eval_rrf(regen_eval_set=regen_eval_set)


@app.command(name="all")
def all_cmd() -> None:
    """Encode + run dense + RRF evals end-to-end (default)."""
    console.rule("[bold magenta]BGE-M3 dense + RRF baselines[/]")
    encode()
    eval_dense()
    eval_rrf()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("all")
    app()
