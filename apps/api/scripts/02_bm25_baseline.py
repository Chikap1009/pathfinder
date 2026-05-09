"""BM25 baseline — index profiles + jobs and evaluate against the seed eval set.

Idempotent. Run with:
    uv run python scripts/02_bm25_baseline.py            # index + eval
    uv run python scripts/02_bm25_baseline.py index      # only build the indexes
    uv run python scripts/02_bm25_baseline.py eval       # only run eval (assumes indexes exist)
    uv run python scripts/02_bm25_baseline.py eval --regen-eval-set

Outputs (under data/processed/bm25/):
    profiles_index/          BM25S persisted index for profiles canonical_text
    jobs_index/              BM25S persisted index for jobs canonical_text
    runs/<timestamp>/
        bm25_run_overall.trec
        bm25_run_overall.json   per-task metrics
        bm25_run.json           run document (parameters, dataset version, scores)
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import bm25s
import duckdb
import typer
from rich.console import Console
from rich.table import Table

from app.eval import ir_metrics, testset_gen


def _find_repo_root() -> Path:
    """Walk up until we find pnpm-workspace.yaml (repo root marker)."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pnpm-workspace.yaml").exists():
            return parent
    raise RuntimeError("repo root marker not found")


REPO_ROOT = _find_repo_root()
DUCKDB_PATH = REPO_ROOT / "data" / "processed" / "pathfinder.duckdb"
BM25_DIR = REPO_ROOT / "data" / "processed" / "bm25"
RUNS_DIR = BM25_DIR / "runs"

# BM25S BM25+ params (Lucene-style defaults; per ADR-0001 / canonical plan).
BM25_K1 = 1.5
BM25_B = 0.75
BM25_METHOD = "lucene"  # bm25s supported methods: lucene | atire | bm25l | bm25+ | robertson
TOP_K = 100  # candidate budget for Recall@100

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)


# ─────────────────────────────────────────────────────────────────────────────
# Index build
# ─────────────────────────────────────────────────────────────────────────────


def _build_index(
    docs: list[str],
    doc_ids: list[str],
    out_dir: Path,
    label: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]index[/] {label}: tokenising {len(docs):,} docs…")
    t0 = time.perf_counter()
    tokens = bm25s.tokenize(docs, stopwords="en", show_progress=False)
    retriever = bm25s.BM25(method=BM25_METHOD, k1=BM25_K1, b=BM25_B)
    retriever.index(tokens, show_progress=False)
    retriever.save(str(out_dir), corpus=doc_ids)
    elapsed = time.perf_counter() - t0
    console.print(f"  → saved → {out_dir} ({elapsed:.2f}s)")


def build_indexes() -> None:
    if not DUCKDB_PATH.exists():
        raise FileNotFoundError(f"{DUCKDB_PATH} missing — run scripts/01_etl.py first.")
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)

    profiles = con.execute("SELECT id, canonical_text FROM profiles").fetchall()
    jobs = con.execute("SELECT id, canonical_text FROM jobs").fetchall()
    con.close()

    BM25_DIR.mkdir(parents=True, exist_ok=True)
    _build_index(
        docs=[t for _, t in profiles],
        doc_ids=[i for i, _ in profiles],
        out_dir=BM25_DIR / "profiles_index",
        label="profiles",
    )
    _build_index(
        docs=[t for _, t in jobs],
        doc_ids=[i for i, _ in jobs],
        out_dir=BM25_DIR / "jobs_index",
        label="jobs",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Eval
# ─────────────────────────────────────────────────────────────────────────────


def _load_index(path: Path) -> bm25s.BM25:
    if not path.exists():
        raise FileNotFoundError(f"BM25 index not built: {path} (run `… index` first).")
    return bm25s.BM25.load(str(path), load_corpus=True)


def _search(retriever: bm25s.BM25, query: str, k: int = TOP_K) -> dict[str, float]:
    tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
    docs, scores = retriever.retrieve(tokens, k=k, show_progress=False)
    out: dict[str, float] = {}
    for i in range(docs.shape[1]):
        d = docs[0, i]
        # When loaded with load_corpus=True bm25s wraps each entry as
        # {'id': <row_index>, 'text': <our_actual_doc_id_string>}.
        # The 'text' key holds the string id we passed at index time.
        if isinstance(d, dict):
            doc_id = str(d.get("text") or d.get("doc_id") or d.get("id"))
        else:
            doc_id = str(d)
        out[doc_id] = float(scores[0, i])
    return out


def run_eval(regen_eval_set: bool = False) -> dict:
    testset_gen.generate(force=regen_eval_set)

    queries_df = testset_gen.load_queries()
    qrels = testset_gen.load_qrels()

    profiles_idx = _load_index(BM25_DIR / "profiles_index")
    jobs_idx = _load_index(BM25_DIR / "jobs_index")

    runs: dict[str, dict[str, float]] = {}
    t0 = time.perf_counter()
    for row in queries_df.itertuples():
        if row.task == "candidate_search":
            runs[row.qid] = _search(profiles_idx, row.text)
        elif row.task == "job_search":
            runs[row.qid] = _search(jobs_idx, row.text)
    elapsed = time.perf_counter() - t0
    avg_ms = (elapsed / max(len(queries_df), 1)) * 1000
    console.print(
        f"[cyan]search[/] {len(queries_df)} queries in {elapsed:.2f}s ({avg_ms:.1f} ms/query)"
    )

    metrics = ir_metrics.per_task_scores(qrels, runs, queries_df, name="bm25_baseline")

    # Persist artifacts
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    ir_metrics.write_trec_run(runs, run_dir / "bm25_run_overall.trec", run_name="bm25")
    ir_metrics.write_metrics_json(metrics, run_dir / "bm25_run_overall.json")

    run_doc = {
        "run_id": ts,
        "name": "bm25_baseline",
        "params": {
            "method": BM25_METHOD,
            "k1": BM25_K1,
            "b": BM25_B,
            "top_k": TOP_K,
            "stopwords": "en",
            "stemmer": None,
        },
        "n_queries": len(queries_df),
        "avg_latency_ms": round(avg_ms, 2),
        "metrics": metrics,
    }
    with (run_dir / "bm25_run.json").open("w", encoding="utf-8") as f:
        json.dump(run_doc, f, indent=2)

    # Pretty-print
    t = Table(title=f"BM25 baseline — {ts}", show_lines=False)
    t.add_column("split", style="cyan")
    for m in ir_metrics.METRICS:
        t.add_column(m, justify="right")
    for split, vals in metrics.items():
        t.add_row(split, *[f"{vals.get(m, 0):.4f}" for m in ir_metrics.METRICS])
    console.print(t)
    console.print(f"[green]✓[/] artifacts → {run_dir}")
    return run_doc


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def index() -> None:
    """Build the BM25 indexes only."""
    build_indexes()


@app.command()
def eval(
    regen_eval_set: bool = typer.Option(False, "--regen-eval-set", help="Regenerate the eval set."),
) -> None:
    """Run BM25 eval (uses existing indexes)."""
    run_eval(regen_eval_set=regen_eval_set)


@app.command()
def all() -> None:
    """Build indexes + run eval end-to-end (default)."""
    console.rule("[bold magenta]BM25 baseline — index + eval[/]")
    build_indexes()
    run_eval()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("all")
    app()
