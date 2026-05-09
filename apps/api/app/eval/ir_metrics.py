"""Thin ranx wrapper for the standard PathFinder IR metric panel.

Computes Recall@10 / Recall@100 / MRR@10 / MAP / nDCG@10, returns both per-query
and corpus-aggregated values, and emits a TREC-style run file for archival.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from ranx import Qrels, Run, evaluate

METRICS = ("recall@10", "recall@100", "mrr@10", "map", "ndcg@10")


def to_run(
    runs: dict[str, dict[str, float]],
    name: str = "run",
) -> Run:
    """`runs[qid][doc_id] = score` → ranx.Run."""
    return Run(runs, name=name)


def to_qrels(qrels: dict[str, dict[str, int]]) -> Qrels:
    return Qrels(qrels)


def score(
    qrels_dict: dict[str, dict[str, int]],
    runs_dict: dict[str, dict[str, float]],
    name: str = "run",
    metrics: Iterable[str] = METRICS,
) -> dict[str, float]:
    qrels = to_qrels(qrels_dict)
    run = to_run(runs_dict, name=name)
    return {m: float(evaluate(qrels, run, m)) for m in metrics}


def per_task_scores(
    qrels_dict: dict[str, dict[str, int]],
    runs_dict: dict[str, dict[str, float]],
    queries_df: pd.DataFrame,
    name: str = "run",
) -> dict[str, dict[str, float]]:
    """Aggregate metrics overall + per-task split + per-stratum split (if present)."""
    out: dict[str, dict[str, float]] = {"overall": score(qrels_dict, runs_dict, name=name)}

    # Per-task split (candidate_search / job_search)
    for task in sorted(queries_df["task"].unique()):
        task_qids = set(queries_df.loc[queries_df["task"] == task, "qid"])
        sub_qrels = {q: r for q, r in qrels_dict.items() if q in task_qids}
        sub_runs = {q: r for q, r in runs_dict.items() if q in task_qids}
        if sub_qrels:
            out[task] = score(sub_qrels, sub_runs, name=f"{name}::{task}")

    # Per-stratum split (original / paraphrase)
    if "stratum" in queries_df.columns:
        # Treat NaN / missing as "original" so the split is exhaustive.
        strat_col = queries_df["stratum"].fillna("original")
        for stratum in sorted(strat_col.unique()):
            qids = set(queries_df.loc[strat_col == stratum, "qid"])
            sub_qrels = {q: r for q, r in qrels_dict.items() if q in qids}
            sub_runs = {q: r for q, r in runs_dict.items() if q in qids}
            if sub_qrels:
                out[f"stratum:{stratum}"] = score(
                    sub_qrels, sub_runs, name=f"{name}::stratum:{stratum}"
                )

    return out


def write_trec_run(
    runs_dict: dict[str, dict[str, float]],
    out_path: Path,
    run_name: str,
    top_k: int = 100,
) -> None:
    """Emit standard TREC run file: qid Q0 doc_id rank score run_name."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for qid, doc_scores in runs_dict.items():
            ranked = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            for rank, (doc_id, score_val) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score_val:.6f} {run_name}\n")


def write_metrics_json(metrics: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
