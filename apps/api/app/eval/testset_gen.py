"""Deterministic eval-set generator.

Generates 100 stratified queries (50 candidate-search + 50 job-search) and TREC
qrels, grounded in the actual corpus. Uses a fixed RNG seed so the eval set is
stable across runs.

No LLM required — queries are templated from real entities + their top-N skills.
A second pass adds graded-relevant labels via skill-overlap Jaccard. This gives
us the first ablation row without paying the rate-limit / API-key cost.

Outputs (under data/eval/):
    queries.jsonl      [{qid, task, text, anchor_id}]
    qrels.txt          TREC format: qid 0 doc_id rel
    metadata.json      generation params, RNG seed, corpus row counts

To regenerate: just delete the files and re-run.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd
from rich.console import Console


def _find_repo_root() -> Path:
    """Walk up until we find pnpm-workspace.yaml (repo root marker)."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pnpm-workspace.yaml").exists():
            return parent
    raise RuntimeError("repo root marker not found")


REPO_ROOT = _find_repo_root()
DUCKDB_PATH = REPO_ROOT / "data" / "processed" / "pathfinder.duckdb"
EVAL_DIR = REPO_ROOT / "data" / "eval"

SEED = 42
QUERIES_PER_TASK = 50
RELEVANCE_THRESHOLD = 0.5  # Jaccard threshold for graded relevance (qrel = 1)

Task = Literal["candidate_search", "job_search"]
console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def _proficiency_label(ord_int: int) -> str:
    return {
        1: "Beginner",
        2: "Advanced Beginner",
        3: "Competent",
        4: "Proficient",
        5: "Expert",
    }[ord_int]


# ─────────────────────────────────────────────────────────────────────────────
# Eval-set construction
# ─────────────────────────────────────────────────────────────────────────────


def _build_candidate_queries(
    con: duckdb.DuckDBPyConnection,
    rng: random.Random,
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    """50 candidate-search queries.

    For each anchor profile:
      - take its top-3 highest-proficiency skills (deterministic, name-tiebroken)
      - compose query: "{Skill1} {Skill2} {Skill3} ({MinProficiency}+)"
      - gold (qrel=2): the anchor profile
      - graded relevant (qrel=1): profiles with skill-set Jaccard ≥ THRESHOLD
        AND at least min(2, len(query_skills)) of the query skills present
    """
    # Persons that have ≥ 3 skills (otherwise the query is trivial).
    persons = con.execute(
        """
        SELECT person_id,
               LIST(skill_name ORDER BY proficiency DESC, skill_name ASC)
                   AS skill_names_by_prof,
               LIST(proficiency ORDER BY proficiency DESC, skill_name ASC)
                   AS profs_by_prof
        FROM profile_skills
        WHERE category != 'soft'              -- soft skills make poor query terms
        GROUP BY person_id
        HAVING COUNT(*) >= 3
        """
    ).fetchall()

    if len(persons) < QUERIES_PER_TASK:
        raise RuntimeError(f"Only {len(persons)} persons with ≥ 3 non-soft skills — too few.")

    rng.shuffle(persons)
    selected = persons[:QUERIES_PER_TASK]

    # Pre-compute every person's full skill set for relevance scoring.
    skill_index_rows = con.execute("SELECT person_id, skill_id FROM profile_skills").fetchall()
    person_skills: dict[str, set[str]] = defaultdict(set)
    for pid, sid in skill_index_rows:
        person_skills[pid].add(sid)

    # Skill-id lookup so we can match query terms to the person index.
    name_to_id_rows = con.execute(
        "SELECT DISTINCT skill_name, skill_id FROM profile_skills"
    ).fetchall()
    skill_name_to_id = dict(name_to_id_rows)

    queries: list[dict] = []
    qrels: dict[str, dict[str, int]] = {}

    for i, (anchor_id, names, profs) in enumerate(selected):
        top_names = list(names[:3])
        top_profs = list(profs[:3])
        min_ord = min(top_profs)
        query_text = ", ".join(top_names) + f" engineer at {_proficiency_label(min_ord)} or higher"
        qid = f"cand_{i:03d}"
        query_skill_ids = {skill_name_to_id.get(n, "") for n in top_names if n in skill_name_to_id}
        anchor_skills = person_skills.get(anchor_id, set())

        # Score every person; pick out graded-relevant.
        rels: dict[str, int] = {}
        for pid, sk in person_skills.items():
            if not query_skill_ids:
                continue
            overlap = query_skill_ids & sk
            if pid == anchor_id:
                rels[pid] = 2
                continue
            if len(overlap) < min(2, len(query_skill_ids)):
                continue
            jacc = _jaccard(query_skill_ids, sk)
            if jacc >= RELEVANCE_THRESHOLD:
                rels[pid] = 1

        # Guard: keep gold even if anchor's full skill set somehow excluded query.
        rels.setdefault(anchor_id, 2)

        queries.append(
            {
                "qid": qid,
                "task": "candidate_search",
                "text": query_text,
                "anchor_id": anchor_id,
                "min_proficiency_ordinal": min_ord,
                "query_skills": top_names,
                "anchor_skill_count": len(anchor_skills),
            }
        )
        qrels[qid] = rels

    return queries, qrels


def _build_job_queries(
    con: duckdb.DuckDBPyConnection,
    rng: random.Random,
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    """50 job-search queries.

    For each anchor job:
      - take its designation/title + top-3 must-have skills (or all skills if < 3)
      - compose query: "{Designation} with {S1}, {S2}, {S3}"
      - gold (qrel=2): the anchor job
      - graded relevant (qrel=1): jobs with skill-set Jaccard ≥ THRESHOLD
        AND same/similar designation
    """
    jobs = con.execute(
        """
        SELECT j.id              AS job_id,
               COALESCE(NULLIF(j.designation, ''), NULLIF(j.title, '')) AS title,
               LIST(s.skill_name ORDER BY s.priority ASC, s.skill_name ASC) AS skills
        FROM jobs j
        JOIN job_skills s ON j.id = s.job_id
        WHERE COALESCE(j.designation, j.title) IS NOT NULL
        GROUP BY j.id, COALESCE(NULLIF(j.designation, ''), NULLIF(j.title, ''))
        HAVING COUNT(s.skill_id) >= 1
        """
    ).fetchall()

    if len(jobs) < QUERIES_PER_TASK:
        raise RuntimeError(f"Only {len(jobs)} jobs with title + skills — too few.")

    rng.shuffle(jobs)
    selected = jobs[:QUERIES_PER_TASK]

    job_skills_rows = con.execute("SELECT job_id, skill_id FROM job_skills").fetchall()
    job_skills: dict[str, set[str]] = defaultdict(set)
    for jid, sid in job_skills_rows:
        job_skills[jid].add(sid)

    name_to_id_rows = con.execute("SELECT DISTINCT skill_name, skill_id FROM job_skills").fetchall()
    skill_name_to_id = dict(name_to_id_rows)

    designation_rows = con.execute("SELECT id, designation, title FROM jobs").fetchall()
    job_title: dict[str, str] = {}
    for jid, d, t in designation_rows:
        job_title[jid] = (d or t or "").strip().lower()

    queries: list[dict] = []
    qrels: dict[str, dict[str, int]] = {}

    for i, (anchor_id, title, names) in enumerate(selected):
        top_names = list(names)[:3]
        query_text = f"{title} role with {', '.join(top_names)}"
        qid = f"job_{i:03d}"

        query_skill_ids = {skill_name_to_id.get(n, "") for n in top_names if n in skill_name_to_id}
        anchor_designation = job_title.get(anchor_id, "")

        rels: dict[str, int] = {}
        for jid, sk in job_skills.items():
            if not query_skill_ids:
                continue
            if jid == anchor_id:
                rels[jid] = 2
                continue
            overlap = query_skill_ids & sk
            if not overlap:
                continue
            other_designation = job_title.get(jid, "")
            # Graded-relevant requires SOME designation similarity (substring match)
            # AND ≥ 1 skill overlap. Substring is a deliberately-loose proxy here;
            # the cross-encoder rerank ablation is what tightens recall in later rows.
            shared_desig = bool(anchor_designation) and (
                anchor_designation in other_designation or other_designation in anchor_designation
            )
            jacc = _jaccard(query_skill_ids, sk)
            if shared_desig and jacc >= RELEVANCE_THRESHOLD:
                rels[jid] = 1
        rels.setdefault(anchor_id, 2)

        queries.append(
            {
                "qid": qid,
                "task": "job_search",
                "text": query_text,
                "anchor_id": anchor_id,
                "designation": title,
                "query_skills": top_names,
            }
        )
        qrels[qid] = rels

    return queries, qrels


# ─────────────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────────────


def write_eval_set(
    queries: list[dict],
    qrels: dict[str, dict[str, int]],
    out_dir: Path = EVAL_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "queries.jsonl").open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # TREC qrels: qid 0 doc_id rel
    with (out_dir / "qrels.txt").open("w", encoding="utf-8") as f:
        for qid, rels in qrels.items():
            for doc_id, rel in rels.items():
                f.write(f"{qid} 0 {doc_id} {rel}\n")

    meta = {
        "seed": SEED,
        "queries_per_task": QUERIES_PER_TASK,
        "relevance_threshold_jaccard": RELEVANCE_THRESHOLD,
        "task_breakdown": {
            "candidate_search": sum(1 for q in queries if q["task"] == "candidate_search"),
            "job_search": sum(1 for q in queries if q["task"] == "job_search"),
        },
        "qrels_stats": {
            "total_query_doc_pairs": sum(len(r) for r in qrels.values()),
            "avg_relevants_per_query": round(
                sum(len(r) for r in qrels.values()) / max(len(qrels), 1), 2
            ),
        },
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def generate(force: bool = False) -> dict:
    """Generate or reuse the eval set. Returns the metadata dict."""
    if (EVAL_DIR / "queries.jsonl").exists() and not force:
        console.print(f"[dim]eval set exists at {EVAL_DIR} — reusing (--force to regenerate)[/]")
        with (EVAL_DIR / "metadata.json").open(encoding="utf-8") as f:
            return json.load(f)

    if not DUCKDB_PATH.exists():
        raise FileNotFoundError(f"{DUCKDB_PATH} missing — run scripts/01_etl.py first.")

    rng = random.Random(SEED)
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)

    cand_q, cand_qrels = _build_candidate_queries(con, rng)
    job_q, job_qrels = _build_job_queries(con, rng)

    queries = cand_q + job_q
    qrels = {**cand_qrels, **job_qrels}
    con.close()

    write_eval_set(queries, qrels)

    meta = json.loads((EVAL_DIR / "metadata.json").read_text(encoding="utf-8"))
    console.print(f"[green]✓[/] wrote eval set ({len(queries)} queries) → {EVAL_DIR}")
    return meta


def load_queries() -> pd.DataFrame:
    p = EVAL_DIR / "queries.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run app.eval.testset_gen.generate() first.")
    return pd.read_json(p, lines=True)


def load_qrels() -> dict[str, dict[str, int]]:
    p = EVAL_DIR / "qrels.txt"
    out: dict[str, dict[str, int]] = defaultdict(dict)
    with p.open(encoding="utf-8") as f:
        for line in f:
            qid, _, doc_id, rel = line.strip().split()
            out[qid][doc_id] = int(rel)
    return dict(out)
