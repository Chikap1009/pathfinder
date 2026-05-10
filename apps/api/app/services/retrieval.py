"""Retrieval orchestrator — the production pipeline.

Wraps BM25 + dense + KG + cross-encoder rerank into a single typed engine
used by the /v1/search endpoint. Mirrors the offline ablation scripts but
emits structured results + per-stage timings/scores ready for SSE streaming.

Initialization (lazy, on first call): loads BM25 indexes, dense embeddings,
canonical skill map, and DuckDB metadata for per-result snippets. The
expensive ML models (BGE-M3, bge-reranker-v2-m3) load on first encode/rerank
call via their existing service-level lru_cache.

Streaming usage:
    engine = get_engine()
    async for ev in engine.search_stream(query, top_k=10):
        ...

Synchronous usage:
    response = engine.search(query, top_k=10)
"""

from __future__ import annotations

import asyncio
import functools
import json
import re
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import bm25s
import duckdb
import numpy as np

from app.api.v1.schemas import (
    IntentResult,
    JobDetail,
    JobSkill,
    PipelineMode,
    ProfileDetail,
    ProfileSkill,
    SearchResponse,
    SearchResult,
    SearchResultEntity,
    StageEvent,
    StageScores,
    TargetSide,
    TimingMs,
)
from app.core.logging import get_logger
from app.services.kg import search_jobs_by_skills, search_persons_by_skills
from app.services.skills import _slug

log = get_logger(__name__)


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
SKILLS_PARQUET = REPO_ROOT / "data" / "processed" / "skills" / "canonical.parquet"

CANDIDATE_BUDGET = 100
RRF_K = 60
RERANK_TOP_N = 25
SNIPPET_MAX_CHARS = 220


# ─── Dense in-memory index (mirrors scripts/03_dense_baseline.py) ───────────


class _DenseIndex:
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


# ─── Helpers ────────────────────────────────────────────────────────────────


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
    out: dict[str, float] = {}
    for run in runs:
        for rank, (doc_id, _) in enumerate(
            sorted(run.items(), key=lambda kv: kv[1], reverse=True), start=1
        ):
            out[doc_id] = out.get(doc_id, 0.0) + 1.0 / (k + rank)
    return out


def _truncate(s: str, n: int = SNIPPET_MAX_CHARS) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


# ─── Engine ─────────────────────────────────────────────────────────────────


class RetrievalEngine:
    """Loaded once per process via `get_engine()`."""

    def __init__(self) -> None:
        self.bm25_p = self._load_bm25(BM25_DIR / "profiles_index", "profiles")
        self.bm25_j = self._load_bm25(BM25_DIR / "jobs_index", "jobs")
        self.dense_p = _DenseIndex(DENSE_DIR / "profiles.npy", DENSE_DIR / "profiles_ids.json")
        self.dense_j = _DenseIndex(DENSE_DIR / "jobs.npy", DENSE_DIR / "jobs_ids.json")
        self.skill_map = self._load_canonical_skill_map()
        self.entity_meta = self._load_entity_meta()
        log.info(
            "retrieval_engine_ready",
            profiles=len(self.dense_p.doc_ids),
            jobs=len(self.dense_j.doc_ids),
            canonical_skills=len(self.skill_map),
        )

    @staticmethod
    def _load_bm25(path: Path, label: str) -> bm25s.BM25:
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 index missing at {path} — run scripts/02_bm25_baseline.py."
            )
        return bm25s.BM25.load(str(path), load_corpus=True)

    @staticmethod
    def _load_canonical_skill_map() -> dict[str, str]:
        if not SKILLS_PARQUET.exists():
            raise FileNotFoundError(
                f"{SKILLS_PARQUET} missing — run scripts/06_skill_canonicalize.py."
            )
        import pandas as pd

        df = pd.read_parquet(SKILLS_PARQUET)
        return dict(zip(df["legacy_id"], df["canonical_id"], strict=True))

    @staticmethod
    def _load_entity_meta() -> dict[str, dict[str, Any]]:
        """One pass over DuckDB: doc_id → {kind, fields, snippet}."""
        if not DUCKDB_PATH.exists():
            raise FileNotFoundError(f"{DUCKDB_PATH} missing — run scripts/01_etl.py.")
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        meta: dict[str, dict[str, Any]] = {}
        for row in con.execute(
            "SELECT id, name, years_experience, skill_summary, canonical_text FROM profiles"
        ).fetchall():
            meta[row[0]] = {
                "kind": "person",
                "name": row[1],
                "years_experience": float(row[2] or 0.0),
                "snippet": _truncate(row[3] or row[4]),
                "canonical_text": row[4],
            }
        for row in con.execute(
            "SELECT id, title, designation, industry, city, country, "
            "experience_lower, experience_upper, canonical_text "
            "FROM jobs"
        ).fetchall():
            meta[row[0]] = {
                "kind": "job",
                "title": row[1],
                "designation": row[2],
                "industry": row[3],
                "city": row[4],
                "country": row[5],
                "experience_lower": int(row[6] or 0),
                "experience_upper": int(row[7] or 0),
                "snippet": _truncate(row[8]),
                "canonical_text": row[8],
            }
        con.close()
        return meta

    # ─── Query-time helpers ─────────────────────────────────────────────

    def _query_canonical_skills(self, query_skills: list[str]) -> list[str]:
        out: list[str] = []
        for name in query_skills or []:
            canonical = self.skill_map.get(_slug(name))
            if canonical is not None:
                out.append(canonical)
        return out

    def _kg_search(self, target: TargetSide, skill_ids: list[str]) -> dict[str, float]:
        if not skill_ids:
            return {}
        if target == TargetSide.candidates:
            rows = search_persons_by_skills(skill_ids, top_k=CANDIDATE_BUDGET)
        else:
            rows = search_jobs_by_skills(skill_ids, top_k=CANDIDATE_BUDGET)
        return {r["doc_id"]: float(r["score"]) for r in rows}

    def _build_result(
        self,
        rank: int,
        doc_id: str,
        final_score: float,
        stage_scores: StageScores,
        intent: IntentResult,
    ) -> SearchResult:
        meta = self.entity_meta.get(doc_id, {"kind": "person"})
        # Derive matched_skills: intersection of intent.query_skills (canonical) with
        # this entity's canonical skills. We keep raw_name display for readability.
        matched_canonical = set(self._query_canonical_skills(intent.query_skills))
        # entity_meta doesn't carry skill list; cheap re-lookup against canonical_text.
        # For a more accurate version we'd join against the parquet — defer to /profile/{id}.
        text = meta.get("canonical_text", "") or ""
        matched_display = (
            [s for s in (intent.query_skills or []) if s.lower() in text.lower()]
            if matched_canonical
            else []
        )

        entity = SearchResultEntity(
            kind=meta["kind"],
            id=doc_id,
            name=meta.get("name"),
            designation=meta.get("designation"),
            title=meta.get("title"),
            industry=meta.get("industry"),
            city=meta.get("city"),
            country=meta.get("country"),
            years_experience=meta.get("years_experience"),
            matched_skills=matched_display,
            snippet=meta.get("snippet", ""),
        )
        return SearchResult(
            rank=rank,
            score=final_score,
            entity=entity,
            stage_scores=stage_scores,
        )

    # ─── Public sync entrypoint ─────────────────────────────────────────

    def search(
        self,
        query: str,
        intent: IntentResult,
        *,
        pipeline: PipelineMode = PipelineMode.rrf3_rerank,
        top_k: int = 10,
    ) -> SearchResponse:
        timings = TimingMs()
        target = intent.target_side
        bm25_idx = self.bm25_p if target == TargetSide.candidates else self.bm25_j
        dense_idx = self.dense_p if target == TargetSide.candidates else self.dense_j

        # ── Encode query for dense / rerank ────────────────────────────
        # Use intent.free_text_intent for dense; raw query for rerank context.
        from app.services.dense import encode_dense

        t = time.perf_counter()
        qvec = encode_dense(
            [intent.free_text_intent or query],
            batch_size=1,
            max_length=512,
            show_progress=False,
        )[0]
        timings.encode = (time.perf_counter() - t) * 1000

        # ── Stage 1A: BM25 ─────────────────────────────────────────────
        bm25_run: dict[str, float] = {}
        if pipeline in {
            PipelineMode.bm25,
            PipelineMode.rrf,
            PipelineMode.rrf3,
            PipelineMode.rrf3_rerank,
        }:
            t = time.perf_counter()
            bm25_run = _bm25_search(bm25_idx, query)
            timings.bm25 = (time.perf_counter() - t) * 1000

        # ── Stage 1B: Dense ────────────────────────────────────────────
        dense_run: dict[str, float] = {}
        if pipeline in {
            PipelineMode.dense,
            PipelineMode.rrf,
            PipelineMode.rrf3,
            PipelineMode.rrf3_rerank,
        }:
            t = time.perf_counter()
            dense_run = dense_idx.search(qvec)
            timings.dense = (time.perf_counter() - t) * 1000

        # ── Stage 1C: KG ───────────────────────────────────────────────
        kg_run: dict[str, float] = {}
        if pipeline in {PipelineMode.rrf3, PipelineMode.rrf3_rerank}:
            t = time.perf_counter()
            sids = self._query_canonical_skills(intent.query_skills)
            kg_run = self._kg_search(target, sids) if sids else {}
            timings.kg = (time.perf_counter() - t) * 1000

        # ── Stage 2: RRF fuse ──────────────────────────────────────────
        if pipeline == PipelineMode.bm25:
            fused = bm25_run
        elif pipeline == PipelineMode.dense:
            fused = dense_run
        else:
            t = time.perf_counter()
            channels = [r for r in (bm25_run, dense_run, kg_run) if r]
            fused = _rrf_fuse(channels) if channels else {}
            timings.rrf = (time.perf_counter() - t) * 1000

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:CANDIDATE_BUDGET]

        # ── Stage 3: cross-encoder rerank ──────────────────────────────
        rerank_scores: dict[str, float] = {}
        if pipeline == PipelineMode.rrf3_rerank and ranked:
            from app.services.rerank import rerank_pairs

            t = time.perf_counter()
            head = ranked[:RERANK_TOP_N]
            pairs = [
                (query, self.entity_meta.get(d, {}).get("canonical_text", "")) for d, _ in head
            ]
            scores = rerank_pairs(pairs, batch_size=32, normalize=True)
            for (doc_id, _), s in zip(head, scores, strict=True):
                rerank_scores[doc_id] = float(s)
            timings.rerank = (time.perf_counter() - t) * 1000

            # Rebuild ranked list: rerank head followed by tail at preserved order
            head_resorted = sorted(head, key=lambda kv: rerank_scores[kv[0]], reverse=True)
            ranked = head_resorted + ranked[RERANK_TOP_N:]

        # ── Build typed results ────────────────────────────────────────
        results: list[SearchResult] = []
        for rank, (doc_id, fused_score) in enumerate(ranked[:top_k], start=1):
            stage_scores = StageScores(
                bm25=bm25_run.get(doc_id),
                dense=dense_run.get(doc_id),
                kg=kg_run.get(doc_id),
                rrf=fused.get(doc_id),
                rerank=rerank_scores.get(doc_id),
            )
            final = rerank_scores.get(doc_id, fused_score)
            results.append(self._build_result(rank, doc_id, final, stage_scores, intent))

        timings.total = (
            timings.intent
            + timings.encode
            + timings.bm25
            + timings.dense
            + timings.kg
            + timings.rrf
            + timings.rerank
        )

        return SearchResponse(
            query=query,
            pipeline=pipeline,
            intent=intent,
            n_candidates_retrieved=len(fused),
            results=results,
            timings_ms=timings,
        )

    # ─── Streaming entrypoint ───────────────────────────────────────────

    async def search_stream(
        self,
        query: str,
        intent: IntentResult,
        *,
        pipeline: PipelineMode = PipelineMode.rrf3_rerank,
        top_k: int = 10,
    ) -> AsyncIterator[StageEvent]:
        """Per-stage SSE events. Internally runs the sync pipeline in a thread to avoid blocking
        the event loop, and yields after each stage completes."""

        # Yield intent immediately
        yield StageEvent(type="intent", intent=intent)

        # Run the heavyweight pipeline in a thread
        loop = asyncio.get_running_loop()
        # We don't currently break the sync pipeline into per-stage callbacks (too much
        # plumbing for a v1). Instead we yield "stage_start" before scheduling, then
        # after the run yield "stage_done" events with timing pulled from TimingMs.
        yield StageEvent(type="stage_start", stage="all")
        t0 = time.perf_counter()
        response: SearchResponse = await loop.run_in_executor(
            None, lambda: self.search(query, intent, pipeline=pipeline, top_k=top_k)
        )
        elapsed = (time.perf_counter() - t0) * 1000

        # Emit per-stage timings as separate events for the client UI
        for stage_name, val in (
            ("encode", response.timings_ms.encode),
            ("bm25", response.timings_ms.bm25),
            ("dense", response.timings_ms.dense),
            ("kg", response.timings_ms.kg),
            ("rrf", response.timings_ms.rrf),
            ("rerank", response.timings_ms.rerank),
        ):
            if val > 0:
                yield StageEvent(type="stage_done", stage=stage_name, elapsed_ms=val)

        yield StageEvent(
            type="results",
            n_candidates=response.n_candidates_retrieved,
            elapsed_ms=elapsed,
            results=response.results,
        )
        yield StageEvent(type="done", elapsed_ms=elapsed)


# ─── Singleton ──────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def get_engine() -> RetrievalEngine:
    return RetrievalEngine()


# ─── Detail loaders for /v1/profile + /v1/job ──────────────────────────────


def get_profile_detail(person_id: str) -> ProfileDetail | None:
    if not DUCKDB_PATH.exists():
        return None
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        row = con.execute(
            "SELECT id, name, years_experience, skill_summary, potential_roles, canonical_text "
            "FROM profiles WHERE id = ?",
            [person_id],
        ).fetchone()
        if row is None:
            return None
        skill_rows = con.execute(
            "SELECT skill_id, skill_name, category, proficiency_label, proficiency "
            "FROM profile_skills WHERE person_id = ? "
            "ORDER BY proficiency DESC, skill_name ASC",
            [person_id],
        ).fetchall()
        return ProfileDetail(
            id=row[0],
            name=row[1],
            years_experience=float(row[2] or 0.0),
            skill_summary=row[3],
            potential_roles=list(row[4]) if row[4] is not None else [],
            skills=[
                ProfileSkill(
                    skill_id=s[0],
                    skill_name=s[1],
                    category=s[2],
                    proficiency_label=s[3],
                    proficiency=int(s[4]),
                )
                for s in skill_rows
            ],
            canonical_text=row[5] or "",
        )
    finally:
        con.close()


def get_job_detail(job_id: str) -> JobDetail | None:
    if not DUCKDB_PATH.exists():
        return None
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        row = con.execute(
            "SELECT id, source, title, designation, industry, city, country, "
            "experience_lower, experience_upper, raw_text, enhanced_text, canonical_text "
            "FROM jobs WHERE id = ?",
            [job_id],
        ).fetchone()
        if row is None:
            return None
        skill_rows = con.execute(
            "SELECT skill_id, skill_name, priority, category "
            "FROM job_skills WHERE job_id = ? ORDER BY priority ASC, skill_name ASC",
            [job_id],
        ).fetchall()
        return JobDetail(
            id=row[0],
            source=row[1],
            title=row[2],
            designation=row[3],
            industry=row[4],
            city=row[5],
            country=row[6],
            experience_lower=int(row[7] or 0),
            experience_upper=int(row[8] or 0),
            raw_text=row[9],
            enhanced_text=row[10],
            canonical_text=row[11] or "",
            skills=[
                JobSkill(
                    skill_id=s[0],
                    skill_name=s[1],
                    priority=s[2],
                    category=s[3],
                )
                for s in skill_rows
            ],
        )
    finally:
        con.close()
