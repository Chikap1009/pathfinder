"""Pydantic schemas for the v1 API surface.

These types are the contract between FastAPI and the Next.js client. The
client gets typed via `pnpm --filter web run openapi:gen` which reads
http://localhost:8000/openapi.json and emits TypeScript types.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

# ─── Common enums ────────────────────────────────────────────────────────────


class TargetSide(StrEnum):
    candidates = "candidates"
    jobs = "jobs"


class PipelineMode(StrEnum):
    bm25 = "bm25"
    dense = "dense"
    rrf = "rrf"
    rrf3 = "rrf3"
    rrf3_rerank = "rrf3_rerank"


# ─── Intent router output ────────────────────────────────────────────────────


class IntentResult(BaseModel):
    """Structured-output schema for the intent classifier."""

    target_side: TargetSide = Field(
        ...,
        description=(
            "Whether the query is searching for candidates (people) or for "
            "jobs/roles. If ambiguous, prefer 'candidates' for verbs like "
            "'find', 'who knows', 'looking for', and 'jobs' for verbs like "
            "'roles', 'positions', 'openings'."
        ),
    )
    query_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Atomic skill names mentioned in the query. Title-cased, "
            "deduplicated. Examples: 'Python', 'Selenium', 'Risk Management'. "
            "Skip generic words like 'engineer', 'developer', 'role'."
        ),
    )
    min_proficiency: Literal[
        "Beginner", "Advanced Beginner", "Competent", "Proficient", "Expert", "Any"
    ] = Field(
        default="Any",
        description=(
            "Minimum proficiency level requested in the query (Dreyfus 5-stage). "
            "Use 'Any' if not specified or unclear."
        ),
    )
    designation_hint: str | None = Field(
        default=None,
        description=(
            "If the query names a specific role or designation (e.g. 'Test "
            "Manager', 'Senior Java Developer'), capture it verbatim. Else null."
        ),
    )
    free_text_intent: str = Field(
        ...,
        description=(
            "The residual semantic chunk after stripping skills + designation. "
            "Used as the dense-retrieval query. Can be a paraphrased one-liner."
        ),
    )


# ─── Search request / response ───────────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000, description="Free-text user query.")
    target_side: TargetSide | None = Field(
        default=None,
        description=(
            "Force the side to search (candidates/jobs). If null, the intent classifier picks one."
        ),
    )
    pipeline: PipelineMode = Field(
        default=PipelineMode.rrf3_rerank,
        description="Which retrieval pipeline configuration to run.",
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return.")


class StageScores(BaseModel):
    """Per-stage retrieval scores for one document — used in the explanation drawer."""

    bm25: float | None = None
    dense: float | None = None
    kg: float | None = None
    rrf: float | None = None
    rerank: float | None = None


class SearchResultEntity(BaseModel):
    """Type-discriminated entity payload (Person OR Job)."""

    kind: Literal["person", "job"]
    id: str
    name: str | None = None
    designation: str | None = None
    title: str | None = None
    industry: str | None = None
    city: str | None = None
    country: str | None = None
    years_experience: float | None = None
    matched_skills: list[str] = Field(default_factory=list)
    snippet: str = Field(default="", description="Truncated canonical_text for preview.")


class SearchResult(BaseModel):
    rank: int
    score: float
    entity: SearchResultEntity
    stage_scores: StageScores


class TimingMs(BaseModel):
    """End-to-end stage timings in milliseconds."""

    intent: float = 0.0
    encode: float = 0.0
    bm25: float = 0.0
    dense: float = 0.0
    kg: float = 0.0
    rrf: float = 0.0
    rerank: float = 0.0
    total: float = 0.0


class SearchResponse(BaseModel):
    query: str
    pipeline: PipelineMode
    intent: IntentResult
    n_candidates_retrieved: int
    results: list[SearchResult]
    timings_ms: TimingMs


# ─── Streaming SSE event types ───────────────────────────────────────────────


class StageEvent(BaseModel):
    """A single SSE chunk emitted during /v1/search/stream."""

    type: Literal[
        "intent",
        "stage_start",
        "stage_done",
        "results",
        "error",
        "done",
    ]
    stage: Literal["intent", "encode", "bm25", "dense", "kg", "rrf", "rerank", "all"] | None = None
    n_candidates: int | None = None
    elapsed_ms: float | None = None
    intent: IntentResult | None = None
    results: list[SearchResult] | None = None
    message: str | None = None


# ─── Profile / Job detail (for the explanation drawer) ───────────────────────


class ProfileSkill(BaseModel):
    skill_id: str
    skill_name: str
    category: Literal["core", "secondary", "soft"]
    proficiency_label: str
    proficiency: int  # 1..5 ordinal


class ProfileDetail(BaseModel):
    id: str
    name: str | None = None
    years_experience: float = 0.0
    skill_summary: str | None = None
    potential_roles: list[str] = Field(default_factory=list)
    skills: list[ProfileSkill] = Field(default_factory=list)
    canonical_text: str = ""


class JobSkill(BaseModel):
    skill_id: str
    skill_name: str
    priority: Literal["must_have", "good_to_have"]
    category: str  # primary | secondary | must_have | good_to_have


class JobDetail(BaseModel):
    id: str
    source: Literal["demands_csv", "jd_zip"]
    title: str | None = None
    designation: str | None = None
    industry: str | None = None
    city: str | None = None
    country: str | None = None
    experience_lower: int = 0
    experience_upper: int = 0
    skills: list[JobSkill] = Field(default_factory=list)
    raw_text: str | None = None
    enhanced_text: str | None = None
    canonical_text: str = ""


# ─── Evaluation summary (the /eval dashboard) ────────────────────────────────


class CorpusStats(BaseModel):
    profiles: int
    jobs: int
    canonical_skills: int
    has_skill_edges: int
    requires_skill_edges: int


class KGStats(BaseModel):
    persons: int
    jobs: int
    skills: int
    roles: int
    designations: int
    industries: int
    locations: int
    has_skill: int
    requires_skill: int
    can_fill: int
    is_designation: int
    at_location: int
    in_industry: int

    @property
    def total_nodes(self) -> int:
        return (
            self.persons
            + self.jobs
            + self.skills
            + self.roles
            + self.designations
            + self.industries
            + self.locations
        )


class EvalSetStats(BaseModel):
    """Composition of the deterministic eval set."""

    total_queries: int
    candidate_search: int
    job_search: int
    paraphrase_stratum: int
    seed: int = 42


class AblationRow(BaseModel):
    """One configuration's metrics on one stratum."""

    name: str
    pipeline: Literal[
        "bm25", "dense", "rrf", "rerank25", "rerank50", "kg_only", "rrf3", "rrf3_rerank"
    ]
    stratum: Literal["overall", "original", "paraphrase", "candidate_search", "job_search"]
    ndcg_at_10: float | None = None
    recall_at_10: float | None = None
    recall_at_100: float | None = None
    mrr_at_10: float | None = None
    map_score: float | None = None
    latency_ms: float | None = None
    is_best_for_stratum: bool = False


class LatencyStage(BaseModel):
    stage: str
    ms_per_query: float
    notes: str | None = None


class EvalTargets(BaseModel):
    recall_at_100: float = 0.97
    ndcg_at_10: float = 0.55
    mrr_at_10: float = 0.65
    ragas_faithfulness: float = 0.95
    p95_latency_ms: float = 2000


class EvalSummary(BaseModel):
    """Snapshot of the entire offline-eval state. Powers the /eval dashboard."""

    corpus: CorpusStats
    kg: KGStats
    eval_set: EvalSetStats
    ablation: list[AblationRow]
    latency_budget: list[LatencyStage]
    targets: EvalTargets
    notes: list[str] = Field(default_factory=list)
