"""Job — the demand side of the corpus.

Two source formats unified into one entity:
  - demands_data.csv (1,081 rows, structured: location / designation / skills lists)
  - jd_dataset.zip   (289 jobs, semi-structured: raw JSON + LLM-enhanced markdown)

We do NOT join across sources (no shared id). Each becomes a Job row with a
`source` discriminator and a per-source id namespace.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.models.skill import JobSkill

JobSource = Literal["demands_csv", "jd_zip"]


# ─── Source-1 wire types: demands_data.csv ───────────────────────────────────


class DemandRaw(BaseModel):
    """Wire-equivalent of one row in demands_data.csv."""

    id: str
    state: str | None = None
    city: str | None = None
    country: str | None = None
    job_title: str | None = None
    primary_skills: str | None = None
    secondary_skills: str | None = None
    experience_lower: int = 0
    experience_upper: int = 0
    designation: str | None = None


# ─── Source-2 wire types: jd_dataset.zip ─────────────────────────────────────


class JDRawPayload(BaseModel):
    """Parsed contents of `{folder_id}/raw_jd.txt` (the JSON file)."""

    industry: str | None = None
    raw_jd: str = ""


class JDEnhancedSkillRequirements(BaseModel):
    must_have: list[str] = Field(default_factory=list)
    good_to_have: list[str] = Field(default_factory=list)
    educational_qualifications: list[str] = Field(default_factory=list)


class JDEnhancedPayload(BaseModel):
    """Parsed contents of `{folder_id}/enhanced_job_description.md`."""

    job_title: str | None = None
    location_city: str | None = None
    location_country: str | None = None
    client_industry: str | None = None
    detailed_responsibilities: list[str] = Field(default_factory=list)
    skill_requirements: JDEnhancedSkillRequirements = Field(
        default_factory=JDEnhancedSkillRequirements
    )
    other_requirements: str | None = None


# ─── Unified Job entity (the indexed shape) ──────────────────────────────────


class Job(BaseModel):
    """Parsed job entity — what gets indexed in BM25 / Qdrant / Neo4j."""

    id: str = Field(
        ...,
        description="Namespaced: 'job_demand_<demand_id>' or 'job_jd_<folder_id>'.",
    )
    source: JobSource

    title: str | None = None
    designation: str | None = None
    industry: str | None = None
    city: str | None = None
    country: str | None = None
    experience_lower: int = 0
    experience_upper: int = 0

    skills: list[JobSkill] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    educational_qualifications: list[str] = Field(default_factory=list)
    other_requirements: str | None = None

    raw_text: str | None = Field(
        default=None,
        description="Original JD prose (raw_jd.txt), if from jd_zip.",
    )
    enhanced_text: str | None = Field(
        default=None,
        description="LLM-enhanced markdown (enhanced_job_description.md), if from jd_zip.",
    )
    canonical_text: str = Field(
        ...,
        description="Joined text used as the BM25 + BGE-M3 index input.",
    )
