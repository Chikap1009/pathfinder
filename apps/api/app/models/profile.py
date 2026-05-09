"""Profile — the candidate side of the corpus.

Mirrors `app/core/schema_maps/profiles.yaml` (1,782-row CSV with 3-way skill
categorisation and Dreyfus 5-stage proficiency tags).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.skill import PersonSkill


class ProfileRaw(BaseModel):
    """Wire-equivalent of one row in profiles.csv after normalisation.

    Fields here mirror CSV columns one-for-one; downstream code converts to the
    richer `Profile` (parsed skills, computed canonical text).
    """

    id: int
    name: str | None = None
    core_skills: str | None = None  # raw "Python (Advanced Beginner), SQL (Beginner)"
    secondary_skills: str | None = None
    soft_skills: str | None = None
    years_of_experience: float = 0.0
    potential_roles: str | None = None  # comma-separated role titles
    skill_summary: str | None = None  # ~70-word prose


class Profile(BaseModel):
    """Parsed profile entity — what gets indexed in BM25 / Qdrant / Neo4j."""

    id: str = Field(..., description="Namespaced: 'person_<csv_id>' to avoid Job collisions.")
    raw_id: int
    name: str | None = None
    years_experience: float = 0.0
    skills: list[PersonSkill] = Field(default_factory=list)
    potential_roles: list[str] = Field(default_factory=list)
    skill_summary: str | None = None
    canonical_text: str = Field(
        ...,
        description="Joined text used as the BM25 + BGE-M3 index input.",
    )
