"""Skill — the join entity between Person and Job.

Skills are canonicalised once at ETL time (alias YAML → exact match → ESCO API
lookup → embedding-cosine fallback) and shared across both sides. Person has
HAS_SKILL with proficiency; Job has REQUIRES_SKILL with priority.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ProficiencyLabel = Literal[
    "Beginner",
    "Advanced Beginner",
    "Competent",
    "Proficient",
    "Expert",
]
ProficiencyOrdinal = Literal[1, 2, 3, 4, 5]
SkillCategory = Literal["core", "secondary", "soft"]
SkillPriority = Literal["must_have", "good_to_have"]
SkillSource = Literal["resume", "jd_must", "jd_nice", "demand_primary", "demand_secondary"]

# Dreyfus 5-stage skill-acquisition model (Beginner → Expert).
# Confirmed on the live profiles.csv (10,483 / 8,977 / 3,871 / 3,524 / 3,356 occurrences).
PROFICIENCY_ORDINAL: dict[ProficiencyLabel, ProficiencyOrdinal] = {
    "Beginner": 1,
    "Advanced Beginner": 2,
    "Competent": 3,
    "Proficient": 4,
    "Expert": 5,
}


class Skill(BaseModel):
    """Canonical skill node. One per de-duplicated, ESCO-mapped skill name."""

    id: str = Field(..., description="Slug: e.g. 'python', 'azure-devops', 'regulatory-affairs'.")
    name: str = Field(..., description="Display name, title-cased.")
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative spellings / abbreviations seen in source data.",
    )
    esco_uri: str | None = Field(
        default=None,
        description="ESCO concept URI when canonicalised against the EU ESCO ontology.",
    )
    kind: Literal["tech", "soft", "domain", "tool"] = "tech"


class PersonSkill(BaseModel):
    """Edge payload for (Person)-[:HAS_SKILL]->(Skill)."""

    skill_id: str
    category: SkillCategory
    proficiency_label: ProficiencyLabel
    proficiency: ProficiencyOrdinal
    source: SkillSource = "resume"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_chunk_id: str | None = None


class JobSkill(BaseModel):
    """Edge payload for (Job)-[:REQUIRES_SKILL]->(Skill)."""

    skill_id: str
    priority: SkillPriority
    category: Literal["primary", "secondary", "must_have", "good_to_have"]
    source: SkillSource
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_chunk_id: str | None = None
