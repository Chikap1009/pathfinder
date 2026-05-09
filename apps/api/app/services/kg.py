"""Neo4j knowledge-graph service wrapper.

Lazy-loaded singleton driver, helper for sync (ingest scripts) and async (FastAPI
request path) usage. Exposes the canonical Cypher templates used by:
  - `scripts/07_kg_build.py` — entity + edge ingest from DuckDB
  - `scripts/08_kg_baseline.py` — KG retrieval channel for the ablation row
  - `app/api/v1/graph.py` (Week 4) — KG-backed search endpoints

Schema: see ADR-0003 (`docs/decisions/0003-two-sided-corpus.md`).
"""

from __future__ import annotations

import functools
from typing import Any

from app.core.logging import get_logger
from app.core.settings import get_settings

log = get_logger(__name__)


# ─── Driver singletons ──────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def get_driver_sync() -> Any:
    """Sync Neo4j driver — for batch ingest scripts."""
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError(
            "neo4j driver missing. Run `uv sync --extra graph` from apps/api/."
        ) from exc

    s = get_settings()
    drv = GraphDatabase.driver(
        s.neo4j_uri,
        auth=(s.neo4j_user, s.neo4j_password.get_secret_value()),
    )
    log.info("neo4j_driver_ready", uri=s.neo4j_uri, mode="sync")
    return drv


@functools.lru_cache(maxsize=1)
def get_driver_async() -> Any:
    """Async Neo4j driver — for FastAPI request handlers."""
    try:
        from neo4j import AsyncGraphDatabase
    except ImportError as exc:
        raise RuntimeError(
            "neo4j driver missing. Run `uv sync --extra graph` from apps/api/."
        ) from exc

    s = get_settings()
    drv = AsyncGraphDatabase.driver(
        s.neo4j_uri,
        auth=(s.neo4j_user, s.neo4j_password.get_secret_value()),
    )
    log.info("neo4j_driver_ready", uri=s.neo4j_uri, mode="async")
    return drv


# ─── Schema constraints + indexes ────────────────────────────────────────────

CONSTRAINTS_CYPHER: list[str] = [
    "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT job_id IF NOT EXISTS FOR (j:Job) REQUIRE j.id IS UNIQUE",
    "CREATE CONSTRAINT skill_id IF NOT EXISTS FOR (s:Skill) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT role_id IF NOT EXISTS FOR (r:Role) REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT designation_name IF NOT EXISTS FOR (d:Designation) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE",
    "CREATE CONSTRAINT industry_name IF NOT EXISTS FOR (i:Industry) REQUIRE i.name IS UNIQUE",
]

INDEXES_CYPHER: list[str] = [
    "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)",
    "CREATE INDEX job_designation IF NOT EXISTS FOR (j:Job) ON (j.designation)",
    "CREATE INDEX job_industry IF NOT EXISTS FOR (j:Job) ON (j.industry)",
    "CREATE INDEX skill_name IF NOT EXISTS FOR (s:Skill) ON (s.name)",
    "CREATE INDEX has_skill_proficiency IF NOT EXISTS FOR ()-[r:HAS_SKILL]-() ON (r.proficiency)",
    "CREATE INDEX has_skill_category IF NOT EXISTS FOR ()-[r:HAS_SKILL]-() ON (r.category)",
    "CREATE INDEX requires_skill_priority IF NOT EXISTS FOR ()-[r:REQUIRES_SKILL]-() ON (r.priority)",
]


def init_schema() -> None:
    """Idempotent. Apply all schema constraints + indexes."""
    drv = get_driver_sync()
    s = get_settings()
    with drv.session(database=s.neo4j_database) as sess:
        for stmt in CONSTRAINTS_CYPHER + INDEXES_CYPHER:
            sess.run(stmt)
    log.info(
        "kg_schema_initialised",
        n_constraints=len(CONSTRAINTS_CYPHER),
        n_indexes=len(INDEXES_CYPHER),
    )


def reset_database() -> None:
    """Drops EVERYTHING. Used for fresh re-ingest. Idempotent."""
    drv = get_driver_sync()
    s = get_settings()
    with drv.session(database=s.neo4j_database) as sess:
        sess.run("MATCH (n) DETACH DELETE n")
    log.warning("kg_database_wiped")


def health_check() -> dict[str, Any]:
    """Quick `RETURN 1` round-trip for /health endpoints."""
    drv = get_driver_sync()
    s = get_settings()
    with drv.session(database=s.neo4j_database) as sess:
        ok = sess.run("RETURN 1 AS ok").single()["ok"]
        n_nodes = sess.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        n_rels = sess.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    return {"ok": ok == 1, "nodes": int(n_nodes), "relationships": int(n_rels)}


# ─── Entity ingest helpers (sync, batched) ───────────────────────────────────

# UNWIND-style batch upserts: pass a list of dicts as `$rows` for max throughput.
# Aim for batches of 1k-5k rows; Neo4j's bolt protocol handles them well.

UPSERT_PERSONS = """
UNWIND $rows AS row
MERGE (p:Person {id: row.id})
SET p.name = row.name,
    p.years_experience = row.years_experience,
    p.canonical_text = row.canonical_text
"""

UPSERT_JOBS = """
UNWIND $rows AS row
MERGE (j:Job {id: row.id})
SET j.title = row.title,
    j.designation = row.designation,
    j.industry = row.industry,
    j.experience_lower = row.experience_lower,
    j.experience_upper = row.experience_upper,
    j.source = row.source,
    j.canonical_text = row.canonical_text
"""

UPSERT_SKILLS = """
UNWIND $rows AS row
MERGE (s:Skill {id: row.id})
SET s.name = row.name,
    s.source = row.source,
    s.esco_uri = row.esco_uri,
    s.confidence = row.confidence
"""

UPSERT_ROLES = """
UNWIND $rows AS row
MERGE (r:Role {id: row.id})
SET r.title = row.title
"""

UPSERT_DESIGNATIONS = """
UNWIND $rows AS row
MERGE (d:Designation {name: row.name})
"""

UPSERT_LOCATIONS = """
UNWIND $rows AS row
MERGE (l:Location {id: row.id})
SET l.city = row.city, l.country = row.country
"""

UPSERT_INDUSTRIES = """
UNWIND $rows AS row
MERGE (i:Industry {name: row.name})
"""


# ─── Edge ingest helpers ─────────────────────────────────────────────────────

UPSERT_HAS_SKILL = """
UNWIND $rows AS row
MATCH (p:Person {id: row.person_id})
MATCH (s:Skill  {id: row.skill_id})
MERGE (p)-[r:HAS_SKILL {category: row.category}]->(s)
SET r.proficiency = row.proficiency,
    r.proficiency_label = row.proficiency_label,
    r.source = row.source
"""

UPSERT_REQUIRES_SKILL = """
UNWIND $rows AS row
MATCH (j:Job   {id: row.job_id})
MATCH (s:Skill {id: row.skill_id})
MERGE (j)-[r:REQUIRES_SKILL {priority: row.priority}]->(s)
SET r.category = row.category,
    r.source = row.source
"""

UPSERT_CAN_FILL = """
UNWIND $rows AS row
MATCH (p:Person {id: row.person_id})
MATCH (r:Role   {id: row.role_id})
MERGE (p)-[e:CAN_FILL {rank: row.rank}]->(r)
"""

UPSERT_IS_DESIGNATION = """
UNWIND $rows AS row
MATCH (j:Job        {id: row.job_id})
MATCH (d:Designation {name: row.designation})
MERGE (j)-[:IS_DESIGNATION]->(d)
"""

UPSERT_AT_LOCATION = """
UNWIND $rows AS row
MATCH (j:Job      {id: row.job_id})
MATCH (l:Location {id: row.location_id})
MERGE (j)-[:AT_LOCATION]->(l)
"""

UPSERT_IN_INDUSTRY = """
UNWIND $rows AS row
MATCH (j:Job      {id: row.job_id})
MATCH (i:Industry {name: row.industry})
MERGE (j)-[:IN_INDUSTRY]->(i)
"""


# ─── Retrieval Cypher templates (used by 08_kg_baseline.py + Week 4 endpoints) ──

# Given a list of canonical skill IDs (from the query), return the top-K Persons
# ranked by total proficiency-weighted skill overlap.
PERSON_BY_SKILL_OVERLAP = """
UNWIND $skill_ids AS sid
MATCH (p:Person)-[r:HAS_SKILL]->(s:Skill {id: sid})
WITH p, s, r,
     CASE r.category WHEN 'core' THEN 3.0
                     WHEN 'secondary' THEN 1.5
                     WHEN 'soft' THEN 0.5
                     ELSE 1.0 END AS cat_weight
WITH p,
     sum(toFloat(coalesce(r.proficiency, 1)) * cat_weight) AS score,
     count(DISTINCT s) AS skill_overlap
WHERE skill_overlap >= $min_overlap
RETURN p.id AS doc_id, score, skill_overlap
ORDER BY score DESC, skill_overlap DESC
LIMIT $top_k
"""

# Given a list of canonical skill IDs, return the top-K Jobs ranked by total
# priority-weighted skill overlap.
JOB_BY_SKILL_OVERLAP = """
UNWIND $skill_ids AS sid
MATCH (j:Job)-[r:REQUIRES_SKILL]->(s:Skill {id: sid})
WITH j, s, r,
     CASE r.priority WHEN 'must_have' THEN 3.0
                     WHEN 'good_to_have' THEN 1.0
                     ELSE 1.0 END AS prio_weight
WITH j, sum(prio_weight) AS score, count(DISTINCT s) AS skill_overlap
WHERE skill_overlap >= $min_overlap
RETURN j.id AS doc_id, score, skill_overlap
ORDER BY score DESC, skill_overlap DESC
LIMIT $top_k
"""


def search_persons_by_skills(
    skill_ids: list[str],
    *,
    top_k: int = 100,
    min_overlap: int = 1,
) -> list[dict[str, Any]]:
    """Synchronous wrapper for `PERSON_BY_SKILL_OVERLAP`."""
    drv = get_driver_sync()
    s = get_settings()
    with drv.session(database=s.neo4j_database) as sess:
        result = sess.run(
            PERSON_BY_SKILL_OVERLAP,
            skill_ids=skill_ids,
            top_k=top_k,
            min_overlap=min_overlap,
        )
        return [dict(r) for r in result]


def search_jobs_by_skills(
    skill_ids: list[str],
    *,
    top_k: int = 100,
    min_overlap: int = 1,
) -> list[dict[str, Any]]:
    drv = get_driver_sync()
    s = get_settings()
    with drv.session(database=s.neo4j_database) as sess:
        result = sess.run(
            JOB_BY_SKILL_OVERLAP,
            skill_ids=skill_ids,
            top_k=top_k,
            min_overlap=min_overlap,
        )
        return [dict(r) for r in result]
