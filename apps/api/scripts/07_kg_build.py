"""KG ingest — DuckDB → Neo4j. Idempotent.

Loads Persons, Jobs, Skills, Roles, Designations, Locations, Industries and
their edges (HAS_SKILL, REQUIRES_SKILL, CAN_FILL, IS_DESIGNATION, AT_LOCATION,
IN_INDUSTRY) into Neo4j 5 from the DuckDB tables produced by `01_etl.py` and
the canonical skill mapping produced by `06_skill_canonicalize.py`.

Idempotent. Run with:
    uv run python scripts/07_kg_build.py            # full ingest (preserves existing data)
    uv run python scripts/07_kg_build.py --reset    # WIPE + re-ingest (use after schema changes)
    uv run python scripts/07_kg_build.py --dry-run  # validate inputs without writing

Outputs:
    Neo4j has the schema initialised, all entities + edges loaded, and a
    canonical skill table joining the legacy IDs from the ETL with the
    canonical IDs from the canonicalisation pipeline.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import duckdb
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from app.services.kg import (
    UPSERT_AT_LOCATION,
    UPSERT_CAN_FILL,
    UPSERT_DESIGNATIONS,
    UPSERT_HAS_SKILL,
    UPSERT_IN_INDUSTRY,
    UPSERT_INDUSTRIES,
    UPSERT_IS_DESIGNATION,
    UPSERT_JOBS,
    UPSERT_LOCATIONS,
    UPSERT_PERSONS,
    UPSERT_REQUIRES_SKILL,
    UPSERT_ROLES,
    UPSERT_SKILLS,
    get_driver_sync,
    health_check,
    init_schema,
    reset_database,
)
from app.services.skills import _slug


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pnpm-workspace.yaml").exists():
            return parent
    raise RuntimeError("repo root marker not found")


REPO_ROOT = _find_repo_root()
DUCKDB_PATH = REPO_ROOT / "data" / "processed" / "pathfinder.duckdb"
SKILLS_PARQUET = REPO_ROOT / "data" / "processed" / "skills" / "canonical.parquet"

BATCH_SIZE = 2_000
console = Console()
app = typer.Typer(add_completion=False, help=__doc__)


# ─── Helpers ────────────────────────────────────────────────────────────────


def _batched(rows: list[dict], size: int = BATCH_SIZE):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def _run_batched(cypher: str, rows: list[dict], label: str) -> int:
    """Run a Cypher UNWIND statement in batches; return rows ingested."""
    if not rows:
        console.print(f"  [dim]{label}:[/] (no rows)")
        return 0
    drv = get_driver_sync()
    n = 0
    t0 = time.perf_counter()
    with drv.session() as sess:
        for batch in _batched(rows):
            sess.run(cypher, rows=batch)
            n += len(batch)
    elapsed = time.perf_counter() - t0
    console.print(
        f"  [cyan]{label}:[/] {n:,} rows in {elapsed:.2f}s ({n / max(elapsed, 1e-3):,.0f} row/s)"
    )
    return n


def _load_skill_legacy_to_canonical() -> dict[str, str]:
    """Build legacy_skill_id → canonical_skill_id map from canonical.parquet."""
    if not SKILLS_PARQUET.exists():
        raise FileNotFoundError(
            f"{SKILLS_PARQUET} missing — run scripts/06_skill_canonicalize.py first."
        )
    df = pd.read_parquet(SKILLS_PARQUET)
    return dict(zip(df["legacy_id"], df["canonical_id"], strict=True))


# ─── Build payloads ─────────────────────────────────────────────────────────


def _build_payloads(con: duckdb.DuckDBPyConnection) -> dict[str, list[dict]]:
    """Read DuckDB views; project into UNWIND-ready payloads."""
    legacy_to_canon = _load_skill_legacy_to_canonical()

    # ── Skills: distinct canonical skills with metadata ─────────────────
    skills_df = pd.read_parquet(SKILLS_PARQUET)
    # Aggregate to canonical level (multiple raw names may map to one canonical).
    skills_canon = skills_df.groupby("canonical_id", as_index=False).agg(
        canonical_name=("canonical_name", "first"),
        source=("source", lambda v: ",".join(sorted(set(v)))),
        esco_uri=("esco_uri", "first"),
        confidence=("confidence", "max"),
    )
    skill_rows = [
        {
            "id": r["canonical_id"],
            "name": r["canonical_name"],
            "source": r["source"],
            "esco_uri": r["esco_uri"] if pd.notna(r["esco_uri"]) else None,
            "confidence": float(r["confidence"])
            if not (isinstance(r["confidence"], float) and math.isnan(r["confidence"]))
            else 0.0,
        }
        for _, r in skills_canon.iterrows()
    ]

    # ── Persons ──────────────────────────────────────────────────────────
    person_rows = [
        {
            "id": r[0],
            "name": r[1] if r[1] is not None else "",
            "years_experience": float(r[2] or 0.0),
            "canonical_text": r[3] or "",
        }
        for r in con.execute(
            "SELECT id, name, years_experience, canonical_text FROM profiles"
        ).fetchall()
    ]

    # ── Jobs ─────────────────────────────────────────────────────────────
    job_rows = [
        {
            "id": r[0],
            "title": r[1] if r[1] is not None else "",
            "designation": r[2] if r[2] is not None else "",
            "industry": r[3] if r[3] is not None else "",
            "experience_lower": int(r[4] or 0),
            "experience_upper": int(r[5] or 0),
            "source": r[6],
            "canonical_text": r[7] or "",
        }
        for r in con.execute(
            "SELECT id, title, designation, industry, experience_lower, "
            "experience_upper, source, canonical_text FROM jobs"
        ).fetchall()
    ]

    # ── Roles (from profiles.potential_roles) ────────────────────────────
    role_set: set[tuple[str, str]] = set()
    can_fill_rows: list[dict] = []
    for pid, roles in con.execute(
        "SELECT id, potential_roles FROM profiles WHERE potential_roles IS NOT NULL"
    ).fetchall():
        if roles is None:
            continue
        # Stored as a list[str] in the parquet; pandas may roundtrip it as np.ndarray.
        for rank, raw_title in enumerate(list(roles), start=1):
            title = str(raw_title).strip()
            if not title:
                continue
            rid = _slug(title)
            role_set.add((rid, title))
            can_fill_rows.append({"person_id": pid, "role_id": rid, "rank": rank})
    role_rows = [{"id": rid, "title": title} for rid, title in sorted(role_set)]

    # ── Designations (job-side) ──────────────────────────────────────────
    designation_set: set[str] = set()
    is_designation_rows: list[dict] = []
    for jid, desig in con.execute(
        "SELECT id, designation FROM jobs WHERE designation IS NOT NULL AND designation <> ''"
    ).fetchall():
        d = str(desig).strip()
        if not d:
            continue
        designation_set.add(d)
        is_designation_rows.append({"job_id": jid, "designation": d})
    designation_rows = [{"name": d} for d in sorted(designation_set)]

    # ── Industries (job-side) ────────────────────────────────────────────
    industry_set: set[str] = set()
    in_industry_rows: list[dict] = []
    for jid, ind in con.execute(
        "SELECT id, industry FROM jobs WHERE industry IS NOT NULL AND industry <> ''"
    ).fetchall():
        i = str(ind).strip()
        if not i:
            continue
        industry_set.add(i)
        in_industry_rows.append({"job_id": jid, "industry": i})
    industry_rows = [{"name": i} for i in sorted(industry_set)]

    # ── Locations (job-side, city + country) ─────────────────────────────
    location_set: set[tuple[str, str, str]] = set()  # (id, city, country)
    at_location_rows: list[dict] = []
    for jid, city, country in con.execute("SELECT id, city, country FROM jobs").fetchall():
        c = (city or "").strip() if city else ""
        co = (country or "").strip() if country else ""
        if not c and not co:
            continue
        loc_id = _slug(f"{c}-{co}")
        location_set.add((loc_id, c, co))
        at_location_rows.append({"job_id": jid, "location_id": loc_id})
    location_rows = [{"id": lid, "city": c, "country": co} for lid, c, co in sorted(location_set)]

    # ── HAS_SKILL (Person → Skill) ───────────────────────────────────────
    has_skill_rows: list[dict] = []
    missing_legacy = 0
    for r in con.execute(
        "SELECT person_id, skill_id, category, proficiency_label, proficiency, source "
        "FROM profile_skills"
    ).fetchall():
        legacy_sid = r[1]
        canonical_sid = legacy_to_canon.get(legacy_sid)
        if canonical_sid is None:
            missing_legacy += 1
            continue
        has_skill_rows.append(
            {
                "person_id": r[0],
                "skill_id": canonical_sid,
                "category": r[2],
                "proficiency_label": r[3],
                "proficiency": int(r[4]),
                "source": r[5],
            }
        )

    # ── REQUIRES_SKILL (Job → Skill) ─────────────────────────────────────
    requires_skill_rows: list[dict] = []
    for r in con.execute(
        "SELECT job_id, skill_id, priority, category, source FROM job_skills"
    ).fetchall():
        legacy_sid = r[1]
        canonical_sid = legacy_to_canon.get(legacy_sid)
        if canonical_sid is None:
            missing_legacy += 1
            continue
        requires_skill_rows.append(
            {
                "job_id": r[0],
                "skill_id": canonical_sid,
                "priority": r[2],
                "category": r[3],
                "source": r[4],
            }
        )

    if missing_legacy:
        console.print(
            f"  [yellow]warning:[/] {missing_legacy} skill rows had no canonical mapping "
            "— legacy_id absent from canonical.parquet (re-run 06_skill_canonicalize.py?)"
        )

    return {
        "skills": skill_rows,
        "persons": person_rows,
        "jobs": job_rows,
        "roles": role_rows,
        "designations": designation_rows,
        "industries": industry_rows,
        "locations": location_rows,
        "has_skill": has_skill_rows,
        "requires_skill": requires_skill_rows,
        "can_fill": can_fill_rows,
        "is_designation": is_designation_rows,
        "in_industry": in_industry_rows,
        "at_location": at_location_rows,
    }


# ─── Main ───────────────────────────────────────────────────────────────────


def run(reset: bool = False, dry_run: bool = False) -> None:
    if not DUCKDB_PATH.exists():
        raise FileNotFoundError(f"{DUCKDB_PATH} missing — run scripts/01_etl.py first.")

    console.rule("[bold magenta]KG ingest — DuckDB → Neo4j[/]")
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    payloads = _build_payloads(con)
    con.close()

    # Show what we're about to write
    t = Table(title="payloads", show_lines=False)
    t.add_column("collection", style="cyan")
    t.add_column("rows", justify="right")
    for k, v in payloads.items():
        t.add_row(k, f"{len(v):,}")
    console.print(t)
    if dry_run:
        console.print("[yellow]--dry-run set; not writing to Neo4j.[/]")
        return

    # Connect, schema, optionally wipe
    if reset:
        reset_database()
    init_schema()

    # Nodes first (constraints rely on .id), then edges.
    _run_batched(UPSERT_SKILLS, payloads["skills"], "skills")
    _run_batched(UPSERT_PERSONS, payloads["persons"], "persons")
    _run_batched(UPSERT_JOBS, payloads["jobs"], "jobs")
    _run_batched(UPSERT_ROLES, payloads["roles"], "roles")
    _run_batched(UPSERT_DESIGNATIONS, payloads["designations"], "designations")
    _run_batched(UPSERT_INDUSTRIES, payloads["industries"], "industries")
    _run_batched(UPSERT_LOCATIONS, payloads["locations"], "locations")
    _run_batched(UPSERT_HAS_SKILL, payloads["has_skill"], "HAS_SKILL")
    _run_batched(UPSERT_REQUIRES_SKILL, payloads["requires_skill"], "REQUIRES_SKILL")
    _run_batched(UPSERT_CAN_FILL, payloads["can_fill"], "CAN_FILL")
    _run_batched(UPSERT_IS_DESIGNATION, payloads["is_designation"], "IS_DESIGNATION")
    _run_batched(UPSERT_IN_INDUSTRY, payloads["in_industry"], "IN_INDUSTRY")
    _run_batched(UPSERT_AT_LOCATION, payloads["at_location"], "AT_LOCATION")

    # Health
    h = health_check()
    console.print(
        f"\n[green]✓[/] graph ready: nodes={h['nodes']:,}, relationships={h['relationships']:,}"
    )


def main(
    reset: bool = typer.Option(False, "--reset", help="WIPE + re-ingest."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate inputs only."),
) -> None:
    run(reset=reset, dry_run=dry_run)


if __name__ == "__main__":
    typer.run(main)
