"""ETL pipeline — load profiles + demands + JDs into Parquet + DuckDB.

Idempotent. Run with:
    uv run python scripts/01_etl.py            # all stages
    uv run python scripts/01_etl.py profiles   # one stage
    uv run python scripts/01_etl.py demands
    uv run python scripts/01_etl.py jds
    uv run python scripts/01_etl.py register   # rebuild DuckDB views from existing Parquet

Outputs:
    data/interim/profiles.parquet      # 1 row per Profile
    data/interim/profile_skills.parquet  # exploded HAS_SKILL edges
    data/interim/demands.parquet       # 1 row per demand record (Job, source=demands_csv)
    data/interim/jds.parquet           # 1 row per JD bundle (Job, source=jd_zip)
    data/interim/job_skills.parquet    # exploded REQUIRES_SKILL edges from both Job sources
    data/processed/pathfinder.duckdb   # tables + canonical_text views
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from app.core.schema_loader import load_schema_map
from app.models.skill import PROFICIENCY_ORDINAL

REPO_API_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = REPO_API_DIR.parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
INTERIM_DIR = REPO_ROOT / "data" / "interim"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DUCKDB_PATH = PROCESSED_DIR / "pathfinder.duckdb"

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Skill normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

PROFICIENCY_RE = re.compile(
    r"^\s*(?P<name>.+?)\s*"
    r"\(\s*(?P<prof>Beginner|Advanced Beginner|Competent|Proficient|Expert)\s*\)\s*$"
)


def slugify(s: str) -> str:
    """Slugify a skill / role name for use as a node id."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s or "unknown"


def _clean_str(v: Any) -> str | None:
    """Coerce a pandas cell to a clean str-or-None (handles NaN, empty, whitespace)."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip()
    return s if s else None


def split_csv_list(raw: Any) -> list[str]:
    """Split a comma-separated cell, strip whitespace, drop empties."""
    s = _clean_str(raw)
    if s is None:
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


def parse_proficiency_skills(raw: Any) -> list[tuple[str, str, int]]:
    """Parse 'Python (Advanced Beginner), SQL (Beginner)' → [(name, label, ordinal), …]."""
    raw = _clean_str(raw)
    if raw is None:
        return []
    out: list[tuple[str, str, int]] = []
    # Split on commas that are NOT inside parentheses.
    chunks = re.split(r",(?![^(]*\))", str(raw))
    for ch in chunks:
        m = PROFICIENCY_RE.match(ch)
        if m:
            name = m.group("name").strip()
            label = m.group("prof").strip()
            out.append((name, label, PROFICIENCY_ORDINAL[label]))  # type: ignore[index]
        else:
            # No proficiency tag — default to Beginner so we don't lose the skill.
            n = ch.strip()
            if n:
                out.append((n, "Beginner", 1))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — profiles.csv
# ─────────────────────────────────────────────────────────────────────────────


def etl_profiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    schema = load_schema_map("profiles")  # noqa: F841 — reserved for future column renames
    src = RAW_DIR / "profiles.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing — run scripts/00_inspect_csv.py first.")

    df = pd.read_csv(src)
    console.print(f"[cyan]profiles.csv[/]: {len(df):,} rows")

    profiles_rows: list[dict[str, Any]] = []
    skill_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        person_id = f"person_{int(row['id'])}"
        name = _clean_str(row.get("name"))
        yoe_raw = row.get("years_of_experience")
        years = float(yoe_raw) if not pd.isna(yoe_raw) else 0.0
        summary = _clean_str(row.get("skill_summary"))
        roles = split_csv_list(row.get("potential_roles"))

        # Skills with proficiency
        for col, category in (
            ("core_skills", "core"),
            ("secondary_skills", "secondary"),
            ("soft_skills", "soft"),
        ):
            for skill_name, label, ordinal in parse_proficiency_skills(row.get(col)):
                skill_rows.append(
                    {
                        "person_id": person_id,
                        "skill_id": slugify(skill_name),
                        "skill_name": skill_name,
                        "category": category,
                        "proficiency_label": label,
                        "proficiency": ordinal,
                        "source": "resume",
                    }
                )

        # Canonical text per profiles.yaml template
        canonical = "\n".join(
            line
            for line in (
                name or "",
                f"Skills: {_clean_str(row.get('core_skills')) or ''}",
                f"Secondary skills: {_clean_str(row.get('secondary_skills')) or ''}",
                f"Soft skills: {_clean_str(row.get('soft_skills')) or ''}",
                f"Years experience: {years}",
                f"Potential roles: {_clean_str(row.get('potential_roles')) or ''}",
                f"Summary: {summary or ''}",
            )
            if line.strip()
        ).strip()

        profiles_rows.append(
            {
                "id": person_id,
                "raw_id": int(row["id"]),
                "name": name,
                "years_experience": years,
                "potential_roles": roles,
                "skill_summary": summary,
                "canonical_text": canonical,
            }
        )

    profiles_df = pd.DataFrame(profiles_rows)
    skills_df = pd.DataFrame(skill_rows)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    profiles_df.to_parquet(INTERIM_DIR / "profiles.parquet", index=False)
    skills_df.to_parquet(INTERIM_DIR / "profile_skills.parquet", index=False)
    console.print(
        f"  → wrote profiles.parquet ({len(profiles_df):,}) "
        f"+ profile_skills.parquet ({len(skills_df):,})"
    )
    return profiles_df, skills_df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — demands_data.csv
# ─────────────────────────────────────────────────────────────────────────────

COUNTRY_ALIASES = {
    "USA": "United States",
    "United States of America": "United States",
    "UK": "United Kingdom",
    "Britain": "United Kingdom",
}


def _norm_country(c: Any) -> str | None:
    if pd.isna(c):
        return None
    c = str(c).strip()
    return COUNTRY_ALIASES.get(c, c) if c else None


def etl_demands() -> tuple[pd.DataFrame, pd.DataFrame]:
    schema = load_schema_map("demands")  # noqa: F841
    src = RAW_DIR / "demands_data.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing.")

    df = pd.read_csv(src)
    console.print(f"[cyan]demands_data.csv[/]: {len(df):,} rows")

    demand_rows: list[dict[str, Any]] = []
    skill_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        raw_demand_id = str(row["id"]).strip()
        job_id = f"job_demand_{slugify(raw_demand_id)}"

        designation = _clean_str(row.get("designation"))
        title = _clean_str(row.get("job_title")) or designation
        city = _clean_str(row.get("city"))
        country = _norm_country(row.get("country"))
        exp_lo_raw = row.get("experience_lower")
        exp_hi_raw = row.get("experience_upper")
        exp_lo = int(exp_lo_raw) if not pd.isna(exp_lo_raw) else 0
        exp_hi = int(exp_hi_raw) if not pd.isna(exp_hi_raw) else 0

        primary = split_csv_list(row.get("primary_skills"))
        secondary = split_csv_list(row.get("secondary_skills"))

        for s in primary:
            skill_rows.append(
                {
                    "job_id": job_id,
                    "skill_id": slugify(s),
                    "skill_name": s,
                    "priority": "must_have",
                    "category": "primary",
                    "source": "demand_primary",
                }
            )
        for s in secondary:
            skill_rows.append(
                {
                    "job_id": job_id,
                    "skill_id": slugify(s),
                    "skill_name": s,
                    "priority": "good_to_have",
                    "category": "secondary",
                    "source": "demand_secondary",
                }
            )

        canonical = (
            f"Designation: {designation or ''}\n"
            f"Primary skills: {', '.join(primary)}\n"
            f"Secondary skills: {', '.join(secondary)}\n"
            f"Location: {city or ''}, {country or ''}\n"
            f"Experience: {exp_lo}-{exp_hi} years"
        ).strip()

        demand_rows.append(
            {
                "id": job_id,
                "source": "demands_csv",
                "raw_id": raw_demand_id,
                "title": title,
                "designation": designation,
                "industry": None,
                "city": city,
                "country": country,
                "experience_lower": exp_lo,
                "experience_upper": exp_hi,
                "responsibilities": [],
                "educational_qualifications": [],
                "other_requirements": None,
                "raw_text": None,
                "enhanced_text": None,
                "canonical_text": canonical,
            }
        )

    demands_df = pd.DataFrame(demand_rows)
    skills_df = pd.DataFrame(skill_rows)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    demands_df.to_parquet(INTERIM_DIR / "demands.parquet", index=False)
    skills_df.to_parquet(INTERIM_DIR / "demand_skills.parquet", index=False)
    console.print(
        f"  → wrote demands.parquet ({len(demands_df):,}) "
        f"+ demand_skills.parquet ({len(skills_df):,})"
    )
    return demands_df, skills_df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — jd_dataset.zip
# ─────────────────────────────────────────────────────────────────────────────

H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
H3_RE = re.compile(r"^\*\*(.+?)\*\*\s*$", re.MULTILINE)
BULLET_RE = re.compile(r"^\s*[-*]\s+(.+?)\s*$", re.MULTILINE)


def _parse_md_sections(md: str) -> dict[str, str]:
    """Split an enhanced JD markdown into a {h2_title: body} dict."""
    parts: dict[str, str] = {}
    matches = list(H2_RE.finditer(md))
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        parts[title] = md[start:end].strip()
    return parts


def _parse_skill_requirements(body: str) -> tuple[list[str], list[str], list[str]]:
    """Split 'Skill Requirements' into Must-Have / Good-to-Have / Educational lists.

    Two formats observed in the JD bundle:
    1. Bold subheaders: `**Must Have:**` / `**Good to Have:**` / `**Educational Qualifications:**`
    2. Flat bullet list under `## Skill Requirements` with no subheaders.

    For (2) we default the whole list to must_have. For (1) we bucket per subheader.
    Bullet bodies are kept as raw text — atomic skill extraction is the job of the
    downstream canonicalisation script (05_skill_canonicalize.py).
    """
    must: list[str] = []
    good: list[str] = []
    edu: list[str] = []

    # Strip CRs that bleed in from windows-style line endings.
    body = body.replace("\r\n", "\n").replace("\r", "\n")

    # Locate **bold subheaders**; if none, fall back to the flat-list interpretation.
    header_re = re.compile(r"^\*\*\s*(.+?)\s*\*\*\s*:?\s*$", re.MULTILINE)
    headers = list(header_re.finditer(body))

    if not headers:
        bullets = [b.strip() for b in BULLET_RE.findall(body) if b.strip()]
        return bullets, [], []

    # Walk header → body slices.
    for i, m in enumerate(headers):
        header = m.group(1).strip().lower()
        sec_start = m.end()
        sec_end = headers[i + 1].start() if i + 1 < len(headers) else len(body)
        section_body = body[sec_start:sec_end]
        bullets = [b.strip() for b in BULLET_RE.findall(section_body) if b.strip()]
        if "must" in header or "required" in header:
            must = bullets
        elif "good" in header or "nice" in header or "preferred" in header:
            good = bullets
        elif "education" in header:
            edu = bullets
        # Other unrecognised subheaders (e.g. "Experience:", "Certifications:") are ignored —
        # the prose stays in `other_requirements` via the parent section split.
    return must, good, edu


def _parse_jd_location(loc: str) -> tuple[str | None, str | None]:
    """'Pune, India' → ('Pune', 'India')."""
    if not loc:
        return None, None
    parts = [p.strip() for p in loc.split(",", 1)]
    city = parts[0] or None
    country = (parts[1] if len(parts) > 1 else "") or None
    return city, _norm_country(country)


def _iter_jd_folders(zf: zipfile.ZipFile) -> Iterator[str]:
    seen: set[str] = set()
    for n in zf.namelist():
        head = n.split("/")[0]
        if head and head not in seen and "/" in n:
            seen.add(head)
            yield head


def etl_jds() -> tuple[pd.DataFrame, pd.DataFrame]:
    schema = load_schema_map("jds")  # noqa: F841
    src = RAW_DIR / "jd_dataset.zip"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing.")

    z = zipfile.ZipFile(src)
    folders = sorted(_iter_jd_folders(z), key=lambda x: int(x) if x.isdigit() else 0)
    console.print(f"[cyan]jd_dataset.zip[/]: {len(folders)} jobs")

    jd_rows: list[dict[str, Any]] = []
    skill_rows: list[dict[str, Any]] = []

    for folder in folders:
        job_id = f"job_jd_{folder}"
        try:
            raw = json.loads(z.read(f"{folder}/raw_jd.txt").decode("utf-8", errors="replace"))
        except Exception:
            raw = {}
        raw_industry = (raw.get("industry") or "").strip() or None
        raw_text = raw.get("raw_jd") or ""

        try:
            md = z.read(f"{folder}/enhanced_job_description.md").decode("utf-8", errors="replace")
        except KeyError:
            console.print(f"  [yellow]skip {folder}: missing enhanced_job_description.md[/]")
            continue

        sections = _parse_md_sections(md)
        title = _clean_str(sections.get("Job Title"))
        loc_text = _clean_str(sections.get("Location")) or ""
        city, country = _parse_jd_location(loc_text)
        industry = _clean_str(sections.get("Client Industry")) or raw_industry
        responsibilities = [
            b.strip() for b in BULLET_RE.findall(sections.get("Detailed Responsibilities") or "")
        ]
        skill_body = sections.get("Skill Requirements") or ""
        must, good, edu = _parse_skill_requirements(skill_body)
        other = _clean_str(sections.get("Other Requirements"))

        for s in must:
            skill_rows.append(
                {
                    "job_id": job_id,
                    "skill_id": slugify(s),
                    "skill_name": s,
                    "priority": "must_have",
                    "category": "must_have",
                    "source": "jd_must",
                }
            )
        for s in good:
            skill_rows.append(
                {
                    "job_id": job_id,
                    "skill_id": slugify(s),
                    "skill_name": s,
                    "priority": "good_to_have",
                    "category": "good_to_have",
                    "source": "jd_nice",
                }
            )

        canonical_parts = [
            f"Title: {title or ''}",
            f"Industry: {industry or ''}",
            f"Location: {loc_text}",
            "Responsibilities:",
            *(f"- {r}" for r in responsibilities),
            f"Must have skills: {', '.join(must)}",
            f"Good to have skills: {', '.join(good)}",
            f"Other requirements: {other or ''}",
        ]
        canonical = "\n".join(canonical_parts).strip()

        jd_rows.append(
            {
                "id": job_id,
                "source": "jd_zip",
                "raw_id": folder,
                "title": title,
                "designation": title,  # fallback when designation absent
                "industry": industry,
                "city": city,
                "country": country,
                "experience_lower": 0,
                "experience_upper": 0,
                "responsibilities": responsibilities,
                "educational_qualifications": edu,
                "other_requirements": other,
                "raw_text": raw_text,
                "enhanced_text": md,
                "canonical_text": canonical,
            }
        )

    jds_df = pd.DataFrame(jd_rows)
    skills_df = pd.DataFrame(skill_rows)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    jds_df.to_parquet(INTERIM_DIR / "jds.parquet", index=False)
    skills_df.to_parquet(INTERIM_DIR / "jd_skills.parquet", index=False)
    console.print(
        f"  → wrote jds.parquet ({len(jds_df):,}) + jd_skills.parquet ({len(skills_df):,})"
    )
    return jds_df, skills_df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — DuckDB tables materialized from the Parquet outputs
#
# We use CREATE OR REPLACE TABLE (not VIEW) so the .duckdb file is
# self-contained and portable: the production deploy ships only the
# .duckdb to the HF Space and does not have access to the interim parquets.
# ─────────────────────────────────────────────────────────────────────────────


def register_duckdb() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DUCKDB_PATH))

    interim = INTERIM_DIR.as_posix()
    con.execute(
        f"""
        -- raw stage tables (materialized — no parquet path dependency at read time)
        CREATE OR REPLACE TABLE profiles        AS SELECT * FROM '{interim}/profiles.parquet';
        CREATE OR REPLACE TABLE profile_skills  AS SELECT * FROM '{interim}/profile_skills.parquet';
        CREATE OR REPLACE TABLE demands         AS SELECT * FROM '{interim}/demands.parquet';
        CREATE OR REPLACE TABLE demand_skills   AS SELECT * FROM '{interim}/demand_skills.parquet';
        CREATE OR REPLACE TABLE jds             AS SELECT * FROM '{interim}/jds.parquet';
        CREATE OR REPLACE TABLE jd_skills       AS SELECT * FROM '{interim}/jd_skills.parquet';

        -- unified Job table (demands + jds)
        CREATE OR REPLACE TABLE jobs AS
            SELECT * FROM demands
            UNION ALL BY NAME
            SELECT * FROM jds;

        -- unified job_skills (demands + jds)
        CREATE OR REPLACE TABLE job_skills AS
            SELECT * FROM demand_skills
            UNION ALL BY NAME
            SELECT * FROM jd_skills;

        -- canonical Skill catalog (deduped across both sides)
        CREATE OR REPLACE TABLE skills AS
            WITH all_skills AS (
                SELECT skill_id, skill_name FROM profile_skills
                UNION ALL
                SELECT skill_id, skill_name FROM job_skills
            )
            SELECT skill_id, MIN(skill_name) AS skill_name, COUNT(*) AS occurrences
            FROM all_skills
            GROUP BY skill_id;

        CHECKPOINT;
        """
    )

    # Show counts
    t = Table(title="DuckDB tables", show_lines=False)
    t.add_column("table", style="cyan")
    t.add_column("rows", justify="right")
    for view in (
        "profiles",
        "profile_skills",
        "demands",
        "demand_skills",
        "jds",
        "jd_skills",
        "jobs",
        "job_skills",
        "skills",
    ):
        n = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
        t.add_row(view, f"{n:,}")
    console.print(t)
    con.close()
    console.print(f"[green]✓[/] DuckDB at [cyan]{DUCKDB_PATH}[/]")


# ─────────────────────────────────────────────────────────────────────────────
# Typer CLI
# ─────────────────────────────────────────────────────────────────────────────

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def profiles() -> None:
    """ETL profiles.csv only."""
    etl_profiles()


@app.command()
def demands() -> None:
    """ETL demands_data.csv only."""
    etl_demands()


@app.command()
def jds() -> None:
    """ETL jd_dataset.zip only."""
    etl_jds()


@app.command()
def register() -> None:
    """Rebuild DuckDB views over existing Parquet files."""
    register_duckdb()


@app.command()
def all() -> None:
    """Run every stage end-to-end (default)."""
    console.rule("[bold magenta]PathFinder ETL — full pipeline[/]")
    etl_profiles()
    etl_demands()
    etl_jds()
    register_duckdb()
    sys.exit(0)


if __name__ == "__main__":
    # Default to `all` if no subcommand given.
    if len(sys.argv) == 1:
        sys.argv.append("all")
    app()
