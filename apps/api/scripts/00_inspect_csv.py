"""DAY-1 schema introspection for profiles.csv.

Idempotent. Run with:
    uv run python scripts/00_inspect_csv.py
    uv run python scripts/00_inspect_csv.py --csv path/to/profiles.csv
    uv run python scripts/00_inspect_csv.py --no-download   # skip remote fetch

What it does:
  1. Tries to download profiles.csv from the IITMandiHack60 repo (main + master).
  2. Inspects: row count, columns + dtypes, null %, sample rows, value cardinality.
  3. Heuristically maps columns to canonical fields (id, name, headline, summary,
     skills, experience, education, location, years_experience, current_company,
     current_title, industry).
  4. Emits app/core/schema_map.yaml. Downstream code reads ONLY from this YAML.
  5. STOPs and asks the user to confirm any ambiguous mappings before ETL.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import typer
import yaml
from rich import box
from rich.console import Console
from rich.table import Table

REPO_API_DIR = Path(__file__).resolve().parents[1]              # apps/api/
REPO_ROOT = REPO_API_DIR.parents[1]                             # pathfinder/
DEFAULT_CSV = REPO_ROOT / "data" / "raw" / "profiles.csv"
SCHEMA_MAP_PATH = REPO_API_DIR / "app" / "core" / "schema_map.yaml"

REMOTE_CANDIDATES = [
    "https://raw.githubusercontent.com/anonymous-devx/IITMandiHack60/main/profiles.csv",
    "https://raw.githubusercontent.com/anonymous-devx/IITMandiHack60/master/profiles.csv",
    # Mirrored CDN paths (best-effort fallbacks):
    "https://cdn.jsdelivr.net/gh/anonymous-devx/IITMandiHack60@main/profiles.csv",
    "https://cdn.jsdelivr.net/gh/anonymous-devx/IITMandiHack60@master/profiles.csv",
]

# ─────────────────────────────────────────────────────────────────────────────
# Heuristics: column-name regex + sample-value parsers per canonical field.
# Each canonical field maps to:
#   - name_patterns: regex(s) the column NAME must match (case-insensitive)
#   - value_check:   optional callable(series) → confidence boost (0..1)
# Matches are scored; best match wins per canonical field.
# ─────────────────────────────────────────────────────────────────────────────

CANONICAL_FIELDS: dict[str, dict[str, Any]] = {
    "id": {
        "name_patterns": [r"^id$", r"^profile_?id$", r"^user_?id$", r"^uuid$", r"^_id$"],
        "kind": "primary_key",
        "required": True,
    },
    "name": {
        "name_patterns": [r"^(full_?)?name$", r"^display_?name$", r"^person$"],
        "kind": "person_name",
        "required": True,
    },
    "headline": {
        "name_patterns": [r"^headline$", r"^tagline$", r"^title$", r"^heading$"],
        "kind": "short_text",
        "required": False,
    },
    "summary": {
        "name_patterns": [r"^summary$", r"^bio$", r"^about$", r"^description$", r"^profile$"],
        "kind": "long_text",
        "required": False,
    },
    "skills": {
        "name_patterns": [r"^skills?$", r"^skill_?list$", r"^technologies$", r"^tech$"],
        "kind": "list_or_csv",
        "required": False,
    },
    "experience": {
        "name_patterns": [r"^experiences?$", r"^work_?history$", r"^positions?$", r"^career$"],
        "kind": "json_or_text",
        "required": False,
    },
    "education": {
        "name_patterns": [r"^educations?$", r"^schools?$", r"^degrees?$", r"^academic"],
        "kind": "json_or_text",
        "required": False,
    },
    "location": {
        "name_patterns": [r"^location$", r"^city$", r"^region$", r"^geo$", r"^address$", r"^based_?in$"],
        "kind": "short_text",
        "required": False,
    },
    "years_experience": {
        "name_patterns": [r"^years?_?(of_)?exp(erience)?$", r"^total_?exp", r"^yoe$"],
        "kind": "numeric",
        "required": False,
    },
    "current_company": {
        "name_patterns": [r"^current_?company$", r"^company$", r"^employer$", r"^org(anisation)?$"],
        "kind": "short_text",
        "required": False,
    },
    "current_title": {
        "name_patterns": [r"^current_?title$", r"^role$", r"^position$", r"^job_?title$"],
        "kind": "short_text",
        "required": False,
    },
    "industry": {
        "name_patterns": [r"^industry$", r"^sector$", r"^domain$", r"^vertical$"],
        "kind": "short_text",
        "required": False,
    },
}


# ─── Value-shape detectors ───────────────────────────────────────────────────

def _looks_like_json(series: pd.Series, sample_n: int = 25) -> float:
    sample = series.dropna().astype(str).head(sample_n)
    if sample.empty:
        return 0.0
    hits = 0
    for v in sample:
        v = v.strip()
        if not v:
            continue
        if (v.startswith("[") and v.endswith("]")) or (v.startswith("{") and v.endswith("}")):
            try:
                json.loads(v)
                hits += 1
            except Exception:  # noqa: BLE001
                pass
    return hits / len(sample) if len(sample) else 0.0


def _looks_like_delimited_list(series: pd.Series, sample_n: int = 25) -> float:
    sample = series.dropna().astype(str).head(sample_n)
    if sample.empty:
        return 0.0
    hits = sum(
        1 for v in sample
        if any(d in v for d in (",", ";", "|"))
        and len(re.split(r"[,;|]", v)) >= 2
    )
    return hits / len(sample)


def _looks_numeric(series: pd.Series) -> float:
    if pd.api.types.is_numeric_dtype(series):
        return 1.0
    coerced = pd.to_numeric(series, errors="coerce")
    return float(coerced.notna().mean())


def _avg_token_count(series: pd.Series, sample_n: int = 50) -> float:
    sample = series.dropna().astype(str).head(sample_n)
    if sample.empty:
        return 0.0
    return float(sample.str.split().str.len().mean())


# ─── Mapping core ────────────────────────────────────────────────────────────

def _score_column_for_field(
    column: str,
    series: pd.Series,
    field: str,
    spec: dict[str, Any],
) -> tuple[float, list[str]]:
    """Return (score in [0, 1], notes). Higher = better fit."""
    score = 0.0
    notes: list[str] = []
    col_norm = column.strip().lower().replace(" ", "_")

    # Name-regex match
    name_score = 0.0
    for pat in spec["name_patterns"]:
        if re.fullmatch(pat, col_norm):
            name_score = 1.0
            notes.append(f"exact-name-match `{pat}`")
            break
        if re.search(pat, col_norm):
            name_score = max(name_score, 0.6)
            notes.append(f"partial-name-match `{pat}`")
    score += 0.6 * name_score

    # Value-shape check
    kind = spec["kind"]
    if kind == "primary_key":
        # Primary key: high uniqueness + non-null
        n = len(series)
        uniq = series.nunique(dropna=True) / n if n else 0.0
        nonnull = series.notna().mean()
        v_score = (uniq * 0.7) + (nonnull * 0.3)
        notes.append(f"uniqueness={uniq:.2f} nonnull={nonnull:.2f}")
    elif kind == "numeric":
        v_score = _looks_numeric(series)
        notes.append(f"numeric_ratio={v_score:.2f}")
    elif kind == "json_or_text":
        json_r = _looks_like_json(series)
        delim_r = _looks_like_delimited_list(series) * 0.5
        # JSON is a stronger positive than delim; long text is also fine
        toks = _avg_token_count(series)
        text_r = min(toks / 30.0, 1.0)
        v_score = max(json_r, delim_r, text_r * 0.5)
        notes.append(f"json={json_r:.2f} delim={delim_r:.2f} avg_tokens={toks:.1f}")
    elif kind == "list_or_csv":
        json_r = _looks_like_json(series)
        delim_r = _looks_like_delimited_list(series)
        v_score = max(json_r * 0.8, delim_r)
        notes.append(f"json={json_r:.2f} delim={delim_r:.2f}")
    elif kind == "long_text":
        toks = _avg_token_count(series)
        v_score = min(toks / 50.0, 1.0)
        notes.append(f"avg_tokens={toks:.1f}")
    elif kind == "short_text":
        toks = _avg_token_count(series)
        # Penalise very long values (likely a different column)
        v_score = 1.0 - abs(toks - 4) / 20.0
        v_score = max(0.0, min(v_score, 1.0))
        notes.append(f"avg_tokens={toks:.1f}")
    elif kind == "person_name":
        toks = _avg_token_count(series)
        v_score = 1.0 if 1.5 <= toks <= 5 else 0.4
        notes.append(f"avg_tokens={toks:.1f}")
    else:
        v_score = 0.5
    score += 0.4 * v_score

    return score, notes


def map_columns(df: pd.DataFrame, min_score: float = 0.45) -> dict[str, Any]:
    """Greedy assignment: each canonical field gets the best-scoring column.

    Fields whose best match scores below `min_score` are reported as `column: None`
    with confidence `absent` — we never force a column when nothing fits, since the
    plan is dataset-agnostic and false-positives confuse downstream code.
    """
    columns = list(df.columns)
    used: set[str] = set()
    mapping: dict[str, dict[str, Any]] = {}
    for field, spec in CANONICAL_FIELDS.items():
        best: tuple[str | None, float, list[str]] = (None, 0.0, [])
        for col in columns:
            if col in used:
                continue
            score, notes = _score_column_for_field(col, df[col], field, spec)
            if score > best[1]:
                best = (col, score, notes)
        col, score, notes = best
        if score < min_score:
            confidence = "absent"
            chosen_col: str | None = None
        else:
            confidence = "high" if score >= 0.7 else "medium"
            chosen_col = col
            if chosen_col is not None:
                used.add(chosen_col)
        mapping[field] = {
            "column": chosen_col,
            "score": round(score, 3),
            "confidence": confidence,
            "kind": spec["kind"],
            "required": spec["required"],
            "notes": notes,
        }

    unmapped = [c for c in columns if c not in used]
    return {"fields": mapping, "unmapped_columns": unmapped, "total_columns": len(columns)}


# ─── IO helpers ──────────────────────────────────────────────────────────────

def try_download(target: Path, urls: list[str], console: Console) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            console.print(f"[dim]→ trying[/] {url}")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "PathFinder-Bootstrap/0.1 (https://github.com/<owner>/pathfinder)"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp, target.open("wb") as out:
                shutil.copyfileobj(resp, out)
            if target.exists() and target.stat().st_size > 0:
                console.print(f"[green]✓[/] downloaded → {target} ({target.stat().st_size:,} bytes)")
                return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            console.print(f"[yellow]✗[/] {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]✗[/] unexpected: {exc}")
    return False


def render_summary(df: pd.DataFrame, console: Console) -> None:
    console.rule("[bold]CSV summary[/]")
    console.print(f"rows: [cyan]{len(df):,}[/]   columns: [cyan]{len(df.columns)}[/]")

    t = Table(title="Columns", box=box.SIMPLE_HEAVY)
    t.add_column("name", style="cyan")
    t.add_column("dtype")
    t.add_column("null %", justify="right")
    t.add_column("uniq", justify="right")
    t.add_column("sample", style="dim")
    for col in df.columns:
        s = df[col]
        nulls = s.isna().mean() * 100
        uniq = s.nunique(dropna=True)
        sample = ""
        for v in s.dropna().astype(str):
            sample = v.replace("\n", " ")[:80]
            if sample:
                break
        t.add_row(col, str(s.dtype), f"{nulls:.1f}%", f"{uniq:,}", sample)
    console.print(t)

    console.rule("[bold]Sample rows[/]")
    sample_rows = df.head(5).to_dict(orient="records")
    for i, row in enumerate(sample_rows, 1):
        console.print(f"[bold cyan]row {i}[/]")
        for k, v in row.items():
            v_str = str(v).replace("\n", " ")
            if len(v_str) > 200:
                v_str = v_str[:200] + "…"
            console.print(f"  [dim]{k}:[/] {v_str}")
        console.print()


def render_mapping(mapping: dict[str, Any], console: Console) -> None:
    console.rule("[bold]Proposed schema mapping[/]")
    t = Table(box=box.SIMPLE_HEAVY)
    t.add_column("canonical field", style="cyan")
    t.add_column("→")
    t.add_column("CSV column")
    t.add_column("conf", style="bold")
    t.add_column("score", justify="right")
    t.add_column("required", justify="center")
    t.add_column("notes", style="dim")
    for field, m in mapping["fields"].items():
        conf = m["confidence"]
        conf_color = {"high": "green", "medium": "yellow", "low": "red"}[conf]
        t.add_row(
            field,
            "→",
            str(m["column"]),
            f"[{conf_color}]{conf}[/]",
            f"{m['score']:.3f}",
            "Y" if m["required"] else "·",
            "; ".join(m["notes"]),
        )
    console.print(t)
    if mapping["unmapped_columns"]:
        console.print(
            f"[dim]Unmapped columns ({len(mapping['unmapped_columns'])}):[/] "
            + ", ".join(mapping["unmapped_columns"])
        )


def _to_yaml_safe(obj: Any) -> Any:
    """Convert numpy / pandas scalars to native Python types so PyYAML can serialise."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _to_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_yaml_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_to_yaml_safe(v) for v in obj.tolist()]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_schema_map(
    csv_path: Path,
    df: pd.DataFrame,
    mapping: dict[str, Any],
    out: Path,
) -> None:
    payload = {
        "version": 1,
        "generated_by": "scripts/00_inspect_csv.py",
        "source_csv": str(csv_path),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
        "fields": {
            f: {
                "column": m["column"],
                "score": float(m["score"]),
                "confidence": m["confidence"],
                "kind": m["kind"],
                "required": bool(m["required"]),
                "notes": list(m["notes"]),
            }
            for f, m in mapping["fields"].items()
        },
        "unmapped_columns": list(mapping["unmapped_columns"]),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_to_yaml_safe(payload), f, sort_keys=False, allow_unicode=True)


# ─── Typer CLI ────────────────────────────────────────────────────────────────

app = typer.Typer(add_completion=False, no_args_is_help=False, help=__doc__)


@app.command()
def main(
    csv: Path = typer.Option(DEFAULT_CSV, "--csv", help="Path to profiles.csv"),
    download: bool = typer.Option(True, "--download/--no-download", help="Try to download if missing"),
    out: Path = typer.Option(SCHEMA_MAP_PATH, "--out", help="Output schema_map.yaml path"),
    overwrite: bool = typer.Option(True, "--overwrite/--no-overwrite", help="Overwrite existing schema_map.yaml"),
) -> None:
    console = Console()
    console.rule("[bold magenta]PathFinder · Day 1 · CSV introspection[/]")

    # 1) Acquire CSV
    if not csv.exists():
        if download:
            console.print(f"[yellow]CSV missing at {csv}; trying remote sources…[/]")
            ok = try_download(csv, REMOTE_CANDIDATES, console)
            if not ok:
                console.print()
                console.print(
                    "[red bold]Could not download profiles.csv automatically.[/]\n"
                    f"Place it at [cyan]{csv}[/] manually then re-run this script.\n"
                    "Hint: clone the IITMandiHack60 repo or copy the file from the dataset bundle."
                )
                raise typer.Exit(code=2)
        else:
            console.print(f"[red]CSV not found at {csv} (--no-download set).[/]")
            raise typer.Exit(code=2)

    # 2) Read it
    try:
        df = pd.read_csv(csv, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed to parse CSV: {exc}[/]")
        # Retry with python engine (more tolerant of malformed rows)
        df = pd.read_csv(csv, engine="python", on_bad_lines="warn")

    # 3) Inspect
    render_summary(df, console)

    # 4) Map columns
    mapping = map_columns(df)
    render_mapping(mapping, console)

    # 5) Persist
    if out.exists() and not overwrite:
        console.print(f"[yellow]{out} exists; --no-overwrite → leaving untouched.[/]")
    else:
        write_schema_map(csv, df, mapping, out)
        console.print(f"[green]✓[/] wrote schema map → [cyan]{out}[/]")

    # 6) Stop & confirm
    console.rule("[bold]ACTION REQUIRED[/]")
    needs_attention = [
        f for f, m in mapping["fields"].items()
        if m["confidence"] != "high" and (m["required"] or m["column"] is not None)
    ]
    if needs_attention:
        console.print(
            "[yellow bold]Review these mappings before proceeding to ETL:[/]"
        )
        for f in needs_attention:
            m = mapping["fields"][f]
            console.print(f"  · [cyan]{f}[/] → {m['column']} (conf=[bold]{m['confidence']}[/], score={m['score']:.2f})")
        console.print()
        console.print(
            "Edit [cyan]apps/api/app/core/schema_map.yaml[/] manually if any "
            "mapping is wrong, then run scripts/01_etl.py."
        )
    else:
        console.print("[green]All required fields mapped with high confidence.[/]")

    sys.exit(0)


if __name__ == "__main__":
    app()
