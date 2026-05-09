"""Skill canonicalisation runner — alias YAML → ESCO API → BGE-M3 cosine.

Idempotent. Run with:
    uv run python scripts/06_skill_canonicalize.py            # full pipeline
    uv run python scripts/06_skill_canonicalize.py --no-esco  # skip the ESCO API leg
    uv run python scripts/06_skill_canonicalize.py --no-embed # skip the embedding fallback
    uv run python scripts/06_skill_canonicalize.py --limit 50 # debug: only first 50 skills

Reads the deduplicated `skills` view from `data/processed/pathfinder.duckdb` and
writes:
    data/processed/skills/canonical.parquet   raw_name | canonical_id | canonical_name | source | esco_uri | confidence
    data/processed/skills/canonical.json      same data (small enough to keep both)
    data/processed/skills/stats.json          counts by source, ESCO hit rate, etc.

Cached: ESCO calls are persisted to `data/processed/skills/esco_cache.jsonl` so
subsequent runs are free.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from app.services.skills import (
    _embedding_cluster,
    _esco_match,
    _normalise,
    _slug,
    load_alias_table,
)


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pnpm-workspace.yaml").exists():
            return parent
    raise RuntimeError("repo root marker not found")


REPO_ROOT = _find_repo_root()
DUCKDB_PATH = REPO_ROOT / "data" / "processed" / "pathfinder.duckdb"
OUT_DIR = REPO_ROOT / "data" / "processed" / "skills"
ESCO_CACHE_PATH = OUT_DIR / "esco_cache.jsonl"

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)


# ─── Cached ESCO lookups ────────────────────────────────────────────────────


def _load_esco_cache() -> dict[str, dict | None]:
    cache: dict[str, dict | None] = {}
    if not ESCO_CACHE_PATH.exists():
        return cache
    with ESCO_CACHE_PATH.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            cache[row["raw_name"]] = row.get("hit")  # may be None if no match
    return cache


def _append_esco_cache(raw_name: str, hit: dict | None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with ESCO_CACHE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"raw_name": raw_name, "hit": hit}) + "\n")


# ─── Pipeline ───────────────────────────────────────────────────────────────


def run(
    use_esco: bool = True,
    use_embed: bool = True,
    limit: int | None = None,
) -> None:
    if not DUCKDB_PATH.exists():
        raise FileNotFoundError(f"{DUCKDB_PATH} missing — run scripts/01_etl.py first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    rows = con.execute(
        "SELECT skill_id AS legacy_id, skill_name AS raw_name, occurrences "
        "FROM skills ORDER BY occurrences DESC, raw_name ASC"
    ).fetchall()
    con.close()
    if limit:
        rows = rows[:limit]
    console.print(f"[cyan]canonicalise[/] {len(rows):,} unique skills")

    # ─── Layer 1: alias table ────────────────────────────────────────────
    alias_map, _ = load_alias_table()
    layer1: dict[str, dict] = {}
    layer1_unmatched: list[tuple[str, str, int]] = []
    for legacy_id, raw_name, occ in rows:
        n = _normalise(raw_name)
        if n in alias_map:
            canon = alias_map[n]
            layer1[raw_name] = {
                "raw_name": raw_name,
                "legacy_id": legacy_id,
                "occurrences": int(occ),
                "canonical_id": _slug(canon),
                "canonical_name": canon,
                "source": "alias",
                "esco_uri": None,
                "confidence": 1.0,
            }
        else:
            layer1_unmatched.append((legacy_id, raw_name, int(occ)))
    console.print(
        f"  layer 1 (alias):     [green]{len(layer1):,} matched[/], {len(layer1_unmatched):,} unmatched"
    )

    # ─── Layer 2: ESCO API ───────────────────────────────────────────────
    layer2: dict[str, dict] = {}
    layer2_unmatched: list[tuple[str, str, int]] = []
    if use_esco:
        esco_cache = _load_esco_cache()
        console.print(f"  esco cache: {len(esco_cache):,} cached lookups loaded")
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]ESCO[/] {task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as prog:
            tid = prog.add_task("query layer 2", total=len(layer1_unmatched))
            for legacy_id, raw_name, occ in layer1_unmatched:
                if raw_name in esco_cache:
                    hit_dict = esco_cache[raw_name]
                else:
                    hit = _esco_match(raw_name)
                    hit_dict = (
                        {"title": hit.title, "uri": hit.uri, "score": hit.score}
                        if hit is not None
                        else None
                    )
                    _append_esco_cache(raw_name, hit_dict)
                    esco_cache[raw_name] = hit_dict
                    time.sleep(0.05)  # gentle throttle
                if hit_dict is not None:
                    layer2[raw_name] = {
                        "raw_name": raw_name,
                        "legacy_id": legacy_id,
                        "occurrences": occ,
                        "canonical_id": _slug(hit_dict["title"]),
                        "canonical_name": hit_dict["title"],
                        "source": "esco",
                        "esco_uri": hit_dict["uri"],
                        "confidence": float(hit_dict["score"]),
                    }
                else:
                    layer2_unmatched.append((legacy_id, raw_name, occ))
                prog.update(tid, advance=1)
        console.print(
            f"  layer 2 (ESCO):      [green]{len(layer2):,} matched[/], {len(layer2_unmatched):,} unmatched"
        )
    else:
        layer2_unmatched = layer1_unmatched
        console.print("  layer 2 (ESCO):      [dim]skipped (--no-esco)[/]")

    # ─── Layer 3: BGE-M3 embedding cosine ───────────────────────────────
    layer3: dict[str, dict] = {}
    if use_embed and layer2_unmatched:
        raw_names = [r[1] for r in layer2_unmatched]
        cluster_map = _embedding_cluster(raw_names, sim_threshold=0.92)
        for legacy_id, raw_name, occ in layer2_unmatched:
            canon = cluster_map.get(raw_name, raw_name)
            layer3[raw_name] = {
                "raw_name": raw_name,
                "legacy_id": legacy_id,
                "occurrences": occ,
                "canonical_id": _slug(canon),
                "canonical_name": canon,
                "source": "embedding" if canon != raw_name else "raw",
                "esco_uri": None,
                "confidence": 0.92 if canon != raw_name else 0.5,
            }
        console.print(
            f"  layer 3 (embedding): "
            f"[green]{sum(1 for v in layer3.values() if v['source'] == 'embedding'):,} clustered[/], "
            f"{sum(1 for v in layer3.values() if v['source'] == 'raw'):,} raw"
        )
    else:
        for legacy_id, raw_name, occ in layer2_unmatched:
            layer3[raw_name] = {
                "raw_name": raw_name,
                "legacy_id": legacy_id,
                "occurrences": occ,
                "canonical_id": _slug(raw_name),
                "canonical_name": raw_name,
                "source": "raw",
                "esco_uri": None,
                "confidence": 0.5,
            }
        if not use_embed:
            console.print("  layer 3 (embedding): [dim]skipped (--no-embed)[/]")

    # ─── Aggregate + persist ─────────────────────────────────────────────
    all_rows = {**layer1, **layer2, **layer3}
    df = pd.DataFrame(list(all_rows.values()))

    df.to_parquet(OUT_DIR / "canonical.parquet", index=False)
    (OUT_DIR / "canonical.json").write_text(
        df.to_json(orient="records", indent=2), encoding="utf-8"
    )

    src_counts = Counter(df["source"])
    canon_count = df["canonical_id"].nunique()
    stats = {
        "total_raw_skills": len(df),
        "distinct_canonical_skills": int(canon_count),
        "compression_ratio": round(len(df) / max(canon_count, 1), 3),
        "by_source": dict(src_counts),
        "esco_uris_resolved": int((df["esco_uri"].fillna("") != "").sum()),
        "alias_yaml_size": len(load_alias_table()[0]),
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # Pretty print summary
    t = Table(title="canonicalisation summary", show_lines=False)
    t.add_column("metric", style="cyan")
    t.add_column("value", justify="right")
    for k, v in stats.items():
        t.add_row(k, str(v))
    console.print(t)
    console.print(f"[green]✓[/] artifacts → {OUT_DIR}")


@app.command()
def main(
    no_esco: bool = typer.Option(False, "--no-esco", help="Skip layer 2 (ESCO API)."),
    no_embed: bool = typer.Option(False, "--no-embed", help="Skip layer 3 (BGE-M3 cosine)."),
    limit: int | None = typer.Option(None, "--limit", help="Debug: only first N skills."),
) -> None:
    run(use_esco=not no_esco, use_embed=not no_embed, limit=limit)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("main")  # default subcommand
    app()
