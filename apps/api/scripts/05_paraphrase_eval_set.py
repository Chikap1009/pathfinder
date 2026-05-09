"""Paraphrase the deterministic eval set with Gemini → 100 natural-language queries.

Idempotent. Run with:
    uv run python scripts/05_paraphrase_eval_set.py            # 1 paraphrase per existing query
    uv run python scripts/05_paraphrase_eval_set.py --regen    # discard cache and re-call Gemini

Pre-requisite:
    Add `GEMINI_API_KEY=...` to repo-root `.env` (free key:
    https://aistudio.google.com/apikey).

For each existing query in `data/eval/queries.jsonl`, generates ONE paraphrase in
recruiter-natural language and appends it to `queries.jsonl` with qid `<orig>_p`.
The `qrels.txt` is extended so paraphrased queries inherit their source query's
gold + relevant doc set (the anchor entity is the same).

Cost: 100 queries x ~150 tokens via Gemini 2.5 Flash-Lite. Free tier ceiling
is 1,000 RPD; with retries we may need 2 days to fill the full stratum.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Literal

import typer
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from app.eval import testset_gen
from app.services.llm import structured


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pnpm-workspace.yaml").exists():
            return parent
    raise RuntimeError("repo root marker not found")


REPO_ROOT = _find_repo_root()
EVAL_DIR = REPO_ROOT / "data" / "eval"
PARAPHRASE_CACHE = EVAL_DIR / "paraphrase_cache.jsonl"

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)


# ─── Output schema (Instructor / Pydantic) ───────────────────────────────────


class ParaphrasedQuery(BaseModel):
    """A natural-language recruiter-style paraphrase of a structured query."""

    paraphrased_text: str = Field(
        ...,
        description=(
            "A natural recruiter phrasing of the input query, 8-25 words. "
            "Preserve the same intent, skills, and proficiency level. "
            "Vary sentence structure, vocabulary, and tone — sound like a recruiter "
            "writing in Slack or a hiring manager describing a role to a friend, "
            "not a search engine input."
        ),
        min_length=15,
        max_length=400,
    )
    style: Literal["recruiter_brief", "casual_seeking", "formal_role_brief", "skills_focus"] = (
        Field(
            ...,
            description="The style category of the paraphrase (for downstream stratification).",
        )
    )


# ─── Prompts ─────────────────────────────────────────────────────────────────

CANDIDATE_SYSTEM_PROMPT = """\
You are a senior technical recruiter writing search queries to find candidates.
You will see a structured query and rewrite it as natural recruiter language.

Rules:
- KEEP the same intent. Same skills. Same proficiency requirement.
- Vary sentence structure, word choice, and tone.
- Sound like a real recruiter — NOT a search engine.
- 8 to 25 words. Single sentence preferred.
- Don't add new requirements. Don't drop required ones.
- Do NOT mention the word "engineer" if the original didn't list a target role; \
infer a sensible role from the skills if natural.
"""

JOB_SYSTEM_PROMPT = """\
You are a senior technical recruiter searching for jobs that fit a candidate.
You will see a structured query and rewrite it as natural recruiter language.

Rules:
- KEEP the same intent. Same designation. Same skills.
- Vary sentence structure and tone.
- Sound like a real recruiter — NOT a search engine.
- 8 to 25 words. Single sentence preferred.
- Don't add new requirements. Don't drop required ones.
"""


def _user_prompt(orig_text: str, query_skills: list[str], task: str) -> str:
    examples = (
        '"Senior Java engineer who can mentor a backend team"\n'
        '"anyone competent in regulatory affairs + brand marketing for a cosmetics PM seat?"\n'
        '"Looking for a strong Selenium tester with Azure exposure for a Bengaluru contract"\n'
        '"Need someone who can run procurement + supply chain at a senior IC level"'
    )
    return (
        f"Original query (task={task}):\n  {orig_text}\n\n"
        f"Skills present in the query: {', '.join(query_skills)}\n\n"
        f"Example styles of recruiter natural language:\n{examples}\n\n"
        f"Now produce ONE paraphrase. Pick a style that best fits the query."
    )


# ─── Cache helpers (so re-runs after a partial failure are cheap) ────────────


def _load_cache() -> dict[str, dict]:
    if not PARAPHRASE_CACHE.exists():
        return {}
    cache: dict[str, dict] = {}
    with PARAPHRASE_CACHE.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            cache[row["source_qid"]] = row
    return cache


def _append_cache(row: dict) -> None:
    PARAPHRASE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with PARAPHRASE_CACHE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ─── Main ────────────────────────────────────────────────────────────────────


def generate_paraphrases(force: bool = False) -> dict:
    queries_df = testset_gen.load_queries()
    qrels = testset_gen.load_qrels()
    cache = {} if force else _load_cache()

    console.print(
        f"[cyan]paraphrase[/] {len(queries_df)} source queries · "
        f"cached: {len(cache)} · to-call: {len(queries_df) - len(cache)}"
    )
    if force and PARAPHRASE_CACHE.exists():
        PARAPHRASE_CACHE.unlink()

    # Filter to non-paraphrased queries (in case this script gets re-run after partial)
    source_queries = [
        r for r in queries_df.to_dict(orient="records") if not r["qid"].endswith("_p")
    ]
    to_generate = [q for q in source_queries if q["qid"] not in cache]

    failures: list[tuple[str, str]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]paraphrasing[/] {task.description}"),
        TextColumn("· {task.completed}/{task.total}"),
        console=console,
    ) as prog:
        task_id = prog.add_task("via Gemini Flash-Lite", total=len(to_generate))
        for q in to_generate:
            sys_prompt = (
                CANDIDATE_SYSTEM_PROMPT if q["task"] == "candidate_search" else JOB_SYSTEM_PROMPT
            )
            user = _user_prompt(q["text"], q.get("query_skills") or [], q["task"])
            try:
                paraphrase = structured(
                    group="paraphraser",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user},
                    ],
                    response_model=ParaphrasedQuery,
                    temperature=0.4,
                    max_tokens=256,
                )
                row = {
                    "source_qid": q["qid"],
                    "paraphrased_qid": f"{q['qid']}_p",
                    "task": q["task"],
                    "anchor_id": q["anchor_id"],
                    "original_text": q["text"],
                    "paraphrased_text": paraphrase.paraphrased_text,
                    "style": paraphrase.style,
                    "query_skills": q.get("query_skills"),
                }
                _append_cache(row)
                cache[q["qid"]] = row
            except Exception as exc:
                failures.append((q["qid"], str(exc)[:200]))
                prog.console.print(f"[yellow]⚠ {q['qid']}: {str(exc)[:100]}[/]")
            prog.update(task_id, advance=1)
            # Gemini Flash-Lite free tier = 15 RPM. Sleep 4.2s between requests
            # to stay safely under (~14.3 RPM steady-state).
            time.sleep(4.2)

    if failures:
        console.print(f"[yellow]⚠ {len(failures)} queries failed:[/]")
        for qid, err in failures[:5]:
            console.print(f"   {qid}: {err}")

    # Write augmented eval set
    paraphrased_count = _write_augmented_eval_set(queries_df, qrels, cache)
    return {
        "source_queries": len(source_queries),
        "cached_or_generated": len(cache),
        "paraphrased_added_to_eval_set": paraphrased_count,
        "failures": len(failures),
    }


def _write_augmented_eval_set(
    queries_df, qrels: dict[str, dict[str, int]], cache: dict[str, dict]
) -> int:
    """Append paraphrased queries to queries.jsonl and qrels.txt."""
    queries_path = EVAL_DIR / "queries.jsonl"
    qrels_path = EVAL_DIR / "qrels.txt"

    # Build the (deduplicated) full set: source rows from existing file + paraphrases.
    existing_rows: list[dict] = []
    seen_qids: set[str] = set()
    if queries_path.exists():
        with queries_path.open(encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                row = json.loads(line)
                # Drop any pre-existing paraphrase rows so we always emit the latest.
                if row["qid"].endswith("_p"):
                    continue
                if row["qid"] in seen_qids:
                    continue
                existing_rows.append(row)
                seen_qids.add(row["qid"])

    paraphrase_rows: list[dict] = []
    new_qrels: dict[str, dict[str, int]] = {}
    for src_qid, p in cache.items():
        new_qid = p["paraphrased_qid"]
        # Mirror qrels: paraphrased query inherits the source query's relevances.
        if src_qid in qrels:
            new_qrels[new_qid] = dict(qrels[src_qid])
        paraphrase_rows.append(
            {
                "qid": new_qid,
                "task": p["task"],
                "text": p["paraphrased_text"],
                "anchor_id": p["anchor_id"],
                "stratum": "paraphrase",
                "paraphrase_style": p["style"],
                "source_qid": src_qid,
                "query_skills": p.get("query_skills"),
            }
        )

    # Persist queries.jsonl (rewritten, deduped, original + paraphrase rows)
    with queries_path.open("w", encoding="utf-8") as f:
        for r in existing_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in paraphrase_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Append paraphrase qrels (preserving any earlier entries)
    existing_qrels_text = qrels_path.read_text(encoding="utf-8") if qrels_path.exists() else ""
    paraphrase_qid_set = {r["qid"] for r in paraphrase_rows}
    keep_lines = [
        ln
        for ln in existing_qrels_text.splitlines()
        if ln.strip() and ln.split()[0] not in paraphrase_qid_set
    ]
    with qrels_path.open("w", encoding="utf-8") as f:
        for ln in keep_lines:
            f.write(ln + "\n")
        for qid, rels in new_qrels.items():
            for doc_id, rel in rels.items():
                f.write(f"{qid} 0 {doc_id} {rel}\n")

    # Update metadata
    meta_path = EVAL_DIR / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["task_breakdown"] = {
        "candidate_search": sum(1 for r in existing_rows if r["task"] == "candidate_search"),
        "job_search": sum(1 for r in existing_rows if r["task"] == "job_search"),
        "paraphrase": len(paraphrase_rows),
    }
    meta["total_queries"] = len(existing_rows) + len(paraphrase_rows)
    meta["paraphrase_generator"] = {
        "model_group": "paraphraser (gemini-2.5-flash-lite)",
        "n_styles": 4,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    console.print(
        f"[green]✓[/] eval set augmented: "
        f"{len(existing_rows)} originals + {len(paraphrase_rows)} paraphrases "
        f"= {len(existing_rows) + len(paraphrase_rows)} total → {queries_path}"
    )
    return len(paraphrase_rows)


@app.command(name="run")
def run_cmd(
    regen: bool = typer.Option(False, "--regen", help="Discard cache; recall Gemini for all 100."),
) -> None:
    summary = generate_paraphrases(force=regen)
    console.print()
    console.print("[bold]Summary:[/]")
    for k, v in summary.items():
        console.print(f"  {k}: {v}")


@app.command(name="show")
def show_cmd(n: int = 10) -> None:
    """Print the first N paraphrases for a quick sanity check."""
    cache = _load_cache()
    for i, (_qid, p) in enumerate(list(cache.items())[:n]):
        console.print(f"[cyan]{p['paraphrased_qid']}[/] [{p['style']}]")
        console.print(f"  orig: {p['original_text']}")
        console.print(f"  para: {p['paraphrased_text']}")
        console.print()
        if i >= n - 1:
            break


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("run")
    app()
