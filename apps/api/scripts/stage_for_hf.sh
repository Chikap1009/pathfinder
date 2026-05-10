#!/usr/bin/env bash
# Pre-stage the data artifacts that the Docker build needs.
#
# The Hugging Face Spaces deployment workflow does a `git subtree split
# --prefix=apps/api`, which strips everything outside `apps/api/`. The
# Dockerfile in apps/api needs to find `apps/api/data/` to COPY into the
# image — this script copies just the artifacts the runtime needs from the
# repo-root /data into apps/api/data/.
#
# Usage:
#   ./scripts/stage_for_hf.sh             # populates apps/api/data/
#   ./scripts/stage_for_hf.sh --clean     # remove the staged copy
#
# Idempotent. Run from anywhere inside the repo.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "$HOME/pathfinder")"
SRC="$REPO_ROOT/data"
DST="$REPO_ROOT/apps/api/data"

case "${1:-}" in
  --clean)
    echo "removing staged data: $DST"
    rm -rf "$DST"
    exit 0
    ;;
esac

if [ ! -d "$SRC" ]; then
  echo "ERROR: $SRC missing — run scripts/01_etl.py + 02_bm25_baseline.py + 03_dense_baseline.py + 06_skill_canonicalize.py first."
  exit 1
fi

mkdir -p "$DST/processed/bm25" "$DST/processed/dense" "$DST/processed/skills" "$DST/eval" "$DST/aliases"

# BM25 indexes.
cp -r "$SRC/processed/bm25/profiles_index" "$DST/processed/bm25/"
cp -r "$SRC/processed/bm25/jobs_index"     "$DST/processed/bm25/"

# Dense embeddings + ids.
cp "$SRC/processed/dense/profiles.npy"      "$DST/processed/dense/"
cp "$SRC/processed/dense/profiles_ids.json" "$DST/processed/dense/"
cp "$SRC/processed/dense/jobs.npy"          "$DST/processed/dense/"
cp "$SRC/processed/dense/jobs_ids.json"     "$DST/processed/dense/"

# Canonical skills (only the parquet — JSON is redundant in production).
cp "$SRC/processed/skills/canonical.parquet" "$DST/processed/skills/"

# DuckDB warehouse.
cp "$SRC/processed/pathfinder.duckdb" "$DST/processed/"

# Eval set (queries + qrels) so /v1/eval/summary numbers stay reproducible.
cp "$SRC/eval/queries.jsonl" "$SRC/eval/qrels.txt" "$SRC/eval/metadata.json" "$DST/eval/" 2>/dev/null || true

# Skill alias YAML.
cp "$SRC/aliases/skills.yaml" "$DST/aliases/" 2>/dev/null || true

echo "staged data:"
du -sh "$DST"/* 2>/dev/null || true
echo
echo "✓ apps/api/data/ ready for Docker build."
