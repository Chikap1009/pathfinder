# PathFinder

> Intent-aware, explainable hybrid retrieval for two-sided talent matching.

[![CI](https://github.com/Chikap1009/pathfinder/actions/workflows/ci.yml/badge.svg)](https://github.com/Chikap1009/pathfinder/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Next.js](https://img.shields.io/badge/next.js-16-black)

<!-- HERO_GIF_PLACEHOLDER — replace with apps/web/public/demo.gif -->

## What this is

PathFinder is a **two-sided retrieval engine** that matches candidates to jobs and jobs
to candidates. It was built to demonstrate production-grade retrieval engineering on a
realistic recruiter / talent-marketplace corpus — not toy data, not a single embedding
model thrown at a vector DB.

It understands queries on both sides:

- **Candidate-side**: _"Senior Python developer with cloud experience at Competent or
  higher"_ &middot; _"Find candidates who could fill a Regulatory Affairs Manager role"_
- **Job-side**: _"Show me jobs in Bengaluru asking for Selenium + Azure"_ &middot;
  _"Find roles similar to this Test Manager position but with React"_
- **Match**: _"Best candidates for job DBS-2025-2591399 and explain the fit"_

Each query is decomposed into structural filters (skills, proficiency, location,
experience range, designation) and a semantic intent (the prose summary or job
description). It retrieves through three parallel channels — **BM25**, **BGE-M3 dense**,
and a **Neo4j knowledge graph** (`Person → HAS_SKILL → Skill ← REQUIRES_SKILL ← Job`
with ESCO-canonicalised skills, locations, designations, and industries) — fuses with
**Reciprocal Rank Fusion** (k=60), and optionally reranks with a **bge-reranker-v2-m3**
cross-encoder. Every result carries a per-stage score breakdown plus matched-skill
evidence, streamed to the frontend over SSE so users see each stage land as it
completes.

## Dataset

The corpus is **two-sided**, sourced from a public dataset repository on GitHub:

| File | Rows / Items | Side | Notes |
| ---- | -----------: | ---- | ----- |
| `profiles.csv`     | 1,782 | candidate | id, name, skills (3-way: core / secondary / soft, with **Dreyfus 5-stage** proficiency tags: Beginner / Advanced Beginner / Competent / Proficient / Expert), years_of_experience, potential_roles, skill_summary |
| `demands_data.csv` | 1,081 | demand    | id, city / state / country, primary / secondary skills, experience range, designation |
| `jd_dataset.zip`   |   289 | JD        | per-job folder with `raw_jd.txt` (JSON: industry + raw text) and `enhanced_job_description.md` (LLM-enhanced sections: title, location, responsibilities, must-have / good-to-have skills) |

**Combined**: ~3,150 documents indexed across both sides. Skill nodes are the **join key** between people and jobs. See [docs/decisions/0003-two-sided-corpus.md](docs/decisions/0003-two-sided-corpus.md).

## Targets vs achieved

| Target | Goal | Achieved (RRF3) |
| ------ | ---: | --------------: |
| nDCG@10 | ≥ 0.55 | **0.557** ✅ |
| Recall@100 | ≥ 0.70 | **0.700** ✅ |
| MRR@10 | ≥ 0.55 | **0.591** ✅ |
| p95 latency (RRF3) | < 2 s | **30 ms** ✅ |
| p95 latency (full pipeline) | < 2 s | **315 ms** ✅ |

## Live links

| Surface | URL |
| ------- | --- |
| Web app (Vercel) | <https://pathfinder-web-wheat.vercel.app> |
| API (HF Spaces)  | <https://chikap1009-pathfinder-api.hf.space> |
| API docs (Swagger) | <https://chikap1009-pathfinder-api.hf.space/docs> |
| Eval dashboard   | <https://pathfinder-web-wheat.vercel.app/eval> |

> **Cold-start note:** the HF Space free tier sleeps after ~50 min idle; the keep-alive cron pings it every 5 min so the first request after a long pause may take ~30 s while BGE-M3 + the cross-encoder load.

## Architecture

<!-- ARCHITECTURE_SVG_PLACEHOLDER — replace with docs/diagrams/architecture.svg -->

See [docs/architecture.md](docs/architecture.md) and the
[ADR folder](docs/decisions/) for component-by-component justification.

## Ablation matrix

7 retrieval configurations × 3 strata, 119 held-out queries (50 candidate-search,
50 job-search, 19 natural-language paraphrase). Frozen snapshot — the live
[/eval dashboard](https://pathfinder-web-wheat.vercel.app/eval) shows the same
numbers plus the per-stratum breakdowns.

### Overall (mean across all 3 strata)

| Configuration                               | nDCG@10 |  R@10 | R@100 | MRR@10 | Latency |
| ------------------------------------------- | ------: | ----: | ----: | -----: | ------: |
| BM25                                        |   0.540 | 0.514 | 0.704 |  0.567 |  0.2 ms |
| BGE-M3 dense                                |   0.527 | 0.516 | 0.703 |  0.548 |  2.3 ms |
| RRF (BM25 + dense)                          |   0.551 | 0.521 | 0.696 |  0.585 |  2.5 ms |
| Cross-encoder rerank top-25                 |   0.538 | 0.535 | 0.696 |  0.551 |  285 ms |
| KG channel only                             |   0.425 | 0.443 | 0.686 |  0.456 |   25 ms |
| **RRF3 (BM25 + dense + KG)** 🏆             | **0.557** | **0.540** | 0.700 | **0.591** | **30 ms** |
| Full pipeline (RRF3 + rerank top-25)        |   0.544 | 0.536 | 0.700 |  0.565 |  315 ms |

**Key findings:**

1. **RRF3 wins on the overall mean** — adding the KG channel as a third RRF
   input lifts nDCG@10 from 0.551 → 0.557 and MRR@10 from 0.585 → 0.591 over
   the 2-channel RRF baseline. The graph signal genuinely complements the
   lexical + dense channels.
2. **The full pipeline trades nDCG for relevance ordering** — appending the
   cross-encoder rerank reduces nDCG slightly (0.557 → 0.544) on the overall
   mean but materially improves precision on the natural-language paraphrase
   stratum where lexical anchors are weak. Different best for different query
   distributions; the live dashboard surfaces both.
3. **KG-only is impressively strong on job_search** because `REQUIRES_SKILL`
   edges map directly to query skills. On candidate_search the ranking
   quality drops because most profiles match ≥ 1 query skill, and proficiency
   weighting alone can't fully order them.
4. **The cross-encoder is the latency tax** — 285 ms vs 0.2–30 ms for the
   retrieval-only configs. Worth it on hard queries; skip it for snappy
   demos by selecting the `rrf3` pipeline.

Encoding latency (BGE-M3 dense, FP16 on RTX 4060): ~1.7 ms / query, ~10 ms / doc
at index time. Cross-encoder (bge-reranker-v2-m3, FP16): ~14 ms per (query, doc)
pair at batch=32 on RTX 4060; ~80 ms on free-tier CPU. KG Cypher (AuraDB Free
over Bolt-TLS): ~25 ms / query for proficiency-weighted skill-overlap.

> *Paraphrase stratum size note:* generated 19/100 paraphrases before hitting
> the Gemini Flash-Lite free-tier daily quota. Stratum will grow once quota
> refills; methodology and ranking already established.

### Knowledge graph (Neo4j 5 Community)

Two-sided schema with skills as the join key. Loaded from DuckDB via
`scripts/07_kg_build.py` in ~26 s.

| Node label   | Count | Source                                      |
| ------------ | ----: | ------------------------------------------- |
| `Person`     | 1,782 | profiles.csv                                |
| `Job`        | 1,370 | demands_data.csv (1,081) + jd_dataset (289) |
| `Skill`      | 4,553 | canonical (alias 80 / cosine 581 / raw 4,498) |
| `Role`       |   834 | profiles.potential_roles                    |
| `Designation`|   415 | demands.designation                         |
| `Industry`   |    53 | jds.client_industry                         |
| `Location`   |   131 | demands.{city, country}                     |

| Relationship           | Count  |
| ---------------------- | -----: |
| `HAS_SKILL`            | 30,369 |
| `REQUIRES_SKILL`       |  5,129 |
| `CAN_FILL`             |  7,283 |
| `IS_DESIGNATION`       |  1,317 |
| `AT_LOCATION`          |  1,352 |
| `IN_INDUSTRY`          |    281 |

**Total: 9,138 nodes, 45,731 relationships.**

### Per-task & per-stratum splits

The live [/eval dashboard](https://pathfinder-web-wheat.vercel.app/eval)
shows the same ablation matrix broken out by candidate-search vs job-search,
plus the original-stratum (lexical-anchor) and paraphrase-stratum numbers.

**Honest reading of the dense vs BM25 gap:** BGE-M3 dense alone trails BM25
by ~1.3 pp nDCG@10 on the original (lexical-anchor) stratum because the eval
set was built from skill-name tokens so ground-truth relevance is reproducible
without an LLM judge — handing BM25 a structural advantage on lexical-overlap
matches. The semantic lift shows up exactly where you'd expect it: on the
natural-language paraphrase stratum, where dense + cross-encoder rerank are
the difference between misses and hits. RRF over both channels neutralises
the gap on either distribution at almost no latency cost.

Eval set: 119 deterministic queries (seed=42) split 50 / 50 / 19 across
candidate-search / job-search / paraphrase strata. Reproducible from
`apps/api/scripts/{02_bm25_baseline,03_dense_baseline,04_rerank_baseline,08_kg_baseline}.py`.

## Quick start

```bash
# 0. Prerequisites: WSL2 Ubuntu (or macOS / Linux), Node 20+, pnpm 9+, uv 0.11+, gh.
#    Optional: Docker Desktop if you want to run Qdrant / Neo4j locally
#    instead of pointing at AuraDB Free.
git clone https://github.com/Chikap1009/pathfinder.git && cd pathfinder
cp .env.example .env       # only GEMINI_API_KEY + NEO4J_* are required for retrieval

# 1. (optional) Bring up local infra if you don't want to use AuraDB Free
make up                    # qdrant + neo4j + redis

# 2. Place datasets — copy the three files into data/raw/:
#    profiles.csv, demands_data.csv, jd_dataset.zip
uv --directory apps/api run python scripts/00_inspect_csv.py
#    → emits apps/api/app/core/schema_map.yaml — review before continuing

# 3. ETL + index
uv --directory apps/api run python scripts/01_etl.py
uv --directory apps/api run python scripts/02_bm25_baseline.py
uv --directory apps/api run python scripts/03_dense_baseline.py
uv --directory apps/api run python scripts/06_skill_canonicalize.py
uv --directory apps/api run python scripts/07_kg_build.py --reset

# 4. Run the dev stack
pnpm install
make dev                   # FastAPI :8000 + Next.js :3000
```

For the full free-tier deploy (Vercel + HF Spaces + AuraDB Free), follow
[docs/deployment.md](docs/deployment.md) — combined cost $0/mo.

## Repo layout

```
pathfinder/
├── apps/
│   ├── api/        FastAPI 0.115+ + LangGraph + BGE-M3 + Qdrant + Neo4j
│   └── web/        Next.js 16 + shadcn/ui + Vercel AI SDK
├── packages/
│   └── shared-types/   (zod schemas mirrored across api/web — optional)
├── docs/
│   ├── architecture.md
│   ├── eval-methodology.md
│   └── decisions/      ADRs (one page per decision)
├── data/               raw / interim / processed / eval (gitignored)
├── docker-compose.yml  qdrant + neo4j + redis (+ api / web profile)
├── Makefile            up / down / dev / lint / test
└── pnpm-workspace.yaml
```

## Tech stack

### Live in the deployed pipeline

| Layer | Pick | Why |
| ----- | ---- | --- |
| Sparse retriever | **BM25S** (BM25+) | ~500× faster than `rank_bm25` on BEIR; no JVM. |
| Dense embedding | **BAAI/bge-m3** (568M, MIT) | Strong on lexical-light queries; 8192 ctx; runs FP16 on free-tier CPU. |
| Reranker | **bge-reranker-v2-m3** | Same family as the encoder; nDCG 0.6965 on NVIDIA RAG benchmark. |
| Dense index | **In-memory NumPy cosine** | Corpus is 3 k docs — Qdrant overhead isn't worth it; matrix multiply beats network round-trips on free CPU. |
| Fusion | **RRF (k=60)** | Score-agnostic; combines BM25 + dense + KG ranks. |
| Knowledge graph | **Neo4j 5** on AuraDB Free | 9.1 k nodes / 45.7 k rels; Cypher templates over Bolt-TLS. |
| Skill ontology | **ESCO** + alias YAML for canonicalisation | Free CC-BY; cross-side join key. |
| Intent extraction | **Instructor + Pydantic v2 + Gemini 2.5 Flash-Lite** via LiteLLM | Native `responseSchema`; ~150 ms p50 with `lru_cache` for duplicate queries. |
| Storage | **DuckDB** for entity meta, parquet for indexes | Self-contained — no DB server in the deploy. |
| Frontend | **Next.js 16** + Tailwind v4 + shadcn (new-york) | Industry default. |
| Streaming UI | **Vercel AI SDK 5** (typed UIMessage data parts) | Per-stage progress chips via SSE. |
| API | **FastAPI 0.115** + Pydantic v2 + uv | Async-native; auto-generated OpenAPI schema feeds the frontend types. |
| Deploy | **Vercel** (web) + **HF Spaces / Docker** (api) + **AuraDB Free** (kg) | Three free tiers; combined $0/mo with a keep-alive cron. |

### Used offline (data prep / eval only — not loaded at request time)

| Layer | Pick | Why |
| ----- | ---- | --- |
| Vector DB | Qdrant | Used during dense-index build for ablation experiments; deploy ships the cosine matrix instead. |
| KG extraction | LlamaIndex `SchemaLLMPathExtractor` | Triple-validated; novel types quarantined. |
| Eval | RAGAS 0.2 (Gemini judge) + `ranx` | Faithfulness / ContextRecall / nDCG / Recall on the frozen 119-query set. |
| Paraphrase generation | Gemini 2.5 Flash-Lite | Generated the 19-query NL stratum. |

## Deployment

Three free-tier targets — Vercel (web), Hugging Face Spaces / Docker (api),
Neo4j AuraDB Free (kg) — combined cost **$0/mo**. The HF Space Docker image
fetches precomputed retrieval indexes from a public GitHub release at build
time, so no LFS quota is consumed. A keep-alive cron pings `/health` and the
graph every 5 min to prevent free-tier sleeps. End-to-end guide:
[docs/deployment.md](docs/deployment.md).

## Licence

MIT. Model attributions: BGE-M3 (MIT), bge-reranker-v2-m3 (MIT), ESCO (CC-BY).
Dataset attribution: derived from a public GitHub dataset repository — see
[docs/decisions/0003-two-sided-corpus.md](docs/decisions/0003-two-sided-corpus.md).
