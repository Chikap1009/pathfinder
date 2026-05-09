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
and **BGE-M3 learned-sparse** — fuses with **Reciprocal Rank Fusion**, reranks with a
**bge-reranker-v2-m3** cross-encoder, and traverses a **Neo4j knowledge graph**
(`Person → HAS_SKILL → Skill ← REQUIRES_SKILL ← Job`, with ESCO-canonicalised skills,
locations, designations, and industries) for relational queries. Every result carries
a per-stage score breakdown, the matched skill / role evidence, a KG path trace, and
a natural-language explanation that's post-hoc faithfulness-checked with RAGAS.

## Dataset

The corpus is **two-sided**, sourced from a public dataset repository on GitHub:

| File | Rows / Items | Side | Notes |
| ---- | -----------: | ---- | ----- |
| `profiles.csv`     | 1,782 | candidate | id, name, skills (3-way: core / secondary / soft, with **Dreyfus 5-stage** proficiency tags: Beginner / Advanced Beginner / Competent / Proficient / Expert), years_of_experience, potential_roles, skill_summary |
| `demands_data.csv` | 1,081 | demand    | id, city / state / country, primary / secondary skills, experience range, designation |
| `jd_dataset.zip`   |   289 | JD        | per-job folder with `raw_jd.txt` (JSON: industry + raw text) and `enhanced_job_description.md` (LLM-enhanced sections: title, location, responsibilities, must-have / good-to-have skills) |

**Combined**: ~3,150 documents indexed across both sides. Skill nodes are the **join key** between people and jobs. See [docs/decisions/0003-two-sided-corpus.md](docs/decisions/0003-two-sided-corpus.md).

## Targets

Recall@100 ≥ 0.97 &nbsp;·&nbsp; nDCG@10 ≥ 0.55 &nbsp;·&nbsp; RAGAS Faithfulness ≥ 0.95 &nbsp;·&nbsp; p95 latency &lt; 2 s.

## Live links

| Surface | URL |
| ------- | --- |
| Web app (Vercel) | <!-- LINK_PLACEHOLDER --> |
| API (HF Spaces)  | <!-- LINK_PLACEHOLDER --> |
| KG demo (AuraDB) | <!-- LINK_PLACEHOLDER --> |
| Eval dashboard   | <!-- LINK_PLACEHOLDER --> |

## Architecture

<!-- ARCHITECTURE_SVG_PLACEHOLDER — replace with docs/diagrams/architecture.svg -->

See [docs/architecture.md](docs/architecture.md) and the
[ADR folder](docs/decisions/) for component-by-component justification.

## Ablation table

Each row is evaluated separately on **candidate-search** and **job-search**
(match-pair scoring lands in Week 3 with the KG row). RAGAS faithfulness is
populated once the explanation generator ships (Week 5).

### Overall (mean of candidate + job tasks)

| Config                              | nDCG@10 | Recall@100 | MRR@10 | RAGAS Faithfulness | p95 latency |
| ----------------------------------- | ------: | ---------: | -----: | -----------------: | ----------- |
| **BM25 only (baseline)**            |   0.604 |      0.770 |  0.634 |              _TBD_ | 0.1 ms / q  |
| + BGE-M3 dense                      |   _TBD_ |      _TBD_ |  _TBD_ |              _TBD_ | _TBD_       |
| + RRF (k=60)                        |   _TBD_ |      _TBD_ |  _TBD_ |              _TBD_ | _TBD_       |
| + cross-encoder rerank              |   _TBD_ |      _TBD_ |  _TBD_ |              _TBD_ | _TBD_       |
| + KG augmentation                   |   _TBD_ |      _TBD_ |  _TBD_ |              _TBD_ | _TBD_       |
| + DAT fusion (ablation)             |   _TBD_ |      _TBD_ |  _TBD_ |              _TBD_ | _TBD_       |

### BM25-only baseline split by task

| Task              | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | MAP   |
| ----------------- | ------: | --------: | ---------: | -----: | ----: |
| Candidate search  |   0.309 |     0.316 |      0.549 |  0.305 | 0.308 |
| Job search        |   0.899 |     0.845 |      0.990 |  0.963 | 0.932 |

The asymmetry is intentional: candidate queries are bag-of-skills against verbose
profile prose (BM25 over-matches common terms), while job queries include the
designation as a strong lexical key. BGE-M3 dense + cross-encoder rerank are
specifically expected to lift the candidate-search row — see
[docs/eval-methodology.md](docs/eval-methodology.md) for the protocol.

Eval set: 100 deterministic queries (seed=42) generated from real corpus
anchors via [`app/eval/testset_gen.py`](apps/api/app/eval/testset_gen.py).
Reproducible via `uv --directory apps/api run python scripts/02_bm25_baseline.py`.

## Quick start

```bash
# 0. Prerequisites: WSL2 Ubuntu, Node 20+ (via fnm), pnpm 9+, uv 0.11+, Docker Desktop, gh.
git clone https://github.com/Chikap1009/pathfinder.git && cd pathfinder
cp .env.example .env       # fill in API keys for Groq / Gemini / Cerebras / Langfuse

# 1. Bring up local infra
make up                    # qdrant + neo4j + redis

# 2. Place datasets — copy the three files into data/raw/:
#    profiles.csv, demands_data.csv, jd_dataset.zip
uv --directory apps/api run python scripts/00_inspect_csv.py
#    → emits apps/api/app/core/schema_map.yaml — review before continuing

# 3. ETL + index
uv --directory apps/api run python scripts/01_etl.py
uv --directory apps/api run python scripts/02_bm25_index.py
uv --directory apps/api run python scripts/03_dense_index.py

# 4. Run the dev stack
pnpm install
make dev                   # FastAPI :8000 + Next.js :3000
```

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

| Layer | Pick | Why |
| ----- | ---- | --- |
| Sparse retriever | **BM25S** (BM25+) | ~500× faster than `rank_bm25` on BEIR; no JVM. |
| Dense / sparse / multi-vector | **BAAI/bge-m3** (568M, MIT) | Three retrieval modes in one forward pass; 8192 ctx; 1.1 GB FP16. |
| Reranker | **bge-reranker-v2-m3** | Same family; nDCG 0.6965 on NVIDIA RAG benchmark. |
| Fusion | **RRF (k=60)** + DAT ablation | Score-agnostic; native in Qdrant Query API. |
| Vector DB | **Qdrant** | Sparse + dense + multivector + RRF in a single Query API call. |
| Knowledge graph | **Neo4j 5 Community** + AuraDB Free demo | Cypher + APOC; demo-subset on AuraDB. |
| KG extraction | LlamaIndex `SchemaLLMPathExtractor` | Triple-validated; novel types quarantined. |
| Skill ontology | **ESCO** + StackOverflow tag synonyms + alias YAML | Free CC-BY; 13.9 k skills. |
| Text2Cypher | Groq `llama-3.3-70b-versatile` (+ Gemini 2.5 Pro retry) | Sub-second TTFT; dynamic few-shot. |
| Intent extraction | Instructor + Pydantic v2 + Gemini 2.5 Flash-Lite | Native `responseSchema`. |
| Eval | RAGAS 0.2 (Gemini judge) + `ranx` | Faithfulness / ContextRecall / nDCG / Recall. |
| Frontend | Next.js 16 + Tailwind v4 + shadcn (new-york) | Industry default. |
| Streaming UI | Vercel AI SDK 5 (typed UIMessage data parts) | Per-stage progress chips. |
| Observability | Langfuse Cloud Hobby (50 k obs / mo) | OTLP-native. |

## Licence

MIT. Model attributions: BGE-M3 (MIT), bge-reranker-v2-m3 (MIT), ESCO (CC-BY).
Dataset attribution: derived from a public GitHub dataset repository — see
[docs/decisions/0003-two-sided-corpus.md](docs/decisions/0003-two-sided-corpus.md).
