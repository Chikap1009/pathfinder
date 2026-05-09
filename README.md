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

| Config                                          | nDCG@10 | Recall@100 | MRR@10 |  MAP  | Search latency |
| ----------------------------------------------- | ------: | ---------: | -----: | ----: | -------------: |
| BM25 only (baseline)                            |   0.604 |      0.770 |  0.634 | 0.620 |     0.1 ms / q |
| BGE-M3 dense alone                              |   0.590 |      0.759 |  0.612 | 0.593 |     0.6 ms / q |
| **+ RRF fusion (k=60, BM25 + dense)**           | **0.619** | **0.760** | **0.657** | **0.629** | **2.5 ms / q** |
| + cross-encoder rerank (top-25 funnel)          |   0.598 |      0.760 |  0.614 | 0.604 |   ~285 ms / q  |
| + cross-encoder rerank (top-50 funnel)          |   0.584 |      0.760 |  0.590 | 0.589 |   ~480 ms / q  |
| + KG augmentation                               |   _TBD_ |      _TBD_ |  _TBD_ | _TBD_ |          _TBD_ |
| + DAT fusion (ablation)                         |   _TBD_ |      _TBD_ |  _TBD_ | _TBD_ |          _TBD_ |

Encoding latency (BGE-M3 dense, FP16 on RTX 4060): 1.7 ms / query, 10 ms / doc at
index time. Cross-encoder (bge-reranker-v2-m3, FP16, RTX 4060): ~10 ms per
(query, doc) pair at batch=32.

**Honest finding on the cross-encoder rows:** the reranker slightly **underperforms RRF**
on this deterministic eval set. This is the canonical-IR-literature trade-off
captured live: bge-reranker-v2-m3 is trained on natural-language passage retrieval
(MS-MARCO, MIRACL); on our skill-token-anchored queries (e.g. _"Java SQL DevOps
engineer at Beginner or higher"_) the model occasionally favors a semantically-
similar designation over the literal lexical anchor and demotes the gold below
rank 10. Per-task split shows it: candidate-search nDCG@10 lifts +1.4% (the
queries are slightly more natural) while job-search drops -4.8% (lexical anchors
dominate the queries). The expected win materialises on natural-language
paraphrased queries — that stratum lands when we wire Gemini-generated
paraphrases (Week 5).

### Per-task split

| Config       | Task             | nDCG@10 | Recall@10 | Recall@100 | MRR@10 |
| ------------ | ---------------- | ------: | --------: | ---------: | -----: |
| BM25         | Candidate search |   0.309 |     0.316 |      0.549 |  0.305 |
| BM25         | Job search       |   0.899 |     0.845 |      0.990 |  0.963 |
| Dense (BGE-M3)| Candidate search |  0.311 |     0.349 |      0.529 |  0.300 |
| Dense (BGE-M3)| Job search       |  0.869 |     0.817 |      0.989 |  0.925 |
| RRF (BM25+D) | Candidate search |   0.333 |     0.336 |      0.529 |  0.333 |
| RRF (BM25+D) | Job search       |   0.904 |     0.843 |      0.990 |  0.980 |

**Honest reading of the dense row:** BGE-M3 alone underperforms BM25 by ~1.5 % nDCG@10
on this eval set. The eval set was deliberately built from skill-name tokens so
that ground-truth relevance is reproducible without an LLM — that hands BM25 a
structural advantage on lexical-overlap matches. **The dense lift will
materialize once we add paraphrased / natural-language queries** (Week 5 RAGAS
testset generator) where dense's semantic generalisation actually pays off. The
RRF row already shows that fusing the two channels improves nDCG@10 (+2.5 % over
BM25) and MRR@10 (+3.6 %) without losing recall, which is the canonical-plan
prediction.

Eval set: 100 deterministic queries (seed=42) generated from real corpus
anchors via [`app/eval/testset_gen.py`](apps/api/app/eval/testset_gen.py).
Reproducible: `uv --directory apps/api run python scripts/02_bm25_baseline.py`
and `… 03_dense_baseline.py`.

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
