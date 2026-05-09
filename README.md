# PathFinder

> Intent-Aware and Explainable Hybrid Retrieval System for people / profile search.

[![CI](https://github.com/Chikap1009/pathfinder/actions/workflows/ci.yml/badge.svg)](https://github.com/Chikap1009/pathfinder/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Next.js](https://img.shields.io/badge/next.js-15.5-black)

<!-- HERO_GIF_PLACEHOLDER — replace with apps/web/public/demo.gif (Week 6) -->

## Elevator pitch

PathFinder is a recruiter-grade people search engine over a **skills-first** profile
corpus. It understands queries like _"Senior Python developer with cloud experience at
Competent or higher"_ or _"Candidates who could fill a Regulatory Affairs Manager role,
strong on legal research"_, decomposes them into structural filters (skill + proficiency
+ role-fit) and semantic intent (the `skill_summary` prose), retrieves through three
parallel channels (BM25, BGE-M3 dense, BGE-M3 learned-sparse), fuses with **Reciprocal
Rank Fusion**, reranks with a **bge-reranker-v2-m3** cross-encoder, augments with a
**Neo4j** knowledge graph (Person → HAS_SKILL → Skill, Person → CAN_FILL → Role,
Skill ↔ ESCO ontology), and explains every result with citations, a per-stage score
breakdown, and the matched skill / role evidence.

**Dataset**: 1,782 profiles × 8 columns from HCLTech IIT Mandi Hack60 PS-1 (id, name,
core_skills, secondary_skills, soft_skills, years_of_experience, potential_roles,
skill_summary). Skills carry one of four proficiency tags: Beginner / Advanced Beginner /
Competent / Advanced. See [docs/decisions/0002-skills-pivot.md](docs/decisions/0002-skills-pivot.md).

**Targets**: Recall@100 ≥ 0.97 · nDCG@10 ≥ 0.55 · RAGAS Faithfulness ≥ 0.95 · p95 latency &lt; 2 s.

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

## Ablation table (filled in Week 2 / 5)

| Config                              | nDCG@10 | Recall@100 | RAGAS Faithfulness | p95 latency |
| ----------------------------------- | ------- | ---------- | ------------------ | ----------- |
| BM25 only (baseline)                | _TBD_   | _TBD_      | _TBD_              | _TBD_       |
| + BGE-M3 dense                      | _TBD_   | _TBD_      | _TBD_              | _TBD_       |
| + RRF (k=60)                        | _TBD_   | _TBD_      | _TBD_              | _TBD_       |
| + cross-encoder rerank              | _TBD_   | _TBD_      | _TBD_              | _TBD_       |
| + KG augmentation                   | _TBD_   | _TBD_      | _TBD_              | _TBD_       |
| + DAT fusion (ablation)             | _TBD_   | _TBD_      | _TBD_              | _TBD_       |

## Quick start

```bash
# 0. Prerequisites: WSL2 Ubuntu, Node 20+ (via fnm), pnpm 9+, uv 0.11+, Docker Desktop, gh.
git clone https://github.com/Chikap1009/pathfinder.git && cd pathfinder
cp .env.example .env       # fill in API keys for Groq / Gemini / Cerebras / Langfuse

# 1. Bring up local infra
make up                    # qdrant + neo4j + redis

# 2. Place dataset
#    Copy IITMandiHack60/profiles.csv to data/raw/profiles.csv
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
│   ├── api/        FastAPI 0.136 + LangGraph 1.1 + BGE-M3 + Qdrant + Neo4j
│   └── web/        Next.js 15.5 + shadcn/ui + Vercel AI SDK 5
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
| Knowledge graph | **Neo4j 5 Community** + AuraDB Free demo | Cypher + APOC; 5 k-profile demo on AuraDB. |
| KG extraction | LlamaIndex `SchemaLLMPathExtractor` | Triple-validated; novel types quarantined. |
| Skill ontology | **ESCO** + StackOverflow tag synonyms + alias YAML | Free CC-BY; 13.9 k skills. |
| Text2Cypher | Groq `llama-3.3-70b-versatile` (+ Gemini 2.5 Pro retry) | Sub-second TTFT; dynamic few-shot. |
| Intent extraction | Instructor + Pydantic v2 + Gemini 2.5 Flash-Lite | Native `responseSchema`. |
| Eval | RAGAS 0.2 (Gemini judge) + `ranx` | Faithfulness / ContextRecall / nDCG / Recall. |
| Frontend | Next.js 15.5 + Tailwind v4 + shadcn (new-york) | Industry default. |
| Streaming UI | Vercel AI SDK 5 (typed UIMessage data parts) | Per-stage progress chips. |
| Observability | Langfuse Cloud Hobby (50 k obs / mo) | OTLP-native. |

## Licence

MIT. Model attributions: BGE-M3 (MIT), bge-reranker-v2-m3 (MIT), ESCO (CC-BY).

## Acknowledgments

Built for **HCLTech IIT Mandi Hack60 PS-1**: _An Intent-Aware and Explainable Hybrid
Retrieval System_.
