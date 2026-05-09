---
title: PathFinder API
emoji: 🔎
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Intent-aware explainable hybrid retrieval — FastAPI backend.
---

# PathFinder API

FastAPI backend powering the PathFinder hybrid retrieval pipeline.

## Quick start (local)

```bash
# from repo root:  cd apps/api
uv python install 3.12          # one-time
uv sync                         # core deps only — fast, ~30s
uv run uvicorn app.main:app --reload --port 8000

# Then: open http://localhost:8000/docs
```

For full retrieval pipeline (Week 2+):

```bash
uv sync --all-extras            # adds ml, graph, llm, eval, jobs, obs
```

## Modules

```
app/
├── main.py                     FastAPI app, CORS, lifespan singletons
├── core/
│   ├── settings.py             pydantic-settings .env loader
│   ├── logging.py              structlog → OTel → Langfuse
│   ├── otel.py                 OpenTelemetry wiring
│   └── schema_map.yaml         Detected CSV column mapping (Day-1 Step 5)
├── api/v1/
│   ├── search.py               POST /v1/search       (Week 4)
│   ├── profile.py              GET  /v1/profile/{id} (Week 4)
│   ├── explain.py              POST /v1/explain      (Week 5)
│   ├── graph.py                GET  /v1/graph/...    (Week 4)
│   ├── eval.py                 GET  /v1/eval/run     (Week 5)
│   └── index.py                POST /v1/index/...    (Week 2)
├── graphs/
│   ├── retrieval_graph.py      LangGraph StateGraph (Week 4)
│   └── explain_graph.py
├── services/
│   ├── vector.py               BGE-M3 + Qdrant
│   ├── kg.py                   Neo4j driver + Cypher templates
│   ├── rerank.py               bge-reranker-v2-m3
│   ├── llm.py                  LiteLLM proxy client
│   ├── extractor.py            SchemaLLMPathExtractor wrapper
│   └── skills.py               ESCO + alias-table canonicalisation
├── models/                     Pydantic v2 schemas
└── eval/
    ├── ragas_pipeline.py
    ├── ir_metrics.py
    ├── testset_gen.py
    └── ablation.py
```

## Scripts (idempotent, Typer CLI)

| Script | Purpose |
| ------ | ------- |
| `00_inspect_csv.py`       | DAY 1: schema introspection → `app/core/schema_map.yaml` |
| `01_etl.py`               | CSV → DuckDB + Parquet, section-parsed |
| `02_bm25_index.py`        | BM25S index over canonical text |
| `03_dense_index.py`       | BGE-M3 → Qdrant (dense + sparse + ColBERT) |
| `04_kg_build.py`          | LLM extractor → Neo4j |
| `05_skill_canonicalize.py`| ESCO + alias YAML → canonical skill IDs |
| `06_ppr_offline.py`       | NetworkX Personalized PageRank → `:NEIGHBOR` edges |
| `07_eval_run.py`          | RAGAS + ranx ablation table |

## Deployment (HF Spaces)

`Dockerfile` exposes port 7860 (HF default). The repo-level
`.github/workflows/deploy-api.yml` does a `git subtree split` and force-pushes this
folder as a Space root.

Env keys live in HF Space *Settings → Variables and secrets* — never commit a `.env`.
