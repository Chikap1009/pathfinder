# PathFinder — Architecture

> One-pager. Refresh per ADR; deep details live in [decisions/](decisions/).

## Top-level view

```
                ┌──────────────────────────────────────────────────────────┐
User query ───► │  STAGE 0 — Intent router (Instructor + Pydantic)         │
                │  classifies: structural | semantic | hybrid | clarify    │
                │  extracts JobQuery {location, min_years, skills, ...}    │
                └──────────────────────────────────────────────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              ▼                           ▼                           ▼
    ┌─────────────────┐       ┌─────────────────┐         ┌─────────────────┐
    │ STAGE 1A — BM25 │       │ STAGE 1B — Dense│         │ STAGE 1C — KG   │
    │ BM25S BM25+     │       │ BGE-M3 dense    │         │ Text2Cypher     │
    │ over Qdrant     │       │ + BGE-M3 sparse │         │ Llama-3.3-70B   │
    │ sparse vectors  │       │ HNSW M=16       │         │ on Neo4j        │
    │ → top 100       │       │ ef_search=128   │         │ → candidate IDs │
    └─────────────────┘       │ → top 100       │         └─────────────────┘
                              └─────────────────┘
              │                           │                           │
              └───────────────┬───────────┴───────────────────────────┘
                              ▼
                  ┌─────────────────────────┐
                  │ STAGE 2 — RRF fusion    │
                  │ k=60 (default),         │
                  │ DAT alt for ablation    │
                  │ → top 50 unique         │
                  └─────────────────────────┘
                              ▼
                  ┌─────────────────────────┐
                  │ STAGE 3 — Cross-encoder │
                  │ bge-reranker-v2-m3      │
                  │ FP16 on RTX 4060        │
                  │ → top 10                │
                  └─────────────────────────┘
                              ▼
                  ┌─────────────────────────┐
                  │ STAGE 4 — Explanation   │
                  │ Gemini 2.5 Flash, JSON  │
                  │ schema-bound, post-hoc  │
                  │ NER-grounded faithful-  │
                  │ ness check via RAGAS    │
                  └─────────────────────────┘
                              ▼
                          UI (SSE)
```

## Latency budget (target p95 < 2 s)

| Stage | Budget | Notes |
| ----- | -----: | ----- |
| Intent + filter extraction | 150 ms | Groq Llama-3.1-8B-Instant |
| BGE-M3 encode (single query) | 25 ms | Local FP16 on RTX 4060 |
| BM25 + dense retrieval (100 each) | 20 ms | Qdrant Query API parallel |
| KG Cypher (when invoked) | 250 ms | Groq Llama-3.3-70B + Bolt round-trip |
| RRF fusion | 5 ms | Server-side in Qdrant |
| Cross-encoder rerank (top 50) | 150 ms | bge-reranker-v2-m3 FP16, batch |
| Explanation generation (top 10) | 800 ms | Gemini 2.5 Flash, JSON schema |
| RAGAS faithfulness check | 200 ms | Async, post-hoc; non-blocking on UI |
| **Total** | **~1.6 s p95** | |

## Knowledge graph schema

Pivoted from the original LinkedIn-style design after Day-1 schema introspection
revealed the dataset is skills-and-roles-focused (no company / industry / location
data). See [decisions/0002-skills-pivot.md](decisions/0002-skills-pivot.md).

**Nodes**: `Person, Skill, Role`. Each carries an `embedding` (BGE-M3 dense, 1024-d).

Relations (with `confidence`, `evidence_chunk_id`):

- `(Person)-[:HAS_SKILL {category, proficiency, source}]→(Skill)`
  · `category ∈ {core, secondary, soft}` · `proficiency ∈ {1..4}`
  (1 = Beginner · 2 = Advanced Beginner · 3 = Competent · 4 = Advanced)
- `(Person)-[:CAN_FILL {rank}]→(Role)` *(rank = position in `potential_roles` list)*
- `(Skill)-[:RELATED_TO {weight}]→(Skill)` *(seeded from ESCO `relatedSkill`)*
- `(Skill)-[:CHILD_OF]→(Skill)` *(ESCO hierarchy)*
- `(Person)-[:NEIGHBOR {ppr_score}]→(Person)` *(precomputed top-25 via NetworkX PPR)*

Dynamic schema policy: novel labels → `(:_PendingType {name, count, examples})`,
promoted to first-class once `count > 50` and a critic LLM validates. (This stays
relevant when we ingest ESCO skills and their long-tail relatedness data — novel
sub-categories will surface there.)

## Deployment topology

- **Web (Next.js)** → Vercel Hobby; auto-deploy on push.
- **API (FastAPI)** → HF Spaces Docker (port 7860); GitHub Actions `keepalive.yml` pings `/health` every 5 min.
- **Qdrant** → self-hosted in Docker locally; mirror to free Qdrant Cloud (1 GB) for the demo subset.
- **Neo4j** → self-hosted Neo4j 5 Community in Docker; AuraDB Free (5 k-profile demo subset only).
- **Redis** → Upstash Redis (10 k cmd / day) in production; local Docker for dev.
- **Observability** → Langfuse Cloud Hobby (50 k observations / month) via OTLP.

See [eval-methodology.md](eval-methodology.md) for the evaluation pipeline.
