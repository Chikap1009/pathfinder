# PathFinder — From Scratch

> A complete walkthrough of the project, written for someone who has never built a search engine, never touched a knowledge graph, and never heard of half the acronyms. By the end, you should be able to explain every line of every file and answer any reasonable interview question about the system.

**Table of contents**

1. [What problem are we solving?](#1-what-problem-are-we-solving)
2. [Background concepts (zero-to-functional understanding)](#2-background-concepts)
3. [The PathFinder architecture in one diagram](#3-the-pathfinder-architecture)
4. [The dataset](#4-the-dataset)
5. [The data pipeline (scripts 00→08)](#5-the-data-pipeline)
6. [The backend (FastAPI, retrieval engine, services)](#6-the-backend)
7. [The knowledge graph (Neo4j, Cypher, AuraDB)](#7-the-knowledge-graph)
8. [The frontend (Next.js, SSE, shadcn)](#8-the-frontend)
9. [Evaluation methodology and the ablation matrix](#9-evaluation)
10. [Deployment ($0/mo three-tier free stack)](#10-deployment)
11. [Problems we hit and how we solved them](#11-problems-we-hit)
12. [Technologies — a reference dictionary](#12-technologies-reference)
13. [The final result and the demo narrative](#13-the-final-result)
14. [Honest reflection — what's shipped, what's not, what's next](#14-honest-reflection)

---

## 1. What problem are we solving?

### The hiring matching problem

Recruiters and hiring managers spend most of their day on a deceptively hard search problem:

- **Recruiter side:** "I have an opening for a Senior Python developer in Bengaluru with cloud experience at Competent or higher proficiency — find me 10 candidates from our database."
- **Candidate side:** "I'm a Test Manager with 5 years' Selenium and Azure experience — show me jobs that need exactly this skill mix."
- **Match side:** "Here's job DBS-2025-2591399 and candidate person_181457 — is this a good fit? Why?"

This is **two-sided**: the same engine has to retrieve candidates given a job description and retrieve jobs given a candidate profile. It also has to handle natural-language phrasing (`"who knows Java microservices?"`) just as well as structured filters (`location=Bengaluru, experience≥5, must_have=React`).

### Why is this hard?

If you've used Google or any search box, you might think this is a solved problem. It isn't, for three reasons:

**1. Keyword search fails on synonyms and abbreviations.**
A candidate's profile says "Test Automation Specialist." A job posting asks for "QA Engineer with Selenium." Keyword search sees no overlap. They're talking about the same job.

**2. Pure semantic similarity ignores structural facts.**
A modern AI approach (embeddings, which we'll explain shortly) sees "Test Automation Specialist" and "QA Engineer with Selenium" as semantically close — good. But it also thinks "Senior Java Developer in Bangalore with 8 years' experience" and "Java intern in Mumbai" are quite similar — bad. The years-of-experience and location are critical hard constraints, not soft fuzzy ones.

**3. The user wants to know *why*.**
A recruiter looking at result #3 wants to see: this candidate matched because they have Selenium (Advanced), Azure (Beginner), and 5y experience. Pure search ranks without explanation. We need per-stage scores and matched evidence.

PathFinder addresses all three with a three-channel hybrid retrieval pipeline, an intent classifier in front, and a structured response that carries per-stage scores plus matched skills so the UI can render explainability.

### The corpus

We work with three datasets bundled into one two-sided corpus:

- `profiles.csv` — 1,782 candidate profiles, each with name, years of experience, skills (with Dreyfus 5-stage proficiency: Beginner / Advanced Beginner / Competent / Proficient / Expert), potential roles, and a textual skill summary.
- `demands_data.csv` — 1,081 demand records (think: HR system entries) with primary/secondary skill requirements, experience range, designation, and location.
- `jd_dataset.zip` — 289 job descriptions, each with a raw JSON and an LLM-enhanced markdown version (title, location, responsibilities, must-have/good-to-have skills, industry).

Combined: ~3,150 documents across both sides. Skills act as the join key — both candidates and jobs reference the same canonical skill IDs after we normalize them via the ESCO taxonomy.

---

## 2. Background concepts

If you've never seen these before, read this section carefully. The rest of the document assumes you understand these.

### 2.1 Information Retrieval (IR)

Information Retrieval is the field of computer science concerned with finding relevant documents from a large collection given a query. Google is IR. Spotify search is IR. PathFinder is IR.

The classic setup:
- **Corpus**: a collection of documents. (PathFinder: 1,782 profiles + 1,370 jobs.)
- **Query**: a user-supplied string. ("Senior Python developer in Bangalore.")
- **Retrieval**: pull the top-K documents (typically K=10 or K=100) ranked by relevance.

Retrieval is hard because "relevance" is fuzzy. A naive approach is keyword matching: return any document containing the query words. But "developer" matches every profile, "senior" matches half, and the result is useless.

### 2.2 Sparse retrieval — BM25

**Sparse retrieval** assigns each document a sparse vector — a list of (term, weight) pairs, one weight per word in the entire vocabulary. Most weights are zero (hence "sparse"). The classic algorithm is:

**TF-IDF** (Term Frequency – Inverse Document Frequency):
- TF = how often the term appears in this document.
- IDF = how rare the term is across the whole corpus.
- Score = TF × IDF. Common words (the, and, with) get low scores; distinctive words (Selenium, Kubernetes) get high scores.

**BM25** is an improved version of TF-IDF that adds two crucial corrections:
- Document length normalization (a 10,000-word document shouldn't dominate just because it contains "Selenium" 50 times).
- Term frequency saturation (the 11th mention of "Selenium" doesn't help as much as the 2nd).

BM25 is decades old but still the strongest baseline in IR. On most benchmarks, naive dense embedding models barely beat it. We use **BM25S** (the library), a pure-Python implementation that's ~500× faster than `rank_bm25` because it pre-computes term statistics as sparse matrices.

**In PathFinder:** `app/services/retrieval.py` loads two BM25 indexes — one for profiles, one for jobs — built by `scripts/02_bm25_baseline.py`. The function `_bm25_search(retriever, query)` tokenizes the query and returns the top-100 doc IDs with scores.

### 2.3 Dense retrieval — embeddings and transformers

**Dense retrieval** uses a neural network to map each document and each query to a fixed-size vector (typically 384 to 1024 numbers). Documents and queries that mean similar things get nearby vectors.

How? A transformer model (BERT-like) reads the text and produces an embedding — a numeric "thumbprint" of the meaning. We compare embeddings using cosine similarity (how aligned they are in vector space).

The model we use is **BGE-M3** (BAAI General Embedding, Multilingual, Multi-Function, Multi-Granularity). It's a 568-million-parameter transformer that produces:
- 1,024-dimensional dense embeddings.
- Optional learned-sparse embeddings (we don't use this in production).
- Optional multi-vector / ColBERT embeddings (we don't use this either).

We use just the dense mode. Each profile and each job is encoded once at index time; queries are encoded on the fly at retrieval time.

**Why "dense"?** Because nearly every dimension carries information — unlike BM25, where most positions are zero.

**Cosine similarity**: think of the embeddings as arrows from the origin. Cosine measures the angle between two arrows. 1.0 = same direction = same meaning; 0.0 = perpendicular = unrelated; -1.0 = opposite. In practice, we get values like 0.3 to 0.8 for relevant pairs.

**In PathFinder:** `_DenseIndex` (also in `retrieval.py`) loads two NumPy arrays — `profiles.npy` (1,782 × 1,024) and `jobs.npy` (1,370 × 1,024) — pre-computed by `scripts/03_dense_baseline.py`. At query time, we encode the query with BGE-M3, normalize it, take the dot product with the corpus matrix, and pick the top-K. With only ~3 k docs, this is a sub-millisecond operation on CPU.

> **Why not Qdrant?** Qdrant is a vector database optimized for billions of vectors with HNSW indexing. For 3 k vectors, the network round-trip costs more than the math. We use in-memory NumPy cosine. Qdrant *was* used during indexing experiments but is not in the production query path.

### 2.4 Cross-encoders — the reranker

A **bi-encoder** (like BGE-M3) encodes the query and each document independently, then compares them. Fast (you pre-compute doc embeddings once), but less accurate.

A **cross-encoder** processes the query and a candidate document together — concatenated into a single input — and outputs a relevance score. Much more accurate, much slower (you have to run the model once per (query, doc) pair).

**Use pattern:** retrieve top-100 with the fast bi-encoder, then rerank the top-25 with the cross-encoder. You get cross-encoder accuracy at bi-encoder speed (sort of).

We use **bge-reranker-v2-m3** — same family as BGE-M3, designed for reranking. nDCG 0.6965 on the NVIDIA RAG benchmark. On free-tier CPU it takes ~285 ms to rerank 25 candidates; on a GPU it would be ~14 ms.

**In PathFinder:** `app/services/rerank.py` loads the model lazily on first use, accepts a (query, [docs]) batch, and returns scores per doc.

### 2.5 Knowledge graphs and Neo4j

A **knowledge graph** is a database where data is stored as nodes (entities) and edges (relationships between them).

- A **node** has a label (`Person`, `Job`, `Skill`) and properties (name, years_experience).
- An **edge** has a type (`HAS_SKILL`, `REQUIRES_SKILL`, `AT_LOCATION`) and optional properties (confidence, proficiency).

Why a graph instead of just SQL tables?

Imagine asking: "Find me candidates who have *all* the skills required by job J42, who are located within commuting distance, in industries similar to J42's industry." In SQL, that's a multi-way join across 5+ tables with complex WHERE clauses. In a graph, you traverse: `(J42)-[:REQUIRES_SKILL]->(skill)<-[:HAS_SKILL]-(person)-[:AT_LOCATION]->(city)`. The query language reads exactly like the question.

**Neo4j** is the industry-standard graph database. It uses **Cypher** as its query language. Cypher syntax looks like ASCII art:

```cypher
MATCH (p:Person)-[:HAS_SKILL]->(s:Skill)<-[:REQUIRES_SKILL]-(j:Job {id: 'J42'})
RETURN p, count(DISTINCT s) AS overlap
ORDER BY overlap DESC
LIMIT 10
```

That query finds the top 10 people whose skills overlap with the skills required by job J42.

**AuraDB Free** is Neo4j's hosted free tier: 50,000 nodes, 175,000 relationships, auto-pauses after 3 days idle, connects via Bolt protocol over TLS (URI starts with `neo4j+s://`). PathFinder uses an AuraDB Free instance — we have 9,138 nodes and 45,731 relationships, comfortably under the cap.

**In PathFinder:** `app/services/kg.py` defines Cypher templates for the two retrieval queries we use — `search_persons_by_skills(skill_ids)` and `search_jobs_by_skills(skill_ids)`. These are *template-based*, not LLM-generated (no Text2Cypher in production).

### 2.6 Rank fusion — RRF

We have three channels (BM25, dense, KG), each producing its own ranked list of candidate IDs with its own score scale. BM25 scores are unbounded log-likelihoods, dense scores are bounded cosine in `[-1, 1]`, KG scores are integer skill-overlap counts. We can't just average them.

**Reciprocal Rank Fusion (RRF)** solves this. It throws away the scores and uses only the *ranks*:

```
RRF_score(doc) = Σ over channels [ 1 / (k + rank_in_channel) ]
```

Where `k=60` is a tuning constant (60 is the convention from the original Cormack et al. 2009 paper).

Intuition: a document ranked #1 by BM25, #5 by dense, and #2 by KG gets a high fused score. A document ranked #87 by all three gets a low one. Because we only use ranks, we don't care about score scales.

We call this `RRF3` when fusing all three channels, `RRF2` for BM25 + dense only.

**In PathFinder:** `_rrf_fuse(runs, k=60)` in `retrieval.py`. The function takes a list of rank dictionaries and returns a single fused dictionary.

### 2.7 Intent classification

Before we retrieve anything, we want to know: is the user looking for candidates or for jobs? Which canonical skills do they mention? Is there a minimum proficiency requirement?

We could write rules ("if the query starts with 'find candidates', set target_side='candidates'"), but natural language has too much variation. Instead, we use a small LLM call.

We use **Gemini 2.5 Flash-Lite** — Google's cheap, fast LLM (free tier on the Gemini API). We wrap it with **Instructor**, a library that forces the LLM to return JSON matching a specific **Pydantic** schema. So instead of free-form text, we get back:

```python
IntentResult(
    target_side="candidates",
    query_skills=["Selenium", "Azure"],
    min_proficiency="Any",
    designation_hint="Test Manager",
    free_text_intent="Test Manager Selenium Azure",
)
```

This takes ~150 ms p50 on the free tier. We cache identical queries via Python's `lru_cache` so duplicate requests are free.

**In PathFinder:** `app/services/intent.py` defines the system prompt and the `classify_intent(query)` function. It uses `app/services/llm.py` which is a LiteLLM wrapper for provider-agnostic LLM calls.

### 2.8 RAG (Retrieval-Augmented Generation)

A buzzword you'll hear constantly. **RAG = Retrieval + LLM**.

The pattern: instead of expecting an LLM to know everything from training, you retrieve relevant documents from a fresh corpus and stuff them into the LLM's prompt. The LLM generates an answer grounded in the retrieved docs.

PathFinder is **the retrieval half of a RAG system**. We don't generate natural-language answers (yet — that's the "explanation generator" feature we discussed as a future addition). We return ranked documents with per-stage scores.

If you wanted a full RAG, you'd add a step: pass the top-3 results to an LLM with a prompt like "Given these candidates, explain why each one matches the recruiter's query." That's straightforward to add but not currently shipped.

### 2.9 The web stack — FastAPI, Pydantic, DuckDB

**FastAPI** is a modern Python web framework. You write Python functions decorated with `@app.post("/search")` and FastAPI turns them into HTTP endpoints. It uses Python's type hints to auto-generate OpenAPI/Swagger documentation. Fast (built on async Python via `uvicorn`/`starlette`), well-typed, and extremely productive.

**Pydantic v2** is a data validation library. You declare data shapes as Python classes:

```python
class SearchRequest(BaseModel):
    query: str
    pipeline: Literal["bm25", "dense", "rrf", "rrf3", "rrf3_rerank"]
    top_k: int = 10
```

FastAPI uses Pydantic to validate every incoming request and produce JSON-schema OpenAPI specs. The frontend can then generate matching TypeScript types from the OpenAPI spec.

**DuckDB** is an embedded analytical SQL database — think SQLite but optimized for analytics. It runs in-process, requires no server, reads Parquet files directly, and executes complex aggregations on millions of rows in milliseconds. We use it as the single source of truth for entity metadata — when the retrieval engine needs to fetch a person's name or a job's description, it queries DuckDB.

The DuckDB file (`pathfinder.duckdb`) lives in `data/processed/` and is regenerated by `scripts/01_etl.py`.

### 2.10 The frontend stack — Next.js, SSE, Vercel AI SDK

**Next.js 16** is the React framework. It does server-side rendering, client-side hydration, file-based routing (`app/search/page.tsx` is the `/search` route), and integrates with Vercel for one-click deploys.

**Server-Sent Events (SSE)** is a browser-native protocol where the server keeps an HTTP connection open and streams text events to the client one at a time. Unlike WebSockets, SSE is one-way (server → client) and HTTP-native.

In PathFinder, `POST /v1/search/stream` returns an SSE stream where each event is a pipeline stage completing — first `intent`, then `bm25`, then `dense`, then `kg`, then `rrf`, then `rerank`, then `results`. The frontend renders these as progress chips so the user sees retrieval happen in real time (Perplexity-style).

**Vercel AI SDK 5** provides typed React hooks for consuming streaming AI responses. We use it on the frontend to subscribe to the SSE stream and update the UI as each stage lands.

### 2.11 The infra stack — Vercel, Hugging Face Spaces, AuraDB

**Vercel** hosts Next.js apps. Free tier (Hobby): unlimited static hosting, generous edge-function quotas, auto-deploys from GitHub. Our frontend lives at `https://pathfinder-web-wheat.vercel.app`.

**Hugging Face Spaces** hosts ML demos. Free tier: 2 vCPU, 16 GB RAM, sleeps after ~50 min idle. SDK options include Gradio, Streamlit, and **Docker** (which we use because FastAPI isn't Gradio). Our backend lives at `https://chikap1009-pathfinder-api.hf.space`.

**Docker** packages an application along with its dependencies into a portable image. HF Spaces takes our `Dockerfile`, builds the image (installs Python, torch, transformers, copies our code), and runs it on their CPU instances.

**uv** is a fast Python package manager (Rust-based, ~10× faster than pip). We use it to declare dependencies in `pyproject.toml` and lock them in `uv.lock`. The Dockerfile calls `uv sync --frozen` to install reproducibly.

**GitHub Actions** runs CI/CD. Our `.github/workflows/keepalive.yml` pings `/health` and the Neo4j instance every 5 minutes so neither free-tier service times out from idle.

---

## 3. The PathFinder architecture

Now that you understand the pieces, here's how they fit together.

```
                            ┌──────────────────────────────────────┐
   user query  ────────────►│  STAGE 0: Intent classifier          │
                            │  Gemini Flash-Lite + Instructor      │
                            │  → {target_side, skills, designation,│
                            │     min_proficiency, free_text}      │
                            └──────────────────────────────────────┘
                                            │
                            ┌───────────────┼────────────────┐
                            ▼               ▼                ▼
                  ┌─────────────────┐ ┌──────────────┐ ┌──────────────┐
                  │ STAGE 1A: BM25  │ │ STAGE 1B:    │ │ STAGE 1C: KG │
                  │ BM25S over      │ │ Dense        │ │ Cypher over  │
                  │ canonical text  │ │ BGE-M3 →     │ │ Neo4j        │
                  │ → top-100       │ │ NumPy cosine │ │ → ranked     │
                  └─────────────────┘ │ → top-100    │ │   doc IDs    │
                                      └──────────────┘ └──────────────┘
                            │               │                │
                            └───────────────┼────────────────┘
                                            ▼
                            ┌──────────────────────────────────────┐
                            │  STAGE 2: RRF (k=60)                  │
                            │  fuse rank lists → 1 ranked list      │
                            └──────────────────────────────────────┘
                                            │
                                            ▼
                            ┌──────────────────────────────────────┐
                            │  STAGE 3: Cross-encoder rerank       │
                            │  bge-reranker-v2-m3 FP16 on top-25   │
                            │  (skipped if pipeline=rrf3)          │
                            └──────────────────────────────────────┘
                                            │
                                            ▼
                            ┌──────────────────────────────────────┐
                            │  STAGE 4: Build SearchResult objects │
                            │  - matched skills (from intent)      │
                            │  - per-stage scores                  │
                            │  - entity meta from DuckDB           │
                            │  - snippet (first 220 chars)         │
                            └──────────────────────────────────────┘
                                            │
                                            ▼
                                       to the frontend
                              via SSE (`/v1/search/stream`)
                              or as one JSON (`/v1/search`)
```

Each stage produces an event on the SSE stream so the frontend can render progress chips. The user sees retrieval happen, not a 1-second spinner.

### Pipeline modes

The `pipeline` parameter on the request lets you skip stages:

| Mode | Stages used | When to use |
|---|---|---|
| `bm25` | 1A only | Lexical baseline; sub-millisecond |
| `dense` | 1B only | Semantic baseline; ~2 ms |
| `rrf` | 1A + 1B + 2 | BM25 + dense fusion |
| `rrf3` | 1A + 1B + 1C + 2 | Three-channel fusion (winner on overall mean) |
| `rrf3_rerank` | 1A + 1B + 1C + 2 + 3 | Full pipeline; best on paraphrase stratum |

We expose all five so the frontend can let users compare.

---

## 4. The dataset

### Profiles (1,782 candidates)

Source: `profiles.csv`. Columns:
- `id` — `person_<numeric>` slug.
- `name` — full name.
- `years_experience` — float.
- `skills` — JSON blob with three categories: core, secondary, soft. Each skill has a name and a Dreyfus 5-stage proficiency.
- `potential_roles` — list of job titles this profile could fill.
- `skill_summary` — LLM-generated natural-language summary of the candidate.
- `canonical_text` — concatenation of summary + roles + skills, used as the indexing text.

### Demands (1,081 records)

Source: `demands_data.csv`. Internal HR-system entries. Columns:
- `id` — `dem_<numeric>` slug.
- `title` — job title.
- `designation` — formal designation (e.g., "Senior Engineer L3").
- `industry`, `city`, `country`.
- `experience_lower`, `experience_upper` — min/max years required.
- `primary_skills`, `secondary_skills` — skill lists with importance levels.
- `canonical_text` — concatenation used as indexing text.

### JDs (289 job descriptions)

Source: `jd_dataset.zip`. Each entry is a folder with two files:
- `raw_jd.txt` — JSON with industry and raw description text.
- `enhanced_job_description.md` — LLM-enhanced markdown breaking out title, location, responsibilities, must-haves, good-to-haves.

After ETL, demands and JDs are unioned into a single `jobs` table (1,081 + 289 = 1,370 records).

### Skills (4,553 canonical)

Skills appear in both profiles (with proficiency) and jobs (with priority). We need them to refer to the same conceptual entity in the knowledge graph. So we run `scripts/06_skill_canonicalize.py` which:
1. Collects every distinct skill string.
2. Matches each against the **ESCO** (European Skills, Competences, Qualifications and Occupations) taxonomy — a free, EU-maintained ontology of ~13,900 occupational skills.
3. Falls back to alias YAML (manual hand-curated synonyms) for skills ESCO doesn't have.
4. Falls back to a cosine-similarity match using BGE-M3 embeddings for the rest.
5. Outputs `canonical.parquet` mapping every raw skill ID to a canonical ID.

The canonical set has 4,553 unique skills. Both candidates (`HAS_SKILL`) and jobs (`REQUIRES_SKILL`) connect to nodes in this set.

### Dreyfus 5-stage proficiency

A classic skill-acquisition model from the 1980s. Five stages:
1. **Beginner** — knows the rules, no experience.
2. **Advanced Beginner** — has hands-on experience but still rule-bound.
3. **Competent** — can plan and execute independently.
4. **Proficient** — sees the big picture, deviates from rules when needed.
5. **Expert** — intuitive grasp; the rules are now invisible.

In PathFinder, proficiency is stored on the `HAS_SKILL` edge as an integer 1–5, with a textual label.

---

## 5. The data pipeline

Eight scripts under `apps/api/scripts/`, run in order. Each is a Typer CLI — invoke via `uv --directory apps/api run python scripts/XX_name.py`.

### `00_inspect_csv.py` — schema introspection

Reads the raw CSVs and outputs `apps/api/app/core/schema_maps/<file>.yaml` describing every column's type and sample values. Run this once when you first place the data files in `data/raw/`. The output YAML is reviewed manually before continuing.

### `01_etl.py` — Extract, Transform, Load

This is the most important script. It does three things:

**Extract**: reads `profiles.csv`, `demands_data.csv`, `jd_dataset.zip`.

**Transform**:
- Section-parses each profile (splits the summary into core/secondary/soft skill groups).
- Normalizes locations (handles "Bengaluru" vs "Bangalore" via a city alias map).
- Normalizes experience ranges ("3-5 years" → lower=3, upper=5).
- Builds `canonical_text` for each record (concatenation of fields used for indexing).
- Writes intermediate parquets to `data/interim/`.

**Load**: writes a DuckDB file at `data/processed/pathfinder.duckdb` with tables:
- `profiles` (1,782 rows)
- `profile_skills` (30,369 rows — one per (person, skill) edge)
- `demands` (1,081 rows)
- `demand_skills`
- `jds` (289 rows)
- `jd_skills`
- `jobs` (1,370 rows — union of demands + jds)
- `job_skills` (5,129 rows)
- `skills` (4,553 canonical skill records)

> **Important fix during deployment:** the original ETL created VIEWS that pointed at the interim parquet files via absolute paths (`SELECT * FROM '/home/chirag/.../interim/profiles.parquet'`). When we shipped the `.duckdb` file to the production container, those parquet paths didn't exist and the views broke. Fix: changed `CREATE VIEW` → `CREATE OR REPLACE TABLE` so the data is materialized into the .duckdb file itself. We also added a `CHECKPOINT` to flush DuckDB's write-ahead log. This was the single bug that blocked deployment for ~30 minutes.

### `02_bm25_baseline.py` — BM25 indexing

Reads the `canonical_text` column from DuckDB. Tokenizes with stopword removal (English stopwords list from `bm25s`) and Porter stemming (so "developing" matches "developer"). Builds two BM25S indexes:
- `data/processed/bm25/profiles_index/` for candidates.
- `data/processed/bm25/jobs_index/` for jobs.

Each index is a directory of small files: a sparse term-frequency matrix, the vocabulary, params (k1=1.5, b=0.75 — standard BM25 hyperparameters), and the original corpus.

### `03_dense_baseline.py` — Dense embedding

Loads BGE-M3 (downloads ~1.1 GB on first run if not cached). For each row in `profiles` and `jobs`, encodes the `canonical_text` with the dense head of BGE-M3, normalizes to unit length, and stacks into a matrix:
- `data/processed/dense/profiles.npy` — (1,782, 1024) NumPy array, FP32.
- `data/processed/dense/profiles_ids.json` — list of doc IDs in the same order.
- `data/processed/dense/jobs.npy`, `jobs_ids.json` — same for jobs.

Total size: ~7 MB profiles + ~5 MB jobs.

### `04_rerank_baseline.py` — Cross-encoder eval

Runs the cross-encoder rerank pipeline against the eval set and writes the metrics to `data/processed/rerank/runs/<timestamp>/`. This is used to populate the ablation matrix; it's not in the request path.

### `05_paraphrase_eval_set.py` — Paraphrase generation

Takes the original 100-query eval set and asks Gemini Flash-Lite to paraphrase each query into a more natural-language form. ("Find candidates with Selenium and Azure who are Test Managers" → "I'm looking for a test management role person who knows Selenium automation and Azure cloud.") Generated 19/100 before hitting the Gemini free-tier daily quota. The 19 paraphrases form the "paraphrase stratum" in our eval set.

### `06_skill_canonicalize.py` — ESCO canonicalization

Builds the canonical skill catalog described above. Three-tier match (ESCO → alias YAML → BGE-M3 cosine) and outputs `canonical.parquet`.

### `07_kg_build.py` — Build the Neo4j graph

The big one. Connects to Neo4j (URI from `NEO4J_URI` env var), wipes everything if `--reset` is passed, and writes:

**Nodes:**
- `Person` (1,782) with properties: id, name, years_experience.
- `Job` (1,370) with properties: id, title, designation, city, country, exp range, source.
- `Skill` (4,553) with properties: id (canonical), name.
- `Role` (834) — distinct potential roles from profiles.
- `Designation` (415) — distinct designations from demands.
- `Industry` (53) — distinct industries.
- `Location` (131) — distinct (city, country) pairs.

**Relationships:**
- `(Person)-[:HAS_SKILL {proficiency, label}]->(Skill)` × 30,369
- `(Job)-[:REQUIRES_SKILL {priority}]->(Skill)` × 5,129
- `(Person)-[:CAN_FILL]->(Role)` × 7,283
- `(Job)-[:IS_DESIGNATION]->(Designation)` × 1,317
- `(Job)-[:AT_LOCATION]->(Location)` × 1,352
- `(Job)-[:IN_INDUSTRY]->(Industry)` × 281

**Total: 9,138 nodes, 45,731 relationships.** Builds in ~30 s over the AuraDB Bolt connection.

### `08_kg_baseline.py` — KG retrieval eval

Runs the KG-only retrieval channel against the eval set and writes metrics. Used to populate the "KG channel only" row in the ablation matrix.

---

## 6. The backend

The FastAPI app lives in `apps/api/app/`. Module structure:

```
app/
├── main.py                  # FastAPI entrypoint, lifespan, CORS, routers
├── core/
│   ├── settings.py          # pydantic-settings .env loader (Settings class)
│   ├── logging.py           # structlog → OTel → Langfuse (Langfuse not active)
│   └── otel.py              # OpenTelemetry tracer (no exporter in prod)
├── api/v1/
│   ├── schemas.py           # Pydantic models for request/response
│   ├── search.py            # POST /v1/search and /v1/search/stream
│   ├── profile.py           # GET /v1/profile/{id}
│   ├── job.py               # GET /v1/job/{id}
│   └── eval.py              # GET /v1/eval/summary
└── services/
    ├── retrieval.py         # RetrievalEngine — the main orchestrator
    ├── intent.py            # classify_intent() — Gemini call
    ├── dense.py             # BGE-M3 lazy loader
    ├── rerank.py            # bge-reranker-v2-m3 lazy loader
    ├── kg.py                # Cypher templates + Neo4j driver wrapper
    ├── skills.py            # canonical skill helpers (_slug normalization)
    └── llm.py               # LiteLLM wrapper + Instructor integration
```

### `main.py` — entrypoint and lifespan

FastAPI uses an **asynccontextmanager** called `lifespan` to run setup/teardown around the server's lifecycle. Ours does three things:

1. **Eager singletons**: opens the Qdrant client (optional; absent in prod), opens the Neo4j async driver. These are cheap and surface misconfiguration early.
2. **Lazy slots**: initializes `state["bge_m3"] = None` etc. so the warmup task can populate them.
3. **Warmup task**: spawns an asyncio Task that calls `get_engine()`, `get_dense_encoder()`, and `get_reranker()` in a thread. This loads the BM25 indexes, NumPy arrays, BGE-M3, and the cross-encoder while the server is already accepting requests. Crucially, we *schedule* the task — we don't *await* it — so uvicorn can start serving immediately. The first `/v1/search` will likely arrive after warmup completes.

We define routes:
- `GET /` → `{"name": "PathFinder API", "docs": "/docs"}`
- `GET /health` → status JSON used by the keep-alive cron.
- The v1 routers: `/v1/search`, `/v1/profile`, `/v1/job`, `/v1/eval`.

CORS is configured from `settings.cors_origin_list` so the Vercel frontend can call us.

### `services/retrieval.py` — the engine

This is the largest file (~500 lines) and the heart of the system. Key pieces:

**Path resolution.** `_find_repo_root()` walks up from `__file__` until it finds a `pnpm-workspace.yaml` marker. In production we touch this file in the Dockerfile so the function resolves to `/home/user/app/`. From there:

```python
REPO_ROOT = _find_repo_root()
DUCKDB_PATH = REPO_ROOT / "data" / "processed" / "pathfinder.duckdb"
BM25_DIR = REPO_ROOT / "data" / "processed" / "bm25"
DENSE_DIR = REPO_ROOT / "data" / "processed" / "dense"
SKILLS_PARQUET = REPO_ROOT / "data" / "processed" / "skills" / "canonical.parquet"
```

**Constants.**
- `CANDIDATE_BUDGET = 100` — top-K from each channel.
- `RRF_K = 60` — RRF tuning constant.
- `RERANK_TOP_N = 25` — how many candidates the cross-encoder sees.
- `SNIPPET_MAX_CHARS = 220` — truncated text in result cards.

**`_DenseIndex`** class. Loads a `.npy` matrix, normalizes rows to unit length, stores the doc-ID list. The `search(q, k)` method takes a query vector, normalizes it, computes `matrix @ q` (a single matrix-vector multiply — this is the dense similarity), uses `np.argpartition` for fast top-K selection.

**`_bm25_search(retriever, query, k)`** function. Tokenizes the query with `bm25s.tokenize` (stopwords + stemming), calls `retriever.retrieve(tokens, k=k)`, and converts the result to a `{doc_id: score}` dict.

**`_rrf_fuse(runs, k=60)`** function. Takes a list of rank dicts, iterates them, and accumulates `1 / (k + rank)` per doc.

**`RetrievalEngine`** class. Constructor loads:
- `self.bm25_p`, `self.bm25_j` — BM25 indexes for profiles, jobs.
- `self.dense_p`, `self.dense_j` — dense indexes.
- `self.skill_map` — dict from legacy skill ID slug → canonical ID (loaded from `canonical.parquet`).
- `self.entity_meta` — dict from doc_id → metadata dict (name, snippet, etc.) — populated by one pass over the DuckDB `profiles` and `jobs` tables.

Key methods:
- `search(query, intent, pipeline, top_k)` — synchronous, returns a `SearchResponse`.
- `search_stream(query, intent, pipeline, top_k)` — async generator yielding `StageEvent`s.
- `_query_canonical_skills(query_skills)` — maps the intent's raw skill names to canonical IDs.
- `_kg_search(target, skill_ids)` — calls into `services/kg.py`.
- `_build_result(rank, doc_id, score, stage_scores, intent)` — constructs the final `SearchResult` Pydantic object with matched_skills derived from the intersection of intent skills and the entity's canonical text.

**`get_engine()`** is a `functools.lru_cache`-wrapped factory so the engine is loaded exactly once per process.

`get_profile_detail(person_id)` and `get_job_detail(job_id)` are separate functions that query DuckDB for the full detail view used by the result drawer (skill list with proficiency, potential roles, etc.).

### `services/intent.py` — the LLM intent classifier

The system prompt is a careful piece of prompt engineering — it tells Gemini exactly how to classify and what fields to extract. Key rules:
- Atomic skill names only (no "Java developer" → "Java" + designation "developer").
- Title-case the skill names for consistency.
- `target_side`: "candidates" for "find" verbs, "jobs" for "openings" / "roles".
- `min_proficiency`: one of the 5 Dreyfus levels or "Any".

The function is `lru_cache`-wrapped so the same query string is free on repeat. There's a 6-second hard timeout — if Gemini doesn't respond, we fall back to a regex-based heuristic classifier.

### `services/llm.py` — LiteLLM wrapper

**LiteLLM** is a meta-library that gives you a single API for OpenAI, Anthropic, Google Gemini, Groq, Cerebras, Together, and many others. Our config (`MODEL_GROUPS`) groups models by purpose:

```python
MODEL_GROUPS = {
    "fast-small":    ["groq/llama-3.1-8b-instant", "gemini/gemini-2.5-flash-lite"],
    "big-reasoner":  ["groq/llama-3.3-70b-versatile", "gemini/gemini-2.5-pro"],
    "judge":         ["gemini/gemini-2.5-flash"],
    "explainer":     ["gemini/gemini-2.5-flash"],
    "paraphraser":   ["gemini/gemini-2.5-flash-lite"],
}
```

When code asks for "fast-small", LiteLLM tries Groq first (faster TTFT), falls back to Gemini if Groq is rate-limited or unauthorized. In production we only have `GEMINI_API_KEY` set, so Gemini handles everything.

`llm.py` wraps this with **Instructor** to force structured Pydantic output. Calling code looks like:

```python
result: IntentResult = call_with_schema(
    model_group="fast-small",
    schema=IntentResult,
    messages=[{"role": "system", "content": SYSTEM_PROMPT}, ...],
)
```

### `services/dense.py`, `services/rerank.py` — model loaders

Simple modules with `@functools.lru_cache`-wrapped factories:

```python
@functools.lru_cache(maxsize=1)
def get_dense_encoder() -> BGEM3FlagModel:
    from FlagEmbedding import BGEM3FlagModel
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
```

First call loads the model (~30 s cold start on free CPU); subsequent calls return the cached instance. The cross-encoder works the same way.

### `services/kg.py` — Neo4j wrapper

Defines Cypher templates and a thin wrapper around the Neo4j Python driver. Two main functions:

```python
def search_persons_by_skills(skill_ids: list[str], top_k: int) -> list[dict]:
    """
    MATCH (p:Person)-[r:HAS_SKILL]->(s:Skill)
    WHERE s.id IN $skill_ids
    WITH p, sum(r.proficiency) AS score
    ORDER BY score DESC
    LIMIT $top_k
    RETURN p.id AS doc_id, score
    """
```

The score is the sum of proficiency levels across matched skills. A candidate with Expert-level Selenium (5) and Competent Azure (3) gets a score of 8 for the query `[Selenium, Azure]`.

`search_jobs_by_skills` is symmetric but traverses `(Job)-[:REQUIRES_SKILL]->(Skill)`.

These are **template-based** — fixed Cypher with parameter substitution. No LLM in the loop (no Text2Cypher). That keeps latency at ~25 ms and makes the queries auditable.

### `services/skills.py`

Two helpers:
- `_slug(name)` — normalizes a skill string for matching: lowercase, strip non-alphanumeric, collapse whitespace.
- Other internal utilities for the canonical mapping.

### `api/v1/schemas.py` — request/response models

Pydantic v2 classes. The main ones:

- `SearchRequest` — `{query, pipeline, top_k, target_side?}`
- `IntentResult` — `{target_side, query_skills, min_proficiency, designation_hint, free_text_intent}`
- `SearchResponse` — `{query, pipeline, intent, n_candidates_retrieved, results, timings}`
- `SearchResult` — `{rank, score, entity, stage_scores}`
- `SearchResultEntity` — discriminated union of Person | Job
- `StageScores` — `{bm25, dense, kg, rrf, rerank}` (any may be null)
- `ProfileDetail`, `JobDetail` — full record schemas for the drawer
- `EvalSummary` — the eval endpoint's response (corpus stats + KG stats + ablation matrix + latency budget)

FastAPI introspects these and serves an OpenAPI schema at `/openapi.json`. The frontend pulls that schema and runs `openapi-typescript` to generate matching TS types.

### `api/v1/search.py` — the endpoint

```python
@router.post("", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    intent = _resolve_intent(req)
    engine = get_engine()
    return engine.search(req.query, intent, pipeline=req.pipeline, top_k=req.top_k)
```

The `/stream` variant returns a `StreamingResponse` with SSE-formatted events emitted as each stage completes. The frontend subscribes to this stream and updates the UI.

### `api/v1/profile.py`, `api/v1/job.py`

Single-record detail endpoints. Both call `get_profile_detail(id)` / `get_job_detail(id)` from `services/retrieval.py`, which queries DuckDB for the full record plus the skill list with proficiency labels.

### `api/v1/eval.py`

Returns the canonical eval snapshot — the ablation matrix shown on the `/eval` dashboard. The numbers are currently **hard-coded constants** in this module, mirroring the JSON artifacts under `data/processed/*/runs/<timestamp>/`. A future iteration could read the latest run JSON at request time; for now they're frozen in code so the dashboard is reproducible per commit.

---

## 7. The knowledge graph

We covered the schema above. A few extra notes on practical KG operation.

### Why a graph at all for this corpus?

A reasonable critic asks: "You have 1,782 profiles and 1,370 jobs. That's 2.4 million pairs, fits in memory, just brute-force compute similarity. Why bring in a database?"

Three answers:

**1. Relational queries are first-class.** "Find candidates qualified for jobs in Bangalore that require Python at Competent+." In SQL this is 4 joins. In Cypher it's:

```cypher
MATCH (j:Job)-[:AT_LOCATION]->(:Location {city: "Bangalore"})
MATCH (j)-[:REQUIRES_SKILL]->(s:Skill {name: "Python"})
MATCH (p:Person)-[r:HAS_SKILL]->(s)
WHERE r.proficiency >= 3
RETURN DISTINCT p
```

That reads exactly like the prose question.

**2. Scalability path.** This corpus is small. A real talent platform has 10M profiles and 1M jobs. The hybrid retrieval pattern (BM25 + dense for top-K, KG for filtering and explanation) scales by adding hardware to each channel independently. The KG handles the structural constraints that pure vector search can't.

**3. Explainability.** When we say "this candidate matched because they have these 3 of your 5 required skills with these proficiencies and they're in your target city," that explanation is a *graph path*. We can literally return the path traversed: `Person → HAS_SKILL → Skill ← REQUIRES_SKILL ← Job`. That's the structural backbone of explainability.

### AuraDB Free quirks

Setting up AuraDB Free taught us several non-obvious things:

- **Free instances don't auto-pause for the first 3 days but then pause indefinitely** unless you ping them.
- **The instance ID is the username and the database name.** The legacy `neo4j` username doesn't work. Instance `3090ad05` means `NEO4J_USER=3090ad05`, `NEO4J_DATABASE=3090ad05`.
- **The auto-generated password is shown exactly once.** Save it. (We learned this the hard way after the third re-setup.)
- **Connection URI uses `+s`** (`neo4j+s://`) for TLS — required for AuraDB. Plain `neo4j://` won't work.

### KG retrieval algorithm

The Cypher template does proficiency-weighted skill overlap. Pseudocode:

```
for each query_skill in [Selenium, Azure]:
    find all Person nodes with HAS_SKILL → query_skill
    for each Person:
        score += proficiency_value (1..5)
return Persons sorted by score, top 100
```

This is *not* sophisticated. It doesn't account for skill rarity (Selenium might be everywhere; a rare skill should weight more). It doesn't reward exact skill-set matches. But it's a strong baseline and the RRF fusion lets the other channels compensate for its weaknesses.

The matrix shows KG-only nDCG@10 of 0.425 — the weakest standalone channel. But adding it to RRF lifts the fused result from 0.551 (BM25+dense) to 0.557 (BM25+dense+KG). That marginal lift is exactly what RRF is designed to extract: the KG sees overlap patterns the lexical/semantic channels miss.

---

## 8. The frontend

`apps/web/` is a Next.js 16 App Router project. Structure:

```
apps/web/
├── app/                       # Routes (file-based)
│   ├── layout.tsx             # Root layout: top-nav, fonts, metadata
│   ├── page.tsx               # / — hero + KPI cards
│   ├── search/page.tsx        # /search — main interactive UI
│   ├── eval/page.tsx          # /eval — KPI tiles + ablation matrix
│   ├── architecture/page.tsx  # /architecture — pipeline ASCII diagram
│   ├── result/[id]/page.tsx   # /result/{person_id} — full profile detail
│   ├── graph/page.tsx         # /graph — 404 (was a placeholder)
│   └── globals.css            # Tailwind v4 base styles
├── components/
│   ├── layout/top-nav.tsx
│   ├── result-drawer.tsx      # Vaul-based bottom sheet for result detail
│   └── ui/                    # shadcn primitives
├── lib/
│   ├── api.ts                 # Typed fetcher to the backend
│   └── types.ts               # Re-exports of openapi-typescript schemas
├── public/
└── next.config.ts
```

### Pages walkthrough

**`/` (home)**
A marketing-style landing page. Hero text, three KPI cards (candidates / jobs / ESCO skills), four feature tiles (Hybrid retrieval / Knowledge graph / Cross-encoder rerank / Explainable). Server-rendered, no interactive elements beyond nav.

**`/search`**
The main feature. A search input + pipeline selector. On submit, opens an SSE connection to `POST /v1/search/stream` and renders each stage event as a progress chip ("Encode • 1201 ms", "BM25 • 0.7 ms", etc.) above the results. Results render as cards with rank, score, name, role, skill chips, and a snippet. Clicking a result opens the `result-drawer` with the full profile detail.

**`/eval`**
Server-rendered dashboard. Fetches `/v1/eval/summary` at build time, displays:
- 4 KPI tiles: nDCG@10 (best), Recall@100, MRR@10, Full pipeline p95 latency.
- 2 panels: Corpus stats, Knowledge graph stats.
- Ablation matrix: a `<table>` showing each row of the 7-config × 3-stratum matrix, with the best row in each stratum highlighted.

**`/architecture`**
Static stage table + ASCII pipeline diagram. We updated this during the polish pass to match the deployed pipeline (no Groq, no Qdrant, no Text2Cypher).

**`/result/[id]`**
Dynamic route. Fetches `/v1/profile/{id}` server-side, renders the full detail: name, years, skill summary, potential roles, core/secondary/soft skills with proficiency labels.

**`/graph`**
Was a placeholder ("Sigma.js + graphology ForceAtlas2 over the full 50 k-profile graph. Wired Week 6"). We removed it from the nav and made the route return 404, because it was misleading.

### SSE streaming implementation

The Vercel AI SDK 5 abstracts SSE consumption into a hook:

```typescript
const { messages, sendMessage } = useChat({
  api: "/api/search-proxy",
});
```

But because our SSE format is custom (per-stage events), we use a lower-level `EventSource` wrapper:

```typescript
const es = new EventSource(`${API_BASE_URL}/v1/search/stream?...`);
es.addEventListener("stage_done", (ev) => {
  const stage = JSON.parse(ev.data);
  setStages((prev) => [...prev, stage]);
});
es.addEventListener("results", (ev) => {
  setResults(JSON.parse(ev.data));
});
```

Each event has a `type` field (`intent`, `stage_start`, `stage_done`, `results`, `done`, `error`) that maps to a discriminated union in the TypeScript types generated from the OpenAPI spec.

### Styling

**Tailwind v4** for utility-first CSS. **shadcn/ui** (new-york preset) for accessible primitives (buttons, dialogs, scroll areas, tabs). **Vaul** for the bottom-sheet drawer that holds result details on mobile. **Lucide** for icons.

The dark theme is the default. Color tokens (`background`, `foreground`, `primary`, `muted-foreground`, `border`) are CSS custom properties defined in `globals.css`.

### Type-safety from backend to frontend

We use **openapi-typescript** to compile the FastAPI-generated OpenAPI spec into TS types. The pipeline:

1. Backend exposes `/openapi.json`.
2. A pre-commit hook runs `openapi-typescript https://chikap1009-pathfinder-api.hf.space/openapi.json -o apps/web/lib/api-schema.ts`.
3. `apps/web/lib/types.ts` re-exports the generated types with friendly aliases.
4. The `fetchSearch()` function in `lib/api.ts` is fully typed end-to-end.

If the backend changes a response shape, TypeScript complains at compile time on the frontend. Zero runtime surprises.

---

## 9. Evaluation

A retrieval system is only as good as its evaluation. Anyone can claim "our search is great"; you only believe them if they show numbers on a frozen test set.

### The test set — 119 queries

Built by `apps/api/app/eval/testset_gen.py`. Three strata:

**Stratum 1 — Candidate-search (50 queries).** Skill-token-anchored. Each query targets a specific (skill, proficiency) combination present in the corpus. Ground truth: the set of profiles in DuckDB that have that exact skill at that or higher proficiency.

**Stratum 2 — Job-search (50 queries).** Symmetric. Each query targets a (skill, location/industry/designation) combination. Ground truth: jobs in DuckDB matching those filters.

**Stratum 3 — Paraphrase (19 queries).** Generated by Gemini Flash-Lite paraphrasing strata 1 and 2 into natural language. ("Find candidates with Python at Competent or higher" → "I need someone who really knows Python, not a beginner.") Tests semantic generalization beyond keyword overlap.

All queries are deterministic — seed=42, reproducible from the script.

### Metrics

**Recall@K.** Of all the relevant docs in the corpus, what fraction did we put in the top K? Range [0, 1], higher is better. We report R@10 and R@100.

**Precision@K.** Of the docs we put in the top K, what fraction are relevant? Less informative for short ranked lists; we don't report this directly.

**Mean Reciprocal Rank (MRR).** For each query, take the rank of the first relevant doc and compute 1/rank. Average over queries. So if your first relevant doc is at position 1, MRR contribution is 1.0; position 2 → 0.5; position 10 → 0.1. Measures "did we put a relevant doc near the top?"

**nDCG@K (normalized Discounted Cumulative Gain).** The most informative metric. Each relevant doc earns gain (higher gain for "more relevant" docs if you have graded relevance — we use binary 0/1). Gain is discounted by log of position (a doc at position 10 contributes less than at position 1). Then normalized against an ideal ranking. Range [0, 1], higher is better. **nDCG@10 is our headline metric.**

**Mean Average Precision (MAP).** Average of precisions at each rank where a relevant doc appears. Robust to recall.

We compute these with **ranx**, a fast Python IR-evaluation library.

### The ablation matrix

Seven retrieval configurations, evaluated separately on all 3 strata and on the overall mean. The 7 configs:

| Config | Channels | Latency |
|---|---|---|
| BM25 | 1A | 0.2 ms |
| BGE-M3 dense | 1B | 2.3 ms |
| RRF (BM25 + dense) | 1A + 1B + RRF | 2.5 ms |
| Cross-encoder rerank top-25 | RRF + rerank | 285 ms |
| KG channel only | 1C | 25 ms |
| RRF3 (BM25 + dense + KG) | 1A + 1B + 1C + RRF | 30 ms |
| Full pipeline | RRF3 + rerank | 315 ms |

Each row × stratum combination is one cell with five metrics + latency. Total: 7 configs × 4 cuts (overall + 3 strata) × 5 metrics ≈ 140 numbers. They're all in `data/processed/*/runs/<timestamp>/<stratum>.json` and surfaced in `app/api/v1/eval.py`.

### Headline result

On the overall mean (mean of all 3 strata):
- **RRF3 is the best**: nDCG@10 = 0.557, MRR@10 = 0.591.
- Adding the cross-encoder rerank *reduces* overall nDCG slightly (0.557 → 0.544) but improves it on the paraphrase stratum where lexical anchors are weak.

**Honest reading.** The eval set is mostly skill-token-anchored, which gives BM25 a structural advantage on lexical-overlap matches. BGE-M3 dense trails BM25 by ~1.3 pp nDCG@10 on the original stratum. RRF over both channels neutralizes the gap. The cross-encoder's value is most visible on paraphrases. None of this is a surprise — it's exactly what you'd predict from the IR literature.

What the matrix *demonstrates* is that adding the KG as a third RRF channel improves the overall mean nDCG by +0.6 pp over the 2-channel baseline. Small but real. The KG signal is non-redundant with BM25 and dense.

---

## 10. Deployment

Three free tiers, $0/mo combined.

### Vercel (frontend)

Vercel detects the Next.js project, builds it with `next build`, and serves the static + edge-function output from their CDN. Auto-deploys on every push to `main`. The env variable `NEXT_PUBLIC_API_BASE_URL` points the frontend at the HF Space.

URL pattern: `https://pathfinder-web-<random>.vercel.app`. Ours is `pathfinder-web-wheat.vercel.app`.

### Hugging Face Spaces (backend)

HF Spaces lets you host ML demos. We use the **Docker** SDK so we can ship FastAPI + torch + transformers without HF's gradio/streamlit wrapper.

Our `apps/api/Dockerfile`:

```dockerfile
FROM python:3.12-slim AS base

# Cache-friendly env
ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# System deps
RUN apt-get update && apt-get install -y curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# uv (fast Python pkg manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

# Non-root user (HF Space requirement)
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Layer 1 — core deps
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

# Layer 2 — ML deps with CPU torch
RUN uv pip install --no-cache \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch==2.11.0 transformers sentence-transformers FlagEmbedding \
    bm25s PyStemmer scikit-learn neo4j ranx litellm instructor \
    google-generativeai opentelemetry-api opentelemetry-sdk \
    opentelemetry-instrumentation-fastapi opentelemetry-exporter-otlp langfuse

# Layer 3 — application code
COPY app ./app
COPY scripts ./scripts
COPY litellm.config.yaml ./

# Layer 4 — data artifacts from a GitHub release (HF LFS rejects binaries)
ARG DATA_RELEASE_URL=https://github.com/Chikap1009/pathfinder/releases/download/data-v1/pathfinder-data-v1.tgz
ARG DATA_REV=3
RUN mkdir -p data && curl -fsSL "$DATA_RELEASE_URL" | tar -xz -C data

# Marker so _find_repo_root() resolves to /home/user/app
RUN touch /home/user/app/pnpm-workspace.yaml \
    && chown -R user:user /home/user/app

USER user
EXPOSE 7860
CMD ["sh", "-c", "uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
```

Each layer is intentionally separated so unrelated changes don't invalidate the slow `torch` install.

URL pattern: `https://<username>-<space-name>.hf.space`. Ours is `chikap1009-pathfinder-api.hf.space`.

Free tier specs: 2 vCPU, 16 GB RAM, sleeps after ~50 min idle.

### Neo4j AuraDB Free

A managed Neo4j instance. Setup: 30 seconds at https://console.neo4j.io/. URI: `neo4j+s://3090ad05.databases.neo4j.io`. Connection from the HF Space via the Bolt protocol, authenticated with `NEO4J_USER=3090ad05` + the auto-generated password.

The KG is built by running `scripts/07_kg_build.py --reset` locally with the AuraDB env vars exported. Takes ~30 seconds to load 9 k nodes and 46 k relationships.

### Keep-alive cron

`.github/workflows/keepalive.yml` runs on a 5-minute cron schedule (`*/5 * * * *`):

```yaml
- name: Ping HF Space
  run: curl -fsS "$API_URL/health"

- name: Ping AuraDB
  run: |
    pip install neo4j
    python -c "
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        '$NEO4J_URI',
        auth=('$NEO4J_USER', '$NEO4J_PASSWORD')
    )
    with driver.session() as s:
        s.run('MATCH (n) RETURN count(n) LIMIT 1').single()
    "
```

This prevents the HF Space from sleeping (which would cause a 30-second cold start on the next request) and keeps AuraDB from auto-pausing after 3 idle days.

---

## 11. Problems we hit

A complete catalog of the bugs we encountered during deployment, with the fix and the lesson.

### Problem 1: HF Spaces rejected our binary files

**Symptom:** First push to the Space's git repo (`git push space main`) returned:
```
remote: Your push was rejected because it contains binary files.
remote: Please use Xet storage.
```

**Root cause:** HF Spaces free tier doesn't allow binary files >a few MB in the git LFS quota. Our `data/processed/` contained ~15 MB of NumPy arrays and DuckDB files.

**Fix:** instead of bundling data in the git repo, we:
1. Created a GitHub release (`data-v1`) and uploaded the data tarball there.
2. Modified the Dockerfile to `curl | tar -xz` the tarball at build time.
3. Used `huggingface_hub.upload_folder()` with `ignore_patterns=["data/**"]` so the Space only gets the source code.

**Lesson:** Free-tier deployments have hidden quotas. Plan for binary distribution separately from source.

### Problem 2: DuckDB views referencing absolute paths

**Symptom:** `/v1/search` returned 500. `/health` and `/v1/eval/summary` returned 200. After adding a debug endpoint, we saw:
```
_duckdb.IOException: IO Error: No files found that match the pattern
"/home/chirag/pathfinder/data/interim/profiles.parquet"
```

**Root cause:** `scripts/01_etl.py` created DuckDB *views* over the interim parquet files (`CREATE VIEW profiles AS SELECT * FROM '/home/chirag/.../profiles.parquet'`). These views store the absolute path. When we shipped the `.duckdb` file to the container, the interim parquets weren't there.

**Fix:** Two-part:
1. Immediate: open the duckdb in WSL and materialize every view as a table (`CREATE TABLE _new AS SELECT * FROM view; DROP VIEW view; ALTER TABLE _new RENAME TO view`). Re-tarred and re-uploaded.
2. Root cause: patched `01_etl.py` to use `CREATE OR REPLACE TABLE` instead of `CREATE OR REPLACE VIEW`, and added a `CHECKPOINT` at the end. Now future ETLs produce a portable .duckdb.

**Lesson:** Database files can carry implicit dependencies on the filesystem layout where they were built. Always materialize before shipping.

### Problem 3: AuraDB Free uses the instance ID as username and database

**Symptom:** Even with the correct password, `scripts/07_kg_build.py` got `Neo4j.ClientError.Security.Unauthorized`.

**Root cause:** Legacy AuraDB instances used `neo4j` as the default username and `neo4j` as the default database. New free instances use the instance ID (e.g., `3090ad05`) for both. We were setting `NEO4J_USER=neo4j`.

**Fix:** `export NEO4J_USER=3090ad05`, `export NEO4J_DATABASE=3090ad05`. The connection succeeded.

**Lesson:** Read the connection-info file the cloud provider gives you carefully. The username isn't always what you expect.

### Problem 4: GitHub username casing mismatch

**Symptom:** `huggingface_hub.upload_folder(repo_id="Chikap1009/pathfinder-api")` returned 401 Unauthorized.

**Root cause:** The Space was created under `ChiKap1009` (capital K), not `Chikap1009`. HF returns a 307 redirect to the canonical casing, but the upload_folder call doesn't follow redirects.

**Fix:** `repo_id="ChiKap1009/pathfinder-api"`. Upload succeeded.

**Lesson:** Cloud provider usernames can be case-sensitive in unexpected places. Check the canonical form.

### Problem 5: Empty tarball uploaded to GitHub release

**Symptom:** `gh release upload data-v1 /tmp/pathfinder-data-v1.tgz` succeeded, but `curl -fsIL <url> | grep content-length` showed 111 bytes.

**Root cause:** The `tar czf ... -C data .` command was run when `apps/api/data/` was empty (had been wiped by a previous `rm`). `tar` succeeded with no files, producing a 111-byte empty archive.

**Fix:** Re-ran `stage_for_hf.sh` to repopulate `apps/api/data/`, then re-tarred (got 8.9 MB), then `gh release upload --clobber` to replace the asset.

**Lesson:** Verify file sizes at every step. Empty tarballs upload as fast as full ones; you only notice when the downstream consumer fails.

### Problem 6: Docker layer caching prevented data refetch

**Symptom:** After fixing the DuckDB issue and re-uploading the tarball, the HF Space rebuild still produced a container with the old (broken) data.

**Root cause:** The Dockerfile line `RUN curl ... | tar -xz` is cached by Docker. Same line text → same layer hash → use cached result.

**Fix:** Added `ARG DATA_REV=N` to the Dockerfile and an `echo "data-rev=$DATA_REV"` step. Bumping `DATA_REV` changes the layer hash and forces re-execution. We're currently at `DATA_REV=3`.

**Lesson:** Docker caches aggressively. Plan for cache invalidation when the source is an external URL.

### Problem 7: Internal Server Error masked the real exception

**Symptom:** `/v1/search` returned `Internal Server Error` (the FastAPI default 500 body). No useful debug info.

**Root cause:** FastAPI's default exception handler logs the traceback to stdout but returns a generic 500 response. We had no easy access to the HF Space's stdout logs (the new HF UI hides them behind owner-only API endpoints we couldn't easily reach).

**Fix:** Added a temporary `/debug/diag` endpoint that calls `get_engine()` inside a try/except and returns the traceback in the JSON response. This is how we discovered the DuckDB path issue. After fixing, we removed the endpoint.

**Lesson:** When you can't access logs, you can write a debug endpoint that surfaces them. Just remember to remove it.

### Problem 8: Resume-readiness gaps (the polish round)

After deployment was working, an honest audit revealed:

- The `/graph` page was a placeholder ("Wired Week 6, 50 k-profile graph"). Corpus is 3.2 k. Misleading.
- The `/architecture` page listed Groq, Qdrant HNSW, Text2Cypher — none in the deployed pipeline.
- The README had `<!-- LINK_PLACEHOLDER -->` tokens instead of live URLs.
- The README's tech stack table included Langfuse, Cerebras, Groq — none configured in the deploy.
- The README's ablation matrix was from an older 100-query snapshot showing nDCG 0.621; the live dashboard was the 119-query snapshot showing nDCG 0.557.

**Fix (polish commit):**
1. Removed `/graph` from the nav and made the route return 404.
2. Rewrote `/architecture` to match the actual deployed pipeline (Gemini, in-memory NumPy, Cypher templates).
3. Filled in the README's live URLs.
4. Split README tech stack into "Live" vs "Used offline (data prep / eval only)".
5. Replaced the README ablation matrix with the current 119-query overall mean from the live dashboard, plus a goal-vs-achieved targets table.

**Lesson:** What you ship and what you document must match. A senior engineer reading your README will check 2-3 specific claims against the live system. Any mismatch erodes trust faster than a missing feature would.

---

## 12. Technologies — reference dictionary

Quick one-paragraph explanations of every library/service in the project. Use this as a glossary.

**Python 3.12** — language version. Newer syntax (PEP 695 generics) and faster startup.

**uv** — Rust-based Python package manager. ~10× faster than pip. Manages virtual envs, locks dependencies in `uv.lock`, runs scripts with `uv run`.

**FastAPI 0.115** — async Python web framework. Type-hint-driven, generates OpenAPI specs, integrates with Pydantic.

**Pydantic v2** — data validation library. Declares schemas as Python classes; validates at boundaries (HTTP, env vars, function args).

**pydantic-settings** — loads env vars into a Pydantic Settings model. Single source of config truth.

**DuckDB** — embedded analytical SQL database. Reads Parquet directly, runs aggregations on millions of rows in-process.

**NumPy** — array math. We use it for the dense retrieval (cosine via matrix-vector multiplication).

**pandas** — used briefly in `01_etl.py` for parquet roundtrips. Not in the request path.

**PyTorch 2.11 (CPU)** — deep learning framework. Used by BGE-M3 and the cross-encoder. We install the CPU-only wheel (~750 MB) since HF Spaces free tier has no GPU.

**transformers** — Hugging Face library wrapping pretrained transformer models with a consistent API.

**sentence-transformers** — convenience wrapper around transformers for sentence-embedding models.

**FlagEmbedding** — BAAI's library for BGE-family models (BGE-M3, bge-reranker-v2-m3). Provides the dense encoder and reranker.

**bm25s** — fast Python BM25 implementation. Uses pre-computed sparse matrices for ~500× speedup over `rank_bm25`.

**PyStemmer** — C-backed Porter stemmer. Used by bm25s.

**scikit-learn** — used for utility functions (`cosine_similarity`) in the skill canonicalization script.

**neo4j (Python driver)** — official Bolt-protocol client. Supports sync + async drivers.

**ranx** — IR evaluation library. Computes nDCG, MRR, Recall, MAP, etc., from runs and qrels.

**LiteLLM** — provider-agnostic LLM client. Lets us call Groq/Gemini/OpenAI/etc. through one API.

**Instructor** — adds Pydantic schema enforcement to LiteLLM. Ensures the LLM returns parseable JSON matching our types.

**google-generativeai** — Google's Gemini SDK. Used directly for the paraphrase script.

**OpenTelemetry** — vendor-neutral observability framework. We instrument FastAPI with auto-instrumentation; no exporter is active in production (would log to Langfuse if configured).

**Langfuse** — LLM observability platform. We pass through traces (configured) but no key is set in production, so it's a no-op.

**Next.js 16** — React framework with App Router, server components, file-based routing, edge functions. Hosted on Vercel.

**React 19** — UI library. Server components by default in Next.js 16.

**TypeScript 5.9** — adds static types to JavaScript. Used end-to-end in the frontend.

**Tailwind CSS v4** — utility-first CSS framework. We write classes like `flex items-center gap-2 text-sm`.

**shadcn/ui (new-york)** — copy-paste component library built on Radix UI primitives. We have Button, Card, Dialog, ScrollArea, Tabs, Tooltip, etc.

**Vaul** — bottom-sheet drawer component (used for the mobile result detail view).

**Lucide React** — icon library. Tree-shakeable SVG icons.

**cmdk** — command palette component (the `⌘K` palette in the top nav).

**Vercel AI SDK 5** — typed React hooks for consuming streaming AI responses. We use it as a thin wrapper over `EventSource` for our custom SSE format.

**openapi-typescript** — generates TS types from an OpenAPI 3.1 schema. Run as a pre-commit hook against `/openapi.json`.

**Vercel (Hobby)** — frontend hosting. Free for personal projects.

**Hugging Face Spaces (Docker, CPU basic)** — backend hosting. Free tier: 2 vCPU, 16 GB RAM.

**Neo4j AuraDB Free** — managed Neo4j. 50 k nodes / 175 k relationships, auto-pauses after 3 idle days.

**GitHub Actions** — CI/CD. We use it for the keep-alive cron and the optional deploy-api workflow.

**Docker** — containerization. The Dockerfile builds an image that bundles Python + uv + deps + our code.

**huggingface_hub** — Python SDK for HF. We use `upload_folder()` to push the Space.

**gh (GitHub CLI)** — for managing releases (`gh release upload`) and secrets (`gh secret set`).

**WSL2 (Ubuntu)** — Windows Subsystem for Linux. The dev environment we work in.

---

## 13. The final result

### What's live

| Layer | URL | Free tier |
|---|---|---|
| Frontend (Next.js) | https://pathfinder-web-wheat.vercel.app | Vercel Hobby |
| Backend (FastAPI) | https://chikap1009-pathfinder-api.hf.space | HF Spaces (Docker, CPU basic) |
| API docs | https://chikap1009-pathfinder-api.hf.space/docs | (same) |
| Knowledge graph | Neo4j AuraDB Free `3090ad05` | AuraDB Free |

### Headline numbers

| Metric | Goal | Achieved (RRF3, 119-query overall mean) |
|---|---:|---:|
| nDCG@10 | ≥ 0.55 | **0.557** ✅ |
| Recall@100 | ≥ 0.70 | **0.700** ✅ |
| MRR@10 | ≥ 0.55 | **0.591** ✅ |
| p95 latency (RRF3) | < 2 s | **30 ms** ✅ |
| p95 latency (full pipeline) | < 2 s | **315 ms** ✅ |
| Cost | $0 / mo | **$0** ✅ |

### Corpus and KG stats

| | Count |
|---|---:|
| Profiles | 1,782 |
| Jobs (demands + JDs) | 1,370 |
| Canonical skills | 4,553 |
| HAS_SKILL edges | 30,369 |
| REQUIRES_SKILL edges | 5,129 |
| Total KG nodes | 9,138 |
| Total KG relationships | 45,731 |
| Eval queries | 119 (50 + 50 + 19) |

### The demo narrative (use this when presenting)

> "PathFinder is an explainable hybrid retrieval system for two-sided talent matching. Given a natural-language query, it classifies intent, retrieves through three parallel channels — BM25, BGE-M3 dense embeddings, and a Neo4j knowledge graph with ESCO-canonicalized skills — fuses with Reciprocal Rank Fusion, optionally reranks with a cross-encoder, and streams per-stage scores back to the frontend so every result is auditable.
>
> The corpus is 1,782 candidate profiles and 1,370 job records joined through 4,553 canonical skills with Dreyfus 5-stage proficiency labels. On a frozen 119-query test set across three strata, the three-channel RRF configuration hits nDCG@10 of 0.557 and Recall@100 of 0.700 in 30 ms; the full pipeline with cross-encoder rerank hits 0.591 MRR@10 in 315 ms p95 — well inside the 2-second target.
>
> The whole system runs on three free tiers — Vercel for the Next.js frontend, Hugging Face Spaces with Docker for the FastAPI backend, and Neo4j AuraDB Free for the graph — combined cost $0 a month, kept warm by a 5-minute keep-alive cron."

---

## 14. Honest reflection

### What's deployed vs aspirational

The architecture documents and the original README describe a more ambitious system than what's currently in production. Here's the honest split:

**Shipped:**
- Three-channel hybrid retrieval (BM25 + dense + KG).
- RRF fusion (RRF3 with k=60).
- Cross-encoder rerank (lazy-loaded, used for `rrf3_rerank` pipeline).
- Intent classification with Gemini Flash-Lite.
- SSE streaming with per-stage events.
- Frozen 119-query ablation matrix.
- Two-sided support (candidate ↔ job).
- Vercel + HF + AuraDB free-tier deployment.
- Per-stage scores + matched skills in results.

**Not shipped (was in the original architecture):**
- **LLM-generated natural-language explanations** ("This candidate matches because they have Selenium at Expert and Azure at Competent, satisfying your must-haves; their 5 years of experience meets your minimum.") — this is the headline missing feature.
- **RAGAS post-hoc faithfulness check** on those explanations — depends on the explanation generator existing first.
- **Knowledge graph explorer page** — was a placeholder; now removed.
- **BGE-M3 learned-sparse and multi-vector channels** — we only use dense.
- **ColBERT multivec retrieval** — not implemented.
- **Text2Cypher with Llama-3.3-70B** — we use template Cypher.
- **Qdrant** — replaced with in-memory NumPy (defensible: 3 k vectors).
- **Langfuse observability** — configured pass-through but no key in production.
- **DAT (Distribution-Aware Tau) fusion** as an alternative to RRF — never implemented.
- **Paraphrase stratum at 100 queries** — only 19, due to Gemini free-tier daily quota.

### Trade-offs we'd defend

**1. In-memory NumPy over Qdrant.** Justified by corpus size (3 k docs). Qdrant's overhead would dominate on free-tier CPU. A senior engineer would expect this answer; we have it ready.

**2. Cypher templates over Text2Cypher.** Faster, more auditable, no LLM in the request path. The KG queries we evaluate are simple skill-overlap traversals; LLM-generated Cypher would be overkill and would introduce latency + a failure mode (malformed Cypher).

**3. Gemini-only over multi-provider.** LiteLLM is configured for Groq + Gemini fallback, but we only set `GEMINI_API_KEY` in the Space. Gemini Flash-Lite is on the generous free tier and adequate for intent classification.

**4. CPU-only.** Free tier has no GPU. The cross-encoder is therefore the bottleneck (~285 ms vs ~14 ms on a 4060). Worth it for the demo; would need a paid tier for real throughput.

### Roadmap (if you wanted to extend it)

In rough order of value:

1. **The explanation generator.** Wire `services/llm.py` into a `/v1/explain` endpoint that takes a (query, result, intent) tuple and returns a 2-sentence natural-language explanation. This turns "explainable hybrid retrieval" from a claim into a demonstration. ~1 day.

2. **A real KG explorer.** Add `/v1/graph/sample` that returns a curated subgraph; render with sigma.js on the frontend. ~1 day.

3. **Paraphrase stratum to 100 queries.** Re-run `05_paraphrase_eval_set.py` once Gemini quota resets. Re-run eval, update ablation matrix. ~1 hour.

4. **Result-card stage-score bars.** The data is in the response; the frontend just renders a flat list. Add a visualization (small horizontal bars for BM25 / dense / KG / RRF / rerank). ~2 hours.

5. **Top-50 retrieval funnel into rerank.** Currently rerank sees top-25; bumping to top-50 might lift nDCG on the paraphrase stratum at the cost of ~150 ms more latency.

6. **Cache layer.** Add Redis (or in-process LRU) in front of `/v1/search` keyed on `(query, pipeline, top_k)`. Demo-friendly: identical queries become instant.

### What this project shows about your skills

For an interviewer who's read this document, the project demonstrates:

- **Information retrieval fundamentals.** Sparse vs dense, RRF, cross-encoder, evaluation.
- **ML engineering.** Model loading, embedding indexing, latency budgeting, fp16 inference.
- **Backend engineering.** Async FastAPI, lifespan singletons, lazy model loading, SSE streaming.
- **Frontend engineering.** Next.js App Router, typed end-to-end via OpenAPI, streaming UI.
- **Data engineering.** ETL with DuckDB, parquet, idempotent scripts, schema introspection.
- **Knowledge graph design.** Schema modeling, ESCO canonicalization, Cypher templating, AuraDB ops.
- **Production deployment.** Docker layering, free-tier constraints, $0/mo three-tier stack, keep-alive operations.
- **Evaluation methodology.** Frozen test set, multi-stratum strata, ablation matrix, honest reporting of weaknesses.
- **Debugging skills.** The 8 problems above show real, sub-1-hour root-cause-and-fix loops on opaque cloud issues.

That's a lot for any candidate, let alone a fresher. Use this document to internalize the project, then when an interviewer asks "tell me about a project you built," you'll have specific facts at every level of abstraction — from "Recall@100 is 0.700" to "we materialize DuckDB views as tables so the .duckdb file is portable" to "the explanation generator is the next feature on the roadmap."

Good luck.
