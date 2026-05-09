# ADR-0003 — Two-sided corpus: Person + Job + JD; reintroduce Job-side relations

- **Status**: Accepted
- **Date**: 2026-05-09
- **Supersedes**: parts of [ADR-0002](0002-skills-pivot.md) — specifically, the
  decision to drop `LOCATED_IN` and `IN_INDUSTRY`. Those edges return on the **Job**
  node label. Person-side conclusions of ADR-0002 stand.
- **Affects**: `apps/api/app/core/schema_map.yaml` (split into per-source maps),
  `app/services/kg.py`, `scripts/01_etl.py`, `scripts/02_bm25_index.py`,
  `scripts/04_kg_build.py`, eval-set generator, README, web UI.

## Context

ADR-0002 was written believing the dataset was a single skills-focused profile CSV.
A subsequent inventory of the public source repository (initiated by the user
asking _"isn't the dataset too small?"_) revealed **three** data assets, not one:

| File | Items | Side | Format |
| ---- | ----: | ---- | ------ |
| `profiles.csv`     | 1,782 | candidates | CSV — skills (with proficiency), summary, suggested roles |
| `demands_data.csv` | 1,081 | demands    | CSV — id, city/state/country, primary/secondary skills, experience range, designation |
| `jd_dataset.zip`   |   289 | JDs        | per-job folder with `raw_jd.txt` (industry + raw text) and `enhanced_job_description.md` (LLM-enhanced sections) |

**Combined**: ~3,150 documents. The corpus is **two-sided** — candidates and jobs
share `Skill` as the join key — and includes structured location, designation,
industry, and experience-range fields **on the job side**.

This is a meaningfully larger and richer corpus than ADR-0002 assumed. The skills
pivot for Person nodes still holds (the candidate side genuinely lacks
company / location / industry data). But the Job side has all of it, plus a free
LLM-enhanced JD that's perfect material for BGE-M3 dense indexing.

## Decision

PathFinder is reframed from "people search" to **two-sided talent matching**.
The KG and indexing pipelines are extended to cover both sides.

### Knowledge graph — final two-sided schema

**Nodes** (each with `embedding: 1024-d BGE-M3`)

| Label | Properties | Source |
| ----- | ---------- | ------ |
| `Person`        | `id, name, years_experience` | profiles.csv |
| `Job`           | `id, title, designation, industry, experience_lower, experience_upper, raw_text, enhanced_text` | demands_data.csv (structured) ∪ jd_dataset (text) — joined where possible by id |
| `Skill`         | `id, name, esco_uri, kind ∈ {tech, soft}` | parsed from both sides; canonicalised via ESCO |
| `Role`          | `id, title` | profiles.potential_roles |
| `Designation`   | `name` | demands_data.designation |
| `Location`      | `city, state, country` | demands_data |
| `Industry`      | `name` | jd raw_jd.industry + ESCO industry codes |
| `_PendingType`  | `name, count, examples` | quarantine for novel labels |

**Relations** (with `confidence`, `evidence_chunk_id`)

Person side (from ADR-0002, unchanged):

- `(Person)-[:HAS_SKILL {category, proficiency, source}]→(Skill)`
- `(Person)-[:CAN_FILL {rank}]→(Role)`
- `(Person)-[:NEIGHBOR {ppr_score}]→(Person)` *(precomputed top-25)*

Job side (new in this ADR):

- `(Job)-[:REQUIRES_SKILL {priority ∈ {must_have, good_to_have}, category ∈ {primary, secondary}}]→(Skill)`
- `(Job)-[:IS_DESIGNATION]→(Designation)`
- `(Job)-[:AT_LOCATION]→(Location)`
- `(Job)-[:IN_INDUSTRY]→(Industry)`
- `(Job)-[:NEIGHBOR {ppr_score}]→(Job)` *(precomputed top-25)*

ESCO-seeded (unchanged):

- `(Skill)-[:RELATED_TO {weight}]→(Skill)`
- `(Skill)-[:CHILD_OF]→(Skill)`

**Derived (the killer talking-point)**:

- `(Person)-[:MATCHES {score, reason_chunks, computed_at}]→(Job)` — precomputed
  offline as the top-K bipartite matches per Person, scored as a weighted blend of:
  - Skill overlap (Jaccard over canonical skill IDs, weighted by proficiency
    on the Person side and `priority` on the Job side)
  - Designation similarity (Person.potential_roles ↔ Job.designation cosine on
    BGE-M3 embeddings)
  - Free-text similarity (skill_summary ↔ enhanced_job_description.md cosine)
  - Soft penalty for `years_experience < experience_lower`

### Retrieval pipeline — both-direction

The intent router gains a 5th class: `match` (person↔job pair scoring). The four
existing classes (`structural`, `semantic`, `hybrid`, `clarify`) all gain a target
side flag (candidates / jobs). The Qdrant collection grows to two named vector
spaces (`profiles_v1` and `jobs_v1`); the cross-encoder reranker is shared.

### Eval set — restratified

Replace the single 200-query set with three 100-query sets:

| Task                                                         | Count | Anchors |
| ------------------------------------------------------------ | ----: | ------- |
| **Candidate search** (find people for a role)                |   100 | skill + proficiency + role-fit + location-on-job |
| **Job search** (find roles for a candidate)                  |   100 | skill + experience range + industry + designation |
| **Match** (rank the other side given one anchor)             |   100 | Person → top-10 Jobs and Job → top-10 Persons; gold from skill overlap |

Hard negatives, RAGAS judge, IR metrics (`ranx`) all unchanged.

## Alternatives considered

| Option | Why rejected |
| ------ | ------------ |
| **Stay people-only (ignore demands.csv + jd_dataset)** | Wastes ~1,400 documents and the most natural way to demonstrate `(Job)-[:REQUIRES_SKILL]→(Skill)` graph traversal. Weakens the "this is a real recruiter system" pitch. |
| **Index only `enhanced_job_description.md` (skip the structured demands.csv)** | Loses location / experience-range / designation filters, which are exactly the kind of structural-query signals the intent router was designed for. |
| **Synthesize Person-side location / industry from prose** | Same hallucination concern as ADR-0002. Now even less necessary — the Job side gives us those edges directly, and matching across the Skill bridge produces the multi-hop queries the original plan promised. |

## Consequences

### Positive

- **Corpus volume tripled** to ~3,150 documents. Cross-encoder over the full corpus
  remains feasible; latency budget unchanged.
- **`(Person)-[:MATCHES]→(Job)` is the demo's killer feature** — every recruiter-
  style query has both directions and a confidence-weighted explanation rooted in
  actual edge overlap.
- **Restores Job-side `LOCATED_IN` / `IN_INDUSTRY`** — graph viz and Cypher demos
  are no longer empty for those relations.
- **Two-sided eval is more credible** than people-only — it's the standard
  formulation in the talent-matching IR literature (Diaz et al., DocChat 2024).
- **Free LLM-enhanced JDs** in `jd_dataset` give us a strong dense-retrieval signal
  on the job side without LLM re-indexing cost.

### Negative / risks

- **Pipeline complexity grows** — three ETL paths instead of one, two embedding
  collections, two Cypher template families. Mitigated by the existing
  `schema_map.yaml` indirection: each source has its own field map under
  `app/core/schema_maps/{profiles,demands,jds}.yaml`.
- **Job-side dataset is messy** (NaN job_title, designation overloaded as both
  job_title and seniority indicator). Section parser must be defensive; a
  `_PendingType` quarantine for unknown designation tokens is the safety valve.
- **Person ↔ Job id collisions** — Person ids are short ints (`181457`); Job ids
  are long strings (`DBS-__DBS-__2025__2591399`). Namespace internally as
  `person_<id>` / `job_<id>` to avoid Qdrant / Neo4j collisions.

## Validation plan

The Week-2 ablation table now contains 3 task columns × 6 config rows = 18 cells.
Success criteria from the project plan apply per task:

| Task             | Recall@100 | nDCG@10 | RAGAS Faithfulness |
| ---------------- | ---------- | ------- | ------------------ |
| Candidate search | ≥ 0.97     | ≥ 0.55  | ≥ 0.95             |
| Job search       | ≥ 0.97     | ≥ 0.55  | ≥ 0.95             |
| Match            | ≥ 0.92     | ≥ 0.50  | ≥ 0.95             |

Match metrics are intentionally lower because the bipartite-matching gold is
noisier than direct retrieval relevance.
