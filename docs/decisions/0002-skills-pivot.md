# ADR-0002 — Pivot to a skills-centric KG schema (Person side)

- **Status**: Accepted, partially superseded by [ADR-0003](0003-two-sided-corpus.md)
  on 2026-05-09 (the same day) once the full dataset was inventoried — Job-side
  relations `LOCATED_IN`, `IN_INDUSTRY`, and a new `REQUIRES_SKILL` edge are
  re-introduced for the Job node label. The Person-side conclusions in this ADR
  remain in force.
- **Date**: 2026-05-09
- **Deciders**: PathFinder lead engineer (with explicit user confirmation at the
  Day-1 STOP-AND-CONFIRM checkpoint after running `scripts/00_inspect_csv.py`).
- **Affects**: `apps/api/app/core/schema_map.yaml`, `app/services/kg.py`,
  `scripts/04_kg_build.py`, `docs/architecture.md`, demo queries in README and
  `apps/web/app/page.tsx`.

## Context

The initial design assumed a LinkedIn-style profile schema: `id, name, headline,
summary, skills, experience JSON, education JSON, location, years_experience,
current_company, current_title, industry`. The KG schema followed: `Person -[:WORKED_AT]→
Company`, `LOCATED_IN`, `STUDIED_AT`, `IN_INDUSTRY`, `SIMILAR_TO (Company-Company)`,
`COLLABORATED_WITH`.

The actual `profiles.csv` (sourced from a public dataset repository on GitHub) has
**1,782 rows × 8 columns** — confirmed via `scripts/00_inspect_csv.py` on 2026-05-09:

| Column | Format |
| ------ | ------ |
| `id` | unique int |
| `name` | string, often blank or anonymised |
| `core_skills` | comma-list with proficiency: `"Python (Advanced Beginner), SQL (Beginner)"` |
| `secondary_skills` | same format |
| `soft_skills` | same format (often NaN) |
| `years_of_experience` | float (mostly `0.0` — fresher-heavy) |
| `potential_roles` | comma-list of suggested role titles |
| `skill_summary` | ~70-word prose summary |

There is **no source data** for company, industry, location, education, or career
history. There is, however, **richer skill data** than the original plan assumed:
3-way categorisation (core / secondary / soft) with 4 ordinal proficiency levels
(`Beginner < Advanced Beginner < Competent < Advanced`) plus a curated list of roles
each candidate could fill.

The skills-only Person side is actually a feature, not a bug: it forces a richer
skill-ontology story (ESCO canonicalisation, proficiency-aware ranking) that's a
stronger interview narrative than yet-another LinkedIn-clone search.

## Decision

Pivot the KG schema to a **skills-and-roles** ontology. Drop relations whose source
fields don't exist; keep + enrich relations the data does support; add new relations
the data uniquely enables.

### Knowledge graph — Person-side schema (extended on the Job side in ADR-0003)

**Nodes**

| Label | Properties |
| ----- | ---------- |
| `Person` | `id, name, years_experience, embedding (1024-d BGE-M3 over canonical_text)` |
| `Skill` | `id, name, esco_uri?, embedding (1024-d BGE-M3 over name)` |
| `Role` | `id, title, embedding` |
| `_PendingType` | `name, count, examples` *(quarantine for novel labels — kept from plan)* |

**Relations**

| Triple | Properties | Source |
| ------ | ---------- | ------ |
| `(Person)-[:HAS_SKILL]→(Skill)` | `category ∈ {core, secondary, soft}, proficiency ∈ {1..4}, evidence_chunk_id, source` | parsed from `core_skills` / `secondary_skills` / `soft_skills` |
| `(Person)-[:CAN_FILL]→(Role)` | `rank` (position in the comma-list, 1 = best), `evidence_chunk_id` | `potential_roles` |
| `(Skill)-[:RELATED_TO]→(Skill)` | `weight` | seeded from ESCO `relatedSkill` |
| `(Skill)-[:CHILD_OF]→(Skill)` | — | ESCO hierarchy |
| `(Person)-[:NEIGHBOR]→(Person)` | `ppr_score` | precomputed top-25 via NetworkX Personalized PageRank |

### Relations dropped

`WORKED_AT, LOCATED_IN, HELD_ROLE, STUDIED_AT, IN_INDUSTRY, SIMILAR_TO (Company),
COLLABORATED_WITH`. Each is tied to fields the dataset does not have. Synthesizing
them from the prose summary via LLM would produce hallucinated edges — directly
violating the plan's faithfulness goals (RAGAS ≥ 0.95).

## Alternatives considered

| Option | Why rejected |
| ------ | ------------ |
| **Synthesize missing fields with an LLM** (extract location/company/industry from `skill_summary`) | The text is a generic skill summary, not a résumé — no anchor entities. Hallucination rate would be high; would inflate corpus signal artificially and tank RAGAS faithfulness. Defeats the explainability goal. |
| **Keep original schema, leave most edges empty** | Misleads graph viz and downstream Cypher; "0 results" looks like a bug. Demo lands worse than the corrected schema. |
| **Augment with an external dataset** (LinkedIn scrape, Crunchbase) | TOS / privacy risk; out of scope; not free; recruiter would correctly flag the demo as not-the-stated-dataset. |

## Consequences

### Positive

- **Skill ontology + ESCO becomes the differentiating story.** ESCO ships ~13.9 k
  skills with hierarchy + relatedness — perfectly maps to `Skill` nodes,
  `RELATED_TO`, and `CHILD_OF`. Few portfolio projects ship a real ontology
  integration; this becomes a recruiter talking-point.
- **Proficiency-aware ranking** is a unique surface area: the proficiency tags
  parsed from each cell give us a 4-level ordinal that we can use both as a
  hard filter (`min_proficiency=Competent`) and as a soft re-ranking signal.
- **Role-fit search** is data-native: `(Person)-[:CAN_FILL]→(Role)` lets us
  answer queries like _"Find candidates for an X role"_ without inventing the
  edges.
- **Smaller corpus (1,782 vs 50k)** means cross-encoder over the **full** corpus is
  feasible (vs the planned top-50 → top-10 funnel). We retain the funnel for
  realism and benchmark vs full-corpus rerank as an ablation row.
- **No fake company/location data** → RAGAS faithfulness is defendable; we never
  generate explanations citing entities that aren't in the source.

### Negative / risks

- **No company / industry / location** queries to demo. Mitigated by curating the
  Week-1 200-query eval set around skill + proficiency + role-fit strata (the
  eval-set script already accepts a stratum YAML — Week 1 update).
- **`name` field is sparse** — many rows are blank or anonymised. Mitigated by
  always rendering the `id` alongside the name, and treating name as a low-signal
  display field for explanations.
- **`years_of_experience` is mostly 0.0** — using it as a hard filter would empty
  most queries. Treat as a soft re-rank signal.

### Operational changes (Week 1)

- `scripts/01_etl.py` — section-parse the proficiency-tagged skill cells:
  regex `^(?P<name>.+?)\s*\((?P<prof>Beginner|Advanced Beginner|Competent|Advanced)\)\s*$`.
- `scripts/04_kg_build.py` — three deterministic passes (skills, roles, ESCO
  augmentation), skipping the `SchemaLLMPathExtractor` from the plan since we no
  longer need to extract relations from prose. The extractor is **kept in the
  codebase** as a stub for the eval-set generation and Week-5 explainability —
  the *technique* still lands as a portfolio talking-point.
- `apps/web/app/page.tsx` — example query replaced with skill-and-role phrasing.
- `docs/eval-methodology.md` §1 — strata table updated to remove
  "city / fintech / 5+ years" anchors and add "skill cluster + proficiency +
  role-fit" anchors.

## Validation plan

The Week-2 ablation table records the same five rows (BM25, +dense, +RRF,
+rerank, +KG, +DAT). Success criteria are unchanged from the plan: nDCG@10 ≥ 0.55,
Recall@100 ≥ 0.97, RAGAS Faithfulness ≥ 0.95, p95 &lt; 2 s. We additionally add a
**Week-5 ablation row** for the `proficiency-aware rerank` (multiply cross-encoder
score by `0.7 + 0.075 × proficiency_ordinal`) — interview-defensible since this is
a feature only this dataset enables.
