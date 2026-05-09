# PathFinder — Evaluation methodology

> Locked Week 5; numbers refresh per ablation run via `apps/api/scripts/07_eval_run.py`.

## 1. Eval set construction

300 stratified synthetic queries — three 100-query sets, one per task. Every query
is grounded in a real anchor entity from the corpus (`profiles.csv`,
`demands_data.csv`, or a parsed JD), so relevance judgments are reproducible
without a human pass. See [decisions/0003-two-sided-corpus.md](decisions/0003-two-sided-corpus.md)
for the two-sided framing.

### Task A — Candidate search (100): find people who fit a need

| Stratum                                | Count | Generator anchor                                                              |
| -------------------------------------- | ----: | ----------------------------------------------------------------------------- |
| Single skill + proficiency             |    25 | "Find Competent+ Python engineers"                                            |
| Skill cluster                          |    20 | "Python + SQL + cloud, all Advanced Beginner or higher"                       |
| Role-fit                               |    20 | "Candidates who could fill a Regulatory Affairs Manager role"                 |
| Soft-skill weighted                    |    10 | "Strong analytical thinking + competent in legal research"                    |
| Paraphrase / non-technical             |    10 | natural recruiter phrasing                                                    |
| Negation / disqualifier                |     8 | "Python engineers but not regulatory-focused profiles"                        |
| Typo / abbreviation                    |     7 | "py + sql adv-beg+ devops"                                                    |

### Task B — Job search (100): find roles for a candidate

| Stratum                                | Count | Generator anchor                                                              |
| -------------------------------------- | ----: | ----------------------------------------------------------------------------- |
| Skill + experience range               |    25 | "Java backend roles, 5+ years"                                                |
| Skill + location                       |    25 | "Selenium roles in Bengaluru / Pune"                                          |
| Skill + designation                    |    20 | "Senior Technical Lead positions in DevOps"                                   |
| Skill + industry                       |    15 | "ServiceNow ITBM roles in any industry"                                       |
| Negation / disqualifier                |    10 | "Backend roles but not testing-focused"                                       |
| Typo / abbreviation                    |     5 | "k8s + tf sr lead bengaluru"                                                  |

### Task C — Match (100): rank the other side given an anchor

| Stratum                                                      | Count | Anchor type |
| ------------------------------------------------------------ | ----: | ----------- |
| Person → top-10 Jobs (skill-rich Person)                     |    35 | Person id   |
| Person → top-10 Jobs (skill-sparse Person, fresher)          |    15 | Person id   |
| Job → top-10 Persons (skill-rich Job)                        |    35 | Job id      |
| Job → top-10 Persons (skill-sparse Job)                      |    15 | Job id      |

For each query the source profile is gold (`qrel = 2`); rule-based + semantic-similarity
finds up to 3 graded relevants (`qrel = 1`). Stored as TREC qrels.

**Hard-negative mining** (Karthik et al. 2025, +19 % MRR@10): for each (query, positive),
score all candidates with BM25 + multi-embedder ensemble; keep candidates `D` where
`d(Q, D) < d(Q, PD)` AND `d(PD, D) > τ`. Cross-encoder denoise: drop candidates with
reranker score &gt; 0.7 (likely false negatives).

## 2. Metrics + thresholds

### Retrieval (`ranx`)

| Metric | Target |
| ------ | ------ |
| Recall@100 | ≥ 0.97 |
| Recall@10  | ≥ 0.85 |
| MRR@10     | ≥ 0.65 |
| MAP        | ≥ 0.55 |
| nDCG@10    | ≥ 0.55 |

### RAGAS (Gemini judge — different family from the BGE generator)

| Metric              | Target |
| ------------------- | ------ |
| Faithfulness        | ≥ 0.95 |
| Context Recall      | ≥ 0.97 |
| Context Precision   | ≥ 0.85 |
| Answer Relevancy    | ≥ 0.92 |
| Noise Sensitivity   | ≤ 0.10 |

### Explanation-specific

| Metric                              | Target |
| ----------------------------------- | ------ |
| Explanation Faithfulness (RAGAS)    | ≥ 0.97 |
| NER-grounding (every entity in src) | ≥ 0.99 |

### Latency

| Stage                       | Target |
| --------------------------- | ------ |
| End-to-end p50              | &lt; 600 ms |
| End-to-end p95              | &lt; 2 s |

## 3. Ablation table (filled in Week 2 / 5)

| Config                              | nDCG@10 | Recall@100 | RAGAS Faithfulness | p95 latency |
| ----------------------------------- | ------- | ---------- | ------------------ | ----------- |
| BM25 only (baseline)                |         |            |                    |             |
| + BGE-M3 dense                      |         |            |                    |             |
| + RRF (k=60)                        |         |            |                    |             |
| + cross-encoder rerank              |         |            |                    |             |
| + KG augmentation                   |         |            |                    |             |
| + DAT fusion                        | (target +2-3 % over RRF) |  |  |  |

## 4. Judge alignment

We run **`AspectCritic.align_and_validate()`** against a 30-sample hand-labelled set;
we ship only when judge ↔ human agreement ≥ 0.8 (Cohen's κ).

## 5. Reproducibility

- Seeds: 42 (NumPy), 1337 (PyTorch), 0 (transformers).
- Eval run is invoked via `make eval` → writes JSON to `data/eval/runs/<timestamp>/` and
  uploads to Langfuse as a "dataset run" so trends show in the dashboard.
