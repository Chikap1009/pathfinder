# PathFinder — Evaluation methodology

> Locked Week 5; numbers refresh per ablation run via `apps/api/scripts/07_eval_run.py`.

## 1. Eval set construction

200 stratified synthetic queries grounded in the actual `profiles.csv` (skills-pivot
post-ADR-0002 — strata reflect the real fields: skill cluster + proficiency + role-fit):

| Stratum                                | Count | Generator anchor                                                              |
| -------------------------------------- | ----: | ----------------------------------------------------------------------------- |
| Single skill + proficiency             |    50 | "Find Competent+ Python engineers"                                            |
| Skill cluster                          |    40 | "Python + SQL + cloud, all Advanced Beginner or higher"                       |
| Role-fit                               |    40 | "Candidates who could fill a Regulatory Affairs Manager role"                 |
| Soft-skill weighted                    |    20 | "Strong analytical thinking + competent in legal research"                    |
| Paraphrase / non-technical             |    25 | natural recruiter phrasing of the above                                       |
| Negation / disqualifier                |    15 | "Python engineers but not regulatory-focused profiles"                        |
| Typo / abbreviation                    |    10 | "py + sql adv-beg+ devops"                                                    |

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
