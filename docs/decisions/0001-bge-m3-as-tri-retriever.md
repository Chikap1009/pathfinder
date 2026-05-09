# ADR-0001 — BGE-M3 as a single-model tri-retriever

- **Status**: Accepted
- **Date**: 2026-05-09
- **Deciders**: PathFinder lead engineer
- **Affects**: `apps/api/app/services/vector.py`, indexing scripts `02_*`, `03_*`,
  hardware (RTX 4060 8 GB VRAM), free-tier model budget.

## Context

PathFinder's hybrid retrieval pipeline needs three orthogonal candidate signals to fuse via RRF:

1. **Lexical** — exact-token matching on rare technical entities (`k8s`, `gRPC`,
   `LangGraph`, model names, library versions). Necessary because dense embeddings
   alone consistently miss out-of-vocabulary technical jargon at the long tail.
2. **Dense semantic** — paraphrase / out-of-vocabulary recall ("ML engineer" matches
   "machine-learning practitioner").
3. **Multi-vector / late-interaction** — fine-grained per-token matching (ColBERT-style)
   for queries where bag-of-vectors miss subtle distinctions.

We must operate in **8 GB VRAM** alongside a cross-encoder reranker, with **MIT-compatible
licensing** (commercial demo on Vercel + HF Spaces), under a **$0/month** budget.

## Decision

Adopt **`BAAI/bge-m3`** (568M, MIT-licensed) as the single embedding model that emits
**all three** retrieval representations in **one forward pass**:

- `dense` (1024-d float32 / FP16) → HNSW index in Qdrant, `cosine` distance.
- `sparse` (learned BM25-style lexical weights via the embedder's sparse head) →
  Qdrant sparse vector index.
- `colbert_vecs` (per-token 1024-d) → Qdrant **multivector** index, MaxSim scoring.

Configuration:

```python
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, devices=["cuda:0"])
out = model.encode(
    texts,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
    batch_size=32,
    max_length=8192,
)
```

The classical **BM25** signal is provided separately by **BM25S** over the canonical
profile text — it is *not* replaced by the BGE-M3 sparse head. We therefore have **four**
retrieval signals (BM25S, BGE-M3 sparse, BGE-M3 dense, BGE-M3 ColBERT) but only **one
embedding model** to load. RRF fuses the four lists.

## Alternatives considered

| Option | Why rejected |
| ------ | ------------ |
| **`intfloat/e5-mistral-7b-instruct`** | 7B → won't fit alongside reranker in 8 GB VRAM. |
| **`Salesforce/SFR-Embedding-Mistral`** | 7B; same VRAM problem; MIT but heavy. |
| **Jina-embeddings-v3** | CC-BY-NC license — blocks the public demo. |
| **OpenAI `text-embedding-3-large`** | $0.13/M tokens; not free. Indexing 50 k profiles = ~$10. |
| **`sentence-transformers/all-MiniLM-L6-v2`** | Strong baseline but no sparse/ColBERT heads; would need a separate model for late interaction. |
| **SPLADE-v3 (separate)** + dense (separate) | Two models = 2× VRAM and 2× indexing pipelines for marginal gain over BGE-M3 sparse head. |

## Consequences

### Positive

- **One model, three signals** → clean architectural narrative for interviews; aligns with
  BGE-M3's published paper (arXiv 2402.03216) framing.
- VRAM: ~1.1 GB FP16 for BGE-M3 + ~2.3 GB for `bge-reranker-v2-m3` = ~3.4 GB → leaves ~4.5 GB
  headroom on the 4060 for KV cache and Ollama backstops.
- Multilingual out-of-the-box (100+ languages) — gives "production-ready" surface area
  for an internship narrative even though `profiles.csv` is likely English-only.
- 8 192-token context handles full résumé prose without chunking gymnastics.
- MIT license — usable on a commercial-style demo without legal asterisks.

### Negative / risks

- **ColBERT vectors are large** — ~50× the dense vector size. Qdrant multivector storage
  mitigation: store ColBERT only for the top 5 k profiles (offline pre-filter by dense
  retrieval), or accept the 1.5 GB overhead. **Mitigation in code**: feature-flag
  `RETRIEVAL_COLBERT_ENABLED` defaults `False` until ablation shows ≥ 1 % nDCG@10 lift.
- BGE-M3 sparse head is **not** drop-in identical to true SPLADE — minor recall gap on
  rare entities. **Mitigation**: keep BM25S in the fusion (covers the long tail).
- Single-model risk: a regression in BGE-M3 affects 3 of 4 retrieval signals.
  **Mitigation**: Voyage-3-large is wired as the embedder fallback in `litellm.config.yaml`
  (`embedder` model group).

### Operational

- Indexing 50 k profiles with `batch_size=32, max_length=8192, FP16` ≈ 12 minutes on a 4060.
  Acceptable for a one-off + nightly delta job.
- HF Spaces (CPU) cannot run BGE-M3 at &lt; 200 ms latency. Mitigation: query embedding
  happens **client-of-API** on the local dev box during demo, OR we switch to Voyage at
  query-time when deployed (slower but free 200 M tokens). Both modes covered by
  `app/services/vector.py:embed_query()`.

## Validation plan

The Week-2 ablation table (`docs/eval-methodology.md` §3) records the marginal contribution
of each BGE-M3 signal: dense alone, dense + sparse, dense + sparse + ColBERT, and the full
4-way fusion with BM25S. We re-evaluate this ADR if BGE-M3 sparse contributes &lt; 0.5 % nDCG@10
over BM25S+dense alone — in that case we drop the sparse head to save indexing time.
