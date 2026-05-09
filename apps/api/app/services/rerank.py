"""Cross-encoder reranker service — `BAAI/bge-reranker-v2-m3` FP16.

Lazy-loaded singleton: 568M params, ~2.3 GB FP16, loads in ~30 s on RTX 4060
the first time. Subsequent calls reuse the in-memory model (so the rerank-
ablation benchmarks aren't dominated by load time).

Pairs naturally with `app/services/dense.py` (same model family, MIT-licensed,
both fit in 8 GB VRAM with KV-cache headroom — see ADR-0001).

Implementation note: we use `sentence_transformers.CrossEncoder` rather than
`FlagEmbedding.FlagReranker` because the FlagReranker tokenisation path calls
`tokenizer.prepare_for_model(...)` which was removed in transformers ≥ 5.0.
CrossEncoder loads the same `BAAI/bge-reranker-v2-m3` weights and applies the
canonical sigmoid head, so scores are byte-identical for the same input.
"""

from __future__ import annotations

import functools
from typing import Any

from app.core.logging import get_logger

log = get_logger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


@functools.lru_cache(maxsize=1)
def get_reranker(
    model_name: str = DEFAULT_MODEL_NAME,
    use_fp16: bool = True,
) -> Any:
    """Load and cache the cross-encoder reranker. Heavy — call once per process."""
    try:
        import torch
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "Cross-encoder deps missing. Run `uv sync --extra ml` from apps/api/."
        ) from exc

    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"

    if cuda_available:
        log.info(
            "reranker_loading",
            model=model_name,
            cuda_device_name=torch.cuda.get_device_name(0),
            fp16=use_fp16,
        )
    else:
        log.warning("reranker_cpu_fallback", reason="cuda_unavailable", fp16=use_fp16)

    # CrossEncoder uses transformers AutoModelForSequenceClassification under the hood.
    # max_length=512 is the bge-reranker-v2-m3 default; longer pairs get truncated.
    reranker = CrossEncoder(
        model_name,
        device=device,
        max_length=512,
        automodel_args={"torch_dtype": "float16"} if use_fp16 and cuda_available else None,
    )
    log.info("reranker_ready", model=model_name, device=device, fp16=use_fp16)
    return reranker


def rerank_pairs(
    pairs: list[tuple[str, str]],
    *,
    batch_size: int = 32,
    normalize: bool = True,
) -> list[float]:
    """Score a batch of (query, doc) pairs. Returns sigmoid scores in [0, 1]."""
    reranker = get_reranker()
    # CrossEncoder.predict returns numpy array; sentence_transformers applies sigmoid
    # via `activation_fn="sigmoid"` when configured on the model card.
    scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return [float(s) for s in scores]
