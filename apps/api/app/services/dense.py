"""BGE-M3 dense / sparse / ColBERT encoder service.

Lazy-loaded singleton: the model loads on first call (~30s on RTX 4060 FP16,
~3 min on CPU). Subsequent calls reuse the in-memory model.

Wraps `FlagEmbedding.BGEM3FlagModel` — a single forward pass returns dense +
learned-sparse + ColBERT vectors. Dense alone for now (Day-2 part 3); the
sparse / ColBERT heads activate in Week 2 alongside Qdrant multivector indexing.

Hardware-aware: prefers CUDA when `torch.cuda.is_available()`, otherwise falls
back to CPU with a structured-log warning.
"""

from __future__ import annotations

import functools
from typing import Any

from app.core.logging import get_logger

log = get_logger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024


@functools.lru_cache(maxsize=1)
def get_dense_encoder(
    model_name: str = DEFAULT_MODEL_NAME,
    use_fp16: bool = True,
) -> Any:
    """Load and cache the BGE-M3 model. Heavy — call once per process."""
    try:
        import torch
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise RuntimeError("BGE-M3 deps missing. Run `uv sync --extra ml` from apps/api/.") from exc

    if torch.cuda.is_available():
        device = "cuda:0"
        log.info(
            "bge_m3_loading",
            device=device,
            cuda_device_name=torch.cuda.get_device_name(0),
            cuda_capability=torch.cuda.get_device_capability(0),
            fp16=use_fp16,
        )
    else:
        device = "cpu"
        # FP16 on CPU is supported by FlagEmbedding but slower than FP32; keep FP16
        # for memory parity with GPU runs so embeddings stay byte-identical.
        log.warning("bge_m3_cpu_fallback", reason="cuda_unavailable", fp16=use_fp16)

    model = BGEM3FlagModel(
        model_name,
        use_fp16=use_fp16,
        devices=[device] if device.startswith("cuda") else None,
    )
    log.info("bge_m3_ready", model=model_name, device=device)
    return model


def encode_dense(
    texts: list[str],
    *,
    batch_size: int = 16,
    max_length: int = 8192,
    show_progress: bool = False,
) -> Any:
    """Encode a batch of texts → numpy array (n, 1024) float32."""
    model = get_dense_encoder()
    out = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    # FlagEmbedding returns a dict-like; extract dense vectors.
    return out["dense_vecs"]
