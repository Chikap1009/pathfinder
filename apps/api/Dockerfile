# PathFinder API — production image (Hugging Face Spaces target).
#
# HF Spaces docker SDK expects:
#   - port 7860 (overridable via $PORT)
#   - non-root /home/user workdir
#   - all data bundled (no network at runtime except outbound to Neo4j AuraDB,
#     LLM providers, and Langfuse).
#
# Build context: apps/api/ (with apps/api/data/ pre-staged via stage_for_hf.sh).
#
# We deliberately do NOT use `uv sync --extra ml` because the lockfile pins
# torch==2.11.0+cu130 (CUDA 13 wheel ~2.5 GB) which fails to import on a
# CPU-only host and bloats the image. Instead we install core deps from the
# lock and add CPU torch + ML libs via explicit uv pip installs.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# ──── Layer 1 — Core deps only (no ML extras) ─────────────────────────────
COPY pyproject.toml uv.lock* README.md ./

# Lockfile has torch+cu130 pinned in the [ml] extra. We install ONLY core deps
# here so we don't pay the CUDA wheel cost. ML libs come next via uv pip.
RUN uv sync --frozen --no-dev

# ──── Layer 2 — ML libs with CPU torch ────────────────────────────────────
# Pin versions matching pyproject.toml [ml]/[graph]/[llm] extras.
RUN uv pip install --no-cache \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch==2.11.0 \
        transformers \
        sentence-transformers \
        FlagEmbedding \
        bm25s \
        PyStemmer \
        scikit-learn \
        neo4j \
        ranx \
        litellm \
        instructor \
        google-generativeai \
        opentelemetry-api \
        opentelemetry-sdk \
        opentelemetry-instrumentation-fastapi \
        opentelemetry-exporter-otlp \
        langfuse

# ──── Layer 3 — Application source ────────────────────────────────────────
COPY app  ./app
COPY scripts ./scripts
COPY litellm.config.yaml ./

# ──── Layer 4 — Pre-computed data artifacts (~15 MB) ──────────────────────
COPY data ./data

# Marker so `_find_repo_root()` resolves correctly at /home/user/app/
RUN touch /home/user/app/pnpm-workspace.yaml \
    && chown -R user:user /home/user/app

USER user
EXPOSE 7860

# Honour $PORT (HF Spaces sets it to 7860 by default).
CMD ["sh", "-c", "uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
