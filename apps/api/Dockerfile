# PathFinder API — production image (Hugging Face Spaces target).
#
# Data artifacts (BM25 indexes, dense embeddings, canonical skill parquet,
# DuckDB warehouse) are downloaded from a public GitHub release asset at
# build time rather than bundled in the repo, because HF Space free-tier
# LFS storage rejects binary files >a few MB.

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

RUN uv sync --frozen --no-dev

# ──── Layer 2 — ML libs with CPU torch ────────────────────────────────────
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

# ──── Layer 4 — Data artifacts from public GitHub release ────────────────
ARG DATA_RELEASE_URL=https://github.com/Chikap1009/pathfinder/releases/download/data-v1/pathfinder-data-v1.tgz
ARG DATA_REV=3
RUN echo "data-rev=$DATA_REV" && mkdir -p data && curl -fsSL "$DATA_RELEASE_URL" | tar -xz -C data && ls -la data/processed/

# Marker so `_find_repo_root()` resolves correctly at /home/user/app/
RUN touch /home/user/app/pnpm-workspace.yaml \
    && chown -R user:user /home/user/app

USER user
EXPOSE 7860
CMD ["sh", "-c", "uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
