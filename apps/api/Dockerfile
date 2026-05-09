# PathFinder API — HF Spaces docker image (port 7860).
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/root/.cache/huggingface

# System deps — keep slim; add build-essentials only if a wheel needs compiling.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (single static binary)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# Layer 1: dep manifests (cache-friendly)
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev --extra ml --extra graph --extra llm --extra eval --extra jobs --extra obs || \
    uv sync         --no-dev --extra ml --extra graph --extra llm --extra eval --extra jobs --extra obs

# Layer 2: source
COPY app  ./app
COPY scripts ./scripts

EXPOSE 7860
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
