"""FastAPI entrypoint — singletons via lifespan, lazy ML model loading.

Heavy clients (Qdrant, Neo4j) open eagerly on startup so the first request is fast.
Heavy models (BGE-M3, bge-reranker-v2-m3) load *lazily* on first /v1/search to keep
boot time bounded — important for HF Spaces free tier where slow boots cause 502s.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.core.logging import configure_logging, get_logger
from app.core.otel import configure_otel
from app.core.settings import get_settings

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging()
    configure_otel(app)
    log.info(
        "startup",
        version=__version__,
        env=settings.app_env,
        cors=settings.cors_origin_list,
    )

    # ───── Eager singletons (cheap; surface mis-config early) ────────────────
    state: dict[str, Any] = {}

    # Qdrant client — optional in dev when Docker is not yet up.
    try:
        from qdrant_client import AsyncQdrantClient

        state["qdrant"] = AsyncQdrantClient(
            url=str(settings.qdrant_url),
            api_key=(
                settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
            ),
        )
        log.info("qdrant_ready", url=str(settings.qdrant_url))
    except ImportError:
        log.warning("qdrant_skipped", reason="qdrant-client not installed (sync --extra graph)")
    except Exception as exc:
        log.warning("qdrant_init_failed", error=str(exc))

    # Neo4j driver
    try:
        from neo4j import AsyncGraphDatabase

        state["neo4j"] = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(
                settings.neo4j_user,
                settings.neo4j_password.get_secret_value(),
            ),
        )
        log.info("neo4j_ready", uri=settings.neo4j_uri)
    except ImportError:
        log.warning("neo4j_skipped", reason="neo4j driver not installed (sync --extra graph)")
    except Exception as exc:
        log.warning("neo4j_init_failed", error=str(exc))

    # Lazy slots — populated on first request (see app.services.vector / .rerank)
    state["bge_m3"] = None
    state["bge_reranker"] = None

    app.state.svc = state

    try:
        yield
    finally:
        log.info("shutdown")
        if (q := state.get("qdrant")) is not None:
            await q.close()
        if (n := state.get("neo4j")) is not None:
            await n.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="PathFinder API",
        version=__version__,
        description="Intent-aware explainable hybrid retrieval for people search.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ───── Routes ────────────────────────────────────────────────────────────
    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, Any]:
        svc = getattr(app.state, "svc", {})
        return {
            "status": "ok",
            "version": __version__,
            "env": settings.app_env,
            "qdrant": "ready" if svc.get("qdrant") is not None else "absent",
            "neo4j": "ready" if svc.get("neo4j") is not None else "absent",
        }

    @app.get("/", tags=["meta"], include_in_schema=False)
    async def root() -> dict[str, str]:
        return {"name": "PathFinder API", "docs": "/docs"}

    # API v1 routers — wired in subsequent weeks.
    # from app.api.v1 import search, profile, explain, graph, eval as eval_api, index
    # for r in (search, profile, explain, graph, eval_api, index):
    #     app.include_router(r.router, prefix="/v1")

    return app


app = create_app()
