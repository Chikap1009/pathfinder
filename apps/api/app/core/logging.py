"""structlog configuration — JSON in prod, human-readable in dev.

Bound logger picks up the active OTel trace_id automatically when the OTel SDK is
configured (see app/core/otel.py).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from app.core.settings import get_settings


def configure_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.is_dev:
        renderer: Any = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Tame noisy stdlib loggers; pipe them through structlog if needed later.
    for name in ("uvicorn", "uvicorn.access", "httpx", "neo4j"):
        logging.getLogger(name).setLevel(level)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
