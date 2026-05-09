"""OpenTelemetry → Langfuse OTLP wiring.

Tracing is **opt-in**: it activates only when both LANGFUSE_PUBLIC_KEY and
LANGFUSE_SECRET_KEY are present (or OTEL_EXPORTER_OTLP_ENDPOINT is set).
This keeps local dev runs free of network noise.
"""

from __future__ import annotations

import base64
import os
from typing import Any

from app.core.logging import get_logger
from app.core.settings import get_settings

log = get_logger(__name__)

_initialized = False


def configure_otel(app: Any | None = None) -> bool:
    """Wire OTel + FastAPI instrumentation. Returns True if tracing is active."""
    global _initialized
    if _initialized:
        return True

    settings = get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        log.info("otel_skipped", reason="langfuse_keys_missing")
        return False

    try:
        # Lazy import — `opentelemetry-*` is in the [obs] extra.
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        log.info("otel_skipped", reason="otel_packages_missing")
        return False

    pk = settings.langfuse_public_key.get_secret_value()
    sk = settings.langfuse_secret_key.get_secret_value()
    auth = base64.b64encode(f"{pk}:{sk}".encode()).decode()

    endpoint = (
        str(settings.otel_exporter_otlp_endpoint)
        if settings.otel_exporter_otlp_endpoint
        else f"{str(settings.langfuse_host).rstrip('/')}/api/public/otel/v1/traces"
    )

    os.environ.setdefault(
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS", f"Authorization=Basic {auth}"
    )

    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.otel_service_name})
    )
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=endpoint,
                headers={"Authorization": f"Basic {auth}"},
            )
        )
    )
    trace.set_tracer_provider(provider)

    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
        except ImportError:
            log.info("otel_fastapi_instrument_skipped", reason="package_missing")

    _initialized = True
    log.info("otel_configured", endpoint=endpoint)
    return True
