"""POST /v1/search and POST /v1/search/stream — the production retrieval endpoint.

Both routes share the same orchestrator (`app.services.retrieval.RetrievalEngine`).
The `/stream` variant emits Server-Sent Events with one chunk per pipeline stage
(intent, encode, bm25, dense, kg, rrf, rerank, results, done) so the Vercel AI SDK 5
client can render Perplexity-style progress chips.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.v1.schemas import IntentResult, SearchRequest, SearchResponse, StageEvent
from app.core.logging import get_logger
from app.services.intent import classify_intent
from app.services.retrieval import get_engine

log = get_logger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


def _resolve_intent(req: SearchRequest) -> IntentResult:
    """Run the intent classifier; honour an explicit target_side override on the request."""
    intent = classify_intent(req.query)
    if req.target_side is not None and req.target_side != intent.target_side:
        log.debug(
            "intent_override",
            classifier=intent.target_side.value,
            override=req.target_side.value,
        )
        intent = intent.model_copy(update={"target_side": req.target_side})
    return intent


@router.post(
    "",
    response_model=SearchResponse,
    summary="Run the full hybrid retrieval pipeline (synchronous)",
    description=(
        "End-to-end retrieval over the chosen `pipeline` configuration. Emits the typed "
        "results + per-stage timings + per-result stage scores in a single response. "
        "Use `POST /v1/search/stream` for incremental streaming."
    ),
)
def search(req: SearchRequest) -> SearchResponse:
    intent = _resolve_intent(req)
    engine = get_engine()
    return engine.search(req.query, intent, pipeline=req.pipeline, top_k=req.top_k)


@router.post(
    "/stream",
    summary="Run the full hybrid retrieval pipeline with SSE per-stage events",
    description=(
        "Server-Sent Events stream. Each event has `type` ∈ {intent, stage_start, "
        "stage_done, results, done, error} and is emitted as soon as the corresponding "
        "stage completes, so the client can show Perplexity-style progress chips."
    ),
    response_class=StreamingResponse,
)
async def search_stream(req: SearchRequest) -> StreamingResponse:
    intent = _resolve_intent(req)
    engine = get_engine()

    async def event_stream() -> AsyncIterator[bytes]:
        try:
            async for ev in engine.search_stream(
                req.query, intent, pipeline=req.pipeline, top_k=req.top_k
            ):
                yield _sse_format(ev)
        except Exception as exc:
            log.exception("search_stream_error")
            yield _sse_format(StageEvent(type="error", message=str(exc)[:300]))

    headers: dict[str, str] = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # disables nginx buffering when behind a proxy
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


def _sse_format(ev: StageEvent) -> bytes:
    """Serialize a StageEvent as a single SSE message."""
    payload = ev.model_dump_json(exclude_none=True)
    return f"event: {ev.type}\ndata: {payload}\n\n".encode()


# ─── A tiny health endpoint so the client can ping the engine ────────────────


@router.get("/health", summary="Ping the retrieval engine")
def search_health() -> dict[str, Any]:
    eng = get_engine()
    return {
        "ok": True,
        "profiles": len(eng.dense_p.doc_ids),
        "jobs": len(eng.dense_j.doc_ids),
        "canonical_skills": len(eng.skill_map),
    }
