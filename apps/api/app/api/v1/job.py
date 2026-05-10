"""GET /v1/job/{id} — full job detail for the explanation drawer."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.v1.schemas import JobDetail
from app.services.retrieval import get_job_detail

router = APIRouter(prefix="/job", tags=["job"])


@router.get(
    "/{job_id}",
    response_model=JobDetail,
    summary="Full job detail (skills + responsibilities + raw + enhanced text)",
)
def job(job_id: str) -> JobDetail:
    detail = get_job_detail(job_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return detail
