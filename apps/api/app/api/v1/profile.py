"""GET /v1/profile/{id} — full profile detail for the explanation drawer."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.v1.schemas import ProfileDetail
from app.services.retrieval import get_profile_detail

router = APIRouter(prefix="/profile", tags=["profile"])


@router.get(
    "/{person_id}",
    response_model=ProfileDetail,
    summary="Full profile detail (skills + summary + roles)",
)
def profile(person_id: str) -> ProfileDetail:
    detail = get_profile_detail(person_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Person {person_id!r} not found")
    return detail
