"""LiteLLM-based provider-agnostic LLM client.

Routes requests through `litellm` so application code talks to model groups
(`fast-small`, `big-reasoner`, `judge`, `explainer`) defined in
`apps/api/litellm.config.yaml` rather than hardcoding provider SDKs.

For Day-2 we use `litellm` directly without the proxy server — the proxy adds
ops surface (cost tracking, request log) we don't need yet. Same import path
either way; swap `completion()` calls to point at the proxy when we deploy.

Wraps Instructor for structured-output (Pydantic) calls.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.logging import get_logger
from app.core.settings import get_settings

log = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


# ─── Provider model strings (resolved by litellm) ────────────────────────────
# Mirrors the model_list groups from litellm.config.yaml. We list the primary
# choice here; fallbacks happen via `litellm.completion(... fallbacks=[...])`.

MODEL_GROUPS: dict[str, list[str]] = {
    "fast-small": [
        "groq/llama-3.1-8b-instant",
        "gemini/gemini-2.5-flash-lite",
    ],
    "big-reasoner": [
        "groq/llama-3.3-70b-versatile",
        "gemini/gemini-2.5-pro",
    ],
    "judge": [
        "gemini/gemini-2.5-flash",
    ],
    "explainer": [
        "gemini/gemini-2.5-flash",
    ],
    "paraphraser": [
        # Cheap + native JSON-schema support via google-generativeai.
        "gemini/gemini-2.5-flash-lite",
    ],
}


def _resolve_provider_keys() -> None:
    """Push pydantic-settings secrets into env so litellm can read them."""
    import os

    s = get_settings()
    if s.groq_api_key:
        os.environ["GROQ_API_KEY"] = s.groq_api_key.get_secret_value()
    if s.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = s.gemini_api_key.get_secret_value()
    if s.google_api_key:
        os.environ["GOOGLE_API_KEY"] = s.google_api_key.get_secret_value()
    if s.openai_api_key:
        os.environ["OPENAI_API_KEY"] = s.openai_api_key.get_secret_value()
    if s.cerebras_api_key:
        os.environ["CEREBRAS_API_KEY"] = s.cerebras_api_key.get_secret_value()


def _api_key_for(model: str) -> str | None:
    """Return the API key matching a `provider/model` string, or None."""
    s = get_settings()
    provider = model.split("/", 1)[0]
    match provider:
        case "groq":
            return s.groq_api_key.get_secret_value() if s.groq_api_key else None
        case "gemini" | "google":
            if s.gemini_api_key:
                return s.gemini_api_key.get_secret_value()
            if s.google_api_key:
                return s.google_api_key.get_secret_value()
            return None
        case "openai":
            return s.openai_api_key.get_secret_value() if s.openai_api_key else None
        case "cerebras":
            return s.cerebras_api_key.get_secret_value() if s.cerebras_api_key else None
        case _:
            return None


def _has_provider_for(group: str) -> tuple[bool, str | None]:
    """Return (any_key_present, missing_provider_name) for a model group."""
    s = get_settings()
    for model in MODEL_GROUPS.get(group, []):
        provider = model.split("/", 1)[0]
        match provider:
            case "groq":
                if s.groq_api_key:
                    return True, None
            case "gemini" | "google":
                if s.gemini_api_key or s.google_api_key:
                    return True, None
            case "openai":
                if s.openai_api_key:
                    return True, None
            case "cerebras":
                if s.cerebras_api_key:
                    return True, None
    return False, ", ".join({m.split("/", 1)[0] for m in MODEL_GROUPS.get(group, [])})


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def complete(
    group: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    response_format: dict[str, Any] | None = None,
) -> str:
    """Plain text completion through the configured model group."""
    _resolve_provider_keys()
    has_key, missing = _has_provider_for(group)
    if not has_key:
        raise RuntimeError(
            f"No API key available for model group '{group}' "
            f"(need one of: {missing}). Add to .env and rerun."
        )

    import litellm

    primary, *fallbacks = MODEL_GROUPS[group]
    log.debug("llm_complete", group=group, primary=primary, n_messages=len(messages))
    resp = litellm.completion(
        model=primary,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        fallbacks=fallbacks,
        num_retries=2,
        api_key=_api_key_for(primary),
    )
    return resp.choices[0].message.content or ""


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=5, max=70),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def structured[T: BaseModel](
    group: str,
    messages: list[dict[str, str]],
    response_model: type[T],
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> T:
    """Structured-output completion: parses response into a Pydantic model.

    Uses Instructor's litellm patch for transparent retry + validation.
    """
    _resolve_provider_keys()
    has_key, missing = _has_provider_for(group)
    if not has_key:
        raise RuntimeError(
            f"No API key available for model group '{group}' "
            f"(need one of: {missing}). Add to .env and rerun."
        )

    import instructor
    from litellm import completion as lite_completion

    client = instructor.from_litellm(lite_completion)
    primary, *_fallbacks = MODEL_GROUPS[group]
    log.debug("llm_structured", group=group, model=primary, schema=response_model.__name__)
    return client.chat.completions.create(
        model=primary,
        messages=messages,
        response_model=response_model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=2,
        api_key=_api_key_for(primary),
    )
