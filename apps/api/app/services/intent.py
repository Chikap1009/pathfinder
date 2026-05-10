"""Intent classifier — query → IntentResult via Instructor + Gemini Flash-Lite.

Used by the /v1/search endpoint to decide:
  - Which side of the corpus to search (candidates vs jobs)
  - Which canonical skills the query mentions
  - Whether a minimum proficiency was requested
  - The free-text semantic chunk (for dense retrieval)

Cheap (~150 ms / call on Gemini 2.5 Flash-Lite). Cached via lru_cache for
identical query strings within a process — duplicate requests are free.
"""

from __future__ import annotations

import functools

from app.api.v1.schemas import IntentResult, TargetSide
from app.core.logging import get_logger
from app.services.llm import MODEL_GROUPS, _api_key_for, _resolve_provider_keys

log = get_logger(__name__)

# Hard timeout for the LLM call (seconds). The intent path must never block search;
# if the LLM doesn't respond fast we fall back to the heuristic classifier.
INTENT_LLM_TIMEOUT_S = 6.0


SYSTEM_PROMPT = """\
You classify a recruiter / user search query into a structured IntentResult.

Rules:
- target_side: 'candidates' if looking for people, 'jobs' if looking for roles/openings.
  Ambiguous? Prefer 'candidates' for verbs like 'find', 'who knows', 'looking for'.
  Prefer 'jobs' for verbs like 'roles', 'positions', 'openings', 'jobs available'.

- query_skills: only ATOMIC skill names. Title-case them, dedupe. Skip generic words
  like 'engineer', 'developer', 'role', 'someone', 'person'. Domain-specific terms
  like 'Procurement', 'Risk Management', 'Selenium' DO count as skills.

- min_proficiency: one of Beginner / Advanced Beginner / Competent / Proficient / Expert,
  or 'Any' if unclear or not specified. The Dreyfus 5-stage scale.
  'senior' / 'lead' usually maps to Proficient or Expert.
  'junior' / 'entry-level' / 'fresher' maps to Beginner or Advanced Beginner.

- designation_hint: capture verbatim if the query names a specific role title
  (e.g. 'Test Manager', 'Senior Java Developer', 'Data Engineer'). Else null.

- free_text_intent: the residual semantic chunk after stripping skills + designation.
  Used as the dense-retrieval query. Can be the original query lightly cleaned up.

Be conservative. Do NOT invent skills the user didn't mention. Empty list is fine.
"""

EXAMPLES = [
    {
        "query": "Senior Python developer with cloud experience at Competent or higher",
        "result": {
            "target_side": "candidates",
            "query_skills": ["Python", "Cloud"],
            "min_proficiency": "Competent",
            "designation_hint": "Senior Python Developer",
            "free_text_intent": "senior Python developer with cloud experience",
        },
    },
    {
        "query": "Find a Test Manager role in Bengaluru with Selenium and Azure",
        "result": {
            "target_side": "jobs",
            "query_skills": ["Selenium", "Azure"],
            "min_proficiency": "Any",
            "designation_hint": "Test Manager",
            "free_text_intent": "Test Manager role in Bengaluru with Selenium and Azure",
        },
    },
    {
        "query": "anyone competent in regulatory affairs + brand marketing for cosmetics?",
        "result": {
            "target_side": "candidates",
            "query_skills": ["Regulatory Affairs", "Brand Marketing"],
            "min_proficiency": "Competent",
            "designation_hint": None,
            "free_text_intent": "regulatory affairs and brand marketing in cosmetics",
        },
    },
]


def _build_user_prompt(query: str) -> str:
    examples_text = "\n\n".join(f"Input: {e['query']}\nOutput: {e['result']!r}" for e in EXAMPLES)
    return f"""Examples:
{examples_text}

Now classify:
Input: {query}
Output:"""


def _classify_via_llm(query: str) -> IntentResult:
    """Single-shot LLM call with a hard timeout. No retries (search must not block)."""
    import instructor
    from litellm import completion as lite_completion

    _resolve_provider_keys()
    client = instructor.from_litellm(lite_completion)
    primary, *_ = MODEL_GROUPS["paraphraser"]
    return client.chat.completions.create(
        model=primary,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(query)},
        ],
        response_model=IntentResult,
        temperature=0.0,
        max_tokens=512,
        max_retries=0,
        timeout=INTENT_LLM_TIMEOUT_S,
        api_key=_api_key_for(primary),
    )


@functools.lru_cache(maxsize=512)
def classify_intent(query: str) -> IntentResult:
    """Cached intent classification. LLM-first with fast heuristic fallback."""
    log.debug("intent_classify", query=query[:80])
    try:
        return _classify_via_llm(query)
    except Exception as exc:
        log.warning("intent_llm_fallback", error=str(exc)[:200])
        return _heuristic_fallback(query)


def _heuristic_fallback(query: str) -> IntentResult:
    """If the LLM is down / quota-exhausted, fall back to a simple keyword classifier."""
    q = query.lower()
    job_signals = (
        "role",
        "roles",
        "position",
        "positions",
        "opening",
        "openings",
        "vacancy",
        "vacancies",
        "job",
        "jobs",
    )
    target = TargetSide.jobs if any(w in q for w in job_signals) else TargetSide.candidates
    return IntentResult(
        target_side=target,
        query_skills=[],
        min_proficiency="Any",
        designation_hint=None,
        free_text_intent=query.strip(),
    )
