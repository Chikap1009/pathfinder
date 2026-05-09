"""Skill canonicalisation pipeline — alias YAML → ESCO API → BGE-M3 cosine.

Three-layer canonicaliser that maps raw skill names from the corpus
(`profile_skills`, `job_skills`) to a canonical form with optional ESCO URI.

Layer 1 — hand-curated alias YAML at `data/aliases/skills.yaml`. Deterministic;
covers obvious abbreviations (k8s → Kubernetes, js → JavaScript, …).

Layer 2 — ESCO REST API search. ESCO is the EU Skills/Competences/Qualifications
ontology (CC-BY, ~13,900 skills with hierarchy + relatedness). We query
`/search?type=skill&text=...&language=en` and accept the top hit if its name
matches our input within a small Levenshtein / token-set threshold.

Layer 3 — BGE-M3 dense embedding cosine fallback against the corpus skill
vocabulary itself (already canonicalised so far). Lets us collapse near-misses
that ESCO doesn't cover (e.g. domain-specific jargon).

The pipeline is offline + idempotent: results are persisted to
`data/processed/skills/canonical.parquet` so subsequent runs (and Neo4j ingest)
are deterministic.
"""

from __future__ import annotations

import functools
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

from app.core.logging import get_logger

log = get_logger(__name__)


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pnpm-workspace.yaml").exists():
            return parent
    raise RuntimeError("repo root marker not found")


REPO_ROOT = _find_repo_root()
ALIAS_YAML = REPO_ROOT / "data" / "aliases" / "skills.yaml"

ESCO_API = "https://esco.ec.europa.eu/api/search"
ESCO_TIMEOUT_S = 8.0
ESCO_THROTTLE_S = 0.05  # 20 req / s, well under any soft cap


# ─── Output type ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CanonicalSkill:
    raw_name: str  # what we observed in the data
    canonical_id: str  # slug(canonical_name)
    canonical_name: str
    source: str  # "alias" | "esco" | "embedding" | "raw"
    esco_uri: str | None
    confidence: float  # 0..1


# ─── Layer 1: alias YAML ────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def load_alias_table() -> tuple[dict[str, str], dict[str, str]]:
    """Returns (alias_lower → canonical_name, canonical_lower → canonical_name)."""
    if not ALIAS_YAML.exists():
        return {}, {}
    with ALIAS_YAML.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    alias_to_canon: dict[str, str] = {}
    canon_lower_to_canon: dict[str, str] = {}
    for raw_canonical, aliases in data.items():
        canonical = str(raw_canonical).strip()
        if not canonical:
            continue
        canon_lower_to_canon[canonical.lower()] = canonical
        alias_to_canon[canonical.lower()] = canonical
        for alias in aliases or []:
            alias_to_canon[str(alias).strip().lower()] = canonical
    return alias_to_canon, canon_lower_to_canon


def _normalise(s: str) -> str:
    """Strip / lowercase / collapse whitespace for fuzzy comparison."""
    # Unicode dash range (U+2010 .. U+2015) -> ASCII hyphen-minus.
    s = re.sub(r"[‐-―]", "-", s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


# ─── Layer 2: ESCO API ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EscoHit:
    title: str
    uri: str
    score: float


def _esco_search(term: str, *, top_k: int = 5) -> list[EscoHit]:
    """Hit ESCO `/search?type=skill&text=...`. Returns parsed top hits."""
    try:
        r = httpx.get(
            ESCO_API,
            params={
                "type": "skill",
                "text": term,
                "language": "en",
                "limit": top_k,
                # Some ESCO instances require a User-Agent header; set one.
            },
            headers={"User-Agent": "PathFinder/0.1 (research-prototype)"},
            timeout=ESCO_TIMEOUT_S,
        )
        if r.status_code != 200:
            log.debug("esco_non_200", status=r.status_code, term=term)
            return []
        body = r.json()
    except httpx.HTTPError as exc:
        log.debug("esco_http_error", error=str(exc), term=term)
        return []

    # ESCO's response shape: { "_embedded": { "results": [ { "title": ..., "uri": ... } ] } }
    results = (body.get("_embedded") or {}).get("results") or []
    out: list[EscoHit] = []
    for item in results[:top_k]:
        title = (item.get("title") or item.get("preferredLabel") or "").strip()
        uri = item.get("uri") or item.get("href") or ""
        if title and uri:
            # ESCO doesn't return a similarity score; we'll compute one below.
            out.append(EscoHit(title=title, uri=uri, score=0.0))
    return out


def _token_set_similarity(a: str, b: str) -> float:
    """Token-set Jaccard, simple and fast."""
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _esco_match(term: str, *, threshold: float = 0.7) -> EscoHit | None:
    """Return the ESCO hit whose title is most similar to `term`, if above threshold."""
    hits = _esco_search(term, top_k=5)
    if not hits:
        return None
    best: tuple[EscoHit | None, float] = (None, 0.0)
    for h in hits:
        sim = _token_set_similarity(term, h.title)
        if sim > best[1]:
            best = (EscoHit(title=h.title, uri=h.uri, score=sim), sim)
    if best[0] is not None and best[1] >= threshold:
        return best[0]
    return None


# ─── Layer 3: BGE-M3 embedding cosine fallback ──────────────────────────────


def _embedding_cluster(
    raw_names: list[str],
    *,
    sim_threshold: float = 0.92,
) -> dict[str, str]:
    """Cluster raw skill names by BGE-M3 cosine similarity; return raw → cluster_canonical.

    Greedy single-pass clusterer:
      1. Sort raw names by length (descending) — longer forms tend to be canonical.
      2. For each name, compare against existing cluster centroids; if cosine > threshold
         attach to that cluster, else open a new cluster with this name as centroid.

    The raw_names list should already have alias-table + ESCO matches removed
    (we only fall back here for the long tail).
    """
    if not raw_names:
        return {}

    # Lazy import — keeps the module importable when [ml] extra isn't installed.
    import numpy as np

    from app.services.dense import encode_dense

    log.info("skill_embedding_cluster", n=len(raw_names), threshold=sim_threshold)
    vecs = np.asarray(encode_dense(raw_names, batch_size=64, max_length=64), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs = vecs / norms

    # Sort by length descending so longer canonical forms get the centroid slot.
    order = sorted(range(len(raw_names)), key=lambda i: -len(raw_names[i]))
    cluster_centroids: list[np.ndarray] = []
    cluster_names: list[str] = []
    raw_to_cluster: dict[str, str] = {}

    for idx in order:
        v = vecs[idx]
        name = raw_names[idx]
        if not cluster_centroids:
            cluster_centroids.append(v)
            cluster_names.append(name)
            raw_to_cluster[name] = name
            continue
        sims = np.stack(cluster_centroids) @ v
        best_idx = int(np.argmax(sims))
        if float(sims[best_idx]) >= sim_threshold:
            raw_to_cluster[name] = cluster_names[best_idx]
        else:
            cluster_centroids.append(v)
            cluster_names.append(name)
            raw_to_cluster[name] = name
    return raw_to_cluster


# ─── Public canonicalise() — single skill at a time, used in unit tests ─────


def canonicalise_one(
    raw_name: str,
    *,
    use_esco: bool = True,
) -> CanonicalSkill:
    """Canonicalise a single skill (no embedding fallback at this layer)."""
    raw = raw_name.strip()
    if not raw:
        return CanonicalSkill(raw, "", "", "raw", None, 0.0)

    alias_map, _canon_map = load_alias_table()
    n = _normalise(raw)
    if n in alias_map:
        canon = alias_map[n]
        return CanonicalSkill(raw, _slug(canon), canon, "alias", None, 1.0)

    if use_esco:
        hit = _esco_match(raw)
        if hit is not None:
            time.sleep(ESCO_THROTTLE_S)
            return CanonicalSkill(
                raw, _slug(hit.title), hit.title, "esco", hit.uri, float(hit.score)
            )

    # Fallback: keep the raw name as canonical (Title Cased), no source enrichment.
    titled = " ".join(p.capitalize() if p.islower() else p for p in raw.split())
    return CanonicalSkill(raw, _slug(titled), titled, "raw", None, 0.5)


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s or "unknown"
