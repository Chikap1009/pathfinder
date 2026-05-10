// Typed fetch client for the FastAPI backend.
// All schemas are imported from the auto-generated `types/api.d.ts` so any
// breakage at the API/UI boundary surfaces as a type error.

import type { components, paths } from "@/types/api";

export const BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

// ─── Re-exported schemas (the types we touch in app code) ────────────────────

export type SearchRequest = components["schemas"]["SearchRequest"];
export type SearchResponse = components["schemas"]["SearchResponse"];
export type SearchResult = components["schemas"]["SearchResult"];
export type SearchResultEntity = components["schemas"]["SearchResultEntity"];
export type StageScores = components["schemas"]["StageScores"];
export type IntentResult = components["schemas"]["IntentResult"];
export type ProfileDetail = components["schemas"]["ProfileDetail"];
export type JobDetail = components["schemas"]["JobDetail"];
export type TimingMs = components["schemas"]["TimingMs"];
export type EvalSummary = components["schemas"]["EvalSummary"];
export type AblationRow = components["schemas"]["AblationRow"];
export type CorpusStats = components["schemas"]["CorpusStats"];
export type KGStats = components["schemas"]["KGStats"];
export type EvalSetStats = components["schemas"]["EvalSetStats"];
export type LatencyStage = components["schemas"]["LatencyStage"];

// SSE event payload — not in the OpenAPI schema (FastAPI doesn't serialize the
// StreamingResponse type), so we mirror the Python pydantic model manually.
export type StageName = "intent" | "encode" | "bm25" | "dense" | "kg" | "rrf" | "rerank" | "all";
export type StageEvent = {
  type: "intent" | "stage_start" | "stage_done" | "results" | "error" | "done";
  stage?: StageName;
  n_candidates?: number;
  elapsed_ms?: number;
  intent?: IntentResult;
  results?: SearchResult[];
  message?: string;
};

export type TargetSide = SearchRequest["target_side"];
export type PipelineMode = NonNullable<SearchRequest["pipeline"]>;

export type Health = {
  status: "ok";
  version: string;
  env: "development" | "staging" | "production";
  qdrant: "ready" | "absent";
  neo4j: "ready" | "absent";
};

// ─── Generic fetch helpers ───────────────────────────────────────────────────

async function _fetch<T>(path: string, init: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...(init.headers ?? {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${init.method ?? "GET"} ${path} → ${res.status} ${body.slice(0, 200)}`);
  }
  return (await res.json()) as T;
}

// ─── Typed endpoints ─────────────────────────────────────────────────────────

export async function getHealth(): Promise<Health> {
  return _fetch<Health>("/health", { method: "GET" });
}

export async function postSearch(req: SearchRequest): Promise<SearchResponse> {
  return _fetch<SearchResponse>("/v1/search", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function getProfile(id: string): Promise<ProfileDetail> {
  type Resp = paths["/v1/profile/{person_id}"]["get"]["responses"]["200"]["content"]["application/json"];
  return _fetch<Resp>(`/v1/profile/${encodeURIComponent(id)}`, { method: "GET" });
}

export async function getJob(id: string): Promise<JobDetail> {
  type Resp = paths["/v1/job/{job_id}"]["get"]["responses"]["200"]["content"]["application/json"];
  return _fetch<Resp>(`/v1/job/${encodeURIComponent(id)}`, { method: "GET" });
}

export async function getEvalSummary(): Promise<EvalSummary> {
  return _fetch<EvalSummary>("/v1/eval/summary", { method: "GET" });
}

// ─── SSE search-stream ───────────────────────────────────────────────────────

/**
 * POST /v1/search/stream as SSE; calls `onEvent` for each StageEvent until done.
 * Returns an `AbortController` so the caller can cancel mid-stream.
 *
 * Use the native `fetch` Streams API instead of `EventSource` because we POST
 * a JSON body (EventSource only supports GET).
 */
export function postSearchStream(
  req: SearchRequest,
  onEvent: (ev: StageEvent) => void,
  onError?: (err: Error) => void,
): AbortController {
  const ac = new AbortController();
  void (async () => {
    try {
      const res = await fetch(`${BASE}/v1/search/stream`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          accept: "text/event-stream",
        },
        body: JSON.stringify(req),
        signal: ac.signal,
        cache: "no-store",
      });
      if (!res.ok || !res.body) {
        throw new Error(`POST /v1/search/stream → ${res.status}`);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buf = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        // SSE messages are separated by a blank line; each message has
        // `event: ...\n` and `data: ...\n` lines.
        const parts = buf.split("\n\n");
        buf = parts.pop() ?? "";
        for (const part of parts) {
          const ev = parseSseMessage(part);
          if (ev) onEvent(ev);
        }
      }
    } catch (err) {
      if ((err as { name?: string })?.name === "AbortError") return;
      onError?.(err as Error);
    }
  })();
  return ac;
}

function parseSseMessage(raw: string): StageEvent | null {
  const lines = raw.split("\n");
  let dataPayload: string | null = null;
  for (const line of lines) {
    if (line.startsWith("data:")) {
      dataPayload = line.slice(5).trim();
    }
  }
  if (!dataPayload) return null;
  try {
    return JSON.parse(dataPayload) as StageEvent;
  } catch {
    return null;
  }
}
