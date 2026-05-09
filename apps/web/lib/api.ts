// Thin fetch wrapper for the FastAPI backend.
// Replace with a typed client (openapi-typescript / openapi-fetch) once the
// backend exposes /openapi.json with the v1 routers wired (Week 4).

const BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export async function apiGet<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

export async function apiPost<TBody, TRes>(
  path: string,
  body: TBody,
  init?: RequestInit,
): Promise<TRes> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    body: JSON.stringify(body),
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`POST ${path} → ${res.status}`);
  return res.json() as Promise<TRes>;
}

export type Health = {
  status: "ok";
  version: string;
  env: "development" | "staging" | "production";
  qdrant: "ready" | "absent";
  neo4j: "ready" | "absent";
};
