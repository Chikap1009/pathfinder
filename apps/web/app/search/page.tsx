"use client";

import { useCallback, useState } from "react";
import { Search, Loader2, ChevronRight, MapPin, Briefcase } from "lucide-react";
import {
  postSearchStream,
  type IntentResult,
  type PipelineMode,
  type SearchResult,
  type StageEvent,
} from "@/lib/api";
import { ResultDrawer } from "@/components/result-drawer";
import { cn } from "@/lib/utils";

const STAGE_LABELS: Record<string, string> = {
  intent: "Intent",
  encode: "Encode",
  bm25: "BM25",
  dense: "Dense",
  kg: "KG",
  rrf: "RRF",
  rerank: "Rerank",
  all: "Pipeline",
};

const PIPELINE_OPTIONS: { value: PipelineMode; label: string; hint: string }[] = [
  { value: "rrf3_rerank", label: "Full pipeline", hint: "BM25+Dense+KG → rerank" },
  { value: "rrf3", label: "RRF3", hint: "BM25+Dense+KG fusion" },
  { value: "rrf", label: "RRF", hint: "BM25+Dense fusion" },
  { value: "dense", label: "Dense only", hint: "BGE-M3 only" },
  { value: "bm25", label: "BM25 only", hint: "Lexical only" },
];

const EXAMPLE_QUERIES = [
  "Senior Python developer with cloud experience at Competent or higher",
  "Test Manager role in Bengaluru with Selenium and Azure",
  "Anyone competent in regulatory affairs and brand marketing for cosmetics",
  "Java SpringBoot Microservices engineer",
];

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [pipeline, setPipeline] = useState<PipelineMode>("rrf3_rerank");
  const [running, setRunning] = useState(false);
  const [intent, setIntent] = useState<IntentResult | null>(null);
  const [stages, setStages] = useState<{ stage: string; ms: number }[]>([]);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [nCandidates, setNCandidates] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<SearchResult | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [abortCtl, setAbortCtl] = useState<AbortController | null>(null);

  const submit = useCallback(
    (q: string) => {
      const trimmed = q.trim();
      if (!trimmed || running) return;

      // Cancel any in-flight request
      abortCtl?.abort();

      setRunning(true);
      setError(null);
      setIntent(null);
      setStages([]);
      setResults([]);
      setNCandidates(null);

      const ctl = postSearchStream(
        { query: trimmed, pipeline, top_k: 10 },
        (ev: StageEvent) => {
          switch (ev.type) {
            case "intent":
              if (ev.intent) setIntent(ev.intent);
              break;
            case "stage_done":
              if (ev.stage && typeof ev.elapsed_ms === "number") {
                setStages((s) => [...s, { stage: ev.stage as string, ms: ev.elapsed_ms as number }]);
              }
              break;
            case "results":
              if (ev.results) setResults(ev.results);
              if (typeof ev.n_candidates === "number") setNCandidates(ev.n_candidates);
              break;
            case "done":
              setRunning(false);
              break;
            case "error":
              setError(ev.message ?? "Unknown error");
              setRunning(false);
              break;
            default:
              break;
          }
        },
        (err) => {
          setError(err.message);
          setRunning(false);
        },
      );
      setAbortCtl(ctl);
    },
    [pipeline, running, abortCtl],
  );

  return (
    <section className="mx-auto flex w-full max-w-5xl flex-col gap-6 px-4 py-12">
      <header className="flex flex-col gap-2">
        <h1 className="font-mono text-3xl font-semibold tracking-tight">Search</h1>
        <p className="text-muted-foreground">
          Two-sided hybrid retrieval — candidates, jobs, or candidate ↔ job match. Streaming
          per-stage scores via SSE.
        </p>
      </header>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit(query);
        }}
        className="border-border bg-card flex items-center gap-2 rounded-lg border p-2"
      >
        <Search className="text-muted-foreground ml-2 h-4 w-4" />
        <input
          type="text"
          name="q"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Try: 'Senior Python developer with cloud experience at Competent+'"
          className="text-foreground placeholder:text-muted-foreground flex-1 bg-transparent px-2 py-2 outline-none"
          disabled={running}
        />
        <select
          value={pipeline}
          onChange={(e) => setPipeline(e.target.value as PipelineMode)}
          className="border-border bg-background text-muted-foreground hover:text-foreground rounded border px-2 py-1 text-xs font-mono"
          disabled={running}
          aria-label="Pipeline mode"
        >
          {PIPELINE_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
        <button
          type="submit"
          disabled={running || !query.trim()}
          className="bg-primary text-primary-foreground hover:bg-primary/90 inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50"
        >
          {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
          Search
        </button>
      </form>

      {/* Example query chips */}
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-muted-foreground mr-1 text-xs">Try:</span>
        {EXAMPLE_QUERIES.map((q) => (
          <button
            key={q}
            type="button"
            onClick={() => {
              setQuery(q);
              submit(q);
            }}
            disabled={running}
            className="border-border hover:bg-accent hover:text-accent-foreground rounded-full border px-2.5 py-1 text-xs transition-colors disabled:opacity-50"
          >
            {q}
          </button>
        ))}
      </div>

      {error ? (
        <div className="border-destructive/30 bg-destructive/5 text-destructive rounded border p-3 text-sm">
          {error}
        </div>
      ) : null}

      {/* Intent + stage chips */}
      {intent || stages.length > 0 ? (
        <div className="border-border bg-card rounded-lg border p-4">
          {intent ? (
            <div className="text-muted-foreground mb-3 flex flex-wrap items-center gap-1.5 text-xs">
              <span className="text-foreground font-medium">intent:</span>
              <span className="border-border rounded border px-1.5 py-0.5 font-mono text-[10px]">
                target = {intent.target_side}
              </span>
              {intent.designation_hint ? (
                <span className="border-border rounded border px-1.5 py-0.5 font-mono text-[10px]">
                  designation = {intent.designation_hint}
                </span>
              ) : null}
              {intent.min_proficiency !== "Any" ? (
                <span className="border-border rounded border px-1.5 py-0.5 font-mono text-[10px]">
                  ≥ {intent.min_proficiency}
                </span>
              ) : null}
              {(intent.query_skills?.length ?? 0) > 0 ? (
                <span className="border-border rounded border px-1.5 py-0.5 font-mono text-[10px]">
                  skills = {(intent.query_skills ?? []).join(", ")}
                </span>
              ) : null}
            </div>
          ) : null}
          <div className="flex flex-wrap items-center gap-1.5">
            {stages.map((s) => (
              <span
                key={s.stage}
                className="border-primary/30 bg-primary/5 text-foreground rounded-full border px-2.5 py-1 font-mono text-[10px]"
              >
                {STAGE_LABELS[s.stage] ?? s.stage} · {s.ms.toFixed(1)} ms
              </span>
            ))}
            {running ? (
              <span className="text-muted-foreground inline-flex items-center gap-1.5 text-[10px] font-mono">
                <Loader2 className="h-3 w-3 animate-spin" /> running…
              </span>
            ) : null}
            {nCandidates !== null ? (
              <span className="text-muted-foreground ml-auto text-[10px] font-mono">
                {nCandidates} candidates · {results.length} returned
              </span>
            ) : null}
          </div>
        </div>
      ) : null}

      {/* Results list */}
      {results.length > 0 ? (
        <ul className="flex flex-col gap-2">
          {results.map((r) => (
            <ResultRow
              key={`${r.entity.kind}:${r.entity.id}`}
              result={r}
              onOpen={() => {
                setSelected(r);
                setDrawerOpen(true);
              }}
            />
          ))}
        </ul>
      ) : (
        !running && stages.length === 0 ? (
          <div className="border-border bg-muted/20 text-muted-foreground rounded-lg border border-dashed p-8 text-center text-sm">
            <p className="font-mono">[ retrieval pipeline ready ]</p>
            <p className="mt-2">
              Submit a query above. Stages stream live as chips:{" "}
              <span className="text-foreground">
                intent → encode → bm25 → dense → kg → rrf → rerank
              </span>
              .
            </p>
          </div>
        ) : null
      )}

      <ResultDrawer open={drawerOpen} onClose={() => setDrawerOpen(false)} result={selected} />
    </section>
  );
}

// ─── Result row ──────────────────────────────────────────────────────────────

function ResultRow({ result, onOpen }: { result: SearchResult; onOpen: () => void }) {
  const e = result.entity;
  const title =
    e.kind === "person"
      ? e.name?.trim() || `Person ${e.id}`
      : (e.designation ?? e.title ?? `Job ${e.id}`);
  const sub =
    e.kind === "person"
      ? `${e.years_experience ?? 0}y experience · ${e.id}`
      : [e.industry, e.city, e.country].filter(Boolean).join(" · ");

  return (
    <li>
      <button
        type="button"
        onClick={onOpen}
        className="border-border bg-card hover:bg-accent/30 group flex w-full items-start gap-4 rounded-lg border p-4 text-left transition-colors"
      >
        <span className="bg-primary/10 text-primary flex h-9 w-9 shrink-0 items-center justify-center rounded font-mono text-xs">
          #{result.rank}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="border-border text-muted-foreground rounded border px-1.5 py-0.5 font-mono text-[10px]">
              {e.kind}
            </span>
            <h3 className="truncate font-medium">{title}</h3>
          </div>
          {sub ? (
            <p className="text-muted-foreground mt-1 flex items-center gap-1 text-xs">
              {e.kind === "person" ? (
                <Briefcase className="h-3 w-3" />
              ) : (
                <MapPin className="h-3 w-3" />
              )}
              {sub}
            </p>
          ) : null}
          {e.snippet ? (
            <p className="text-muted-foreground mt-2 line-clamp-2 text-sm">{e.snippet}</p>
          ) : null}
          {(e.matched_skills?.length ?? 0) > 0 ? (
            <div className="mt-2 flex flex-wrap gap-1">
              {(e.matched_skills ?? []).map((s) => (
                <span
                  key={s}
                  className="border-primary/30 bg-primary/5 text-foreground rounded border px-1.5 py-0.5 text-[10px]"
                >
                  {s}
                </span>
              ))}
            </div>
          ) : null}
        </div>
        <div className="text-muted-foreground flex flex-col items-end gap-1 font-mono text-xs">
          <span className="tabular-nums">{result.score.toFixed(3)}</span>
          <ChevronRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
        </div>
      </button>
    </li>
  );
}
