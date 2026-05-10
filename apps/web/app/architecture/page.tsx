export const metadata = { title: "Architecture · PathFinder" };

const STAGES = [
  { stage: "0",  name: "Intent router",          budget: "150 ms", model: "Gemini 2.5 Flash-Lite + Instructor (lru-cached)" },
  { stage: "1A", name: "BM25 retrieval",         budget: "0.2 ms", model: "BM25S BM25+, k1=1.5, b=0.75" },
  { stage: "1B", name: "Dense retrieval",        budget: "2.3 ms", model: "BGE-M3 (FP16) → in-memory NumPy cosine" },
  { stage: "1C", name: "KG retrieval",           budget: "25 ms",  model: "Cypher templates over Neo4j AuraDB Free" },
  { stage: "2",  name: "RRF3 fusion",            budget: "0.1 ms", model: "k=60, BM25 + dense + KG ranks" },
  { stage: "3",  name: "Cross-encoder rerank",   budget: "285 ms", model: "bge-reranker-v2-m3 FP16, top-25 funnel" },
  { stage: "4",  name: "Per-stage scores → SSE", budget: "5 ms",   model: "FastAPI StreamingResponse" },
] as const;

export default function ArchitecturePage() {
  return (
    <section className="mx-auto flex w-full max-w-5xl flex-col gap-8 px-4 py-12">
      <header>
        <h1 className="font-mono text-3xl font-semibold tracking-tight">Architecture</h1>
        <p className="text-muted-foreground">
          Measured per-stage latency on the live deploy (rrf3_rerank pipeline,
          119-query overall mean): <strong className="text-foreground">315 ms p95 end-to-end</strong>,
          well under the 2 s target. Stage timings are emitted on every search
          via Server-Sent Events.
        </p>
      </header>

      <div className="border-border bg-card overflow-hidden rounded-lg border">
        <table className="w-full text-sm">
          <thead className="border-border bg-muted/40 border-b text-left">
            <tr>
              <th className="px-4 py-3 font-medium">Stage</th>
              <th className="px-4 py-3 font-medium">Pipeline step</th>
              <th className="px-4 py-3 font-medium">Budget</th>
              <th className="px-4 py-3 font-medium">Model / library</th>
            </tr>
          </thead>
          <tbody>
            {STAGES.map(({ stage, name, budget, model }) => (
              <tr key={stage} className="border-border border-b last:border-b-0">
                <td className="text-primary px-4 py-3 font-mono text-xs">{stage}</td>
                <td className="px-4 py-3">{name}</td>
                <td className="text-muted-foreground px-4 py-3 font-mono">{budget}</td>
                <td className="text-muted-foreground px-4 py-3 font-mono text-xs">{model}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <pre className="border-border bg-muted/20 text-muted-foreground overflow-x-auto rounded-lg border p-6 text-xs leading-relaxed">
{`        query
          │
          ▼
   intent (Gemini)
          │
   ┌──────┼──────┐
   ▼      ▼      ▼
  BM25  dense   KG          ← three parallel channels
   │      │      │
   └──────┼──────┘
          ▼
       RRF k=60               ← rank-fusion (RRF3)
          │
          ▼
   cross-encoder rerank       ← bge-reranker-v2-m3, top-25 funnel
          │
          ▼
   results + per-stage
   scores via SSE`}
      </pre>

      <footer className="text-muted-foreground text-xs font-mono">
        See{" "}
        <a
          href="https://github.com/Chikap1009/pathfinder/blob/main/docs/architecture.md"
          target="_blank"
          rel="noreferrer"
          className="text-foreground hover:underline"
        >
          docs/architecture.md
        </a>{" "}
        and{" "}
        <a
          href="https://github.com/Chikap1009/pathfinder/tree/main/docs/decisions"
          target="_blank"
          rel="noreferrer"
          className="text-foreground hover:underline"
        >
          docs/decisions/
        </a>{" "}
        for the deeper write-up.
      </footer>
    </section>
  );
}
