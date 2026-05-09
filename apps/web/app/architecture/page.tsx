export const metadata = { title: "Architecture · PathFinder" };

const STAGES = [
  { stage: "0",  name: "Intent router",       budget: "150 ms", model: "Groq Llama-3.1-8B-Instant" },
  { stage: "1A", name: "BM25 (BM25S BM25+)",  budget: "5 ms",   model: "BM25S, k1=1.5, b=0.75" },
  { stage: "1B", name: "Dense + sparse",      budget: "20 ms",  model: "BGE-M3, Qdrant HNSW M=16" },
  { stage: "1C", name: "Text2Cypher (KG)",    budget: "250 ms", model: "Groq Llama-3.3-70B-versatile" },
  { stage: "2",  name: "RRF fusion",          budget: "5 ms",   model: "k=60, server-side in Qdrant" },
  { stage: "3",  name: "Cross-encoder",       budget: "150 ms", model: "bge-reranker-v2-m3 FP16" },
  { stage: "4",  name: "Explanation",         budget: "800 ms", model: "Gemini 2.5 Flash, JSON schema" },
  { stage: "5",  name: "RAGAS faithfulness",  budget: "200 ms", model: "Gemini judge (different family)" },
] as const;

export default function ArchitecturePage() {
  return (
    <section className="mx-auto flex w-full max-w-5xl flex-col gap-8 px-4 py-12">
      <header>
        <h1 className="font-mono text-3xl font-semibold tracking-tight">Architecture</h1>
        <p className="text-muted-foreground">
          Latency budget targets <strong className="text-foreground">p95 &lt; 2 s</strong> on
          a single RTX 4060. Each stage is independently swappable behind the
          LiteLLM proxy + Qdrant Query API.
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

      <div className="border-border bg-muted/20 text-muted-foreground rounded-lg border border-dashed p-8 text-center text-sm font-mono">
        [ interactive Excalidraw / Mermaid diagram placeholder ]
      </div>

      <footer className="text-muted-foreground text-xs font-mono">
        See <code>docs/architecture.md</code> and <code>docs/decisions/*.md</code> for the deeper write-up.
      </footer>
    </section>
  );
}
