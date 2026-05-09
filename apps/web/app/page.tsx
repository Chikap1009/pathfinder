import Link from "next/link";
import { ArrowRight, Network, BarChart3, FileCog2, Sparkles } from "lucide-react";

const FEATURES = [
  {
    title: "Hybrid retrieval",
    desc: "BM25 + BGE-M3 dense / learned-sparse + ColBERT, fused with RRF (k = 60).",
    icon: Sparkles,
  },
  {
    title: "Knowledge graph",
    desc: "Neo4j 5; Text2Cypher with dynamic few-shot for relational sub-queries.",
    icon: Network,
  },
  {
    title: "Cross-encoder rerank",
    desc: "bge-reranker-v2-m3 FP16 — top 50 → top 10, p95 < 2 s end-to-end.",
    icon: FileCog2,
  },
  {
    title: "Explainable",
    desc: "Per-stage score breakdown, KG path, BM25 term highlights, RAGAS-validated NL.",
    icon: BarChart3,
  },
] as const;

export default function Home() {
  return (
    <section className="mx-auto flex w-full max-w-5xl flex-col gap-16 px-4 py-16 md:py-24">
      <header className="flex flex-col gap-6">
        <span className="border-border bg-muted/40 text-muted-foreground inline-flex w-fit items-center gap-2 rounded-full border px-3 py-1 text-xs font-mono">
          <span className="bg-primary inline-block h-1.5 w-1.5 rounded-full" />
          HCLTech IIT Mandi Hack60 · PS-1
        </span>
        <h1 className="font-mono text-4xl font-semibold leading-tight tracking-tight md:text-6xl">
          Find the right person.
          <br />
          <span className="text-muted-foreground">Explain why they're right.</span>
        </h1>
        <p className="text-muted-foreground max-w-2xl text-lg leading-relaxed">
          PathFinder is an intent-aware, explainable hybrid retrieval engine for a
          skills-and-roles profile corpus. It understands queries like{" "}
          <em className="text-foreground not-italic">
            "Senior Python developer with cloud experience at Competent or higher"
          </em>{" "}
          or{" "}
          <em className="text-foreground not-italic">
            "Find candidates who could fill a Regulatory Affairs Manager role"
          </em>
          , decomposes them, retrieves through three parallel channels, fuses with
          RRF, reranks with a cross-encoder, traverses a Neo4j knowledge graph
          (Person → HAS_SKILL → Skill, Person → CAN_FILL → Role, Skill ↔ ESCO),
          and returns ranked results with citations, matched-skill evidence, and
          per-stage scores.
        </p>
        <div className="flex flex-wrap gap-3">
          <Link
            href="/search"
            className="bg-primary text-primary-foreground hover:bg-primary/90 inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors"
          >
            Try a search <ArrowRight className="h-4 w-4" />
          </Link>
          <Link
            href="/architecture"
            className="border-border hover:bg-accent hover:text-accent-foreground inline-flex items-center gap-2 rounded-md border px-4 py-2 text-sm font-medium transition-colors"
          >
            Architecture
          </Link>
        </div>
      </header>

      <div className="grid gap-4 md:grid-cols-2">
        {FEATURES.map(({ title, desc, icon: Icon }) => (
          <div
            key={title}
            className="border-border bg-card text-card-foreground rounded-lg border p-5"
          >
            <div className="text-primary mb-3 inline-flex h-9 w-9 items-center justify-center rounded-md bg-primary/10">
              <Icon className="h-4 w-4" />
            </div>
            <h3 className="mb-1 font-medium">{title}</h3>
            <p className="text-muted-foreground text-sm">{desc}</p>
          </div>
        ))}
      </div>

      <footer className="border-border text-muted-foreground border-t pt-8 text-xs font-mono">
        Targets: Recall@100 ≥ 0.97 · nDCG@10 ≥ 0.55 · RAGAS Faithfulness ≥ 0.95 · p95 &lt; 2 s.
      </footer>
    </section>
  );
}
