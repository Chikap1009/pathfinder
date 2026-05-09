import { Search } from "lucide-react";

export const metadata = { title: "Search · PathFinder" };

export default function SearchPage() {
  return (
    <section className="mx-auto flex w-full max-w-5xl flex-col gap-8 px-4 py-12">
      <header className="flex flex-col gap-2">
        <h1 className="font-mono text-3xl font-semibold tracking-tight">Search</h1>
        <p className="text-muted-foreground">
          Two-sided search — candidates, jobs, or candidate ↔ job match. Streaming
          Perplexity-style results. Wired Week 4.
        </p>
      </header>

      <form className="border-border bg-card flex items-center gap-2 rounded-lg border p-2">
        <Search className="text-muted-foreground ml-2 h-4 w-4" />
        <input
          name="q"
          type="text"
          placeholder="Try: ‘Senior Python developer with cloud experience at Competent+’ or ‘Candidates who could fill a Regulatory Affairs Manager role’"
          className="text-foreground placeholder:text-muted-foreground flex-1 bg-transparent px-2 py-2 outline-none"
          disabled
        />
        <button
          type="submit"
          disabled
          className="bg-primary text-primary-foreground rounded-md px-4 py-2 text-sm opacity-50"
        >
          Search
        </button>
      </form>

      <div className="border-border bg-muted/20 text-muted-foreground rounded-lg border border-dashed p-8 text-center text-sm">
        <p className="font-mono">[ retrieval pipeline placeholder ]</p>
        <p className="mt-2">
          Stages will stream as chips: <span className="text-foreground">intent → BM25 → dense → KG → fuse → rerank → explain</span>.
        </p>
      </div>
    </section>
  );
}
