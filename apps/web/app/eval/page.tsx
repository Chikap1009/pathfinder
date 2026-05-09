export const metadata = { title: "Eval · PathFinder" };

const ABLATION = [
  { config: "BM25 only (baseline)",       ndcg: "—", recall: "—", faith: "—", p95: "—" },
  { config: "+ BGE-M3 dense",             ndcg: "—", recall: "—", faith: "—", p95: "—" },
  { config: "+ RRF (k=60)",               ndcg: "—", recall: "—", faith: "—", p95: "—" },
  { config: "+ cross-encoder rerank",     ndcg: "—", recall: "—", faith: "—", p95: "—" },
  { config: "+ KG augmentation",          ndcg: "—", recall: "—", faith: "—", p95: "—" },
  { config: "+ DAT fusion (ablation)",    ndcg: "—", recall: "—", faith: "—", p95: "—" },
] as const;

export default function EvalPage() {
  return (
    <section className="mx-auto flex w-full max-w-5xl flex-col gap-8 px-4 py-12">
      <header>
        <h1 className="font-mono text-3xl font-semibold tracking-tight">Evaluation</h1>
        <p className="text-muted-foreground">
          200-query stratified eval set; RAGAS judge from a different model family
          to mitigate self-bias. Numbers fill in Week 2 / 5.
        </p>
      </header>

      <div className="border-border bg-card overflow-hidden rounded-lg border">
        <table className="w-full text-sm">
          <thead className="border-border bg-muted/40 border-b text-left">
            <tr>
              <th className="px-4 py-3 font-medium">Configuration</th>
              <th className="px-4 py-3 font-medium">nDCG@10</th>
              <th className="px-4 py-3 font-medium">Recall@100</th>
              <th className="px-4 py-3 font-medium">RAGAS Faithfulness</th>
              <th className="px-4 py-3 font-medium">p95 latency</th>
            </tr>
          </thead>
          <tbody>
            {ABLATION.map(({ config, ndcg, recall, faith, p95 }) => (
              <tr key={config} className="border-border border-b last:border-b-0">
                <td className="px-4 py-3 font-mono text-xs">{config}</td>
                <td className="text-muted-foreground px-4 py-3">{ndcg}</td>
                <td className="text-muted-foreground px-4 py-3">{recall}</td>
                <td className="text-muted-foreground px-4 py-3">{faith}</td>
                <td className="text-muted-foreground px-4 py-3">{p95}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-muted-foreground text-xs font-mono">
        Targets: Recall@100 ≥ 0.97 · nDCG@10 ≥ 0.55 · Faithfulness ≥ 0.95 · p95 &lt; 2 s.
      </p>
    </section>
  );
}
