export const metadata = { title: "Graph · PathFinder" };

export default function GraphPage() {
  return (
    <section className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 py-12">
      <header>
        <h1 className="font-mono text-3xl font-semibold tracking-tight">
          Knowledge graph explorer
        </h1>
        <p className="text-muted-foreground">
          Sigma.js + graphology ForceAtlas2 over the full 50 k-profile graph.
          Wired Week 6. Result subgraphs are rendered with React Flow inside the
          explanation drawer.
        </p>
      </header>

      <div className="border-border bg-muted/20 text-muted-foreground flex aspect-video w-full items-center justify-center rounded-lg border border-dashed text-sm font-mono">
        [ Sigma.js full-KG canvas placeholder ]
      </div>
    </section>
  );
}
