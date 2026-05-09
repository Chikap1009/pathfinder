type Props = { params: Promise<{ id: string }> };

export const metadata = { title: "Profile · PathFinder" };

export default async function ResultPage({ params }: Props) {
  const { id } = await params;
  return (
    <section className="mx-auto flex w-full max-w-5xl flex-col gap-6 px-4 py-12">
      <header>
        <p className="text-muted-foreground font-mono text-xs">profile / {id}</p>
        <h1 className="font-mono text-3xl font-semibold tracking-tight">
          Profile detail
        </h1>
        <p className="text-muted-foreground mt-2">
          Full profile + explanation drawer (per-stage scores, KG path, term
          highlights). Wired Week 5.
        </p>
      </header>

      <div className="border-border bg-muted/20 text-muted-foreground rounded-lg border border-dashed p-12 text-center text-sm font-mono">
        [ explanation drawer placeholder ]
      </div>
    </section>
  );
}
