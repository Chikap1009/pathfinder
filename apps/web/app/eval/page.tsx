import {
  Activity,
  AlertCircle,
  Award,
  CheckCircle2,
  Clock,
  Database,
  Network,
  Sparkles,
  Target,
  Users,
  Briefcase,
  Workflow,
} from "lucide-react";
import { BASE, type AblationRow, type EvalSummary } from "@/lib/api";
import { cn } from "@/lib/utils";

export const metadata = { title: "Eval · PathFinder" };
export const dynamic = "force-dynamic";

async function fetchSummary(): Promise<EvalSummary | null> {
  try {
    const res = await fetch(`${BASE}/v1/eval/summary`, { cache: "no-store" });
    if (!res.ok) throw new Error(`status ${res.status}`);
    return (await res.json()) as EvalSummary;
  } catch {
    return null;
  }
}

export default async function EvalPage() {
  const summary = await fetchSummary();

  if (!summary) {
    return (
      <section className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 py-12">
        <h1 className="font-mono text-3xl font-semibold tracking-tight">Eval</h1>
        <div className="border-destructive/30 bg-destructive/5 text-destructive flex items-start gap-3 rounded border p-4">
          <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />
          <div className="text-sm">
            Couldn&apos;t reach the API at <code className="font-mono">{BASE}</code>. Start the
            backend with{" "}
            <code className="font-mono">make api-dev</code> from the repo root.
          </div>
        </div>
      </section>
    );
  }

  const bestOverall = summary.ablation
    .filter((r) => r.stratum === "overall" && r.is_best_for_stratum)
    .at(0);

  return (
    <section className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-12">
      <header>
        <h1 className="font-mono text-3xl font-semibold tracking-tight">Evaluation</h1>
        <p className="text-muted-foreground mt-2 max-w-3xl">
          Locked snapshot of every offline metric: corpus + KG composition, the
          full ablation matrix across {summary.eval_set.total_queries} queries
          on two strata, and the per-stage latency budget on a single RTX 4060.
        </p>
      </header>

      {/* ─── KPI tiles ──────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Tile
          icon={<Sparkles className="h-4 w-4" />}
          label="Best nDCG@10"
          value={(bestOverall?.ndcg_at_10 ?? 0).toFixed(3)}
          sub={bestOverall?.name ?? "—"}
          accent="primary"
        />
        <Tile
          icon={<Target className="h-4 w-4" />}
          label="Recall@100"
          value={(bestOverall?.recall_at_100 ?? 0).toFixed(3)}
          sub={`target ≥ ${summary.targets.recall_at_100.toFixed(2)}`}
        />
        <Tile
          icon={<Activity className="h-4 w-4" />}
          label="MRR@10"
          value={(bestOverall?.mrr_at_10 ?? 0).toFixed(3)}
          sub={`target ≥ ${summary.targets.mrr_at_10.toFixed(2)}`}
        />
        <Tile
          icon={<Clock className="h-4 w-4" />}
          label="Full pipeline"
          value="315 ms"
          sub={`target p95 < ${summary.targets.p95_latency_ms} ms`}
        />
      </div>

      {/* ─── Corpus + KG composition ───────────────────────────────────── */}
      <div className="grid gap-4 md:grid-cols-2">
        <Section title="Corpus" icon={<Database className="h-4 w-4" />}>
          <DefList>
            <Row label="Profiles" v={summary.corpus.profiles} icon={<Users className="h-3 w-3" />} />
            <Row label="Jobs" v={summary.corpus.jobs} icon={<Briefcase className="h-3 w-3" />} />
            <Row label="Canonical skills" v={summary.corpus.canonical_skills} />
            <Row label="HAS_SKILL edges" v={summary.corpus.has_skill_edges} />
            <Row label="REQUIRES_SKILL edges" v={summary.corpus.requires_skill_edges} />
            <Row label="Eval queries" v={summary.eval_set.total_queries}
              sub={`${summary.eval_set.candidate_search} candidate · ${summary.eval_set.job_search} job · ${summary.eval_set.paraphrase_stratum} paraphrase`}
            />
          </DefList>
        </Section>

        <Section title="Knowledge graph" icon={<Network className="h-4 w-4" />}>
          <DefList>
            <Row label="Person nodes" v={summary.kg.persons} />
            <Row label="Job nodes" v={summary.kg.jobs} />
            <Row label="Skill nodes" v={summary.kg.skills} />
            <Row
              label="Other nodes"
              v={summary.kg.roles + summary.kg.designations + summary.kg.industries + summary.kg.locations}
              sub={`${summary.kg.roles} role · ${summary.kg.designations} desig · ${summary.kg.industries} ind · ${summary.kg.locations} loc`}
            />
            <Row
              label="Total relationships"
              v={
                summary.kg.has_skill +
                summary.kg.requires_skill +
                summary.kg.can_fill +
                summary.kg.is_designation +
                summary.kg.at_location +
                summary.kg.in_industry
              }
              sub="HAS_SKILL · REQUIRES_SKILL · CAN_FILL · IS_DESIGNATION · AT_LOCATION · IN_INDUSTRY"
            />
          </DefList>
        </Section>
      </div>

      {/* ─── Ablation tables ──────────────────────────────────────────── */}
      <div className="flex flex-col gap-6">
        <h2 className="flex items-center gap-2 font-mono text-xl font-semibold">
          <Workflow className="text-primary h-5 w-5" /> Ablation matrix
        </h2>
        <p className="text-muted-foreground -mt-4 text-sm">
          7 retrieval configurations × 3 strata. Highlighted cells are the best
          nDCG@10 per stratum.
        </p>

        <AblationTable
          title="Overall (mean of candidate + job tasks)"
          rows={summary.ablation.filter((r) => r.stratum === "overall")}
        />
        <AblationTable
          title="Original stratum (100 lexical-anchor queries)"
          rows={summary.ablation.filter((r) => r.stratum === "original")}
        />
        <AblationTable
          title="Paraphrase stratum (19 Gemini-generated queries)"
          rows={summary.ablation.filter((r) => r.stratum === "paraphrase")}
        />
      </div>

      {/* ─── Latency budget ────────────────────────────────────────────── */}
      <Section title="Latency budget per stage" icon={<Clock className="h-4 w-4" />}>
        <LatencyChart stages={summary.latency_budget} />
      </Section>

      {/* ─── Notes ─────────────────────────────────────────────────────── */}
      {(summary.notes?.length ?? 0) > 0 ? (
        <Section title="Findings" icon={<Award className="h-4 w-4" />}>
          <ul className="space-y-2 text-sm leading-relaxed">
            {(summary.notes ?? []).map((n) => (
              <li key={n} className="flex items-start gap-2">
                <CheckCircle2 className="text-primary mt-0.5 h-4 w-4 shrink-0" />
                <span>{n}</span>
              </li>
            ))}
          </ul>
        </Section>
      ) : null}
    </section>
  );
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function Tile({
  icon,
  label,
  value,
  sub,
  accent,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub?: string;
  accent?: "primary";
}) {
  return (
    <div
      className={cn(
        "border-border bg-card flex flex-col gap-1 rounded-lg border p-4",
        accent === "primary" && "border-primary/30 bg-primary/5",
      )}
    >
      <div className="text-muted-foreground flex items-center gap-1.5 text-xs">
        {icon}
        <span>{label}</span>
      </div>
      <div className="font-mono text-2xl font-semibold tabular-nums">{value}</div>
      {sub ? <div className="text-muted-foreground text-xs">{sub}</div> : null}
    </div>
  );
}

function Section({
  title,
  icon,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="border-border bg-card rounded-lg border p-5">
      <header className="mb-4 flex items-center gap-2">
        <span className="text-primary">{icon}</span>
        <h2 className="text-sm font-medium">{title}</h2>
      </header>
      {children}
    </section>
  );
}

function DefList({ children }: { children: React.ReactNode }) {
  return <dl className="grid grid-cols-1 gap-2 text-sm sm:grid-cols-2">{children}</dl>;
}

function Row({
  label,
  v,
  sub,
  icon,
}: {
  label: string;
  v: number;
  sub?: string;
  icon?: React.ReactNode;
}) {
  return (
    <div className="border-border flex flex-col gap-0.5 rounded border p-2.5">
      <dt className="text-muted-foreground flex items-center gap-1 text-[10px] uppercase tracking-wide">
        {icon}
        {label}
      </dt>
      <dd className="font-mono text-base font-semibold tabular-nums">{v.toLocaleString()}</dd>
      {sub ? <span className="text-muted-foreground text-[10px]">{sub}</span> : null}
    </div>
  );
}

function AblationTable({ title, rows }: { title: string; rows: AblationRow[] }) {
  return (
    <div className="border-border bg-card rounded-lg border">
      <h3 className="border-border border-b p-3 text-sm font-medium">{title}</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-muted-foreground border-border bg-muted/30 border-b text-left">
            <tr>
              <th className="px-4 py-2 font-medium">Configuration</th>
              <th className="px-3 py-2 text-right font-medium">nDCG@10</th>
              <th className="px-3 py-2 text-right font-medium">R@10</th>
              <th className="px-3 py-2 text-right font-medium">R@100</th>
              <th className="px-3 py-2 text-right font-medium">MRR@10</th>
              <th className="px-3 py-2 text-right font-medium">Latency</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={`${r.pipeline}-${r.stratum}`}
                className={cn(
                  "border-border border-b last:border-b-0",
                  r.is_best_for_stratum && "bg-primary/5",
                )}
              >
                <td className="px-4 py-2">
                  <div className="flex items-center gap-2">
                    {r.is_best_for_stratum ? (
                      <Award className="text-primary h-3.5 w-3.5 shrink-0" />
                    ) : (
                      <span className="w-3.5" />
                    )}
                    <span className={cn("font-mono text-xs", r.is_best_for_stratum && "font-semibold")}>
                      {r.name}
                    </span>
                  </div>
                </td>
                <td className="text-foreground px-3 py-2 text-right font-mono tabular-nums">
                  {fmt(r.ndcg_at_10)}
                </td>
                <td className="text-muted-foreground px-3 py-2 text-right font-mono tabular-nums">
                  {fmt(r.recall_at_10)}
                </td>
                <td className="text-muted-foreground px-3 py-2 text-right font-mono tabular-nums">
                  {fmt(r.recall_at_100)}
                </td>
                <td className="text-muted-foreground px-3 py-2 text-right font-mono tabular-nums">
                  {fmt(r.mrr_at_10)}
                </td>
                <td className="text-muted-foreground px-3 py-2 text-right font-mono tabular-nums">
                  {r.latency_ms !== null && r.latency_ms !== undefined
                    ? `${r.latency_ms < 10 ? r.latency_ms.toFixed(1) : r.latency_ms.toFixed(0)} ms`
                    : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function fmt(v: number | null | undefined): string {
  return v === null || v === undefined ? "—" : v.toFixed(3);
}

function LatencyChart({
  stages,
}: {
  stages: NonNullable<EvalSummary["latency_budget"]>;
}) {
  const max = Math.max(...stages.map((s) => s.ms_per_query));
  return (
    <ul className="space-y-2">
      {stages.map((s) => {
        const pct = Math.max(2, (s.ms_per_query / max) * 100);
        return (
          <li key={s.stage} className="flex items-center gap-3 text-sm">
            <span className="w-56 truncate font-mono text-xs">{s.stage}</span>
            <div className="bg-muted h-2 flex-1 overflow-hidden rounded">
              <div className="bg-primary h-full" style={{ width: `${pct}%` }} />
            </div>
            <span className="text-muted-foreground w-20 text-right font-mono text-xs tabular-nums">
              {s.ms_per_query < 10
                ? `${s.ms_per_query.toFixed(1)} ms`
                : `${s.ms_per_query.toFixed(0)} ms`}
            </span>
            {s.notes ? (
              <span className="text-muted-foreground hidden text-[10px] sm:block w-72 truncate">
                {s.notes}
              </span>
            ) : null}
          </li>
        );
      })}
    </ul>
  );
}
