"use client";

import { useEffect, useState } from "react";
import { Drawer } from "vaul";
import { X, ExternalLink, Sparkles, MapPin, Briefcase, Zap, Clock } from "lucide-react";
import {
  getJob,
  getProfile,
  type JobDetail,
  type ProfileDetail,
  type SearchResult,
} from "@/lib/api";
import { cn } from "@/lib/utils";

type Props = {
  open: boolean;
  onClose: () => void;
  result: SearchResult | null;
};

export function ResultDrawer({ open, onClose, result }: Props) {
  const [profile, setProfile] = useState<ProfileDetail | null>(null);
  const [job, setJob] = useState<JobDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open || !result) return;
    setLoading(true);
    setError(null);
    setProfile(null);
    setJob(null);
    const id = result.entity.id;
    const fetcher = result.entity.kind === "person" ? getProfile(id) : getJob(id);
    fetcher
      .then((d) => {
        if (result.entity.kind === "person") setProfile(d as ProfileDetail);
        else setJob(d as JobDetail);
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, [open, result]);

  return (
    <Drawer.Root open={open} onOpenChange={(o) => !o && onClose()} direction="right">
      <Drawer.Portal>
        <Drawer.Overlay className="fixed inset-0 z-40 bg-black/40 backdrop-blur-sm" />
        <Drawer.Content className="border-border bg-background fixed top-0 right-0 z-50 flex h-full w-full max-w-2xl flex-col border-l outline-none">
          <Drawer.Title className="sr-only">
            {result?.entity.kind === "person"
              ? `Profile ${result.entity.name ?? result.entity.id}`
              : `Job ${result?.entity.designation ?? result?.entity.id ?? ""}`}
          </Drawer.Title>

          <header className="border-border bg-background/80 sticky top-0 flex items-start justify-between gap-3 border-b p-5 backdrop-blur">
            <div className="min-w-0 flex-1">
              {result ? <DrawerHeader result={result} /> : null}
            </div>
            <button
              type="button"
              onClick={onClose}
              className="hover:bg-muted text-muted-foreground rounded-md p-2 transition-colors"
              aria-label="Close drawer"
            >
              <X className="h-4 w-4" />
            </button>
          </header>

          <div className="flex-1 overflow-y-auto p-5">
            {result ? <StageScoreBar result={result} /> : null}

            {loading ? (
              <div className="text-muted-foreground py-12 text-center text-sm font-mono">
                loading detail…
              </div>
            ) : error ? (
              <div className="text-destructive border-destructive/30 bg-destructive/5 my-4 rounded border p-3 text-sm">
                {error}
              </div>
            ) : profile ? (
              <ProfileBody profile={profile} matched={result?.entity.matched_skills ?? []} />
            ) : job ? (
              <JobBody job={job} matched={result?.entity.matched_skills ?? []} />
            ) : null}
          </div>
        </Drawer.Content>
      </Drawer.Portal>
    </Drawer.Root>
  );
}

// ─── Header (rank + title + score) ───────────────────────────────────────────

function DrawerHeader({ result }: { result: SearchResult }) {
  const e = result.entity;
  const title =
    e.kind === "person"
      ? e.name?.trim() || `Person ${e.id}`
      : (e.designation ?? e.title ?? `Job ${e.id}`);
  const sub =
    e.kind === "person"
      ? `${e.years_experience ?? 0} years experience · ${e.id}`
      : [e.industry, e.city, e.country].filter(Boolean).join(" · ");
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2">
        <span className="bg-primary/10 text-primary flex h-7 w-7 items-center justify-center rounded font-mono text-xs">
          #{result.rank}
        </span>
        <span className="border-border text-muted-foreground rounded border px-2 py-0.5 font-mono text-[10px]">
          {e.kind}
        </span>
        <span className="text-muted-foreground font-mono text-xs">
          score {result.score.toFixed(4)}
        </span>
      </div>
      <h2 className="text-lg font-semibold leading-tight">{title}</h2>
      {sub ? <p className="text-muted-foreground text-xs">{sub}</p> : null}
    </div>
  );
}

// ─── Per-stage score bars (the explanation centerpiece) ─────────────────────

function StageScoreBar({ result }: { result: SearchResult }) {
  const stages: { name: string; key: keyof typeof result.stage_scores; color: string }[] = [
    { name: "BM25", key: "bm25", color: "bg-chart-1" },
    { name: "Dense", key: "dense", color: "bg-chart-2" },
    { name: "KG", key: "kg", color: "bg-chart-3" },
    { name: "RRF", key: "rrf", color: "bg-chart-4" },
    { name: "Rerank", key: "rerank", color: "bg-chart-5" },
  ];
  // Visual scaling: each stage's bar width = score / (max-stage score) * 100%.
  // Use a per-stage max so different scales (BM25 ~10s, dense ~1.0, RRF ~0.03) render.
  const present = stages
    .map((s) => ({ ...s, value: (result.stage_scores[s.key] as number | null | undefined) ?? null }))
    .filter((s) => s.value !== null && s.value !== undefined);

  if (present.length === 0) return null;

  // Scale each present stage to its own [0,1] using a heuristic max. We expose the raw
  // value as text so the user can still see the absolute number.
  const heuristicMax: Record<string, number> = {
    BM25: 30,
    Dense: 1.0,
    KG: 50,
    RRF: 0.05,
    Rerank: 1.0,
  };

  return (
    <section className="border-border mb-5 rounded-lg border p-4">
      <header className="mb-3 flex items-center gap-2">
        <Sparkles className="text-primary h-4 w-4" />
        <h3 className="text-sm font-medium">Per-stage scores</h3>
      </header>
      <ul className="space-y-2">
        {present.map((s) => {
          const max = heuristicMax[s.name] ?? 1;
          const width = Math.min(100, Math.max(2, ((s.value ?? 0) / max) * 100));
          return (
            <li key={s.name} className="flex items-center gap-3">
              <span className="text-muted-foreground w-16 text-xs font-mono">{s.name}</span>
              <div className="bg-muted h-2 flex-1 overflow-hidden rounded">
                <div
                  className={cn("h-full transition-all", s.color)}
                  style={{ width: `${width}%` }}
                />
              </div>
              <span className="text-muted-foreground w-20 text-right font-mono text-xs tabular-nums">
                {s.value !== null && s.value !== undefined ? s.value.toFixed(3) : "—"}
              </span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

// ─── Profile body ────────────────────────────────────────────────────────────

function ProfileBody({ profile, matched }: { profile: ProfileDetail; matched: string[] }) {
  const matchedSet = new Set(matched.map((s) => s.toLowerCase()));
  const skills = profile.skills ?? [];
  const potentialRoles = profile.potential_roles ?? [];
  const grouped = {
    core: skills.filter((s) => s.category === "core"),
    secondary: skills.filter((s) => s.category === "secondary"),
    soft: skills.filter((s) => s.category === "soft"),
  };
  return (
    <div className="space-y-5">
      {profile.skill_summary ? (
        <Section icon={<ExternalLink className="h-4 w-4" />} title="Skill summary">
          <p className="text-sm leading-relaxed">{profile.skill_summary}</p>
        </Section>
      ) : null}

      {potentialRoles.length > 0 ? (
        <Section icon={<Briefcase className="h-4 w-4" />} title="Potential roles">
          <div className="flex flex-wrap gap-1.5">
            {potentialRoles.map((r) => (
              <span
                key={r}
                className="border-border bg-muted/50 rounded border px-2 py-0.5 text-xs"
              >
                {r}
              </span>
            ))}
          </div>
        </Section>
      ) : null}

      {(["core", "secondary", "soft"] as const).map((cat) =>
        grouped[cat].length > 0 ? (
          <Section
            key={cat}
            icon={<Zap className="h-4 w-4" />}
            title={`${cat} skills · ${grouped[cat].length}`}
          >
            <ul className="grid grid-cols-1 gap-1 text-sm sm:grid-cols-2">
              {grouped[cat].map((s) => {
                const isMatch = matchedSet.has(s.skill_name.toLowerCase());
                return (
                  <li
                    key={s.skill_id}
                    className={cn(
                      "border-border flex items-center justify-between gap-2 rounded border px-2 py-1.5",
                      isMatch && "border-primary/50 bg-primary/5",
                    )}
                  >
                    <span className="truncate">{s.skill_name}</span>
                    <span className="text-muted-foreground shrink-0 font-mono text-[10px]">
                      {s.proficiency_label}
                    </span>
                  </li>
                );
              })}
            </ul>
          </Section>
        ) : null,
      )}
    </div>
  );
}

// ─── Job body ────────────────────────────────────────────────────────────────

function JobBody({ job, matched }: { job: JobDetail; matched: string[] }) {
  const matchedSet = new Set(matched.map((s) => s.toLowerCase()));
  const jobSkills = job.skills ?? [];
  const must = jobSkills.filter((s) => s.priority === "must_have");
  const good = jobSkills.filter((s) => s.priority === "good_to_have");

  return (
    <div className="space-y-5">
      <Section icon={<MapPin className="h-4 w-4" />} title="Where + when">
        <dl className="grid grid-cols-2 gap-2 text-sm">
          <Field label="Location">{[job.city, job.country].filter(Boolean).join(", ") || "—"}</Field>
          <Field label="Industry">{job.industry || "—"}</Field>
          <Field label="Designation">{job.designation || "—"}</Field>
          <Field label="Source">{job.source}</Field>
          <Field label="Experience">
            {job.experience_lower ?? 0}–{job.experience_upper ?? 0} years
          </Field>
        </dl>
      </Section>

      {must.length > 0 ? (
        <Section icon={<Zap className="h-4 w-4" />} title={`Must-have skills · ${must.length}`}>
          <SkillList skills={must} matched={matchedSet} />
        </Section>
      ) : null}

      {good.length > 0 ? (
        <Section icon={<Sparkles className="h-4 w-4" />} title={`Good to have · ${good.length}`}>
          <SkillList skills={good} matched={matchedSet} />
        </Section>
      ) : null}

      {job.enhanced_text ? (
        <Section icon={<Clock className="h-4 w-4" />} title="LLM-enhanced JD">
          <pre className="text-muted-foreground border-border bg-muted/30 max-h-72 overflow-auto rounded border p-3 text-xs leading-relaxed whitespace-pre-wrap">
            {job.enhanced_text}
          </pre>
        </Section>
      ) : null}
    </div>
  );
}

function SkillList({
  skills,
  matched,
}: {
  skills: NonNullable<JobDetail["skills"]>;
  matched: Set<string>;
}) {
  return (
    <ul className="grid grid-cols-1 gap-1 text-sm">
      {skills.map((s) => {
        const isMatch = matched.has(s.skill_name.toLowerCase());
        return (
          <li
            key={s.skill_id}
            className={cn(
              "border-border rounded border px-2 py-1.5 text-sm",
              isMatch && "border-primary/50 bg-primary/5",
            )}
          >
            {s.skill_name}
          </li>
        );
      })}
    </ul>
  );
}

// ─── Tiny presentation helpers ───────────────────────────────────────────────

function Section({
  icon,
  title,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section>
      <header className="text-muted-foreground mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-wide">
        <span className="text-primary">{icon}</span>
        {title}
      </header>
      {children}
    </section>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5">
      <dt className="text-muted-foreground text-[10px] uppercase tracking-wide">{label}</dt>
      <dd className="text-sm">{children}</dd>
    </div>
  );
}
