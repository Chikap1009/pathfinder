import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft, MapPin, Briefcase } from "lucide-react";
import { BASE, type JobDetail, type ProfileDetail } from "@/lib/api";

type Props = { params: Promise<{ id: string }> };

export const metadata = { title: "Profile · PathFinder" };
export const dynamic = "force-dynamic";

async function fetchEntity(
  id: string,
): Promise<{ kind: "person"; data: ProfileDetail } | { kind: "job"; data: JobDetail } | null> {
  // The id namespace tells us which endpoint to hit.
  if (id.startsWith("person_")) {
    const res = await fetch(`${BASE}/v1/profile/${encodeURIComponent(id)}`, {
      cache: "no-store",
    });
    if (res.status === 404) return null;
    if (!res.ok) throw new Error(`profile fetch ${res.status}`);
    return { kind: "person", data: (await res.json()) as ProfileDetail };
  }
  if (id.startsWith("job_")) {
    const res = await fetch(`${BASE}/v1/job/${encodeURIComponent(id)}`, { cache: "no-store" });
    if (res.status === 404) return null;
    if (!res.ok) throw new Error(`job fetch ${res.status}`);
    return { kind: "job", data: (await res.json()) as JobDetail };
  }
  return null;
}

export default async function ResultPage({ params }: Props) {
  const { id } = await params;
  let entity: Awaited<ReturnType<typeof fetchEntity>> = null;
  try {
    entity = await fetchEntity(id);
  } catch (err) {
    return (
      <section className="mx-auto flex w-full max-w-4xl flex-col gap-6 px-4 py-12">
        <div className="border-destructive/30 bg-destructive/5 text-destructive rounded border p-4">
          Failed to fetch <code>{id}</code>: {(err as Error).message}
        </div>
      </section>
    );
  }
  if (!entity) notFound();

  return (
    <section className="mx-auto flex w-full max-w-4xl flex-col gap-6 px-4 py-12">
      <Link
        href="/search"
        className="text-muted-foreground hover:text-foreground inline-flex w-fit items-center gap-1.5 text-sm"
      >
        <ArrowLeft className="h-4 w-4" /> back to search
      </Link>

      {entity.kind === "person" ? <ProfileView profile={entity.data} /> : <JobView job={entity.data} />}
    </section>
  );
}

function ProfileView({ profile }: { profile: ProfileDetail }) {
  const skills = profile.skills ?? [];
  const potentialRoles = profile.potential_roles ?? [];
  const grouped = {
    core: skills.filter((s) => s.category === "core"),
    secondary: skills.filter((s) => s.category === "secondary"),
    soft: skills.filter((s) => s.category === "soft"),
  };
  return (
    <article className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <p className="text-muted-foreground text-xs font-mono">{profile.id}</p>
        <h1 className="font-mono text-3xl font-semibold tracking-tight">
          {profile.name?.trim() || `Person ${profile.id}`}
        </h1>
        <p className="text-muted-foreground inline-flex items-center gap-1 text-sm">
          <Briefcase className="h-3.5 w-3.5" />
          {profile.years_experience ?? 0} years experience
        </p>
      </header>

      {profile.skill_summary ? (
        <section className="border-border bg-card rounded-lg border p-5">
          <h2 className="text-muted-foreground mb-2 text-xs font-medium uppercase tracking-wide">
            Skill summary
          </h2>
          <p className="leading-relaxed">{profile.skill_summary}</p>
        </section>
      ) : null}

      {potentialRoles.length > 0 ? (
        <section>
          <h2 className="text-muted-foreground mb-2 text-xs font-medium uppercase tracking-wide">
            Potential roles
          </h2>
          <div className="flex flex-wrap gap-1.5">
            {potentialRoles.map((r) => (
              <span
                key={r}
                className="border-border bg-muted/40 rounded-full border px-2.5 py-1 text-xs"
              >
                {r}
              </span>
            ))}
          </div>
        </section>
      ) : null}

      {(["core", "secondary", "soft"] as const).map((cat) =>
        grouped[cat].length > 0 ? (
          <section key={cat}>
            <h2 className="text-muted-foreground mb-2 text-xs font-medium uppercase tracking-wide">
              {cat} skills · {grouped[cat].length}
            </h2>
            <ul className="grid grid-cols-1 gap-1.5 sm:grid-cols-2 md:grid-cols-3">
              {grouped[cat].map((s) => (
                <li
                  key={s.skill_id}
                  className="border-border flex items-center justify-between gap-2 rounded border px-2 py-1.5 text-sm"
                >
                  <span className="truncate">{s.skill_name}</span>
                  <span className="text-muted-foreground shrink-0 font-mono text-[10px]">
                    {s.proficiency_label}
                  </span>
                </li>
              ))}
            </ul>
          </section>
        ) : null,
      )}
    </article>
  );
}

function JobView({ job }: { job: JobDetail }) {
  const jobSkills = job.skills ?? [];
  const must = jobSkills.filter((s) => s.priority === "must_have");
  const good = jobSkills.filter((s) => s.priority === "good_to_have");
  return (
    <article className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <p className="text-muted-foreground text-xs font-mono">{job.id}</p>
        <h1 className="font-mono text-3xl font-semibold tracking-tight">
          {job.designation ?? job.title ?? `Job ${job.id}`}
        </h1>
        <p className="text-muted-foreground inline-flex items-center gap-1 text-sm">
          <MapPin className="h-3.5 w-3.5" />
          {[job.city, job.country].filter(Boolean).join(", ") || "—"} · {job.industry || "—"}
        </p>
      </header>

      {must.length > 0 ? (
        <section>
          <h2 className="text-muted-foreground mb-2 text-xs font-medium uppercase tracking-wide">
            Must have · {must.length}
          </h2>
          <ul className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
            {must.map((s) => (
              <li
                key={s.skill_id}
                className="border-border rounded border px-2 py-1.5 text-sm"
              >
                {s.skill_name}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {good.length > 0 ? (
        <section>
          <h2 className="text-muted-foreground mb-2 text-xs font-medium uppercase tracking-wide">
            Good to have · {good.length}
          </h2>
          <ul className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
            {good.map((s) => (
              <li
                key={s.skill_id}
                className="border-border rounded border px-2 py-1.5 text-sm"
              >
                {s.skill_name}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {job.enhanced_text ? (
        <section className="border-border bg-card rounded-lg border p-5">
          <h2 className="text-muted-foreground mb-2 text-xs font-medium uppercase tracking-wide">
            LLM-enhanced JD
          </h2>
          <pre className="text-muted-foreground max-h-96 overflow-auto whitespace-pre-wrap text-xs leading-relaxed">
            {job.enhanced_text}
          </pre>
        </section>
      ) : null}
    </article>
  );
}
