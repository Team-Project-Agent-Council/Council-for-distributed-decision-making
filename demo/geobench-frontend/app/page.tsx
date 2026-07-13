import Link from "next/link";
import { ArrowRight, Sparkles, Brain, Eye, GitBranch, Map } from "lucide-react";

export default function Home() {
  return (
    <div className="mx-auto max-w-6xl px-4 sm:px-6 py-12 sm:py-20">
      {/* Hero */}
      <section className="relative overflow-hidden rounded-3xl px-8 py-16 sm:px-14 sm:py-24">
        <div
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(circle at 20% 20%, rgba(198,239,56,0.18) 0%, rgba(17,17,17,0) 55%), linear-gradient(135deg, #111 0%, #1a1a1a 100%)",
          }}
        />
        <div className="relative z-10 max-w-3xl">
          <span
            className="inline-flex items-center gap-1.5 text-xs font-bold tracking-[0.15em] uppercase mb-6 px-3 py-1 rounded-full"
            style={{ background: "rgba(198,239,56,0.15)", color: "#c6ef38" }}
          >
            <Sparkles size={12} />
            Progressive Narrowing
          </span>
          <h1 className="text-4xl sm:text-6xl font-black tracking-tight mb-5 text-white">
            Where was this image taken?
          </h1>
          <p className="text-lg leading-relaxed mb-8 max-w-2xl" style={{ color: "rgba(255,255,255,0.65)" }}>
            GeoBench runs a council of five specialist vision-language agents on a single Google
            Street View image. Each agent looks at a different layer of evidence — text, terrain,
            flora, infrastructure, GeoGuessr meta — then a judge narrows the world down to a
            region, then to a country.
          </p>

          <div className="flex flex-wrap gap-3">
            <Link
              href="/test-council"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full font-bold text-sm transition-transform hover:scale-105"
              style={{ background: "#c6ef38", color: "#111" }}
            >
              <Sparkles size={15} />
              Run the Council
              <ArrowRight size={15} />
            </Link>
            <Link
              href="/council"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full font-semibold text-sm transition-colors"
              style={{
                background: "rgba(255,255,255,0.08)",
                color: "rgba(255,255,255,0.9)",
                border: "1px solid rgba(255,255,255,0.15)",
              }}
            >
              Meet the agents
            </Link>
          </div>
        </div>
      </section>

      {/* Pipeline overview */}
      <section className="mt-16 sm:mt-24">
        <div className="mb-10">
          <span
            className="inline-block text-xs font-bold tracking-[0.15em] uppercase mb-3"
            style={{ color: "#c6ef38" }}
          >
            How it works
          </span>
          <h2 className="text-3xl sm:text-4xl font-black tracking-tight">
            From five perspectives to one location
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <PipelineCard
            number="1"
            icon={<Eye size={20} />}
            title="Independent assessment"
            body="Linguistic, Landscape, Botanics, Regulatory and Meta each propose candidate countries in parallel — no peeking at the others."
          />
          <PipelineCard
            number="2"
            icon={<Brain size={20} />}
            title="Region consensus check"
            body="A judge maps every candidate to a world region. If all agents agree, the council fast-tracks to the country phase."
          />
          <PipelineCard
            number="3"
            icon={<GitBranch size={20} />}
            title="Hypothesis testing"
            body="Otherwise, the judge generates 2–4 region hypotheses. Each specialist scores them in isolation on a 5-level confidence scale."
          />
          <PipelineCard
            number="4"
            icon={<Map size={20} />}
            title="Country determination"
            body="Specialists re-assess within the chosen region. The judge weighs every country hypothesis and outputs the final answer plus coordinates."
          />
        </div>
      </section>

      {/* CTA */}
      <section className="mt-16 sm:mt-24 rounded-3xl px-8 py-12 sm:px-14 sm:py-16 text-center"
        style={{
          background:
            "linear-gradient(135deg, rgba(198,239,56,0.08) 0%, rgba(198,239,56,0.02) 100%)",
          border: "1px solid rgba(198,239,56,0.2)",
        }}
      >
        <h2 className="text-2xl sm:text-3xl font-black tracking-tight mb-3">
          Try it on your own image
        </h2>
        <p className="text-base mb-7 max-w-xl mx-auto" style={{ color: "var(--muted-foreground)" }}>
          Upload any street-level photo or pick a random one from the dataset, then watch the
          council narrow it down phase by phase in real time.
        </p>
        <Link
          href="/test-council"
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full font-bold text-sm transition-transform hover:scale-105"
          style={{ background: "#c6ef38", color: "#111" }}
        >
          <Sparkles size={15} />
          Run the Council
          <ArrowRight size={15} />
        </Link>
      </section>
    </div>
  );
}

function PipelineCard({
  number,
  icon,
  title,
  body,
}: {
  number: string;
  icon: React.ReactNode;
  title: string;
  body: string;
}) {
  return (
    <div
      className="rounded-2xl p-5"
      style={{
        background: "var(--card)",
        border: "1px solid var(--border)",
      }}
    >
      <div className="flex items-center gap-3 mb-3">
        <div
          className="flex h-9 w-9 items-center justify-center rounded-xl font-bold text-sm"
          style={{
            background: "rgba(198,239,56,0.12)",
            color: "#c6ef38",
            border: "1px solid rgba(198,239,56,0.25)",
          }}
        >
          {number}
        </div>
        <div style={{ color: "#c6ef38" }}>{icon}</div>
      </div>
      <h3 className="font-bold text-base mb-2">{title}</h3>
      <p className="text-sm leading-relaxed" style={{ color: "var(--muted-foreground)" }}>
        {body}
      </p>
    </div>
  );
}
