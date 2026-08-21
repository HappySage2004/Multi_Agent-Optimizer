"use client";

/** Shared card chrome for the inspector tabs, so the four tabs stay visually identical. */

export function InspectorCard({
  title,
  badge,
  badgeTone = "neutral",
  description,
  children,
}: {
  title: string;
  badge?: string;
  badgeTone?: "neutral" | "active" | "dark" | "warning";
  description?: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="space-y-3 rounded-xl border border-zinc-200/50 bg-white p-4 shadow-xs">
      <div className="space-y-1.5">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-bold tracking-wider text-zinc-700 uppercase">{title}</span>
          {badge ? <Badge tone={badgeTone}>{badge}</Badge> : null}
        </div>
        {description ? (
          <p className="text-[11px] leading-relaxed text-zinc-400">{description}</p>
        ) : null}
      </div>
      {children}
    </div>
  );
}

export function InspectorSection({
  title,
  meta,
  children,
}: {
  title: string;
  meta?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-3 rounded-xl border border-zinc-200/50 bg-white p-4 shadow-xs">
      <div className="flex items-baseline justify-between gap-2 text-[11px]">
        <span className="font-bold text-zinc-600">{title}</span>
        {meta ? <span className="font-medium text-zinc-400">{meta}</span> : null}
      </div>
      {children}
    </div>
  );
}

export function Badge({
  tone = "neutral",
  children,
}: {
  tone?: "neutral" | "active" | "dark" | "warning";
  children: React.ReactNode;
}) {
  const tones = {
    neutral: "bg-zinc-100/70 text-zinc-600 border border-zinc-200/50",
    active: "bg-emerald-50 text-emerald-700 border border-emerald-200/60",
    dark: "bg-zinc-700 text-white",
    warning: "bg-amber-50 text-amber-700 border border-amber-200/60",
  } as const;
  return (
    <span
      className={`shrink-0 rounded px-2 py-0.5 text-[10px] font-semibold whitespace-nowrap ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

/**
 * What a tab shows before its stage has run. Names the stage that produces the data, so
 * an empty panel is never mistaken for a zero result.
 */
export function AwaitingStage({ stage, detail }: { stage: string; detail?: string }) {
  return (
    <div className="rounded-xl border border-dashed border-zinc-200 bg-white/60 p-6 text-center">
      <p className="text-[11px] font-semibold text-zinc-500">Awaiting {stage}</p>
      <p className="mt-1 text-[10px] leading-relaxed text-zinc-400">
        {detail ?? "Run a campaign brief to populate this panel."}
      </p>
    </div>
  );
}

/** Flags a panel whose numbers came from an unimplemented specialist. */
export function StubNotice({ stage }: { stage: string }) {
  return (
    <p className="rounded-lg border border-amber-200/70 bg-amber-50 px-3 py-2 text-[10px] leading-relaxed text-amber-800">
      {stage} is a stub. These are deterministic placeholders derived from screen IDs, not
      analysis of the data.
    </p>
  );
}

/** Horizontal 0-1 score bar used by the D2 sub-score breakdown. */
export function ScoreBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-20 shrink-0 text-[9px] font-medium tracking-wide text-zinc-400 uppercase">
        {label}
      </span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-zinc-100">
        <div
          className="h-full rounded-full bg-violet-950"
          style={{ width: `${Math.round(Math.min(1, Math.max(0, value)) * 100)}%` }}
        />
      </div>
      <span className="w-8 shrink-0 text-right font-mono text-[9px] text-zinc-500">
        {value.toFixed(2)}
      </span>
    </div>
  );
}
