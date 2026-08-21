"use client";

/**
 * D1: Audience Profiling Engine (UI.md §2 Panel 3).
 *
 * The mockup's "proximity clusters" and 24-hour footfall sparkline come from POI and
 * ridership features the Data Agent owns. What the API exposes today is the resolved
 * spec, the candidate-pool aggregates, and per-time-block demand from the economics
 * rows — so the chart plots forecast impression share by real `dim_slot` block, and the
 * cluster tags reflect the inventory mix actually in the candidate pool.
 */

import {
  AwaitingStage,
  InspectorCard,
  InspectorSection,
  StubNotice,
} from "@/components/inspector/InspectorShell";
import { BuildingIcon, InterchangeIcon, RetailIcon } from "@/components/ui/Icon";
import { audienceLabel, geographyLabel, type TimeBlockRollup } from "@/lib/derive";
import { formatCompact, formatCurrency, formatDate, formatNumber, titleCase } from "@/lib/format";
import type {
  ArtifactReference,
  CampaignSpec,
  ScreenCandidate,
  ScreenCandidatesSummary,
} from "@/lib/types";

export function TabAudienceD1({
  spec,
  candidates,
  candidatesRef,
  rollups,
}: {
  spec: CampaignSpec | null;
  candidates: ScreenCandidate[];
  candidatesRef: ArtifactReference | undefined;
  rollups: TimeBlockRollup[];
}) {
  if (!spec) {
    return (
      <AwaitingStage
        stage="brief intake"
        detail="The campaign spec is produced by stage 1 before any profiling runs."
      />
    );
  }

  const summary = (candidatesRef?.summary ?? {}) as ScreenCandidatesSummary;

  return (
    <>
      <InspectorCard
        title="Audience Profiling"
        badge={candidatesRef ? "Active" : "Pending"}
        badgeTone={candidatesRef ? "active" : "neutral"}
        description="Inferred from the resolved brief, the eligible inventory, and per-time-block demand forecasts."
      />

      <InspectorSection title="Resolved Brief" meta={spec.optimization_goal.toUpperCase()}>
        <dl className="space-y-1.5 text-[11px]">
          <Row label="Objective" value={spec.campaign_objective} />
          {spec.industry_vertical ? (
            <Row label="Vertical" value={spec.industry_vertical} />
          ) : null}
          <Row label="Audience" value={audienceLabel(spec)} />
          <Row label="Geography" value={geographyLabel(spec)} />
          <Row
            label="Flight"
            value={`${formatDate(spec.start_date)} • ${spec.duration_days} days`}
          />
          <Row label="Budget" value={formatCurrency(spec.budget)} />
        </dl>

        {spec.missing_information.length > 0 ? (
          <div className="rounded-lg border border-amber-200/70 bg-amber-50 px-3 py-2">
            <span className="text-[10px] font-bold tracking-wider text-amber-800 uppercase">
              Not specified in the brief
            </span>
            <ul className="mt-1 list-disc space-y-0.5 pl-4 text-[10px] text-amber-700">
              {spec.missing_information.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        ) : null}
      </InspectorSection>

      {candidatesRef ? (
        <InspectorSection
          title="Candidate Pool"
          meta={`${formatNumber(candidatesRef.rows)} screens`}
        >
          {candidatesRef.provenance === "stub" ? (
            <StubNotice stage="Relevance scoring (Data Agent)" />
          ) : null}
          <div className="grid grid-cols-2 gap-2 text-[11px]">
            <Stat
              label="Eligible in geography"
              value={
                summary.eligible_screens !== undefined
                  ? formatNumber(summary.eligible_screens)
                  : "—"
              }
            />
            <Stat
              label="Shortlisted"
              value={summary.candidates !== undefined ? formatNumber(summary.candidates) : "—"}
            />
            <Stat
              label="Mean relevance"
              value={
                summary.relevance_mean !== undefined ? summary.relevance_mean.toFixed(3) : "—"
              }
            />
            <Stat
              label="Relevance range"
              value={
                summary.relevance_min !== undefined && summary.relevance_max !== undefined
                  ? `${summary.relevance_min.toFixed(2)}-${summary.relevance_max.toFixed(2)}`
                  : "—"
              }
            />
          </div>
        </InspectorSection>
      ) : null}

      <InventoryMix candidates={candidates} />

      <FootfallChart rollups={rollups} />
    </>
  );
}

/**
 * The inventory mix standing in for the mockup's proximity clusters: real `screen_type`
 * counts from the candidate pool rather than POI categories the API does not expose yet.
 */
function InventoryMix({ candidates }: { candidates: ScreenCandidate[] }) {
  if (candidates.length === 0) {
    return (
      <InspectorSection title="Inventory Mix">
        <p className="text-[10px] leading-relaxed text-zinc-400">
          Populated from the <code className="font-mono">screen_candidates</code> artifact once
          the Data Agent has run.
        </p>
      </InspectorSection>
    );
  }

  const counts = new Map<string, number>();
  for (const candidate of candidates) {
    const key = candidate.screen_type ?? "unknown";
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  const icons = [BuildingIcon, InterchangeIcon, RetailIcon];

  const zones = new Set(candidates.map((c) => c.zone_id).filter(Boolean));

  return (
    <InspectorSection
      title="Inventory Mix"
      meta={`Top ${candidates.length} of pool • ${zones.size} ${zones.size === 1 ? "zone" : "zones"}`}
    >
      <div className="flex flex-wrap gap-1.5">
        {ranked.map(([type, count], index) => {
          const Icon = icons[index % icons.length];
          const primary = index === 0;
          return (
            <span
              key={type}
              className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[10px] font-medium ${
                primary
                  ? "bg-zinc-700 text-zinc-100"
                  : "border border-zinc-200/50 bg-zinc-100/70 text-zinc-600"
              }`}
            >
              <Icon className={`h-3 w-3 ${primary ? "text-violet-300" : "text-zinc-400"}`} />
              {titleCase(type)} ({count})
            </span>
          );
        })}
      </div>
    </InspectorSection>
  );
}

/**
 * Forecast impression share per `dim_slot` block. The commuter peaks (blocks 2 and 5)
 * are highlighted in violet, matching the mockup's peak emphasis.
 */
function FootfallChart({ rollups }: { rollups: TimeBlockRollup[] }) {
  const hasData = rollups.some((r) => r.impressions > 0);
  if (!hasData) {
    return (
      <InspectorSection title="Forecast Impressions by Time Block">
        <p className="text-[10px] leading-relaxed text-zinc-400">
          Populated from the optimizer&rsquo;s allocations once stage 4 has run.
        </p>
      </InspectorSection>
    );
  }

  const peak = Math.max(...rollups.map((r) => r.impressionShare));

  return (
    <InspectorSection title="Forecast Impressions by Time Block" meta="dim_slot">
      <div className="flex h-24 items-end gap-2 border-b border-zinc-100 pt-2 pb-1">
        {rollups.map((rollup) => {
          // Scale to the tallest bar so a single-block package is still readable.
          const height = peak > 0 ? (rollup.impressionShare / peak) * 100 : 0;
          return (
            <div
              key={rollup.id}
              className="group relative flex-1"
              title={`${rollup.label} — ${formatCompact(rollup.impressions)} impressions (${Math.round(rollup.impressionShare * 100)}%)`}
            >
              <div
                className={`w-full rounded-t-sm ${
                  !rollup.selected
                    ? "bg-zinc-200/60"
                    : rollup.isPeak
                      ? "bg-violet-950"
                      : "bg-zinc-300"
                }`}
                style={{ height: `${Math.max(height, rollup.selected ? 6 : 3)}%`, minHeight: 3 }}
              />
            </div>
          );
        })}
      </div>

      <div className="flex gap-2 text-[9px] font-medium tracking-tight text-zinc-400">
        {rollups.map((rollup) => (
          <span key={rollup.id} className="flex-1 text-center">
            {rollup.label.slice(0, 5)}
            {rollup.isPeak ? <span className="block text-violet-950">peak</span> : null}
          </span>
        ))}
      </div>

      <div className="flex items-center gap-3 pt-1 text-[9px] text-zinc-400">
        <LegendSwatch className="bg-violet-950" label="Peak block, bought" />
        <LegendSwatch className="bg-zinc-300" label="Bought" />
        <LegendSwatch className="bg-zinc-200/60" label="Not bought" />
      </div>
    </InspectorSection>
  );
}

function LegendSwatch({ className, label }: { className: string; label: string }) {
  return (
    <span className="flex items-center gap-1">
      <span className={`h-2 w-2 rounded-sm ${className}`} />
      {label}
    </span>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-3">
      <dt className="shrink-0 text-zinc-400">{label}</dt>
      <dd className="text-right font-medium text-zinc-600">{value}</dd>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-zinc-50/80 px-2.5 py-2">
      <span className="block text-[9px] font-medium tracking-wide text-zinc-400 uppercase">
        {label}
      </span>
      <span className="font-semibold text-zinc-700">{value}</span>
    </div>
  );
}
