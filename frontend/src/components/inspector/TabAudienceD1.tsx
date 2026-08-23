"use client";

/**
 * D1: Audience Profiling Engine (UI.md §2 Panel 3).
 *
 * What the rep needs from this tab is "who did we aim at, and what did that leave us to
 * work with" — the resolved brief, how much inventory survived it, and what kind of
 * inventory that is.
 *
 * Two things were removed as reporting on the model rather than on the campaign: the
 * mean/range of the relevance score (a 0-1 number with no scale a client understands, and
 * nothing the rep can act on), and the forecast-impressions-by-time-block chart, which
 * duplicated D4's allocation view a stage too early.
 */

import {
  AwaitingStage,
  InspectorCard,
  InspectorSection,
  Stat,
  StubNotice,
} from "@/components/inspector/InspectorShell";
import { BuildingIcon, InterchangeIcon, RetailIcon } from "@/components/ui/Icon";
import { audienceLabel, geographyLabel } from "@/lib/derive";
import { formatCurrency, formatDate, formatNumber, titleCase } from "@/lib/format";
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
}: {
  spec: CampaignSpec | null;
  candidates: ScreenCandidate[];
  candidatesRef: ArtifactReference | undefined;
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
        description="Who the campaign is aimed at, and the inventory that reaches them."
      />

      <InspectorSection title="Resolved Brief" meta={spec.optimization_goal.toUpperCase()}>
        <dl className="space-y-1 text-[11px]">
          <Row label="Objective" value={spec.campaign_objective} />
          {spec.industry_vertical ? <Row label="Vertical" value={spec.industry_vertical} /> : null}
          <Row label="Audience" value={audienceLabel(spec)} />
          <Row label="Geography" value={geographyLabel(spec)} />
          <Row
            label="Flight"
            value={`${formatDate(spec.start_date)} • ${spec.duration_days} days`}
          />
          <Row label="Budget" value={formatCurrency(spec.budget)} />
        </dl>

        {spec.missing_information.length > 0 ? (
          <div className="rounded-lg border border-amber-200/70 bg-amber-50 px-2.5 py-1.5">
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
            <StubNotice stage="Relevance scoring (relevance engine)" />
          ) : null}
          <div className="grid grid-cols-2 gap-2">
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
          </div>
          <p className="text-[10px] leading-relaxed text-zinc-400">
            These screens were scored and shortlisted against the audience, location, timing
            and context set out in the campaign brief. The strongest of them go through to
            pricing.
          </p>
        </InspectorSection>
      ) : null}

      <InventoryMix candidates={candidates} />
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
          Populated once the relevance engine has scored the eligible inventory.
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

  // Counted on the name so this figure and the labels in D2 agree. Vehicle-mounted
  // screens have no zone at all, so they drop out rather than counting as one.
  const zones = new Set(candidates.map((c) => c.zone_name ?? c.zone_id).filter(Boolean));

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
              className={`flex items-center gap-1.5 rounded-md px-2 py-1 text-[10px] font-medium ${
                primary
                  ? "border border-zinc-800 bg-zinc-50 text-zinc-800"
                  : "border border-zinc-200/50 bg-zinc-100/70 text-zinc-600"
              }`}
            >
              <Icon className={`h-3 w-3 ${primary ? "text-violet-950" : "text-zinc-400"}`} />
              {titleCase(type)} ({count})
            </span>
          );
        })}
      </div>
    </InspectorSection>
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
