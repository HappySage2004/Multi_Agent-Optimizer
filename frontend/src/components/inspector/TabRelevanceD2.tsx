"use client";

/**
 * D2: Campaign-Screen Relevance Scorer (UI.md §2 Panel 3).
 *
 * Ranked candidate list with percentage-fit pills, backed by real
 * `screen_candidates` rows. Each row shows the four sub-scores that compose the
 * relevance score and the reasons the Data Agent recorded — SOLUTION.md §31 requires a
 * traceable reason per recommendation, so the reasons are shown verbatim rather than
 * summarized.
 */

import { useState } from "react";

import {
  AwaitingStage,
  InspectorCard,
  InspectorSection,
  ScoreBar,
  StubNotice,
} from "@/components/inspector/InspectorShell";
import { CheckIcon, ChevronRightIcon, WarningIcon } from "@/components/ui/Icon";
import { formatNumber, titleCase } from "@/lib/format";
import type { Allocation, ArtifactReference, ScreenCandidate } from "@/lib/types";

export function TabRelevanceD2({
  candidates,
  candidatesRef,
  allocations,
  loading,
}: {
  candidates: ScreenCandidate[];
  candidatesRef: ArtifactReference | undefined;
  allocations: Allocation[];
  loading: boolean;
}) {
  if (!candidatesRef) {
    return (
      <AwaitingStage
        stage="the Data Agent (stage 2)"
        detail="Relevance scoring produces the screen_candidates artifact this panel reads."
      />
    );
  }

  const packagedScreenIds = new Set(allocations.map((a) => a.screen_id));

  return (
    <>
      <InspectorCard
        title="Relevance Matrix"
        badge={`${formatNumber(candidatesRef.rows)} ranked`}
        badgeTone="dark"
        description="Weighted affinity score per screen: audience match, geography, transit and context."
      >
        {candidatesRef.provenance === "stub" ? (
          <StubNotice stage="Relevance scoring (Data Agent)" />
        ) : null}
      </InspectorCard>

      {loading && candidates.length === 0 ? (
        <InspectorSection title="Candidates">
          <p className="text-[10px] text-zinc-400">Loading candidate rows…</p>
        </InspectorSection>
      ) : null}

      {candidates.length > 0 ? (
        <div className="space-y-2">
          <div className="flex items-baseline justify-between px-1 text-[10px]">
            <span className="font-bold tracking-wider text-zinc-400 uppercase">
              Top {candidates.length} of {formatNumber(candidatesRef.rows)}
            </span>
            <span className="font-medium text-zinc-400">
              {packagedScreenIds.size} in package
            </span>
          </div>

          {candidates.map((candidate, index) => (
            <CandidateRow
              key={candidate.screen_id}
              candidate={candidate}
              rank={index + 1}
              inPackage={packagedScreenIds.has(candidate.screen_id)}
            />
          ))}
        </div>
      ) : null}
    </>
  );
}

function CandidateRow({
  candidate,
  rank,
  inPackage,
}: {
  candidate: ScreenCandidate;
  rank: number;
  inPackage: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const fit = `${(candidate.relevance_score * 100).toFixed(1)}% Fit`;

  return (
    <div
      className={`rounded-xl border bg-white shadow-xs ${
        inPackage ? "border-violet-950/20" : "border-zinc-200/50"
      }`}
    >
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        className="flex w-full items-center justify-between gap-2 p-3.5 text-left"
        aria-expanded={expanded}
      >
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="font-mono text-[9px] text-zinc-300">#{rank}</span>
            <span className="truncate font-bold text-zinc-700">{candidate.screen_id}</span>
            {inPackage ? (
              <span className="flex shrink-0 items-center gap-0.5 rounded bg-emerald-50 px-1.5 py-px text-[9px] font-semibold text-emerald-700">
                <CheckIcon className="h-2 w-2" strokeWidth={3} />
                Bought
              </span>
            ) : null}
          </div>
          <div className="mt-0.5 truncate text-[10px] text-zinc-400">
            {[
              candidate.screen_type ? titleCase(candidate.screen_type) : null,
              candidate.zone_id,
              candidate.city_id,
            ]
              .filter(Boolean)
              .join(" • ")}
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-1.5">
          {!candidate.hard_constraints_passed ? (
            <WarningIcon className="h-3.5 w-3.5 text-amber-500" />
          ) : null}
          <span
            className={`rounded-md px-2.5 py-1 text-xs font-black whitespace-nowrap ${
              inPackage
                ? "bg-violet-950 text-white"
                : "border border-zinc-200/50 bg-zinc-100/80 font-bold text-zinc-600"
            }`}
          >
            {fit}
          </span>
          <ChevronRightIcon
            className={`h-3 w-3 text-zinc-300 transition-transform ${expanded ? "rotate-90" : ""}`}
          />
        </div>
      </button>

      {expanded ? (
        <div className="space-y-2.5 border-t border-zinc-100 px-3.5 py-3">
          <div className="space-y-1.5">
            <ScoreBar label="Audience" value={candidate.audience_match_score} />
            <ScoreBar label="Geography" value={candidate.geography_score} />
            <ScoreBar label="Transit" value={candidate.transit_score} />
            <ScoreBar label="Context" value={candidate.contextual_score} />
          </div>

          {candidate.reasons.length > 0 ? (
            <div className="space-y-1 border-t border-zinc-100 pt-2">
              <span className="text-[9px] font-bold tracking-wider text-zinc-400 uppercase">
                Reasons
              </span>
              <ul className="list-disc space-y-0.5 pl-4 text-[10px] leading-relaxed text-zinc-500">
                {candidate.reasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            </div>
          ) : null}

          {!candidate.hard_constraints_passed ? (
            <p className="rounded border border-amber-200/70 bg-amber-50 px-2 py-1 text-[10px] font-medium text-amber-800">
              Failed hard-constraint filtering — not eligible for the package.
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
