"use client";

/**
 * D2: why each recommended screen is in the package (UI.md §2 Panel 3).
 *
 * This used to list the top 60 of a 250-screen ranked pool. Fifty of those rows were
 * screens the optimizer then declined to buy, which is a debugging view of the relevance
 * engine rather than something a rep can take to a client — they get asked "why this
 * screen", never "why is screen #47 ranked above screen #48". So the list is now exactly
 * the screens the optimizer recommended, and the tab is titled for that.
 *
 * The per-screen reasons stay verbatim: SOLUTION.md §31 requires a traceable reason per
 * recommendation, and they are the answer to the only question this tab exists for.
 */

import { useState } from "react";

import {
  AwaitingStage,
  InspectorCard,
  InspectorSection,
  ScoreBar,
  StubNotice,
} from "@/components/inspector/InspectorShell";
import { ChevronRightIcon, WarningIcon } from "@/components/ui/Icon";
import { placeLabel } from "@/lib/derive";
import { titleCase } from "@/lib/format";
import type { Allocation, ArtifactReference, ScreenCandidate } from "@/lib/types";

export function TabRelevanceD2({
  recommended,
  candidatesRef,
  allocations,
  loading,
}: {
  /** Candidate rows for the screens the optimizer bought, in rank order. */
  recommended: ScreenCandidate[];
  candidatesRef: ArtifactReference | undefined;
  allocations: Allocation[];
  loading: boolean;
}) {
  if (!candidatesRef) {
    return (
      <AwaitingStage
        stage="the relevance engine (stage 2)"
        detail="Screens are scored against the brief before any of them can be recommended."
      />
    );
  }

  if (allocations.length === 0) {
    return (
      <AwaitingStage
        stage="the optimizer (stage 4)"
        detail="Screens have been scored, but none is recommended until the optimizer has built a package."
      />
    );
  }

  return (
    <>
      <InspectorCard
        title="Recommended Screens"
        badge={`${recommended.length} selected`}
        badgeTone="dark"
        description="Every screen in this package, and how well each one matches the brief on audience, location, timing and context."
      >
        {candidatesRef.provenance === "stub" ? (
          <StubNotice stage="Relevance scoring (relevance engine)" />
        ) : null}
      </InspectorCard>

      {loading && recommended.length === 0 ? (
        <InspectorSection title="Recommended Screens">
          <p className="text-[10px] text-zinc-400">Loading screen details…</p>
        </InspectorSection>
      ) : null}

      {recommended.length > 0 ? (
        <div className="space-y-1.5">
          <div className="px-1 text-[10px] font-bold tracking-wider text-zinc-400 uppercase">
            Best match first — tap a screen for its reasoning
          </div>

          {recommended.map((candidate, index) => (
            <CandidateRow key={candidate.screen_id} candidate={candidate} rank={index + 1} />
          ))}
        </div>
      ) : null}
    </>
  );
}

function CandidateRow({ candidate, rank }: { candidate: ScreenCandidate; rank: number }) {
  const [expanded, setExpanded] = useState(false);
  const fit = `${Math.round(candidate.relevance_score * 100)}% Fit`;

  return (
    <div className="rounded-xl border border-violet-950/15 bg-white shadow-xs">
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        className="flex w-full items-center justify-between gap-2 p-2.5 text-left"
        aria-expanded={expanded}
      >
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="font-mono text-[10px] text-zinc-300">#{rank}</span>
            {/* The place first — a rep can picture "East Commons Station"; a screen id is
                a lookup key they read out only when booking. */}
            <span className="truncate text-[11px] font-bold text-zinc-700">
              {placeLabel(candidate)}
            </span>
          </div>
          <div className="mt-0.5 truncate text-[10px] text-zinc-400">
            {[
              candidate.screen_type ? titleCase(candidate.screen_type) : null,
              candidate.zone_name ?? candidate.zone_id,
              candidate.screen_id,
            ]
              .filter(Boolean)
              .join(" • ")}
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-1.5">
          {!candidate.hard_constraints_passed ? (
            <WarningIcon className="h-3 w-3 text-amber-500" />
          ) : null}
          <span className="rounded-md bg-violet-950 px-2 py-0.5 text-[11px] font-bold whitespace-nowrap text-white">
            {fit}
          </span>
          <ChevronRightIcon
            className={`h-3 w-3 text-zinc-300 transition-transform ${expanded ? "rotate-90" : ""}`}
          />
        </div>
      </button>

      {expanded ? (
        <div className="space-y-2 border-t border-zinc-100 px-2.5 py-2">
          {candidate.reasons.length > 0 ? (
            <ul className="list-disc space-y-0.5 pl-4 text-[10px] leading-relaxed text-zinc-500">
              {candidate.reasons.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          ) : null}

          <div className="space-y-1 border-t border-zinc-100 pt-2">
            <span className="text-[10px] font-bold tracking-wider text-zinc-400 uppercase">
              Fit breakdown
            </span>
            <ScoreBar label="Audience" value={candidate.audience_match_score} />
            <ScoreBar label="Location" value={candidate.geography_score} />
            <ScoreBar label="Surroundings" value={candidate.contextual_score} />
            <ScoreBar label="Time of day" value={candidate.time_of_day_score} />
            <ScoreBar label="Track record" value={candidate.historical_performance_score} />
          </div>

          {/* Volume percentile, kept visually apart: it is reported, not weighted into
              the relevance score. Saying so is the point — a rep who reads it as part of
              the fit will quote the wrong reason. */}
          <div className="space-y-1 border-t border-zinc-100 pt-2">
            <ScoreBar label="Audience size" value={candidate.transit_score} />
            <p className="text-[10px] leading-relaxed text-zinc-400">
              How busy this screen is next to the others considered. Shown for context — it
              does not feed the fit score above.
            </p>
          </div>

          {!candidate.hard_constraints_passed ? (
            <p className="rounded border border-amber-200/70 bg-amber-50 px-2 py-1 text-[10px] font-medium text-amber-800">
              This screen did not clear the brief&rsquo;s hard requirements.
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
