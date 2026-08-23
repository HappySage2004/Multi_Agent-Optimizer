"use client";

/**
 * Right panel: the D1-D4 tab container (UI.md §2 Panel 3).
 *
 * Tab switching is pure client state, so it is instantaneous — all four views read from
 * run data already in memory and none of them fetches on activation.
 */

import { TabAudienceD1 } from "@/components/inspector/TabAudienceD1";
import { TabOptimizerD4 } from "@/components/inspector/TabOptimizerD4";
import { TabPricingD3 } from "@/components/inspector/TabPricingD3";
import { TabRelevanceD2 } from "@/components/inspector/TabRelevanceD2";
import { WarningIcon } from "@/components/ui/Icon";
import type { RunData } from "@/hooks/useCampaignRun";
import { priceGuardrail, rotationRows, timeBlockRollups } from "@/lib/derive";

export const INSPECTOR_TABS = [
  { id: "d1", label: "D1: Audience" },
  { id: "d2", label: "D2: Relevance" },
  { id: "d3", label: "D3: Pricing" },
  { id: "d4", label: "D4: Optimizer" },
] as const;

export type InspectorTabId = (typeof INSPECTOR_TABS)[number]["id"];

/**
 * An artifact the run claims to have but which would not load.
 *
 * This was silently swallowed: the fetch failure became an empty row array while the
 * artifact reference still reported its row count, so a tab said "250 screens" and drew
 * none of them. A stage that simply has not run is a different thing and stays the tabs'
 * own empty state — this only fires when a recorded artifact is unreadable.
 */
function ArtifactErrors({ errors }: { errors: RunData["artifactErrors"] }) {
  const entries = Object.entries(errors);
  if (entries.length === 0) return null;

  return (
    <div className="space-y-1.5 rounded-lg border border-amber-200/70 bg-amber-50 p-3">
      <div className="flex items-center gap-1.5 text-[13px] font-bold text-amber-800">
        <WarningIcon className="h-3.5 w-3.5 shrink-0" />
        Stored data could not be read
      </div>
      {entries.map(([kind, detail]) => (
        <p key={kind} className="text-[12px] leading-relaxed text-amber-800">
          <span className="font-mono font-semibold">{kind}</span>: {detail}
        </p>
      ))}
      <p className="text-[12px] leading-relaxed text-amber-700">
        The figures below are therefore incomplete — they are not zeroes.
      </p>
    </div>
  );
}

export function InspectorPanel({
  runData,
  activeTab,
  onTabChange,
}: {
  runData: RunData;
  activeTab: InspectorTabId;
  onTabChange: (tab: InspectorTabId) => void;
}) {
  const { run, candidates, packagedCandidates, economics, loadingArtifacts, artifactErrors } =
    runData;
  const allocations = run?.optimization?.package?.allocations ?? [];

  const rollups = timeBlockRollups(allocations, economics);
  const guardrail = priceGuardrail(economics, allocations);
  const rotation = rotationRows(allocations, economics, packagedCandidates);

  return (
    <aside className="flex h-full flex-col border-l border-zinc-200/60 bg-zinc-50/40">
      <div
        role="tablist"
        aria-label="Reasoning engines"
        className="flex gap-1 border-b border-zinc-100 bg-white p-2"
      >
        {INSPECTOR_TABS.map((tab) => {
          const active = tab.id === activeTab;
          return (
            <button
              key={tab.id}
              role="tab"
              aria-selected={active}
              type="button"
              onClick={() => onTabChange(tab.id)}
              className={`flex-1 rounded-lg py-2 text-[13px] transition-all ${
                active
                  ? "bg-violet-950 font-bold text-white shadow-xs"
                  : "font-medium text-zinc-500 hover:bg-zinc-100/60"
              }`}
            >
              {tab.label}
            </button>
          );
        })}
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto p-4 text-[14px]">
        <ArtifactErrors errors={artifactErrors} />

        {activeTab === "d1" ? (
          <TabAudienceD1
            spec={run?.campaign_spec ?? null}
            candidates={candidates}
            candidatesRef={run?.artifacts.screen_candidates}
            rollups={rollups}
          />
        ) : null}

        {activeTab === "d2" ? (
          <TabRelevanceD2
            candidates={candidates}
            candidatesRef={run?.artifacts.screen_candidates}
            allocations={allocations}
            loading={loadingArtifacts}
          />
        ) : null}

        {activeTab === "d3" ? (
          <TabPricingD3
            guardrail={guardrail}
            rollups={rollups}
            economicsRef={run?.artifacts.screen_economics}
            loading={loadingArtifacts}
          />
        ) : null}

        {activeTab === "d4" ? (
          <TabOptimizerD4
            optimization={run?.optimization ?? null}
            validation={run?.validation ?? null}
            rows={rotation}
          />
        ) : null}
      </div>
    </aside>
  );
}
