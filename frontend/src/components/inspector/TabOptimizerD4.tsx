"use client";

/**
 * D4: what the package buys (UI.md §2 Panel 3).
 *
 * Two sections were removed as engineering diagnostics rather than sales material: the
 * solver log (HiGHS output, gaps, timings) and the 17-check validation list. The validator
 * still runs and still gates the answer — the Master Agent will not report a package that
 * fails it — but a rep reading `reach_matches_pool_cap: pass` learns nothing they can say
 * to a client.
 *
 * What survives is stated in campaign terms: what it costs, who it reaches, and how the
 * airtime is split across each screen's rotation loop.
 */

import {
  AwaitingStage,
  InspectorCard,
  InspectorSection,
  Stat,
} from "@/components/inspector/InspectorShell";
import { CheckIcon, WarningIcon } from "@/components/ui/Icon";
import { ROTATION_LOOP_SLOTS, type RotationRow } from "@/lib/derive";
import { formatCompact, formatCurrency, formatNumber, formatPercent, titleCase } from "@/lib/format";
import type { OptimizationResult } from "@/lib/types";

/**
 * Plain names for the hard constraints the checker reports. An unmapped key falls back to
 * title case rather than being hidden — a constraint nobody named is still one the rep
 * needs to see the status of.
 */
const CONSTRAINT_LABELS: Record<string, string> = {
  budget: "Within budget",
  budget_respected: "Within budget",
  availability: "Screens are available",
  availability_respected: "Screens are available",
  dates: "Runs on the requested dates",
  geography: "Inside the requested area",
  screen_count: "Screen count respected",
  slot_cap: "Daily airtime cap respected",
  wear_out: "Repetition kept in check",
  zone_coverage: "Required zones covered",
};

export function TabOptimizerD4({
  optimization,
  rows,
}: {
  optimization: OptimizationResult | null;
  rows: RotationRow[];
}) {
  if (!optimization) {
    return (
      <AwaitingStage
        stage="the optimizer (stage 4)"
        detail="The optimizer chooses the screens, time blocks and airtime this panel reports."
      />
    );
  }

  const pkg = optimization.package;
  const solved = optimization.status === "optimal" || optimization.status === "feasible";

  return (
    <>
      <InspectorCard
        title="Recommended Plan"
        // "Optimal" and "feasible" are solver words. Both mean the same thing to a rep:
        // this is a valid plan inside the brief's limits.
        badge={solved ? "Plan ready" : titleCase(optimization.status)}
        badgeTone={solved ? "active" : "warning"}
        description="The best mix of screens, time blocks and airtime the budget, availability and dates allow."
      />

      {optimization.infeasibility ? (
        <InspectorSection title="No plan fits the brief">
          <p className="text-[11px] leading-relaxed text-red-700">
            {optimization.infeasibility.explanation}
          </p>
        </InspectorSection>
      ) : null}

      {pkg ? (
        <>
          <InspectorSection title="What This Plan Buys">
            <div className="grid grid-cols-2 gap-2">
              <Stat label="Spend" value={formatCurrency(pkg.total_cost)} />
              <Stat label="Budget used" value={formatPercent(pkg.budget_utilization)} />
              <Stat label="People reached" value={formatCompact(pkg.expected_reach)} />
              <Stat
                label="Times seen, each"
                value={`${pkg.expected_frequency.toFixed(1)}x`}
              />
              <Stat label="Total views" value={formatCompact(pkg.gross_impressions_viewed)} />
              <Stat label="Screen placements" value={formatNumber(pkg.allocations.length)} />
            </div>
            <p className="text-[10px] leading-relaxed text-zinc-400">
              People reached counts each person once, even when several screens on the same
              platform or route show them the ad. Total views counts every showing.
            </p>
          </InspectorSection>

          <ConstraintStatus status={pkg.constraint_status} />

          <RotationMatrix rows={rows} />
        </>
      ) : null}
    </>
  );
}

/** Hard constraints are enforced in code; this reports what the checker found. */
function ConstraintStatus({ status }: { status: Record<string, boolean> }) {
  const entries = Object.entries(status);
  if (entries.length === 0) return null;

  const failures = entries.filter(([, passed]) => !passed);

  return (
    <InspectorSection
      title="Brief Requirements"
      meta={failures.length === 0 ? "All met" : `${failures.length} not met`}
    >
      <div className="flex flex-wrap gap-1.5">
        {entries.map(([name, passed]) => (
          <span
            key={name}
            className={`flex items-center gap-1 rounded-md px-2 py-1 text-[10px] font-medium ${
              passed
                ? "border border-emerald-200/60 bg-emerald-50 text-emerald-700"
                : "border border-red-200/60 bg-red-50 text-red-700"
            }`}
          >
            {passed ? (
              <CheckIcon className="h-2.5 w-2.5" strokeWidth={3} />
            ) : (
              <WarningIcon className="h-2.5 w-2.5" />
            )}
            {CONSTRAINT_LABELS[name] ?? titleCase(name)}
          </span>
        ))}
      </div>
    </InspectorSection>
  );
}

function RotationMatrix({ rows }: { rows: RotationRow[] }) {
  if (rows.length === 0) {
    return (
      <InspectorSection title="Airtime Per Screen">
        <p className="text-[10px] text-zinc-400">No placements in this plan.</p>
      </InspectorSection>
    );
  }

  return (
    <InspectorSection
      title="Airtime Per Screen"
      meta={`${rows.length} ${rows.length === 1 ? "placement" : "placements"}`}
    >
      <p className="text-[10px] leading-relaxed text-zinc-400">
        Each screen runs a loop of {ROTATION_LOOP_SLOTS} ad slots. The filled blocks are how
        many of them this campaign holds — hold two of six and the ad appears on every third
        pass of the loop, all day.
      </p>

      <div className="space-y-2">
        {rows.map((row) => (
          <div key={`${row.screenId}-${row.timeBlockId}`} className="space-y-1">
            <div className="flex items-baseline justify-between gap-2 text-[10px]">
              <span className="truncate font-semibold text-zinc-700">{row.screenId}</span>
              <span className="shrink-0 font-medium text-zinc-400">{row.timeBlockLabel}</span>
            </div>

            <div className="grid grid-cols-6 gap-1 text-center text-[10px] font-bold">
              {row.slots.map((active, index) => (
                <div
                  key={index}
                  className={`rounded p-1 ${
                    active
                      ? "bg-violet-950 text-white shadow-xs"
                      : index < (row.maxSlotsPerDay ?? ROTATION_LOOP_SLOTS)
                        ? "bg-zinc-100/70 text-zinc-400"
                        : "bg-zinc-50 text-zinc-300"
                  }`}
                  title={
                    active
                      ? `Slot ${index + 1} — held by this campaign`
                      : index < (row.maxSlotsPerDay ?? ROTATION_LOOP_SLOTS)
                        ? `Slot ${index + 1} — free, not taken`
                        : `Slot ${index + 1} — already sold for these dates`
                  }
                >
                  {index + 1}
                </div>
              ))}
            </div>

            <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-zinc-400">
              <span>
                {row.slotsPerDay} of {row.maxSlotsPerDay ?? ROTATION_LOOP_SLOTS} slots
              </span>
              <span>{formatCurrency(row.pricePerSlotPerDay, 2)}/slot/day</span>
              <span>{formatCurrency(row.lineCost)} total</span>
              <span>{formatCompact(row.expectedImpressions)} views</span>
              {row.relevanceScore !== null ? (
                <span>{Math.round(row.relevanceScore * 100)}% fit</span>
              ) : null}
            </div>
          </div>
        ))}
      </div>
    </InspectorSection>
  );
}
