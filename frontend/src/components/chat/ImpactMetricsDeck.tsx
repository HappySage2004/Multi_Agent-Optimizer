"use client";

/**
 * The 4-card metric grid from UI.md §2: Est. Reach (highlighted), Target Spend,
 * Screen Count, Effective CPM.
 *
 * Every figure comes from the optimizer's package. Effective CPM is the one derived
 * value — cost over impressions — and it is labelled as such.
 */

import { type PackageMetrics } from "@/lib/derive";
import { formatCompact, formatCurrency, formatPercent } from "@/lib/format";

export function ImpactMetricsDeck({ metrics }: { metrics: PackageMetrics }) {
  const headroom = metrics.budget - metrics.totalCost;

  return (
    <div className="my-3 grid grid-cols-2 gap-3 lg:grid-cols-4">
      <div className="rounded-xl border border-zinc-600 bg-zinc-700 p-3 text-white shadow-xs">
        <span className="text-[9px] font-bold tracking-wider text-zinc-300 uppercase">
          Est. Reach
        </span>
        <p className="mt-0.5 text-base font-black text-white">
          {formatCompact(metrics.expectedReach)}
        </p>
        <span className="text-[10px] font-medium text-emerald-300">
          {metrics.expectedFrequency > 0
            ? `${metrics.expectedFrequency.toFixed(2)}x avg frequency`
            : "Deduplicated reach"}
        </span>
      </div>

      <MetricCard
        label="Target Spend"
        value={formatCurrency(metrics.totalCost)}
        footnote={
          headroom >= 0
            ? `${formatPercent(metrics.budgetUtilization)} of ${formatCurrency(metrics.budget)} budget`
            : `Over budget by ${formatCurrency(Math.abs(headroom))}`
        }
        footnoteClass={headroom >= 0 ? "text-zinc-400" : "text-red-600 font-semibold"}
      />

      <MetricCard
        label="Screen Count"
        value={`${metrics.screenCount} ${metrics.screenCount === 1 ? "Unit" : "Units"}`}
        footnote={
          metrics.screenTypeBreakdown.length > 0
            ? metrics.screenTypeBreakdown.map((t) => `${t.count} ${t.label}`).join(" + ")
            : `${metrics.allocationCount} allocated lines`
        }
        footnoteClass="text-violet-950 font-semibold"
      />

      <MetricCard
        label="Effective CPM"
        value={metrics.effectiveCpm === null ? "—" : formatCurrency(metrics.effectiveCpm, 2)}
        footnote={
          metrics.effectiveCpm === null
            ? "No forecast impressions"
            : `${formatCompact(metrics.expectedImpressions)} impressions`
        }
        footnoteClass="text-emerald-700 font-semibold"
      />
    </div>
  );
}

function MetricCard({
  label,
  value,
  footnote,
  footnoteClass,
}: {
  label: string;
  value: string;
  footnote: string;
  footnoteClass: string;
}) {
  return (
    <div className="rounded-xl border border-zinc-200/50 bg-zinc-50/70 p-3">
      <span className="text-[9px] font-bold tracking-wider text-zinc-400 uppercase">{label}</span>
      <p className="mt-0.5 text-base font-bold text-zinc-700">{value}</p>
      <span className={`block truncate text-[10px] ${footnoteClass}`} title={footnote}>
        {footnote}
      </span>
    </div>
  );
}
