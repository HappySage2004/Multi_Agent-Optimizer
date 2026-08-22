"use client";

/**
 * Executive synthesis card from UI.md §2: screen distribution and the rotation-slot
 * choices the optimizer made, stated from the package rather than described generically.
 */

import { ChevronRightIcon } from "@/components/ui/Icon";
import { type PackageMetrics, type TimeBlockRollup } from "@/lib/derive";
import { formatCompact, formatCurrency, formatPercent } from "@/lib/format";
import type { CampaignSpec } from "@/lib/types";

export function StrategySummaryCard({
  metrics,
  rollups,
  spec,
  onOpenInspector,
}: {
  metrics: PackageMetrics;
  rollups: TimeBlockRollup[];
  spec: CampaignSpec;
  onOpenInspector: () => void;
}) {
  const selected = rollups.filter((r) => r.selected);
  const peaks = selected.filter((r) => r.isPeak);
  const maxSlots = Math.max(...selected.map((r) => r.slotsPerDay), 0);

  return (
    <div className="space-y-2 rounded-xl border border-zinc-200/50 bg-zinc-50/60 p-4">
      <div className="flex items-center justify-between text-[11px] font-bold text-zinc-700">
        <span>Recommended Media Strategy</span>
        <button
          type="button"
          onClick={onOpenInspector}
          className="flex cursor-pointer items-center gap-1 text-[10px] font-semibold text-violet-950 hover:underline"
        >
          Inspector Panel
          <ChevronRightIcon className="h-3 w-3" />
        </button>
      </div>

      <p className="text-[11px] leading-relaxed text-zinc-500">
        {describeDistribution(metrics)} across{" "}
        <strong className="font-semibold text-zinc-600">{metrics.allocationCount}</strong>{" "}
        {metrics.allocationCount === 1 ? "line" : "lines"} in{" "}
        {selected.length === 0 ? "no time block" : joinLabels(selected.map((r) => r.label))}
        {peaks.length > 0
          ? `, concentrating on the ${peaks.length === 1 ? "commuter peak" : "commuter peaks"} ${joinLabels(peaks.map((r) => r.label))}`
          : ""}
        . Optimized for{" "}
        <strong className="font-semibold text-zinc-600">{spec.optimization_goal}</strong> over a{" "}
        {spec.duration_days}-day flight at {formatPercent(metrics.budgetUtilization)} budget
        utilization, with up to {maxSlots} rotation{" "}
        {maxSlots === 1 ? "slot" : "slots"} per screen per day.
      </p>

      <dl className="grid grid-cols-2 gap-x-4 gap-y-1 pt-1 text-[10px] sm:grid-cols-4">
        <Fact label="Solver" value={metrics.optimizationMethod} />
        <Fact
          label="Viewed exposures"
          value={formatCompact(metrics.grossImpressionsViewed)}
        />
        <Fact label="Spend" value={formatCurrency(metrics.totalCost)} />
        <Fact
          label="Objective"
          value={metrics.expectedReach > 0 ? formatCompact(metrics.expectedReach) : "—"}
        />
      </dl>
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="font-medium tracking-wide text-zinc-400 uppercase">{label}</dt>
      <dd className="truncate font-semibold text-zinc-600" title={value}>
        {value}
      </dd>
    </div>
  );
}

/** "18 metro_platform + 20 bus_stop screens", or a plain count when types are unknown. */
function describeDistribution(metrics: PackageMetrics): string {
  if (metrics.screenTypeBreakdown.length === 0) {
    return `Allocated ${metrics.screenCount} screens`;
  }
  const parts = metrics.screenTypeBreakdown.map((t) => `${t.count} ${t.label}`);
  return `Allocated ${joinLabels(parts)} screens`;
}

function joinLabels(items: string[]): string {
  if (items.length === 0) return "";
  if (items.length === 1) return items[0];
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}
