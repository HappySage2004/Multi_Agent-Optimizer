"use client";

/**
 * D3: how each recommended screen was priced (UI.md §2 Panel 3).
 *
 * The tab used to lead with pool-wide aggregates — mean price, mean occupancy, mean
 * booking probability across 750 rows. None of that survives contact with a client, who
 * asks about one screen: *why is this one $103 and that one $75?* So the aggregates are
 * gone and the tab is a per-screen answer instead.
 *
 * Each row shows the range comparable screens have actually sold for, where this quote
 * landed inside it, and the two or three things that put it there. The band comes from the
 * ML agent's price model; nothing is re-derived here.
 */

import {
  AwaitingStage,
  InspectorCard,
  InspectorSection,
} from "@/components/inspector/InspectorShell";
import { type PricingLine } from "@/lib/derive";
import { formatCurrency } from "@/lib/format";
import type { ArtifactReference } from "@/lib/types";

export function TabPricingD3({
  lines,
  economicsRef,
  hasAllocations,
  loading,
}: {
  lines: PricingLine[];
  economicsRef: ArtifactReference | undefined;
  hasAllocations: boolean;
  loading: boolean;
}) {
  if (!economicsRef) {
    return (
      <AwaitingStage
        stage="pricing (stage 3)"
        detail="Each shortlisted screen is priced against its own market comparables before the package is built."
      />
    );
  }

  if (!hasAllocations) {
    return (
      <AwaitingStage
        stage="the optimizer (stage 4)"
        detail="Screens have been priced, but this tab reports the quotes for the screens actually recommended."
      />
    );
  }

  const total = lines.reduce((sum, line) => sum + line.lineCost, 0);

  return (
    <>
      <InspectorCard
        title="How Each Screen Is Priced"
        badge={`${lines.length} ${lines.length === 1 ? "line" : "lines"}`}
        badgeTone="dark"
        description="Every quote is anchored to what comparable screens in the same area, of the same type and daypart, have actually sold for."
      />

      {loading && lines.length === 0 ? (
        <InspectorSection title="Screen Pricing">
          <p className="text-[10px] text-zinc-400">Loading pricing detail…</p>
        </InspectorSection>
      ) : null}

      {lines.length > 0 ? (
        <div className="space-y-1.5">
          <div className="flex items-baseline justify-between px-1 text-[10px]">
            <span className="font-bold tracking-wider text-zinc-400 uppercase">
              Highest cost first
            </span>
            <span className="font-medium text-zinc-400">{formatCurrency(total)} total</span>
          </div>

          {lines.map((line) => (
            <PricingRow key={`${line.screenId}-${line.timeBlockLabel}`} line={line} />
          ))}
        </div>
      ) : null}
    </>
  );
}

function PricingRow({ line }: { line: PricingLine }) {
  return (
    <div className="space-y-2 rounded-xl border border-zinc-200/50 bg-white p-2.5 shadow-xs">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate text-[11px] font-bold text-zinc-700">{line.place}</div>
          <div className="mt-0.5 truncate text-[10px] text-zinc-400">
            {line.timeBlockLabel} • {line.slotsPerDay}
            {line.maxSlotsPerDay !== null ? ` of ${line.maxSlotsPerDay}` : ""} slots/day •{" "}
            {line.screenId}
          </div>
        </div>
        <div className="shrink-0 text-right">
          <div className="text-[11px] font-bold text-zinc-800">
            {formatCurrency(line.lineCost)}
          </div>
          <div className="text-[10px] text-zinc-400">
            {formatCurrency(line.paid, 2)}/slot/day
          </div>
        </div>
      </div>

      <BandBar line={line} />

      <ul className="space-y-0.5 text-[10px] leading-relaxed text-zinc-500">
        {line.drivers.map((driver) => (
          <li key={driver} className="flex gap-1.5">
            <span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-zinc-300" />
            <span>{driver}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * Where this quote sits in the range comparable screens have sold for.
 *
 * A quote above the cap is drawn pinned to the right rather than off the end, and labelled
 * — the demand-value premium is allowed to exceed the cap by design (an under-priced
 * screen's own comparables are what understate it), so this is a real state, not a bug.
 */
function BandBar({ line }: { line: PricingLine }) {
  return (
    <div>
      <div className="relative pt-1.5 pb-1">
        <div className="h-1.5 overflow-hidden rounded-full bg-zinc-100">
          <div
            className="h-full rounded-full bg-violet-950/15"
            style={{ width: `${line.paidPosition * 100}%` }}
          />
        </div>
        <div
          className="absolute top-0.5 -translate-x-1/2"
          style={{ left: `${line.paidPosition * 100}%` }}
        >
          <span
            className={`block h-3.5 w-0.5 rounded ${line.aboveCap ? "bg-amber-500" : "bg-violet-950"}`}
          />
        </div>
      </div>

      <div className="flex justify-between text-[10px] text-zinc-400">
        <span>Low {formatCurrency(line.floor, 0)}</span>
        <span>Typical {formatCurrency(line.target, 0)}</span>
        <span className={line.aboveCap ? "font-semibold text-amber-700" : undefined}>
          High {formatCurrency(line.cap, 0)}
        </span>
      </div>
    </div>
  );
}
