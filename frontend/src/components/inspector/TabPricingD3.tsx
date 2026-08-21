"use client";

/**
 * D3: Demand Forecasting & Dynamic Pricing Guardrails (UI.md §2 Panel 3).
 *
 * The gauge shows the floor/target/cap band the ML Agent produced, averaged over the
 * priced inventory, with a marker for the price the optimizer actually paid. The
 * six-block grid is real `dim_slot` occupancy: slots bought over slots forecast
 * available, for the screens in the package.
 */

import {
  AwaitingStage,
  InspectorCard,
  InspectorSection,
  StubNotice,
} from "@/components/inspector/InspectorShell";
import { type PriceGuardrail, type TimeBlockRollup } from "@/lib/derive";
import { formatCompact, formatCurrency, formatNumber, formatPercent } from "@/lib/format";
import type { ArtifactReference, ScreenEconomicsSummary } from "@/lib/types";

export function TabPricingD3({
  guardrail,
  rollups,
  economicsRef,
  loading,
}: {
  guardrail: PriceGuardrail | null;
  rollups: TimeBlockRollup[];
  economicsRef: ArtifactReference | undefined;
  loading: boolean;
}) {
  if (!economicsRef) {
    return (
      <AwaitingStage
        stage="the ML Agent (stage 3)"
        detail="Demand forecasting and pricing produce the screen_economics artifact this panel reads."
      />
    );
  }

  const summary = (economicsRef.summary ?? {}) as ScreenEconomicsSummary;

  return (
    <>
      <InspectorCard
        title="Pricing Guardrails"
        badge="Floor / Target / Cap"
        badgeTone="dark"
        description="Price boundaries per screen and time block, from the demand forecast and market price model."
      >
        {economicsRef.provenance === "stub" ? (
          <StubNotice stage="Demand and pricing (ML Agent)" />
        ) : null}
      </InspectorCard>

      <InspectorSection
        title="Forecast Aggregates"
        meta={`${formatNumber(economicsRef.rows)} rows`}
      >
        <div className="grid grid-cols-2 gap-2 text-[11px]">
          <Stat
            label="Screens priced"
            value={summary.screens !== undefined ? formatNumber(summary.screens) : "—"}
          />
          <Stat
            label="Mean price"
            value={summary.price_mean !== undefined ? formatCurrency(summary.price_mean, 2) : "—"}
          />
          <Stat
            label="Impressions / slot / day"
            value={
              summary.impressions_per_slot_day_mean !== undefined
                ? formatCompact(summary.impressions_per_slot_day_mean)
                : "—"
            }
          />
          <Stat
            label="Min confidence"
            value={
              summary.confidence_min !== undefined ? summary.confidence_min.toFixed(3) : "—"
            }
          />
        </div>
        {summary.time_blocks && summary.time_blocks.length > 0 ? (
          <p className="text-[10px] text-zinc-400">
            Priced blocks: {summary.time_blocks.join(", ")}
          </p>
        ) : null}
      </InspectorSection>

      {loading && !guardrail ? (
        <InspectorSection title="Slot Price Guardrail">
          <p className="text-[10px] text-zinc-400">Loading economics rows…</p>
        </InspectorSection>
      ) : null}

      {guardrail ? <GuardrailGauge guardrail={guardrail} /> : null}

      <OccupancyGrid rollups={rollups} />
    </>
  );
}

function GuardrailGauge({ guardrail }: { guardrail: PriceGuardrail }) {
  // The band is split floor->target->cap, so segment widths mirror the real spread.
  const span = guardrail.cap - guardrail.floor;
  const targetPct = span > 0 ? ((guardrail.target - guardrail.floor) / span) * 100 : 50;

  return (
    <InspectorSection
      title="Slot Price Guardrail"
      meta={`${formatCurrency(guardrail.target, 2)} target`}
    >
      <div className="relative pt-2 pb-1">
        <div className="flex h-2 overflow-hidden rounded-full bg-zinc-100">
          <div className="bg-zinc-200" style={{ width: `${targetPct}%` }} />
          <div className="bg-violet-950" style={{ width: `${Math.max(100 - targetPct, 0)}%` }} />
        </div>

        {guardrail.paidPosition !== null ? (
          <div
            className="absolute top-0 flex -translate-x-1/2 flex-col items-center"
            style={{ left: `${guardrail.paidPosition * 100}%` }}
            title={`Volume-weighted price paid: ${formatCurrency(guardrail.paid ?? 0, 2)}`}
          >
            <span className="h-4 w-0.5 rounded bg-zinc-700" />
          </div>
        ) : null}
      </div>

      <div className="flex justify-between text-[10px] font-medium text-zinc-400">
        <span>Floor: {formatCurrency(guardrail.floor, 0)}</span>
        <span className="font-bold text-violet-950">
          Target: {formatCurrency(guardrail.target, 0)}
        </span>
        <span>Cap: {formatCurrency(guardrail.cap, 0)}</span>
      </div>

      <dl className="grid grid-cols-2 gap-2 border-t border-zinc-100 pt-2 text-[10px]">
        <div>
          <dt className="text-zinc-400">Weighted price paid</dt>
          <dd className="font-semibold text-zinc-700">
            {guardrail.paid !== null ? formatCurrency(guardrail.paid, 2) : "No package yet"}
          </dd>
        </div>
        <div>
          <dt className="text-zinc-400">Mean booking probability</dt>
          <dd className="font-semibold text-zinc-700">
            {guardrail.meanBookingProbability !== null
              ? formatPercent(guardrail.meanBookingProbability)
              : "—"}
          </dd>
        </div>
      </dl>

      <p className="text-[9px] leading-relaxed text-zinc-400">
        Band is the mean floor/target/cap across the{" "}
        {formatNumber(guardrail.screensPriced)} screens in this package. The marker is total
        spend over total slots bought. Pool-wide mean price is shown above.
      </p>
    </InspectorSection>
  );
}

/** The 6-block occupancy grid. Always all six real `dim_slot` blocks. */
function OccupancyGrid({ rollups }: { rollups: TimeBlockRollup[] }) {
  const anyPriced = rollups.some((r) => r.demandIndex !== null || r.selected);

  return (
    <InspectorSection title="6-Block Time Occupancy" meta="dim_slot">
      <div className="grid grid-cols-3 gap-1.5 text-center text-[10px] font-medium">
        {rollups.map((rollup) => {
          const label =
            rollup.occupancy !== null
              ? formatPercent(rollup.occupancy, 0)
              : rollup.demandIndex !== null
                ? `idx ${rollup.demandIndex.toFixed(2)}`
                : "—";
          return (
            <div
              key={rollup.id}
              title={describeBlock(rollup)}
              className={`rounded p-2 ${
                rollup.selected
                  ? "bg-violet-950 font-bold text-white"
                  : rollup.demandIndex !== null
                    ? "bg-zinc-100 text-zinc-700"
                    : "bg-zinc-100/60 text-zinc-400"
              }`}
            >
              <span className="block">{rollup.label}</span>
              <span className="block text-[9px] opacity-80">{label}</span>
            </div>
          );
        })}
      </div>

      <p className="text-[9px] leading-relaxed text-zinc-400">
        {anyPriced
          ? "Violet = bought by the optimizer, showing slots bought over slots available. Grey with an index = priced but not bought."
          : "No blocks priced yet."}
      </p>
    </InspectorSection>
  );
}

function describeBlock(rollup: TimeBlockRollup): string {
  const parts = [rollup.label];
  if (rollup.isPeak) parts.push("commuter peak");
  if (rollup.selected) {
    parts.push(`${rollup.screens} screens`, `${rollup.slotsPerDay} slots/day`);
    parts.push(`${formatCompact(rollup.impressions)} impressions`);
    parts.push(formatCurrency(rollup.cost));
  } else {
    parts.push("not bought");
  }
  if (rollup.meanPrice !== null) parts.push(`mean price ${formatCurrency(rollup.meanPrice, 2)}`);
  return parts.join(" • ");
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
