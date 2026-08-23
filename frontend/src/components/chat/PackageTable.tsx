"use client";

/**
 * The Headline Package Overview — every line of inventory being offered.
 *
 * Rendered by the UI, not written by the agent. Three reasons that split is worth it: a
 * component cannot mis-transcribe a price, the numbers come from the same run record the
 * validator checked, and a 12-row Markdown table costs output tokens on every single turn.
 * The agent writes the sentence above it; this owns the figures.
 *
 * WHEN IT APPEARS. Only under an assistant turn that actually built or rebuilt a package —
 * `AssistantBlock` passes a run only when `message.runId` is set, which the hook sets only
 * when `pipeline_ran` is true. A plain enquiry ("what does viewability mean?") resolves to
 * the same run but carries no `runId`, so no table. A rebuild gets a fresh one showing the
 * updated plan, which is what makes two of them in a transcript a useful before/after.
 */

import { useState } from "react";

import { ChevronRightIcon } from "@/components/ui/Icon";
import { type PackageLineRow, packageLineRows, packagePlaces } from "@/lib/derive";
import { formatCurrency, formatNumber } from "@/lib/format";
import type { CampaignSpec, OptimizedPackage, ScreenCandidate } from "@/lib/types";

/** Rows past this are collapsed behind a toggle. Enough to read a plan without scrolling. */
const VISIBLE_ROWS = 12;

export function PackageTable({
  pkg,
  spec,
  candidates,
}: {
  pkg: OptimizedPackage;
  spec: CampaignSpec;
  /** Candidate rows for the bought screens — where the zone name and screen type come from. */
  candidates: ScreenCandidate[];
}) {
  const [expanded, setExpanded] = useState(false);

  // Same helper the printed proposal uses, so the chat table and the client PDF cannot
  // disagree about what is being sold.
  const rows: PackageLineRow[] = packageLineRows(pkg, candidates);

  if (rows.length === 0) return null;

  const visible = expanded ? rows : rows.slice(0, VISIBLE_ROWS);
  const hidden = rows.length - visible.length;
  const places = packagePlaces(rows);
  // `screen_ids` is a Python @property and never crosses the wire, so count distinct
  // screens here. One screen can hold several lines across time blocks.
  const screenCount = new Set(rows.map((r) => r.screenId)).size;

  return (
    <section className="overflow-hidden rounded-xl border border-zinc-200/70 bg-white shadow-xs">
      <header className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 border-b border-zinc-200/70 bg-zinc-50/70 px-4 py-2.5">
        <div className="space-y-0.5">
          <h3 className="text-[11px] font-bold text-zinc-700">Headline Package Overview</h3>
          <p className="text-[10px] text-zinc-500">
            {screenCount} {screenCount === 1 ? "screen" : "screens"} ·{" "}
            {rows.length} {rows.length === 1 ? "line" : "lines"} · {spec.duration_days} days
            {places.length > 0 ? ` · ${places.join(", ")}` : ""}
          </p>
        </div>
        <div className="text-right">
          <div className="text-[13px] font-bold text-violet-950">
            {formatCurrency(pkg.total_cost, 2)}
          </div>
          <div className="text-[10px] text-zinc-400">total campaign cost</div>
        </div>
      </header>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-left">
          <thead>
            <tr className="border-b border-zinc-200/70 text-[9px] tracking-wider text-zinc-400 uppercase">
              <Th>Screen ID</Th>
              <Th>Location</Th>
              <Th>Screen Type</Th>
              <Th>Time Block</Th>
              <Th align="right">Slots/Day</Th>
              <Th align="right">Price/Slot/Day</Th>
              <Th align="right">Line Cost ({spec.duration_days} Days)</Th>
              <Th align="right">Viewed Exposures</Th>
            </tr>
          </thead>
          <tbody>
            {visible.map((row) => (
              <tr key={row.key} className="border-b border-zinc-100 last:border-0 hover:bg-zinc-50/60">
                <Td>
                  <span className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-[10px] font-medium text-zinc-600">
                    {row.screenId}
                  </span>
                </Td>
                <Td className="font-medium text-zinc-700">{row.place}</Td>
                <Td>{row.screenType}</Td>
                <Td className="whitespace-nowrap">{row.timeBlock}</Td>
                <Td align="right" className="tabular-nums">
                  {row.slotsPerDay}
                </Td>
                <Td align="right" className="tabular-nums">
                  {formatCurrency(row.pricePerSlotPerDay, 2)}
                </Td>
                <Td align="right" className="tabular-nums font-medium text-zinc-700">
                  {formatCurrency(row.lineCost, 2)}
                </Td>
                <Td align="right" className="tabular-nums">
                  {formatNumber(row.viewedExposures)}
                </Td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr className="border-t border-zinc-200 bg-zinc-50/70 text-[11px] font-bold text-zinc-700">
              <Td colSpan={6}>Total{expanded || hidden === 0 ? "" : ` (all ${rows.length} lines)`}</Td>
              <Td align="right" className="tabular-nums">
                {formatCurrency(pkg.total_cost, 2)}
              </Td>
              <Td align="right" className="tabular-nums">
                {formatNumber(pkg.gross_impressions_viewed)}
              </Td>
            </tr>
          </tfoot>
        </table>
      </div>

      {hidden > 0 ? (
        <button
          type="button"
          onClick={() => setExpanded(true)}
          className="flex w-full cursor-pointer items-center justify-center gap-1 border-t border-zinc-200/70 bg-white py-2 text-[10px] font-semibold text-violet-950 hover:bg-zinc-50"
        >
          Show {hidden} more {hidden === 1 ? "line" : "lines"}
          <ChevronRightIcon className="h-3 w-3 rotate-90" />
        </button>
      ) : null}
    </section>
  );
}

function Th({
  children,
  align = "left",
}: {
  children: React.ReactNode;
  align?: "left" | "right";
}) {
  return (
    <th
      scope="col"
      className={`px-3 py-2 font-semibold ${align === "right" ? "text-right" : "text-left"}`}
    >
      {children}
    </th>
  );
}

function Td({
  children,
  align = "left",
  className = "",
  colSpan,
}: {
  children: React.ReactNode;
  align?: "left" | "right";
  className?: string;
  colSpan?: number;
}) {
  return (
    <td
      colSpan={colSpan}
      className={`px-3 py-1.5 text-[11px] text-zinc-500 ${align === "right" ? "text-right" : "text-left"} ${className}`}
    >
      {children}
    </td>
  );
}
