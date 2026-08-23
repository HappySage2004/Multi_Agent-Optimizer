"use client";

/**
 * The client-facing proposal, laid out for paper.
 *
 * WHY THIS EXISTS. "Export Proposal PDF" used to call `window.print()` against the live
 * workspace — a three-column resizable app with two scroll containers and a chat
 * transcript. The browser rendered whatever happened to be in the viewport, which is why
 * the PDF was unreadable. Printing needs its own document, so this is it: hidden on screen,
 * the only thing on the page under `@media print` (see globals.css).
 *
 * WHAT A CLIENT MAY SEE. What they are buying and what it costs — screens, places, times,
 * per-line cost, totals, and the delivery estimate. Nothing about HOW the price was
 * reached. Specifically absent, and deliberately: price bands and where in one we quoted,
 * occupancy, booking probability, relevance scores, pool keys, the solver and its gap,
 * pricing levers, and the internal reasoning from the chat answer. Those are the seller's
 * position in a negotiation the client is on the other side of.
 */

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

import {
  type PackageMetrics,
  flightEndDate,
  packageLineRows,
  packagePlaces,
} from "@/lib/derive";
import { formatCurrency, formatDate, formatNumber } from "@/lib/format";
import type { CampaignSpec, OptimizedPackage, ScreenCandidate } from "@/lib/types";

export function ProposalDocument({
  pkg,
  spec,
  candidates,
  metrics,
  title,
}: {
  pkg: OptimizedPackage;
  spec: CampaignSpec;
  candidates: ScreenCandidate[];
  metrics: PackageMetrics;
  title: string;
}) {
  // Portalled to <body> so the print stylesheet can hide `body > *` and re-show this one
  // branch. Rendered inside the workspace tree it would inherit the app's scroll
  // containers and flex layout, which is what made the old print output unusable.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const rows = packageLineRows(pkg, candidates);
  const places = packagePlaces(rows);
  const endDate = flightEndDate(spec.start_date, spec.duration_days);

  if (!mounted) return null;

  return createPortal(
    <article
      id="proposal-document"
      aria-hidden
      // Off-screen rather than `display: none`: a hidden subtree is not laid out, and the
      // print stylesheet needs real geometry to paginate the table.
      className="pointer-events-none absolute -left-[200vw] top-0 w-[190mm] bg-white text-[10pt] text-zinc-800"
    >
      <header className="mb-6 border-b-2 border-zinc-800 pb-3">
        <p className="text-[8pt] font-bold tracking-[0.18em] text-zinc-500 uppercase">
          Transit Media Proposal
        </p>
        <h1 className="mt-1 text-[17pt] leading-tight font-bold text-zinc-900">{title}</h1>
        <p className="mt-1 text-[9pt] text-zinc-500">
          Prepared {formatDate(new Date().toISOString().slice(0, 10))}
        </p>
      </header>

      <section className="mb-6">
        <H2>Campaign at a glance</H2>
        <dl className="grid grid-cols-2 gap-x-8 gap-y-1.5">
          <Fact label="Flight" value={`${formatDate(spec.start_date)} – ${formatDate(endDate)}`} />
          <Fact label="Duration" value={`${spec.duration_days} days`} />
          <Fact
            label="Screens"
            value={`${metrics.screenCount} across ${rows.length} placements`}
          />
          <Fact label="Locations" value={places.length > 0 ? places.join(", ") : "—"} />
          <Fact
            label="Formats"
            value={
              metrics.screenTypeBreakdown.length > 0
                ? metrics.screenTypeBreakdown
                    .map((t) => `${t.count} × ${titleWords(t.label)}`)
                    .join(", ")
                : "—"
            }
          />
          <Fact label="Investment" value={formatCurrency(metrics.totalCost, 2)} />
        </dl>
      </section>

      <section className="mb-6">
        <H2>What this delivers</H2>
        <div className="grid grid-cols-3 gap-3">
          <Metric
            label="People reached"
            value={formatNumber(metrics.expectedReach)}
            note="Distinct individuals. Screens sharing a stop or route see the same people, so this is not a sum of views."
          />
          <Metric
            label="Total views"
            value={formatNumber(metrics.grossImpressionsViewed)}
            note="Every occasion the creative is seen across the flight."
          />
          <Metric
            label="Average frequency"
            value={`${metrics.expectedFrequency.toFixed(1)}×`}
            note="How often the average person reached sees the campaign."
          />
        </div>
        {metrics.effectiveCpm !== null ? (
          <p className="mt-2 text-[9pt] text-zinc-500">
            Effective cost per thousand views:{" "}
            <strong className="font-semibold text-zinc-700">
              {formatCurrency(metrics.effectiveCpm, 2)}
            </strong>
          </p>
        ) : null}
        <p className="mt-1.5 text-[8pt] leading-relaxed text-zinc-400">
          Delivery figures are estimates based on scheduled transit service and historical
          ridership for these locations. They are not a guarantee of performance.
        </p>
      </section>

      <section>
        <H2>Schedule of inventory</H2>
        <table className="w-full border-collapse text-[8.5pt]">
          <thead>
            <tr className="border-b-2 border-zinc-300 text-left">
              <Th>Screen</Th>
              <Th>Location</Th>
              <Th>Format</Th>
              <Th>Time of day</Th>
              <Th align="right">Slots/day</Th>
              <Th align="right">Rate/slot/day</Th>
              <Th align="right">Cost ({spec.duration_days} days)</Th>
              <Th align="right">Est. views</Th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              // `break-inside-avoid` keeps a row off a page boundary; without it the
              // printer splits cells mid-line, which is half of why the old PDF was
              // unreadable.
              <tr key={row.key} className="break-inside-avoid border-b border-zinc-100">
                <Td className="font-mono text-[7.5pt]">{row.screenId}</Td>
                <Td className="font-medium">{row.place}</Td>
                <Td>{row.screenType}</Td>
                <Td className="whitespace-nowrap">{row.timeBlock}</Td>
                <Td align="right">{row.slotsPerDay}</Td>
                <Td align="right">{formatCurrency(row.pricePerSlotPerDay, 2)}</Td>
                <Td align="right">{formatCurrency(row.lineCost, 2)}</Td>
                <Td align="right">{formatNumber(row.viewedExposures)}</Td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr className="border-t-2 border-zinc-800 font-bold">
              <Td colSpan={6}>Total investment</Td>
              <Td align="right">{formatCurrency(metrics.totalCost, 2)}</Td>
              <Td align="right">{formatNumber(metrics.grossImpressionsViewed)}</Td>
            </tr>
          </tfoot>
        </table>
      </section>

      <footer className="mt-8 border-t border-zinc-200 pt-3 text-[8pt] text-zinc-400">
        Rates are held for 14 days from the date of this proposal and are subject to
        inventory remaining available at the time of booking.
      </footer>
    </article>,
    document.body,
  );
}

function H2({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="mb-2 text-[8pt] font-bold tracking-[0.14em] text-zinc-400 uppercase">
      {children}
    </h2>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex gap-2 border-b border-zinc-100 pb-1">
      <dt className="w-24 shrink-0 text-zinc-500">{label}</dt>
      <dd className="font-medium text-zinc-800">{value}</dd>
    </div>
  );
}

function Metric({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="break-inside-avoid rounded border border-zinc-200 p-2.5">
      <div className="text-[14pt] leading-none font-bold text-zinc-900">{value}</div>
      <div className="mt-1 text-[8.5pt] font-semibold text-zinc-600">{label}</div>
      <p className="mt-1 text-[7.5pt] leading-snug text-zinc-400">{note}</p>
    </div>
  );
}

function Th({ children, align = "left" }: { children: React.ReactNode; align?: "left" | "right" }) {
  return (
    <th
      scope="col"
      className={`px-1.5 py-1.5 font-semibold text-zinc-600 ${align === "right" ? "text-right" : "text-left"}`}
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
      className={`px-1.5 py-1 ${align === "right" ? "text-right" : "text-left"} ${className}`}
    >
      {children}
    </td>
  );
}

/** `metro_station` -> `Metro Station`. The breakdown is keyed on the raw screen_type. */
function titleWords(value: string): string {
  return value
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}
