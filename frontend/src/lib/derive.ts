/**
 * Turns a run record into the values the UI displays.
 *
 * Two rules, both from SOLUTION.md §31:
 *   1. Nothing here invents a number. Every field traces to the run record, the
 *      optimizer's package, or an artifact row. When the data is absent the helper
 *      returns null and the component shows an empty state.
 *   2. Arithmetic is limited to restating what the optimizer already computed —
 *      effective CPM from cost and impressions, shares of a total. No re-optimization,
 *      no re-forecasting.
 */

import { titleCase } from "./format";
import type {
  Allocation,
  CampaignSpec,
  OptimizedPackage,
  Provenance,
  RunRecord,
  ScreenCandidate,
  ScreenEconomics,
} from "./types";

/** The six real time blocks from `datasets/dim_slot.csv`. */
export const TIME_BLOCKS = [
  { id: "1", label: "00:00-04:00", daypart: "night" },
  { id: "2", label: "04:00-08:00", daypart: "morning" },
  { id: "3", label: "08:00-12:00", daypart: "midday" },
  { id: "4", label: "12:00-16:00", daypart: "afternoon" },
  { id: "5", label: "16:00-20:00", daypart: "evening" },
  { id: "6", label: "20:00-24:00", daypart: "night" },
] as const;

/** Commuter peaks, per the `dim_slot` daypart mapping used by the ML Agent. */
export const PEAK_BLOCK_IDS = new Set(["2", "5"]);

export function timeBlockLabel(id: string): string {
  return TIME_BLOCKS.find((b) => b.id === id)?.label ?? `Block ${id}`;
}

/**
 * "16:00-20:00 (Evening)" — the clock hours and the daypart together.
 *
 * What the package table shows. A rep reads a time and a part of the day; "Block 5" is an
 * internal id and means nothing on a client call.
 */
export function timeBlockDaypartLabel(id: string): string {
  const block = TIME_BLOCKS.find((b) => b.id === id);
  if (!block) return `Block ${id}`;
  return `${block.label} (${titleCase(block.daypart)})`;
}

/** Slots in one rotation loop. Matches the D4 six-slot matrix in UI.md. */
export const ROTATION_LOOP_SLOTS = 6;

// --------------------------------------------------------------- package lines

/**
 * One purchasable line, labelled for a person rather than for the pipeline.
 *
 * Shared by the in-chat `PackageTable` and the printed proposal so the two can never
 * disagree about what is being sold. Everything here is client-safe: what the screen is,
 * where it is, when it runs, what it costs. Nothing about how the price was derived.
 */
export interface PackageLineRow {
  key: string;
  screenId: string;
  /** Zone name where there is one, else the corridor. Never a raw zone id. */
  place: string;
  screenType: string;
  timeBlock: string;
  slotsPerDay: number;
  pricePerSlotPerDay: number;
  lineCost: number;
  viewedExposures: number;
}

export function packageLineRows(
  pkg: OptimizedPackage,
  candidates: ScreenCandidate[] = [],
): PackageLineRow[] {
  const byScreen = new Map(candidates.map((c) => [c.screen_id, c]));
  return pkg.allocations
    .map((a, index) => {
      const candidate = byScreen.get(a.screen_id);
      return {
        key: `${a.screen_id}-${a.time_block_id}-${index}`,
        screenId: a.screen_id,
        place: placeLabel(candidate),
        screenType: candidate?.screen_type ? titleCase(candidate.screen_type) : "—",
        timeBlock: timeBlockDaypartLabel(a.time_block_id),
        slotsPerDay: a.slots_per_day,
        pricePerSlotPerDay: a.price_per_slot_per_day,
        lineCost: lineCost(a),
        viewedExposures: a.viewed_exposures,
      };
    })
    .sort((a, b) => b.viewedExposures - a.viewedExposures);
}

/**
 * Where a screen is, in the words a client would use.
 *
 * The stop or station name FIRST — an advertiser can picture "East Commons Station"; a zone
 * is a planning unit and a zone id is meaningless to them. Vehicle-mounted screens have no
 * fixed location, so their corridor names them. The ids are last and only exist so a cell
 * is never blank.
 */
export function placeLabel(candidate: ScreenCandidate | undefined): string {
  if (!candidate) return "—";
  return (
    candidate.location_name ??
    candidate.corridor_id ??
    candidate.zone_name ??
    candidate.zone_id ??
    "—"
  );
}

/** Distinct named places in a package, for a summary line. */
export function packagePlaces(rows: PackageLineRow[]): string[] {
  return [...new Set(rows.map((r) => r.place))].filter((p) => p !== "—");
}

/** Inclusive flight end: a 30-day flight starting on the 1st runs through the 30th. */
export function flightEndDate(startIso: string, durationDays: number): string {
  const start = new Date(`${startIso}T00:00:00Z`);
  if (Number.isNaN(start.getTime())) return startIso;
  start.setUTCDate(start.getUTCDate() + Math.max(1, durationDays) - 1);
  return start.toISOString().slice(0, 10);
}

// ------------------------------------------------------------- package metrics

export interface PackageMetrics {
  totalCost: number;
  budget: number;
  /** Optimizer's own figure, not recomputed. */
  budgetUtilization: number;
  /** Gross VIEWED exposures. Never label this as people in the UI. */
  grossImpressionsViewed: number;
  expectedReach: number;
  expectedFrequency: number;
  screenCount: number;
  allocationCount: number;
  /** Cost per thousand VIEWED exposures. The one derived ratio, from cost and exposures. */
  effectiveCpm: number | null;
  /** Screen counts by `screen_type`, resolved through the candidate rows. */
  screenTypeBreakdown: { label: string; count: number }[];
  optimizationMethod: string;
}

export function packageMetrics(
  pkg: OptimizedPackage,
  spec: CampaignSpec,
  candidates: ScreenCandidate[] = [],
): PackageMetrics {
  const screenIds = new Set(pkg.allocations.map((a) => a.screen_id));
  const typeById = new Map(candidates.map((c) => [c.screen_id, c.screen_type]));

  const counts = new Map<string, number>();
  for (const id of screenIds) {
    const type = typeById.get(id);
    if (!type) continue; // Unknown type: omitted rather than bucketed as "Other".
    counts.set(type, (counts.get(type) ?? 0) + 1);
  }

  return {
    totalCost: pkg.total_cost,
    budget: spec.budget,
    budgetUtilization: pkg.budget_utilization,
    grossImpressionsViewed: pkg.gross_impressions_viewed,
    expectedReach: pkg.expected_reach,
    expectedFrequency: pkg.expected_frequency,
    screenCount: screenIds.size,
    allocationCount: pkg.allocations.length,
    effectiveCpm:
      pkg.gross_impressions_viewed > 0
        ? (pkg.total_cost / pkg.gross_impressions_viewed) * 1000
        : null,
    screenTypeBreakdown: [...counts.entries()]
      .map(([label, count]) => ({ label, count }))
      .sort((a, b) => b.count - a.count),
    optimizationMethod: pkg.optimization_method,
  };
}

/** `price x slots x days`, mirroring `Allocation.line_cost` on the backend. */
export function lineCost(allocation: Allocation): number {
  return allocation.price_per_slot_per_day * allocation.slots_per_day * allocation.duration_days;
}

// ------------------------------------------------------ per-time-block rollups

export interface TimeBlockRollup {
  id: string;
  label: string;
  isPeak: boolean;
  /** True when the optimizer bought into this block. */
  selected: boolean;
  allocations: number;
  screens: number;
  slotsPerDay: number;
  impressions: number;
  cost: number;
  /** Share of total package impressions, 0-1. Drives the D1 bar heights. */
  impressionShare: number;
  /**
   * Mean committed-slot fraction across the flight for this block's priced screens, 0-1.
   * This is real market occupancy from live bookings — how much of the inventory is
   * already sold. Not to be confused with audience volume.
   */
  marketOccupancy: number | null;
  /** Mean recommended price over the economics rows for this block, if priced. */
  meanPrice: number | null;
  /** Bought slots as a share of forecast available slots, 0-1, if both are known. */
  occupancy: number | null;
}

/**
 * One row per real time block, always all six, so the D1 chart and D3 grid keep a
 * stable shape whether or not the optimizer bought into a block.
 */
export function timeBlockRollups(
  allocations: Allocation[],
  economics: ScreenEconomics[] = [],
): TimeBlockRollup[] {
  const totalImpressions = allocations.reduce((sum, a) => sum + a.viewed_exposures, 0);

  return TIME_BLOCKS.map(({ id, label }) => {
    const blockAllocations = allocations.filter((a) => a.time_block_id === id);
    // Infeasible rows carry no pricing, so every aggregate below must exclude them.
    const blockEconomics = economics.filter((e) => e.time_block_id === id && e.feasible);

    const impressions = blockAllocations.reduce((sum, a) => sum + a.viewed_exposures, 0);
    const boughtSlots = blockAllocations.reduce((sum, a) => sum + a.slots_per_day, 0);

    // Capacity is only meaningful for the screens actually in the package.
    const packagedScreenIds = new Set(blockAllocations.map((a) => a.screen_id));
    const availableSlots = blockEconomics
      .filter((e) => packagedScreenIds.has(e.screen_id))
      .reduce((sum, e) => sum + e.max_slots_per_day, 0);

    return {
      id,
      label,
      isPeak: PEAK_BLOCK_IDS.has(id),
      selected: blockAllocations.length > 0,
      allocations: blockAllocations.length,
      screens: packagedScreenIds.size,
      slotsPerDay: boughtSlots,
      impressions,
      cost: blockAllocations.reduce((sum, a) => sum + lineCost(a), 0),
      impressionShare: totalImpressions > 0 ? impressions / totalImpressions : 0,
      marketOccupancy: mean(
        blockEconomics.map((e) => e.occupancy_rate).filter((v): v is number => v !== null),
      ),
      meanPrice: mean(
        blockEconomics.map((e) => e.pricing?.recommended_price ?? null).filter((v): v is number => v !== null),
      ),
      occupancy: availableSlots > 0 ? boughtSlots / availableSlots : null,
    };
  });
}

// -------------------------------------------------------------- price guardrail

export interface PriceGuardrail {
  floor: number;
  target: number;
  cap: number;
  /** Volume-weighted mean price the optimizer actually paid, if a package exists. */
  paid: number | null;
  /** Where `paid` sits between floor and cap, 0-1. Null when it lands outside. */
  paidPosition: number | null;
  screensPriced: number;
  meanBookingProbability: number | null;
}

/**
 * Mean floor/target/cap across the priced inventory — the band the optimizer had to
 * work inside, which is what the D3 gauge shows.
 */
export function priceGuardrail(
  economics: ScreenEconomics[],
  allocations: Allocation[] = [],
): PriceGuardrail | null {
  // Only feasible rows have a band; sold-out rows carry `pricing: null`.
  const priced = economics.filter((e) => e.feasible && e.pricing !== null);
  if (priced.length === 0) return null;

  const floor = mean(priced.map((e) => e.pricing!.floor));
  const target = mean(priced.map((e) => e.pricing!.target));
  const cap = mean(priced.map((e) => e.pricing!.cap));
  if (floor === null || target === null || cap === null) return null;

  const slots = allocations.reduce((sum, a) => sum + a.slots_per_day * a.duration_days, 0);
  const paid = slots > 0 ? allocations.reduce((sum, a) => sum + lineCost(a), 0) / slots : null;

  return {
    floor,
    target,
    cap,
    paid,
    paidPosition: paid !== null && cap > floor ? clamp01((paid - floor) / (cap - floor)) : null,
    screensPriced: new Set(priced.map((e) => e.screen_id)).size,
    meanBookingProbability: mean(priced.map((e) => e.pricing!.booking_probability)),
  };
}

// --------------------------------------------------- per-screen pricing (D3)

export interface PricingLine {
  screenId: string;
  place: string;
  screenType: string | null;
  timeBlockLabel: string;
  floor: number;
  target: number;
  cap: number;
  /** What the optimizer actually agreed per slot per day. */
  paid: number;
  /** Where `paid` sits between floor and cap, 0-1, clamped. */
  paidPosition: number;
  /** True when the demand premium carried the quote past the cap — the one legal case. */
  aboveCap: boolean;
  slotsPerDay: number;
  maxSlotsPerDay: number | null;
  lineCost: number;
  /** Plain-language reasons this screen is priced where it is. Never empty. */
  drivers: string[];
}

/**
 * Why each screen in the package costs what it costs, one line per bought screen x block.
 *
 * This is the sales rep’s answer to "why is this one $103 and that one $75", so the
 * drivers are sentences rather than model fields. Nothing is computed here that the ML
 * agent did not already decide — the thresholds only choose which sentence to show.
 */
export function pricingLines(
  allocations: Allocation[],
  economics: ScreenEconomics[] = [],
  candidates: ScreenCandidate[] = [],
): PricingLine[] {
  const byKey = new Map(economics.map((e) => [`${e.screen_id}|${e.time_block_id}`, e]));
  const byScreen = new Map(candidates.map((c) => [c.screen_id, c]));

  return allocations
    .map((a) => {
      const row = byKey.get(`${a.screen_id}|${a.time_block_id}`);
      const band = row?.pricing ?? null;
      if (!band) return null;

      const paid = a.price_per_slot_per_day;
      const span = band.cap - band.floor;

      return {
        screenId: a.screen_id,
        place: placeLabel(byScreen.get(a.screen_id)),
        screenType: byScreen.get(a.screen_id)?.screen_type ?? null,
        timeBlockLabel: timeBlockLabel(a.time_block_id),
        floor: band.floor,
        target: band.target,
        cap: band.cap,
        paid,
        paidPosition: span > 0 ? clamp01((paid - band.floor) / span) : 0.5,
        aboveCap: paid > band.cap,
        slotsPerDay: a.slots_per_day,
        maxSlotsPerDay: row?.max_slots_per_day ?? null,
        lineCost: lineCost(a),
        drivers: priceDrivers(row, band.floor, band.cap, paid),
      } satisfies PricingLine;
    })
    .filter((line): line is PricingLine => line !== null)
    .sort((a, b) => b.lineCost - a.lineCost);
}

/**
 * The two or three sentences that actually explain a quote.
 *
 * Deliberately not every model field: occupancy is what moves the price inside the band,
 * seasonality and the demand premium are the only two things that move the band itself,
 * and everything else is diagnostics the rep cannot act on.
 */
function priceDrivers(
  row: ScreenEconomics | undefined,
  floor: number,
  cap: number,
  paid: number,
): string[] {
  const drivers: string[] = [];

  const occupancy = row?.occupancy_rate ?? null;
  if (occupancy !== null) {
    const pct = Math.round(occupancy * 100);
    drivers.push(
      occupancy >= 0.66
        ? `In demand — ${pct}% of this screen’s slots are already sold for these dates, so it prices near the top of its range.`
        : occupancy <= 0.33
          ? `Wide open — only ${pct}% of this screen’s slots are sold for these dates, so it prices near the bottom of its range.`
          : `Moderately booked — ${pct}% of this screen’s slots are sold for these dates, which puts it mid-range.`,
    );
  }

  const seasonality = row?.seasonality_multiplier ?? null;
  if (seasonality !== null && Math.abs(seasonality - 1) >= 0.01) {
    const delta = Math.round((seasonality - 1) * 100);
    drivers.push(
      delta > 0
        ? `Timing premium of ${delta}% — these dates run hotter than this screen’s average.`
        : `Timing discount of ${Math.abs(delta)}% — these dates run quieter than this screen’s average.`,
    );
  }

  const premium = row?.demand_premium ?? null;
  if (premium !== null && premium > 1.001) {
    drivers.push(
      `Under-priced for its audience — it delivers more than comparable screens have been charging, so a ${Math.round((premium - 1) * 100)}% uplift applies.`,
    );
  }

  drivers.push(
    paid > cap
      ? `Agreed at ${paid.toFixed(2)} per slot per day, above the ${cap.toFixed(0)} ceiling for comparable screens — the under-pricing correction above is the reason.`
      : `Agreed at ${paid.toFixed(2)} per slot per day, inside the ${floor.toFixed(0)}–${cap.toFixed(0)} range comparable screens have sold for.`,
  );

  return drivers;
}

// ----------------------------------------------------------- rotation loop (D4)

export interface RotationRow {
  screenId: string;
  timeBlockId: string;
  timeBlockLabel: string;
  slotsPerDay: number;
  maxSlotsPerDay: number | null;
  /** Which of the six loop slots are bought. Index 0 = Slot 1. */
  slots: boolean[];
  pricePerSlotPerDay: number;
  lineCost: number;
  expectedImpressions: number;
  relevanceScore: number | null;
}

/**
 * Rotation-loop allocation per line. `slots_per_day` is a count, so the first N of the
 * six loop slots are marked — the optimizer does not name specific slots.
 */
export function rotationRows(
  allocations: Allocation[],
  economics: ScreenEconomics[] = [],
  candidates: ScreenCandidate[] = [],
): RotationRow[] {
  const capacity = new Map(
    economics.map((e) => [`${e.screen_id}|${e.time_block_id}`, e.max_slots_per_day]),
  );
  const relevance = new Map(candidates.map((c) => [c.screen_id, c.relevance_score]));

  return allocations
    .map((a) => ({
      screenId: a.screen_id,
      timeBlockId: a.time_block_id,
      timeBlockLabel: timeBlockLabel(a.time_block_id),
      slotsPerDay: a.slots_per_day,
      maxSlotsPerDay: capacity.get(`${a.screen_id}|${a.time_block_id}`) ?? null,
      slots: Array.from({ length: ROTATION_LOOP_SLOTS }, (_, i) => i < a.slots_per_day),
      pricePerSlotPerDay: a.price_per_slot_per_day,
      lineCost: lineCost(a),
      expectedImpressions: a.viewed_exposures,
      relevanceScore: relevance.get(a.screen_id) ?? null,
    }))
    .sort((a, b) => b.expectedImpressions - a.expectedImpressions);
}

// ----------------------------------------------------------------- provenance

export interface ProvenanceInfo {
  provenance: Provenance;
  stubStages: string[];
  /** Set only when a stage was a stub. Surfaced verbatim in the UI. */
  note: string | null;
}

export function provenanceInfo(run: RunRecord | null): ProvenanceInfo {
  const stubStages = run?.stub_stages ?? [];
  if (stubStages.length === 0) {
    return { provenance: "computed", stubStages: [], note: null };
  }
  return {
    provenance: "stub",
    stubStages,
    note:
      `Illustrative only — ${stubStages.join(", ")} came from unimplemented ` +
      `specialist stubs. The numbers are deterministic placeholders derived from ` +
      `screen IDs, not analysis of the data.`,
  };
}

/** Geography as the spec resolved it, for the D1 header. */
export function geographyLabel(spec: CampaignSpec): string {
  const parts = [...spec.zone_ids, ...spec.corridor_ids, ...spec.city_ids];
  return parts.length > 0 ? parts.join(", ") : "Unresolved";
}

/** Audience target as a short human phrase. Omits anything intake left null. */
export function audienceLabel(spec: CampaignSpec): string {
  const target = spec.target_audience;
  const parts: string[] = [];
  if (target.age_range) parts.push(`Ages ${target.age_range[0]}-${target.age_range[1]}`);
  if (target.income_range) {
    parts.push(`Income ${target.income_range[0]}-${target.income_range[1]}`);
  }
  if (target.commuter) parts.push("Commuters");
  if (target.occupations.length > 0) parts.push(target.occupations.join(", "));
  return parts.length > 0 ? parts.join(" • ") : "No audience filter specified";
}

// -------------------------------------------------------------------- helpers

function mean(values: number[]): number | null {
  const finite = values.filter((v) => Number.isFinite(v));
  if (finite.length === 0) return null;
  return finite.reduce((sum, v) => sum + v, 0) / finite.length;
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}
