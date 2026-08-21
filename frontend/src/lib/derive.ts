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

/** Slots in one rotation loop. Matches the D4 six-slot matrix in UI.md. */
export const ROTATION_LOOP_SLOTS = 6;

// ------------------------------------------------------------- package metrics

export interface PackageMetrics {
  totalCost: number;
  budget: number;
  /** Optimizer's own figure, not recomputed. */
  budgetUtilization: number;
  expectedImpressions: number;
  expectedReach: number;
  expectedFrequency: number;
  screenCount: number;
  allocationCount: number;
  /** Cost per thousand impressions. The one derived ratio, from cost and impressions. */
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
    expectedImpressions: pkg.expected_impressions,
    expectedReach: pkg.expected_reach,
    expectedFrequency: pkg.expected_frequency,
    screenCount: screenIds.size,
    allocationCount: pkg.allocations.length,
    effectiveCpm:
      pkg.expected_impressions > 0 ? (pkg.total_cost / pkg.expected_impressions) * 1000 : null,
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
  /** Mean `demand_index` over the economics rows for this block, if priced. */
  demandIndex: number | null;
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
  const totalImpressions = allocations.reduce((sum, a) => sum + a.expected_impressions, 0);

  return TIME_BLOCKS.map(({ id, label }) => {
    const blockAllocations = allocations.filter((a) => a.time_block_id === id);
    const blockEconomics = economics.filter((e) => e.time_block_id === id);

    const impressions = blockAllocations.reduce((sum, a) => sum + a.expected_impressions, 0);
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
      demandIndex: mean(blockEconomics.map((e) => e.demand_forecast.demand_index)),
      meanPrice: mean(blockEconomics.map((e) => e.pricing.recommended_price)),
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
  if (economics.length === 0) return null;

  const floor = mean(economics.map((e) => e.pricing.floor));
  const target = mean(economics.map((e) => e.pricing.target));
  const cap = mean(economics.map((e) => e.pricing.cap));
  if (floor === null || target === null || cap === null) return null;

  const slots = allocations.reduce((sum, a) => sum + a.slots_per_day * a.duration_days, 0);
  const paid = slots > 0 ? allocations.reduce((sum, a) => sum + lineCost(a), 0) / slots : null;

  return {
    floor,
    target,
    cap,
    paid,
    paidPosition: paid !== null && cap > floor ? clamp01((paid - floor) / (cap - floor)) : null,
    screensPriced: new Set(economics.map((e) => e.screen_id)).size,
    meanBookingProbability: mean(economics.map((e) => e.pricing.booking_probability)),
  };
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
      expectedImpressions: a.expected_impressions,
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
