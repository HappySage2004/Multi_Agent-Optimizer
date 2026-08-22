/**
 * Mirrors the backend Pydantic contracts. Change both together.
 *
 *   CampaignSpec / AudienceTarget  <- app/models/campaign.py
 *   ScreenCandidate                <- app/models/screens.py
 *   ScreenEconomics                <- app/models/economics.py
 *   OptimizationResult             <- app/models/optimization.py
 *   ValidationResult               <- app/models/recommendation.py
 *   ArtifactReference / Provenance <- app/models/artifacts.py
 *   Session / Upload / run wire    <- app/api/schemas.py
 */

/** `stub` = placeholder from an unimplemented specialist. `computed` = real analysis. */
export type Provenance = "stub" | "computed";

export type OptimizationGoal = "reach" | "frequency" | "awareness" | "conversion";
export type SolveStatus = "optimal" | "feasible" | "infeasible" | "error";
export type CheckStatus = "pass" | "fail" | "skipped";

// ---------------------------------------------------------------- campaign spec

export interface AudienceTarget {
  age_range: [number, number] | null;
  income_range: [number, number] | null;
  occupations: string[];
  commuter: boolean | null;
  other_attributes: Record<string, unknown>;
}

export interface CampaignSpec {
  campaign_objective: string;
  industry_vertical: string | null;
  ad_type: string | null;

  city_ids: string[];
  zone_ids: string[];
  corridor_ids: string[];

  target_audience: AudienceTarget;

  /** ISO date, e.g. "2026-10-01". */
  start_date: string;
  duration_days: number;
  budget: number;

  requested_num_screens: number | null;

  preferred_dayparts: string[];
  preferred_time_blocks: string[];

  optimization_goal: OptimizationGoal;

  hard_constraints: Record<string, unknown>;
  soft_preferences: Record<string, unknown>;

  original_query: string | null;
  /** Fields intake could not determine. Never silently invented. */
  missing_information: string[];
}

// -------------------------------------------------------------------- artifacts

export interface ArtifactReference {
  artifact_id: string;
  kind: string;
  path: string;
  rows: number;
  columns: string[];
  /** Aggregates only — never row-level data. */
  summary: Record<string, unknown>;
  provenance: Provenance;
  created_at: string;
}

/** The two artifact kinds the inspector reads rows from. */
export type ArtifactKind = "screen_candidates" | "screen_economics";

export interface ArtifactRowsOut<TRow> {
  run_id: string;
  kind: string;
  artifact_id: string;
  provenance: Provenance;
  total_rows: number;
  returned_rows: number;
  columns: string[];
  summary: Record<string, unknown>;
  rows: TRow[];
}

// ----------------------------------------------------------- screen candidates

/**
 * One row of the `screen_candidates` artifact. Drives the D2 Relevance tab.
 *
 * `relevance_score` is the weighted sum of exactly five components:
 *   0.40 audience + 0.20 geography + 0.15 contextual + 0.15 time_of_day
 * + 0.10 historical_performance.
 * `transit_score` is a reported volume percentile and is NOT in that sum.
 */
export interface ScreenCandidate {
  screen_id: string;
  relevance_score: number;

  audience_match_score: number;
  geography_score: number;
  contextual_score: number;
  /** Audience volume as a percentile of the eligible pool. Diagnostic, not weighted. */
  transit_score: number;
  time_of_day_score: number;
  historical_performance_score: number;

  /** Must cite real feature values — no generic "highly relevant" text. */
  reasons: string[];
  /** Sub-scores that fell back to a neutral default, and why. Empty is good. */
  defaults_applied: string[];
  hard_constraints_passed: boolean;

  /**
   * The physical-audience unit: location_id for stop-mounted screens, corridor_id for
   * vehicle-mounted ones. Screens sharing it see the SAME people, so never sum
   * impressions across a shared pool_key and call the result reach.
   */
  pool_key: string | null;
  /**
   * How many partitions the pool's audience was divided into for this screen's figure. 1
   * for stop-mounted screens; the vehicles working the corridor for vehicle-mounted ones.
   */
  pool_partition_count: number;
  /**
   * PEOPLE PASSING on a typical day, keyed `{block}_{weekday|weekend}` — all 12
   * combinations present. NOT viewed exposures: no viewability discount here, and a
   * whole-block daily figure, not per slot. Block 1 (00:00-04:00) is always 0: no
   * scheduled service starts then, which means "not modelled", not "nobody there".
   */
  impressions_by_block: Record<string, number>;
  impressions_weekday: number;
  impressions_weekend: number;

  city_id: string | null;
  zone_id: string | null;
  corridor_id: string | null;
  screen_type: string | null;
}

/** Aggregates the relevance engine puts on the `screen_candidates` reference. */
export interface ScreenCandidatesSummary {
  eligible_screens?: number;
  candidates?: number;
  relevance_min?: number;
  relevance_mean?: number;
  relevance_max?: number;
  /** Time blocks this campaign's audience is active in, e.g. ["2", "5"]. */
  preferred_time_blocks?: string[];
  day_type_focus?: string | null;
  audience_terms?: string[];
  distinct_audience_pools?: number;
  /** Deduplicated daily audience. Always well below the naive sum. */
  pooled_daily_audience?: number;
  naive_daily_audience?: number;
  demand_source?: string;
  defaults_applied?: string[];
}

// ------------------------------------------------------------ screen economics

export interface DemandForecastSummary {
  viewed_exposures_per_slot_per_day: number;
  demand_index: number;
  confidence: number;
}

export interface PricingRecommendation {
  floor: number;
  target: number;
  cap: number;
  recommended_price: number;
  booking_probability: number;
  confidence: number;
}

export interface TimeSlotAvailability {
  date: string;
  time_block_id: string;
  available_slots: number;
}

/**
 * One row of the `screen_economics` artifact -- one candidate screen x time block.
 * Drives the D3 Pricing tab.
 *
 * Rows with no purchasable slot are RETAINED with `feasible: false` and `pricing: null`,
 * so the UI can show what was excluded and why. Always branch on `feasible` before
 * reading `pricing`.
 */
export interface ScreenEconomics {
  screen_id: string;
  time_block_id: string;

  feasible: boolean;

  /** Empty by design: `max_slots_per_day` is the availability contract. */
  availability: TimeSlotAvailability[];
  /** Slots purchasable EVERY day of the flight -- the tightest single day, not a mean. */
  max_slots_per_day: number;
  /** Mean committed-slot fraction across the window, 0-1. */
  occupancy_rate: number | null;
  /** Absolute price at each slot count 1-6, null beyond availability. Flat by design. */
  price_by_slot_count: Record<string, number | null>;

  demand_forecast: DemandForecastSummary | null;
  /** null on infeasible rows. */
  pricing: PricingRecommendation | null;

  /**
   * VIEWED exposures one purchased slot earns on one day. A block is a 4-hour window in
   * which all 6 rotation slots cycle continuously, so holding k slots puts the creative on
   * k of every 6 loop passes and exposures are linear in k. Scales with slots x days.
   */
  viewed_exposures_per_slot_per_day: number;
  /**
   * Distinct people PASSING this screen's pool during this block on a typical day. Upstream
   * truth, carried for traceability — NOT the reach ceiling, because not everyone who
   * passes looks.
   */
  daily_unique_audience: number;
  /**
   * THE REACH CEILING: distinct people who look, = daily_unique_audience x viewability.
   * Does not scale with slots or days.
   */
  reachable_daily_audience: number;
  /** Share of passers-by assumed to look at this screen type. ASSUMED, no ground truth. */
  viewability_factor: number | null;
  /** Physical-audience unit; dedupe on this before reporting reach. */
  pool_key: string | null;
  expected_revenue: number;
  confidence: number;

  seasonality_multiplier: number | null;
  /** location_match | zone_match | none | not_applicable */
  event_match_type: string | null;
  /**
   * NOT client-facing and NOT campaign reach. Pricing-internal heuristic with mismatched
   * fixed/mobile units. Do not render this as an audience figure.
   */
  pricing_internal_reach_proxy: number | null;
  reach_owner: string;
  /** Which fallbacks fired and which adjustments applied, for this row. */
  assumptions: string[];
}

/** Aggregates the ML Agent puts on the `screen_economics` reference. */
export interface ScreenEconomicsSummary {
  rows?: number;
  feasible_rows?: number;
  screens_priced?: number;
  time_blocks?: string[];
  price_min?: number;
  price_mean?: number;
  price_max?: number;
  occupancy_mean?: number;
  booking_probability_mean?: number;
  viewed_exposures_per_slot_per_day_mean?: number;
  reachable_daily_audience_total_naive?: number;
  /** Which model produced the audience volume on this artifact. */
  demand_model?: string;
}

// ----------------------------------------------------------------- optimization

export interface Allocation {
  screen_id: string;
  time_block_id: string;
  slots_per_day: number;
  duration_days: number;
  price_per_slot_per_day: number;
  /** Gross VIEWED exposures over the flight. Exposures, not people. */
  viewed_exposures: number;
  expected_revenue: number;
}

export interface OptimizedPackage {
  allocations: Allocation[];
  total_cost: number;
  /** Total VIEWED exposures. Internal — never render this as a number of people. */
  gross_impressions_viewed: number;
  /**
   * DISTINCT PEOPLE, deduplicated by (pool_key, time block) and capped at each pool's
   * reachable daily audience. Saturates. This is the client-facing audience figure.
   */
  expected_reach: number;
  expected_frequency: number;
  budget_utilization: number;
  constraint_status: Record<string, boolean>;
  objective_value: number;
  optimization_method: string;

  /**
   * Reach under the solver's saturation curve, with an ASSUMED constant. Comparison only —
   * `expected_reach` is the definition this system stands behind.
   */
  curve_reach_diagnostic?: number | null;
  /** Coverage groups the plan could not satisfy, and by how much. Report it. */
  unmet_coverage?: Record<string, number>;
  /** Viewed exposures beyond the advisory wear-out cap. Non-zero is expected on long flights. */
  wear_out_exposures_over_cap?: number;
}

export interface InfeasibilityReport {
  status: "infeasible" | "error";
  reason_codes: string[];
  explanation: string;
  relaxation_options: string[];
}

/** Exactly one of `package` / `infeasibility` is set. */
export interface OptimizationResult {
  status: SolveStatus;
  package: OptimizedPackage | null;
  infeasibility: InfeasibilityReport | null;
  solver_log: string[];
}

// ------------------------------------------------------------------ validation

export interface ValidationCheck {
  name: string;
  status: CheckStatus;
  detail: string;
  expected: string | null;
  observed: string | null;
}

export interface ValidationResult {
  passed: boolean;
  checks: ValidationCheck[];
}

// ------------------------------------------------------------- runs & sessions

export interface Session {
  id: string;
  title: string;
  created_at: string | null;
  updated_at: string | null;
}

export interface Upload {
  id: string;
  session_id: string;
  filename: string;
  content_type: string | null;
  size_bytes: number;
  stored_path: string;
}

/** Full record from GET /runs/{run_id} — localDB/runs.json. */
export interface RunRecord {
  id: string;
  created_at: string;
  session_id: string | null;
  status: string;
  campaign_spec: CampaignSpec;
  artifacts: Partial<Record<ArtifactKind, ArtifactReference>>;
  optimization: OptimizationResult | null;
  validation: ValidationResult | null;
  stub_stages: string[];
}

/** Compact view from GET /runs and the SSE `done` event. */
export interface RunSnapshot {
  run_id: string;
  status: string;
  artifacts: Record<string, { artifact_id: string; rows: number; provenance: Provenance }>;
  optimization_status: string | null;
  validated: boolean;
  stub_stages: string[];
}

export interface TokenUsage {
  [key: string]: unknown;
}

export interface CampaignRunOut {
  session_id: string | null;
  run_id: string | null;
  answer: string;
  stub_stages: string[];
  provenance: Provenance;
  run_state: RunSnapshot | null;
  token_usage: TokenUsage | null;
}

export interface HealthOut {
  status: string;
  tables: number;
  ridership_actuals_provisioned: boolean;
  gemini_api_key_configured: boolean;
  master_model: string;
  specialist_model: string;
}

// ------------------------------------------------------------------ SSE events

/** `event: update` — one graph node produced a delta. Progress only, no content. */
export interface StreamUpdateEvent {
  node: string;
  summary: {
    messages?: number;
    tool_calls?: (string | null)[];
    tool_result_for?: string | null;
  };
}

/** `event: done` — terminal success. */
export interface StreamDoneEvent {
  session_id: string;
  /** The session's title after this run named it from the brief. */
  session_title: string | null;
  run_id: string | null;
  answer: string;
  run_state: RunSnapshot | null;
  token_usage: TokenUsage | null;
}

/** `event: error` — the run aborted; `detail` is safe to show the user. */
export interface StreamErrorEvent {
  status: number;
  detail: string;
}
