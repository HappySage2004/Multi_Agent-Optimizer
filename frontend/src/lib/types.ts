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

/** One row of the `screen_candidates` artifact. Drives the D2 Relevance tab. */
export interface ScreenCandidate {
  screen_id: string;
  relevance_score: number;

  audience_match_score: number;
  geography_score: number;
  contextual_score: number;
  transit_score: number;

  /** Must cite real feature values — no generic "highly relevant" text. */
  reasons: string[];
  hard_constraints_passed: boolean;

  city_id: string | null;
  zone_id: string | null;
  screen_type: string | null;
}

/** Aggregates the Data Agent puts on the `screen_candidates` reference. */
export interface ScreenCandidatesSummary {
  eligible_screens?: number;
  candidates?: number;
  relevance_min?: number;
  relevance_mean?: number;
  relevance_max?: number;
}

// ------------------------------------------------------------ screen economics

export interface DemandForecastSummary {
  expected_impressions: number;
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

/** One row of the `screen_economics` artifact. Drives the D3 Pricing tab. */
export interface ScreenEconomics {
  screen_id: string;
  time_block_id: string;

  availability: TimeSlotAvailability[];
  max_slots_per_day: number;

  demand_forecast: DemandForecastSummary;
  pricing: PricingRecommendation;

  /** Per slot per day, so the optimizer can scale by slots x duration. */
  expected_impressions: number;
  expected_revenue: number;
  confidence: number;
}

/** Aggregates the ML Agent puts on the `screen_economics` reference. */
export interface ScreenEconomicsSummary {
  rows?: number;
  screens?: number;
  time_blocks?: string[];
  price_mean?: number;
  impressions_per_slot_day_mean?: number;
  confidence_min?: number;
}

// ----------------------------------------------------------------- optimization

export interface Allocation {
  screen_id: string;
  time_block_id: string;
  slots_per_day: number;
  duration_days: number;
  price_per_slot_per_day: number;
  expected_impressions: number;
  expected_revenue: number;
}

export interface OptimizedPackage {
  allocations: Allocation[];
  total_cost: number;
  expected_impressions: number;
  expected_reach: number;
  expected_frequency: number;
  budget_utilization: number;
  constraint_status: Record<string, boolean>;
  objective_value: number;
  optimization_method: string;
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
