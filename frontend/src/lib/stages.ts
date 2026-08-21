/**
 * The six pipeline stages from SOLUTION.md §3, and how to infer the live one from the
 * SSE `update` events.
 *
 * The stream reports top-level graph nodes only, so a specialist's own tools
 * (`build_screen_candidates`, `estimate_screen_economics`, `optimize_package`) are
 * invisible — they run inside the Master's `task` tool. What we can see is the *order* of
 * `task` calls, and that order is guaranteed: stages are strictly sequential, dependent
 * tools refuse to run early via `run_state.missing_prerequisite()`, and the Master's
 * prompt forbids more than one `task` call per turn. So the Nth `task` is the Nth
 * specialist.
 */

export type StageId = "intake" | "candidates" | "economics" | "optimization" | "verification" | "recommendation";

export interface StageDefinition {
  id: StageId;
  /** Short label for the progress rail. */
  label: string;
  /** Which agent owns the stage. */
  owner: string;
}

export const STAGES: StageDefinition[] = [
  { id: "intake", label: "Brief intake", owner: "Master" },
  { id: "candidates", label: "Screen candidates", owner: "Data Agent" },
  { id: "economics", label: "Demand & pricing", owner: "ML Agent" },
  { id: "optimization", label: "Optimization", owner: "OR Agent" },
  { id: "verification", label: "Verification", owner: "Master" },
  { id: "recommendation", label: "Recommendation", owner: "Master" },
];

/** Master-owned tools that pin a stage directly. */
const TOOL_STAGES: Record<string, StageId> = {
  resolve_geography_terms: "intake",
  create_campaign_spec: "intake",
  verify_package: "verification",
  inspect_package: "verification",
  check_explanations: "verification",
};

/** The specialist stages, in delegation order. */
const DELEGATION_ORDER: StageId[] = ["candidates", "economics", "optimization"];

export function stageForTool(toolName: string, taskCallIndex: number): StageId | null {
  if (toolName === "task") {
    return DELEGATION_ORDER[Math.min(taskCallIndex, DELEGATION_ORDER.length - 1)];
  }
  return TOOL_STAGES[toolName] ?? null;
}

export function stageIndex(stage: StageId): number {
  return STAGES.findIndex((s) => s.id === stage);
}

/** Which artifact kind, if any, proves a stage actually produced output. */
export const STAGE_ARTIFACT: Partial<Record<StageId, string>> = {
  candidates: "screen_candidates",
  economics: "screen_economics",
};
