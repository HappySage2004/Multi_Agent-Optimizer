/**
 * The six pipeline stages from SOLUTION.md §3, and how to infer the live one from the
 * SSE `update` events.
 *
 * Two kinds of stage, resolved differently:
 *
 * Master-owned stages call their tool directly, so the tool name pins the stage exactly.
 * That now includes relevance scoring: `build_screen_candidates` is a deterministic engine
 * the Master invokes itself, not a delegation.
 *
 * Delegated stages run inside the Master's `task` tool, so the specialist's own tools
 * (`estimate_screen_economics`, `optimize_package`) are invisible in the stream. What we
 * can see is the *order* of `task` calls, and that order is guaranteed: stages are
 * strictly sequential, dependent tools refuse to run early via
 * `run_state.missing_prerequisite()`, and the Master's prompt forbids more than one `task`
 * call per turn. So the Nth `task` is the Nth specialist.
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
  { id: "candidates", label: "Screen candidates", owner: "Relevance engine" },
  { id: "economics", label: "Audience & pricing", owner: "ML Agent" },
  { id: "optimization", label: "Optimization", owner: "OR Agent" },
  { id: "verification", label: "Verification", owner: "Master" },
  { id: "recommendation", label: "Recommendation", owner: "Master" },
];

/**
 * The tool whose call means a new pipeline run has begun.
 *
 * Needed because most of the Master's tools are read-only and a follow-up turn calls them
 * to answer a question — `inspect_package` maps to the verification stage, but seeing it
 * on its own means the agent is reading an existing package, not building one.
 * `create_campaign_spec` is the only tool that creates a run, so it is the unambiguous
 * signal. Until it fires, the stage rail stays hidden.
 */
export const PIPELINE_ENTRY_TOOL = "create_campaign_spec";

/** Master-owned tools that pin a stage directly. */
const TOOL_STAGES: Record<string, StageId> = {
  resolve_geography_terms: "intake",
  create_campaign_spec: "intake",
  describe_inventory: "candidates",
  build_screen_candidates: "candidates",
  describe_relevance_model: "candidates",
  verify_package: "verification",
  inspect_package: "verification",
  check_explanations: "verification",
};

/** The delegated stages, in `task` call order. Relevance is no longer among them. */
const DELEGATION_ORDER: StageId[] = ["economics", "optimization"];

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
