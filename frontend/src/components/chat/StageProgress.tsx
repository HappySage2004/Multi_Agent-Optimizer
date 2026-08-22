"use client";

/**
 * Live pipeline progress while the orchestration streams.
 *
 * A paced run takes ~90s because of the shared Gemini rate limiter, so the user needs to
 * see which stage is running rather than an undifferentiated spinner.
 */

import { CheckIcon, SpinnerIcon } from "@/components/ui/Icon";
import type { StageState } from "@/hooks/useCampaignRun";
import { STAGES } from "@/lib/stages";

export function StageProgress({
  stages,
  toolTrail,
  onCancel,
}: {
  stages: StageState[];
  toolTrail: string[];
  onCancel: () => void;
}) {
  const lastTool = toolTrail.at(-1);

  return (
    <div className="flex max-w-3xl gap-3">
      <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-zinc-800 bg-zinc-50 text-[11px] font-bold text-zinc-800 shadow-xs">
        AI
      </div>

      <div className="w-full space-y-3 rounded-xl border border-zinc-200/50 bg-zinc-50/60 p-4">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-bold text-zinc-700">Orchestrating pipeline</span>
          <button
            type="button"
            onClick={onCancel}
            className="text-[10px] font-semibold text-zinc-400 transition-colors hover:text-zinc-700"
          >
            Cancel
          </button>
        </div>

        <ol className="space-y-1.5">
          {stages.map((stage, index) => {
            const definition = STAGES[index];
            return (
              <li key={stage.id} className="flex items-center gap-2 text-[11px]">
                <StageMarker status={stage.status} />
                <span
                  className={
                    stage.status === "active"
                      ? "font-semibold text-zinc-800"
                      : stage.status === "complete"
                        ? "text-zinc-500"
                        : "text-zinc-400"
                  }
                >
                  {definition.label}
                </span>
                <span className="text-[10px] text-zinc-400">{definition.owner}</span>
              </li>
            );
          })}
        </ol>

        {lastTool ? (
          <p className="border-t border-zinc-200/60 pt-2 text-[10px] text-zinc-400">
            Last tool call: <span className="font-mono text-zinc-500">{lastTool}</span>
            {toolTrail.length > 1 ? ` (${toolTrail.length} total)` : ""}
          </p>
        ) : null}
      </div>
    </div>
  );
}

function StageMarker({ status }: { status: StageState["status"] }) {
  if (status === "complete") {
    return (
      <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-emerald-50 text-emerald-700">
        <CheckIcon className="h-2.5 w-2.5" strokeWidth={3} />
      </span>
    );
  }
  if (status === "active") {
    return (
      <span className="flex h-4 w-4 shrink-0 items-center justify-center text-violet-950">
        <SpinnerIcon className="h-3.5 w-3.5 animate-spin" />
      </span>
    );
  }
  return (
    <span className="flex h-4 w-4 shrink-0 items-center justify-center">
      <span className="h-1.5 w-1.5 rounded-full bg-zinc-300" />
    </span>
  );
}
