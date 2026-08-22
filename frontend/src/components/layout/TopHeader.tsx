"use client";

/**
 * Centre-panel header: campaign title, orchestration status, Reset and Export.
 * Per UI.md §2 Panel 2.
 */

import { DownloadDocIcon, SpinnerIcon } from "@/components/ui/Icon";
import type { RunStatus } from "@/hooks/useCampaignRun";
import type { Provenance } from "@/lib/types";

export function TopHeader({
  title,
  status,
  provenance,
  hasPackage,
  onReset,
  onExport,
}: {
  title: string;
  status: RunStatus;
  provenance: Provenance;
  hasPackage: boolean;
  onReset: () => void;
  onExport: () => void;
}) {
  return (
    <header className="sticky top-0 z-10 flex h-14 items-center justify-between border-b border-zinc-100 bg-white px-6">
      <div className="flex min-w-0 items-center gap-3">
        <h1
          className="truncate text-[13px] font-semibold tracking-tight text-zinc-900"
          title={title}
        >
          {title}
        </h1>
        <StatusBadge status={status} hasPackage={hasPackage} />
        {provenance === "stub" && hasPackage ? (
          <span className="flex shrink-0 items-center gap-1 rounded-full border border-amber-200/70 bg-amber-50 px-2 py-0.5 text-[10px] font-semibold text-amber-700">
            Illustrative
          </span>
        ) : null}
      </div>

      <div className="flex shrink-0 items-center gap-2">
        <button
          type="button"
          onClick={onReset}
          className="rounded-lg px-3 py-1.5 text-xs font-medium text-zinc-500 transition-colors hover:text-zinc-800"
        >
          Reset
        </button>
        <button
          type="button"
          onClick={onExport}
          disabled={!hasPackage}
          className="flex items-center gap-1.5 rounded-lg border border-zinc-800 bg-zinc-50 px-3.5 py-1.5 text-xs font-semibold text-zinc-800 shadow-xs transition-colors hover:bg-zinc-100 disabled:cursor-not-allowed disabled:border-zinc-200 disabled:bg-white disabled:text-zinc-300"
        >
          <DownloadDocIcon className="h-3.5 w-3.5" />
          <span>Export Proposal PDF</span>
        </button>
      </div>
    </header>
  );
}

function StatusBadge({ status, hasPackage }: { status: RunStatus; hasPackage: boolean }) {
  if (status === "streaming") {
    return (
      <span className="flex shrink-0 items-center gap-1 rounded-full border border-violet-200/70 bg-violet-50 px-2 py-0.5 text-[10px] font-semibold text-violet-950">
        <SpinnerIcon className="h-2.5 w-2.5 animate-spin" />
        Orchestrating
      </span>
    );
  }
  if (status === "error") {
    return (
      <span className="flex shrink-0 items-center gap-1 rounded-full border border-red-200/60 bg-red-50 px-2 py-0.5 text-[10px] font-semibold text-red-700">
        <span className="h-1.5 w-1.5 rounded-full bg-red-600" />
        Failed
      </span>
    );
  }

  const label = hasPackage ? "Orchestrated" : "Ready";
  return (
    <span className="flex shrink-0 items-center gap-1 rounded-full border border-zinc-200/60 bg-zinc-100/70 px-2 py-0.5 text-[10px] font-semibold text-zinc-600">
      <span className={`h-1.5 w-1.5 rounded-full ${hasPackage ? "bg-violet-950" : "bg-zinc-400"}`} />
      {label}
    </span>
  );
}
