"use client";

/**
 * D4: Impressions & Rotation Loop Optimizer (UI.md §2 Panel 3).
 *
 * The rotation matrix marks `slots_per_day` of the six loop slots per line — the
 * optimizer allocates a slot *count*, not named slots, so the first N are marked and the
 * caption says so. Below it: the constraint status, solver log and the Master Agent's
 * validation checks, since a package the validator rejected must never look clean.
 */

import {
  AwaitingStage,
  InspectorCard,
  InspectorSection,
} from "@/components/inspector/InspectorShell";
import { CheckIcon, WarningIcon } from "@/components/ui/Icon";
import { ROTATION_LOOP_SLOTS, type RotationRow } from "@/lib/derive";
import { formatCompact, formatCurrency, formatNumber, formatPercent, titleCase } from "@/lib/format";
import type { OptimizationResult, ValidationResult } from "@/lib/types";

export function TabOptimizerD4({
  optimization,
  validation,
  rows,
}: {
  optimization: OptimizationResult | null;
  validation: ValidationResult | null;
  rows: RotationRow[];
}) {
  if (!optimization) {
    return (
      <AwaitingStage
        stage="the OR Agent (stage 4)"
        detail="The optimizer produces the allocation package this panel reads."
      />
    );
  }

  const pkg = optimization.package;
  const statusTone =
    optimization.status === "optimal" || optimization.status === "feasible"
      ? "active"
      : "warning";

  return (
    <>
      <InspectorCard
        title="Reach Optimizer"
        badge={titleCase(optimization.status)}
        badgeTone={statusTone}
        description="Multi-screen, multi-slot allocation under the budget, inventory and date constraints."
      />

      {optimization.infeasibility ? (
        <InspectorSection title="Infeasible">
          <p className="text-[13px] leading-relaxed text-red-700">
            {optimization.infeasibility.explanation}
          </p>
          {optimization.infeasibility.reason_codes.length > 0 ? (
            <div className="flex flex-wrap gap-1.5">
              {optimization.infeasibility.reason_codes.map((code) => (
                <span
                  key={code}
                  className="rounded border border-red-200 bg-red-50 px-2 py-0.5 font-mono text-[12px] font-semibold text-red-700"
                >
                  {code}
                </span>
              ))}
            </div>
          ) : null}
        </InspectorSection>
      ) : null}

      {pkg ? (
        <>
          <InspectorSection title="Package Totals" meta={pkg.optimization_method}>
            <div className="grid grid-cols-2 gap-2 text-[13px]">
              <Stat label="Spend" value={formatCurrency(pkg.total_cost)} />
              <Stat label="Budget used" value={formatPercent(pkg.budget_utilization)} />
              <Stat
                label="Viewed exposures"
                value={formatCompact(pkg.gross_impressions_viewed)}
              />
              <Stat label="Reach" value={formatCompact(pkg.expected_reach)} />
              <Stat label="Frequency" value={pkg.expected_frequency.toFixed(2)} />
              <Stat label="Lines" value={formatNumber(pkg.allocations.length)} />
            </div>
          </InspectorSection>

          <ConstraintStatus status={pkg.constraint_status} />

          <RotationMatrix rows={rows} />

          {optimization.solver_log.length > 0 ? (
            <InspectorSection title="Solver Log">
              <pre className="overflow-x-auto rounded-lg bg-zinc-50 p-2.5 font-mono text-[12px] leading-relaxed text-zinc-500">
                {optimization.solver_log.join("\n")}
              </pre>
            </InspectorSection>
          ) : null}
        </>
      ) : null}

      <ValidationPanel validation={validation} />
    </>
  );
}

/** Hard constraints are enforced in code; this reports what the checker found. */
function ConstraintStatus({ status }: { status: Record<string, boolean> }) {
  const entries = Object.entries(status);
  if (entries.length === 0) return null;

  return (
    <InspectorSection title="Hard Constraints">
      <div className="flex flex-wrap gap-1.5">
        {entries.map(([name, passed]) => (
          <span
            key={name}
            className={`flex items-center gap-1 rounded-md px-2 py-1 text-[12px] font-medium ${
              passed
                ? "border border-emerald-200/60 bg-emerald-50 text-emerald-700"
                : "border border-red-200/60 bg-red-50 text-red-700"
            }`}
          >
            {passed ? (
              <CheckIcon className="h-2.5 w-2.5" strokeWidth={3} />
            ) : (
              <WarningIcon className="h-2.5 w-2.5" />
            )}
            {titleCase(name)}
          </span>
        ))}
      </div>
    </InspectorSection>
  );
}

function RotationMatrix({ rows }: { rows: RotationRow[] }) {
  if (rows.length === 0) {
    return (
      <InspectorSection title={`Rotation Loop Allocation (${ROTATION_LOOP_SLOTS} Slots/Loop)`}>
        <p className="text-[12px] text-zinc-400">No allocations in this package.</p>
      </InspectorSection>
    );
  }

  return (
    <InspectorSection
      title={`Rotation Loop Allocation (${ROTATION_LOOP_SLOTS} Slots/Loop)`}
      meta={`${rows.length} ${rows.length === 1 ? "line" : "lines"}`}
    >
      <div className="space-y-2.5">
        {rows.map((row) => (
          <div key={`${row.screenId}-${row.timeBlockId}`} className="space-y-1.5">
            <div className="flex items-baseline justify-between gap-2 text-[12px]">
              <span className="truncate font-semibold text-zinc-700">{row.screenId}</span>
              <span className="shrink-0 font-medium text-zinc-400">{row.timeBlockLabel}</span>
            </div>

            <div className="grid grid-cols-6 gap-1 text-center text-[12px] font-bold">
              {row.slots.map((active, index) => (
                <div
                  key={index}
                  className={`rounded p-1.5 ${
                    active
                      ? "bg-violet-950 text-white shadow-xs"
                      : index < (row.maxSlotsPerDay ?? ROTATION_LOOP_SLOTS)
                        ? "bg-zinc-100/70 text-zinc-400"
                        : "bg-zinc-50 text-zinc-300"
                  }`}
                  title={
                    active
                      ? `Slot ${index + 1} — bought`
                      : index < (row.maxSlotsPerDay ?? ROTATION_LOOP_SLOTS)
                        ? `Slot ${index + 1} — available, not bought`
                        : `Slot ${index + 1} — beyond this screen's daily capacity`
                  }
                >
                  {index + 1}
                </div>
              ))}
            </div>

            <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-zinc-400">
              <span>
                {row.slotsPerDay}
                {row.maxSlotsPerDay !== null ? `/${row.maxSlotsPerDay}` : ""} slots/day
              </span>
              <span>{formatCurrency(row.pricePerSlotPerDay, 2)}/slot/day</span>
              <span>{formatCurrency(row.lineCost)} line</span>
              <span>{formatCompact(row.expectedImpressions)} impr</span>
              {row.relevanceScore !== null ? (
                <span>{(row.relevanceScore * 100).toFixed(1)}% fit</span>
              ) : null}
            </div>
          </div>
        ))}
      </div>

      <p className="border-t border-zinc-100 pt-2 text-[11px] leading-relaxed text-zinc-400">
        The optimizer allocates a slot <em>count</em> per screen and time block, not named
        slots, so the first N of the loop are marked. Faded cells are beyond that
        screen&rsquo;s forecast daily capacity.
      </p>
    </InspectorSection>
  );
}

/** The Master Agent validates every specialist output before answering. */
function ValidationPanel({ validation }: { validation: ValidationResult | null }) {
  if (!validation) {
    return (
      <InspectorSection title="Validation">
        <p className="text-[12px] text-zinc-400">
          Not verified yet — the Master Agent runs stage 5 after optimization.
        </p>
      </InspectorSection>
    );
  }

  const failures = validation.checks.filter((c) => c.status === "fail");

  return (
    <InspectorSection
      title="Validation"
      meta={
        validation.passed
          ? `${validation.checks.length} checks passed`
          : `${failures.length} of ${validation.checks.length} FAILED`
      }
    >
      <div className="space-y-1.5">
        {validation.checks.map((check) => (
          <div key={check.name} className="flex gap-2 text-[12px]">
            <span className="mt-px shrink-0">
              {check.status === "pass" ? (
                <CheckIcon className="h-3 w-3 text-emerald-600" strokeWidth={3} />
              ) : check.status === "fail" ? (
                <WarningIcon className="h-3 w-3 text-red-600" />
              ) : (
                <span className="block h-3 w-3 text-center leading-3 text-zinc-300">–</span>
              )}
            </span>
            <div className="min-w-0">
              <span
                className={`font-mono font-semibold ${
                  check.status === "fail" ? "text-red-700" : "text-zinc-600"
                }`}
              >
                {check.name}
              </span>
              <p className="leading-relaxed text-zinc-400">{check.detail}</p>
              {check.status === "fail" && check.expected ? (
                <p className="text-red-600">
                  expected {check.expected}, observed {check.observed}
                </p>
              ) : null}
            </div>
          </div>
        ))}
      </div>
    </InspectorSection>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-zinc-50/80 px-2.5 py-2">
      <span className="block text-[11px] font-medium tracking-wide text-zinc-400 uppercase">
        {label}
      </span>
      <span className="font-semibold text-zinc-700">{value}</span>
    </div>
  );
}
