"use client";

/**
 * The chat stream from UI.md §2: right-aligned user bubbles, left-aligned AI response
 * blocks, with the metrics deck and strategy card rendered inline under the answer that
 * produced them.
 */

import { useEffect, useRef } from "react";

import { ClarificationCard } from "@/components/chat/ClarificationCard";
import { ImpactMetricsDeck } from "@/components/chat/ImpactMetricsDeck";
import { Markdown } from "@/components/chat/Markdown";
import { PackageTable } from "@/components/chat/PackageTable";
import { StageProgress, ThinkingIndicator } from "@/components/chat/StageProgress";
import { StrategySummaryCard } from "@/components/chat/StrategySummaryCard";
import { PaperclipIcon, WarningIcon } from "@/components/ui/Icon";
import type { ChatMessage, RunData, RunStatus, StageState } from "@/hooks/useCampaignRun";
import { packageMetrics, provenanceInfo, timeBlockRollups } from "@/lib/derive";
import type { ClarificationRequest } from "@/lib/types";

export function ChatFeed({
  messages,
  runData,
  status,
  stages,
  toolTrail,
  error,
  pendingQuestions,
  onCancel,
  onOpenInspector,
  onDismissError,
  onAnswerClarification,
}: {
  messages: ChatMessage[];
  runData: RunData;
  status: RunStatus;
  stages: StageState[];
  toolTrail: string[];
  error: string | null;
  pendingQuestions: ClarificationRequest | null;
  onCancel: () => void;
  onOpenInspector: () => void;
  onDismissError: () => void;
  onAnswerClarification: (reply: string) => void;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Follow the stream as stages land and the answer arrives.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, status, stages, pendingQuestions]);

  return (
    <div className="flex-1 space-y-6 overflow-y-auto p-6 pb-28">
      {messages.length === 0 && status !== "streaming" ? <EmptyState /> : null}

      {messages.map((message) =>
        message.role === "user" ? (
          <UserBubble key={message.id} message={message} />
        ) : (
          <AssistantBlock
            key={message.id}
            message={message}
            runData={runData}
            onOpenInspector={onOpenInspector}
          />
        ),
      )}

      {/* Anchored after the transcript rather than inside the message that asked: the
          round belongs to the session, and a rep who scrolls up should not lose it. */}
      {pendingQuestions ? (
        <div className="flex max-w-3xl gap-3">
          <div className="mt-0.5 h-7 w-7 shrink-0" aria-hidden />
          <div className="w-full">
            <ClarificationCard
              request={pendingQuestions}
              disabled={status === "streaming"}
              onSubmit={onAnswerClarification}
            />
          </div>
        </div>
      ) : null}

      {/* The stage rail only appears once the pipeline is genuinely underway; a follow-up
          question never enters it and gets the compact indicator instead. */}
      {status === "streaming" ? (
        stages.some((stage) => stage.status !== "pending") ? (
          <StageProgress stages={stages} toolTrail={toolTrail} onCancel={onCancel} />
        ) : (
          <ThinkingIndicator
            hasPackage={Boolean(runData.run?.optimization?.package)}
            onCancel={onCancel}
          />
        )
      ) : null}

      {error ? <ErrorBlock detail={error} onDismiss={onDismissError} /> : null}

      <div ref={bottomRef} />
    </div>
  );
}

function UserBubble({ message }: { message: ChatMessage }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-lg space-y-2 rounded-2xl rounded-tr-xs border border-violet-900 bg-violet-950 p-4 text-xs leading-relaxed text-zinc-100 shadow-xs">
        <p className="whitespace-pre-wrap">{message.text}</p>
        {message.attachments && message.attachments.length > 0 ? (
          <ul className="space-y-1 border-t border-violet-900/60 pt-2">
            {message.attachments.map((name) => (
              <li key={name} className="flex items-center gap-1.5 text-[10px] text-violet-200">
                <PaperclipIcon className="h-3 w-3 shrink-0" />
                <span className="truncate">{name}</span>
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    </div>
  );
}

function AssistantBlock({
  message,
  runData,
  onOpenInspector,
}: {
  message: ChatMessage;
  runData: RunData;
  onOpenInspector: () => void;
}) {
  // The deck belongs to the run this message reported on, not whatever is loaded now.
  const run = runData.run && runData.run.id === message.runId ? runData.run : null;
  const pkg = run?.optimization?.package ?? null;
  const infeasibility = run?.optimization?.infeasibility ?? null;
  const provenance = provenanceInfo(run);

  const metrics =
    run && pkg ? packageMetrics(pkg, run.campaign_spec, runData.packagedCandidates) : null;
  const rollups = pkg ? timeBlockRollups(pkg.allocations, runData.economics) : [];

  return (
    <div className="flex max-w-3xl gap-3">
      <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-zinc-800 bg-zinc-50 text-[11px] font-bold text-zinc-800 shadow-xs">
        AI
      </div>

      <div className="w-full space-y-4 text-xs leading-relaxed text-zinc-700">
        {/* Only a session that predates transcript storage lands here. Everything since
            restores its actual prose, so this note has to stay narrow rather than becoming
            the generic "this is old" label. */}
        {message.restored ? (
          <p className="rounded-lg border border-zinc-200/60 bg-zinc-50/70 px-3 py-2 text-[11px] text-zinc-500">
            Rebuilt from run history. This campaign ran before conversations were saved, so
            the package below is the stored result but the agent&rsquo;s written answer was
            never recorded.
          </p>
        ) : null}

        {/* First, above the prose: the package is what the rep came for. Absent on a
            plain enquiry, which answers off the existing run and carries no runId. */}
        {run && pkg ? (
          <PackageTable
            pkg={pkg}
            spec={run.campaign_spec}
            candidates={runData.packagedCandidates}
          />
        ) : null}

        {message.text ? <Markdown>{message.text}</Markdown> : null}

        {provenance.note ? (
          <div className="flex gap-2 rounded-lg border border-amber-200/70 bg-amber-50 px-3 py-2 text-[11px] text-amber-800">
            <WarningIcon className="mt-px h-3.5 w-3.5 shrink-0" />
            <p className="leading-relaxed">{provenance.note}</p>
          </div>
        ) : null}

        {infeasibility ? <InfeasibilityBlock report={infeasibility} /> : null}

        {metrics && run ? (
          <>
            <ImpactMetricsDeck metrics={metrics} />
            <StrategySummaryCard
              metrics={metrics}
              rollups={rollups}
              spec={run.campaign_spec}
              onOpenInspector={onOpenInspector}
            />
          </>
        ) : null}
      </div>
    </div>
  );
}

/**
 * Infeasibility is explicit, per SOLUTION.md §31: status, reason codes and relaxation
 * options — never a fabricated package.
 */
function InfeasibilityBlock({
  report,
}: {
  report: NonNullable<NonNullable<RunData["run"]>["optimization"]>["infeasibility"];
}) {
  if (!report) return null;
  return (
    <div className="space-y-2 rounded-xl border border-red-200/60 bg-red-50/70 p-4">
      <div className="flex items-center gap-1.5 text-[11px] font-bold text-red-800">
        <WarningIcon className="h-3.5 w-3.5" />
        No feasible package
      </div>
      <p className="text-[11px] leading-relaxed text-red-700">{report.explanation}</p>

      {report.reason_codes.length > 0 ? (
        <div className="flex flex-wrap gap-1.5">
          {report.reason_codes.map((code) => (
            <span
              key={code}
              className="rounded border border-red-200 bg-white px-2 py-0.5 font-mono text-[10px] font-semibold text-red-700"
            >
              {code}
            </span>
          ))}
        </div>
      ) : null}

      {report.relaxation_options.length > 0 ? (
        <div className="space-y-1 pt-1">
          <span className="text-[10px] font-bold tracking-wider text-red-700 uppercase">
            Relaxation options
          </span>
          <ul className="list-disc space-y-0.5 pl-4 text-[11px] text-red-700">
            {report.relaxation_options.map((option) => (
              <li key={option}>{option}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

function ErrorBlock({ detail, onDismiss }: { detail: string; onDismiss: () => void }) {
  return (
    <div className="flex max-w-3xl gap-3">
      <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-red-100 text-red-700">
        <WarningIcon className="h-4 w-4" />
      </div>
      <div className="w-full space-y-2 rounded-xl border border-red-200/60 bg-red-50/70 p-4">
        <div className="flex items-start justify-between gap-3">
          <span className="text-[11px] font-bold text-red-800">Run failed</span>
          <button
            type="button"
            onClick={onDismiss}
            className="text-[10px] font-semibold text-red-400 hover:text-red-700"
          >
            Dismiss
          </button>
        </div>
        <p className="text-[11px] leading-relaxed whitespace-pre-wrap text-red-700">{detail}</p>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex max-w-2xl flex-col gap-3 pt-8">
      <h2 className="text-sm font-bold text-zinc-700">Describe the campaign</h2>
      <p className="text-xs leading-relaxed text-zinc-500">
        Write the brief in plain language. Intake normalizes it into a campaign spec, then the
        Data, ML and OR agents build a costed media package against real inventory.
      </p>
      <div className="space-y-2 rounded-xl border border-zinc-200/50 bg-zinc-50/60 p-4">
        <span className="text-[10px] font-bold tracking-wider text-zinc-400 uppercase">
          Example brief
        </span>
        <p className="text-[11px] leading-relaxed text-zinc-500">
          &ldquo;I have $50,000 for a 30-day campaign starting 2026-10-01 targeting commuters
          aged 18-34 in the Downtown Core zone of Las Hackland. Consumer tech product launch.
          Optimize for reach.&rdquo;
        </p>
      </div>
      <p className="text-[10px] leading-relaxed text-zinc-400">
        A full orchestration takes about 90 seconds — the model calls are deliberately paced
        to stay inside the provider rate limit.
      </p>
    </div>
  );
}
