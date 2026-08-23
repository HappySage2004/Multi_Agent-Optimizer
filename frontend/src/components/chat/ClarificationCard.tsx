"use client";

/**
 * The pre-flight clarification gate, rendered as selectable options.
 *
 * Shown once per brief, before the pipeline runs, when the agent found a gap big enough to
 * make the ranking meaningless — a brief with no audience and no industry leaves 0.80 of the
 * relevance weight sitting at a flat 0.5. Answering takes a few clicks; not answering costs
 * a 90-second run whose headline claim is a constant.
 *
 * Three things the layout is deliberately doing:
 *
 * 1. `understood` leads, so the rep sees the questions are narrow gaps rather than a request
 *    to start over.
 * 2. The defer option carries the agent's actual recommendation in its own text, so picking
 *    it is an informed choice. The backend guarantees that text exists.
 * 3. "Build it with your best guesses" is always available and never disabled. A rep in a
 *    hurry must be able to leave with one click, which is the whole reason this is a card
 *    and not a blocking modal.
 */

import { useMemo, useState } from "react";

import { ArrowRightIcon, CheckIcon } from "@/components/ui/Icon";
import type { ClarificationRequest, ClarifyingOption, ClarifyingQuestion } from "@/lib/types";

/** What the rep picked for one question. `custom` carries their typed text. */
interface Selection {
  key: string;
  customText: string;
}

export function ClarificationCard({
  request,
  disabled,
  onSubmit,
}: {
  request: ClarificationRequest;
  /** True while a run is streaming — the card stays visible but stops accepting input. */
  disabled: boolean;
  /** Receives the composed reply text, sent as the next turn's message. */
  onSubmit: (reply: string) => void;
}) {
  const [selections, setSelections] = useState<Record<string, Selection>>({});

  const answered = request.questions.filter((q) => isResolved(q, selections[q.id])).length;
  const allAnswered = answered === request.questions.length;

  const summary = useMemo(
    () => composeReply(request.questions, selections),
    [request.questions, selections],
  );

  function choose(questionId: string, key: string) {
    setSelections((prev) => ({
      ...prev,
      [questionId]: { key, customText: prev[questionId]?.customText ?? "" },
    }));
  }

  function setCustomText(questionId: string, text: string) {
    setSelections((prev) => ({
      ...prev,
      // Typing implies choosing the custom option, so a rep who tabs straight into the
      // field never ends up with text that is silently ignored.
      [questionId]: { key: prev[questionId]?.key ?? "D", customText: text },
    }));
  }

  return (
    <div className="space-y-3 rounded-xl border border-violet-200/70 bg-violet-50/40 p-4">
      <div className="space-y-1.5">
        <span className="text-[10px] font-bold tracking-wider text-violet-900 uppercase">
          Before I build this
        </span>
        <p className="text-[11px] leading-relaxed text-zinc-600">{request.understood}</p>
      </div>

      <div className="space-y-3">
        {request.questions.map((question, index) => (
          <QuestionBlock
            key={question.id}
            index={index + 1}
            total={request.questions.length}
            question={question}
            selection={selections[question.id]}
            disabled={disabled}
            onChoose={(key) => choose(question.id, key)}
            onCustomText={(text) => setCustomText(question.id, text)}
          />
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-violet-200/60 pt-3">
        <button
          type="button"
          disabled={disabled || !allAnswered}
          onClick={() => onSubmit(summary)}
          className="flex cursor-pointer items-center gap-1.5 rounded-lg bg-violet-950 px-3 py-1.5 text-[11px] font-semibold text-zinc-50 shadow-xs hover:bg-violet-900 disabled:cursor-not-allowed disabled:bg-zinc-300 disabled:text-zinc-500"
        >
          Build the package
          <ArrowRightIcon className="h-3 w-3" />
        </button>

        {/* Never disabled by the answer count: leaving in one click is the point. */}
        <button
          type="button"
          disabled={disabled}
          onClick={() => onSubmit(JUST_BUILD_IT)}
          className="cursor-pointer rounded-lg border border-violet-200 bg-white px-3 py-1.5 text-[11px] font-semibold text-violet-950 hover:bg-violet-50 disabled:cursor-not-allowed disabled:text-zinc-400"
        >
          Build it with your best guesses
        </button>

        <span className="text-[10px] text-zinc-400">
          {allAnswered
            ? "All set."
            : `${answered} of ${request.questions.length} answered`}
        </span>
      </div>
    </div>
  );
}

function QuestionBlock({
  index,
  total,
  question,
  selection,
  disabled,
  onChoose,
  onCustomText,
}: {
  index: number;
  total: number;
  question: ClarifyingQuestion;
  selection: Selection | undefined;
  disabled: boolean;
  onChoose: (key: string) => void;
  onCustomText: (text: string) => void;
}) {
  const custom = question.options.find((o) => o.kind === "custom");
  const showTextField = custom && selection?.key === custom.key;

  return (
    <fieldset className="space-y-1.5" disabled={disabled}>
      <legend className="text-[11px] font-semibold text-zinc-700">
        {total > 1 ? `${index}. ` : ""}
        {question.question}
      </legend>

      <div className="grid gap-1.5 sm:grid-cols-2">
        {question.options.map((option) => (
          <OptionButton
            key={option.key}
            option={option}
            selected={selection?.key === option.key}
            disabled={disabled}
            onClick={() => onChoose(option.key)}
          />
        ))}
      </div>

      {showTextField ? (
        <input
          type="text"
          autoFocus
          value={selection?.customText ?? ""}
          onChange={(event) => onCustomText(event.target.value)}
          placeholder="Type your answer…"
          aria-label={`Your own answer for: ${question.question}`}
          className="w-full rounded-lg border border-violet-300 bg-white px-2.5 py-1.5 text-[11px] text-zinc-700 placeholder:text-zinc-400 focus:border-violet-500 focus:outline-none"
        />
      ) : null}
    </fieldset>
  );
}

function OptionButton({
  option,
  selected,
  disabled,
  onClick,
}: {
  option: ClarifyingOption;
  selected: boolean;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={selected}
      className={`flex cursor-pointer gap-2 rounded-lg border p-2 text-left transition-colors disabled:cursor-not-allowed ${
        selected
          ? "border-violet-500 bg-white shadow-xs"
          : "border-zinc-200/70 bg-white/60 hover:border-violet-300 hover:bg-white"
      }`}
    >
      <span
        className={`mt-px flex h-4 w-4 shrink-0 items-center justify-center rounded border text-[9px] font-bold ${
          selected
            ? "border-violet-500 bg-violet-950 text-zinc-50"
            : "border-zinc-300 bg-white text-zinc-400"
        }`}
      >
        {selected ? <CheckIcon className="h-2.5 w-2.5" /> : option.key}
      </span>
      <span className="min-w-0 space-y-0.5">
        <span className="block text-[11px] font-semibold text-zinc-700">{option.label}</span>
        {option.detail ? (
          <span className="block text-[10px] leading-relaxed text-zinc-500">{option.detail}</span>
        ) : null}
      </span>
    </button>
  );
}

// --------------------------------------------------------------------- reply text

export const JUST_BUILD_IT =
  "Just build it — use your recommendation for everything you asked about.";

function isResolved(question: ClarifyingQuestion, selection: Selection | undefined): boolean {
  if (!selection) return false;
  const option = question.options.find((o) => o.key === selection.key);
  if (!option) return false;
  // A custom pick with an empty box is not an answer.
  return option.kind === "custom" ? selection.customText.trim().length > 0 : true;
}

/**
 * The selections as a sentence per question, sent as the next user message.
 *
 * Deliberately prose rather than JSON: it goes into the transcript the rep reads back, and
 * into the agent's own message history. `field` is included because that is the token the
 * agent matches against the question it asked.
 */
function composeReply(
  questions: ClarifyingQuestion[],
  selections: Record<string, Selection>,
): string {
  const lines = questions.map((question) => {
    const selection = selections[question.id];
    const option = question.options.find((o) => o.key === selection?.key);

    if (!option) return `- ${question.field}: no preference — use your recommendation.`;
    if (option.kind === "custom") {
      return `- ${question.field}: ${selection?.customText.trim()}`;
    }
    if (option.kind === "defer") {
      return `- ${question.field}: your call — go with your recommendation.`;
    }
    return `- ${question.field}: ${option.value ?? option.label}`;
  });

  return ["Answering your questions:", ...lines].join("\n");
}
