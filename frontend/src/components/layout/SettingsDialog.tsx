"use client";

/**
 * The Settings modal behind the sidebar's gear button. One setting so far: which model
 * runs the orchestration.
 *
 * The choice is the rep's rather than a deploy-time env var because the two providers fail
 * differently. Gemini's free tier allows ~20 requests/day/model and one orchestration
 * costs 15-20 of them, so a demo runs out of Gemini before it runs out of questions —
 * having to edit `.env` and restart uvicorn mid-demo is the thing this avoids.
 *
 * Providers with no credentials are shown disabled with the reason attached, not hidden:
 * "AZURE_OPENAI_API_KEY is not set" is actionable, a silently absent option is not.
 */

import { useEffect, useRef } from "react";

import { CheckIcon, SettingsIcon, SpinnerIcon, WarningIcon } from "@/components/ui/Icon";
import type { ModelSelection, ModelsOut, ProviderInfo } from "@/lib/types";

export function SettingsDialog({
  open,
  models,
  selection,
  loadError,
  onSelect,
  onClose,
}: {
  open: boolean;
  /** Null until `GET /models` answers. */
  models: ModelsOut | null;
  /** Null while the catalogue is still loading, or when running the backend default. */
  selection: ModelSelection | null;
  loadError: string | null;
  onSelect: (selection: ModelSelection) => void;
  onClose: () => void;
}) {
  const panelRef = useRef<HTMLDivElement>(null);

  // Escape closes. Registered only while open so the handler is not a permanent listener
  // competing with the prompt bar's own keys.
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  useEffect(() => {
    if (open) panelRef.current?.focus();
  }, [open]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-900/25 p-6 backdrop-blur-[2px]"
      // Backdrop click closes; the guard keeps a click inside the panel from bubbling out.
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label="Settings"
        tabIndex={-1}
        className="w-full max-w-lg overflow-hidden rounded-2xl border border-zinc-200/70 bg-white shadow-xl outline-none"
      >
        <header className="flex items-center justify-between border-b border-zinc-100 px-5 py-3.5">
          <div className="flex items-center gap-2">
            <SettingsIcon className="h-4 w-4 text-zinc-400" />
            <h2 className="text-xs font-bold tracking-tight text-zinc-800">Settings</h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg px-2 py-1 text-[11px] font-medium text-zinc-400 transition-colors hover:bg-zinc-50 hover:text-zinc-700"
          >
            Close
          </button>
        </header>

        <div className="max-h-[70vh] space-y-3 overflow-y-auto px-5 py-4">
          <div>
            <div className="text-[10px] font-bold tracking-wider text-zinc-400 uppercase">
              Orchestration Model
            </div>
            <p className="mt-1 text-[11px] leading-relaxed text-zinc-500">
              Runs the Master Agent and both specialists. Applies from the next turn — a package
              already built is not re-priced by switching.
            </p>
          </div>

          {loadError ? (
            <div className="flex items-start gap-2 rounded-xl border border-amber-200/70 bg-amber-50/70 p-3 text-[11px] leading-relaxed font-medium text-amber-800">
              <WarningIcon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>Could not load the model list. {loadError}</span>
            </div>
          ) : !models ? (
            <div className="flex items-center gap-2 px-1 py-4 text-[11px] font-medium text-zinc-400">
              <SpinnerIcon className="h-3.5 w-3.5" />
              Loading models…
            </div>
          ) : (
            models.providers.map((provider) => (
              <ProviderBlock
                key={provider.id}
                provider={provider}
                selection={selection}
                isBackendDefault={provider.id === models.default_provider}
                onSelect={onSelect}
              />
            ))
          )}
        </div>
      </div>
    </div>
  );
}

function ProviderBlock({
  provider,
  selection,
  isBackendDefault,
  onSelect,
}: {
  provider: ProviderInfo;
  selection: ModelSelection | null;
  isBackendDefault: boolean;
  onSelect: (selection: ModelSelection) => void;
}) {
  return (
    <section className="rounded-xl border border-zinc-100 bg-zinc-50/60 p-3">
      <div className="flex items-baseline justify-between gap-2">
        <div className="text-[11px] font-semibold text-zinc-700">{provider.label}</div>
        <div className="flex items-center gap-1.5 text-[10px] font-medium text-zinc-400">
          {isBackendDefault ? <span>server default</span> : null}
          {/* The cap is why a run takes ~90s on Gemini and far less on Azure, so it is
              worth showing next to the choice rather than hiding in a log line. */}
          <span>{provider.requests_per_minute}/min cap</span>
        </div>
      </div>

      {provider.configured ? (
        <div className="mt-2 space-y-1">
          {provider.models.map((option) => {
            const active = selection?.provider === provider.id && selection?.model === option.id;
            return (
              <button
                key={option.id}
                type="button"
                onClick={() => onSelect({ provider: provider.id, model: option.id })}
                aria-pressed={active}
                className={`flex w-full items-center justify-between gap-2 rounded-lg border px-2.5 py-2 text-left transition-colors ${
                  active
                    ? "border-violet-950/20 bg-violet-950 text-white"
                    : "border-transparent bg-white text-zinc-600 hover:border-zinc-200/70 hover:text-zinc-900"
                }`}
              >
                <span className="min-w-0">
                  <span className="block truncate text-[11px] font-semibold">{option.label}</span>
                  {option.description ? (
                    <span
                      className={`block truncate text-[10px] ${
                        active ? "text-violet-300" : "text-zinc-400"
                      }`}
                      title={option.description}
                    >
                      {option.description}
                    </span>
                  ) : null}
                </span>
                {active ? <CheckIcon className="h-3.5 w-3.5 shrink-0 text-violet-300" /> : null}
              </button>
            );
          })}
        </div>
      ) : (
        <div className="mt-2 flex items-start gap-2 rounded-lg bg-white px-2.5 py-2 text-[10px] leading-relaxed font-medium text-amber-700">
          <WarningIcon className="mt-0.5 h-3 w-3 shrink-0" />
          <span>{provider.unconfigured_reason}</span>
        </div>
      )}
    </section>
  );
}
