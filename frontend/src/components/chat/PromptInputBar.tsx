"use client";

/**
 * Floating prompt bar from UI.md §2: attachment button, text input, submit.
 * Fixed to `bottom-4 left-6 right-6` inside the centre panel.
 */

import { useRef, useState } from "react";

import { ArrowRightIcon, PaperclipIcon, SpinnerIcon } from "@/components/ui/Icon";
import type { Upload } from "@/lib/types";

/** Mirrors ALLOWED_SUFFIXES in backend/app/api/uploads.py. */
const ACCEPTED_FILE_TYPES = ".pdf,.txt,.md,.csv,.docx,.xlsx,.pptx";

export function PromptInputBar({
  busy,
  hasPackage,
  pendingUploads,
  onSubmit,
  onAttach,
  onRemoveUpload,
}: {
  busy: boolean;
  hasPackage: boolean;
  pendingUploads: Upload[];
  onSubmit: (query: string) => void;
  onAttach: (file: File) => void;
  onRemoveUpload: (uploadId: string) => void;
}) {
  const [value, setValue] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || busy) return;
    onSubmit(trimmed);
    setValue("");
  };

  return (
    <div className="absolute bottom-4 left-6 right-6">
      {pendingUploads.length > 0 ? (
        <div className="mb-2 flex flex-wrap gap-1.5">
          {pendingUploads.map((upload) => (
            <span
              key={upload.id}
              className="flex items-center gap-1.5 rounded-lg border border-zinc-200 bg-white px-2 py-1 text-[10px] font-medium text-zinc-600 shadow-xs"
            >
              <PaperclipIcon className="h-3 w-3 text-zinc-400" />
              <span className="max-w-[180px] truncate">{upload.filename}</span>
              <button
                type="button"
                onClick={() => onRemoveUpload(upload.id)}
                aria-label={`Remove ${upload.filename}`}
                className="text-zinc-300 hover:text-zinc-600"
              >
                &times;
              </button>
            </span>
          ))}
        </div>
      ) : null}

      <form
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
        className="flex items-center gap-2 rounded-2xl border border-zinc-200 bg-white p-2 shadow-lg shadow-zinc-100 transition-all focus-within:border-violet-950 focus-within:ring-2 focus-within:ring-violet-950/10"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPTED_FILE_TYPES}
          className="hidden"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onAttach(file);
            // Reset so re-picking the same file fires onChange again.
            event.target.value = "";
          }}
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          aria-label="Attach a campaign document"
          title="Attach a brief, RFP or client deck"
          className="rounded-xl p-2 text-zinc-400 transition-colors hover:text-zinc-600"
        >
          <PaperclipIcon className="h-4 w-4" />
        </button>

        <input
          type="text"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          disabled={busy}
          placeholder={
            hasPackage
              ? "Refine plan, e.g. 'Shift 15% budget from bus stops to inner metro coaches'..."
              : "Describe the campaign, e.g. '$50,000 over 30 days from 2026-10-01 targeting commuters in Downtown Core'..."
          }
          className="flex-1 bg-transparent px-1 text-xs text-zinc-700 outline-none placeholder:text-zinc-400 disabled:cursor-not-allowed"
        />

        <button
          type="submit"
          disabled={busy || value.trim().length === 0}
          className="flex items-center gap-1.5 rounded-xl bg-violet-950 p-2 px-4 text-xs font-semibold text-white shadow-xs transition-colors hover:bg-violet-900 disabled:cursor-not-allowed disabled:bg-zinc-300"
        >
          {busy ? (
            <>
              <SpinnerIcon className="h-3.5 w-3.5 animate-spin" />
              <span>Running</span>
            </>
          ) : (
            <>
              <span>{hasPackage ? "Refine" : "Plan"}</span>
              <ArrowRightIcon className="h-3.5 w-3.5 text-violet-300" />
            </>
          )}
        </button>
      </form>
    </div>
  );
}
