"use client";

/**
 * Left panel: brand, New Campaign, session history, inventory status, Settings.
 * Per UI.md §2 Panel 1.
 */

import { ChatIcon, PlusIcon, SettingsIcon, TrashIcon } from "@/components/ui/Icon";
import { formatNumber } from "@/lib/format";
import type { HealthOut, Session } from "@/lib/types";

export function Sidebar({
  sessions,
  activeSessionId,
  health,
  onNewCampaign,
  onSelectSession,
  onDeleteSession,
}: {
  sessions: Session[];
  activeSessionId: string | null;
  health: HealthOut | null;
  onNewCampaign: () => void;
  onSelectSession: (sessionId: string) => void;
  onDeleteSession: (sessionId: string) => void;
}) {
  return (
    <aside className="flex h-full flex-col justify-between border-r border-zinc-200/60 bg-white p-3.5">
      <div className="flex flex-1 flex-col space-y-5 overflow-hidden">
        <div className="flex items-center gap-2.5 px-1 pt-1">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-violet-950 text-xs font-bold tracking-wider text-white shadow-xs">
            AQ
          </div>
          <div>
            <div className="text-xs font-bold tracking-tight text-zinc-800">AgentIQ</div>
            <div className="text-[10px] font-medium text-zinc-400">Urban Media Engine</div>
          </div>
        </div>

        <button
          type="button"
          onClick={onNewCampaign}
          className="flex w-full shrink-0 items-center justify-center gap-2 rounded-xl bg-violet-950 px-3 py-2 text-xs font-semibold text-white shadow-xs transition-colors hover:bg-violet-900"
        >
          <PlusIcon className="h-3.5 w-3.5 text-violet-300" />
          <span>New Campaign</span>
        </button>

        <div className="flex-1 space-y-1 overflow-y-auto pr-1">
          <div className="mb-1.5 px-2 text-[10px] font-bold tracking-wider text-zinc-400 uppercase">
            Previous Sessions
          </div>

          {sessions.length === 0 ? (
            <p className="px-2 text-[11px] leading-relaxed text-zinc-400">
              No sessions yet. Start a campaign to create one.
            </p>
          ) : (
            sessions.map((session) => (
              <SessionRow
                key={session.id}
                session={session}
                active={session.id === activeSessionId}
                onSelect={() => onSelectSession(session.id)}
                onDelete={() => onDeleteSession(session.id)}
              />
            ))
          )}
        </div>
      </div>

      <div className="shrink-0 space-y-1.5 border-t border-zinc-100 pt-3">
        <InventoryStatus health={health} />

        <button
          type="button"
          className="flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-xs font-medium text-zinc-500 transition-colors hover:bg-zinc-50 hover:text-zinc-800"
        >
          <SettingsIcon className="h-4 w-4 text-zinc-400" />
          <span>Settings</span>
        </button>
      </div>
    </aside>
  );
}

function SessionRow({
  session,
  active,
  onSelect,
  onDelete,
}: {
  session: Session;
  active: boolean;
  onSelect: () => void;
  onDelete: () => void;
}) {
  return (
    <div
      className={`group flex items-center gap-1 rounded-lg transition-colors ${
        active
          ? "border border-zinc-200/50 bg-zinc-100/70 text-zinc-800"
          : "border border-transparent text-zinc-500 hover:bg-zinc-50 hover:text-zinc-800"
      }`}
    >
      <button
        type="button"
        onClick={onSelect}
        className="flex min-w-0 flex-1 items-center gap-2 p-2 text-left"
      >
        <ChatIcon
          className={`h-3.5 w-3.5 shrink-0 ${active ? "text-violet-950" : "text-zinc-400"}`}
        />
        <span className={`truncate text-[11px] ${active ? "font-semibold" : ""}`}>
          {session.title}
        </span>
      </button>

      <button
        type="button"
        onClick={onDelete}
        aria-label={`Delete session ${session.title}`}
        title="Delete session"
        className="mr-1.5 rounded p-1 text-zinc-300 opacity-0 transition-opacity group-hover:opacity-100 hover:text-zinc-600 focus:opacity-100"
      >
        <TrashIcon className="h-3 w-3" />
      </button>
    </div>
  );
}

/**
 * Reports what /health actually returns. The screen total is the real inventory count
 * from the reference layer; the dot only pulses when the backend answered.
 */
function InventoryStatus({ health }: { health: HealthOut | null }) {
  const online = health !== null;
  return (
    <div className="space-y-1 rounded-xl border border-zinc-100 bg-zinc-50/80 p-2.5 text-[10px]">
      <div className="flex items-center justify-between">
        <span className="font-medium text-zinc-400">Active Inventory</span>
        {online ? (
          <span className="flex items-center gap-1 font-semibold text-emerald-700">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-600" />
            {formatNumber(SCREEN_COUNT)}
          </span>
        ) : (
          <span className="flex items-center gap-1 font-semibold text-zinc-400">
            <span className="h-1.5 w-1.5 rounded-full bg-zinc-300" />
            Offline
          </span>
        )}
      </div>

      {online ? (
        <div className="flex items-center justify-between">
          <span className="font-medium text-zinc-400">Tables</span>
          <span className="font-semibold text-zinc-600">{health.tables}</span>
        </div>
      ) : null}

      {online && !health.gemini_api_key_configured ? (
        <div className="pt-0.5 leading-snug font-medium text-amber-700">
          GEMINI_API_KEY not configured — agent endpoints return 503.
        </div>
      ) : null}

      {online && !health.ridership_actuals_provisioned ? (
        <div className="pt-0.5 leading-snug font-medium text-amber-700">
          ridership_actuals.csv not provisioned.
        </div>
      ) : null}
    </div>
  );
}

/**
 * Screens in the reference layer (`screen_facts()`), per DATASETS.md. /health reports
 * table count rather than row counts, so this is the documented constant, not a guess.
 */
const SCREEN_COUNT = 11_163;
