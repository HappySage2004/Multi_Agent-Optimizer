"use client";

/**
 * The workspace state machine: sessions, the streaming orchestration, and the run data
 * every panel reads from.
 *
 * Chat transcripts are not persisted server-side — localDB stores sessions, runs and
 * uploads, but no message log. So a transcript lives in memory for the browser session,
 * and switching to an older session rehydrates what *is* durable: the brief from
 * `campaign_spec.original_query` plus the full package from the run record. The agent's
 * prose answer is not recoverable, and the restored message says so rather than
 * inventing one.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  ApiError,
  createSession,
  deleteSession,
  getArtifactRows,
  getHealth,
  getRun,
  listRuns,
  listSessions,
  stageUpload,
  streamCampaign,
} from "@/lib/api";
import {
  PIPELINE_ENTRY_TOOL,
  STAGES,
  type StageId,
  stageForTool,
  stageIndex,
} from "@/lib/stages";
import type {
  HealthOut,
  RunRecord,
  ScreenCandidate,
  ScreenEconomics,
  Session,
  TokenUsage,
  Upload,
} from "@/lib/types";

/**
 * The placeholder a session carries until a brief names it. The backend renames the
 * session when a run starts (`_ensure_session`), so this is what the client sees only
 * before the first brief.
 */
const DEFAULT_SESSION_TITLE = "New Campaign";

/** Ranked rows pulled for the D2 candidate list. */
const RANKED_ROW_LIMIT = 60;
/**
 * Cap for the screen-filtered pulls. A package holds at most `max_screens_in_package`
 * (120) screens, and economics rows are per screen *and* time block, so this leaves room
 * for the widest package the optimizer can return.
 */
const PACKAGED_ROW_LIMIT = 1000;

export type RunStatus = "idle" | "streaming" | "done" | "error";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  /** Prose. Empty for a restored assistant message, which has no recoverable answer. */
  text: string;
  /** Set on assistant messages that carry a package, so the deck renders inline. */
  runId?: string;
  /** True when rehydrated from a run record rather than streamed live. */
  restored?: boolean;
  /** Filenames attached to a user message. */
  attachments?: string[];
}

export interface StageState {
  id: StageId;
  status: "pending" | "active" | "complete";
}

export interface RunData {
  run: RunRecord | null;
  /** Top-ranked candidates, for the D2 list. A sample of the pool, in rank order. */
  candidates: ScreenCandidate[];
  /** Candidate rows for the screens actually bought, for the deck and D4. */
  packagedCandidates: ScreenCandidate[];
  /** Economics rows for the screens actually bought, for D3 and D4. */
  economics: ScreenEconomics[];
  /** True while the artifact rows behind the inspector are still loading. */
  loadingArtifacts: boolean;
}

const EMPTY_RUN_DATA: RunData = {
  run: null,
  candidates: [],
  packagedCandidates: [],
  economics: [],
  loadingArtifacts: false,
};

let messageSeq = 0;
function nextMessageId(): string {
  messageSeq += 1;
  return `msg-${messageSeq}`;
}

export function useCampaignRun() {
  const [health, setHealth] = useState<HealthOut | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  /** Transcript per session id, in memory only. */
  const [transcripts, setTranscripts] = useState<Record<string, ChatMessage[]>>({});
  const [runData, setRunData] = useState<RunData>(EMPTY_RUN_DATA);

  const [status, setStatus] = useState<RunStatus>("idle");
  const [activeStage, setActiveStage] = useState<StageId | null>(null);
  const [completedStages, setCompletedStages] = useState<StageId[]>([]);
  const [toolTrail, setToolTrail] = useState<string[]>([]);
  const [tokenUsage, setTokenUsage] = useState<TokenUsage | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [pendingUploads, setPendingUploads] = useState<Upload[]>([]);

  const abortRef = useRef<AbortController | null>(null);
  /** Sessions already rehydrated, so switching back does not refetch. */
  const hydratedRef = useRef<Set<string>>(new Set());

  const messages = useMemo(
    () => (activeSessionId ? transcripts[activeSessionId] ?? [] : []),
    [transcripts, activeSessionId],
  );

  const appendMessage = useCallback((sessionId: string, message: ChatMessage) => {
    setTranscripts((prev) => ({ ...prev, [sessionId]: [...(prev[sessionId] ?? []), message] }));
  }, []);

  // ------------------------------------------------------------------ bootstrap

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const [healthResult, sessionsResult] = await Promise.allSettled([
        getHealth(),
        listSessions(),
      ]);
      if (cancelled) return;

      if (healthResult.status === "fulfilled") setHealth(healthResult.value);
      if (sessionsResult.status === "fulfilled") {
        setSessions(sessionsResult.value);
      } else {
        setError(describeError(sessionsResult.reason));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // --------------------------------------------------------- artifacts & runs

  /**
   * Load the run record plus the artifact rows the inspector needs.
   *
   * Two different slices are needed. D2 wants the top of the ranked pool; the deck, D3
   * and D4 want the rows for the screens the optimizer actually bought, which are
   * scattered through the artifact and so have to be requested by id.
   */
  const loadRun = useCallback(async (runId: string) => {
    setRunData((prev) => ({ ...prev, loadingArtifacts: true }));
    try {
      const run = await getRun(runId);
      setRunData({ ...EMPTY_RUN_DATA, run, loadingArtifacts: true });

      const allocations = run.optimization?.package?.allocations ?? [];
      const packagedIds = [...new Set(allocations.map((a) => a.screen_id))];
      const hasCandidates = Boolean(run.artifacts.screen_candidates);
      const hasEconomics = Boolean(run.artifacts.screen_economics);

      // Artifacts are absent until their stage runs; a 404 here is expected, not an error.
      const [ranked, packaged, economics] = await Promise.all([
        hasCandidates
          ? getArtifactRows(runId, "screen_candidates", RANKED_ROW_LIMIT).catch(() => null)
          : null,
        hasCandidates && packagedIds.length > 0
          ? getArtifactRows(
              runId,
              "screen_candidates",
              PACKAGED_ROW_LIMIT,
              packagedIds,
            ).catch(() => null)
          : null,
        hasEconomics
          ? getArtifactRows(
              runId,
              "screen_economics",
              packagedIds.length > 0 ? PACKAGED_ROW_LIMIT : RANKED_ROW_LIMIT,
              packagedIds.length > 0 ? packagedIds : undefined,
            ).catch(() => null)
          : null,
      ]);

      setRunData({
        run,
        candidates: ranked?.rows ?? [],
        packagedCandidates: packaged?.rows ?? [],
        economics: economics?.rows ?? [],
        loadingArtifacts: false,
      });
    } catch (cause) {
      setRunData({ ...EMPTY_RUN_DATA });
      setError(describeError(cause));
    }
  }, []);

  /**
   * Rebuild what a past session left behind: its latest run's brief and package. The
   * prose answer was never persisted, so the restored assistant message carries none.
   */
  const hydrateSession = useCallback(
    async (sessionId: string) => {
      if (hydratedRef.current.has(sessionId)) return;
      hydratedRef.current.add(sessionId);
      try {
        const runs = await listRuns(sessionId);
        if (runs.length === 0) return;

        const latest = runs[runs.length - 1];
        const run = await getRun(latest.run_id);
        const restored: ChatMessage[] = [
          {
            id: nextMessageId(),
            role: "user",
            text: run.campaign_spec.original_query ?? run.campaign_spec.campaign_objective,
          },
          { id: nextMessageId(), role: "assistant", text: "", runId: run.id, restored: true },
        ];
        setTranscripts((prev) => ({ ...prev, [sessionId]: prev[sessionId] ?? restored }));
        await loadRun(run.id);
      } catch {
        // A session with no readable run just opens empty.
      }
    },
    [loadRun],
  );

  // ------------------------------------------------------------------ sessions

  const selectSession = useCallback(
    (sessionId: string) => {
      if (sessionId === activeSessionId) return;
      setActiveSessionId(sessionId);
      setRunData({ ...EMPTY_RUN_DATA });
      setPendingUploads([]);
      setError(null);
      setStatus("idle");
      setActiveStage(null);
      setCompletedStages([]);
      setToolTrail([]);
      void hydrateSession(sessionId);
    },
    [activeSessionId, hydrateSession],
  );

  const newCampaign = useCallback(async () => {
    try {
      const session = await createSession();
      setSessions((prev) => [session, ...prev]);
      hydratedRef.current.add(session.id); // Brand new: nothing to rehydrate.
      setActiveSessionId(session.id);
      setTranscripts((prev) => ({ ...prev, [session.id]: [] }));
      setRunData({ ...EMPTY_RUN_DATA });
      setPendingUploads([]);
      setError(null);
      setStatus("idle");
      setActiveStage(null);
      setCompletedStages([]);
      setToolTrail([]);
      return session;
    } catch (cause) {
      setError(describeError(cause));
      return null;
    }
  }, []);

  const removeSession = useCallback(
    async (sessionId: string) => {
      try {
        await deleteSession(sessionId);
        setSessions((prev) => prev.filter((s) => s.id !== sessionId));
        setTranscripts((prev) => {
          const next = { ...prev };
          delete next[sessionId];
          return next;
        });
        if (sessionId === activeSessionId) {
          setActiveSessionId(null);
          setRunData({ ...EMPTY_RUN_DATA });
        }
      } catch (cause) {
        setError(describeError(cause));
      }
    },
    [activeSessionId],
  );

  /** Clear the current session's transcript without deleting its server-side history. */
  const resetTranscript = useCallback(() => {
    if (!activeSessionId) return;
    setTranscripts((prev) => ({ ...prev, [activeSessionId]: [] }));
    setRunData({ ...EMPTY_RUN_DATA });
    setPendingUploads([]);
    setStatus("idle");
    setActiveStage(null);
    setCompletedStages([]);
    setToolTrail([]);
    setError(null);
  }, [activeSessionId]);

  // ------------------------------------------------------------------- uploads

  const attachFile = useCallback(
    async (file: File) => {
      let sessionId = activeSessionId;
      if (!sessionId) {
        const session = await newCampaign();
        if (!session) return;
        sessionId = session.id;
      }
      try {
        const upload = await stageUpload(sessionId, file);
        setPendingUploads((prev) => [...prev, upload]);
      } catch (cause) {
        setError(describeError(cause));
      }
    },
    [activeSessionId, newCampaign],
  );

  const removePendingUpload = useCallback((uploadId: string) => {
    setPendingUploads((prev) => prev.filter((u) => u.id !== uploadId));
  }, []);

  // ----------------------------------------------------------------- the run

  const submit = useCallback(
    async (query: string) => {
      const trimmed = query.trim();
      if (!trimmed || status === "streaming") return;

      let sessionId = activeSessionId;
      if (!sessionId) {
        const session = await newCampaign();
        if (!session) return;
        sessionId = session.id;
      }

      const attachments = pendingUploads;
      appendMessage(sessionId, {
        id: nextMessageId(),
        role: "user",
        text: trimmed,
        attachments: attachments.map((u) => u.filename),
      });

      setStatus("streaming");
      setError(null);
      // Deliberately not pre-set to "intake". A follow-up question never enters the
      // pipeline, so the stage rail must stay hidden until a run is actually created —
      // otherwise every question shows a tracker frozen on "Brief intake".
      setActiveStage(null);
      setCompletedStages([]);
      setToolTrail([]);
      setTokenUsage(null);
      setPendingUploads([]);

      const controller = new AbortController();
      abortRef.current = controller;

      // `task` calls are counted so the Nth delegation maps to the Nth specialist.
      let taskCalls = 0;
      // Flipped by the tool that creates a run. Until then this is a follow-up turn and
      // the read-only tools it calls must not drive the stage rail.
      let pipelineStarted = false;
      const seen = new Set<StageId>();

      const advanceTo = (stage: StageId) => {
        setActiveStage(stage);
        // Everything before the live stage is finished, by sequential construction.
        const upto = stageIndex(stage);
        for (const s of STAGES.slice(0, upto)) seen.add(s.id);
        setCompletedStages([...seen]);
      };

      try {
        await streamCampaign(
          { query: trimmed, session_id: sessionId, upload_ids: attachments.map((u) => u.id) },
          {
            onUpdate: (event) => {
              for (const name of event.summary.tool_calls ?? []) {
                if (!name) continue;
                setToolTrail((prev) => [...prev, name]);
                if (name === PIPELINE_ENTRY_TOOL) pipelineStarted = true;
                const stage = stageForTool(name, taskCalls);
                if (name === "task") taskCalls += 1;
                if (stage && pipelineStarted) advanceTo(stage);
              }
            },
            onDone: (event) => {
              // The backend named the session from this brief; adopt its title so the
              // sidebar and the header agree with what is on disk.
              if (event.session_title) {
                setSessions((prev) =>
                  prev.map((s) =>
                    s.id === sessionId ? { ...s, title: event.session_title as string } : s,
                  ),
                );
              }
              // The backend reports whether this turn rebuilt the package. A follow-up
              // resolves to the same run, so attaching it to the message would repeat the
              // metrics deck under every answer — and completing the stage rail would
              // claim a pipeline that never ran.
              const rebuilt = event.pipeline_ran !== false;
              if (rebuilt && pipelineStarted) {
                seen.add("verification");
                setCompletedStages([...seen, "recommendation"]);
              }
              setActiveStage(null);
              setTokenUsage(event.token_usage);
              setStatus("done");

              appendMessage(sessionId, {
                id: nextMessageId(),
                role: "assistant",
                text: event.answer,
                runId: rebuilt ? event.run_id ?? undefined : undefined,
              });
              // Only refetch when there is something new to fetch.
              if (rebuilt && event.run_id) void loadRun(event.run_id);
            },
            onError: (event) => {
              setStatus("error");
              setActiveStage(null);
              setError(event.detail);
            },
          },
          controller.signal,
        );
      } catch (cause) {
        if (!controller.signal.aborted) {
          setStatus("error");
          setActiveStage(null);
          setError(describeError(cause));
        }
      } finally {
        abortRef.current = null;
      }
    },
    [activeSessionId, appendMessage, loadRun, newCampaign, pendingUploads, status],
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setStatus("idle");
    setActiveStage(null);
  }, []);

  const stageStates: StageState[] = useMemo(
    () =>
      STAGES.map((stage) => ({
        id: stage.id,
        status:
          stage.id === activeStage
            ? "active"
            : completedStages.includes(stage.id)
              ? "complete"
              : "pending",
      })),
    [activeStage, completedStages],
  );

  /**
   * Sidebar titles. The backend names a session from the brief the moment a run starts,
   * but that name only reaches the client on the `done` event a minute or so later. Until
   * then a still-unnamed session borrows its first user message, which the row shortens
   * with CSS — so there is no second copy of the backend's title heuristic over here.
   */
  const displaySessions = useMemo(
    () =>
      sessions.map((session) => {
        if (session.title !== DEFAULT_SESSION_TITLE) return session;
        const brief = transcripts[session.id]?.find((m) => m.role === "user")?.text.trim();
        return brief ? { ...session, title: brief } : session;
      }),
    [sessions, transcripts],
  );

  const activeSession = useMemo(
    () => displaySessions.find((s) => s.id === activeSessionId) ?? null,
    [displaySessions, activeSessionId],
  );

  return {
    health,
    sessions: displaySessions,
    activeSession,
    activeSessionId,
    messages,
    runData,
    status,
    stageStates,
    activeStage,
    toolTrail,
    tokenUsage,
    error,
    pendingUploads,
    selectSession,
    newCampaign,
    removeSession,
    resetTranscript,
    attachFile,
    removePendingUpload,
    submit,
    cancel,
    dismissError: useCallback(() => setError(null), []),
  };
}

function describeError(cause: unknown): string {
  if (cause instanceof ApiError) {
    // 503 without a key and 429 on quota are the two the user will actually hit.
    return cause.status ? `${cause.status}: ${cause.message}` : cause.message;
  }
  return cause instanceof Error ? cause.message : "Unexpected error.";
}
