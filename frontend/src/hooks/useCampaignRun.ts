"use client";

/**
 * The workspace state machine: sessions, the streaming orchestration, and the run data
 * every panel reads from.
 *
 * Transcripts are persisted server-side (localDB/messages.json, written by the campaign
 * endpoints), so switching session or reloading the page restores the actual conversation
 * — prose and all — rather than a placeholder. Sessions that predate that storage have no
 * messages, so they still fall back to reconstructing the brief from
 * `campaign_spec.original_query`; that path is the only one that shows a package with no
 * answer, and it says so.
 *
 * Run data is cached **per session**. It used to live in a single slot that
 * `selectSession` cleared, while `hydratedRef` suppressed the refetch — so the first visit
 * to a session filled the inspector and every visit after it showed empty D1-D4 tabs. The
 * cache is what makes the "already hydrated" guard correct.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  ApiError,
  clearMessages,
  createSession,
  deleteSession,
  getArtifactRows,
  getClarification,
  getHealth,
  getRun,
  listMessages,
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
  ArtifactKind,
  ChatMessageRecord,
  ClarificationRequest,
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
  /** Prose. Empty only on a legacy turn reconstructed from a run record. */
  text: string;
  /** Set on assistant messages that carry a package, so the deck renders inline. */
  runId?: string;
  /**
   * True when reconstructed from a run record because no transcript was stored — the one
   * case where the answer genuinely cannot be recovered.
   */
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
  /**
   * Why an artifact the run *claims* to have could not be read, keyed by kind.
   *
   * A missing reference means the stage never ran, which the tabs already handle. This is
   * the other case: the run record says 250 rows and the fetch failed anyway — usually a
   * 410, because runs.json was committed while backend/artifacts/ is gitignored. That used
   * to be swallowed into an empty array, so the tab claimed rows it was not showing.
   */
  artifactErrors: Partial<Record<ArtifactKind, string>>;
}

const EMPTY_RUN_DATA: RunData = {
  run: null,
  candidates: [],
  packagedCandidates: [],
  economics: [],
  loadingArtifacts: false,
  artifactErrors: {},
};

let messageSeq = 0;
function nextMessageId(): string {
  messageSeq += 1;
  return `msg-local-${messageSeq}`;
}

/** A stored message as the chat feed wants it. */
function toChatMessage(record: ChatMessageRecord): ChatMessage {
  return {
    id: record.id,
    role: record.role,
    text: record.text,
    // The same rule the live stream applies: a follow-up resolves to the run it talked
    // about, but only a rebuild owns the metrics deck.
    runId: record.pipeline_ran === false ? undefined : record.run_id ?? undefined,
    attachments: record.attachments.length > 0 ? record.attachments : undefined,
  };
}

export function useCampaignRun() {
  const [health, setHealth] = useState<HealthOut | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  /** Transcript per session id. Loaded from the server, then kept live in place. */
  const [transcripts, setTranscripts] = useState<Record<string, ChatMessage[]>>({});
  /** Run data per session id — see the header for why this is not a single slot. */
  const [runDataBySession, setRunDataBySession] = useState<Record<string, RunData>>({});

  const [status, setStatus] = useState<RunStatus>("idle");
  const [activeStage, setActiveStage] = useState<StageId | null>(null);
  const [completedStages, setCompletedStages] = useState<StageId[]>([]);
  const [toolTrail, setToolTrail] = useState<string[]>([]);
  const [tokenUsage, setTokenUsage] = useState<TokenUsage | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [pendingUploads, setPendingUploads] = useState<Upload[]>([]);
  /**
   * The open clarification round, per session.
   *
   * Per-session rather than a single slot for the same reason `runDataBySession` is: the
   * rep can switch away from an unanswered question and back, and a shared slot would show
   * one session's questions under another session's transcript.
   */
  const [clarificationBySession, setClarificationBySession] = useState<
    Record<string, ClarificationRequest | null>
  >({});

  const abortRef = useRef<AbortController | null>(null);
  /** Sessions already loaded, so switching back reads the cache instead of refetching. */
  const hydratedRef = useRef<Set<string>>(new Set());

  const messages = useMemo(
    () => (activeSessionId ? transcripts[activeSessionId] ?? [] : []),
    [transcripts, activeSessionId],
  );

  const runData = useMemo(
    () => (activeSessionId ? runDataBySession[activeSessionId] ?? EMPTY_RUN_DATA : EMPTY_RUN_DATA),
    [runDataBySession, activeSessionId],
  );

  const pendingQuestions = useMemo(
    () => (activeSessionId ? clarificationBySession[activeSessionId] ?? null : null),
    [clarificationBySession, activeSessionId],
  );

  const appendMessage = useCallback((sessionId: string, message: ChatMessage) => {
    setTranscripts((prev) => ({ ...prev, [sessionId]: [...(prev[sessionId] ?? []), message] }));
  }, []);

  const patchRunData = useCallback((sessionId: string, patch: Partial<RunData>) => {
    setRunDataBySession((prev) => ({
      ...prev,
      [sessionId]: { ...(prev[sessionId] ?? EMPTY_RUN_DATA), ...patch },
    }));
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
   * Load the run record plus the artifact rows the inspector needs, into `sessionId`'s
   * slot.
   *
   * Two different slices are needed. D2 wants the top of the ranked pool; the deck, D3
   * and D4 want the rows for the screens the optimizer actually bought, which are
   * scattered through the artifact and so have to be requested by id.
   *
   * `prefetched` is passed when the caller already fetched the record — the session-open
   * path used to fetch the same run twice, once to find it and once in here.
   */
  const loadRun = useCallback(
    async (sessionId: string, runId: string, prefetched?: RunRecord) => {
      patchRunData(sessionId, { loadingArtifacts: true });
      try {
        const run = prefetched ?? (await getRun(runId));
        patchRunData(sessionId, { ...EMPTY_RUN_DATA, run, loadingArtifacts: true });

        const allocations = run.optimization?.package?.allocations ?? [];
        const packagedIds = [...new Set(allocations.map((a) => a.screen_id))];
        const artifactErrors: Partial<Record<ArtifactKind, string>> = {};

        /**
         * A *missing reference* means the stage never ran, which the tabs render as an
         * empty state. A *failed fetch* against a reference that exists is different, and
         * has to be said out loud rather than collapsed into the same empty array.
         */
        const pull = async <T,>(
          kind: ArtifactKind,
          present: boolean,
          fetcher: () => Promise<{ rows: T[] }>,
        ): Promise<T[]> => {
          if (!present) return [];
          try {
            return (await fetcher()).rows;
          } catch (cause) {
            artifactErrors[kind] = describeError(cause);
            return [];
          }
        };

        const hasCandidates = Boolean(run.artifacts.screen_candidates);
        const hasEconomics = Boolean(run.artifacts.screen_economics);

        const [candidates, packagedCandidates, economics] = await Promise.all([
          pull<ScreenCandidate>("screen_candidates", hasCandidates, () =>
            getArtifactRows(runId, "screen_candidates", RANKED_ROW_LIMIT),
          ),
          pull<ScreenCandidate>("screen_candidates", hasCandidates && packagedIds.length > 0, () =>
            getArtifactRows(runId, "screen_candidates", PACKAGED_ROW_LIMIT, packagedIds),
          ),
          pull<ScreenEconomics>("screen_economics", hasEconomics, () =>
            getArtifactRows(
              runId,
              "screen_economics",
              packagedIds.length > 0 ? PACKAGED_ROW_LIMIT : RANKED_ROW_LIMIT,
              packagedIds.length > 0 ? packagedIds : undefined,
            ),
          ),
        ]);

        patchRunData(sessionId, {
          run,
          candidates,
          packagedCandidates,
          economics,
          loadingArtifacts: false,
          artifactErrors,
        });
      } catch (cause) {
        patchRunData(sessionId, { ...EMPTY_RUN_DATA });
        setError(describeError(cause));
      }
    },
    [patchRunData],
  );

  /**
   * Load a session's stored transcript and the package its last turn reported on.
   *
   * Marked hydrated up front so a fast double-click cannot fetch twice, and un-marked on
   * failure so a session whose load failed (backend down, say) retries on the next visit
   * rather than staying empty for the life of the tab.
   */
  const hydrateSession = useCallback(
    async (sessionId: string) => {
      if (hydratedRef.current.has(sessionId)) return;
      hydratedRef.current.add(sessionId);
      try {
        const stored = await listMessages(sessionId);

        // Questions arrive on the SSE `done` event, so without this a reload would leave
        // the rep reading an answer whose options are no longer on screen.
        const pending = await getClarification(sessionId);
        setClarificationBySession((prev) => ({ ...prev, [sessionId]: pending }));

        if (stored.length > 0) {
          setTranscripts((prev) => ({ ...prev, [sessionId]: stored.map(toChatMessage) }));
          // The most recent turn that carried a package is the one the panels show.
          const withRun = [...stored].reverse().find((m) => m.run_id);
          if (withRun?.run_id) await loadRun(sessionId, withRun.run_id);
          return;
        }

        // No stored transcript: a session from before messages were persisted. Rebuild
        // what is still durable — the brief and the package — and mark the assistant turn
        // `restored`, because its prose is genuinely unrecoverable.
        const runs = await listRuns(sessionId);
        if (runs.length === 0) return;

        const run = await getRun(runs[runs.length - 1].run_id);
        setTranscripts((prev) => ({
          ...prev,
          [sessionId]: [
            {
              id: nextMessageId(),
              role: "user",
              text: run.campaign_spec.original_query ?? run.campaign_spec.campaign_objective,
            },
            { id: nextMessageId(), role: "assistant", text: "", runId: run.id, restored: true },
          ],
        }));
        await loadRun(sessionId, run.id, run);
      } catch (cause) {
        hydratedRef.current.delete(sessionId);
        setError(describeError(cause));
      }
    },
    [loadRun],
  );

  // ------------------------------------------------------------------ sessions

  /** Per-turn UI state that belongs to the stream, not to a session. */
  const resetTurnState = useCallback(() => {
    setStatus("idle");
    setActiveStage(null);
    setCompletedStages([]);
    setToolTrail([]);
    setPendingUploads([]);
    setError(null);
  }, []);

  const selectSession = useCallback(
    (sessionId: string) => {
      if (sessionId === activeSessionId) return;
      setActiveSessionId(sessionId);
      resetTurnState();
      void hydrateSession(sessionId);
    },
    [activeSessionId, hydrateSession, resetTurnState],
  );

  const newCampaign = useCallback(async () => {
    try {
      const session = await createSession();
      setSessions((prev) => [session, ...prev]);
      hydratedRef.current.add(session.id); // Brand new: nothing to load.
      setActiveSessionId(session.id);
      setTranscripts((prev) => ({ ...prev, [session.id]: [] }));
      setRunDataBySession((prev) => ({ ...prev, [session.id]: { ...EMPTY_RUN_DATA } }));
      resetTurnState();
      return session;
    } catch (cause) {
      setError(describeError(cause));
      return null;
    }
  }, [resetTurnState]);

  const removeSession = useCallback(
    async (sessionId: string) => {
      try {
        // The backend cascades the transcript, runs and uploads.
        await deleteSession(sessionId);
        setSessions((prev) => prev.filter((s) => s.id !== sessionId));
        hydratedRef.current.delete(sessionId);
        setTranscripts((prev) => {
          const next = { ...prev };
          delete next[sessionId];
          return next;
        });
        setRunDataBySession((prev) => {
          const next = { ...prev };
          delete next[sessionId];
          return next;
        });
        if (sessionId === activeSessionId) setActiveSessionId(null);
      } catch (cause) {
        setError(describeError(cause));
      }
    },
    [activeSessionId],
  );

  /**
   * Clear the current session's transcript. Durable now, so it survives a reload — but the
   * session's runs and packages are deliberately left intact, which is what makes this
   * different from deleting the session.
   */
  const resetTranscript = useCallback(async () => {
    if (!activeSessionId) return;
    const sessionId = activeSessionId;
    setTranscripts((prev) => ({ ...prev, [sessionId]: [] }));
    setRunDataBySession((prev) => ({ ...prev, [sessionId]: { ...EMPTY_RUN_DATA } }));
    resetTurnState();
    try {
      await clearMessages(sessionId);
    } catch (cause) {
      setError(describeError(cause));
    }
  }, [activeSessionId, resetTurnState]);

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
      // Optimistic: shown immediately. The backend persists its own copy of this same
      // message before it invokes the agent, so a reload reads it from there.
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
              // Null clears a round the agent has moved past; a value opens a new one.
              // Always assigned, never merged — a stale question card is worse than none.
              setClarificationBySession((prev) => ({
                ...prev,
                [sessionId]: event.pending_questions ?? null,
              }));

              const rebuilt = event.pipeline_ran !== false;
              if (rebuilt && pipelineStarted) {
                seen.add("verification");
                setCompletedStages([...seen, "recommendation"]);
              }
              setActiveStage(null);
              setTokenUsage(event.token_usage);
              setStatus("done");

              appendMessage(sessionId, {
                // The id the backend stored, so this message and its persisted copy are
                // the same message rather than two that happen to match.
                id: event.message_id ?? nextMessageId(),
                role: "assistant",
                text: event.answer,
                runId: rebuilt ? event.run_id ?? undefined : undefined,
              });
              // Only refetch when there is something new to fetch.
              if (rebuilt && event.run_id) void loadRun(sessionId, event.run_id);
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

  /**
   * Send the rep's answers to an open clarification round.
   *
   * Just `submit` with composed text: the reply is an ordinary turn, and the agent already
   * has the original brief in its message history. The card is cleared optimistically so it
   * cannot be answered twice while the run streams; `onDone` is what re-opens a round if
   * the agent somehow asks again.
   */
  const answerClarification = useCallback(
    async (reply: string) => {
      if (!activeSessionId || status === "streaming") return;
      setClarificationBySession((prev) => ({ ...prev, [activeSessionId]: null }));
      await submit(reply);
    },
    [activeSessionId, status, submit],
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
    pendingQuestions,
    selectSession,
    newCampaign,
    removeSession,
    resetTranscript,
    attachFile,
    removePendingUpload,
    submit,
    answerClarification,
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
