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
  listModels,
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
  ModelSelection,
  ModelsOut,
  RunRecord,
  ScreenCandidate,
  ScreenEconomics,
  Session,
  TokenUsage,
  Upload,
} from "@/lib/types";

/**
 * Where the rep's model choice is remembered.
 *
 * localStorage rather than the backend: the selection is a per-operator preference, and
 * putting it on the server would make one rep's switch change everyone's next run. It is
 * always re-validated against `GET /models` on load — a key that has since been removed,
 * or a deployment renamed, would otherwise send a model id the backend rejects with a 400.
 */
const MODEL_SELECTION_KEY = "agentiq.model-selection";

function readStoredSelection(): ModelSelection | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(MODEL_SELECTION_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<ModelSelection>;
    if (typeof parsed.provider === "string" && typeof parsed.model === "string") {
      return { provider: parsed.provider, model: parsed.model };
    }
  } catch {
    // Corrupt or unreadable storage is not worth surfacing; fall back to the default.
  }
  return null;
}

/** True when the catalogue still offers this exact choice on a configured provider. */
function isSelectable(catalogue: ModelsOut, choice: ModelSelection): boolean {
  const provider = catalogue.providers.find((p) => p.id === choice.provider);
  return Boolean(provider?.configured && provider.models.some((m) => m.id === choice.model));
}

/** Ranked rows pulled for the D2 candidate list. */
const RANKED_ROW_LIMIT = 60;
/**
 * Cap for the screen-filtered pulls. A package holds at most `max_screens_in_package`
 * (120) screens, and economics rows are per screen *and* time block, so this leaves room
 * for the widest package the optimizer can return.
 */
const PACKAGED_ROW_LIMIT = 1000;

/**
 * How many times the repair effect may try to load one run. Three, because the failure it
 * exists for is transient (a request that lost a race with a session switch, a backend
 * still starting up) — and because an unbounded retry against a genuinely broken run would
 * be a request loop.
 */
const MAX_RUN_LOAD_ATTEMPTS = 3;

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
  const [models, setModels] = useState<ModelsOut | null>(null);
  const [modelsError, setModelsError] = useState<string | null>(null);
  /**
   * Null until the catalogue lands, and again whenever the stored choice is no longer
   * offered. Null means "send no provider/model and let the backend pick", which is what
   * keeps a turn working while /models is still in flight.
   */
  const [modelSelection, setModelSelection] = useState<ModelSelection | null>(null);
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
  /** Sessions whose transcript load has been attempted, so a revisit reads the cache. */
  const hydratedRef = useRef<Set<string>>(new Set());
  /** `sessionId:runId` pairs being fetched right now, so renders cannot stack duplicates. */
  const runLoadInFlight = useRef<Set<string>>(new Set());
  /**
   * Attempts per `sessionId:runId`. The repair effect below is bounded by this rather than
   * by a "tried once" flag: one transient failure must not leave the inspector empty for
   * the life of the tab, but a permanent one must not spin either.
   */
  const runLoadAttempts = useRef<Map<string, number>>(new Map());

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

  /**
   * Adopt a title the backend reports, so the sidebar and the centre header never disagree.
   *
   * Both the `session` and `done` events carry one: the first is provisional (from the
   * brief), the second may be the campaign objective intake resolved. Applying them the
   * same way is what keeps the rail showing a campaign name rather than a chat log.
   */
  const adoptSessionTitle = useCallback((sessionId: string, title: string | null | undefined) => {
    if (!title) return;
    setSessions((prev) => prev.map((s) => (s.id === sessionId ? { ...s, title } : s)));
  }, []);

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
      const [healthResult, sessionsResult, modelsResult] = await Promise.allSettled([
        getHealth(),
        listSessions(),
        listModels(),
      ]);
      if (cancelled) return;

      if (healthResult.status === "fulfilled") setHealth(healthResult.value);
      if (sessionsResult.status === "fulfilled") {
        setSessions(sessionsResult.value);
      } else {
        setError(describeError(sessionsResult.reason));
      }

      // The model list failing is not a run-blocking error — omitting provider/model runs
      // the backend default — so it goes to its own slot rather than the chat error banner.
      if (modelsResult.status === "fulfilled") {
        const catalogue = modelsResult.value;
        setModels(catalogue);
        setModelsError(null);
        const stored = readStoredSelection();
        setModelSelection(
          stored && isSelectable(catalogue, stored)
            ? stored
            : { provider: catalogue.default_provider, model: catalogue.default_model },
        );
      } else {
        setModelsError(describeError(modelsResult.reason));
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
      // The effect below may ask for the same run on several renders before the first
      // fetch resolves, so collapse concurrent calls and cap total attempts.
      const key = `${sessionId}:${runId}`;
      if (runLoadInFlight.current.has(key)) return;
      const attempts = runLoadAttempts.current.get(key) ?? 0;
      if (attempts >= MAX_RUN_LOAD_ATTEMPTS) return;
      runLoadInFlight.current.add(key);
      runLoadAttempts.current.set(key, attempts + 1);

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
      } finally {
        runLoadInFlight.current.delete(key);
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
        // The mark stays: a persistent failure must not spin. The repair effect below
        // still retries the *package* once the transcript is in hand, which is the case
        // that actually recovers.
        setError(describeError(cause));
      }
    },
    [loadRun],
  );

  /**
   * The run the active session's transcript says the panels should be showing.
   *
   * Derived from the transcript rather than tracked separately, so it is correct whether
   * the message arrived live or came back from localDB.
   */
  const expectedRunId = useMemo(() => {
    const list = activeSessionId ? transcripts[activeSessionId] : undefined;
    if (!list) return null;
    for (let i = list.length - 1; i >= 0; i -= 1) {
      if (list[i].runId) return list[i].runId as string;
    }
    return null;
  }, [transcripts, activeSessionId]);

  /**
   * Keep the inspector in step with the active session, declaratively.
   *
   * Hydration used to fire only from `selectSession`'s click handler, guarded by a ref. Any
   * path that left a session active without its run data — a remount that reset the cache
   * but not the guard, a load that raced a session switch, a first paint that never saw a
   * click — left D1-D4 permanently empty while the transcript sat there describing a
   * package. Stating the invariant ("the loaded run matches the one the transcript names")
   * and repairing it is what makes that unreachable, rather than one more special case.
   *
   * Bounded by `runLoadAttempts` inside `loadRun`, so this cannot loop.
   */
  useEffect(() => {
    const sessionId = activeSessionId;
    if (!sessionId) return;

    if (!hydratedRef.current.has(sessionId)) {
      void hydrateSession(sessionId);
      return;
    }
    if (expectedRunId && runDataBySession[sessionId]?.run?.id !== expectedRunId) {
      void loadRun(sessionId, expectedRunId);
    }
  }, [activeSessionId, expectedRunId, runDataBySession, hydrateSession, loadRun]);

  // -------------------------------------------------------------------- model

  /**
   * Switch the model the next turn runs on. Persisted immediately, so a reload keeps it.
   *
   * Deliberately not applied to the turn in flight: the graph is compiled per selection
   * and swapping mid-stream would leave half a run on each provider.
   */
  const selectModel = useCallback((choice: ModelSelection) => {
    setModelSelection(choice);
    try {
      window.localStorage.setItem(MODEL_SELECTION_KEY, JSON.stringify(choice));
    } catch {
      // Private-mode storage refusals must not lose the in-memory choice.
    }
  }, []);

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
      // Loading is the effect's job, keyed on the active session — see above for why this
      // is not done here any more.
    },
    [activeSessionId, resetTurnState],
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
        for (const key of [...runLoadAttempts.current.keys()]) {
          if (key.startsWith(`${sessionId}:`)) runLoadAttempts.current.delete(key);
        }
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
          {
            query: trimmed,
            session_id: sessionId,
            upload_ids: attachments.map((u) => u.id),
            // Omitted while the catalogue is still loading, which runs the backend default.
            provider: modelSelection?.provider,
            model: modelSelection?.model,
          },
          {
            onSession: (event) => adoptSessionTitle(sessionId, event.session_title),
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
              // The final name, which is the resolved campaign objective when intake
              // produced one -- the same string the centre header shows.
              adoptSessionTitle(sessionId, event.session_title);
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
    [
      activeSessionId,
      adoptSessionTitle,
      appendMessage,
      loadRun,
      modelSelection,
      newCampaign,
      pendingUploads,
      status,
    ],
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

  const activeSession = useMemo(
    () => sessions.find((s) => s.id === activeSessionId) ?? null,
    [sessions, activeSessionId],
  );

  return {
    health,
    models,
    modelsError,
    modelSelection,
    selectModel,
    sessions,
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
