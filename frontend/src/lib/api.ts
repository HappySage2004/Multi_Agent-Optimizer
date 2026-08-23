/**
 * The single boundary between the frontend and FastAPI. Nothing else fetches.
 *
 * Endpoints (backend/app/api/):
 *   GET    /health
 *   POST   /sessions            GET /sessions            PATCH/DELETE /sessions/{id}
 *   GET    /sessions/{id}/messages   POST/DELETE the same path
 *   GET    /messages/{id}       PATCH/DELETE /messages/{id}
 *   POST   /uploads             GET /uploads?session_id=
 *   POST   /campaign/run        POST /campaign/stream (SSE)
 *   GET    /runs/{id}           GET /runs/{id}/artifacts/{kind}?limit=
 */

import type {
  ArtifactKind,
  ArtifactRowsOut,
  CampaignRunOut,
  ChatMessageRecord,
  ClarificationOut,
  ClarificationRequest,
  HealthOut,
  NewChatMessage,
  RunRecord,
  RunSnapshot,
  ScreenCandidate,
  ScreenEconomics,
  Session,
  StreamDoneEvent,
  StreamErrorEvent,
  StreamUpdateEvent,
  Upload,
} from "./types";

/**
 * `127.0.0.1`, not `localhost`, and the difference is worth seconds on every panel load.
 *
 * uvicorn binds IPv4 only. `localhost` resolves to `::1` first on Windows, and a SYN to
 * `[::1]:8000` here is *dropped* rather than refused — so the client waits out a connect
 * timeout before falling back to IPv4. Measured on the four-request session-open path,
 * host as the only variable: 8.18s via `localhost` against 0.053s via `127.0.0.1`.
 *
 * Set NEXT_PUBLIC_API_BASE_URL to override; keep it on 127.0.0.1 unless uvicorn is bound
 * to `::` as well.
 */
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

/** An error carrying the backend's own status and `detail`, which is safe to display. */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { ...(init?.body ? { "content-type": "application/json" } : {}), ...init?.headers },
      cache: "no-store",
    });
  } catch {
    // A network-level failure has no status; the backend is most likely not running.
    throw new ApiError(0, `Cannot reach the API at ${API_BASE_URL}. Is the backend running?`);
  }

  if (!response.ok) {
    throw new ApiError(response.status, await readDetail(response));
  }
  return (await response.json()) as T;
}

/** FastAPI puts the message in `detail`, which is either a string or a validation array. */
async function readDetail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    const detail = body.detail;
    if (typeof detail === "string") return detail;
    if (detail) return JSON.stringify(detail);
  } catch {
    // Not JSON — fall through to the status line.
  }
  return `${response.status} ${response.statusText}`;
}

// --------------------------------------------------------------------- health

export function getHealth(): Promise<HealthOut> {
  return request<HealthOut>("/health");
}

// ------------------------------------------------------------------- sessions

export function listSessions(): Promise<Session[]> {
  return request<Session[]>("/sessions");
}

export function createSession(title = "New Campaign"): Promise<Session> {
  return request<Session>("/sessions", { method: "POST", body: JSON.stringify({ title }) });
}

/** Rename a session. The sidebar titles a session from the brief that started it. */
export function updateSession(sessionId: string, title: string): Promise<Session> {
  return request<Session>(`/sessions/${encodeURIComponent(sessionId)}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
}

export function deleteSession(sessionId: string): Promise<{ deleted: string }> {
  return request<{ deleted: string }>(`/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
}

// ------------------------------------------------------------------- messages

/**
 * The transcript. Persisted server-side by the campaign endpoints, so these are a read
 * and amend surface rather than the write path for a normal turn — the browser does not
 * have to POST its own messages for them to survive a reload.
 */
export function listMessages(sessionId: string): Promise<ChatMessageRecord[]> {
  return request<ChatMessageRecord[]>(`/sessions/${encodeURIComponent(sessionId)}/messages`);
}

export function createMessage(
  sessionId: string,
  message: NewChatMessage,
): Promise<ChatMessageRecord> {
  return request<ChatMessageRecord>(`/sessions/${encodeURIComponent(sessionId)}/messages`, {
    method: "POST",
    body: JSON.stringify(message),
  });
}

export function updateMessage(
  messageId: string,
  patch: Partial<NewChatMessage>,
): Promise<ChatMessageRecord> {
  return request<ChatMessageRecord>(`/messages/${encodeURIComponent(messageId)}`, {
    method: "PATCH",
    body: JSON.stringify(patch),
  });
}

export function deleteMessage(messageId: string): Promise<{ deleted: string }> {
  return request<{ deleted: string }>(`/messages/${encodeURIComponent(messageId)}`, {
    method: "DELETE",
  });
}

/** Clear a session's transcript. Its runs and packages are left intact. */
export function clearMessages(sessionId: string): Promise<{ deleted: number }> {
  return request<{ deleted: number }>(`/sessions/${encodeURIComponent(sessionId)}/messages`, {
    method: "DELETE",
  });
}

// -------------------------------------------------------------------- uploads

/** Multipart, so this one bypasses the JSON content-type in `request`. */
export async function stageUpload(sessionId: string, file: File): Promise<Upload> {
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("file", file);

  const response = await fetch(`${API_BASE_URL}/uploads`, { method: "POST", body: form });
  if (!response.ok) throw new ApiError(response.status, await readDetail(response));
  return (await response.json()) as Upload;
}

export function listUploads(sessionId: string): Promise<Upload[]> {
  return request<Upload[]>(`/uploads?session_id=${encodeURIComponent(sessionId)}`);
}

// ----------------------------------------------------------------------- runs

export function getRun(runId: string): Promise<RunRecord> {
  return request<RunRecord>(`/runs/${encodeURIComponent(runId)}`);
}

export function listRuns(sessionId?: string): Promise<RunSnapshot[]> {
  const query = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
  return request<RunSnapshot[]>(`/runs${query}`);
}

/** Row type per artifact kind, so callers get the right shape without a cast. */
interface ArtifactRowType {
  screen_candidates: ScreenCandidate;
  screen_economics: ScreenEconomics;
}

/**
 * `screenIds` restricts the rows to specific screens. The ones in a package sit anywhere
 * in a 250- or 750-row artifact, so a plain top-N slice would mostly miss them.
 */
export function getArtifactRows<K extends ArtifactKind>(
  runId: string,
  kind: K,
  limit = 25,
  screenIds?: string[],
): Promise<ArtifactRowsOut<ArtifactRowType[K]>> {
  const params = new URLSearchParams({ limit: String(limit) });
  for (const id of screenIds ?? []) params.append("screen_ids", id);

  return request<ArtifactRowsOut<ArtifactRowType[K]>>(
    `/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(kind)}?${params}`,
  );
}

// ------------------------------------------------------------------- campaign

export interface CampaignQuery {
  query: string;
  session_id?: string | null;
  upload_ids?: string[];
}

/**
 * The session's open clarifying questions, or null.
 *
 * Needed on hydration: the questions ride in on the SSE `done` event, so a reload or a
 * session switch would otherwise leave the rep looking at an answer that refers to options
 * no longer on screen.
 */
export async function getClarification(sessionId: string): Promise<ClarificationRequest | null> {
  const out = await request<ClarificationOut>(
    `/sessions/${encodeURIComponent(sessionId)}/clarification`,
  );
  return out.pending_questions;
}

/** Blocking run. Returns the answer but no stage progress; prefer `streamCampaign`. */
export function runCampaign(payload: CampaignQuery): Promise<CampaignRunOut> {
  return request<CampaignRunOut>("/campaign/run", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export interface StreamHandlers {
  onUpdate?: (event: StreamUpdateEvent) => void;
  onDone?: (event: StreamDoneEvent) => void;
  onError?: (event: StreamErrorEvent) => void;
}

/**
 * SSE over POST, so `EventSource` (GET-only) is out and we parse the stream ourselves.
 *
 * The backend emits `update` events per graph node, then exactly one terminal `done` or
 * `error`. A run takes ~90s because of the shared Gemini rate limiter, so `signal` is how
 * the UI cancels one.
 */
export async function streamCampaign(
  payload: CampaignQuery,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/campaign/stream`, {
      method: "POST",
      headers: { "content-type": "application/json", accept: "text/event-stream" },
      body: JSON.stringify(payload),
      signal,
    });
  } catch {
    // An aborted run is a user action, not a failure worth surfacing.
    if (signal?.aborted) return;
    throw new ApiError(0, `Cannot reach the API at ${API_BASE_URL}. Is the backend running?`);
  }

  // A pre-stream rejection (503 no API key, 404 unknown session) arrives as plain JSON.
  if (!response.ok) throw new ApiError(response.status, await readDetail(response));
  if (!response.body) throw new ApiError(502, "The API returned an empty stream.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Events are separated by a blank line; anything after the last one is partial.
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";
      for (const frame of frames) dispatchFrame(frame, handlers);
    }
    if (buffer.trim()) dispatchFrame(buffer, handlers);
  } finally {
    reader.releaseLock();
  }
}

function dispatchFrame(frame: string, handlers: StreamHandlers): void {
  let name = "message";
  const dataLines: string[] = [];

  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) name = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return;

  let data: unknown;
  try {
    data = JSON.parse(dataLines.join("\n"));
  } catch {
    return; // A malformed frame should not kill the stream.
  }

  if (name === "update") handlers.onUpdate?.(data as StreamUpdateEvent);
  else if (name === "done") handlers.onDone?.(data as StreamDoneEvent);
  else if (name === "error") handlers.onError?.(data as StreamErrorEvent);
}
