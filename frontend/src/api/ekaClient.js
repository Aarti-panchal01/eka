/**
 * ekaClient.js — thin fetch wrapper over the Eka FastAPI backend.
 *
 * No dependencies. Uses browser-native `fetch`, `FormData`, `Audio`, and
 * `URL.createObjectURL`. Every route here was checked directly against:
 *   backend/main.py                  (router prefixes)
 *   backend/api/routes/chat.py
 *   backend/api/routes/memory.py
 *   backend/api/routes/voice.py
 *   backend/api/routes/goals.py
 *   backend/api/routes/reflections.py
 *   backend/api/routes/insights.py
 *   backend/api/routes/preferences.py
 *   backend/models/schemas.py
 *
 * The backend has no auth — every request identifies the caller via a
 * `user_id` that the frontend generates and persists itself. Several
 * mutating routes (memory update/priority/delete) take `user_id` as a
 * REQUIRED QUERY PARAM even though they're PUT/DELETE — that's not a bug,
 * don't "fix" it by moving it into the body.
 */

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/** Base URL of the Eka API. Only VITE_-prefixed env vars reach the browser. */
const BASE_URL = import.meta.env.VITE_EKA_API_URL || "http://localhost:8000";

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/**
 * Thrown by every failed API call. `detail` mirrors whatever FastAPI put in
 * the JSON body's `detail` field:
 *   - HTTPException(...)   -> detail is a string
 *   - 422 validation error -> detail is an array of {loc, msg, type}
 * `message` is always a human-readable string derived from either shape.
 */
export class EkaApiError extends Error {
  constructor(message, { status, detail, path } = {}) {
    super(message);
    this.name = "EkaApiError";
    this.status = status ?? null;
    this.detail = detail ?? null;
    this.path = path ?? null;
  }
}

/**
 * Turns a FastAPI error body into one readable string.
 * Handles both HTTPException ({"detail": "..."}) and pydantic 422 validation
 * errors ({"detail": [{"loc": [...], "msg": "...", "type": "..."}]}).
 */
function extractErrorMessage(data, response) {
  const detail = data && typeof data === "object" ? data.detail : undefined;

  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }

  if (Array.isArray(detail) && detail.length > 0) {
    return detail
      .map((err) => {
        const loc = Array.isArray(err.loc)
          ? err.loc.filter((part) => part !== "body").join(".")
          : "";
        const msg = err.msg || err.type || "invalid value";
        return loc ? `${loc}: ${msg}` : msg;
      })
      .join("; ");
  }

  if (typeof data === "string" && data.trim()) {
    return data;
  }

  return `${response.status} ${response.statusText || "Request failed"}`;
}

// ---------------------------------------------------------------------------
// Core request helper
// ---------------------------------------------------------------------------

/**
 * Makes one request and returns parsed JSON (or null for empty/204 bodies).
 * Throws EkaApiError on any non-2xx response or network failure.
 *
 * @param {string} method       HTTP method
 * @param {string} path         path beginning with "/", e.g. "/chat/send"
 * @param {any}    [body]       plain object (JSON-encoded) or a FormData instance
 * @param {object} [opts]
 * @param {boolean} [opts.isFormData]  pass true when body is a FormData — skips
 *                                     JSON.stringify and the Content-Type header
 *                                     (the browser sets the multipart boundary)
 * @param {AbortSignal} [opts.signal]
 */
async function apiCall(method, path, body = null, { isFormData = false, signal } = {}) {
  const url = `${BASE_URL}${path}`;
  const headers = {};
  let finalBody;

  if (isFormData) {
    finalBody = body; // browser sets multipart/form-data + boundary itself
  } else if (body !== null && body !== undefined) {
    headers["Content-Type"] = "application/json";
    finalBody = JSON.stringify(body);
  }

  let response;
  try {
    response = await fetch(url, { method, headers, body: finalBody, signal });
  } catch (networkErr) {
    throw new EkaApiError(
      `Could not reach Eka API at ${url}: ${networkErr.message}`,
      { status: 0, detail: networkErr.message, path }
    );
  }

  // FastAPI returns 204 (and sometimes an empty 200) with no body.
  if (response.status === 204) {
    return null;
  }

  const rawText = await response.text();
  let data = null;
  if (rawText) {
    try {
      data = JSON.parse(rawText);
    } catch {
      data = rawText; // non-JSON body (shouldn't happen on this API, but don't crash)
    }
  }

  if (!response.ok) {
    throw new EkaApiError(extractErrorMessage(data, response), {
      status: response.status,
      detail: data && typeof data === "object" ? data.detail : data,
      path,
    });
  }

  return data;
}

// ---------------------------------------------------------------------------
// Query-string helper
// ---------------------------------------------------------------------------

/**
 * Builds a "?a=1&b=2" query string, silently dropping any param whose value
 * is null, undefined, or "" (so callers can pass optional filters straight
 * through without hand-checking each one).
 */
function qs(params = {}) {
  const usp = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === "") continue;
    usp.append(key, value);
  }
  const s = usp.toString();
  return s ? `?${s}` : "";
}

// ---------------------------------------------------------------------------
// user_id helper — the backend has no auth, so the frontend mints one
// ---------------------------------------------------------------------------

const USER_ID_KEY = "eka_user_id";

function fallbackUuid() {
  // crypto.randomUUID() is unavailable on very old browsers / insecure
  // (non-HTTPS, non-localhost) origins. RFC4122-ish fallback.
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * Returns the persistent anonymous user id for this browser, creating one
 * on first use. Every ekaAPI call that touches user data needs this — the
 * backend has no login/session, `user_id` is the entire identity model.
 */
export function getOrCreateUserId() {
  let id = null;
  try {
    id = localStorage.getItem(USER_ID_KEY);
  } catch {
    // localStorage can throw in some locked-down/private-browsing contexts
  }
  if (id) return id;

  id =
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : fallbackUuid();

  try {
    localStorage.setItem(USER_ID_KEY, id);
  } catch {
    // best-effort persistence; caller still gets a usable id for this session
  }
  return id;
}

// ---------------------------------------------------------------------------
// Voice playback (not routed through apiCall — audio/wav, not JSON)
// ---------------------------------------------------------------------------

/**
 * POST /voice/tts. Returns a Blob (audio/wav) — this bypasses apiCall
 * entirely because the success response is binary, not JSON.
 *
 * On 503 (TTS unavailable — bad/missing SARVAM_API_KEY, no credits, etc.)
 * this throws an EkaApiError whose message says voice is unavailable, so
 * callers can catch that specific case and just keep showing text.
 */
async function synthesize(text, mode = "founder") {
  const path = "/voice/tts";
  const url = `${BASE_URL}${path}`;

  let response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, mode }),
    });
  } catch (networkErr) {
    throw new EkaApiError(`Could not reach Eka API at ${url}: ${networkErr.message}`, {
      status: 0,
      detail: networkErr.message,
      path,
    });
  }

  if (response.status === 503) {
    let detail = "Text-to-speech is currently unavailable (check SARVAM_API_KEY and credits).";
    try {
      const data = await response.json();
      if (data && typeof data.detail === "string") detail = data.detail;
    } catch {
      // fall through with the default message
    }
    throw new EkaApiError(`Voice is unavailable: ${detail}`, {
      status: 503,
      detail,
      path,
    });
  }

  if (!response.ok) {
    let data = null;
    try {
      data = await response.json();
    } catch {
      // non-JSON error body; extractErrorMessage falls back to statusText
    }
    throw new EkaApiError(extractErrorMessage(data, response), {
      status: response.status,
      detail: data && typeof data === "object" ? data.detail : data,
      path,
    });
  }

  return response.blob();
}

/**
 * Plays an audio Blob (e.g. the one from `synthesize`) and returns a Promise
 * that resolves when playback finishes and rejects on playback error. The
 * object URL is revoked in both cases so it never leaks.
 *
 * The returned Promise ALSO has a `.stop()` method attached to it (it's
 * still a normal Promise — you can `await` it — but you can also call
 * `.stop()` on the same value to interrupt playback early; `.stop()`
 * resolves the promise rather than rejecting it, since stopping on purpose
 * isn't an error).
 *
 * @param {Blob} audioBlob
 * @param {object} [opts]
 * @param {number} [opts.playbackSpeed=1.0]
 * @returns {Promise<void> & { stop: () => void }}
 */
function playAudio(audioBlob, { playbackSpeed = 1.0 } = {}) {
  const url = URL.createObjectURL(audioBlob);
  const audioEl = new Audio(url);
  audioEl.playbackRate = playbackSpeed;

  let settled = false;
  let resolveFn = null;

  const cleanup = () => URL.revokeObjectURL(url);

  const promise = new Promise((resolve, reject) => {
    resolveFn = resolve;

    audioEl.addEventListener("ended", () => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve();
    });
    audioEl.addEventListener("error", () => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error("Audio playback failed"));
    });
    audioEl.play().catch((err) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(err);
    });
  });

  promise.stop = () => {
    if (settled) return;
    settled = true;
    audioEl.pause();
    cleanup();
    resolveFn();
  };

  return promise;
}

// ---------------------------------------------------------------------------
// ekaAPI
// ---------------------------------------------------------------------------

/**
 * Drop routing internals from a chat response before the UI ever sees them.
 *
 * Stripped here rather than hidden at render time on purpose: a field that
 * never enters the app cannot leak through a new component, a debug panel, or
 * a console.log of the response object. The UI keeps what it legitimately
 * shows — the reply, the session, the mode, and how many memories were used.
 *
 * `degraded` and the `*:heuristic` markers are genuinely useful when debugging
 * a bad answer, so they remain visible in the Network tab and in backend logs.
 * They are just not product surface.
 */
const INTERNAL_FIELDS = [
  "provider",
  "complexity",
  "sentiment",
  "degraded",
  "llm_backend",
  "latency_ms",
  "message_id",
];

function stripInternals(res) {
  if (!res || typeof res !== "object") return res;
  const clean = {};
  for (const [key, value] of Object.entries(res)) {
    if (INTERNAL_FIELDS.includes(key)) continue;
    // Catches anything future the backend adds that names a fallback tier,
    // e.g. "ranker": "heuristic".
    if (typeof value === "string" && value.includes("heuristic")) continue;
    clean[key] = value;
  }
  return clean;
}

export const ekaAPI = {
  // ===================================================================
  // CHAT — prefix /chat  (backend/api/routes/chat.py)
  // ===================================================================

  /**
   * POST /chat/send — the main endpoint. Runs the full RAG pipeline.
   * `sessionId` may be null/undefined for a brand-new conversation; the
   * backend creates one and returns its id in `response.session_id` —
   * always store that back, every reply carries the current session id.
   * @returns {Promise<import(".").ChatResponse>}
   */
  sendMessage(sessionId, userId, message, mode = "founder") {
    return apiCall("POST", "/chat/send", {
      message,
      user_id: userId,
      session_id: sessionId || null,
      mode,
    }).then(stripInternals);
  },

  /** GET /chat/sessions?user_id=&include_archived=&limit= */
  getSessions(userId, { includeArchived = false, limit = 50 } = {}) {
    return apiCall(
      "GET",
      `/chat/sessions${qs({ user_id: userId, include_archived: includeArchived, limit })}`
    );
  },

  /** POST /chat/sessions  body: {user_id, mode, title} */
  createSession(userId, mode = "founder", title = null) {
    return apiCall("POST", "/chat/sessions", { user_id: userId, mode, title });
  },

  /** GET /chat/sessions/{sessionId} */
  getSession(sessionId) {
    return apiCall("GET", `/chat/sessions/${encodeURIComponent(sessionId)}`);
  },

  /** PATCH /chat/sessions/{sessionId}  body: any of {title, mode, archived} */
  updateSession(sessionId, updates) {
    return apiCall("PATCH", `/chat/sessions/${encodeURIComponent(sessionId)}`, updates);
  },

  /** GET /chat/sessions/{sessionId}/messages?limit= */
  getMessages(sessionId, limit = 200) {
    return apiCall(
      "GET",
      `/chat/sessions/${encodeURIComponent(sessionId)}/messages${qs({ limit })}`
    );
  },

  /**
   * DELETE /chat/sessions/{sessionId} — soft delete (sets archived=true).
   * No user_id query param here; the backend looks the session up by id alone.
   */
  archiveSession(sessionId) {
    return apiCall("DELETE", `/chat/sessions/${encodeURIComponent(sessionId)}`);
  },

  // ===================================================================
  // MEMORY — prefix /memory  (backend/api/routes/memory.py)
  // ===================================================================

  /**
   * GET /memory?user_id=&topic=&priority=&source=&q=&date_from=&date_to=&skip=&limit=
   * NOTE: the response is an ENVELOPE, not a bare array:
   *   { items: MemoryResponse[], total, skip, limit }
   * Destructure `.items` — do not `.map()` the return value directly.
   */
  getMemories(userId, filters = {}) {
    const {
      topic,
      priority,
      source,
      q,
      dateFrom,
      dateTo,
      skip = 0,
      limit = 20,
    } = filters;
    return apiCall(
      "GET",
      `/memory${qs({
        user_id: userId,
        topic,
        priority,
        source,
        q,
        date_from: dateFrom,
        date_to: dateTo,
        skip,
        limit,
      })}`
    );
  },

  /** GET /memory/{memoryId}?user_id= (user_id optional here, but always pass it) */
  getMemory(memoryId, userId) {
    return apiCall("GET", `/memory/${encodeURIComponent(memoryId)}${qs({ user_id: userId })}`);
  },

  /**
   * POST /memory  body: MemoryCreate
   *   {user_id, title, content, source='manual', topic, tags=[], importance=5, user_priority='normal'}
   */
  createMemory(memoryData) {
    return apiCall("POST", "/memory", memoryData);
  },

  /**
   * PUT /memory/{memoryId}?user_id=  body: partial MemoryUpdate.
   * `user_id` is a REQUIRED QUERY PARAM even though this is a PUT with a
   * JSON body — do not put it inside `updates`.
   */
  updateMemory(memoryId, updates, userId) {
    return apiCall(
      "PUT",
      `/memory/${encodeURIComponent(memoryId)}${qs({ user_id: userId })}`,
      updates
    );
  },

  /** DELETE /memory/{memoryId}?user_id= — user_id required in the query string. */
  deleteMemory(memoryId, userId) {
    return apiCall("DELETE", `/memory/${encodeURIComponent(memoryId)}${qs({ user_id: userId })}`);
  },

  /**
   * PUT /memory/{memoryId}/priority?user_id=  body: {priority}
   * priority is one of "high" | "normal" | "low" | "excluded" — "excluded"
   * removes the memory from retrieval entirely.
   */
  updateMemoryPriority(memoryId, priority, userId) {
    return apiCall(
      "PUT",
      `/memory/${encodeURIComponent(memoryId)}/priority${qs({ user_id: userId })}`,
      { priority }
    );
  },

  /**
   * POST /memory/search  body: {text, user_id, limit}
   * Returns a bare array of MemorySearchResult — unlike getMemories, this
   * one is NOT wrapped in an {items,...} envelope.
   */
  searchMemories(queryText, userId, limit = 20) {
    return apiCall("POST", "/memory/search", { text: queryText, user_id: userId, limit });
  },

  /**
   * POST /memory/upload — multipart/form-data with fields:
   * file, user_id, title, topic, importance. Accepts .txt/.md/.markdown/.pdf,
   * max 5MB (enforced server-side; oversized files come back as a 413 detail string).
   */
  uploadFile(file, userId, { title, topic, importance = 6 } = {}) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("user_id", userId);
    if (title) formData.append("title", title);
    if (topic) formData.append("topic", topic);
    formData.append("importance", String(importance));
    return apiCall("POST", "/memory/upload", formData, { isFormData: true });
  },

  // ===================================================================
  // VOICE — prefix /voice  (backend/api/routes/voice.py)
  // ===================================================================

  /**
   * POST /voice/stt — multipart/form-data, field name "file".
   * @param {Blob} audioBlob   e.g. from MediaRecorder
   * @param {string} [filename]
   * @returns {Promise<{text: string, backend: string|null}>}
   */
  transcribe(audioBlob, filename = "recording.webm") {
    const formData = new FormData();
    formData.append("file", audioBlob, filename);
    return apiCall("POST", "/voice/stt", formData, { isFormData: true });
  },

  /**
   * POST /voice/tts — see `synthesize` above. Returns a raw audio/wav Blob,
   * NOT JSON, so it deliberately does not go through `apiCall`. Throws
   * EkaApiError with a "Voice is unavailable" message on 503.
   * @returns {Promise<Blob>}
   */
  synthesize,

  /**
   * Plays a Blob returned by `synthesize`. See `playAudio` above for the
   * exact resolve/reject/stop() contract.
   */
  playAudio,

  /** GET /voice/voices -> {voices: {mode: speakerName}, configured: bool} */
  getVoices() {
    return apiCall("GET", "/voice/voices");
  },

  // ===================================================================
  // GOALS — prefix /goals  (backend/api/routes/goals.py)
  // ===================================================================

  /** GET /goals?user_id=&status=  status is one of active|completed|paused */
  getGoals(userId, status) {
    return apiCall("GET", `/goals${qs({ user_id: userId, status })}`);
  },

  /** POST /goals  body: GoalCreate (must include user_id) */
  createGoal(goalData) {
    return apiCall("POST", "/goals", goalData);
  },

  /** GET /goals/{goalId} */
  getGoal(goalId) {
    return apiCall("GET", `/goals/${encodeURIComponent(goalId)}`);
  },

  /** PUT /goals/{goalId}  body: partial GoalUpdate */
  updateGoal(goalId, updates) {
    return apiCall("PUT", `/goals/${encodeURIComponent(goalId)}`, updates);
  },

  /**
   * PUT /goals/{goalId}/progress  body: {current_value}
   * Moves current_value only; the backend auto-completes the goal once it
   * reaches target_value.
   */
  updateGoalProgress(goalId, currentValue) {
    return apiCall("PUT", `/goals/${encodeURIComponent(goalId)}/progress`, {
      current_value: currentValue,
    });
  },

  /** DELETE /goals/{goalId} */
  deleteGoal(goalId) {
    return apiCall("DELETE", `/goals/${encodeURIComponent(goalId)}`);
  },

  // ===================================================================
  // REFLECTIONS — prefix /reflections  (backend/api/routes/reflections.py)
  // ===================================================================

  /** GET /reflections?user_id=&limit=&date_from=&date_to= */
  getReflections(userId, { limit = 30, dateFrom, dateTo } = {}) {
    return apiCall(
      "GET",
      `/reflections${qs({ user_id: userId, limit, date_from: dateFrom, date_to: dateTo })}`
    );
  },

  /**
   * POST /reflections  body: ReflectionCreate
   *   {user_id, date?, mood?, challenges_faced?, learnings?, gratitude?,
   *    mode_used?, request_commentary?}
   * When request_commentary is true, the backend has Eka write
   * eka_commentary on the created reflection via the LLM (may be slow).
   */
  createReflection(reflectionData) {
    return apiCall("POST", "/reflections", reflectionData);
  },

  /**
   * GET /reflections/by-date/{date}?user_id=  date is "YYYY-MM-DD".
   * Returns null (not a 404) when there's no reflection for that day.
   */
  getReflectionByDate(userId, date) {
    return apiCall(
      "GET",
      `/reflections/by-date/${encodeURIComponent(date)}${qs({ user_id: userId })}`
    );
  },

  /** GET /reflections/{id} */
  getReflection(id) {
    return apiCall("GET", `/reflections/${encodeURIComponent(id)}`);
  },

  /** PUT /reflections/{id}  body: partial ReflectionUpdate */
  updateReflection(id, updates) {
    return apiCall("PUT", `/reflections/${encodeURIComponent(id)}`, updates);
  },

  /** DELETE /reflections/{id} */
  deleteReflection(id) {
    return apiCall("DELETE", `/reflections/${encodeURIComponent(id)}`);
  },

  // ===================================================================
  // INSIGHTS — prefix /insights  (backend/api/routes/insights.py)
  // ===================================================================

  /**
   * GET /insights/daily/{date}?user_id=&force=  date is "YYYY-MM-DD".
   * Generates the insight on first call for that day; pass force:true to
   * recompute even if it's already cached.
   */
  getDailyInsight(userId, date, { force = false } = {}) {
    return apiCall(
      "GET",
      `/insights/daily/${encodeURIComponent(date)}${qs({ user_id: userId, force })}`
    );
  },

  /** GET /insights/daily?user_id=  — shortcut for "today", no date in the path. */
  getTodayInsight(userId) {
    return apiCall("GET", `/insights/daily${qs({ user_id: userId })}`);
  },

  /**
   * GET /insights/weekly?user_id=&week_start=
   * `weekStart` ("YYYY-MM-DD") is optional — omit it and the backend
   * defaults to the current week's Monday.
   */
  getWeeklyInsight(userId, weekStart) {
    return apiCall("GET", `/insights/weekly${qs({ user_id: userId, week_start: weekStart })}`);
  },

  // ===================================================================
  // PREFERENCES — prefix /preferences  (backend/api/routes/preferences.py)
  // ===================================================================

  /** GET /preferences?user_id=  — creates a defaults row on first call. */
  getPreferences(userId) {
    return apiCall("GET", `/preferences${qs({ user_id: userId })}`);
  },

  /**
   * PUT /preferences  body: PreferencesUpdate — user_id goes IN THE BODY
   * here (unlike memory's query-param pattern), because there's no
   * {preference_id} in the path to attach a query param to.
   */
  updatePreferences(userId, updates) {
    return apiCall("PUT", "/preferences", { user_id: userId, ...updates });
  },

  // ===================================================================
  // SYSTEM
  // ===================================================================

  /** GET /health */
  checkHealth() {
    return apiCall("GET", "/health");
  },

  /** GET / */
  getRoot() {
    return apiCall("GET", "/");
  },
};

export default ekaAPI;

// Named exports for the individual pieces, in case a caller wants them
// without going through the ekaAPI namespace object.
export { apiCall, qs };
