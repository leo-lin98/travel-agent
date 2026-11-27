/**
 * The browser's calls to the backend (interface-contract.md §2). `fetchFn` is passed in so the
 * caller decides which fetch runs. A failed fetch or read, or an unparseable body, resolves as
 * `network_error`; any other exception propagates, so a bug is never shown as network trouble.
 */
import { errorFromBody, uiError } from "./errors.js";
import { takeEvents } from "./stream.js";

const NETWORK_ERROR = { type: "error", error: uiError("network_error") };

/** Marks a fetch, read, or parse that failed; never a value the backend can send. */
const FAILED = Symbol("transport failed");

/** Runs `operation`, returning FAILED when it throws a fetch or read failure (TypeError). */
async function transport(operation) {
  try {
    return await operation();
  } catch (error) {
    if (error instanceof TypeError) {
      return FAILED;
    }
    throw error;
  }
}

/** Runs `parse`, returning FAILED when the text is not JSON (SyntaxError). */
function parsed(parse) {
  try {
    return parse();
  } catch (error) {
    if (error instanceof SyntaxError) {
      return FAILED;
    }
    throw error;
  }
}

/** Cancels a stream that broke the contract, which may still be sending, and fails the turn. */
async function abandon(reader) {
  await transport(() => reader.cancel());
  return NETWORK_ERROR;
}

/**
 * Reads NDJSON events, reporting statuses live, until the terminal event or the end. After the
 * terminal event the server closes the stream (contract rule 2), so it is not cancelled.
 */
async function readTurnEvents(reader, onStatus) {
  let buffer = "";
  for (;;) {
    const chunk = await transport(() => reader.read());
    if (chunk === FAILED || chunk.done) {
      return NETWORK_ERROR;
    }
    const taken = parsed(() => takeEvents(buffer + chunk.value));
    if (taken === FAILED) {
      return abandon(reader);
    }
    buffer = taken.rest;
    for (const event of taken.events) {
      if (event.type === "status") {
        onStatus(event.status);
      } else if (event.type === "result") {
        return event;
      } else if (event.type === "error") {
        return errorEvent(event);
      } else {
        return abandon(reader);
      }
    }
  }
}

/** The outcome of an error event; one without a string code and message is an internal error. */
function errorEvent(event) {
  if (typeof event.code !== "string" || typeof event.message !== "string") {
    return { type: "error", error: uiError("internal_error") };
  }
  return { type: "error", error: { code: event.code, message: event.message } };
}

/** The `{type: "error"}` outcome of a non-OK response. */
async function httpError(response) {
  const text = await transport(() => response.text());
  return { type: "error", error: text === FAILED ? uiError("network_error") : errorFromBody(text) };
}

/**
 * Posts one turn and streams its events. Calls `onStatus` with each status as it arrives and
 * resolves with the result event or `{type: "error", error: {code, message}}`.
 */
export async function streamTurn(fetchFn, body, onStatus) {
  const response = await transport(() =>
    fetchFn("/api/turn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
  if (response === FAILED) {
    return NETWORK_ERROR;
  }
  if (!response.ok) {
    return httpError(response);
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  return readTurnEvents(reader, onStatus);
}

/** Posts one audio clip as the raw body; resolves with the transcript or the error. */
export async function transcribe(fetchFn, clip) {
  const response = await transport(() =>
    fetchFn("/api/transcribe", {
      method: "POST",
      headers: { "Content-Type": clip.type },
      body: clip,
    }),
  );
  if (response === FAILED) {
    return NETWORK_ERROR;
  }
  if (response.status !== 200) {
    return httpError(response);
  }
  const text = await transport(() => response.text());
  const reply = text === FAILED ? FAILED : parsed(() => JSON.parse(text));
  if (reply === FAILED) {
    return NETWORK_ERROR;
  }
  if (typeof reply?.transcript !== "string") {
    return { type: "error", error: uiError("internal_error") };
  }
  return { type: "transcript", text: reply.transcript };
}
