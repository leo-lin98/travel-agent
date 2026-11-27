import { useId, useRef, useState } from "react";

import { streamTurn, transcribe } from "./api.js";
import { Conversation, LastResult } from "./components.jsx";
import { appendTranscript } from "./draft.js";
import { errorText, uiError } from "./errors.js";
import { MAX_CLIP_SECONDS, startClip } from "./microphone.js";
import { requestDeviceLocation } from "./location.js";
import { turnBody } from "./request.js";
import { STORAGE_KEY, UNREADABLE, displayMessages, readStored, storedValue } from "./storage.js";
import { isResultEvent } from "./validate.js";

/** Longest message the user can send (interface-contract.md MAX_MESSAGE_CHARS). */
const MAX_MESSAGE_CHARS = 2000;

const NO_LOCATION = { state: "not_shared", deviceLocation: null };

/** Reads the stored conversation; storage that cannot be accessed is unreadable, not a crash. */
function readSavedConversation() {
  try {
    return readStored(globalThis.sessionStorage.getItem(STORAGE_KEY));
  } catch (error) {
    console.error("stored_conversation_read_failed", { error });
    return UNREADABLE;
  }
}

/**
 * The website chat (ui-contract.md). `activity` is one of "idle", "turn", "requesting_microphone",
 * "recording", or "transcribing"; only one runs at a time.
 */
export default function App() {
  const messageId = useId();
  const [saved, setSaved] = useState(readSavedConversation);
  const [error, setError] = useState(() =>
    saved.readable ? null : uiError("stored_context_unreadable"),
  );
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(null);
  const [status, setStatus] = useState("");
  const [activity, setActivity] = useState("idle");
  const [location, setLocation] = useState(NO_LOCATION);
  const stopRecording = useRef(null);

  const idle = activity === "idle";
  const canSend = idle && saved.readable && draft.trim() !== "";
  const canRecord = saved.readable && (idle || activity === "recording");

  /** Shows `turnError` and puts `message` back in the input; storage is left as it was. */
  function failTurn(message, turnError) {
    setDraft(message);
    setError(turnError);
  }

  async function sendTurn(event) {
    event.preventDefault();
    const message = draft;
    setDraft("");
    setPending(message);
    setActivity("turn");
    try {
      const body = turnBody(
        saved.continuation,
        message,
        Intl.DateTimeFormat().resolvedOptions().timeZone,
        location.deviceLocation,
      );
      const outcome = await streamTurn(globalThis.fetch, body, setStatus);
      if (outcome.type === "error") {
        failTurn(message, outcome.error);
        return;
      }
      if (!isResultEvent(outcome)) {
        console.error("turn_result_malformed");
        failTurn(message, uiError("internal_error"));
        return;
      }
      const { continuation, result, estimate_labels: estimateLabels } = outcome;
      globalThis.sessionStorage.setItem(
        STORAGE_KEY,
        storedValue(continuation, result, estimateLabels),
      );
      setSaved({
        readable: true,
        continuation,
        lastResult: { result, estimate_labels: estimateLabels },
      });
      setError(null);
    } catch (failure) {
      console.error("turn_failed_in_browser", { error: failure });
      failTurn(message, uiError("internal_error"));
    } finally {
      setStatus("");
      setPending(null);
      setActivity("idle");
    }
  }

  async function recordMessage() {
    setActivity("requesting_microphone");
    let session;
    try {
      session = await startClip(globalThis.navigator.mediaDevices, MAX_CLIP_SECONDS);
    } catch {
      setActivity("idle");
      setError(uiError("microphone_denied"));
      return;
    }
    try {
      stopRecording.current = session.stop;
      setActivity("recording");
      const clip = await session.clip;
      stopRecording.current = null;
      setActivity("transcribing");
      const outcome = await transcribe(globalThis.fetch, clip);
      if (outcome.type === "error") {
        setError(outcome.error);
        return;
      }
      setDraft((current) => appendTranscript(current, outcome.text));
      setError(null);
    } catch (failure) {
      console.error("transcription_failed_in_browser", { error: failure });
      setError(uiError("internal_error"));
    } finally {
      stopRecording.current = null;
      setActivity("idle");
    }
  }

  function toggleRecording() {
    if (activity === "recording") {
      stopRecording.current?.();
    } else {
      recordMessage();
    }
  }

  /** Asks for the device location, or withdraws consent when it is already shared. */
  function toggleLocation() {
    if (location.state === "shared") {
      setLocation(NO_LOCATION);
      return;
    }
    requestDeviceLocation(globalThis.navigator.geolocation, setLocation);
  }

  function startNewConversation() {
    try {
      globalThis.sessionStorage.removeItem(STORAGE_KEY);
    } catch (failure) {
      console.error("stored_conversation_remove_failed", { error: failure });
      setError(uiError("internal_error"));
      return;
    }
    setSaved(readStored(null));
    setError(null);
    setDraft("");
  }

  return (
    <main className="chat">
      <h1>Travel agent</h1>
      <Conversation messages={displayMessages(saved.continuation)} pending={pending} />
      {saved.readable && (
        <LastResult lastResult={saved.lastResult} request={saved.continuation?.request} />
      )}
      <p role="status" data-testid="turn-status" className="status">
        {status}
      </p>
      {error !== null && (
        <p role="alert" data-error-code={error.code} className="error">
          {errorText(error)}
        </p>
      )}
      <form onSubmit={sendTurn} className="composer">
        <label htmlFor={messageId}>Message</label>
        <textarea
          id={messageId}
          value={draft}
          maxLength={MAX_MESSAGE_CHARS}
          readOnly={activity === "turn"}
          rows={3}
          onChange={(event) => setDraft(event.target.value)}
        />
        <div className="actions">
          <button type="submit" disabled={!canSend}>
            Send
          </button>
          <button type="button" disabled={!canRecord} onClick={toggleRecording}>
            {activity === "recording" ? "Stop recording" : "Record message"}
          </button>
          <button type="button" aria-pressed={location.state === "shared"} onClick={toggleLocation}>
            Share my location
          </button>
          <button type="button" disabled={activity !== "idle"} onClick={startNewConversation}>
            New conversation
          </button>
        </div>
      </form>
      <p data-testid="location-notice" aria-live="polite" className="notice">
        {location.state === "denied" &&
          "Your location was not shared. Please tell us your starting location."}
      </p>
    </main>
  );
}
