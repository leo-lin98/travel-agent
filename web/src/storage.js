/** Pure reading and writing of the stored conversation (interface-contract.md §3, O4). */
import { isObject, parseJson } from "./json.js";
import { FRESH_CONTINUATION } from "./request.js";
import { isRenderableResult } from "./validate.js";

export const STORAGE_KEY = "travel-agent.conversation";

/** A stored conversation that cannot be read or rendered (ui-contract.md §7). */
export const UNREADABLE = { readable: false, continuation: null, lastResult: null };

/** Whether a stored `last_result` is null, or complete and renderable with `continuation`. */
function isStoredResult(lastResult, continuation) {
  if (lastResult === null) {
    return true;
  }
  return (
    isObject(lastResult) &&
    isObject(lastResult.estimate_labels) &&
    isRenderableResult(
      lastResult.result,
      lastResult.estimate_labels,
      isObject(continuation) ? continuation.request : undefined,
    )
  );
}

/**
 * Reads the raw stored value. Nothing stored is a fresh conversation. A value that is not a JSON
 * object, or whose last result cannot be rendered, is unreadable (TR1). Otherwise the
 * continuation is used as stored, and the backend validates it.
 */
export function readStored(raw) {
  if (raw === null) {
    return { readable: true, continuation: FRESH_CONTINUATION, lastResult: null };
  }
  const value = parseJson(raw);
  if (!isObject(value) || !isStoredResult(value.last_result, value.continuation)) {
    return UNREADABLE;
  }
  return { readable: true, continuation: value.continuation, lastResult: value.last_result };
}

/** The value stored after a result event, with exactly the two contract keys. */
export function storedValue(continuation, result, estimateLabels) {
  return JSON.stringify({ continuation, last_result: { result, estimate_labels: estimateLabels } });
}

/** The chat messages of a possibly invalid continuation that can be shown. */
export function displayMessages(continuation) {
  if (!isObject(continuation) || !Array.isArray(continuation.messages)) {
    return [];
  }
  return continuation.messages.filter(
    (message) =>
      isObject(message) && typeof message.role === "string" && typeof message.text === "string",
  );
}
