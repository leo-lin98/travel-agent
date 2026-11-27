/** Pure mapping of backend and browser errors to the alert's code and text (ui-contract.md §6). */
import { isObject, parseJson } from "./json.js";

const UI_MESSAGES = {
  microphone_denied:
    "We could not use your microphone. Allow microphone access to record, or type your message.",
  network_error: "We could not reach the travel agent. Please try again.",
  internal_error: "Something went wrong in this page. Your conversation is kept; please try again.",
  stored_context_unreadable:
    "Your saved conversation could not be read. Start a new conversation to continue.",
};

const OVERFLOW_TEXT = "This conversation is full. Start a new conversation to keep planning.";

/** A browser-side error with its fixed message. */
export function uiError(code) {
  return { code, message: UI_MESSAGES[code] };
}

/** The `{code, message}` of an HTTP error body, or a network error when the body has neither. */
export function errorFromBody(text) {
  const body = parseJson(text);
  if (isObject(body) && typeof body.code === "string" && typeof body.message === "string") {
    return { code: body.code, message: body.message };
  }
  return uiError("network_error");
}

/** The alert text: the error's message, except for overflow, which asks for a new conversation. */
export function errorText(error) {
  return error.code === "continuation_overflow" ? OVERFLOW_TEXT : error.message;
}
