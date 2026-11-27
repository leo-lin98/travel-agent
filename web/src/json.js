/** Pure helpers for reading untrusted JSON text. */

/** Parses JSON, returning undefined (never a JSON value) when the text is not JSON. */
export function parseJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

/** Whether `value` is a plain JSON object (not null and not an array). */
export const isObject = (value) =>
  typeof value === "object" && value !== null && !Array.isArray(value);
