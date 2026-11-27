import { describe, expect, test } from "vitest";

import { FRESH_CONTINUATION } from "./request.js";
import { displayMessages, readStored, storedValue } from "./storage.js";

describe("readStored", () => {
  test("nothing stored is a fresh conversation", () => {
    expect(readStored(null)).toEqual({
      readable: true,
      continuation: FRESH_CONTINUATION,
      lastResult: null,
    });
  });

  test("a stored object gives its continuation and last result as stored", () => {
    const continuation = { messages: [], extra: "kept for the backend to judge" };
    const result = { kind: "clarification", message: "m", questions: [{ question: "Where?" }] };
    const lastResult = { result, estimate_labels: {} };
    expect(readStored(JSON.stringify({ continuation, last_result: lastResult }))).toEqual({
      readable: true,
      continuation,
      lastResult,
    });
  });

  test("a stored invalid continuation with no last result is kept for the backend to judge", () => {
    const continuation = { messages: "tampered" };
    expect(readStored(JSON.stringify({ continuation, last_result: null }))).toEqual({
      readable: true,
      continuation,
      lastResult: null,
    });
  });

  const clarification = { kind: "clarification", message: "m", questions: [{ question: "Q?" }] };
  test.each([
    ["a missing last result", { continuation: {} }],
    ["a last result that is not an object", { continuation: {}, last_result: "done" }],
    [
      "a result that cannot be rendered",
      { continuation: {}, last_result: { result: { kind: "clarification" }, estimate_labels: {} } },
    ],
    [
      "options without the continuation labels",
      {
        continuation: { request: null },
        last_result: { result: { kind: "validated_options", options: [] }, estimate_labels: {} },
      },
    ],
    [
      "a clarification without estimate labels",
      { continuation: {}, last_result: { result: clarification } },
    ],
  ])("%s is unreadable", (_name, value) => {
    expect(readStored(JSON.stringify(value))).toEqual({
      readable: false,
      continuation: null,
      lastResult: null,
    });
  });

  test.each(["{not json", "null", "[]", "5", '"text"'])("%s is unreadable, not replaced", (raw) => {
    expect(readStored(raw)).toEqual({ readable: false, continuation: null, lastResult: null });
  });
});

test("the stored value has exactly the continuation and the last result", () => {
  const continuation = { messages: [{ role: "user", text: "hi" }] };
  const result = { kind: "clarification", message: "Where?", questions: [] };
  const labels = { "ev-a": "estimate" };
  expect(JSON.parse(storedValue(continuation, result, labels))).toEqual({
    continuation,
    last_result: { result, estimate_labels: labels },
  });
});

describe("displayMessages", () => {
  test("shows messages with a string role and text, in order", () => {
    const continuation = {
      messages: [
        { role: "user", text: "One" },
        { role: 3, text: "bad role" },
        { role: "assistant" },
        { role: "assistant", text: "Two" },
      ],
    };
    expect(displayMessages(continuation)).toEqual([
      { role: "user", text: "One" },
      { role: "assistant", text: "Two" },
    ]);
  });

  test.each([undefined, null, {}, { messages: "no" }, { messages: [null, 1] }])(
    "shows nothing it cannot read in %j",
    (continuation) => {
      expect(displayMessages(continuation)).toEqual([]);
    },
  );
});
