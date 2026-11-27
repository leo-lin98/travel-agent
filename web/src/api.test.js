import { describe, expect, test } from "vitest";

import { streamTurn, transcribe } from "./api.js";
import { uiError } from "./errors.js";

const BODY = { continuation: {}, message: "hi", user_timezone: "UTC", device_location: null };
const RESULT = {
  type: "result",
  result: { kind: "clarification" },
  continuation: { messages: [] },
  estimate_labels: {},
};

/** A response whose body delivers `chunks` one at a time. */
function streamed(chunks) {
  const encoder = new TextEncoder();
  const body = new ReadableStream({
    start(controller) {
      chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)));
      controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { "Content-Type": "application/x-ndjson" } });
}

const ndjson = (...events) => events.map((event) => `${JSON.stringify(event)}\n`).join("");

/** A fetch double that records its calls and answers with `respond()`. */
function fakeFetch(respond) {
  const calls = [];
  const fetchFn = async (url, init) => {
    calls.push({ url, init });
    return respond();
  };
  return { calls, fetchFn };
}

describe("streamTurn", () => {
  test("posts the body as JSON to /api/turn", async () => {
    const { calls, fetchFn } = fakeFetch(() => streamed([ndjson(RESULT)]));
    await streamTurn(fetchFn, BODY, () => {});
    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe("/api/turn");
    expect(calls[0].init.method).toBe("POST");
    expect(calls[0].init.headers).toEqual({ "Content-Type": "application/json" });
    expect(JSON.parse(calls[0].init.body)).toEqual(BODY);
  });

  test("reports each status as it arrives and returns the result event", async () => {
    const statuses = [];
    const text = ndjson(
      { type: "status", status: "Researching" },
      { type: "status", status: "Planning" },
      RESULT,
    );
    const { fetchFn } = fakeFetch(() =>
      streamed([text.slice(0, 7), text.slice(7, 60), text.slice(60)]),
    );
    const outcome = await streamTurn(fetchFn, BODY, (status) => statuses.push(status));
    expect(statuses).toEqual(["Researching", "Planning"]);
    expect(outcome).toEqual(RESULT);
  });

  test("returns an error event's code and message", async () => {
    const event = { type: "error", code: "internal_error", message: "Something went wrong." };
    const { fetchFn } = fakeFetch(() => streamed([ndjson(event)]));
    expect(await streamTurn(fetchFn, BODY, () => {})).toEqual({
      type: "error",
      error: { code: "internal_error", message: "Something went wrong." },
    });
  });

  test.each([
    ["no code", { type: "error", message: "Something went wrong." }],
    ["a numeric code", { type: "error", code: 500, message: "Something went wrong." }],
    ["no message", { type: "error", code: "internal_error" }],
  ])("an error event with %s is an internal error", async (_name, event) => {
    const { fetchFn } = fakeFetch(() => streamed([ndjson(event)]));
    expect(await streamTurn(fetchFn, BODY, () => {})).toEqual({
      type: "error",
      error: uiError("internal_error"),
    });
  });

  test("returns an HTTP error body's code and message", async () => {
    const body = { code: "continuation_overflow", message: "Too long." };
    const { fetchFn } = fakeFetch(() => new Response(JSON.stringify(body), { status: 422 }));
    expect(await streamTurn(fetchFn, BODY, () => {})).toEqual({ type: "error", error: body });
  });

  test("a stream that ends without a terminal event is a network error", async () => {
    const { fetchFn } = fakeFetch(() => streamed([ndjson({ type: "status", status: "Planning" })]));
    expect(await streamTurn(fetchFn, BODY, () => {})).toEqual({
      type: "error",
      error: uiError("network_error"),
    });
  });

  test("an event outside the contract is a network error, not skipped", async () => {
    const { fetchFn } = fakeFetch(() => streamed([ndjson({ type: "debug" }, RESULT)]));
    expect(await streamTurn(fetchFn, BODY, () => {})).toEqual({
      type: "error",
      error: uiError("network_error"),
    });
  });

  test("a failure outside fetch, reading, and parsing is not reported as a network error", async () => {
    const { fetchFn } = fakeFetch(() => streamed([ndjson({ type: "status", status: "Planning" })]));
    const bug = new TypeError("bug in the status handler");
    await expect(
      streamTurn(fetchFn, BODY, () => {
        throw bug;
      }),
    ).rejects.toBe(bug);
  });

  test.each([
    ["an event outside the contract", ndjson({ type: "debug" })],
    ["an unparseable line", "{broken\n"],
  ])("the stream is cancelled when reading stops at %s", async (_name, text) => {
    let cancelled;
    const wasCancelled = new Promise((resolve) => (cancelled = resolve));
    const body = new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(text));
      },
      cancel() {
        cancelled(true);
      },
    });
    const { fetchFn } = fakeFetch(() => new Response(body, { status: 200 }));
    expect(await streamTurn(fetchFn, BODY, () => {})).toEqual({
      type: "error",
      error: uiError("network_error"),
    });
    expect(await wasCancelled).toBe(true);
  });

  test("a failed fetch or an unreadable line is a network error", async () => {
    const failing = async () => {
      throw new TypeError("Failed to fetch");
    };
    const { fetchFn } = fakeFetch(() => streamed(["{broken\n"]));
    const expected = { type: "error", error: uiError("network_error") };
    expect(await streamTurn(failing, BODY, () => {})).toEqual(expected);
    expect(await streamTurn(fetchFn, BODY, () => {})).toEqual(expected);
  });
});

describe("transcribe", () => {
  const clip = new Blob([new Uint8Array([1, 2, 3])], { type: "audio/webm;codecs=opus" });

  test("posts the raw clip with its type and returns the transcript", async () => {
    const { calls, fetchFn } = fakeFetch(
      () => new Response(JSON.stringify({ transcript: "From Shibuya." }), { status: 200 }),
    );
    expect(await transcribe(fetchFn, clip)).toEqual({ type: "transcript", text: "From Shibuya." });
    expect(calls[0].url).toBe("/api/transcribe");
    expect(calls[0].init.method).toBe("POST");
    expect(calls[0].init.headers).toEqual({ "Content-Type": "audio/webm;codecs=opus" });
    expect(calls[0].init.body).toBe(clip);
  });

  test("returns the error of any non-200 response", async () => {
    const body = { code: "transcription_failed", message: "Try again." };
    const { fetchFn } = fakeFetch(() => new Response(JSON.stringify(body), { status: 502 }));
    expect(await transcribe(fetchFn, clip)).toEqual({ type: "error", error: body });
  });

  test("a 200 without a string transcript is an internal error", async () => {
    const { fetchFn } = fakeFetch(() => new Response(JSON.stringify({ transcript: 5 })));
    expect(await transcribe(fetchFn, clip)).toEqual({
      type: "error",
      error: uiError("internal_error"),
    });
  });

  test("a 200 whose body is not JSON is a network error", async () => {
    const { fetchFn } = fakeFetch(() => new Response("<html>proxy page</html>"));
    expect(await transcribe(fetchFn, clip)).toEqual({
      type: "error",
      error: uiError("network_error"),
    });
  });

  test("a failed fetch is a network error", async () => {
    const failing = async () => {
      throw new TypeError("Failed to fetch");
    };
    expect(await transcribe(failing, clip)).toEqual({
      type: "error",
      error: uiError("network_error"),
    });
  });
});
