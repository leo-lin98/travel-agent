/**
 * Shared helpers for the Phase 3 browser acceptance tests (artifacts/phase-3/ui-contract.md).
 *
 * Page shims installed here change only what the browser cannot produce offline: a synthetic
 * microphone (a Web Audio tone, because the fake media device's getUserMedia never resolves in
 * headless Chromium here), a denied microphone, and the pacing of the turn stream so each status
 * is visible for a moment. They never change the content the backend sends.
 */
import { expect } from "./fixtures.js";

export const STORAGE_KEY = "travel-agent.conversation";
export const MAX_MESSAGE_CHARS = 2000;
export const MAX_CLIP_MS = 30_000;

/** Fixed texts of the offline demo script (backend/scripts/demo_offline.py). */
export const DEMO = {
  transcript:
    "I'm free from 3pm and have dinner in Omotesando at 7pm. " +
    "I have 3000 to spend and would like something quiet.",
  clarification: "Two quick questions before I plan your afternoon.",
  questions: ["Where will you start?", "Which currency is your 3000 budget in?"],
  optionsReply: "Here are three quiet ways to spend the afternoon before dinner.",
};

export const FIRST_MESSAGE = "I'm free from 3pm until dinner in Omotesando at 7pm.";
export const SECOND_MESSAGE = "From Shibuya Station, and the budget is in yen.";

export const FRESH_CONTINUATION = {
  schema_version: 1,
  messages: [],
  request: {
    schema_version: 1,
    gap: { start_time: null, start_location: null },
    commitment: { location: null, time: null, timezone: null },
    budget: null,
    transport_modes: [],
    constraints: [],
    unresolved_questions: [],
  },
};

export const conversation = (page) => page.getByRole("log", { name: "Conversation" });
export const messages = (page) => conversation(page).getByTestId("message");
export const userMessages = (page) =>
  conversation(page).locator('[data-testid="message"][data-role="user"]');
export const options = (page) => page.getByRole("region", { name: "Itinerary options" });
export const option = (page, name) => options(page).getByRole("article", { name, exact: true });
export const questions = (page) => page.getByRole("list", { name: "Questions to answer" });
export const input = (page) => page.getByRole("textbox", { name: "Message" });
export const alert = (page) => page.getByRole("alert");

const isTurn = (request) => request.url().endsWith("/api/turn") && request.method() === "POST";
const isTranscribe = (request) =>
  request.url().endsWith("/api/transcribe") && request.method() === "POST";

/** Records the JSON body of every /api/turn request the page sends. */
export function collectTurnBodies(page) {
  const bodies = [];
  page.on("request", (request) => {
    if (isTurn(request)) {
      bodies.push(request.postDataJSON());
    }
  });
  return bodies;
}

/** Sends the current input; resolves with the request body once the response has finished. */
export async function send(page) {
  const requested = page.waitForRequest(isTurn);
  await page.getByRole("button", { name: "Send" }).click();
  const request = await requested;
  const response = await request.response();
  await response.finished();
  return request.postDataJSON();
}

/** Types `text` and sends it; resolves with the request body. */
export async function sendTyped(page, text) {
  await input(page).fill(text);
  return send(page);
}

/** Records one clip and waits until the transcript is in the input; returns the input value. */
export async function recordClip(page) {
  await page.getByRole("button", { name: "Record message" }).click();
  const uploaded = page.waitForResponse((response) => isTranscribe(response.request()));
  await page.getByRole("button", { name: "Stop recording" }).click();
  await uploaded;
  return input(page);
}

/** Runs the demo's two turns by typing and waits for the three options. */
export async function completeTypedFlow(page) {
  await sendTyped(page, FIRST_MESSAGE);
  await expect(questions(page).getByRole("listitem")).toHaveCount(2);
  await sendTyped(page, SECOND_MESSAGE);
  await expect(options(page).getByRole("article")).toHaveCount(3);
}

export const readStorage = (page) =>
  page.evaluate((key) => globalThis.sessionStorage.getItem(key), STORAGE_KEY);

/** Puts `value` in session storage before the app loads, unless the key already exists. */
export async function seedStorage(page, value) {
  await page.addInitScript(
    ([key, stored]) => {
      if (globalThis.sessionStorage.getItem(key) === null) {
        globalThis.sessionStorage.setItem(key, stored);
      }
    },
    [STORAGE_KEY, value],
  );
}

/** Replaces getUserMedia with a synthetic tone, so recording works without a real device. */
export async function useSyntheticMicrophone(page) {
  await page.addInitScript(() => {
    globalThis.navigator.mediaDevices.getUserMedia = async () => {
      const context = new globalThis.AudioContext();
      const tone = context.createOscillator();
      const destination = context.createMediaStreamDestination();
      tone.connect(destination);
      tone.start();
      return destination.stream;
    };
  });
}

/** Makes getUserMedia fail as when the user denies the microphone. */
export async function denyMicrophone(page) {
  await page.addInitScript(() => {
    globalThis.navigator.mediaDevices.getUserMedia = async () => {
      throw new globalThis.DOMException("Permission denied", "NotAllowedError");
    };
  });
}

/** Delivers each /api/turn NDJSON line `delayMs` after the previous one, unchanged. */
export async function paceTurnStream(page, delayMs) {
  await page.addInitScript((delay) => {
    const original = globalThis.fetch.bind(globalThis);
    globalThis.fetch = async (resource, init) => {
      const response = await original(resource, init);
      const url = typeof resource === "string" ? resource : resource.url;
      if (!url.endsWith("/api/turn") || !response.ok || response.body === null) {
        return response;
      }
      const reader = response.body.getReader();
      const decoder = new globalThis.TextDecoder();
      const encoder = new globalThis.TextEncoder();
      let buffer = "";
      const paced = new globalThis.ReadableStream({
        async pull(controller) {
          for (;;) {
            const end = buffer.indexOf("\n");
            if (end >= 0) {
              const line = buffer.slice(0, end + 1);
              buffer = buffer.slice(end + 1);
              await new Promise((resolve) => globalThis.setTimeout(resolve, delay));
              controller.enqueue(encoder.encode(line));
              return;
            }
            const { done, value } = await reader.read();
            if (done) {
              if (buffer) {
                controller.enqueue(encoder.encode(buffer));
              }
              controller.close();
              return;
            }
            buffer += decoder.decode(value, { stream: true });
          }
        },
      });
      return new globalThis.Response(paced, {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    };
  }, delayMs);
}

/** Logs every change of the turn-status text ("" when absent) in page order. */
export async function logTurnStatus(page) {
  await page.addInitScript(() => {
    globalThis.turnStatusLog = [""];
    const record = () => {
      const element = globalThis.document.querySelector('[data-testid="turn-status"]');
      const text = element === null ? "" : element.textContent.trim();
      if (text !== globalThis.turnStatusLog.at(-1)) {
        globalThis.turnStatusLog.push(text);
      }
    };
    new globalThis.MutationObserver(record).observe(globalThis.document, {
      subtree: true,
      childList: true,
      characterData: true,
    });
  });
}

/** The statuses shown during each finished turn, from the turn-status log. */
export async function shownStatusesPerTurn(page) {
  const log = await page.evaluate(() => globalThis.turnStatusLog);
  const turns = [];
  let current = [];
  for (const text of log.slice(1)) {
    if (text === "") {
      turns.push(current);
      current = [];
    } else {
      current.push(text);
    }
  }
  return turns;
}

/** The last NDJSON event of a finished /api/turn response body. */
export function lastEvent(body) {
  const lines = body.trim().split("\n");
  return JSON.parse(lines.at(-1));
}
