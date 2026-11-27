/**
 * Phase 3 browser acceptance: a backend restart between turns (ui-contract.md §7).
 *
 * The test cannot restart the demo that Playwright's webServer owns, so after turn 1 it starts
 * a second demo process on another 127.0.0.1 port, one that never saw turn 1, and routes the
 * page's next /api/turn there with page.route and route.fetch. That is the closest honest
 * equivalent: the next turn is served by a different backend process, using only the browser's
 * stored context. route.fetch buffers the response, so this test does not check live status.
 * The request goes only to 127.0.0.1, so it stays offline.
 */
import { spawn } from "node:child_process";

import { DEMO_ARGS, DEMO_COMMAND, DEMO_PORT, HOST } from "../offline-servers.js";
import { expect, test } from "./fixtures.js";
import {
  FIRST_MESSAGE,
  SECOND_MESSAGE,
  messages,
  options,
  questions,
  sendTyped,
} from "./helpers.js";

const RESTARTED_PORT = DEMO_PORT + 1;
const START_TIMEOUT_MS = 60_000;

/** Starts a demo backend on `port` and resolves once uvicorn reports it is listening. */
function startDemo(port) {
  const args = DEMO_ARGS.map((arg) => (arg === String(DEMO_PORT) ? String(port) : arg));
  const child = spawn(DEMO_COMMAND, args, { stdio: ["ignore", "ignore", "pipe"] });
  const ready = new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error("restarted demo did not start")),
      START_TIMEOUT_MS,
    );
    child.stderr.on("data", (chunk) => {
      if (chunk.toString().includes(`Uvicorn running on http://${HOST}:${port}`)) {
        clearTimeout(timer);
        resolve();
      }
    });
    child.on("exit", (code) => {
      clearTimeout(timer);
      reject(new Error(`restarted demo exited with ${code}`));
    });
  });
  return { child, ready };
}

test("a different backend process continues the conversation from the browser's context", async ({
  page,
}) => {
  test.setTimeout(START_TIMEOUT_MS + 30_000);
  await page.goto("/");
  await sendTyped(page, FIRST_MESSAGE);
  await expect(questions(page).getByRole("listitem")).toHaveCount(2);

  const restarted = startDemo(RESTARTED_PORT);
  try {
    await restarted.ready;
    let served = 0;
    await page.route("**/api/turn", async (route) => {
      served += 1;
      const response = await route.fetch({ url: `http://${HOST}:${RESTARTED_PORT}/api/turn` });
      await route.fulfill({ response });
    });

    await sendTyped(page, SECOND_MESSAGE);

    expect(served).toBe(1);
    await expect(messages(page)).toHaveCount(4);
    await expect(options(page).getByRole("article")).toHaveCount(3);
  } finally {
    restarted.child.kill("SIGTERM");
  }
});
