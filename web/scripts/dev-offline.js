/**
 * Runs the offline demo backend and the Vite dev server together on 127.0.0.1.
 *
 * The demo backend uses only scripted test doubles; this is for local inspection only.
 * When the demo backend exits, the Vite server closes and this process exits with its code.
 */
import { spawn } from "node:child_process";

import { createServer } from "vite";

import { DEMO_ARGS, DEMO_COMMAND } from "../offline-servers.js";

const demo = spawn(DEMO_COMMAND, DEMO_ARGS, { stdio: "inherit" });
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => demo.kill(signal));
}

const starting = createServer();
demo.on("exit", async (code) => {
  try {
    await (await starting).close();
  } finally {
    process.exit(code ?? 1);
  }
});

try {
  const vite = await starting;
  await vite.listen();
  vite.printUrls();
} catch (error) {
  demo.kill("SIGTERM");
  throw error;
}
