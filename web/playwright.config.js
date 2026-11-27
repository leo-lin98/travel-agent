import { defineConfig, devices } from "@playwright/test";

import { DEMO_ARGS, DEMO_COMMAND, DEMO_PORT, HOST, WEB_PORT } from "./offline-servers.js";

const webOrigin = `http://${HOST}:${WEB_PORT}`;

export default defineConfig({
  testDir: "tests",
  outputDir: "../artifacts/phase-3-browser-results",
  reporter: [
    ["list"],
    ["html", { outputFolder: "../artifacts/phase-3-browser", open: "never" }],
    ["junit", { outputFile: "../artifacts/phase-3-browser.xml" }],
  ],
  use: { baseURL: webOrigin },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: [
    {
      name: "demo",
      command: [DEMO_COMMAND, ...DEMO_ARGS].join(" "),
      wait: {
        stderr: new RegExp(`Uvicorn running on http://${HOST.replaceAll(".", "\\.")}:${DEMO_PORT}`),
      },
      reuseExistingServer: false,
    },
    {
      name: "vite",
      command: "pnpm exec vite",
      url: webOrigin,
      reuseExistingServer: false,
    },
  ],
});
