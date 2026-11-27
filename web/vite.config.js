import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

import { DEMO_PORT, HOST, WEB_PORT } from "./offline-servers.js";

export default defineConfig({
  plugins: [react()],
  server: {
    host: HOST,
    port: WEB_PORT,
    strictPort: true,
    proxy: { "/api": `http://${HOST}:${DEMO_PORT}` },
  },
  test: { include: ["src/**/*.test.{js,jsx}"] },
});
