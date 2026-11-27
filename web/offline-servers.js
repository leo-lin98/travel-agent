/** Hosts, ports, and commands for the offline demo backend and the Vite dev server. */

export const HOST = "127.0.0.1";
export const DEMO_PORT = 8765;
export const WEB_PORT = 5173;
export const DEMO_COMMAND = "uv";
export const DEMO_ARGS = [
  "run",
  "--project",
  "../backend",
  "python",
  "../backend/scripts/demo_offline.py",
  "--port",
  String(DEMO_PORT),
];
