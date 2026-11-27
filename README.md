# travel-agent

A travel-planning chat feature for the website.

Project documentation lives in [`docs/`](docs/), starting with the [requirements](docs/requirements.md), [implementation plan](docs/plan.md), [architecture](docs/architecture.md), [hosting decisions](docs/hosting.md), and [decision log](docs/decisions.md). Individual phase documents live in [docs/phased-development/](docs/phased-development/), linked in order from the plan.

The [planned files and directories](docs/plan.md#project-structure-and-configuration) are scaffolded, and empty folders contain `.gitkeep` markers so Git preserves the structure. Phases 0–2 are implemented in `backend/`: contracts, feasibility checks, option selection, the LangGraph agent loop, and the replay script, with offline tests. The other files are still empty placeholders filled in by their owning phases.

```text
backend/
├── src/travel_agent/
│   └── providers/       # external connectors
├── scripts/             # replay, evaluation, and offline demo runners
└── tests/
    ├── fixtures/        # synthetic cases and scripted responses
    ├── offline/         # network-free checks
    └── online/          # explicit live integration checks
web/
├── src/                 # React / JavaScript frontend
└── tests/               # browser and component checks
deploy/                  # deployment configuration
artifacts/               # local reports
```

## Backend setup and checks

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14. Run all commands from the repository root.

```bash
uv sync --locked --project backend            # create backend/.venv; fails if backend/uv.lock is stale
uv lock --check --project backend             # verify the lockfile is current
uv run --project backend pytest backend/tests/offline -q --junitxml=artifacts/phase-0.xml
uv run --project backend pytest backend/tests/offline/test_feasibility.py backend/tests/offline/test_selection.py -q --junitxml=artifacts/phase-1.xml
uv run --project backend python backend/scripts/replay.py --case backend/tests/fixtures/deadline_boundary.json --output artifacts/phase-1-example.json   # inspectable offline replay report
uv run --project backend pytest backend/tests/offline/test_agent.py -q --junitxml=artifacts/phase-2.xml
uv run --project backend python backend/scripts/replay.py --case backend/tests/fixtures/replan_conversation.json --output artifacts/phase-2-example.json   # scripted multi-turn conversation replay
uv run --project backend pytest backend/tests/offline/test_api.py -q --junitxml=artifacts/phase-3-api.xml
uv run --project backend python backend/scripts/demo_offline.py --port 8765   # offline demo backend on 127.0.0.1 only
uv run --project backend ruff check backend   # lint
uv run --project backend ruff format --check backend
uv run --project backend pyright -p backend   # strict type check of backend/src, backend/tests, and backend/scripts
```

Setup needs network access once: uv may download CPython 3.14, and the first pyright run may fetch Node through nodeenv. After that the checks run offline.

Offline tests run with `--disable-socket --allow-unix-socket`, so any TCP or UDP connection fails and no provider keys are needed. Unix sockets stay allowed because the asyncio event loop used by the API tests needs a local socket pair.

`replay.py` tells a revision case from a conversation case by the file's keys, so both use the same `--case` and `--output` options. A conversation case runs each turn through the agent loop with a scripted model, scripted providers, and a fixed clock. It exits with an error and writes no report if the loop makes a request the script does not cover or leaves scripted replies unused.

`demo_offline.py` serves the chat API (`POST /api/turn`, `POST /api/transcribe`) with a scripted model, scripted providers, a fixed clock, and a fake transcriber. It needs no network or provider keys, reads no environment variables, and always binds `127.0.0.1`; `--port` is required. The model follows a fixed two-turn script whatever the user types: a clarification, then three validated options. A third turn in one conversation ends with an `internal_error` stream event and a `demo_script_ended` line on stderr; start a new conversation to replay it.

## Frontend setup and checks

Requires Node and [pnpm](https://pnpm.io/) (the version is pinned in `web/package.json` under `packageManager`), plus the backend setup above. Run these commands from `web/`.

```bash
pnpm install --frozen-lockfile        # install the pinned dependencies; fails if pnpm-lock.yaml is stale
pnpm exec playwright install chromium # one-time download of the Chromium browser for Playwright
pnpm lint                             # ESLint
pnpm format:check                     # Prettier check; `pnpm format` rewrites files
pnpm test:unit                        # Vitest component logic tests in src/
pnpm test:offline                     # Playwright browser tests in tests/, against the offline demo
pnpm dev:offline                      # offline demo backend and Vite dev server on 127.0.0.1
```

Setup needs network access once, for the packages and the Chromium download. After that the checks run offline.

`test:offline` starts its own demo backend (port 8765) and Vite dev server (port 5173), so stop `dev:offline` first. Vite binds `127.0.0.1` and proxies `/api` to the demo at `127.0.0.1:8765`. Browser tests import `test` and `expect` from `tests/fixtures.js`, which aborts any request to a host other than `127.0.0.1` and fails the test; ESLint rejects direct `@playwright/test` imports in `tests/`. Reports go to `artifacts/phase-3-browser/` (HTML), `artifacts/phase-3-browser.xml` (JUnit), and `artifacts/phase-3-browser-results/` (traces and failure context).

`dev:offline` runs `backend/scripts/demo_offline.py` with only scripted test doubles and a fake transcriber. It is for local inspection only; it never calls a provider and needs no keys. Stop it with Ctrl+C, which stops both servers.
