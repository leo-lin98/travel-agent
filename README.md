# travel-agent

A travel-planning chat feature for the website.

Project documentation lives in [`docs/`](docs/), starting with the [requirements](docs/requirements.md), [implementation plan](docs/plan.md), [architecture](docs/architecture.md), [hosting decisions](docs/hosting.md), and [decision log](docs/decisions.md). Individual phase documents live in [docs/phased-development/](docs/phased-development/), linked in order from the plan.

The [planned files and directories](docs/plan.md#project-structure-and-configuration) are scaffolded with empty file placeholders. Empty folders contain `.gitkeep` markers so Git preserves the structure; implementation, dependencies, and tooling will be added in their owning phases.

```text
backend/
├── src/travel_agent/
│   └── providers/       # external connectors
├── scripts/             # replay and evaluation runners
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
uv run --project backend ruff check backend   # lint
uv run --project backend ruff format --check backend
uv run --project backend pyright -p backend   # strict type check of backend/src, backend/tests, and backend/scripts
```

Setup needs network access once: uv may download CPython 3.14, and the first pyright run may fetch Node through nodeenv. After that the checks run offline.

Offline tests run with `--disable-socket`, so any outbound network call fails and no provider keys are needed.
