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
