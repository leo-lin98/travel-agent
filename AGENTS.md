# Repository Guidelines

## Collaboration & Decision Ownership

Read this `AGENTS.md` before taking any action. Keep it updated as requirements and decisions evolve; distinguish confirmed decisions from open questions.

The user owns consequential system design decisions and will implement the project to develop engineering skills. Ask about ambiguity that materially changes behavior, scope, security boundaries, compatibility, or acceptance criteria. Resolve routine implementation details within the approved specification and record consequential choices. Push back on unnecessary complexity. Do not turn every hypothetical edge case into a requirements question.

The requested deliverable is a phased implementation plan for a chat feature on the website, indexed at `docs/plan.md` with individual phases in `docs/phased-development/`. Gather requirements and user decisions before finalizing it. Do not implement the feature unless asked.

## Decision Logging

Whenever the user confirms or changes a project decision, update [docs/decisions.md](docs/decisions.md) in the same task without waiting for a separate request to log it. Begin with future decisions; do not backfill earlier decisions unless the user asks.

- Record the date, confirmed decision, and the problem or circumstances that required the choice.
- Include the user's stated reasoning, alternatives discussed, and accepted tradeoffs when available. Mark missing reasoning as not recorded; never invent reasons or attribute agent suggestions to the user.
- Link to affected requirements, architecture, or phase documents instead of duplicating their full specification. Update those documents when the confirmed decision changes them.
- Keep proposals and unanswered questions distinct from confirmed decisions. If confirmation is ambiguous and materially affects the project, clarify before recording it as confirmed.
- Preserve decision history. When a new decision replaces an earlier one, append the new decision and mark the earlier entry as superseded with a link to its replacement.
- Keep entries free of secrets and sensitive values.

## Product and Architecture Context

Read [docs/requirements.md](docs/requirements.md) for confirmed product requirements before planning or implementing changes. Read [docs/plan.md](docs/plan.md) for the selected stack, shared execution rules, and phase sequencing. Before working on a phase, read its linked document in [docs/phased-development/](docs/phased-development/) for deliverables, implementation steps, tests, and acceptance criteria. Read [docs/architecture.md](docs/architecture.md) for the agent loop, tool contracts, data lifecycle, and provider-data permissions. [docs/hosting.md](docs/hosting.md) owns dated hosting research. Keep those as the sources of truth rather than duplicating the product specification here.

The user confirmed four development principles:

1. Preserve firm product constraints, including one gap, arrival buffer, stated budget, verified opening/access, distinct alternatives, and replanning before departure.
2. Give the agent discretion over research, tool selection, necessary clarification, and repairing proposals after validation fails.
3. Keep Python authoritative over schedules, costs, evidence requirements, and hard-constraint rejection.
4. Use realistic evaluations to discover missing rules and justify additional complexity.

Use one minimal LangGraph agent/tool loop. Do not add specialized agents, fixed questionnaires, or a node for every edge case. Keep the selected stack unless a consequential change is approved. No feature implementation is authorized by this documentation work.

The plan must use the general phase format of the portfolio project's `docs/rag-plan.md`, without importing its content or design. Group phases into **Offline testing**, **Online testing**, and **Deployment**. Every phase must be locally testable without network access after setup, produce inspectable results, and require no later phase to verify it. Live provider and deployment checks are separate additional gates. Deployment includes rate limiting and security hardening. Do not claim scripted offline tests establish live model quality.

## Project Structure & Module Organization

This repository is currently a minimal scaffold for `travel-agent`:

- `README.md`: project entry point with a link to the planning documentation.
- `LICENSE`: repository licensing terms.
- `AGENTS.md`: contributor and agent guidance.
- `docs/decisions.md`: log of future confirmed user decisions and their context; initially empty.
- `docs/requirements.md`: confirmed product requirements and stable requirement IDs.
- `docs/plan.md`: selected stack, shared execution rules, and phase index in the three requested buckets.
- `docs/phased-development/`: individual phase documents owning deliverables, implementation steps, tests, and acceptance criteria.
- `docs/architecture.md`: minimal agent loop, Python authority, tool/data boundaries, and integration gates.
- `docs/hosting.md`: selected Render and Langfuse Cloud setup, accepted cold-start tradeoff, and deferred database research.

The files and directories for `backend/`, `web/`, `deploy/`, and `artifacts/` are scaffolded as documented in `docs/plan.md`, with empty implementation files and `.gitkeep` markers preserving empty folders. Source files and manifests are placeholders; no feature implementation, tests, assets, or configured development commands exist yet. The user authorized creating the listed files, not implementing the planned phases. Document new directories in `README.md`. Consult `docs/requirements.md`, `docs/plan.md`, and `docs/architecture.md` before architecture changes.

## Build, Test, and Development Commands

No build, test, lint, or local development commands are configured. Do not assume `npm test`, `pytest`, or another framework is available.

Useful checks from the repository root:

- `git status --short`: review changed and untracked files.
- `git diff --check`: detect whitespace errors in tracked changes.
- `git diff`: inspect unstaged changes before review.

When introducing tooling, declare dependencies in the appropriate project manifest and document reproducible setup, run, lint, and test commands. Install dependencies into the project environment, never globally.

## Coding Style & Naming Conventions

Python is selected; formatting, linting, indentation, and naming conventions are not yet established. Match surrounding style when code exists.

Keep changes focused. Prefer small, single-purpose functions and explicit parameters. Avoid mutating inputs or global state, speculative abstractions, and unrelated refactors. Validate external input at system boundaries and report actionable errors.

## Testing Guidelines

Pytest is selected but not configured; no coverage threshold exists. Add requirement-derived tests for new core logic, including meaningful failure cases. Prefer fast tests for pure functions and integration tests for external boundaries. Document test naming and execution commands when configuring tests. Run applicable lint and test checks before requesting review.

## Commit & Pull Request Guidelines

Git history contains only `Initial commit`; no commit convention is established. Use concise, imperative commit subjects describing the change. Agents must not commit, push, merge, or publish without explicit authorization.

Pull requests should explain the problem, scope, key decisions, and verification results. Link relevant issues and include screenshots for visual changes.

## Security & Configuration

Never commit or expose secrets. Do not inspect or modify `.env` files, secret or credential files, private keys, or cloud authentication directories. Keep configuration examples free of sensitive values.
