# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

`AGENTS.md`, imported above, owns the working rules: decision ownership, decision logging, and commit and security policy. This file adds the commands and the architecture picture that span several documents.

## Current State

Phases 0 and 1 are implemented in the backend: `pyproject.toml` and `uv.lock`, the typed contracts (`contracts.py`), Python feasibility checks (`feasibility.py`), option selection (`selection.py`), the offline replay script (`scripts/replay.py`), and their tests in `backend/tests/offline/`. Everything else is still a 0-byte placeholder or `.gitkeep`: `agent.py`, `api.py`, `security.py`, `telemetry.py`, `providers/`, `scripts/evaluate.py`, `backend/tests/online/`, and all of `web/` and `deploy/`, including `package.json` and `Dockerfile`. The root `.gitignore` excludes `docs/`, `artifacts/`, `backend/tests/cassettes/`, and `.DS_Store`, so the planning documents exist locally but are not committed. Feature implementation needs explicit user authorization.

## Commands

The offline suite, single-test, and replay commands below work now. The live checks are targets for Phases 4 and 5 and do not exist yet. `README.md` lists setup, lint, format, and type-check commands. Run backend commands from the repository root.

```bash
# Offline suite (no network or provider keys)
uv run --project backend pytest backend/tests/offline -q --junitxml=artifacts/<phase-report>.xml

# Single test file or test
uv run --project backend pytest backend/tests/offline/test_contracts.py -q
uv run --project backend pytest backend/tests/offline/test_contracts.py::<test_name> -q

# Inspectable offline replay
uv run --project backend python backend/scripts/replay.py --case backend/tests/fixtures/<case>.json --output artifacts/<report>.json

# Live checks: explicit gates that call real providers and can incur API charges
uv run --project backend pytest backend/tests/online/test_providers.py -q --junitxml=artifacts/phase-4-live.xml
uv run --project backend python backend/scripts/evaluate.py --cases backend/tests/fixtures/live_cases.json --output artifacts/phase-5-summary.json
```

The exact command and report name for each phase are in its document in `docs/phased-development/`. The frontend package manager, linter, formatter, and type checker have not been selected. Browser commands will run from `web/`.

## Architecture

V1 is a website chat feature that fills one same-day gap before one fixed commitment, returning up to three distinct feasible itineraries.

**The model proposes and Python decides.** One bounded LangGraph loop lets the model choose research, tools, clarification, and repairs. It has no fixed tool sequence, specialized agents, or per-edge-case nodes. Python owns schedule and cost arithmetic, evidence sufficiency, hard-constraint rejection, and the final gate. Model output and schema-valid claims are never accepted as facts.

**Request flow:**

1. The React client sends a turn with its `sessionStorage` context to FastAPI (`api.py`). Speech-to-text uses ElevenLabs, but its transport is not yet defined.
2. FastAPI validates the context as untrusted input and rebuilds request-scoped state.
3. The loop (`agent.py`) calls OpenAI and the connectors in `providers/` (Google Maps, Tavily, Open-Meteo).
4. Proposals go to `feasibility.py`. Failures return to the loop as structured feedback, so the model can repair them within the execution budget.
5. `selection.py` picks only from validated candidates, in order: AI Recommended, then Cheapest of the remainder, then best 50/50 travel score of the remainder.

A turn ends in one of four distinct results: clarification, validated options, no feasible result (which asks before retrying and never relaxes constraints), or execution failure. Hitting the budget or a provider failure is not proof that no itinerary exists.

**State lifecycle:**

- V1 has no backend database or checkpoint store. Backend state lasts for one request.
- Continuation data is the minimum permitted context held in the browser. It never proves that evidence or candidates were verified.
- Langfuse conversation-text tracing is the one approved persistent exception for live traffic. It must not block the response path, and it excludes raw audio.
- Local evaluation cassettes in `backend/tests/cassettes/` are a separate, test-only copy of live evaluation runs. They stay on the developer's machine, are never committed or deployed, and never record real user traffic.

**Testing and gates:**

- Phases run in order across three buckets: Offline (0–3), Online (4–5), and Deployment (6–7). Each phase must be verifiable without later phases and must leave a report in `artifacts/`.
- Offline tests use synthetic fixtures and fixed clocks, and they fail on any outbound network call. They do not establish live model quality or provider availability.
- Some work is blocked until its gate in `docs/plan.md` clears. Google Maps data permissions block live Phase 4 integration, and the trace-field policy blocks content-bearing traces in Phase 5. Do not assume Maps responses may be passed to the model, stored, or traced.

## Where to Look

| Question | Document |
|---|---|
| Product behavior and requirement IDs (R1–R17; R5 is deferred to V2) | `docs/requirements.md` |
| Stack, gates, and V1 exclusions | `docs/plan.md` |
| Deliverables, tests, and acceptance for a phase | `docs/phased-development/phase-N-*.md` |
| Loop, tool boundaries, data lifecycle, and provider permissions | `docs/architecture.md` |
| Render and Langfuse hosting research | `docs/hosting.md` |
| Confirmed decisions and what superseded them | `docs/decisions.md` |
