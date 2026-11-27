"""Offline replay: run one case file and write an inspectable JSON report.

Usage (from the repository root):
    uv run --project backend python backend/scripts/replay.py --case CASE.json --output OUT.json

The case file's keys decide what is replayed; there is no mode flag.

- A revision case holds ``request``, ``evidence``, ``candidates``, and ``recommendation``.
  Its report holds ``checked`` (passing candidates), ``rejected`` (validation feedback),
  and ``options`` (selected options in slot order), each as contract-model JSON.
- A conversation case holds a fixed ``clock_start``, the ``user_timezone``, ``limits``, the first
  ``continuation``, scripted ``web``, ``places``, ``routes``, and ``weather`` answers, and
  ``turns`` of user messages with scripted model replies. Each turn runs through
  ``run_turn`` with the previous turn's continuation and the case's user timezone. Its
  report holds ``turns``, each a ``user_message`` and its ``TurnOutcome``. An unscripted
  provider request, a model call past the scripted replies, or unused replies is an error
  and writes no report.

No network access is used.
"""

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from travel_agent.agent import (
    AgentDependencies,
    ContinuationOverflowError,
    ExecutionLimits,
    MapsPlace,
    MapsRoute,
    ModelMessage,
    ModelReply,
    ToolSpec,
    TurnOutcome,
    WeatherPeriod,
    WebPage,
    run_turn,
)
from travel_agent.contracts import (
    CandidateItinerary,
    CheckedItinerary,
    ContinuationPayload,
    ContractDatetime,
    Evidence,
    IanaTimezone,
    Location,
    PlanningRequest,
    Recommendation,
    TransportMode,
    TurnStatus,
    ValidatedOption,
    ValidatedOptions,
    ValidationFeedback,
)
from travel_agent.feasibility import UnusableInputError, check_candidate
from travel_agent.selection import SelectionError, select_options

type RouteKey = tuple[float, float, float, float, TransportMode]
type PointKey = tuple[float, float]


class ReplayCase(BaseModel):
    """One revision case file: strict, immutable, and closed to unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    request: PlanningRequest
    evidence: tuple[Evidence, ...]
    candidates: tuple[CandidateItinerary, ...]
    recommendation: Recommendation


class ReplayReport(BaseModel):
    """The revision replay output: checked and rejected candidates in case order, then options."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    checked: tuple[CheckedItinerary, ...]
    rejected: tuple[ValidationFeedback, ...]
    options: tuple[ValidatedOption, ...]


class _Script(BaseModel):
    """Base for conversation-case parts: strict, immutable, and closed to unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ScriptedSearch(_Script):
    """The pages a web search for exactly ``query`` returns."""

    query: str
    pages: tuple[WebPage, ...]


class ScriptedPlace(_Script):
    """The place a Maps lookup for exactly ``query`` returns; None means no match."""

    query: str
    place: MapsPlace | None


class ScriptedRoute(_Script):
    """The route between two coordinates by one mode; labels are not matched."""

    origin: Location
    destination: Location
    mode: TransportMode
    route: MapsRoute


class ScriptedForecast(_Script):
    """The forecast periods for one point's exact coordinates."""

    point: Location
    periods: tuple[WeatherPeriod, ...]


class ScriptedTurn(_Script):
    """One user message and the model replies the turn consumes, in order."""

    user_message: str
    replies: Annotated[tuple[ModelReply, ...], Field(min_length=1)]


def route_key(origin: Location, destination: Location, mode: TransportMode) -> RouteKey:
    """Key a route by endpoint coordinates and mode."""
    return (origin.latitude, origin.longitude, destination.latitude, destination.longitude, mode)


def point_key(point: Location) -> PointKey:
    """Key a forecast by the point's coordinates."""
    return (point.latitude, point.longitude)


class ConversationCase(_Script):
    """One conversation case file: a fixed clock, scripted answers, and the turns to run."""

    clock_start: ContractDatetime
    user_timezone: IanaTimezone
    limits: ExecutionLimits
    continuation: ContinuationPayload
    web: tuple[ScriptedSearch, ...]
    places: tuple[ScriptedPlace, ...]
    routes: tuple[ScriptedRoute, ...]
    weather: tuple[ScriptedForecast, ...]
    turns: Annotated[tuple[ScriptedTurn, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _unique_scripts(self) -> Self:
        """Reject two scripted answers for the same request, which would be ambiguous."""
        keyed: dict[str, list[object]] = {
            "web query": [item.query for item in self.web],
            "place query": [item.query for item in self.places],
            "route": [route_key(item.origin, item.destination, item.mode) for item in self.routes],
            "forecast point": [point_key(item.point) for item in self.weather],
        }
        for what, keys in keyed.items():
            duplicates = sorted(repr(key) for key, count in Counter(keys).items() if count > 1)
            if duplicates:
                raise ValueError(f"duplicate scripted {what}: {', '.join(duplicates)}")
        return self


class ConversationTurn(_Script):
    """One replayed turn: the user's message and the turn's outcome."""

    user_message: str
    outcome: TurnOutcome


class ConversationReport(_Script):
    """The conversation replay output: every turn in order."""

    turns: tuple[ConversationTurn, ...]


CASE_ADAPTER: TypeAdapter[ReplayCase | ConversationCase] = TypeAdapter(
    ReplayCase | ConversationCase
)


class ScriptMismatchError(Exception):
    """The loop's requests did not match the case script (deliberately not a ProviderError)."""


class FixedClock:
    """A clock that always reads the case's ``clock_start``."""

    def __init__(self, start: datetime) -> None:
        self._start = start

    def now(self) -> datetime:
        return self._start


class ScriptedModel:
    """Returns one turn's scripted replies in order and fails past the end."""

    def __init__(self, replies: tuple[ModelReply, ...]) -> None:
        self._replies = replies
        self.calls = 0

    def respond(
        self, messages: tuple[ModelMessage, ...], tools: tuple[ToolSpec, ...]
    ) -> ModelReply:
        if self.calls >= len(self._replies):
            raise ScriptMismatchError(
                f"model called {self.calls + 1} times; turn script has {len(self._replies)}"
            )
        reply = self._replies[self.calls]
        self.calls += 1
        return reply


class ScriptedWebResearch:
    """Answers only scripted web searches."""

    def __init__(self, searches: tuple[ScriptedSearch, ...]) -> None:
        self._pages = {item.query: item.pages for item in searches}

    def search(self, query: str) -> tuple[WebPage, ...]:
        if query not in self._pages:
            raise ScriptMismatchError(f"unscripted web search {query!r}")
        return self._pages[query]


class ScriptedMaps:
    """Answers only scripted place lookups and routes."""

    def __init__(
        self, places: tuple[ScriptedPlace, ...], routes: tuple[ScriptedRoute, ...]
    ) -> None:
        self._places = {item.query: item.place for item in places}
        self._routes = {
            route_key(item.origin, item.destination, item.mode): item.route for item in routes
        }

    def find_place(self, query: str) -> MapsPlace | None:
        if query not in self._places:
            raise ScriptMismatchError(f"unscripted place query {query!r}")
        return self._places[query]

    def route(self, origin: Location, destination: Location, mode: TransportMode) -> MapsRoute:
        key = route_key(origin, destination, mode)
        if key not in self._routes:
            raise ScriptMismatchError(f"unscripted route {key!r}")
        return self._routes[key]


class ScriptedWeather:
    """Answers only scripted forecast points; the window is not matched."""

    def __init__(self, forecasts: tuple[ScriptedForecast, ...]) -> None:
        self._periods = {point_key(item.point): item.periods for item in forecasts}

    def forecast(
        self, point: Location, window_start: datetime, window_end: datetime
    ) -> tuple[WeatherPeriod, ...]:
        key = point_key(point)
        if key not in self._periods:
            raise ScriptMismatchError(f"unscripted forecast {key!r}")
        return self._periods[key]


class DiscardingStatusSink:
    """Drops live statuses; the report reads them from ``TurnOutcome.statuses``."""

    def emit(self, status: TurnStatus) -> None:
        """Ignore the status."""


def replay(case: ReplayCase) -> ReplayReport:
    """Check every candidate in case order, then select options from those that passed."""
    results = tuple(
        check_candidate(case.request, candidate, case.evidence) for candidate in case.candidates
    )
    checked = tuple(item for item in results if isinstance(item, CheckedItinerary))
    rejected = tuple(item for item in results if isinstance(item, ValidationFeedback))
    options = select_options(checked, case.recommendation) if checked else ()
    return ReplayReport(checked=checked, rejected=rejected, options=options)


def replay_turn(
    case: ConversationCase, turn: ScriptedTurn, continuation: ContinuationPayload
) -> TurnOutcome:
    """Run one scripted turn and require that it used every scripted reply."""
    model = ScriptedModel(turn.replies)
    dependencies = AgentDependencies(
        model=model,
        web=ScriptedWebResearch(case.web),
        maps=ScriptedMaps(case.places, case.routes),
        weather=ScriptedWeather(case.weather),
        clock=FixedClock(case.clock_start),
        status_sink=DiscardingStatusSink(),
    )
    outcome = run_turn(
        continuation, turn.user_message, case.user_timezone, None, dependencies, case.limits
    )
    if model.calls != len(turn.replies):
        raise ScriptMismatchError(
            f"turn used {model.calls} of {len(turn.replies)} scripted replies"
        )
    return outcome


def replay_conversation(case: ConversationCase) -> ConversationReport:
    """Run the turns in order, passing each turn's continuation to the next."""
    turns: tuple[ConversationTurn, ...] = ()
    continuation = case.continuation
    for index, turn in enumerate(case.turns, start=1):
        try:
            outcome = replay_turn(case, turn, continuation)
        except ScriptMismatchError as error:
            raise ScriptMismatchError(f"turn {index}: {error}") from error
        turns = (*turns, ConversationTurn(user_message=turn.user_message, outcome=outcome))
        continuation = outcome.continuation
    return ConversationReport(turns=turns)


def summarise(report: ReplayReport) -> str:
    """Return a short human-readable summary of a revision replay report."""
    lines = [
        f"checked: {len(report.checked)}, rejected: {len(report.rejected)}, "
        f"options: {len(report.options)}"
    ]
    for item in report.checked:
        total = item.total_cost
        estimate = " (includes estimates)" if item.total_includes_estimates else ""
        arrival = item.arrival_at_commitment.isoformat()
        lines.append(
            f"  pass {item.candidate.candidate_id}: arrive {arrival}, "
            f"total {total.amount} {total.currency}{estimate}"
        )
    for feedback in report.rejected:
        constraints = ", ".join(sorted({failure.constraint for failure in feedback.failures}))
        lines.append(f"  fail {feedback.candidate_id}: {constraints}")
    for option in report.options:
        labels = ", ".join(option.labels)
        lines.append(f"  option {option.itinerary.candidate.candidate_id}: {labels}")
    return "\n".join(lines)


def summarise_conversation(report: ConversationReport) -> str:
    """Return a short per-turn summary of a conversation replay report."""
    lines: list[str] = []
    for index, turn in enumerate(report.turns, start=1):
        outcome = turn.outcome
        statuses = " > ".join(outcome.statuses) or "none"
        lines.append(
            f"turn {index}: {outcome.result.kind}; model calls {outcome.model_calls}, "
            f"tool calls {len(outcome.tool_calls)}; statuses: {statuses}"
        )
        for feedback in outcome.validation_feedback:
            constraints = ", ".join(sorted({failure.constraint for failure in feedback.failures}))
            lines.append(f"  fail {feedback.candidate_id}: {constraints}")
        if isinstance(outcome.result, ValidatedOptions):
            for option in outcome.result.options:
                labels = ", ".join(option.labels)
                lines.append(f"  option {option.itinerary.candidate.candidate_id}: {labels}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse the required --case and --output paths."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--case", type=Path, required=True, help="replay case JSON file")
    parser.add_argument("--output", type=Path, required=True, help="report JSON file to write")
    return parser.parse_args()


def load_case(case_path: Path) -> ReplayCase | ConversationCase:
    """Read a case file and tell its kind apart by schema; exit on an invalid file."""
    try:
        return CASE_ADAPTER.validate_json(case_path.read_bytes())
    except ValidationError as error:
        raise SystemExit(f"invalid replay case {case_path}: {error}") from error


def run_revision_case(case: ReplayCase, case_path: Path) -> tuple[ReplayReport, str]:
    """Replay a revision case and summarise it; exit on unusable input."""
    try:
        report = replay(case)
    except (UnusableInputError, SelectionError) as error:
        raise SystemExit(
            f"unusable replay case {case_path}: {type(error).__name__}: {error}"
        ) from error
    return report, summarise(report)


def run_conversation_case(
    case: ConversationCase, case_path: Path
) -> tuple[ConversationReport, str]:
    """Replay a conversation case and summarise it; exit when the script does not fit."""
    try:
        report = replay_conversation(case)
    except (ScriptMismatchError, ContinuationOverflowError, ValidationError) as error:
        raise SystemExit(
            f"unusable replay case {case_path}: {type(error).__name__}: {error}"
        ) from error
    return report, summarise_conversation(report)


def main() -> None:
    """Load the case, replay it, write the report, and print a summary."""
    args = parse_args()
    case_path: Path = args.case
    output_path: Path = args.output
    case = load_case(case_path)
    report, summary = (
        run_revision_case(case, case_path)
        if isinstance(case, ReplayCase)
        else run_conversation_case(case, case_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(summary)
    print(f"report written to {output_path}")


if __name__ == "__main__":
    main()
