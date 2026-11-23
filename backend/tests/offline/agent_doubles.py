"""Deterministic offline doubles for the Phase 2 agent loop.

The scripted model returns a fixed sequence of replies and records what it was sent.
Scripted providers answer only scripted requests; anything else raises
``UnscriptedCallError`` so a test fails loudly instead of the loop seeing a provider
failure. ``ManualClock`` only moves when a double or a test advances it.
No double opens a network connection.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

from travel_agent.agent import (
    MapsPlace,
    MapsRoute,
    ModelMessage,
    ModelReply,
    ProviderError,
    ToolSpec,
    WeatherPeriod,
    WebPage,
)
from travel_agent.contracts import Location, TransportMode, TurnStatus

type RouteKey = tuple[float, float, float, float, TransportMode]
type PointKey = tuple[float, float]


class UnscriptedCallError(AssertionError):
    """A double received a request its script does not cover."""


class ManualClock:
    """A clock that changes only through ``advance``."""

    def __init__(self, start: datetime) -> None:
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)


class ScriptedModel:
    """Returns ``replies`` in order; each reply advances ``clock`` by ``seconds_per_reply``."""

    def __init__(
        self, replies: Sequence[ModelReply], clock: ManualClock, seconds_per_reply: float
    ) -> None:
        self._replies = tuple(replies)
        self._clock = clock
        self._seconds_per_reply = seconds_per_reply
        self.received: list[tuple[tuple[ModelMessage, ...], tuple[ToolSpec, ...]]] = []

    @property
    def remaining(self) -> int:
        return len(self._replies) - len(self.received)

    def respond(
        self, messages: tuple[ModelMessage, ...], tools: tuple[ToolSpec, ...]
    ) -> ModelReply:
        index = len(self.received)
        self.received.append((messages, tools))
        if index >= len(self._replies):
            raise UnscriptedCallError(f"model called {index + 1} times; script has {index}")
        self._clock.advance(self._seconds_per_reply)
        return self._replies[index]


class FailingModel:
    """A model whose every call fails like an unavailable provider."""

    def __init__(self) -> None:
        self.calls = 0

    def respond(
        self, messages: tuple[ModelMessage, ...], tools: tuple[ToolSpec, ...]
    ) -> ModelReply:
        self.calls += 1
        raise ProviderError("synthetic model outage")


class ScriptedWebResearch:
    """Returns the scripted pages for an exact query; each search advances the clock."""

    def __init__(
        self,
        pages_by_query: Mapping[str, tuple[WebPage, ...]],
        clock: ManualClock,
        seconds_per_call: float,
    ) -> None:
        self._pages_by_query = dict(pages_by_query)
        self._clock = clock
        self._seconds_per_call = seconds_per_call
        self.queries: list[str] = []

    def search(self, query: str) -> tuple[WebPage, ...]:
        self.queries.append(query)
        if query not in self._pages_by_query:
            raise UnscriptedCallError(f"unscripted web search {query!r}")
        self._clock.advance(self._seconds_per_call)
        return self._pages_by_query[query]


class FailingWebResearch:
    """A web research provider whose every search fails."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str) -> tuple[WebPage, ...]:
        self.queries.append(query)
        raise ProviderError("synthetic web research outage")


def route_key(origin: Location, destination: Location, mode: TransportMode) -> RouteKey:
    """Key a route by endpoint coordinates and mode; labels are ignored."""
    return (origin.latitude, origin.longitude, destination.latitude, destination.longitude, mode)


class ScriptedMaps:
    """Answers scripted place queries (a None place means Maps has no match) and routes."""

    def __init__(
        self,
        places_by_query: Mapping[str, MapsPlace | None],
        routes: Mapping[RouteKey, MapsRoute],
    ) -> None:
        self._places_by_query = dict(places_by_query)
        self._routes = dict(routes)
        self.place_queries: list[str] = []
        self.route_requests: list[RouteKey] = []

    def find_place(self, query: str) -> MapsPlace | None:
        self.place_queries.append(query)
        if query not in self._places_by_query:
            raise UnscriptedCallError(f"unscripted place query {query!r}")
        return self._places_by_query[query]

    def route(self, origin: Location, destination: Location, mode: TransportMode) -> MapsRoute:
        key = route_key(origin, destination, mode)
        self.route_requests.append(key)
        if key not in self._routes:
            raise UnscriptedCallError(f"unscripted route {key!r}")
        return self._routes[key]


class FailingMaps:
    """A Maps provider whose every call fails."""

    def __init__(self) -> None:
        self.calls = 0

    def find_place(self, query: str) -> MapsPlace | None:
        self.calls += 1
        raise ProviderError("synthetic maps outage")

    def route(self, origin: Location, destination: Location, mode: TransportMode) -> MapsRoute:
        self.calls += 1
        raise ProviderError("synthetic maps outage")


class ScriptedWeather:
    """Returns scripted forecast periods for a point's exact coordinates."""

    def __init__(self, periods_by_point: Mapping[PointKey, tuple[WeatherPeriod, ...]]) -> None:
        self._periods_by_point = dict(periods_by_point)
        self.requests: list[tuple[PointKey, datetime, datetime]] = []

    def forecast(
        self, point: Location, window_start: datetime, window_end: datetime
    ) -> tuple[WeatherPeriod, ...]:
        key = (point.latitude, point.longitude)
        self.requests.append((key, window_start, window_end))
        if key not in self._periods_by_point:
            raise UnscriptedCallError(f"unscripted forecast {key!r}")
        return self._periods_by_point[key]


class FailingWeather:
    """A weather provider whose every forecast fails."""

    def __init__(self) -> None:
        self.calls = 0

    def forecast(
        self, point: Location, window_start: datetime, window_end: datetime
    ) -> tuple[WeatherPeriod, ...]:
        self.calls += 1
        raise ProviderError("synthetic weather outage")


class RecordingStatusSink:
    """Records every status the loop emits, in order."""

    def __init__(self) -> None:
        self.statuses: list[TurnStatus] = []

    def emit(self, status: TurnStatus) -> None:
        self.statuses.append(status)
