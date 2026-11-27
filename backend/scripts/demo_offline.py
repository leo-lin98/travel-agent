"""Offline demo backend: the chat API wired to scripted doubles (Phase 3, D4).

Usage (from the repository root):
    uv run --project backend python backend/scripts/demo_offline.py --port PORT

The server always binds 127.0.0.1, so the demo cannot listen publicly. It needs no network,
no provider keys, and reads no environment variables. Phase 6 keeps ``scripts/`` out of the
release image, so these doubles never reach a deployment.

The scripted model follows a fixed two-turn script whatever the user types and ignores the
device location. It is stateless: the turn is the number of user messages in the history
minus one, and the reply is the number of tool-call messages so far. A fresh session always
starts at turn 1, and refreshes, restarts, and parallel sessions do not interfere. Asking it
for a reply past the script raises ``DemoScriptEndedError`` after one stderr log line.
"""

import argparse
import json
import sys
from collections.abc import Mapping
from datetime import datetime
from typing import Final

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from travel_agent.agent import (
    AgentDependencies,
    ConversationMessage,
    ExecutionLimits,
    MapsPlace,
    MapsRoute,
    ModelMessage,
    ModelReply,
    StatusSink,
    ToolCall,
    ToolCallReply,
    ToolCallsMessage,
    ToolSpec,
    WeatherPeriod,
    WebPage,
)
from travel_agent.api import ApiLimits, create_app
from travel_agent.contracts import ChatRole, Location, TransportMode

HOST: Final[str] = "127.0.0.1"

type _Doc = Mapping[str, object]

_DAY: Final[str] = "2026-11-20"
_ZONE: Final[str] = "Asia/Tokyo"
_RETRIEVED: Final[str] = f"{_DAY}T13:00:00+09:00"
_ALL_DAY: Final[_Doc] = {
    "applies_from": f"{_DAY}T00:00:00+09:00",
    "applies_until": f"{_DAY}T23:59:00+09:00",
}

TRANSCRIPT: Final[str] = (
    "I'm free from 3pm and have dinner in Omotesando at 7pm. "
    "I have 3000 to spend and would like something quiet."
)


class DemoScriptEndedError(Exception):
    """The demo model was asked for a reply its fixed script does not have."""


class UnscriptedDemoCallError(Exception):
    """A demo provider was asked for something the script never requests."""


# --- script data ------------------------------------------------------------------------


def _parsed[M: BaseModel](model: type[M], document: _Doc) -> M:
    """Validate a JSON-shaped document into a strict model, as the wire would."""
    return model.model_validate_json(json.dumps(document))


def _location(label: str, latitude: float, longitude: float, place_id: str | None) -> _Doc:
    return {"label": label, "latitude": latitude, "longitude": longitude, "place_id": place_id}


_START: Final = _location("Shibuya Station", 35.658, 139.7016, None)
_DINNER: Final = _location("Dinner in Omotesando", 35.6652, 139.7123, "ChIJdemoOmotesandoDinner")
_GARDEN: Final = _location("Demo Quiet Garden", 35.6764, 139.6993, "ChIJdemoQuietGarden")
_MUSEUM: Final = _location("Demo Print Museum", 35.662, 139.709, "ChIJdemoPrintMuseum")
_TEA: Final = _location("Demo Tea House", 35.66, 139.705, "ChIJdemoTeaHouse")


def _place(location: _Doc, opens: str, closes: str) -> MapsPlace:
    return _parsed(
        MapsPlace,
        {
            **{key: location[key] for key in ("place_id", "label", "latitude", "longitude")},
            "maps_url": f"https://maps.google.com/?q=place_id:{location['place_id']}",
            "retrieved_at": _RETRIEVED,
            "opening": {
                "claim": {
                    "kind": "opening_access",
                    "status": "open",
                    "open_intervals": [
                        {
                            "opens_at": f"{_DAY}T{opens}:00+09:00",
                            "closes_at": f"{_DAY}T{closes}:00+09:00",
                        }
                    ],
                },
                **_ALL_DAY,
            },
        },
    )


def _pages(url: str, title: str, content: str) -> tuple[WebPage, ...]:
    page = {"url": url, "title": title, "content": content, "retrieved_at": _RETRIEVED}
    return (_parsed(WebPage, page),)


_PAGES: Final[Mapping[str, tuple[WebPage, ...]]] = {
    "quiet things to do near Omotesando": _pages(
        "https://guide.example.com/omotesando-quiet",
        "Quiet afternoons near Omotesando",
        "Gardens, small museums, and tea houses within a short walk.",
    ),
    "Demo Quiet Garden": _pages(
        "https://garden.example.org/visit",
        "Demo Quiet Garden - Visit",
        "Open daily 09:00-21:00. Admission JPY 500. Most visitors stay about an hour.",
    ),
    "Demo Print Museum": _pages(
        "https://museum.example.org/",
        "Demo Print Museum",
        "Woodblock prints. Allow about 75 minutes. Ticket prices are not listed online.",
    ),
    "Demo Tea House": _pages(
        "https://tea.example.org/",
        "Demo Tea House",
        "A seasonal tea set is JPY 800. A visit takes about 45 minutes.",
    ),
}
_PLACES: Final[Mapping[str, MapsPlace]] = {
    "Demo Quiet Garden": _place(_GARDEN, "09:00", "21:00"),
    "Demo Print Museum": _place(_MUSEUM, "10:00", "18:00"),
    "Demo Tea House": _place(_TEA, "11:00", "19:00"),
}


def _walk(minutes: int, meters: int) -> MapsRoute:
    route: _Doc = {
        "duration_minutes": minutes,
        "distance_meters": meters,
        "transfers": (),
        "transit": (),
    }
    return _parsed(MapsRoute, route)


type _RouteKey = tuple[float, float, float, float, TransportMode]


def _route_key(origin: Location, destination: Location, mode: TransportMode) -> _RouteKey:
    return (origin.latitude, origin.longitude, destination.latitude, destination.longitude, mode)


def _point(location: _Doc) -> tuple[float, float]:
    parsed = _parsed(Location, location)
    return (parsed.latitude, parsed.longitude)


def _walk_key(origin: _Doc, destination: _Doc) -> _RouteKey:
    return _route_key(
        _parsed(Location, origin), _parsed(Location, destination), TransportMode.WALKING
    )


# Walking legs per stop: (minutes, meters) from the start, then to dinner.
_LEGS: Final[Mapping[str, tuple[tuple[int, int], tuple[int, int]]]] = {
    "garden": ((25, 2000), (25, 1900)),
    "museum": ((15, 1100), (12, 900)),
    "tea": ((10, 700), (20, 1500)),
}
_STOPS: Final[Mapping[str, _Doc]] = {"garden": _GARDEN, "museum": _MUSEUM, "tea": _TEA}
_ROUTES: Final[Mapping[_RouteKey, MapsRoute]] = {
    key: route
    for name, (into, out) in _LEGS.items()
    for key, route in (
        (_walk_key(_START, _STOPS[name]), _walk(*into)),
        (_walk_key(_STOPS[name], _DINNER), _walk(*out)),
    )
}
_FORECASTS: Final[Mapping[tuple[float, float], tuple[WeatherPeriod, ...]]] = {
    _point(_GARDEN): (
        _parsed(
            WeatherPeriod,
            {
                "applies_from": f"{_DAY}T15:00:00+09:00",
                "applies_until": f"{_DAY}T18:00:00+09:00",
                "rain": True,
                "source_url": "https://forecast.example.com/tokyo",
                "retrieved_at": _RETRIEVED,
            },
        ),
    ),
}


# --- the model script -------------------------------------------------------------------


def _call(call_id: str, name: str, arguments: _Doc) -> ToolCall:
    return ToolCall(call_id=call_id, name=name, arguments=json.dumps(arguments))


def _reply(*calls: ToolCall) -> ModelReply:
    return ToolCallReply(kind="tool_calls", calls=calls)


def _request(start: _Doc | None, currency: str | None, open_questions: tuple[_Doc, ...]) -> _Doc:
    return {
        "request": {
            "schema_version": 1,
            "gap": {"start_time": f"{_DAY}T15:00:00+09:00", "start_location": start},
            "commitment": {
                "location": _DINNER,
                "time": f"{_DAY}T19:00:00+09:00",
                "timezone": _ZONE,
            },
            "budget": {"amount": "3000", "currency": currency},
            "transport_modes": ["walking"],
            "constraints": [],
            "unresolved_questions": list(open_questions),
        }
    }


_START_QUESTION: Final[_Doc] = {"topic": "start_location", "question": "Where will you start?"}
_CURRENCY_QUESTION: Final[_Doc] = {
    "topic": "budget_currency",
    "question": "Which currency is your 3000 budget in?",
}


def _scope(stop: str) -> _Doc:
    return {"place_id": _STOPS[stop]["place_id"], "fallback_key": None, **_ALL_DAY}


def _sourced(evidence_id: str, result_id: str, stop: str, claim: _Doc, finding: str) -> _Doc:
    topic = "visit_duration" if claim["kind"] == "visit_duration" else "cost"
    return {
        "evidence": {
            "kind": "sourced_finding",
            "evidence_id": evidence_id,
            "result_id": result_id,
            "source_kind": "official_website",
            "topic": topic,
            "claim": claim,
            "scope": _scope(stop),
            "finding": finding,
        }
    }


def _minutes(minutes: int) -> _Doc:
    return {"kind": "visit_duration", "minutes": minutes}


def _yen(amount: str) -> _Doc:
    return {"kind": "cost_amount", "amount": {"amount": amount, "currency": "JPY"}}


def _research(stop: str) -> tuple[ToolCall, ...]:
    """Find the stop on Maps, search the web for it, and route to and from it on foot."""
    label = str(_STOPS[stop]["label"])
    place = {"kind": "place", "place_id": _STOPS[stop]["place_id"]}
    return (
        _call(f"{stop}-find", "find_place", {"query": label}),
        _call(f"{stop}-search", "web_search", {"query": label}),
        _call(
            f"{stop}-in",
            "get_route",
            {"origin": {"kind": "start"}, "destination": place, "mode": "walking"},
        ),
        _call(
            f"{stop}-out",
            "get_route",
            {"origin": place, "destination": {"kind": "commitment"}, "mode": "walking"},
        ),
    )


def _candidate(stop: str, visit_minutes: int, cost: str, cost_id: str) -> _Doc:
    location = _STOPS[stop]
    (into_minutes, into_meters), (out_minutes, out_meters) = _LEGS[stop]
    return {
        "candidate": {
            "candidate_id": f"cand-{stop}",
            "stops": [
                {
                    **{
                        key: location[key] for key in ("place_id", "label", "latitude", "longitude")
                    },
                    "fallback_key": None,
                    "visit_minutes": visit_minutes,
                    "evidence_ids": (),
                }
            ],
            "legs": [
                {
                    "mode": "walking",
                    "duration_minutes": minutes,
                    "distance_meters": meters,
                    "transfers": (),
                    "evidence_ids": (),
                }
                for minutes, meters in ((into_minutes, into_meters), (out_minutes, out_meters))
            ],
            "costs": [
                {
                    "description": f"{location['label']} entry",
                    "amount": {"amount": cost, "currency": "JPY"},
                    "evidence_ids": [cost_id],
                }
            ],
        }
    }


_MUSEUM_GUESS: Final[_Doc] = {
    "evidence": {
        "kind": "estimate",
        "evidence_id": "ev-museum-ticket",
        "topic": "cost",
        "claim": _yen("1200"),
        "label": "ai_best_guess",
        "scope": _scope("museum"),
        "basis": "The museum does not list ticket prices; similar small museums charge this.",
        "based_on": [],
    }
}

_TURN_1: Final[tuple[ModelReply, ...]] = (
    _reply(_call("t1-search", "web_search", {"query": "quiet things to do near Omotesando"})),
    _reply(
        _call(
            "t1-update",
            "update_request",
            _request(None, None, (_START_QUESTION, _CURRENCY_QUESTION)),
        )
    ),
    _reply(
        _call(
            "t1-clarify",
            "ask_clarification",
            {
                "message": "Two quick questions before I plan your afternoon.",
                "questions": [_START_QUESTION, _CURRENCY_QUESTION],
            },
        )
    ),
)
_TURN_2: Final[tuple[ModelReply, ...]] = (
    _reply(_call("t2-update", "update_request", _request(_START, "JPY", ()))),
    _reply(*_research("garden"), *_research("museum")),
    _reply(
        _call(
            "garden-weather",
            "get_weather",
            {"point": {"kind": "place", "place_id": _GARDEN["place_id"]}},
        ),
        _call(
            "garden-stay",
            "record_evidence",
            _sourced(
                "ev-garden-stay",
                "web-1",
                "garden",
                _minutes(60),
                "Most visitors stay about an hour.",
            ),
        ),
        _call(
            "garden-fee",
            "record_evidence",
            _sourced("ev-garden-fee", "web-1", "garden", _yen("500"), "Admission JPY 500."),
        ),
        _call(
            "museum-stay",
            "record_evidence",
            _sourced("ev-museum-stay", "web-2", "museum", _minutes(75), "Allow about 75 minutes."),
        ),
        _call("museum-ticket", "record_evidence", _MUSEUM_GUESS),
    ),
    _reply(
        _call("check-garden", "check_candidate", _candidate("garden", 90, "500", "ev-garden-fee"))
    ),
    _reply(
        _call(
            "check-museum", "check_candidate", _candidate("museum", 90, "1200", "ev-museum-ticket")
        )
    ),
    _reply(*_research("tea")),
    _reply(
        _call(
            "tea-stay",
            "record_evidence",
            _sourced(
                "ev-tea-stay", "web-3", "tea", _minutes(45), "A visit takes about 45 minutes."
            ),
        ),
        _call(
            "tea-fee",
            "record_evidence",
            _sourced("ev-tea-fee", "web-3", "tea", _yen("800"), "A seasonal tea set is JPY 800."),
        ),
    ),
    _reply(_call("check-tea", "check_candidate", _candidate("tea", 60, "800", "ev-tea-fee"))),
    _reply(
        _call(
            "t2-submit",
            "submit_options",
            {
                "message": "Here are three quiet ways to spend the afternoon before dinner.",
                "recommendation": {
                    "candidate_id": "cand-garden",
                    "reason": "The quietest option, with time to linger before dinner.",
                },
                "candidate_ids": ["cand-garden", "cand-museum", "cand-tea"],
            },
        )
    ),
)
SCRIPT: Final[tuple[tuple[ModelReply, ...], ...]] = (_TURN_1, _TURN_2)


# --- doubles ----------------------------------------------------------------------------


class DemoModel:
    """Replies from ``SCRIPT`` at the position the history shows; holds no state."""

    def respond(
        self, messages: tuple[ModelMessage, ...], tools: tuple[ToolSpec, ...]
    ) -> ModelReply:
        turn = (
            sum(
                1
                for message in messages
                if isinstance(message, ConversationMessage)
                and message.message.role == ChatRole.USER
            )
            - 1
        )
        reply = sum(1 for message in messages if isinstance(message, ToolCallsMessage))
        if turn >= len(SCRIPT) or reply >= len(SCRIPT[turn]):
            log = {"event": "demo_script_ended", "turn": turn + 1, "reply": reply + 1}
            print(json.dumps(log), file=sys.stderr)
            raise DemoScriptEndedError(
                f"the demo script has {len(SCRIPT)} turns; asked for turn {turn + 1}, "
                f"reply {reply + 1}"
            )
        return SCRIPT[turn][reply]


class DemoWebResearch:
    """Answers only the script's web searches."""

    def search(self, query: str) -> tuple[WebPage, ...]:
        if query not in _PAGES:
            raise UnscriptedDemoCallError(f"unscripted web search {query!r}")
        return _PAGES[query]


class DemoMaps:
    """Answers only the script's place lookups and walking routes."""

    def find_place(self, query: str) -> MapsPlace | None:
        if query not in _PLACES:
            raise UnscriptedDemoCallError(f"unscripted place query {query!r}")
        return _PLACES[query]

    def route(self, origin: Location, destination: Location, mode: TransportMode) -> MapsRoute:
        key = _route_key(origin, destination, mode)
        if key not in _ROUTES:
            raise UnscriptedDemoCallError(f"unscripted route {key!r}")
        return _ROUTES[key]


class DemoWeather:
    """Answers only the script's forecast points."""

    def forecast(
        self, point: Location, window_start: datetime, window_end: datetime
    ) -> tuple[WeatherPeriod, ...]:
        key = (point.latitude, point.longitude)
        if key not in _FORECASTS:
            raise UnscriptedDemoCallError(f"unscripted forecast {key!r}")
        return _FORECASTS[key]


class DemoClock:
    """Always reads the script's outing date at 13:00 Tokyo time, so its times are today."""

    def now(self) -> datetime:
        return datetime.fromisoformat(_RETRIEVED)


class DemoTranscriber:
    """Returns one fixed English transcript (a complete turn-1 request) for any clip."""

    def transcribe(self, audio: bytes, content_type: str) -> str:
        return TRANSCRIPT


def _dependencies_for(sink: StatusSink) -> AgentDependencies:
    return AgentDependencies(
        model=DemoModel(),
        web=DemoWebResearch(),
        maps=DemoMaps(),
        weather=DemoWeather(),
        clock=DemoClock(),
        status_sink=sink,
    )


# Provisional limits; real values are tuned in later phases (O2).
_EXECUTION_LIMITS: Final = _parsed(
    ExecutionLimits,
    {
        "max_model_calls": 20,
        "max_tool_calls": 40,
        "max_candidates": 5,
        "max_evidence_items": 30,
        "max_elapsed_seconds": 120.0,
    },
)
_API_LIMITS: Final = _parsed(
    ApiLimits,
    {"max_turn_body_bytes": 262144, "max_message_chars": 2000, "max_audio_bytes": 5242880},
)


def build_demo_app() -> FastAPI:
    """The chat API wired to the demo doubles and provisional limits."""
    return create_app(_dependencies_for, DemoTranscriber(), _EXECUTION_LIMITS, _API_LIMITS)


def parse_args() -> argparse.Namespace:
    """Parse the required --port."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--port", type=int, required=True, help="port to listen on at 127.0.0.1")
    return parser.parse_args()


def main() -> None:
    """Serve the demo app on 127.0.0.1 at the given port."""
    port: int = parse_args().port
    uvicorn.run(build_demo_app(), host=HOST, port=port)


if __name__ == "__main__":
    main()
