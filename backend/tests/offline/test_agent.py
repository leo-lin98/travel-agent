"""Phase 2 acceptance tests for the minimal agent loop and revision replay.

Expectations derive from the Phase 2 document (tests, edge cases, acceptance), the
architecture's agent/Python split, docs/decisions.md (2026-11-18: typed model protocol,
web evidence cites a result id, revisions carry the request and chat but not evidence;
2026-11-17: malformed tool calls, evidence gathering and status, Place ID lookup, tool
tracing fields), ledger rulings O1-O8, Phase 1 obligations FR3/FR4, and
artifacts/phase-2/interface-contract.md.

The scripted model dictates every tool call, so these tests check loop mechanics,
Python authority, and outcomes, not the quality of a real model. All data is synthetic
(2026-11-20, Asia/Tokyo) with a manual clock and no network access.
"""

import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import pytest
from agent_doubles import (
    FailingMaps,
    FailingModel,
    FailingWeather,
    FailingWebResearch,
    ManualClock,
    RecordingStatusSink,
    RouteKey,
    ScriptedMaps,
    ScriptedModel,
    ScriptedWeather,
    ScriptedWebResearch,
    route_key,
)
from pydantic import TypeAdapter, ValidationError

from travel_agent.agent import (
    AgentDependencies,
    AssistantTextMessage,
    CandidatePassed,
    CandidateRejected,
    ContinuationOverflowError,
    ConversationMessage,
    DatetimeObservation,
    DeviceLocationMessage,
    EvidenceObservation,
    ExecutionLimits,
    InstructionsMessage,
    LoopNoticeMessage,
    MapsPlace,
    MapsRoute,
    ModelMessage,
    ModelReply,
    PlaceObservation,
    RequestMessage,
    RouteObservation,
    TextReply,
    ToolCall,
    ToolCallReply,
    ToolCallStatus,
    ToolError,
    ToolErrorCode,
    ToolResult,
    ToolResultMessage,
    TurnOutcome,
    WeatherObservation,
    WeatherPeriod,
    WebPage,
    WebSearchObservation,
    run_turn,
)
from travel_agent.contracts import (
    MAX_CONTINUATION_MESSAGES,
    MAX_TEXT_CHARS,
    CandidateItinerary,
    ChatMessage,
    ChatRole,
    Clarification,
    ConstraintId,
    ContinuationPayload,
    Estimate,
    EstimateLabel,
    ExecutionFailure,
    ExecutionFailureReason,
    Location,
    NoFeasibleResult,
    PlanningRequest,
    SourcedFinding,
    SourceKind,
    TransportMode,
    TurnResult,
    TurnStatus,
    UnresolvedTopic,
    ValidatedOptions,
    ValidationFeedback,
)

type Doc = dict[str, object]

TOKYO = ZoneInfo("Asia/Tokyo")
CLOCK_START = datetime(2026, 11, 20, 14, 0, tzinfo=TOKYO)

RESEARCHING = TurnStatus.RESEARCHING
PLANNING = TurnStatus.PLANNING
GENERATING = TurnStatus.GENERATING_ITINERARY

GARDEN = "ChIJsyntheticQuietGarden01"
CAFE = "ChIJsyntheticCornerCafe001"
TRAIL = "ChIJsyntheticRidgeTrail001"
DINNER_PLACE = "ChIJsyntheticGinzaDinner0001"

START_LAT, START_LNG = 35.658, 139.7016
DINNER_LAT, DINNER_LNG = 35.6717, 139.765
GARDEN_LAT, GARDEN_LNG = 35.6764, 139.6993
CAFE_LAT, CAFE_LNG = 35.665, 139.712
TRAIL_LAT, TRAIL_LNG = 35.7012, 139.7203
SPRING_LAT, SPRING_LNG = 35.689, 139.692

Q_GARDEN = "Synthetic Quiet Garden"
Q_GARDEN_NOTICE = "Synthetic Quiet Garden closure notice"
Q_FARE = "Synthetic Metro fare Garden-mae to Ginza"
Q_TAXI = "Tokyo taxi fare per km"
Q_TRAIL = "quiet trail near Shibuya"
Q_CAFE = "Synthetic Corner Cafe"
Q_MISSING_PLACE = "Synthetic Hidden Spring"
Q_SPRING = "Synthetic Hidden Spring official site"
Q_SPRING_BLOG = "Synthetic Hidden Spring blog"

ALL_TOOLS = frozenset(
    {
        "web_search",
        "find_place",
        "get_route",
        "get_weather",
        "record_evidence",
        "update_request",
        "check_candidate",
        "ask_clarification",
        "submit_options",
        "report_no_feasible_result",
        "get_datetime",
    }
)

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPLAY_SCRIPT = BACKEND_DIR / "scripts" / "replay.py"
CONVERSATION_CASE = BACKEND_DIR / "tests" / "fixtures" / "replan_conversation.json"


# --- document builders (pure; every parameter explicit) -----------------------------


def at(clock: str) -> str:
    """Return an ISO datetime on the fixed outing date in Tokyo."""
    return f"2026-11-20T{clock}:00+09:00"


def tokyo(hour: int, minute: int) -> datetime:
    return datetime(2026, 11, 20, hour, minute, tzinfo=TOKYO)


def location_doc(label: str, latitude: float, longitude: float, place_id: str | None) -> Doc:
    return {"label": label, "latitude": latitude, "longitude": longitude, "place_id": place_id}


START_DOC = location_doc("Synthetic Shibuya start", START_LAT, START_LNG, None)
DINNER_DOC = location_doc("Synthetic Ginza dinner", DINNER_LAT, DINNER_LNG, DINNER_PLACE)


def jpy_budget(amount: str) -> Doc:
    return {"amount": amount, "currency": "JPY"}


def request_doc(
    commitment_time: str, budget: Doc | None, transport_modes: list[str], constraints: list[str]
) -> Doc:
    """Free from 15:00 near Shibuya before a Ginza dinner at `commitment_time`."""
    return {
        "schema_version": 1,
        "gap": {"start_time": at("15:00"), "start_location": START_DOC},
        "commitment": {"location": DINNER_DOC, "time": commitment_time, "timezone": "Asia/Tokyo"},
        "budget": budget,
        "transport_modes": transport_modes,
        "constraints": constraints,
        "unresolved_questions": [],
    }


def garden_request() -> Doc:
    """Dinner 19:00 (cutoff 18:45), JPY 3000 cap, any mode, constraint 0 = a garden."""
    return request_doc(at("19:00"), jpy_budget("3000"), [], ["Include a garden"])


def simple_request() -> Doc:
    """Dinner 19:00, JPY 3000 cap, any mode, no explicit constraints."""
    return request_doc(at("19:00"), jpy_budget("3000"), [], [])


EMPTY_REQUEST: Doc = {
    "schema_version": 1,
    "gap": {"start_time": None, "start_location": None},
    "commitment": {"location": None, "time": None, "timezone": None},
    "budget": None,
    "transport_modes": [],
    "constraints": [],
    "unresolved_questions": [],
}


def continuation(messages: list[tuple[str, str]], request: Doc) -> ContinuationPayload:
    document = {
        "schema_version": 1,
        "messages": [{"role": role, "text": text} for role, text in messages],
        "request": request,
    }
    return ContinuationPayload.model_validate_json(json.dumps(document))


def open_claim(intervals: list[tuple[str, str]]) -> Doc:
    return {
        "kind": "opening_access",
        "status": "open",
        "open_intervals": [{"opens_at": at(o), "closes_at": at(c)} for o, c in intervals],
    }


def closed_claim() -> Doc:
    return {"kind": "opening_access", "status": "closed", "open_intervals": []}


def duration_claim(minutes: int) -> Doc:
    return {"kind": "visit_duration", "minutes": minutes}


def cost_claim(amount: str, currency: str) -> Doc:
    return {"kind": "cost_amount", "amount": {"amount": amount, "currency": currency}}


def taxi_rate_claim(base: str, per_km: str) -> Doc:
    return {
        "kind": "taxi_rate",
        "base_fare": {"amount": base, "currency": "JPY"},
        "per_km": {"amount": per_km, "currency": "JPY"},
        "valid_from": "2026-04-01",
        "valid_until": "2027-03-31",
    }


def constraint_claim(index: int) -> Doc:
    return {"kind": "constraint", "constraint_index": index}


def day_scope(place_id: str | None) -> Doc:
    return {
        "place_id": place_id,
        "fallback_key": None,
        "applies_from": at("00:00"),
        "applies_until": at("23:59"),
    }


def sourced_record(
    evidence_id: str, result_id: str, source_kind: str, topic: str, claim: Doc, place_id: str | None
) -> Doc:
    """A web finding the model records by citing a search result id (D2)."""
    return {
        "kind": "sourced_finding",
        "evidence_id": evidence_id,
        "result_id": result_id,
        "source_kind": source_kind,
        "topic": topic,
        "claim": claim,
        "scope": day_scope(place_id),
        "finding": "Synthetic finding read from the cited page.",
    }


def estimate_record(
    evidence_id: str, topic: str, claim: Doc, label: str, place_id: str | None, based_on: list[str]
) -> Doc:
    return {
        "kind": "estimate",
        "evidence_id": evidence_id,
        "topic": topic,
        "claim": claim,
        "label": label,
        "scope": day_scope(place_id),
        "basis": "We couldn't find a better estimate; this is the AI's best guess.",
        "based_on": based_on,
    }


def stop_doc(
    place_id: str,
    latitude: float,
    longitude: float,
    visit_minutes: int,
    evidence_ids: list[str],
) -> Doc:
    return {
        "place_id": place_id,
        "label": f"Stop {place_id[-6:]}",
        "latitude": latitude,
        "longitude": longitude,
        "fallback_key": None,
        "visit_minutes": visit_minutes,
        "evidence_ids": evidence_ids,
    }


def garden_stop(visit_minutes: int, evidence_ids: list[str]) -> Doc:
    return stop_doc(GARDEN, GARDEN_LAT, GARDEN_LNG, visit_minutes, evidence_ids)


def leg_doc(
    mode: str, minutes: int, meters: int, transfers: list[tuple[int, int]], evidence_ids: list[str]
) -> Doc:
    return {
        "mode": mode,
        "duration_minutes": minutes,
        "distance_meters": meters,
        "transfers": [{"available_minutes": a, "walk_minutes": w} for a, w in transfers],
        "evidence_ids": evidence_ids,
    }


def cost_doc(description: str, amount: str, currency: str, evidence_ids: list[str]) -> Doc:
    return {
        "description": description,
        "amount": {"amount": amount, "currency": currency},
        "evidence_ids": evidence_ids,
    }


def candidate_doc(candidate_id: str, stops: list[Doc], legs: list[Doc], costs: list[Doc]) -> Doc:
    return {"candidate_id": candidate_id, "stops": stops, "legs": legs, "costs": costs}


WALK_START_TO_GARDEN = leg_doc("walking", 20, 1500, [], [])
WALK_GARDEN_TO_DINNER = leg_doc("walking", 70, 5200, [], [])
TAXI_GARDEN_TO_DINNER = leg_doc("taxi_rideshare", 15, 5000, [], [])
TRANSIT_GARDEN_TO_DINNER = leg_doc("public_transit", 30, 5000, [(5, 3)], ["ev-fare"])


def garden_candidate(candidate_id: str, visit_minutes: int) -> Doc:
    """Walk to the garden, transit to dinner (JPY 180), JPY 500 admission; cites ev-*."""
    return candidate_doc(
        candidate_id,
        [garden_stop(visit_minutes, ["ev-garden-constraint"])],
        [WALK_START_TO_GARDEN, TRANSIT_GARDEN_TO_DINNER],
        [cost_doc("Garden admission", "500", "JPY", ["ev-admission"])],
    )


def walking_garden_candidate(candidate_id: str, costs: list[Doc]) -> Doc:
    """A 60-minute garden visit, walking both ways, citing nothing on the stop."""
    return candidate_doc(
        candidate_id, [garden_stop(60, [])], [WALK_START_TO_GARDEN, WALK_GARDEN_TO_DINNER], costs
    )


def walking_cafe_candidate(candidate_id: str, costs: list[Doc]) -> Doc:
    return candidate_doc(
        candidate_id,
        [stop_doc(CAFE, CAFE_LAT, CAFE_LNG, 30, [])],
        [leg_doc("walking", 15, 1200, [], []), leg_doc("walking", 60, 4500, [], [])],
        costs,
    )


def replaced(document: object, path: tuple[str | int, ...], value: object) -> object:
    """Return a copy of `document` with `value` set at `path`."""
    if not path:
        return value
    head, tail = path[0], path[1:]
    if isinstance(document, dict) and isinstance(head, str):
        mapping = cast(dict[str, object], document)
        return {**mapping, head: replaced(mapping.get(head), tail, value)}
    if isinstance(document, list) and isinstance(head, int):
        items = cast(list[object], document)
        return [replaced(item, tail, value) if i == head else item for i, item in enumerate(items)]
    raise TypeError(f"path segment {head!r} does not match the document shape")


# --- provider data -----------------------------------------------------------------


def page(url: str, title: str | None, content: str) -> WebPage:
    document = {"url": url, "title": title, "content": content, "retrieved_at": at("14:30")}
    return WebPage.model_validate_json(json.dumps(document))


GARDEN_OFFICIAL = page(
    "https://garden.example.org/visit",
    "Synthetic Quiet Garden - Visit",
    "Open daily 09:00-21:00. Admission JPY 500. A quiet walled garden.",
)
GARDEN_GUIDE = page(
    "https://guide.example.com/quiet-garden", None, "Visitors usually spend about an hour."
)
GARDEN_NOTICE = page(
    "https://city.example.jp/notices/quiet-garden",
    "City notice - Synthetic Quiet Garden",
    "The garden is closed today for maintenance.",
)
METRO_FARE = page(
    "https://metro.example.jp/fares",
    "Synthetic Metro - Fares",
    "Garden-mae to Ginza: adult IC fare JPY 180.",
)
TAXI_RATES = page(
    "https://taxi.example.jp/fares",
    "Synthetic Taxi Association - Fares",
    "From 2026-04-01: JPY 500 base fare, then JPY 300 per km.",
)
TRAIL_ARTICLE = page(
    "https://hiking.example.com/ridge-trail",
    "A quiet walk: the Synthetic Ridge Trail",
    "The Synthetic Ridge Trail is a short wooded trail north of Shibuya.",
)
CAFE_MENU = page("https://cafe.example.com/menu", "Corner Cafe - Menu", "Coffee USD 5.")
SPRING_OFFICIAL = page(
    "https://spring.example.org/",
    "Synthetic Hidden Spring - Official",
    "Open 08:00-18:00. The spring is at 35.689, 139.692.",
)
SPRING_BLOG = page(
    "https://blog.example.com/hidden-spring",
    "A hidden spring in Tokyo",
    "We visited around noon; it seemed open 08:00-18:00. Located at 35.689, 139.692.",
)

PAGES: dict[str, tuple[WebPage, ...]] = {
    Q_GARDEN: (GARDEN_OFFICIAL, GARDEN_GUIDE),
    Q_GARDEN_NOTICE: (GARDEN_NOTICE,),
    Q_FARE: (METRO_FARE,),
    Q_TAXI: (TAXI_RATES,),
    Q_TRAIL: (TRAIL_ARTICLE,),
    Q_CAFE: (CAFE_MENU,),
    Q_SPRING: (SPRING_OFFICIAL,),
    Q_SPRING_BLOG: (SPRING_BLOG,),
}


def maps_place(
    place_id: str, label: str, latitude: float, longitude: float, opening: Doc | None
) -> MapsPlace:
    document = {
        "place_id": place_id,
        "label": label,
        "latitude": latitude,
        "longitude": longitude,
        "maps_url": f"https://maps.google.com/?q=place_id:{place_id}",
        "retrieved_at": at("14:35"),
        "opening": None
        if opening is None
        else {"claim": opening, "applies_from": at("00:00"), "applies_until": at("23:59")},
    }
    return MapsPlace.model_validate_json(json.dumps(document))


GARDEN_ON_MAPS = maps_place(
    GARDEN, "Synthetic Quiet Garden", GARDEN_LAT, GARDEN_LNG, open_claim([("09:00", "21:00")])
)
CAFE_ON_MAPS = maps_place(
    CAFE, "Synthetic Corner Cafe", CAFE_LAT, CAFE_LNG, open_claim([("08:00", "20:00")])
)
TRAIL_ON_MAPS = maps_place(
    TRAIL, "Synthetic Ridge Trail", TRAIL_LAT, TRAIL_LNG, open_claim([("06:00", "20:00")])
)

PLACES: dict[str, MapsPlace | None] = {
    Q_GARDEN: GARDEN_ON_MAPS,
    Q_CAFE: CAFE_ON_MAPS,
    "Synthetic Ridge Trail": TRAIL_ON_MAPS,
    Q_MISSING_PLACE: None,
}


def point(latitude: float, longitude: float) -> Location:
    return Location.model_validate_json(
        json.dumps(location_doc("point", latitude, longitude, None))
    )


START = point(START_LAT, START_LNG)
DINNER = point(DINNER_LAT, DINNER_LNG)
GARDEN_POINT = point(GARDEN_LAT, GARDEN_LNG)
CAFE_POINT = point(CAFE_LAT, CAFE_LNG)
TRAIL_POINT = point(TRAIL_LAT, TRAIL_LNG)
SPRING_POINT = point(SPRING_LAT, SPRING_LNG)


def maps_route(
    minutes: int, meters: int, transfers: list[tuple[int, int]], transit: list[tuple[str, str, str]]
) -> MapsRoute:
    document = {
        "duration_minutes": minutes,
        "distance_meters": meters,
        "transfers": [{"available_minutes": a, "walk_minutes": w} for a, w in transfers],
        "transit": [
            {"line": line, "board_station": board, "alight_station": alight}
            for line, board, alight in transit
        ],
    }
    return MapsRoute.model_validate_json(json.dumps(document))


WALKING = TransportMode.WALKING
TRANSIT = TransportMode.PUBLIC_TRANSIT
TAXI = TransportMode.TAXI_RIDESHARE

ROUTES: dict[RouteKey, MapsRoute] = {
    route_key(START, GARDEN_POINT, WALKING): maps_route(20, 1500, [], []),
    route_key(GARDEN_POINT, DINNER, WALKING): maps_route(70, 5200, [], []),
    route_key(GARDEN_POINT, DINNER, TRANSIT): maps_route(
        30, 5000, [(5, 3)], [("Synthetic Metro Line", "Garden-mae", "Ginza")]
    ),
    route_key(GARDEN_POINT, DINNER, TAXI): maps_route(15, 5000, [], []),
    route_key(START, CAFE_POINT, WALKING): maps_route(15, 1200, [], []),
    route_key(CAFE_POINT, DINNER, WALKING): maps_route(60, 4500, [], []),
    route_key(START, TRAIL_POINT, WALKING): maps_route(30, 2500, [], []),
    route_key(TRAIL_POINT, DINNER, WALKING): maps_route(40, 3000, [], []),
    route_key(START, SPRING_POINT, WALKING): maps_route(25, 2000, [], []),
    route_key(SPRING_POINT, DINNER, WALKING): maps_route(60, 4800, [], []),
}


def rain_period(source_url: str) -> WeatherPeriod:
    document = {
        "applies_from": at("15:00"),
        "applies_until": at("19:00"),
        "rain": True,
        "source_url": source_url,
        "retrieved_at": at("14:40"),
    }
    return WeatherPeriod.model_validate_json(json.dumps(document))


FORECAST_URL = "https://forecast.example.com/tokyo"
FORECASTS = {
    (START_LAT, START_LNG): (rain_period(FORECAST_URL),),
    (GARDEN_LAT, GARDEN_LNG): (rain_period(FORECAST_URL),),
}


# --- model replies -----------------------------------------------------------------


def call(call_id: str, name: str, arguments: Doc) -> ToolCall:
    return ToolCall(call_id=call_id, name=name, arguments=json.dumps(arguments))


def raw_call(call_id: str, name: str, arguments: str) -> ToolCall:
    return ToolCall(call_id=call_id, name=name, arguments=arguments)


def reply(*calls: ToolCall) -> ModelReply:
    return ToolCallReply(kind="tool_calls", calls=calls)


def text_reply(text: str) -> ModelReply:
    return TextReply(kind="text", text=text)


START_POINT: Doc = {"kind": "start"}
COMMITMENT_POINT: Doc = {"kind": "commitment"}


def place_point(place_id: str) -> Doc:
    return {"kind": "place", "place_id": place_id}


def search(call_id: str, query: str) -> ToolCall:
    return call(call_id, "web_search", {"query": query})


def find(call_id: str, query: str) -> ToolCall:
    return call(call_id, "find_place", {"query": query})


def route(call_id: str, origin: Doc, destination: Doc, mode: str) -> ToolCall:
    return call(call_id, "get_route", {"origin": origin, "destination": destination, "mode": mode})


def weather(call_id: str, at_point: Doc) -> ToolCall:
    return call(call_id, "get_weather", {"point": at_point})


def record(call_id: str, evidence: Doc) -> ToolCall:
    return call(call_id, "record_evidence", {"evidence": evidence})


def update(call_id: str, request: Doc) -> ToolCall:
    return call(call_id, "update_request", {"request": request})


def check(call_id: str, candidate: Doc) -> ToolCall:
    return call(call_id, "check_candidate", {"candidate": candidate})


def clarify(call_id: str, message: str, questions: list[tuple[str, str]]) -> ToolCall:
    return call(
        call_id,
        "ask_clarification",
        {
            "message": message,
            "questions": [{"topic": topic, "question": text} for topic, text in questions],
        },
    )


def submit(call_id: str, message: str, recommended: str, candidate_ids: list[str]) -> ToolCall:
    return call(
        call_id,
        "submit_options",
        {
            "message": message,
            "recommendation": {"candidate_id": recommended, "reason": "Closest match."},
            "candidate_ids": candidate_ids,
        },
    )


def no_fit(call_id: str) -> ToolCall:
    return call(
        call_id,
        "report_no_feasible_result",
        {
            "message": "No plan fits your time and budget.",
            "retry_question": "Would you like me to try again?",
        },
    )


def garden_basics(prefix: str) -> list[ModelReply]:
    """Look the garden up, search its pages (web-1 official, web-2 guide), record hours
    (official), a sourced 60-minute visit, and the official JPY 500 admission."""
    return [
        reply(find(f"{prefix}-find", Q_GARDEN), search(f"{prefix}-search", Q_GARDEN)),
        reply(
            record(
                f"{prefix}-hours",
                sourced_record(
                    "ev-garden-hours",
                    "web-1",
                    "official_website",
                    "opening_access",
                    open_claim([("09:00", "21:00")]),
                    GARDEN,
                ),
            ),
            record(
                f"{prefix}-duration",
                sourced_record(
                    "ev-garden-duration",
                    "web-2",
                    "other_web",
                    "visit_duration",
                    duration_claim(60),
                    GARDEN,
                ),
            ),
            record(
                f"{prefix}-admission",
                sourced_record(
                    "ev-admission",
                    "web-1",
                    "official_website",
                    "cost",
                    cost_claim("500", "JPY"),
                    GARDEN,
                ),
            ),
        ),
    ]


def garden_research(prefix: str) -> list[ModelReply]:
    """`garden_basics`, constraint support, both routes, and the operator's fare (web-3)."""
    return [
        *garden_basics(prefix),
        reply(
            record(
                f"{prefix}-constraint",
                sourced_record(
                    "ev-garden-constraint",
                    "web-1",
                    "official_website",
                    "constraint",
                    constraint_claim(0),
                    GARDEN,
                ),
            ),
            route(f"{prefix}-route-in", START_POINT, place_point(GARDEN), "walking"),
            route(f"{prefix}-route-out", place_point(GARDEN), COMMITMENT_POINT, "public_transit"),
            search(f"{prefix}-fare-search", Q_FARE),
        ),
        reply(
            record(
                f"{prefix}-fare",
                sourced_record(
                    "ev-fare", "web-3", "official_website", "cost", cost_claim("180", "JPY"), None
                ),
            )
        ),
    ]


def walking_routes(prefix: str) -> ModelReply:
    return reply(
        route(f"{prefix}-route-in", START_POINT, place_point(GARDEN), "walking"),
        route(f"{prefix}-route-out", place_point(GARDEN), COMMITMENT_POINT, "walking"),
    )


def garden_plan(prefix: str) -> list[ModelReply]:
    """A complete successful plan for `garden_request`: research, check, submit."""
    return [
        *garden_research(prefix),
        reply(check(f"{prefix}-check", garden_candidate("cand-garden", 60))),
        reply(
            submit(f"{prefix}-submit", "Here is a garden visit.", "cand-garden", ["cand-garden"])
        ),
    ]


# --- harness -----------------------------------------------------------------------


@dataclass(frozen=True)
class Harness:
    clock: ManualClock
    model: ScriptedModel
    web: ScriptedWebResearch
    maps: ScriptedMaps
    weather: ScriptedWeather
    sink: RecordingStatusSink
    dependencies: AgentDependencies


def harness_with(
    clock: ManualClock, model: ScriptedModel, web: ScriptedWebResearch, maps: ScriptedMaps
) -> Harness:
    forecasts = ScriptedWeather(FORECASTS)
    sink = RecordingStatusSink()
    dependencies = AgentDependencies(
        model=model, web=web, maps=maps, weather=forecasts, clock=clock, status_sink=sink
    )
    return Harness(clock, model, web, maps, forecasts, sink, dependencies)


def harness(replies: Sequence[ModelReply]) -> Harness:
    clock = ManualClock(CLOCK_START)
    return harness_with(
        clock,
        ScriptedModel(replies, clock, 0),
        ScriptedWebResearch(PAGES, clock, 0),
        ScriptedMaps(PLACES, ROUTES),
    )


def limits(
    model_calls: int, tool_calls: int, candidates: int, evidence: int, elapsed: float
) -> ExecutionLimits:
    document = {
        "max_model_calls": model_calls,
        "max_tool_calls": tool_calls,
        "max_candidates": candidates,
        "max_evidence_items": evidence,
        "max_elapsed_seconds": elapsed,
    }
    return ExecutionLimits.model_validate_json(json.dumps(document))


GENEROUS = limits(40, 80, 10, 60, 600.0)
USER_TEXT = "Plan something before dinner."
USER_TZ = "Asia/Tokyo"


def run(h: Harness, start: ContinuationPayload, user_message: str) -> TurnOutcome:
    return run_turn(start, user_message, USER_TZ, None, h.dependencies, GENEROUS)


def run_garden(replies: Sequence[ModelReply]) -> tuple[Harness, TurnOutcome]:
    h = harness(replies)
    return h, run(h, continuation([], garden_request()), USER_TEXT)


def run_simple(replies: Sequence[ModelReply]) -> tuple[Harness, TurnOutcome]:
    h = harness(replies)
    return h, run(h, continuation([], simple_request()), USER_TEXT)


# --- inspection helpers ----------------------------------------------------------------


def options_of(result: TurnResult) -> ValidatedOptions:
    assert isinstance(result, ValidatedOptions), result
    return result


def failure_of(result: TurnResult) -> ExecutionFailure:
    assert isinstance(result, ExecutionFailure), result
    return result


def clarification_of(result: TurnResult) -> Clarification:
    assert isinstance(result, Clarification), result
    return result


def last_history(model: ScriptedModel) -> tuple[ModelMessage, ...]:
    return model.received[-1][0]


def result_for(model: ScriptedModel, call_id: str) -> ToolResult:
    """The observation the model was sent for `call_id`."""
    for message in last_history(model):
        if isinstance(message, ToolResultMessage) and message.call_id == call_id:
            return message.result
    raise AssertionError(f"no tool result was sent for call {call_id!r}")


def error_for(model: ScriptedModel, call_id: str) -> ToolError:
    result = result_for(model, call_id)
    assert isinstance(result, ToolError), result
    return result


def rejection_for(model: ScriptedModel, call_id: str) -> ValidationFeedback:
    result = result_for(model, call_id)
    assert isinstance(result, CandidateRejected), result
    return result.feedback


def reasons(feedback: ValidationFeedback) -> set[tuple[ConstraintId, str]]:
    return {(failure.constraint, failure.inputs.get("reason", "")) for failure in feedback.failures}


def evidence_by_id(outcome: TurnOutcome, evidence_id: str) -> SourcedFinding | Estimate:
    matches = [item for item in outcome.evidence if item.evidence_id == evidence_id]
    assert len(matches) == 1, [item.evidence_id for item in outcome.evidence]
    return matches[0]


def candidate(document: Doc) -> CandidateItinerary:
    return CandidateItinerary.model_validate_json(json.dumps(document))


def request(document: Doc) -> PlanningRequest:
    return PlanningRequest.model_validate_json(json.dumps(document))


# --- AC1: one loop; the model chooses the path ----------------------------------------


def test_loop_is_built_with_langgraph() -> None:
    assert "langgraph" in sys.modules


def test_successful_plan_returns_checked_options() -> None:
    _, outcome = run_garden(garden_plan("r"))

    options = options_of(outcome.result).options
    assert len(options) == 1
    checked = options[0].itinerary
    assert checked.candidate == candidate(garden_candidate("cand-garden", 60))
    assert checked.total_cost.amount == Decimal("680")
    assert checked.total_cost.currency == "JPY"
    assert checked.total_includes_estimates is False
    assert checked.arrival_at_commitment == tokyo(18, 45)
    assert checked.leave_by == (tokyo(16, 55), tokyo(18, 15))


def test_model_may_research_in_a_different_order() -> None:
    reordered = [
        reply(search("a1", Q_FARE), search("a2", Q_GARDEN)),
        reply(
            route("a3", place_point(GARDEN), COMMITMENT_POINT, "public_transit"),
            find("a4", Q_GARDEN),
        ),
        reply(
            route("a5", place_point(GARDEN), COMMITMENT_POINT, "public_transit"),
            route("a6", START_POINT, place_point(GARDEN), "walking"),
        ),
        reply(
            record(
                "a7",
                sourced_record(
                    "ev-fare", "web-1", "official_website", "cost", cost_claim("180", "JPY"), None
                ),
            ),
            record(
                "a8",
                sourced_record(
                    "ev-garden-hours",
                    "web-2",
                    "official_website",
                    "opening_access",
                    open_claim([("09:00", "21:00")]),
                    GARDEN,
                ),
            ),
            record(
                "a9",
                sourced_record(
                    "ev-garden-duration",
                    "web-3",
                    "other_web",
                    "visit_duration",
                    duration_claim(60),
                    GARDEN,
                ),
            ),
            record(
                "a10",
                sourced_record(
                    "ev-garden-constraint",
                    "web-2",
                    "official_website",
                    "constraint",
                    constraint_claim(0),
                    GARDEN,
                ),
            ),
            record(
                "a11",
                sourced_record(
                    "ev-admission",
                    "web-2",
                    "official_website",
                    "cost",
                    cost_claim("500", "JPY"),
                    GARDEN,
                ),
            ),
        ),
        reply(check("a12", garden_candidate("cand-garden", 60))),
        reply(submit("a13", "Here is a garden visit.", "cand-garden", ["cand-garden"])),
    ]

    h, outcome = run_garden(reordered)

    route_error = error_for(h.model, "a3")
    assert route_error.code == ToolErrorCode.UNKNOWN_PLACE
    options = options_of(outcome.result).options
    assert [option.itinerary.total_cost.amount for option in options] == [Decimal("680")]


def test_model_may_end_a_turn_without_any_research() -> None:
    h, outcome = run_garden([reply(no_fit("n1"))])

    assert isinstance(outcome.result, NoFeasibleResult)
    assert outcome.model_calls == 1
    assert h.web.queries == []
    assert h.maps.place_queries == []
    assert outcome.statuses == ()


# --- AC2: typed tool contracts and observations ----------------------------------------


def test_every_model_call_offers_the_same_typed_tools() -> None:
    h, _ = run_garden(garden_plan("r"))

    first_tools = h.model.received[0][1]
    assert {spec.name for spec in first_tools} == ALL_TOOLS
    assert len(first_tools) == len(ALL_TOOLS)
    for spec in first_tools:
        assert spec.description
        assert spec.parameters["type"] == "object"
        assert "properties" in spec.parameters
    assert all(tools == first_tools for _, tools in h.model.received)


def test_first_model_call_gets_the_conversation_then_the_request() -> None:
    prior = [("user", "I have dinner in Ginza at 7pm."), ("assistant", "Where are you starting?")]
    start = continuation(prior, garden_request())
    h = harness([reply(no_fit("n1"))])

    run(h, start, "Near Shibuya station.")

    history = last_history(h.model)
    conversation = [m.message for m in history if isinstance(m, ConversationMessage)]
    assert conversation == [
        *start.messages,
        ChatMessage(role=ChatRole.USER, text="Near Shibuya station."),
    ]
    requests = [m.request for m in history if isinstance(m, RequestMessage)]
    assert requests == [start.request]


def test_research_observations_are_typed_and_python_built() -> None:
    h, _ = run_garden(garden_plan("r"))

    found = result_for(h.model, "r-find")
    assert isinstance(found, PlaceObservation)
    assert (found.place.place_id, found.place.latitude, found.place.longitude) == (
        GARDEN,
        GARDEN_LAT,
        GARDEN_LNG,
    )

    results = result_for(h.model, "r-search")
    assert isinstance(results, WebSearchObservation)
    assert [(item.result_id, item.url) for item in results.results] == [
        ("web-1", GARDEN_OFFICIAL.url),
        ("web-2", GARDEN_GUIDE.url),
    ]
    assert results.results[0].retrieved_at == GARDEN_OFFICIAL.retrieved_at

    routed = result_for(h.model, "r-route-out")
    assert isinstance(routed, RouteObservation)
    assert routed.leg.mode == TRANSIT
    assert (routed.leg.duration_minutes, routed.leg.distance_meters) == (30, 5000)
    assert [(t.available_minutes, t.walk_minutes) for t in routed.leg.transfers] == [(5, 3)]
    assert routed.leg.evidence_ids == ()
    assert [(s.line, s.board_station, s.alight_station) for s in routed.transit] == [
        ("Synthetic Metro Line", "Garden-mae", "Ginza")
    ]

    checked = result_for(h.model, "r-check")
    assert isinstance(checked, CandidatePassed)
    assert checked.checked.total_cost.amount == Decimal("680")


def test_recorded_web_evidence_copies_url_and_time_from_the_cited_result() -> None:
    h, outcome = run_garden(garden_plan("r"))

    observed = result_for(h.model, "r-hours")
    assert isinstance(observed, EvidenceObservation)
    stored = evidence_by_id(outcome, "ev-garden-hours")
    assert observed.evidence == stored
    assert isinstance(stored, SourcedFinding)
    assert stored.source.url == GARDEN_OFFICIAL.url
    assert stored.source.title == GARDEN_OFFICIAL.title
    assert stored.source.kind == SourceKind.OFFICIAL_WEBSITE
    assert stored.retrieved_at == GARDEN_OFFICIAL.retrieved_at

    duration = evidence_by_id(outcome, "ev-garden-duration")
    assert isinstance(duration, SourcedFinding)
    assert duration.source.url == GARDEN_GUIDE.url
    assert duration.source.kind == SourceKind.OTHER_WEB


def test_request_without_locations_gives_a_structured_route_error() -> None:
    h = harness([reply(route("q1", START_POINT, COMMITMENT_POINT, "walking")), reply(no_fit("n1"))])

    run(h, continuation([], EMPTY_REQUEST), "Help me plan.")

    assert error_for(h.model, "q1").code == ToolErrorCode.REQUEST_INCOMPLETE
    assert h.maps.route_requests == []


# --- AC3: malformed tool calls ----------------------------------------------------------

UNPARSEABLE = '{"query": '
VALID_RECORD = sourced_record(
    "ev-x", "web-1", "official_website", "visit_duration", duration_claim(60), GARDEN
)


@pytest.mark.parametrize(
    ("name", "arguments", "code", "location"),
    [
        pytest.param("book_taxi", '{"query": "x"}', ToolErrorCode.UNKNOWN_TOOL, None, id="unknown"),
        pytest.param(
            "web_search", UNPARSEABLE, ToolErrorCode.UNPARSEABLE_ARGUMENTS, None, id="unparseable"
        ),
        pytest.param(
            "web_search",
            json.dumps({"q": Q_GARDEN}),
            ToolErrorCode.INVALID_ARGUMENTS,
            "query",
            id="missing-field",
        ),
        pytest.param(
            "web_search", json.dumps([Q_GARDEN]), ToolErrorCode.INVALID_ARGUMENTS, None, id="array"
        ),
        pytest.param(
            "record_evidence",
            json.dumps({"evidence": {**VALID_RECORD, "source_kind": "google_maps"}}),
            ToolErrorCode.INVALID_ARGUMENTS,
            "source_kind",
            id="maps-kind-from-web",
        ),
        pytest.param(
            "record_evidence",
            json.dumps({"evidence": {**VALID_RECORD, "url": "https://garden.example.org/"}}),
            ToolErrorCode.INVALID_ARGUMENTS,
            "url",
            id="model-supplied-url",
        ),
        pytest.param(
            "record_evidence",
            json.dumps({"evidence": {**VALID_RECORD, "topic": "cost"}}),
            ToolErrorCode.INVALID_ARGUMENTS,
            "claim",
            id="claim-topic-mismatch",
        ),
        pytest.param(
            "update_request",
            json.dumps({"request": replaced(simple_request(), ("commitment", "timezone"), "Mars")}),
            ToolErrorCode.INVALID_ARGUMENTS,
            "timezone",
            id="invalid-request",
        ),
        pytest.param(
            "submit_options",
            json.dumps({"message": "Done.", "candidate_ids": ["cand-garden"]}),
            ToolErrorCode.INVALID_ARGUMENTS,
            "recommendation",
            id="terminal-missing-field",
        ),
    ],
)
def test_malformed_call_is_not_executed_and_the_model_can_correct_it(
    name: str, arguments: str, code: ToolErrorCode, location: str | None
) -> None:
    start = continuation([], simple_request())
    h = harness(
        [
            reply(raw_call("bad", name, arguments)),
            reply(search("fixed", Q_GARDEN)),
            reply(no_fit("n1")),
        ]
    )

    outcome = run(h, start, USER_TEXT)

    error = error_for(h.model, "bad")
    assert error.code == code
    assert error.tool_name == name
    assert error.message
    if location is not None:
        assert any(location in detail.location for detail in error.details), error.details
    assert h.web.queries == [Q_GARDEN]
    assert outcome.evidence == ()
    assert outcome.continuation.request == start.request
    first = outcome.tool_calls[0]
    assert (first.tool_name, first.status, first.arguments) == (
        name,
        ToolCallStatus.MALFORMED,
        arguments,
    )
    assert outcome.statuses == (RESEARCHING,)
    assert isinstance(outcome.result, NoFeasibleResult)


def test_repeated_malformed_calls_end_in_limit_termination() -> None:
    h = harness([reply(raw_call(f"bad-{i}", "book_taxi", "{}")) for i in range(5)])

    outcome = run_turn(
        continuation([], simple_request()),
        USER_TEXT,
        USER_TZ,
        None,
        h.dependencies,
        limits(40, 3, 5, 10, 600.0),
    )

    assert failure_of(outcome.result).reason == ExecutionFailureReason.EXECUTION_LIMIT_REACHED
    assert [entry.status for entry in outcome.tool_calls] == [ToolCallStatus.MALFORMED] * 3
    assert outcome.statuses == ()


def test_malformed_calls_count_toward_the_tool_budget() -> None:
    h = harness(
        [
            reply(
                raw_call("bad-1", "book_taxi", "{}"),
                raw_call("bad-2", "web_search", UNPARSEABLE),
                search("ok", Q_GARDEN),
            )
        ]
    )

    outcome = run_turn(
        continuation([], simple_request()),
        USER_TEXT,
        USER_TZ,
        None,
        h.dependencies,
        limits(40, 2, 5, 10, 600.0),
    )

    assert failure_of(outcome.result).reason == ExecutionFailureReason.EXECUTION_LIMIT_REACHED
    assert h.web.queries == []
    assert len(outcome.tool_calls) == 2


def test_text_reply_is_not_a_result_and_the_model_is_asked_to_act() -> None:
    h = harness(
        [
            text_reply("Your total will be JPY 100."),
            reply(clarify("c1", "Where are you starting?", [("start_location", "Where from?")])),
        ]
    )

    outcome = run(h, continuation([], EMPTY_REQUEST), "Plan my afternoon.")

    assert isinstance(outcome.result, Clarification)
    assert outcome.model_calls == 2
    second = h.model.received[1][0]
    assert AssistantTextMessage(kind="assistant_text", text="Your total will be JPY 100.") in second
    assert any(isinstance(m, LoopNoticeMessage) and m.code == "no_action" for m in second)


# --- AC4: rejection followed by repair ------------------------------------------------------


def test_rejected_proposal_is_repaired_in_the_same_run() -> None:
    replies = [
        *garden_research("r"),
        reply(check("long", garden_candidate("cand-garden-long", 200))),
        reply(check("short", garden_candidate("cand-garden-short", 60))),
        reply(submit("s", "A shorter visit fits.", "cand-garden-short", ["cand-garden-short"])),
    ]

    h, outcome = run_garden(replies)

    rejected = rejection_for(h.model, "long")
    assert rejected.candidate_id == "cand-garden-long"
    assert (ConstraintId.ARRIVAL_BUFFER, "") in reasons(rejected)
    assert isinstance(result_for(h.model, "short"), CandidatePassed)
    assert outcome.validation_feedback == (rejected,)
    ids = [o.itinerary.candidate.candidate_id for o in options_of(outcome.result).options]
    assert ids == ["cand-garden-short"]


def test_rejection_reports_every_failure_at_once() -> None:
    late_and_uncited = replaced(
        garden_candidate("cand-garden-bad", 200), ("stops", 0, "evidence_ids"), ["ev-unrecorded"]
    )
    replies = [
        *garden_research("r"),
        reply(check("bad", cast(Doc, late_and_uncited))),
        reply(no_fit("n1")),
    ]

    h, _ = run_garden(replies)

    found = reasons(rejection_for(h.model, "bad"))
    assert (ConstraintId.ARRIVAL_BUFFER, "") in found
    assert (ConstraintId.MISSING_EVIDENCE, "unknown_evidence_id") in found
    assert (ConstraintId.EXPLICIT_CONSTRAINT, "") in found


# --- AC5: clarification before planning ------------------------------------------------------

CURRENCY_QUESTION = ("budget_currency", "Which currency is your 3000 budget in?")


def unresolved_currency_request() -> Doc:
    document = request_doc(at("19:00"), {"amount": "3000", "currency": None}, [], [])
    return {
        **document,
        "unresolved_questions": [{"topic": CURRENCY_QUESTION[0], "question": CURRENCY_QUESTION[1]}],
    }


def test_missing_currency_is_clarified_then_planning_continues() -> None:
    first = harness(
        [
            reply(update("u1", unresolved_currency_request())),
            reply(clarify("c1", "Which currency should I use?", [CURRENCY_QUESTION])),
        ]
    )

    asked = run(first, continuation([], EMPTY_REQUEST), "Dinner in Ginza at 7. I have 3000.")

    questions = clarification_of(asked.result).questions
    assert [q.topic for q in questions] == [UnresolvedTopic.BUDGET_CURRENCY]
    assert GENERATING not in asked.statuses
    budget = asked.continuation.request.budget
    assert budget is not None and budget.currency is None

    second = harness([reply(update("u2", garden_request())), *garden_plan("r")])
    planned = run(second, asked.continuation, "Yen. And I'd like a garden.")

    carried = [m.request for m in second.model.received[0][0] if isinstance(m, RequestMessage)]
    assert carried == [asked.continuation.request]
    assert len(options_of(planned.result).options) == 1
    final_budget = planned.continuation.request.budget
    assert final_budget is not None and final_budget.currency == "JPY"


def test_checking_with_an_unresolved_currency_is_a_structured_error() -> None:
    h = harness(
        [
            *garden_basics("r"),
            walking_routes("w"),
            reply(check("k", walking_garden_candidate("cand-walk", []))),
            reply(clarify("c1", "Which currency should I use?", [CURRENCY_QUESTION])),
        ]
    )

    outcome = run(h, continuation([], unresolved_currency_request()), USER_TEXT)

    assert error_for(h.model, "k").code == ToolErrorCode.BUDGET_CURRENCY_UNRESOLVED
    assert isinstance(outcome.result, Clarification)
    assert outcome.validation_feedback == ()


RELAXING_QUESTION = ("other", "What makes a place relaxing for you?")


def test_vague_constraint_is_clarified_before_planning() -> None:
    vague = {
        **simple_request(),
        "unresolved_questions": [{"topic": RELAXING_QUESTION[0], "question": RELAXING_QUESTION[1]}],
    }
    h = harness(
        [
            reply(update("u1", vague)),
            reply(clarify("c1", "Happy to help.", [RELAXING_QUESTION])),
        ]
    )

    outcome = run(h, continuation([], simple_request()), "Something relaxing before dinner.")

    questions = clarification_of(outcome.result).questions
    assert [(q.topic, q.question) for q in questions] == [
        (UnresolvedTopic.OTHER, RELAXING_QUESTION[1])
    ]
    assert outcome.statuses == ()
    assert "check_candidate" not in [entry.tool_name for entry in outcome.tool_calls]


def test_plan_against_a_vague_constraint_without_sourced_support_is_rejected() -> None:
    relaxing = request_doc(at("19:00"), jpy_budget("3000"), [], ["Something relaxing"])
    guess = estimate_record(
        "ev-relaxing-guess", "constraint", constraint_claim(0), "ai_best_guess", GARDEN, []
    )
    planned = candidate_doc(
        "cand-relaxing",
        [garden_stop(60, ["ev-relaxing-guess"])],
        [WALK_START_TO_GARDEN, WALK_GARDEN_TO_DINNER],
        [],
    )
    h = harness(
        [
            *garden_basics("r"),
            reply(record("g", guess)),
            walking_routes("w"),
            reply(check("k", planned)),
            reply(clarify("c1", "What makes a place relaxing for you?", [RELAXING_QUESTION])),
        ]
    )

    outcome = run(h, continuation([], relaxing), "Something relaxing before dinner.")

    assert (ConstraintId.EXPLICIT_CONSTRAINT, "") in reasons(rejection_for(h.model, "k"))
    assert isinstance(outcome.result, Clarification)
    assert outcome.statuses == (RESEARCHING, PLANNING)


# --- AC6: stale values, revisions, carry-forward ------------------------------------------------


def test_revision_carries_request_and_conversation_but_regathers_evidence() -> None:
    first_h, first = run_garden(garden_plan("r"))
    assert first.continuation.messages == (
        ChatMessage(role=ChatRole.USER, text=USER_TEXT),
        ChatMessage(role=ChatRole.ASSISTANT, text="Here is a garden visit."),
    )
    revised_request = request_doc(at("18:00"), jpy_budget("3000"), [], ["Include a garden"])
    revision = "My dinner moved to 6pm."
    second_h = harness(
        [
            reply(update("u1", revised_request)),
            reply(submit("old-submit", "Same plan.", "cand-garden", ["cand-garden"])),
            reply(check("old-check", garden_candidate("cand-garden", 60))),
            *garden_research("t2"),
            reply(check("new-check", garden_candidate("cand-garden-6pm", 60))),
            reply(submit("new-submit", "Updated plan.", "cand-garden-6pm", ["cand-garden-6pm"])),
        ]
    )

    second = run(second_h, first.continuation, revision)

    opening = second_h.model.received[0][0]
    conversation = [m.message for m in opening if isinstance(m, ConversationMessage)]
    assert conversation == [
        *first.continuation.messages,
        ChatMessage(role=ChatRole.USER, text=revision),
    ]
    assert [m.request for m in opening if isinstance(m, RequestMessage)] == [
        first.continuation.request
    ]

    assert error_for(second_h.model, "old-submit").code == ToolErrorCode.UNKNOWN_CANDIDATE
    stale = reasons(rejection_for(second_h.model, "old-check"))
    assert (ConstraintId.MISSING_EVIDENCE, "unknown_place") in stale
    assert (ConstraintId.MISSING_EVIDENCE, "unverified_leg") in stale
    assert (ConstraintId.MISSING_EVIDENCE, "unknown_evidence_id") in stale

    assert second_h.web.queries == first_h.web.queries == [Q_GARDEN, Q_FARE]
    assert second_h.maps.place_queries == [Q_GARDEN]

    option = options_of(second.result).options[0].itinerary
    assert option.candidate.candidate_id == "cand-garden-6pm"
    assert option.arrival_at_commitment <= tokyo(17, 45)

    before, after = first.continuation.request, second.continuation.request
    assert after.commitment.time == tokyo(18, 0)
    assert (after.gap, after.budget, after.constraints, after.transport_modes) == (
        before.gap,
        before.budget,
        before.constraints,
        before.transport_modes,
    )
    assert (after.commitment.location, after.commitment.timezone) == (
        before.commitment.location,
        before.commitment.timezone,
    )
    assert second.continuation.messages == (
        *first.continuation.messages,
        ChatMessage(role=ChatRole.USER, text=revision),
        ChatMessage(role=ChatRole.ASSISTANT, text="Updated plan."),
    )


def test_superseded_request_value_is_used_by_the_final_gate() -> None:
    earlier_dinner = request_doc(at("17:00"), jpy_budget("3000"), [], ["Include a garden"])
    replies = [
        *garden_research("r"),
        reply(check("k", garden_candidate("cand-garden", 60))),
        reply(update("u", earlier_dinner)),
        reply(submit("s", "Here is a garden visit.", "cand-garden", ["cand-garden"])),
        reply(no_fit("n1")),
    ]

    h, outcome = run_garden(replies)

    assert isinstance(result_for(h.model, "k"), CandidatePassed)
    gate = error_for(h.model, "s")
    assert gate.code == ToolErrorCode.GATE_REJECTED
    assert [item.candidate_id for item in gate.feedback] == ["cand-garden"]
    assert (ConstraintId.ARRIVAL_BUFFER, "") in reasons(gate.feedback[0])
    assert outcome.validation_feedback == gate.feedback
    assert isinstance(outcome.result, NoFeasibleResult)
    assert outcome.continuation.request == request(earlier_dinner)


def test_route_from_a_superseded_start_is_not_accepted() -> None:
    moved_start = replaced(
        simple_request(),
        ("gap", "start_location"),
        location_doc("Synthetic Harajuku start", 35.6702, 139.7027, None),
    )
    replies = [
        *garden_basics("r"),
        walking_routes("w"),
        reply(update("u", cast(Doc, moved_start))),
        reply(check("k", walking_garden_candidate("cand-walk", []))),
        reply(no_fit("n1")),
    ]

    h, _ = run_simple(replies)

    stale = rejection_for(h.model, "k")
    legs = [
        failure.inputs.get("leg_index")
        for failure in stale.failures
        if failure.inputs.get("reason") == "unverified_leg"
    ]
    assert legs == ["0"]


# --- AC7: unknown references ---------------------------------------------------------------


def test_unknown_candidate_in_a_submission_is_a_structured_error() -> None:
    replies = [
        *garden_research("r"),
        reply(check("k", garden_candidate("cand-garden", 60))),
        reply(submit("s1", "Done.", "cand-imaginary", ["cand-imaginary"])),
        reply(submit("s2", "Done.", "cand-garden", ["cand-garden"])),
    ]

    h, outcome = run_garden(replies)

    error = error_for(h.model, "s1")
    assert error.code == ToolErrorCode.UNKNOWN_CANDIDATE
    assert any("cand-imaginary" in detail.problem for detail in error.details), error.details
    assert len(options_of(outcome.result).options) == 1


def two_walking_plans(prefix: str) -> list[ModelReply]:
    """Research and pass a walking garden plan and a walking cafe plan (no budget)."""
    coffee = cost_doc("Coffee", "5", "USD", ["ev-coffee"])
    admission = cost_doc("Garden admission", "500", "JPY", ["ev-admission"])
    return [
        *garden_basics(prefix),
        walking_routes(prefix),
        reply(
            find(f"{prefix}-cafe-find", Q_CAFE),
            search(f"{prefix}-cafe-search", Q_CAFE),
            route(f"{prefix}-cafe-in", START_POINT, place_point(CAFE), "walking"),
            route(f"{prefix}-cafe-out", place_point(CAFE), COMMITMENT_POINT, "walking"),
        ),
        reply(
            record(
                f"{prefix}-coffee",
                sourced_record(
                    "ev-coffee", "web-3", "official_website", "cost", cost_claim("5", "USD"), CAFE
                ),
            ),
            record(
                f"{prefix}-cafe-duration",
                estimate_record(
                    "ev-cafe-duration",
                    "visit_duration",
                    duration_claim(30),
                    "ai_best_guess",
                    CAFE,
                    [],
                ),
            ),
        ),
        reply(
            check(f"{prefix}-garden", walking_garden_candidate("cand-garden-walk", [admission])),
            check(f"{prefix}-cafe", walking_cafe_candidate("cand-cafe-walk", [coffee])),
        ),
    ]


NO_BUDGET = request_doc(at("19:00"), None, [], [])


def test_recommendation_outside_the_submission_becomes_repair_feedback() -> None:
    h = harness(
        [
            *two_walking_plans("p"),
            reply(submit("s1", "Done.", "cand-cafe-walk", ["cand-garden-walk"])),
            reply(submit("s2", "Done.", "cand-garden-walk", ["cand-garden-walk"])),
        ]
    )

    outcome = run(h, continuation([], NO_BUDGET), USER_TEXT)

    assert isinstance(result_for(h.model, "p-cafe"), CandidatePassed)
    assert error_for(h.model, "s1").code == ToolErrorCode.RECOMMENDATION_NOT_FOUND
    ids = [o.itinerary.candidate.candidate_id for o in options_of(outcome.result).options]
    assert ids == ["cand-garden-walk"]


def test_mixed_currencies_without_a_budget_become_repair_feedback() -> None:
    h = harness(
        [
            *two_walking_plans("p"),
            reply(
                submit("s1", "Done.", "cand-garden-walk", ["cand-garden-walk", "cand-cafe-walk"])
            ),
            reply(submit("s2", "Done.", "cand-garden-walk", ["cand-garden-walk"])),
        ]
    )

    outcome = run(h, continuation([], NO_BUDGET), USER_TEXT)

    assert isinstance(result_for(h.model, "p-garden"), CandidatePassed)
    assert isinstance(result_for(h.model, "p-cafe"), CandidatePassed)
    assert error_for(h.model, "s1").code == ToolErrorCode.MIXED_CURRENCY
    assert len(options_of(outcome.result).options) == 1


def test_unknown_result_id_is_rejected_and_nothing_is_stored() -> None:
    h = harness(
        [
            reply(search("s", Q_GARDEN)),
            reply(
                record(
                    "r",
                    sourced_record(
                        "ev-x",
                        "web-9",
                        "official_website",
                        "visit_duration",
                        duration_claim(60),
                        GARDEN,
                    ),
                )
            ),
            reply(no_fit("n1")),
        ]
    )

    outcome = run(h, continuation([], simple_request()), USER_TEXT)

    error = error_for(h.model, "r")
    assert error.code == ToolErrorCode.UNKNOWN_RESULT_ID
    assert outcome.evidence == ()
    assert [entry.status for entry in outcome.tool_calls][1] == ToolCallStatus.ERROR


def test_estimate_based_on_unknown_evidence_is_rejected() -> None:
    derived = estimate_record(
        "ev-derived", "cost", cost_claim("900", "JPY"), "estimate", GARDEN, ["ev-never-recorded"]
    )
    h = harness([reply(record("r", derived)), reply(no_fit("n1"))])

    outcome = run(h, continuation([], simple_request()), USER_TEXT)

    assert error_for(h.model, "r").code == ToolErrorCode.UNKNOWN_EVIDENCE_ID
    assert outcome.evidence == ()


def test_duplicate_evidence_id_is_rejected() -> None:
    guess = estimate_record(
        "ev-guess", "visit_duration", duration_claim(45), "ai_best_guess", GARDEN, []
    )
    h = harness([reply(record("r1", guess)), reply(record("r2", guess)), reply(no_fit("n1"))])

    outcome = run(h, continuation([], simple_request()), USER_TEXT)

    assert isinstance(result_for(h.model, "r1"), EvidenceObservation)
    assert error_for(h.model, "r2").code == ToolErrorCode.DUPLICATE_EVIDENCE_ID
    assert len(outcome.evidence) == 1


def guess_with_id(evidence_id: str) -> Doc:
    return estimate_record(
        evidence_id, "visit_duration", duration_claim(45), "ai_best_guess", GARDEN, []
    )


def sourced_with_id(evidence_id: str) -> Doc:
    return sourced_record(
        evidence_id, "web-1", "other_web", "visit_duration", duration_claim(60), GARDEN
    )


@pytest.mark.parametrize(
    ("evidence_id", "build"),
    [
        pytest.param("web-1", guess_with_id, id="web-estimate"),
        pytest.param("maps-1", guess_with_id, id="maps-estimate"),
        pytest.param("weather-1", guess_with_id, id="weather-estimate"),
        pytest.param("maps-7", sourced_with_id, id="maps-sourced"),
    ],
)
def test_reserved_evidence_id_prefix_is_rejected_and_can_be_corrected(
    evidence_id: str, build: Callable[[str], Doc]
) -> None:
    h = harness(
        [
            reply(search("s", Q_GARDEN)),
            reply(record("bad", build(evidence_id))),
            reply(record("good", build("ev-corrected"))),
            reply(no_fit("n1")),
        ]
    )

    outcome = run(h, continuation([], simple_request()), USER_TEXT)

    error = error_for(h.model, "bad")
    assert error.code == ToolErrorCode.INVALID_ARGUMENTS
    assert any("evidence_id" in detail.location for detail in error.details), error.details
    assert isinstance(result_for(h.model, "good"), EvidenceObservation)
    assert [item.evidence_id for item in outcome.evidence] == ["ev-corrected"]
    assert [(entry.call_id, entry.status) for entry in outcome.tool_calls][1:3] == [
        ("bad", ToolCallStatus.MALFORMED),
        ("good", ToolCallStatus.SUCCESS),
    ]


def test_reserved_prefixed_evidence_id_counts_toward_the_tool_budget() -> None:
    h = harness(
        [reply(record("bad", guess_with_id("maps-1")), record("good", guess_with_id("ev-ok")))]
    )

    outcome = run_turn(
        continuation([], simple_request()),
        USER_TEXT,
        USER_TZ,
        None,
        h.dependencies,
        limits(40, 1, 10, 60, 600.0),
    )

    assert failure_of(outcome.result).reason == ExecutionFailureReason.EXECUTION_LIMIT_REACHED
    assert outcome.evidence == ()


def test_ids_that_only_resemble_reserved_prefixes_are_accepted() -> None:
    h = harness(
        [
            reply(
                record("a", guess_with_id("website-hours")), record("b", guess_with_id("Maps-1"))
            ),
            reply(no_fit("n1")),
        ]
    )

    outcome = run(h, continuation([], simple_request()), USER_TEXT)

    assert [item.evidence_id for item in outcome.evidence] == ["website-hours", "Maps-1"]


def test_route_to_a_place_not_looked_up_is_rejected() -> None:
    h = harness(
        [reply(route("q", START_POINT, place_point(GARDEN), "walking")), reply(no_fit("n1"))]
    )

    run(h, continuation([], simple_request()), USER_TEXT)

    assert error_for(h.model, "q").code == ToolErrorCode.UNKNOWN_PLACE
    assert h.maps.route_requests == []


def test_place_missing_on_maps_is_a_structured_error() -> None:
    h = harness([reply(find("f", Q_MISSING_PLACE)), reply(no_fit("n1"))])

    outcome = run(h, continuation([], simple_request()), USER_TEXT)

    assert error_for(h.model, "f").code == ToolErrorCode.PLACE_NOT_FOUND
    assert outcome.evidence == ()


@pytest.mark.parametrize(
    ("path", "value", "constraint", "reason", "index"),
    [
        pytest.param(
            ("stops", 0, "place_id"),
            "ChIJsyntheticNeverLooked01",
            ConstraintId.MISSING_EVIDENCE,
            "unknown_place",
            ("stop_index", "0"),
            id="place-not-looked-up",
        ),
        pytest.param(
            ("stops", 0, "latitude"),
            35.6801,
            ConstraintId.MISSING_EVIDENCE,
            "place_mismatch",
            ("stop_index", "0"),
            id="moved-coordinates",
        ),
        pytest.param(
            ("legs", 0, "duration_minutes"),
            10,
            ConstraintId.MISSING_EVIDENCE,
            "unverified_leg",
            ("leg_index", "0"),
            id="shortened-leg",
        ),
        pytest.param(
            ("legs", 1, "mode"),
            "taxi_rideshare",
            ConstraintId.MISSING_EVIDENCE,
            "unverified_leg",
            ("leg_index", "1"),
            id="mode-without-route",
        ),
        pytest.param(
            ("stops", 0, "evidence_ids"),
            ["ev-unrecorded"],
            ConstraintId.MISSING_EVIDENCE,
            "unknown_evidence_id",
            ("evidence_id", "ev-unrecorded"),
            id="uncited-evidence",
        ),
    ],
)
def test_candidate_values_must_come_from_this_requests_providers(
    path: tuple[str | int, ...],
    value: object,
    constraint: ConstraintId,
    reason: str,
    index: tuple[str, str],
) -> None:
    altered = cast(Doc, replaced(walking_garden_candidate("cand-walk", []), path, value))
    replies = [
        *garden_basics("r"),
        walking_routes("w"),
        reply(check("ok", walking_garden_candidate("cand-walk-ok", []))),
        reply(check("bad", altered)),
        reply(no_fit("n1")),
    ]

    h, outcome = run_simple(replies)

    assert isinstance(result_for(h.model, "ok"), CandidatePassed)
    feedback = rejection_for(h.model, "bad")
    key, expected = index
    assert any(
        failure.constraint == constraint
        and failure.inputs.get("reason") == reason
        and failure.inputs.get(key) == expected
        for failure in feedback.failures
    ), feedback.failures
    assert outcome.validation_feedback == (feedback,)


# --- AC8: distinct unsuccessful outcomes --------------------------------------------------------


def test_web_provider_failure_is_an_execution_failure() -> None:
    clock = ManualClock(CLOCK_START)
    model = ScriptedModel([reply(search("s", Q_GARDEN)), reply(no_fit("n1"))], clock, 0)
    sink = RecordingStatusSink()
    dependencies = AgentDependencies(
        model=model,
        web=FailingWebResearch(),
        maps=ScriptedMaps(PLACES, ROUTES),
        weather=ScriptedWeather(FORECASTS),
        clock=clock,
        status_sink=sink,
    )
    start = continuation([], simple_request())

    outcome = run_turn(start, USER_TEXT, USER_TZ, None, dependencies, GENEROUS)

    assert failure_of(outcome.result).reason == ExecutionFailureReason.PROVIDER_FAILURE
    assert [(r.tool_name, r.status) for r in outcome.tool_calls] == [
        ("web_search", ToolCallStatus.ERROR)
    ]
    assert len(model.received) == 1
    assert outcome.continuation.request == start.request
    assert [m.role for m in outcome.continuation.messages] == [ChatRole.USER, ChatRole.ASSISTANT]


def test_maps_failure_is_an_execution_failure() -> None:
    clock = ManualClock(CLOCK_START)
    dependencies = AgentDependencies(
        model=ScriptedModel([reply(find("f", Q_GARDEN)), reply(no_fit("n1"))], clock, 0),
        web=ScriptedWebResearch(PAGES, clock, 0),
        maps=FailingMaps(),
        weather=ScriptedWeather(FORECASTS),
        clock=clock,
        status_sink=RecordingStatusSink(),
    )

    outcome = run_turn(
        continuation([], simple_request()), USER_TEXT, USER_TZ, None, dependencies, GENEROUS
    )

    assert failure_of(outcome.result).reason == ExecutionFailureReason.PROVIDER_FAILURE


def test_model_failure_is_an_execution_failure() -> None:
    clock = ManualClock(CLOCK_START)
    dependencies = AgentDependencies(
        model=FailingModel(),
        web=ScriptedWebResearch(PAGES, clock, 0),
        maps=ScriptedMaps(PLACES, ROUTES),
        weather=ScriptedWeather(FORECASTS),
        clock=clock,
        status_sink=RecordingStatusSink(),
    )

    outcome = run_turn(
        continuation([], simple_request()), USER_TEXT, USER_TZ, None, dependencies, GENEROUS
    )

    assert failure_of(outcome.result).reason == ExecutionFailureReason.PROVIDER_FAILURE


def test_no_feasible_result_asks_before_retrying_and_the_conversation_continues() -> None:
    replies = [
        *garden_research("r"),
        reply(check("long", garden_candidate("cand-garden-long", 200))),
        reply(no_fit("n1")),
    ]
    _, first = run_garden(replies)

    assert isinstance(first.result, NoFeasibleResult)
    assert first.result.retry_question == "Would you like me to try again?"
    assert first.continuation.request == request(garden_request())

    second_h = harness(garden_plan("t2"))
    second = run(second_h, first.continuation, "Yes, please try again.")

    assert len(options_of(second.result).options) == 1
    assert len(second.continuation.messages) == 4


# --- AC9: evidence-gathering behaviours -------------------------------------------------------


def notice_and_duration(prefix: str, notice: Doc) -> list[ModelReply]:
    """Look the garden up on Maps, read an official page, and record hours and a duration."""
    return [
        reply(find(f"{prefix}-find", Q_GARDEN), search(f"{prefix}-search", Q_GARDEN_NOTICE)),
        reply(
            record(f"{prefix}-hours", notice),
            record(
                f"{prefix}-duration",
                estimate_record(
                    "ev-garden-duration",
                    "visit_duration",
                    duration_claim(60),
                    "ai_best_guess",
                    GARDEN,
                    [],
                ),
            ),
        ),
        walking_routes(prefix),
    ]


def test_official_closure_wins_over_google_maps_hours() -> None:
    closure = sourced_record(
        "ev-garden-notice",
        "web-1",
        "official_authority_notice",
        "opening_access",
        closed_claim(),
        GARDEN,
    )
    replies = [
        *notice_and_duration("r", closure),
        reply(check("k", walking_garden_candidate("cand-walk", []))),
        reply(no_fit("n1")),
    ]

    h, _ = run_simple(replies)

    found = result_for(h.model, "r-find")
    assert isinstance(found, PlaceObservation)
    assert [item.source.kind for item in found.evidence if isinstance(item, SourcedFinding)] == [
        SourceKind.GOOGLE_MAPS
    ]
    assert (ConstraintId.OPENING_ACCESS, "closed") in reasons(rejection_for(h.model, "k"))


def test_official_opening_wins_over_a_google_maps_closure() -> None:
    clock = ManualClock(CLOCK_START)
    closed_on_maps = maps_place(
        GARDEN, "Synthetic Quiet Garden", GARDEN_LAT, GARDEN_LNG, closed_claim()
    )
    replies = [
        *garden_basics("r"),
        walking_routes("w"),
        reply(check("k", walking_garden_candidate("cand-walk", []))),
        reply(submit("s", "Open per the official site.", "cand-walk", ["cand-walk"])),
    ]
    h = harness_with(
        clock,
        ScriptedModel(replies, clock, 0),
        ScriptedWebResearch(PAGES, clock, 0),
        ScriptedMaps({Q_GARDEN: closed_on_maps}, ROUTES),
    )

    outcome = run(h, continuation([], simple_request()), USER_TEXT)

    assert len(options_of(outcome.result).options) == 1


def test_google_maps_hours_are_the_fallback_without_an_official_source() -> None:
    replies = [
        reply(find("f", Q_GARDEN), search("s", Q_GARDEN)),
        reply(
            record(
                "d",
                sourced_record(
                    "ev-garden-duration",
                    "web-2",
                    "other_web",
                    "visit_duration",
                    duration_claim(60),
                    GARDEN,
                ),
            )
        ),
        walking_routes("w"),
        reply(check("k", walking_garden_candidate("cand-walk", []))),
        reply(submit("s2", "Open per Google Maps.", "cand-walk", ["cand-walk"])),
    ]

    h, outcome = run_simple(replies)

    found = result_for(h.model, "f")
    assert isinstance(found, PlaceObservation)
    assert len(found.evidence) == 1
    maps_item = found.evidence[0]
    assert isinstance(maps_item, SourcedFinding)
    assert maps_item.evidence_id == "maps-1"
    assert maps_item.source.kind == SourceKind.GOOGLE_MAPS
    assert maps_item.source.url == GARDEN_ON_MAPS.maps_url
    assert maps_item.retrieved_at == GARDEN_ON_MAPS.retrieved_at
    assert maps_item.scope.place_id == GARDEN
    assert len(options_of(outcome.result).options) == 1


def test_taxi_fare_is_calculated_from_distance_and_a_current_rate() -> None:
    taxi_plan = candidate_doc(
        "cand-taxi",
        [garden_stop(60, [])],
        [WALK_START_TO_GARDEN, TAXI_GARDEN_TO_DINNER],
        [cost_doc("Garden admission", "500", "JPY", ["ev-admission"])],
    )
    replies = [
        *garden_basics("r"),
        reply(
            route("in", START_POINT, place_point(GARDEN), "walking"),
            route("out", place_point(GARDEN), COMMITMENT_POINT, "taxi_rideshare"),
            search("rates", Q_TAXI),
        ),
        reply(
            record(
                "rate",
                sourced_record(
                    "ev-taxi-rate",
                    "web-3",
                    "official_website",
                    "cost",
                    taxi_rate_claim("500", "300"),
                    None,
                ),
            )
        ),
        reply(check("k", taxi_plan)),
        reply(submit("s", "Take a taxi to dinner.", "cand-taxi", ["cand-taxi"])),
    ]

    _, outcome = run_simple(replies)

    checked = options_of(outcome.result).options[0].itinerary
    assert checked.total_cost.amount == Decimal("2500")
    assert checked.total_includes_estimates is True


def test_transit_fare_comes_from_the_operators_official_site() -> None:
    _, outcome = run_garden(garden_plan("r"))

    fare = evidence_by_id(outcome, "ev-fare")
    assert isinstance(fare, SourcedFinding)
    assert fare.source.url == METRO_FARE.url
    assert fare.source.kind == SourceKind.OFFICIAL_WEBSITE
    checked = options_of(outcome.result).options[0].itinerary
    assert checked.candidate.legs[1].evidence_ids == ("ev-fare",)
    assert checked.total_cost.amount == Decimal("680")


def test_missing_price_reaches_the_options_as_an_ai_best_guess() -> None:
    guessed = cost_doc("Garden admission", "800", "JPY", ["ev-admission-guess"])
    replies = [
        reply(find("f", Q_GARDEN), search("s", Q_GARDEN)),
        reply(
            record(
                "d",
                sourced_record(
                    "ev-garden-duration",
                    "web-2",
                    "other_web",
                    "visit_duration",
                    duration_claim(60),
                    GARDEN,
                ),
            ),
            record(
                "g",
                estimate_record(
                    "ev-admission-guess",
                    "cost",
                    cost_claim("800", "JPY"),
                    "ai_best_guess",
                    GARDEN,
                    [],
                ),
            ),
        ),
        walking_routes("w"),
        reply(check("k", walking_garden_candidate("cand-walk", [guessed]))),
        reply(submit("s2", "Admission is an estimate.", "cand-walk", ["cand-walk"])),
    ]

    _, outcome = run_simple(replies)

    checked = options_of(outcome.result).options[0].itinerary
    assert checked.total_cost.amount == Decimal("800")
    assert checked.total_includes_estimates is True
    cited = checked.candidate.costs[0].evidence_ids
    labels = [evidence_by_id(outcome, evidence_id) for evidence_id in cited]
    assert [item.label for item in labels if isinstance(item, Estimate)] == [
        EstimateLabel.AI_BEST_GUESS
    ]


def test_stop_found_through_web_research_gets_its_place_id_from_maps() -> None:
    trail_plan = candidate_doc(
        "cand-trail",
        [stop_doc(TRAIL, TRAIL_LAT, TRAIL_LNG, 45, [])],
        [leg_doc("walking", 30, 2500, [], []), leg_doc("walking", 40, 3000, [], [])],
        [],
    )
    replies = [
        reply(search("s", Q_TRAIL)),
        reply(find("f", "Synthetic Ridge Trail")),
        reply(
            record(
                "d",
                sourced_record(
                    "ev-trail-duration",
                    "web-1",
                    "other_web",
                    "visit_duration",
                    duration_claim(45),
                    TRAIL,
                ),
            ),
            route("in", START_POINT, place_point(TRAIL), "walking"),
            route("out", place_point(TRAIL), COMMITMENT_POINT, "walking"),
        ),
        reply(check("k", trail_plan)),
        reply(submit("s2", "A quiet trail walk.", "cand-trail", ["cand-trail"])),
    ]

    h, outcome = run_simple(replies)

    assert h.maps.place_queries == ["Synthetic Ridge Trail"]
    stops = options_of(outcome.result).options[0].itinerary.candidate.stops
    assert [stop.place_id for stop in stops] == [TRAIL]


def test_rain_forecast_adds_a_warning_without_rejecting_the_plan() -> None:
    replies = [
        *garden_research("r"),
        reply(weather("wx", place_point(GARDEN))),
        reply(check("k", garden_candidate("cand-garden", 60))),
        reply(submit("s", "Bring an umbrella.", "cand-garden", ["cand-garden"])),
    ]

    h, outcome = run_garden(replies)

    forecast = result_for(h.model, "wx")
    assert isinstance(forecast, WeatherObservation)
    assert [item.evidence_id for item in forecast.evidence] == ["weather-1"]
    item = forecast.evidence[0]
    assert isinstance(item, SourcedFinding)
    assert str(item.source.url) == FORECAST_URL
    assert item.scope.place_id == GARDEN
    assert options_of(outcome.result).options[0].itinerary.weather_warnings != ()


# --- D4: stops with no Place ID use a fallback key ----------------------------------------


def spring_key(result_id: str) -> Doc:
    return {"result_id": result_id, "latitude": SPRING_LAT, "longitude": SPRING_LNG}


def fallback_scope(key: Doc) -> Doc:
    return {
        "place_id": None,
        "fallback_key": key,
        "applies_from": at("00:00"),
        "applies_until": at("23:59"),
    }


def fallback_point(key: Doc) -> Doc:
    return {"kind": "fallback", "fallback_key": key}


def spring_candidate(candidate_id: str, key: Doc, visit_minutes: int) -> Doc:
    """The spring found on the web (no Place ID), walking both ways, nothing priced."""
    spring: Doc = {
        "place_id": None,
        "label": "Synthetic Hidden Spring",
        "latitude": SPRING_LAT,
        "longitude": SPRING_LNG,
        "fallback_key": key,
        "visit_minutes": visit_minutes,
        "evidence_ids": [],
    }
    return candidate_doc(
        candidate_id,
        [spring],
        [leg_doc("walking", 25, 2000, [], []), leg_doc("walking", 60, 4800, [], [])],
        [],
    )


def spring_research(prefix: str, query: str, source_kind: str) -> list[ModelReply]:
    """Maps has no match; the opening comes from the page (web-1) the spring was found on."""
    key = spring_key("web-1")
    hours = sourced_record(
        "ev-spring-hours",
        "web-1",
        source_kind,
        "opening_access",
        open_claim([("08:00", "18:00")]),
        None,
    )
    visit = estimate_record(
        "ev-spring-duration", "visit_duration", duration_claim(45), "ai_best_guess", None, []
    )
    return [
        reply(find(f"{prefix}-find", Q_MISSING_PLACE), search(f"{prefix}-search", query)),
        reply(
            record(f"{prefix}-hours", {**hours, "scope": fallback_scope(key)}),
            record(f"{prefix}-duration", {**visit, "scope": fallback_scope(key)}),
        ),
        reply(
            route(f"{prefix}-in", START_POINT, fallback_point(key), "walking"),
            route(f"{prefix}-out", fallback_point(key), COMMITMENT_POINT, "walking"),
        ),
    ]


def test_stop_without_a_place_id_is_planned_with_a_fallback_key() -> None:
    replies = [
        *spring_research("r", Q_SPRING, "official_website"),
        reply(check("k", spring_candidate("cand-spring", spring_key("web-1"), 45))),
        reply(submit("s", "A hidden spring.", "cand-spring", ["cand-spring"])),
    ]

    h, outcome = run_simple(replies)

    assert error_for(h.model, "r-find").code == ToolErrorCode.PLACE_NOT_FOUND
    routed = result_for(h.model, "r-in")
    assert isinstance(routed, RouteObservation)
    assert (routed.destination.latitude, routed.destination.longitude) == (SPRING_LAT, SPRING_LNG)
    stops = options_of(outcome.result).options[0].itinerary.candidate.stops
    assert [(stop.place_id, stop.fallback_key) for stop in stops] == [
        (None, candidate(spring_candidate("x", spring_key("web-1"), 45)).stops[0].fallback_key)
    ]


def test_fallback_stop_with_only_non_official_opening_evidence_is_rejected() -> None:
    replies = [
        *spring_research("r", Q_SPRING_BLOG, "other_web"),
        reply(check("k", spring_candidate("cand-spring", spring_key("web-1"), 45))),
        reply(no_fit("n")),
    ]

    h, outcome = run_simple(replies)

    assert (ConstraintId.OPENING_ACCESS, "unverified") in reasons(rejection_for(h.model, "k"))
    assert isinstance(outcome.result, NoFeasibleResult)


def test_fallback_stop_must_come_from_a_result_in_this_request() -> None:
    replies = [
        *spring_research("r", Q_SPRING, "official_website"),
        reply(check("k", spring_candidate("cand-spring", spring_key("web-7"), 45))),
        reply(no_fit("n")),
    ]

    h, _ = run_simple(replies)

    feedback = rejection_for(h.model, "k")
    assert any(
        f.constraint == ConstraintId.MISSING_EVIDENCE
        and f.inputs.get("reason") == "unknown_result_id"
        and f.inputs.get("stop_index") == "0"
        for f in feedback.failures
    ), feedback.failures


def test_route_to_a_fallback_point_needs_a_result_from_this_request() -> None:
    h = harness(
        [
            reply(route("q", START_POINT, fallback_point(spring_key("web-1")), "walking")),
            reply(no_fit("n")),
        ]
    )

    run(h, continuation([], simple_request()), USER_TEXT)

    assert error_for(h.model, "q").code == ToolErrorCode.UNKNOWN_RESULT_ID
    assert h.maps.route_requests == []


def test_evidence_scoped_to_an_unknown_fallback_result_is_rejected() -> None:
    guess = estimate_record(
        "ev-spring-duration", "visit_duration", duration_claim(45), "ai_best_guess", None, []
    )
    h = harness(
        [
            reply(record("r", {**guess, "scope": fallback_scope(spring_key("web-1"))})),
            reply(no_fit("n")),
        ]
    )

    outcome = run(h, continuation([], simple_request()), USER_TEXT)

    assert error_for(h.model, "r").code == ToolErrorCode.UNKNOWN_RESULT_ID
    assert outcome.evidence == ()


def test_plans_with_the_same_fallback_stop_are_one_option() -> None:
    replies = [
        *spring_research("r", Q_SPRING, "official_website"),
        reply(
            check("k1", spring_candidate("cand-spring", spring_key("web-1"), 45)),
            check("k2", spring_candidate("cand-spring-long", spring_key("web-1"), 60)),
        ),
        reply(submit("s", "Done.", "cand-spring", ["cand-spring", "cand-spring-long"])),
    ]

    h, outcome = run_simple(replies)

    assert isinstance(result_for(h.model, "k2"), CandidatePassed)
    ids = [o.itinerary.candidate.candidate_id for o in options_of(outcome.result).options]
    assert ids == ["cand-spring"]


# --- AC10: status updates -------------------------------------------------------------------


def test_statuses_change_only_when_the_activity_changes() -> None:
    h, outcome = run_garden(garden_plan("r"))

    assert outcome.statuses == (RESEARCHING, PLANNING, GENERATING)
    assert tuple(h.sink.statuses) == outcome.statuses


def test_status_returns_from_planning_to_researching() -> None:
    replies = [
        *garden_research("r"),
        reply(check("long", garden_candidate("cand-garden-long", 200))),
        reply(search("again", Q_GARDEN)),
        reply(check("short", garden_candidate("cand-garden-short", 60))),
        reply(submit("s", "A shorter visit fits.", "cand-garden-short", ["cand-garden-short"])),
    ]

    h, outcome = run_garden(replies)

    assert outcome.statuses == (RESEARCHING, PLANNING, RESEARCHING, PLANNING, GENERATING)
    assert tuple(h.sink.statuses) == outcome.statuses


def test_clarification_turn_never_reaches_generating_itinerary() -> None:
    h = harness(
        [
            reply(weather("wx", START_POINT)),
            reply(clarify("c1", "Which currency?", [CURRENCY_QUESTION])),
        ]
    )

    outcome = run(h, continuation([], unresolved_currency_request()), USER_TEXT)

    assert isinstance(outcome.result, Clarification)
    assert outcome.statuses == (RESEARCHING,)
    assert h.sink.statuses == [RESEARCHING]


def test_gate_rejection_moves_from_generating_back_to_planning() -> None:
    replies = [
        *garden_research("r"),
        reply(check("k1", garden_candidate("cand-garden", 60))),
        reply(check("k2", garden_candidate("cand-garden", 200))),
        reply(submit("s1", "Done.", "cand-garden", ["cand-garden"])),
        reply(check("k3", garden_candidate("cand-garden-short", 60))),
        reply(submit("s2", "Done.", "cand-garden-short", ["cand-garden-short"])),
    ]

    h, outcome = run_garden(replies)

    assert error_for(h.model, "s1").code == ToolErrorCode.GATE_REJECTED
    assert outcome.statuses == (RESEARCHING, PLANNING, GENERATING, PLANNING, GENERATING)
    assert isinstance(outcome.result, ValidatedOptions)


# --- AC11: execution limits with a fixed clock --------------------------------------------------


def test_model_call_limit_ends_the_turn() -> None:
    h = harness([reply(search("a", Q_GARDEN)), reply(search("b", Q_FARE)), reply(no_fit("n"))])

    outcome = run_turn(
        continuation([], simple_request()),
        USER_TEXT,
        USER_TZ,
        None,
        h.dependencies,
        limits(2, 80, 10, 60, 600.0),
    )

    assert failure_of(outcome.result).reason == ExecutionFailureReason.EXECUTION_LIMIT_REACHED
    assert len(h.model.received) == 2
    assert outcome.model_calls == 2


def test_tool_call_limit_ends_the_turn_before_the_excess_call() -> None:
    h = harness([reply(search("a", Q_GARDEN), search("b", Q_FARE), find("c", Q_GARDEN))])

    outcome = run_turn(
        continuation([], simple_request()),
        USER_TEXT,
        USER_TZ,
        None,
        h.dependencies,
        limits(40, 2, 10, 60, 600.0),
    )

    assert failure_of(outcome.result).reason == ExecutionFailureReason.EXECUTION_LIMIT_REACHED
    assert h.maps.place_queries == []
    assert [entry.call_id for entry in outcome.tool_calls] == ["a", "b"]


def test_candidate_limit_ends_the_turn_before_the_excess_check() -> None:
    replies = [
        *garden_research("r"),
        reply(check("k1", garden_candidate("cand-a", 60))),
        reply(check("k2", garden_candidate("cand-b", 70))),
    ]
    h = harness(replies)

    outcome = run_turn(
        continuation([], garden_request()),
        USER_TEXT,
        USER_TZ,
        None,
        h.dependencies,
        limits(40, 80, 1, 60, 600.0),
    )

    assert failure_of(outcome.result).reason == ExecutionFailureReason.EXECUTION_LIMIT_REACHED
    assert [r.call_id for r in outcome.tool_calls if r.tool_name == "check_candidate"] == ["k1"]


def test_evidence_limit_ends_the_turn() -> None:
    first = estimate_record(
        "ev-a", "visit_duration", duration_claim(45), "ai_best_guess", GARDEN, []
    )
    second = estimate_record(
        "ev-b", "visit_duration", duration_claim(50), "ai_best_guess", CAFE, []
    )
    h = harness([reply(record("r1", first)), reply(record("r2", second)), reply(no_fit("n"))])

    outcome = run_turn(
        continuation([], simple_request()),
        USER_TEXT,
        USER_TZ,
        None,
        h.dependencies,
        limits(40, 80, 10, 1, 600.0),
    )

    assert failure_of(outcome.result).reason == ExecutionFailureReason.EXECUTION_LIMIT_REACHED
    assert [item.evidence_id for item in outcome.evidence] == ["ev-a"]


@pytest.mark.parametrize(
    ("search_seconds", "ends_on_limit"),
    [pytest.param(60.0, False, id="equal-continues"), pytest.param(60.5, True, id="over-ends")],
)
def test_elapsed_time_limit_is_checked_before_the_next_model_call(
    search_seconds: float, ends_on_limit: bool
) -> None:
    clock = ManualClock(CLOCK_START)
    model = ScriptedModel([reply(search("a", Q_GARDEN)), reply(no_fit("n"))], clock, 0)
    h = harness_with(
        clock,
        model,
        ScriptedWebResearch(PAGES, clock, search_seconds),
        ScriptedMaps(PLACES, ROUTES),
    )

    outcome = run_turn(
        continuation([], simple_request()),
        USER_TEXT,
        USER_TZ,
        None,
        h.dependencies,
        limits(40, 80, 10, 60, 60.0),
    )

    if ends_on_limit:
        assert failure_of(outcome.result).reason == ExecutionFailureReason.EXECUTION_LIMIT_REACHED
        assert len(model.received) == 1
    else:
        assert isinstance(outcome.result, NoFeasibleResult)


def test_elapsed_time_limit_is_checked_before_each_tool_call() -> None:
    clock = ManualClock(CLOCK_START)
    model = ScriptedModel([reply(search("a", Q_GARDEN), search("b", Q_FARE))], clock, 0)
    h = harness_with(
        clock, model, ScriptedWebResearch(PAGES, clock, 61), ScriptedMaps(PLACES, ROUTES)
    )

    outcome = run_turn(
        continuation([], simple_request()),
        USER_TEXT,
        USER_TZ,
        None,
        h.dependencies,
        limits(40, 80, 10, 60, 60.0),
    )

    assert failure_of(outcome.result).reason == ExecutionFailureReason.EXECUTION_LIMIT_REACHED
    assert h.web.queries == [Q_GARDEN]


VALID_LIMITS: Doc = {
    "max_model_calls": 5,
    "max_tool_calls": 10,
    "max_candidates": 3,
    "max_evidence_items": 20,
    "max_elapsed_seconds": 30.0,
}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("max_model_calls", 0, id="no-model-calls"),
        pytest.param("max_tool_calls", -1, id="negative-tool-calls"),
        pytest.param("max_candidates", 0, id="no-candidates"),
        pytest.param("max_candidates", 21, id="more-candidates-than-selection-allows"),
        pytest.param("max_evidence_items", 0, id="no-evidence"),
        pytest.param("max_elapsed_seconds", 0, id="no-time"),
        pytest.param("max_elapsed_seconds", "30", id="text-seconds"),
        pytest.param("max_model_calls", None, id="null"),
    ],
)
def test_execution_limits_reject_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        ExecutionLimits.model_validate_json(json.dumps({**VALID_LIMITS, field: value}))


@pytest.mark.parametrize("field", sorted(VALID_LIMITS))
def test_execution_limits_have_no_defaults(field: str) -> None:
    document = {key: value for key, value in VALID_LIMITS.items() if key != field}
    with pytest.raises(ValidationError):
        ExecutionLimits.model_validate_json(json.dumps(document))


def test_execution_limits_accept_explicit_valid_values() -> None:
    parsed = ExecutionLimits.model_validate_json(json.dumps(VALID_LIMITS))

    assert (parsed.max_candidates, parsed.max_elapsed_seconds) == (3, 30.0)


# --- AC12: final values are Python's ----------------------------------------------------------


def test_explanatory_text_cannot_change_the_checked_values() -> None:
    replies = [
        *garden_research("r"),
        reply(check("k", garden_candidate("cand-garden", 60))),
        reply(
            submit(
                "s",
                "Total JPY 100, arriving at 16:00, with a free museum stop on the way.",
                "cand-garden",
                ["cand-garden"],
            )
        ),
    ]

    h, outcome = run_garden(replies)

    passed = result_for(h.model, "k")
    assert isinstance(passed, CandidatePassed)
    options = options_of(outcome.result).options
    assert [option.itinerary for option in options] == [passed.checked]
    assert options[0].itinerary.total_cost.amount == Decimal("680")
    assert len(options[0].itinerary.candidate.stops) == 1


def test_only_submitted_candidates_become_options() -> None:
    h = harness(
        [
            *two_walking_plans("p"),
            reply(submit("s", "Done.", "cand-garden-walk", ["cand-garden-walk"])),
        ]
    )

    outcome = run(h, continuation([], NO_BUDGET), USER_TEXT)

    assert isinstance(result_for(h.model, "p-cafe"), CandidatePassed)
    ids = [o.itinerary.candidate.candidate_id for o in options_of(outcome.result).options]
    assert ids == ["cand-garden-walk"]


def test_gate_rechecks_the_latest_proposal_for_a_candidate_id() -> None:
    replies = [
        *garden_research("r"),
        reply(check("k1", garden_candidate("cand-garden", 60))),
        reply(check("k2", garden_candidate("cand-garden", 200))),
        reply(submit("s", "Done.", "cand-garden", ["cand-garden"])),
        reply(no_fit("n")),
    ]

    h, outcome = run_garden(replies)

    gate = error_for(h.model, "s")
    assert gate.code == ToolErrorCode.GATE_REJECTED
    assert (ConstraintId.ARRIVAL_BUFFER, "") in reasons(gate.feedback[0])
    assert isinstance(outcome.result, NoFeasibleResult)


# --- tracing fields, redundancy, continuation bounds ---------------------------------------------


def test_tool_calls_are_logged_with_tracing_fields() -> None:
    clock = ManualClock(CLOCK_START)
    finish = no_fit("t3")
    replies = [
        reply(raw_call("t1", "book_taxi", "{}")),
        reply(search("t2", Q_GARDEN)),
        reply(finish),
    ]
    model = ScriptedModel(replies, clock, 0)
    h = harness_with(
        clock, model, ScriptedWebResearch(PAGES, clock, 2.5), ScriptedMaps(PLACES, ROUTES)
    )

    outcome = run(h, continuation([], simple_request()), USER_TEXT)

    logged = [
        (r.position, r.call_id, r.tool_name, r.status, r.latency_seconds, r.arguments)
        for r in outcome.tool_calls
    ]
    assert logged == [
        (1, "t1", "book_taxi", ToolCallStatus.MALFORMED, 0.0, "{}"),
        (2, "t2", "web_search", ToolCallStatus.SUCCESS, 2.5, json.dumps({"query": Q_GARDEN})),
        (3, "t3", "report_no_feasible_result", ToolCallStatus.SUCCESS, 0.0, finish.arguments),
    ]


def test_redundant_research_is_counted_and_still_executed() -> None:
    h = harness(
        [
            reply(
                search("a", Q_GARDEN),
                raw_call("b", "web_search", '{ "query":"Synthetic Quiet Garden" }'),
                find("c", Q_GARDEN),
                find("d", Q_GARDEN),
                search("e", Q_FARE),
            ),
            reply(no_fit("n")),
        ]
    )

    outcome = run(h, continuation([], simple_request()), USER_TEXT)

    assert outcome.redundant_research_calls == 2
    assert h.web.queries == [Q_GARDEN, Q_GARDEN, Q_FARE]
    assert h.maps.place_queries == [Q_GARDEN, Q_GARDEN]
    assert all(r.status == ToolCallStatus.SUCCESS for r in outcome.tool_calls)


@pytest.mark.parametrize(
    ("prior_count", "fits"),
    [
        pytest.param(MAX_CONTINUATION_MESSAGES - 2, True, id="exactly-fills"),
        pytest.param(MAX_CONTINUATION_MESSAGES - 1, False, id="one-over"),
    ],
)
def test_continuation_message_count_is_never_trimmed(prior_count: int, fits: bool) -> None:
    roles = ("user", "assistant")
    prior = [(roles[i % 2], f"Earlier message {i}.") for i in range(prior_count)]
    start = continuation(prior, simple_request())
    h = harness([reply(no_fit("n"))])

    if fits:
        outcome = run(h, start, "One more try.")
        assert outcome.continuation.messages[:prior_count] == start.messages
        assert len(outcome.continuation.messages) == MAX_CONTINUATION_MESSAGES
    else:
        with pytest.raises(ContinuationOverflowError):
            run(h, start, "One more try.")
        assert h.model.received == []


@pytest.mark.parametrize(
    ("user_chars", "fits"),
    [pytest.param(10, True, id="exactly-fills"), pytest.param(11, False, id="one-over")],
)
def test_continuation_text_reserves_room_for_the_reply(user_chars: int, fits: bool) -> None:
    # 29,990 prior characters + the user message + a reply of up to MAX_TEXT_CHARS
    # must fit in MAX_CONTINUATION_TEXT_CHARS (32,000).
    assert MAX_TEXT_CHARS == 2000
    prior = [("user", "x" * 2000)] * 14 + [("assistant", "y" * 1990)]
    start = continuation(prior, simple_request())
    h = harness([reply(no_fit("n"))])

    if fits:
        outcome = run(h, start, "z" * user_chars)
        assert outcome.continuation.messages[:15] == start.messages
    else:
        with pytest.raises(ContinuationOverflowError):
            run(h, start, "z" * user_chars)
        assert h.model.received == []


@pytest.mark.parametrize(
    "message", [pytest.param("", id="empty"), pytest.param("x" * 2001, id="long")]
)
def test_invalid_user_message_is_rejected_before_any_model_call(message: str) -> None:
    h = harness([reply(no_fit("n"))])

    with pytest.raises(ValidationError):
        run(h, continuation([], simple_request()), message)
    assert h.model.received == []


# --- R-A: the request must be for today, after the turn starts -----------------------------


def harness_at(now: datetime, replies: Sequence[ModelReply]) -> Harness:
    clock = ManualClock(now)
    return harness_with(
        clock,
        ScriptedModel(replies, clock, 0),
        ScriptedWebResearch(PAGES, clock, 0),
        ScriptedMaps(PLACES, ROUTES),
    )


def same_day_failures(feedback: ValidationFeedback) -> dict[str, dict[str, str]]:
    """SAME_DAY_GAP failures that carry a reason, keyed by reason."""
    return {
        failure.inputs["reason"]: dict(failure.inputs)
        for failure in feedback.failures
        if failure.constraint == ConstraintId.SAME_DAY_GAP and "reason" in failure.inputs
    }


def dated_request(gap_start: str, commitment_time: str) -> Doc:
    document = replaced(garden_request(), ("gap", "start_time"), gap_start)
    return cast(Doc, replaced(document, ("commitment", "time"), commitment_time))


def check_once(now: datetime, request_document: Doc) -> ValidationFeedback:
    """Check the garden plan once, with no research, at turn start `now`."""
    h = harness_at(
        now, [reply(check("k", garden_candidate("cand-garden", 60))), reply(no_fit("n"))]
    )
    run(h, continuation([], request_document), USER_TEXT)
    return rejection_for(h.model, "k")


def test_commitment_tomorrow_is_repair_feedback() -> None:
    tomorrow = dated_request("2026-11-21T15:00:00+09:00", "2026-11-21T19:00:00+09:00")

    found = same_day_failures(check_once(tokyo(14, 0), tomorrow))

    assert set(found) == {"commitment_not_today"}
    assert found["commitment_not_today"]["commitment_time"] == "2026-11-21T19:00:00+09:00"
    assert found["commitment_not_today"]["now"] == "2026-11-20T14:00:00+09:00"


def test_commitment_already_past_today_is_repair_feedback() -> None:
    found = same_day_failures(check_once(tokyo(19, 30), garden_request()))

    assert set(found) == {"commitment_passed", "gap_start_passed"}
    assert found["commitment_passed"]["commitment_time"] == "2026-11-20T19:00:00+09:00"
    assert found["commitment_passed"]["now"] == "2026-11-20T19:30:00+09:00"


def test_gap_start_in_the_past_is_repair_feedback() -> None:
    found = same_day_failures(check_once(tokyo(15, 30), garden_request()))

    assert set(found) == {"gap_start_passed"}
    assert found["gap_start_passed"]["gap_start"] == "2026-11-20T15:00:00+09:00"
    assert found["gap_start_passed"]["now"] == "2026-11-20T15:30:00+09:00"


def test_gap_starting_exactly_at_turn_start_passes() -> None:
    h = harness_at(tokyo(15, 0), garden_plan("r"))

    outcome = run(h, continuation([], garden_request()), USER_TEXT)

    assert len(options_of(outcome.result).options) == 1


def test_commitment_after_midnight_in_its_timezone_is_not_today() -> None:
    # 14:30 UTC is 23:30 in Tokyo; 00:30 Tokyo is the next local day but the same UTC day.
    late_evening = dated_request("2026-11-20T23:40:00+09:00", "2026-11-21T00:30:00+09:00")

    found = same_day_failures(check_once(datetime(2026, 11, 20, 14, 30, tzinfo=UTC), late_evening))

    assert set(found) == {"commitment_not_today"}
    assert found["commitment_not_today"]["now"] == "2026-11-20T23:30:00+09:00"


def test_same_local_day_counts_even_when_the_utc_day_differs() -> None:
    # 23:00 UTC on the 19th is 08:00 on the 20th in Tokyo, the commitment's local day.
    morning = dated_request("2026-11-20T08:30:00+09:00", "2026-11-20T10:00:00+09:00")

    feedback = check_once(datetime(2026, 11, 19, 23, 0, tzinfo=UTC), morning)

    assert [f for f in feedback.failures if f.constraint == ConstraintId.SAME_DAY_GAP] == []


def test_final_gate_applies_the_same_day_check() -> None:
    tomorrow = dated_request("2026-11-21T15:00:00+09:00", "2026-11-21T19:00:00+09:00")
    replies = [
        *garden_research("r"),
        reply(check("k", garden_candidate("cand-garden", 60))),
        reply(update("u", tomorrow)),
        reply(submit("s", "Done.", "cand-garden", ["cand-garden"])),
        reply(no_fit("n")),
    ]

    h, outcome = run_garden(replies)

    assert isinstance(result_for(h.model, "k"), CandidatePassed)
    gate = error_for(h.model, "s")
    assert gate.code == ToolErrorCode.GATE_REJECTED
    assert "commitment_not_today" in same_day_failures(gate.feedback[0])
    assert isinstance(outcome.result, NoFeasibleResult)


def test_past_gap_start_is_repaired_with_update_request_in_the_same_run() -> None:
    later_start = cast(Doc, replaced(garden_request(), ("gap", "start_time"), at("15:45")))
    replies = [
        *garden_research("r"),
        reply(check("k1", garden_candidate("cand-garden", 60))),
        reply(update("u", later_start)),
        reply(check("k2", garden_candidate("cand-garden-later", 60))),
        reply(submit("s", "Starting at 15:45.", "cand-garden-later", ["cand-garden-later"])),
    ]
    h = harness_at(tokyo(15, 30), replies)

    outcome = run(h, continuation([], garden_request()), USER_TEXT)

    assert set(same_day_failures(rejection_for(h.model, "k1"))) == {"gap_start_passed"}
    assert isinstance(result_for(h.model, "k2"), CandidatePassed)
    ids = [o.itinerary.candidate.candidate_id for o in options_of(outcome.result).options]
    assert ids == ["cand-garden-later"]


# --- D6 / O9: get_datetime gives the turn-start time in the user's timezone -----------------


def now_call(call_id: str) -> ToolCall:
    return call(call_id, "get_datetime", {})


def datetime_for(model: ScriptedModel, call_id: str) -> DatetimeObservation:
    result = result_for(model, call_id)
    assert isinstance(result, DatetimeObservation), result
    return result


def iso_now(observation: DatetimeObservation) -> str:
    """The `now` field exactly as it is serialised for the model."""
    return cast(dict[str, str], json.loads(observation.model_dump_json()))["now"]


def run_in_zone(now: datetime, user_timezone: str, replies: Sequence[ModelReply]) -> Harness:
    h = harness_at(now, replies)
    run_turn(
        continuation([], simple_request()), USER_TEXT, user_timezone, None, h.dependencies, GENEROUS
    )
    return h


def test_get_datetime_returns_the_turn_start_in_the_user_zone_even_as_the_clock_moves() -> None:
    clock = ManualClock(CLOCK_START)
    replies = [reply(now_call("t1")), reply(now_call("t2")), reply(no_fit("n"))]
    h = harness_with(
        clock,
        ScriptedModel(replies, clock, 30),
        ScriptedWebResearch(PAGES, clock, 0),
        ScriptedMaps(PLACES, ROUTES),
    )

    run_turn(
        continuation([], simple_request()),
        USER_TEXT,
        "Europe/London",
        None,
        h.dependencies,
        GENEROUS,
    )

    for call_id in ("t1", "t2"):
        observed = datetime_for(h.model, call_id)
        assert iso_now(observed) == "2026-11-20T05:00:00+00:00"
        assert observed.now == CLOCK_START
        assert observed.timezone == "Europe/London"


@pytest.mark.parametrize(
    ("now", "zone", "expected"),
    [
        pytest.param(
            datetime(2026, 3, 8, 6, 30, tzinfo=UTC),
            "America/New_York",
            "2026-03-08T01:30:00-05:00",
            id="new-york-before-spring-forward",
        ),
        pytest.param(
            datetime(2026, 3, 8, 7, 30, tzinfo=UTC),
            "America/New_York",
            "2026-03-08T03:30:00-04:00",
            id="new-york-after-spring-forward",
        ),
        pytest.param(
            datetime(2026, 11, 1, 5, 30, tzinfo=UTC),
            "America/New_York",
            "2026-11-01T01:30:00-04:00",
            id="new-york-first-1-30-before-fall-back",
        ),
        pytest.param(
            datetime(2026, 11, 1, 6, 30, tzinfo=UTC),
            "America/New_York",
            "2026-11-01T01:30:00-05:00",
            id="new-york-second-1-30-after-fall-back",
        ),
        pytest.param(
            datetime(2026, 3, 8, 6, 30, tzinfo=UTC),
            "Asia/Tokyo",
            "2026-03-08T15:30:00+09:00",
            id="tokyo-march",
        ),
        pytest.param(
            datetime(2026, 11, 1, 6, 30, tzinfo=UTC),
            "Asia/Tokyo",
            "2026-11-01T15:30:00+09:00",
            id="tokyo-november",
        ),
    ],
)
def test_get_datetime_applies_the_zone_rules_for_the_date(
    now: datetime, zone: str, expected: str
) -> None:
    h = run_in_zone(now, zone, [reply(now_call("t")), reply(no_fit("n"))])

    observed = datetime_for(h.model, "t")
    assert iso_now(observed) == expected
    assert observed.timezone == zone


def test_same_day_check_stays_in_the_commitment_zone_when_the_user_zone_differs() -> None:
    # 14:30 UTC: 23:30 on the 20th in Tokyo (commitment), 09:30 on the 20th in New York (user).
    late_evening = dated_request("2026-11-20T23:40:00+09:00", "2026-11-21T00:30:00+09:00")
    replies = [
        reply(now_call("t")),
        reply(check("k", garden_candidate("cand-garden", 60))),
        reply(no_fit("n")),
    ]
    h = harness_at(datetime(2026, 11, 20, 14, 30, tzinfo=UTC), replies)

    run_turn(
        continuation([], late_evening),
        USER_TEXT,
        "America/New_York",
        None,
        h.dependencies,
        GENEROUS,
    )

    assert iso_now(datetime_for(h.model, "t")) == "2026-11-20T09:30:00-05:00"
    found = same_day_failures(rejection_for(h.model, "k"))
    assert set(found) == {"commitment_not_today"}
    assert found["commitment_not_today"]["now"] == "2026-11-20T23:30:00+09:00"


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param('{"timezone": "UTC"}', id="extra-key"),
        pytest.param("[]", id="not-an-object"),
        pytest.param("", id="empty-string"),
    ],
)
def test_get_datetime_with_arguments_is_not_executed(arguments: str) -> None:
    h = harness(
        [
            reply(raw_call("bad", "get_datetime", arguments)),
            reply(now_call("ok")),
            reply(no_fit("n")),
        ]
    )

    outcome = run(h, continuation([], simple_request()), USER_TEXT)

    error = error_for(h.model, "bad")
    assert error.code in {ToolErrorCode.INVALID_ARGUMENTS, ToolErrorCode.UNPARSEABLE_ARGUMENTS}
    if arguments != "":
        assert error.code == ToolErrorCode.INVALID_ARGUMENTS
    assert outcome.tool_calls[0].status == ToolCallStatus.MALFORMED
    assert isinstance(result_for(h.model, "ok"), DatetimeObservation)


def test_get_datetime_counts_toward_the_tool_budget() -> None:
    h = harness([reply(now_call("t"), search("s", Q_GARDEN))])

    outcome = run_turn(
        continuation([], simple_request()),
        USER_TEXT,
        USER_TZ,
        None,
        h.dependencies,
        limits(40, 1, 10, 60, 600.0),
    )

    assert failure_of(outcome.result).reason == ExecutionFailureReason.EXECUTION_LIMIT_REACHED
    assert h.web.queries == []
    assert [(entry.call_id, entry.tool_name) for entry in outcome.tool_calls] == [
        ("t", "get_datetime")
    ]


def test_get_datetime_emits_no_status_and_is_not_redundant_research() -> None:
    h = harness([reply(now_call("t1"), now_call("t2")), reply(no_fit("n"))])

    outcome = run(h, continuation([], simple_request()), USER_TEXT)

    assert outcome.statuses == ()
    assert h.sink.statuses == []
    assert outcome.redundant_research_calls == 0
    assert [(e.tool_name, e.status) for e in outcome.tool_calls][:2] == [
        ("get_datetime", ToolCallStatus.SUCCESS),
        ("get_datetime", ToolCallStatus.SUCCESS),
    ]


@pytest.mark.parametrize("zone", ["Mars/Base", "asia/tokyo", "+09:00", "UTC+9", ""])
def test_invalid_user_timezone_is_rejected_before_any_model_call(zone: str) -> None:
    h = harness([reply(no_fit("n"))])

    with pytest.raises(ValidationError):
        run_turn(
            continuation([], simple_request()), USER_TEXT, zone, None, h.dependencies, GENEROUS
        )
    assert h.model.received == []


def test_model_uses_get_datetime_to_fix_a_request_within_the_run() -> None:
    tomorrow = dated_request("2026-11-21T15:00:00+09:00", "2026-11-21T19:00:00+09:00")
    replies = [
        *garden_research("r"),
        reply(check("k1", garden_candidate("cand-garden", 60))),
        reply(now_call("t")),
        reply(update("u", garden_request())),
        reply(check("k2", garden_candidate("cand-garden-today", 60))),
        reply(submit("s", "Planned for today.", "cand-garden-today", ["cand-garden-today"])),
    ]
    h = harness(replies)

    outcome = run(h, continuation([], tomorrow), "Dinner in Ginza at 7.")

    assert "commitment_not_today" in same_day_failures(rejection_for(h.model, "k1"))
    assert iso_now(datetime_for(h.model, "t")).startswith("2026-11-20T14:00:00")
    assert isinstance(result_for(h.model, "k2"), CandidatePassed)
    ids = [o.itinerary.candidate.candidate_id for o in options_of(outcome.result).options]
    assert ids == ["cand-garden-today"]


def test_instructions_and_tool_description_point_the_model_to_get_datetime() -> None:
    h = harness([reply(no_fit("n"))])

    run(h, continuation([], simple_request()), USER_TEXT)

    messages, tools = h.model.received[0]
    instructions = [m.text for m in messages if isinstance(m, InstructionsMessage)]
    assert len(instructions) == 1 and "get_datetime" in instructions[0]
    spec = next(tool for tool in tools if tool.name == "get_datetime")
    assert spec.description
    assert spec.parameters.get("properties") == {}


# --- R-B: out-of-range datetimes are rejected at the boundary --------------------------------


def test_continuation_with_a_year_9999_commitment_is_rejected_at_the_boundary() -> None:
    far_future = replaced(garden_request(), ("commitment", "time"), "9999-11-20T19:00:00+09:00")

    with pytest.raises(ValidationError) as exc_info:
        continuation([], cast(Doc, far_future))

    assert any(detail["loc"][-1] == "time" for detail in exc_info.value.errors())


def test_update_request_with_a_year_9999_time_is_a_malformed_call() -> None:
    far_future = replaced(garden_request(), ("commitment", "time"), "9999-11-20T19:00:00+09:00")
    h = harness([reply(update("u", cast(Doc, far_future))), reply(no_fit("n"))])

    outcome = run(h, continuation([], garden_request()), USER_TEXT)

    assert error_for(h.model, "u").code == ToolErrorCode.INVALID_ARGUMENTS
    assert outcome.continuation.request == request(garden_request())


# --- untested paths from the T2 review ---------------------------------------------------------


def test_calls_after_a_successful_terminal_call_are_not_processed() -> None:
    h = harness([reply(no_fit("n"), search("s", Q_GARDEN))])

    outcome = run(h, continuation([], simple_request()), USER_TEXT)

    assert isinstance(outcome.result, NoFeasibleResult)
    assert h.web.queries == []
    assert [entry.call_id for entry in outcome.tool_calls] == ["n"]


def test_checking_an_incomplete_request_is_a_structured_error() -> None:
    h = harness([reply(check("k", garden_candidate("cand-garden", 60))), reply(no_fit("n"))])

    outcome = run(h, continuation([], EMPTY_REQUEST), USER_TEXT)

    assert error_for(h.model, "k").code == ToolErrorCode.REQUEST_INCOMPLETE
    assert outcome.validation_feedback == ()


def test_submitting_after_the_request_became_incomplete_is_a_structured_error() -> None:
    no_time = cast(Doc, replaced(garden_request(), ("commitment", "time"), None))
    replies = [
        *garden_research("r"),
        reply(check("k", garden_candidate("cand-garden", 60))),
        reply(update("u", no_time)),
        reply(submit("s", "Done.", "cand-garden", ["cand-garden"])),
        reply(no_fit("n")),
    ]

    h, outcome = run_garden(replies)

    assert error_for(h.model, "s").code == ToolErrorCode.REQUEST_INCOMPLETE
    assert isinstance(outcome.result, NoFeasibleResult)


@pytest.mark.parametrize(
    "research",
    [
        pytest.param(find("f", Q_GARDEN), id="maps"),
        pytest.param(weather("w", START_POINT), id="weather"),
    ],
)
def test_python_built_evidence_counts_toward_the_evidence_cap(research: ToolCall) -> None:
    first = estimate_record(
        "ev-a", "visit_duration", duration_claim(45), "ai_best_guess", GARDEN, []
    )
    h = harness([reply(record("r", first)), reply(research), reply(no_fit("n"))])

    outcome = run_turn(
        continuation([], simple_request()),
        USER_TEXT,
        USER_TZ,
        None,
        h.dependencies,
        limits(40, 80, 10, 1, 600.0),
    )

    assert failure_of(outcome.result).reason == ExecutionFailureReason.EXECUTION_LIMIT_REACHED
    assert [item.evidence_id for item in outcome.evidence] == ["ev-a"]


def test_weather_provider_failure_is_an_execution_failure() -> None:
    clock = ManualClock(CLOCK_START)
    dependencies = AgentDependencies(
        model=ScriptedModel([reply(weather("w", START_POINT)), reply(no_fit("n"))], clock, 0),
        web=ScriptedWebResearch(PAGES, clock, 0),
        maps=ScriptedMaps(PLACES, ROUTES),
        weather=FailingWeather(),
        clock=clock,
        status_sink=RecordingStatusSink(),
    )

    outcome = run_turn(
        continuation([], simple_request()), USER_TEXT, USER_TZ, None, dependencies, GENEROUS
    )

    assert failure_of(outcome.result).reason == ExecutionFailureReason.PROVIDER_FAILURE


def test_route_provider_failure_is_an_execution_failure() -> None:
    clock = ManualClock(CLOCK_START)
    model = ScriptedModel(
        [reply(route("q", START_POINT, COMMITMENT_POINT, "walking")), reply(no_fit("n"))], clock, 0
    )
    dependencies = AgentDependencies(
        model=model,
        web=ScriptedWebResearch(PAGES, clock, 0),
        maps=FailingMaps(),
        weather=ScriptedWeather(FORECASTS),
        clock=clock,
        status_sink=RecordingStatusSink(),
    )

    outcome = run_turn(
        continuation([], simple_request()), USER_TEXT, USER_TZ, None, dependencies, GENEROUS
    )

    assert failure_of(outcome.result).reason == ExecutionFailureReason.PROVIDER_FAILURE


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(("gap", "start_time"), id="gap-start"),
        pytest.param(("commitment", "time"), id="commitment"),
    ],
)
def test_weather_without_the_outing_window_is_a_structured_error(path: tuple[str, ...]) -> None:
    missing = cast(Doc, replaced(simple_request(), path, None))
    h = harness([reply(weather("w", START_POINT)), reply(no_fit("n"))])

    run(h, continuation([], missing), USER_TEXT)

    assert error_for(h.model, "w").code == ToolErrorCode.REQUEST_INCOMPLETE
    assert h.weather.requests == []


# --- AC13: conversation replay -----------------------------------------------------------------

REPLY_LIST = TypeAdapter(tuple[ModelReply, ...])


def test_conversation_case_fixture_is_valid_contract_data() -> None:
    case = cast(dict[str, object], json.loads(CONVERSATION_CASE.read_text(encoding="utf-8")))

    assert set(case) == {
        "clock_start",
        "user_timezone",
        "limits",
        "continuation",
        "web",
        "places",
        "routes",
        "weather",
        "turns",
    }
    ExecutionLimits.model_validate_json(json.dumps(case["limits"]))
    ContinuationPayload.model_validate_json(json.dumps(case["continuation"]))
    for search_entry in cast(list[dict[str, object]], case["web"]):
        TypeAdapter(tuple[WebPage, ...]).validate_json(json.dumps(search_entry["pages"]))
    for place_entry in cast(list[dict[str, object]], case["places"]):
        if place_entry["place"] is not None:
            MapsPlace.model_validate_json(json.dumps(place_entry["place"]))
    for route_entry in cast(list[dict[str, object]], case["routes"]):
        Location.model_validate_json(json.dumps(route_entry["origin"]))
        Location.model_validate_json(json.dumps(route_entry["destination"]))
        MapsRoute.model_validate_json(json.dumps(route_entry["route"]))
    for forecast_entry in cast(list[dict[str, object]], case["weather"]):
        Location.model_validate_json(json.dumps(forecast_entry["point"]))
        TypeAdapter(tuple[WeatherPeriod, ...]).validate_json(json.dumps(forecast_entry["periods"]))
    turns = cast(list[dict[str, object]], case["turns"])
    assert len(turns) == 3
    for turn in turns:
        assert set(turn) == {"user_message", "replies"}
        REPLY_LIST.validate_json(json.dumps(turn["replies"]))


def conversation_case() -> dict[str, object]:
    return cast(dict[str, object], json.loads(CONVERSATION_CASE.read_text(encoding="utf-8")))


def case_turns(case: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], case["turns"])


def with_turn_replies(case: dict[str, object], index: int, replies: list[object]) -> str:
    turns = [
        {**turn, "replies": replies} if i == index else turn
        for i, turn in enumerate(case_turns(case))
    ]
    return json.dumps({**case, "turns": turns})


def turn_replies(case: dict[str, object], index: int) -> list[object]:
    return cast(list[object], case_turns(case)[index]["replies"])


def malformed_json(case: dict[str, object]) -> str:
    return json.dumps(case)[:-10]


def unknown_top_level_key(case: dict[str, object]) -> str:
    return json.dumps({**case, "notes": "not part of either case schema"})


def empty_turns(case: dict[str, object]) -> str:
    return json.dumps({**case, "turns": []})


def unscripted_web_search(case: dict[str, object]) -> str:
    web = cast(list[dict[str, object]], case["web"])
    return json.dumps({**case, "web": [entry for entry in web if entry["query"] != Q_FARE]})


def too_few_model_replies(case: dict[str, object]) -> str:
    return with_turn_replies(case, 1, turn_replies(case, 1)[:-1])


def leftover_model_replies(case: dict[str, object]) -> str:
    extra = json.loads(reply(no_fit("extra")).model_dump_json())
    return with_turn_replies(case, 0, [*turn_replies(case, 0), extra])


def duplicate_scripted_query(case: dict[str, object]) -> str:
    web = cast(list[object], case["web"])
    return json.dumps({**case, "web": [*web, web[0]]})


@pytest.mark.parametrize(
    "broken",
    [
        pytest.param(malformed_json, id="malformed-json"),
        pytest.param(unknown_top_level_key, id="matches-neither-schema"),
        pytest.param(empty_turns, id="empty-turns"),
        pytest.param(unscripted_web_search, id="unscripted-web-search"),
        pytest.param(too_few_model_replies, id="too-few-replies"),
        pytest.param(leftover_model_replies, id="leftover-replies"),
        pytest.param(duplicate_scripted_query, id="duplicate-scripted-query"),
    ],
)
def test_unusable_conversation_case_fails_without_output(
    tmp_path: Path, broken: Callable[[dict[str, object]], str]
) -> None:
    case_path = tmp_path / "case.json"
    case_path.write_text(broken(conversation_case()), encoding="utf-8")
    output = tmp_path / "out" / "report.json"

    completed = subprocess.run(
        [sys.executable, str(REPLAY_SCRIPT), "--case", str(case_path), "--output", str(output)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 1, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr.strip()
    assert not output.exists()


def test_replay_writes_conversation_updates_feedback_statuses_and_options(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "phase-2-example.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPLAY_SCRIPT),
            "--case",
            str(CONVERSATION_CASE),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = cast(dict[str, object], json.loads(output.read_text(encoding="utf-8")))
    assert set(report) == {"turns"}
    turns = cast(list[dict[str, object]], report["turns"])
    assert [set(turn) for turn in turns] == [{"user_message", "outcome"}] * 3
    outcomes = [TurnOutcome.model_validate_json(json.dumps(turn["outcome"])) for turn in turns]
    clarified, planned, revised = outcomes

    first_call = clarified.tool_calls[0]
    assert (first_call.tool_name, first_call.status) == ("get_datetime", ToolCallStatus.SUCCESS)

    assert {q.topic for q in clarification_of(clarified.result).questions} == {
        UnresolvedTopic.BUDGET_CURRENCY,
        UnresolvedTopic.OTHER,
    }
    assert GENERATING not in clarified.statuses
    assert clarified.statuses != ()

    assert [f.candidate_id for f in planned.validation_feedback] == ["cand-garden-long"]
    assert (ConstraintId.ARRIVAL_BUFFER, "") in reasons(planned.validation_feedback[0])
    planned_ids = [o.itinerary.candidate.candidate_id for o in options_of(planned.result).options]
    assert planned_ids == ["cand-garden"]

    statuses = revised.statuses
    assert any(
        statuses[i : i + 3] == (PLANNING, RESEARCHING, PLANNING) for i in range(len(statuses))
    ), statuses
    stale = {pair for feedback in revised.validation_feedback for pair in reasons(feedback)}
    assert (ConstraintId.MISSING_EVIDENCE, "unknown_place") in stale
    assert (ConstraintId.MISSING_EVIDENCE, "unknown_evidence_id") in stale
    final = options_of(revised.result).options
    assert [o.itinerary.candidate.candidate_id for o in final] == ["cand-garden-revised"]
    assert final[0].itinerary.arrival_at_commitment <= tokyo(17, 15)

    before, after = planned.continuation.request, revised.continuation.request
    assert after.commitment.time == tokyo(17, 30)
    assert (after.gap, after.budget, after.constraints, after.transport_modes) == (
        before.gap,
        before.budget,
        before.constraints,
        before.transport_modes,
    )
    assert [len(o.continuation.messages) for o in outcomes] == [2, 4, 6]
    assert revised.continuation.messages[:4] == planned.continuation.messages


# --- Phase 3 O1: device location -------------------------------------------------------

DEVICE_LOCATION = Location(
    label="Device location", latitude=35.6595, longitude=139.7005, place_id=None
)


def opening_after(start: ContinuationPayload, device_location: Location | None) -> Harness:
    """Run one clarification turn from `start` and return its harness."""
    h = harness([reply(clarify("c", "Where should we start?", [("start_location", "Where?")]))])
    run_turn(start, USER_TEXT, USER_TZ, device_location, h.dependencies, GENEROUS)
    return h


def test_no_device_location_keeps_the_phase_2_opening_history() -> None:
    start = continuation([("user", "Hi"), ("assistant", "Hello.")], EMPTY_REQUEST)

    opening = opening_after(start, None).model.received[0][0]

    assert [m.kind for m in opening] == [
        "instructions",
        "conversation",
        "conversation",
        "conversation",
        "request",
    ]


def test_device_location_reaches_the_model_as_user_supplied_context_only() -> None:
    start = continuation([], EMPTY_REQUEST)

    h = opening_after(start, DEVICE_LOCATION)

    opening = h.model.received[0][0]
    assert [m.kind for m in opening] == [
        "instructions",
        "conversation",
        "request",
        "device_location",
    ]
    supplied = [m for m in opening if isinstance(m, DeviceLocationMessage)]
    assert [m.location for m in supplied] == [DEVICE_LOCATION]
    requests = [m.request for m in opening if isinstance(m, RequestMessage)]
    assert [r.gap.start_location for r in requests] == [None]


def test_device_location_is_not_applied_to_the_request_without_the_model() -> None:
    start = continuation([], EMPTY_REQUEST)
    h = harness([reply(clarify("c", "Where should we start?", [("start_location", "Where?")]))])

    outcome = run_turn(start, USER_TEXT, USER_TZ, DEVICE_LOCATION, h.dependencies, GENEROUS)

    assert outcome.continuation.request == start.request
