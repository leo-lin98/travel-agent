"""Phase 3 acceptance tests for the website chat API and the offline demo.

Expectations derive from the Phase 3 document (status updates, malformed client context,
transcription failure, multiple turns, fresh session, backend restart, timezone with and
without daylight saving time, bounded sizes, no provider keys in the browser), R2/R3/R9/R13,
docs/decisions.md (2026-11-26 D1-D5, 2026-11-23 get_datetime), ledger rulings O1-O4, and
artifacts/phase-3/interface-contract.md.

Scripted doubles dictate every model reply, so these tests check the API boundary and the
demo wiring, not real model or transcription quality. All data is synthetic and offline.
"""

import asyncio
import gc
import importlib.util
import json
import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from agent_doubles import (
    ManualClock,
    RecordingStatusSink,
    ScriptedMaps,
    ScriptedModel,
    ScriptedWeather,
    ScriptedWebResearch,
    route_key,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx2 import Response
from pydantic import AwareDatetime, BaseModel, ConfigDict, JsonValue, TypeAdapter
from starlette.types import Message, Scope

from travel_agent.agent import (
    AgentDependencies,
    AgentModel,
    ConversationMessage,
    DatetimeObservation,
    DeviceLocationMessage,
    ExecutionLimits,
    MapsPlace,
    MapsRoute,
    ModelMessage,
    ModelReply,
    ProviderError,
    RequestMessage,
    StatusSink,
    ToolCall,
    ToolCallReply,
    ToolResultMessage,
    ToolSpec,
    TurnOutcome,
    WeatherPeriod,
    WebPage,
    run_turn,
)
from travel_agent.api import ApiLimits, TranscriptionError, create_app
from travel_agent.contracts import (
    MAX_CONTINUATION_MESSAGES,
    MAX_MESSAGE_CHARS,
    ChatMessage,
    ChatRole,
    ContinuationPayload,
    EstimateLabel,
    ExecutionFailure,
    ExecutionFailureReason,
    Location,
    OptionLabel,
    TransportMode,
    TurnResult,
    TurnStatus,
    ValidatedOptions,
    stop_key,
)

type Doc = dict[str, object]
type Event = dict[str, JsonValue]

BACKEND_DIR = Path(__file__).resolve().parents[2]
CONVERSATION_CASE = BACKEND_DIR / "tests" / "fixtures" / "replan_conversation.json"
DEMO_SCRIPT = BACKEND_DIR / "scripts" / "demo_offline.py"

RESEARCHING = TurnStatus.RESEARCHING.value
PLANNING = TurnStatus.PLANNING.value
GENERATING = TurnStatus.GENERATING_ITINERARY.value

LEAK = "leak-marker-7f3a9c"
ENV_SECRET = "sk-test-canary-91d2e"
PROVIDER_KEY_VARIABLES = (
    "OPENAI_API_KEY",
    "GOOGLE_MAPS_API_KEY",
    "TAVILY_API_KEY",
    "ELEVENLABS_API_KEY",
    "LANGFUSE_SECRET_KEY",
)
STREAM_WAIT_SECONDS = 5.0
TOKYO_CLOCK = datetime(2026, 11, 20, 5, 0, tzinfo=UTC)
USER_TZ = "Asia/Tokyo"
AUDIO = b"\x1aE\xdf\xa3synthetic-webm-clip"
AUDIO_TYPE = "audio/webm;codecs=opus"

_EVENT: TypeAdapter[Event] = TypeAdapter(dict[str, JsonValue])
_TURN_RESULT: TypeAdapter[TurnResult] = TypeAdapter(TurnResult)
_LABELS: TypeAdapter[dict[str, EstimateLabel]] = TypeAdapter(dict[str, EstimateLabel])

LIMITS = ExecutionLimits.model_validate_json(
    json.dumps(
        {
            "max_model_calls": 40,
            "max_tool_calls": 80,
            "max_candidates": 10,
            "max_evidence_items": 60,
            "max_elapsed_seconds": 600.0,
        }
    )
)
API_LIMITS = ApiLimits.model_validate_json(
    json.dumps({"max_turn_body_bytes": 20000, "max_message_chars": 500, "max_audio_bytes": 1000})
)

DEVICE_LOCATION = Location(
    label="Current location", latitude=35.6595, longitude=139.7005, place_id=None
)
EMPTY_REQUEST: Doc = {
    "schema_version": 1,
    "gap": {"start_time": None, "start_location": None},
    "commitment": {"location": None, "time": None, "timezone": None},
    "budget": None,
    "transport_modes": [],
    "constraints": [],
    "unresolved_questions": [],
}


# --- the Phase 2 conversation fixture ---------------------------------------------------


class _Fixture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScriptedSearch(_Fixture):
    query: str
    pages: tuple[WebPage, ...]


class ScriptedPlace(_Fixture):
    query: str
    place: MapsPlace | None


class ScriptedRoute(_Fixture):
    origin: Location
    destination: Location
    mode: TransportMode
    route: MapsRoute


class ScriptedForecast(_Fixture):
    point: Location
    periods: tuple[WeatherPeriod, ...]


class ScriptedTurn(_Fixture):
    user_message: str
    replies: tuple[ModelReply, ...]


class ConversationFixture(_Fixture):
    """The replan conversation: clarification, then options, then a revision."""

    clock_start: AwareDatetime
    user_timezone: str
    limits: ExecutionLimits
    continuation: ContinuationPayload
    web: tuple[ScriptedSearch, ...]
    places: tuple[ScriptedPlace, ...]
    routes: tuple[ScriptedRoute, ...]
    weather: tuple[ScriptedForecast, ...]
    turns: tuple[ScriptedTurn, ...]


def load_conversation() -> ConversationFixture:
    return ConversationFixture.model_validate_json(CONVERSATION_CASE.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Script:
    """Doubles for one backend process: the model plays `turns` in order across requests."""

    model: ScriptedModel
    dependencies_for: Callable[[StatusSink], AgentDependencies]


def script_for(case: ConversationFixture, turns: Sequence[ScriptedTurn]) -> Script:
    clock = ManualClock(case.clock_start)
    model = ScriptedModel([r for turn in turns for r in turn.replies], clock, 0)
    web = ScriptedWebResearch({s.query: s.pages for s in case.web}, clock, 0)
    maps = ScriptedMaps(
        {p.query: p.place for p in case.places},
        {route_key(r.origin, r.destination, r.mode): r.route for r in case.routes},
    )
    weather = ScriptedWeather(
        {(f.point.latitude, f.point.longitude): f.periods for f in case.weather}
    )
    return Script(model, dependencies_with(model, clock, web, maps, weather))


def dependencies_with(
    model: AgentModel,
    clock: ManualClock,
    web: ScriptedWebResearch,
    maps: ScriptedMaps,
    weather: ScriptedWeather,
) -> Callable[[StatusSink], AgentDependencies]:
    def build(sink: StatusSink) -> AgentDependencies:
        return AgentDependencies(
            model=model, web=web, maps=maps, weather=weather, clock=clock, status_sink=sink
        )

    return build


def no_provider_dependencies(
    model: AgentModel, clock: ManualClock
) -> Callable[[StatusSink], AgentDependencies]:
    """Dependencies whose providers answer nothing (any provider call fails the test)."""
    return dependencies_with(
        model, clock, ScriptedWebResearch({}, clock, 0), ScriptedMaps({}, {}), ScriptedWeather({})
    )


def reference_outcomes(case: ConversationFixture) -> list[TurnOutcome]:
    """Run the fixture directly through `run_turn` with its own doubles."""
    script = script_for(case, case.turns)
    outcomes: list[TurnOutcome] = []
    state = case.continuation
    for turn in case.turns:
        outcome = run_turn(
            state,
            turn.user_message,
            case.user_timezone,
            None,
            script.dependencies_for(RecordingStatusSink()),
            case.limits,
        )
        outcomes.append(outcome)
        state = outcome.continuation
    return outcomes


AI_GUESS_ADMISSION: Doc = {
    "evidence": {
        "kind": "estimate",
        "evidence_id": "ev-admission",
        "topic": "cost",
        "claim": {"kind": "cost_amount", "amount": {"amount": "500", "currency": "JPY"}},
        "label": "ai_best_guess",
        "scope": {
            "place_id": "ChIJsyntheticQuietGarden01",
            "fallback_key": None,
            "applies_from": "2026-11-20T00:00:00+09:00",
            "applies_until": "2026-11-20T23:59:00+09:00",
        },
        "basis": "No current admission price was found.",
        "based_on": [],
    }
}


def with_guessed_admission(case: ConversationFixture) -> ConversationFixture:
    """The fixture with turn 2's admission price recorded as an AI best guess."""

    def swap(call: ToolCall) -> ToolCall:
        if call.call_id != "t2-admission":
            return call
        return ToolCall(
            call_id=call.call_id, name=call.name, arguments=json.dumps(AI_GUESS_ADMISSION)
        )

    def swapped(reply: ModelReply) -> ModelReply:
        if not isinstance(reply, ToolCallReply):
            return reply
        return ToolCallReply(kind="tool_calls", calls=tuple(swap(c) for c in reply.calls))

    turns = tuple(
        ScriptedTurn(user_message=t.user_message, replies=tuple(swapped(r) for r in t.replies))
        for t in case.turns
    )
    return case.model_copy(update={"turns": turns})


# --- small scripted turns -----------------------------------------------------------------


def call(call_id: str, name: str, arguments: Doc) -> ToolCall:
    return ToolCall(call_id=call_id, name=name, arguments=json.dumps(arguments))


def reply(*calls: ToolCall) -> ModelReply:
    return ToolCallReply(kind="tool_calls", calls=calls)


CLARIFY = reply(
    call(
        "clarify",
        "ask_clarification",
        {
            "message": "Where are you starting from?",
            "questions": [{"topic": "start_location", "question": "Where are you starting?"}],
        },
    )
)
NOW = reply(call("now", "get_datetime", {}))
SEARCH = reply(call("search", "web_search", {"query": "quiet places"}))


def clarifying_model(clock: ManualClock) -> ScriptedModel:
    return ScriptedModel([CLARIFY], clock, 0)


class RaisingModel:
    """Returns `replies` in order, then raises `error` on the next call."""

    def __init__(self, replies: Sequence[ModelReply], error: Exception) -> None:
        self._replies = tuple(replies)
        self._error = error
        self.calls = 0

    def respond(
        self, messages: tuple[ModelMessage, ...], tools: tuple[ToolSpec, ...]
    ) -> ModelReply:
        index = self.calls
        self.calls += 1
        if index < len(self._replies):
            return self._replies[index]
        raise self._error


class GatedModel:
    """Before its second reply, waits for a status event to reach the client."""

    def __init__(self, replies: Sequence[ModelReply], status_delivered: threading.Event) -> None:
        self._replies = tuple(replies)
        self._status_delivered = status_delivered
        self.calls = 0
        self.status_seen_mid_turn: bool | None = None

    def respond(
        self, messages: tuple[ModelMessage, ...], tools: tuple[ToolSpec, ...]
    ) -> ModelReply:
        index = self.calls
        self.calls += 1
        if index == 1:
            self.status_seen_mid_turn = self._status_delivered.wait(STREAM_WAIT_SECONDS)
        return self._replies[index]


class RecordingTranscriber:
    """Returns `transcript` and records every clip it is given."""

    def __init__(self, transcript: str) -> None:
        self._transcript = transcript
        self.calls: list[tuple[bytes, str]] = []

    def transcribe(self, audio: bytes, content_type: str) -> str:
        self.calls.append((audio, content_type))
        return self._transcript


class RaisingTranscriber:
    """Raises `error` for every clip."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.calls = 0

    def transcribe(self, audio: bytes, content_type: str) -> str:
        self.calls += 1
        raise self._error


def unused_transcriber() -> RecordingTranscriber:
    return RecordingTranscriber("unused")


def turn_app(dependencies_for: Callable[[StatusSink], AgentDependencies]) -> TestClient:
    return TestClient(create_app(dependencies_for, unused_transcriber(), LIMITS, API_LIMITS))


def speech_app(transcriber: RecordingTranscriber | RaisingTranscriber) -> TestClient:
    clock = ManualClock(TOKYO_CLOCK)
    app = create_app(
        no_provider_dependencies(clarifying_model(clock), clock), transcriber, LIMITS, API_LIMITS
    )
    return TestClient(app, raise_server_exceptions=False)


# --- request and response helpers -----------------------------------------------------------


def continuation_doc(messages: Sequence[tuple[str, str]], request: Doc) -> Doc:
    return {
        "schema_version": 1,
        "messages": [{"role": role, "text": text} for role, text in messages],
        "request": request,
    }


FRESH: Doc = continuation_doc([], EMPTY_REQUEST)


def turn_doc(
    continuation: Doc | ContinuationPayload,
    message: str,
    user_timezone: str,
    device_location: Location | None,
) -> Doc:
    return {
        "continuation": continuation
        if isinstance(continuation, dict)
        else continuation.model_dump(mode="json"),
        "message": message,
        "user_timezone": user_timezone,
        "device_location": None
        if device_location is None
        else device_location.model_dump(mode="json"),
    }


def post_turn(client: TestClient, document: Doc) -> Response:
    return post_turn_bytes(client, json.dumps(document).encode())


def post_turn_bytes(client: TestClient, body: bytes) -> Response:
    return client.post("/api/turn", content=body, headers={"content-type": "application/json"})


def post_audio(client: TestClient, audio: bytes, content_type: str | None) -> Response:
    headers = {} if content_type is None else {"content-type": content_type}
    return client.post("/api/transcribe", content=audio, headers=headers)


def events_of(response: Response) -> list[Event]:
    """Parse an NDJSON turn stream: status events, then exactly one terminal event."""
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert response.text.endswith("\n")
    events = [_EVENT.validate_json(line) for line in response.text.splitlines()]
    assert events, "empty stream"
    for event in events[:-1]:
        assert set(event) == {"type", "status"}, event
        assert event["type"] == "status"
    assert events[-1]["type"] in ("result", "error"), events[-1]
    return events


def statuses_of(events: list[Event]) -> list[JsonValue]:
    return [event["status"] for event in events[:-1]]


@dataclass(frozen=True)
class Final:
    result: TurnResult
    continuation: ContinuationPayload
    estimate_labels: dict[str, EstimateLabel]


def final_of(events: list[Event]) -> Final:
    last = events[-1]
    assert last["type"] == "result", last
    assert set(last) == {"type", "result", "continuation", "estimate_labels"}
    return Final(
        _TURN_RESULT.validate_json(json.dumps(last["result"])),
        ContinuationPayload.model_validate_json(json.dumps(last["continuation"])),
        _LABELS.validate_json(json.dumps(last["estimate_labels"])),
    )


def assert_error(response: Response, status: int, code: str) -> None:
    """An error is exactly {code, message} with no request values or internals."""
    assert response.status_code == status, response.text
    body = _EVENT.validate_json(response.text)
    assert set(body) == {"code", "message"}, body
    assert body["code"] == code
    assert isinstance(body["message"], str) and body["message"]
    for marker in (LEAK, ENV_SECRET, "Traceback", 'File "'):
        assert marker not in response.text


def options_of(result: TurnResult) -> ValidatedOptions:
    assert isinstance(result, ValidatedOptions), result
    return result


@pytest.fixture
def provider_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake provider keys in the environment; none may ever reach a response."""
    for name in PROVIDER_KEY_VARIABLES:
        monkeypatch.setenv(name, ENV_SECRET)


# --- turns: status stream and results --------------------------------------------------------


def test_each_turn_streams_its_statuses_in_order_then_one_result() -> None:
    case = load_conversation()
    client = turn_app(script_for(case, case.turns).dependencies_for)
    state: Doc | ContinuationPayload = case.continuation
    streamed: list[list[JsonValue]] = []
    kinds: list[str] = []
    for turn in case.turns:
        events = events_of(post_turn(client, turn_doc(state, turn.user_message, USER_TZ, None)))
        final = final_of(events)
        streamed.append(statuses_of(events))
        kinds.append(final.result.kind)
        state = final.continuation

    assert kinds == ["clarification", "validated_options", "validated_options"]
    assert streamed == [
        [RESEARCHING],
        [RESEARCHING, PLANNING, GENERATING],
        [PLANNING, RESEARCHING, PLANNING, GENERATING],
    ]


def test_result_events_carry_the_turn_result_and_next_continuation_unchanged() -> None:
    case = load_conversation()
    expected = reference_outcomes(case)
    client = turn_app(script_for(case, case.turns).dependencies_for)
    state: Doc | ContinuationPayload = case.continuation
    for turn, reference in zip(case.turns, expected, strict=True):
        final = final_of(
            events_of(post_turn(client, turn_doc(state, turn.user_message, USER_TZ, None)))
        )
        assert final.result == reference.result
        assert final.continuation == reference.continuation
        assert final.estimate_labels == {}
        state = final.continuation


def test_the_next_turn_gives_the_model_the_previous_turns_context() -> None:
    case = load_conversation()
    script = script_for(case, case.turns[:2])
    client = turn_app(script.dependencies_for)
    first = final_of(
        events_of(post_turn(client, turn_doc(FRESH, case.turns[0].user_message, USER_TZ, None)))
    )
    calls_before = len(script.model.received)

    events_of(
        post_turn(client, turn_doc(first.continuation, case.turns[1].user_message, USER_TZ, None))
    )

    opening = script.model.received[calls_before][0]
    assert [m.message for m in opening if isinstance(m, ConversationMessage)] == [
        *first.continuation.messages,
        ChatMessage(role=ChatRole.USER, text=case.turns[1].user_message),
    ]
    assert [m.request for m in opening if isinstance(m, RequestMessage)] == [
        first.continuation.request
    ]


def test_a_new_backend_instance_accepts_a_continuation_from_a_previous_one() -> None:
    case = load_conversation()
    expected = reference_outcomes(case)
    before_restart = turn_app(script_for(case, case.turns[:1]).dependencies_for)
    first = final_of(
        events_of(
            post_turn(before_restart, turn_doc(FRESH, case.turns[0].user_message, USER_TZ, None))
        )
    )

    restarted = script_for(case, case.turns[1:2])
    after_restart = turn_app(restarted.dependencies_for)
    second = final_of(
        events_of(
            post_turn(
                after_restart,
                turn_doc(first.continuation, case.turns[1].user_message, USER_TZ, None),
            )
        )
    )

    assert second.result == expected[1].result
    assert second.continuation == expected[1].continuation
    opening = restarted.model.received[0][0]
    conversation = [m.message for m in opening if isinstance(m, ConversationMessage)]
    assert conversation[:2] == list(first.continuation.messages)


def test_status_is_streamed_while_the_turn_is_still_running() -> None:
    watcher = StreamWatcher()
    clock = ManualClock(TOKYO_CLOCK)
    model = GatedModel([SEARCH, CLARIFY], watcher.status_delivered)
    web = ScriptedWebResearch({"quiet places": ()}, clock, 0)
    app = create_app(
        dependencies_with(model, clock, web, ScriptedMaps({}, {}), ScriptedWeather({})),
        unused_transcriber(),
        LIMITS,
        API_LIMITS,
    )
    body = json.dumps(turn_doc(FRESH, "Plan my afternoon.", USER_TZ, None)).encode()

    asyncio.run(post_over_asgi(app, body, watcher))

    assert model.status_seen_mid_turn is True
    assert watcher.status_code == 200
    lines = [_EVENT.validate_json(line) for line in watcher.buffer.splitlines()]
    assert [line["type"] for line in lines] == ["status", "result"]


class StreamWatcher:
    """Collects an ASGI response and signals once a complete status line has arrived."""

    def __init__(self) -> None:
        self.buffer = b""
        self.status_code: int | None = None
        self.status_delivered = threading.Event()

    def start(self, status_code: int) -> None:
        self.status_code = status_code

    def chunk(self, data: bytes) -> None:
        self.buffer += data
        complete = self.buffer.split(b"\n")[:-1]
        if any(_EVENT.validate_json(line).get("type") == "status" for line in complete):
            self.status_delivered.set()


async def post_over_asgi(app: FastAPI, body: bytes, watcher: StreamWatcher) -> None:
    """POST /api/turn over raw ASGI, so chunks are observed as the app sends them."""
    finished = asyncio.Event()
    body_sent = False

    async def receive() -> Message:
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await finished.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        if message["type"] == "http.response.start":
            watcher.start(cast(int, message["status"]))
        elif message["type"] == "http.response.body":
            watcher.chunk(cast(bytes, message.get("body", b"")))
            if not message.get("more_body", False):
                finished.set()

    await asyncio.wait_for(app(turn_scope(body), receive, send), timeout=4 * STREAM_WAIT_SECONDS)


def turn_scope(body: bytes) -> Scope:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/turn",
        "raw_path": b"/api/turn",
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8000),
    }


class AbandonedTurnModel:
    """Researches once, then fails with `error` only after the test releases it."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.calls = 0
        self.release = threading.Event()
        self.released_in_time: bool | None = None

    def respond(
        self, messages: tuple[ModelMessage, ...], tools: tuple[ToolSpec, ...]
    ) -> ModelReply:
        self.calls += 1
        if self.calls == 1:
            return SEARCH
        self.released_in_time = self.release.wait(STREAM_WAIT_SECONDS)
        raise self._error


async def disconnect_mid_turn(app: FastAPI, body: bytes, model: AbandonedTurnModel) -> bool:
    """Disconnect after the first status line, then let the abandoned turn fail.

    Returns whether a status line arrived before the disconnect. Waits (bounded) until no
    task other than this one is left, so an unretrieved worker exception is logged here.
    """
    status_seen = asyncio.Event()
    received = b""
    body_sent = False

    async def receive() -> Message:
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.wait_for(status_seen.wait(), timeout=STREAM_WAIT_SECONDS)
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        nonlocal received
        if message["type"] == "http.response.body":
            received += cast(bytes, message.get("body", b""))
            lines = received.split(b"\n")[:-1]
            if any(_EVENT.validate_json(line).get("type") == "status" for line in lines):
                status_seen.set()

    await asyncio.wait_for(app(turn_scope(body), receive, send), timeout=4 * STREAM_WAIT_SECONDS)
    model.release.set()
    deadline = asyncio.get_running_loop().time() + STREAM_WAIT_SECONDS
    current = asyncio.current_task()
    while {t for t in asyncio.all_tasks() if t is not current}:
        assert asyncio.get_running_loop().time() < deadline, "turn worker did not finish"
        await asyncio.sleep(0.01)
    return status_seen.is_set()


def test_a_turn_abandoned_by_a_disconnect_never_logs_its_exception_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = ManualClock(TOKYO_CLOCK)
    model = AbandonedTurnModel(RuntimeError(f"user text {LEAK}"))
    web = ScriptedWebResearch({"quiet places": ()}, clock, 0)
    app = create_app(
        dependencies_with(model, clock, web, ScriptedMaps({}, {}), ScriptedWeather({})),
        unused_transcriber(),
        LIMITS,
        API_LIMITS,
    )
    body = json.dumps(turn_doc(FRESH, "Plan my afternoon.", USER_TZ, None)).encode()

    with caplog.at_level(logging.DEBUG):
        status_before_disconnect = asyncio.run(disconnect_mid_turn(app, body, model))
        gc.collect()

    assert status_before_disconnect is True
    assert model.calls == 2
    assert model.released_in_time is True
    assert LEAK not in caplog.text
    assert not any(LEAK in record.getMessage() for record in caplog.records)


def test_estimate_labels_name_the_estimates_behind_displayed_costs() -> None:
    case = with_guessed_admission(load_conversation())
    client = turn_app(script_for(case, case.turns[:2]).dependencies_for)
    first = final_of(
        events_of(post_turn(client, turn_doc(FRESH, case.turns[0].user_message, USER_TZ, None)))
    )

    second = final_of(
        events_of(
            post_turn(
                client, turn_doc(first.continuation, case.turns[1].user_message, USER_TZ, None)
            )
        )
    )

    option = options_of(second.result).options[0].itinerary
    assert option.total_includes_estimates is True
    assert second.estimate_labels == {"ev-admission": EstimateLabel.AI_BEST_GUESS}


def test_provider_failure_is_a_result_event_with_a_continuation(provider_secrets: None) -> None:
    clock = ManualClock(TOKYO_CLOCK)
    model = RaisingModel([], ProviderError(f"upstream said {LEAK} {ENV_SECRET}"))
    client = turn_app(no_provider_dependencies(model, clock))

    response = post_turn(client, turn_doc(FRESH, "Plan my afternoon.", USER_TZ, None))

    final = final_of(events_of(response))
    assert isinstance(final.result, ExecutionFailure)
    assert final.result.reason == ExecutionFailureReason.PROVIDER_FAILURE
    assert [m.role for m in final.continuation.messages] == [ChatRole.USER, ChatRole.ASSISTANT]
    assert LEAK not in response.text
    assert ENV_SECRET not in response.text


def test_failure_after_streaming_starts_ends_with_an_error_event(provider_secrets: None) -> None:
    clock = ManualClock(TOKYO_CLOCK)
    model = RaisingModel([SEARCH], RuntimeError(f"bug near {LEAK} {ENV_SECRET}"))
    web = ScriptedWebResearch({"quiet places": ()}, clock, 0)
    client = TestClient(
        create_app(
            dependencies_with(model, clock, web, ScriptedMaps({}, {}), ScriptedWeather({})),
            unused_transcriber(),
            LIMITS,
            API_LIMITS,
        ),
        raise_server_exceptions=False,
    )

    response = post_turn(client, turn_doc(FRESH, "Plan my afternoon.", USER_TZ, None))

    events = events_of(response)
    assert statuses_of(events) == [RESEARCHING]
    error = events[-1]
    assert set(error) == {"type", "code", "message"}
    assert error["code"] == "internal_error"
    for marker in (LEAK, ENV_SECRET, "Traceback", "RuntimeError"):
        assert marker not in response.text


def test_dependency_setup_failure_before_streaming_is_an_internal_error(
    provider_secrets: None,
) -> None:
    def failing_setup(sink: StatusSink) -> AgentDependencies:
        raise RuntimeError(f"bad config {LEAK} {ENV_SECRET}")

    client = TestClient(
        create_app(failing_setup, unused_transcriber(), LIMITS, API_LIMITS),
        raise_server_exceptions=False,
    )

    response = post_turn(client, fresh_turn())

    assert response.headers["content-type"].startswith("application/json")
    assert_error(response, 500, "internal_error")


# --- turns: rejected before streaming --------------------------------------------------------


def fresh_turn() -> Doc:
    return turn_doc(FRESH, "Plan my afternoon.", USER_TZ, None)


def without(document: Doc, key: str) -> Doc:
    return {k: v for k, v in document.items() if k != key}


def with_key(document: Doc, key: str, value: object) -> Doc:
    return {**document, key: value}


MALFORMED_BODIES: dict[str, bytes] = {
    "not_json": b"{not json",
    "json_array": b"[]",
    "missing_device_location": json.dumps(without(fresh_turn(), "device_location")).encode(),
    "missing_continuation": json.dumps(without(fresh_turn(), "continuation")).encode(),
    "unknown_top_level_field": json.dumps(with_key(fresh_turn(), LEAK, True)).encode(),
    "claimed_evidence_in_continuation": json.dumps(
        with_key(fresh_turn(), "continuation", {**FRESH, "evidence": [{"finding": LEAK}]})
    ).encode(),
    "unknown_field_in_message": json.dumps(
        with_key(
            fresh_turn(),
            "continuation",
            {**FRESH, "messages": [{"role": "user", "text": "Hi", "verified": LEAK}]},
        )
    ).encode(),
    "message_not_a_string": json.dumps(with_key(fresh_turn(), "message", {"text": LEAK})).encode(),
    "empty_message": json.dumps(with_key(fresh_turn(), "message", "")).encode(),
    "messages_not_a_list": json.dumps(
        with_key(fresh_turn(), "continuation", {**FRESH, "messages": {"text": LEAK}})
    ).encode(),
    "unknown_role": json.dumps(
        with_key(
            fresh_turn(),
            "continuation",
            {**FRESH, "messages": [{"role": "system", "text": LEAK}]},
        )
    ).encode(),
    "unsupported_schema_version": json.dumps(
        with_key(fresh_turn(), "continuation", {**FRESH, "schema_version": 2})
    ).encode(),
    "device_location_out_of_range": json.dumps(
        with_key(
            fresh_turn(),
            "device_location",
            {"label": LEAK, "latitude": 91.0, "longitude": 139.7, "place_id": None},
        )
    ).encode(),
    "timezone_not_a_string": json.dumps(with_key(fresh_turn(), "user_timezone", 9)).encode(),
}


@pytest.mark.parametrize("body", MALFORMED_BODIES.values(), ids=MALFORMED_BODIES.keys())
def test_malformed_client_context_is_rejected_before_the_model_runs(body: bytes) -> None:
    clock = ManualClock(TOKYO_CLOCK)
    model = clarifying_model(clock)
    client = turn_app(no_provider_dependencies(model, clock))

    response = post_turn_bytes(client, body)

    assert_error(response, 422, "invalid_request")
    assert model.received == []


def test_message_longer_than_the_limit_is_rejected_and_the_limit_itself_is_accepted() -> None:
    clock = ManualClock(TOKYO_CLOCK)
    model = ScriptedModel([CLARIFY], clock, 0)
    client = turn_app(no_provider_dependencies(model, clock))
    limit = API_LIMITS.max_message_chars

    too_long = post_turn(client, turn_doc(FRESH, "a" * (limit + 1), USER_TZ, None))
    at_limit = post_turn(client, turn_doc(FRESH, "a" * limit, USER_TZ, None))

    assert_error(too_long, 422, "message_too_long")
    assert final_of(events_of(at_limit)).result.kind == "clarification"


@pytest.mark.parametrize("zone", ["asia/tokyo", "UTC+09:00", "+09:00", "", "Mars/Olympus_Mons"])
def test_invalid_timezone_is_rejected_before_the_model_runs(zone: str) -> None:
    clock = ManualClock(TOKYO_CLOCK)
    model = clarifying_model(clock)
    client = turn_app(no_provider_dependencies(model, clock))

    response = post_turn(client, turn_doc(FRESH, "Plan my afternoon.", zone, None))

    assert_error(response, 422, "invalid_timezone")
    assert model.received == []


def chat(count: int) -> list[tuple[str, str]]:
    return [("user" if i % 2 == 0 else "assistant", f"Message {i}.") for i in range(count)]


def test_continuation_overflow_has_a_stable_code_and_the_cap_itself_is_accepted() -> None:
    clock = ManualClock(TOKYO_CLOCK)
    model = ScriptedModel([CLARIFY], clock, 0)
    client = turn_app(no_provider_dependencies(model, clock))
    full = continuation_doc(chat(MAX_CONTINUATION_MESSAGES - 1), EMPTY_REQUEST)
    room = continuation_doc(chat(MAX_CONTINUATION_MESSAGES - 2), EMPTY_REQUEST)

    overflow = post_turn(client, turn_doc(full, "One more thing.", USER_TZ, None))
    calls_after_overflow = len(model.received)
    accepted = post_turn(client, turn_doc(room, "One more thing.", USER_TZ, None))

    assert_error(overflow, 422, "continuation_overflow")
    assert calls_after_overflow == 0
    assert len(final_of(events_of(accepted)).continuation.messages) == MAX_CONTINUATION_MESSAGES


def test_body_larger_than_the_limit_is_rejected_and_the_limit_itself_is_accepted() -> None:
    clock = ManualClock(TOKYO_CLOCK)
    model = ScriptedModel([CLARIFY], clock, 0)
    client = turn_app(no_provider_dependencies(model, clock))
    body = json.dumps(fresh_turn()).encode()
    at_limit = body + b" " * (API_LIMITS.max_turn_body_bytes - len(body))

    too_large = post_turn_bytes(client, at_limit + b" ")
    calls_after_rejection = len(model.received)
    accepted = post_turn_bytes(client, at_limit)

    assert_error(too_large, 413, "body_too_large")
    assert calls_after_rejection == 0
    assert final_of(events_of(accepted)).result.kind == "clarification"


# --- user timezone through get_datetime ------------------------------------------------------


@pytest.mark.parametrize(
    ("instant", "zone", "expected"),
    [
        # America/New_York leaves daylight saving time at 2026-11-01 06:00 UTC.
        (datetime(2026, 11, 1, 5, 30, tzinfo=UTC), "America/New_York", "2026-11-01T01:30:00-04:00"),
        (datetime(2026, 11, 1, 6, 30, tzinfo=UTC), "America/New_York", "2026-11-01T01:30:00-05:00"),
        # Asia/Tokyo has no daylight saving time: the offset is the same at both instants.
        (datetime(2026, 11, 1, 5, 30, tzinfo=UTC), "Asia/Tokyo", "2026-11-01T14:30:00+09:00"),
        (datetime(2026, 11, 1, 6, 30, tzinfo=UTC), "Asia/Tokyo", "2026-11-01T15:30:00+09:00"),
    ],
)
def test_browser_timezone_reaches_get_datetime(instant: datetime, zone: str, expected: str) -> None:
    clock = ManualClock(instant)
    model = ScriptedModel([NOW, CLARIFY], clock, 0)
    client = turn_app(no_provider_dependencies(model, clock))

    final_of(events_of(post_turn(client, turn_doc(FRESH, "What can I do now?", zone, None))))

    observed = [
        m.result
        for m in model.received[1][0]
        if isinstance(m, ToolResultMessage) and isinstance(m.result, DatetimeObservation)
    ]
    assert [o.model_dump(mode="json")["now"] for o in observed] == [expected]


# --- device location (R9, O1) ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("device_location", "expected"),
    [(DEVICE_LOCATION, [DEVICE_LOCATION]), (None, [])],
    ids=["granted", "denied"],
)
def test_device_location_is_passed_to_the_turn_only_when_granted(
    device_location: Location | None, expected: list[Location]
) -> None:
    clock = ManualClock(TOKYO_CLOCK)
    model = clarifying_model(clock)
    client = turn_app(no_provider_dependencies(model, clock))

    final = final_of(
        events_of(
            post_turn(client, turn_doc(FRESH, "Plan my afternoon.", USER_TZ, device_location))
        )
    )

    opening = model.received[0][0]
    assert [m.location for m in opening if isinstance(m, DeviceLocationMessage)] == expected
    assert final.continuation.request.gap.start_location is None


# --- speech (D3) ------------------------------------------------------------------------------


def test_transcription_returns_the_transcript_for_the_exact_clip() -> None:
    transcriber = RecordingTranscriber("I'm free from 3pm near Shibuya.")
    client = speech_app(transcriber)

    response = post_audio(client, AUDIO, AUDIO_TYPE)

    assert response.status_code == 200, response.text
    assert _EVENT.validate_json(response.text) == {"transcript": "I'm free from 3pm near Shibuya."}
    assert transcriber.calls == [(AUDIO, AUDIO_TYPE)]


@pytest.mark.parametrize(
    "transcriber",
    [
        RaisingTranscriber(TranscriptionError(f"provider said {LEAK} {ENV_SECRET}")),
        RecordingTranscriber(""),
        RecordingTranscriber("   "),
    ],
    ids=["transcription_error", "empty_transcript", "blank_transcript"],
)
def test_transcription_failure_is_an_explicit_error(
    transcriber: RecordingTranscriber | RaisingTranscriber, provider_secrets: None
) -> None:
    response = post_audio(speech_app(transcriber), AUDIO, AUDIO_TYPE)

    assert_error(response, 502, "transcription_failed")


def test_unexpected_transcriber_failure_is_an_internal_error(provider_secrets: None) -> None:
    response = post_audio(
        speech_app(RaisingTranscriber(RuntimeError(f"bug near {LEAK} {ENV_SECRET}"))),
        AUDIO,
        AUDIO_TYPE,
    )

    assert_error(response, 500, "internal_error")


class NonTextTranscriber:
    """Breaks the Transcriber protocol by returning no text, so the endpoint itself fails."""

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio: bytes, content_type: str) -> str:
        self.calls += 1
        return cast(str, None)


def test_unexpected_failure_after_transcribing_is_an_internal_error() -> None:
    clock = ManualClock(TOKYO_CLOCK)
    transcriber = NonTextTranscriber()
    client = TestClient(
        create_app(
            no_provider_dependencies(clarifying_model(clock), clock),
            transcriber,
            LIMITS,
            API_LIMITS,
        ),
        raise_server_exceptions=False,
    )

    response = post_audio(client, AUDIO, AUDIO_TYPE)

    assert transcriber.calls == 1
    assert response.headers["content-type"].startswith("application/json")
    assert_error(response, 500, "internal_error")


def test_audio_larger_than_the_limit_is_rejected_and_the_limit_itself_is_accepted() -> None:
    transcriber = RecordingTranscriber("A transcript.")
    client = speech_app(transcriber)
    limit = API_LIMITS.max_audio_bytes

    too_large = post_audio(client, b"\x01" * (limit + 1), AUDIO_TYPE)
    calls_after_rejection = len(transcriber.calls)
    at_limit = post_audio(client, b"\x01" * limit, AUDIO_TYPE)

    assert_error(too_large, 413, "audio_too_large")
    assert calls_after_rejection == 0
    assert at_limit.status_code == 200, at_limit.text


def test_empty_audio_is_rejected_without_transcribing() -> None:
    transcriber = RecordingTranscriber("A transcript.")

    response = post_audio(speech_app(transcriber), b"", AUDIO_TYPE)

    assert_error(response, 400, "empty_audio")
    assert transcriber.calls == []


@pytest.mark.parametrize("content_type", [None, "application/octet-stream", "text/plain"])
def test_non_audio_upload_is_rejected_without_transcribing(content_type: str | None) -> None:
    transcriber = RecordingTranscriber("A transcript.")

    response = post_audio(speech_app(transcriber), AUDIO, content_type)

    assert_error(response, 415, "unsupported_audio_type")
    assert transcriber.calls == []


def test_audio_and_transcript_are_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    transcript = f"Transcript {LEAK}"
    client = speech_app(RecordingTranscriber(transcript))
    audio = b"audio-" + LEAK.encode()

    with caplog.at_level(logging.DEBUG):
        response = post_audio(client, audio, AUDIO_TYPE)

    assert response.status_code == 200, response.text
    assert LEAK not in caplog.text


# --- offline demo (D4) -------------------------------------------------------------------------


def demo_client() -> TestClient:
    spec = importlib.util.spec_from_file_location("demo_offline", DEMO_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build = cast(Callable[[], FastAPI], module.build_demo_app)
    return TestClient(build(), raise_server_exceptions=False)


def demo_turn(
    client: TestClient, state: Doc | ContinuationPayload, message: str, located: bool
) -> list[Event]:
    location = DEVICE_LOCATION if located else None
    return events_of(post_turn(client, turn_doc(state, message, USER_TZ, location)))


def has_run(statuses: list[JsonValue], run: list[JsonValue]) -> bool:
    return any(statuses[i : i + len(run)] == run for i in range(len(statuses)))


def test_demo_completes_a_clarification_then_three_distinct_options() -> None:
    client = demo_client()

    first_events = demo_turn(client, FRESH, "Plan something before dinner.", False)
    first = final_of(first_events)
    second_events = demo_turn(client, first.continuation, "Shibuya, and yen.", False)
    second = final_of(second_events)

    assert first.result.kind == "clarification"
    assert RESEARCHING in statuses_of(first_events)
    assert GENERATING not in statuses_of(first_events)
    statuses = statuses_of(second_events)
    assert has_run(statuses, [PLANNING, RESEARCHING, PLANNING]), statuses
    assert statuses[-1] == GENERATING
    options = options_of(second.result).options
    assert sorted(label for o in options for label in o.labels) == sorted(OptionLabel)
    assert all(len(o.labels) == 1 for o in options)
    plans = [frozenset(stop_key(s) for s in o.itinerary.candidate.stops) for o in options]
    assert len(set(plans)) == len(plans) == 3
    assert any(o.itinerary.weather_warnings for o in options)
    assert EstimateLabel.AI_BEST_GUESS in second.estimate_labels.values()
    assert len(second.continuation.messages) == 4


def test_demo_follows_its_script_whatever_the_user_says() -> None:
    client = demo_client()

    typed = final_of(demo_turn(client, FRESH, "Plan something before dinner.", False))
    other = final_of(demo_turn(client, FRESH, "Something completely different.", True))

    assert other.result == typed.result
    assert other.estimate_labels == typed.estimate_labels


def test_demo_reports_running_past_its_script_and_restarts_for_a_fresh_session() -> None:
    client = demo_client()
    first = final_of(demo_turn(client, FRESH, "Plan something before dinner.", False))
    second = final_of(demo_turn(client, first.continuation, "Shibuya, and yen.", False))

    past_end = demo_turn(client, second.continuation, "And after dinner?", False)
    again = final_of(demo_turn(client, FRESH, "Plan something before dinner.", False))

    assert past_end[-1]["type"] == "error"
    assert past_end[-1]["code"] == "internal_error"
    assert again.result == first.result


def test_demo_transcript_can_start_a_spoken_conversation() -> None:
    client = demo_client()

    response = post_audio(client, AUDIO, AUDIO_TYPE)

    assert response.status_code == 200, response.text
    transcript = _EVENT.validate_json(response.text)["transcript"]
    assert isinstance(transcript, str) and 0 < len(transcript.strip()) <= MAX_MESSAGE_CHARS
    first = final_of(demo_turn(client, FRESH, transcript, False))
    assert first.result.kind == "clarification"
