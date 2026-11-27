"""FastAPI app factory for the website chat (Phase 3).

``create_app`` requires every dependency to be injected: the per-turn agent dependencies,
the transcriber, and explicit limits. Nothing here reads the environment or selects test
doubles; the offline demo wires its doubles in ``backend/scripts/demo_offline.py``.

Error bodies are exactly ``{"code", "message"}`` with fixed text per code; they never carry
request values or exception details. Audio is held in memory for one request only, and
logs never contain audio, transcripts, or message text.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Mapping
from enum import StrEnum
from typing import Annotated, Final, Literal, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter, ValidationError
from starlette.concurrency import run_in_threadpool

from travel_agent.agent import (
    AgentDependencies,
    ContinuationOverflowError,
    ExecutionLimits,
    StatusSink,
    TurnOutcome,
    require_room,
    run_turn,
)
from travel_agent.contracts import (
    MAX_MESSAGE_CHARS,
    ChatMessage,
    ChatRole,
    ContinuationPayload,
    Estimate,
    EstimateLabel,
    IanaTimezone,
    Location,
    TurnResult,
    TurnStatus,
    ValidatedOptions,
)

_LOGGER: Final = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """The transcriber could not produce a transcript; its message is never sent to clients."""


class Transcriber(Protocol):
    """Speech to text for one recorded clip."""

    def transcribe(self, audio: bytes, content_type: str) -> str:
        """Return the transcript, or raise ``TranscriptionError``."""
        ...


class ApiLimits(BaseModel):
    """Request size limits; explicit, validated, and without defaults (O2)."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    max_turn_body_bytes: Annotated[int, Field(ge=1)]
    max_message_chars: Annotated[int, Field(ge=1, le=MAX_MESSAGE_CHARS)]
    max_audio_bytes: Annotated[int, Field(ge=1)]


type DependencyFactory = Callable[[StatusSink], AgentDependencies]


class ErrorCode(StrEnum):
    """Stable error codes the browser may rely on."""

    BODY_TOO_LARGE = "body_too_large"
    INVALID_REQUEST = "invalid_request"
    MESSAGE_TOO_LONG = "message_too_long"
    INVALID_TIMEZONE = "invalid_timezone"
    CONTINUATION_OVERFLOW = "continuation_overflow"
    INTERNAL_ERROR = "internal_error"
    AUDIO_TOO_LARGE = "audio_too_large"
    UNSUPPORTED_AUDIO_TYPE = "unsupported_audio_type"
    EMPTY_AUDIO = "empty_audio"
    TRANSCRIPTION_FAILED = "transcription_failed"


_ERRORS: Final[Mapping[ErrorCode, tuple[int, str]]] = {
    ErrorCode.BODY_TOO_LARGE: (413, "The request is too large."),
    ErrorCode.INVALID_REQUEST: (422, "The request is not valid. Please start a new conversation."),
    ErrorCode.MESSAGE_TOO_LONG: (422, "The message is too long. Please shorten it."),
    ErrorCode.INVALID_TIMEZONE: (422, "Your timezone could not be recognised."),
    ErrorCode.CONTINUATION_OVERFLOW: (
        422,
        "This conversation is full. Please start a new conversation.",
    ),
    ErrorCode.INTERNAL_ERROR: (500, "Something went wrong on our side. Please try again."),
    ErrorCode.AUDIO_TOO_LARGE: (413, "The recording is too long. Please record a shorter one."),
    ErrorCode.UNSUPPORTED_AUDIO_TYPE: (415, "The upload is not an audio recording."),
    ErrorCode.EMPTY_AUDIO: (400, "The recording is empty. Please try again."),
    ErrorCode.TRANSCRIPTION_FAILED: (
        502,
        "The recording could not be transcribed. Please try again or type your message.",
    ),
}


def _error(code: ErrorCode) -> JSONResponse:
    status, message = _ERRORS[code]
    return JSONResponse({"code": code.value, "message": message}, status_code=status)


async def _bounded_body(request: Request, limit: int) -> bytes | None:
    """Read the body into memory, or return None as soon as it exceeds ``limit`` bytes."""
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


_INTERNAL_ERROR_MESSAGE: Final[str] = _ERRORS[ErrorCode.INTERNAL_ERROR][1]
_TIMEZONE: Final[TypeAdapter[str]] = TypeAdapter(IanaTimezone)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _TurnBody(_Model):
    """The browser's turn request; every key is required and the context is untrusted."""

    continuation: ContinuationPayload
    message: Annotated[str, StringConstraints(min_length=1)]
    user_timezone: str
    device_location: Location | None


class _StatusEvent(_Model):
    type: Literal["status"]
    status: TurnStatus


class _ResultEvent(_Model):
    type: Literal["result"]
    result: TurnResult
    continuation: ContinuationPayload
    estimate_labels: dict[str, EstimateLabel]


class _ErrorEvent(_Model):
    type: Literal["error"]
    code: Literal[ErrorCode.INTERNAL_ERROR]
    message: str


def _line(event: _StatusEvent | _ResultEvent | _ErrorEvent) -> bytes:
    return event.model_dump_json().encode() + b"\n"


def _estimate_labels(outcome: TurnOutcome) -> dict[str, EstimateLabel]:
    """Label every estimate cited by a displayed option's costs or legs (R2, R13)."""
    if not isinstance(outcome.result, ValidatedOptions):
        return {}
    estimates = {
        item.evidence_id: item.label for item in outcome.evidence if isinstance(item, Estimate)
    }
    cited = (
        evidence_id
        for option in outcome.result.options
        for item in (*option.itinerary.candidate.costs, *option.itinerary.candidate.legs)
        for evidence_id in item.evidence_ids
    )
    return {
        evidence_id: estimates[evidence_id] for evidence_id in cited if evidence_id in estimates
    }


class _QueueSink:
    """Hands statuses from the turn's worker thread to the event loop, in order."""

    def __init__(
        self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[TurnStatus | None]
    ) -> None:
        self._loop = loop
        self._queue = queue

    def emit(self, status: TurnStatus) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, status)


async def _terminal_line(
    body: _TurnBody, dependencies: AgentDependencies, execution_limits: ExecutionLimits
) -> bytes:
    """Run the turn in a worker thread and return its one terminal line; never raises.

    Catching here means the outcome is always collected, even when the client has
    disconnected and nobody awaits it, so an exception is only ever logged by type.
    """
    try:
        outcome = await run_in_threadpool(
            run_turn,
            body.continuation,
            body.message,
            body.user_timezone,
            body.device_location,
            dependencies,
            execution_limits,
        )
        return _line(
            _ResultEvent(
                type="result",
                result=outcome.result,
                continuation=outcome.continuation,
                estimate_labels=_estimate_labels(outcome),
            )
        )
    except Exception as error:
        _LOGGER.error("turn_failed", extra={"error_type": type(error).__name__})
        return _line(
            _ErrorEvent(
                type="error", code=ErrorCode.INTERNAL_ERROR, message=_INTERNAL_ERROR_MESSAGE
            )
        )


async def _turn_events(
    queue: asyncio.Queue[TurnStatus | None],
    body: _TurnBody,
    dependencies: AgentDependencies,
    execution_limits: ExecutionLimits,
) -> AsyncIterator[bytes]:
    """Run the turn off the event loop, streaming its statuses and then one terminal event."""
    worker = asyncio.ensure_future(_terminal_line(body, dependencies, execution_limits))
    worker.add_done_callback(lambda _: queue.put_nowait(None))
    while (status := await queue.get()) is not None:
        yield _line(_StatusEvent(type="status", status=status))
    yield await worker


async def _start_turn(
    request: Request,
    dependencies_for: DependencyFactory,
    execution_limits: ExecutionLimits,
    api_limits: ApiLimits,
) -> Response:
    """Reject a bad request before streaming (contract order), else start the stream."""
    raw = await _bounded_body(request, api_limits.max_turn_body_bytes)
    if raw is None:
        return _error(ErrorCode.BODY_TOO_LARGE)
    try:
        body = _TurnBody.model_validate_json(raw)
    except ValidationError:
        return _error(ErrorCode.INVALID_REQUEST)
    if len(body.message) > api_limits.max_message_chars:
        return _error(ErrorCode.MESSAGE_TOO_LONG)
    try:
        _TIMEZONE.validate_python(body.user_timezone, strict=True)
    except ValidationError:
        return _error(ErrorCode.INVALID_TIMEZONE)
    try:
        require_room(body.continuation, ChatMessage(role=ChatRole.USER, text=body.message))
    except ContinuationOverflowError:
        return _error(ErrorCode.CONTINUATION_OVERFLOW)
    queue: asyncio.Queue[TurnStatus | None] = asyncio.Queue()
    dependencies = dependencies_for(_QueueSink(asyncio.get_running_loop(), queue))
    return StreamingResponse(
        _turn_events(queue, body, dependencies, execution_limits),
        media_type="application/x-ndjson",
    )


async def _transcribe(
    request: Request, transcriber: Transcriber, api_limits: ApiLimits
) -> JSONResponse:
    audio = await _bounded_body(request, api_limits.max_audio_bytes)
    if audio is None:
        return _error(ErrorCode.AUDIO_TOO_LARGE)
    content_type = request.headers.get("content-type")
    if content_type is None or not content_type.startswith("audio/"):
        return _error(ErrorCode.UNSUPPORTED_AUDIO_TYPE)
    if not audio:
        return _error(ErrorCode.EMPTY_AUDIO)
    try:
        transcript = await run_in_threadpool(transcriber.transcribe, audio, content_type)
    except TranscriptionError:
        _LOGGER.warning("transcription_failed")
        return _error(ErrorCode.TRANSCRIPTION_FAILED)
    if not transcript.strip():
        _LOGGER.warning("transcription_empty")
        return _error(ErrorCode.TRANSCRIPTION_FAILED)
    return JSONResponse({"transcript": transcript})


def create_app(
    dependencies_for: DependencyFactory,
    transcriber: Transcriber,
    execution_limits: ExecutionLimits,
    api_limits: ApiLimits,
) -> FastAPI:
    """Build the chat API around injected dependencies and limits."""
    app = FastAPI()

    @app.post("/api/turn")
    async def turn(request: Request) -> Response:  # pyright: ignore[reportUnusedFunction]
        try:
            return await _start_turn(request, dependencies_for, execution_limits, api_limits)
        except Exception as error:
            _LOGGER.error("turn_setup_error", extra={"error_type": type(error).__name__})
            return _error(ErrorCode.INTERNAL_ERROR)

    @app.post("/api/transcribe")
    async def transcribe(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        try:
            return await _transcribe(request, transcriber, api_limits)
        except Exception as error:
            _LOGGER.error("transcribe_error", extra={"error_type": type(error).__name__})
            return _error(ErrorCode.INTERNAL_ERROR)

    return app
