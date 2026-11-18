"""Typed boundary contracts shared by the agent loop, Python checks, and the API.

Every model is strict, frozen, and rejects unknown keys. No field has a default:
nullable fields need an explicit ``null`` so missing essentials stay visible.
These types validate shape and internal consistency only and never treat claims
as facts; schedule, cost, and evidence checks belong to ``feasibility.py``.
"""

from collections import Counter
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Final, Literal, cast
from zoneinfo import available_timezones

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    ValidationInfo,
    field_validator,
)

CONTRACT_SCHEMA_VERSION: Final[int] = 1
MAX_MESSAGE_CHARS: Final[int] = 2000
MAX_CONTINUATION_MESSAGES: Final[int] = 40
MAX_CONTINUATION_TEXT_CHARS: Final[int] = 32000
MAX_CONSTRAINTS: Final[int] = 20
MAX_CONSTRAINT_CHARS: Final[int] = 500
MAX_UNRESOLVED_QUESTIONS: Final[int] = 10
MAX_STOPS: Final[int] = 8
MAX_TEXT_CHARS: Final[int] = 2000
MAX_LABEL_CHARS: Final[int] = 200
MAX_ID_CHARS: Final[int] = 64
MAX_FEEDBACK_INPUTS: Final[int] = 20
MAX_OPEN_INTERVALS: Final[int] = 8
MAX_TRANSFERS: Final[int] = 10

_MAX_PLACE_ID_CHARS: Final[int] = 512
_MAX_DAY_MINUTES: Final[int] = 1440
# Exact-case keys; requires the tzdata package where the OS has no tz database.
_IANA_TIMEZONES: Final[frozenset[str]] = frozenset(available_timezones())


# --- validators -----------------------------------------------------------------


def _require_iana_timezone(value: str) -> str:
    """Reject strings that are not an exact, case-sensitive IANA time zone key."""
    if value not in _IANA_TIMEZONES:
        raise ValueError(
            f"unknown IANA time zone {value!r}; an exact IANA name such as 'Asia/Tokyo' is required"
        )
    return value


def _require_supported_schema_version(value: object) -> object:
    """Reject any schema_version other than the integer CONTRACT_SCHEMA_VERSION."""
    if type(value) is not int or value != CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {value!r}; supported: {CONTRACT_SCHEMA_VERSION}"
        )
    return value


def _require_unique[T](values: tuple[T, ...]) -> tuple[T, ...]:
    """Reject tuples that repeat an item."""
    duplicates = sorted(str(item) for item, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate entries are not allowed: {duplicates}")
    return values


# --- scalar types ---------------------------------------------------------------

Identifier = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_ID_CHARS, pattern=r"^[A-Za-z0-9_-]+$"),
]
PlaceId = Annotated[
    str,
    StringConstraints(min_length=1, max_length=_MAX_PLACE_ID_CHARS, pattern=r"^[A-Za-z0-9_-]+$"),
]
CurrencyCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
IanaTimezone = Annotated[
    str, StringConstraints(min_length=1), AfterValidator(_require_iana_timezone)
]
_SchemaVersion = Annotated[Literal[1], BeforeValidator(_require_supported_schema_version)]

_LabelText = Annotated[str, StringConstraints(min_length=1, max_length=MAX_LABEL_CHARS)]
_ConstraintText = Annotated[str, StringConstraints(min_length=1, max_length=MAX_CONSTRAINT_CHARS)]
_MessageText = Annotated[str, StringConstraints(min_length=1, max_length=MAX_MESSAGE_CHARS)]
_FreeText = Annotated[str, StringConstraints(min_length=1, max_length=MAX_TEXT_CHARS)]
_Latitude = Annotated[float, Field(ge=-90, le=90, allow_inf_nan=False)]
_Longitude = Annotated[float, Field(ge=-180, le=180, allow_inf_nan=False)]
_NonNegativeAmount = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]

RequirementId = Literal[
    "R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "R10",
    "R11", "R12", "R13", "R14", "R15", "R16", "R17",
]  # fmt: skip


# --- enumerations ---------------------------------------------------------------


class TransportMode(StrEnum):
    WALKING = "walking"
    PUBLIC_TRANSIT = "public_transit"
    TAXI_RIDESHARE = "taxi_rideshare"


class TurnStatus(StrEnum):
    RESEARCHING = "Researching"
    PLANNING = "Planning"
    GENERATING_ITINERARY = "Generating itinerary"


class UnresolvedTopic(StrEnum):
    START_LOCATION = "start_location"
    GAP_START_TIME = "gap_start_time"
    COMMITMENT_LOCATION = "commitment_location"
    COMMITMENT_TIME = "commitment_time"
    COMMITMENT_TIMEZONE = "commitment_timezone"
    BUDGET_CURRENCY = "budget_currency"
    OTHER = "other"


class SourceKind(StrEnum):
    """Where a finding came from; a venue's or operator's own site is OFFICIAL_WEBSITE."""

    OFFICIAL_WEBSITE = "official_website"
    OFFICIAL_SOCIAL_MEDIA = "official_social_media"
    OFFICIAL_AUTHORITY_NOTICE = "official_authority_notice"
    GOOGLE_MAPS = "google_maps"
    OTHER_WEB = "other_web"


class EvidenceTopic(StrEnum):
    OPENING_ACCESS = "opening_access"
    VISIT_DURATION = "visit_duration"
    COST = "cost"
    WEATHER = "weather"
    CONSTRAINT = "constraint"


class OpeningStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


class EstimateLabel(StrEnum):
    ESTIMATE = "estimate"
    AI_BEST_GUESS = "ai_best_guess"


class ConstraintId(StrEnum):
    ARRIVAL_BUFFER = "arrival_buffer"
    TRANSFER_TIME = "transfer_time"
    OPENING_ACCESS = "opening_access"
    VISIT_DURATION = "visit_duration"
    BUDGET = "budget"
    TRANSPORT_AVAILABILITY = "transport_availability"
    EXPLICIT_CONSTRAINT = "explicit_constraint"
    MISSING_EVIDENCE = "missing_evidence"
    SAME_DAY_GAP = "same_day_gap"


class OptionLabel(StrEnum):
    AI_RECOMMENDED = "ai_recommended"
    CHEAPEST = "cheapest"
    BEST_TRAVEL = "best_travel"


class ExecutionFailureReason(StrEnum):
    PROVIDER_FAILURE = "provider_failure"
    EXECUTION_LIMIT_REACHED = "execution_limit_reached"
    INTERNAL_ERROR = "internal_error"


class ChatRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class TurnResultKind(StrEnum):
    CLARIFICATION = "clarification"
    VALIDATED_OPTIONS = "validated_options"
    NO_FEASIBLE_RESULT = "no_feasible_result"
    EXECUTION_FAILURE = "execution_failure"


# --- models ---------------------------------------------------------------------


class _Contract(BaseModel):
    """Base for every contract: strict, immutable, and closed to unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Location(_Contract):
    label: _LabelText
    latitude: _Latitude
    longitude: _Longitude
    place_id: PlaceId | None


class Money(_Contract):
    amount: _NonNegativeAmount
    currency: CurrencyCode


class Gap(_Contract):
    """The free time being planned; either part may still be unknown."""

    start_time: AwareDatetime | None
    start_location: Location | None


class Commitment(_Contract):
    """The fixed event that ends the gap; ``timezone`` is the commitment-local zone."""

    location: Location | None
    time: AwareDatetime | None
    timezone: IanaTimezone | None


class Budget(_Contract):
    """A stated spending cap; ``currency`` is None until the conversation establishes it."""

    amount: _NonNegativeAmount
    currency: CurrencyCode | None


class UnresolvedQuestion(_Contract):
    topic: UnresolvedTopic
    question: _ConstraintText


class PlanningRequest(_Contract):
    """Untrusted boundary payload describing one gap before one commitment."""

    schema_version: _SchemaVersion
    gap: Gap
    commitment: Commitment
    budget: Budget | None
    transport_modes: Annotated[tuple[TransportMode, ...], AfterValidator(_require_unique)]
    constraints: Annotated[tuple[_ConstraintText, ...], Field(max_length=MAX_CONSTRAINTS)]
    unresolved_questions: Annotated[
        tuple[UnresolvedQuestion, ...], Field(max_length=MAX_UNRESOLVED_QUESTIONS)
    ]


class SourceReference(_Contract):
    url: HttpUrl
    kind: SourceKind
    title: _LabelText | None


class EvidenceScope(_Contract):
    """What the evidence applies to; ``place_id`` is None for non-place evidence."""

    place_id: PlaceId | None
    applies_from: AwareDatetime
    applies_until: AwareDatetime

    @field_validator("applies_until")
    @classmethod
    def _window_is_ordered(cls, applies_until: datetime, info: ValidationInfo) -> datetime:
        applies_from = info.data.get("applies_from")
        if isinstance(applies_from, datetime) and applies_from > applies_until:
            raise ValueError(
                f"applies_until {applies_until.isoformat()} is before "
                f"applies_from {applies_from.isoformat()}"
            )
        return applies_until


class OpenInterval(_Contract):
    """A span during which a place is open or accessible."""

    opens_at: AwareDatetime
    closes_at: AwareDatetime

    @field_validator("closes_at")
    @classmethod
    def _closes_after_opening(cls, closes_at: datetime, info: ValidationInfo) -> datetime:
        opens_at = info.data.get("opens_at")
        if isinstance(opens_at, datetime) and closes_at <= opens_at:
            raise ValueError(
                f"closes_at {closes_at.isoformat()} must be after opens_at {opens_at.isoformat()}"
            )
        return closes_at


class OpeningAccessClaim(_Contract):
    """The place is open or accessible exactly during ``open_intervals`` within the scope."""

    kind: Literal["opening_access"]
    status: OpeningStatus
    open_intervals: Annotated[tuple[OpenInterval, ...], Field(max_length=MAX_OPEN_INTERVALS)]

    @field_validator("open_intervals")
    @classmethod
    def _intervals_match_status(
        cls, open_intervals: tuple[OpenInterval, ...], info: ValidationInfo
    ) -> tuple[OpenInterval, ...]:
        status = info.data.get("status")
        if status == OpeningStatus.CLOSED and open_intervals:
            raise ValueError(
                f"status 'closed' requires no open_intervals; got {len(open_intervals)}"
            )
        if status == OpeningStatus.OPEN and not open_intervals:
            raise ValueError("status 'open' requires at least one open interval; got none")
        return open_intervals


class VisitDurationClaim(_Contract):
    kind: Literal["visit_duration"]
    minutes: Annotated[int, Field(ge=1, le=_MAX_DAY_MINUTES)]


class CostAmountClaim(_Contract):
    kind: Literal["cost_amount"]
    amount: Money


class TaxiRateClaim(_Contract):
    """A distance-based taxi rate valid on inclusive local dates; no time-based charge."""

    kind: Literal["taxi_rate"]
    base_fare: Money
    per_km: Money
    valid_from: date
    valid_until: date

    @field_validator("per_km")
    @classmethod
    def _per_km_matches_base_currency(cls, per_km: Money, info: ValidationInfo) -> Money:
        base_fare = info.data.get("base_fare")
        if isinstance(base_fare, Money) and base_fare.currency != per_km.currency:
            raise ValueError(
                f"per_km currency {per_km.currency!r} differs from "
                f"base_fare currency {base_fare.currency!r}"
            )
        return per_km

    @field_validator("valid_until")
    @classmethod
    def _validity_is_ordered(cls, valid_until: date, info: ValidationInfo) -> date:
        valid_from = info.data.get("valid_from")
        if isinstance(valid_from, date) and valid_from > valid_until:
            raise ValueError(
                f"valid_until {valid_until.isoformat()} is before "
                f"valid_from {valid_from.isoformat()}"
            )
        return valid_until


class WeatherClaim(_Contract):
    kind: Literal["weather"]
    rain: bool


class ConstraintClaim(_Contract):
    """Supports ``PlanningRequest.constraints[constraint_index]``."""

    kind: Literal["constraint"]
    constraint_index: Annotated[int, Field(ge=0, lt=MAX_CONSTRAINTS)]


EvidenceClaim = Annotated[
    OpeningAccessClaim
    | VisitDurationClaim
    | CostAmountClaim
    | TaxiRateClaim
    | WeatherClaim
    | ConstraintClaim,
    Field(discriminator="kind"),
]

_CLAIM_KINDS_BY_TOPIC: Final[Mapping[EvidenceTopic, frozenset[str]]] = {
    EvidenceTopic.OPENING_ACCESS: frozenset({"opening_access"}),
    EvidenceTopic.VISIT_DURATION: frozenset({"visit_duration"}),
    EvidenceTopic.COST: frozenset({"cost_amount", "taxi_rate"}),
    EvidenceTopic.WEATHER: frozenset({"weather"}),
    EvidenceTopic.CONSTRAINT: frozenset({"constraint"}),
}


def _require_claim_matches_topic(claim: EvidenceClaim, info: ValidationInfo) -> EvidenceClaim:
    """Reject a claim whose kind is not allowed for the evidence topic."""
    topic = info.data.get("topic")
    if isinstance(topic, EvidenceTopic) and claim.kind not in _CLAIM_KINDS_BY_TOPIC[topic]:
        allowed = sorted(_CLAIM_KINDS_BY_TOPIC[topic])
        raise ValueError(
            f"claim kind {claim.kind!r} does not match topic {topic.value!r}; allowed: {allowed}"
        )
    return claim


class SourcedFinding(_Contract):
    """A finding read from a cited source; ``finding`` is untrusted data, never instructions."""

    kind: Literal["sourced_finding"]
    evidence_id: Identifier
    topic: EvidenceTopic
    claim: EvidenceClaim
    source: SourceReference
    retrieved_at: AwareDatetime
    scope: EvidenceScope
    finding: _FreeText

    @field_validator("claim")
    @classmethod
    def _claim_matches_topic(cls, claim: EvidenceClaim, info: ValidationInfo) -> EvidenceClaim:
        return _require_claim_matches_topic(claim, info)


class Estimate(_Contract):
    """A labeled estimate derived from other evidence or, as AI best guess, from none."""

    kind: Literal["estimate"]
    evidence_id: Identifier
    topic: EvidenceTopic
    claim: EvidenceClaim
    label: EstimateLabel
    scope: EvidenceScope
    basis: _FreeText
    based_on: tuple[Identifier, ...]

    @field_validator("claim")
    @classmethod
    def _claim_matches_topic(cls, claim: EvidenceClaim, info: ValidationInfo) -> EvidenceClaim:
        return _require_claim_matches_topic(claim, info)


Evidence = Annotated[SourcedFinding | Estimate, Field(discriminator="kind")]


class ActivityStop(_Contract):
    place_id: PlaceId
    label: _LabelText
    latitude: _Latitude
    longitude: _Longitude
    visit_minutes: Annotated[int, Field(ge=1, le=_MAX_DAY_MINUTES)]
    evidence_ids: tuple[Identifier, ...]


class TransferWalk(_Contract):
    """One connection inside a leg: the minutes it allows versus the walk it needs."""

    available_minutes: Annotated[int, Field(ge=0, le=_MAX_DAY_MINUTES)]
    walk_minutes: Annotated[int, Field(ge=0, le=_MAX_DAY_MINUTES)]


class TravelLeg(_Contract):
    """One leg; ``duration_minutes`` includes waits and transfer walks."""

    mode: TransportMode
    duration_minutes: Annotated[int, Field(ge=0, le=_MAX_DAY_MINUTES)]
    distance_meters: Annotated[int, Field(ge=0)]
    transfers: Annotated[tuple[TransferWalk, ...], Field(max_length=MAX_TRANSFERS)]
    evidence_ids: tuple[Identifier, ...]


class CostItem(_Contract):
    description: _LabelText
    amount: Money
    evidence_ids: Annotated[tuple[Identifier, ...], Field(min_length=1)]


class CandidateItinerary(_Contract):
    """A model proposal, not facts; legs run start -> each stop -> commitment."""

    candidate_id: Identifier
    stops: Annotated[tuple[ActivityStop, ...], Field(min_length=1, max_length=MAX_STOPS)]
    legs: tuple[TravelLeg, ...]
    costs: tuple[CostItem, ...]

    @field_validator("legs")
    @classmethod
    def _legs_reach_commitment(
        cls, legs: tuple[TravelLeg, ...], info: ValidationInfo
    ) -> tuple[TravelLeg, ...]:
        stops = info.data.get("stops")
        if isinstance(stops, tuple):
            stop_count = len(cast(tuple[ActivityStop, ...], stops))
            if len(legs) != stop_count + 1:
                raise ValueError(
                    f"legs must have {stop_count + 1} entries for {stop_count} stops "
                    f"(start -> each stop -> commitment); got {len(legs)}"
                )
        return legs


class CheckedItinerary(_Contract):
    """Python-checked schedule and totals; ``leave_by[0]`` is the starting location."""

    candidate: CandidateItinerary
    leave_by: tuple[AwareDatetime, ...]
    arrival_at_commitment: AwareDatetime
    total_cost: Money
    total_includes_estimates: bool
    weather_warnings: tuple[_FreeText, ...]

    @field_validator("leave_by")
    @classmethod
    def _leave_by_per_departure(
        cls, leave_by: tuple[datetime, ...], info: ValidationInfo
    ) -> tuple[datetime, ...]:
        candidate = info.data.get("candidate")
        if isinstance(candidate, CandidateItinerary):
            stop_count = len(candidate.stops)
            if len(leave_by) != stop_count + 1:
                raise ValueError(
                    f"leave_by must have {stop_count + 1} entries for {stop_count} stops "
                    f"(starting location plus each stop); got {len(leave_by)}"
                )
        return leave_by


class ValidatedOption(_Contract):
    itinerary: CheckedItinerary
    labels: Annotated[tuple[OptionLabel, ...], Field(min_length=1), AfterValidator(_require_unique)]
    recommendation_reason: _FreeText | None


class Recommendation(_Contract):
    """The model's chosen candidate and its required reason; selection only consumes it."""

    candidate_id: Identifier
    reason: _FreeText


class ConstraintFailure(_Contract):
    """One failed hard constraint with the inputs that caused it, for model repair."""

    constraint: ConstraintId
    message: _FreeText
    inputs: Annotated[
        dict[_LabelText, _FreeText], Field(min_length=1, max_length=MAX_FEEDBACK_INPUTS)
    ]


class ValidationFeedback(_Contract):
    candidate_id: Identifier
    failures: Annotated[tuple[ConstraintFailure, ...], Field(min_length=1)]


class Clarification(_Contract):
    kind: Literal["clarification"]
    message: _FreeText
    questions: Annotated[
        tuple[UnresolvedQuestion, ...], Field(min_length=1, max_length=MAX_UNRESOLVED_QUESTIONS)
    ]


class ValidatedOptions(_Contract):
    kind: Literal["validated_options"]
    message: _FreeText
    options: Annotated[tuple[ValidatedOption, ...], Field(min_length=1, max_length=3)]


class NoFeasibleResult(_Contract):
    """No candidate passed; asks before retrying and never relaxes constraints."""

    kind: Literal["no_feasible_result"]
    message: _FreeText
    retry_question: _FreeText


class ExecutionFailure(_Contract):
    """The turn could not finish; this is not evidence that no itinerary exists."""

    kind: Literal["execution_failure"]
    message: _FreeText
    reason: ExecutionFailureReason


TurnResult = Annotated[
    Clarification | ValidatedOptions | NoFeasibleResult | ExecutionFailure,
    Field(discriminator="kind"),
]


class ChatMessage(_Contract):
    role: ChatRole
    text: _MessageText


class ContinuationPayload(_Contract):
    """Untrusted browser context; it carries no evidence and proves nothing was verified."""

    schema_version: _SchemaVersion
    messages: Annotated[tuple[ChatMessage, ...], Field(max_length=MAX_CONTINUATION_MESSAGES)]
    request: PlanningRequest

    @field_validator("messages")
    @classmethod
    def _total_text_is_bounded(cls, messages: tuple[ChatMessage, ...]) -> tuple[ChatMessage, ...]:
        total = sum(len(message.text) for message in messages)
        if total > MAX_CONTINUATION_TEXT_CHARS:
            raise ValueError(
                f"messages contain {total} characters in total; "
                f"limit is {MAX_CONTINUATION_TEXT_CHARS}"
            )
        return messages


class EvalCase(_Contract):
    """One scripted evaluation conversation; ``device_location`` None means unavailable."""

    case_id: Identifier
    requirement_ids: Annotated[tuple[RequirementId, ...], Field(min_length=1)]
    now: AwareDatetime
    device_location: Location | None
    user_messages: Annotated[tuple[_MessageText, ...], Field(min_length=1)]
    expected_outcome: TurnResultKind
    notes: _FreeText
