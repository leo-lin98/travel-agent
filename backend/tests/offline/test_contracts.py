"""Phase 0 acceptance tests for the typed boundary contracts.

Expectations derive from docs/requirements.md (R1-R17, R5 deferred), the Phase 0
document, and artifacts/phase-0/interface-contract.md. Every payload is validated
through JSON, as it would arrive at a system boundary.
"""

import json
from collections.abc import Callable
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import pytest
from pydantic import TypeAdapter, ValidationError

from travel_agent.contracts import (
    MAX_CONSTRAINT_CHARS,
    MAX_CONSTRAINTS,
    MAX_CONTINUATION_MESSAGES,
    MAX_CONTINUATION_TEXT_CHARS,
    MAX_LABEL_CHARS,
    MAX_MESSAGE_CHARS,
    MAX_OPEN_INTERVALS,
    MAX_STOPS,
    MAX_TEXT_CHARS,
    MAX_TRANSFERS,
    MAX_UNRESOLVED_QUESTIONS,
    CandidateItinerary,
    Clarification,
    ConstraintId,
    ContinuationPayload,
    Estimate,
    EstimateLabel,
    EvalCase,
    Evidence,
    ExecutionFailure,
    ExecutionFailureReason,
    NoFeasibleResult,
    OpeningAccessClaim,
    OpeningStatus,
    OptionLabel,
    PlanningRequest,
    Recommendation,
    SourcedFinding,
    SourceKind,
    TaxiRateClaim,
    TransportMode,
    TurnResult,
    TurnStatus,
    UnresolvedTopic,
    ValidatedOptions,
    ValidationFeedback,
    stop_key,
)

type JsonPath = tuple[str | int, ...]

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
TOKYO = ZoneInfo("Asia/Tokyo")
ARRIVAL_BUFFER = timedelta(minutes=15)

EVIDENCE_LIST = TypeAdapter(tuple[Evidence, ...])
TURN_RESULT = TypeAdapter(TurnResult)
EVAL_CASES = TypeAdapter(tuple[EvalCase, ...])

V1_REQUIREMENT_IDS = frozenset(
    {f"R{number}" for number in range(1, 18) if number != 5},
)
# User decision 2026-11-17: the dev eval set must cover R1-R4 and R6-R14 only.
EVAL_REQUIRED_REQUIREMENT_IDS = frozenset(
    {f"R{number}" for number in range(1, 15) if number != 5},
)
TURN_RESULT_KINDS = frozenset(
    {"clarification", "validated_options", "no_feasible_result", "execution_failure"},
)


# --- helpers (pure; never mutate their inputs) ---------------------------------


def read_text(name: str) -> str:
    """Return the raw text of a fixture file."""
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def read_fixture(name: str) -> object:
    """Return a fixture file parsed as plain JSON data."""
    return json.loads(read_text(name))


def to_json(document: object) -> str:
    """Serialize plain JSON data for boundary validation."""
    return json.dumps(document)


def replaced(document: object, path: JsonPath, value: object) -> object:
    """Return a copy of `document` with `value` set at `path` (adds a missing final key)."""
    if not path:
        return value
    head = path[0]
    tail = path[1:]
    if isinstance(document, dict) and isinstance(head, str):
        mapping = cast(dict[str, object], document)
        if not tail:
            return {**mapping, head: value}
        return {**mapping, head: replaced(mapping[head], tail, value)}
    if isinstance(document, list) and isinstance(head, int):
        items = cast(list[object], document)
        return [
            replaced(item, tail, value) if index == head else item
            for index, item in enumerate(items)
        ]
    raise TypeError(f"path segment {head!r} does not match the fixture document shape")


def without(document: object, path: JsonPath) -> object:
    """Return a copy of `document` with the key at `path` removed."""
    head = path[0]
    tail = path[1:]
    if isinstance(document, dict) and isinstance(head, str):
        mapping = cast(dict[str, object], document)
        if not tail:
            return {key: item for key, item in mapping.items() if key != head}
        return {**mapping, head: without(mapping[head], tail)}
    if isinstance(document, list) and isinstance(head, int):
        items = cast(list[object], document)
        return [without(item, tail) if index == head else item for index, item in enumerate(items)]
    raise TypeError(f"path segment {head!r} does not match the fixture document shape")


def value_at(document: object, path: JsonPath) -> object:
    """Return the value stored at `path`."""
    if not path:
        return document
    head = path[0]
    if isinstance(document, dict) and isinstance(head, str):
        return value_at(cast(dict[str, object], document)[head], path[1:])
    if isinstance(document, list) and isinstance(head, int):
        return value_at(cast(list[object], document)[head], path[1:])
    raise TypeError(f"path segment {head!r} does not match the fixture document shape")


def error_fields(error: ValidationError) -> set[str | int]:
    """Return every location segment named by a validation error."""
    return {segment for detail in error.errors() for segment in detail["loc"]}


def messages_at(error: ValidationError, field: str) -> list[str]:
    """Return the messages of errors whose location ends with `field`."""
    return [
        detail["msg"] for detail in error.errors() if detail["loc"] and detail["loc"][-1] == field
    ]


def chat_messages(count: int, length: int) -> list[object]:
    """Return `count` user messages whose text is `length` characters long."""
    return [{"role": "user", "text": "x" * length} for _ in range(count)]


def valid_eval_case(case_id: str, requirement_ids: list[str]) -> dict[str, object]:
    """Return a minimal valid eval case document."""
    return {
        "case_id": case_id,
        "requirement_ids": requirement_ids,
        "now": "2026-11-20T14:30:00+09:00",
        "device_location": None,
        "user_messages": ["I'm at Shibuya station with dinner in Ginza at 7pm."],
        "expected_outcome": "validated_options",
        "notes": "Synthetic case.",
    }


# --- planning request (R1, R7, R13) ---------------------------------------------


def test_planning_request_separates_gap_from_commitment() -> None:
    request = PlanningRequest.model_validate_json(read_text("planning_request.json"))

    assert request.gap.start_time == datetime(2026, 11, 20, 15, 45, tzinfo=TOKYO)
    assert request.gap.start_location is not None
    assert request.commitment.time == datetime(2026, 11, 20, 19, 0, tzinfo=TOKYO)
    assert request.commitment.location is not None
    assert request.commitment.timezone == "Asia/Tokyo"
    assert request.budget is not None
    assert request.budget.amount == Decimal("3000")
    assert request.budget.currency == "JPY"
    assert request.transport_modes == (TransportMode.WALKING, TransportMode.PUBLIC_TRANSIT)
    assert request.constraints == ("Include one garden or park", "A coffee stop before dinner")


def test_missing_essentials_stay_unresolved_instead_of_invented() -> None:
    request = PlanningRequest.model_validate_json(read_text("planning_request_unresolved.json"))

    assert request.commitment.time is None
    assert request.budget is not None
    assert request.budget.amount == Decimal("5000")
    assert request.budget.currency is None
    assert {question.topic for question in request.unresolved_questions} == {
        UnresolvedTopic.COMMITMENT_TIME,
        UnresolvedTopic.BUDGET_CURRENCY,
    }


@pytest.mark.parametrize(
    "path",
    [
        ("gap", "start_time"),
        ("gap", "start_location"),
        ("commitment", "location"),
        ("commitment", "time"),
        ("commitment", "timezone"),
    ],
)
def test_each_essential_can_be_unknown(path: JsonPath) -> None:
    document = replaced(read_fixture("planning_request.json"), path, None)

    request = PlanningRequest.model_validate_json(to_json(document))

    assert value_at(request.model_dump(mode="json"), path) is None


def test_no_stated_budget_means_no_cap() -> None:
    document = replaced(read_fixture("planning_request.json"), ("budget",), None)

    assert PlanningRequest.model_validate_json(to_json(document)).budget is None


@pytest.mark.parametrize("path", [("budget",), ("unresolved_questions",), ("gap", "start_time")])
def test_request_fields_have_no_silent_defaults(path: JsonPath) -> None:
    document = without(read_fixture("planning_request.json"), path)

    with pytest.raises(ValidationError) as exc_info:
        PlanningRequest.model_validate_json(to_json(document))

    assert path[-1] in error_fields(exc_info.value)


@pytest.mark.parametrize("path", [("gap", "start_time"), ("commitment", "time")])
def test_naive_request_datetimes_are_rejected(path: JsonPath) -> None:
    document = replaced(read_fixture("planning_request.json"), path, "2026-11-20T19:00:00")

    with pytest.raises(ValidationError):
        PlanningRequest.model_validate_json(to_json(document))


@pytest.mark.parametrize("timezone", ["UTC+9", "Mars/Olympus", "", "tokyo"])
def test_commitment_timezone_must_be_iana(timezone: str) -> None:
    document = replaced(read_fixture("planning_request.json"), ("commitment", "timezone"), timezone)

    with pytest.raises(ValidationError):
        PlanningRequest.model_validate_json(to_json(document))


@pytest.mark.parametrize("timezone", ["asia/tokyo", "ASIA/TOKYO", "Asia/tokyo"])
def test_commitment_timezone_is_case_sensitive(timezone: str) -> None:
    """R1: the zone must resolve identically on every OS, including case-sensitive Linux."""
    path: JsonPath = ("commitment", "timezone")
    document = replaced(read_fixture("planning_request.json"), path, timezone)

    with pytest.raises(ValidationError) as exc_info:
        PlanningRequest.model_validate_json(to_json(document))

    assert "timezone" in error_fields(exc_info.value)
    assert timezone in str(exc_info.value)


def test_commitment_timezone_accepts_exact_iana_name() -> None:
    request = PlanningRequest.model_validate_json(read_text("planning_request.json"))

    assert request.commitment.timezone == "Asia/Tokyo"


def test_commitment_timezone_accepts_other_iana_zones() -> None:
    document = replaced(
        read_fixture("planning_request.json"), ("commitment", "timezone"), "America/New_York"
    )

    request = PlanningRequest.model_validate_json(to_json(document))

    assert request.commitment.timezone == "America/New_York"


@pytest.mark.parametrize("mode", ["car", "bicycle", "Walking"])
def test_unsupported_transport_mode_is_rejected(mode: str) -> None:
    document = replaced(read_fixture("planning_request.json"), ("transport_modes",), [mode])

    with pytest.raises(ValidationError):
        PlanningRequest.model_validate_json(to_json(document))


def test_all_three_transport_choices_are_supported() -> None:
    modes = ["walking", "public_transit", "taxi_rideshare"]
    document = replaced(read_fixture("planning_request.json"), ("transport_modes",), modes)

    request = PlanningRequest.model_validate_json(to_json(document))

    assert {mode.value for mode in request.transport_modes} == set(modes)
    assert {mode.value for mode in TransportMode} == set(modes)


def test_duplicate_transport_modes_are_rejected() -> None:
    document = replaced(
        read_fixture("planning_request.json"), ("transport_modes",), ["walking", "walking"]
    )

    with pytest.raises(ValidationError):
        PlanningRequest.model_validate_json(to_json(document))


# --- schema version, strictness, immutability -----------------------------------


@pytest.mark.parametrize("fixture", ["planning_request.json", "continuation_payload.json"])
@pytest.mark.parametrize("version", [0, 2, "1", 1.5, None])
def test_unsupported_schema_version_is_rejected_actionably(fixture: str, version: object) -> None:
    document = replaced(read_fixture(fixture), ("schema_version",), version)
    model = PlanningRequest if fixture == "planning_request.json" else ContinuationPayload

    with pytest.raises(ValidationError) as exc_info:
        model.model_validate_json(to_json(document))

    assert "schema_version" in error_fields(exc_info.value)
    assert "schema_version" in str(exc_info.value)


@pytest.mark.parametrize("fixture", ["planning_request.json", "continuation_payload.json"])
def test_missing_schema_version_is_rejected(fixture: str) -> None:
    document = without(read_fixture(fixture), ("schema_version",))
    model = PlanningRequest if fixture == "planning_request.json" else ContinuationPayload

    with pytest.raises(ValidationError) as exc_info:
        model.model_validate_json(to_json(document))

    assert "schema_version" in error_fields(exc_info.value)


@pytest.mark.parametrize(
    ("fixture", "path"),
    [
        ("planning_request.json", ("assumed_currency",)),
        ("planning_request.json", ("commitment", "verified")),
        ("planning_request.json", ("gap", "start_location", "country")),
        ("candidate_itinerary.json", ("stops", 0, "opening_verified")),
        ("candidate_itinerary.json", ("legs", 0, "feasible")),
        ("validation_feedback.json", ("passed",)),
        ("turn_result_execution_failure.json", ("debug",)),
    ],
)
def test_unknown_fields_are_rejected(fixture: str, path: JsonPath) -> None:
    document = replaced(read_fixture(fixture), path, True)
    validators: dict[str, Callable[[str], object]] = {
        "planning_request.json": PlanningRequest.model_validate_json,
        "candidate_itinerary.json": CandidateItinerary.model_validate_json,
        "validation_feedback.json": ValidationFeedback.model_validate_json,
        "turn_result_execution_failure.json": TURN_RESULT.validate_json,
    }

    with pytest.raises(ValidationError):
        validators[fixture](to_json(document))


def test_numeric_strings_are_not_coerced() -> None:
    path: JsonPath = ("stops", 0, "visit_minutes")
    document = replaced(read_fixture("candidate_itinerary.json"), path, "60")

    with pytest.raises(ValidationError):
        CandidateItinerary.model_validate_json(to_json(document))


@pytest.mark.parametrize("attribute", ["constraints", "budget", "gap"])
def test_contract_models_are_immutable(attribute: str) -> None:
    request = PlanningRequest.model_validate_json(read_text("planning_request.json"))

    with pytest.raises(ValidationError):
        setattr(request, attribute, None)


# --- locations, place IDs, money (R2, R12, R13) ---------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [("latitude", 90.0001), ("latitude", -91), ("longitude", 180.5), ("longitude", -181)],
)
def test_out_of_range_coordinates_are_rejected(field: str, value: float) -> None:
    location_path: JsonPath = ("gap", "start_location", field)
    stop_path: JsonPath = ("stops", 0, field)
    request = replaced(read_fixture("planning_request.json"), location_path, value)
    candidate = replaced(read_fixture("candidate_itinerary.json"), stop_path, value)

    with pytest.raises(ValidationError):
        PlanningRequest.model_validate_json(to_json(request))
    with pytest.raises(ValidationError):
        CandidateItinerary.model_validate_json(to_json(candidate))


@pytest.mark.parametrize(("latitude", "longitude"), [(90, 180), (-90, -180), (0, 0)])
def test_boundary_coordinates_are_accepted(latitude: float, longitude: float) -> None:
    template = read_fixture("planning_request.json")
    document = replaced(template, ("gap", "start_location", "latitude"), latitude)
    document = replaced(document, ("gap", "start_location", "longitude"), longitude)

    request = PlanningRequest.model_validate_json(to_json(document))

    assert request.gap.start_location is not None
    assert request.gap.start_location.latitude == latitude


def test_activity_stop_requires_place_id_key() -> None:
    document = without(read_fixture("candidate_itinerary.json"), ("stops", 0, "place_id"))

    with pytest.raises(ValidationError) as exc_info:
        CandidateItinerary.model_validate_json(to_json(document))

    assert "place_id" in error_fields(exc_info.value)


@pytest.mark.parametrize("place_id", [None, "", "has spaces", "semi;colon"])
def test_activity_stop_place_id_must_be_valid(place_id: object) -> None:
    path: JsonPath = ("stops", 1, "place_id")
    document = replaced(read_fixture("candidate_itinerary.json"), path, place_id)

    with pytest.raises(ValidationError):
        CandidateItinerary.model_validate_json(to_json(document))


@pytest.mark.parametrize("amount", ["-1", "-0.01", "abc", "NaN", "Infinity"])
def test_cost_amount_must_be_finite_and_non_negative(amount: str) -> None:
    document = replaced(
        read_fixture("candidate_itinerary.json"), ("costs", 0, "amount", "amount"), amount
    )

    with pytest.raises(ValidationError):
        CandidateItinerary.model_validate_json(to_json(document))


def test_zero_cost_is_allowed() -> None:
    path: JsonPath = ("costs", 0, "amount", "amount")
    document = replaced(read_fixture("candidate_itinerary.json"), path, "0")

    candidate = CandidateItinerary.model_validate_json(to_json(document))

    assert candidate.costs[0].amount.amount == Decimal("0")


@pytest.mark.parametrize("currency", ["jpy", "JP", "JPYY", "¥", "", None])
def test_cost_currency_must_be_iso_4217_shaped(currency: object) -> None:
    document = replaced(
        read_fixture("candidate_itinerary.json"), ("costs", 0, "amount", "currency"), currency
    )

    with pytest.raises(ValidationError):
        CandidateItinerary.model_validate_json(to_json(document))


@pytest.mark.parametrize(
    ("path", "value"),
    [(("budget", "amount"), "-5"), (("budget", "currency"), "yen"), (("budget", "currency"), "US")],
)
def test_invalid_budget_is_rejected(path: JsonPath, value: str) -> None:
    document = replaced(read_fixture("planning_request.json"), path, value)

    with pytest.raises(ValidationError):
        PlanningRequest.model_validate_json(to_json(document))


# --- evidence (R9, R10, R13) ----------------------------------------------------


def test_evidence_fixture_distinguishes_sourced_findings_from_estimates() -> None:
    evidence = EVIDENCE_LIST.validate_json(read_text("evidence.json"))

    sourced = [item for item in evidence if isinstance(item, SourcedFinding)]
    estimates = [item for item in evidence if isinstance(item, Estimate)]
    assert len(sourced) + len(estimates) == len(evidence) == 11
    assert {item.label for item in estimates} == {
        EstimateLabel.ESTIMATE,
        EstimateLabel.AI_BEST_GUESS,
    }
    assert {item.source.kind for item in sourced} == {
        SourceKind.OFFICIAL_WEBSITE,
        SourceKind.OTHER_WEB,
        SourceKind.GOOGLE_MAPS,
    }
    retrieved_at = datetime(2026, 11, 20, 14, 30, tzinfo=TOKYO)
    assert all(item.retrieved_at == retrieved_at for item in sourced)


def test_source_kinds_cover_official_sources_and_maps_fallback() -> None:
    assert {kind.value for kind in SourceKind} == {
        "official_website",
        "official_social_media",
        "official_authority_notice",
        "google_maps",
        "other_web",
    }


def test_estimate_labels_include_ai_best_guess() -> None:
    assert {label.value for label in EstimateLabel} == {"estimate", "ai_best_guess"}


def test_untrusted_source_text_is_kept_as_plain_data() -> None:
    evidence = EVIDENCE_LIST.validate_json(read_text("evidence.json"))

    duration = next(item for item in evidence if item.evidence_id == "ev-garden-duration")

    assert isinstance(duration, SourcedFinding)
    assert "Ignore previous instructions" in duration.finding


@pytest.mark.parametrize(
    ("index", "path", "value"),
    [
        (0, ("source",), None),
        (0, ("retrieved_at",), "2026-11-20T14:30:00"),
        (0, ("scope", "applies_from"), "2026-11-20T09:00:00"),
        (0, ("scope", "applies_from"), "2026-11-21T00:30:00+09:00"),
        (0, ("source", "url"), "javascript:alert(1)"),
        (0, ("source", "url"), "ftp://garden.example.org/visit"),
        (0, ("source", "kind"), "blog"),
        (0, ("topic",), "vibes"),
        (0, ("kind",), "fact"),
        (2, ("label",), "guess"),
        (2, ("source",), {"url": "https://x.example", "kind": "other_web", "title": None}),
        (2, ("verified",), True),
    ],
)
def test_malformed_evidence_is_rejected(index: int, path: JsonPath, value: object) -> None:
    document = replaced(read_fixture("evidence.json"), (index, *path), value)

    with pytest.raises(ValidationError):
        EVIDENCE_LIST.validate_json(to_json(document))


def test_sourced_finding_requires_a_source() -> None:
    document = without(read_fixture("evidence.json"), (0, "source"))

    with pytest.raises(ValidationError) as exc_info:
        EVIDENCE_LIST.validate_json(to_json(document))

    assert "source" in error_fields(exc_info.value)


# --- candidate itinerary (R2, R9, R10, R13) ------------------------------------


def test_candidate_itinerary_fixture_loads() -> None:
    candidate = CandidateItinerary.model_validate_json(read_text("candidate_itinerary.json"))

    assert [stop.place_id for stop in candidate.stops] == [
        "ChIJsyntheticGardenMuseum0001",
        "ChIJsyntheticKissaCafe00001",
    ]
    assert len(candidate.legs) == len(candidate.stops) + 1
    assert candidate.legs[-1].mode == TransportMode.PUBLIC_TRANSIT


def test_candidate_must_include_travel_to_the_commitment() -> None:
    document = read_fixture("candidate_itinerary.json")
    legs = cast(list[object], value_at(document, ("legs",)))
    document = replaced(document, ("legs",), legs[:-1])

    with pytest.raises(ValidationError):
        CandidateItinerary.model_validate_json(to_json(document))


def test_candidate_requires_at_least_one_stop() -> None:
    document = replaced(read_fixture("candidate_itinerary.json"), ("stops",), [])
    document = replaced(document, ("legs",), [value_at(document, ("legs", 0))])

    with pytest.raises(ValidationError):
        CandidateItinerary.model_validate_json(to_json(document))


def test_candidate_stop_count_is_bounded() -> None:
    template = read_fixture("candidate_itinerary.json")
    stop = value_at(template, ("stops", 0))
    leg = value_at(template, ("legs", 0))
    document = replaced(template, ("stops",), [stop] * (MAX_STOPS + 1))
    document = replaced(document, ("legs",), [leg] * (MAX_STOPS + 2))

    with pytest.raises(ValidationError):
        CandidateItinerary.model_validate_json(to_json(document))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("stops", 0, "visit_minutes"), 0),
        (("stops", 0, "visit_minutes"), 1441),
        (("legs", 0, "duration_minutes"), -1),
        (("legs", 0, "distance_meters"), -1),
        (("legs", 0, "mode"), "car"),
        (("costs", 0, "evidence_ids"), []),
    ],
)
def test_invalid_candidate_values_are_rejected(path: JsonPath, value: object) -> None:
    document = replaced(read_fixture("candidate_itinerary.json"), path, value)

    with pytest.raises(ValidationError):
        CandidateItinerary.model_validate_json(to_json(document))


def test_fixture_evidence_references_resolve() -> None:
    candidate = CandidateItinerary.model_validate_json(read_text("candidate_itinerary.json"))
    evidence = EVIDENCE_LIST.validate_json(read_text("evidence.json"))
    evidence_ids = {item.evidence_id for item in evidence}

    referenced = (
        {ref for stop in candidate.stops for ref in stop.evidence_ids}
        | {ref for leg in candidate.legs for ref in leg.evidence_ids}
        | {ref for cost in candidate.costs for ref in cost.evidence_ids}
    )

    assert referenced <= evidence_ids


# --- validation feedback --------------------------------------------------------


def test_validation_feedback_names_failed_constraints_and_inputs() -> None:
    feedback = ValidationFeedback.model_validate_json(read_text("validation_feedback.json"))

    assert [failure.constraint for failure in feedback.failures] == [
        ConstraintId.ARRIVAL_BUFFER,
        ConstraintId.TRANSFER_TIME,
    ]
    assert feedback.failures[0].inputs["latest_allowed_arrival"] == "2026-11-20T18:45:00+09:00"
    assert all(failure.message for failure in feedback.failures)


def test_feedback_inputs_cannot_be_changed_after_validation() -> None:
    feedback = ValidationFeedback.model_validate_json(read_text("validation_feedback.json"))
    inputs = feedback.failures[0].inputs
    # The cast reaches past the read-only type to try what a careless caller could do.
    writable = cast(dict[str, str], inputs)

    with pytest.raises(TypeError):
        writable["latest_allowed_arrival"] = "tampered"

    assert inputs["latest_allowed_arrival"] == "2026-11-20T18:45:00+09:00"


def test_feedback_inputs_serialise_as_a_json_object() -> None:
    feedback = ValidationFeedback.model_validate_json(read_text("validation_feedback.json"))

    dumped = json.loads(feedback.model_dump_json())

    assert dumped == read_fixture("validation_feedback.json")
    assert ValidationFeedback.model_validate_json(feedback.model_dump_json()) == feedback


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("failures",), []),
        (("failures", 0, "inputs"), {}),
        (("failures", 0, "constraint"), "looks_bad"),
        (("failures", 0, "message"), ""),
    ],
)
def test_unactionable_feedback_is_rejected(path: JsonPath, value: object) -> None:
    document = replaced(read_fixture("validation_feedback.json"), path, value)

    with pytest.raises(ValidationError):
        ValidationFeedback.model_validate_json(to_json(document))


# --- turn status and turn result (R2, R3, R14) ----------------------------------


def test_turn_status_values_are_exact() -> None:
    assert [status.value for status in TurnStatus] == [
        "Researching",
        "Planning",
        "Generating itinerary",
    ]


@pytest.mark.parametrize(
    ("fixture", "expected_type"),
    [
        ("turn_result_clarification.json", Clarification),
        ("turn_result_validated_options.json", ValidatedOptions),
        ("turn_result_no_feasible_result.json", NoFeasibleResult),
        ("turn_result_execution_failure.json", ExecutionFailure),
    ],
)
def test_each_turn_result_kind_loads_as_its_own_type(
    fixture: str, expected_type: type[object]
) -> None:
    result = TURN_RESULT.validate_json(read_text(fixture))

    assert isinstance(result, expected_type)


def test_turn_result_is_discriminated_by_exactly_four_kinds() -> None:
    schema = TURN_RESULT.json_schema()

    assert schema["discriminator"]["propertyName"] == "kind"
    assert set(schema["discriminator"]["mapping"]) == TURN_RESULT_KINDS


@pytest.mark.parametrize("kind", ["partial_options", "relaxed_options", "", None])
def test_unknown_turn_result_kind_is_rejected(kind: object) -> None:
    document = replaced(read_fixture("turn_result_no_feasible_result.json"), ("kind",), kind)

    with pytest.raises(ValidationError):
        TURN_RESULT.validate_json(to_json(document))


def test_turn_result_fields_cannot_mix_kinds() -> None:
    document = replaced(
        read_fixture("turn_result_execution_failure.json"), ("retry_question",), "Try again?"
    )

    with pytest.raises(ValidationError):
        TURN_RESULT.validate_json(to_json(document))


@pytest.mark.parametrize("count", [0, 4])
def test_validated_options_hold_one_to_three_options(count: int) -> None:
    document = read_fixture("turn_result_validated_options.json")
    option = value_at(document, ("options", 0))
    document = replaced(document, ("options",), [option] * count)

    with pytest.raises(ValidationError):
        TURN_RESULT.validate_json(to_json(document))


@pytest.mark.parametrize("labels", [[], ["cheapest", "cheapest"], ["fastest"]])
def test_option_labels_are_known_and_unique(labels: list[str]) -> None:
    path: JsonPath = ("options", 0, "labels")
    document = replaced(read_fixture("turn_result_validated_options.json"), path, labels)

    with pytest.raises(ValidationError):
        TURN_RESULT.validate_json(to_json(document))


def test_option_label_values() -> None:
    assert {label.value for label in OptionLabel} == {"ai_recommended", "cheapest", "best_travel"}


def test_checked_itinerary_needs_a_leave_by_time_per_stop_and_start() -> None:
    path: JsonPath = ("options", 0, "itinerary", "leave_by")
    document = read_fixture("turn_result_validated_options.json")
    leave_by = cast(list[object], value_at(document, path))
    document = replaced(document, path, leave_by[1:])

    with pytest.raises(ValidationError):
        TURN_RESULT.validate_json(to_json(document))


@pytest.mark.parametrize(
    "path",
    [
        ("options", 0, "itinerary", "arrival_at_commitment"),
        ("options", 0, "itinerary", "leave_by", 0),
    ],
)
def test_checked_times_must_be_timezone_aware(path: JsonPath) -> None:
    template = read_fixture("turn_result_validated_options.json")
    document = replaced(template, path, "2026-11-20T18:45:00")

    with pytest.raises(ValidationError):
        TURN_RESULT.validate_json(to_json(document))


@pytest.mark.parametrize("retry_question", ["", None])
def test_no_feasible_result_must_ask_before_retrying(retry_question: object) -> None:
    document = replaced(
        read_fixture("turn_result_no_feasible_result.json"), ("retry_question",), retry_question
    )

    with pytest.raises(ValidationError):
        TURN_RESULT.validate_json(to_json(document))


def test_clarification_requires_a_question() -> None:
    document = replaced(read_fixture("turn_result_clarification.json"), ("questions",), [])

    with pytest.raises(ValidationError):
        TURN_RESULT.validate_json(to_json(document))


def test_execution_failure_reasons_are_distinct_from_no_feasible_result() -> None:
    assert {reason.value for reason in ExecutionFailureReason} == {
        "provider_failure",
        "execution_limit_reached",
        "internal_error",
    }


def test_checked_fixture_schedule_and_totals_are_consistent() -> None:
    request = PlanningRequest.model_validate_json(read_text("planning_request.json"))
    result = TURN_RESULT.validate_json(read_text("turn_result_validated_options.json"))
    candidate = CandidateItinerary.model_validate_json(read_text("candidate_itinerary.json"))
    assert isinstance(result, ValidatedOptions)
    itinerary = result.options[0].itinerary
    stops = itinerary.candidate.stops
    legs = itinerary.candidate.legs

    for index, stop in enumerate(stops):
        arrival = itinerary.leave_by[index] + timedelta(minutes=legs[index].duration_minutes)
        assert arrival + timedelta(minutes=stop.visit_minutes) == itinerary.leave_by[index + 1]
    final_arrival = itinerary.leave_by[-1] + timedelta(minutes=legs[-1].duration_minutes)

    assert itinerary.candidate == candidate
    assert itinerary.leave_by[0] == request.gap.start_time
    assert final_arrival == itinerary.arrival_at_commitment
    assert request.commitment.time is not None
    assert itinerary.arrival_at_commitment == request.commitment.time - ARRIVAL_BUFFER
    # Cost items plus the 180 metro fare the transit leg cites (evidence.json ev-transit-fare).
    assert itinerary.total_cost.amount == sum(
        (cost.amount.amount for cost in itinerary.candidate.costs), Decimal(0)
    ) + Decimal("180")
    assert request.budget is not None
    assert itinerary.total_cost.amount <= request.budget.amount
    assert itinerary.total_includes_estimates


# --- continuation payload (R11, R16) --------------------------------------------


def test_continuation_payload_fixture_loads() -> None:
    payload = ContinuationPayload.model_validate_json(read_text("continuation_payload.json"))
    request = PlanningRequest.model_validate_json(read_text("planning_request.json"))

    assert len(payload.messages) == 3
    assert payload.request == request


@pytest.mark.parametrize(
    "path",
    [
        ("verified",),
        ("candidates",),
        ("evidence",),
        ("checked_options",),
        ("messages", 0, "verified"),
        ("request", "verified"),
        ("request", "commitment", "location", "verified"),
    ],
)
def test_continuation_cannot_assert_verification(path: JsonPath) -> None:
    document = replaced(read_fixture("continuation_payload.json"), path, True)

    with pytest.raises(ValidationError):
        ContinuationPayload.model_validate_json(to_json(document))


@pytest.mark.parametrize("role", ["system", "tool", ""])
def test_continuation_message_roles_are_user_or_assistant(role: str) -> None:
    document = replaced(read_fixture("continuation_payload.json"), ("messages", 0, "role"), role)

    with pytest.raises(ValidationError):
        ContinuationPayload.model_validate_json(to_json(document))


def test_continuation_message_count_is_bounded() -> None:
    template = read_fixture("continuation_payload.json")
    at_limit = replaced(template, ("messages",), chat_messages(MAX_CONTINUATION_MESSAGES, 1))
    over_limit = replaced(template, ("messages",), chat_messages(MAX_CONTINUATION_MESSAGES + 1, 1))

    accepted = ContinuationPayload.model_validate_json(to_json(at_limit))
    assert len(accepted.messages) == MAX_CONTINUATION_MESSAGES
    with pytest.raises(ValidationError):
        ContinuationPayload.model_validate_json(to_json(over_limit))


def test_continuation_message_at_max_length_is_accepted() -> None:
    messages = chat_messages(1, MAX_MESSAGE_CHARS)
    document = replaced(read_fixture("continuation_payload.json"), ("messages",), messages)

    payload = ContinuationPayload.model_validate_json(to_json(document))

    assert len(payload.messages[0].text) == MAX_MESSAGE_CHARS


@pytest.mark.parametrize("length", [MAX_MESSAGE_CHARS + 1, 0])
def test_continuation_message_length_is_bounded(length: int) -> None:
    messages = chat_messages(1, length)
    document = replaced(read_fixture("continuation_payload.json"), ("messages",), messages)

    with pytest.raises(ValidationError):
        ContinuationPayload.model_validate_json(to_json(document))


def test_continuation_total_text_is_bounded() -> None:
    full_messages = MAX_CONTINUATION_TEXT_CHARS // MAX_MESSAGE_CHARS
    remainder = MAX_CONTINUATION_TEXT_CHARS - full_messages * MAX_MESSAGE_CHARS
    at_limit = chat_messages(full_messages, MAX_MESSAGE_CHARS) + (
        chat_messages(1, remainder) if remainder else []
    )
    over_limit = [*at_limit, *chat_messages(1, 1)]
    template = read_fixture("continuation_payload.json")
    assert len(over_limit) <= MAX_CONTINUATION_MESSAGES

    ContinuationPayload.model_validate_json(to_json(replaced(template, ("messages",), at_limit)))
    with pytest.raises(ValidationError):
        ContinuationPayload.model_validate_json(
            to_json(replaced(template, ("messages",), over_limit))
        )


# --- request size bounds --------------------------------------------------------


def test_constraint_count_is_bounded() -> None:
    template = read_fixture("planning_request.json")
    at_limit = replaced(template, ("constraints",), ["quiet"] * MAX_CONSTRAINTS)
    over_limit = replaced(template, ("constraints",), ["quiet"] * (MAX_CONSTRAINTS + 1))

    accepted = PlanningRequest.model_validate_json(to_json(at_limit))
    assert len(accepted.constraints) == MAX_CONSTRAINTS
    with pytest.raises(ValidationError):
        PlanningRequest.model_validate_json(to_json(over_limit))


def test_constraint_length_is_bounded() -> None:
    template = read_fixture("planning_request.json")
    at_limit = replaced(template, ("constraints",), ["x" * MAX_CONSTRAINT_CHARS])
    over_limit = replaced(template, ("constraints",), ["x" * (MAX_CONSTRAINT_CHARS + 1)])

    PlanningRequest.model_validate_json(to_json(at_limit))
    with pytest.raises(ValidationError):
        PlanningRequest.model_validate_json(to_json(over_limit))


def test_unresolved_question_count_is_bounded() -> None:
    question = {"topic": "other", "question": "Which museum did you mean?"}
    document = replaced(
        read_fixture("planning_request.json"),
        ("unresolved_questions",),
        [question] * (MAX_UNRESOLVED_QUESTIONS + 1),
    )

    with pytest.raises(ValidationError):
        PlanningRequest.model_validate_json(to_json(document))


# --- evaluation cases (Phase 0 deliverable; files written by T3) -----------------


def test_eval_case_contract_accepts_a_valid_case() -> None:
    cases = EVAL_CASES.validate_json(to_json([valid_eval_case("dev-001", ["R1", "R8"])]))

    assert cases[0].case_id == "dev-001"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("requirement_ids",), ["R5"]),
        (("requirement_ids",), ["R18"]),
        (("requirement_ids",), []),
        (("expected_outcome",), "partial_options"),
        (("user_messages",), []),
        (("now",), "2026-11-20T14:30:00"),
        (("case_id",), ""),
    ],
)
def test_invalid_eval_case_is_rejected(path: JsonPath, value: object) -> None:
    document = replaced(valid_eval_case("dev-001", ["R1"]), path, value)

    with pytest.raises(ValidationError):
        EVAL_CASES.validate_json(to_json([document]))


@pytest.mark.parametrize("fixture", ["eval_cases.json", "live_cases.json"])
def test_eval_case_file_is_valid_with_unique_ids(fixture: str) -> None:
    cases = EVAL_CASES.validate_json(read_text(fixture))
    case_ids = [case.case_id for case in cases]

    assert cases
    assert len(case_ids) == len(set(case_ids))
    assert {case.expected_outcome.value for case in cases} <= TURN_RESULT_KINDS


def test_dev_and_held_out_eval_sets_are_disjoint() -> None:
    dev_ids = {case.case_id for case in EVAL_CASES.validate_json(read_text("eval_cases.json"))}
    held_out_ids = {case.case_id for case in EVAL_CASES.validate_json(read_text("live_cases.json"))}

    assert dev_ids.isdisjoint(held_out_ids)


def test_dev_eval_set_covers_every_conversational_requirement() -> None:
    """R1-R4 and R6-R14 must be covered; R15-R17 may be tagged but are not required."""
    cases = EVAL_CASES.validate_json(read_text("eval_cases.json"))

    covered = {requirement for case in cases for requirement in case.requirement_ids}

    assert EVAL_REQUIRED_REQUIREMENT_IDS <= covered
    assert covered <= V1_REQUIREMENT_IDS


def test_eval_case_may_tag_infrastructure_requirements() -> None:
    document = valid_eval_case("dev-002", ["R1", "R15", "R16", "R17"])

    cases = EVAL_CASES.validate_json(to_json([document]))

    assert set(cases[0].requirement_ids) == {"R1", "R15", "R16", "R17"}


# --- error locations and actionable messages for cross-field rules --------------


def test_short_legs_error_points_at_legs() -> None:
    document = read_fixture("candidate_itinerary.json")
    legs = cast(list[object], value_at(document, ("legs",)))
    document = replaced(document, ("legs",), legs[:-1])

    with pytest.raises(ValidationError) as exc_info:
        CandidateItinerary.model_validate_json(to_json(document))

    assert messages_at(exc_info.value, "legs")


def test_leg_count_error_names_the_counts() -> None:
    document = read_fixture("candidate_itinerary.json")
    leg = value_at(document, ("legs", 0))
    document = replaced(document, ("legs",), [leg] * 5)

    with pytest.raises(ValidationError) as exc_info:
        CandidateItinerary.model_validate_json(to_json(document))

    messages = messages_at(exc_info.value, "legs")
    assert any("5" in message and "3" in message for message in messages)


def test_duplicate_transport_mode_error_names_the_mode() -> None:
    document = replaced(
        read_fixture("planning_request.json"), ("transport_modes",), ["walking", "walking"]
    )

    with pytest.raises(ValidationError) as exc_info:
        PlanningRequest.model_validate_json(to_json(document))

    assert any("walking" in message for message in messages_at(exc_info.value, "transport_modes"))


def test_continuation_total_text_error_points_at_messages() -> None:
    full_messages = MAX_CONTINUATION_TEXT_CHARS // MAX_MESSAGE_CHARS
    over_limit = [*chat_messages(full_messages, MAX_MESSAGE_CHARS), *chat_messages(1, 1)]
    document = replaced(read_fixture("continuation_payload.json"), ("messages",), over_limit)

    with pytest.raises(ValidationError) as exc_info:
        ContinuationPayload.model_validate_json(to_json(document))

    assert messages_at(exc_info.value, "messages")


def test_leave_by_count_error_points_at_leave_by() -> None:
    path: JsonPath = ("options", 0, "itinerary", "leave_by")
    document = read_fixture("turn_result_validated_options.json")
    leave_by = cast(list[object], value_at(document, path))
    document = replaced(document, path, leave_by[1:])

    with pytest.raises(ValidationError) as exc_info:
        TURN_RESULT.validate_json(to_json(document))

    assert messages_at(exc_info.value, "leave_by")


def test_reversed_scope_window_error_points_at_applies_until() -> None:
    path: JsonPath = (0, "scope", "applies_from")
    document = replaced(read_fixture("evidence.json"), path, "2026-11-21T00:30:00+09:00")

    with pytest.raises(ValidationError) as exc_info:
        EVIDENCE_LIST.validate_json(to_json(document))

    assert messages_at(exc_info.value, "applies_until")


def test_zero_length_scope_window_is_accepted() -> None:
    path: JsonPath = (0, "scope", "applies_from")
    document = replaced(read_fixture("evidence.json"), path, "2026-11-20T23:59:00+09:00")

    evidence = EVIDENCE_LIST.validate_json(to_json(document))

    assert evidence[0].scope.applies_from == evidence[0].scope.applies_until


# --- validation feedback input bounds (orchestrator hardening decision) ---------


def feedback_with_inputs(inputs: dict[str, str]) -> str:
    """Return validation feedback JSON whose first failure carries `inputs`."""
    path: JsonPath = ("failures", 0, "inputs")
    return to_json(replaced(read_fixture("validation_feedback.json"), path, inputs))


def test_feedback_input_key_and_value_at_limits_are_accepted() -> None:
    inputs = {"k" * MAX_LABEL_CHARS: "v" * MAX_TEXT_CHARS}

    feedback = ValidationFeedback.model_validate_json(feedback_with_inputs(inputs))

    assert feedback.failures[0].inputs == inputs


@pytest.mark.parametrize(
    "inputs",
    [
        {"k" * (MAX_LABEL_CHARS + 1): "value"},
        {"key": "v" * (MAX_TEXT_CHARS + 1)},
        {"": "value"},
        {"key": ""},
    ],
)
def test_feedback_input_key_and_value_are_bounded(inputs: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        ValidationFeedback.model_validate_json(feedback_with_inputs(inputs))


# --- evidence claims, transfers, recommendation (Phase 1, decision D1) ----------
# Contract: artifacts/phase-1/interface-contract.md section 1.

NO_INTERVALS: list[object] = []
OPEN_INTERVAL: dict[str, object] = {
    "opens_at": "2026-11-20T09:00:00+09:00",
    "closes_at": "2026-11-20T18:00:00+09:00",
}
TAXI_RATE_EVIDENCE: dict[str, object] = {
    "kind": "sourced_finding",
    "evidence_id": "ev-taxi-rate",
    "topic": "cost",
    "source": {"url": "https://taxi.example.jp/fares", "kind": "official_website", "title": None},
    "retrieved_at": "2026-11-20T14:30:00+09:00",
    "scope": {
        "place_id": None,
        "fallback_key": None,
        "applies_from": "2026-11-20T00:00:00+09:00",
        "applies_until": "2026-11-20T23:59:00+09:00",
    },
    "finding": "Base fare JPY 500 plus JPY 400 per km.",
    "claim": {
        "kind": "taxi_rate",
        "base_fare": {"amount": "500", "currency": "JPY"},
        "per_km": {"amount": "400", "currency": "JPY"},
        "valid_from": "2026-04-01",
        "valid_until": "2027-03-31",
    },
}


def test_every_fixture_evidence_item_carries_a_typed_claim() -> None:
    evidence = EVIDENCE_LIST.validate_json(read_text("evidence.json"))

    garden_hours = next(item for item in evidence if item.evidence_id == "ev-garden-hours")
    assert isinstance(garden_hours.claim, OpeningAccessClaim)
    assert garden_hours.claim.status == OpeningStatus.OPEN
    assert garden_hours.claim.open_intervals[0].opens_at == datetime(
        2026, 11, 20, 9, 0, tzinfo=TOKYO
    )
    assert {item.claim.kind for item in evidence} == {
        "opening_access",
        "visit_duration",
        "cost_amount",
        "constraint",
        "weather",
    }


def test_opening_status_values() -> None:
    assert {status.value for status in OpeningStatus} == {"open", "closed"}


@pytest.mark.parametrize(
    ("index", "claim"),
    [
        (0, {"kind": "visit_duration", "minutes": 60}),
        (1, {"kind": "opening_access", "status": "closed", "open_intervals": NO_INTERVALS}),
        (2, {"kind": "weather", "rain": True}),
        (7, {"kind": "constraint", "constraint_index": 0}),
        (10, {"kind": "cost_amount", "amount": {"amount": "1", "currency": "JPY"}}),
    ],
)
def test_claim_kind_must_match_topic(index: int, claim: dict[str, object]) -> None:
    document = replaced(read_fixture("evidence.json"), (index, "claim"), claim)

    with pytest.raises(ValidationError) as exc_info:
        EVIDENCE_LIST.validate_json(to_json(document))

    assert "claim" in error_fields(exc_info.value)


def test_evidence_requires_a_claim_key() -> None:
    for index in (0, 2):
        document = without(read_fixture("evidence.json"), (index, "claim"))
        with pytest.raises(ValidationError):
            EVIDENCE_LIST.validate_json(to_json(document))


def test_taxi_rate_is_a_cost_claim_variant() -> None:
    evidence = EVIDENCE_LIST.validate_json(to_json([TAXI_RATE_EVIDENCE]))

    claim = evidence[0].claim
    assert isinstance(claim, TaxiRateClaim)
    assert claim.base_fare.amount == Decimal("500")
    assert claim.per_km.amount == Decimal("400")
    assert claim.valid_until.isoformat() == "2027-03-31"


@pytest.mark.parametrize(
    ("path", "value", "field"),
    [
        (("claim", "per_km", "currency"), "USD", "per_km"),
        (("claim", "valid_until"), "2026-03-31", "valid_until"),
        (("claim", "valid_from"), "2026-13-01", "valid_from"),
        (("claim", "base_fare", "amount"), "-1", "amount"),
    ],
)
def test_invalid_taxi_rate_is_rejected(path: JsonPath, value: object, field: str) -> None:
    document = replaced(TAXI_RATE_EVIDENCE, path, value)

    with pytest.raises(ValidationError) as exc_info:
        EVIDENCE_LIST.validate_json(to_json([document]))

    assert field in error_fields(exc_info.value)


@pytest.mark.parametrize(
    ("claim", "field"),
    [
        (
            {"kind": "opening_access", "status": "closed", "open_intervals": [OPEN_INTERVAL]},
            "open_intervals",
        ),
        (
            {"kind": "opening_access", "status": "open", "open_intervals": NO_INTERVALS},
            "open_intervals",
        ),
        (
            {
                "kind": "opening_access",
                "status": "open",
                "open_intervals": [
                    {
                        "opens_at": "2026-11-20T18:00:00+09:00",
                        "closes_at": "2026-11-20T09:00:00+09:00",
                    }
                ],
            },
            "closes_at",
        ),
        (
            {
                "kind": "opening_access",
                "status": "open",
                "open_intervals": [
                    {
                        "opens_at": "2026-11-20T09:00:00+09:00",
                        "closes_at": "2026-11-20T09:00:00+09:00",
                    }
                ],
            },
            "closes_at",
        ),
        (
            {
                "kind": "opening_access",
                "status": "open",
                "open_intervals": [
                    {"opens_at": "2026-11-20T09:00:00", "closes_at": "2026-11-20T18:00:00+09:00"}
                ],
            },
            "opens_at",
        ),
        (
            {
                "kind": "opening_access",
                "status": "open",
                "open_intervals": [OPEN_INTERVAL] * (MAX_OPEN_INTERVALS + 1),
            },
            "open_intervals",
        ),
        ({"kind": "opening_access", "status": "maybe", "open_intervals": NO_INTERVALS}, "status"),
    ],
)
def test_invalid_opening_claim_is_rejected(claim: dict[str, object], field: str) -> None:
    document = replaced(read_fixture("evidence.json"), (0, "claim"), claim)

    with pytest.raises(ValidationError) as exc_info:
        EVIDENCE_LIST.validate_json(to_json(document))

    assert field in error_fields(exc_info.value)


def test_closed_claim_without_intervals_is_accepted() -> None:
    claim: dict[str, object] = {
        "kind": "opening_access",
        "status": "closed",
        "open_intervals": NO_INTERVALS,
    }
    document = replaced(read_fixture("evidence.json"), (0, "claim"), claim)

    evidence = EVIDENCE_LIST.validate_json(to_json(document))

    assert isinstance(evidence[0].claim, OpeningAccessClaim)
    assert evidence[0].claim.status == OpeningStatus.CLOSED


@pytest.mark.parametrize(
    ("index", "claim"),
    [
        (1, {"kind": "visit_duration", "minutes": 0}),
        (1, {"kind": "visit_duration", "minutes": 1441}),
        (1, {"kind": "visit_duration", "minutes": "60"}),
        (8, {"kind": "constraint", "constraint_index": -1}),
        (8, {"kind": "constraint", "constraint_index": MAX_CONSTRAINTS}),
        (10, {"kind": "weather", "rain": "yes"}),
        (10, {"kind": "weather", "rain": True, "verified": True}),
    ],
)
def test_invalid_claim_values_are_rejected(index: int, claim: dict[str, object]) -> None:
    document = replaced(read_fixture("evidence.json"), (index, "claim"), claim)

    with pytest.raises(ValidationError):
        EVIDENCE_LIST.validate_json(to_json(document))


def test_leg_transfers_hold_available_and_walk_minutes() -> None:
    candidate = CandidateItinerary.model_validate_json(read_text("candidate_itinerary.json"))

    transfer = candidate.legs[-1].transfers[0]
    assert (transfer.available_minutes, transfer.walk_minutes) == (6, 4)
    assert candidate.legs[0].transfers == ()


@pytest.mark.parametrize(
    "transfers",
    [
        [{"available_minutes": -1, "walk_minutes": 4}],
        [{"available_minutes": 3, "walk_minutes": -1}],
        [{"available_minutes": 3}],
        [{"available_minutes": 3, "walk_minutes": 10, "ok": True}],
        [{"available_minutes": 3, "walk_minutes": 10}] * (MAX_TRANSFERS + 1),
    ],
)
def test_invalid_transfers_are_rejected(transfers: list[object]) -> None:
    document = replaced(
        read_fixture("candidate_itinerary.json"), ("legs", 2, "transfers"), transfers
    )

    with pytest.raises(ValidationError):
        CandidateItinerary.model_validate_json(to_json(document))


def test_leg_requires_a_transfers_key() -> None:
    document = without(read_fixture("candidate_itinerary.json"), ("legs", 0, "transfers"))

    with pytest.raises(ValidationError) as exc_info:
        CandidateItinerary.model_validate_json(to_json(document))

    assert "transfers" in error_fields(exc_info.value)


def test_recommendation_names_a_candidate_and_a_reason() -> None:
    recommendation = Recommendation.model_validate_json(
        to_json({"candidate_id": "cand-garden-cafe", "reason": "Matches the garden request."})
    )

    assert recommendation.candidate_id == "cand-garden-cafe"
    assert recommendation.reason == "Matches the garden request."


@pytest.mark.parametrize(
    "document",
    [
        {"candidate_id": "cand-garden-cafe", "reason": ""},
        {"candidate_id": "cand-garden-cafe", "reason": None},
        {"candidate_id": "cand-garden-cafe"},
        {"candidate_id": "not an id", "reason": "Fine."},
        {"candidate_id": "cand-garden-cafe", "reason": "x" * (MAX_TEXT_CHARS + 1)},
        {"candidate_id": "cand-garden-cafe", "reason": "Fine.", "labels": ["cheapest"]},
    ],
)
def test_invalid_recommendation_is_rejected(document: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Recommendation.model_validate_json(to_json(document))


# --- stops with no Place ID (user decision D4, 2026-11-18) -----------------------
# Stop 0 of candidate_itinerary.json sits at (35.6764, 139.6993).

FALLBACK_KEY: dict[str, object] = {"result_id": "web-1", "latitude": 35.6764, "longitude": 139.6993}


def fallback_candidate(place_id: object, fallback_key: object) -> object:
    """candidate_itinerary.json with stop 0's identity replaced."""
    document = replaced(
        read_fixture("candidate_itinerary.json"), ("stops", 0, "place_id"), place_id
    )
    return replaced(document, ("stops", 0, "fallback_key"), fallback_key)


def test_stop_without_place_id_is_identified_by_a_fallback_key() -> None:
    candidate = CandidateItinerary.model_validate_json(
        to_json(fallback_candidate(None, FALLBACK_KEY))
    )

    stop = candidate.stops[0]
    assert stop.place_id is None
    assert stop.fallback_key is not None
    assert stop.fallback_key.result_id == "web-1"
    assert (stop.fallback_key.latitude, stop.fallback_key.longitude) == (35.6764, 139.6993)


def test_place_id_stops_have_no_fallback_key() -> None:
    candidate = CandidateItinerary.model_validate_json(read_text("candidate_itinerary.json"))

    assert all(stop.fallback_key is None for stop in candidate.stops)


@pytest.mark.parametrize(
    ("place_id", "fallback_key"),
    [
        pytest.param("ChIJsyntheticGardenMuseum0001", FALLBACK_KEY, id="both"),
        pytest.param(None, None, id="neither"),
    ],
)
def test_stop_needs_exactly_one_identity(place_id: object, fallback_key: object) -> None:
    with pytest.raises(ValidationError) as exc_info:
        CandidateItinerary.model_validate_json(to_json(fallback_candidate(place_id, fallback_key)))

    assert messages_at(exc_info.value, "fallback_key")


def test_stop_requires_the_fallback_key_field() -> None:
    document = without(read_fixture("candidate_itinerary.json"), ("stops", 0, "fallback_key"))

    with pytest.raises(ValidationError) as exc_info:
        CandidateItinerary.model_validate_json(to_json(document))

    assert "fallback_key" in error_fields(exc_info.value)


@pytest.mark.parametrize("field", ["latitude", "longitude"])
def test_fallback_key_coordinates_must_match_the_stop(field: str) -> None:
    moved = {**FALLBACK_KEY, field: 35.0}

    with pytest.raises(ValidationError) as exc_info:
        CandidateItinerary.model_validate_json(to_json(fallback_candidate(None, moved)))

    assert messages_at(exc_info.value, "fallback_key")


@pytest.mark.parametrize(
    ("field", "value"),
    [("result_id", "has spaces"), ("result_id", ""), ("latitude", 91.0), ("longitude", None)],
)
def test_invalid_fallback_key_is_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        CandidateItinerary.model_validate_json(
            to_json(fallback_candidate(None, {**FALLBACK_KEY, field: value}))
        )


def test_evidence_scope_may_name_a_fallback_key() -> None:
    document = replaced(read_fixture("evidence.json"), (0, "scope", "place_id"), None)
    document = replaced(document, (0, "scope", "fallback_key"), FALLBACK_KEY)

    evidence = EVIDENCE_LIST.validate_json(to_json(document))

    assert evidence[0].scope.place_id is None
    assert evidence[0].scope.fallback_key is not None
    assert evidence[0].scope.fallback_key.result_id == "web-1"


def test_evidence_scope_cannot_name_both_identities() -> None:
    document = replaced(read_fixture("evidence.json"), (0, "scope", "fallback_key"), FALLBACK_KEY)

    with pytest.raises(ValidationError) as exc_info:
        EVIDENCE_LIST.validate_json(to_json(document))

    assert messages_at(exc_info.value, "fallback_key")


def test_evidence_scope_requires_the_fallback_key_field() -> None:
    document = without(read_fixture("evidence.json"), (0, "scope", "fallback_key"))

    with pytest.raises(ValidationError) as exc_info:
        EVIDENCE_LIST.validate_json(to_json(document))

    assert "fallback_key" in error_fields(exc_info.value)


@pytest.mark.parametrize("latitude", [0.0, -0.0])
def test_signed_zero_coordinates_give_the_same_stop_key(latitude: float) -> None:
    key = {"result_id": "web-1", "latitude": latitude, "longitude": 1.0}
    document = replaced(fallback_candidate(None, key), ("stops", 0, "latitude"), latitude)
    document = replaced(document, ("stops", 0, "longitude"), 1.0)

    candidate = CandidateItinerary.model_validate_json(to_json(document))

    assert stop_key(candidate.stops[0]) == "web-1@0.0,1.0"


# --- bounded datetimes (orchestrator ruling R-B: years 2000-2100, no tz conversion) ---

OUT_OF_RANGE_TIMES = [
    pytest.param("9999-11-20T19:00:00+09:00", id="year-9999"),
    pytest.param("0001-01-01T00:00:00+09:00", id="year-1"),
    pytest.param("1999-12-31T23:59:59+00:00", id="just-before-2000"),
    pytest.param("2101-01-01T00:00:00+00:00", id="just-after-2100"),
]
BOUNDARY_TIMES = [
    pytest.param("2000-01-01T00:00:00+14:00", id="first-local-instant-of-2000"),
    pytest.param("2100-12-31T23:59:59-12:00", id="last-local-second-of-2100"),
]


def names_the_year_range(messages: list[str]) -> bool:
    """True when some message at the field states the supported 2000-2100 range."""
    return any("2000" in message and "2100" in message for message in messages)


REQUEST_TIME_PATHS = [("gap", "start_time"), ("commitment", "time")]
EVIDENCE_TIME_PATHS = [
    (0, "retrieved_at"),
    (0, "scope", "applies_from"),
    (0, "scope", "applies_until"),
    (0, "claim", "open_intervals", 0, "opens_at"),
    (0, "claim", "open_intervals", 0, "closes_at"),
]


@pytest.mark.parametrize("value", OUT_OF_RANGE_TIMES)
@pytest.mark.parametrize("path", REQUEST_TIME_PATHS)
def test_request_times_outside_the_supported_years_are_rejected(path: JsonPath, value: str) -> None:
    document = replaced(read_fixture("planning_request.json"), path, value)

    with pytest.raises(ValidationError) as exc_info:
        PlanningRequest.model_validate_json(to_json(document))

    assert names_the_year_range(messages_at(exc_info.value, str(path[-1])))


@pytest.mark.parametrize("value", OUT_OF_RANGE_TIMES)
@pytest.mark.parametrize("path", EVIDENCE_TIME_PATHS)
def test_evidence_times_outside_the_supported_years_are_rejected(
    path: JsonPath, value: str
) -> None:
    document = replaced(read_fixture("evidence.json"), path, value)

    with pytest.raises(ValidationError) as exc_info:
        EVIDENCE_LIST.validate_json(to_json(document))

    assert names_the_year_range(messages_at(exc_info.value, str(path[-1])))


@pytest.mark.parametrize("value", OUT_OF_RANGE_TIMES)
def test_continuation_times_outside_the_supported_years_are_rejected(value: str) -> None:
    path: JsonPath = ("request", "commitment", "time")
    document = replaced(read_fixture("continuation_payload.json"), path, value)

    with pytest.raises(ValidationError) as exc_info:
        ContinuationPayload.model_validate_json(to_json(document))

    assert names_the_year_range(messages_at(exc_info.value, "time"))


@pytest.mark.parametrize("value", BOUNDARY_TIMES)
@pytest.mark.parametrize("path", REQUEST_TIME_PATHS)
def test_request_times_at_the_supported_year_boundaries_are_accepted(
    path: JsonPath, value: str
) -> None:
    document = replaced(read_fixture("planning_request.json"), path, value)

    PlanningRequest.model_validate_json(to_json(document))


@pytest.mark.parametrize("value", BOUNDARY_TIMES)
def test_evidence_times_at_the_supported_year_boundaries_are_accepted(value: str) -> None:
    document = replaced(read_fixture("evidence.json"), (0, "retrieved_at"), value)

    EVIDENCE_LIST.validate_json(to_json(document))


@pytest.mark.parametrize("value", BOUNDARY_TIMES)
def test_continuation_times_at_the_supported_year_boundaries_are_accepted(value: str) -> None:
    path: JsonPath = ("request", "commitment", "time")
    document = replaced(read_fixture("continuation_payload.json"), path, value)

    ContinuationPayload.model_validate_json(to_json(document))
