"""Phase 1 acceptance tests for Python feasibility checks.

Expectations derive from docs/requirements.md (R1, R2, R7-R10, R13), the Phase 1
document's edge cases, the 2026-11-18 decisions (D1 typed claims, D2 latest leave-by),
ledger rulings O1-O8, and artifacts/phase-1/interface-contract.md section 2.
All data is synthetic with a fixed date (2026-11-20, Asia/Tokyo); no clock or network.
"""

import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import pytest
from pydantic import TypeAdapter

from travel_agent.contracts import (
    CandidateItinerary,
    CheckedItinerary,
    ConstraintFailure,
    ConstraintId,
    Evidence,
    PlanningRequest,
    TurnResult,
    ValidatedOptions,
    ValidationFeedback,
)
from travel_agent.feasibility import (
    DuplicateEvidenceIdError,
    IncompleteRequestError,
    UnresolvedBudgetCurrencyError,
    UnusableInputError,
    check_candidate,
)

type Doc = dict[str, object]
type JsonPath = tuple[str | int, ...]

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
TOKYO = ZoneInfo("Asia/Tokyo")
EVIDENCE_LIST = TypeAdapter(tuple[Evidence, ...])
TURN_RESULT = TypeAdapter(TurnResult)

PARK = "ChIJsyntheticPark000001"
MUSEUM = "ChIJsyntheticMuseum00001"
MOUNTAIN = "ChIJsyntheticMountain0001"
OTHER_PLACE = "ChIJsyntheticElsewhere001"

WEBSITE = "official_website"
SOCIAL = "official_social_media"
AUTHORITY = "official_authority_notice"
MAPS = "google_maps"
OTHER_WEB = "other_web"
OFFICIAL_KINDS = (WEBSITE, SOCIAL, AUTHORITY)


# --- builders (pure; every parameter explicit) ------------------------------------


def at(clock: str) -> str:
    """Return an ISO datetime on the fixed outing date in Tokyo."""
    return f"2026-11-20T{clock}:00+09:00"


def tokyo(hour: int, minute: int) -> datetime:
    """Return an aware datetime on the fixed outing date in Tokyo."""
    return datetime(2026, 11, 20, hour, minute, tzinfo=TOKYO)


DAY_FROM = at("00:00")
DAY_UNTIL = at("23:59")


def replaced(document: object, path: JsonPath, value: object) -> object:
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


def request_doc() -> Doc:
    """Gap 15:00-19:00 dinner (cutoff 18:45), JPY 3000 cap, every mode, no constraints."""
    return {
        "schema_version": 1,
        "gap": {
            "start_time": at("15:00"),
            "start_location": {
                "label": "Synthetic start",
                "latitude": 35.658,
                "longitude": 139.7016,
                "place_id": None,
            },
        },
        "commitment": {
            "location": {
                "label": "Synthetic dinner venue",
                "latitude": 35.6717,
                "longitude": 139.765,
                "place_id": "ChIJsyntheticDinner00001",
            },
            "time": at("19:00"),
            "timezone": "Asia/Tokyo",
        },
        "budget": {"amount": "3000", "currency": "JPY"},
        "transport_modes": ["walking", "public_transit", "taxi_rideshare"],
        "constraints": [],
        "unresolved_questions": [],
    }


def scope(place_id: str | None, applies_from: str, applies_until: str) -> Doc:
    return {
        "place_id": place_id,
        "fallback_key": None,
        "applies_from": applies_from,
        "applies_until": applies_until,
    }


def sourced(evidence_id: str, topic: str, kind: str, evidence_scope: Doc, claim: Doc) -> Doc:
    return {
        "kind": "sourced_finding",
        "evidence_id": evidence_id,
        "topic": topic,
        "source": {"url": f"https://{evidence_id}.example.org/", "kind": kind, "title": None},
        "retrieved_at": at("14:30"),
        "scope": evidence_scope,
        "finding": "Synthetic finding.",
        "claim": claim,
    }


def estimate(evidence_id: str, topic: str, label: str, evidence_scope: Doc, claim: Doc) -> Doc:
    return {
        "kind": "estimate",
        "evidence_id": evidence_id,
        "topic": topic,
        "label": label,
        "scope": evidence_scope,
        "basis": "Synthetic basis.",
        "based_on": [],
        "claim": claim,
    }


def open_claim(intervals: list[tuple[str, str]]) -> Doc:
    return {
        "kind": "opening_access",
        "status": "open",
        "open_intervals": [{"opens_at": at(o), "closes_at": at(c)} for o, c in intervals],
    }


def closed_claim() -> Doc:
    return {"kind": "opening_access", "status": "closed", "open_intervals": []}


def opening(evidence_id: str, kind: str, place_id: str, claim: Doc) -> Doc:
    """Opening evidence scoped to the whole outing day."""
    return sourced(evidence_id, "opening_access", kind, scope(place_id, DAY_FROM, DAY_UNTIL), claim)


def duration(evidence_id: str, kind: str, place_id: str, minutes: int) -> Doc:
    claim: Doc = {"kind": "visit_duration", "minutes": minutes}
    return sourced(evidence_id, "visit_duration", kind, scope(place_id, DAY_FROM, DAY_UNTIL), claim)


def money(amount: str, currency: str) -> Doc:
    return {"amount": amount, "currency": currency}


def price(evidence_id: str, place_id: str | None, amount: str, currency: str) -> Doc:
    claim: Doc = {"kind": "cost_amount", "amount": money(amount, currency)}
    return sourced(evidence_id, "cost", WEBSITE, scope(place_id, DAY_FROM, DAY_UNTIL), claim)


def taxi_rate(evidence_id: str, base: str, per_km: str, valid_from: str, valid_until: str) -> Doc:
    claim: Doc = {
        "kind": "taxi_rate",
        "base_fare": money(base, "JPY"),
        "per_km": money(per_km, "JPY"),
        "valid_from": valid_from,
        "valid_until": valid_until,
    }
    return sourced(evidence_id, "cost", WEBSITE, scope(None, DAY_FROM, DAY_UNTIL), claim)


def constraint_support(evidence_id: str, place_id: str | None, index: int) -> Doc:
    claim: Doc = {"kind": "constraint", "constraint_index": index}
    return sourced(evidence_id, "constraint", WEBSITE, scope(place_id, DAY_FROM, DAY_UNTIL), claim)


def weather(evidence_id: str, applies_from: str, applies_until: str, rain: bool) -> Doc:
    claim: Doc = {"kind": "weather", "rain": rain}
    return sourced(
        evidence_id, "weather", OTHER_WEB, scope(None, applies_from, applies_until), claim
    )


def stop(place_id: str, visit_minutes: int, evidence_ids: list[str]) -> Doc:
    return {
        "place_id": place_id,
        "label": f"Stop {place_id[-4:]}",
        "latitude": 35.66,
        "longitude": 139.70,
        "fallback_key": None,
        "visit_minutes": visit_minutes,
        "evidence_ids": evidence_ids,
    }


def leg(mode: str, minutes: int, meters: int, transfers: list[tuple[int, int]]) -> Doc:
    return {
        "mode": mode,
        "duration_minutes": minutes,
        "distance_meters": meters,
        "transfers": [{"available_minutes": a, "walk_minutes": w} for a, w in transfers],
        "evidence_ids": [],
    }


def cost(description: str, amount: str, currency: str, evidence_ids: list[str]) -> Doc:
    return {
        "description": description,
        "amount": money(amount, currency),
        "evidence_ids": evidence_ids,
    }


def candidate_doc(stops: list[Doc], legs: list[Doc], costs: list[Doc]) -> Doc:
    return {"candidate_id": "cand-test", "stops": stops, "legs": legs, "costs": costs}


def park_candidate(visit_minutes: int, first_leg: int, last_leg: int) -> Doc:
    """One park stop, walking both ways, JPY 500 admission."""
    return candidate_doc(
        [stop(PARK, visit_minutes, ["ev-park-hours", "ev-park-duration"])],
        [leg("walking", first_leg, 2000, []), leg("walking", last_leg, 2000, [])],
        [cost("Park admission", "500", "JPY", ["ev-park-price"])],
    )


def park_evidence(intervals: list[tuple[str, str]]) -> list[Doc]:
    """Official park hours, a sourced 60-minute visit, and an official JPY 500 price."""
    return [
        opening("ev-park-hours", WEBSITE, PARK, open_claim(intervals)),
        duration("ev-park-duration", OTHER_WEB, PARK, 60),
        price("ev-park-price", PARK, "500", "JPY"),
    ]


def cite(candidate: Doc, cited: list[str]) -> Doc:
    """Return `candidate` with its first stop citing exactly `cited`."""
    return cast(Doc, replaced(candidate, ("stops", 0, "evidence_ids"), cited))


def without_id(evidence: list[Doc], evidence_id: str) -> list[Doc]:
    return [item for item in evidence if item["evidence_id"] != evidence_id]


def run(
    request: object, candidate: object, evidence: list[Doc]
) -> CheckedItinerary | ValidationFeedback:
    """Validate every input at its boundary, then run the feasibility check."""
    return check_candidate(
        PlanningRequest.model_validate_json(json.dumps(request)),
        CandidateItinerary.model_validate_json(json.dumps(candidate)),
        EVIDENCE_LIST.validate_json(json.dumps(evidence)),
    )


def checked(result: CheckedItinerary | ValidationFeedback) -> CheckedItinerary:
    assert isinstance(result, CheckedItinerary), f"expected a pass, got {result!r}"
    return result


def failures_of(result: CheckedItinerary | ValidationFeedback) -> tuple[ConstraintFailure, ...]:
    assert isinstance(result, ValidationFeedback), f"expected a rejection, got {result!r}"
    return result.failures


def constraint_ids(result: CheckedItinerary | ValidationFeedback) -> set[ConstraintId]:
    return {failure.constraint for failure in failures_of(result)}


def only_failure(
    result: CheckedItinerary | ValidationFeedback, constraint: ConstraintId
) -> ConstraintFailure:
    matching = [f for f in failures_of(result) if f.constraint == constraint]
    assert len(matching) == 1, f"expected one {constraint} failure, got {failures_of(result)!r}"
    return matching[0]


def reasons(result: CheckedItinerary | ValidationFeedback, constraint: ConstraintId) -> set[str]:
    return {f.inputs["reason"] for f in failures_of(result) if f.constraint == constraint}


WIDE_OPEN = [("09:00", "21:00")]


# --- arrival cutoff and leave-by (R2, R8, D2, AC1) ----------------------------------


def test_arrival_exactly_at_cutoff_is_accepted() -> None:
    request = replaced(request_doc(), ("gap", "start_time"), at("15:45"))
    candidate = park_candidate(120, 30, 30)  # 15:45 +30 -> 16:15, +120 -> 18:15, +30 -> 18:45

    result = checked(run(request, candidate, park_evidence(WIDE_OPEN)))

    assert result.arrival_at_commitment == tokyo(18, 45)
    assert result.leave_by == (tokyo(15, 45), tokyo(18, 15))


def test_arrival_one_minute_after_cutoff_is_rejected() -> None:
    request = replaced(request_doc(), ("gap", "start_time"), at("15:45"))
    candidate = park_candidate(121, 30, 30)

    result = run(request, candidate, park_evidence(WIDE_OPEN))

    assert constraint_ids(result) == {ConstraintId.ARRIVAL_BUFFER}
    failure = only_failure(result, ConstraintId.ARRIVAL_BUFFER)
    assert datetime.fromisoformat(failure.inputs["earliest_arrival"]) == tokyo(18, 46)
    assert datetime.fromisoformat(failure.inputs["arrival_cutoff"]) == tokyo(18, 45)


def test_leave_by_is_the_latest_feasible_departure() -> None:
    # Earliest arrival 17:00 leaves 105 spare minutes; D2 puts leave-by as late as possible.
    result = checked(run(request_doc(), park_candidate(60, 30, 30), park_evidence(WIDE_OPEN)))

    assert result.leave_by == (tokyo(16, 45), tokyo(18, 15))
    assert result.arrival_at_commitment == tokyo(18, 45)


def test_leave_by_respects_the_closing_time() -> None:
    result = checked(
        run(request_doc(), park_candidate(60, 30, 30), park_evidence([("09:00", "18:00")]))
    )

    assert result.leave_by == (tokyo(16, 30), tokyo(18, 0))
    assert result.arrival_at_commitment == tokyo(18, 30)


def test_multi_stop_leave_by_chain_runs_backward_from_cutoff() -> None:
    candidate = candidate_doc(
        [
            stop(PARK, 30, []),
            stop(MUSEUM, 45, []),
        ],
        [leg("walking", 20, 1500, []), leg("walking", 15, 1000, []), leg("walking", 25, 1800, [])],
        [cost("Park admission", "500", "JPY", ["ev-park-price"])],
    )
    evidence = [
        *park_evidence(WIDE_OPEN),
        opening("ev-museum-hours", WEBSITE, MUSEUM, open_claim([("09:00", "17:30")])),
        duration("ev-museum-duration", OTHER_WEB, MUSEUM, 45),
    ]
    evidence = [
        *without_id(evidence, "ev-park-duration"),
        duration("ev-park-d30", OTHER_WEB, PARK, 30),
    ]

    result = checked(run(request_doc(), candidate, evidence))

    assert result.leave_by == (tokyo(15, 40), tokyo(16, 30), tokyo(17, 30))
    assert result.arrival_at_commitment == tokyo(17, 55)


def test_leaving_the_last_stop_at_leave_by_arrives_by_cutoff() -> None:
    for intervals in (WIDE_OPEN, [("09:00", "18:00")], [("09:00", "17:00")]):
        result = checked(run(request_doc(), park_candidate(60, 30, 30), park_evidence(intervals)))
        last_leg = result.candidate.legs[-1].duration_minutes

        assert result.leave_by[-1] + timedelta(minutes=last_leg) == result.arrival_at_commitment
        assert result.arrival_at_commitment <= tokyo(18, 45)
        assert result.leave_by[0] >= tokyo(15, 0)


def test_early_arrival_waits_for_opening() -> None:
    # Arrive 15:30, wait until 17:00, visit until 18:00, arrive 18:30.
    result = checked(
        run(request_doc(), park_candidate(60, 30, 30), park_evidence([("17:00", "21:00")]))
    )

    assert result.leave_by == (tokyo(16, 45), tokyo(18, 15))


def test_waiting_time_counts_toward_the_arrival_cutoff() -> None:
    # Opens 17:16: visit 17:16-18:16, arrive 18:46.
    result = run(request_doc(), park_candidate(60, 30, 30), park_evidence([("17:16", "21:00")]))

    assert constraint_ids(result) == {ConstraintId.ARRIVAL_BUFFER}


def test_visit_uses_a_later_open_interval_after_a_break() -> None:
    result = checked(
        run(
            request_doc(),
            park_candidate(60, 30, 30),
            park_evidence([("09:00", "15:20"), ("16:00", "21:00")]),
        )
    )

    assert result.leave_by == (tokyo(16, 45), tokyo(18, 15))


def test_full_chain_includes_the_leg_to_the_commitment() -> None:
    # Visit ends 16:30; a 136-minute final leg arrives 18:46.
    result = run(request_doc(), park_candidate(60, 30, 136), park_evidence(WIDE_OPEN))

    assert constraint_ids(result) == {ConstraintId.ARRIVAL_BUFFER}


def test_visit_that_cannot_finish_before_closing_is_rejected() -> None:
    # Arrive 15:30, closes 16:00, needs 60 minutes.
    result = run(request_doc(), park_candidate(60, 30, 30), park_evidence([("09:00", "16:00")]))

    failure = only_failure(result, ConstraintId.OPENING_ACCESS)
    assert failure.inputs["reason"] == "outside_open_interval"
    assert failure.inputs["stop_index"] == "0"
    assert failure.inputs["place_id"] == PARK


def test_checked_times_use_the_commitment_timezone() -> None:
    request = replaced(request_doc(), ("gap", "start_time"), "2026-11-20T06:00:00+00:00")

    result = checked(run(request, park_candidate(60, 30, 30), park_evidence(WIDE_OPEN)))

    assert all(value.utcoffset() == timedelta(hours=9) for value in result.leave_by)
    assert result.arrival_at_commitment.utcoffset() == timedelta(hours=9)


NY_DAY_FROM = "2026-11-01T00:00:00-04:00"
NY_DAY_UNTIL = "2026-11-01T23:00:00-05:00"


def test_schedule_uses_elapsed_time_across_a_dst_fall_back() -> None:
    # New York falls back at 02:00 EDT on 2026-11-01, so wall clock repeats 01:00-02:00.
    # Start 00:30 EDT = 04:30Z; commitment 04:00 EST = 09:00Z; cutoff 08:45Z (255 real minutes).
    # Plan: 60 walk + 60 visit + 120 walk = 240 real minutes, arriving 08:30Z: accepted.
    # Wall-clock arithmetic would arrive at 04:30 local, after the 03:45 cutoff.
    request = replaced(
        replaced(
            replaced(request_doc(), ("gap", "start_time"), "2026-11-01T00:30:00-04:00"),
            ("commitment", "time"),
            "2026-11-01T04:00:00-05:00",
        ),
        ("commitment", "timezone"),
        "America/New_York",
    )
    request = replaced(request, ("budget",), None)
    candidate = candidate_doc(
        [stop(PARK, 60, ["ev-park-hours", "ev-park-duration"])],
        [leg("walking", 60, 4000, []), leg("walking", 120, 8000, [])],
        [cost("Park admission", "5", "USD", ["ev-park-price"])],
    )
    ny_day = scope(PARK, NY_DAY_FROM, NY_DAY_UNTIL)
    evidence = [
        sourced(
            "ev-park-hours",
            "opening_access",
            WEBSITE,
            ny_day,
            {
                "kind": "opening_access",
                "status": "open",
                "open_intervals": [{"opens_at": NY_DAY_FROM, "closes_at": NY_DAY_UNTIL}],
            },
        ),
        sourced(
            "ev-park-duration",
            "visit_duration",
            OTHER_WEB,
            ny_day,
            {
                "kind": "visit_duration",
                "minutes": 60,
            },
        ),
        sourced(
            "ev-park-price",
            "cost",
            WEBSITE,
            ny_day,
            {
                "kind": "cost_amount",
                "amount": money("5", "USD"),
            },
        ),
    ]

    result = checked(run(request, candidate, evidence))

    # Backward from 08:45Z: leave the park by 06:45Z (the second 01:45, EST), so arrive
    # by 05:45Z and leave the start by 04:45Z (00:45 EDT).
    assert [value.isoformat() for value in result.leave_by] == [
        "2026-11-01T00:45:00-04:00",
        "2026-11-01T01:45:00-05:00",
    ]
    assert result.arrival_at_commitment.isoformat() == "2026-11-01T03:45:00-05:00"


# --- same-day gap and unusable input (R1, R13) --------------------------------------


@pytest.mark.parametrize("gap_start", ["2026-11-19T23:30:00+09:00", at("18:45"), at("19:30")])
def test_gap_must_start_before_cutoff_on_the_commitment_day(gap_start: str) -> None:
    request = replaced(request_doc(), ("gap", "start_time"), gap_start)

    result = run(request, park_candidate(60, 30, 30), park_evidence(WIDE_OPEN))

    failure = only_failure(result, ConstraintId.SAME_DAY_GAP)
    assert datetime.fromisoformat(failure.inputs["gap_start"]) == datetime.fromisoformat(gap_start)
    assert ConstraintId.ARRIVAL_BUFFER not in constraint_ids(result)


def test_same_day_uses_the_commitment_local_date() -> None:
    # 2026-11-19T16:00Z is 2026-11-20 01:00 in Tokyo: the same local day.
    request = replaced(request_doc(), ("gap", "start_time"), "2026-11-19T16:00:00+00:00")

    checked(run(request, park_candidate(60, 30, 30), park_evidence(WIDE_OPEN)))


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
def test_missing_essentials_raise_instead_of_guessing(path: JsonPath) -> None:
    request = replaced(request_doc(), path, None)

    with pytest.raises(IncompleteRequestError):
        run(request, park_candidate(60, 30, 30), park_evidence(WIDE_OPEN))


def test_budget_without_currency_raises() -> None:
    request = replaced(request_doc(), ("budget", "currency"), None)

    with pytest.raises(UnresolvedBudgetCurrencyError):
        run(request, park_candidate(60, 30, 30), park_evidence(WIDE_OPEN))


def test_duplicate_evidence_ids_raise() -> None:
    evidence = [*park_evidence(WIDE_OPEN), price("ev-park-price", PARK, "500", "JPY")]

    with pytest.raises(DuplicateEvidenceIdError):
        run(request_doc(), park_candidate(60, 30, 30), evidence)


def test_unusable_input_errors_share_a_base_class() -> None:
    assert issubclass(IncompleteRequestError, UnusableInputError)
    assert issubclass(UnresolvedBudgetCurrencyError, UnusableInputError)
    assert issubclass(DuplicateEvidenceIdError, UnusableInputError)
    assert issubclass(UnusableInputError, ValueError)


# --- transfer walks (R9, AC2) ------------------------------------------------------


def transit_candidate(transfers: list[tuple[int, int]]) -> Doc:
    """Park candidate whose second leg is public transit citing the JPY 200 fare."""
    transit_leg = {**leg("public_transit", 30, 6000, transfers), "evidence_ids": ["ev-fare"]}
    return cast(Doc, replaced(park_candidate(60, 30, 30), ("legs", 1), transit_leg))


def transit_evidence() -> list[Doc]:
    """Park evidence plus an official, plan-wide JPY 200 transit fare."""
    return [*park_evidence(WIDE_OPEN), price("ev-fare", None, "200", "JPY")]


def test_three_minute_transfer_needing_ten_minutes_walk_is_rejected() -> None:
    result = run(request_doc(), transit_candidate([(6, 4), (3, 10)]), transit_evidence())

    failure = only_failure(result, ConstraintId.TRANSFER_TIME)
    assert failure.inputs["leg_index"] == "1"
    assert failure.inputs["transfer_index"] == "1"
    assert failure.inputs["available_minutes"] == "3"
    assert failure.inputs["walk_minutes"] == "10"


def test_transfer_walk_equal_to_available_time_passes() -> None:
    checked(run(request_doc(), transit_candidate([(10, 10)]), transit_evidence()))


# --- opening and access (R9, O2, AC3) ----------------------------------------------


def evidence_ids(items: list[Doc]) -> list[str]:
    return [cast(str, item["evidence_id"]) for item in items]


def park_with_opening(opening_items: list[Doc]) -> CheckedItinerary | ValidationFeedback:
    """Replace the park's opening evidence; the stop cites every supplied item."""
    evidence = [*without_id(park_evidence(WIDE_OPEN), "ev-park-hours"), *opening_items]
    candidate = cite(park_candidate(60, 30, 30), ["ev-park-duration", *evidence_ids(opening_items)])
    return run(request_doc(), candidate, evidence)


def test_official_closed_place_is_rejected() -> None:
    result = park_with_opening([opening("ev-park-hours", WEBSITE, PARK, closed_claim())])

    failure = only_failure(result, ConstraintId.OPENING_ACCESS)
    assert failure.inputs["reason"] == "closed"


def test_place_without_opening_evidence_is_unverified() -> None:
    result = park_with_opening([])

    assert reasons(result, ConstraintId.OPENING_ACCESS) == {"unverified"}


@pytest.mark.parametrize(
    "item",
    [
        opening("ev-park-hours", OTHER_WEB, PARK, open_claim(WIDE_OPEN)),
        estimate(
            "ev-park-hours",
            "opening_access",
            "ai_best_guess",
            scope(PARK, DAY_FROM, DAY_UNTIL),
            open_claim(WIDE_OPEN),
        ),
        estimate(
            "ev-park-hours",
            "opening_access",
            "estimate",
            scope(PARK, DAY_FROM, DAY_UNTIL),
            open_claim(WIDE_OPEN),
        ),
        opening("ev-park-hours", WEBSITE, OTHER_PLACE, open_claim(WIDE_OPEN)),
    ],
    ids=["other-web", "ai-best-guess", "estimate", "different-place"],
)
def test_non_verifying_opening_evidence_leaves_place_unverified(item: Doc) -> None:
    assert reasons(park_with_opening([item]), ConstraintId.OPENING_ACCESS) == {"unverified"}


def mountain_run(opening_items: list[Doc]) -> CheckedItinerary | ValidationFeedback:
    candidate = candidate_doc(
        [stop(MOUNTAIN, 60, [])],
        [leg("walking", 30, 2000, []), leg("walking", 30, 2000, [])],
        [cost("Trail fee", "0", "JPY", ["ev-mountain-fee"])],
    )
    evidence = [
        *opening_items,
        duration("ev-mountain-duration", OTHER_WEB, MOUNTAIN, 60),
        price("ev-mountain-fee", MOUNTAIN, "0", "JPY"),
    ]
    return run(request_doc(), candidate, evidence)


def test_closed_mountain_is_rejected() -> None:
    notice = opening("ev-trail-notice", AUTHORITY, MOUNTAIN, closed_claim())

    assert reasons(mountain_run([notice]), ConstraintId.OPENING_ACCESS) == {"closed"}


def test_unverified_mountain_is_rejected() -> None:
    blog = opening("ev-trail-blog", OTHER_WEB, MOUNTAIN, open_claim(WIDE_OPEN))

    assert reasons(mountain_run([blog]), ConstraintId.OPENING_ACCESS) == {"unverified"}


def test_mountain_open_per_authority_notice_passes() -> None:
    notice = opening("ev-trail-notice", AUTHORITY, MOUNTAIN, open_claim([("06:00", "18:00")]))

    checked(mountain_run([notice]))


def test_google_maps_alone_verifies_when_no_official_source_exists() -> None:
    maps = opening("ev-park-maps", MAPS, PARK, open_claim(WIDE_OPEN))

    checked(park_with_opening([maps]))


@pytest.mark.parametrize("official_kind", OFFICIAL_KINDS)
def test_official_closure_beats_google_maps(official_kind: str) -> None:
    maps = opening("ev-park-maps", MAPS, PARK, open_claim(WIDE_OPEN))
    official = opening("ev-park-official", official_kind, PARK, closed_claim())

    assert reasons(park_with_opening([maps, official]), ConstraintId.OPENING_ACCESS) == {"closed"}


def test_official_opening_beats_a_google_maps_closure() -> None:
    maps = opening("ev-park-maps", MAPS, PARK, closed_claim())
    official = opening("ev-park-official", WEBSITE, PARK, open_claim(WIDE_OPEN))

    checked(park_with_opening([maps, official]))


def test_official_hours_beat_google_maps_hours() -> None:
    # Maps says open late; the official site closes at 16:00, so the visit cannot fit.
    maps = opening("ev-park-maps", MAPS, PARK, open_claim(WIDE_OPEN))
    official = opening("ev-park-official", SOCIAL, PARK, open_claim([("09:00", "16:00")]))

    assert reasons(park_with_opening([maps, official]), ConstraintId.OPENING_ACCESS) == {
        "outside_open_interval"
    }


def test_uncited_official_closure_still_applies() -> None:
    # The stop cites only its Maps listing; Python still sees the official closure.
    maps = opening("ev-park-hours", MAPS, PARK, open_claim(WIDE_OPEN))
    official = opening("ev-park-closure", AUTHORITY, PARK, closed_claim())
    evidence = [*without_id(park_evidence(WIDE_OPEN), "ev-park-hours"), maps, official]

    result = run(request_doc(), park_candidate(60, 30, 30), evidence)

    assert reasons(result, ConstraintId.OPENING_ACCESS) == {"closed"}


def test_same_official_intervals_in_a_different_order_do_not_conflict() -> None:
    split_day = [("09:00", "12:00"), ("13:00", "21:00")]
    website = opening("ev-park-website", WEBSITE, PARK, open_claim(split_day))
    notice = opening("ev-park-notice", AUTHORITY, PARK, open_claim(split_day[::-1]))

    checked(park_with_opening([website, notice]))


def test_official_source_that_does_not_apply_to_the_visit_is_not_used() -> None:
    old_post = sourced(
        "ev-park-old-post",
        "opening_access",
        SOCIAL,
        scope(PARK, "2026-10-01T00:00:00+09:00", "2026-10-31T23:59:00+09:00"),
        closed_claim(),
    )
    maps = opening("ev-park-maps", MAPS, PARK, open_claim(WIDE_OPEN))

    checked(park_with_opening([old_post, maps]))


def test_official_source_outside_the_outing_window_does_not_verify() -> None:
    morning_only = sourced(
        "ev-park-morning",
        "opening_access",
        WEBSITE,
        scope(PARK, at("08:00"), at("12:00")),
        open_claim([("08:00", "12:00")]),
    )

    assert reasons(park_with_opening([morning_only]), ConstraintId.OPENING_ACCESS) == {"unverified"}


def test_partial_window_official_closure_applies() -> None:
    # A notice closing the park 16:00-17:00 overlaps the outing, so it decides opening.
    maps = opening("ev-park-maps", MAPS, PARK, open_claim(WIDE_OPEN))
    notice = sourced(
        "ev-park-notice",
        "opening_access",
        AUTHORITY,
        scope(PARK, at("16:00"), at("17:00")),
        closed_claim(),
    )

    assert reasons(park_with_opening([maps, notice]), ConstraintId.OPENING_ACCESS) == {"closed"}


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (open_claim(WIDE_OPEN), closed_claim()),
        (open_claim(WIDE_OPEN), open_claim([("09:00", "18:00")])),
    ],
    ids=["open-vs-closed", "different-hours"],
)
def test_disagreeing_official_sources_are_rejected(first: Doc, second: Doc) -> None:
    website = opening("ev-park-site", WEBSITE, PARK, first)
    social = opening("ev-park-social", SOCIAL, PARK, second)
    maps = opening("ev-park-maps", MAPS, PARK, open_claim(WIDE_OPEN))

    assert reasons(park_with_opening([website, social, maps]), ConstraintId.OPENING_ACCESS) == {
        "conflicting_sources"
    }


def test_agreeing_official_sources_pass() -> None:
    website = opening("ev-park-site", WEBSITE, PARK, open_claim(WIDE_OPEN))
    notice = opening("ev-park-notice", AUTHORITY, PARK, open_claim(WIDE_OPEN))

    checked(park_with_opening([website, notice]))


# --- visit duration (R10, O6, AC4) -------------------------------------------------


def park_with_duration(
    visit_minutes: int, items: list[Doc]
) -> CheckedItinerary | ValidationFeedback:
    """Replace the park's duration evidence; the stop cites every supplied item."""
    evidence = [*without_id(park_evidence(WIDE_OPEN), "ev-park-duration"), *items]
    candidate = cite(park_candidate(visit_minutes, 30, 30), ["ev-park-hours", *evidence_ids(items)])
    return run(request_doc(), candidate, evidence)


def test_visit_matching_sourced_duration_passes() -> None:
    checked(park_with_duration(60, [duration("ev-park-duration", OTHER_WEB, PARK, 60)]))


def test_visit_longer_than_sourced_duration_passes() -> None:
    checked(park_with_duration(90, [duration("ev-park-duration", OTHER_WEB, PARK, 60)]))


def test_visit_shorter_than_sourced_duration_is_rejected() -> None:
    result = park_with_duration(59, [duration("ev-park-duration", OTHER_WEB, PARK, 60)])

    failure = only_failure(result, ConstraintId.VISIT_DURATION)
    assert failure.inputs["visit_minutes"] == "59"
    assert failure.inputs["required_minutes"] == "60"
    assert failure.inputs["stop_index"] == "0"


def test_missing_duration_evidence_is_rejected() -> None:
    result = park_with_duration(60, [])

    failure = only_failure(result, ConstraintId.MISSING_EVIDENCE)
    assert failure.inputs["reason"] == "no_visit_duration"
    assert failure.inputs["place_id"] == PARK


def ai_duration(minutes: int) -> Doc:
    claim: Doc = {"kind": "visit_duration", "minutes": minutes}
    return estimate(
        "ev-park-guess", "visit_duration", "ai_best_guess", scope(PARK, DAY_FROM, DAY_UNTIL), claim
    )


def test_ai_best_guess_duration_is_allowed() -> None:
    checked(park_with_duration(60, [ai_duration(45)]))


def test_ai_best_guess_duration_still_bounds_the_visit() -> None:
    result = park_with_duration(30, [ai_duration(45)])

    assert only_failure(result, ConstraintId.VISIT_DURATION).inputs["required_minutes"] == "45"


def test_sourced_duration_takes_precedence_over_a_best_guess() -> None:
    items = [duration("ev-park-duration", OTHER_WEB, PARK, 45), ai_duration(90)]

    checked(park_with_duration(60, items))


def test_duration_for_another_place_is_not_used() -> None:
    result = park_with_duration(60, [duration("ev-park-duration", OTHER_WEB, OTHER_PLACE, 60)])

    assert reasons(result, ConstraintId.MISSING_EVIDENCE) == {"no_visit_duration"}


# --- costs and budget (R13, O3, O6, O7, AC5) ----------------------------------------


def with_budget(amount: str | None) -> Doc:
    budget = None if amount is None else {"amount": amount, "currency": "JPY"}
    return cast(Doc, replaced(request_doc(), ("budget",), budget))


def test_sourced_total_is_not_labelled_an_estimate() -> None:
    result = checked(run(request_doc(), park_candidate(60, 30, 30), park_evidence(WIDE_OPEN)))

    assert result.total_cost.amount == Decimal("500")
    assert result.total_cost.currency == "JPY"
    assert result.total_includes_estimates is False


def test_total_equal_to_budget_passes() -> None:
    result = checked(run(with_budget("500"), park_candidate(60, 30, 30), park_evidence(WIDE_OPEN)))

    assert result.total_cost.amount == Decimal("500")


def test_total_above_budget_is_rejected() -> None:
    result = run(with_budget("499"), park_candidate(60, 30, 30), park_evidence(WIDE_OPEN))

    failure = only_failure(result, ConstraintId.BUDGET)
    assert failure.inputs["reason"] == "over_budget"
    assert Decimal(failure.inputs["total_cost"]) == Decimal("500")
    assert Decimal(failure.inputs["budget"]) == Decimal("499")


def test_no_budget_applies_no_cap_but_still_totals() -> None:
    candidate = replaced(
        park_candidate(60, 30, 30), ("costs", 0), cost("Park admission", "90000", "JPY", ["ev-big"])
    )
    evidence = [*park_evidence(WIDE_OPEN), price("ev-big", PARK, "90000", "JPY")]

    result = checked(run(with_budget(None), candidate, evidence))

    assert result.total_cost.amount == Decimal("90000")
    assert result.total_cost.currency == "JPY"


def test_budget_with_no_priced_items_totals_zero() -> None:
    candidate = replaced(park_candidate(60, 30, 30), ("costs",), [])

    result = checked(run(request_doc(), candidate, park_evidence(WIDE_OPEN)))

    assert result.total_cost.amount == Decimal("0")
    assert result.total_cost.currency == "JPY"
    assert result.total_includes_estimates is False


def test_no_budget_and_nothing_priced_has_no_currency() -> None:
    candidate = replaced(park_candidate(60, 30, 30), ("costs",), [])

    result = run(with_budget(None), candidate, park_evidence(WIDE_OPEN))

    assert reasons(result, ConstraintId.MISSING_EVIDENCE) == {"no_cost_currency"}


def ai_price(amount: str) -> Doc:
    claim: Doc = {"kind": "cost_amount", "amount": money(amount, "JPY")}
    return estimate(
        "ev-park-guess", "cost", "ai_best_guess", scope(PARK, DAY_FROM, DAY_UNTIL), claim
    )


def test_ai_best_guess_price_is_counted_and_labelled() -> None:
    candidate = replaced(
        park_candidate(60, 30, 30),
        ("costs", 0),
        cost("Park admission", "800", "JPY", ["ev-park-guess"]),
    )
    evidence = [*park_evidence(WIDE_OPEN), ai_price("800")]

    result = checked(run(request_doc(), candidate, evidence))

    assert result.total_cost.amount == Decimal("800")
    assert result.total_includes_estimates is True


def test_ai_best_guess_price_counts_toward_the_cap() -> None:
    candidate = replaced(
        park_candidate(60, 30, 30),
        ("costs", 0),
        cost("Park admission", "800", "JPY", ["ev-park-guess"]),
    )
    evidence = [*park_evidence(WIDE_OPEN), ai_price("800")]

    result = run(with_budget("799"), candidate, evidence)

    assert reasons(result, ConstraintId.BUDGET) == {"over_budget"}


def test_cost_that_disagrees_with_its_evidence_is_rejected() -> None:
    candidate = replaced(park_candidate(60, 30, 30), ("costs", 0, "amount", "amount"), "300")

    result = run(request_doc(), candidate, park_evidence(WIDE_OPEN))

    failure = only_failure(result, ConstraintId.MISSING_EVIDENCE)
    assert failure.inputs["reason"] == "cost_mismatch"
    assert failure.inputs["cost_index"] == "0"


def test_cost_citing_no_price_evidence_is_rejected() -> None:
    candidate = replaced(
        park_candidate(60, 30, 30), ("costs", 0, "evidence_ids"), ["ev-park-hours"]
    )

    result = run(request_doc(), candidate, park_evidence(WIDE_OPEN))

    assert reasons(result, ConstraintId.MISSING_EVIDENCE) == {"no_cost_evidence"}


def test_outdated_price_is_not_current() -> None:
    stale = sourced(
        "ev-park-price",
        "cost",
        WEBSITE,
        scope(PARK, "2025-01-01T00:00:00+09:00", "2025-12-31T23:59:00+09:00"),
        {"kind": "cost_amount", "amount": money("500", "JPY")},
    )
    evidence = [*without_id(park_evidence(WIDE_OPEN), "ev-park-price"), stale]

    result = run(request_doc(), park_candidate(60, 30, 30), evidence)

    assert reasons(result, ConstraintId.MISSING_EVIDENCE) == {"no_cost_evidence"}


def test_cost_in_a_different_currency_from_the_budget_is_rejected() -> None:
    candidate = replaced(park_candidate(60, 30, 30), ("costs", 0, "amount", "currency"), "USD")
    evidence = [
        *without_id(park_evidence(WIDE_OPEN), "ev-park-price"),
        price("ev-park-price", PARK, "500", "USD"),
    ]

    result = run(request_doc(), candidate, evidence)

    assert reasons(result, ConstraintId.BUDGET) == {"currency_mismatch"}


def test_mixed_cost_currencies_without_a_budget_are_rejected() -> None:
    candidate = replaced(
        park_candidate(60, 30, 30),
        ("costs",),
        [
            cost("Park admission", "500", "JPY", ["ev-park-price"]),
            cost("Snack", "3", "USD", ["ev-snack"]),
        ],
    )
    evidence = [*park_evidence(WIDE_OPEN), price("ev-snack", PARK, "3", "USD")]

    result = run(with_budget(None), candidate, evidence)

    assert reasons(result, ConstraintId.BUDGET) == {"currency_mismatch"}


def test_currency_mismatch_is_reported_alongside_other_cost_failures() -> None:
    # Cost 0 cites no price; cost 1 is a correctly evidenced EUR amount on a JPY budget.
    candidate = replaced(
        park_candidate(60, 30, 30),
        ("costs",),
        [
            cost("Park admission", "500", "JPY", ["ev-park-hours"]),
            cost("Snack", "3", "EUR", ["ev-snack"]),
        ],
    )
    evidence = [*park_evidence(WIDE_OPEN), price("ev-snack", PARK, "3", "EUR")]

    result = run(request_doc(), candidate, evidence)

    missing = only_failure(result, ConstraintId.MISSING_EVIDENCE)
    assert missing.inputs["reason"] == "no_cost_evidence"
    assert missing.inputs["cost_index"] == "0"
    mismatch = only_failure(result, ConstraintId.BUDGET)
    assert mismatch.inputs["reason"] == "currency_mismatch"
    assert mismatch.inputs["currencies"] == "EUR,JPY"


# --- taxi fares (R7, R13, O7, AC5) -------------------------------------------------


def taxi_candidate(meters: int) -> Doc:
    return cast(
        Doc,
        replaced(park_candidate(60, 30, 30), ("legs", 0), leg("taxi_rideshare", 15, meters, [])),
    )


CURRENT_RATE = taxi_rate("ev-taxi-rate", "500", "400", "2026-04-01", "2027-03-31")


def test_taxi_fare_is_computed_from_distance_and_current_rate() -> None:
    result = checked(
        run(request_doc(), taxi_candidate(2500), [*park_evidence(WIDE_OPEN), CURRENT_RATE])
    )

    # 500 + 400 x 2.5 km = 1500, plus 500 admission.
    assert result.total_cost.amount == Decimal("2000")
    assert result.total_includes_estimates is True


def test_taxi_fare_uses_exact_fractional_kilometres() -> None:
    result = checked(
        run(request_doc(), taxi_candidate(2345), [*park_evidence(WIDE_OPEN), CURRENT_RATE])
    )

    # 500 + 400 x 2.345 = 1438 exactly.
    assert result.total_cost.amount == Decimal("1938")


def test_taxi_fare_counts_toward_the_cap() -> None:
    result = run(
        with_budget("1999"), taxi_candidate(2500), [*park_evidence(WIDE_OPEN), CURRENT_RATE]
    )

    assert reasons(result, ConstraintId.BUDGET) == {"over_budget"}


@pytest.mark.parametrize(
    ("valid_from", "valid_until"),
    [("2025-04-01", "2026-11-19"), ("2026-11-21", "2027-03-31")],
    ids=["expired", "not-yet-valid"],
)
def test_taxi_rate_not_valid_on_the_outing_date_is_not_current(
    valid_from: str, valid_until: str
) -> None:
    rate = taxi_rate("ev-taxi-rate", "500", "400", valid_from, valid_until)

    result = run(request_doc(), taxi_candidate(2500), [*park_evidence(WIDE_OPEN), rate])

    failure = only_failure(result, ConstraintId.MISSING_EVIDENCE)
    assert failure.inputs["reason"] == "no_taxi_rate"
    assert failure.inputs["leg_index"] == "0"


def test_taxi_rate_valid_through_the_outing_date_applies() -> None:
    rate = taxi_rate("ev-taxi-rate", "500", "400", "2026-11-20", "2026-11-20")

    checked(run(request_doc(), taxi_candidate(2500), [*park_evidence(WIDE_OPEN), rate]))


def test_taxi_leg_without_a_rate_is_rejected() -> None:
    result = run(request_doc(), taxi_candidate(2500), park_evidence(WIDE_OPEN))

    assert reasons(result, ConstraintId.MISSING_EVIDENCE) == {"no_taxi_rate"}


def test_estimated_taxi_rate_is_not_current_pricing() -> None:
    guessed = estimate(
        "ev-taxi-rate",
        "cost",
        "ai_best_guess",
        scope(None, DAY_FROM, DAY_UNTIL),
        cast(Doc, CURRENT_RATE["claim"]),
    )

    result = run(request_doc(), taxi_candidate(2500), [*park_evidence(WIDE_OPEN), guessed])

    assert reasons(result, ConstraintId.MISSING_EVIDENCE) == {"no_taxi_rate"}


def test_conflicting_taxi_rates_are_rejected() -> None:
    other = taxi_rate("ev-taxi-rate-2", "600", "400", "2026-04-01", "2027-03-31")

    result = run(
        request_doc(), taxi_candidate(2500), [*park_evidence(WIDE_OPEN), CURRENT_RATE, other]
    )

    assert reasons(result, ConstraintId.MISSING_EVIDENCE) == {"conflicting_taxi_rates"}


@pytest.mark.parametrize("per_km", ["400", "400.00"], ids=["same-scale", "different-scale"])
def test_same_taxi_price_with_different_validity_windows_is_not_a_conflict(per_km: str) -> None:
    same_price = taxi_rate("ev-taxi-rate-2", "500", per_km, "2026-10-01", "2026-12-31")

    result = checked(
        run(
            request_doc(),
            taxi_candidate(2500),
            [*park_evidence(WIDE_OPEN), CURRENT_RATE, same_price],
        )
    )

    # One price, fare counted once: 500 + 400 x 2.5 km = 1500, plus 500 admission.
    assert result.total_cost.amount == Decimal("2000")


def test_different_per_km_among_applicable_taxi_rates_is_a_conflict() -> None:
    other = taxi_rate("ev-taxi-rate-2", "500", "450", "2026-10-01", "2026-12-31")

    result = run(
        request_doc(), taxi_candidate(2500), [*park_evidence(WIDE_OPEN), CURRENT_RATE, other]
    )

    failure = only_failure(result, ConstraintId.MISSING_EVIDENCE)
    assert failure.inputs["reason"] == "conflicting_taxi_rates"
    assert failure.inputs["leg_index"] == "0"


# --- transit fares (R13: the cap covers transport; the leg cites its fare) ----------


def transit_fare_guess(amount: str) -> Doc:
    claim: Doc = {"kind": "cost_amount", "amount": money(amount, "JPY")}
    return estimate("ev-fare", "cost", "ai_best_guess", scope(None, DAY_FROM, DAY_UNTIL), claim)


def test_transit_fare_cited_by_its_leg_is_added_to_the_total() -> None:
    result = checked(run(request_doc(), transit_candidate([]), transit_evidence()))

    assert result.total_cost.amount == Decimal("700")  # 500 admission + 200 fare
    assert result.total_includes_estimates is False


def test_transit_fare_counts_toward_the_cap() -> None:
    result = run(with_budget("699"), transit_candidate([]), transit_evidence())

    failure = only_failure(result, ConstraintId.BUDGET)
    assert failure.inputs["reason"] == "over_budget"
    assert Decimal(failure.inputs["total_cost"]) == Decimal("700")


def test_transit_leg_without_a_cited_fare_is_rejected() -> None:
    # The fare exists in the evidence, but the leg does not cite it.
    candidate = replaced(transit_candidate([]), ("legs", 1, "evidence_ids"), [])

    result = run(request_doc(), candidate, transit_evidence())

    failure = only_failure(result, ConstraintId.MISSING_EVIDENCE)
    assert failure.inputs["reason"] == "no_transit_fare"
    assert failure.inputs["leg_index"] == "1"


def test_transit_leg_citing_only_non_fare_evidence_is_rejected() -> None:
    candidate = replaced(transit_candidate([]), ("legs", 1, "evidence_ids"), ["ev-park-hours"])

    result = run(request_doc(), candidate, transit_evidence())

    assert reasons(result, ConstraintId.MISSING_EVIDENCE) == {"no_transit_fare"}


def test_outdated_transit_fare_is_not_current() -> None:
    stale = sourced(
        "ev-fare",
        "cost",
        WEBSITE,
        scope(None, "2025-01-01T00:00:00+09:00", "2025-12-31T23:59:00+09:00"),
        {"kind": "cost_amount", "amount": money("200", "JPY")},
    )

    result = run(request_doc(), transit_candidate([]), [*park_evidence(WIDE_OPEN), stale])

    assert reasons(result, ConstraintId.MISSING_EVIDENCE) == {"no_transit_fare"}


def test_ai_best_guess_transit_fare_is_counted_and_labelled() -> None:
    evidence = [*park_evidence(WIDE_OPEN), transit_fare_guess("250")]

    result = checked(run(request_doc(), transit_candidate([]), evidence))

    assert result.total_cost.amount == Decimal("750")
    assert result.total_includes_estimates is True


def test_leg_citing_two_different_fares_is_rejected() -> None:
    candidate = replaced(transit_candidate([]), ("legs", 1, "evidence_ids"), ["ev-fare", "ev-f2"])
    evidence = [*transit_evidence(), price("ev-f2", None, "220", "JPY")]

    result = run(request_doc(), candidate, evidence)

    failure = only_failure(result, ConstraintId.MISSING_EVIDENCE)
    assert failure.inputs["reason"] == "conflicting_transit_fares"
    assert failure.inputs["leg_index"] == "1"


def test_leg_citing_the_same_fare_twice_counts_it_once() -> None:
    candidate = replaced(transit_candidate([]), ("legs", 1, "evidence_ids"), ["ev-fare", "ev-f2"])
    evidence = [*transit_evidence(), price("ev-f2", None, "200.00", "JPY")]

    result = checked(run(request_doc(), candidate, evidence))

    assert result.total_cost.amount == Decimal("700")


def test_transit_fare_in_another_currency_from_the_budget_is_rejected() -> None:
    evidence = [*park_evidence(WIDE_OPEN), price("ev-fare", None, "2", "USD")]

    result = run(request_doc(), transit_candidate([]), evidence)

    assert reasons(result, ConstraintId.BUDGET) == {"currency_mismatch"}


def test_transit_fare_sets_the_currency_when_nothing_else_is_priced() -> None:
    candidate = replaced(transit_candidate([]), ("costs",), [])

    result = checked(run(with_budget(None), candidate, transit_evidence()))

    assert result.total_cost.amount == Decimal("200")
    assert result.total_cost.currency == "JPY"


def test_different_taxi_price_outside_the_outing_date_is_not_a_conflict() -> None:
    expired = taxi_rate("ev-taxi-rate-old", "450", "350", "2025-04-01", "2026-03-31")

    result = checked(
        run(
            request_doc(),
            taxi_candidate(2500),
            [*park_evidence(WIDE_OPEN), CURRENT_RATE, expired],
        )
    )

    # Only the current rate applies: 500 + 400 x 2.5 km = 1500, plus 500 admission.
    assert result.total_cost.amount == Decimal("2000")


def test_taxi_fare_alone_sets_the_total_currency_without_a_budget() -> None:
    candidate = replaced(taxi_candidate(2500), ("costs",), [])

    result = checked(run(with_budget(None), candidate, [*park_evidence(WIDE_OPEN), CURRENT_RATE]))

    assert result.total_cost.amount == Decimal("1500")
    assert result.total_cost.currency == "JPY"
    assert result.total_includes_estimates is True


# --- weather (R9, AC6) -------------------------------------------------------------


def test_rain_alone_adds_a_warning_without_rejecting() -> None:
    rain = weather("ev-rain", at("16:00"), at("20:00"), True)

    result = checked(
        run(request_doc(), park_candidate(60, 30, 30), [*park_evidence(WIDE_OPEN), rain])
    )

    assert len(result.weather_warnings) == 1
    assert result.weather_warnings[0].strip()


def test_dry_forecast_adds_no_warning() -> None:
    dry = weather("ev-dry", at("16:00"), at("20:00"), False)

    result = checked(
        run(request_doc(), park_candidate(60, 30, 30), [*park_evidence(WIDE_OPEN), dry])
    )

    assert result.weather_warnings == ()


def test_rain_outside_the_outing_adds_no_warning() -> None:
    rain = weather("ev-rain", at("20:00"), at("23:00"), True)

    result = checked(
        run(request_doc(), park_candidate(60, 30, 30), [*park_evidence(WIDE_OPEN), rain])
    )

    assert result.weather_warnings == ()


def test_rain_does_not_replace_the_opening_check() -> None:
    rain = weather("ev-rain", at("16:00"), at("20:00"), True)
    closed = opening("ev-park-hours", AUTHORITY, PARK, closed_claim())
    evidence = [*without_id(park_evidence(WIDE_OPEN), "ev-park-hours"), closed, rain]

    result = run(request_doc(), park_candidate(60, 30, 30), evidence)

    assert constraint_ids(result) == {ConstraintId.OPENING_ACCESS}


# --- explicit constraints and transport (R7, R9, O1, AC7) ---------------------------


def with_constraints(constraints: list[str]) -> Doc:
    return cast(Doc, replaced(request_doc(), ("constraints",), constraints))


def test_constraint_without_supporting_evidence_is_rejected() -> None:
    result = run(
        with_constraints(["Include a park"]), park_candidate(60, 30, 30), park_evidence(WIDE_OPEN)
    )

    failure = only_failure(result, ConstraintId.EXPLICIT_CONSTRAINT)
    assert failure.inputs["constraint_index"] == "0"
    assert failure.inputs["constraint"] == "Include a park"


def test_cited_constraint_evidence_satisfies_the_constraint() -> None:
    candidate = cite(park_candidate(60, 30, 30), ["ev-park-is-park"])
    evidence = [*park_evidence(WIDE_OPEN), constraint_support("ev-park-is-park", PARK, 0)]

    checked(run(with_constraints(["Include a park"]), candidate, evidence))


def test_uncited_constraint_evidence_does_not_count() -> None:
    evidence = [*park_evidence(WIDE_OPEN), constraint_support("ev-park-is-park", PARK, 0)]

    result = run(with_constraints(["Include a park"]), park_candidate(60, 30, 30), evidence)

    assert constraint_ids(result) == {ConstraintId.EXPLICIT_CONSTRAINT}


def test_constraint_evidence_must_name_the_right_constraint() -> None:
    candidate = cite(park_candidate(60, 30, 30), ["ev-coffee"])
    evidence = [*park_evidence(WIDE_OPEN), constraint_support("ev-coffee", PARK, 1)]

    result = run(with_constraints(["Include a park", "Get coffee"]), candidate, evidence)

    failure = only_failure(result, ConstraintId.EXPLICIT_CONSTRAINT)
    assert failure.inputs["constraint_index"] == "0"


def test_constraint_evidence_for_a_place_not_in_the_plan_does_not_count() -> None:
    candidate = cite(park_candidate(60, 30, 30), ["ev-elsewhere"])
    evidence = [*park_evidence(WIDE_OPEN), constraint_support("ev-elsewhere", OTHER_PLACE, 0)]

    result = run(with_constraints(["Include a park"]), candidate, evidence)

    assert constraint_ids(result) == {ConstraintId.EXPLICIT_CONSTRAINT}


# An Estimate supports a constraint only when based_on is non-empty and every id in it
# is a SourcedFinding (decision 2026-11-18); a bare AI best guess never does.


def estimated_support(evidence_id: str, label: str, based_on: list[str]) -> Doc:
    claim: Doc = {"kind": "constraint", "constraint_index": 0}
    item = estimate(evidence_id, "constraint", label, scope(PARK, DAY_FROM, DAY_UNTIL), claim)
    return {**item, "based_on": based_on}


def run_with_estimated_support(extra: list[Doc]) -> CheckedItinerary | ValidationFeedback:
    """Park plan with constraint 0 'Include a park', citing the estimate 'ev-support'."""
    candidate = cite(
        park_candidate(60, 30, 30), ["ev-park-hours", "ev-park-duration", "ev-support"]
    )
    return run(with_constraints(["Include a park"]), candidate, [*park_evidence(WIDE_OPEN), *extra])


@pytest.mark.parametrize("label", ["estimate", "ai_best_guess"])
def test_estimate_without_a_basis_does_not_support_a_constraint(label: str) -> None:
    result = run_with_estimated_support([estimated_support("ev-support", label, [])])

    failure = only_failure(result, ConstraintId.EXPLICIT_CONSTRAINT)
    assert failure.inputs["constraint_index"] == "0"


@pytest.mark.parametrize("label", ["estimate", "ai_best_guess"])
def test_estimate_based_on_sourced_findings_supports_a_constraint(label: str) -> None:
    support = estimated_support("ev-support", label, ["ev-park-hours", "ev-park-price"])

    checked(run_with_estimated_support([support]))


def test_estimate_based_on_unknown_evidence_does_not_support_a_constraint() -> None:
    result = run_with_estimated_support([estimated_support("ev-support", "estimate", ["ev-nope"])])

    assert constraint_ids(result) == {ConstraintId.EXPLICIT_CONSTRAINT}


def test_estimate_based_only_on_another_estimate_does_not_support_a_constraint() -> None:
    other = estimated_support("ev-other-guess", "ai_best_guess", [])
    support = estimated_support("ev-support", "estimate", ["ev-other-guess"])

    result = run_with_estimated_support([other, support])

    assert constraint_ids(result) == {ConstraintId.EXPLICIT_CONSTRAINT}


def test_estimate_with_any_unsourced_basis_does_not_support_a_constraint() -> None:
    other = estimated_support("ev-other-guess", "ai_best_guess", [])
    support = estimated_support("ev-support", "estimate", ["ev-park-hours", "ev-other-guess"])

    result = run_with_estimated_support([other, support])

    assert constraint_ids(result) == {ConstraintId.EXPLICIT_CONSTRAINT}


def test_leg_mode_outside_the_requested_modes_is_rejected() -> None:
    request = replaced(request_doc(), ("transport_modes",), ["walking"])

    result = run(request, transit_candidate([]), transit_evidence())

    failure = only_failure(result, ConstraintId.TRANSPORT_AVAILABILITY)
    assert failure.inputs["leg_index"] == "1"
    assert failure.inputs["mode"] == "public_transit"


def test_unstated_transport_modes_do_not_restrict_legs() -> None:
    request = replaced(request_doc(), ("transport_modes",), [])

    checked(run(request, transit_candidate([]), transit_evidence()))


# --- structured results (AC8) ------------------------------------------------------


def test_independent_failures_are_all_reported() -> None:
    request = replaced(with_budget("100"), ("transport_modes",), ["walking"])

    result = run(request, transit_candidate([(3, 10)]), transit_evidence())

    assert isinstance(result, ValidationFeedback)
    assert result.candidate_id == "cand-test"
    assert {
        ConstraintId.BUDGET,
        ConstraintId.TRANSPORT_AVAILABILITY,
        ConstraintId.TRANSFER_TIME,
    } <= constraint_ids(result)


def test_unknown_cited_evidence_id_is_reported() -> None:
    candidate = cite(park_candidate(60, 30, 30), ["ev-park-hours", "ev-does-not-exist"])

    result = run(request_doc(), candidate, park_evidence(WIDE_OPEN))

    failure = only_failure(result, ConstraintId.MISSING_EVIDENCE)
    assert failure.inputs["reason"] == "unknown_evidence_id"
    assert failure.inputs["evidence_id"] == "ev-does-not-exist"


def test_checked_itinerary_keeps_the_candidate_unchanged() -> None:
    candidate = park_candidate(60, 30, 30)

    result = checked(run(request_doc(), candidate, park_evidence(WIDE_OPEN)))

    assert result.candidate == CandidateItinerary.model_validate_json(json.dumps(candidate))


def test_phase_0_sample_itinerary_checks_to_its_sample_result() -> None:
    fixture = (FIXTURES_DIR / "turn_result_validated_options.json").read_text(encoding="utf-8")
    expected = TURN_RESULT.validate_json(fixture)
    assert isinstance(expected, ValidatedOptions)
    sample = expected.options[0].itinerary

    result = check_candidate(
        PlanningRequest.model_validate_json((FIXTURES_DIR / "planning_request.json").read_text()),
        CandidateItinerary.model_validate_json(
            (FIXTURES_DIR / "candidate_itinerary.json").read_text()
        ),
        EVIDENCE_LIST.validate_json((FIXTURES_DIR / "evidence.json").read_text()),
    )

    result = checked(result)
    assert result.candidate == sample.candidate
    assert result.leave_by == sample.leave_by
    assert result.arrival_at_commitment == sample.arrival_at_commitment
    assert result.total_cost == sample.total_cost
    assert result.total_includes_estimates is True
    assert len(result.weather_warnings) == 1


# --- stops with no Place ID (user decision D4, 2026-11-18) -------------------------
# A fallback stop is identified by the research result it came from plus coordinates read
# from that page. Its opening/access needs an applicable official source: no Maps fallback.

FALLBACK_KEY: Doc = {"result_id": "web-1", "latitude": 35.66, "longitude": 139.70}
FALLBACK_STOP_KEY = "web-1@35.66,139.7"


def as_fallback_stop(candidate: Doc, key: Doc) -> Doc:
    """Return `candidate` with stop 0 identified by `key` instead of a Place ID."""
    without_place = replaced(candidate, ("stops", 0, "place_id"), None)
    return cast(Doc, replaced(without_place, ("stops", 0, "fallback_key"), key))


def scoped_to(item: Doc, key: Doc) -> Doc:
    """Return evidence `item` scoped to the fallback key instead of a Place ID."""
    evidence_scope = cast(Doc, item["scope"])
    return {**item, "scope": {**evidence_scope, "place_id": None, "fallback_key": key}}


def fallback_park() -> Doc:
    return as_fallback_stop(park_candidate(60, 30, 30), FALLBACK_KEY)


def test_fallback_stop_with_an_official_source_is_checked_like_a_place() -> None:
    evidence = [scoped_to(item, FALLBACK_KEY) for item in park_evidence(WIDE_OPEN)]

    result = checked(run(request_doc(), fallback_park(), evidence))

    assert result.candidate.stops[0].place_id is None
    assert result.total_cost.amount == Decimal("500")
    assert result.leave_by == (tokyo(16, 45), tokyo(18, 15))


@pytest.mark.parametrize("kind", [MAPS, OTHER_WEB])
def test_fallback_stop_without_an_official_opening_source_is_rejected(kind: str) -> None:
    evidence = [
        scoped_to(opening("ev-park-hours", kind, PARK, open_claim(WIDE_OPEN)), FALLBACK_KEY),
        scoped_to(duration("ev-park-duration", OTHER_WEB, PARK, 60), FALLBACK_KEY),
        scoped_to(price("ev-park-price", PARK, "500", "JPY"), FALLBACK_KEY),
    ]

    result = run(request_doc(), fallback_park(), evidence)

    failure = only_failure(result, ConstraintId.OPENING_ACCESS)
    assert failure.inputs["reason"] == "unverified"
    assert failure.inputs["stop_index"] == "0"
    assert failure.inputs["place_id"] == FALLBACK_STOP_KEY


def test_maps_hours_do_not_override_an_official_source_for_a_fallback_stop() -> None:
    evidence = [
        scoped_to(opening("ev-maps-hours", MAPS, PARK, closed_claim()), FALLBACK_KEY),
        *(scoped_to(item, FALLBACK_KEY) for item in park_evidence(WIDE_OPEN)),
    ]

    checked(run(request_doc(), fallback_park(), evidence))


@pytest.mark.parametrize(
    "evidence_key",
    [
        pytest.param({**FALLBACK_KEY, "result_id": "web-2"}, id="other-result"),
        pytest.param({**FALLBACK_KEY, "latitude": 35.661}, id="other-coordinates"),
    ],
)
def test_evidence_for_another_fallback_key_does_not_apply(evidence_key: Doc) -> None:
    evidence = [scoped_to(item, evidence_key) for item in park_evidence(WIDE_OPEN)]

    result = run(request_doc(), fallback_park(), evidence)

    assert only_failure(result, ConstraintId.OPENING_ACCESS).inputs["reason"] == "unverified"


def test_place_id_evidence_does_not_apply_to_a_fallback_stop() -> None:
    result = run(request_doc(), fallback_park(), park_evidence(WIDE_OPEN))

    failure = only_failure(result, ConstraintId.OPENING_ACCESS)
    assert failure.inputs["reason"] == "unverified"
    assert failure.inputs["place_id"] == FALLBACK_STOP_KEY


def test_evidence_scoped_at_zero_applies_to_a_stop_at_negative_zero() -> None:
    stop_at_minus_zero: Doc = {"result_id": "web-1", "latitude": -0.0, "longitude": 139.70}
    candidate = as_fallback_stop(park_candidate(60, 30, 30), stop_at_minus_zero)
    candidate = cast(Doc, replaced(candidate, ("stops", 0, "latitude"), -0.0))
    scope_at_zero: Doc = {"result_id": "web-1", "latitude": 0.0, "longitude": 139.70}
    evidence = [scoped_to(item, scope_at_zero) for item in park_evidence(WIDE_OPEN)]

    checked(run(request_doc(), candidate, evidence))
