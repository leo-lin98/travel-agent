"""Phase 1 acceptance tests for distinct option selection.

Expectations derive from docs/requirements.md R2, the Phase 1 document's distinctness
edge cases, the 2026-11-18 decisions (D3 min-max travel score, D4 strict slots), ledger
rulings O4-O5, and artifacts/phase-1/interface-contract.md section 3.
Inputs are synthetic CheckedItinerary values; selection never re-checks feasibility.
"""

import json
import subprocess
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import pytest
from pydantic import TypeAdapter

from travel_agent.contracts import (
    CandidateItinerary,
    CheckedItinerary,
    ConstraintId,
    Evidence,
    OptionLabel,
    PlanningRequest,
    Recommendation,
    ValidatedOption,
    ValidationFeedback,
)
from travel_agent.selection import (
    MAX_SELECTION_CANDIDATES,
    DuplicateCandidateIdError,
    MixedCurrencyError,
    RecommendationNotFoundError,
    SelectionError,
    TooManyCandidatesError,
    select_options,
)

type Doc = dict[str, object]

P1 = "ChIJsyntheticPlaceOne0001"
P2 = "ChIJsyntheticPlaceTwo0002"
P3 = "ChIJsyntheticPlaceThree03"
P4 = "ChIJsyntheticPlaceFour004"
P5 = "ChIJsyntheticPlaceFive005"
HOTEL = "ChIJsyntheticGrandHotel01"
HOTEL_CAFE = "ChIJsyntheticHotelCafe001"
SORTS_FIRST = "ChIJsyntheticAaaSortsFirst"
SORTS_SECOND = "ChIJsyntheticZzzSortsLast0"

AI = OptionLabel.AI_RECOMMENDED
CHEAPEST = OptionLabel.CHEAPEST
BEST_TRAVEL = OptionLabel.BEST_TRAVEL


# --- builders (pure; every parameter explicit) ------------------------------------


def stop(place_id: str, label: str) -> Doc:
    return {
        "place_id": place_id,
        "label": label,
        "latitude": 35.66,
        "longitude": 139.70,
        "visit_minutes": 30,
        "evidence_ids": [],
    }


def leg(mode: str, minutes: int, meters: int) -> Doc:
    return {
        "mode": mode,
        "duration_minutes": minutes,
        "distance_meters": meters,
        "transfers": [],
        "evidence_ids": [],
    }


def checked_from(
    candidate_id: str, stops: list[Doc], legs: list[Doc], total: str, currency: str
) -> CheckedItinerary:
    """Build a CheckedItinerary through JSON; schedule values are placeholders."""
    document = {
        "candidate": {"candidate_id": candidate_id, "stops": stops, "legs": legs, "costs": []},
        "leave_by": ["2026-11-20T15:00:00+09:00"] * (len(stops) + 1),
        "arrival_at_commitment": "2026-11-20T18:00:00+09:00",
        "total_cost": {"amount": total, "currency": currency},
        "total_includes_estimates": False,
        "weather_warnings": [],
    }
    return CheckedItinerary.model_validate_json(json.dumps(document))


def plan(
    candidate_id: str, places: list[str], minutes: int, meters: int, total: str
) -> CheckedItinerary:
    """A walking plan whose first leg carries all travel minutes and distance (JPY)."""
    stops = [stop(place_id, f"Place {place_id[-4:]}") for place_id in places]
    legs = [leg("walking", minutes, meters)] + [leg("walking", 0, 0) for _ in places]
    return checked_from(candidate_id, stops, legs, total, "JPY")


def recommend(candidate_id: str) -> Recommendation:
    return Recommendation.model_validate_json(
        json.dumps({"candidate_id": candidate_id, "reason": "Best match for the request."})
    )


def ids(options: tuple[ValidatedOption, ...]) -> list[str]:
    return [option.itinerary.candidate.candidate_id for option in options]


def labels(options: tuple[ValidatedOption, ...]) -> list[tuple[OptionLabel, ...]]:
    return [option.labels for option in options]


# --- strict slots (R2, D4, O5) -------------------------------------------------------


def test_three_distinct_plans_fill_the_three_slots_in_order() -> None:
    checked = (
        plan("rec", [P1], 60, 3000, "2000"),
        plan("cheap", [P2], 90, 5000, "500"),
        plan("quick", [P3], 20, 1000, "1500"),
        plan("slow", [P4], 100, 6000, "1000"),
    )

    options = select_options(checked, recommend("rec"))

    assert ids(options) == ["rec", "cheap", "quick"]
    assert labels(options) == [(AI,), (CHEAPEST,), (BEST_TRAVEL,)]


def test_only_the_ai_option_carries_the_recommendation_reason() -> None:
    checked = (
        plan("rec", [P1], 60, 3000, "2000"),
        plan("cheap", [P2], 90, 5000, "500"),
        plan("quick", [P3], 20, 1000, "1500"),
    )

    options = select_options(checked, recommend("rec"))

    assert options[0].recommendation_reason == "Best match for the request."
    assert [option.recommendation_reason for option in options[1:]] == [None, None]


def test_options_keep_the_checked_itineraries_unchanged() -> None:
    checked = (plan("rec", [P1], 60, 3000, "2000"), plan("cheap", [P2], 90, 5000, "500"))

    options = select_options(checked, recommend("rec"))

    assert [option.itinerary for option in options] == list(checked)


def test_cheapest_is_chosen_among_the_remaining_plans() -> None:
    # The recommended plan is the cheapest overall; it keeps only its AI label.
    checked = (
        plan("rec", [P1], 60, 3000, "100"),
        plan("mid", [P2], 60, 3000, "700"),
        plan("pricey", [P3], 60, 3000, "900"),
    )

    options = select_options(checked, recommend("rec"))

    assert ids(options) == ["rec", "mid", "pricey"]
    assert labels(options) == [(AI,), (CHEAPEST,), (BEST_TRAVEL,)]


def test_best_travel_excludes_the_cheapest_pick() -> None:
    checked = (
        plan("rec", [P1], 60, 3000, "2000"),
        plan("cheap-and-quick", [P2], 10, 500, "100"),
        plan("medium", [P3], 40, 2000, "900"),
        plan("slow", [P4], 90, 6000, "800"),
    )

    options = select_options(checked, recommend("rec"))

    assert ids(options) == ["rec", "cheap-and-quick", "medium"]


def test_one_distinct_plan_gives_one_option() -> None:
    checked = (plan("rec", [P1], 60, 3000, "2000"),)

    options = select_options(checked, recommend("rec"))

    assert ids(options) == ["rec"]
    assert labels(options) == [(AI,)]


def test_two_distinct_plans_leave_the_best_travel_slot_empty() -> None:
    checked = (
        plan("rec", [P1], 60, 3000, "2000"),
        plan("other", [P2], 10, 500, "900"),
        plan("other-again", [P2], 5, 100, "800"),
    )

    options = select_options(checked, recommend("rec"))

    assert ids(options) == ["rec", "other-again"]
    assert labels(options) == [(AI,), (CHEAPEST,)]


# --- distinctness (R2, O4) ----------------------------------------------------------


def test_same_places_in_another_order_or_transport_are_one_plan() -> None:
    reordered_stops = [stop(P2, "Place 0002"), stop(P1, "Place 0001")]
    by_transit = [
        leg("public_transit", 5, 3000),
        leg("taxi_rideshare", 5, 3000),
        leg("walking", 1, 50),
    ]
    checked = (
        plan("rec", [P1, P2], 60, 3000, "2000"),
        checked_from("reordered", reordered_stops, by_transit, "100", "JPY"),
        plan("same-again", [P1, P2], 30, 1000, "200"),
    )

    options = select_options(checked, recommend("rec"))

    assert ids(options) == ["rec"]


def test_same_place_under_different_names_is_one_plan() -> None:
    checked = (
        plan("rec", [P1], 60, 3000, "2000"),
        checked_from(
            "renamed",
            [stop(P1, "Totally different display name")],
            [leg("walking", 10, 500), leg("walking", 10, 500)],
            "100",
            "JPY",
        ),
    )

    options = select_options(checked, recommend("rec"))

    assert ids(options) == ["rec"]


def test_plans_differing_in_one_activity_place_are_distinct() -> None:
    checked = (
        plan("rec", [P1, P2], 60, 3000, "2000"),
        plan("swap-one", [P1, P3], 60, 3000, "1000"),
    )

    options = select_options(checked, recommend("rec"))

    assert ids(options) == ["rec", "swap-one"]


def test_cafe_inside_a_hotel_and_the_hotel_are_distinct() -> None:
    checked = (
        checked_from(
            "hotel",
            [stop(HOTEL, "Grand Hotel")],
            [leg("walking", 10, 500), leg("walking", 10, 500)],
            "1500",
            "JPY",
        ),
        checked_from(
            "hotel-cafe",
            [stop(HOTEL_CAFE, "Lobby Cafe at Grand Hotel")],
            [leg("walking", 10, 500), leg("walking", 10, 500)],
            "900",
            "JPY",
        ),
    )

    options = select_options(checked, recommend("hotel"))

    assert ids(options) == ["hotel", "hotel-cafe"]


def test_cheapest_never_repeats_the_recommended_plan_under_another_id() -> None:
    checked = (
        plan("rec", [P1, P2], 60, 3000, "2000"),
        plan("rec-reordered", [P2, P1], 60, 3000, "100"),
        plan("other", [P3], 60, 3000, "900"),
    )

    options = select_options(checked, recommend("rec"))

    assert ids(options) == ["rec", "other"]


def test_cheapest_group_is_represented_by_its_cheapest_member() -> None:
    checked = (
        plan("rec", [P1], 60, 3000, "2000"),
        plan("group-a", [P2, P3], 60, 3000, "300"),
        plan("group-b", [P3, P2], 60, 3000, "200"),
        plan("other", [P4], 60, 3000, "900"),
    )

    options = select_options(checked, recommend("rec"))

    assert ids(options) == ["rec", "group-b", "other"]


def test_best_travel_group_is_represented_by_its_best_member() -> None:
    checked = (
        plan("rec", [P1], 60, 1000, "2000"),
        plan("cheap", [P2], 60, 1000, "100"),
        plan("g-slow", [P3, P4], 50, 1000, "900"),
        plan("g-fast", [P4, P3], 20, 1000, "900"),
        plan("h", [P5], 30, 1000, "900"),
    )

    options = select_options(checked, recommend("rec"))

    assert ids(options) == ["rec", "cheap", "g-fast"]


# --- cheapest ties (routine choice RC7) -----------------------------------------------


def test_cheapest_tie_goes_to_fewer_travel_minutes() -> None:
    checked = (
        plan("rec", [P1], 60, 3000, "2000"),
        plan("slower", [P2], 50, 1000, "500"),
        plan("faster", [P3], 40, 9000, "500"),
    )

    assert ids(select_options(checked, recommend("rec")))[1] == "faster"


def test_cheapest_tie_then_goes_to_shorter_distance() -> None:
    checked = (
        plan("rec", [P1], 60, 3000, "2000"),
        plan("farther", [P2], 40, 2000, "500"),
        plan("nearer", [P3], 40, 1000, "500"),
    )

    assert ids(select_options(checked, recommend("rec")))[1] == "nearer"


def test_cheapest_tie_then_goes_to_sorted_place_ids() -> None:
    checked = (
        plan("rec", [P1], 60, 3000, "2000"),
        plan("z-id-first-place", [SORTS_FIRST], 40, 1000, "500"),
        plan("a-id-last-place", [SORTS_SECOND], 40, 1000, "500"),
    )

    assert ids(select_options(checked, recommend("rec")))[1] == "z-id-first-place"


# --- best travel score (R2, D3, AC9) ------------------------------------------------


def test_travel_score_balances_distance_and_minutes_equally() -> None:
    # Remaining after the cheapest pick: balanced (0.25 + 0.25) beats lopsided (0 + 1).
    checked = (
        plan("rec", [P1], 60, 3000, "2000"),
        plan("cheap", [P2], 60, 3000, "100"),
        plan("shortest", [P3], 100, 1000, "900"),
        plan("fastest", [P4], 20, 5000, "900"),
        plan("balanced", [P5], 40, 2000, "900"),
    )

    assert ids(select_options(checked, recommend("rec")))[2] == "balanced"


def test_travel_score_uses_exact_arithmetic_for_ties() -> None:
    # Distance range 1000..2000, minutes 10..110 among the remaining plans.
    # "a": 0.5*0.2 + 0.5*0.1 = 0.15; "b": 0.5*0.0 + 0.5*0.3 = 0.15 -> exact tie.
    # Floating point makes "a" 0.15000000000000002; the tie must go to fewer minutes ("a").
    checked = (
        plan("rec", [P1], 60, 1500, "2000"),
        plan("cheap", [P2], 60, 1500, "100"),
        plan("x", [P3], 110, 1000, "900"),
        plan("y", [P4], 10, 2000, "900"),
        plan("a", [P5], 20, 1200, "900"),
        plan("b", [HOTEL], 40, 1000, "900"),
    )

    assert ids(select_options(checked, recommend("rec")))[2] == "a"


def test_travel_score_normalises_over_the_remaining_plans_only() -> None:
    # Over b1/b2 only: both score 0.5, tie -> fewer minutes -> b2.
    # Wrongly including the cheapest pick (distance 0, 100 minutes) would make b1 win.
    checked = (
        plan("rec", [P1], 45, 1500, "2000"),
        plan("cheap", [P2], 100, 0, "100"),
        plan("b1", [P3], 50, 1000, "500"),
        plan("b2", [P4], 40, 2000, "600"),
    )

    assert ids(select_options(checked, recommend("rec"))) == ["rec", "cheap", "b2"]


def test_equal_measures_score_zero_and_fall_to_lower_cost() -> None:
    checked = (
        plan("rec", [P1], 60, 3000, "2000"),
        plan("cheap", [P2], 60, 3000, "100"),
        plan("costlier", [P3], 30, 1000, "900"),
        plan("cheaper", [P4], 30, 1000, "500"),
    )

    options = select_options(checked, recommend("rec"))

    assert ids(options)[1:] == ["cheap", "cheaper"]


def test_full_tie_falls_to_sorted_place_ids() -> None:
    checked = (
        plan("rec", [P1], 60, 3000, "2000"),
        plan("cheap", [P2], 60, 3000, "100"),
        plan("z-id-first-place", [SORTS_FIRST], 30, 1000, "900"),
        plan("a-id-last-place", [SORTS_SECOND], 30, 1000, "900"),
    )

    assert ids(select_options(checked, recommend("rec")))[2] == "z-id-first-place"


def test_travel_totals_count_every_leg_through_the_commitment() -> None:
    def one_stop(candidate_id: str, place_id: str, legs: list[Doc]) -> CheckedItinerary:
        return checked_from(candidate_id, [stop(place_id, "Stop")], legs, "900", "JPY")

    checked = (
        plan("rec", [P1], 60, 3000, "2000"),
        plan("cheap", [P2], 60, 3000, "100"),
        one_stop("even", P3, [leg("walking", 10, 400), leg("walking", 10, 400)]),
        one_stop("heavy-last-leg", P4, [leg("walking", 1, 50), leg("taxi_rideshare", 60, 5000)]),
        one_stop("single", P5, [leg("walking", 25, 1100), leg("walking", 0, 0)]),
    )

    # Totals: even 20 min / 800 m, single 25 / 1100, heavy 61 / 5050 -> even wins.
    # Dropping the final leg to the commitment would wrongly make "heavy-last-leg" win.
    assert ids(select_options(checked, recommend("rec")))[2] == "even"


# --- invalid input (O5, bounded collections) --------------------------------------


def test_unknown_recommendation_raises() -> None:
    checked = (plan("rec", [P1], 60, 3000, "2000"),)

    with pytest.raises(RecommendationNotFoundError):
        select_options(checked, recommend("not-validated"))


def test_empty_candidate_set_raises_for_any_recommendation() -> None:
    with pytest.raises(RecommendationNotFoundError):
        select_options((), recommend("rec"))


def test_duplicate_candidate_ids_raise() -> None:
    checked = (plan("rec", [P1], 60, 3000, "2000"), plan("rec", [P2], 60, 3000, "100"))

    with pytest.raises(DuplicateCandidateIdError):
        select_options(checked, recommend("rec"))


def test_candidate_count_is_bounded() -> None:
    at_limit = tuple(
        plan(f"cand-{index}", [f"ChIJsyntheticBulk{index:04d}"], 60, 3000, "100")
        for index in range(MAX_SELECTION_CANDIDATES)
    )
    over_limit = (*at_limit, plan("one-more", [P1], 60, 3000, "100"))

    assert len(select_options(at_limit, recommend("cand-0"))) == 3
    with pytest.raises(TooManyCandidatesError):
        select_options(over_limit, recommend("cand-0"))


def test_mixed_total_currencies_raise() -> None:
    usd = checked_from(
        "usd", [stop(P2, "Two")], [leg("walking", 10, 500), leg("walking", 10, 500)], "5", "USD"
    )
    checked = (plan("rec", [P1], 60, 3000, "2000"), usd)

    with pytest.raises(MixedCurrencyError):
        select_options(checked, recommend("rec"))


def test_selection_errors_share_a_base_class() -> None:
    for error in (
        RecommendationNotFoundError,
        DuplicateCandidateIdError,
        TooManyCandidatesError,
        MixedCurrencyError,
    ):
        assert issubclass(error, SelectionError)
    assert issubclass(SelectionError, ValueError)


# --- replay script on the deadline-boundary case (AC10) ------------------------------
# Runs backend/scripts/replay.py as a subprocess with the test interpreter, end to end.

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPLAY_SCRIPT = BACKEND_DIR / "scripts" / "replay.py"
DEADLINE_CASE = BACKEND_DIR / "tests" / "fixtures" / "deadline_boundary.json"
TOKYO = ZoneInfo("Asia/Tokyo")


def test_deadline_case_fixture_is_valid_contract_data() -> None:
    case = cast(dict[str, object], json.loads(DEADLINE_CASE.read_text(encoding="utf-8")))

    assert set(case) == {"request", "evidence", "candidates", "recommendation"}
    PlanningRequest.model_validate_json(json.dumps(case["request"]))
    TypeAdapter(tuple[Evidence, ...]).validate_json(json.dumps(case["evidence"]))
    candidates = TypeAdapter(tuple[CandidateItinerary, ...]).validate_json(
        json.dumps(case["candidates"])
    )
    recommendation = Recommendation.model_validate_json(json.dumps(case["recommendation"]))
    assert recommendation.candidate_id in {item.candidate_id for item in candidates}


def test_replay_writes_schedule_totals_selections_and_failure_reasons(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "phase-1-example.json"

    completed = subprocess.run(
        [sys.executable, str(REPLAY_SCRIPT), "--case", str(DEADLINE_CASE), "--output", str(output)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = cast(dict[str, object], json.loads(output.read_text(encoding="utf-8")))
    assert set(report) == {"checked", "rejected", "options"}
    passed = TypeAdapter(tuple[CheckedItinerary, ...]).validate_json(json.dumps(report["checked"]))
    rejected = TypeAdapter(tuple[ValidationFeedback, ...]).validate_json(
        json.dumps(report["rejected"])
    )
    options = TypeAdapter(tuple[ValidatedOption, ...]).validate_json(json.dumps(report["options"]))

    assert [item.candidate.candidate_id for item in passed] == [
        "cand-garden-on-time",
        "cand-museum-transit",
        "cand-museum-taxi",
        "cand-hotel-cafe",
        "cand-hotel",
    ]
    on_time = passed[0]
    assert on_time.arrival_at_commitment == datetime(2026, 11, 20, 18, 45, tzinfo=TOKYO)
    assert on_time.leave_by == (
        datetime(2026, 11, 20, 15, 45, tzinfo=TOKYO),
        datetime(2026, 11, 20, 18, 15, tzinfo=TOKYO),
    )
    assert on_time.total_cost.amount == Decimal("1200")
    assert passed[2].total_cost.amount == Decimal("2900")
    assert passed[2].total_includes_estimates is True

    assert [item.candidate_id for item in rejected] == ["cand-garden-late"]
    assert {failure.constraint for failure in rejected[0].failures} == {ConstraintId.ARRIVAL_BUFFER}

    assert ids(options) == ["cand-garden-on-time", "cand-hotel", "cand-hotel-cafe"]
    assert labels(options) == [(AI,), (CHEAPEST,), (BEST_TRAVEL,)]
