"""Pure Python selection of up to three distinct options from checked itineraries (Phase 1).

Input is only Python-validated ``CheckedItinerary`` values; selection never re-checks
feasibility. Slots are strict and filled in order (R2, decision D4): AI Recommended,
then Cheapest of the remaining plans, then Best travel of the remaining plans. Each
option carries one label, and fewer options are returned when fewer distinct plans
exist.

Two itineraries are the same plan when they visit the same set of activity-stop Place
IDs (ledger O4); labels, coordinates, stop order, and transport are ignored. Each slot
skips every itinerary whose plan was already shown, using the plan key as a lookup.
"""

from collections import Counter
from decimal import Decimal
from fractions import Fraction
from typing import Final

from travel_agent.contracts import (
    CheckedItinerary,
    OptionLabel,
    Recommendation,
    ValidatedOption,
)

MAX_SELECTION_CANDIDATES: Final[int] = 20

type PlanKey = frozenset[str]
type CheapestKey = tuple[Decimal, int, int, tuple[str, ...], str]
type BestTravelKey = tuple[Fraction, int, int, Decimal, tuple[str, ...], str]

_HALF: Final[Fraction] = Fraction(1, 2)


# --- errors ------------------------------------------------------------------------


class SelectionError(ValueError):
    """The checked itineraries or the recommendation cannot be selected from."""


class RecommendationNotFoundError(SelectionError):
    """The recommendation does not reference any checked itinerary."""


class DuplicateCandidateIdError(SelectionError):
    """Two checked itineraries share a ``candidate_id``."""


class TooManyCandidatesError(SelectionError):
    """More checked itineraries than ``MAX_SELECTION_CANDIDATES``."""


class MixedCurrencyError(SelectionError):
    """Checked itinerary totals are in more than one currency, so they cannot be compared."""


# --- per-itinerary measures ----------------------------------------------------------


def _plan_key(itinerary: CheckedItinerary) -> PlanKey:
    """The set of activity-stop Place IDs; equal keys mean the same plan (O4)."""
    return frozenset(stop.place_id for stop in itinerary.candidate.stops)


def _sorted_places(itinerary: CheckedItinerary) -> tuple[str, ...]:
    return tuple(sorted(_plan_key(itinerary)))


def _travel_minutes(itinerary: CheckedItinerary) -> int:
    """Total minutes over every leg, including the final leg to the commitment."""
    return sum(leg.duration_minutes for leg in itinerary.candidate.legs)


def _distance_meters(itinerary: CheckedItinerary) -> int:
    """Total meters over every leg, including the final leg to the commitment."""
    return sum(leg.distance_meters for leg in itinerary.candidate.legs)


def _candidate_id(itinerary: CheckedItinerary) -> str:
    return itinerary.candidate.candidate_id


# --- input checks ------------------------------------------------------------------


def _require_bounded(checked: tuple[CheckedItinerary, ...]) -> None:
    if len(checked) > MAX_SELECTION_CANDIDATES:
        raise TooManyCandidatesError(
            f"got {len(checked)} checked itineraries; at most "
            f"{MAX_SELECTION_CANDIDATES} may be selected from"
        )


def _require_unique_ids(checked: tuple[CheckedItinerary, ...]) -> None:
    counts = Counter(_candidate_id(itinerary) for itinerary in checked)
    duplicates = sorted(candidate_id for candidate_id, count in counts.items() if count > 1)
    if duplicates:
        raise DuplicateCandidateIdError(f"duplicate candidate_id values: {duplicates}")


def _require_one_currency(checked: tuple[CheckedItinerary, ...]) -> None:
    currencies = sorted({itinerary.total_cost.currency for itinerary in checked})
    if len(currencies) > 1:
        raise MixedCurrencyError(f"total_cost currencies differ: {currencies}")


def _recommended(
    checked: tuple[CheckedItinerary, ...], recommendation: Recommendation
) -> CheckedItinerary:
    """The checked itinerary the recommendation names (O5), else an explicit error."""
    for itinerary in checked:
        if _candidate_id(itinerary) == recommendation.candidate_id:
            return itinerary
    raise RecommendationNotFoundError(
        f"recommendation candidate_id {recommendation.candidate_id!r} is not among the "
        f"checked itineraries {[_candidate_id(itinerary) for itinerary in checked]}"
    )


# --- slots -------------------------------------------------------------------------


def _unused(
    checked: tuple[CheckedItinerary, ...], used: frozenset[PlanKey]
) -> tuple[CheckedItinerary, ...]:
    """Every itinerary whose plan has not been shown yet (all members of each group)."""
    return tuple(itinerary for itinerary in checked if _plan_key(itinerary) not in used)


def _cheapest_key(itinerary: CheckedItinerary) -> CheapestKey:
    return (
        itinerary.total_cost.amount,
        _travel_minutes(itinerary),
        _distance_meters(itinerary),
        _sorted_places(itinerary),
        _candidate_id(itinerary),
    )


def _cheapest(eligible: tuple[CheckedItinerary, ...]) -> CheckedItinerary | None:
    """The cheapest eligible itinerary (RC7); see ``select_options`` for the tie chain."""
    return min(eligible, key=_cheapest_key, default=None)


def _normalised(value: int, low: int, high: int) -> Fraction:
    """Min-max rescale of ``value`` to 0..1 over ``[low, high]``; 0 when ``low == high``."""
    if high == low:
        return Fraction(0)
    return Fraction(value - low, high - low)


def _best_travel_keys(eligible: tuple[CheckedItinerary, ...]) -> tuple[BestTravelKey, ...]:
    """Sort keys for the best-travel slot; ranges are computed once over ``eligible``."""
    distances = tuple(_distance_meters(itinerary) for itinerary in eligible)
    minutes = tuple(_travel_minutes(itinerary) for itinerary in eligible)
    d_low, d_high = min(distances), max(distances)
    t_low, t_high = min(minutes), max(minutes)
    return tuple(
        (
            _HALF * _normalised(distance, d_low, d_high)
            + _HALF * _normalised(minute, t_low, t_high),
            minute,
            distance,
            itinerary.total_cost.amount,
            _sorted_places(itinerary),
            _candidate_id(itinerary),
        )
        for itinerary, distance, minute in zip(eligible, distances, minutes, strict=True)
    )


def _best_travel(eligible: tuple[CheckedItinerary, ...]) -> CheckedItinerary | None:
    """The eligible itinerary with the lowest 50/50 travel score; see ``select_options``."""
    if not eligible:
        return None
    keys = _best_travel_keys(eligible)
    best_index = min(range(len(eligible)), key=lambda index: keys[index])
    return eligible[best_index]


def _option(itinerary: CheckedItinerary, label: OptionLabel) -> ValidatedOption:
    return ValidatedOption(itinerary=itinerary, labels=(label,), recommendation_reason=None)


def select_options(
    checked: tuple[CheckedItinerary, ...],
    recommendation: Recommendation,
) -> tuple[ValidatedOption, ...]:
    """Pick up to three distinct options in slot order: AI, Cheapest, Best travel.

    Itineraries with the same set of activity-stop Place IDs are one plan (O4). Each
    slot considers only itineraries whose plan has not been shown yet, and each option
    carries one label (D4).

    1. AI Recommended: the itinerary the recommendation names, carrying its reason.
    2. Cheapest (RC7): lowest total cost; ties go to fewer travel minutes, then shorter
       distance, then the sorted activity Place IDs.
    3. Best travel (R2, D3, RC8): lowest 50/50 travel score. Over the eligible
       itineraries only (every member of every unused plan), total distance ``d`` and
       total travel minutes ``t`` (all legs through the commitment) are rescaled to
       0..1::

           d_norm = (d - d_min) / (d_max - d_min)    # 0 when d_max == d_min
           t_norm = (t - t_min) / (t_max - t_min)    # 0 when t_max == t_min
           score  = 0.5 * d_norm + 0.5 * t_norm      # lower is better

       Arithmetic is exact (``Fraction``), so equal scores compare equal. Ties go to
       fewer travel minutes, then shorter distance, then lower total cost, then the
       sorted activity Place IDs.

    In both tie chains a final ``candidate_id`` key only picks a representative within
    one plan, since members of a plan share their Place IDs.

    Input checks run first, in order, raising a ``SelectionError`` subclass: too many
    itineraries, duplicate ids, mixed total currencies, recommendation not found.
    """
    _require_bounded(checked)
    _require_unique_ids(checked)
    _require_one_currency(checked)
    recommended = _recommended(checked, recommendation)
    ai_option = ValidatedOption(
        itinerary=recommended,
        labels=(OptionLabel.AI_RECOMMENDED,),
        recommendation_reason=recommendation.reason,
    )
    used_after_ai = frozenset({_plan_key(recommended)})
    cheapest = _cheapest(_unused(checked, used_after_ai))
    if cheapest is None:
        return (ai_option,)
    used_after_cheapest = used_after_ai | {_plan_key(cheapest)}
    best_travel = _best_travel(_unused(checked, used_after_cheapest))
    if best_travel is None:
        return (ai_option, _option(cheapest, OptionLabel.CHEAPEST))
    return (
        ai_option,
        _option(cheapest, OptionLabel.CHEAPEST),
        _option(best_travel, OptionLabel.BEST_TRAVEL),
    )
