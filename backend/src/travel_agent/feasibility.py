"""Pure Python feasibility checks for one candidate itinerary (Phase 1).

The model proposes; this module decides. It reads only typed evidence claims, never
free text, and returns either a ``CheckedItinerary`` or ``ValidationFeedback`` listing
every independent failure so the model can repair them in one round.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Final
from zoneinfo import ZoneInfo

from travel_agent.contracts import (
    ActivityStop,
    Budget,
    CandidateItinerary,
    CheckedItinerary,
    ConstraintClaim,
    ConstraintFailure,
    ConstraintId,
    CostAmountClaim,
    CostItem,
    Estimate,
    Evidence,
    Money,
    OpeningAccessClaim,
    OpeningStatus,
    OpenInterval,
    PlanningRequest,
    SourcedFinding,
    SourceKind,
    TaxiRateClaim,
    TransportMode,
    TravelLeg,
    ValidationFeedback,
    VisitDurationClaim,
    WeatherClaim,
    scope_key,
    stop_key,
)

ARRIVAL_BUFFER: Final[timedelta] = timedelta(minutes=15)
_OFFICIAL_SOURCES: Final[frozenset[SourceKind]] = frozenset(
    {
        SourceKind.OFFICIAL_WEBSITE,
        SourceKind.OFFICIAL_SOCIAL_MEDIA,
        SourceKind.OFFICIAL_AUTHORITY_NOTICE,
    }
)
_METERS_PER_KM: Final[Decimal] = Decimal(1000)
_RAIN_WARNING: Final[str] = "Rain is forecast during the outing; plan for wet conditions."


# --- unusable input ----------------------------------------------------------------


class UnusableInputError(ValueError):
    """The inputs cannot be checked at all; the model cannot repair this by replanning."""


class IncompleteRequestError(UnusableInputError):
    """An essential request field is still None."""


class UnresolvedBudgetCurrencyError(UnusableInputError):
    """A budget exists but its currency is unknown (R13: ask before planning)."""


class DuplicateEvidenceIdError(UnusableInputError):
    """The evidence tuple repeats an evidence_id, so citations would be ambiguous."""


# --- derived context ---------------------------------------------------------------


@dataclass(frozen=True)
class _Context:
    """Request values resolved once per evaluation.

    ``gap_start`` and ``cutoff`` are UTC so arithmetic uses elapsed time across DST
    (RC13); ``tz`` is used only to format output and to read local dates.
    """

    tz: ZoneInfo
    gap_start: datetime
    cutoff: datetime
    outing_date: date
    budget: Budget | None
    stop_keys: frozenset[str]


@dataclass(frozen=True)
class _CostOutcome:
    """Row 9 result: failures, or a total when every price is computable."""

    failures: tuple[ConstraintFailure, ...]
    total: Money | None
    includes_estimates: bool


@dataclass(frozen=True)
class _ScheduleOutcome:
    """Rows 7-8 result: failures, or leave-by times and the commitment arrival."""

    failures: tuple[ConstraintFailure, ...]
    leave_by: tuple[datetime, ...]
    arrival: datetime | None


def _require_essentials(request: PlanningRequest) -> tuple[datetime, datetime, str]:
    """Return gap start, commitment time, and timezone, or raise naming what is missing."""
    gap, commitment = request.gap, request.commitment
    missing = [
        name
        for name, value in (
            ("gap.start_time", gap.start_time),
            ("gap.start_location", gap.start_location),
            ("commitment.location", commitment.location),
            ("commitment.time", commitment.time),
            ("commitment.timezone", commitment.timezone),
        )
        if value is None
    ]
    if missing or gap.start_time is None or commitment.time is None or commitment.timezone is None:
        raise IncompleteRequestError(
            f"request is missing essential fields {missing}; ask the user before checking"
        )
    return gap.start_time, commitment.time, commitment.timezone


def _require_budget_currency(budget: Budget | None) -> None:
    """Raise when a budget is stated without a currency."""
    if budget is not None and budget.currency is None:
        raise UnresolvedBudgetCurrencyError(
            f"budget amount {budget.amount} has no currency; ask the user before checking"
        )


def _require_unique_evidence_ids(evidence: tuple[Evidence, ...]) -> None:
    """Raise when two evidence items share an evidence_id."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in evidence:
        if item.evidence_id in seen:
            duplicates.add(item.evidence_id)
        seen.add(item.evidence_id)
    if duplicates:
        raise DuplicateEvidenceIdError(f"evidence repeats evidence_id values {sorted(duplicates)}")


def _build_context(request: PlanningRequest, candidate: CandidateItinerary) -> _Context:
    """Validate the request and derive the outing window in absolute (UTC) time."""
    gap_start, commitment_time, timezone = _require_essentials(request)
    _require_budget_currency(request.budget)
    tz = ZoneInfo(timezone)
    local_commitment = commitment_time.astimezone(tz)
    return _Context(
        tz=tz,
        gap_start=gap_start.astimezone(UTC),
        cutoff=commitment_time.astimezone(UTC) - ARRIVAL_BUFFER,
        outing_date=local_commitment.date(),
        budget=request.budget,
        stop_keys=frozenset(stop_key(stop) for stop in candidate.stops),
    )


# --- failure construction ----------------------------------------------------------


def _failure(constraint: ConstraintId, message: str, inputs: dict[str, str]) -> ConstraintFailure:
    """Build one structured failure; ``inputs`` values are already strings."""
    return ConstraintFailure(constraint=constraint, message=message, inputs=inputs)


def _iso(value: datetime, tz: ZoneInfo) -> str:
    """Format a datetime as ISO 8601 in the commitment timezone."""
    return value.astimezone(tz).isoformat()


# --- applicability -----------------------------------------------------------------


def _overlaps_outing(item: Evidence, context: _Context) -> bool:
    """True when the evidence scope window overlaps the outing window (RC2)."""
    return item.scope.applies_from <= context.cutoff and item.scope.applies_until >= (
        context.gap_start
    )


def _applies_to_plan(item: Evidence, context: _Context) -> bool:
    """True for plan-wide or plan-stop evidence whose window overlaps the outing."""
    key = scope_key(item.scope)
    return _overlaps_outing(item, context) and (key is None or key in context.stop_keys)


def _applies_to_place(item: Evidence, key: str, context: _Context) -> bool:
    """True for evidence about the stop with ``stop_key`` ``key`` that overlaps the outing."""
    return scope_key(item.scope) == key and _overlaps_outing(item, context)


def _taxi_rate_applies(item: Evidence, claim: TaxiRateClaim, context: _Context) -> bool:
    """True for a plan-wide sourced rate whose validity dates cover the outing date (RC1)."""
    return (
        isinstance(item, SourcedFinding)
        and scope_key(item.scope) is None
        and claim.valid_from <= context.outing_date <= claim.valid_until
    )


# --- row 1: same-day gap -----------------------------------------------------------


def _check_same_day_gap(context: _Context) -> tuple[ConstraintFailure, ...]:
    """The gap starts on the commitment's local day and before the arrival cutoff."""
    local_start_date = context.gap_start.astimezone(context.tz).date()
    if local_start_date == context.outing_date and context.gap_start < context.cutoff:
        return ()
    return (
        _failure(
            ConstraintId.SAME_DAY_GAP,
            "The gap must start on the commitment day, before the arrival cutoff.",
            {
                "gap_start": _iso(context.gap_start, context.tz),
                "arrival_cutoff": _iso(context.cutoff, context.tz),
            },
        ),
    )


# --- rows 2-3: transport and transfers ---------------------------------------------


def _check_transport_modes(
    legs: tuple[TravelLeg, ...], allowed: tuple[TransportMode, ...]
) -> tuple[ConstraintFailure, ...]:
    """Every leg uses a requested mode; no stated modes means unrestricted."""
    if not allowed:
        return ()
    return tuple(
        _failure(
            ConstraintId.TRANSPORT_AVAILABILITY,
            "This leg uses a transport mode the user did not allow.",
            {"leg_index": str(index), "mode": leg.mode.value},
        )
        for index, leg in enumerate(legs)
        if leg.mode not in allowed
    )


def _check_transfers(legs: tuple[TravelLeg, ...]) -> tuple[ConstraintFailure, ...]:
    """Every transfer walk fits the connection time it has (equal passes)."""
    return tuple(
        _failure(
            ConstraintId.TRANSFER_TIME,
            "A transfer needs a longer walk than the connection allows.",
            {
                "leg_index": str(leg_index),
                "transfer_index": str(transfer_index),
                "available_minutes": str(transfer.available_minutes),
                "walk_minutes": str(transfer.walk_minutes),
            },
        )
        for leg_index, leg in enumerate(legs)
        for transfer_index, transfer in enumerate(leg.transfers)
        if transfer.walk_minutes > transfer.available_minutes
    )


# --- row 4: explicit constraints ---------------------------------------------------


def _cited_ids(candidate: CandidateItinerary) -> tuple[str, ...]:
    """Every evidence id cited by stops, legs, or costs, first-seen order, no repeats."""
    groups = (
        *(stop.evidence_ids for stop in candidate.stops),
        *(leg.evidence_ids for leg in candidate.legs),
        *(cost.evidence_ids for cost in candidate.costs),
    )
    return tuple(dict.fromkeys(evidence_id for group in groups for evidence_id in group))


def _grounded(item: Evidence, sourced_ids: frozenset[str]) -> bool:
    """True for a sourced finding, or an estimate based only on existing sourced findings.

    A bare AI best guess (empty ``based_on``) never counts (decision 2026-11-18).
    """
    if isinstance(item, SourcedFinding):
        return True
    return bool(item.based_on) and all(i in sourced_ids for i in item.based_on)


def _check_explicit_constraints(
    constraints: tuple[str, ...],
    cited: tuple[Evidence, ...],
    sourced_ids: frozenset[str],
    context: _Context,
) -> tuple[ConstraintFailure, ...]:
    """Each request constraint has cited, applicable, grounded supporting evidence (O1)."""
    supported = frozenset(
        item.claim.constraint_index
        for item in cited
        if isinstance(item.claim, ConstraintClaim)
        and _applies_to_plan(item, context)
        and _grounded(item, sourced_ids)
    )
    return tuple(
        _failure(
            ConstraintId.EXPLICIT_CONSTRAINT,
            "No cited evidence supports this requested constraint.",
            {"constraint_index": str(index), "constraint": text},
        )
        for index, text in enumerate(constraints)
        if index not in supported
    )


# --- row 5: opening and access -----------------------------------------------------


def _opening_findings(
    evidence: tuple[Evidence, ...], key: str, context: _Context
) -> Iterator[tuple[SourceKind, OpeningAccessClaim]]:
    """Yield (source kind, claim) for applicable sourced opening findings at a stop."""
    for item in evidence:
        if (
            isinstance(item, SourcedFinding)
            and isinstance(item.claim, OpeningAccessClaim)
            and _applies_to_place(item, key, context)
        ):
            yield item.source.kind, item.claim


def _opening_key(claim: OpeningAccessClaim) -> tuple[OpeningStatus, frozenset[OpenInterval]]:
    """Comparable value of an opening claim; interval order is irrelevant (RC5)."""
    return claim.status, frozenset(claim.open_intervals)


def _decide_opening(
    evidence: tuple[Evidence, ...], stop: ActivityStop, context: _Context
) -> OpeningAccessClaim | str:
    """Return the deciding claim (O2 precedence) or a rejection reason code.

    A fallback-key stop has no Google Maps tier: only official findings verify it
    (decision 2026-11-18).
    """
    findings = tuple(_opening_findings(evidence, stop_key(stop), context))
    official = tuple(claim for kind, claim in findings if kind in _OFFICIAL_SOURCES)
    maps = tuple(
        claim
        for kind, claim in findings
        if kind == SourceKind.GOOGLE_MAPS and stop.place_id is not None
    )
    tier = official or maps
    if not tier:
        return "unverified"
    if len({_opening_key(claim) for claim in tier}) > 1:
        return "conflicting_sources"
    if tier[0].status == OpeningStatus.CLOSED:
        return "closed"
    return tier[0]


def _opening_failure(stop_index: int, key: str, reason: str) -> ConstraintFailure:
    """Build an OPENING_ACCESS failure for one stop; ``place_id`` carries its ``stop_key``."""
    return _failure(
        ConstraintId.OPENING_ACCESS,
        "This stop is not verified open for the planned visit.",
        {"stop_index": str(stop_index), "place_id": key, "reason": reason},
    )


# --- row 6: visit duration ---------------------------------------------------------


def _required_minutes(evidence: tuple[Evidence, ...], key: str, context: _Context) -> int | None:
    """Evidenced visit minutes: max of sourced findings, else max of estimates (RC6)."""
    applicable = tuple(
        item
        for item in evidence
        if isinstance(item.claim, VisitDurationClaim) and _applies_to_place(item, key, context)
    )
    sourced = tuple(
        item.claim.minutes
        for item in applicable
        if isinstance(item, SourcedFinding) and isinstance(item.claim, VisitDurationClaim)
    )
    estimated = tuple(
        item.claim.minutes
        for item in applicable
        if isinstance(item, Estimate) and isinstance(item.claim, VisitDurationClaim)
    )
    tier = sourced or estimated
    return max(tier) if tier else None


def _check_visit_duration(
    stop_index: int, stop: ActivityStop, evidence: tuple[Evidence, ...], context: _Context
) -> tuple[ConstraintFailure, ...]:
    """The planned visit is at least as long as the evidenced duration (O6, R10)."""
    key = stop_key(stop)
    required = _required_minutes(evidence, key, context)
    if required is None:
        return (
            _failure(
                ConstraintId.MISSING_EVIDENCE,
                "No evidence gives a visit duration for this stop.",
                {
                    "reason": "no_visit_duration",
                    "stop_index": str(stop_index),
                    "place_id": key,
                },
            ),
        )
    if stop.visit_minutes >= required:
        return ()
    return (
        _failure(
            ConstraintId.VISIT_DURATION,
            "The planned visit is shorter than the evidenced duration.",
            {
                "stop_index": str(stop_index),
                "place_id": key,
                "visit_minutes": str(stop.visit_minutes),
                "required_minutes": str(required),
            },
        ),
    )


# --- rows 7-8: schedule (D2) -------------------------------------------------------


def _earliest_start(
    arrival: datetime, visit: timedelta, intervals: tuple[OpenInterval, ...]
) -> datetime | None:
    """Earliest visit start at or after ``arrival`` that fits wholly inside an interval."""
    starts = tuple(
        max(arrival, _utc(interval.opens_at))
        for interval in intervals
        if max(arrival, _utc(interval.opens_at)) + visit <= _utc(interval.closes_at)
    )
    return min(starts) if starts else None


def _latest_departure(
    bound: datetime, visit: timedelta, intervals: tuple[OpenInterval, ...]
) -> datetime:
    """Latest departure no later than ``bound`` whose whole visit fits an interval.

    The forward pass already proved a fitting visit exists, so the choice is non-empty.
    """
    return max(
        min(bound, _utc(interval.closes_at))
        for interval in intervals
        if min(bound, _utc(interval.closes_at)) - visit >= _utc(interval.opens_at)
    )


def _utc(value: datetime) -> datetime:
    """Return ``value`` in UTC so timedelta arithmetic is elapsed time."""
    return value.astimezone(UTC)


def _minutes(value: int) -> timedelta:
    """Return ``value`` minutes as a timedelta."""
    return timedelta(minutes=value)


def _schedule(
    candidate: CandidateItinerary,
    intervals: tuple[tuple[OpenInterval, ...], ...],
    context: _Context,
) -> _ScheduleOutcome:
    """Forward pass for earliest times, then backward pass from the cutoff for leave-by."""
    stops, legs = candidate.stops, candidate.legs
    departure = context.gap_start
    for index, (stop, leg, stop_intervals) in enumerate(
        zip(stops, legs[:-1], intervals, strict=True)
    ):
        start = _earliest_start(
            departure + _minutes(leg.duration_minutes), _minutes(stop.visit_minutes), stop_intervals
        )
        if start is None:
            return _ScheduleOutcome(
                (_opening_failure(index, stop_key(stop), "outside_open_interval"),), (), None
            )
        departure = start + _minutes(stop.visit_minutes)
    earliest_arrival = departure + _minutes(legs[-1].duration_minutes)
    if earliest_arrival > context.cutoff:
        failure = _failure(
            ConstraintId.ARRIVAL_BUFFER,
            "The earliest arrival is after the arrival cutoff.",
            {
                "earliest_arrival": _iso(earliest_arrival, context.tz),
                "arrival_cutoff": _iso(context.cutoff, context.tz),
            },
        )
        return _ScheduleOutcome((failure,), (), None)
    latest_arrival = context.cutoff
    reversed_leave_by: list[datetime] = []
    for stop, next_leg, stop_intervals in zip(
        reversed(stops), reversed(legs[1:]), reversed(intervals), strict=True
    ):
        leave = _latest_departure(
            latest_arrival - _minutes(next_leg.duration_minutes),
            _minutes(stop.visit_minutes),
            stop_intervals,
        )
        reversed_leave_by.append(leave)
        latest_arrival = leave - _minutes(stop.visit_minutes)
    reversed_leave_by.append(latest_arrival - _minutes(legs[0].duration_minutes))
    leave_by = tuple(value.astimezone(context.tz) for value in reversed(reversed_leave_by))
    arrival = (reversed_leave_by[0] + _minutes(legs[-1].duration_minutes)).astimezone(context.tz)
    return _ScheduleOutcome((), leave_by, arrival)


# --- rows 9-10: costs and budget ---------------------------------------------------


def _check_cost_item(
    cost_index: int, cost: CostItem, by_id: dict[str, Evidence], context: _Context
) -> ConstraintFailure | bool:
    """Return a failure, or whether the item's matching evidence is only estimates."""
    claims = tuple(
        (item, item.claim)
        for item in (by_id[i] for i in cost.evidence_ids if i in by_id)
        if isinstance(item.claim, CostAmountClaim) and _applies_to_plan(item, context)
    )
    if not claims:
        return _failure(
            ConstraintId.MISSING_EVIDENCE,
            "No cited, applicable evidence prices this cost item.",
            {"reason": "no_cost_evidence", "cost_index": str(cost_index)},
        )
    matching = tuple(item for item, claim in claims if claim.amount == cost.amount)
    if not matching:
        return _failure(
            ConstraintId.MISSING_EVIDENCE,
            "The cost amount does not match its cited evidence.",
            {"reason": "cost_mismatch", "cost_index": str(cost_index)},
        )
    return all(isinstance(item, Estimate) for item in matching)


@dataclass(frozen=True)
class _TransitFare:
    """One transit leg's fare and whether only estimates support it."""

    fare: Money
    is_estimate: bool


def _check_transit_leg(
    leg_index: int, leg: TravelLeg, by_id: dict[str, Evidence], context: _Context
) -> ConstraintFailure | _TransitFare:
    """Return a failure, or the one distinct fare the leg itself cites (R13)."""
    items = tuple(
        item
        for item in (by_id[i] for i in leg.evidence_ids if i in by_id)
        if isinstance(item.claim, CostAmountClaim) and _applies_to_plan(item, context)
    )
    amounts = frozenset(
        item.claim.amount for item in items if isinstance(item.claim, CostAmountClaim)
    )
    if len(amounts) != 1:
        reason = "no_transit_fare" if not amounts else "conflicting_transit_fares"
        return _failure(
            ConstraintId.MISSING_EVIDENCE,
            "A transit leg must cite exactly one current fare.",
            {"reason": reason, "leg_index": str(leg_index)},
        )
    (fare,) = amounts
    return _TransitFare(fare=fare, is_estimate=all(isinstance(item, Estimate) for item in items))


@dataclass(frozen=True)
class _TaxiPrice:
    """The price part of a taxi rate; validity dates only decide applicability (RC12)."""

    base_fare: Money
    per_km: Money


def _taxi_prices(evidence: tuple[Evidence, ...], context: _Context) -> frozenset[_TaxiPrice]:
    """Distinct prices among taxi rates that apply on the outing date (Money equality)."""
    return frozenset(
        _TaxiPrice(base_fare=item.claim.base_fare, per_km=item.claim.per_km)
        for item in evidence
        if isinstance(item.claim, TaxiRateClaim) and _taxi_rate_applies(item, item.claim, context)
    )


def _taxi_fare(price: _TaxiPrice, distance_meters: int) -> Decimal:
    """Exact distance-based fare: base + per_km x km, no rounding (O7, RC10)."""
    return price.base_fare.amount + price.per_km.amount * Decimal(distance_meters) / _METERS_PER_KM


def _check_taxi_legs(
    legs: tuple[TravelLeg, ...], prices: frozenset[_TaxiPrice]
) -> tuple[ConstraintFailure, ...]:
    """Each taxi leg has exactly one distinct applicable price."""
    if len(prices) == 1:
        return ()
    reason = "no_taxi_rate" if not prices else "conflicting_taxi_rates"
    return tuple(
        _failure(
            ConstraintId.MISSING_EVIDENCE,
            "A taxi leg needs exactly one current taxi rate.",
            {"reason": reason, "leg_index": str(index)},
        )
        for index, leg in enumerate(legs)
        if leg.mode == TransportMode.TAXI_RIDESHARE
    )


def _seen_currencies(currencies: frozenset[str], budget: Budget | None) -> frozenset[str]:
    """Priced-item currencies plus the budget currency when one is stated."""
    if budget is None or budget.currency is None:
        return currencies
    return currencies | frozenset({budget.currency})


def _check_currency(
    seen: frozenset[str], has_priced_item: bool, budget: Budget | None
) -> tuple[ConstraintFailure, ...]:
    """One currency across costs, taxi rates, and budget (O3); some currency must exist (RC4).

    Runs independently of evidence failures, so the model sees a mismatch in the same round.
    """
    if len(seen) > 1:
        return (
            _failure(
                ConstraintId.BUDGET,
                "Costs must all use one currency, matching the budget currency.",
                {"reason": "currency_mismatch", "currencies": ",".join(sorted(seen))},
            ),
        )
    if budget is None and not has_priced_item:
        return (
            _failure(
                ConstraintId.MISSING_EVIDENCE,
                "No priced item or budget sets the total currency.",
                {"reason": "no_cost_currency"},
            ),
        )
    return ()


def _check_costs(
    candidate: CandidateItinerary, evidence: tuple[Evidence, ...], context: _Context
) -> _CostOutcome:
    """Validate every price and the currency, then compute the total once when all pass."""
    by_id = {item.evidence_id: item for item in evidence}
    item_results = tuple(
        _check_cost_item(index, cost, by_id, context) for index, cost in enumerate(candidate.costs)
    )
    transit_results = tuple(
        _check_transit_leg(index, leg, by_id, context)
        for index, leg in enumerate(candidate.legs)
        if leg.mode == TransportMode.PUBLIC_TRANSIT
    )
    transit_fares = tuple(result for result in transit_results if isinstance(result, _TransitFare))
    taxi_legs = tuple(leg for leg in candidate.legs if leg.mode == TransportMode.TAXI_RIDESHARE)
    prices = _taxi_prices(evidence, context) if taxi_legs else frozenset[_TaxiPrice]()
    seen = _seen_currencies(
        frozenset(
            {cost.amount.currency for cost in candidate.costs}
            | {price.base_fare.currency for price in prices}
            | {result.fare.currency for result in transit_fares}
        ),
        context.budget,
    )
    has_priced_item = bool(candidate.costs or taxi_legs or transit_results)
    failures = (
        *(result for result in item_results if isinstance(result, ConstraintFailure)),
        *(result for result in transit_results if isinstance(result, ConstraintFailure)),
        *_check_taxi_legs(candidate.legs, prices),
        *_check_currency(seen, has_priced_item, context.budget),
    )
    if failures:
        return _CostOutcome(tuple(failures), None, False)
    # With no failures, ``seen`` holds exactly one currency and ``prices`` exactly one
    # price whenever taxi legs exist.
    (currency,) = seen
    fares = tuple(_taxi_fare(price, leg.distance_meters) for price in prices for leg in taxi_legs)
    amount = (
        sum((cost.amount.amount for cost in candidate.costs), Decimal(0))
        + sum(fares, Decimal(0))
        + sum((result.fare.amount for result in transit_fares), Decimal(0))
    )
    includes_estimates = (
        bool(taxi_legs)
        or any(result is True for result in item_results)
        or any(result.is_estimate for result in transit_fares)
    )
    return _CostOutcome((), Money(amount=amount, currency=currency), includes_estimates)


def _check_budget(total: Money, budget: Budget | None) -> tuple[ConstraintFailure, ...]:
    """A total above the budget fails; equal passes; no budget means no cap (R13)."""
    if budget is None or total.amount <= budget.amount:
        return ()
    return (
        _failure(
            ConstraintId.BUDGET,
            "The total cost is above the budget.",
            {
                "reason": "over_budget",
                "total_cost": str(total.amount),
                "budget": str(budget.amount),
            },
        ),
    )


# --- row 11 and weather ------------------------------------------------------------


def _check_unknown_ids(
    cited_ids: tuple[str, ...], known: frozenset[str]
) -> tuple[ConstraintFailure, ...]:
    """One failure per cited id that is not in the evidence tuple."""
    return tuple(
        _failure(
            ConstraintId.MISSING_EVIDENCE,
            "The candidate cites evidence that does not exist.",
            {"reason": "unknown_evidence_id", "evidence_id": evidence_id},
        )
        for evidence_id in cited_ids
        if evidence_id not in known
    )


def _weather_warnings(evidence: tuple[Evidence, ...], context: _Context) -> tuple[str, ...]:
    """One warning per applicable rain forecast; rain never rejects (AC6)."""
    return tuple(
        _RAIN_WARNING
        for item in evidence
        if isinstance(item.claim, WeatherClaim)
        and item.claim.rain
        and _applies_to_plan(item, context)
    )


# --- entry point -------------------------------------------------------------------


def _flatten(groups: Iterable[tuple[ConstraintFailure, ...]]) -> tuple[ConstraintFailure, ...]:
    """Concatenate failure groups in order."""
    return tuple(failure for group in groups for failure in group)


def check_candidate(
    request: PlanningRequest,
    candidate: CandidateItinerary,
    evidence: tuple[Evidence, ...],
) -> CheckedItinerary | ValidationFeedback:
    """Check one candidate against the request and evidence; Python decides feasibility.

    Raises ``UnusableInputError`` subclasses when the inputs cannot be checked at all.
    Otherwise returns every independent failure, skipping only dependent checks:
    the schedule needs a same-day gap and an ``open`` decision for every stop, and
    the budget cap needs computable costs.
    """
    context = _build_context(request, candidate)
    _require_unique_evidence_ids(evidence)
    cited_ids = _cited_ids(candidate)
    known_ids = frozenset(item.evidence_id for item in evidence)
    sourced_ids = frozenset(
        item.evidence_id for item in evidence if isinstance(item, SourcedFinding)
    )
    cited_set = frozenset(cited_ids)
    cited = tuple(item for item in evidence if item.evidence_id in cited_set)

    gap_failures = _check_same_day_gap(context)
    decisions = tuple(_decide_opening(evidence, stop, context) for stop in candidate.stops)
    opening_failures = tuple(
        _opening_failure(index, stop_key(stop), decision)
        for index, (stop, decision) in enumerate(zip(candidate.stops, decisions, strict=True))
        if isinstance(decision, str)
    )
    open_claims = tuple(d for d in decisions if isinstance(d, OpeningAccessClaim))
    schedule = (
        _schedule(candidate, tuple(claim.open_intervals for claim in open_claims), context)
        if not gap_failures and len(open_claims) == len(candidate.stops)
        else _ScheduleOutcome((), (), None)
    )
    costs = _check_costs(candidate, evidence, context)
    budget_failures = _check_budget(costs.total, context.budget) if costs.total is not None else ()

    failures = _flatten(
        (
            gap_failures,
            _check_transport_modes(candidate.legs, request.transport_modes),
            _check_transfers(candidate.legs),
            _check_explicit_constraints(request.constraints, cited, sourced_ids, context),
            opening_failures,
            *(
                _check_visit_duration(index, stop, evidence, context)
                for index, stop in enumerate(candidate.stops)
            ),
            schedule.failures,
            costs.failures,
            budget_failures,
            _check_unknown_ids(cited_ids, known_ids),
        )
    )
    if failures or schedule.arrival is None or costs.total is None:
        return ValidationFeedback(candidate_id=candidate.candidate_id, failures=failures)
    return CheckedItinerary(
        candidate=candidate,
        leave_by=schedule.leave_by,
        arrival_at_commitment=schedule.arrival,
        total_cost=costs.total,
        total_includes_estimates=costs.includes_estimates,
        weather_warnings=_weather_warnings(evidence, context),
    )
