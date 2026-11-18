"""Offline replay: check one case's candidates and select options, then write a JSON report.

Usage (from the repository root):
    uv run --project backend python backend/scripts/replay.py --case CASE.json --output OUT.json

The case file holds ``request``, ``evidence``, ``candidates``, and ``recommendation``.
The report holds ``checked`` (passing candidates), ``rejected`` (validation feedback),
and ``options`` (selected options in slot order), each as contract-model JSON.
No network access is used.
"""

import argparse
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from travel_agent.contracts import (
    CandidateItinerary,
    CheckedItinerary,
    Evidence,
    PlanningRequest,
    Recommendation,
    ValidatedOption,
    ValidationFeedback,
)
from travel_agent.feasibility import UnusableInputError, check_candidate
from travel_agent.selection import SelectionError, select_options


class ReplayCase(BaseModel):
    """One replay case file: strict, immutable, and closed to unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    request: PlanningRequest
    evidence: tuple[Evidence, ...]
    candidates: tuple[CandidateItinerary, ...]
    recommendation: Recommendation


class ReplayReport(BaseModel):
    """The replay output: checked and rejected candidates in case order, then options."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    checked: tuple[CheckedItinerary, ...]
    rejected: tuple[ValidationFeedback, ...]
    options: tuple[ValidatedOption, ...]


def replay(case: ReplayCase) -> ReplayReport:
    """Check every candidate in case order, then select options from those that passed."""
    results = tuple(
        check_candidate(case.request, candidate, case.evidence) for candidate in case.candidates
    )
    checked = tuple(item for item in results if isinstance(item, CheckedItinerary))
    rejected = tuple(item for item in results if isinstance(item, ValidationFeedback))
    options = select_options(checked, case.recommendation) if checked else ()
    return ReplayReport(checked=checked, rejected=rejected, options=options)


def summarise(report: ReplayReport) -> str:
    """Return a short human-readable summary of a replay report."""
    lines = [
        f"checked: {len(report.checked)}, rejected: {len(report.rejected)}, "
        f"options: {len(report.options)}"
    ]
    for item in report.checked:
        total = item.total_cost
        estimate = " (includes estimates)" if item.total_includes_estimates else ""
        arrival = item.arrival_at_commitment.isoformat()
        lines.append(
            f"  pass {item.candidate.candidate_id}: arrive {arrival}, "
            f"total {total.amount} {total.currency}{estimate}"
        )
    for feedback in report.rejected:
        constraints = ", ".join(sorted({failure.constraint for failure in feedback.failures}))
        lines.append(f"  fail {feedback.candidate_id}: {constraints}")
    for option in report.options:
        labels = ", ".join(option.labels)
        lines.append(f"  option {option.itinerary.candidate.candidate_id}: {labels}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse the required --case and --output paths."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--case", type=Path, required=True, help="replay case JSON file")
    parser.add_argument("--output", type=Path, required=True, help="report JSON file to write")
    return parser.parse_args()


def main() -> None:
    """Load the case, replay it, write the report, and print a summary."""
    args = parse_args()
    case_path: Path = args.case
    output_path: Path = args.output
    try:
        case = ReplayCase.model_validate_json(case_path.read_bytes())
    except ValidationError as error:
        raise SystemExit(f"invalid replay case {case_path}: {error}") from error
    try:
        report = replay(case)
    except (UnusableInputError, SelectionError) as error:
        raise SystemExit(
            f"unusable replay case {case_path}: {type(error).__name__}: {error}"
        ) from error
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(summarise(report))
    print(f"report written to {output_path}")


if __name__ == "__main__":
    main()
