"""Validate a future *official* House outcome receipt before registered scoring.

This is a source-consistency preflight, not the registered scoring program.
No live AP counts, forecast errors or official 2026 outcome are embedded here.
An analyst must transcribe the final initial-election House division and
the American Presidency Project's matching 2026 midterm seat-change entry,
then inspect both original sources and acknowledge their final status.

Only after that independent evidence exists may this command prepare a
new, non-overwriting handoff to ``../score_after_election.py``::

    python post_election/final_outcome_handoff.py \\
        --evidence PATH_TO_SOURCE_RECEIPT.json \\
        --output post_election/results/verified_source_handoff.json

The original registered scorer must still be invoked separately. Merely
setting the acknowledgment flags in JSON does *not* verify the publishers'
webpages; source inspection and documented provenance are human obligations.
"""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Final

from analysis import LOCK

HISTORIAN_URL: Final[str] = "https://history.house.gov/Institution/Party-Divisions/Party-Divisions/"
PRESIDENCY_URL: Final[str] = (
    "https://www.presidency.ucsb.edu/statistics/data/"
    "seats-congress-gainedlost-the-presidents-party-mid-term-elections"
)
ELECTION_DAY: Final[date] = date(2026, 11, 3)


def exact_int(value: Any, description: str) -> int:
    """Reject missing, boolean and non-integral seat counts."""

    if type(value) is not int:
        raise TypeError(f"{description} must be a signed integer, not a boolean")
    return value


def required_date(value: Any, description: str) -> date:
    """Require an explicit ISO calendar date after election day."""

    if not isinstance(value, str):
        raise TypeError(f"{description} must be a YYYY-MM-DD string")
    parsed: date = date.fromisoformat(value)
    if parsed.isoformat() != value or parsed < ELECTION_DAY:
        raise ValueError(f"{description} must be a post-election ISO date")
    return parsed


@dataclass(frozen=True, slots=True)
class OfficialSourceHandoff:
    """Validated numerical bridge from the official sources to the scorer."""

    historian_retrieved: date
    presidency_retrieved: date
    republican_initial_seats: int
    democratic_initial_seats: int
    republican_seat_change: int
    model_seats_defended: int


def validate_source_receipt(
    raw: dict[str, Any], manifest: dict[str, Any], *, reviewed_on: date
) -> OfficialSourceHandoff:
    """Compare independently published source values against locked inputs.

    A receipt with an AP snapshot, a contemporary member count, missing
    confirmation flags or inconsistent party divisions is rejected.
    Neither the locked point forecast nor its interval is inspected here;
    they remain the responsibility of the independent registered scorer.
    """

    if not isinstance(raw, dict):
        raise TypeError("The evidence receipt must be a JSON object")
    if raw.get("record_kind") != "final_2026_initial_house_party_divisions":
        raise ValueError("Use only the final initial-election party-division source class")
    if raw.get("election_date") != ELECTION_DAY.isoformat():
        raise ValueError("The election date does not match the locked target")
    if raw.get("analyst_confirmed_direct_source_inspection") is not True:
        raise ValueError("Direct manual verification of both official sources is required")
    if type(reviewed_on) is not date or reviewed_on < ELECTION_DAY:
        raise ValueError("The independent review must happen after the election")

    historian: Any = raw.get("house_historian")
    presidency: Any = raw.get("american_presidency_project")
    if not isinstance(historian, dict) or not isinstance(presidency, dict):
        raise TypeError("Both official source records must be objects")
    if historian.get("source_url") != HISTORIAN_URL:
        raise ValueError("The House Historian initial-election source URL is required")
    if presidency.get("source_url") != PRESIDENCY_URL:
        raise ValueError("The American Presidency Project midterm source URL is required")
    if exact_int(historian.get("congress"), "Congress number") != 120:
        raise ValueError("The 2026 election must map to the 120th Congress")
    if historian.get("initial_election_party_division") is not True:
        raise ValueError("Current House membership cannot replace initial-election results")
    if historian.get("published_final") is not True or presidency.get("published_final") is not True:
        raise ValueError("Both named sources must publish a final outcome")
    if exact_int(presidency.get("election_year"), "Presidency Project year") != 2026:
        raise ValueError("Only the 2026 presidency-party midterm row is relevant")

    h_date: date = required_date(historian.get("retrieved_on"), "House Historian retrieval")
    p_date: date = required_date(presidency.get("retrieved_on"), "Presidency Project retrieval")
    if max(h_date, p_date) > reviewed_on:
        raise ValueError("Official retrieval cannot postdate the analyst's review")
    dem: int = exact_int(historian.get("democratic_initial_seats"), "Democratic seats")
    gop: int = exact_int(historian.get("republican_initial_seats"), "Republican seats")
    other: int = exact_int(historian.get("other_initial_seats"), "Other seats")
    if min(dem, gop, other) < 0 or dem + gop + other != 435:
        raise ValueError("The Historian's 120th initial House division must total 435")
    if other != 0:
        raise ValueError("Nonzero other-party seats require explicit caucus alignment review")

    defended: int = exact_int(
        manifest.get("scenario", {}).get("president_party_seats_defended"),
        "Locked GOP seats defended",
    )
    app_defended: int = exact_int(
        presidency.get("house_seats_to_defend"), "Presidency Project defended seats"
    )
    app_change: int = exact_int(
        presidency.get("republican_house_seat_change"), "Presidency Project seat change"
    )
    if defended != 220 or app_defended != defended:
        raise ValueError("The independently published defended-seat baseline must match 220")
    if gop - defended != app_change:
        raise ValueError("Historian party division and published seat change disagree")
    if manifest.get("target", {}).get("observed_2026_outcome") is not None:
        raise ValueError("The registered forecast lock must not contain an observed target")
    return OfficialSourceHandoff(h_date, p_date, gop, dem, app_change, defended)


def prepare_handoff(
    evidence: OfficialSourceHandoff, *, reviewed_on: date
) -> dict[str, Any]:
    """Provide auditable source provenance and a separate human-run command."""

    command: list[str] = [
        "python", "score_after_election.py",
        "--actual-change", str(evidence.republican_seat_change),
        "--official-url", HISTORIAN_URL,
        "--source-retrieved-on", evidence.historian_retrieved.isoformat(),
        "--confirmed-final",
        "--output", "post_election/verified_2026_score.json",
    ]
    return {
        "report_type": "source_consistency_handoff_not_a_forecast_score",
        "official_forecast_score_computed": False,
        "source_class": "initial_house_party_divisions_not_current_membership",
        "election_date": ELECTION_DAY.isoformat(),
        "analyst_reviewed_on": reviewed_on.isoformat(),
        "historian_url": HISTORIAN_URL,
        "historian_retrieved_on": evidence.historian_retrieved.isoformat(),
        "presidency_project_url": PRESIDENCY_URL,
        "presidency_project_retrieved_on": evidence.presidency_retrieved.isoformat(),
        "republican_initial_election_seats": evidence.republican_initial_seats,
        "democratic_initial_election_seats": evidence.democratic_initial_seats,
        "locked_republican_seats_defended": evidence.model_seats_defended,
        "verified_source_republican_seat_change": evidence.republican_seat_change,
        "registered_scorer_command_not_executed": shlex.join(command),
        "warning": (
            "This preflight checks transcribed evidence for arithmetic and source consistency "
            "but does not fetch or certify official webpages. Review original published figures "
            "before running the unchanged registered scorer."
        ),
    }


def main() -> None:
    """Create a new, non-overwriting source receipt only after manual review."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True,
                        help="Locally archived JSON transcriptions from both final sources")
    parser.add_argument("--output", type=Path, required=True,
                        help="New, previously nonexistent JSON handoff report")
    args = parser.parse_args()
    today: date = datetime.now(timezone.utc).date()
    raw: dict[str, Any] = json.loads(args.evidence.read_text(encoding="utf-8"))
    manifest: dict[str, Any] = json.loads(LOCK.read_text(encoding="utf-8"))
    evidence = validate_source_receipt(raw, manifest, reviewed_on=today)
    report = prepare_handoff(evidence, reviewed_on=today)
    destination: Path = args.output.resolve()
    if destination.suffix.lower() != ".json":
        raise ValueError("The output must be a JSON file")
    if destination.exists():
        raise FileExistsError("A source handoff must not overwrite an existing report")
    # Exclusive creation prevents a race with another writer or an existing report.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print("Validated official-source receipt. Registered scorer NOT executed.")


if __name__ == "__main__":
    main()
