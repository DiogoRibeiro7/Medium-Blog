"""Compare archived AP House live snapshots without scoring an election.

The historical model, its registered interval and the initial-election
outcome definition remain fixed. This program reads local, dated AP records;
it never fetches current results or substitutes them for an official score.

Examples::

    python snapshot_history.py --write
    python snapshot_history.py --check
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from analysis import LOCK, ROOT, HouseSnapshot, diagnostics, load_inputs, write_json

ARCHIVE: Final[Path] = ROOT / "data"
REPORT: Final[Path] = ROOT / "results" / "snapshot_history.json"


def dated_snapshots(folder: Path = ARCHIVE) -> list[tuple[Path, HouseSnapshot]]:
    """Load all dated AP records and validate their chronological identities."""

    records: list[tuple[Path, HouseSnapshot]] = []
    for source in sorted(folder.glob("ap_house_live_*.json")):
        data: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
        snapshot: HouseSnapshot = HouseSnapshot.from_dict(data)
        date: str = snapshot.reported_at_utc[:10]
        if source.name != f"ap_house_live_{date}.json":
            raise ValueError(f"File name disagrees with the AP reporting date: {source.name}")
        records.append((source, snapshot))
    if len(records) < 2:
        raise ValueError("At least two archived live snapshots are required")
    records.sort(key=lambda pair: pair[1].reported_at_utc)
    timestamps: list[datetime] = [
        datetime.fromisoformat(record.reported_at_utc.replace("Z", "+00:00"))
        for _, record in records
    ]
    if any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:], strict=False)):
        raise ValueError("AP reporting timestamps must be unique and increasing")
    return records


def build_history(
    records: list[tuple[Path, HouseSnapshot]], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Calculate explicitly conditional changes between successive source dates."""

    entries: list[dict[str, Any]] = []
    for filename, snapshot in records:
        report: dict[str, Any] = diagnostics(snapshot, manifest)
        entries.append({
            "archived_source_filename": filename.name,
            "reported_at_utc": snapshot.reported_at_utc,
            "retrieved_at_utc": snapshot.retrieved_at_utc,
            "democratic_called": snapshot.democratic_won,
            "republican_called": snapshot.republican_won,
            "total_called": snapshot.democratic_won + snapshot.republican_won,
            "democratic_leading_uncalled": snapshot.democratic_leading,
            "republican_leading_uncalled": snapshot.republican_leading,
            "total_uncalled": snapshot.uncalled,
            "republican_total_if_leads_hold": snapshot.republican_trend,
            "conditional_republican_seat_change": report["illustrative_change_if_all_leads_hold"],
            "conditional_ols_forecast_minus_scenario": report["candidate_scenario_differences"]["ols"][
                "if_leads_hold_predicted_minus_conditional_change"
            ],
            "conditional_ols_interval_inclusion": report["if_leads_hold_interval_inclusion"],
        })

    movements: list[dict[str, Any]] = []
    for earlier, later in zip(entries, entries[1:], strict=False):
        movements.append({
            "from_snapshot": earlier["archived_source_filename"],
            "to_snapshot": later["archived_source_filename"],
            "additional_called_races": later["total_called"] - earlier["total_called"],
            "change_in_uncalled_races": later["total_uncalled"] - earlier["total_uncalled"],
            "change_in_conditional_republican_seats": (
                later["republican_total_if_leads_hold"] - earlier["republican_total_if_leads_hold"]
            ),
            "change_in_conditional_ols_signed_difference": (
                later["conditional_ols_forecast_minus_scenario"] -
                earlier["conditional_ols_forecast_minus_scenario"]
            ),
        })

    return {
        "report_type": "unofficial_ap_live_snapshot_history_not_final_scoring",
        "official_initial_election_outcome_scored": False,
        "source_name": "Associated Press",
        "source_url": "https://apnews.com/hub/elections",
        "historical_model_input_lock": "../results/forecast_lock.json",
        "registered_house_seats_defended": manifest["scenario"]["president_party_seats_defended"],
        "frozen_ols_point": manifest["primary_forecast"]["predicted_seat_change"],
        "frozen_ols_95_interval": manifest["primary_forecast"]["model_based_95_prediction_interval"],
        "snapshots": entries,
        "movements": movements,
        "caveats": [
            "All seat totals and differences derived from leads are hypothetical, not observations.",
            "Changing leads and newly called races can jointly alter the aggregate trend.",
            "The initial-election House Historian-aligned target remains unavailable here.",
            "A new AP snapshot cannot authorize modification of the registered forecast.",
        ],
    }


def main() -> None:
    """Create or verify the deterministic longitudinal evidence report."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--write", action="store_true", help="Write archived-snapshot diagnostics")
    commands.add_argument("--check", action="store_true", help="Verify the archived report without editing")
    args = parser.parse_args()
    _, manifest = load_inputs(lock_path=LOCK)
    record = build_history(dated_snapshots(), manifest)
    if args.write:
        write_json(record, REPORT)
    else:
        expected: dict[str, Any] = json.loads(REPORT.read_text(encoding="utf-8"))
        if record != expected:
            raise ValueError("Snapshot chronology is stale; regenerate and review the dated changes")
    print(f"Verified {len(record['snapshots'])} AP live records; no official election outcome scored")


if __name__ == "__main__":
    main()
