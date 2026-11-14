"""Country-deletion sensitivity for matched tax-cut event-time contrasts.

Delete each country from the complete outcome panel, then re-run all matching
steps. The resulting ranges describe sample influence; they are not bootstrap
confidence intervals or estimates of frequentist sampling uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from causal_tax_cuts.analysis import Key, Observation, load_panel
from causal_tax_cuts.event_study import EventTimeSummary, estimate_event_study


@dataclass(frozen=True, slots=True)
class CountryDeletionEffect:
    """An event-time point estimate with one country's entire history omitted."""

    excluded_country: str
    event_time: int
    status: str
    mean_contrast: float | None
    shift_from_full: float | None
    n_matched_events: int | None
    n_treated_countries: int | None
    pretrend_slope_gap: float | None


@dataclass(frozen=True, slots=True)
class CountryInfluenceSummary:
    """Range and sign sensitivity among successful full re-matched deletions."""

    event_time: int
    full_mean_contrast: float
    minimum_deletion_mean: float | None
    maximum_deletion_mean: float | None
    max_absolute_shift: float | None
    most_influential_country: str | None
    n_successful_deletions: int
    n_failed_deletions: int
    n_sign_reversals: int
    min_matched_events: int | None
    max_matched_events: int | None


@dataclass(frozen=True, slots=True)
class CountryInfluenceAudit:
    """Explicit full-panel sample and failed-deletion coverage."""

    n_countries: int
    n_full_matched_events: int
    n_full_treated_countries: int
    n_failed_country_deletions: int
    failed_countries: tuple[str, ...]
    country_order: tuple[str, ...]


def _summarize_deletions(
    full: EventTimeSummary,
    rows: list[CountryDeletionEffect],
) -> CountryInfluenceSummary:
    """Build descriptive extrema without treating failures as zero effects."""
    valid = [row for row in rows if row.status == "success"]
    if not valid:
        return CountryInfluenceSummary(
            full.event_time,
            full.mean_contrast,
            None,
            None,
            None,
            None,
            0,
            len(rows),
            0,
            None,
            None,
        )

    effects = [row.mean_contrast for row in valid]
    shifts = [row.shift_from_full for row in valid]
    counts = [row.n_matched_events for row in valid]
    if any(value is None for value in (*effects, *shifts, *counts)):
        raise ValueError("Successful deletion must include all numeric diagnostics")
    # The guard above validates the optional fields; these filters give the
    # type checker concrete numeric lists without inventing missing values.
    values = [value for value in effects if value is not None]
    differences = [value for value in shifts if value is not None]
    matched_counts = [value for value in counts if value is not None]
    most_influential = min(
        valid,
        key=lambda row: (
            -abs(row.shift_from_full if row.shift_from_full is not None else 0.0),
            row.excluded_country,
        ),
    ).excluded_country

    return CountryInfluenceSummary(
        event_time=full.event_time,
        full_mean_contrast=full.mean_contrast,
        minimum_deletion_mean=min(values),
        maximum_deletion_mean=max(values),
        max_absolute_shift=max(abs(value) for value in differences),
        most_influential_country=most_influential,
        n_successful_deletions=len(valid),
        n_failed_deletions=len(rows) - len(valid),
        n_sign_reversals=sum(full.mean_contrast * value < 0 for value in values),
        min_matched_events=min(matched_counts),
        max_matched_events=max(matched_counts),
    )


def estimate_country_influence(
    panel: dict[Key, Observation],
    *,
    lags: int = 5,
    max_horizon: int = 5,
    max_donors: int = 5,
    min_events: int = 5,
) -> tuple[
    list[CountryDeletionEffect],
    list[CountryInfluenceSummary],
    CountryInfluenceAudit,
]:
    """Re-estimate fixed-cohort matching after deleting each country entirely.

    Countries are excluded from *both* the potential treated-event and donor
    pools; a country's role is not determined only by its own reform history.
    Failure due to insufficient matches is recorded, never replaced by zero.
    Every other ValueError is propagated so actual data or code defects fail.
    """
    countries = tuple(sorted({country for country, _ in panel}))
    if len(countries) < 3:
        raise ValueError("Country influence requires at least three countries")
    full_events, full_summaries, full_audit = estimate_event_study(
        panel,
        lags=lags,
        max_horizon=max_horizon,
        max_donors=max_donors,
        min_events=min_events,
    )
    if not full_events:
        raise ValueError("Cannot compare against an empty full matched cohort")

    per_country: list[CountryDeletionEffect] = []
    failed_countries: list[str] = []
    for excluded in countries:
        reduced = {
            key: observation for key, observation in panel.items() if key[0] != excluded
        }
        try:
            _, deletion_summaries, audit = estimate_event_study(
                reduced,
                lags=lags,
                max_horizon=max_horizon,
                max_donors=max_donors,
                min_events=min_events,
            )
        except ValueError as exc:
            # A structurally non-estimable deletion is a diagnostic, while
            # unexpected exceptions must still fail the research pipeline.
            if "no placebo summaries reported" not in str(exc):
                raise
            failed_countries.append(excluded)
            per_country.extend(
                CountryDeletionEffect(
                    excluded,
                    summary.event_time,
                    "insufficient_matched_events",
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                for summary in full_summaries
            )
            continue

        by_event_time = {row.event_time: row for row in deletion_summaries}
        for full in full_summaries:
            deleted = by_event_time[full.event_time]
            if not math.isfinite(deleted.mean_contrast):
                raise ValueError("Non-finite country-deletion contrast")
            per_country.append(
                CountryDeletionEffect(
                    excluded_country=excluded,
                    event_time=full.event_time,
                    status="success",
                    mean_contrast=deleted.mean_contrast,
                    shift_from_full=deleted.mean_contrast - full.mean_contrast,
                    n_matched_events=audit.n_matched_events,
                    n_treated_countries=audit.n_treated_countries,
                    pretrend_slope_gap=audit.mean_pretrend_slope_gap,
                )
            )

    aggregated = [
        _summarize_deletions(
            full,
            [row for row in per_country if row.event_time == full.event_time],
        )
        for full in full_summaries
    ]
    audit = CountryInfluenceAudit(
        n_countries=len(countries),
        n_full_matched_events=full_audit.n_matched_events,
        n_full_treated_countries=full_audit.n_treated_countries,
        n_failed_country_deletions=len(failed_countries),
        failed_countries=tuple(failed_countries),
        country_order=countries,
    )
    return per_country, aggregated, audit


def _write_csv(
    destination: Path, rows: list[CountryDeletionEffect] | list[CountryInfluenceSummary]
) -> None:
    """Write fixed-column dataclass rows, using blanks for failed deletions."""
    if not rows:
        raise ValueError(f"No rows available for {destination.name}")
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def save_country_influence(
    output: Path,
    deleted: list[CountryDeletionEffect],
    summary: list[CountryInfluenceSummary],
    audit: CountryInfluenceAudit,
    *,
    source_panel: Path,
    outcome: str,
    covariates: tuple[str, ...],
    lags: int,
    max_horizon: int,
) -> None:
    """Persist descriptive country-deletion tables with honest uncertainty."""
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "country_deletions.csv", deleted)
    _write_csv(output / "country_influence_summary.csv", summary)
    metadata: dict[str, object] = {
        **asdict(audit),
        "source_panel": str(source_panel),
        "outcome": outcome,
        "covariates": list(covariates),
        "lags": lags,
        "max_horizon": max_horizon,
        "procedure": "delete one country completely, then re-match all events",
        "status": "country_deletion_sensitivity_not_cluster_confidence_interval",
        "inference": "none; no bootstrap confidence intervals or p-values",
        "warning": (
            "Country-deletion extrema are descriptive influence ranges, not "
            "uncertainty bands. Removing countries also changes the treated "
            "event cohort and potentially changes all donor pools."
        ),
    }
    (output / "country_influence_audit.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Run complete country-deletion influence checks on an audited panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--outcome", default="top1_share_pct")
    parser.add_argument("--covariates", nargs="*", default=[])
    parser.add_argument("--lags", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--max-donors", type=int, default=5)
    parser.add_argument("--min-events", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    covariates: tuple[str, ...] = tuple(args.covariates)
    if args.outcome in {"country", "year", "treated"} or len(covariates) != len(
        set(covariates)
    ):
        parser.error("Outcome must be a measurement and covariates must be unique")
    panel = load_panel(args.panel, args.outcome, covariates)
    deleted, summary, audit = estimate_country_influence(
        panel,
        lags=args.lags,
        max_horizon=args.horizon,
        max_donors=args.max_donors,
        min_events=args.min_events,
    )
    save_country_influence(
        args.output,
        deleted,
        summary,
        audit,
        source_panel=args.panel,
        outcome=args.outcome,
        covariates=covariates,
        lags=args.lags,
        max_horizon=args.horizon,
    )
    print(
        f"Saved {len(summary)} horizons across {audit.n_countries} deletions; "
        f"{audit.n_failed_country_deletions} deletion(s) not estimable"
    )
    print("CAUTION: country-deletion ranges are not confidence intervals")


if __name__ == "__main__":
    main()
