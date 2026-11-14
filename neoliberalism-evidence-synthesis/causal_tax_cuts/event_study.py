"""Balanced-cohort event-time placebo contrasts for the tax-cut study.

Pre-event horizons are descriptive diagnostics of matching quality, not
confirmatory tests of parallel trends. This is an independent method and not
a reimplementation of the published PanelMatch estimator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median

from causal_tax_cuts.analysis import Key, Observation, history, load_panel, rank_donors


@dataclass(frozen=True, slots=True)
class EventTimeContrast:
    """One event's donor-adjusted outcome change relative to year t-1."""

    country: str
    reform_year: int
    event_time: int
    contrast: float
    donors: tuple[str, ...]
    pretrend_slope_gap: float


@dataclass(frozen=True, slots=True)
class EventTimeSummary:
    """Descriptive mean across a fixed matched cohort at one event time."""

    event_time: int
    mean_contrast: float
    median_contrast: float
    n_events: int
    n_treated_countries: int


@dataclass(frozen=True, slots=True)
class MatchingAudit:
    """Counts and pre-period slope balance for the restricted matched sample."""

    n_panel_reform_events: int
    n_eligible_events: int
    n_matched_events: int
    n_unmatched_eligible_events: int
    n_treated_countries: int
    min_donors: int
    max_donors: int
    mean_pretrend_slope_gap: float


def _pretrend_slope_gap(gaps: dict[int, float], lags: int) -> float:
    """Calculate a treated-minus-donor pre-period OLS slope, per year."""
    pre_periods = range(-lags, 0)
    mean_year = fmean(pre_periods)
    mean_gap = fmean(gaps[year] for year in pre_periods)
    denominator = sum((year - mean_year) ** 2 for year in pre_periods)
    return (
        sum((year - mean_year) * (gaps[year] - mean_gap) for year in pre_periods)
        / denominator
    )


def estimate_event_study(
    panel: dict[Key, Observation],
    *,
    lags: int = 5,
    max_horizon: int = 5,
    max_donors: int = 5,
    min_events: int = 5,
) -> tuple[list[EventTimeContrast], list[EventTimeSummary], MatchingAudit]:
    """Match full observation windows and calculate before/after event contrasts.

    A single donor set per reform is reused for all event times. Donors must
    match the preceding treatment path, have complete outcomes over the whole
    event window, and have no reform during t through t+max_horizon. The
    treated countries are permitted later reforms. Without covariates,
    ranking intentionally retains every eligible donor.

    Pre-period event times are -lags through -2; year -1 is the zero baseline.
    Post-period event times are 0 through max_horizon.
    """
    if lags < 3 or max_horizon < 0 or max_donors < 1 or min_events < 1:
        raise ValueError("Require lags >= 3, horizon >= 0, donors >= 1 and events >= 1")
    if not panel:
        raise ValueError("Panel contains no observations")

    offsets = tuple(range(-lags, max_horizon + 1))
    event_times = tuple(range(-lags, -1)) + tuple(range(max_horizon + 1))
    countries = sorted({country for country, _ in panel})
    reform_events = sorted(
        (row for row in panel.values() if row.treated == 1),
        key=lambda row: (row.year, row.country),
    )

    event_contrasts: list[EventTimeContrast] = []
    matched_donor_counts: list[int] = []
    pretrend_slopes: list[float] = []
    eligible_events = 0
    for reform in reform_events:
        country, year = reform.country, reform.year
        treatment_history = history(panel, country, year, lags)
        if treatment_history is None or treatment_history[0] != 0:
            continue
        if any((country, year + offset) not in panel for offset in offsets):
            continue
        eligible_events += 1

        donors = [
            candidate
            for candidate in countries
            if candidate != country
            and history(panel, candidate, year, lags) == treatment_history
            and all((candidate, year + offset) in panel for offset in offsets)
            and all(
                panel[(candidate, following_year)].treated == 0
                for following_year in range(year, year + max_horizon + 1)
            )
        ]
        if not donors:
            continue

        selected = rank_donors(panel, country, donors, year, lags, max_donors)
        donor_names = tuple(name for name, _ in selected)
        matched_donor_counts.append(len(donor_names))

        # Compare the outcome levels of the same treated/donor set at each
        # event time, then subtract their difference in the baseline year.
        outcome_gaps = {
            offset: (
                panel[(country, year + offset)].outcome
                - fmean(panel[(donor, year + offset)].outcome for donor in donor_names)
            )
            for offset in offsets
        }
        slope_gap = _pretrend_slope_gap(outcome_gaps, lags)
        pretrend_slopes.append(slope_gap)
        for offset in event_times:
            contrast = outcome_gaps[offset] - outcome_gaps[-1]
            if not math.isfinite(contrast):
                raise ValueError("Non-finite placebo contrast")
            event_contrasts.append(
                EventTimeContrast(
                    country,
                    year,
                    offset,
                    contrast,
                    donor_names,
                    slope_gap,
                )
            )

    n_matched = len(matched_donor_counts)
    if n_matched < min_events:
        raise ValueError(
            f"Only {n_matched} matched reform events (minimum {min_events}); "
            "no placebo summaries reported"
        )

    summaries = [
        EventTimeSummary(
            offset,
            fmean(row.contrast for row in event_contrasts if row.event_time == offset),
            median(row.contrast for row in event_contrasts if row.event_time == offset),
            n_matched,
            len({row.country for row in event_contrasts}),
        )
        for offset in event_times
    ]
    audit = MatchingAudit(
        n_panel_reform_events=len(reform_events),
        n_eligible_events=eligible_events,
        n_matched_events=n_matched,
        n_unmatched_eligible_events=eligible_events - n_matched,
        n_treated_countries=len({row.country for row in event_contrasts}),
        min_donors=min(matched_donor_counts),
        max_donors=max(matched_donor_counts),
        mean_pretrend_slope_gap=fmean(pretrend_slopes),
    )
    return event_contrasts, summaries, audit


def save_event_study(
    output: Path,
    contrasts: list[EventTimeContrast],
    summaries: list[EventTimeSummary],
    audit: MatchingAudit,
    *,
    source_panel: Path,
    outcome: str,
    covariates: tuple[str, ...],
    lags: int,
    max_horizon: int,
) -> None:
    """Write the event-level contrasts, aggregate series and explicit caveats."""
    if not contrasts or not summaries:
        raise ValueError("Refusing to save an empty event-time analysis")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "event_time_contrasts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "country",
                "reform_year",
                "event_time",
                "contrast",
                "donors",
                "pretrend_slope_gap",
            )
        )
        for row in contrasts:
            writer.writerow(
                (
                    row.country,
                    row.reform_year,
                    row.event_time,
                    f"{row.contrast:.12g}",
                    ";".join(row.donors),
                    f"{row.pretrend_slope_gap:.12g}",
                )
            )
    with (output / "event_time_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(summaries[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in summaries)

    metadata: dict[str, object] = {
        **asdict(audit),
        "status": "descriptive_placebo_not_validated_causal_inference",
        "source_panel": str(source_panel),
        "outcome": outcome,
        "covariates": list(covariates),
        "lags": lags,
        "max_horizon": max_horizon,
        "baseline": "year t-1",
        "cohort": "same reform events and donors at every event time",
        "donor_rule": "matching lagged treatment history; no cuts through t+horizon",
        "inference": "none; no p-values or confidence intervals",
        "warning": (
            "Failure to detect a pretrend is not evidence of parallel trends; "
            "future-exposure censoring and unmeasured confounding remain."
        ),
    }
    (output / "event_study_audit.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Run balanced placebo diagnostics on an existing audited country panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--outcome", default="top1_share_pct")
    parser.add_argument("--covariates", nargs="*", default=[])
    parser.add_argument("--lags", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--max-donors", type=int, default=5)
    parser.add_argument("--min-events", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    covariates: tuple[str, ...] = tuple(args.covariates)
    if args.outcome in {"country", "year", "treated"} or len(covariates) != len(
        set(covariates)
    ):
        parser.error("Outcome must be a measurement and covariates must be unique")
    panel = load_panel(args.panel, args.outcome, covariates)
    contrasts, summaries, audit = estimate_event_study(
        panel,
        lags=args.lags,
        max_horizon=args.horizon,
        max_donors=args.max_donors,
        min_events=args.min_events,
    )
    save_event_study(
        args.output,
        contrasts,
        summaries,
        audit,
        source_panel=args.panel,
        outcome=args.outcome,
        covariates=covariates,
        lags=args.lags,
        max_horizon=args.horizon,
    )
    print(
        f"Saved {len(summaries)} event-time diagnostics for "
        f"{audit.n_matched_events} matched events; "
        f"mean pretrend slope gap {audit.mean_pretrend_slope_gap:.5f} "
        f"outcome units/year"
    )
    print("CAUTION: pre-treatment diagnostics do not prove parallel trends")


if __name__ == "__main__":
    main()
