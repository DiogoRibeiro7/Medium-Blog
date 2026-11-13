"""Exploratory matched differences-in-differences for major high-income tax cuts.

This implementation is an independently specified sensitivity analysis, *not* a
replication of Hope and Limberg's PanelMatch estimates. It deliberately reports
no p-values or confidence intervals until appropriate country-block inference
has been implemented and independently checked.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import TypeAlias

import numpy as np

Key: TypeAlias = tuple[str, int]


@dataclass(frozen=True, slots=True)
class Observation:
    """An observed country-year with a recorded reform and an outcome."""

    country: str
    year: int
    treated: int
    outcome: float
    covariates: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class EventEffect:
    """Matched treated-minus-control change for one reform and one horizon."""

    country: str
    year: int
    horizon: int
    effect: float
    donors: tuple[str, ...]
    mean_distance: float | None
    pretrend_gap: float


@dataclass(frozen=True, slots=True)
class HorizonSummary:
    """Descriptive ATT across matched events, without inferential claims."""

    horizon: int
    att: float
    median_effect: float
    n_events: int
    n_treated_countries: int
    n_unmatched: int
    mean_pretrend_gap: float


def load_panel(
    path: Path, outcome: str, covariates: tuple[str, ...]
) -> dict[Key, Observation]:
    """Load a canonical panel, rejecting ambiguous, incomplete or invalid cells.

    The binary exposure is an *event in that year*, not an absorbing indicator.
    All percentages must be supplied in the units intended for interpretation.
    """
    required = {"country", "year", "treated", outcome, *covariates}
    panel: dict[Key, Observation] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = required.difference(reader.fieldnames or [])
            raise ValueError(f"Panel is missing required columns: {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            country = (row["country"] or "").strip().upper()
            if not country:
                raise ValueError(f"Missing country on line {line}")
            try:
                year = int(row["year"])
                treated_text = str(row["treated"]).strip()
                if treated_text not in {"0", "1"}:
                    raise ValueError("treated must be exactly 0 or 1")
                treated = int(treated_text)
                value = float(row[outcome])
                features = tuple(float(row[column]) for column in covariates)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid numeric value on line {line}: {exc}"
                ) from exc
            if not 1900 <= year <= 2100:
                raise ValueError(f"Year outside accepted range on line {line}: {year}")
            if not all(math.isfinite(number) for number in (value, *features)):
                raise ValueError(f"Non-finite value on line {line}")
            if outcome == "top1_share_pct" and not 1.0 <= value <= 100.0:
                raise ValueError(
                    f"Top-1% income share on line {line} must be expressed in "
                    "percentage points, not a 0-to-1 fraction"
                )
            key = (country, year)
            if key in panel:
                raise ValueError(f"Duplicate country-year key: {key}")
            panel[key] = Observation(country, year, treated, value, features)
    if not panel:
        raise ValueError("Panel contains no observations")
    return panel


def history(
    panel: dict[Key, Observation], country: str, year: int, lags: int
) -> tuple[int, ...] | None:
    """Get the full pre-treatment reform path or flag its incomplete coverage."""
    previous = [panel.get((country, year - lag)) for lag in range(1, lags + 1)]
    if any(observation is None for observation in previous):
        return None
    return tuple(
        observation.treated for observation in previous if observation is not None
    )


def _covariate_trajectory(
    panel: dict[Key, Observation], country: str, year: int, lags: int
) -> np.ndarray:
    """Flatten lagged covariates only; never use post-treatment information."""
    values = [panel[(country, year - lag)].covariates for lag in range(1, lags + 1)]
    return np.asarray([value for group in values for value in group], dtype=float)


def rank_donors(
    panel: dict[Key, Observation],
    treated_country: str,
    donors: list[str],
    year: int,
    lags: int,
    max_donors: int,
) -> list[tuple[str, float | None]]:
    """Use ridge-regularised Mahalanobis distance on lagged covariate paths.

    Covariance is computed across eligible candidates and the focal treated
    observation, then regularised because 18 countries cannot stably estimate
    an unrestricted covariance matrix for dozens of lagged features.
    """
    target = _covariate_trajectory(panel, treated_country, year, lags)
    if target.size == 0:
        # Without covariates, taking the first K alphabetical countries would
        # inject a wholly arbitrary donor-selection rule: use all eligible.
        return [(country, None) for country in sorted(donors)]
    features = np.vstack(
        [
            target,
            *(_covariate_trajectory(panel, country, year, lags) for country in donors),
        ]
    )
    standard_deviation = np.std(features, axis=0, ddof=0)
    standard_deviation = np.where(standard_deviation < 1e-10, 1.0, standard_deviation)
    standardized = (features - np.mean(features, axis=0)) / standard_deviation
    covariance = np.atleast_2d(np.cov(standardized, rowvar=False))
    # A fixed ridge shrinkage makes the estimate defined when features exceed
    # the number of donor countries; it is a declared analytic choice.
    regularized = 0.75 * covariance + 0.25 * np.eye(features.shape[1])
    inverse = np.linalg.pinv(regularized, rcond=1e-10)
    ranked: list[tuple[str, float | None]] = []
    for index, country in enumerate(donors, start=1):
        difference = standardized[0] - standardized[index]
        squared_distance = float(difference @ inverse @ difference)
        ranked.append((country, math.sqrt(max(0.0, squared_distance))))
    return sorted(ranked, key=lambda item: (float(item[1]), item[0]))[:max_donors]


def estimate(
    panel: dict[Key, Observation],
    *,
    lags: int = 5,
    max_horizon: int = 5,
    max_donors: int = 5,
    min_events: int = 5,
) -> tuple[list[EventEffect], list[HorizonSummary]]:
    """Estimate an event-onset contrast against countries without a new cut.

    Treated reform paths must be observed at t-1..t-L, with no event at t-1.
    Controls have identical historical event paths and no event through t+h.
    Treated countries may make subsequent reforms: the estimand is the effect
    of reform *onset and subsequent observed policy*, not an isolated one-off.
    Censoring future-exposed donors is a non-random, explicit design choice.
    """
    if lags < 3 or max_horizon < 0 or max_donors < 1 or min_events < 1:
        raise ValueError("Require lags >= 3, horizon >= 0, donors >= 1 and events >= 1")
    countries = sorted({country for country, _ in panel})
    events = sorted(
        (observation for observation in panel.values() if observation.treated == 1),
        key=lambda item: (item.year, item.country),
    )
    effects: list[EventEffect] = []
    summaries: list[HorizonSummary] = []
    for horizon in range(max_horizon + 1):
        eligible_events = 0
        for event in events:
            year = event.year
            country = event.country
            treatment_path = history(panel, country, year, lags)
            if treatment_path is None or treatment_path[0] != 0:
                continue
            if any(
                (country, observed_year) not in panel
                for observed_year in (year - 3, year - 1, year + horizon)
            ):
                continue
            eligible_events += 1
            donors: list[str] = []
            for candidate in countries:
                if (
                    candidate == country
                    or history(panel, candidate, year, lags) != treatment_path
                ):
                    continue
                if any(
                    (candidate, observed_year) not in panel
                    for observed_year in (year - 3, year - 1, year + horizon)
                ):
                    continue
                # A donor's post-event tax cut would contaminate the no-new-cut
                # counterfactual at this horizon; censor it transparently.
                if any(
                    (candidate, target_year) not in panel
                    or panel[(candidate, target_year)].treated != 0
                    for target_year in range(year, year + horizon + 1)
                ):
                    continue
                donors.append(candidate)
            if not donors:
                continue
            selected = rank_donors(panel, country, donors, year, lags, max_donors)
            donor_countries = tuple(name for name, _ in selected)
            treated_change = (
                panel[(country, year + horizon)].outcome
                - panel[(country, year - 1)].outcome
            )
            donor_change = fmean(
                panel[(name, year + horizon)].outcome - panel[(name, year - 1)].outcome
                for name in donor_countries
            )
            pretrend_gap = (
                panel[(country, year - 1)].outcome
                - panel[(country, year - 3)].outcome
                - fmean(
                    panel[(name, year - 1)].outcome - panel[(name, year - 3)].outcome
                    for name in donor_countries
                )
            )
            distances = [distance for _, distance in selected if distance is not None]
            effects.append(
                EventEffect(
                    country,
                    year,
                    horizon,
                    treated_change - donor_change,
                    donor_countries,
                    fmean(distances) if distances else None,
                    pretrend_gap,
                )
            )
        selected_effects = [item for item in effects if item.horizon == horizon]
        if len(selected_effects) < min_events:
            raise ValueError(
                f"Horizon {horizon}: only {len(selected_effects)} matched reform events "
                f"(minimum {min_events}); no ATT reported"
            )
        values = [item.effect for item in selected_effects]
        summaries.append(
            HorizonSummary(
                horizon,
                fmean(values),
                median(values),
                len(values),
                len({item.country for item in selected_effects}),
                eligible_events - len(values),
                fmean(item.pretrend_gap for item in selected_effects),
            )
        )
    return effects, summaries


def _save_results(
    output: Path,
    effects: list[EventEffect],
    summaries: list[HorizonSummary],
    metadata: dict[str, object],
) -> None:
    """Store auditable matched sets and point estimates without fake inference."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "event_effects.csv").open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.writer(destination)
        writer.writerow(
            (
                "country",
                "year",
                "horizon",
                "effect",
                "donors",
                "mean_distance",
                "pretrend_gap",
            )
        )
        for item in effects:
            writer.writerow(
                (
                    item.country,
                    item.year,
                    item.horizon,
                    f"{item.effect:.12g}",
                    ";".join(item.donors),
                    "" if item.mean_distance is None else f"{item.mean_distance:.12g}",
                    f"{item.pretrend_gap:.12g}",
                )
            )
    with (output / "att_by_horizon.csv").open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=list(asdict(summaries[0])))
        writer.writeheader()
        for item in summaries:
            writer.writerow(asdict(item))
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Run the exploratory estimate on a user-supplied real, audited panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel", type=Path, required=True, help="Canonical real-data CSV"
    )
    parser.add_argument("--outcome", default="top1_share_pct")
    parser.add_argument("--covariates", nargs="*", default=[])
    parser.add_argument("--lags", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--max-donors", type=int, default=5)
    parser.add_argument("--min-events", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    covariates = tuple(args.covariates)
    if args.outcome in {"country", "year", "treated"} or len(covariates) != len(
        set(covariates)
    ):
        parser.error("Outcome must be a measurement and covariates must be unique")
    panel = load_panel(args.panel, args.outcome, covariates)
    effects, summaries = estimate(
        panel,
        lags=args.lags,
        max_horizon=args.horizon,
        max_donors=args.max_donors,
        min_events=args.min_events,
    )
    metadata: dict[str, object] = {
        "status": "exploratory_point_estimates_not_paper_replication",
        "inference": "none; no credible standard errors or p-values computed",
        "source_panel": str(args.panel),
        "outcome": args.outcome,
        "covariates": list(covariates),
        "lags": args.lags,
        "max_horizon": args.horizon,
        "max_donors": args.max_donors,
        "n_countries": len({country for country, _ in panel}),
        "n_observations": len(panel),
        "n_raw_event_country_years": sum(item.treated for item in panel.values()),
        "donor_rule": "same lagged reform history; no reforms during t through t+h",
        "treated_rule": "reform at t; no reform at t-1; subsequent reforms permitted",
    }
    _save_results(args.output, effects, summaries, metadata)
    print(
        f"Saved {len(effects)} matched contrasts across {len(summaries)} horizons to {args.output}"
    )
    print(
        "CAUTION: exploratory point estimates; causal identification and inference not established"
    )


if __name__ == "__main__":
    main()
