"""Deterministic, synthetic finite-population sampling experiments.

The first three studies are implemented; the remaining six are specified in
RESEARCH_PLAN.md. No published evidence is claimed until the full suite exists.
"""

from __future__ import annotations

import argparse
import csv
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

SEED = 20261102
STUDIES = ("srs", "strata", "nonresponse")
METHODS = {
    "srs": ("sample_mean",),
    "strata": ("unweighted", "poststratified", "ht", "hajek"),
    "nonresponse": ("respondents", "ipw_known", "ipw_misspecified"),
}
FIELDNAMES = ("study", "scenario", "replicate", "method", "estimate", "target", "error")


@dataclass(frozen=True)
class Observation:
    """One estimator from one independently seeded Monte Carlo replicate."""

    study: str
    scenario: str
    replicate: int
    method: str
    estimate: float
    target: float

    def as_row(self) -> dict[str, str | int | float]:
        """Return the exact archive schema, including signed estimation error."""
        return {
            "study": self.study,
            "scenario": self.scenario,
            "replicate": self.replicate,
            "method": self.method,
            "estimate": self.estimate,
            "target": self.target,
            "error": self.estimate - self.target,
        }


def rng_for(study: str, replicate: int, seed: int = SEED) -> np.random.Generator:
    """Create a stable stream independent of execution or study order."""
    if study not in STUDIES or replicate < 0 or seed < 0:
        raise ValueError("Unknown study, negative replicate or negative seed")
    # CRC32 is deterministic across Python processes; hash() is not.
    key = zlib.crc32(study.encode("utf-8"))
    return np.random.default_rng(np.random.SeedSequence([seed, key, replicate]))


def srs(replicate: int, seed: int = SEED) -> list[Observation]:
    """Sample without replacement from one fixed finite population."""
    population_rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    y = population_rng.normal(loc=4.0, scale=2.5, size=1000)
    sample = rng_for("srs", replicate, seed).choice(len(y), size=120, replace=False)
    return [Observation("srs", "n120_N1000", replicate, "sample_mean", float(y[sample].mean()), float(y.mean()))]


def strata(replicate: int, seed: int = SEED) -> list[Observation]:
    """Disproportionate stratified design with known inclusion probabilities."""
    generator = rng_for("strata", replicate, seed)
    population_rng = np.random.default_rng(np.random.SeedSequence([seed, 202]))
    low = population_rng.normal(0.0, 1.0, size=800)
    high = population_rng.normal(5.0, 1.0, size=200)
    # Deliberate oversampling: 80/800 and 80/200.
    low_sample = low[generator.choice(800, size=80, replace=False)]
    high_sample = high[generator.choice(200, size=80, replace=False)]
    target = float((low.sum() + high.sum()) / 1000)
    unweighted = float(np.concatenate((low_sample, high_sample)).mean())
    weighted_total = float((800 / 80) * low_sample.sum() + (200 / 80) * high_sample.sum())
    post = float(0.8 * low_sample.mean() + 0.2 * high_sample.mean())
    # Here calibrated weights sum to known N, making HT and Hájek coincide.
    estimates = {"unweighted": unweighted, "poststratified": post,
                 "ht": weighted_total / 1000, "hajek": weighted_total / (800 + 200)}
    return [Observation("strata", "oversampled_high", replicate, name, value, target)
            for name, value in estimates.items()]


def nonresponse(replicate: int, seed: int = SEED) -> list[Observation]:
    """Observed-X response weighting with an intentionally omitted Y driver.

    The 'known' IPW uses the true response probabilities conditional on BOTH
    observed X and latent outcome noise, solely as an oracle benchmark.
    Such probabilities are not available to an actual analyst.
    """
    generator = rng_for("nonresponse", replicate, seed)
    x = generator.normal(size=2500)
    epsilon = generator.normal(size=2500)
    y = 2.0 + x + epsilon
    # Response depends on both measured predictor and unmeasured outcome shock.
    logits = -0.55 + 0.8 * x + 0.8 * epsilon
    probability = 1.0 / (1.0 + np.exp(-logits))
    included = generator.random(len(x)) < probability
    if not np.any(included):
        raise RuntimeError("No respondents in synthetic replicate")
    observed = y[included]
    oracle_weight = 1.0 / probability[included]
    # Deliberate misspecification: omits the response-dependence on epsilon.
    wrong_probability = 1.0 / (1.0 + np.exp(-(-0.55 + 0.8 * x[included])))
    wrong_weight = 1.0 / wrong_probability
    estimates = {
        "respondents": float(observed.mean()),
        "ipw_known": float(np.average(observed, weights=oracle_weight)),
        "ipw_misspecified": float(np.average(observed, weights=wrong_weight)),
    }
    target = float(y.mean())
    return [Observation("nonresponse", "outcome_dependent", replicate, name, value, target)
            for name, value in estimates.items()]


RUNNERS: dict[str, Callable[[int, int], list[Observation]]] = {
    "srs": srs, "strata": strata, "nonresponse": nonresponse,
}


def simulate(study: str, repetitions: int, seed: int = SEED) -> list[Observation]:
    """Run one experiment, enforcing its complete method/replicate contract."""
    if study not in RUNNERS or repetitions <= 0:
        raise ValueError("Unknown study or invalid repetitions")
    rows = [row for replicate in range(repetitions) for row in RUNNERS[study](replicate, seed)]
    assert len(rows) == repetitions * len(METHODS[study])
    if any(not math.isfinite(row.estimate) or not math.isfinite(row.target) for row in rows):
        raise ValueError("Nonfinite simulation result")
    return rows


def write_evidence(rows: list[Observation], output: Path, suffix: str = "") -> None:
    """Write replicates and Monte Carlo summaries with explicit uncertainty."""
    if not rows:
        raise ValueError("Cannot archive an empty experiment")
    output.mkdir(parents=True, exist_ok=True)
    study = rows[0].study
    if any(row.study != study for row in rows):
        raise ValueError("Mixed studies")
    with (output / f"{study}_replicates{suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(row.as_row() for row in rows)
    with (output / f"{study}_summary{suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("study", "scenario", "method", "repetitions", "mean_estimate", "mean_target",
                  "bias", "rmse", "sd_error", "mcse_bias", "ci_low", "ci_high")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in METHODS[study]:
            chosen = [row for row in rows if row.method == method]
            errors = np.asarray([row.estimate - row.target for row in chosen])
            bias = float(errors.mean())
            sd = float(errors.std(ddof=1)) if len(errors) > 1 else 0.0
            mcse = sd / math.sqrt(len(errors))
            writer.writerow({"study": study, "scenario": chosen[0].scenario, "method": method,
                             "repetitions": len(errors), "mean_estimate": float(np.mean([r.estimate for r in chosen])),
                             "mean_target": float(np.mean([r.target for r in chosen])),
                             "bias": bias, "rmse": float(np.sqrt(np.mean(errors**2))),
                             "sd_error": sd, "mcse_bias": mcse,
                             "ci_low": bias - 1.96 * mcse, "ci_high": bias + 1.96 * mcse})


def main() -> None:
    """CLI, with quick output isolated from eventual canonical evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("Seed must be nonnegative")
    if args.repetitions is not None and args.repetitions <= 0:
        parser.error("Repetitions must be positive")
    # Explicitly avoid default-path canonical artifacts while evidence is provisional.
    if not args.quick and args.output == Path(__file__).resolve().parent / "data":
        parser.error("Full provisional runs need an explicit --output directory")
    count = args.repetitions if args.repetitions is not None else (3 if args.quick else 300)
    suffix = "_quick" if args.quick else ""
    for study in STUDIES:
        write_evidence(simulate(study, count, args.seed), args.output, suffix)
        print(f"{study}: {count} repetitions")


if __name__ == "__main__":
    main()
