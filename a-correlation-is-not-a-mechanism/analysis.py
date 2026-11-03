"""Deterministic simulation studies for observational causal identification.

Only the first three experiments are implemented in this phase. All numerical
outputs are synthetic; none estimates an effect in an empirical population.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


DEFAULT_SEED = 20261102
STUDY_IDS = (
    "confounding", "simpson", "collider", "mediator", "instrument",
    "backdoor", "overlap", "nonidentification", "prediction",
)


@dataclass(frozen=True)
class StudyResult:
    """Single Monte Carlo estimate and its ground-truth target."""

    study: str
    scenario: str
    method: str
    replicate: int
    estimate: float
    target: float

    @property
    def error(self) -> float:
        """Signed estimation error relative to the documented estimand."""

        return self.estimate - self.target


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope with intercept, guarding against constant regressors."""

    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be equally sized one-dimensional arrays")
    centered = x - np.mean(x)
    denominator = float(centered @ centered)
    if denominator <= 0:
        raise ValueError("regressor must vary")
    return float(centered @ (y - np.mean(y)) / denominator)


def confounding_experiment(
    rng: np.random.Generator, replicate: int, sample_size: int = 1000
) -> list[StudyResult]:
    """Compare naive and adjusted slopes in a linear common-cause model.

    Structural equations: T = a U + eps_T; Y = tau T + b U + eps_Y.
    The naive population slope equals tau + ab/(a^2 + 1), while correctly
    adjusting for U identifies tau in this structural model.
    """

    if sample_size < 4:
        raise ValueError("sample_size must be at least four")
    tau, a, b = 1.0, 1.2, 1.5
    u = rng.normal(size=sample_size)
    t = a * u + rng.normal(size=sample_size)
    y = tau * t + b * u + rng.normal(size=sample_size)
    naive = _slope(t, y)
    adjusted = float(np.linalg.lstsq(np.column_stack((np.ones(sample_size), t, u)), y, rcond=None)[0][1])
    return [
        StudyResult("confounding", "common_cause", "naive", replicate, naive, tau),
        StudyResult("confounding", "common_cause", "adjusted", replicate, adjusted, tau),
    ]


def confounding_theoretical_bias(a: float = 1.2, b: float = 1.5) -> float:
    """Exact population omitted-variable bias with unit treatment noise."""

    return a * b / (a * a + 1.0)


def simpson_experiment(
    rng: np.random.Generator, replicate: int, sample_size: int = 4000
) -> list[StudyResult]:
    """Simpson reversal driven by treatment propensity and baseline risk.

    Within either stratum, treatment increases response probability by 0.1.
    Assignment favors high-risk stratum 0, whereas stratum 1 has much higher
    baseline response, reversing the marginal association.
    """

    if sample_size < 100:
        raise ValueError("sample_size must be at least 100")
    stratum = rng.binomial(1, 0.5, sample_size)
    p_treat = np.where(stratum == 1, 0.15, 0.85)
    treatment = rng.binomial(1, p_treat)
    baseline = np.where(stratum == 1, 0.8, 0.1)
    response = rng.binomial(1, baseline + 0.1 * treatment)
    marginal = float(response[treatment == 1].mean() - response[treatment == 0].mean())
    standardized = 0.0
    for group in (0, 1):
        mask = stratum == group
        treated = mask & (treatment == 1)
        untreated = mask & (treatment == 0)
        if not treated.any() or not untreated.any():
            raise RuntimeError("empty stratum-treatment cell; increase sample size")
        standardized += 0.5 * float(response[treated].mean() - response[untreated].mean())
    return [
        StudyResult("simpson", "stratified_assignment", "marginal", replicate, marginal, 0.1),
        StudyResult("simpson", "stratified_assignment", "standardized", replicate, standardized, 0.1),
    ]


def simpson_exact_marginal_difference() -> float:
    """Exact marginal treated-minus-untreated response probability.

    P(stratum=1|T=1)=0.15 and P(stratum=1|T=0)=0.85 by Bayes' rule.
    """

    treated = 0.15 * 0.9 + 0.85 * 0.2
    untreated = 0.85 * 0.8 + 0.15 * 0.1
    return treated - untreated


def collider_experiment(
    rng: np.random.Generator, replicate: int, sample_size: int = 1500
) -> list[StudyResult]:
    """Estimate independent causes' regression slope before/after collider adjustment.

    Independent standard Gaussian T and U cause C=T+U+eps; outcome Y=U+eps_Y.
    The causal effect of T on Y is zero. Conditioning on C induces a negative
    association: coefficient -1/(1+sigma_C^2), with sigma_C^2=1 here.
    """

    if sample_size < 4:
        raise ValueError("sample_size must be at least four")
    t = rng.normal(size=sample_size)
    u = rng.normal(size=sample_size)
    c = t + u + rng.normal(size=sample_size)
    y = u + rng.normal(size=sample_size)
    marginal = _slope(t, y)
    conditioned = float(np.linalg.lstsq(np.column_stack((np.ones(sample_size), t, c)), y, rcond=None)[0][1])
    return [
        StudyResult("collider", "gaussian_common_effect", "marginal", replicate, marginal, 0.0),
        StudyResult("collider", "gaussian_common_effect", "conditioned", replicate, conditioned, 0.0),
    ]


def collider_theoretical_slope(collider_noise_variance: float = 1.0) -> float:
    """Population T coefficient when regressing Y on T and the collider C."""

    if collider_noise_variance < 0 or not math.isfinite(collider_noise_variance):
        raise ValueError("collider noise variance must be finite and nonnegative")
    return -1.0 / (1.0 + collider_noise_variance)


def mediator_experiment(
    rng: np.random.Generator, replicate: int, sample_size: int = 1500
) -> list[StudyResult]:
    """Distinguish randomized total and controlled-direct effects.

    T is randomized; M=0.8*T+eps_M; Y=T+1.5*M+eps_Y. The total
    effect is 2.2, whereas the controlled direct effect is 1.0. Both
    are identified here because there is no mediator-outcome confounding.
    """

    if sample_size < 4:
        raise ValueError("sample_size must be at least four")
    treatment = rng.binomial(1, 0.5, size=sample_size).astype(float)
    mediator = 0.8 * treatment + rng.normal(size=sample_size)
    outcome = treatment + 1.5 * mediator + rng.normal(size=sample_size)
    total = _slope(treatment, outcome)
    direct = float(np.linalg.lstsq(
        np.column_stack((np.ones(sample_size), treatment, mediator)),
        outcome, rcond=None,
    )[0][1])
    return [
        StudyResult("mediator", "randomized_mediation", "total", replicate, total, 2.2),
        StudyResult("mediator", "randomized_mediation", "direct", replicate, direct, 1.0),
    ]


def instrument_experiment(
    rng: np.random.Generator, replicate: int, sample_size: int = 2000
) -> list[StudyResult]:
    """Contrast a valid instrument against direct-path exclusion violation.

    Z is independent of U. T=0.8*Z+U+eps_T. Y=1.2*T+gamma*Z+1.5*U+eps_Y.
    The population Wald estimand is 1.2+gamma/0.8, whereas the structural
    intervention effect of T is 1.2 in both scenarios.
    """

    if sample_size < 4:
        raise ValueError("sample_size must be at least four")
    instrument = rng.normal(size=sample_size)
    latent = rng.normal(size=sample_size)
    treatment = 0.8 * instrument + latent + rng.normal(size=sample_size)
    denominator = _slope(instrument, treatment)
    if abs(denominator) < 1e-12:
        raise RuntimeError("near-zero sample first stage")
    results: list[StudyResult] = []
    shared_noise = rng.normal(size=sample_size)
    for label, gamma in (("valid_exclusion", 0.0), ("exclusion_violation", 0.4)):
        outcome = 1.2 * treatment + gamma * instrument + 1.5 * latent + shared_noise
        wald = _slope(instrument, outcome) / denominator
        results.append(StudyResult("instrument", label, "wald", replicate, wald, 1.2))
    return results


def instrument_theoretical_wald(gamma: float, first_stage: float = 0.8) -> float:
    """Population instrumental-variable ratio for this linear model."""

    if first_stage == 0.0 or not math.isfinite(first_stage) or not math.isfinite(gamma):
        raise ValueError("finite nonzero first stage and finite direct path required")
    return 1.2 + gamma / first_stage


def backdoor_experiment(
    rng: np.random.Generator, replicate: int, sample_size: int = 1800
) -> list[StudyResult]:
    """Compare an admissible adjustment set with a collider-augmented set.

    U causes both T and Y. V causes Y and the collider C, while T also
    causes C. Adjusting for U blocks confounding; adding C opens T->C<-V->Y.
    """

    if sample_size < 4:
        raise ValueError("sample_size must be at least four")
    latent = rng.normal(size=sample_size)
    competing_cause = rng.normal(size=sample_size)
    treatment = 0.9 * latent + rng.normal(size=sample_size)
    collider = treatment + competing_cause + rng.normal(size=sample_size)
    outcome = 1.0 * treatment + 1.2 * latent + 1.4 * competing_cause + rng.normal(size=sample_size)
    naive = _slope(treatment, outcome)
    adjusted = float(np.linalg.lstsq(
        np.column_stack((np.ones(sample_size), treatment, latent)), outcome, rcond=None,
    )[0][1])
    collider_adjusted = float(np.linalg.lstsq(
        np.column_stack((np.ones(sample_size), treatment, latent, collider)), outcome, rcond=None,
    )[0][1])
    return [
        StudyResult("backdoor", "confounder_and_collider", "naive", replicate, naive, 1.0),
        StudyResult("backdoor", "confounder_and_collider", "valid_adjustment", replicate, adjusted, 1.0),
        StudyResult("backdoor", "confounder_and_collider", "collider_adjustment", replicate, collider_adjusted, 1.0),
    ]


def backdoor_theoretical_collider_bias() -> float:
    """Exact linear-Gaussian collider-adjustment bias after controlling U."""

    # After controlling U, T's residual is independent of V, and C=T+V+eps.
    # Conditioning on C gives coefficient -1.4/(1+1) on T through V.
    return -0.7


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size for finite nonnegative weights."""

    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("weights must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and nonnegative")
    total = float(np.sum(weights))
    if total == 0:
        raise ValueError("at least one weight must be positive")
    return total**2 / float(np.dot(weights, weights))


def overlap_experiment(
    rng: np.random.Generator, replicate: int, sample_size: int = 4000
) -> list[StudyResult]:
    """Compare covariate overlap and inverse-probability weighting.

    X is balanced binary; treatment assignment depends strongly on X.
    Potential outcomes are Y(t)=2X+t+noise and the true ATE is one.
    The true propensity is known by construction, so uncertainty arises
    from positivity/finite samples rather than propensity-model fitting.
    """

    if sample_size < 100:
        raise ValueError("sample_size must be at least 100")
    covariate = rng.binomial(1, 0.5, size=sample_size)
    noise = rng.normal(size=sample_size)
    rows: list[StudyResult] = []
    # Share covariates and potential outcomes across scenarios.
    for label, low in (("adequate", 0.25), ("poor", 0.01)):
        propensity = np.where(covariate == 1, 1.0 - low, low)
        treatment = rng.binomial(1, propensity)
        outcome = 2.0 * covariate + treatment + noise
        if not treatment.any() or np.all(treatment):
            raise RuntimeError("empty treatment arm")
        naive = float(outcome[treatment == 1].mean() - outcome[treatment == 0].mean())
        # Horvitz-Thompson estimator of both potential-outcome means.
        ipw = float(np.mean(treatment * outcome / propensity - (1 - treatment) * outcome / (1 - propensity)))
        rows.extend((
            StudyResult("overlap", label, "naive", replicate, naive, 1.0),
            StudyResult("overlap", label, "ipw", replicate, ipw, 1.0),
        ))
    return rows


def overlap_theoretical_ess_fraction(low_probability: float) -> float:
    """Large-sample ESS/n for inverse assignment weights in binary X design."""

    if not 0 < low_probability <= 0.5:
        raise ValueError("low_probability must be in (0, 0.5]")
    # W = T/e(X) + (1-T)/(1-e(X)), E[W]=2,
    # E[W²]=1/e+1/(1-e) under balanced symmetric strata.
    return 4.0 / (1.0 / low_probability + 1.0 / (1.0 - low_probability))


def nonidentification_experiment(
    rng: np.random.Generator, replicate: int, sample_size: int = 2000
) -> list[StudyResult]:
    """Two models with identical observational (T,Y) laws but different ATEs.

    Model A: T~N(0,1), Y=T+eps, eps~N(0,4).
    Model B: U~N(0,1), T=0.5U+sqrt(0.75)eps_T,
             Y=2U+eps_Y, eps_Y~N(0,1).
    Both yield Var(T)=1, Var(Y)=5, Cov(T,Y)=1, while ATEs are 1 and 0.
    """

    if sample_size < 4:
        raise ValueError("sample_size must be at least four")
    treatment_a = rng.normal(size=sample_size)
    outcome_a = treatment_a + 2.0 * rng.normal(size=sample_size)
    latent = rng.normal(size=sample_size)
    treatment_b = 0.5 * latent + math.sqrt(0.75) * rng.normal(size=sample_size)
    outcome_b = 2.0 * latent + rng.normal(size=sample_size)
    return [
        StudyResult("nonidentification", "causal_model", "observed_slope", replicate, _slope(treatment_a, outcome_a), 1.0),
        StudyResult("nonidentification", "confounded_model", "observed_slope", replicate, _slope(treatment_b, outcome_b), 0.0),
    ]


def prediction_experiment(
    rng: np.random.Generator, replicate: int, sample_size: int = 1500
) -> list[StudyResult]:
    """Show that a predictive proxy is not itself a manipulable cause.

    U causes both a measured proxy X and outcome Y, but not treatment T.
    Intervening on measured X leaves Y unchanged, while intervening on T
    changes Y by two. Predictive regression assigns a nonzero X coefficient.
    """

    if sample_size < 4:
        raise ValueError("sample_size must be at least four")
    latent = rng.normal(size=sample_size)
    proxy = latent + 0.5 * rng.normal(size=sample_size)
    treatment = rng.binomial(1, 0.5, size=sample_size).astype(float)
    outcome = 2.0 * treatment + 3.0 * latent + rng.normal(size=sample_size)
    coefficients = np.linalg.lstsq(
        np.column_stack((np.ones(sample_size), treatment, proxy)), outcome, rcond=None,
    )[0]
    return [
        StudyResult("prediction", "proxy_not_intervention", "proxy_association", replicate, float(coefficients[2]), 0.0),
        StudyResult("prediction", "proxy_not_intervention", "treatment_effect", replicate, float(coefficients[1]), 2.0),
    ]


EXPERIMENTS: dict[str, Callable[[np.random.Generator, int, int], list[StudyResult]]] = {
    "confounding": confounding_experiment,
    "simpson": simpson_experiment,
    "collider": collider_experiment,
    "mediator": mediator_experiment,
    "instrument": instrument_experiment,
    "backdoor": backdoor_experiment,
    "overlap": overlap_experiment,
    "nonidentification": nonidentification_experiment,
    "prediction": prediction_experiment,
}
DEFAULT_SIZES = {
    "confounding": 1000, "simpson": 4000, "collider": 1500,
    "mediator": 1500, "instrument": 2000, "backdoor": 1800,
    "overlap": 4000, "nonidentification": 2000, "prediction": 1500,
}


def simulate(study: str, repetitions: int = 300, seed: int = DEFAULT_SEED) -> list[StudyResult]:
    """Generate independent replicates with study-specific stable random streams."""

    if study not in EXPERIMENTS:
        raise ValueError(f"unknown study: {study}")
    if repetitions < 2:
        raise ValueError("repetitions must be at least two")
    study_index = STUDY_IDS.index(study)
    children = np.random.SeedSequence([seed, study_index]).spawn(repetitions)
    return [
        result
        for i, child in enumerate(children)
        for result in EXPERIMENTS[study](np.random.default_rng(child), i, DEFAULT_SIZES[study])
    ]


def summarize(results: Iterable[StudyResult]) -> list[dict[str, str | float | int]]:
    """Compute Monte Carlo bias and uncertainty for each named estimator."""

    groups: dict[tuple[str, str, str], list[StudyResult]] = {}
    for result in results:
        groups.setdefault((result.study, result.scenario, result.method), []).append(result)
    rows: list[dict[str, str | float | int]] = []
    for (study, scenario, method), observations in sorted(groups.items()):
        estimates = np.array([item.estimate for item in observations], dtype=float)
        target = observations[0].target
        rows.append({
            "study": study,
            "scenario": scenario,
            "method": method,
            "repetitions": len(observations),
            "target": target,
            "mean_estimate": float(np.mean(estimates)),
            "bias": float(np.mean(estimates) - target),
            "monte_carlo_se": float(np.std(estimates, ddof=1) / math.sqrt(len(estimates))),
            "sampling_sd": float(np.std(estimates, ddof=1)),
        })
    return rows


def _write_csv(destination: Path, records: list[dict[str, object]]) -> None:
    """Write a nonempty rectangular table with stable column order."""

    if not records:
        raise ValueError("cannot archive an empty table")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    """CLI entry point: write generated study tables only to explicit output dirs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=300)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    for study in STUDY_IDS:
        results = simulate(study, repetitions=args.repetitions, seed=args.seed)
        _write_csv(args.output_dir / f"{study}_replicates.csv", [vars(item) | {"error": item.error} for item in results])
        _write_csv(args.output_dir / f"{study}_summary.csv", summarize(results))


if __name__ == "__main__":
    main()
