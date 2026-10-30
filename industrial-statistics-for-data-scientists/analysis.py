"""Synthetic industrial-statistics study with reproducible decisions.

Eight figures connect rational subgrouping, Xbar/R charts, capability,
crossed gauge R&R, inspection misclassification, guard bands and a
randomized replicated 2x2 factorial experiment. Synthetic latent truths
permit evaluation of decisions without claiming real plant performance.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import TypeAlias

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
Row: TypeAlias = dict[str, str | float | int]

MASTER_SEED = 20261030
TARGET = 10.0
LSL = 9.2
USL = 10.8
MU_BASELINE = 10.04
MU_SHIFT = 0.36
SIGMA_PROCESS = 0.20
GAUGE_BIAS = 0.10
SIGMA_REPEAT = 0.09
SIGMA_OPERATOR = 0.08
SIGMA_INTERACTION = 0.04

SUBGROUP_SIZE = 5
N_SUBGROUPS = 120
N_BASELINE = 70
D2_5 = 2.326
A2_5 = 0.577
D3_5 = 0.0
D4_5 = 2.115

N_PARTS = 20
N_OPERATORS = 3
N_REPLICATES = 3
INSPECTION_MU = 10.20
INSPECTION_SIGMA = 0.25
FALSE_ACCEPT_COST = 50.0
FALSE_REJECT_COST = 1.0
DOE_REPLICATES = 6
DOE_CENTERPOINTS = 4
DOE_SAMPLE_SIZE = 5
TRUE_DOE_COEFFICIENTS = np.array([10.05, 0.14, -0.10, 0.18], dtype=np.float64)


@dataclass(frozen=True)
class GaugeDesign:
    """Crossed part/operator/repetition measurements and latent truth."""

    values: FloatArray
    part_truth: FloatArray
    operator_offsets: FloatArray


@dataclass(frozen=True)
class IndustrialStudy:
    """Seeded measurements, fitted summaries and data for eight figures."""

    true_process: FloatArray
    measured_process: FloatArray
    subgroup_rows: list[Row]
    limits: dict[str, float]
    capability_rows: list[Row]
    gauge_design: GaugeDesign
    gauge_components: dict[str, float]
    gauge_studies: FloatArray
    gauge_rows: list[Row]
    gauge_component_rows: list[Row]
    inspection_true: FloatArray
    inspection_raw: FloatArray
    inspection_corrected: FloatArray
    inspection_rows: list[Row]
    guardband_rows: list[Row]
    factorial_rows: list[Row]
    factorial_effect_rows: list[Row]
    factorial_setting_rows: list[Row]
    estimated_beta: FloatArray
    beta_standard_errors: FloatArray
    bias_estimate: float
    summary_rows: list[Row]


def generator(namespace: int) -> np.random.Generator:
    """Create independent, stable seed namespaces for study components."""
    if not isinstance(namespace, int) or isinstance(namespace, bool) or namespace < 0:
        raise ValueError("namespace must be a nonnegative integer")
    return np.random.default_rng(np.random.SeedSequence((MASTER_SEED, namespace)))


def _validate_vector(values: FloatArray) -> FloatArray:
    """Return a finite float vector with at least two observations."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError("expected a finite one-dimensional vector of length >= 2")
    return array


def normal_nonconformance(mu: float, sigma: float) -> float:
    """Calculate two-sided specification failure under a Gaussian model."""
    if not all(np.isfinite(x) for x in (mu, sigma)) or sigma <= 0.0:
        raise ValueError("mu must be finite and sigma must be positive and finite")
    distribution = NormalDist(mu=mu, sigma=sigma)
    return float(distribution.cdf(LSL) + 1.0 - distribution.cdf(USL))


def capability(mu: float, sigma: float) -> dict[str, float]:
    """Return population Cp, Cpk, Cpm for fixed, two-sided specifications.

    These formulas are meaningful as normal-theory capability only after a
    stable reference process has been established. This function does not
    assert stability or estimate an uncertain process from a sample.
    """
    if not all(np.isfinite(x) for x in (mu, sigma)) or sigma <= 0:
        raise ValueError("mu must be finite and sigma must be positive and finite")
    return {
        "cp": float((USL - LSL) / (6.0 * sigma)),
        "cpk": float(min(USL - mu, mu - LSL) / (3.0 * sigma)),
        "cpm": float(
            (USL - LSL) / (6.0 * np.sqrt(sigma * sigma + (mu - TARGET) ** 2))
        ),
    }


def process_stream(rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """Simulate latent part dimensions and one stable biased gauge.

    The physical process mean rises after subgroup 70. Each rational
    subgroup consists of five near-contemporaneous independent parts.
    Gauge repeatability noise is independent across readings; the common
    additive bias is fixed through the monitoring period.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    truth = np.asarray(
        rng.normal(MU_BASELINE, SIGMA_PROCESS, (N_SUBGROUPS, SUBGROUP_SIZE)),
        dtype=np.float64,
    )
    truth[N_BASELINE:] += MU_SHIFT
    observed = truth + GAUGE_BIAS + rng.normal(0.0, SIGMA_REPEAT, truth.shape)
    return truth, np.asarray(observed, dtype=np.float64)


def xbar_r_limits(baseline: FloatArray) -> dict[str, float]:
    """Estimate phase-I Xbar/R limits from unshifted five-part subgroups."""
    array = np.asarray(baseline, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 20 or array.shape[1] != SUBGROUP_SIZE:
        raise ValueError("expected >= 20 rational subgroups of exactly five parts")
    if not np.isfinite(array).all():
        raise ValueError("all subgroup measurements must be finite")
    xbar_bar = float(array.mean())
    ranges = np.ptp(array, axis=1)
    rbar = float(np.mean(ranges))
    return {
        "xbar_center": xbar_bar,
        "range_center": rbar,
        "within_sigma": rbar / D2_5,
        "xbar_lcl": xbar_bar - A2_5 * rbar,
        "xbar_ucl": xbar_bar + A2_5 * rbar,
        "range_lcl": D3_5 * rbar,
        "range_ucl": D4_5 * rbar,
    }


def _capability_row(name: str, values: FloatArray, sigma: float, note: str) -> Row:
    """Create a labelled *descriptive* capability estimate from one sample."""
    vector = _validate_vector(values)
    mu = float(vector.mean())
    metrics = capability(mu, sigma)
    return {
        "population": name,
        "mean": mu,
        "sigma_used": sigma,
        "cp": metrics["cp"],
        "cpk": metrics["cpk"],
        "cpm": metrics["cpm"],
        "normal_model_nonconformance": normal_nonconformance(mu, sigma),
        "methodological_note": note,
    }


def simulate_gauge_design(rng: np.random.Generator) -> GaugeDesign:
    """Generate a balanced crossed random-effects gauge study.

    Twenty parts span a wider inspection range; three operators each
    measure every part three times. The operator effects are centered
    within this particular illustrative operator set. A common additive
    bias cannot be recovered from this study alone.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    parts = np.asarray(rng.uniform(9.3, 10.7, N_PARTS), dtype=np.float64)
    operators = np.asarray(rng.normal(0.0, SIGMA_OPERATOR, N_OPERATORS), dtype=np.float64)
    operators -= operators.mean()
    interactions = rng.normal(0.0, SIGMA_INTERACTION, (N_PARTS, N_OPERATORS))
    noise = rng.normal(0.0, SIGMA_REPEAT, (N_PARTS, N_OPERATORS, N_REPLICATES))
    values = (
        parts[:, None, None]
        + operators[None, :, None]
        + interactions[:, :, None]
        + GAUGE_BIAS
        + noise
    )
    return GaugeDesign(np.asarray(values, dtype=np.float64), parts, operators)


def crossed_anova_components(values: FloatArray) -> dict[str, float]:
    """Method-of-moments random-effects decomposition for a balanced design.

    Model Y_ijk = mu + P_i + O_j + (PO)_ij + E_ijk. Expected mean squares
    give variance components; negative ANOVA differences are clipped at
    zero (a convention, not an unbiased constrained estimator).
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or min(array.shape) < 2 or not np.isfinite(array).all():
        raise ValueError("expected finite (parts, operators, replicates) with >= 2 each")
    n_parts, n_ops, n_reps = array.shape
    grand = float(array.mean())
    part_mean = array.mean(axis=(1, 2))
    operator_mean = array.mean(axis=(0, 2))
    cell_mean = array.mean(axis=2)
    interaction_residual = (
        cell_mean - part_mean[:, None] - operator_mean[None, :] + grand
    )
    ms_part = float(
        n_ops * n_reps * np.sum((part_mean - grand) ** 2) / (n_parts - 1)
    )
    ms_operator = float(
        n_parts * n_reps * np.sum((operator_mean - grand) ** 2) / (n_ops - 1)
    )
    ms_interaction = float(
        n_reps * np.sum(interaction_residual**2) / ((n_parts - 1) * (n_ops - 1))
    )
    ms_repeat = float(
        np.sum((array - cell_mean[:, :, None]) ** 2) / (n_parts * n_ops * (n_reps - 1))
    )
    components = {
        "repeatability": ms_repeat,
        "part_operator": max(0.0, (ms_interaction - ms_repeat) / n_reps),
        "operator": max(0.0, (ms_operator - ms_interaction) / (n_parts * n_reps)),
        "part": max(0.0, (ms_part - ms_interaction) / (n_ops * n_reps)),
        "ms_part": ms_part,
        "ms_operator": ms_operator,
        "ms_part_operator": ms_interaction,
        "ms_repeat": ms_repeat,
    }
    gauge_variance = sum(components[key] for key in ("repeatability", "operator", "part_operator"))
    total_variance = gauge_variance + components["part"]
    components["gauge_sd"] = float(np.sqrt(gauge_variance))
    components["percent_study_variation"] = (
        100.0 * np.sqrt(gauge_variance / total_variance) if total_variance > 0 else float("nan")
    )
    return components


def inspection_classification(
    true: FloatArray, readings: FloatArray, *, guardband: float = 0.0
) -> dict[str, float | int]:
    """Evaluate specification acceptance against simulated latent truth."""
    truth = _validate_vector(true)
    measured = _validate_vector(readings)
    if truth.shape != measured.shape:
        raise ValueError("true and measured vectors must be the same size")
    if not np.isfinite(guardband) or guardband < 0 or guardband >= (USL - LSL) / 2:
        raise ValueError("guardband must be nonnegative and narrower than half tolerance")
    true_good = (truth >= LSL) & (truth <= USL)
    accepted = (measured >= LSL + guardband) & (measured <= USL - guardband)
    n_good = int(true_good.sum())
    n_bad = int((~true_good).sum())
    if min(n_good, n_bad) == 0:
        raise ValueError("need both conforming and nonconforming parts")
    fa = int(((~true_good) & accepted).sum())
    fr = int((true_good & (~accepted)).sum())
    n = int(truth.size)
    return {
        "parts": n,
        "truly_conforming": n_good,
        "truly_nonconforming": n_bad,
        "false_accepts": fa,
        "false_rejects": fr,
        "false_accepts_per_10000": 10_000 * fa / n,
        "false_rejects_per_10000": 10_000 * fr / n,
        "false_accepts_given_bad": fa / n_bad,
        "false_rejects_given_good": fr / n_good,
        "illustrative_loss_per_10000": 10_000 * (
            FALSE_ACCEPT_COST * fa + FALSE_REJECT_COST * fr
        ) / n,
    }


def _doe_model(speed: float, temperature: float) -> float:
    """True mean dimension under two coded production settings."""
    if not np.isfinite(speed) or not np.isfinite(temperature):
        raise ValueError("coded factors must be finite")
    beta = TRUE_DOE_COEFFICIENTS
    return float(beta[0] + beta[1] * speed + beta[2] * temperature + beta[3] * speed * temperature)


def simulate_factorial(
    rng: np.random.Generator, bias_estimate: float
) -> tuple[list[Row], FloatArray, FloatArray, list[Row], list[Row]]:
    """Randomize replicated 2x2 corners, intersperse fixed center runs, fit OLS.

    Every run reports the bias-corrected mean of five *new* physical parts.
    The true within-part sigma is held constant across settings by design.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    if not np.isfinite(bias_estimate):
        raise ValueError("bias estimate must be finite")
    corner_design = [
        (float(a), float(b))
        for _ in range(DOE_REPLICATES)
        for a in (-1, 1)
        for b in (-1, 1)
    ]
    randomized = [corner_design[i] for i in rng.permutation(len(corner_design))]
    # Centerpoints are scheduled between randomized factorial runs as
    # stability/curvature checks, not treated as fully randomized corners.
    interval = len(randomized) // DOE_CENTERPOINTS
    design: list[tuple[float, float]] = []
    for index in range(0, len(randomized), interval):
        design.append((0.0, 0.0))
        design.extend(randomized[index : index + interval])
    if len(design) != DOE_REPLICATES * 4 + DOE_CENTERPOINTS:
        raise RuntimeError("unexpected factorial run count")
    rows: list[Row] = []
    predictors: list[list[float]] = []
    responses: list[float] = []
    for order, (speed, temperature) in enumerate(design, start=1):
        mu = _doe_model(speed, temperature)
        physical_parts = rng.normal(mu, SIGMA_PROCESS, DOE_SAMPLE_SIZE)
        observed_parts = (
            physical_parts + GAUGE_BIAS + rng.normal(0.0, SIGMA_REPEAT, DOE_SAMPLE_SIZE)
        )
        corrected_response = float(np.mean(observed_parts) - bias_estimate)
        predictor = [1.0, speed, temperature, speed * temperature]
        predictors.append(predictor)
        responses.append(corrected_response)
        rows.append(
            {
                "run_order": order,
                "speed_coded": speed,
                "temperature_coded": temperature,
                "centerpoint": int(speed == 0.0 and temperature == 0.0),
                "true_run_mean": float(np.mean(physical_parts)),
                "measured_corrected_mean": corrected_response,
                "true_setting_mean": mu,
                "n_parts": DOE_SAMPLE_SIZE,
            }
        )
    x = np.asarray(predictors, dtype=np.float64)
    y = np.asarray(responses, dtype=np.float64)
    estimates, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ estimates
    residual_variance = float(residuals @ residuals / (len(y) - x.shape[1]))
    cov = residual_variance * np.linalg.inv(x.T @ x)
    standard_errors = np.asarray(np.sqrt(np.diag(cov)), dtype=np.float64)
    coefficient_rows: list[Row] = []
    for index, label in enumerate(("intercept", "speed", "temperature", "interaction")):
        coefficient_rows.append(
            {
                "term": label,
                "true_beta": float(TRUE_DOE_COEFFICIENTS[index]),
                "estimated_beta": float(estimates[index]),
                "standard_error": float(standard_errors[index]),
                "effect_for_coded_plus_vs_minus": (
                    float(2 * estimates[index]) if index > 0 else float("nan")
                ),
            }
        )
    setting_rows: list[Row] = []
    for speed in (-1.0, 1.0):
        for temperature in (-1.0, 1.0):
            vec = np.array([1.0, speed, temperature, speed * temperature], dtype=np.float64)
            fitted = float(vec @ estimates)
            truth = _doe_model(speed, temperature)
            setting_rows.append(
                {
                    "speed_coded": speed,
                    "temperature_coded": temperature,
                    "true_mean": truth,
                    "fitted_mean": fitted,
                    "mean_standard_error": float(np.sqrt(vec @ cov @ vec)),
                    "true_normal_defect_rate": normal_nonconformance(truth, SIGMA_PROCESS),
                    "fitted_normal_defect_rate": normal_nonconformance(fitted, SIGMA_PROCESS),
                    "assumed_process_sigma": SIGMA_PROCESS,
                }
            )
    return rows, np.asarray(estimates, dtype=np.float64), standard_errors, coefficient_rows, setting_rows


def _capability_rows(
    truth: FloatArray, observed: FloatArray, baseline_limits: dict[str, float]
) -> list[Row]:
    """Distinguish known stable truth, gauge-inclusive estimates, and mixtures."""
    true_model = capability(MU_BASELINE, SIGMA_PROCESS)
    true_post = capability(MU_BASELINE + MU_SHIFT, SIGMA_PROCESS)
    rows: list[Row] = [
        {
            "population": "baseline_latent_true_model",
            "mean": MU_BASELINE,
            "sigma_used": SIGMA_PROCESS,
            **true_model,
            "normal_model_nonconformance": normal_nonconformance(MU_BASELINE, SIGMA_PROCESS),
            "methodological_note": "Exact normal theory for a stationary baseline, known by simulation.",
        },
        _capability_row(
            "baseline_latent_sample",
            truth[:N_BASELINE].ravel(),
            xbar_r_limits(truth[:N_BASELINE])["within_sigma"],
            "Estimate of within-subgroup latent truth; reference segment only.",
        ),
        _capability_row(
            "baseline_measured_sample",
            observed[:N_BASELINE].ravel(),
            baseline_limits["within_sigma"],
            "Gauge-inclusive, biased observations; stable reference segment only.",
        ),
        {
            "population": "postshift_latent_true_model",
            "mean": MU_BASELINE + MU_SHIFT,
            "sigma_used": SIGMA_PROCESS,
            **true_post,
            "normal_model_nonconformance": normal_nonconformance(
                MU_BASELINE + MU_SHIFT, SIGMA_PROCESS
            ),
            "methodological_note": "Exact normal theory after the known imposed mean shift.",
        },
        _capability_row(
            "postshift_measured_sample",
            observed[N_BASELINE:].ravel(),
            xbar_r_limits(observed[N_BASELINE:])["within_sigma"],
            "Gauge-inclusive estimate within the postshift regime, not a claim of overall stability.",
        ),
        _capability_row(
            "pooled_measured_descriptive",
            observed.ravel(),
            float(observed.std(ddof=1)),
            "Overall-Ppk-style descriptive index: mixture across two regimes; NOT valid stationary capability.",
        ),
    ]
    return rows


def run_study(*, quick: bool = False) -> IndustrialStudy:
    """Compute all independent datasets, decision outcomes and saved results."""
    if not isinstance(quick, bool):
        raise TypeError("quick must be bool")
    truth, measured = process_stream(generator(10))
    limits = xbar_r_limits(measured[:N_BASELINE])
    subgroup_rows: list[Row] = []
    xbars = measured.mean(axis=1)
    ranges = np.ptp(measured, axis=1)
    for index in range(N_SUBGROUPS):
        subgroup_rows.append(
            {
                "subgroup_index_zero_based": index,
                "n_parts": SUBGROUP_SIZE,
                "regime": "reference" if index < N_BASELINE else "changed",
                "true_subgroup_mean": float(truth[index].mean()),
                "measured_subgroup_mean": float(xbars[index]),
                "measured_subgroup_range": float(ranges[index]),
                "xbar_out_of_control": int(
                    xbars[index] < limits["xbar_lcl"]
                    or xbars[index] > limits["xbar_ucl"]
                ),
                "range_out_of_control": int(
                    ranges[index] < limits["range_lcl"]
                    or ranges[index] > limits["range_ucl"]
                ),
            }
        )
    capability_rows = _capability_rows(truth, measured, limits)

    gauge_design = simulate_gauge_design(generator(20))
    gauge = crossed_anova_components(gauge_design.values)
    gauge_rows: list[Row] = []
    for part in range(N_PARTS):
        for operator in range(N_OPERATORS):
            for repetition in range(N_REPLICATES):
                gauge_rows.append(
                    {
                        "part": part + 1,
                        "operator": operator + 1,
                        "repetition": repetition + 1,
                        "latent_true_part_dimension": float(gauge_design.part_truth[part]),
                        "measured_dimension": float(gauge_design.values[part, operator, repetition]),
                    }
                )
    true_part_variance = float(np.var(gauge_design.part_truth, ddof=1))
    true_components = {
        "repeatability": SIGMA_REPEAT**2,
        "operator": SIGMA_OPERATOR**2,
        "part_operator": SIGMA_INTERACTION**2,
        "part": true_part_variance,
    }
    gauge_component_rows: list[Row] = [
        {
            "source": name,
            "nominal_generating_variance": true_components[name],
            "one_study_anova_variance": gauge[name],
            "nominal_generating_sd": float(np.sqrt(true_components[name])),
            "one_study_anova_sd": float(np.sqrt(gauge[name])),
            "note": (
                "Part term is the realized spread of selected reference parts, "
                "not a production-process variance."
                if name == "part"
                else "The generating variance is a simulation parameter."
            ),
        }
        for name in ("repeatability", "operator", "part_operator", "part")
    ]
    n_gauge_repetitions = 70 if quick else 300
    gauge_results = np.empty((n_gauge_repetitions, 2), dtype=np.float64)
    repeated_rng = generator(21)
    for index in range(n_gauge_repetitions):
        result = crossed_anova_components(simulate_gauge_design(repeated_rng).values)
        gauge_results[index] = [
            result["gauge_sd"],
            result["percent_study_variation"],
        ]

    # A known traceable master can identify overall measurement bias; the
    # crossed R&R design alone has no independent truth for that purpose.
    master_rng = generator(30)
    master_operators = np.tile(np.arange(N_OPERATORS), 60)
    master_observations = (
        TARGET
        + GAUGE_BIAS
        + gauge_design.operator_offsets[master_operators]
        + master_rng.normal(0.0, SIGMA_REPEAT, len(master_operators))
        + master_rng.normal(0.0, SIGMA_INTERACTION, len(master_operators))
    )
    bias_estimate = float(np.mean(master_observations) - TARGET)

    inspection_rng = generator(31)
    n_inspection = 30_000 if quick else 200_000
    inspected_truth = np.asarray(
        inspection_rng.normal(INSPECTION_MU, INSPECTION_SIGMA, n_inspection),
        dtype=np.float64,
    )
    operator_choices = inspection_rng.integers(0, N_OPERATORS, n_inspection)
    raw = np.asarray(
        inspected_truth
        + GAUGE_BIAS
        + gauge_design.operator_offsets[operator_choices]
        + inspection_rng.normal(0.0, SIGMA_INTERACTION, n_inspection)
        + inspection_rng.normal(0.0, SIGMA_REPEAT, n_inspection),
        dtype=np.float64,
    )
    corrected = np.asarray(raw - bias_estimate, dtype=np.float64)
    inspection_rows: list[Row] = []
    for label, readings in (("raw", raw), ("master_bias_corrected", corrected)):
        inspection_rows.append(
            {
                "reading_policy": label,
                "guardband": 0.0,
                **inspection_classification(inspected_truth, readings),
            }
        )
    guardband_rows = [
        {
            "reading_policy": "master_bias_corrected",
            "guardband": float(g),
            **inspection_classification(inspected_truth, corrected, guardband=float(g)),
        }
        for g in np.linspace(0.0, 0.25, 11)
    ]
    (factorial_rows, estimated_beta, standard_errors, coefficient_rows, setting_rows) = (
        simulate_factorial(generator(40), bias_estimate)
    )
    true_gauge_variance = SIGMA_REPEAT**2 + SIGMA_OPERATOR**2 + SIGMA_INTERACTION**2
    summary_rows: list[Row] = [
        {
            "seed": MASTER_SEED,
            "subgroups": N_SUBGROUPS,
            "baseline_subgroups": N_BASELINE,
            "parts_per_subgroup": SUBGROUP_SIZE,
            "latent_sigma_within": SIGMA_PROCESS,
            "known_mean_shift": MU_SHIFT,
            "xbar_lcl": limits["xbar_lcl"],
            "xbar_ucl": limits["xbar_ucl"],
            "range_ucl": limits["range_ucl"],
            "baseline_xbar_alarms": sum(
                int(row["xbar_out_of_control"]) for row in subgroup_rows[:N_BASELINE]
            ),
            "postshift_xbar_alarms": sum(
                int(row["xbar_out_of_control"]) for row in subgroup_rows[N_BASELINE:]
            ),
            "true_baseline_cpk": capability(MU_BASELINE, SIGMA_PROCESS)["cpk"],
            "measured_baseline_cpk": float(capability_rows[2]["cpk"]),
            "pooled_measured_ppk_style": float(capability_rows[-1]["cpk"]),
            "injected_bias": GAUGE_BIAS,
            "master_estimated_bias": bias_estimate,
            "nominal_gauge_sd": float(np.sqrt(true_gauge_variance)),
            "estimated_gauge_sd": gauge["gauge_sd"],
            "one_study_percent_gauge_study_variation": gauge["percent_study_variation"],
            "inspection_count": n_inspection,
            "inspection_true_bad": int(inspection_rows[0]["truly_nonconforming"]),
            "raw_false_accepts_per_10000": float(inspection_rows[0]["false_accepts_per_10000"]),
            "raw_false_rejects_per_10000": float(inspection_rows[0]["false_rejects_per_10000"]),
            "corrected_false_accepts_per_10000": float(inspection_rows[1]["false_accepts_per_10000"]),
            "corrected_false_rejects_per_10000": float(inspection_rows[1]["false_rejects_per_10000"]),
            "true_shifted_normal_defect_probability": normal_nonconformance(
                MU_BASELINE + MU_SHIFT, SIGMA_PROCESS
            ),
            "best_factorial_fitted_normal_defect_probability": min(
                float(row["fitted_normal_defect_rate"]) for row in setting_rows
            ),
        }
    ]
    return IndustrialStudy(
        true_process=truth,
        measured_process=measured,
        subgroup_rows=subgroup_rows,
        limits=limits,
        capability_rows=capability_rows,
        gauge_design=gauge_design,
        gauge_components=gauge,
        gauge_studies=gauge_results,
        gauge_rows=gauge_rows,
        gauge_component_rows=gauge_component_rows,
        inspection_true=inspected_truth,
        inspection_raw=raw,
        inspection_corrected=corrected,
        inspection_rows=inspection_rows,
        guardband_rows=guardband_rows,
        factorial_rows=factorial_rows,
        factorial_effect_rows=coefficient_rows,
        factorial_setting_rows=setting_rows,
        estimated_beta=estimated_beta,
        beta_standard_errors=standard_errors,
        bias_estimate=bias_estimate,
        summary_rows=summary_rows,
    )


def write_csv(path: Path, rows: list[Row]) -> None:
    """Write a deterministic UTF-8 CSV without modifying unrelated outputs."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if not rows:
        raise ValueError("nonempty rows are required")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_figures(study: IndustrialStudy, directory: Path, *, suffix: str = "") -> None:
    """Render eight deterministic figures using full or quick experiment data."""
    if not isinstance(study, IndustrialStudy) or not isinstance(directory, Path):
        raise TypeError("expected IndustrialStudy and Path")
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False}
    )
    navy, teal, orange, red = "#385779", "#4a9290", "#d08e47", "#b64d4a"
    indices = np.arange(N_SUBGROUPS)
    true_mean = study.true_process.mean(axis=1)
    measured_mean = study.measured_process.mean(axis=1)

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.plot(indices, true_mean, color=navy, label="Latent physical subgroup mean", lw=1.6)
    ax.plot(indices, measured_mean, color=orange, label="Biased gauge subgroup mean", alpha=0.88)
    ax.axvline(N_BASELINE - 0.5, ls="--", color=red, label="Known process intervention")
    ax.axhline(MU_BASELINE, ls=":", color=navy, alpha=0.65)
    ax.axhline(MU_BASELINE + MU_SHIFT, ls=":", color=red, alpha=0.65)
    ax.set_title("Time order reveals a regime change that a histogram cannot locate")
    ax.set_xlabel("Subgroup (five contemporaneous parts each)")
    ax.set_ylabel("Dimension (synthetic units)")
    ax.grid(alpha=0.20)
    ax.legend()
    fig.savefig(directory / f"fig1_process_time_order{suffix}.png", dpi=175)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8), sharex=True, constrained_layout=True)
    limits = study.limits
    for ax, field, center, lcl, ucl, ylabel in (
        (
            axes[0],
            "measured_subgroup_mean",
            "xbar_center",
            "xbar_lcl",
            "xbar_ucl",
            "Subgroup mean",
        ),
        (
            axes[1],
            "measured_subgroup_range",
            "range_center",
            "range_lcl",
            "range_ucl",
            "Within-subgroup range",
        ),
    ):
        values = np.array([float(row[field]) for row in study.subgroup_rows])
        signals = (values < limits[lcl]) | (values > limits[ucl])
        ax.plot(indices, values, color=navy, lw=1.0, marker="o", markersize=2.5)
        ax.scatter(indices[signals], values[signals], color=red, zorder=4, s=32)
        ax.axhline(limits[center], color=teal, ls=":", label="Frozen reference center")
        ax.axhline(limits[lcl], color=orange, ls="--", label="Frozen 3σ limits")
        ax.axhline(limits[ucl], color=orange, ls="--")
        ax.axvline(N_BASELINE - 0.5, ls="--", color=red, alpha=0.8)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_title("Phase-I X̄ and R limits remain frozen during the shifted phase")
    axes[-1].set_xlabel("Subgroup index")
    fig.savefig(directory / f"fig2_xbar_r_control_charts{suffix}.png", dpi=175)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    baseline_truth = study.true_process[:N_BASELINE].ravel()
    baseline_measured = study.measured_process[:N_BASELINE].ravel()
    pooled = study.measured_process.ravel()
    axes[0].hist(baseline_truth, bins=24, density=True, alpha=0.65, color=teal, label="Latent truth")
    axes[0].hist(
        baseline_measured, bins=24, density=True, alpha=0.45, color=orange, label="Measured"
    )
    baseline_cpk = float(study.capability_rows[0]["cpk"])
    measured_cpk = float(study.capability_rows[2]["cpk"])
    axes[0].set_title(f"Reference phase: true Cpk={baseline_cpk:.2f}, measured={measured_cpk:.2f}")
    axes[0].legend(fontsize=9)
    axes[1].hist(pooled, bins=28, density=True, alpha=0.78, color=navy)
    descriptive = float(study.capability_rows[-1]["cpk"])
    axes[1].set_title(f"Mixed phases: pooled Ppk-style={descriptive:.2f} (not capability)")
    for ax in axes:
        ax.axvline(LSL, ls="--", color=red, label="Specifications")
        ax.axvline(USL, ls="--", color=red)
        ax.axvline(TARGET, ls=":", color="black", alpha=0.65)
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)
    fig.suptitle("Measurement noise and process mixture change a capability story")
    fig.savefig(directory / f"fig3_capability_vs_mixture{suffix}.png", dpi=175)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    components = study.gauge_component_rows
    positions = np.arange(len(components))
    width = 0.36
    truth_components = [float(r["nominal_generating_sd"]) for r in components]
    fitted_components = [float(r["one_study_anova_sd"]) for r in components]
    axes[0].bar(positions - width / 2, truth_components, width, color=teal, label="Known scale")
    axes[0].bar(positions + width / 2, fitted_components, width, color=orange, label="One ANOVA study")
    axes[0].set_xticks(positions, ["Repeat", "Operator", "Part×operator", "Selected parts"], rotation=14)
    axes[0].set_title("Random-effects variance components")
    axes[0].set_ylabel("Standard deviation (synthetic units)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)
    nominal_gauge = float(np.sqrt(SIGMA_REPEAT**2 + SIGMA_OPERATOR**2 + SIGMA_INTERACTION**2))
    axes[1].hist(study.gauge_studies[:, 0], bins=24, alpha=0.84, color=navy)
    axes[1].axvline(nominal_gauge, ls="--", lw=2, color=red, label="Nominal gauge SD")
    axes[1].axvline(
        study.gauge_components["gauge_sd"], ls=":", lw=2, color=orange, label="Illustrative study"
    )
    axes[1].set_title("Gauge SD varies across independent crossed studies")
    axes[1].set_xlabel("ANOVA-estimated total gauge SD")
    axes[1].set_ylabel("Independent studies")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.suptitle("Gauge R&R estimates have sampling uncertainty")
    fig.savefig(directory / f"fig4_gauge_rr_variance_components{suffix}.png", dpi=175)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    selected = np.argsort(study.gauge_design.part_truth)
    sorted_truth = study.gauge_design.part_truth[selected]
    xs = np.arange(N_PARTS)
    axes[0].plot(xs, sorted_truth, color="black", lw=1.3, ls="--", label="Known part truth")
    for op, (label, color) in enumerate(
        (("Operator 1", navy), ("Operator 2", teal), ("Operator 3", orange))
    ):
        observed = study.gauge_design.values[selected, op, :]
        axes[0].errorbar(
            xs,
            observed.mean(axis=1),
            yerr=observed.std(axis=1, ddof=1),
            fmt="o-",
            markersize=3.5,
            lw=1.1,
            capsize=2,
            color=color,
            label=label,
        )
    axes[0].set_title("Every operator re-measures the same twenty reference parts")
    axes[0].set_ylabel("Measured dimension; bars = within-cell SD")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].grid(alpha=0.19)
    residuals = [
        (
            study.gauge_design.values[:, op, :]
            - study.gauge_design.part_truth[:, None]
            - GAUGE_BIAS
        ).ravel()
        for op in range(N_OPERATORS)
    ]
    box = axes[1].boxplot(
        residuals,
        patch_artist=True,
        tick_labels=["Operator 1", "Operator 2", "Operator 3"],
    )
    for patch, color in zip(box["boxes"], (navy, teal, orange)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].axhline(0.0, color=red, ls=":")
    axes[1].set_title("Operator-dependent residual offsets remain after removing known part truth and bias")
    axes[1].set_ylabel("Measurement minus latent dimension minus fixed bias")
    axes[1].grid(axis="y", alpha=0.2)
    fig.savefig(directory / f"fig5_operator_repeatability_profiles{suffix}.png", dpi=175)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    near = np.flatnonzero(
        (study.inspection_true >= USL - 0.38)
        & (study.inspection_true <= USL + 0.25)
    )
    selection_rng = generator(70)
    chosen = selection_rng.choice(near, size=min(1600, near.size), replace=False)
    true = study.inspection_true[chosen]
    measured = study.inspection_corrected[chosen]
    good = (true >= LSL) & (true <= USL)
    accepted = (measured >= LSL) & (measured <= USL)
    for mask, label, color in (
        (good & accepted, "Correct accept", teal),
        ((~good) & (~accepted), "Correct reject", navy),
        (good & (~accepted), "False reject", orange),
        ((~good) & accepted, "False accept", red),
    ):
        axes[0].scatter(true[mask], measured[mask], s=10, alpha=0.45, color=color, label=label)
    axes[0].axvline(USL, color="black", ls="--")
    axes[0].axhline(USL, color="black", ls="--")
    axes[0].set_xlabel("Known latent dimension")
    axes[0].set_ylabel("Bias-corrected gauge reading")
    axes[0].set_title("Errors concentrate near an inspection limit")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.18)
    positions = np.arange(2)
    width = 0.36
    labels = ["Raw", "Master-corrected"]
    fa = [float(row["false_accepts_per_10000"]) for row in study.inspection_rows]
    fr = [float(row["false_rejects_per_10000"]) for row in study.inspection_rows]
    axes[1].bar(positions - width / 2, fa, width, color=red, label="False accepts")
    axes[1].bar(positions + width / 2, fr, width, color=orange, label="False rejects")
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("Mistakes per 10,000 produced parts")
    axes[1].set_title("Correcting bias changes both types of error")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("A specification decision is made on a measurement, not on truth")
    fig.savefig(directory / f"fig6_inspection_misclassification{suffix}.png", dpi=175)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    bands = np.array([float(row["guardband"]) for row in study.guardband_rows])
    fa = np.array([float(row["false_accepts_per_10000"]) for row in study.guardband_rows])
    fr = np.array([float(row["false_rejects_per_10000"]) for row in study.guardband_rows])
    loss = np.array([float(row["illustrative_loss_per_10000"]) for row in study.guardband_rows])
    axes[0].plot(bands, fa, "-o", color=red, label="False accepts")
    axes[0].plot(bands, fr, "-o", color=navy, label="False rejects")
    axes[0].set_title("Guard bands trade consumer and producer errors")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mistakes per 10,000 parts (log scale)")
    axes[0].legend()
    minimum = int(np.argmin(loss))
    axes[1].plot(bands, loss, "-o", color=orange, label="Illustrative weighted loss")
    axes[1].scatter([bands[minimum]], [loss[minimum]], color=red, s=55, zorder=3)
    axes[1].set_title(f"Assumed loss ratio {FALSE_ACCEPT_COST:.0f}:1; minimum near g={bands[minimum]:.3f}")
    axes[1].set_ylabel("Arbitrary loss units per 10,000 parts")
    for ax in axes:
        ax.set_xlabel("Inward guard band on each specification limit")
        ax.grid(alpha=0.2)
    fig.suptitle("A measurement policy requires explicit error costs")
    fig.savefig(directory / f"fig7_guardband_tradeoff{suffix}.png", dpi=175)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    factorial = [row for row in study.factorial_rows if int(row["centerpoint"]) == 0]
    for temperature, label, color in (
        (-1.0, "Temperature −1", teal),
        (1.0, "Temperature +1", orange),
    ):
        means = []
        errors = []
        for speed in (-1.0, 1.0):
            values = np.array(
                [
                    float(row["measured_corrected_mean"])
                    for row in factorial
                    if float(row["speed_coded"]) == speed
                    and float(row["temperature_coded"]) == temperature
                ]
            )
            means.append(float(np.mean(values)))
            errors.append(float(2.0 * np.std(values, ddof=1) / np.sqrt(len(values))))
        axes[0].errorbar(
            (-1, 1), means, yerr=errors, fmt="o-", capsize=5, color=color, label=label
        )
    axes[0].set_xticks((-1, 1), ("Speed −1", "Speed +1"))
    axes[0].set_ylabel("Corrected mean dimension; bars = ±2 standard errors")
    axes[0].set_title("Nonparallel lines reveal an interaction")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    settings = study.factorial_setting_rows
    labels = [
        f"Speed {int(row['speed_coded']):+d}\nTemp {int(row['temperature_coded']):+d}"
        for row in settings
    ]
    risk = [10_000 * float(row["fitted_normal_defect_rate"]) for row in settings]
    positions = np.arange(len(settings))
    bars = axes[1].bar(positions, risk, color=[navy, teal, orange, red])
    for bar, rate in zip(bars, risk):
        # Place large-bar labels inside to avoid obscuring the reference line.
        inside = rate > 50
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            rate - 7 if inside else rate + 2,
            f"{rate:.1f}",
            va="top" if inside else "bottom",
            ha="center",
            color="white" if inside else "black",
            fontsize=8,
        )
    axes[1].axhline(
        10_000 * normal_nonconformance(MU_BASELINE + MU_SHIFT, SIGMA_PROCESS),
        color="black",
        ls="--",
        label="Shifted production reference",
    )
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("Model-implied nonconforming parts per 10,000")
    axes[1].set_title("Risk projection assumes constant, known physical σ")
    axes[1].set_ylim(0, 265)
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Design experiments to change the process, then validate the tail risk")
    fig.savefig(directory / f"fig8_factorial_interaction_and_risk{suffix}.png", dpi=175)
    plt.close(fig)


def main() -> None:
    """Write the full deterministic evidence, or separately suffixed quick outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller smoke run")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if not isinstance(args.output_root, Path):
        raise TypeError("output root must be a pathlib.Path")
    study = run_study(quick=bool(args.quick))
    suffix = "_quick" if args.quick else ""
    tables = (
        ("summary", study.summary_rows),
        ("process_subgroups", study.subgroup_rows),
        ("capability", study.capability_rows),
        ("gage_rr_study", study.gauge_rows),
        ("gage_rr_components", study.gauge_component_rows),
        ("inspection", study.inspection_rows),
        ("inspection_guardbands", study.guardband_rows),
        ("factorial_runs", study.factorial_rows),
        ("factorial_effects", study.factorial_effect_rows),
        ("factorial_settings", study.factorial_setting_rows),
    )
    for name, rows in tables:
        write_csv(args.output_root / "results" / f"{name}{suffix}.csv", rows)
    make_figures(study, args.output_root / "figures", suffix=suffix)
    for key, value in study.summary_rows[0].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
