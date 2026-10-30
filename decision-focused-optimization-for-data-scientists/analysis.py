"""Reproducible synthetic prediction-to-decision and stochastic LP experiments.

This study deliberately separates prediction error, expected decision loss,
tail-risk, nonanticipativity, and information value. Its generated demand
processes and their parameters are not estimates of real supply chains.

Usage:
    python analysis.py
    python analysis.py --quick
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog
from scipy.stats import norm

FloatArray = NDArray[np.float64]
SEED: int = 20261030
UNDER_SINGLE: float = 8.0
OVER_SINGLE: float = 2.0
UNDER_JOINT: FloatArray = np.array([8.0, 6.0])
OVER_JOINT: FloatArray = np.array([2.0, 3.0])
CAPACITY: float = 190.0
RHO: float = 0.65
JOINT_MEANS: FloatArray = np.array([90.0, 70.0])
JOINT_SIGMAS: FloatArray = np.array([0.34, 0.44])
RISK_WEIGHTS: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
FIGURE_NAMES: tuple[str, ...] = (
    "fig1_rmse_cost_reversal.png",
    "fig2_critical_fractile.png",
    "fig3_loss_surface.png",
    "fig4_saa_convergence.png",
    "fig5_capacity_frontier.png",
    "fig6_out_of_sample_regret.png",
    "fig7_risk_cvar_tradeoff.png",
    "fig8_shift_and_information_value.png",
)
RESULT_NAMES: tuple[str, ...] = (
    "summary.csv",
    "single_product_predictions.csv",
    "critical_fractile.csv",
    "single_product_loss_surface.csv",
    "saa_convergence.csv",
    "capacity_scenarios.csv",
    "allocation_policies.csv",
    "capacity_frontier.csv",
    "stress_tests.csv",
    "perfect_information.csv",
)


@dataclass(frozen=True)
class DemandFit:
    """Log-mean regression plus training-only log residual scale."""

    beta: FloatArray
    sigma: float


@dataclass(frozen=True)
class Study:
    """All generated evidence needed for figures, tables, and the manuscript."""

    fit: DemandFit
    x_test: FloatArray
    y_test: FloatArray
    mean_prediction: FloatArray
    quantile_prediction: FloatArray
    single_summary: dict[str, float]
    critical: list[dict[str, float]]
    loss_surface: list[dict[str, float]]
    convergence: list[dict[str, float]]
    scenarios: FloatArray
    policies: dict[str, FloatArray]
    risk_tradeoff: list[dict[str, float]]
    frontier: list[dict[str, float]]
    stress: list[dict[str, float | str]]
    information: list[dict[str, float | str]]
    baseline_joint: FloatArray
    summary: dict[str, float]
    rows: dict[str, list[dict[str, Any]]]


def _size(n: int, name: str, minimum: int = 1) -> None:
    """Validate sample-size input without bool or fractional coercion."""

    if type(n) is not int or n < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _probability(x: float, name: str, allow_boundary: bool = False) -> None:
    """Require a finite scalar strictly inside (0,1) unless noted."""

    if isinstance(x, (bool, np.bool_)) or not np.isscalar(x):
        raise ValueError(f"{name} must be numeric")
    if not np.isfinite(x) or not (0 <= x <= 1 if allow_boundary else 0 < x < 1):
        raise ValueError(f"{name} out of range")


def single_features(x: FloatArray) -> FloatArray:
    """Full-rank basis for the synthetic conditional log-demand model."""

    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise ValueError("x must be a finite vector in [0,1]")
    return np.column_stack([np.ones(len(x)), np.sin(2 * np.pi * x), x])


def single_log_mean(x: FloatArray) -> FloatArray:
    """Known log-demand conditional mean, used only to generate data."""

    return single_features(x) @ np.array([np.log(85.0), 0.20, 0.12])


def single_batch(rng: np.random.Generator, n: int, sigma: float = 0.45) -> tuple[FloatArray, FloatArray]:
    """Draw positive, conditionally lognormal demand with independent features."""

    _size(n, "n")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    x = rng.uniform(size=n)
    y = np.exp(single_log_mean(x) + sigma * rng.standard_normal(n))
    return x, y


def fit_single(x: FloatArray, demand: FloatArray) -> DemandFit:
    """Fit conditional log demand and its residual SD on a training split."""

    demand = np.asarray(demand, dtype=float)
    if demand.ndim != 1 or len(demand) < 8 or len(demand) != len(x):
        raise ValueError("at least eight aligned training observations required")
    if not np.all(np.isfinite(demand)) or np.any(demand <= 0):
        raise ValueError("training demand must be finite and strictly positive")
    features = single_features(x)
    log_demand = np.log(demand)
    beta = np.linalg.lstsq(features, log_demand, rcond=None)[0]
    sigma = float(np.std(log_demand - features @ beta, ddof=3))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("estimated sigma must be positive")
    return DemandFit(beta, sigma)


def mean_forecast(fit: DemandFit, x: FloatArray) -> FloatArray:
    """Conditional mean forecast under the fitted lognormal approximation."""

    return np.exp(single_features(x) @ fit.beta + 0.5 * fit.sigma**2)


def quantile_forecast(fit: DemandFit, x: FloatArray, tau: float) -> FloatArray:
    """Fitted conditional tau-quantile, the newsvendor decision for that loss."""

    _probability(tau, "tau")
    return np.exp(single_features(x) @ fit.beta + fit.sigma * norm.ppf(tau))


def newsvendor_loss(
    quantity: FloatArray | float, demand: FloatArray, under_cost: float = UNDER_SINGLE,
    over_cost: float = OVER_SINGLE,
) -> FloatArray:
    """Asymmetric positive-part loss; quantity and demand broadcast normally."""

    if not np.isfinite(under_cost) or not np.isfinite(over_cost) or min(under_cost, over_cost) <= 0:
        raise ValueError("shortage and excess costs must be positive")
    quantity = np.asarray(quantity, dtype=float)
    demand = np.asarray(demand, dtype=float)
    if not np.all(np.isfinite(quantity)) or not np.all(np.isfinite(demand)):
        raise ValueError("inputs must be finite")
    if np.any(quantity < 0) or np.any(demand < 0):
        raise ValueError("quantities and demand must be nonnegative")
    return under_cost * np.maximum(demand - quantity, 0) + over_cost * np.maximum(quantity - demand, 0)


def empirical_saa_quantity(demand: FloatArray, tau: float) -> float:
    """Smallest empirical tau-quantile minimizing sample-average newsvendor loss."""

    _probability(tau, "tau")
    demand = np.asarray(demand, dtype=float)
    if demand.ndim != 1 or len(demand) < 1 or not np.all(np.isfinite(demand)) or np.any(demand < 0):
        raise ValueError("demand must be a nonnegative finite vector")
    rank = int(np.ceil(len(demand) * tau))
    return float(np.partition(demand, rank - 1)[rank - 1])


def joint_demands(
    rng: np.random.Generator,
    n: int,
    *,
    rho: float = RHO,
    volatility_factor: float = 1.0,
) -> FloatArray:
    """Draw correlated lognormal outcomes with fixed means.

    Changing the Gaussian copula correlation alters dependence but not the
    marginal distribution. Increasing volatility keeps each marginal mean
    fixed via the minus-sigma-squared-over-two normalization.
    """

    _size(n, "n")
    if not np.isfinite(rho) or not -1 < rho < 1:
        raise ValueError("rho must be strictly between -1 and 1")
    if not np.isfinite(volatility_factor) or volatility_factor <= 0:
        raise ValueError("volatility_factor must be positive")
    independent = rng.standard_normal((n, 2))
    z = np.column_stack([
        independent[:, 0],
        rho * independent[:, 0] + np.sqrt(1 - rho**2) * independent[:, 1],
    ])
    sigma = JOINT_SIGMAS * volatility_factor
    return JOINT_MEANS * np.exp(z * sigma - 0.5 * sigma**2)


def joint_losses(
    quantities: FloatArray, demand: FloatArray,
    under_cost: FloatArray = UNDER_JOINT, over_cost: FloatArray = OVER_JOINT,
) -> FloatArray:
    """Compute each two-product scenario's total shortage and excess cost."""

    quantities = np.asarray(quantities, dtype=float)
    demand = np.asarray(demand, dtype=float)
    if quantities.shape != (2,) or demand.ndim != 2 or demand.shape[1] != 2:
        raise ValueError("expected two quantities and an n-by-two demand matrix")
    if not np.all(np.isfinite(quantities)) or not np.all(np.isfinite(demand)):
        raise ValueError("inputs must be finite")
    if np.any(quantities < 0) or np.any(demand < 0):
        raise ValueError("demand and allocation must be nonnegative")
    return np.sum(
        under_cost * np.maximum(demand - quantities, 0)
        + over_cost * np.maximum(quantities - demand, 0),
        axis=1,
    )


def empirical_cvar(losses: FloatArray, beta: float = 0.90) -> float:
    """Rockafellar-Uryasev upper-tail CVaR of an empirical loss sample."""

    _probability(beta, "beta")
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1 or len(losses) < 1 or not np.all(np.isfinite(losses)):
        raise ValueError("losses must be finite and nonempty")
    eta = float(np.quantile(losses, beta, method="higher"))
    return float(eta + np.maximum(losses - eta, 0).mean() / (1.0 - beta))


def solve_allocation(
    scenarios: FloatArray, capacity: float = CAPACITY,
    *,
    risk_weight: float = 0.0, beta: float = 0.90,
) -> FloatArray:
    """Solve a nonanticipative scenario LP for expected cost / CVaR mixture.

    Decision q is shared across scenarios. Scenario-dependent nonnegative
    slack variables represent shortage and excess. For risk_weight > 0 the
    LP additionally optimizes eta + mean((loss_s - eta)+)/(1-beta).

    The scenario probabilities are equal; this is sample-average
    approximation, not a robust-out-of-sample guarantee.
    """

    scenarios = np.asarray(scenarios, dtype=float)
    if scenarios.ndim != 2 or scenarios.shape[1] != 2 or len(scenarios) < 1:
        raise ValueError("scenarios must have shape (n,2) with n >= 1")
    if not np.all(np.isfinite(scenarios)) or np.any(scenarios < 0):
        raise ValueError("scenarios must be finite, nonnegative")
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("capacity must be positive")
    _probability(risk_weight, "risk_weight", allow_boundary=True)
    _probability(beta, "beta")
    n = len(scenarios)
    n_regular = 2 + 4 * n
    with_risk = risk_weight > 0.0
    n_variables = n_regular + (1 + n if with_risk else 0)
    under_index = np.arange(2, 2 + 2 * n).reshape(n, 2)
    over_index = np.arange(2 + 2 * n, 2 + 4 * n).reshape(n, 2)
    eta_index = n_regular
    tail_index = n_regular + 1 + np.arange(n)
    objective = np.zeros(n_variables)
    for j in range(2):
        objective[under_index[:, j]] = (1 - risk_weight) * UNDER_JOINT[j] / n
        objective[over_index[:, j]] = (1 - risk_weight) * OVER_JOINT[j] / n
    if with_risk:
        objective[eta_index] = risk_weight
        objective[tail_index] = risk_weight / ((1 - beta) * n)
    constraints = 1 + 4 * n + (n if with_risk else 0)
    a = np.zeros((constraints, n_variables))
    b = np.zeros(constraints)
    a[0, :2] = 1.0
    b[0] = capacity
    row = 1
    for s in range(n):
        for j in range(2):
            # u_short >= demand - q, so -q - u_short <= -demand.
            a[row, j] = -1.0
            a[row, under_index[s, j]] = -1.0
            b[row] = -scenarios[s, j]
            row += 1
            # u_excess >= q - demand, so q - u_excess <= demand.
            a[row, j] = 1.0
            a[row, over_index[s, j]] = -1.0
            b[row] = scenarios[s, j]
            row += 1
    if with_risk:
        for s in range(n):
            a[row, under_index[s]] = UNDER_JOINT
            a[row, over_index[s]] = OVER_JOINT
            a[row, eta_index] = -1.0
            a[row, tail_index[s]] = -1.0
            row += 1
    assert row == constraints
    bounds: list[tuple[float | None, float | None]] = (
        [(0.0, capacity), (0.0, capacity)]
        + [(0.0, None)] * (4 * n)
        + ([(None, None)] + [(0.0, None)] * n if with_risk else [])
    )
    solution = linprog(objective, A_ub=a, b_ub=b, bounds=bounds, method="highs")
    if not solution.success or solution.x is None:
        raise RuntimeError(f"scenario LP did not solve: {solution.message}")
    quantities = np.asarray(solution.x[:2], dtype=float)
    if np.any(quantities < -1e-7) or float(quantities.sum()) > capacity + 1e-7:
        raise RuntimeError("LP returned infeasible quantities")
    return np.maximum(quantities, 0.0)


def clairvoyant_cost(demand: FloatArray, capacity: float = CAPACITY) -> FloatArray:
    """Perfect-information two-product cost, choosing q after observing demand.

    Because product 1 has larger shortage penalty, a hindsight oracle fills
    it first up to demand, then fills product 2. This decision is unavailable
    before demand arrives and is used only to bound information value.
    """

    demand = np.asarray(demand, dtype=float)
    if demand.ndim != 2 or demand.shape[1] != 2 or not np.all(np.isfinite(demand)) or np.any(demand < 0):
        raise ValueError("demand must be a finite nonnegative n-by-two array")
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("capacity must be positive")
    q1 = np.minimum(demand[:, 0], capacity)
    q2 = np.minimum(demand[:, 1], np.maximum(capacity - q1, 0))
    return UNDER_JOINT[0] * (demand[:, 0] - q1) + UNDER_JOINT[1] * (demand[:, 1] - q2)


def monte_carlo_saa(
    *,
    replicates: int = 80,
    n_evaluation: int = 30000,
) -> list[dict[str, float]]:
    """Estimate SAA decision regret against a known simulation-only quantile."""

    _size(replicates, "replicates")
    _size(n_evaluation, "n_evaluation", 100)
    x0 = np.array([0.35])
    tau = UNDER_SINGLE / (UNDER_SINGLE + OVER_SINGLE)
    oracle_q = float(np.exp(single_log_mean(x0)[0] + 0.45 * norm.ppf(tau)))
    test_rng = np.random.default_rng(np.random.SeedSequence([SEED, 311]))
    eval_demand = np.exp(single_log_mean(x0)[0] + 0.45 * test_rng.standard_normal(n_evaluation))
    oracle_cost = float(newsvendor_loss(oracle_q, eval_demand).mean())
    rows = []
    for n in (8, 16, 32, 64, 128, 256, 512, 1024):
        outcomes = []
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 312, n]))
        for _ in range(replicates):
            sample = np.exp(single_log_mean(x0)[0] + 0.45 * rng.standard_normal(n))
            q_saa = empirical_saa_quantity(sample, tau)
            outcomes.append((q_saa, float(newsvendor_loss(q_saa, eval_demand).mean() - oracle_cost)))
        matrix = np.asarray(outcomes)
        rows.append({
            "scenarios": float(n),
            "mean_quantity": float(matrix[:, 0].mean()),
            "quantity_sd": float(matrix[:, 0].std(ddof=1)),
            "mean_decision_regret": float(matrix[:, 1].mean()),
            "regret_sd": float(matrix[:, 1].std(ddof=1)),
            "oracle_quantity": oracle_q,
            "oracle_holdout_cost": oracle_cost,
        })
    return rows


def _generator(key: int) -> np.random.Generator:
    """Independent named random number stream for every study component."""

    return np.random.default_rng(np.random.SeedSequence([SEED, key]))


def run_study(*, quick: bool = False) -> Study:
    """Run a complete and deliberately transparent predict/optimize study."""

    if type(quick) is not bool:
        raise ValueError("quick must be bool")
    train_x, train_y = single_batch(_generator(101), 5000)
    fit = fit_single(train_x, train_y)
    test_x, test_y = single_batch(_generator(102), 10000 if quick else 60000)
    tau = UNDER_SINGLE / (UNDER_SINGLE + OVER_SINGLE)
    forecast_mean = mean_forecast(fit, test_x)
    forecast_quantile = quantile_forecast(fit, test_x, tau)
    point_cost = newsvendor_loss(forecast_mean, test_y)
    quantile_cost = newsvendor_loss(forecast_quantile, test_y)
    single_summary = {
        "mean_rmse": float(np.sqrt(np.mean((forecast_mean - test_y)**2))),
        "quantile_rmse": float(np.sqrt(np.mean((forecast_quantile - test_y)**2))),
        "mean_decision_cost": float(point_cost.mean()),
        "quantile_decision_cost": float(quantile_cost.mean()),
        "quantity_mean": float(forecast_mean.mean()),
        "quantity_quantile": float(forecast_quantile.mean()),
        "training_log_sigma": fit.sigma,
    }
    x0 = np.array([0.35])
    eval_fixed = np.exp(single_log_mean(x0)[0] + 0.45 * _generator(103).standard_normal(35000 if not quick else 7000))
    critical: list[dict[str, float]] = []
    for under_cost in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0):
        frac = under_cost / (under_cost + OVER_SINGLE)
        predicted_q = float(quantile_forecast(fit, x0, frac)[0])
        true_q = float(np.exp(single_log_mean(x0)[0] + 0.45 * norm.ppf(frac)))
        critical.append({
            "under_cost": under_cost, "over_cost": OVER_SINGLE, "critical_fractile": frac,
            "fitted_quantity": predicted_q, "true_oracle_quantity": true_q,
            "observed_cost": float(newsvendor_loss(predicted_q, eval_fixed, under_cost, OVER_SINGLE).mean()),
        })
    fixed_mean = float(mean_forecast(fit, x0)[0])
    fixed_quantile = float(quantile_forecast(fit, x0, tau)[0])
    surface = []
    for quantity in np.linspace(30, 190, 161):
        loss = newsvendor_loss(float(quantity), eval_fixed)
        surface.append({
            "quantity": float(quantity), "mean_cost": float(loss.mean()),
            "p95_cost": float(np.quantile(loss, 0.95)),
            "cvar90_cost": empirical_cvar(loss),
        })
    convergence = monte_carlo_saa(replicates=22 if quick else 90, n_evaluation=6000 if quick else 30000)
    # Joint simulation is independent of all single-product training and tests.
    scenarios = joint_demands(_generator(201), 100 if quick else 240)
    baseline = joint_demands(_generator(202), 12000 if quick else 70000)
    high_vol = joint_demands(_generator(203), 12000 if quick else 70000, volatility_factor=1.45)
    high_corr = joint_demands(_generator(204), 12000 if quick else 70000, rho=0.92)
    policies: dict[str, FloatArray] = {
        "Deterministic mean": JOINT_MEANS.copy(),
        "SAA expected cost": solve_allocation(scenarios, risk_weight=0.0),
    }
    for weight in RISK_WEIGHTS[1:]:
        policies[f"Mean/CVaR {weight:.2f}"] = solve_allocation(scenarios, risk_weight=weight)
    risk_tradeoff = []
    for weight in RISK_WEIGHTS:
        label = "SAA expected cost" if weight == 0 else f"Mean/CVaR {weight:.2f}"
        q = policies[label]
        costs = joint_losses(q, baseline)
        risk_tradeoff.append({
            "weight_cvar": weight, "q_product1": float(q[0]), "q_product2": float(q[1]),
            "mean_cost": float(costs.mean()), "cvar90": empirical_cvar(costs),
            "p95": float(np.quantile(costs, 0.95)),
        })
    # Compare allocations on the same test scenarios to eliminate sample mismatch.
    frontier = []
    for q1 in np.linspace(25, CAPACITY, 166):
        q = np.array([q1, CAPACITY - q1])
        losses = joint_losses(q, baseline)
        frontier.append({
            "q_product1": float(q1), "q_product2": float(q[1]),
            "mean_cost": float(losses.mean()), "cvar90": empirical_cvar(losses),
        })
    stress = []
    information = []
    for scenario_name, population in (
        ("Original distribution", baseline),
        ("Conditional volatility × 1.45", high_vol),
        ("Correlation 0.65 → 0.92", high_corr),
    ):
        for label, q in policies.items():
            losses = joint_losses(q, population)
            stress.append({
                "scenario": scenario_name, "policy": label,
                "q_product1": float(q[0]), "q_product2": float(q[1]),
                "mean_cost": float(losses.mean()), "cvar90": empirical_cvar(losses),
                "p95": float(np.quantile(losses, 0.95)),
                "capacity_utilized": float(q.sum() / CAPACITY),
            })
        cost_mean = float(joint_losses(policies["Deterministic mean"], population).mean())
        cost_saa = float(joint_losses(policies["SAA expected cost"], population).mean())
        hindsight = float(clairvoyant_cost(population).mean())
        information.append({
            "scenario": scenario_name,
            "mean_policy_cost": cost_mean,
            "saa_policy_cost": cost_saa,
            "perfect_information_cost": hindsight,
            "observed_vss": cost_mean - cost_saa,
            "observed_evpi": cost_saa - hindsight,
        })
    info_base = information[0]
    summary = {
        "n_single_train": 5000.0, "n_single_test": float(len(test_y)),
        "single_under_cost": UNDER_SINGLE, "single_over_cost": OVER_SINGLE,
        "critical_fractile": tau, **single_summary,
        "fixed_x_mean_decision": fixed_mean, "fixed_x_quantile_decision": fixed_quantile,
        "joint_capacity": CAPACITY, "joint_saa_scenarios": float(len(scenarios)),
        "joint_test_scenarios": float(len(baseline)),
        "joint_risk_neutral_q1": float(policies["SAA expected cost"][0]),
        "joint_risk_neutral_q2": float(policies["SAA expected cost"][1]),
        "joint_mean_policy_expected_cost": info_base["mean_policy_cost"],
        "joint_saa_policy_expected_cost": info_base["saa_policy_cost"],
        "joint_hindsight_expected_cost": info_base["perfect_information_cost"],
        "estimated_vss": info_base["observed_vss"],
        "estimated_evpi": info_base["observed_evpi"],
        "joint_neutral_cvar90": risk_tradeoff[0]["cvar90"],
        "joint_cvar_policy_cvar90": risk_tradeoff[-1]["cvar90"],
    }
    rows: dict[str, list[dict[str, Any]]] = {
        "summary.csv": [{"metric": key, "value": value} for key, value in summary.items()],
        "single_product_predictions.csv": [
            {
                "x": float(test_x[i]), "demand": float(test_y[i]),
                "mean_forecast": float(forecast_mean[i]),
                "quantile_forecast": float(forecast_quantile[i]),
                "mean_policy_cost": float(point_cost[i]),
                "quantile_policy_cost": float(quantile_cost[i]),
            }
            for i in range(min(500, len(test_y)))
        ],
        "critical_fractile.csv": critical,
        "single_product_loss_surface.csv": surface,
        "saa_convergence.csv": convergence,
        "capacity_scenarios.csv": [
            {"scenario": i, "product1_demand": float(values[0]), "product2_demand": float(values[1])}
            for i, values in enumerate(scenarios)
        ],
        "allocation_policies.csv": [
            {
                "policy": name, "product1_quantity": float(q[0]), "product2_quantity": float(q[1]),
                "capacity_used": float(q.sum()),
                "baseline_mean_cost": float(joint_losses(q, baseline).mean()),
                "baseline_cvar90": empirical_cvar(joint_losses(q, baseline)),
            }
            for name, q in policies.items()
        ],
        "capacity_frontier.csv": frontier,
        "stress_tests.csv": stress,
        "perfect_information.csv": information,
    }
    if set(rows) != set(RESULT_NAMES):
        raise RuntimeError("result table manifest mismatch")
    return Study(
        fit, test_x, test_y, forecast_mean, forecast_quantile,
        single_summary, critical, surface, convergence, scenarios,
        policies, risk_tradeoff, frontier, stress, information, baseline,
        summary, rows,
    )


def plot_figures(study: Study, directory: Path, *, quick: bool = False) -> list[Path]:
    """Render eight original figures without side effects on publication data."""

    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 105, "savefig.dpi": 150, "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.19,
    })
    created: list[Path] = []
    blue, orange, green, red = "#477da0", "#d28342", "#128b7e", "#bd5041"

    def save(fig: Any, name: str) -> None:
        filename = name.replace(".png", "_quick.png") if quick else name
        path = directory / filename
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        created.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    labels = ["Mean: squared-error optimal", "80% quantile: cost optimal"]
    a = study.single_summary
    axes[0].bar(labels, [a["mean_rmse"], a["quantile_rmse"]], color=[blue, orange])
    axes[0].set(title="Squared-error comparison", ylabel="Holdout RMSE")
    axes[1].bar(labels, [a["mean_decision_cost"], a["quantile_decision_cost"]], color=[blue, orange])
    axes[1].set(title="Same predictions used as inventory orders", ylabel="Average shortage + excess cost")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=14)
    fig.suptitle("Better prediction under RMSE need not imply better decisions")
    save(fig, FIGURE_NAMES[0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1))
    fractiles = np.array([r["critical_fractile"] for r in study.critical])
    axes[0].plot(fractiles, [r["fitted_quantity"] for r in study.critical], "o-", label="Fitted conditional quantile", color=blue)
    axes[0].plot(fractiles, [r["true_oracle_quantity"] for r in study.critical], "--", label="Generating-model oracle", color=red)
    axes[0].set(xlabel="Shortage / total cost = critical fractile", ylabel="Optimal single-product order", title="Higher shortage cost implies a higher quantile")
    axes[0].legend(frameon=False)
    axes[1].plot(fractiles, [r["observed_cost"] for r in study.critical], "o-", color=orange)
    axes[1].set(xlabel="Critical fractile", ylabel="Holdout cost in scenario-specific cost units", title="Cost levels vary with shortage penalty")
    save(fig, FIGURE_NAMES[1])

    quantities = np.array([r["quantity"] for r in study.loss_surface])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1))
    axes[0].plot(quantities, [r["mean_cost"] for r in study.loss_surface], color=blue, lw=2)
    axes[0].axvline(study.summary["fixed_x_mean_decision"], color=red, ls=":", label="Conditional mean")
    axes[0].axvline(study.summary["fixed_x_quantile_decision"], color=green, ls="--", label="Conditional 80% quantile")
    axes[0].set(xlabel="Single product quantity at X = 0.35", ylabel="Expected asymmetric cost", title="The cost surface selects a different decision")
    axes[0].legend(frameon=False)
    axes[1].plot(quantities, [r["cvar90_cost"] for r in study.loss_surface], label="CVaR₉₀", color=orange)
    axes[1].plot(quantities, [r["p95_cost"] for r in study.loss_surface], label="95th loss percentile", color="#9874a9")
    axes[1].set(xlabel="Order quantity", ylabel="Tail-loss measure", title="Expected cost and tail risk are different objectives")
    axes[1].legend(frameon=False)
    save(fig, FIGURE_NAMES[2])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1))
    n = np.array([r["scenarios"] for r in study.convergence])
    regret = np.array([r["mean_decision_regret"] for r in study.convergence])
    regret_sd = np.array([r["regret_sd"] for r in study.convergence])
    axes[0].plot(n, regret, "o-", color=blue, label="Mean holdout regret")
    axes[0].fill_between(n, np.maximum(regret - regret_sd, 0), regret + regret_sd, color=blue, alpha=0.15, label="±1 SD over SAA replicates")
    axes[0].axhline(0, color=red, ls="--")
    axes[0].set(xscale="log", xlabel="Scenarios available before decision", ylabel="Regret versus generating-model oracle", title="More scenarios reduce decision uncertainty")
    axes[0].legend(frameon=False)
    axes[1].plot(n, [r["quantity_sd"] for r in study.convergence], "o-", color=orange)
    axes[1].set(xscale="log", xlabel="Number of scenarios", ylabel="SD of SAA ordering quantity", title="Scenario sampling creates policy variability")
    save(fig, FIGURE_NAMES[3])

    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    qs = [r["q_product1"] for r in study.frontier]
    ax.plot(qs, [r["mean_cost"] for r in study.frontier], color=blue, lw=2.4, label="Mean cost on capacity boundary")
    ax.plot(qs, [r["cvar90"] for r in study.frontier], color=orange, lw=2, label="90% CVaR on same boundary")
    for label, color in (("SAA expected cost", green), ("Mean/CVaR 1.00", red)):
        q = study.policies[label]
        losses = joint_losses(q, study.baseline_joint)
        ax.scatter([q[0]], [losses.mean() if label == "SAA expected cost" else empirical_cvar(losses)], s=85, color=color, zorder=4, label=label)
    ax.set(xlabel="Product 1 allocation (product 2 = 190 − product 1)", ylabel="Holdout scenario cost", title="Shared capacity creates an allocation tradeoff")
    ax.legend(frameon=False, ncol=2)
    save(fig, FIGURE_NAMES[4])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    selected = ("Deterministic mean", "SAA expected cost", "Mean/CVaR 1.00")
    loss_list = [joint_losses(study.policies[name], study.baseline_joint) for name in selected]
    axes[0].boxplot(loss_list, showfliers=False, tick_labels=["Deterministic", "Risk-neutral SAA", "Pure CVaR"])
    axes[0].set(title="Out-of-sample loss distributions", ylabel="Total cost", ylim=(0, max(np.quantile(a, 0.97) for a in loss_list) * 1.2))
    axes[1].bar(np.arange(3) - 0.18, [np.mean(l) for l in loss_list], 0.35, color=blue, label="Mean")
    axes[1].bar(np.arange(3) + 0.18, [empirical_cvar(l) for l in loss_list], 0.35, color=orange, label="CVaR₉₀")
    axes[1].set(xticks=range(3), xticklabels=["Deterministic", "Risk-neutral SAA", "Pure CVaR"], ylabel="Cost", title="Optimizing expected cost is not minimizing tail cost")
    axes[1].legend(frameon=False)
    save(fig, FIGURE_NAMES[5])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    risk = study.risk_tradeoff
    xs, ys = [r["mean_cost"] for r in risk], [r["cvar90"] for r in risk]
    axes[0].plot(xs, ys, "o-", color=green, lw=2.1)
    for r in risk:
        axes[0].annotate(f"λ={r['weight_cvar']:.2f}", (r["mean_cost"], r["cvar90"]), xytext=(4, 5), textcoords="offset points", fontsize=8)
    axes[0].set(xlabel="Out-of-sample expected cost", ylabel="Out-of-sample CVaR₉₀", title="Risk preferences induce different decision policies")
    axes[1].plot([r["weight_cvar"] for r in risk], [r["q_product1"] for r in risk], "o-", color=blue, label="Product 1")
    axes[1].plot([r["weight_cvar"] for r in risk], [r["q_product2"] for r in risk], "s-", color=orange, label="Product 2")
    axes[1].set(xlabel="CVaR weight λ in optimization objective", ylabel="Capacity allocation", title="Risk appetite changes the optimal allocation")
    axes[1].legend(frameon=False)
    save(fig, FIGURE_NAMES[6])

    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.5))
    environments = [r["scenario"] for r in study.information]
    neutral = [next(r for r in study.stress if r["scenario"] == e and r["policy"] == "SAA expected cost")["cvar90"] for e in environments]
    cautious = [next(r for r in study.stress if r["scenario"] == e and r["policy"] == "Mean/CVaR 1.00")["cvar90"] for e in environments]
    x = np.arange(len(environments))
    axes[0].bar(x - 0.18, neutral, 0.35, color=blue, label="Neutral-policy CVaR")
    axes[0].bar(x + 0.18, cautious, 0.35, color=green, label="CVaR-policy CVaR")
    axes[0].set(xticks=x, xticklabels=["Baseline", "Higher volatility", "Higher correlation"], ylabel="Realized CVaR₉₀", title="Fixed policies under two controlled shifts")
    axes[0].legend(frameon=False, fontsize=9)
    info = study.information[0]
    axes[1].bar(["Stochastic-solution\nvalue (VSS)", "Perfect-information\nvalue (EVPI)"], [info["observed_vss"], info["observed_evpi"]], color=[orange, "#9770aa"])
    axes[1].set(ylabel="Estimated expected-loss improvement", title="Simulation-only value-of-information benchmarks")
    save(fig, FIGURE_NAMES[7])
    return created


def write_results(study: Study, directory: Path, *, quick: bool = False) -> list[Path]:
    """Write ten compact CSV tables, never overwriting full results in quick mode."""

    directory.mkdir(parents=True, exist_ok=True)
    created = []
    for name in RESULT_NAMES:
        rows = study.rows[name]
        if not rows:
            raise ValueError(f"No data for {name}")
        actual = name.replace(".csv", "_quick.csv") if quick else name
        path = directory / actual
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        created.append(path)
    return created


def main() -> None:
    """Command-line entrypoint for a full or smoke-test study."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Suffix small smoke-test artifacts")
    options = parser.parse_args()
    study = run_study(quick=options.quick)
    directory = Path(__file__).resolve().parent
    figs = plot_figures(study, directory / "figures", quick=options.quick)
    tables = write_results(study, directory / "results", quick=options.quick)
    print(f"Generated {len(figs)} figures and {len(tables)} tables.")
    for key, value in study.summary.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
