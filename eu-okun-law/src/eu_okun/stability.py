"""Formal (descriptive) pre-/post-pandemic Okun coefficient-stability tests.

A fully interacted OLS model tests coefficient changes with calendar-aware
Newey-West covariance. Pandemic quarters are not regression outcomes, inputs,
or lag history. Asymptotic p-values are conditional, not causal evidence.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from eu_okun.baseline import fit_hac_covariance
from eu_okun.comparison import common_sample_designs
from eu_okun.contract import EU27, EurostatDataError, quarter_sequence

Panel = dict[tuple[str, str], dict[str, float | None]]
Z_975 = 1.959963984540054


@dataclass(frozen=True)
class StabilityResult:
    """One country/participation specification and its HAC Wald statistics."""

    geo: str
    controls_participation: bool
    pre_n: int
    post_n: int
    pre_gdp_sum: float
    post_gdp_sum: float
    change_post_minus_pre: float
    change_hac_se: float
    ci95_low: float
    ci95_high: float
    p_change: float
    p_change_holm: float
    joint_wald: float
    joint_df: int
    joint_p: float


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down multiplicity correction, valid under arbitrary dependence."""
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues):
        raise ValueError("Every p-value must be finite and between zero and one.")
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(sorted(range(len(pvalues)), key=lambda j: pvalues[j])):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def chi_square_tail(value: float, df: int) -> float:
    """Closed-form chi-square survival for the prespecified df=3 or df=4.

    This avoids introducing SciPy for two fixed dimension counts. The
    distributional approximation assumes conventional HAC Wald regularity.
    """
    if value < 0 or not math.isfinite(value) or df not in (3, 4):
        raise ValueError("Expected finite nonnegative chi-square with 3 or 4 df.")
    half = value / 2.0
    if df == 3:
        return math.erfc(math.sqrt(half)) + math.sqrt(2 * value / math.pi) * math.exp(
            -half
        )
    return math.exp(-half) * (1.0 + half)


def _eligible_periods(
    panel: Panel,
    end: str,
    countries: Sequence[str],
) -> tuple[
    dict[tuple[str, int, bool], tuple[np.ndarray, np.ndarray]],
    list[str],
    dict[tuple[str, int, bool], tuple[np.ndarray, np.ndarray]],
    list[str],
]:
    """Build common country-quarter samples with disjoint input histories."""
    quarter_sequence("2022-Q1", end)
    pre_periods = quarter_sequence("2009-Q1", "2019-Q4")
    post_periods = quarter_sequence("2022-Q1", end)
    pre = {key: row for key, row in panel.items() if key[1] <= "2019-Q4"}
    post = {key: row for key, row in panel.items() if key[1] >= "2022-Q1"}
    pre_design, pre_dates = common_sample_designs(
        pre, pre_periods, countries, (1,), min_quarters=30
    )
    post_design, post_dates = common_sample_designs(
        post, post_periods, countries, (1,), min_quarters=12
    )
    return pre_design, pre_dates, post_design, post_dates


def fit_stability(
    panel: Panel,
    end: str,
    countries: Sequence[str] = EU27,
    hac_bandwidth: int = 4,
) -> tuple[list[StabilityResult], dict[str, object]]:
    """Test GDP-sum change and a joint Wald of all post interactions.

    The model is [X, post*X], with X=[1,g_t,g_{t-1},optional delta p_t].
    Thus the post-minus-pre GDP coefficient-sum difference is exactly
    the sum of the two GDP interaction coefficients. All countries use
    the same pre and post quarters within each subperiod.
    """
    pre, pre_dates, post, post_dates = _eligible_periods(panel, end, countries)
    rows: list[StabilityResult] = []
    periods = pre_dates + post_dates
    for geo in countries:
        for control in (False, True):
            x_pre, y_pre = pre[(geo, 1, control)]
            x_post, y_post = post[(geo, 1, control)]
            width = x_pre.shape[1]
            zeros = np.zeros_like(x_pre)
            x_pooled = np.vstack((
                np.hstack((x_pre, zeros)),
                np.hstack((x_post, x_post)),
            ))
            y_pooled = np.concatenate((y_pre, y_post))
            try:
                beta, covariance, _, _ = fit_hac_covariance(
                    x_pooled, y_pooled, maxlags=hac_bandwidth, periods=periods
                )
                change = float(beta[width + 1] + beta[width + 2])
                contrast = np.zeros(2 * width, dtype=float)
                contrast[width + 1 : width + 3] = 1.0
                variance = float(contrast @ covariance @ contrast)
                if variance <= 0 or not math.isfinite(variance):
                    raise ValueError("Nonpositive HAC change variance.")
                standard_error = math.sqrt(variance)
                p_change = math.erfc(abs(change / standard_error) / math.sqrt(2))
                interaction = beta[width:]
                interaction_cov = covariance[width:, width:]
                if np.linalg.cond(interaction_cov) > 1e12:
                    raise ValueError("Singular/unstable joint Wald covariance.")
                wald = float(
                    interaction @ np.linalg.solve(interaction_cov, interaction)
                )
                joint_p = chi_square_tail(max(0.0, wald), width)
            except (ValueError, np.linalg.LinAlgError) as exc:
                raise EurostatDataError(
                    f"Stability estimation failed for {geo}, participation={control}: {exc}"
                ) from exc
            pre_sum = float(beta[1] + beta[2])
            rows.append(
                StabilityResult(
                    geo=geo,
                    controls_participation=control,
                    pre_n=len(pre_dates),
                    post_n=len(post_dates),
                    pre_gdp_sum=pre_sum,
                    post_gdp_sum=pre_sum + change,
                    change_post_minus_pre=change,
                    change_hac_se=standard_error,
                    ci95_low=change - Z_975 * standard_error,
                    ci95_high=change + Z_975 * standard_error,
                    p_change=p_change,
                    p_change_holm=1.0,
                    joint_wald=wald,
                    joint_df=width,
                    joint_p=joint_p,
                )
            )
    if not rows:
        raise EurostatDataError("No estimable country stability tests.")
    adjusted = holm_adjust([r.p_change for r in rows])
    rows = [replace(r, p_change_holm=p) for r, p in zip(rows, adjusted, strict=True)]
    return rows, {
        "pre_periods": pre_dates,
        "post_periods": post_dates,
        "hac_quarter_bandwidth": hac_bandwidth,
        "gdp_lags": 1,
        "multiplicity_family_size": len(rows),
        "null_sum": "Sum of post-minus-pre GDP coefficients equals zero",
        "null_joint": "All post-period interactions equal zero",
    }

def classical_weaker(review_dir: Path) -> dict[str, bool]:
    """The classical review's weaker-after-2021 rule, recomputed from its tables.

    A country is in the set when every leave-one-out post-2021 sum lies above the upper
    bound of its 2009-2019 95% interval (K = 1, without participation).
    """
    upper: dict[str, float] = {}
    with (review_dir / "grid.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (row["sample"], row["gdp_lags"], row["participation"]) == ("pre_2020", "1",
                                                                          "False"):
                upper[row["geo"]] = float(row["ci_high"])
    out: dict[str, bool] = {}
    with (review_dir / "stability_review.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["participation"] == "False":
                out[row["geo"]] = float(row["post_loo_min"]) > upper[row["geo"]]
    return out


def write_stability(directory: Path, output_dir: Path) -> int:
    """Audit original Eurostat data and write formal HAC stability statistics."""
    import hashlib
    import json
    from dataclasses import asdict

    from eu_okun.audit import audit_snapshot
    from eu_okun.baseline import _read_panel

    audit = audit_snapshot(directory)
    final = audit["analysis_period"]["end"]
    if not isinstance(final, str):
        raise EurostatDataError("Invalid final quarter in audit.")
    rows, metadata = fit_stability(
        _read_panel(directory / "processed" / "eu27_quarterly.csv"), final
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "Fully interacted pre/post OLS with calendar-aware HAC Wald tests",
        "caveat": "Descriptive coefficient stability, not causal identification",
        "snapshot_manifest_sha256": hashlib.sha256(
            (directory / "snapshot_manifest.json").read_bytes()
        ).hexdigest(),
        "design": metadata,
        "results": [asdict(row) for row in rows],
    }
    (output_dir / "stability.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return len(rows)
