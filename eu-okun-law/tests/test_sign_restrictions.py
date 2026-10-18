"""Tests of the sign-restricted SVAR: rotations, MA coefficients, the panel VAR and its
posterior, admissible assignments and identified sets on models with a known structural
matrix, the unbounded-ratio rule, recursive orderings, the robust Bayes summaries, the
deflator snapshot and the report. Synthetic data; no network."""

from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eu_okun.contract import EU27, EurostatDataError, quarter_sequence
from eu_okun.sign_restrictions import (
    DU,
    PI,
    PRIMARY,
    PRIMARY_RESTRICTIONS,
    SHOCKS,
    Draw,
    DrawSets,
    VarData,
    admissible,
    audit_deflator,
    fetch_deflator,
    fit_var,
    haar_rotations,
    identified_sets,
    ma_coefficients,
    posterior_draws,
    recursive_okun,
    summarise,
    target_names,
    var_rows,
)

ROOT = Path(__file__).resolve().parents[1]

# Impact matrix with columns (demand, technology, labour supply, other) that satisfy S1-S3.
B_TRUE = np.array([
    [0.8, 0.6, 0.1, 0.2],     # g
    [0.5, -0.4, -0.3, 0.1],   # pi
    [-0.3, -0.2, 0.25, 0.05],  # du
    [0.05, 0.1, 0.3, -0.1],   # dp
])
A1 = np.array([[0.4, 0.05, -0.1, 0.0], [0.1, 0.5, 0.0, 0.0], [-0.1, 0.0, 0.3, 0.05],
               [0.0, 0.0, 0.05, 0.2]])
A2 = np.array([[0.1, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.1, 0.0],
               [0.0, 0.0, 0.0, 0.05]])


def _a(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    """Stack Phi_1, Phi_2 into the (4 lags, 4) coefficient matrix x_t' = x_{t-1}' A_1 + ..."""
    return np.vstack([phi1.T, phi2.T])


def test_rotations_are_orthonormal_and_ma_recursion_matches_the_companion() -> None:
    q = haar_rotations(500, np.random.default_rng(0))
    assert np.allclose(np.einsum("mij,mkj->mik", q, q), np.eye(4), atol=1e-12)
    assert abs(float(q.mean())) < 0.02
    a = _a(A1, A2)
    psi = ma_coefficients(a, 2, 6)
    comp = np.vstack([np.hstack([A1, A2]), np.hstack([np.eye(4), np.zeros((4, 4))])])
    for h in range(7):
        assert np.allclose(psi[h], np.linalg.matrix_power(comp, h)[:4, :4])


def simulate_panel(countries: tuple[str, ...] = EU27[:12], quarters: int = 120,
                   seed: int = 1) -> VarData:
    rng = np.random.default_rng(seed)
    qs = quarter_sequence("1990-Q1", "2030-Q4")[:quarters]
    x = np.zeros((len(countries), quarters, 4))
    for i in range(len(countries)):
        mu = rng.normal(0, 0.3, 4)
        for t in range(2, quarters):
            x[i, t] = mu + A1 @ x[i, t - 1] + A2 @ x[i, t - 2] + B_TRUE @ rng.normal(size=4)
    return VarData(countries, tuple(qs), x)


def test_panel_var_and_posterior_recover_the_reduced_form() -> None:
    data = simulate_panel()
    y, x, groups = var_rows(data, replace(PRIMARY, exclude_pandemic=False))
    rf = fit_var(y, x, groups, 2)
    # Coefficients on low-variance lags (dp) are imprecise: compare in standard errors.
    se = np.sqrt(np.outer(np.diag(rf.xtx_inv), np.diag(rf.sigma_hat)))
    assert np.max(np.abs(rf.a_hat - _a(A1, A2)) / se) < 4
    assert np.allclose(rf.sigma_hat, B_TRUE @ B_TRUE.T, atol=0.05)
    draws, discarded = posterior_draws(rf, 300, np.random.default_rng(2))
    assert discarded == 0 and len(draws) == 300
    mean_sigma = np.mean([d.sigma for d in draws], axis=0)
    assert np.allclose(mean_sigma, rf.s / (rf.dof - 5), rtol=0.05, atol=0.01)
    assert np.allclose(np.mean([d.a for d in draws], axis=0), rf.a_hat, atol=0.01)


def _true_rotation(sigma: np.ndarray) -> np.ndarray:
    return np.asarray(np.linalg.inv(np.linalg.cholesky(sigma)) @ B_TRUE)


def test_admissible_finds_the_true_rotation_and_its_permutations() -> None:
    sigma = B_TRUE @ B_TRUE.T
    q0 = _true_rotation(sigma)
    assert np.allclose(q0 @ q0.T, np.eye(4), atol=1e-12)
    # The truth with columns shuffled and one sign flipped is the same structural model.
    shuffled = q0[:, [3, 2, 0, 1]] * np.array([1, -1, 1, 1])
    impact = np.einsum("ij,mjk->mik", np.linalg.cholesky(sigma),
                       np.stack([q0, shuffled, np.eye(4)]))
    found = admissible([impact], PRIMARY_RESTRICTIONS)
    rows = {int(i) for idx, _, _ in found for i in idx}
    assert {0, 1} <= rows
    for idx, columns, signs in found:
        for k, rot in enumerate(idx):
            cols = impact[rot][:, columns] * signs[k]
            for s, shock in enumerate(SHOCKS):
                for var, sign in PRIMARY_RESTRICTIONS[shock].items():
                    assert sign * cols[var, s] > 0


def test_identified_sets_contain_the_truth() -> None:
    sigma = B_TRUE @ B_TRUE.T
    draw = Draw(_a(A1, A2), sigma)
    rotations = np.concatenate([_true_rotation(sigma)[None],
                                haar_rotations(20000, np.random.default_rng(3))])
    sets = identified_sets(draw, PRIMARY, rotations)
    assert sets.nonempty and sets.admissible_count > 1
    psi = ma_coefficients(draw.a, 2, 8)
    cum = np.cumsum(psi @ B_TRUE, axis=0)
    for s, shock in enumerate(SHOCKS):
        for h in (0, 4, 8):
            assert sets.lower[f"{shock}:unemployment:{h}"] - 1e-12 <= cum[h, DU, s] \
                <= sets.upper[f"{shock}:unemployment:{h}"] + 1e-12
        assert sets.single[f"{shock}:gdp:0"] == pytest.approx(cum[0, 0, s])   # truth first
    total = sum(np.sum((psi[:9] @ B_TRUE)[:, DU, k] ** 2) for k in range(4))
    shares = [sets.single[f"{shock}:fevd_du:8"] for shock in SHOCKS]
    assert 0 < sum(shares) <= 1 + 1e-12
    assert shares[0] == pytest.approx(np.sum((psi[:9] @ B_TRUE)[:, DU, 0] ** 2) / total)


def test_ratio_sets_are_unbounded_when_the_gdp_response_can_change_sign() -> None:
    sigma = B_TRUE @ B_TRUE.T
    # Strongly mean-reverting GDP: the cumulative GDP response at h = 4 can be of either sign.
    phi1 = A1.copy()
    phi1[0, 0] = -0.9
    draw = Draw(_a(phi1, np.zeros((4, 4))), sigma)
    spec = replace(PRIMARY, lags=2)
    sets = identified_sets(draw, spec, haar_rotations(20000, np.random.default_rng(4)))
    gdp_lo, gdp_hi = sets.lower["demand:gdp:4"], sets.upper["demand:gdp:4"]
    if gdp_lo < 0 < gdp_hi:
        assert sets.lower["demand:okun:4"] == -math.inf and sets.upper["demand:okun:4"] == math.inf
    else:
        assert math.isfinite(sets.lower["demand:okun:4"])


def test_empty_identified_set_when_restrictions_contradict_the_covariance() -> None:
    # pi and du perfectly co-move with g in the same direction for every shock: no rotation
    # can give demand (g+, pi+, du-) since du = g always.
    b = np.array([[1.0, 0, 0, 0], [0.5, 0.2, 0, 0], [1.0, 0, 1e-6, 0], [0.0, 0, 0, 1.0]])
    draw = Draw(np.zeros((8, 4)), b @ b.T + 1e-10 * np.eye(4))
    sets = identified_sets(draw, PRIMARY, haar_rotations(5000, np.random.default_rng(5)))
    assert not sets.nonempty


def test_recursive_orderings_with_g_first_agree() -> None:
    sigma = B_TRUE @ B_TRUE.T
    draw = Draw(_a(A1, A2), sigma)
    first = [recursive_okun(draw, 2, (0, *rest)) for rest in
             ((1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1))]
    for value in first[1:]:
        assert value[4] == pytest.approx(first[0][4])
    psi = ma_coefficients(draw.a, 2, 4)
    column = sigma[:, 0] / math.sqrt(sigma[0, 0])
    cum = np.cumsum(psi @ column, axis=0)
    assert first[0][4] == pytest.approx(cum[4, DU] / cum[4, 0])
    later = recursive_okun(draw, 2, (DU, PI, 0, 3))
    assert later[4] != pytest.approx(first[0][4])


def test_summaries_propagate_infinite_bounds() -> None:
    names = target_names()

    def drawset(lo: float, hi: float) -> DrawSets:
        return DrawSets(True, 10, {n: lo for n in names}, {n: hi for n in names},
                        {n: (lo + hi) / 2 if math.isfinite(lo) else 0.0 for n in names})

    finite = [drawset(-1.0 - 0.01 * k, 1.0 + 0.01 * k) for k in range(100)]
    s = summarise(finite)[0]
    assert s.bounded and s.region_lower <= s.mean_lower < s.mean_upper <= s.region_upper
    some_inf = finite[:94] + [drawset(-math.inf, math.inf)] * 6
    assert not summarise(some_inf)[0].bounded
    few_inf = finite[:97] + [drawset(-math.inf, math.inf)] * 3
    assert summarise(few_inf)[0].bounded


# ----------------------------------------------------------------------------------
# Deflator snapshot, end-to-end estimate and report
# ----------------------------------------------------------------------------------


def _deflator_payload() -> dict[str, Any]:
    quarters = quarter_sequence("2008-Q1", "2008-Q4")
    return {"class": "dataset", "version": "2.0", "updated": "2026-10-16",
            "id": ["freq", "unit", "s_adj", "na_item", "geo", "time"],
            "size": [1, 1, 1, 1, 27, 4],
            "dimension": {
                "freq": {"category": {"index": {"Q": 0}}},
                "unit": {"category": {"index": {"PD_PCH_PRE_NAC": 0}}},
                "s_adj": {"category": {"index": {"SCA": 0}}},
                "na_item": {"category": {"index": {"B1GQ": 0}}},
                "geo": {"category": {"index": {g: i for i, g in enumerate(EU27)}}},
                "time": {"category": {"index": {q: j for j, q in enumerate(quarters)}}}},
            "value": {str(i * 4 + j): 0.5 + 0.01 * i for i in range(27) for j in range(4)},
            "status": {"0": "p"}}


def test_deflator_snapshot_is_frozen_and_audited(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    import eu_okun.sign_restrictions as sr

    payload = _deflator_payload()
    monkeypatch.setattr(sr, "fetch_dataset",
                        lambda spec, start, end: (json.dumps(payload).encode(), payload, "url"))
    manifest = fetch_deflator(tmp_path / "d", "2008-Q1", "2008-Q4")
    assert manifest["dataset"] == "namq_10_gdp"
    audit_deflator(tmp_path / "d")
    csv_path = tmp_path / "d" / "processed" / "eu27_deflator.csv"
    assert csv_path.read_text(encoding="utf-8").splitlines()[1] == "AT,2008-Q1,0.5,p"
    csv_path.write_text(csv_path.read_text(encoding="utf-8").replace("0.5,", "0.6,", 1),
                        encoding="utf-8", newline="\n")
    with pytest.raises(EurostatDataError, match="manifest hash"):
        audit_deflator(tmp_path / "d")


def test_estimate_and_report_end_to_end(tmp_path: Path) -> None:
    from eu_okun.sign_restrictions_report import render, review_from_data, write_tables

    data = simulate_panel(countries=EU27, quarters=80, seed=6)
    review = review_from_data(data, draws=40, rotations=2000, check_draws=5)
    primary = review.results["primary"]
    assert primary.nonempty_share > 0.9 and primary.draws == 40
    assert set(review.results) == {"primary", "tech_no_du", "ls_output_up", "h01", "lags4",
                                   "no_ie", "pandemic"}
    assert len(review.orderings) == 24
    write_tables(review, tmp_path)
    report = render(review, ".", "eu-okun sign-restrictions")
    for text in ("## Summary", "## 1. Design", "## 2. Plausibility", "## 3. Demand against "
                 "technology", "## 4. Labour supply against demand", "## 5. Robustness",
                 "## 6. Recursive orderings", "orthogonalised, not causal"):
        assert text in report, text
    for name in ("summaries.csv", "specifications.csv", "orderings.csv", "provenance.json"):
        assert (tmp_path / name).exists()


@pytest.mark.skipif(os.environ.get("EU_OKUN_REPRODUCE_SVAR") != "1",
                    reason="Full SVAR reproduction runs in the estimates workflow")
def test_committed_sign_restriction_results_reproduce(tmp_path: Path) -> None:
    """The committed tables regenerate from the frozen snapshots (validation gate)."""
    import csv

    from eu_okun.sign_restrictions_report import write_report

    committed = ROOT / "docs" / "sign-restrictions"
    if not (committed / "summaries.csv").exists():
        pytest.skip("No committed sign-restriction results.")
    write_report(ROOT / "snapshots" / "eurostat-2026-10-15",
                 sorted((ROOT / "snapshots").glob("eurostat-deflator-*"))[-1], tmp_path,
                 figures=False)
    for name in ("summaries.csv", "specifications.csv", "orderings.csv"):
        with (committed / name).open(encoding="utf-8") as a, (tmp_path / name).open(
                encoding="utf-8") as b:
            old, new = list(csv.DictReader(a)), list(csv.DictReader(b))
        assert len(old) == len(new), name
        for x, y in zip(old, new, strict=True):
            for key, value in x.items():
                try:
                    assert float(value) == pytest.approx(float(y[key]), rel=1e-6, abs=1e-9)
                except ValueError:
                    assert value == y[key], (name, key)
