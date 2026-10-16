"""Tests of the robustness analysis of amendment 2: summaries from joint draws, pairwise
survival across fits, rank envelopes, focus-pair verdicts and the report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eu_okun.bayes_report import LoadedFit
from eu_okun.bayes_robustness import (
    FOCUS,
    SET_A,
    SET_B,
    FitSummary,
    credible_counts,
    focus_verdicts,
    rank_envelope,
    summarise_fit,
    surviving_pairs,
    write_robustness_report,
)
from eu_okun.contract import EU27

COUNTRIES = ["ES", "LU", "NL", "PT"]


def _summary(countries: list[str], accepted: bool = True) -> dict[str, Any]:
    return {
        "countries": countries,
        "diagnostics": {"accepted": accepted, "max_rhat": 1.004 if accepted else 1.02,
                        "min_ess_bulk": 800.0, "min_ess_tail": 700.0, "divergences": 0,
                        "min_bfmi": 0.8, "tree_depth_saturated": 0, "failures": []},
        "predictive_check": {"coverage_90": 0.9, "p_value": {
            "lag1_autocorrelation": 0.01, "cross_country_correlation": 0.4, "mean_sd": 0.5}},
        "classical_same_rows": {g: {"theta": -0.3, "ci_low": -0.5, "ci_high": -0.1}
                                for g in countries},
    }


def _fit(name: str, means: list[float], sd: float = 0.02, seed: int = 0,
         accepted: bool = True, countries: list[str] | None = None) -> LoadedFit:
    countries = countries or COUNTRIES
    rng = np.random.default_rng(seed)
    theta = np.asarray(means)[None, :] + rng.normal(0, sd, (2000, len(means)))
    b = rng.normal(0.05, 0.02, (2000, len(means), 2))
    return LoadedFit(name, _summary(countries, accepted),
                     {"theta": theta, "b": b, "beta0": theta - b.sum(axis=-1)})


def test_summary_uses_joint_draws_and_checks_the_coefficient_sum() -> None:
    fit = _fit("primary", [-0.5, -0.1, -0.15, -0.25])
    summary = summarise_fit(fit)
    assert summary.identity_error < 1e-12
    row = summary.rows[0]
    assert row.mean == pytest.approx(fit.draws["theta"][:, 0].mean())
    assert row.low < row.median < row.high and row.p_negative == 1.0
    assert row.rank_median == 1 and summary.rows[1].rank_median == 4
    np.testing.assert_allclose(summary.pairwise + summary.pairwise.T + np.eye(4), 1.0)
    broken = _fit("primary", [-0.5, -0.1, -0.15, -0.25])
    broken.draws["beta0"] = broken.draws["beta0"] + 0.01
    assert summarise_fit(broken).identity_error == pytest.approx(0.01)


def _summary_with(pairwise: np.ndarray, intervals: list[tuple[float, float]],
                  name: str = "x") -> FitSummary:
    fit = summarise_fit(_fit(name, [-0.5, -0.1, -0.15, -0.25]))
    fit.pairwise = pairwise
    fit.rows = [type(r)(**{**r.__dict__, "rank_low": lo, "rank_high": hi})
                for r, (lo, hi) in zip(fit.rows, intervals, strict=True)]
    return fit


def test_surviving_pairs_need_the_same_direction_in_every_fit() -> None:
    a = np.zeros((4, 4))
    a[0, 1], a[0, 2], a[3, 1] = 0.99, 0.97, 0.96
    b = a.copy()
    b[0, 2] = 0.90                                       # not credible in the second fit
    fits = [_summary_with(a, [(1, 2)] * 4), _summary_with(b, [(1, 3)] * 4)]
    counts = credible_counts(fits)
    assert counts[0, 1] == 2 and counts[0, 2] == 1
    survived = surviving_pairs(fits)
    assert survived[0, 1] and survived[3, 1] and not survived[0, 2]
    assert int(survived.sum()) == 2
    np.testing.assert_array_equal(rank_envelope(fits)[0], [1, 3])
    with pytest.raises(ValueError):
        surviving_pairs([])


def _classical_rows(pairs: dict[tuple[str, str], tuple[float, float]],
                    ) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    out: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for (a, b), (estimate, holm) in pairs.items():
        for sample in ("full", "pandemic_excluded"):
            out[(sample, a, b)] = [{"block_length": str(k), "estimate": str(estimate),
                                    "holm_p": str(holm if k == 8 else 0.5)}
                                   for k in (4, 8, 12)]
    return out


def test_focus_verdicts_cover_every_case() -> None:
    # Means in FOCUS order ES, LU, NL, PT; tight draws make every gap credible.
    tight = summarise_fit(_fit("primary", [-0.5, -0.1, -0.15, -0.25], sd=0.005))
    loose = summarise_fit(_fit("wide", [-0.5, -0.1, -0.15, -0.25], sd=0.2, seed=1))
    classical = _classical_rows({
        ("ES", "LU"): (-0.4, 0.01),     # apparent, Bayes agrees -> survives
        ("ES", "NL"): (+0.3, 0.01),     # apparent, opposite sign -> contradicted
        ("ES", "PT"): (-0.2, 0.20),     # not apparent, Bayes credible -> Bayesian only
        ("LU", "NL"): (+0.05, 0.01),    # apparent; not credible in the loose fit
        ("LU", "PT"): (+0.10, 0.30),
        ("NL", "PT"): (+0.10, 0.30),
    })
    verdicts = {(v.country_a, v.country_b): v
                for v in focus_verdicts([tight], [tight], classical, tight)}
    assert verdicts[("ES", "LU")].verdict == "survives"
    assert verdicts[("ES", "NL")].verdict == "contradicted (opposite direction)"
    assert verdicts[("ES", "PT")].verdict == "credible only in the Bayesian fits"
    mixed = {(v.country_a, v.country_b): v
             for v in focus_verdicts([tight, loose], [tight, loose], classical, tight)}
    assert mixed[("LU", "NL")].verdict == "does not survive"
    assert mixed[("NL", "PT")].verdict == "not supported by either"
    assert set(FOCUS) == {"ES", "LU", "NL", "PT"} and len(verdicts) == 6


def _write_fixture(directory: Path, name: str, means: np.ndarray, accepted: bool,
                   seed: int) -> None:
    fit = _fit(name, list(means), sd=0.05, seed=seed, accepted=accepted, countries=list(EU27))
    directory.mkdir(parents=True)
    draws = {k: v.reshape(2, 1000, *v.shape[1:]) for k, v in fit.draws.items()}
    np.savez_compressed(directory / "draws.npz", **draws)
    (directory / "fit.json").write_text(json.dumps(fit.summary), encoding="utf-8")


def _write_pairs(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample", "country_a", "country_b", "block_length", "n", "estimate",
                         "ci_low", "ci_high", "sign_p", "holm_p", "estimate_position"])
        for sample in ("full", "pandemic_excluded"):
            for i, a in enumerate(FOCUS):
                for b in FOCUS[i + 1:]:
                    for k in (4, 8, 12):
                        writer.writerow([sample, a, b, k, 40, -0.1, -0.3, 0.1, 0.02, 0.04, 0.5])


def test_report_renders_with_a_failed_fit_and_a_replication(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    means = np.sort(rng.normal(-0.25, 0.12, len(EU27)))
    for k, name in enumerate(SET_B):
        _write_fixture(tmp_path / "fits" / name, name, means + rng.normal(0, 0.01, len(EU27)),
                       accepted=name != "no_pooling", seed=k)
    _write_pairs(tmp_path / "pairs.csv")
    review = write_robustness_report(tmp_path / "fits", tmp_path / "out",
                                     pairs_csv=tmp_path / "pairs.csv", figures=True)
    assert "no_pooling" not in review.set_a and len(review.set_a) == len(SET_A) - 1
    assert review.surviving_b.sum() <= review.surviving_a.sum()
    report = (tmp_path / "out" / "BAYES_ROBUSTNESS.md").read_text(encoding="utf-8")
    for heading in ("## 1. Specifications, convergence and predictive checks",
                    "## 3. Which country differences survive", "## 4. Rank uncertainty",
                    "## 5. Portugal, Spain, the Netherlands and Luxembourg",
                    "## 6. Limitations"):
        assert heading in report
    assert "**failed**" in report
    for name in ("country_summaries.csv", "fits.csv", "surviving_pairs.csv", "focus_pairs.csv",
                 "pairwise_probabilities.csv", "rank_envelope.json", "diagnostics.png",
                 "estimates.png", "pairwise_robustness.png", "rank_intervals.png",
                 "focus_pairs.png"):
        assert (tmp_path / "out" / name).exists()
    again = write_robustness_report(tmp_path / "fits", tmp_path / "out2",
                                    pairs_csv=tmp_path / "pairs.csv", figures=False,
                                    replication_dir=tmp_path / "out")
    assert again.replication is not None
    text = (tmp_path / "out2" / "BAYES_ROBUSTNESS.md").read_text(encoding="utf-8")
    assert "## 6. Replication on a second platform" in text and "## 8. Reproduction" in text
    assert "0 only here" in text and "focus verdicts identical" in text
    with pytest.raises(FileNotFoundError, match="Set A is incomplete"):
        import shutil
        shutil.rmtree(tmp_path / "fits" / "lags_1")
        write_robustness_report(tmp_path / "fits", tmp_path / "out3",
                                pairs_csv=tmp_path / "pairs.csv", figures=False)
