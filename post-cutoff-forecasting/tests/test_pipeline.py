"""End to end on the synthetic series: run, summarise, draw, check."""

import json

import numpy as np
import pandas as pd
import pytest

from postcutoff import foundation, harness
from postcutoff.check import run_checks
from postcutoff.figures import make_all
from postcutoff.summarize import summarize
from postcutoff.synth import write_synthetic


class FakeFoundation:
    """Last week plus a small, deterministic perturbation; stands in for a downloaded model."""

    def __init__(self, spec, offset):
        self.spec = spec
        self.offset = offset
        self.seen_covariates = None

    def provenance(self):
        return {"package": "fake", "repo": "fake/fake", "revision": "0" * 40}

    def forecast(self, contexts, horizon, quantiles, past_covariates=None, future_covariates=None):
        if self.spec.covariates:
            assert past_covariates["holiday"].shape == contexts.shape
            assert future_covariates["holiday"].shape == (contexts.shape[0], horizon)
        med = contexts[:, -168:][:, :horizon] * (1 + self.offset)
        spread = 0.04 * med
        return np.stack([med + (q - 0.5) * 2 * spread for q in quantiles], axis=2)


@pytest.fixture
def project(fast_cfg, monkeypatch):
    write_synthetic(fast_cfg, end="2026-10-08")
    offsets = {"chronos2": 0.01, "chronos2_holidays": 0.005, "timesfm25": -0.01, "tirex": 0.02}
    monkeypatch.setattr(foundation, "load", lambda spec, c, h: FakeFoundation(spec, offsets[spec.name]))
    return fast_cfg


MODELS = [
    "seasonal_naive_week",
    "seasonal_naive_day",
    "mstl_ets",
    "chronos2",
    "chronos2_holidays",
    "timesfm25",
    "tirex",
]


def test_end_to_end(project):
    cfg = project
    scores, manifest = harness.run(cfg, MODELS, limit_per_window=5, log=lambda *_: None)
    assert set(scores["model"]) == set(MODELS)
    assert len(scores) == len(MODELS) * 4 * 5 * 7
    assert manifest["models"]["tirex"]["revision"] == "0" * 40

    out = summarize(cfg)
    rel = pd.DataFrame(out["relative_mae"]["week-ahead"]["vs_reference"])
    assert rel.loc[rel["model"] == "seasonal_naive_week", "ratio"].eq(1.0).all()
    assert np.isfinite(rel["ratio"]).all() and (rel.loc[rel["model"] == "tirex", "ratio"] != 1.0).all()
    assert (cfg.results_dir / "RESULTS.md").read_text().startswith("# Results")

    paths = make_all(cfg)
    assert len(paths) >= 4
    assert run_checks(cfg) == []

    # Reusing stored forecasts gives identical scores.
    again, _ = harness.run(cfg, MODELS, limit_per_window=5, log=lambda *_: None)
    pd.testing.assert_frame_equal(scores.reset_index(drop=True), again.reset_index(drop=True))
    reused = json.loads((cfg.results_dir / "run_manifest.json").read_text())["models"]
    assert all(v.get("reused") for v in reused.values())


def test_check_catches_a_tampered_score(project):
    cfg = project
    harness.run(cfg, ["seasonal_naive_week", "chronos2"], limit_per_window=3, log=lambda *_: None)
    summarize(cfg)
    path = cfg.results_dir / "scores.csv"
    s = pd.read_csv(path)
    i = s.index[s["model"] == "seasonal_naive_week"][0]
    s.loc[i, "sum_abs_err"] *= 1.5
    s.to_csv(path, index=False)
    problems = run_checks(cfg, require_figures=False)
    assert any("seasonal naive" in p for p in problems)
    assert any("does not match scores.csv" in p for p in problems)


def test_check_catches_forecasts_shifted_by_an_hour(project, monkeypatch):
    cfg = project

    class Shifted(FakeFoundation):
        def forecast(self, contexts, horizon, quantiles, **_):
            f = super().forecast(contexts, horizon, quantiles)
            return np.concatenate([f[:, :1], f[:, :-1]], axis=1)  # one hour late

    monkeypatch.setattr(foundation, "load", lambda spec, c, h: Shifted(spec, 0.0))
    harness.run(cfg, ["seasonal_naive_week", "tirex"], limit_per_window=8, log=lambda *_: None)
    summarize(cfg)
    problems = run_checks(cfg, require_figures=False)
    assert problems == ["tirex: forecasts are closest to the target shifted by -1 h"]


def test_run_refuses_weights_changed_inside_the_unseen_window(project, monkeypatch):
    cfg = project

    class Late(FakeFoundation):
        def provenance(self):
            return {**super().provenance(), "weights_last_commit": "2026-01-15T00:00:00+00:00"}

    monkeypatch.setattr(foundation, "load", lambda spec, c, h: Late(spec, 0.0))
    with pytest.raises(ValueError, match="weights last changed"):
        harness.run(cfg, ["seasonal_naive_week", "tirex"], limit_per_window=2, log=lambda *_: None)


def test_summary_has_placebo_and_block_sensitivity(project):
    cfg = project
    harness.run(cfg, ["seasonal_naive_week", "mstl_ets", "chronos2"], limit_per_window=4, log=lambda *_: None)
    out = summarize(cfg)
    for horizon in ("day-ahead", "week-ahead"):
        keys = set(out["change"][horizon])
        assert {"mae_vs_reference", "placebo_mae_vs_reference", "mae_vs_mstl_ets", "wql_vs_reference"} <= keys
        blocks = {r["block_days"] for r in out["block_sensitivity"][horizon]}
        assert blocks == {7, 14, 28, 56}
    placebo = out["change"]["week-ahead"]["placebo_mae_vs_reference"][0]
    assert placebo["before_windows"] == ["2022-23"] and placebo["after_windows"] == ["2023-24"]


def test_smoke_reports_implausible_models(project, monkeypatch):
    cfg = project

    class Broken(FakeFoundation):
        def forecast(self, contexts, horizon, quantiles, **_):
            return np.zeros((contexts.shape[0], horizon, len(quantiles))) + 1.0

    monkeypatch.setattr(foundation, "load", lambda spec, c, h: Broken(spec, 0))
    problems = harness.smoke(cfg, ["seasonal_naive_week", "tirex"], n=2, log=lambda *_: None)
    assert problems == ["tirex: median forecast implausible relative to the last week's level"]
