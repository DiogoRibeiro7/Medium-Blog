"""One-quarter-ahead forecasts: hierarchical Bayesian against classical models.

Implements the evaluation of ``BAYES_FORECAST_PROTOCOL.md`` inside the PR #149
forecasting framework (``forecasting.py``): the same information set (nothing dated t),
the same complete-case quarters, origins, training rows and targets for every model.
The classical point forecasts are exactly the framework's; this module adds their OLS
predictive distributions, the Bayesian predictive distributions from saved draws, proper
scores, a synchronised circular block bootstrap for model differences, country
heterogeneity, and the report.

The forecasts are retrospective: revised Eurostat data, no historical vintages.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun.contract import EU27, EurostatDataError, quarter_sequence
from eu_okun.forecasting import (
    MIN_TRAIN,
    MODEL_FEATURES,
    TEST_START,
    TRAIN_START,
    complete_panel_examples,
)
from eu_okun.predictive import (
    Prediction,
    PredictiveComponents,
    StudentT,
    crps_samples,
    loss_value,
    ols_predictive,
    predict_student_t,
    predictive_log_density,
    predictive_samples,
    read_predictions,
    write_predictions,
)
from eu_okun.review import Panel
from eu_okun.stability import holm_adjust

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
FEATURES = ("du_lag1", "dp_lag1", "gdp_lag1", "gdp_lag2")
CLASSICAL_MODELS: tuple[str, ...] = tuple(MODEL_FEATURES)
BAYES_MODELS: tuple[str, ...] = ("bayes_ar1", "bayes_gdp_ar1", "bayes_gdp_ar1_factor")
MODELS: tuple[str, ...] = (*BAYES_MODELS, *CLASSICAL_MODELS)
PERSISTENCE, AR1 = "zero_change", "unemployment_ar1"
FAMILY: tuple[tuple[str, str], ...] = (
    ("bayes_gdp_ar1", "gdp_unemployment_ar1"),
    ("bayes_ar1", "unemployment_ar1"),
    ("bayes_gdp_ar1", "zero_change"),
    ("bayes_gdp_ar1", "bayes_ar1"),
    ("bayes_gdp_ar1_factor", "bayes_gdp_ar1"),
)
BLOCK = 4
BLOCK_SENSITIVITY = 8
REPLICATES = 999
SEED = 20261020
ALPHA = 0.05
PANDEMIC_TARGETS = ("2020-Q1", "2022-Q2")       # excluded in the sensitivity evaluation
LABELS: dict[str, str] = {
    "bayes_ar1": "Bayes AR(1)",
    "bayes_gdp_ar1": "Bayes AR(1) + GDP lags",
    "bayes_gdp_ar1_factor": "Bayes AR(1) + GDP lags + factor",
    "zero_change": "Persistence (zero change)",
    "historical_mean": "Historical mean",
    "unemployment_ar1": "AR(1)",
    "gdp_lags": "GDP lags",
    "gdp_unemployment_ar1": "AR(1) + GDP lags",
    "gdp_ar1_participation": "AR(1) + GDP lags + participation",
}


# ----------------------------------------------------------------------------------
# Data: the framework's complete-case panel
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastPanel:
    """Common complete-case quarters of the framework and the forecast origins.

    ``y[i, t]`` is the change in quarter t; ``features[name][i, t]`` is a predictor
    dated t-1 or earlier. ``origins`` are column indices of the target quarters.
    """

    countries: tuple[str, ...]
    quarters: tuple[str, ...]
    y: np.ndarray
    features: dict[str, np.ndarray]
    origins: tuple[int, ...]

    @property
    def target_quarters(self) -> list[str]:
        return [self.quarters[j] for j in self.origins]

    def design(self, names: Sequence[str]) -> np.ndarray:
        """Predictors stacked as (N, T, K)."""
        if not names:
            return np.zeros((*self.y.shape, 0))
        return np.stack([self.features[name] for name in names], axis=-1)


def build_forecast_panel(panel: Panel, end: str, countries: Sequence[str] = EU27,
                         ) -> ForecastPanel:
    """The framework's examples arranged as arrays, with its origins (protocol, 2)."""
    examples, common = complete_panel_examples(panel, quarter_sequence(TRAIN_START, end),
                                               countries)
    origins = tuple(i for i, q in enumerate(common) if q >= TEST_START and i >= MIN_TRAIN)
    if len(origins) < 12:
        raise EurostatDataError("Fewer than 12 common forecast origins.")
    y = np.asarray([[examples[g][q].actual_du for q in common] for g in countries])
    features = {name: np.asarray([[getattr(examples[g][q], name) for q in common]
                                  for g in countries]) for name in FEATURES}
    return ForecastPanel(tuple(countries), tuple(common), y, features, origins)


# ----------------------------------------------------------------------------------
# Classical predictive distributions (framework point forecasts)
# ----------------------------------------------------------------------------------


def classical_distribution(fpanel: ForecastPanel, model: str, i: int, column: int,
                           ) -> StudentT:
    """Predictive distribution of a framework model trained on columns before ``column``."""
    y = fpanel.y[i, :column]
    if model == "zero_change":
        if y.size < 12:
            raise ValueError("Zero-change scale needs at least 12 training changes.")
        return StudentT(0.0, float(np.sqrt(np.mean(y**2))), float(y.size))
    names = MODEL_FEATURES[model]
    x = np.column_stack([np.ones(column), *(fpanel.features[n][i, :column] for n in names)])
    x_new = np.asarray([1.0, *(fpanel.features[n][i, column] for n in names)])
    return ols_predictive(x, y, x_new)


def classical_predictions(fpanel: ForecastPanel, model: str) -> list[Prediction]:
    """Every country and origin; training rows are all common quarters before the target."""
    out = []
    for column in fpanel.origins:
        quarter = fpanel.quarters[column]
        for i, geo in enumerate(fpanel.countries):
            dist = classical_distribution(fpanel, model, i, column)
            out.append(predict_student_t("forecast", model, geo, quarter,
                                         float(fpanel.y[i, column]), dist))
    return out


# ----------------------------------------------------------------------------------
# Bayesian predictive distributions from saved draws
# ----------------------------------------------------------------------------------


def bayes_components(draws: Mapping[str, np.ndarray], x_new: np.ndarray,
                     common_factor: bool) -> PredictiveComponents:
    """Location and scale per draw and country; the factor value is unknown (integrated)."""
    centre = draws["alpha"] + np.einsum("nk,snk->sn", x_new, draws["beta"])
    loading = draws["lam"] if common_factor else None
    return PredictiveComponents(np.asarray(centre), np.asarray(draws["sigma"]), loading,
                                np.asarray(draws["nu"]).reshape(-1))


def bayes_forecast_predictions(model: str, fpanel: ForecastPanel, column: int,
                               draws: Mapping[str, np.ndarray], features: Sequence[str],
                               common_factor: bool, replicates: int,
                               seed: int) -> list[Prediction]:
    """Posterior predictive summaries and scores for every country at one origin."""
    x_new = fpanel.design(features)[:, column]
    components = bayes_components(draws, x_new, common_factor)
    samples = predictive_samples(components, replicates, np.random.default_rng(seed))
    actual = fpanel.y[:, column]
    crps = crps_samples(samples, actual)
    log_score = predictive_log_density(components, actual)
    q = np.quantile(samples, QUANTILES, axis=0)
    mean = samples.mean(axis=0)
    pit = np.mean(samples <= actual[None, :], axis=0)
    quarter = fpanel.quarters[column]
    return [Prediction("forecast", model, geo, quarter, float(actual[i]), float(mean[i]),
                       float(q[2, i]), float(q[0, i]), float(q[1, i]), float(q[3, i]),
                       float(q[4, i]), float(crps[i]), float(log_score[i]), float(pit[i]))
            for i, geo in enumerate(fpanel.countries)]


# ----------------------------------------------------------------------------------
# Losses and the synchronised circular block bootstrap
# ----------------------------------------------------------------------------------


def loss_array(predictions: Sequence[Prediction], fpanel: ForecastPanel,
               loss: str) -> np.ndarray:
    """(N, origins) losses; raises unless every country and origin appears once."""
    countries = list(fpanel.countries)
    quarters = fpanel.target_quarters
    out = np.full((len(countries), len(quarters)), np.nan)
    for p in predictions:
        i, j = countries.index(p.geo), quarters.index(p.quarter)
        if np.isfinite(out[i, j]):
            raise ValueError(f"Duplicate forecast {p.model} {p.geo}/{p.quarter}.")
        out[i, j] = loss_value(p, loss)
    if not np.isfinite(out).all():
        raise ValueError("Forecasts do not cover every country and origin.")
    return out


def circular_block_indices(n: int, block: int, replicates: int,
                           rng: np.random.Generator) -> np.ndarray:
    """(replicates, n) indices: circular blocks of ``block`` consecutive quarters.

    The same construction as ``dependence.paired_block_comparisons``; every country of a
    resampled quarter is kept, which preserves cross-country dependence.
    """
    if block < 1 or n < 2 * block:
        raise ValueError("Series too short for the block length.")
    blocks = -(-n // block)
    starts = rng.integers(0, n, size=(replicates, blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return np.asarray(idx.reshape(replicates, -1)[:, :n])


@dataclass(frozen=True)
class BootstrapComparison:
    """Model A against model B over a set of target quarters."""

    model_a: str
    model_b: str
    loss: str
    targets: str
    block: int
    quarters: int
    mean_a: float
    mean_b: float
    ratio: float
    ratio_low: float
    ratio_high: float
    difference: float
    low: float
    high: float
    p_value: float
    p_holm: float = float("nan")
    verdict: str = ""


def bootstrap_comparison(loss_a: np.ndarray, loss_b: np.ndarray, indices: np.ndarray,
                         model_a: str, model_b: str, loss: str, targets: str,
                         block: int) -> BootstrapComparison:
    """Mean of the quarterly cross-country differences, its percentile interval, the
    ratio of mean losses and the null-centred bootstrap p-value."""
    a, b = loss_a.mean(axis=0), loss_b.mean(axis=0)
    d = a - b
    mean = float(d.mean())
    boot = d[indices].mean(axis=1)
    ratios = a[indices].mean(axis=1) / b[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    r_low, r_high = np.quantile(ratios, [0.025, 0.975])
    exceed = int(np.sum(np.abs(boot - mean) >= abs(mean)))
    p = (1 + exceed) / (indices.shape[0] + 1)
    return BootstrapComparison(model_a, model_b, loss, targets, block, d.size,
                               float(a.mean()), float(b.mean()), float(a.mean() / b.mean()),
                               float(r_low), float(r_high), mean, float(low), float(high), p)


def holm_verdicts(tests: Sequence[BootstrapComparison]) -> list[BootstrapComparison]:
    """Holm adjustment over the family and the pre-registered verdicts."""
    adjusted = holm_adjust([t.p_value for t in tests])
    out = []
    for t, p in zip(tests, adjusted, strict=True):
        verdict = (f"{t.model_a if t.difference < 0 else t.model_b} better" if p < ALPHA
                   else "not distinguishable")
        out.append(replace(t, p_holm=p, verdict=verdict))
    return out


# ----------------------------------------------------------------------------------
# The comparison
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastReplication:
    """Tables of an independent run of the same refits (e.g. on another platform)."""

    label: str
    family: dict[tuple[str, str], tuple[float, float, str]]       # ratio, Holm p, verdict
    failed: dict[str, list[str]]                                  # model -> target quarters
    bayes_crps: dict[tuple[str, str, str], float]                 # model, geo, quarter


def load_replication(directory: Path) -> ForecastReplication:
    """Read ``comparisons.csv``, ``refits.json`` and ``forecasts.csv`` of another run."""
    family = {}
    with (directory / "comparisons.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["kind"] == "family":
                family[(row["model_a"], row["model_b"])] = (
                    float(row["ratio"]), float(row["p_holm"]), row["verdict"])
    refits = json.loads((directory / "refits.json").read_text(encoding="utf-8"))["refits"]
    failed: dict[str, list[str]] = {m: [] for m in BAYES_MODELS}
    for record in refits:
        if not record["accepted"]:
            failed[record["model"]].append(record["target"])
    crps = {(p.model, p.geo, p.quarter): p.crps
            for p in read_predictions(directory / "forecasts.csv") if p.model in BAYES_MODELS}
    readme = directory / "README.md"
    label = (readme.read_text(encoding="utf-8").splitlines()[0].lstrip("# ")
             if readme.exists() else directory.name)
    return ForecastReplication(label, family, {k: sorted(v) for k, v in failed.items()}, crps)


@dataclass
class ForecastComparison:
    """Everything the report is generated from."""

    fpanel: ForecastPanel
    predictions: dict[str, list[Prediction]]
    refits: list[dict[str, Any]]
    family: list[BootstrapComparison]
    descriptive: list[BootstrapComparison]
    sensitivity: dict[str, list[BootstrapComparison]] = field(default_factory=dict)
    country: list[dict[str, Any]] = field(default_factory=list)
    snapshot: dict[str, str] = field(default_factory=dict)
    replication: ForecastReplication | None = None

    def losses(self, model: str, loss: str = "crps") -> np.ndarray:
        return loss_array(self.predictions[model], self.fpanel, loss)

    def keep(self, targets: str) -> list[int]:
        """Origin positions of a target set: 'all' or 'excluding 2020-Q1..2022-Q2'."""
        quarters = self.fpanel.target_quarters
        if targets == "all":
            return list(range(len(quarters)))
        first, last = PANDEMIC_TARGETS
        return [j for j, q in enumerate(quarters) if not first <= q <= last]

    def mean_loss(self, model: str, loss: str = "crps", targets: str = "all") -> float:
        values = self.losses(model, loss)[:, self.keep(targets)]
        return float(values.mean(axis=0).mean())

    def coverage(self, model: str, targets: str = "all") -> tuple[float, float]:
        quarters = {self.fpanel.target_quarters[j] for j in self.keep(targets)}
        rows = [p for p in self.predictions[model] if p.quarter in quarters]
        return (float(np.mean([p.q25 <= p.actual <= p.q75 for p in rows])),
                float(np.mean([p.q05 <= p.actual <= p.q95 for p in rows])))


def load_bayes(fits_dir: Path, fpanel: ForecastPanel, model: str,
               ) -> tuple[list[Prediction], list[dict[str, Any]]]:
    """Predictions and refit records of one Bayesian model; every origin is required."""
    rows: list[Prediction] = []
    records = []
    missing = []
    for quarter in fpanel.target_quarters:
        directory = fits_dir / model / quarter
        if not (directory / "origin.json").exists():
            missing.append(quarter)
            continue
        records.append(json.loads((directory / "origin.json").read_text(encoding="utf-8")))
        rows += read_predictions(directory / "predictions.csv")
    if missing:
        raise FileNotFoundError(f"{model}: no refit for {', '.join(missing)} in {fits_dir}.")
    return rows, records


def _comparisons(comparison: ForecastComparison, pairs: Sequence[tuple[str, str]],
                 loss: str, targets: str, block: int) -> list[BootstrapComparison]:
    keep = comparison.keep(targets)
    indices = circular_block_indices(len(keep), block, REPLICATES,
                                     np.random.default_rng(SEED))
    return [bootstrap_comparison(comparison.losses(a, loss)[:, keep],
                                 comparison.losses(b, loss)[:, keep], indices, a, b, loss,
                                 targets, block) for a, b in pairs]


def country_heterogeneity(comparison: ForecastComparison) -> list[dict[str, Any]]:
    """Per-country CRPS/MAE skill and a block-bootstrap interval for comparison 1."""
    indices = circular_block_indices(len(comparison.fpanel.origins), BLOCK, REPLICATES,
                                     np.random.default_rng(SEED))
    crps = {m: comparison.losses(m, "crps") for m in MODELS}
    mae = {m: comparison.losses(m, "absolute_error") for m in MODELS}
    a, b = FAMILY[0]
    rows = []
    for i, geo in enumerate(comparison.fpanel.countries):
        d = crps[a][i] - crps[b][i]
        boot = d[indices].mean(axis=1)
        low, high = np.quantile(boot, [0.025, 0.975])
        row: dict[str, Any] = {"geo": geo, "difference_1": float(d.mean()),
                               "difference_1_low": float(low), "difference_1_high": float(high)}
        for model in MODELS:
            row[f"crps_{model}"] = float(crps[model][i].mean())
            row[f"crps_skill_ar1_{model}"] = 1 - crps[model][i].mean() / crps[AR1][i].mean()
            row[f"crps_skill_persistence_{model}"] = (
                1 - crps[model][i].mean() / crps[PERSISTENCE][i].mean())
            row[f"mae_skill_ar1_{model}"] = 1 - mae[model][i].mean() / mae[AR1][i].mean()
        rows.append(row)
    return rows


def build_comparison(data_dir: Path, fits_dir: Path,
                     replication_dir: Path | None = None) -> ForecastComparison:
    """Classical forecasts from the snapshot, Bayesian ones from the refits, and tests."""
    from eu_okun.audit import audit_snapshot
    from eu_okun.baseline import _read_panel

    audit = audit_snapshot(data_dir)
    end = audit["analysis_period"]["end"]
    if not isinstance(end, str):
        raise EurostatDataError("Invalid study endpoint.")
    snapshot = {"manifest_sha256": str(audit["snapshot_manifest_sha256"]),
                "processed_csv_sha256": str(audit["processed_csv_sha256"])}
    fpanel = build_forecast_panel(_read_panel(data_dir / "processed" / "eu27_quarterly.csv"),
                                  end)
    predictions: dict[str, list[Prediction]] = {}
    refits: list[dict[str, Any]] = []
    for model in BAYES_MODELS:
        rows, records = load_bayes(fits_dir, fpanel, model)
        for record in records:
            if record["snapshot"] != snapshot:
                raise ValueError(f"Refit {model}/{record['target']} used another snapshot.")
        predictions[model], refits = rows, refits + records
    for model in CLASSICAL_MODELS:
        predictions[model] = classical_predictions(fpanel, model)
    comparison = ForecastComparison(fpanel, predictions, refits, [], [], snapshot=snapshot)
    comparison.family = holm_verdicts(_comparisons(comparison, FAMILY, "crps", "all", BLOCK))
    references = [(m, r) for r in (AR1, PERSISTENCE) for m in MODELS if m != r]
    comparison.descriptive = [
        *_comparisons(comparison, references, "crps", "all", BLOCK),
        *_comparisons(comparison, references, "absolute_error", "all", BLOCK),
        *_comparisons(comparison, references, "crps", "excluding pandemic", BLOCK),
    ]
    comparison.sensitivity["block 8"] = holm_verdicts(
        _comparisons(comparison, FAMILY, "crps", "all", BLOCK_SENSITIVITY))
    comparison.sensitivity["excluding pandemic"] = holm_verdicts(
        _comparisons(comparison, FAMILY, "crps", "excluding pandemic", BLOCK))
    failed = sorted({r["target"] for r in refits if not r["accepted"]})
    if failed:
        quarters = comparison.fpanel.target_quarters
        keep = [j for j, q in enumerate(quarters) if q not in failed]
        indices = circular_block_indices(len(keep), BLOCK, REPLICATES,
                                         np.random.default_rng(SEED))
        comparison.sensitivity["without failed refits"] = holm_verdicts([
            bootstrap_comparison(comparison.losses(a)[:, keep], comparison.losses(b)[:, keep],
                                 indices, a, b, "crps", "without failed refits", BLOCK)
            for a, b in FAMILY])
    comparison.country = country_heterogeneity(comparison)
    if replication_dir is not None:
        comparison.replication = load_replication(replication_dir)
    return comparison


# ----------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]], align: str = "") -> list[str]:
    align = align or "l" + "r" * (len(header) - 1)
    marks = ["---" if a == "l" else "---:" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(marks) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines + [""]


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _pct(ratio: float) -> str:
    return f"{(ratio - 1.0) * 100:+.1f}%"


def _verdict(t: BootstrapComparison) -> str:
    if t.verdict == "not distinguishable":
        return t.verdict
    return f"**{LABELS[t.verdict.removesuffix(' better')]} better**"


def _find(tests: Sequence[BootstrapComparison], a: str, b: str, loss: str = "crps",
          targets: str = "all") -> BootstrapComparison:
    return next(t for t in tests if (t.model_a, t.model_b, t.loss, t.targets)
                == (a, b, loss, targets))


def _refit_rows(comparison: ForecastComparison) -> list[list[Any]]:
    rows = []
    for model in BAYES_MODELS:
        records = [r for r in comparison.refits if r["model"] == model]
        first = sum(r["attempts"][0]["accepted"] for r in records)
        retried = sum(len(r["attempts"]) > 1 and r["accepted"] for r in records)
        failed = [r["target"] for r in records if not r["accepted"]]
        minutes = sum(a["runtime_seconds"] for r in records for a in r["attempts"]) / 60
        rows.append([f"`{model}`", len(records), sum(len(r["attempts"]) for r in records),
                     first, retried, ", ".join(failed) or "none",
                     f"{max(r['attempts'][-1]['max_rhat'] for r in records):.3f}",
                     f"{minutes:.0f}"])
    return rows


def render(comparison: ForecastComparison, assets: str, command: str) -> str:
    """Assemble the Markdown report."""
    fp = comparison.fpanel
    quarters = fp.target_quarters
    excl = len(comparison.keep("excluding pandemic"))
    lines = ["# One-quarter-ahead forecasts: hierarchical Bayes against classical models", "",
             "Generated by `eu-okun forecast-report` from the frozen snapshot "
             "`snapshots/eurostat-2026-10-15` and the Bayesian refits. Design, models, scores "
             "and decision rules were pre-registered in "
             "[BAYES_FORECAST_PROTOCOL.md](../BAYES_FORECAST_PROTOCOL.md) before any fit or "
             "score.", "",
             "**Information set.** The forecast for quarter t is issued at the end of t-1 and "
             "uses unemployment, participation and GDP growth up to t-1 only: no value dated "
             "t. **Retrospective, not real-time:** every value is the revised one in the "
             "snapshot; release lags and historical data vintages are not reconstructed. "
             "Better forecasts are not evidence of causal effects.", ""]

    # Summary
    lines += ["## Summary", ""]
    for t in comparison.family:
        lines.append(f"- {LABELS[t.model_a]} vs {LABELS[t.model_b]}: mean CRPS {_pct(t.ratio)} "
                     f"(95% interval {_pct(t.ratio_low)} to {_pct(t.ratio_high)}; Holm p "
                     f"{_p(t.p_holm)}), {_verdict(t)}.")
    decided = [t for t in comparison.family if t.verdict != "not distinguishable"]
    lines += ["", f"{len(decided)} of {len(comparison.family)} pre-registered comparisons "
              f"{'is' if len(decided) == 1 else 'are'} decided at the 5% familywise level "
              f"(synchronised circular block bootstrap, block length {BLOCK}, all "
              f"{len(quarters)} targets).", ""]
    for name, tests in comparison.sensitivity.items():
        changed = [f"{LABELS[t.model_a]} vs {LABELS[t.model_b]}"
                   for t, u in zip(comparison.family, tests, strict=True)
                   if t.verdict != u.verdict]
        lines.append(f"- Sensitivity, {name}: "
                     + (f"verdicts change for {'; '.join(changed)}." if changed
                        else "no verdict changes."))
    lines += _convergence_lines(comparison) + _accuracy_lines(comparison)
    if comparison.replication is not None:
        flips = _replication_flips(comparison)
        lines.append(f"- Replication ({comparison.replication.label}): the same verdicts "
                     + ("for all five comparisons." if not flips else
                        f"except {'; '.join(flips)}.") + " See section 5.")
    lines.append("")

    # 1. Design and records
    lines += ["## 1. Design and computation", "",
              f"- Framework sample: {len(fp.quarters)} common complete-case quarters "
              f"({fp.quarters[0]} to {fp.quarters[-1]}), {len(fp.countries)} countries.",
              f"- Origins: {len(quarters)} targets ({quarters[0]} to {quarters[-1]}); training "
              f"windows {fp.origins[0]} to {fp.origins[-1]} quarters, starting {fp.quarters[0]}.",
              f"- Sensitivity set: {excl} targets outside 2020-Q1..2022-Q2.",
              "- Every model uses identical origins, training rows and targets; the classical "
              "point forecasts are exactly those of `eu-okun forecast`.",
              f"- Snapshot manifest SHA-256 `{comparison.snapshot.get('manifest_sha256', '')}`.",
              ""]
    lines += _table(["Model", "Origins", "Fits run", "Accepted at first attempt",
                     "Accepted after retry", "Failed after retry", "Worst final R-hat",
                     "Sampling minutes"], _refit_rows(comparison), "lrrrrlrr")

    # 2. Pre-registered comparisons
    lines += ["## 2. Pre-registered comparisons", "",
              "Mean CRPS over target quarters of the cross-country mean (pp; lower is "
              "better). Difference A - B with its 95% bootstrap percentile interval; ratio "
              f"with its interval; null-centred bootstrap p-value, Holm-adjusted over the "
              f"five comparisons ({REPLICATES} resamples, seed {SEED}). The verdict uses the "
              "null-centred p-value; when a few quarters dominate, the bootstrap distribution "
              "is skewed and the percentile interval can exclude zero while p > 0.05.", ""]
    rows = [[LABELS[t.model_a], LABELS[t.model_b], f"{t.mean_a:.4f}", f"{t.mean_b:.4f}",
             f"{t.difference:+.4f} [{t.low:+.4f}, {t.high:+.4f}]",
             f"{_pct(t.ratio)} [{_pct(t.ratio_low)}, {_pct(t.ratio_high)}]", _p(t.p_value),
             _p(t.p_holm), _verdict(t)] for t in comparison.family]
    lines += _table(["A", "B", "CRPS A", "CRPS B", "Difference [95%]", "A vs B [95%]", "p",
                     "Holm p", "Verdict"], rows, "llrrrrrrl")
    for name, tests in comparison.sensitivity.items():
        rows = [[LABELS[t.model_a], LABELS[t.model_b], str(t.quarters), _pct(t.ratio),
                 f"[{_pct(t.ratio_low)}, {_pct(t.ratio_high)}]", _p(t.p_holm), _verdict(t)]
                for t in tests]
        lines += [f"Sensitivity, {name}:", ""]
        lines += _table(["A", "B", "Targets", "A vs B", "95% interval", "Holm p", "Verdict"],
                        rows, "llrrrrl")

    # 3. All models
    lines += ["## 3. All models: accuracy, skill and calibration", "",
              "Descriptive. Skill is 1 - score / score of the reference (positive: better "
              "than the reference), with its 95% block-bootstrap interval. Bias is the mean "
              "of the predictive mean minus the outcome.", ""]
    for targets in ("all", "excluding pandemic"):
        rows = []
        for model in MODELS:
            c50, c90 = comparison.coverage(model, targets)
            cells = [LABELS[model], f"{comparison.mean_loss(model, 'crps', targets):.4f}",
                     f"{comparison.mean_loss(model, 'absolute_error', targets):.4f}",
                     f"{np.sqrt(comparison.mean_loss(model, 'squared_error', targets)):.4f}",
                     f"{_bias(comparison, model, targets):+.4f}"]
            for ref in (PERSISTENCE, AR1):
                if model == ref:
                    cells.append("reference")
                    continue
                t = _find(comparison.descriptive, model, ref, "crps", targets)
                cells.append(f"{1 - t.ratio:+.3f} [{1 - t.ratio_high:+.3f}, "
                             f"{1 - t.ratio_low:+.3f}]")
            cells += [f"{c50:.0%}", f"{c90:.0%}"]
            rows.append(cells)
        title = ("All targets" if targets == "all"
                 else f"Targets outside 2020-Q1..2022-Q2 ({excl})")
        lines += [f"**{title}**", ""]
        lines += _table(["Model", "CRPS", "MAE", "RMSE", "Bias",
                         "CRPS skill vs persistence [95%]", "CRPS skill vs AR(1) [95%]",
                         "50% cover", "90% cover"], rows)
    rows = []
    for model in MODELS:
        cells = [LABELS[model]]
        for ref in (PERSISTENCE, AR1):
            if model == ref:
                cells.append("reference")
                continue
            t = _find(comparison.descriptive, model, ref, "absolute_error")
            cells.append(f"{1 - t.ratio:+.3f} [{1 - t.ratio_high:+.3f}, {1 - t.ratio_low:+.3f}]")
        rows.append(cells)
    lines += ["MAE skill over all targets (the framework's point metric):", ""]
    lines += _table(["Model", "MAE skill vs persistence [95%]", "MAE skill vs AR(1) [95%]"],
                    rows)
    lines += [f"![CRPS skill relative to AR(1)]({assets}/skill.png)", "",
              f"![Quarterly CRPS differences]({assets}/quarterly_differences.png)", "",
              *_dominance_lines(comparison), "",
              f"![Calibration]({assets}/calibration.png)", ""]

    # 4. Countries
    lines += ["## 4. Countries (descriptive)", "",
              f"CRPS skill relative to AR(1) by country over the {len(quarters)} targets, and "
              f"the CRPS difference of comparison 1 ({LABELS[FAMILY[0][0]]} minus "
              f"{LABELS[FAMILY[0][1]]}) with its block-bootstrap interval. Alphabetical; no "
              "ranking and no per-country significance claims.", ""]
    rows = []
    for r in comparison.country:
        rows.append([r["geo"], *(f"{r[f'crps_skill_ar1_{m}']:+.3f}" for m in
                                 ("bayes_gdp_ar1", "gdp_unemployment_ar1", "zero_change")),
                     f"{r['difference_1']:+.4f} [{r['difference_1_low']:+.4f}, "
                     f"{r['difference_1_high']:+.4f}]"])
    lines += _table(["Country", f"{LABELS['bayes_gdp_ar1']}",
                     f"{LABELS['gdp_unemployment_ar1']}", f"{LABELS['zero_change']}",
                     "Comparison 1 difference [95%]"], rows)
    better = sum(r["difference_1"] < 0 for r in comparison.country)
    below = sum(r["difference_1_high"] < 0 for r in comparison.country)
    above = sum(r["difference_1_low"] > 0 for r in comparison.country)
    lines += [f"{LABELS[FAMILY[0][0]]} has the lower mean CRPS in {better} of "
              f"{len(comparison.country)} countries; the interval excludes zero in its favour "
              f"in {below} and against it in {above} (unadjusted, descriptive).", "",
              f"![Country skill]({assets}/countries.png)", ""]

    number = 5
    if comparison.replication is not None:
        lines += _replication_section(comparison)
        number += 1
    lines += [
        f"## {number}. Interpretation and limitations", "",
        "- **Improvement, significance, causality.** A lower mean score is an improvement "
        "in this sample; the bootstrap intervals and Holm-adjusted p-values say whether it "
        "is distinguishable from sampling variation; neither says anything causal.",
        "- **Retrospective forecasts.** Revised data, with GDP growth for t-1 assumed known "
        "at the end of t-1. A real-time evaluation needs Eurostat data as published at each "
        "origin.",
        "- **Pandemic quarters** are in training and targets, as in the framework; the "
        "sensitivity set without 2020-Q1..2022-Q2 targets shows how much they drive the "
        "results.",
        "- **Power.** 34 target quarters; 'not distinguishable' never means equal accuracy.",
        "",
        f"## {number + 1}. Reproduction", "",
        "```bash",
        "poetry run eu-okun forecast-bayes-fit --data-dir snapshots/eurostat-2026-10-15 "
        "--output-dir results/forecast-bayes",
        command,
        "```", "",
    ]
    return "\n".join(lines)


def _convergence_lines(comparison: ForecastComparison) -> list[str]:
    """How many refits failed the acceptance criteria, and how narrowly."""
    failed = [r for r in comparison.refits if not r["accepted"]]
    if not failed:
        return [f"- Convergence: all {len(comparison.refits)} refits met the acceptance "
                "criteria."]
    finals = [r["attempts"][-1] for r in failed]
    parameters: dict[str, int] = {}
    for a in finals:
        for failure in a["failures"]:
            if "(" in failure:
                name = failure.rsplit("(", 1)[1].rstrip(")")
                parameters[name] = parameters.get(name, 0) + 1
    common = max(parameters, key=lambda k: parameters[k]) if parameters else "n/a"
    ess = [min(a["min_ess_bulk"], a["min_ess_tail"]) for a in finals]
    by_model = "; ".join(f"`{m}` {sum(r['model'] == m for r in failed)} of "
                         f"{sum(r['model'] == m for r in comparison.refits)}"
                         for m in BAYES_MODELS)
    return [f"- Convergence: {len(failed)} of {len(comparison.refits)} refits failed the "
            f"acceptance criteria after the single retry ({by_model}). In their final "
            "attempts: "
            f"R-hat at most {max(a['max_rhat'] for a in finals):.3f}, ESS at least "
            f"{min(ess):.0f}, {sum(a['divergences'] for a in finals)} divergent transitions; "
            f"the parameter named most often is `{common}`. Their forecasts are scored, as "
            "registered; the sensitivity analysis without their target quarters is in "
            "section 2."]


def _replication_flips(comparison: ForecastComparison) -> list[str]:
    rep_run = comparison.replication
    assert rep_run is not None
    out = []
    for t in comparison.family:
        _, p_holm, verdict = rep_run.family[(t.model_a, t.model_b)]
        if verdict != t.verdict:
            out.append(f"{LABELS[t.model_a]} vs {LABELS[t.model_b]} (Holm p {_p(t.p_holm)} "
                       f"here, {_p(p_holm)} there)")
    return out


def _replication_section(comparison: ForecastComparison) -> list[str]:
    rep_run = comparison.replication
    assert rep_run is not None
    lines = ["## 5. Replication on a second platform", "",
             f"{rep_run.label}: the same snapshot, lock file, seeds, sampler settings and "
             "single retry. Its tables are in `replication-linux/`.", ""]
    rows = []
    for t in comparison.family:
        ratio, p_holm, verdict = rep_run.family[(t.model_a, t.model_b)]
        rows.append([LABELS[t.model_a], LABELS[t.model_b], _pct(t.ratio), _p(t.p_holm),
                     t.verdict, _pct(ratio), _p(p_holm), verdict])
    lines += _table(["A", "B", "A vs B (this run)", "Holm p", "Verdict",
                     "A vs B (replication)", "Holm p", "Verdict"], rows, "llrrlrrl")
    here = {(p.model, p.geo, p.quarter): p.crps for m in BAYES_MODELS
            for p in comparison.predictions[m]}
    gaps = [abs(here[k] - v) for k, v in rep_run.bayes_crps.items() if k in here]
    failed_here = {m: sorted(r["target"] for r in comparison.refits
                             if r["model"] == m and not r["accepted"]) for m in BAYES_MODELS}
    lines += [
        f"- Bayesian forecasts: the CRPS of the {len(gaps)} Bayesian cells differs by "
        f"{np.mean(gaps):.4f} on average and at most {max(gaps):.4f} between the runs.",
        "- Failed refits: " + "; ".join(
            f"`{m}` {len(failed_here[m])} here, {len(rep_run.failed.get(m, []))} there"
            for m in BAYES_MODELS) + ".",
        "- Verdicts: " + ("identical for all five comparisons." if not _replication_flips(
            comparison) else f"identical except {'; '.join(_replication_flips(comparison))}."),
        "",
    ]
    return lines


def _dominance_lines(comparison: ForecastComparison) -> list[str]:
    """Which target quarters carry the mean difference of the first three comparisons."""
    quarters = comparison.fpanel.target_quarters
    out = []
    for a, b in FAMILY[:3]:
        d = comparison.losses(a).mean(axis=0) - comparison.losses(b).mean(axis=0)
        top = np.argsort(np.abs(d))[::-1][:2]
        rest = np.delete(d, top)
        out.append(f"- {LABELS[a]} minus {LABELS[b]}: largest quarterly differences "
                   f"{quarters[top[0]]} ({d[top[0]]:+.3f}) and {quarters[top[1]]} "
                   f"({d[top[1]]:+.3f}); mean difference {d.mean():+.4f} over all "
                   f"{len(quarters)} quarters, {rest.mean():+.4f} without these two.")
    return out


def _best(comparison: ForecastComparison, loss: str, targets: str) -> str:
    """The model with the lowest mean loss."""
    scores = {m: comparison.mean_loss(m, loss, targets) for m in MODELS}
    return min(scores, key=scores.__getitem__)


def _accuracy_lines(comparison: ForecastComparison) -> list[str]:
    """Point accuracy and calibration in one place, kept apart from the tests."""
    out = []
    for targets, label in (("all", "all targets"),
                           ("excluding pandemic", "outside 2020-Q1..2022-Q2")):
        best = {loss: _best(comparison, loss, targets)
                for loss in ("crps", "absolute_error", "squared_error")}
        cover = [comparison.coverage(m, targets)[1] for m in MODELS]
        out.append(f"- Point and probabilistic accuracy, {label}: lowest CRPS "
                   f"{LABELS[best['crps']]}, lowest MAE {LABELS[best['absolute_error']]}, "
                   f"lowest RMSE {LABELS[best['squared_error']]}; 90% intervals cover "
                   f"{min(cover):.0%}-{max(cover):.0%} of outcomes across models.")
    return out


def _bias(comparison: ForecastComparison, model: str, targets: str) -> float:
    quarters = {comparison.fpanel.target_quarters[j] for j in comparison.keep(targets)}
    rows = [p for p in comparison.predictions[model] if p.quarter in quarters]
    return float(np.mean([p.mean - p.actual for p in rows]))


def write_figures(comparison: ForecastComparison, directory: Path, source: str) -> None:
    """Skill, quarterly differences, calibration and country skill."""
    from eu_okun.review_figures import AQUA, BLUE, INK2, MUTED, ORANGE, SURFACE, _finish, _pyplot

    plt = _pyplot()
    from matplotlib.lines import Line2D

    colour = {m: BLUE for m in BAYES_MODELS}
    colour.update({m: ORANGE for m in ("gdp_lags", "gdp_unemployment_ar1",
                                       "gdp_ar1_participation", "unemployment_ar1")})
    colour.update({"zero_change": AQUA, "historical_mean": AQUA})
    others = [m for m in MODELS if m != AR1]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    ys = np.arange(len(others))[::-1]
    for ax, targets in zip(axes, ("all", "excluding pandemic"), strict=True):
        for y, model in zip(ys, others, strict=True):
            t = _find(comparison.descriptive, model, AR1, "crps", targets)
            ax.plot([1 - t.ratio_high, 1 - t.ratio_low], [y, y], color=colour[model], lw=1.8,
                    solid_capstyle="round")
            ax.plot([1 - t.ratio], [y], "o", ms=6.5, color=colour[model], mec=SURFACE, mew=1.2)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_title("All targets" if targets == "all" else "Targets outside 2020-Q1..2022-Q2")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([LABELS[m] for m in others], fontsize=8.5)
    fig.supxlabel("CRPS skill relative to AR(1) (right of zero: better); 95% block-bootstrap "
                  "interval", fontsize=9, color=INK2)
    fig.legend(handles=[
        Line2D([], [], marker="o", color=BLUE, lw=1.8, ms=6, label="Hierarchical Bayes"),
        Line2D([], [], marker="o", color=ORANGE, lw=1.8, ms=6, label="Classical regression"),
        Line2D([], [], marker="o", color=AQUA, lw=1.8, ms=6, label="Benchmark"),
    ], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.09))
    fig.subplots_adjust(wspace=0.08, bottom=0.2)
    _finish(plt, fig, directory / "skill.png", source, credit_y=-0.15)

    quarters = comparison.fpanel.target_quarters
    x = np.arange(len(quarters))
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    first, last = (quarters.index(q) for q in PANDEMIC_TARGETS)
    ax.axvspan(first - 0.5, last + 0.5, color=MUTED, alpha=0.12, lw=0)
    for (a, b), c in zip(FAMILY[:3], (BLUE, ORANGE, AQUA), strict=True):
        d = comparison.losses(a).mean(axis=0) - comparison.losses(b).mean(axis=0)
        ax.plot(x, d, color=c, lw=2, marker="o", ms=4, mec=SURFACE, mew=1,
                label=f"{LABELS[a]} minus {LABELS[b]}")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(quarters[::2], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cross-country mean CRPS difference (pp)")
    ax.set_title("Quarterly differences; shaded: 2020-Q1..2022-Q2")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, fontsize=8.5)
    _finish(plt, fig, directory / "quarterly_differences.png", source, credit_y=-0.3)

    shown = ("bayes_gdp_ar1", "gdp_unemployment_ar1", "unemployment_ar1", "zero_change")
    fig, axes = plt.subplots(1, len(shown), figsize=(12.0, 3.4), sharey=True)
    bins = np.linspace(0, 1, 11)
    for ax, model in zip(axes, shown, strict=True):
        ax.hist([p.pit for p in comparison.predictions[model]], bins=bins, density=True,
                color=colour[model], edgecolor=SURFACE, linewidth=2)
        ax.axhline(1.0, color=INK2, lw=1, ls=(0, (3, 2)))
        c50, c90 = comparison.coverage(model)
        ax.set_title(f"{LABELS[model]}\ncoverage {c50:.0%} / {c90:.0%}", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Density")
    fig.supxlabel("PIT value, all targets; dashed: uniform. Coverage of central 50% / 90% "
                  "intervals in the titles", fontsize=9, color=INK2)
    fig.subplots_adjust(wspace=0.08, bottom=0.2)
    _finish(plt, fig, directory / "calibration.png", source, credit_y=-0.08)

    rows = sorted(comparison.country, key=lambda r: r["geo"])
    ys = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 8.8))
    for y, r in zip(ys, rows, strict=True):
        for model, offset in (("bayes_gdp_ar1", -0.18), ("gdp_unemployment_ar1", 0.18)):
            ax.plot([r[f"crps_skill_ar1_{model}"]], [y + offset], "o", ms=6,
                    color=colour[model], mec=SURFACE, mew=1)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["geo"] for r in rows])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("CRPS skill relative to AR(1), all targets (right of zero: better)")
    ax.set_title("Forecast skill differs widely across countries")
    ax.legend(handles=[
        Line2D([], [], marker="o", color=BLUE, lw=0, ms=6, label=LABELS["bayes_gdp_ar1"]),
        Line2D([], [], marker="o", color=ORANGE, lw=0, ms=6,
               label=LABELS["gdp_unemployment_ar1"]),
    ], loc="upper left", bbox_to_anchor=(0.0, -0.07), fontsize=8.5)
    _finish(plt, fig, directory / "countries.png", source, credit_y=-0.1)


def write_tables(comparison: ForecastComparison, directory: Path) -> None:
    """Every forecast, the tests, country heterogeneity and refit records."""
    directory.mkdir(parents=True, exist_ok=True)
    write_predictions(directory / "forecasts.csv",
                      [p for rows in comparison.predictions.values() for p in rows])
    tests = ([("family", t) for t in comparison.family]
             + [("descriptive", t) for t in comparison.descriptive]
             + [(f"sensitivity: {k}", t) for k, v in comparison.sensitivity.items() for t in v])
    with (directory / "comparisons.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["kind", *asdict(comparison.family[0])],
                                lineterminator="\n")
        writer.writeheader()
        for kind, t in tests:
            writer.writerow({"kind": kind, **asdict(t)})
    with (directory / "countries.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison.country[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparison.country)
    refits = [{k: v for k, v in r.items() if k not in ("environment", "snapshot")}
              for r in comparison.refits]
    environments = sorted({json.dumps(r["environment"], sort_keys=True)
                           for r in comparison.refits})
    (directory / "refits.json").write_text(json.dumps({
        "snapshot": comparison.snapshot, "environments": [json.loads(e) for e in environments],
        "refits": refits}, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_report(data_dir: Path, fits_dir: Path, output_dir: Path, report: Path | None = None,
                 figures: bool = True,
                 replication_dir: Path | None = None) -> ForecastComparison:
    """Build the comparison and write tables, figures and the Markdown report."""
    comparison = build_comparison(data_dir, fits_dir, replication_dir)
    report = report or output_dir / "FORECAST_COMPARISON.md"
    write_tables(comparison, output_dir)
    if figures:
        source = ("Source: Eurostat, snapshot 2026-10-15; retrospective one-quarter-ahead "
                  "forecasts (BAYES_FORECAST_PROTOCOL.md). Not causal.")
        write_figures(comparison, output_dir, source)
    assets = Path(os.path.relpath(output_dir, report.parent)).as_posix()
    command = (f"poetry run eu-okun forecast-report --data-dir {data_dir.as_posix()} "
               f"--fits-dir {fits_dir.as_posix()} --output-dir {output_dir.as_posix()} "
               f"--report {report.as_posix()}"
               + (f" --replication-dir {replication_dir.as_posix()}"
                  if replication_dir is not None else ""))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render(comparison, assets, command), encoding="utf-8", newline="\n")
    return comparison
