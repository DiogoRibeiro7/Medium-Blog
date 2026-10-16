"""Tables, figures and the Markdown report of the predictive comparison.

The Bayesian predictions come from the saved refits (``eu-okun predictive-fit``); the
classical ones are recomputed here from the snapshot, because they are cheap and
deterministic. Every number in the report is computed from these predictions.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from eu_okun.audit import audit_snapshot
from eu_okun.baseline import _read_panel
from eu_okun.predictive import (
    ALPHA,
    BAYES_MODELS,
    CLASSICAL_MODELS,
    EVALUATION_SETS,
    FAMILY,
    LOSSES,
    MODEL_LABELS,
    MODELS,
    NW_LAGS,
    PanelTest,
    Prediction,
    PredictiveData,
    adjust_family,
    build_predictive_data,
    classical_predictions,
    loss_matrix,
    panel_test,
    quarterly_differentials,
    read_predictions,
    write_predictions,
)
from eu_okun.review import read_flags

REFERENCE = "ols_dl"


@dataclass(frozen=True)
class Replication:
    """Tables of an independent run of the same refits (e.g. on another platform)."""

    label: str
    family: dict[tuple[str, str, str], tuple[float, float, str]]  # ratio, Holm p, verdict
    failed: dict[str, list[str]]                                  # set -> target quarters
    bayes_crps: dict[tuple[str, str, str, str], float]            # set, model, geo, quarter


def load_replication(directory: Path) -> Replication:
    """Read ``comparisons.csv``, ``refits.json`` and ``predictions.csv`` of another run."""
    family = {}
    with (directory / "comparisons.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["kind"] == "family":
                family[(row["evaluation"], row["model_a"], row["model_b"])] = (
                    float(row["ratio"]), float(row["p_holm"]), row["verdict"])
    refits = json.loads((directory / "refits.json").read_text(encoding="utf-8"))["refits"]
    failed: dict[str, list[str]] = {name: [] for name in EVALUATION_SETS}
    for record in refits:
        if not record["accepted"] and record["target"] not in failed[record["evaluation"]]:
            failed[record["evaluation"]].append(record["target"])
    crps = {(p.evaluation, p.model, p.geo, p.quarter): p.crps
            for p in read_predictions(directory / "predictions.csv") if p.model in BAYES_MODELS}
    readme = directory / "README.md"
    label = (readme.read_text(encoding="utf-8").splitlines()[0].lstrip("# ")
             if readme.exists() else directory.name)
    return Replication(label, family, {k: sorted(v) for k, v in failed.items()}, crps)


@dataclass
class Comparison:
    """Everything the report is generated from."""

    pdata: dict[str, PredictiveData]
    predictions: dict[tuple[str, str], list[Prediction]]
    refits: list[dict[str, Any]]
    family: list[PanelTest]
    descriptive: list[PanelTest]
    sensitivity: list[PanelTest] = field(default_factory=list)
    failed_quarters: dict[str, list[str]] = field(default_factory=dict)
    snapshot: dict[str, str] = field(default_factory=dict)
    replication: Replication | None = None

    def losses(self, evaluation: str, model: str, loss: str = "crps") -> np.ndarray:
        return loss_matrix(self.predictions[(evaluation, model)], self.pdata[evaluation], loss)

    def mean_loss(self, evaluation: str, model: str, loss: str = "crps") -> float:
        """Mean over target quarters of the cross-country mean loss."""
        values = self.losses(evaluation, model, loss)
        scored = self.pdata[evaluation].scored
        return float(np.mean([values[scored[:, j], j].mean() for j in range(scored.shape[1])]))

    def coverage(self, evaluation: str, model: str) -> tuple[float, float]:
        rows = self.predictions[(evaluation, model)]
        inside50 = np.mean([p.q25 <= p.actual <= p.q75 for p in rows])
        inside90 = np.mean([p.q05 <= p.actual <= p.q95 for p in rows])
        return float(inside50), float(inside90)


# ----------------------------------------------------------------------------------
# Building the comparison
# ----------------------------------------------------------------------------------


def load_bayes(fits_dir: Path, pdata: PredictiveData, model: str,
               ) -> tuple[list[Prediction], list[dict[str, Any]]]:
    """Predictions and origin records of one Bayesian model; every origin is required."""
    rows: list[Prediction] = []
    records: list[dict[str, Any]] = []
    missing = []
    for quarter in pdata.target_quarters:
        directory = fits_dir / pdata.evaluation.name / model / quarter
        if not (directory / "origin.json").exists():
            missing.append(quarter)
            continue
        records.append(json.loads((directory / "origin.json").read_text(encoding="utf-8")))
        rows += read_predictions(directory / "predictions.csv")
    if missing:
        raise FileNotFoundError(f"{pdata.evaluation.name}/{model}: no refit for "
                                f"{', '.join(missing)} in {fits_dir}.")
    return rows, records


def _family_tests(comparison: Comparison, quarters: dict[str, list[int]] | None = None,
                  ) -> list[PanelTest]:
    """The pre-registered family, optionally restricted to some target columns."""
    tests = []
    for evaluation in EVALUATION_SETS:
        scored = comparison.pdata[evaluation].scored
        keep = list(range(scored.shape[1])) if quarters is None else quarters[evaluation]
        for a, b in FAMILY:
            tests.append(panel_test(
                evaluation, a, b, comparison.losses(evaluation, a)[:, keep],
                comparison.losses(evaluation, b)[:, keep], scored[:, keep]))
    return adjust_family(tests)


def build_comparison(data_dir: Path, fits_dir: Path,
                     replication_dir: Path | None = None) -> Comparison:
    """Recompute the classical predictions, load the Bayesian ones and run the tests."""
    audit = audit_snapshot(data_dir)
    csv_path = data_dir / "processed" / "eu27_quarterly.csv"
    panel, flags = _read_panel(csv_path), read_flags(csv_path)
    snapshot = {"manifest_sha256": str(audit["snapshot_manifest_sha256"]),
                "processed_csv_sha256": str(audit["processed_csv_sha256"])}
    pdata = {name: build_predictive_data(panel, flags, s) for name, s in EVALUATION_SETS.items()}
    predictions: dict[tuple[str, str], list[Prediction]] = {}
    refits: list[dict[str, Any]] = []
    for name, data in pdata.items():
        for model in BAYES_MODELS:
            rows, records = load_bayes(fits_dir, data, model)
            for record in records:
                if record["snapshot"] != snapshot:
                    raise ValueError(f"Refit {name}/{model}/{record['target']} used another "
                                     "snapshot.")
            predictions[(name, model)] = rows
            refits += records
        for model in CLASSICAL_MODELS:
            predictions[(name, model)] = classical_predictions(data, model)
    comparison = Comparison(pdata, predictions, refits, [], [], snapshot=snapshot)
    comparison.family = _family_tests(comparison)
    comparison.descriptive = [
        panel_test(name, model, REFERENCE, comparison.losses(name, model, loss),
                   comparison.losses(name, REFERENCE, loss), pdata[name].scored, loss)
        for name in pdata for loss in LOSSES for model in MODELS if model != REFERENCE
    ]
    failed = {name: sorted({r["target"] for r in refits
                            if r["evaluation"] == name and not r["accepted"]}) for name in pdata}
    comparison.failed_quarters = failed
    if any(failed.values()):
        keep = {name: [j for j, q in enumerate(pdata[name].target_quarters)
                       if q not in failed[name]] for name in pdata}
        if all(len(k) >= 8 for k in keep.values()):
            comparison.sensitivity = _family_tests(comparison, keep)
    if replication_dir is not None:
        comparison.replication = load_replication(replication_dir)
    return comparison


# ----------------------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------------------


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]], align: str = "") -> list[str]:
    align = align or "l" + "r" * (len(header) - 1)
    marks = ["---" if a == "l" else "---:" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(marks) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines + [""]


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _label(model: str) -> str:
    return MODEL_LABELS[model]


def _pct(ratio: float) -> str:
    """Relative change of a ratio of mean losses, e.g. 0.93 -> '-7.0%'."""
    return f"{(ratio - 1.0) * 100:+.1f}%"


def _test(tests: Sequence[PanelTest], evaluation: str, a: str, b: str,
          loss: str = "crps") -> PanelTest:
    return next(t for t in tests if t.evaluation == evaluation and t.model_a == a
                and t.model_b == b and t.score == loss)


def _interval(test: PanelTest, level: float = 0.95) -> tuple[float, float]:
    """Unadjusted interval for the mean differential (HLN-scaled, t with T-1 df)."""
    if test.standard_error == 0.0:
        return test.mean_difference, test.mean_difference
    t = test.quarters
    correction = np.sqrt((t - 1) / t)
    half = float(stats.t.ppf(0.5 + level / 2, t - 1)) * test.standard_error / correction
    return test.mean_difference - half, test.mean_difference + half


def _verdict_text(test: PanelTest) -> str:
    if test.verdict == "not distinguishable":
        return "not distinguishable"
    winner = test.verdict.removesuffix(" better")
    return f"**{_label(winner)} better**"


# ----------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------


def _refit_rows(comparison: Comparison) -> list[list[Any]]:
    rows = []
    for evaluation in EVALUATION_SETS:
        for model in BAYES_MODELS:
            records = [r for r in comparison.refits
                       if r["evaluation"] == evaluation and r["model"] == model]
            first_ok = sum(r["attempts"][0]["accepted"] for r in records)
            retried_ok = sum(len(r["attempts"]) > 1 and r["accepted"] for r in records)
            failed = [r["target"] for r in records if not r["accepted"]]
            worst = max(r["attempts"][-1]["max_rhat"] for r in records)
            minutes = sum(a["runtime_seconds"] for r in records for a in r["attempts"]) / 60
            rows.append([evaluation, f"`{model}`", len(records), first_ok, retried_ok,
                         ", ".join(failed) if failed else "none", f"{worst:.3f}",
                         f"{minutes:.0f}"])
    return rows


def render(comparison: Comparison, assets: str, command: str) -> str:
    """Assemble the Markdown report."""
    family = comparison.family
    sets = list(EVALUATION_SETS)
    lines: list[str] = ["# Predictive comparison: hierarchical Bayes against classical models",
                        ""]
    lines += [
        "Generated by `eu-okun predictive-report` from the frozen snapshot "
        "`snapshots/eurostat-2026-10-15` and the expanding-window refits. The design, "
        "models, scores and decision rules were pre-registered in "
        "[PREDICTIVE_PROTOCOL.md](../PREDICTIVE_PROTOCOL.md) before any held-out "
        "prediction was computed.", "",
        "Each prediction for quarter t uses data up to t-1 **plus GDP growth in quarter "
        "t**: it is a conditional prediction on revised data, not a real-time forecast. "
        "Better prediction is not evidence of a causal effect.", "",
    ]

    # Summary
    lines += ["## Summary", ""]
    for evaluation in sets:
        s = EVALUATION_SETS[evaluation]
        lines.append(f"**{evaluation}, {s.label}** (targets {s.first_target} to "
                     f"{s.last_target}, {comparison.pdata[evaluation].scored.sum()} scored "
                     "cells):")
        lines.append("")
        for a, b in FAMILY:
            t = _test(family, evaluation, a, b)
            lines.append(f"- {_label(a)} vs {_label(b)}: mean CRPS {_pct(t.ratio)} "
                         f"(Holm p = {_p(t.p_holm)}), {_verdict_text(t)}.")
        lines.append("")
    decided = [t for t in family if t.verdict != "not distinguishable"]
    lines += [
        f"{len(decided)} of the {len(family)} pre-registered comparisons "
        f"{'is' if len(decided) == 1 else 'are'} decided at the 5% familywise level; the "
        "others are not distinguishable, which with 16 target quarters per set does not mean "
        "equal accuracy.", "",
    ]
    if comparison.sensitivity:
        kept = [t for t in comparison.sensitivity if t.verdict != "not distinguishable"]
        changed = [f"{t.evaluation} {_label(t.model_a)} vs {_label(t.model_b)}"
                   for t, u in zip(family, comparison.sensitivity, strict=True)
                   if t.verdict != u.verdict]
        lines += [
            f"Without the target quarters in which a Bayesian refit failed after the retry, "
            f"{len(kept)} of {len(comparison.sensitivity)} comparisons are decided"
            + (f"; the verdict changes for {'; '.join(changed)}." if changed else
               "; no verdict changes.") + " See section 8.", ""]
    for evaluation in sets:
        ref = comparison.mean_loss(evaluation, "zero_change")
        gdp_models = [m for m in MODELS if m in BAYES_MODELS or m in ("ols_dl", "ardl")]
        beat = [_label(m) for m in gdp_models if comparison.mean_loss(evaluation, m) < ref]
        lines.append(f"- {evaluation}: {len(beat)} of {len(gdp_models)} models that use GDP "
                     f"growth in the target quarter {'has' if len(beat) == 1 else 'have'} a "
                     "lower mean CRPS than zero change"
                     + (f" ({'; '.join(beat)})." if beat else "."))
    lines.append("")
    if comparison.replication is not None:
        flips = _replication_flips(comparison)
        lines += [f"An independent run ({comparison.replication.label}) gives the same verdicts "
                  + ("for all ten comparisons." if not flips else
                     f"except for {'; '.join(flips)}.") + " See section 9.", ""]
    calibration = []
    for evaluation in sets:
        for model in ("bayes_primary", "ols_dl"):
            c50, c90 = comparison.coverage(evaluation, model)
            calibration.append(f"{_label(model)} in {evaluation}: {c50:.0%} and {c90:.0%}")
    lines += ["Empirical coverage of the central 50% and 90% predictive intervals: "
              + "; ".join(calibration) + ".", ""]

    # 1. Design
    lines += [
        "## 1. Design", "",
        "| Set | Targets | Training data for target t | Scored cells |",
        "| --- | --- | --- | ---: |",
    ]
    for evaluation in sets:
        s = EVALUATION_SETS[evaluation]
        training = ("2009-Q2 to t-1" if evaluation == "E1"
                    else "2009-Q2 to 2019-Q4, plus 2022-Q3 to t-1")
        lines.append(f"| {evaluation} ({s.label}) | {s.first_target} to {s.last_target} | "
                     f"{training} | {comparison.pdata[evaluation].scored.sum()} of "
                     f"{comparison.pdata[evaluation].scored.size} |")
    lines += [
        "",
        "Every model is re-estimated at every origin on observations strictly before the "
        "target quarter, and every model is scored on the same cells. Cells whose change "
        "relies on a flagged Eurostat value are not scored. ES and FR are not scored in "
        "E2 (their unemployment series carry 'definition differs' flags from 2021).", "",
        _models_table(), "",
    ]

    # 2. Refits
    lines += ["## 2. Bayesian refits and convergence", "",
              "Acceptance criteria and the single retry with 3000 warm-up iterations are "
              "those of BAYES_PROTOCOL.md, section 6. Predictions from a refit that still "
              "fails are scored (section 8 repeats the tests without those quarters).", ""]
    lines += _table(["Set", "Model", "Origins", "Accepted at first attempt",
                     "Accepted after retry", "Failed after retry", "Worst final R-hat",
                     "Sampling minutes"], _refit_rows(comparison), "llrrrlrr")

    # 3. Family
    lines += ["## 3. Pre-registered comparisons", "",
              "Mean CRPS is the mean over target quarters of the cross-country mean, in "
              "percentage points (lower is better). The difference is A minus B; the "
              f"test is a panel Diebold-Mariano test (Newey-West, {NW_LAGS} lags, "
              "Harvey-Leybourne-Newbold correction, t with 15 df), Holm-adjusted over the "
              f"ten comparisons at {ALPHA:.0%}.", ""]
    rows = []
    for t in family:
        rows.append([t.evaluation, _label(t.model_a), _label(t.model_b), f"{t.mean_a:.4f}",
                     f"{t.mean_b:.4f}", _pct(t.ratio), f"{t.mean_difference:+.4f}",
                     f"{t.statistic:+.2f}", _p(t.p_value), _p(t.p_holm), _verdict_text(t)])
    lines += _table(["Set", "A", "B", "CRPS A", "CRPS B", "A vs B", "Difference",
                     "Statistic", "p", "Holm p", "Verdict"], rows, "lllrrrrrrrl")
    lines += _family_reading(comparison)

    # 4. All models
    lines += ["## 4. All models and scores", "",
              "Descriptive. Ratios are relative to the OLS distributed lag; p-values "
              "(unadjusted) test each model against it with the same panel test.", ""]
    for evaluation in sets:
        rows = []
        for model in MODELS:
            crps = comparison.mean_loss(evaluation, model)
            ref = comparison.mean_loss(evaluation, REFERENCE)
            p = ("" if model == REFERENCE
                 else _p(_test(comparison.descriptive, evaluation, model, REFERENCE).p_value))
            c50, c90 = comparison.coverage(evaluation, model)
            rows.append([_label(model), f"{crps:.4f}", _pct(crps / ref), p,
                         f"{comparison.mean_loss(evaluation, model, 'neg_log_score'):.3f}",
                         f"{np.sqrt(comparison.mean_loss(evaluation, model, 'squared_error')):.4f}",
                         f"{comparison.mean_loss(evaluation, model, 'absolute_error'):.4f}",
                         f"{c50:.0%}", f"{c90:.0%}"])
        lines += [f"**{evaluation}, {EVALUATION_SETS[evaluation].label}**", ""]
        lines += _table(["Model", "CRPS", "vs OLS-DL", "p vs OLS-DL", "Neg. log score",
                         "RMSE", "MAE", "50% cover", "90% cover"], rows)
    lines += [f"![Mean CRPS difference from the OLS distributed lag]({assets}/"
              "crps_differentials.png)", ""]

    # 5. Over time
    lines += ["## 5. Loss differentials over time", "",
              "Cross-country mean CRPS differential in each target quarter (negative: the "
              "first model predicted better). One or two quarters can dominate a mean over "
              "16.", "", f"![Quarterly CRPS differentials]({assets}/quarterly_differentials.png)",
              ""]
    lines += _dominance_text(comparison)

    # 6. Calibration
    lines += ["## 6. Calibration", "",
              "A calibrated predictive distribution has a flat PIT histogram and covers the "
              "outcome 50% and 90% of the time with its central 50% and 90% intervals "
              "(coverage in the tables of section 4).", "",
              f"![PIT histograms]({assets}/calibration.png)", ""]
    lines += _calibration_text(comparison)

    # 7. Countries
    lines += _country_section(comparison)

    # 8. Sensitivity and secondary losses
    lines += ["## 8. Sensitivity and secondary losses", ""]
    if comparison.sensitivity:
        dropped = "; ".join(f"{k}: " + ", ".join(v)
                            for k, v in comparison.failed_quarters.items() if v)
        lines += [
            "Without the target quarters in which any Bayesian refit failed after the retry "
            f"({dropped}), Holm-adjusted again over the ten comparisons:", ""]
        sensitivity_rows: list[list[Any]] = [
            [t.evaluation, _label(t.model_a), _label(t.model_b), t.quarters, _pct(t.ratio),
             _p(t.p_holm), _verdict_text(t)] for t in comparison.sensitivity]
        lines += _table(["Set", "A", "B", "Quarters", "A vs B", "Holm p", "Verdict"],
                        sensitivity_rows, "lllrrrl")
    elif any(comparison.failed_quarters.values()):
        lines += ["Too few quarters remain without failed refits for the sensitivity test.",
                  ""]
    else:
        lines += ["Every Bayesian refit met the acceptance criteria, so the sensitivity "
                  "analysis without failed refits coincides with section 3.", ""]
    lines += ["The family comparisons under the secondary losses (unadjusted p-values, "
              "descriptive):", ""]
    rows = []
    for evaluation in sets:
        for a, b in FAMILY:
            cells = []
            for loss in ("neg_log_score", "squared_error", "absolute_error"):
                t = panel_test(evaluation, a, b, comparison.losses(evaluation, a, loss),
                               comparison.losses(evaluation, b, loss),
                               comparison.pdata[evaluation].scored, loss)
                cells.append(f"{t.mean_difference:+.4f} (p {_p(t.p_value)})")
            rows.append([evaluation, _label(a), _label(b), *cells])
    lines += _table(["Set", "A", "B", "Neg. log score diff.", "Squared error diff.",
                     "Absolute error diff."], rows, "lllrrr")

    # 9. Replication (when supplied), then interpretation and reproduction
    number = 9
    if comparison.replication is not None:
        lines += _replication_section(comparison)
        number += 1
    lines += _interpretation(comparison, number)
    lines += [
        f"## {number + 1}. Reproduction", "",
        "```bash",
        "poetry install --extras \"bayes report\"",
        "poetry run eu-okun predictive-fit --data-dir snapshots/eurostat-2026-10-15 "
        "--output-dir results/predictive",
        command,
        "```", "",
        f"Snapshot manifest SHA-256 `{comparison.snapshot.get('manifest_sha256', '')}`. "
        "The refits run in the separate *EU Okun predictive comparison* workflow; ordinary "
        "pull-request checks never sample. `predictions.csv` holds every scored prediction "
        "of every model, `refits.json` the diagnostics of every refit.", "",
    ]
    return "\n".join(lines)


def _models_table() -> str:
    rows = [
        ("bayes_primary", "Partial pooling, Student-t AR(1) errors, common factor (integrated "
         "out for the target quarter)"),
        ("bayes_no_common_factor", "As primary, without the common factor"),
        ("bayes_gaussian_iid", "Partial pooling, Gaussian i.i.d. errors"),
        ("ols_dl", "OLS per country on 1, g[t], g[t-1], g[t-2]"),
        ("ardl", "OLS per country on 1, y[t-1], g[t], g[t-1], g[t-2]"),
        ("zero_change", "No change; scale of past changes"),
        ("historical_mean", "Mean of past changes"),
        ("unemployment_ar1", "OLS per country on 1, y[t-1]"),
    ]
    out = ["| Model | Name in tables | Uses GDP growth in t | Predictive distribution |",
           "| --- | --- | --- | --- |"]
    for model, text in rows:
        gdp = "yes" if model in ("bayes_primary", "bayes_no_common_factor",
                                 "bayes_gaussian_iid", "ols_dl", "ardl") else "no"
        out.append(f"| `{model}` | {_label(model)} | {gdp} | {text} |")
    return "\n".join(out)


def _family_reading(comparison: Comparison) -> list[str]:
    """Plain statements of what each question's comparisons show (from the verdicts)."""
    lines = []
    for evaluation in EVALUATION_SETS:
        q1 = _test(comparison.family, evaluation, "bayes_primary", "ols_dl")
        q2a = _test(comparison.family, evaluation, "bayes_gaussian_iid", "ols_dl")
        q2b = _test(comparison.family, evaluation, "bayes_primary", "bayes_gaussian_iid")
        q3 = _test(comparison.family, evaluation, "bayes_primary", "zero_change")
        sentences = [
            f"Hierarchical model against classical regressions: {_pct(q1.ratio)} mean CRPS, "
            f"{q1.verdict}.",
            f"Pooling alone: {_pct(q2a.ratio)}, {q2a.verdict}; error model: "
            f"{_pct(q2b.ratio)}, {q2b.verdict}.",
            f"Okun relation against no change: {_pct(q3.ratio)}, {q3.verdict}.",
        ]
        lines += [f"*{evaluation}.* " + " ".join(sentences), ""]
    return lines


def _dominance_text(comparison: Comparison) -> list[str]:
    lines = []
    for evaluation in EVALUATION_SETS:
        pdata = comparison.pdata[evaluation]
        d = quarterly_differentials(comparison.losses(evaluation, "bayes_primary"),
                                    comparison.losses(evaluation, "ols_dl"), pdata.scored)
        order = np.argsort(np.abs(d))[::-1]
        top = [f"{pdata.target_quarters[j]} ({d[j]:+.4f})" for j in order[:2]]
        better = int(np.sum(d < 0))
        lines.append(f"- {evaluation}, Bayes primary minus OLS distributed lag: negative in "
                     f"{better} of {d.size} quarters; largest absolute differentials "
                     f"{' and '.join(top)}.")
    return lines + [""]


def _calibration_text(comparison: Comparison) -> list[str]:
    """Coverage ranges and the direction of systematic errors, per evaluation set."""
    lines = []
    for evaluation in EVALUATION_SETS:
        cover = [comparison.coverage(evaluation, m) for m in MODELS]
        pit = {m: float(np.mean([p.pit for p in comparison.predictions[(evaluation, m)]]))
               for m in MODELS}
        low, high = min(pit.values()), max(pit.values())
        if high < 0.5:
            direction = ("every model's mean PIT is below 0.5: outcomes tended to fall below "
                         "the predictions (unemployment fell faster than predicted)")
        elif low > 0.5:
            direction = ("every model's mean PIT is above 0.5: outcomes tended to lie above "
                         "the predictions (unemployment fell less or rose more than predicted)")
        else:
            direction = "mean PIT values lie on both sides of 0.5"
        lines.append(
            f"- {evaluation}: across the eight models the central 50% intervals cover "
            f"{min(c[0] for c in cover):.0%}-{max(c[0] for c in cover):.0%} of outcomes and "
            f"the 90% intervals {min(c[1] for c in cover):.0%}-{max(c[1] for c in cover):.0%}"
            f"; mean PIT {low:.2f}-{high:.2f}, so {direction}.")
    return lines + [""]


def _country_section(comparison: Comparison) -> list[str]:
    lines = ["## 7. Countries (descriptive)", "",
             "Mean CRPS by country over its scored target quarters, alphabetical. Not a "
             "ranking, and no per-country significance is claimed: each country has at most "
             "16 target quarters.", ""]
    countries = sorted(comparison.pdata["E1"].data.countries)
    header = ["Country"]
    for evaluation in EVALUATION_SETS:
        header += [f"{evaluation} Bayes primary", f"{evaluation} OLS-DL",
                   f"{evaluation} zero change"]
    rows = []
    better = {e: 0 for e in EVALUATION_SETS}
    counted = {e: 0 for e in EVALUATION_SETS}
    for geo in countries:
        row: list[Any] = [geo]
        for evaluation in EVALUATION_SETS:
            pdata = comparison.pdata[evaluation]
            i = list(pdata.data.countries).index(geo)
            if not pdata.scored[i].any():
                row += ["not scored"] * 3
                continue
            values = {m: float(np.nanmean(comparison.losses(evaluation, m)[i]))
                      for m in ("bayes_primary", "ols_dl", "zero_change")}
            row += [f"{values['bayes_primary']:.3f}", f"{values['ols_dl']:.3f}",
                    f"{values['zero_change']:.3f}"]
            counted[evaluation] += 1
            better[evaluation] += values["bayes_primary"] < values["ols_dl"]
        rows.append(row)
    lines += _table(header, rows)
    lines += ["Bayes primary has a lower mean CRPS than the OLS distributed lag in "
              + "; ".join(f"{better[e]} of {counted[e]} countries in {e}"
                          for e in EVALUATION_SETS) + ".", ""]
    return lines


def _replication_flips(comparison: Comparison) -> list[str]:
    """Family comparisons whose verdict differs in the replication run."""
    rep_run = comparison.replication
    assert rep_run is not None
    out = []
    for t in comparison.family:
        _, p_holm, verdict = rep_run.family[(t.evaluation, t.model_a, t.model_b)]
        if verdict != t.verdict:
            out.append(f"{t.evaluation} {_label(t.model_a)} vs {_label(t.model_b)} (Holm p "
                       f"{_p(t.p_holm)} here, {_p(p_holm)} there)")
    return out


def _replication_section(comparison: Comparison) -> list[str]:
    rep_run = comparison.replication
    assert rep_run is not None
    lines = ["## 9. Replication on a second platform", "",
             f"{rep_run.label}: the same snapshot, lock file, seeds, sampler settings and "
             "single retry. Its tables are in `replication-linux/`. MCMC output differs "
             "between platforms in floating-point detail, so borderline convergence "
             "decisions and borderline p-values can differ.", ""]
    rows = []
    for t in comparison.family:
        ratio, p_holm, verdict = rep_run.family[(t.evaluation, t.model_a, t.model_b)]
        rows.append([t.evaluation, _label(t.model_a), _label(t.model_b), _pct(t.ratio),
                     _p(t.p_holm), t.verdict, _pct(ratio), _p(p_holm), verdict])
    lines += _table(["Set", "A", "B", "A vs B (this run)", "Holm p", "Verdict",
                     "A vs B (replication)", "Holm p", "Verdict"], rows, "lllrrlrrl")
    here = {(p.evaluation, p.model, p.geo, p.quarter): p.crps
            for (_, model), predictions in comparison.predictions.items()
            if model in BAYES_MODELS for p in predictions}
    gaps = [abs(here[k] - v) for k, v in rep_run.bayes_crps.items() if k in here]
    failed_here = "; ".join(f"{k}: {', '.join(v)}" for k, v in comparison.failed_quarters.items()
                            if v) or "none"
    failed_there = "; ".join(f"{k}: {', '.join(v)}" for k, v in rep_run.failed.items()
                             if v) or "none"
    flips = _replication_flips(comparison)
    lines += [
        f"- Bayesian predictions: the CRPS of the {len(gaps)} Bayesian cells differs by "
        f"{np.mean(gaps):.4f} on average and at most {max(gaps):.4f} between the runs.",
        f"- Target quarters with a failed refit: this run {failed_here}; replication "
        f"{failed_there}.",
        "- Verdicts: " + ("identical for all ten comparisons." if not flips else
                          f"identical except {'; '.join(flips)}. A verdict that changes "
                          "between runs rests on a p-value at the threshold and is not "
                          "robust."),
        "",
    ]
    return lines


def _interpretation(comparison: Comparison, number: int = 9) -> list[str]:
    lines = [f"## {number}. Interpretation and limitations", ""]
    lines += [
        "- **What is tested.** Conditional predictive accuracy given the quarter's GDP "
        "growth, on revised data. It is neither a real-time forecast nor a causal test: a "
        "model that predicts better is not shown to have structural coefficients.",
        "- **Power.** 16 target quarters per set; averaging within quarters guards against "
        "cross-country dependence but leaves little information. Non-rejection is not "
        "evidence of equal accuracy.",
        "- **Pseudo-out-of-sample.** The model structure and priors were chosen with the "
        "full-sample classical results in view (BAYES_PROTOCOL.md), so E1 is not a blind "
        "test of the model choice; E2 lies after the period the model was designed on.",
        "- **The common factor** is unknown in a target quarter and integrated out. The "
        "posterior predictive checks showed it to be autocorrelated and correlated with "
        "average GDP growth (docs/HIERARCHICAL_BAYES.md); a dynamic factor could predict "
        "it, which is a separate milestone on the roadmap.",
        "- **Not covered:** joint (multivariate) scores, real-time vintages, time-varying "
        "coefficients.",
        "",
    ]
    return lines


# ----------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------


def write_figures(comparison: Comparison, directory: Path, source: str) -> None:
    """CRPS differentials, quarterly differentials and PIT histograms."""
    from eu_okun.review_figures import AQUA, BLUE, INK2, MUTED, ORANGE, SURFACE, _finish, _pyplot

    plt = _pyplot()
    from matplotlib.lines import Line2D

    colour = {m: BLUE for m in BAYES_MODELS}
    colour.update({"ardl": ORANGE, "zero_change": AQUA, "historical_mean": AQUA,
                   "unemployment_ar1": AQUA})
    others = [m for m in MODELS if m != REFERENCE]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=True)
    ys = np.arange(len(others))[::-1]
    for ax, evaluation in zip(axes, EVALUATION_SETS, strict=True):
        for y, model in zip(ys, others, strict=True):
            t = _test(comparison.descriptive, evaluation, model, REFERENCE)
            low, high = _interval(t)
            ax.plot([low, high], [y, y], color=colour[model], lw=1.8, solid_capstyle="round")
            ax.plot([t.mean_difference], [y], marker="o", ms=6.5, color=colour[model],
                    mec=SURFACE, mew=1.2, lw=0)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_title(f"{evaluation}, {EVALUATION_SETS[evaluation].label}")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([_label(m) for m in others], fontsize=8.5)
    fig.supxlabel("Mean CRPS minus that of the OLS distributed lag (pp); 95% interval, "
                  "unadjusted. Left of zero: better.", fontsize=9, color=INK2)
    fig.legend(handles=[
        Line2D([], [], marker="o", color=BLUE, lw=1.8, ms=6, label="Hierarchical Bayes"),
        Line2D([], [], marker="o", color=ORANGE, lw=1.8, ms=6,
               label="Classical, uses GDP growth in t"),
        Line2D([], [], marker="o", color=AQUA, lw=1.8, ms=6,
               label="Benchmark without GDP growth in t"),
    ], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.09))
    fig.subplots_adjust(wspace=0.08, bottom=0.2)
    _finish(plt, fig, directory / "crps_differentials.png", source, credit_y=-0.15)

    pairs = (("bayes_primary", "ols_dl", BLUE), ("bayes_gaussian_iid", "ols_dl", AQUA),
             ("bayes_primary", "zero_change", ORANGE))
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), sharey=True)
    for ax, evaluation in zip(axes, EVALUATION_SETS, strict=True):
        pdata = comparison.pdata[evaluation]
        x = np.arange(len(pdata.targets))
        for a, b, c in pairs:
            d = quarterly_differentials(comparison.losses(evaluation, a),
                                        comparison.losses(evaluation, b), pdata.scored)
            ax.plot(x, d, color=c, lw=2, marker="o", ms=4.5, mec=SURFACE, mew=1)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks(x[::2])
        ax.set_xticklabels(pdata.target_quarters[::2], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{evaluation}, {EVALUATION_SETS[evaluation].label}")
    axes[0].set_ylabel("Cross-country mean CRPS difference (pp)")
    fig.legend(handles=[Line2D([], [], color=c, lw=2, marker="o", ms=5,
                               label=f"{_label(a)} minus {_label(b)}") for a, b, c in pairs],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), fontsize=8.5)
    fig.subplots_adjust(wspace=0.06, bottom=0.27)
    _finish(plt, fig, directory / "quarterly_differentials.png", source, credit_y=-0.16)

    shown = ("bayes_primary", "bayes_gaussian_iid", "ols_dl", "zero_change")
    fig, axes = plt.subplots(2, len(shown), figsize=(12.0, 5.6), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 11)
    for row, evaluation in enumerate(EVALUATION_SETS):
        for col, model in enumerate(shown):
            ax = axes[row, col]
            pit = [p.pit for p in comparison.predictions[(evaluation, model)]]
            ax.hist(pit, bins=bins, density=True, color=BLUE if model in BAYES_MODELS else
                    (ORANGE if model == "ols_dl" else AQUA), edgecolor=SURFACE, linewidth=2)
            ax.axhline(1.0, color=INK2, lw=1, ls=(0, (3, 2)))
            ax.grid(axis="x", visible=False)
            if row == 0:
                ax.set_title(_label(model), fontsize=9.5)
            if col == 0:
                ax.set_ylabel(f"{evaluation}: density")
    fig.supxlabel("PIT value (predictive CDF at the outcome); dashed: uniform", fontsize=9,
                  color=INK2)
    fig.subplots_adjust(hspace=0.25, wspace=0.08)
    _finish(plt, fig, directory / "calibration.png", source, credit_y=-0.03)


# ----------------------------------------------------------------------------------
# Tables and entry point
# ----------------------------------------------------------------------------------


def write_tables(comparison: Comparison, directory: Path) -> None:
    """Predictions of every model, the tests, country scores and refit diagnostics."""
    directory.mkdir(parents=True, exist_ok=True)
    write_predictions(directory / "predictions.csv",
                      [p for rows in comparison.predictions.values() for p in rows])
    tests = ([("family", t) for t in comparison.family]
             + [("descriptive", t) for t in comparison.descriptive]
             + [("sensitivity", t) for t in comparison.sensitivity])
    with (directory / "comparisons.csv").open("w", encoding="utf-8", newline="") as f:
        names = list(asdict(comparison.family[0]))
        writer = csv.DictWriter(f, fieldnames=["kind", *names], lineterminator="\n")
        writer.writeheader()
        for kind, t in tests:
            writer.writerow({"kind": kind, **asdict(t)})
    with (directory / "quarterly_differentials.csv").open("w", encoding="utf-8",
                                                          newline="") as f:
        writer_rows = csv.writer(f, lineterminator="\n")
        writer_rows.writerow(["evaluation", "quarter", *[f"{a}-{b}" for a, b in FAMILY]])
        for evaluation, pdata in comparison.pdata.items():
            series = [quarterly_differentials(comparison.losses(evaluation, a),
                                              comparison.losses(evaluation, b), pdata.scored)
                      for a, b in FAMILY]
            for j, quarter in enumerate(pdata.target_quarters):
                writer_rows.writerow([evaluation, quarter, *(f"{s[j]:.6f}" for s in series)])
    refits = [{k: v for k, v in r.items() if k not in ("environment", "snapshot")}
              for r in comparison.refits]
    environments = sorted({json.dumps(r["environment"], sort_keys=True)
                           for r in comparison.refits})
    (directory / "refits.json").write_text(json.dumps(
        {"snapshot": comparison.snapshot, "environments": [json.loads(e) for e in environments],
         "refits": refits}, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_report(data_dir: Path, fits_dir: Path, output_dir: Path, report: Path | None = None,
                 figures: bool = True, replication_dir: Path | None = None) -> Comparison:
    """Build the comparison and write tables, figures and the Markdown report."""
    comparison = build_comparison(data_dir, fits_dir, replication_dir)
    report = report or output_dir / "PREDICTIVE_COMPARISON.md"
    write_tables(comparison, output_dir)
    if figures:
        source = ("Source: Eurostat, snapshot 2026-10-15; expanding-window refits "
                  "(PREDICTIVE_PROTOCOL.md). Conditional predictions, not causal.")
        write_figures(comparison, output_dir, source)
    assets = Path(os.path.relpath(output_dir, report.parent)).as_posix()
    command = (f"poetry run eu-okun predictive-report --data-dir {data_dir.as_posix()} "
               f"--fits-dir {fits_dir.as_posix()} --output-dir {output_dir.as_posix()} "
               f"--report {report.as_posix()}"
               + (f" --replication-dir {replication_dir.as_posix()}"
                  if replication_dir is not None else ""))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render(comparison, assets, command), encoding="utf-8", newline="\n")
    return comparison

