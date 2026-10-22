"""Diagnostics of the reported Bayesian results (``BAYES_DIAGNOSTICS_PROTOCOL.md``).

Run-to-run stability over further seeds, exact leave-one-out for the cells with high
Pareto k, influential cells and a calibrated prior scale. ``run_diagnostic_fits`` does
every fit into ``results/bayes-diagnostics``; ``write_diagnostics_report`` reads those
fits next to the registered ones and writes ``docs/BAYES_DIAGNOSTICS.md``. Nothing here
replaces a registered result.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun import bayes_checks, bayes_dynamics, bayes_regimes
from eu_okun.audit import audit_snapshot
from eu_okun.baseline import _read_panel
from eu_okun.bayes_artifacts import pointwise_log_likelihood_draws, prior_draws, simulate_outcomes
from eu_okun.bayes_checks import PARETO_K_MAX, PRIOR_CHECK_DRAWS, PRIOR_CHECK_SEED
from eu_okun.bayes_data import BayesData
from eu_okun.bayes_fit import (
    PROTOCOL_SETTINGS,
    FitResult,
    SamplerSettings,
    fit_specification,
    predictive_checks,
    run_specifications,
    save_fit,
)
from eu_okun.bayes_model import LOG_SIGMA_CENTRE, SPECS, ModelSpec, design_for
from eu_okun.bayes_report import LoadedFit, credible_interval, sign_class
from eu_okun.review import read_flags

# Section 2: the registered comparison sets and the fits they involve.
COMPARISON_SETS: dict[str, tuple[tuple[str, str], ...]] = {
    "9.2": bayes_checks.COMPARISONS,
    "11.4": bayes_dynamics.COMPARISONS,
    "12.5": bayes_regimes.COMPARISONS,
}
FITS: tuple[str, ...] = tuple(dict.fromkeys(
    name for pairs in COMPARISON_SETS.values() for pair in pairs for name in pair))
PERSISTENCE_SPECS: tuple[str, ...] = bayes_dynamics.COMPARED
ESTIMAND_SPECS: tuple[str, ...] = ("primary", "ardl_dynamic_factor")
# Section 3.
SEEDS: tuple[int, ...] = (20261030, 20261031, 20261032, 20261033)
RETRY_WARMUP = 3000
# Section 6.
LADDER: tuple[float, ...] = (1.0, 0.75, 0.5, 0.35, 0.25)
PRIOR_SHARE_MAX = 0.01
LARGE_CHANGE = 5.0
DECOMPOSITION: tuple[str, ...] = ("registered prior", "GDP terms set to zero",
                                  "Gaussian innovations", "sigma fixed at 0.3 pp",
                                  "Ireland's cells left out")
STAGES: tuple[str, ...] = ("seeds", "calibration", "exact", "influence")


# ----------------------------------------------------------------------------------
# Cells
# ----------------------------------------------------------------------------------


def cell_label(data: BayesData, i: int, j: int) -> str:
    """The label of the ``log_likelihood`` group: ``"<geo> <quarter>"``."""
    return f"{data.countries[i]} {data.quarters[j]}"


def cell_index(data: BayesData, label: str) -> tuple[int, int]:
    geo, quarter = label.split(" ")
    return data.countries.index(geo), data.quarters.index(quarter)


def leave_out(data: BayesData, labels: Sequence[str]) -> BayesData:
    """The same data with the likelihood terms of ``labels`` removed.

    Only the mask changes: each value stays in ``y``, so a later quarter's AR(1) or
    lagged-change term still conditions on it (protocol, section 4).
    """
    mask = data.mask.copy()
    for label in labels:
        i, j = cell_index(data, label)
        if not mask[i, j]:
            raise ValueError(f"{label} is not in the likelihood.")
        mask[i, j] = False
    return replace(data, mask=mask)


# ----------------------------------------------------------------------------------
# PSIS-LOO by cell
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pointwise:
    """PSIS-LOO of every cell of one fit."""

    spec: str
    labels: tuple[str, ...]
    elpd: np.ndarray
    pareto_k: np.ndarray

    def select(self, cells: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        index = {label: n for n, label in enumerate(self.labels)}
        rows = [index[c] for c in cells]
        return self.elpd[rows], self.pareto_k[rows]


def psis_pointwise(path: Path, spec: str) -> Pointwise:
    """PSIS-LOO by cell from the ``log_likelihood`` group of a NetCDF artifact.

    A cell's value depends only on its own log-likelihood draws and on the relative
    efficiency of the posterior, so a subset of cells has the same values as in
    ``bayes_dynamics.loo_on_cells``.
    """
    import xarray as xr
    from arviz_stats.loo.loo import loo

    with xr.open_datatree(path, engine="h5netcdf") as tree:
        labels = tuple(str(c) for c in tree["log_likelihood"].to_dataset()["cell"].to_numpy())
        # ArviZ's loo carries no annotations.
        result: Any = loo(tree, pointwise=True, var_name="y")  # type: ignore[no-untyped-call]
        return Pointwise(spec, labels, np.asarray(result.elpd_i, dtype=float).reshape(-1),
                         np.asarray(result.pareto_k, dtype=float).reshape(-1))


def comparison_cells(fits_dir: Path) -> dict[tuple[str, str, str], list[str]]:
    """The registered cell set of each comparison (protocol, section 2)."""
    labels = {name: bayes_dynamics.cell_labels(fits_dir / name / "posterior.nc")
              for name in FITS if (fits_dir / name / "posterior.nc").exists()}
    out: dict[tuple[str, str, str], list[str]] = {}
    dynamics = [labels[n] for n in bayes_dynamics.COMPARED if n in labels]
    shared = bayes_dynamics.common_cells(dynamics)
    for key, pairs in COMPARISON_SETS.items():
        for a, b in pairs:
            if a not in labels or b not in labels:
                continue
            if key == "9.2":
                if labels[a] != labels[b]:
                    raise ValueError(f"{a} and {b} were fitted on different cells.")
                out[(key, a, b)] = list(labels[a])
            elif key == "11.4":
                out[(key, a, b)] = shared
            else:
                out[(key, a, b)] = bayes_dynamics.common_cells([labels[a], labels[b]])
    return out


def cells_of_fit(cells: Mapping[tuple[str, str, str], Sequence[str]], name: str) -> set[str]:
    """Every cell that enters one of the fit's comparisons."""
    return {c for (_, a, b), labels in cells.items() if name in (a, b) for c in labels}


# ----------------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------------


def _inputs(data_dir: Path) -> tuple[Any, Any, dict[str, str]]:
    audit = audit_snapshot(data_dir)
    csv_path = data_dir / "processed" / "eu27_quarterly.csv"
    snapshot = {"manifest_sha256": audit["snapshot_manifest_sha256"],
                "processed_csv_sha256": audit["processed_csv_sha256"]}
    return _read_panel(csv_path), read_flags(csv_path), snapshot


def fit_with_retry(data: BayesData, spec: ModelSpec,
                   settings: SamplerSettings = PROTOCOL_SETTINGS) -> FitResult:
    """One fit, retried once with more warm-up if it fails the acceptance criteria."""
    fit = fit_specification(data, spec, settings)
    if not fit.diagnostics.accepted:
        fit = fit_specification(data, spec, replace(settings, warmup=RETRY_WARMUP))
    return fit


def loaded(fit: FitResult) -> LoadedFit:
    """A fit in memory in the form the report modules read."""
    summary = {"countries": list(fit.data.countries),
               "diagnostics": {"accepted": fit.diagnostics.accepted}}
    draws = {k: v.reshape(-1, *v.shape[2:]) for k, v in fit.samples.items()}
    return LoadedFit(fit.spec.name, summary, draws)


def estimand(fit: LoadedFit) -> dict[str, Any]:
    """The quoted estimand (section 2) and the credibly negative countries.

    ``mu_theta`` without a lagged change; the per-draw mean of ``theta_LR`` with one.
    """
    name, total = bayes_dynamics.total_association(fit)
    values = total.mean(axis=1) if name == "theta_LR" else fit.mu_theta()
    low, high = credible_interval(values)
    negative = [geo for i, geo in enumerate(fit.countries)
                if sign_class(total[:, i]) == "negative"]
    return {"name": "mean theta_LR" if name == "theta_LR" else "mu_theta",
            "median": float(np.median(values)), "low": low, "high": high,
            "credibly_negative": negative}


def exact_elpd(fit: FitResult, data: BayesData, label: str) -> float:
    """log of the mean predictive density of one cell over the draws of ``fit``."""
    i, j = cell_index(data, label)
    values = pointwise_log_likelihood_draws(fit.spec, data, fit.samples)[:, :, i, j].reshape(-1)
    top = float(values.max())
    return top + math.log(float(np.mean(np.exp(values - top))))


def _diagnostics(fit: FitResult) -> dict[str, Any]:
    d = fit.diagnostics
    return {"accepted": d.accepted, "warmup": fit.settings.warmup, "max_rhat": d.max_rhat,
            "min_ess": min(d.min_ess_bulk, d.min_ess_tail), "divergences": d.divergences,
            "runtime_seconds": fit.runtime_seconds}


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=1, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def exact_refit(data_dir: Path, name: str, label: str, target: Path) -> None:
    """Section 4: refit without one cell's term and score that cell exactly."""
    panel, flags, _ = _inputs(data_dir)
    spec = SPECS[name]
    data = design_for(spec, panel, flags)
    fit = fit_with_retry(leave_out(data, [label]), spec)
    _write_json(target, {"spec": name, "cell": label, "seed": fit.settings.seed,
                         **_diagnostics(fit), "elpd": exact_elpd(fit, data, label),
                         "estimand": estimand(loaded(fit))})


def joint_refit(data_dir: Path, name: str, labels: Sequence[str], directory: Path) -> None:
    """Section 5: refit without every refitted cell of the specification at once."""
    panel, flags, snapshot = _inputs(data_dir)
    spec = SPECS[name]
    fit = fit_with_retry(leave_out(design_for(spec, panel, flags), labels), spec)
    save_fit(fit, predictive_checks(fit), directory, snapshot, netcdf=False)
    _write_json(directory / "influence.json", {"spec": name, "cells": list(labels),
                                               **_diagnostics(fit),
                                               "estimand": estimand(loaded(fit))})


def calibrated_refit(data_dir: Path, name: str, scale: float, directory: Path) -> None:
    """Section 6 (c): refit with the calibrated prior scale."""
    panel, flags, snapshot = _inputs(data_dir)
    spec = replace(SPECS[name], prior_scale=scale)
    fit = fit_with_retry(design_for(spec, panel, flags), spec)
    save_fit(fit, predictive_checks(fit), directory, snapshot)


# ----------------------------------------------------------------------------------
# Prior calibration (section 6)
# ----------------------------------------------------------------------------------


def _prior_summary(data: BayesData, simulated: np.ndarray,
                   rows: np.ndarray | None = None) -> dict[str, float]:
    mask = data.mask if rows is None else data.mask & rows[:, None]
    cells = simulated[:, mask]
    return {"share_above_5pp": float(np.mean(np.abs(cells) > LARGE_CHANGE)),
            "median_largest_change": float(np.median(np.max(np.abs(cells), axis=1)))}


def decomposition(spec: ModelSpec, data: BayesData, draws: int = PRIOR_CHECK_DRAWS,
                  seed: int = PRIOR_CHECK_SEED) -> dict[str, dict[str, float]]:
    """The registered prior predictive check with one component neutralised at a time.

    Every variant uses the same prior draws and the same innovation stream.
    """
    prior = prior_draws(spec, data, draws, seed)

    def simulate(s: ModelSpec, values: Mapping[str, np.ndarray]) -> np.ndarray:
        return simulate_outcomes(s, data, values, np.random.default_rng(seed))

    full = simulate(spec, prior)
    no_gdp = {**prior, "theta": np.zeros_like(prior["theta"]), "b": np.zeros_like(prior["b"])}
    fixed = {**prior, "sigma": np.full_like(prior["sigma"], math.exp(LOG_SIGMA_CENTRE))}
    ireland = np.asarray([geo != "IE" for geo in data.countries])
    return {
        DECOMPOSITION[0]: _prior_summary(data, full),
        DECOMPOSITION[1]: _prior_summary(data, simulate(spec, no_gdp)),
        DECOMPOSITION[2]: _prior_summary(data, simulate(replace(spec, student_t=False), prior)),
        DECOMPOSITION[3]: _prior_summary(data, simulate(spec, fixed)),
        DECOMPOSITION[4]: _prior_summary(data, full, ireland),
    }


def calibration(data_dir: Path) -> dict[str, Any]:
    """Section 6 (a) and (b): the decomposition, the ladder and the chosen scales."""
    panel, flags, _ = _inputs(data_dir)
    out: dict[str, Any] = {"ladder": {}, "chosen": {}, "criterion_met": {}}
    for name in ESTIMAND_SPECS:
        spec = SPECS[name]
        data = design_for(spec, panel, flags)
        if name == "primary":
            out["decomposition"] = decomposition(spec, data)
        steps = []
        for scale in LADDER:
            check = bayes_checks.prior_predictive_check(replace(spec, prior_scale=scale), data)
            steps.append({"scale": scale, "share_above_5pp": check.share_above_5pp,
                          "median_largest_change": check.quantiles["max_abs_change"][1]})
        out["ladder"][name] = steps
        out["chosen"][name], out["criterion_met"][name] = choose_scale(steps)
    return out


def choose_scale(steps: Sequence[Mapping[str, float]]) -> tuple[float, bool]:
    """Section 6 (b): the widest passing step, or the last step if none passes."""
    for step in steps:
        if step["share_above_5pp"] <= PRIOR_SHARE_MAX:
            return step["scale"], True
    return steps[-1]["scale"], False


# ----------------------------------------------------------------------------------
# The fitting command
# ----------------------------------------------------------------------------------


def high_k_cells(fits_dir: Path, refits_dir: Path) -> dict[str, list[str]]:
    """Section 4: each fit's cells with Pareto k above 0.7 in the local run or any seed run.

    Only accepted fits count, and only cells that enter one of the fit's comparisons.
    """
    cells = comparison_cells(fits_dir)
    out: dict[str, list[str]] = {}
    for name in FITS:
        wanted = cells_of_fit(cells, name)
        labels = bayes_dynamics.cell_labels(fits_dir / name / "posterior.nc")
        runs = [fits_dir / name] + [refits_dir / f"seed-{s}" / name for s in SEEDS]
        flagged: set[str] = set()
        for run in runs:
            path = run / "posterior.nc"
            if not path.exists():
                raise FileNotFoundError(f"No posterior for {run}; run the seeds stage first.")
            summary = json.loads((run / "fit.json").read_text(encoding="utf-8"))
            if not summary["diagnostics"]["accepted"]:
                continue
            point = psis_pointwise(path, name)
            flagged |= {c for c, k in zip(point.labels, point.pareto_k, strict=True)
                        if k > PARETO_K_MAX and c in wanted}
        out[name] = sorted(flagged, key=labels.index)
    return out


def _tasks(stage: str, data_dir: Path, fits_dir: Path, refits_dir: Path,
           ) -> list[tuple[str, Path, Any]]:
    """(description, completion marker, callable) for one stage, in a fixed order."""
    tasks: list[tuple[str, Path, Any]] = []
    if stage == "seeds":
        for seed in SEEDS:
            for name in FITS:
                out = refits_dir / f"seed-{seed}"
                settings = replace(PROTOCOL_SETTINGS, seed=seed)
                tasks.append((f"{name} seed {seed}", out / name / "posterior.nc",
                              lambda o=out, n=name, s=settings: run_specifications(
                                  data_dir, o, [n], s, retry_warmup=RETRY_WARMUP)))
    elif stage == "calibration":
        path = refits_dir / "calibration.json"
        if not path.exists():
            _write_json(path, calibration(data_dir))
        chosen = json.loads(path.read_text(encoding="utf-8"))["chosen"]
        for name in ESTIMAND_SPECS:
            target = refits_dir / "calibrated" / name
            tasks.append((f"{name} at prior scale {chosen[name]}", target / "posterior.nc",
                          lambda n=name, s=chosen[name], t=target: calibrated_refit(
                              data_dir, n, s, t)))
    else:
        path = refits_dir / "cells.json"
        if not path.exists():
            _write_json(path, high_k_cells(fits_dir, refits_dir))
        cells: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8"))
        if stage == "exact":
            for name in FITS:
                for label in cells[name]:
                    target = refits_dir / "exact" / name / f"{label.replace(' ', '_')}.json"
                    tasks.append((f"{name} without {label}", target,
                                  lambda n=name, c=label, t=target: exact_refit(
                                      data_dir, n, c, t)))
        else:
            for name in ESTIMAND_SPECS:
                target = refits_dir / "influence" / name
                tasks.append((f"{name} without {len(cells[name])} cells",
                              target / "influence.json",
                              lambda n=name, t=target: joint_refit(data_dir, n, cells[n], t)))
    return tasks


def run_diagnostic_fits(data_dir: Path, fits_dir: Path, refits_dir: Path,
                        stages: Sequence[str] = STAGES, shard: tuple[int, int] = (0, 1),
                        ) -> list[str]:
    """Run the fits of the named stages; skip those already done (resumable).

    With ``shard = (k, n)`` only every n-th task from the k-th is run, so that n
    processes can share a stage. The exact and influence stages need every seed fit.
    """
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stages: {', '.join(unknown)}.")
    k, n = shard
    if not 0 <= k < n:
        raise ValueError("The shard must be k/n with 0 <= k < n.")
    done = []
    for stage in stages:
        for number, (description, marker, task) in enumerate(
                _tasks(stage, data_dir, fits_dir, refits_dir)):
            if number % n != k or marker.exists():
                continue
            started = time.perf_counter()
            task()
            done.append(f"{stage}: {description} ({time.perf_counter() - started:.0f} s)")
            print(done[-1], flush=True)
    return done


# ----------------------------------------------------------------------------------
# Results of one run
# ----------------------------------------------------------------------------------

LOCAL = "local"
LINUX = "Linux"
REPORTED_SUMMARIES = {"9.2": Path("bayes-checks") / "model_checks.json",
                      "11.4": Path("bayes-dynamics") / "dynamics.json",
                      "12.5": Path("bayes-regimes") / "regimes.json"}


def seed_label(seed: int) -> str:
    return f"seed {seed}"


def _comparison(a: str, b: str, ea: np.ndarray, eb: np.ndarray, ka: np.ndarray,
                kb: np.ndarray) -> dict[str, Any]:
    """The registered verdict (BAYES_PROTOCOL.md, section 9.2) on given pointwise values."""
    def result(name: str, e: np.ndarray, k: np.ndarray) -> bayes_checks.LooResult:
        return bayes_checks.LooResult(name, float(e.sum()), math.nan, math.nan, int(e.size),
                                      int(np.sum(k > PARETO_K_MAX)), tuple(e.tolist()))

    c = bayes_checks.compare_loo(result(a, ea, ka), result(b, eb, kb))
    return {"elpd_diff": c.elpd_diff, "se_diff": c.se_diff, "verdict": c.verdict,
            "reliable": c.reliable}


def _load(directory: Path) -> LoadedFit:
    from eu_okun.bayes_report import load_fit

    return load_fit(directory)


def run_results(run_dir: Path, cells: Mapping[tuple[str, str, str], Sequence[str]],
                ) -> dict[str, Any]:
    """The quantities of section 3 for one run (a directory with one folder per fit)."""
    accepted: dict[str, bool] = {}
    pointwise: dict[str, Pointwise] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for name in FITS:
        summary = json.loads((run_dir / name / "fit.json").read_text(encoding="utf-8"))
        summaries[name] = summary
        accepted[name] = bool(summary["diagnostics"]["accepted"])
        if accepted[name]:
            pointwise[name] = psis_pointwise(run_dir / name / "posterior.nc", name)
    comparisons: dict[str, Any] = {}
    for (key, a, b), labels in cells.items():
        if a in pointwise and b in pointwise:
            ea, ka = pointwise[a].select(labels)
            eb, kb = pointwise[b].select(labels)
            comparisons[f"{key} {a} {b}"] = _comparison(a, b, ea, eb, ka, kb)
        else:
            comparisons[f"{key} {a} {b}"] = None
    persistence: dict[str, Any] = {}
    for name in PERSISTENCE_SPECS:
        p = bayes_dynamics.persistence(summaries[name], name)
        persistence[name] = {"p": p.p_value, "fixed": p.fixed} if accepted[name] else None
    estimands = {name: estimand(_load(run_dir / name)) if accepted[name] else None
                 for name in ESTIMAND_SPECS}
    return {"accepted": accepted, "comparisons": comparisons, "persistence": persistence,
            "estimands": estimands,
            "pareto_high": {n: int(np.sum(p.pareto_k > PARETO_K_MAX))
                            for n, p in pointwise.items()},
            "pointwise": pointwise}


def linux_results(docs_dir: Path) -> dict[str, Any]:
    """The same quantities from the committed Linux replications, where they were kept."""
    comparisons: dict[str, Any] = {}
    for key, relative in REPORTED_SUMMARIES.items():
        path = docs_dir / relative.parent / "replication-linux" / relative.name
        found = json.loads(path.read_text(encoding="utf-8"))["comparisons"]
        for a, b in COMPARISON_SETS[key]:
            row = next((c for c in found if (c["model_a"], c["model_b"]) == (a, b)), None)
            comparisons[f"{key} {a} {b}"] = None if row is None else {
                k: row[k] for k in ("elpd_diff", "se_diff", "verdict", "reliable")}
    dynamics = json.loads((docs_dir / "bayes-dynamics" / "replication-linux" / "dynamics.json")
                          .read_text(encoding="utf-8"))["specifications"]
    persistence = {n: {"p": dynamics[n]["lag1_p_value"], "fixed": dynamics[n]["persistence_fixed"]}
                   for n in PERSISTENCE_SPECS}
    negative = {n: list(dynamics[n]["credibly_negative"]) for n in ESTIMAND_SPECS}
    return {"comparisons": comparisons, "persistence": persistence, "negative": negative}


def check_local_run(local: Mapping[str, Any], docs_dir: Path) -> None:
    """The local run must reproduce the committed comparisons (the registered fits)."""
    for key, relative in REPORTED_SUMMARIES.items():
        found = json.loads((docs_dir / relative).read_text(encoding="utf-8"))["comparisons"]
        for row in found:
            ours = local["comparisons"].get(f"{key} {row['model_a']} {row['model_b']}")
            same = ours is not None and all(
                math.isclose(ours[k], row[k], rel_tol=1e-9, abs_tol=1e-6)
                for k in ("elpd_diff", "se_diff"))
            if not same:
                raise ValueError(f"The fits do not reproduce the committed comparison "
                                 f"{row['model_a']} against {row['model_b']} ({key}).")


# ----------------------------------------------------------------------------------
# The study
# ----------------------------------------------------------------------------------


def _set_change(registered: Sequence[str], other: Sequence[str]) -> dict[str, list[str]]:
    return {"added": sorted(set(other) - set(registered)),
            "dropped": sorted(set(registered) - set(other))}


def _k_of(point: Pointwise, label: str) -> float:
    return float(point.pareto_k[point.labels.index(label)])


def _corrected(a: str, b: str, cell_set: Sequence[str], local: Mapping[str, Any],
               exact: Mapping[tuple[str, str], Mapping[str, Any]],
               ) -> tuple[dict[str, Any] | None, int]:
    """Section 4: the comparison with exact values in place of PSIS for refitted cells."""
    values = {}
    replaced, resolved = 0, True
    for spec in (a, b):
        e, k = local["pointwise"][spec].select(cell_set)
        e = e.copy()
        for n, label in enumerate(cell_set):
            record = exact.get((spec, label))
            if record is not None:
                replaced += 1
                resolved &= bool(record["accepted"])
                e[n] = record["elpd"]
        values[spec] = (e, k)
    if not resolved:
        return None, replaced
    return _comparison(a, b, values[a][0], values[b][0], values[a][1], values[b][1]), replaced


def _rho(fit: LoadedFit) -> list[float] | None:
    """Median and 90% interval of the factor's AR(1) coefficient, if it has one."""
    if "rho_f" not in fit.draws:
        return None
    low, high = credible_interval(fit.draws["rho_f"])
    return [float(np.median(fit.draws["rho_f"])), low, high]


def build_study(fits_dir: Path, refits_dir: Path, data_dir: Path, docs_dir: Path,
                ) -> dict[str, Any]:
    """Every result of sections 3-6, as one JSON-serialisable document."""
    cells = comparison_cells(fits_dir)
    local = run_results(fits_dir, cells)
    check_local_run(local, docs_dir)
    runs: dict[str, Any] = {LOCAL: local, LINUX: linux_results(docs_dir)}
    runs.update({seed_label(s): run_results(refits_dir / f"seed-{s}", cells) for s in SEEDS})
    labels = list(runs)
    sampled = [r for r in labels if r != LINUX]
    panel, flags, _ = _inputs(data_dir)
    refitted: dict[str, list[str]] = json.loads((refits_dir / "cells.json").read_text(
        encoding="utf-8"))
    exact: dict[tuple[str, str], dict[str, Any]] = {}
    for name in FITS:
        for label in refitted[name]:
            path = refits_dir / "exact" / name / f"{label.replace(' ', '_')}.json"
            exact[(name, label)] = json.loads(path.read_text(encoding="utf-8"))

    # Section 4: exact values by cell.
    exact_rows = []
    for (name, label), record in exact.items():
        point = local["pointwise"][name]
        k_runs = [_k_of(runs[r]["pointwise"][name], label) for r in sampled
                  if name in runs[r]["pointwise"]]
        exact_rows.append({
            "spec": name, "cell": label, "k_local": _k_of(point, label),
            "k_max": max(k_runs), "runs_flagged": sum(k > PARETO_K_MAX for k in k_runs),
            "psis": float(point.elpd[point.labels.index(label)]), "exact": record["elpd"],
            "accepted": record["accepted"], "warmup": record["warmup"],
            "max_rhat": record["max_rhat"]})

    # Sections 3 and 4: comparisons.
    comparisons = []
    for (key, a, b), cell_set in cells.items():
        by_run = {r: runs[r]["comparisons"].get(f"{key} {a} {b}") for r in labels}
        present = {r: v for r, v in by_run.items() if v is not None}
        ratios = [v["elpd_diff"] / v["se_diff"] for v in present.values()]
        corrected, replaced = _corrected(a, b, cell_set, local, exact)
        stable = len({v["verdict"] for v in present.values()}) == 1
        changes = corrected is not None and corrected["verdict"] != present[LOCAL]["verdict"]
        comparisons.append({
            "set": key, "a": a, "b": b, "cells": len(cell_set), "runs": by_run,
            "stable": stable, "ratio_min": min(ratios), "ratio_max": max(ratios),
            "unreliable_runs": [r for r, v in present.items() if not v["reliable"]],
            "exact_cells": replaced, "corrected": corrected, "resolved": corrected is not None,
            "verdict_changes": changes,
            "established": stable and corrected is not None and not changes})

    persistence = []
    for name in PERSISTENCE_SPECS:
        by_run = {r: runs[r]["persistence"].get(name) for r in labels}
        fixed = {v["fixed"] for v in by_run.values() if v is not None}
        persistence.append({"spec": name, "runs": by_run, "stable": len(fixed) == 1})

    estimands = []
    for name in ESTIMAND_SPECS:
        registered = local["estimands"][name]
        found = {r: runs[r]["estimands"][name] for r in sampled
                 if runs[r]["estimands"][name] is not None}
        negative = {r: len(e["credibly_negative"]) for r, e in found.items()}
        negative[LINUX] = len(runs[LINUX]["negative"][name])
        estimands.append({"spec": name, "estimand": registered["name"],
                          "registered": {k: registered[k] for k in ("median", "low", "high")},
                          "medians": {r: e["median"] for r, e in found.items()},
                          "negative_counts": {r: negative[r] for r in labels if r in negative},
                          "negative_sets_differ": [r for r, e in found.items() if set(
                              e["credibly_negative"]) != set(registered["credibly_negative"])]})

    pareto = [{"spec": name, "cells": len(local["pointwise"][name].labels),
               "runs": {r: runs[r]["pareto_high"].get(name) for r in sampled}}
              for name in FITS]

    # Section 5: influence.
    influence: dict[str, Any] = {}
    inventory = []
    for name in ESTIMAND_SPECS:
        registered = local["estimands"][name]
        data = design_for(SPECS[name], panel, flags)
        single = []
        for label in refitted[name]:
            record = exact[(name, label)]
            e = record["estimand"]
            single.append({"cell": label, "accepted": record["accepted"],
                           "median": e["median"], "shift": e["median"] - registered["median"],
                           **_set_change(registered["credibly_negative"],
                                         e["credibly_negative"])})
            i, j = cell_index(data, label)
            flagged = sum(1 for other in FITS if label in local["pointwise"][other].labels
                          and _k_of(local["pointwise"][other], label) > PARETO_K_MAX)
            inventory.append({"spec": name, "cell": label, "y": float(data.y[i, j]),
                              "k_local": _k_of(local["pointwise"][name], label),
                              "fits_flagged": flagged})
        directory = refits_dir / "influence" / name
        joint_doc = json.loads((directory / "influence.json").read_text(encoding="utf-8"))
        joint_fit = json.loads((directory / "fit.json").read_text(encoding="utf-8"))
        je = joint_doc["estimand"]
        joint = {"cells": len(joint_doc["cells"]), "accepted": joint_doc["accepted"],
                 "median": je["median"], "low": je["low"], "high": je["high"],
                 "lag1_p": joint_fit["predictive_check"]["p_value"]["lag1_autocorrelation"],
                 **_set_change(registered["credibly_negative"], je["credibly_negative"])}
        rows = [r for r in [*single, joint] if r["accepted"]]
        outside = [r for r in rows if not registered["low"] <= r["median"] <= registered["high"]]
        changed = [r for r in rows if r["added"] or r["dropped"]]
        influence[name] = {
            "registered": {k: registered[k] for k in ("median", "low", "high",
                                                      "credibly_negative")},
            "single": single, "joint": joint, "max_shift": max(abs(r["shift"]) for r in single)
            if single else 0.0, "outside_interval": len(outside), "set_changes": len(changed),
            "sensitive": bool(outside or changed)}

    # Section 6: prior calibration.
    calibration_doc = json.loads((refits_dir / "calibration.json").read_text(encoding="utf-8"))
    primary_data = design_for(SPECS["primary"], panel, flags)
    observed = {"share_above_5pp": float(np.mean(np.abs(primary_data.y[primary_data.mask])
                                                 > LARGE_CHANGE)),
                "largest_change": float(np.max(np.abs(primary_data.y[primary_data.mask])))}
    calibrated: dict[str, Any] = {}
    points: dict[str, Pointwise] = {}
    for name in ESTIMAND_SPECS:
        directory = refits_dir / "calibrated" / name
        summary = json.loads((directory / "fit.json").read_text(encoding="utf-8"))
        fit = _load(directory)
        e = estimand(fit)
        registered = local["estimands"][name]
        out_of_interval = not registered["low"] <= e["median"] <= registered["high"]
        change = _set_change(registered["credibly_negative"], e["credibly_negative"])
        p = bayes_dynamics.persistence(summary, name)
        calibrated[name] = {
            "scale": summary["spec"]["prior_scale"], "accepted": fit.accepted,
            "warmup": summary["settings"]["warmup"], "median": e["median"], "low": e["low"],
            "high": e["high"], "registered": {k: registered[k] for k in ("median", "low",
                                                                         "high")},
            "negative": len(e["credibly_negative"]),
            "negative_registered": len(registered["credibly_negative"]), **change,
            "lag1_p": p.p_value, "lag1_p_registered": local["persistence"][name]["p"],
            "rho_f": _rho(fit), "rho_f_registered": _rho(_load(fits_dir / name)),
            "prior_sensitive": bool(out_of_interval or change["added"] or change["dropped"])}
        if fit.accepted:
            points[name] = psis_pointwise(directory / "posterior.nc", name)
    loo = None
    if len(points) == len(ESTIMAND_SPECS):
        cell_set = cells[("11.4", "ardl_dynamic_factor", "primary")]
        ea, ka = points["ardl_dynamic_factor"].select(cell_set)
        eb, kb = points["primary"].select(cell_set)
        loo = _comparison("ardl_dynamic_factor", "primary", ea, eb, ka, kb)

    return {"runs": labels, "comparisons": comparisons, "persistence": persistence,
            "estimands": estimands, "pareto": pareto, "exact": exact_rows,
            "influence": influence, "inventory": inventory,
            "calibration": {**calibration_doc, "fits": calibrated, "loo": loo,
                            "observed": observed}}


# ----------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------

_table = bayes_dynamics._table
_names = bayes_dynamics._names
_code = bayes_dynamics._code


def _short(verdict: str, a: str, b: str) -> str:
    """A for "A better supported", B for B, = for not distinguishable."""
    if verdict == f"{a} better supported":
        return "A"
    if verdict == f"{b} better supported":
        return "B"
    return "="


def _diff(row: Mapping[str, Any] | None, a: str = "", b: str = "") -> str:
    if row is None:
        return "n/a"
    mark = f" {_short(row['verdict'], a, b)}" if a else ""
    return f"{row['elpd_diff']:+.1f} ({row['se_diff']:.1f}){mark}"


def _pair(c: Mapping[str, Any]) -> str:
    return f"{_code(c['a'])} against {_code(c['b'])}"


def _interval(row: Mapping[str, Any]) -> str:
    return f"{row['median']:+.3f} [{row['low']:+.3f}, {row['high']:+.3f}]"


def _change(row: Mapping[str, Any]) -> str:
    parts = []
    if row["added"]:
        parts.append("adds " + _names(row["added"]))
    if row["dropped"]:
        parts.append("drops " + _names(row["dropped"]))
    return "; ".join(parts) or "none"


def summary_lines(study: Mapping[str, Any]) -> list[str]:
    comparisons = study["comparisons"]
    runs = study["runs"]
    established = [c for c in comparisons if c["established"]]
    lines = [f"- **PSIS-LOO.** {len(established)} of the {len(comparisons)} registered "
             f"comparisons are established: the same verdict in all {len(runs)} runs and "
             f"unchanged by exact leave-one-out."]
    for c in comparisons:
        if c["established"]:
            continue
        reasons = []
        if not c["stable"]:
            reasons.append(f"its verdict differs between runs (difference / SE from "
                           f"{c['ratio_min']:.2f} to {c['ratio_max']:.2f})")
        if c["verdict_changes"]:
            reasons.append(f"exact leave-one-out changes its verdict to "
                           f"'{c['corrected']['verdict']}'")
        if not c["resolved"]:
            reasons.append("a refit it needs was not accepted")
        lines.append(f"  - Not established: {_pair(c)} (section {c['set']}), because "
                     + " and ".join(reasons) + ".")
    exact = study["exact"]
    gap = max(abs(r["exact"] - r["psis"]) for r in exact) if exact else 0.0
    changed = [c for c in comparisons if c["verdict_changes"]]
    lines.append(f"- **Exact leave-one-out.** {len(exact)} cells were refitted across "
                 f"{len({r['spec'] for r in exact})} fits. The largest gap between the exact "
                 f"and PSIS values of a cell is {gap:.2f}. Verdicts changed: "
                 f"{_names([_pair(c) for c in changed])}.")
    unreliable = [c for c in comparisons if c["unreliable_runs"]]
    if unreliable:
        lines.append("- **Reliability labels.** The registered label (more than 1% of cells "
                     "with Pareto k > 0.7) marks a comparison unreliable in some runs only: "
                     + "; ".join(f"{_pair(c)} in {_names(c['unreliable_runs'])}"
                                 for c in unreliable) + ".")
    for p in study["persistence"]:
        found = [v for v in p["runs"].values() if v is not None]
        fixed = sum(v["fixed"] for v in found)
        ps = [v["p"] for v in found]
        state = "stable" if p["stable"] else "not stable"
        lines.append(f"- **Persistence, {_code(p['spec'])}.** Fixed in {fixed} of "
                     f"{len(found)} runs (p from {min(ps):.3f} to {max(ps):.3f}); {state}.")
    for e in study["estimands"]:
        medians = list(e["medians"].values())
        counts = list(e["negative_counts"].values())
        lines.append(f"- **{e['estimand']} in {_code(e['spec'])}.** Registered "
                     f"{_interval(e['registered'])}; the median ranges from {min(medians):+.3f}"
                     f" to {max(medians):+.3f} over the sampled runs. Credibly negative "
                     f"countries: {min(counts)} to {max(counts)} over the runs.")
    for name, inf in study["influence"].items():
        joint = inf["joint"]
        verdict = "sensitive" if inf["sensitive"] else "not sensitive"
        lines.append(f"- **Influence, {_code(name)}.** {len(inf['single'])} cells refitted one "
                     f"at a time move the median by at most {inf['max_shift']:.3f}; without "
                     f"all of them at once it is {_interval(joint)}, and the credibly negative "
                     f"set change is {_change(joint)}. By the registered rule the result is "
                     f"{verdict} to influential cells.")
    cal = study["calibration"]
    dec = cal["decomposition"]
    lines.append("- **Where the prior extremes come from.** Share of simulated `primary` "
                 "changes beyond 5 pp: " + "; ".join(
                     f"{k} {v['share_above_5pp']:.1%}" for k, v in dec.items()) + ".")
    for name, fit in cal["fits"].items():
        met = "meets" if cal["criterion_met"][name] else "does not meet"
        verdict = "prior-sensitive" if fit["prior_sensitive"] else "not prior-sensitive"
        lines.append(f"- **Calibrated prior, {_code(name)}.** Scale {fit['scale']:g} ({met} the "
                     f"criterion); estimand {_interval(fit)} against "
                     f"{_interval(fit['registered'])}; credibly negative set change: "
                     f"{_change(fit)}; persistence p {fit['lag1_p']:.3f} against "
                     f"{fit['lag1_p_registered']:.3f}. By the registered rule the result is "
                     f"{verdict}.")
    return lines


def render(study: Mapping[str, Any], command: str) -> str:
    runs = study["runs"]
    lines = [
        "# Hierarchical Bayes: diagnostics of the reported results", "",
        "Generated by `eu-okun bayes-diagnostics` from the frozen snapshot "
        "`snapshots/eurostat-2026-10-15`. "
        "[BAYES_DIAGNOSTICS_PROTOCOL.md](../BAYES_DIAGNOSTICS_PROTOCOL.md) registered "
        "every check and rule below before any of its fits. The registered results in "
        "[HIERARCHICAL_BAYES.md](HIERARCHICAL_BAYES.md), "
        "[BAYES_MODEL_CHECKS.md](BAYES_MODEL_CHECKS.md), "
        "[BAYES_DYNAMICS.md](BAYES_DYNAMICS.md) and [BAYES_REGIMES.md](BAYES_REGIMES.md) "
        "are unchanged; this report sits alongside them.", "",
        "## Summary", "", *summary_lines(study), "",
    ]

    lines += ["## 1. Run-to-run stability of the registered comparisons", "",
              f"Runs: {_names(runs)}. The local and Linux runs are the registered ones; "
              "the four seed runs repeat every fit with another seed and nothing else "
              "changed. Each cell gives the elpd difference A - B, its SE and the verdict "
              "(A or B better supported, = not distinguishable). The Linux values are "
              "those committed with the reports.", ""]
    header = ["Set", "A", "B", *runs, "Stable"]
    rows = []
    for c in study["comparisons"]:
        rows.append([c["set"], _code(c["a"]), _code(c["b"]),
                     *[_diff(c["runs"][r], c["a"], c["b"]) for r in runs],
                     "yes" if c["stable"] else "**no**"])
    lines += _table(header, rows, "lll" + "r" * len(runs) + "l")
    lines += ["Runs in which the registered label marks a comparison unreliable (more than "
              "1% of its cells with Pareto k > 0.7):", ""]
    lines += [f"- {_pair(c)}: {_names(c['unreliable_runs'])}." for c in study["comparisons"]
              if c["unreliable_runs"]] or ["- none."]
    lines += ["", "Cells with Pareto k > 0.7 by fit and run:", ""]
    sampled = [r for r in runs if r != LINUX]
    lines += _table(["Fit", "Cells", *sampled],
                    [[_code(p["spec"]), p["cells"],
                      *["n/a" if p["runs"][r] is None else p["runs"][r] for r in sampled]]
                     for p in study["pareto"]])

    lines += ["## 2. Persistence verdicts", "",
              "Posterior predictive p-value of the mean lag-1 autocorrelation (section 11.4 "
              "of BAYES_PROTOCOL.md: fixed when it lies in [0.05, 0.95]).", ""]
    rows = []
    for p in study["persistence"]:
        cells = ["n/a" if v is None else f"{v['p']:.3f}" + ("" if v["fixed"] else " misfit")
                 for v in (p["runs"][r] for r in runs)]
        rows.append([_code(p["spec"]), *cells, "yes" if p["stable"] else "**no**"])
    lines += _table(["Specification", *runs, "Stable"], rows, "l" + "r" * len(runs) + "l")

    lines += ["## 3. The quoted estimands across runs", ""]
    rows = []
    for e in study["estimands"]:
        rows.append([_code(e["spec"]), e["estimand"], _interval(e["registered"]),
                     *[f"{e['medians'][r]:+.3f}" if r in e["medians"] else "n/a"
                       for r in sampled[1:]],
                     " / ".join(str(e["negative_counts"].get(r, "n/a")) for r in runs)])
    lines += _table(["Specification", "Estimand", "Registered (local)",
                     *[f"Median, {r}" for r in sampled[1:]],
                     "Credibly negative, " + " / ".join(runs)], rows,
                    "lll" + "r" * (len(sampled) - 1) + "r")

    lines += ["## 4. Exact leave-one-out", "",
              "Each cell with Pareto k > 0.7 in the local run or in a seed run was refitted "
              "with its likelihood term removed and its value kept as conditioning data; "
              "the exact value is the log mean predictive density over the refit's draws.",
              ""]
    rows = [[_code(r["spec"]), r["cell"], f"{r['k_local']:.2f}", f"{r['k_max']:.2f}",
             r["runs_flagged"], f"{r['psis']:.2f}", f"{r['exact']:.2f}",
             f"{r['exact'] - r['psis']:+.2f}", "yes" if r["accepted"] else "**no**"]
            for r in study["exact"]]
    lines += _table(["Fit", "Cell", "k, local", "k, max", "Runs with k > 0.7", "PSIS", "Exact",
                     "Exact - PSIS", "Refit accepted"], rows, "ll" + "r" * 6 + "l")
    lines += ["The registered comparisons with the exact values in place of PSIS (local run, "
              "registered cell sets):", ""]
    rows = []
    for c in study["comparisons"]:
        local = c["runs"][LOCAL]
        corrected = c["corrected"]
        rows.append([c["set"], _code(c["a"]), _code(c["b"]), _diff(local, c["a"], c["b"]),
                     c["exact_cells"],
                     "not resolved" if corrected is None else _diff(corrected, c["a"], c["b"]),
                     "**yes**" if c["verdict_changes"] else "no",
                     "yes" if c["established"] else "**no**"])
    lines += _table(["Set", "A", "B", "PSIS", "Exact cells", "Corrected", "Verdict changes",
                     "Established"], rows, "lllrrrll")

    lines += ["## 5. Influential observations", ""]
    lines += _table(["Fit", "Cell", "Observed change (pp)", "Pareto k", "Fits with k > 0.7"],
                    [[_code(r["spec"]), r["cell"], f"{r['y']:+.2f}", f"{r['k_local']:.2f}",
                      r["fits_flagged"]] for r in study["inventory"]])
    for name, inf in study["influence"].items():
        reg = inf["registered"]
        lines += [f"**{_code(name)}**, registered {_interval(reg)}:", ""]
        rows = [[s["cell"] + ("" if s["accepted"] else " (refit not accepted)"),
                 f"{s['median']:+.3f}", f"{s['shift']:+.3f}", _change(s)]
                for s in inf["single"]]
        joint = inf["joint"]
        rows.append([f"all {joint['cells']} at once"
                     + ("" if joint["accepted"] else " (refit not accepted)"),
                     _interval(joint),
                     f"{joint['median'] - reg['median']:+.3f}", _change(joint)])
        lines += _table(["Left out", "Median", "Shift", "Credibly negative set"], rows, "lrrl")
        lines += [f"Persistence p-value of the joint fit: {joint['lag1_p']:.3f}. By the "
                  f"registered rule: {'sensitive' if inf['sensitive'] else 'not sensitive'} "
                  f"({inf['outside_interval']} medians outside the registered interval, "
                  f"{inf['set_changes']} changes of the credibly negative set).", ""]

    cal = study["calibration"]
    lines += ["## 6. Prior calibration", "",
              "**(a) Where the extremes come from.** The registered prior predictive check of "
              "`primary` (1000 draws, seed 20261018) with one component neutralised at a time; "
              "every variant uses the same prior draws and innovations.", ""]
    lines += _table(["Variant", "Changes beyond 5 pp", "Median largest change (pp)"],
                    [*[[k, f"{v['share_above_5pp']:.1%}", f"{v['median_largest_change']:.1f}"]
                       for k, v in cal["decomposition"].items()],
                     ["**observed**", f"{cal['observed']['share_above_5pp']:.1%}",
                      f"{cal['observed']['largest_change']:.1f} (largest)"]])
    lines += ["**(b) The ladder.** A step passes when at most 1% of simulated changes exceed "
              "5 pp; the calibrated scale is the widest step that passes.", ""]
    rows = []
    for name, steps in cal["ladder"].items():
        for s in steps:
            chosen = " (chosen)" if s["scale"] == cal["chosen"][name] else ""
            rows.append([_code(name), f"{s['scale']:g}{chosen}", f"{s['share_above_5pp']:.1%}",
                         f"{s['median_largest_change']:.1f}"])
    lines += _table(["Specification", "Prior scale", "Changes beyond 5 pp",
                     "Median largest change (pp)"], rows)
    lines += ["**(c) Calibrated fits.**", ""]
    rows = []
    for name, f in cal["fits"].items():
        rows.append([_code(name), f"{f['scale']:g}", "yes" if f["accepted"] else "**no**",
                     _interval(f["registered"]), _interval(f),
                     f"{f['negative_registered']} / {f['negative']}", _change(f),
                     f"{f['lag1_p_registered']:.3f} / {f['lag1_p']:.3f}",
                     "**yes**" if f["prior_sensitive"] else "no"])
    lines += _table(["Specification", "Scale", "Accepted", "Registered", "Calibrated",
                     "Credibly negative", "Set change", "Persistence p", "Prior-sensitive"],
                    rows, "lrlrrrlrl")
    for name, f in cal["fits"].items():
        if f["rho_f"] is not None:
            lines += [f"The scale also narrows the prior of the factor's AR(1) coefficient: in "
                      f"{_code(name)}, rho_f is {f['rho_f_registered'][0]:+.2f} "
                      f"[{f['rho_f_registered'][1]:+.2f}, {f['rho_f_registered'][2]:+.2f}] "
                      f"under the registered priors and {f['rho_f'][0]:+.2f} "
                      f"[{f['rho_f'][1]:+.2f}, {f['rho_f'][2]:+.2f}] under the calibrated "
                      f"scale.", ""]
    if cal["loo"] is not None:
        loo = cal["loo"]
        lines += [f"PSIS-LOO `ardl_dynamic_factor` against `primary` with both calibrated: "
                  f"{loo['elpd_diff']:+.1f} (SE {loo['se_diff']:.1f}), {loo['verdict']}.", ""]

    lines += ["## 7. Limits", "",
              "- Exact leave-one-out is computed in the local run only; the seed runs use the "
              "registered PSIS procedure.",
              "- Six runs bound how often a verdict on its threshold flips; they do not give "
              "its probability.",
              "- The calibrated scale is chosen from the prior predictive alone; it makes the "
              "priors narrower, not better informed.",
              "- No specification identifies a causal GDP shock.", "",
              "## Reproduce", "", "```", command, "```", ""]
    return "\n".join(lines)


def _csv_rows(study: Mapping[str, Any]) -> dict[str, tuple[list[str], list[list[Any]]]]:
    stability = []
    for c in study["comparisons"]:
        for run, row in c["runs"].items():
            stability.append([c["set"], c["a"], c["b"], run,
                              *(["", "", "", ""] if row is None else
                                [row["elpd_diff"], row["se_diff"], row["verdict"],
                                 row["reliable"]])])
    comparisons = [[c["set"], c["a"], c["b"], c["cells"], c["runs"][LOCAL]["elpd_diff"],
                    c["runs"][LOCAL]["se_diff"], c["runs"][LOCAL]["verdict"], c["exact_cells"],
                    *(["", "", "not resolved"] if c["corrected"] is None else
                      [c["corrected"]["elpd_diff"], c["corrected"]["se_diff"],
                       c["corrected"]["verdict"]]),
                    c["stable"], c["established"]] for c in study["comparisons"]]
    exact = [[r["spec"], r["cell"], r["k_local"], r["k_max"], r["runs_flagged"], r["psis"],
              r["exact"], r["accepted"]] for r in study["exact"]]
    influence = []
    for name, inf in study["influence"].items():
        for s in inf["single"]:
            influence.append([name, s["cell"], s["median"], s["shift"], " ".join(s["added"]),
                              " ".join(s["dropped"])])
        j = inf["joint"]
        influence.append([name, f"all {j['cells']}", j["median"],
                          j["median"] - inf["registered"]["median"], " ".join(j["added"]),
                          " ".join(j["dropped"])])
    cal = study["calibration"]
    prior = [["decomposition", "primary", k, v["share_above_5pp"], v["median_largest_change"]]
             for k, v in cal["decomposition"].items()]
    prior += [["ladder", name, s["scale"], s["share_above_5pp"], s["median_largest_change"]]
              for name, steps in cal["ladder"].items() for s in steps]
    return {
        "stability.csv": (["set", "model_a", "model_b", "run", "elpd_diff", "se_diff",
                           "verdict", "reliable"], stability),
        "comparisons.csv": (["set", "model_a", "model_b", "cells", "psis_elpd_diff",
                             "psis_se_diff", "psis_verdict", "exact_cells",
                             "corrected_elpd_diff", "corrected_se_diff", "corrected_verdict",
                             "stable", "established"], comparisons),
        "exact_loo.csv": (["spec", "cell", "pareto_k_local", "pareto_k_max", "runs_flagged",
                           "psis_elpd", "exact_elpd", "refit_accepted"], exact),
        "influence.csv": (["spec", "left_out", "median", "shift", "added", "dropped"],
                          influence),
        "prior_calibration.csv": (["part", "spec", "variant_or_scale", "share_above_5pp",
                                   "median_largest_change"], prior),
    }


def write_outputs(study: Mapping[str, Any], output_dir: Path, report: Path,
                  command: str) -> None:
    import csv

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, (header, rows) in _csv_rows(study).items():
        with (output_dir / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
    document = {**study, "command": command}
    (output_dir / "summary.json").write_text(json.dumps(document, indent=1, allow_nan=False)
                                             + "\n", encoding="utf-8", newline="\n")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render(study, command), encoding="utf-8", newline="\n")


def write_diagnostics_report(fits_dir: Path, refits_dir: Path, data_dir: Path,
                             docs_dir: Path, output_dir: Path, report: Path | None = None,
                             ) -> dict[str, Any]:
    """Build the study from the fits and write the CSV files, the summary and the report."""
    study = build_study(fits_dir, refits_dir, data_dir, docs_dir)
    report = report or output_dir / "BAYES_DIAGNOSTICS.md"
    command = (f"poetry run eu-okun bayes-diagnostics --fits-dir {fits_dir.as_posix()} "
               f"--refits-dir {refits_dir.as_posix()} --data-dir {data_dir.as_posix()} "
               f"--docs-dir {docs_dir.as_posix()} --output-dir {output_dir.as_posix()} "
               f"--report {report.as_posix()}")
    write_outputs(study, output_dir, report, command)
    return study
