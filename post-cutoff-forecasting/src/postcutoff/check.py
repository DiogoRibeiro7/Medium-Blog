"""Consistency checks on a finished run. Each check recomputes something the pipeline
produced by a second, simpler route; any disagreement fails the run."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import Config
from .data import load_snapshot, prepare, sha256


def run_checks(cfg: Config, require_figures: bool = True) -> list[str]:
    problems: list[str] = []
    res = cfg.results_dir
    scores = pd.read_csv(res / "scores.csv")
    origins = pd.read_csv(res / "origins.csv")
    manifest = json.loads((res / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((res / "summary.json").read_text(encoding="utf-8"))

    # 1. The run used the snapshot that is on disk.
    if manifest["data_sha256"] != sha256(cfg.snapshot):
        problems.append("run_manifest.json was produced from a different data snapshot")

    # 2. Every model has every lead day of every scored origin, exactly once.
    days = cfg.horizon // 24
    scored = origins[origins["status"] == "scored"]
    if manifest.get("limit_per_window"):
        scored = scored.groupby("window", group_keys=False).tail(manifest["limit_per_window"])
    models = sorted(scores["model"].unique())
    expected = len(scored) * days
    for m in models:
        sub = scores[scores["model"] == m]
        if len(sub) != expected or sub.duplicated(["origin", "lead_day"]).any():
            problems.append(f"{m}: {len(sub)} score rows, expected {expected}")
        if set(sub["origin"]) != set(scored["origin"]):
            problems.append(f"{m}: scored origins differ from origins.csv")

    # 3. The reference model's errors, recomputed directly from the snapshot. Seasonal
    #    naive with a one-week period forecasts each hour with the hour 168 earlier, as
    #    fed to the models (masked hours replaced), for the first 168 hours.
    ref = cfg.reference_model
    if cfg.models[ref].spec == "SeasonalNaive(168)" and cfg.horizon <= 168 and ref in models:
        prep = prepare(load_snapshot(cfg), cfg)
        pos_of = {pd.Timestamp(t): i for i, t in enumerate(prep.index)}
        sub = scores[scores["model"] == ref]
        worst = 0.0
        for origin, rows in sub.groupby("origin"):
            p = pos_of[pd.Timestamp(origin)]
            y = prep.observed[p : p + cfg.horizon]
            f = prep.inputs[p - 168 : p - 168 + cfg.horizon]
            per_day = np.abs(y - f).reshape(days, 24).sum(axis=1)
            got = rows.sort_values("lead_day")["sum_abs_err"].to_numpy()
            worst = max(worst, float(np.max(np.abs(per_day - got) / np.maximum(per_day, 1.0))))
        if worst > 1e-3:
            problems.append(
                f"seasonal naive errors do not match a direct recomputation (worst relative gap {worst:.2e})"
            )

    # 4. Every ratio and every change in the summary, recomputed from the score rows with
    #    plain sums (no bootstrap code involved).
    days_of = {"day-ahead": [1], "week-ahead": list(range(1, days + 1))}
    column_of = {"mae": "sum_abs_err", "wql": "sum_wql_num"}

    def windows_of(label):
        if label in summary["windows"]:
            return [label]
        return summary["before"] if label.startswith("before") else summary["after"]

    def total(windows, horizon, model, column):
        sub = scores[
            scores["window"].isin(windows) & scores["lead_day"].isin(days_of[horizon]) & (scores["model"] == model)
        ]
        return sub[column].sum()

    def close(a, b):
        return abs(a - b) <= 1e-9 * max(1.0, abs(b))

    for metric in ("mae", "wql"):
        for horizon, tables in summary[f"relative_{metric}"].items():
            for name, rows in tables.items():
                for row in rows:
                    w = windows_of(row["windows"])
                    col = column_of[metric]
                    ratio = total(w, horizon, row["model"], col) / total(w, horizon, row["reference"], col)
                    if not close(ratio, row["ratio"]):
                        problems.append(
                            f"{metric} {horizon} {name}: {row['model']} in {row['windows']} does not match scores.csv"
                        )
                    lo, hi = row["ci95"]
                    tol = 1e-9 * max(1.0, abs(row["ratio"]))
                    if row["model"] != row["reference"] and not (lo - tol <= row["ratio"] <= hi + tol):
                        problems.append(f"{metric} {horizon} {name}: interval for {row['model']} excludes its estimate")
    for horizon, tables in summary["change"].items():
        for name, rows in tables.items():
            col = column_of["wql" if name.startswith("wql") else "mae"]
            for row in rows:
                before = total(row["before_windows"], horizon, row["model"], col) / total(
                    row["before_windows"], horizon, row["reference"], col
                )
                after = total(row["after_windows"], horizon, row["model"], col) / total(
                    row["after_windows"], horizon, row["reference"], col
                )
                if not close(100.0 * (after / before - 1.0), row["change_pct"]):
                    problems.append(f"change {horizon} {name}: {row['model']} does not match scores.csv")

    # 4b. Time alignment of every model's forecasts, when the stored forecasts are present:
    #     the median must be closest to the target at lag 0, not one or two hours off.
    if cfg.forecasts_dir.exists():
        from .harness import alignment

        prep_al = prepare(load_snapshot(cfg), cfg)
        pos_of = {pd.Timestamp(t): i for i, t in enumerate(prep_al.index)}
        for path in sorted(cfg.forecasts_dir.glob("*.npz")):
            stored = np.load(path, allow_pickle=False)
            origins_npz = [pd.Timestamp(str(o)) for o in stored["origins"]]
            qs = [round(float(q), 6) for q in stored["quantiles"]]
            med = np.sort(stored["forecasts"].astype(float), axis=2)[:, :, qs.index(0.5)]
            pos = np.array([pos_of[o] for o in origins_npz])
            Y = prep_al.observed[pos[:, None] + np.arange(med.shape[1])[None, :]]
            lags = alignment(Y, med)
            best = min(lags, key=lags.get)
            if best != 0:
                problems.append(f"{path.stem}: forecasts are closest to the target shifted by {best:+d} h")

    # 5. Ranges.
    m = pd.DataFrame(summary["metrics"])
    if ((m["coverage80"] < 0) | (m["coverage80"] > 1)).any():
        problems.append("coverage outside [0, 1]")
    if (m["mae"] <= 0).any() or (m["mase"] <= 0).any():
        problems.append("non-positive error metric")

    # 6. Foundation models: the exact weights are recorded.
    for name, prov in manifest["models"].items():
        if cfg.models[name].is_foundation and not prov.get("revision"):
            problems.append(f"{name}: weights revision not recorded")

    # 7. Figures.
    if require_figures:
        for f in ["fig1_series.png", "fig2_relative_mae.png", "fig3_change.png", "fig4_lead_day.png"]:
            if not (cfg.figures_dir / f).exists():
                problems.append(f"missing figure {f}")
    return problems
