"""Consistency checks on a finished run. Each failure is listed; the command exits non-zero
if there is any."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import Config
from .harness import read_forecast
from .metrics import daily_errors, wape


def check(cfg: Config, countries: list[str] | None = None, strict: bool = True) -> list[str]:
    from .data import load_prepared

    problems = []
    path = cfg.results_dir / "summary.json"
    if not path.exists():
        return ["results/summary.json is missing; run `swaptails summarize`"]
    s = json.loads(path.read_text(encoding="utf-8"))
    missing = s.get("status", {}).get("missing_runs", [])
    if missing and strict:
        problems.append(f"{len(missing)} forecaster-variant-country runs are missing: {missing[:10]}")
    for llm, cov in s.get("label_coverage", {}).items():
        low = {k: v for k, v in cov.items() if v is not None and v < 1}
        if low and strict:
            problems.append(f"{llm}: not every candidate day answered {low}")
    fc = s.get("forecasting")
    if fc:
        countries = countries or sorted(s["status"]["countries"])
        # Recompute one headline WAPE from the stored forecasts.
        run = fc["headline_runs"][0]
        f, v = run.split("|")
        total_abs = total_y = 0.0
        for k in countries:
            frame = read_forecast(cfg, f, v, k)
            if frame is None:
                continue
            prep = load_prepared(cfg, k)
            d = daily_errors(frame, pd.Series(prep.observed, index=prep.index), cfg.quantiles)
            total_abs += d["abs"].sum()
            total_y += d["actual"].sum()
        if total_y and not np.isclose(wape([total_abs], [total_y]), fc["runs"][run]["wape"], rtol=1e-4, atol=1e-6):
            problems.append(f"{run}: WAPE in summary.json does not match the stored forecasts")
        for run, r in fc["runs"].items():
            if r.get("days", 0) == 0:
                problems.append(f"{run}: no scored days")
            elif not (0 <= r["wape"] < 1):
                problems.append(f"{run}: implausible WAPE {r['wape']}")
            if r.get("cvar95_daily_wape") is not None and r["cvar95_daily_wape"] < r["wape"] * 0.999:
                problems.append(f"{run}: tail error below the average")
    for name in ("RESULTS.md", "run_manifest.json"):
        if not (cfg.results_dir / name).exists():
            problems.append(f"results/{name} is missing")
    return problems
