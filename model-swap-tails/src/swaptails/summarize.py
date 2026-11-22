"""Every number the article uses, written to results/summary.json and results/RESULTS.md.

Three blocks:
1. text step: how often each language model's day type matches what demand did, on all the
   days it was asked about and on the informative ones (where the answer differs from the
   weekday);
2. forecasting step: each forecaster on average (WAPE) and in the tail (P95 and CVaR95 of
   daily WAPE), overall, by country and by kind of day, against the operators' forecasts;
3. the two steps together: how a covariate-capable forecaster's error on the informative
   days moves when the day type comes from a flag, from each language model, or from the
   ex-post truth.
"""

from __future__ import annotations

import json
import math
from datetime import date

import numpy as np
import pandas as pd

from . import __version__
from .config import DAY_TYPES, Config
from .data import load_prepared, read_manifest
from .days import calendars, coverage, day_table
from .harness import read_forecast
from .llm import labels_path, prompt_hash
from .metrics import (
    bootstrap,
    cluster_bootstrap_accuracy,
    daily_errors,
    interval,
    kendall_tau,
    rank,
    summarise,
)

STRATA = (
    "national_holiday",
    "regional_holiday",
    "optional_holiday",
    "adjacent",
    "year_end",
    "ordinary_workday",
    "ordinary_weekend",
)
TSO = "tso|reference"


def _clean(obj):
    """JSON-safe: NaN to None, numpy scalars to Python, dates to ISO strings."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        return None if math.isnan(float(obj)) else round(float(obj), 6)
    if isinstance(obj, date):
        return obj.isoformat()
    return obj


# --------------------------------------------------------------------------- inputs


def load_inputs(cfg: Config, countries: list[str], need_data: bool = True):
    cals = calendars(cfg, countries)
    preps, tables = {}, {}
    for k in countries:
        prep = load_prepared(cfg, k) if need_data else None
        preps[k] = prep
        tables[k] = day_table(cfg, k, cals[k], prep)
    return cals, preps, tables


def collect_errors(cfg: Config, countries: list[str], preps: dict, tables: dict) -> tuple[pd.DataFrame, dict]:
    """Daily errors of every run present, on the days all present runs share in each country,
    plus the operators' forecast on those days where it is complete."""
    frames, status = [], {"missing_runs": [], "countries": {}}
    for k in countries:
        prep = preps[k]
        actual = pd.Series(prep.observed, index=prep.index)
        per_run = {}
        for run in cfg.run_names():
            forecaster, variant = run.split("|")
            fc = read_forecast(cfg, forecaster, variant, k)
            if fc is None:
                status["missing_runs"].append(f"{run}|{k}")
                continue
            per_run[run] = daily_errors(fc, actual, cfg.quantiles)
        if not per_run:
            continue
        shared = sorted(set.intersection(*(set(d.index) for d in per_run.values())))
        for run, d in per_run.items():
            d = d.loc[shared].copy()
            d["run"], d["country"] = run, k
            frames.append(d)
        tso_days = 0
        if prep.tso is not None:
            fc = _tso_frame(prep, shared)
            if not fc.empty:
                d = daily_errors(fc, actual, cfg.quantiles)
                d = d[d["hours"] == fc.groupby("date").size().reindex(d.index)]
                tso_days = len(d)
                d["run"], d["country"] = TSO, k
                frames.append(d)
        status["countries"][k] = {
            "days_scored": len(shared),
            "runs": len(per_run),
            "reference_days": tso_days,
            "reference_coverage": tso_days / len(shared) if shared else float("nan"),
        }
    errors = pd.concat(frames).reset_index().rename(columns={"index": "date"}) if frames else pd.DataFrame()
    return errors, status


def _tso_frame(prep, days: list[date]) -> pd.DataFrame:
    """The operator's forecast for the target hours of the given days, where every hour has a value."""
    keep = np.isin(prep.local_date, np.array(days, dtype=object)) & prep.scorable
    frame = pd.DataFrame({"date": prep.local_date[keep], "timestamp_utc": prep.index[keep], "q0.5": prep.tso[keep]})
    complete = frame.groupby("date")["q0.5"].transform(lambda s: s.notna().all())
    frame = frame[complete].copy()
    for q in ("q0.1", "q0.9"):
        frame[q] = frame["q0.5"]
    return frame


def attach_strata(cfg: Config, errors: pd.DataFrame, tables: dict) -> pd.DataFrame:
    parts = []
    for k, table in tables.items():
        sub = errors[errors["country"] == k]
        if sub.empty:
            continue
        info = table[["category", "truth", "informative", "temperature", "weekday_rule"]]
        sub = sub.join(info, on="date")
        period = table.loc[
            [d for d in table.index if cfg.forecast_first_day <= d <= cfg.forecast_last_day], "temperature"
        ]
        lo, hi = np.nanquantile(period, cfg.extreme_temperature_quantiles) if period.notna().any() else (np.nan, np.nan)
        sub["extreme_temperature"] = (sub["temperature"] < lo) | (sub["temperature"] > hi)
        sub["dst_change"] = sub["hours"] != 24
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- forecasting


def run_meta(cfg: Config, run: str) -> dict:
    if run == TSO:
        return {
            "label": "Operator day-ahead forecast",
            "bloc": "Reference",
            "developer": "national TSOs",
            "variant": "-",
        }
    forecaster, variant = run.split("|")
    spec = cfg.forecasters[forecaster]
    return {
        "label": spec.label,
        "bloc": spec.bloc,
        "developer": spec.developer,
        "variant": variant,
        "forecaster": forecaster,
    }


def forecasting_block(cfg: Config, errors: pd.DataFrame, status: dict) -> dict:
    tail = cfg.tail_quantile
    runs = sorted(errors["run"].unique())
    out = {"runs": {}, "by_country": {}, "by_stratum": {}}
    for run in runs:
        days = errors[errors["run"] == run]
        entry = {**run_meta(cfg, run), **summarise(days, tail)}
        entry["informative"] = summarise(days[days["informative"]], tail)
        entry["extreme_temperature"] = summarise(days[days["extreme_temperature"]], tail)
        entry["dst_change"] = summarise(days[days["dst_change"]], tail)
        sums = days.groupby("country")[["abs", "actual"]].sum()
        entry["mean_country_wape"] = float((sums["abs"] / sums["actual"]).mean())
        out["runs"][run] = entry
        out["by_country"][run] = {k: summarise(g, tail) for k, g in days.groupby("country")}
        out["by_stratum"][run] = {s: summarise(days[days["category"] == s], tail) for s in STRATA}
        out["by_stratum"][run]["informative"] = entry["informative"]
    # Headline comparison: one run per forecaster.
    headline = [cfg.headline(f) for f in cfg.forecasters if cfg.headline(f) in out["runs"]]
    good_ref = [k for k, v in status["countries"].items() if v["reference_coverage"] >= cfg.reference_min_coverage]
    out["headline_runs"] = headline
    out["reference_countries"] = good_ref
    w = {r: out["runs"][r]["wape"] for r in headline}
    c = {r: out["runs"][r]["cvar95_daily_wape"] for r in headline}
    h = {r: out["by_stratum"][r]["national_holiday"].get("wape", float("nan")) for r in headline}
    inf = {r: out["runs"][r]["informative"].get("wape", float("nan")) for r in headline}
    ranks = {"wape": rank(w), "cvar95": rank(c), "national_holiday_wape": rank(h), "informative_wape": rank(inf)}
    out["headline_ranks"] = ranks
    out["rank_agreement"] = {
        "wape_vs_cvar95": kendall_tau(ranks["wape"], ranks["cvar95"]),
        "wape_vs_national_holiday": kendall_tau(ranks["wape"], ranks["national_holiday_wape"]),
        "wape_vs_informative": kendall_tau(ranks["wape"], ranks["informative_wape"]),
    }
    out["winners"] = {
        k: min(v, key=v.get) if v else None
        for k, v in {"wape": w, "cvar95": c, "national_holiday": h, "informative": inf}.items()
    }
    out["by_country_winners"] = {}
    for k in sorted(errors["country"].unique()):
        per = {r: out["by_country"][r].get(k, {}) for r in headline}
        out["by_country_winners"][k] = {
            "wape": min(per, key=lambda r: per[r].get("wape", np.inf)) if per else None,
            "cvar95": min(per, key=lambda r: per[r].get("cvar95_daily_wape", np.inf)) if per else None,
        }
    # Best of each bloc among the headline runs, on average and in the tail.
    out["best_by_bloc"] = {}
    for bloc in ("Europe", "United States", "China"):
        members = [r for r in headline if out["runs"][r]["bloc"] == bloc]
        if members:
            out["best_by_bloc"][bloc] = {
                "by_wape": min(members, key=lambda r: w[r]),
                "by_cvar95": min(members, key=lambda r: c[r]),
            }
    return out


def bootstrap_block(cfg: Config, errors: pd.DataFrame, runs: list[str], countries: list[str]) -> dict:
    """Pooled WAPE and CVaR95 with paired block-bootstrap intervals, and pairwise differences."""
    sub = errors[errors["run"].isin(runs) & errors["country"].isin(countries)]
    if sub.empty:
        return {}
    dates = sorted(sub["date"].unique())
    pos = {d: i for i, d in enumerate(dates)}
    shape = (len(runs), len(countries), len(dates))
    a, y = np.full(shape, np.nan), np.full(shape, np.nan)
    for (run, k), g in sub.groupby(["run", "country"]):
        i, j = runs.index(run), countries.index(k)
        cols = [pos[d] for d in g["date"]]
        a[i, j, cols] = g["abs"].to_numpy()
        y[i, j, cols] = g["actual"].to_numpy()
    valid = ~np.isnan(a).any(axis=0) & ~np.isnan(y).any(axis=0)
    reps = bootstrap(a, y, valid, cfg.tail_quantile, cfg.bootstrap_block_days, cfg.bootstrap_replications, cfg.seed)
    point_w = np.nansum(np.where(valid, a, 0), axis=(1, 2)) / np.nansum(np.where(valid, y, 0), axis=(1, 2))
    out = {"countries": countries, "days": int(valid.sum()), "runs": {}, "differences": {}}
    for i, run in enumerate(runs):
        out["runs"][run] = {
            "wape": float(point_w[i]),
            "wape_ci": interval(reps["wape"][:, i]),
            "cvar95_ci": interval(reps["cvar"][:, i]),
        }
    for i, r1 in enumerate(runs):
        for j, r2 in enumerate(runs):
            if j <= i:
                continue
            dw = reps["wape"][:, i] - reps["wape"][:, j]
            dc = reps["cvar"][:, i] - reps["cvar"][:, j]
            out["differences"][f"{r1} - {r2}"] = {
                "wape_ci": interval(dw),
                "cvar95_ci": interval(dc),
                "share_wape_below_zero": float((dw < 0).mean()),
                "share_cvar95_below_zero": float((dc < 0).mean()),
            }
    return out


def propagation_block(cfg: Config, errors: pd.DataFrame) -> dict:
    """For each forecaster with covariates: WAPE overall and on informative days by variant."""
    out = {}
    for f, spec in cfg.forecasters.items():
        if not spec.covariates:
            continue
        rows = {}
        for v in cfg.variants_for(f):
            days = errors[errors["run"] == f"{f}|{v}"]
            if days.empty:
                continue
            rows[v] = {
                "wape": summarise(days, cfg.tail_quantile).get("wape"),
                "cvar95": summarise(days, cfg.tail_quantile).get("cvar95_daily_wape"),
                "informative_wape": summarise(days[days["informative"]], cfg.tail_quantile).get("wape"),
                "holiday_wape": summarise(
                    days[days["category"].isin(["national_holiday", "regional_holiday"])], cfg.tail_quantile
                ).get("wape"),
            }
        out[f] = rows
    return out


# --------------------------------------------------------------------------- text step


def labels_block(cfg: Config, tables: dict) -> dict:
    out = {"llms": {}, "rules": {}}
    frames = []
    for k, t in tables.items():
        t = t[t["label_year"] & t["truth"].apply(lambda x: isinstance(x, str))].copy()
        t["country"] = k
        frames.append(t)
    days = pd.concat(frames).reset_index()
    cand = days[days["candidate"].notna()].copy()
    cand["cluster"] = [
        f"{k}-{d.isocalendar()[0]}-{d.isocalendar()[1]}" for k, d in zip(cand["country"], cand["date"], strict=True)
    ]
    reps, seed = cfg.bootstrap_replications, cfg.seed

    def score(pred: pd.Series, subset: pd.DataFrame) -> dict:
        p = pred.loc[subset.index]
        correct = (p == subset["truth"]).to_numpy()
        inf = subset["informative"].to_numpy(dtype=bool)
        res = {
            "n": int(len(subset)),
            "accuracy": float(correct.mean()) if len(subset) else float("nan"),
            "accuracy_ci": cluster_bootstrap_accuracy(correct, subset["cluster"].to_numpy(), reps, seed),
            "n_informative": int(inf.sum()),
            "accuracy_informative": float(correct[inf].mean()) if inf.any() else float("nan"),
            "accuracy_informative_ci": cluster_bootstrap_accuracy(
                correct[inf], subset["cluster"].to_numpy()[inf], reps, seed
            ),
            "accuracy_uninformative": float(correct[~inf].mean()) if (~inf).any() else float("nan"),
            "by_reason": {
                r: float(correct[(subset["candidate"] == r).to_numpy()].mean()) for r in subset["candidate"].unique()
            },
            "by_category": {
                c: float(correct[(subset["category"] == c).to_numpy()].mean()) for c in subset["category"].unique()
            },
            "by_country": {
                c: float(correct[(subset["country"] == c).to_numpy()].mean()) for c in subset["country"].unique()
            },
            "by_country_informative": {
                c: float(correct[(subset["country"] == c).to_numpy() & inf].mean())
                if ((subset["country"] == c).to_numpy() & inf).any()
                else float("nan")
                for c in subset["country"].unique()
            },
        }
        return res

    for rule in ("weekday_rule", "flag_rule"):
        out["rules"][rule] = score(cand[rule], cand)
        out["rules"][rule]["accuracy_all_days"] = float((days[rule] == days["truth"]).mean())
    for llm, spec in cfg.llms.items():
        asked = cand[f"asked:{llm}"].astype(bool)
        if not asked.any():
            continue
        sub = cand[asked]
        answers = sub[f"answer:{llm}"].where(sub[f"answer:{llm}"].notna(), "invalid")
        entry = {
            "label": spec.label,
            "bloc": spec.bloc,
            "developer": spec.developer,
            "licence": spec.licence,
            "model": spec.model,
            "coverage": float(asked.mean()),
            "valid_rate": float(sub[f"answer:{llm}"].notna().mean()),
            **score(answers, sub),
            "accuracy_all_days": float((days[f"llm:{llm}"] == days["truth"]).mean()),
            "confusion": _confusion(sub["truth"], answers),
        }
        out["llms"][llm] = entry
    out["days_with_truth"] = int(len(days))
    out["candidate_days"] = int(len(cand))
    out["informative_days"] = int(cand["informative"].sum())
    out["informative_outside_candidates"] = int(days[days["candidate"].isna()]["informative"].sum())
    out["truth_distribution"] = {t: int((cand["truth"] == t).sum()) for t in DAY_TYPES}
    out["prompt_hash"] = prompt_hash(cfg.prompt_version)
    return out


def _confusion(truth: pd.Series, pred: pd.Series) -> dict:
    labels = [*DAY_TYPES, "invalid"]
    return {t: {p: int(((truth == t) & (pred == p)).sum()) for p in labels} for t in DAY_TYPES}


# --------------------------------------------------------------------------- figure data


def figure_data(cfg: Config, errors: pd.DataFrame, preps: dict, forecasting: dict) -> dict:
    """What the figures need, so they can be drawn again without the forecasts."""
    out = {"examples": []}
    best = forecasting.get("best_by_bloc", {})
    show = [v["by_wape"] for v in best.values()]
    for k, start in cfg.examples:
        if k not in preps or preps[k] is None:
            continue
        prep = preps[k]
        days = [start + pd.Timedelta(days=i) for i in range(7)]
        days = [d.date() if hasattr(d, "date") else d for d in days]
        mask = np.isin(prep.local_date, np.array(days, dtype=object))
        series = {"actual": prep.observed[mask].tolist()}
        if prep.tso is not None:
            series["Operator day-ahead"] = prep.tso[mask].tolist()
        for run in show:
            f, v = run.split("|")
            fc = read_forecast(cfg, f, v, k)
            if fc is None:
                continue
            fc = fc.set_index("timestamp_utc")["q0.5"]
            series[f"{cfg.forecasters[f].label} ({v})"] = fc.reindex(prep.index[mask]).tolist()
        out["examples"].append(
            {
                "country": k,
                "start": start,
                "timestamps": [t.isoformat() for t in prep.index[mask]],
                "series": series,
            }
        )
    out["daily_wape_headline"] = {
        run: errors[errors["run"] == run]["wape"].round(5).tolist()
        for run in [*forecasting.get("headline_runs", []), TSO]
        if (errors["run"] == run).any()
    }
    return out


# --------------------------------------------------------------------------- main


def summarize(cfg: Config, countries: list[str] | None = None) -> dict:
    countries = countries or list(cfg.countries)
    countries = [k for k in countries if (cfg.cache_dir / f"load_{k}.parquet").exists()]
    cals, preps, tables = load_inputs(cfg, countries)
    errors, status = collect_errors(cfg, countries, preps, tables)
    summary = {"version": __version__, "config": cfg.path.name, "status": status}
    summary["labels"] = labels_block(cfg, tables)
    summary["label_coverage"] = {llm: {k: coverage(t, llm) for k, t in tables.items()} for llm in cfg.llms}
    if not errors.empty:
        errors = attach_strata(cfg, errors, tables)
        forecasting = forecasting_block(cfg, errors, status)
        summary["forecasting"] = forecasting
        model_runs = [r for r in forecasting["runs"] if r != TSO]
        summary["bootstrap_models"] = bootstrap_block(cfg, errors, model_runs, sorted(status["countries"]))
        refs = forecasting["reference_countries"]
        if refs:
            summary["bootstrap_with_reference"] = bootstrap_block(
                cfg, errors, [*forecasting["headline_runs"], TSO], refs
            )
        summary["propagation"] = propagation_block(cfg, errors)
        fig = figure_data(cfg, errors, preps, forecasting)
        cfg.results_dir.mkdir(parents=True, exist_ok=True)
        (cfg.results_dir / "figure_data.json").write_text(json.dumps(_clean(fig)) + "\n", encoding="utf-8")
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "summary.json").write_text(
        json.dumps(_clean(summary), indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_manifest(cfg)
    write_markdown(cfg, _clean(summary))
    return summary


def write_manifest(cfg: Config) -> None:
    """Provenance of the run: data requests and hashes, model revisions, label provenance."""
    manifest = {"data": read_manifest(cfg), "forecasts": {}, "llms": {}}
    for path in sorted(cfg.forecasts_dir.glob("*.json")):
        manifest["forecasts"][path.stem] = json.loads(path.read_text(encoding="utf-8"))
    for path in sorted(cfg.forecasts_dir.glob("provenance_*.json")):
        manifest["forecasts"][path.stem] = json.loads(path.read_text(encoding="utf-8"))
    for llm in cfg.llms:
        prov = cfg.labels_dir / f"{llm}.provenance.json"
        if prov.exists():
            manifest["llms"][llm] = json.loads(prov.read_text(encoding="utf-8"))
        manifest["llms"].setdefault(llm, {})["labels_file"] = str(labels_path(cfg, llm).relative_to(cfg.root))
    (cfg.results_dir / "run_manifest.json").write_text(
        json.dumps(_clean(manifest), indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


def _pct(x, digits=2) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def _row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(header: list[str]) -> list[str]:
    return [_row(*header), _row(*(["---"] * len(header)))]


def write_markdown(cfg: Config, s: dict) -> None:
    lines = [
        "# Results",
        "",
        "Generated by `swaptails summarize` from config/experiment.toml. "
        "Every number in the article comes from `summary.json`.",
        "",
    ]
    lab = s.get("labels", {})
    if lab.get("llms"):
        lines += [
            "## Text step: day types from a calendar note",
            "",
            f"{lab['candidate_days']} candidate days in {list(cfg.label_years)}, of which "
            f"{lab['informative_days']} informative (demand differed from what the weekday alone implies). "
            f"Informative days outside the candidates: {lab['informative_outside_candidates']}.",
            "",
            *_table(["Model", "Bloc", "Valid", "Candidate days", "Informative days [95% CI]", "All days (pipeline)"]),
        ]
        for name, r in lab["rules"].items():
            lines.append(
                _row(
                    name.replace("_", " "),
                    "rule",
                    "–",
                    _pct(r["accuracy"], 1),
                    _pct(r["accuracy_informative"], 1),
                    _pct(r["accuracy_all_days"], 1),
                )
            )
        for r in sorted(lab["llms"].values(), key=lambda r: -(r["accuracy_informative"] or 0)):
            lo, hi = r["accuracy_informative_ci"]
            lines.append(
                _row(
                    r["label"],
                    r["bloc"],
                    _pct(r["valid_rate"], 1),
                    _pct(r["accuracy"], 1),
                    f"{_pct(r['accuracy_informative'], 1)} [{_pct(lo, 1)}, {_pct(hi, 1)}]",
                    _pct(r["accuracy_all_days"], 1),
                )
            )
        lines.append("")
    fc = s.get("forecasting")
    if fc:
        agree = fc["rank_agreement"]
        lines += [
            "## Forecasting step: average and tail",
            "",
            f"Headline runs: models with covariates use the `{cfg.headline_variant}` variant. Rank agreement "
            f"(Kendall tau) between WAPE and CVaR95: {agree['wape_vs_cvar95']}; between WAPE and national-holiday "
            f"WAPE: {agree['wape_vs_national_holiday']}.",
            "",
            *_table(
                ["Run", "Bloc", "WAPE", "P95 daily", "CVaR95 daily", "National holidays", "Informative days"]
                + ["Rank WAPE", "Rank CVaR95"]
            ),
        ]
        ranks = fc["headline_ranks"]
        runs = sorted(fc["headline_runs"], key=lambda r: fc["runs"][r]["wape"])
        if "tso|reference" in fc["runs"]:
            runs.append("tso|reference")
        for run in runs:
            r = fc["runs"][run]
            lines.append(
                _row(
                    f"{r['label']} ({r['variant']})",
                    r["bloc"],
                    _pct(r["wape"]),
                    _pct(r["p95_daily_wape"]),
                    _pct(r["cvar95_daily_wape"]),
                    _pct(fc["by_stratum"][run]["national_holiday"].get("wape")),
                    _pct(r["informative"].get("wape")),
                    ranks["wape"].get(run, "–"),
                    ranks["cvar95"].get(run, "–"),
                )
            )
        lines += [
            "",
            "## Both steps: where the day type comes from",
            "",
            *_table(["Forecaster", "Variant", "WAPE", "CVaR95", "Informative days", "Holidays"]),
        ]
        for f, rows in s.get("propagation", {}).items():
            for v in cfg.variants_for(f):
                if v in rows:
                    r = rows[v]
                    lines.append(
                        _row(
                            f,
                            v,
                            _pct(r["wape"]),
                            _pct(r["cvar95"]),
                            _pct(r["informative_wape"]),
                            _pct(r["holiday_wape"]),
                        )
                    )
        lines.append("")
    missing = s.get("status", {}).get("missing_runs")
    if missing:
        lines += ["## Missing runs", "", ", ".join(missing), ""]
    (cfg.results_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
