"""Part 2: billionaire counts against WID top wealth shares, and residence counts.

Every number quoted in the Part 2 article is computed here and written to
``results/part2/numbers.json``; ``--check`` recomputes everything and fails if
any number differs. Five figures go to ``figures/part2/``.

Inputs (all hash-checked):

* the 54-country panel and PIP surveys of Part 1 (``part1_poverty_lines``);
* World Inequality Database series extracted by ``fetch_wid.py``
  (``data/wid_series.csv.gz``, ``data/wid_metadata.csv.gz``);
* Forbes 2026 aggregate counts from the frozen person-list audit in
  ``../billionaires-poverty-audit/evidence/forbes-2026-run-42099896845``:
  ``countryOfCitizenship`` (citizenship) and ``country`` (read here as
  residence; see ``residence``).

Usage::

    python part2_wealth_shares.py           # print, write results/part2 and figures/part2
    python part2_wealth_shares.py --check   # fail if any number differs from numbers.json

All results are descriptive, not causal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

import part1_poverty_lines as p1  # noqa: E402
from part1_poverty_lines import (AQUA, BASELINE, BLUE, INK, MUTED, ORANGE, SECONDARY,  # noqa: E402
                                 SURFACE, B, G, _compare, _frame, _line_axis, _official_lines,
                                 _style, at_line, build_panel, fmt, load_panel, load_pip)
from stats_tools import fractional_logit, loo_rmse_fractional, loo_rmse_ols, ols_hc3  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results" / "part2"
FIGURES = HERE / "figures" / "part2"
EVIDENCE = p1.AUDIT / "evidence" / "forbes-2026-run-42099896845"
WID_YEAR = 2024
LINE = 30.0
Z95 = 1.959963984540054

ISO2 = {
    "ALB": "AL", "ARM": "AM", "AUS": "AU", "AUT": "AT", "BEL": "BE", "BGR": "BG", "BRA": "BR",
    "CAN": "CA", "CHE": "CH", "CHL": "CL", "CHN": "CN", "COL": "CO", "CYP": "CY", "CZE": "CZ",
    "DEU": "DE", "DNK": "DK", "EGY": "EG", "ESP": "ES", "EST": "EE", "FIN": "FI", "FRA": "FR",
    "GBR": "GB", "GEO": "GE", "GRC": "GR", "HRV": "HR", "IDN": "ID", "IND": "IN", "IRL": "IE",
    "ISR": "IL", "ITA": "IT", "JPN": "JP", "KAZ": "KZ", "KOR": "KR", "LBN": "LB", "LUX": "LU",
    "MEX": "MX", "MYS": "MY", "NGA": "NG", "NLD": "NL", "NOR": "NO", "NPL": "NP", "PHL": "PH",
    "POL": "PL", "PRT": "PT", "ROU": "RO", "RUS": "RU", "SVK": "SK", "SWE": "SE", "THA": "TH",
    "TUR": "TR", "UKR": "UA", "USA": "US", "VNM": "VN", "ZAF": "ZA",
}
# WID percentile groups used, keyed by the column name given to them here.
WEALTH = {"wT10": "p90p100", "wT1": "p99p100", "wT01": "p99.9p100", "wT001": "p99.99p100",
          "wB50": "p0p50"}
INCOME = {"iT10": "p90p100", "iT1": "p99p100"}
LABEL = {B: "Billionaire density", "wT1": "Top 1% wealth share", "wT10": "Top 10% wealth share",
         "iT10": "Top 10% income share", "wT01": "Top 0.1% wealth share",
         "wT001": "Top 0.01% wealth share", "wB50": "Bottom 50% wealth share", "gini": "Survey Gini",
         "wB50x": "Bottom 50% wealth share"}
MEAN = "log10_mean"
SOURCE = ("Data: World Inequality Database (bulk file of 6 Nov 2026), 2024 shares, adults, equal split; Forbes 2026 "
          "billionaires; World Bank PIP (2021 PPP); WDI GDP per person. 54 countries; descriptive, not causal.")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_wid(data: Path = DATA) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Read the WID extract after checking it against its manifest."""
    manifest = json.loads((data / "wid_manifest.json").read_text(encoding="utf-8"))
    for name, key in (("wid_series.csv.gz", "wid_series_sha256"), ("wid_metadata.csv.gz", "wid_metadata_sha256")):
        if _sha(data / name) != manifest[key]:
            raise ValueError(f"{name} does not match wid_manifest.json")
    series = pd.read_csv(data / "wid_series.csv.gz")
    meta = pd.read_csv(data / "wid_metadata.csv.gz")
    return series, meta, manifest


def load_forbes(evidence: Path = EVIDENCE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the frozen Forbes aggregates after checking their pinned SHA-256."""
    pinned = dict(line.split()[::-1] for line in (evidence / "SHA256SUMS.txt").read_text().splitlines() if line)
    frames = []
    for name in ("counts_citizenship.csv", "counts_api_country.csv"):
        if _sha(evidence / name) != pinned[name]:
            raise ValueError(f"{name} does not match SHA256SUMS.txt")
        frames.append(pd.read_csv(evidence / name))
    return frames[0], frames[1]


def _wid_values(series: pd.DataFrame, variable: str, percentile: str) -> pd.Series:
    x = series[(series.variable == variable) & (series.percentile == percentile)]
    return x.set_index(["country", "year"])["value"]


def build(panel: pd.DataFrame, surveys: pd.DataFrame, heads: pd.DataFrame, series: pd.DataFrame,
          citizenship: pd.DataFrame, residence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Part 1 panel plus WID shares, WID data status and the three billionaire counts."""
    base, wide = build_panel(panel, surveys, heads)
    base["iso2"] = base.iso3.map(ISO2)
    if base.iso2.isna().any():
        raise ValueError("panel country without a WID code")
    for col, pct in WEALTH.items():
        v = _wid_values(series, "shweal992j", pct)
        base[col] = [100 * v.get((c, WID_YEAR), np.nan) for c in base.iso2]
        base[col + "_sy"] = [100 * v.get((c, min(int(y), WID_YEAR)), np.nan)
                             for c, y in zip(base.iso2, base.reporting_year)]
    for col, pct in INCOME.items():
        v = _wid_values(series, "sptinc992j", pct)
        base[col] = [100 * v.get((c, WID_YEAR), np.nan) for c in base.iso2]
    # WID grades each observation; a grade above 0 means the year rests on observed wealth data.
    w = series[(series.variable == "shweal992j") & (series.percentile == "p99p100") & (series.year >= 1995)]
    graded = w[w.data_quality > 0].groupby("country").agg(last_obs=("year", "max"), grade=("data_quality", "max"))
    base["last_obs"] = base.iso2.map(graded.last_obs)
    base["grade"] = base.iso2.map(graded.grade)
    base["observed"] = base.last_obs.notna()
    t1 = _wid_values(series, "shweal992j", "p99p100")
    base["wT1_lo"] = [100 * t1.get((c, int(y)), np.nan) if ok else t
                      for c, y, ok, t in zip(base.iso2, base.last_obs.fillna(WID_YEAR), base.observed, base.wT1)]
    cit = dict(zip(citizenship.country_label, citizenship.billionaires))
    res = dict(zip(residence.country_label, residence.billionaires))
    if not set(base.country) <= set(cit):
        raise ValueError(f"no Forbes citizenship label for {sorted(set(base.country) - set(cit))}")
    base["n_sec"] = base.billionaires.astype(float)
    base["n_cit"] = base.country.map(cit).astype(float)
    base["n_res"] = base.country.map(res).fillna(0).astype(float)
    per10m = 1e7 / base.population
    for k in ("sec", "cit", "res"):
        n = base[f"n_{k}"]
        base[f"B_{k}"] = np.log10((n * per10m).where(n > 0))
        base[f"Bh_{k}"] = np.log10((n + 0.5) * per10m)
    if not np.allclose(base.B_sec, base[B]):
        raise ValueError("secondary density does not reproduce the panel column")
    needed = list(WEALTH) + list(INCOME)
    if base[needed].isna().any().any():
        raise ValueError("missing WID share for a panel country")
    return base.reset_index(drop=True), wide


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def pcorr(frame: pd.DataFrame, x: str, y: str, controls: Sequence[str]) -> dict[str, Any]:
    """Partial correlation of x and y given controls, with a Fisher-z 95% interval."""
    f = frame[[x, y, *controls]].dropna()
    if controls:
        Z = sm.add_constant(f[list(controls)].to_numpy(dtype=float))
        rx = sm.OLS(f[x].to_numpy(dtype=float), Z).fit().resid
        ry = sm.OLS(f[y].to_numpy(dtype=float), Z).fit().resid
    else:
        rx, ry = f[x].to_numpy(dtype=float), f[y].to_numpy(dtype=float)
    r = float(np.corrcoef(rx, ry)[0, 1])
    se = 1.0 / np.sqrt(len(f) - 3 - len(controls))
    z = np.arctanh(r)
    return {"r": r, "ci95": [float(np.tanh(z - Z95 * se)), float(np.tanh(z + Z95 * se))], "n": int(len(f))}


def r2(frame: pd.DataFrame, y: str, cols: Sequence[str]) -> float:
    return ols_hc3(frame, y, cols)["r_squared"]


def concentration(base: pd.DataFrame) -> dict[str, Any]:
    """Is billionaire density a measure of wealth concentration?"""
    out: dict[str, Any] = {"raw": {}, "given_gdp": {}}
    for y in ("wT1", "wT10", "wT01", "wT001", "wB50", "iT10", "iT1", G, MEAN):
        out["raw"][y] = pcorr(base, B, y, [])
    groups = {"all": base, "observed": base[base.observed], "imputed": base[~base.observed]}
    for y in ("wT10", "wT1", "wT01", "wT001", "wB50", "iT10"):
        out["given_gdp"][y] = {g: pcorr(f, B, y, [G]) for g, f in groups.items()}
    out["wT1_raw_gdp"] = pcorr(base, "wT1", G, [])
    out["wT1_summary"] = {"min": float(base.wT1.min()), "max": float(base.wT1.max()),
                          "median": float(base.wT1.median()),
                          "min_iso3": base.loc[base.wT1.idxmin(), "iso3"],
                          "max_iso3": base.loc[base.wT1.idxmax(), "iso3"]}
    for iso in ("USA", "CHE", "NLD", "ZAF", "RUS", "BRA", "MEX", "IND", "SWE", "CHN"):
        row = base.loc[base.iso3 == iso].iloc[0]
        out.setdefault("countries", {})[iso] = {"wT1": float(row.wT1), "wT10": float(row.wT10),
                                                "density": float(10 ** row[B]), "n": float(row.n_sec)}
    return out


def dissociation(base: pd.DataFrame) -> dict[str, Any]:
    """What each measure tracks in the surveys, given GDP and survey type."""
    ctrl = [G, "consumption"]
    out = {}
    for x in (B, "wT1", "wT10", "iT10"):
        out[x] = {y: pcorr(base, x, y, ctrl) for y in (MEAN, "gini", "bottom40")}
    out["gini_corr_raw"] = {x: pcorr(base, x, "gini", [])["r"] for x in ("wT1", "wT10", "iT10")}
    return out


def imputation(base: pd.DataFrame, meta: pd.DataFrame) -> dict[str, Any]:
    """How many wealth shares rest on observed data, and how imputed ones relate to income shares."""
    obs, imp = base[base.observed], base[~base.observed]
    latam = base[base.iso3.isin(["BRA", "CHL", "COL", "MEX"])]
    m = meta[(meta.variable == "shweal992j") & meta.country.isin(base.iso2)]
    return {
        "n_observed": int(len(obs)), "n_imputed": int(len(imp)),
        "imputed_iso3": sorted(imp.iso3.tolist()),
        "observed_iso3": sorted(obs.iso3.tolist()),
        "last_obs_min": int(obs.last_obs.min()), "last_obs_max": int(obs.last_obs.max()),
        "last_obs_median": float(obs.last_obs.median()),
        "grade_counts": {str(int(k)): int(v) for k, v in obs.grade.value_counts().sort_index().items()},
        "r2_wT10_on_iT10": {"observed": r2(obs, "wT10", ["iT10"]), "imputed": r2(imp, "wT10", ["iT10"])},
        "r2_wT1_on_iT10": {"observed": r2(obs, "wT1", ["iT10"]), "imputed": r2(imp, "wT1", ["iT10"])},
        "latam": {"wT10_min": float(latam.wT10.min()), "wT10_max": float(latam.wT10.max()),
                  "wT1_min": float(latam.wT1.min()), "wT1_max": float(latam.wT1.max()),
                  "wB50_min": float(latam.wB50.min()), "wB50_max": float(latam.wB50.max()),
                  "iT10_min": float(latam.iT10.min()), "iT10_max": float(latam.iT10.max())},
        "metadata_avg_quality_zero": int((m.avg_quality == 0).sum()),
        "extrapolated_years": int(WID_YEAR - obs.last_obs.max()),
    }


def _models(f: pd.DataFrame, baseline: Sequence[str], extras: Sequence[str]) -> dict[str, Any]:
    """OLS and fractional-logit results for each extra variable added to a baseline."""
    out: dict[str, Any] = {"baseline": {"r_squared": r2(f, "poverty_pct", baseline),
                                        "loo_ols": loo_rmse_ols(f, "poverty_pct", baseline),
                                        "loo_frac": loo_rmse_fractional(f, "poverty_pct", baseline)}}
    for x in extras:
        cols = [*baseline, x]
        fit = ols_hc3(f, "poverty_pct", cols)
        frac = fractional_logit(f, "poverty_pct", cols, x)
        sd = float(f[x].std())
        out[x] = {"coef": fit["coef"][x], "p": fit["p"][x], "ci95": fit["ci95"][x], "r_squared": fit["r_squared"],
                  "loo_ols": loo_rmse_ols(f, "poverty_pct", cols), "loo_frac": loo_rmse_fractional(f, "poverty_pct", cols),
                  "ame": frac["ame"], "ame_ci95": frac["ci95"], "ame_p": frac["coef_p"], "sd": sd,
                  "ame_sd": frac["ame"] * sd}
    return out


def poverty(base: pd.DataFrame, wide: pd.DataFrame) -> dict[str, Any]:
    """What each measure adds about poverty at $30, given GDP and given GDP and the survey mean."""
    f = at_line(base, wide, LINE)
    extras = (B, "wT1", "wT10", "wB50", "iT10", "gini")
    out = {"gdp": _models(f, [G], extras), "mean_gdp": _models(f, [MEAN, G], extras)}
    both = ols_hc3(f, "poverty_pct", [G, B, "wT1"])
    out["gdp_both"] = {"coef_b": both["coef"][B], "p_b": both["p"][B], "coef_t1": both["coef"]["wT1"],
                       "p_t1": both["p"]["wT1"]}
    cons = ols_hc3(f, "poverty_pct", [MEAN, G, "consumption"])
    out["mean_gdp_consumption"] = {"coef": cons["coef"]["consumption"], "p": cons["p"]["consumption"]}
    o = f[f.observed]
    out["observed_only"] = {"n": int(len(o)), "gdp": _models(o, [G], (B, "wT1")),
                            "mean_gdp": _models(o, [MEAN, G], (B, "wT1", "gini"))}
    return out


def lines(base: pd.DataFrame, wide: pd.DataFrame) -> list[dict[str, Any]]:
    """Fractional-logit effects per standard deviation at every poverty line."""
    rows = []
    for line in wide.columns:
        f = at_line(base, wide, float(line))
        row: dict[str, Any] = {"line": float(line)}
        for x in (B, "wT1", "iT10"):
            sd = float(f[x].std())
            for key, baseline in (("mean_gdp", [MEAN, G]), ("gdp", [G])):
                frac = fractional_logit(f, "poverty_pct", [*baseline, x], x)
                row[f"{x}|{key}"] = {"ame_sd": frac["ame"] * sd, "ci95_sd": [c * sd for c in frac["ci95"]],
                                     "p": frac["coef_p"]}
        rows.append(row)
    return rows


def residence(base: pd.DataFrame, wide: pd.DataFrame, citizenship: pd.DataFrame,
              api: pd.DataFrame) -> dict[str, Any]:
    """Compare the secondary citizenship, Forbes citizenship and Forbes residence counts."""
    allc = api[["country_label", "billionaires"]].merge(
        citizenship[["country_label", "billionaires"]], on="country_label", how="outer",
        suffixes=("_res", "_cit")).fillna(0)
    allc["diff"] = allc.billionaires_res - allc.billionaires_cit
    allc = allc.sort_values(["diff", "country_label"])
    gains = allc[allc["diff"] > 0].sort_values("diff", ascending=False).head(8)
    losses = allc[allc["diff"] < 0].head(8)
    f = at_line(base, wide, LINE)
    sub = f[f.n_res > 0]
    out: dict[str, Any] = {
        "forbes_total": int(citizenship.billionaires.sum()), "residence_assigned": int(api.billionaires.sum()),
        "labels_citizenship": int(len(citizenship)), "labels_residence": int(len(api)),
        "gains": {r.country_label: int(r["diff"]) for _, r in gains.iterrows()},
        "losses": {r.country_label: int(r["diff"]) for _, r in losses.iterrows()},
        "panel_sum": {k: int(f[f"n_{k}"].sum()) for k in ("sec", "cit", "res")},
        "panel_n_differ": int((f.n_res != f.n_cit).sum()),
        "panel_zero_residence": sorted(f.loc[f.n_res == 0, "iso3"].tolist()),
        "n_sub": int(len(sub)),
        "corr_cit_res": pcorr(sub, "B_cit", "B_res", [])["r"],
        "corr_sec_cit": pcorr(f, "B_sec", "B_cit", [])["r"],
        "moves": {},
        "counts": {},
    }
    shift = np.log10((f.n_res + 0.5) / (f.n_cit + 0.5))
    for i in shift.abs().sort_values(ascending=False).index[:16]:
        out["moves"][f.loc[i, "iso3"]] = {"country": f.loc[i, "country"], "cit": int(f.loc[i, "n_cit"]),
                                          "res": int(f.loc[i, "n_res"]), "log10_shift": float(shift[i])}
    gdp_r2 = r2(sub, "poverty_pct", [G])
    for k in ("sec", "cit", "res"):
        col = f"B_{k}"
        raw = ols_hc3(sub, "poverty_pct", [col])
        joint = ols_hc3(sub, "poverty_pct", [G, col])
        jc = ols_hc3(sub, "poverty_pct", [G, col, "consumption"])
        fg = fractional_logit(sub, "poverty_pct", [G, col], col)
        fm = fractional_logit(sub, "poverty_pct", [MEAN, G, col], col)
        half = f"Bh_{k}"
        jh = ols_hc3(f, "poverty_pct", [G, half])
        out["counts"][k] = {
            "raw_r2": raw["r_squared"], "raw_slope": raw["coef"][col],
            "joint_coef": joint["coef"][col], "joint_p": joint["p"][col], "joint_ci95": joint["ci95"][col],
            "incremental_r2": joint["r_squared"] - gdp_r2,
            "type_coef": jc["coef"][col], "type_p": jc["p"][col],
            "frac_gdp": {"ame": fg["ame"], "ci95": fg["ci95"]},
            "frac_mean_gdp": {"ame": fm["ame"], "ci95": fm["ci95"]},
            "corr_gdp": pcorr(sub, col, G, [])["r"],
            "pc_mean": pcorr(sub, col, MEAN, [G, "consumption"]),
            "pc_wT1": pcorr(sub, col, "wT1", [G]),
            "half_joint_coef": jh["coef"][half], "half_joint_p": jh["p"][half],
        }
    return out


def timing(base: pd.DataFrame, wide: pd.DataFrame) -> dict[str, Any]:
    """WID shares at the survey year, and at the last observed year, instead of 2024."""
    f = at_line(base, wide, LINE)
    out = {}
    for col in ("wT1_sy", "wT1_lo"):
        g = ols_hc3(f, "poverty_pct", [G, col])
        m = ols_hc3(f, "poverty_pct", [MEAN, G, col])
        out[col] = {"corr_2024": pcorr(f, "wT1", col, [])["r"],
                    "mean_abs_diff": float((f.wT1 - f[col]).abs().mean()),
                    "b_raw": pcorr(f, B, col, [])["r"], "b_given_gdp": pcorr(f, B, col, [G]),
                    "gini": pcorr(f, col, "gini", [G, "consumption"]),
                    "gdp_coef": g["coef"][col], "gdp_p": g["p"][col],
                    "mean_coef": m["coef"][col], "mean_p": m["p"][col],
                    "mean_loo": loo_rmse_ols(f, "poverty_pct", [MEAN, G, col])}
    out["survey_years"] = [int(f.reporting_year.min()), int(f.reporting_year.max())]
    return out


def compute(panel: pd.DataFrame, surveys: pd.DataFrame, heads: pd.DataFrame, series: pd.DataFrame,
            meta: pd.DataFrame, wid_manifest: dict[str, Any], citizenship: pd.DataFrame,
            api: pd.DataFrame) -> dict[str, Any]:
    base, wide = build(panel, surveys, heads, series, citizenship, api)
    return {
        "n": int(len(base)),
        "wid": {"last_modified": wid_manifest["last_modified"], "archive_bytes": wid_manifest["archive_bytes"],
                "archive_sha256": wid_manifest["archive_sha256"], "year": WID_YEAR,
                "latest_year_by_variable": wid_manifest["latest_year_by_variable"],
                "n_countries": wid_manifest["n_countries"]},
        "concentration": concentration(base),
        "dissociation": dissociation(base),
        "imputation": imputation(base, meta),
        "poverty": poverty(base, wide),
        "lines": lines(base, wide),
        "residence": residence(base, wide, citizenship, api),
        "timing": timing(base, wide),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def frame(fig: plt.Figure, title: str, subtitle: str, **kw: Any) -> None:
    _frame(fig, title, subtitle, source=SOURCE, **kw)


def _density_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10, 30]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def fig_scatter(base: pd.DataFrame, n: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for observed, colour, label in ((True, BLUE, "Wealth share rests on observed data"),
                                    (False, ORANGE, "Wealth share imputed from income shares")):
        g = base[base.observed == observed]
        ax.scatter(10 ** g[B], g.wT1, s=52, color=colour, edgecolors=SURFACE, linewidths=1.2, zorder=3, label=label)
    names = {"USA": ("United States", (8, 2), "left"), "CHE": ("Switzerland", (8, 3), "left"),
             "NLD": ("Netherlands", (8, -3), "left"), "ZAF": ("South Africa", (8, -3), "left"),
             "RUS": ("Russia", (8, 2), "left"), "IND": ("India", (-8, 2), "right"),
             "SWE": ("Sweden", (8, -5), "left"), "BEL": ("Belgium", (8, -3), "left"),
             "NGA": ("Nigeria", (8, -3), "left"), "SVK": ("Slovakia", (8, -3), "left"),
             "CYP": ("Cyprus", (8, 3), "left")}
    for iso, (name, off, ha) in names.items():
        row = base.loc[base.iso3 == iso].iloc[0]
        ax.annotate(name, (10 ** row[B], row.wT1), xytext=off, textcoords="offset points",
                    fontsize=9, color=SECONDARY, va="center", ha=ha)
    lat = base[base.iso3.isin(["BRA", "CHL", "COL", "MEX"])]
    anchor = (1.6, 45.5)
    ax.text(*anchor, "Brazil, Chile, Colombia, Mexico: 39–40%", fontsize=9, color=SECONDARY, ha="center",
            va="bottom")
    for _, row in lat.iterrows():
        ax.plot([anchor[0], 10 ** row[B]], [anchor[1] - 0.3, row.wT1 + 0.9], color=BASELINE, lw=0.8, zorder=2)
    _density_axis(ax)
    ax.xaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10, 30, 100]))
    ax.set_xlim(0.12, 160)
    ax.set_ylim(10, 60)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_xlabel("Billionaires per 10 million people (citizenship, log scale)")
    ax.set_ylabel("Top 1% share of net personal wealth, 2024")
    ax.legend(loc="upper right")
    r = n["concentration"]["raw"]["wT1"]
    frame(fig, "More billionaires per person does not mean more concentrated wealth",
           f"Correlation {fmt(r['r'], 2)} (95% interval {fmt(r['ci95'][0], 2)} to {fmt(r['ci95'][1], 2)}) across 54 countries.",
           top=0.88, bottom=0.15, left=0.09)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_dissociation(n: dict[str, Any], out: Path) -> None:
    d = n["dissociation"]
    rows = [B, "wT1", "wT10", "iT10"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9), sharey=True)
    panels = ((MEAN, "Level: survey mean (log)"), ("gini", "Spread: survey Gini"))
    for ax, (y, title) in zip(axes, panels):
        for i, x in enumerate(rows):
            r = d[x][y]
            colour = BLUE if x == B else ORANGE
            ax.plot(r["ci95"], [i, i], color=colour, lw=2.4, solid_capstyle="round", zorder=2)
            ax.scatter([r["r"]], [i], s=70, color=colour, edgecolors=SURFACE, linewidths=1.5, zorder=3)
            ax.annotate(fmt(r["r"], 2, sign=True), (r["r"], i), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=9, color=SECONDARY)
        ax.axvline(0, color=BASELINE, lw=1.2, zorder=1)
        ax.set_xlim(-0.6, 1.0)
        ax.set_ylim(len(rows) - 0.5, -0.75)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("Partial correlation, given GDP and survey type")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([LABEL[x] for x in rows])
    frame(fig, "Billionaire density goes with the level of welfare, top shares with its spread",
           "Partial correlations with the PIP survey's mean and Gini, holding GDP per person and survey type fixed.\n"
           "Lines are 95% intervals; 54 countries.", top=0.82, bottom=0.18, left=0.2)
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_imputation(base: pd.DataFrame, n: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    im = n["imputation"]
    for observed, colour, label in ((True, BLUE, f"Observed wealth data ({im['n_observed']} countries)"),
                                    (False, ORANGE, f"Imputed ({im['n_imputed']} countries)")):
        g = base[base.observed == observed]
        ax.scatter(g.iT10, g.wT10, s=52, color=colour, edgecolors=SURFACE, linewidths=1.2, zorder=3, label=label)
    for iso, name, off in (("MEX", "Brazil, Chile, Colombia, Mexico", (-150, 11)), ("USA", "United States", (6, 4)),
                           ("ZAF", "South Africa", (-74, -2)), ("NLD", "Netherlands", (6, -4)),
                           ("JPN", "Japan", (5, -9)), ("SWE", "Sweden", (6, 4)), ("RUS", "Russia", (6, 3))):
        row = base.loc[base.iso3 == iso].iloc[0]
        ax.annotate(name, (row.iT10, row.wT10), xytext=off, textcoords="offset points", fontsize=9, color=SECONDARY)
    fmt_pct = FuncFormatter(lambda v, _: f"{v:g}%")
    ax.xaxis.set_major_formatter(fmt_pct)
    ax.yaxis.set_major_formatter(fmt_pct)
    ax.set_xlabel("Top 10% share of pre-tax national income, 2024")
    ax.set_ylabel("Top 10% share of net personal wealth, 2024")
    ax.legend(loc="upper left")
    r2i, r2o = im["r2_wT10_on_iT10"]["imputed"], im["r2_wT10_on_iT10"]["observed"]
    frame(fig, f"For {im['n_imputed']} of the 54 countries, WID's wealth shares follow their income shares",
           f"Where no wealth data exist, WID predicts the wealth distribution from the top 10% income share.\n"
           f"Income share explains {100 * r2i:.0f}% of the variation in imputed wealth shares, {100 * r2o:.0f}% in observed ones.",
           top=0.86, bottom=0.15, left=0.09)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_lines(n: dict[str, Any], out: Path) -> None:
    rows = n["lines"]
    xs = np.array([r["line"] for r in rows])
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    series = ((B, BLUE, "Billionaire density"), ("wT1", ORANGE, "Top 1% wealth share"),
              ("iT10", AQUA, "Top 10% income share"))
    for x, colour, label in series:
        key = f"{x}|mean_gdp"
        ame = np.array([r[key]["ame_sd"] for r in rows])
        lo = np.array([r[key]["ci95_sd"][0] for r in rows])
        hi = np.array([r[key]["ci95_sd"][1] for r in rows])
        ax.fill_between(xs, lo, hi, color=colour, alpha=0.13, lw=0, zorder=1)
        ax.plot(xs, ame, color=colour, lw=2, zorder=3, label=label)
        ax.scatter(xs, ame, s=16, color=colour, zorder=4)
    ax.axhline(0, color=BASELINE, lw=1.2, zorder=2)
    _line_axis(ax)
    top = 4.8
    ax.set_ylim(-3.2, top)
    _official_lines(ax, top - 0.1)
    ax.set_xlabel("Poverty line, 2021 PPP dollars per person per day (log scale)")
    ax.set_ylabel("Change in poverty per standard deviation (points)")
    ax.legend(loc="lower right")
    frame(fig, "Given the survey mean, the top shares behave like measures of spread",
           "Fractional-logit average marginal effect of one standard deviation of each measure, given GDP per person\n"
           "and the survey mean. Bands are 95% intervals; 54 countries; each measure in its own model.",
           top=0.86, bottom=0.16, left=0.09)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_residence(base: pd.DataFrame, n: dict[str, Any], out: Path) -> None:
    moves = n["residence"]["moves"]
    order = sorted(moves, key=lambda k: moves[k]["log10_shift"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.8), gridspec_kw={"width_ratios": [1.45, 1]})
    pop = base.set_index("iso3").population
    for i, iso in enumerate(order):
        m = moves[iso]
        c = (m["cit"] + 0.5) * 1e7 / pop[iso]
        r = (m["res"] + 0.5) * 1e7 / pop[iso]
        ax.plot([c, r], [i, i], color=BASELINE, lw=2, zorder=1)
        ax.scatter([c], [i], s=46, color=BLUE, zorder=3, edgecolors=SURFACE, linewidths=1,
                   label="Citizenship" if i == 0 else None)
        ax.scatter([r], [i], s=46, color=ORANGE, zorder=3, edgecolors=SURFACE, linewidths=1,
                   label="Residence" if i == 0 else None)
        ax.annotate(f"{m['cit']} → {m['res']}", (max(c, r), i), xytext=(7, 0), textcoords="offset points",
                    fontsize=8.5, color=MUTED, va="center")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([moves[k]["country"] for k in order], fontsize=9.5)
    _density_axis(ax)
    ax.set_xlim(0.07, 300)
    ax.xaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
    ax.set_xlabel("Billionaires per 10 million (count + 0.5, log scale)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left")
    ax.set_title("The 16 largest moves among the 54 countries", fontsize=10.5, color=INK, loc="left")
    counts = n["residence"]["counts"]
    labels = {"sec": "Citizenship,\nfirst article", "cit": "Citizenship,\nForbes list", "res": "Residence,\nForbes list"}
    for i, k in enumerate(("sec", "cit", "res")):
        c = counts[k]
        colour = ORANGE if k == "res" else BLUE
        ax2.plot(c["joint_ci95"], [i, i], color=colour, lw=2.4, solid_capstyle="round")
        ax2.scatter([c["joint_coef"]], [i], s=70, color=colour, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax2.annotate(fmt(c["joint_coef"]), (c["joint_coef"], i), xytext=(0, 9), textcoords="offset points",
                     ha="center", fontsize=9, color=SECONDARY)
    ax2.axvline(0, color=BASELINE, lw=1.2)
    ax2.set_yticks(range(3))
    ax2.set_yticklabels([labels[k] for k in ("sec", "cit", "res")], fontsize=9.5)
    ax2.invert_yaxis()
    ax2.set_ylim(2.6, -0.6)
    ax2.set_xlabel("Poverty points at $30 per tenfold")
    ax2.grid(axis="y", visible=False)
    ax2.set_title(f"Coefficient given GDP ({n['residence']['n_sub']} countries)", fontsize=10.5, color=INK, loc="left")
    frame(fig, "Counting billionaires where they live moves countries, not conclusions",
           "Left: billionaires by citizenship and by residence in Forbes's 2026 list. Right: the billionaire coefficient\n"
           "in the first article's model with each count. Lines are 95% intervals.",
           top=0.84, bottom=0.17, left=0.15)
    fig.subplots_adjust(wspace=0.55)
    fig.savefig(out, dpi=250)
    plt.close(fig)


FIGURE_FILES = ("fig1_density_vs_top1.png", "fig2_level_or_spread.png", "fig3_imputed_wealth_shares.png",
                "fig4_given_survey_mean_by_line.png", "fig5_residence_counts.png")


def render(base: pd.DataFrame, n: dict[str, Any], outdir: Path) -> None:
    _style()
    outdir.mkdir(parents=True, exist_ok=True)
    fig_scatter(base, n, outdir / FIGURE_FILES[0])
    fig_dissociation(n, outdir / FIGURE_FILES[1])
    fig_imputation(base, n, outdir / FIGURE_FILES[2])
    fig_lines(n, outdir / FIGURE_FILES[3])
    fig_residence(base, n, outdir / FIGURE_FILES[4])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pc(d: dict[str, Any]) -> str:
    return f"{fmt(d['r'], 3, sign=True)} [{fmt(d['ci95'][0], 2)}, {fmt(d['ci95'][1], 2)}] n={d['n']}"


def summary_lines(n: dict[str, Any]) -> list[str]:
    c, d, im, pv, rs, tm = (n[k] for k in ("concentration", "dissociation", "imputation", "poverty", "residence", "timing"))
    out = [f"WID bulk file {n['wid']['last_modified']} ({n['wid']['archive_bytes']} bytes), shares for {n['wid']['year']}; "
           f"{n['n']} countries",
           "Raw correlations with billionaire density: " + "; ".join(f"{k} {fmt(v['r'], 3)}" for k, v in c["raw"].items()),
           f"Top 1% wealth share: median {c['wT1_summary']['median']:.1f}, range {c['wT1_summary']['min']:.1f} "
           f"({c['wT1_summary']['min_iso3']}) to {c['wT1_summary']['max']:.1f} ({c['wT1_summary']['max_iso3']})"]
    for y, g in c["given_gdp"].items():
        out.append(f"Partial corr B with {y} | GDP: all {_pc(g['all'])}; observed {_pc(g['observed'])}; imputed {_pc(g['imputed'])}")
    for x, ys in d.items():
        if x != "gini_corr_raw":
            out.append(f"{x} | GDP, type: mean {_pc(ys[MEAN])}; gini {_pc(ys['gini'])}; bottom40 {_pc(ys['bottom40'])}")
    out.append(f"Raw corr with PIP Gini: {d['gini_corr_raw']}")
    out.append(f"WID status: {im['n_observed']} observed (last observed {im['last_obs_min']}-{im['last_obs_max']}, "
               f"median {im['last_obs_median']}), {im['n_imputed']} imputed: {', '.join(im['imputed_iso3'])}")
    out.append(f"R2 wT10~iT10 observed {im['r2_wT10_on_iT10']['observed']:.3f} imputed {im['r2_wT10_on_iT10']['imputed']:.3f}; "
               f"wT1~iT10 observed {im['r2_wT1_on_iT10']['observed']:.3f} imputed {im['r2_wT1_on_iT10']['imputed']:.3f}; "
               f"LatAm {im['latam']}; grades {im['grade_counts']}")
    for key, title in (("gdp", "Given GDP"), ("mean_gdp", "Given survey mean and GDP")):
        m = pv[key]
        out.append(f"$30 {title}: baseline R2 {m['baseline']['r_squared']:.3f} LOO {m['baseline']['loo_ols']:.2f} "
                   f"fracLOO {m['baseline']['loo_frac']:.2f}")
        for x, r in m.items():
            if x != "baseline":
                out.append(f"   + {x}: coef {fmt(r['coef'], 2)} p {r['p']:.3f} R2 {r['r_squared']:.3f} LOO {r['loo_ols']:.2f} "
                           f"fracLOO {r['loo_frac']:.2f} AME {fmt(r['ame'], 3)} [{fmt(r['ame_ci95'][0], 3)}, "
                           f"{fmt(r['ame_ci95'][1], 3)}] p {r['ame_p']:.3f} per SD {fmt(r['ame_sd'], 2)}")
    out.append(f"GDP + B + wT1: {pv['gdp_both']}; consumption given mean+GDP: {pv['mean_gdp_consumption']}")
    for key in ("gdp", "mean_gdp"):
        for x, r in pv["observed_only"][key].items():
            if x == "baseline":
                out.append(f"Observed only ({pv['observed_only']['n']}) {key}: R2 {r['r_squared']:.3f} LOO {r['loo_ols']:.2f}")
            else:
                out.append(f"   + {x}: coef {fmt(r['coef'], 2)} p {r['p']:.3f} LOO {r['loo_ols']:.2f}")
    for row in n["lines"]:
        out.append(f"${row['line']:g}: " + "; ".join(
            f"{k} {fmt(v['ame_sd'], 2)} p {v['p']:.3f}" for k, v in row.items() if k != "line"))
    out.append(f"Forbes: {rs['forbes_total']} by citizenship ({rs['labels_citizenship']} labels), {rs['residence_assigned']} with a "
               f"country field ({rs['labels_residence']}); gains {rs['gains']}; losses {rs['losses']}")
    out.append(f"Panel sums {rs['panel_sum']}; {rs['panel_n_differ']} differ; zero residence {rs['panel_zero_residence']}; "
               f"corr cit/res {rs['corr_cit_res']:.4f}; sec/cit {rs['corr_sec_cit']:.4f}")
    for k, v in rs["counts"].items():
        out.append(f"{k} (n={rs['n_sub']}): raw R2 {v['raw_r2']:.3f} slope {fmt(v['raw_slope'])} | joint {fmt(v['joint_coef'], 2)} "
                   f"p {v['joint_p']:.3f} [{fmt(v['joint_ci95'][0], 1)}, {fmt(v['joint_ci95'][1], 1)}] incR2 {v['incremental_r2']:.4f} | "
                   f"+type {fmt(v['type_coef'], 2)} p {v['type_p']:.3f} | AME|GDP {fmt(v['frac_gdp']['ame'], 2)} "
                   f"AME|mean {fmt(v['frac_mean_gdp']['ame'], 2)} | corr GDP {v['corr_gdp']:.3f} | pc mean {_pc(v['pc_mean'])} | "
                   f"pc wT1 {_pc(v['pc_wT1'])} | +0.5 on 54: {fmt(v['half_joint_coef'], 2)} p {v['half_joint_p']:.3f}")
    for k in ("wT1_sy", "wT1_lo"):
        v = tm[k]
        out.append(f"{k}: corr 2024 {v['corr_2024']:.3f}, mean |diff| {v['mean_abs_diff']:.2f}; B raw {fmt(v['b_raw'], 3)} "
                   f"B|GDP {_pc(v['b_given_gdp'])}; gini {_pc(v['gini'])}; given GDP {fmt(v['gdp_coef'], 2)} p {v['gdp_p']:.3f}; "
                   f"given mean {fmt(v['mean_coef'], 2)} p {v['mean_p']:.3f} LOO {v['mean_loo']:.2f}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--figures", type=Path, default=FIGURES)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    panel = load_panel(p1.PANEL)
    surveys, heads, _ = load_pip()
    series, meta, wid_manifest = load_wid()
    citizenship, api = load_forbes()
    numbers = json.loads(json.dumps(compute(panel, surveys, heads, series, meta, wid_manifest, citizenship, api)))
    lines_out = summary_lines(numbers)
    print("\n".join(lines_out))
    target = RESULTS / "numbers.json"
    if args.check:
        problems: list[str] = []
        _compare(numbers, json.loads(target.read_text(encoding="utf-8")), "numbers", problems)
        if problems:
            print("\nREPRODUCTION FAILED:\n" + "\n".join(problems[:50]), file=sys.stderr)
            return 1
        print("\nREPRODUCED: every number matches results/part2/numbers.json")
    else:
        RESULTS.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (RESULTS / "summary.txt").write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    if not args.no_figures:
        base, _ = build(panel, surveys, heads, series, citizenship, api)
        render(base, numbers, args.figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
