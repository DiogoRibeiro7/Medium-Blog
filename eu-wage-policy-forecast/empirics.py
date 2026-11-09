"""Estimates the forecast needs from the data snapshot, with their uncertainty as bootstrap draws.

    python empirics.py

1. How the bottom of the distribution follows the minimum wage. Structure of Earnings Survey (SES)
   waves 2006-2022, full-time employees, monthly earnings in October. The elasticity of P10/median
   with respect to the minimum wage/median (Kaitz index), with country and wave fixed effects:
   linear, and quadratic in log Kaitz (the response grows as the floor nears the middle). Portugal's
   own slope over its five waves. The elasticity of the low-pay share. Bootstrap over countries.
2. How the median follows the minimum wage. Growth of the SES median between waves against growth
   of national-accounts compensation per employee and of the minimum wage, across countries; and
   Portugal's own drift. Two backtests: the SES Kaitz index of 2022 forecast from 2018, and the Banco
   de Portugal base-wage Kaitz index of 2025 forecast from 2019.
3. How uncertain wage growth is. An AR(1) for Portuguese compensation per employee, 1996-2025.
4. Who the transparency duties reach. Employees in enterprises above 250, 150 and 50 persons
   employed (business statistics by size class).
5. The EU reference value. Each country's Kaitz index (OECD, median, 2025) against 60%.

Writes results/empirics_summary.txt (every number), results/empirics.json (point estimates the
forecast reads), results/elasticity_draws.csv, results/spillover_draws.csv, results/coverage.csv and
results/eu_reference.csv and the SES panel (results/ses_panel*.csv).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from inputs import KAITZ_BASE, MW_HISTORY, REFERENCE_KAITZ

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"

ISO3 = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "HR": "HRV", "CY": "CYP", "CZ": "CZE", "DK": "DNK", "EE": "EST",
    "FI": "FIN", "FR": "FRA", "DE": "DEU", "EL": "GRC", "HU": "HUN", "IE": "IRL", "IT": "ITA", "LV": "LVA",
    "LT": "LTU", "LU": "LUX", "MT": "MLT", "NL": "NLD", "PL": "POL", "PT": "PRT", "RO": "ROU", "SK": "SVK",
    "SI": "SVN", "ES": "ESP", "SE": "SWE",
}
EU27 = list(ISO3)
FOURTEEN_PAYMENTS = {"EL", "ES", "PT"}       # Eurostat converts these to a 12-month basis (x 14 / 12)
SES_WAVES = (2006, 2010, 2014, 2018, 2022)
N_BOOT = 2000


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv", low_memory=False)


# ------------------------------------------------------------------------------------------- data
def minimum_wage_as_paid() -> pd.DataFrame:
    """Monthly minimum wage in euro as paid (14-payment countries converted back), by semester."""
    mw = read("minimum_wage_eur")
    mw = mw[mw.geo.isin(EU27)].copy()
    mw["year"] = mw.time.str[:4].astype(int)
    mw["semester"] = mw.time.str[-2:]
    mw["mw_paid"] = np.where(mw.geo.isin(FOURTEEN_PAYMENTS), mw.value * 12 / 14, mw.value)
    return mw[["geo", "year", "semester", "mw_paid"]]


def ses_panel() -> pd.DataFrame:
    mw = minimum_wage_as_paid()
    mw_oct = mw[mw.semester == "S2"].set_index(["geo", "year"])["mw_paid"]
    rows = []
    for y in SES_WAVES:
        s = read(f"ses_deciles_{y}")
        qcol = "quantile" if "quantile" in s.columns else "quant_inc"
        s = s[(s.worktime == "FT") & s.geo.isin(EU27)]
        eur = s[s.unit == "EUR"].pivot_table(index="geo", columns=qcol, values="value")
        pc = s[(s.unit == "PC") & (s[qcol] == "S2")].set_index("geo")["value"]
        for geo, r in eur.iterrows():
            if not {"P10", "MED", "P90"} <= set(r.dropna().index):
                continue
            rows.append({"geo": geo, "wave": y, "p10": r.P10, "median": r.MED, "p90": r.P90,
                         "low_pay_share": pc.get(geo, np.nan), "mw_paid": mw_oct.get((geo, y), np.nan)})
    p = pd.DataFrame(rows)
    p["kaitz"] = p.mw_paid / p["median"]
    p["p10_over_median"] = p.p10 / p["median"]
    return p


def compensation_per_employee() -> pd.DataFrame:
    """National accounts: compensation of employees (D1) per employee, euro per month."""
    na = read("national_accounts_cp").pivot_table(index=["geo", "time"], columns="na_item", values="value")
    emp = read("employment_persons").pivot_table(index=["geo", "time"], columns="na_item", values="value")
    x = na.join(emp)
    x["comp_per_employee"] = x.D1 / x.SAL_DC * 1e3 / 12      # million euro / thousand persons -> euro a month
    return x.reset_index().rename(columns={"time": "year"})[["geo", "year", "comp_per_employee", "SAL_DC"]]


# ------------------------------------------------------------------------------- fixed effects
def _fe_coef(df: pd.DataFrame, y: str, xs: list[str], fe: tuple[str, ...] = ("geo", "wave")) -> np.ndarray:
    """OLS coefficients on xs with dummies for each fixed effect (exact two-way within estimator)."""
    blocks = [df[xs].to_numpy(float)]
    for k, f in enumerate(fe):
        d = pd.get_dummies(df[f]).to_numpy(float)
        blocks.append(d if k == 0 else d[:, 1:])
    coef, *_ = np.linalg.lstsq(np.column_stack(blocks), df[y].to_numpy(float), rcond=None)
    return coef[: len(xs)]


def _country_bootstrap(df: pd.DataFrame, fn, n_boot: int = N_BOOT, seed: int = 1) -> np.ndarray:
    """Resample countries with replacement (a resampled country gets its own fixed effect)."""
    rng = np.random.default_rng(seed)
    geos = df.geo.unique()
    by_geo = {g: df[df.geo == g] for g in geos}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(geos, size=geos.size, replace=True)
        bd = pd.concat([by_geo[g].assign(geo=f"{g}_{k}") for k, g in enumerate(pick)], ignore_index=True)
        out.append(fn(bd))
    return np.asarray(out)


def elasticities(p: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    d = p.dropna(subset=["kaitz", "p10_over_median"]).copy()
    d = d[d.groupby("geo").geo.transform("size") >= 2]
    d["lk"] = np.log(d.kaitz)
    d["lk2"] = d.lk ** 2
    d["ly"] = np.log(d.p10_over_median)
    lin = lambda df: _fe_coef(df, "ly", ["lk"])                       # noqa: E731
    quad = lambda df: _fe_coef(df, "ly", ["lk", "lk2"])               # noqa: E731
    b_lin, b_quad = lin(d), quad(d)
    draws_lin = _country_bootstrap(d, lin)[:, 0]
    draws_quad = _country_bootstrap(d, quad)
    # low-pay share (below two thirds of the median), where positive
    lp = d[d.low_pay_share > 0].copy()
    lp = lp[lp.groupby("geo").geo.transform("size") >= 2]
    lp["ls"] = np.log(lp.low_pay_share)
    low = lambda df: _fe_coef(df, "ls", ["lk"])                       # noqa: E731
    b_low = low(lp)
    draws_low = _country_bootstrap(lp, low)[:, 0]
    # Portugal alone: five waves, slope of log P10/median on log Kaitz
    pt = d[d.geo == "PT"].sort_values("wave")
    X = np.column_stack([np.ones(len(pt)), pt.lk])
    coef, res, *_ = np.linalg.lstsq(X, pt.ly, rcond=None)
    resid = pt.ly - X @ coef
    s2 = float(resid @ resid) / (len(pt) - 2)
    se_pt = float(np.sqrt(s2 / ((pt.lk - pt.lk.mean()) ** 2).sum()))
    rng = np.random.default_rng(2)
    draws_pt = coef[1] + se_pt * rng.standard_t(len(pt) - 2, N_BOOT)
    est = {
        "lin": float(b_lin[0]), "quad_b1": float(b_quad[0]), "quad_b2": float(b_quad[1]),
        "lowpay": float(b_low[0]), "pt_own": float(coef[1]), "pt_own_se": se_pt,
        "n_obs": int(len(d)), "n_countries": int(d.geo.nunique()), "n_obs_lowpay": int(len(lp)),
        "kaitz_min": float(d.kaitz.min()), "kaitz_max": float(d.kaitz.max()),
    }
    draws = pd.DataFrame({"lin": draws_lin, "quad_b1": draws_quad[:, 0], "quad_b2": draws_quad[:, 1],
                          "lowpay": draws_low, "pt_own": draws_pt})
    return est, draws


def local_elasticity(b1: float | np.ndarray, b2: float | np.ndarray, kaitz: float) -> float | np.ndarray:
    return b1 + 2 * b2 * np.log(kaitz)


# ------------------------------------------------------------------------------- the median
def spillover(p: pd.DataFrame, cpe: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """dlog median - dlog avg = theta (dlog MW - dlog avg) + wave effect, between SES waves."""
    x = p.merge(cpe.rename(columns={"year": "wave"})[["geo", "wave", "comp_per_employee"]], on=["geo", "wave"], how="left")
    x = x.sort_values(["geo", "wave"])
    g = x.groupby("geo")
    x["dmed"] = np.log(x["median"]) - np.log(g["median"].shift())
    x["davg"] = np.log(x.comp_per_employee) - np.log(g.comp_per_employee.shift())
    x["dmw"] = np.log(x.mw_paid) - np.log(g.mw_paid.shift())
    x["gap"] = x.wave - g.wave.shift()
    d = x.dropna(subset=["dmed", "davg", "dmw"]).copy()
    d = d[d.gap == 4]
    d["y"] = d.dmed - d.davg
    d["x"] = d.dmw - d.davg

    def fit(df: pd.DataFrame) -> np.ndarray:
        X = np.column_stack([df.x.to_numpy(float), pd.get_dummies(df.wave).to_numpy(float)])
        coef, *_ = np.linalg.lstsq(X, df.y.to_numpy(float), rcond=None)
        return coef

    coef = fit(d)
    theta = float(coef[0])
    waves = sorted(d.wave.unique())
    wave_fx = dict(zip(waves, coef[1:]))
    d["resid"] = d.y - theta * d.x - d.wave.map(wave_fx)
    # Portugal's drift: the median against compensation per employee, net of the spillover, per year.
    # Wave effects are common to all countries; Portugal's drift is its own mean gap, wave effects included.
    pt = d[d.geo == "PT"]
    pt_gap = pt.y - theta * pt.x
    drift = float(pt_gap.mean() / 4)
    drift_se = float(pt_gap.std(ddof=1) / np.sqrt(len(pt_gap)) / 4)
    boot = _country_bootstrap(d, lambda df: fit(df)[:1])[:, 0]
    rng = np.random.default_rng(3)
    drift_draws = drift + drift_se * rng.standard_t(len(pt_gap) - 1, N_BOOT)
    est = {"theta": theta, "theta_ci90": [float(np.percentile(boot, 5)), float(np.percentile(boot, 95))],
           "drift_pt": drift, "drift_pt_se": drift_se, "n_intervals": int(len(d)),
           "pt_intervals": pt[["wave", "dmed", "davg", "dmw"]].round(4).to_dict("records")}
    return est, pd.DataFrame({"theta": boot, "drift": drift_draws}), d


def median_growth(g_avg: float | np.ndarray, g_mw: float | np.ndarray, theta, drift) -> float | np.ndarray:
    """Log growth of the median given log growth of average compensation and of the minimum wage."""
    return g_avg + theta * (g_mw - g_avg) + drift


def backtests(d: pd.DataFrame, p: pd.DataFrame, cpe: pd.DataFrame, theta: float, drift: float) -> dict:
    out = {}
    # (a) SES Kaitz for Portugal, 2018 -> 2022, with theta and drift estimated without the 2022 wave
    d18 = d[d.wave < 2022]
    X = np.column_stack([d18.x.to_numpy(float), pd.get_dummies(d18.wave).to_numpy(float)])
    th18 = float(np.linalg.lstsq(X, d18.y.to_numpy(float), rcond=None)[0][0])
    pt18 = d18[d18.geo == "PT"]
    dr18 = float((pt18.y - th18 * pt18.x).mean() / 4)
    pt = p[p.geo == "PT"].set_index("wave")
    a = cpe[cpe.geo == "PT"].set_index("year").comp_per_employee
    g_avg = np.log(a[2022] / a[2018])
    g_mw = np.log(pt.mw_paid[2022] / pt.mw_paid[2018])
    k_pred = pt.kaitz[2018] * np.exp(g_mw - median_growth(g_avg, g_mw, th18, 4 * dr18))
    out["ses_2022_from_2018"] = {"predicted": float(k_pred), "actual": float(pt.kaitz[2022]),
                                 "naive_avg_growth": float(pt.kaitz[2018] * np.exp(g_mw - g_avg)),
                                 "theta_2006_2018": th18, "drift_2006_2018": dr18}
    # (b) Banco de Portugal base-wage Kaitz, 2019 -> 2025, with the full-sample estimates
    g_avg = np.log(a[2025] / a[2019])
    g_mw = np.log(870.0 / 600.0)
    k_pred = KAITZ_BASE[2019] * np.exp(g_mw - median_growth(g_avg, g_mw, theta, 6 * drift))
    out["base_2025_from_2019"] = {"predicted": float(k_pred), "actual": KAITZ_BASE[2025],
                                  "naive_avg_growth": float(KAITZ_BASE[2019] * np.exp(g_mw - g_avg))}
    return out


# --------------------------------------------------------------------------- wage-growth noise
def growth_process(cpe: pd.DataFrame) -> dict:
    a = cpe[cpe.geo == "PT"].set_index("year").comp_per_employee.sort_index()
    g = np.log(a).diff().dropna()
    g = g[g.index >= 1996]
    y, x = g.values[1:], g.values[:-1]
    X = np.column_stack([np.ones_like(x), x])
    c, rho = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - (c + rho * x)
    return {"rho": float(rho), "sigma": float(resid.std(ddof=2)), "years": [int(g.index[0]), int(g.index[-1])],
            "observed": {int(k): float(v) for k, v in g[g.index >= 2019].items()}}


# ------------------------------------------------------------------------------------ coverage
def coverage() -> pd.DataFrame:
    """Share of employees in enterprises with at least 250, 150 and 50 persons employed.

    Persons employed by size class include owners and unpaid family workers, who are concentrated in
    enterprises with fewer than 10 persons; employees above 10 persons are taken as persons employed,
    and the 0-9 class as the remainder of total employees. The 150 threshold falls inside the 50-249
    class: the share above it is interpolated with a Pareto tail fitted to the shares above 50 and 250."""
    s = read("sbs_size_class")
    rows = []
    for geo in ("PT", "EU27_2020", "ES", "DE", "FR", "IT", "PL"):
        x = s[(s.geo == geo) & (s.indic_sbs == "EMP_NR")].pivot_table(index="time", columns="size_emp", values="value")
        sal = s[(s.geo == geo) & (s.indic_sbs == "SAL_NR") & (s.size_emp == "TOTAL")].set_index("time").value
        x = x.dropna(subset=["50-249", "GE250"])
        years = [y for y in x.index if y in sal.index]
        if not years:
            continue
        y = max(years)
        r = x.loc[y]
        employees = float(sal[y])
        above10 = float(r["10-19"] + r["20-49"] + r["50-249"] + r["GE250"])
        s250 = float(r["GE250"]) / employees
        s50 = float(r["50-249"] + r["GE250"]) / employees
        slope = np.log(s50 / s250) / np.log(50 / 250)            # share above s is proportional to s**slope
        s150 = s250 * (150 / 250) ** slope
        rows.append({"geo": geo, "year": int(y), "employees": employees, "employees_0_9": employees - above10,
                     "share_ge250": s250, "share_ge150": float(s150), "share_ge50": s50, "pareto_slope": float(slope)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ EU reference
def eu_reference(p: pd.DataFrame) -> pd.DataFrame:
    om = read("oecd_min_to_average")
    inv = {v: k for k, v in ISO3.items()}
    om = om[om.REF_AREA.isin(inv) & (om.AGGREGATION_OPERATION == "MEDIAN")].copy()
    om["geo"] = om.REF_AREA.map(inv)
    last = om.sort_values("TIME_PERIOD").groupby("geo").tail(1)
    ses22 = p[p.wave == 2022].set_index("geo")
    rows = []
    for r in last.itertuples():
        k = r.OBS_VALUE / 100
        s = ses22.loc[r.geo] if r.geo in ses22.index else None
        rows.append({"geo": r.geo, "year": int(r.TIME_PERIOD), "kaitz_oecd": k,
                     "dlog_to_reference": float(np.log(REFERENCE_KAITZ / k)),
                     "kaitz_ses_2022": float(s.kaitz) if s is not None else np.nan,
                     "p10_over_median_2022": float(s.p10_over_median) if s is not None else np.nan,
                     "low_pay_share_2022": float(s.low_pay_share) if s is not None else np.nan})
    return pd.DataFrame(rows).sort_values("kaitz_oecd").reset_index(drop=True)


def gender_gap() -> dict:
    g = read("gender_pay_gap")
    g = g[g.geo.isin(["PT", "EU27_2020"])]
    last = g.sort_values("time").groupby("geo").tail(1)
    return {r.geo: {"year": int(r.time), "gap_pct": float(r.value)} for r in last.itertuples()}


# ------------------------------------------------------------------------------------------ main
def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    p = ses_panel()
    p.to_csv(RESULTS / "ses_panel.csv", index=False)
    p[p.geo == "PT"].to_csv(RESULTS / "ses_panel_pt.csv", index=False)
    cpe = compensation_per_employee()
    L = ["# Empirical inputs to the forecast (empirics.py)", ""]

    # check that the snapshot's minimum wage matches the history used in inputs.py
    mw = minimum_wage_as_paid()
    pt_mw = mw[(mw.geo == "PT") & (mw.semester == "S1")].set_index("year").mw_paid.round(0)
    for y, v in MW_HISTORY.items():
        assert abs(pt_mw[y] - v) < 1, (y, pt_mw[y], v)
    L.append("## Portugal minimum wage as paid (January, 14 payments): " +
             ", ".join(f"{y} {pt_mw[y]:.0f}" for y in MW_HISTORY))

    est_e, draws_e = elasticities(p)
    draws_e.to_csv(RESULTS / "elasticity_draws.csv", index=False)
    ci = lambda a: f"[{np.percentile(a, 5):.3f}, {np.percentile(a, 95):.3f}]"  # noqa: E731
    L += ["", "## 1. The bottom: P10/median against the Kaitz index (SES, country and wave fixed effects)",
          f"- Linear elasticity {est_e['lin']:.3f} {ci(draws_e.lin)}, {est_e['n_obs']} obs, {est_e['n_countries']} countries; "
          f"Kaitz range in the sample {est_e['kaitz_min']:.3f}-{est_e['kaitz_max']:.3f}",
          f"- Quadratic in log Kaitz: b1 {est_e['quad_b1']:.3f}, b2 {est_e['quad_b2']:.3f}; local elasticity at Kaitz "
          + ", ".join(f"{k:.2f}: {local_elasticity(est_e['quad_b1'], est_e['quad_b2'], k):.3f} "
                      f"{ci(local_elasticity(draws_e.quad_b1, draws_e.quad_b2, k))}" for k in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70)),
          f"- Portugal's own slope over five waves: {est_e['pt_own']:.3f} (s.e. {est_e['pt_own_se']:.3f}) {ci(draws_e.pt_own)}",
          f"- Low-pay share elasticity: {est_e['lowpay']:.3f} {ci(draws_e.lowpay)}, {est_e['n_obs_lowpay']} obs"]

    est_s, draws_s, d = spillover(p, cpe)
    draws_s.to_csv(RESULTS / "spillover_draws.csv", index=False)
    L += ["", "## 2. The median: growth between SES waves against compensation per employee and the minimum wage",
          f"- Spillover theta {est_s['theta']:.3f} [{est_s['theta_ci90'][0]:.3f}, {est_s['theta_ci90'][1]:.3f}], "
          f"{est_s['n_intervals']} four-year intervals",
          f"- Portugal's drift of the median against compensation per employee, net of the spillover: "
          f"{100 * est_s['drift_pt']:.2f} pt a year (s.e. {100 * est_s['drift_pt_se']:.2f})",
          "- Portugal intervals (wave, dlog median, dlog compensation per employee, dlog minimum wage): " +
          "; ".join(f"{r['wave']}: {r['dmed']:.3f}/{r['davg']:.3f}/{r['dmw']:.3f}" for r in est_s["pt_intervals"])]
    bt = backtests(d, p, cpe, est_s["theta"], est_s["drift_pt"])
    a, b = bt["ses_2022_from_2018"], bt["base_2025_from_2019"]
    L += [f"- Backtest, SES Kaitz 2022 from 2018 (estimates from 2006-2018 only, theta {a['theta_2006_2018']:.3f}, "
          f"drift {100 * a['drift_2006_2018']:.2f} pt): predicted {a['predicted']:.3f}, actual {a['actual']:.3f}, "
          f"median growing like compensation per employee {a['naive_avg_growth']:.3f}",
          f"- Backtest, Banco de Portugal base-wage Kaitz 2025 from 2019: predicted {b['predicted']:.3f}, "
          f"actual {b['actual']:.3f}, median growing like compensation per employee {b['naive_avg_growth']:.3f}"]

    gp = growth_process(cpe)
    L += ["", "## 3. Wage-growth noise: AR(1) of log growth of compensation per employee, Portugal",
          f"- {gp['years'][0]}-{gp['years'][1]}: rho {gp['rho']:.3f}, innovation sd {100 * gp['sigma']:.2f} pt",
          "- Observed log growth: " + ", ".join(f"{y} {100 * v:.2f}" for y, v in gp["observed"].items())]

    cov = coverage()
    cov.to_csv(RESULTS / "coverage.csv", index=False)
    L += ["", "## 4. Employees in enterprises above the transparency thresholds (business economy, B-S excl. O and S94)"]
    for r in cov.itertuples():
        L.append(f"- {r.geo} {r.year}: {r.employees / 1e6:.2f} m employees; >=250 {100 * r.share_ge250:.1f}%, "
                 f">=150 {100 * r.share_ge150:.1f}% (interpolated), >=50 {100 * r.share_ge50:.1f}%")

    eu = eu_reference(p)
    eu.to_csv(RESULTS / "eu_reference.csv", index=False)
    below = eu[eu.kaitz_oecd < REFERENCE_KAITZ]
    L += ["", f"## 5. Kaitz index against the {100 * REFERENCE_KAITZ:.0f}% reference value (OECD, median, full-time)",
          "- " + "; ".join(f"{r.geo} {100 * r.kaitz_oecd:.1f} ({r.year})" for r in eu.itertuples()),
          f"- {len(below)} of {len(eu)} countries below the reference; median gap {100 * below.dlog_to_reference.median():.1f} log points"]

    gg = gender_gap()
    L += ["", "## 6. Unadjusted gender pay gap (Eurostat, hourly, B-S excl. O)",
          "- " + "; ".join(f"{k} {v['gap_pct']:.1f}% ({v['year']})" for k, v in gg.items())]

    sal = cpe[(cpe.geo == "PT")].set_index("year").SAL_DC
    out = {"elasticity": est_e, "spillover": est_s, "backtests": bt, "growth": gp,
           "pt_employees_2025": float(sal[2025] * 1e3),
           "pt_ses_2022": p[(p.geo == "PT") & (p.wave == 2022)].iloc[0][["p10", "median", "p90", "mw_paid", "kaitz",
                                                                          "p10_over_median", "low_pay_share"]].astype(float).to_dict(),
           "pt_comp_per_employee": {int(k): float(v) for k, v in cpe[cpe.geo == "PT"].set_index("year").comp_per_employee.items()
                                    if k >= 2018},
           "gender_gap": gg}
    (RESULTS / "empirics.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    (RESULTS / "empirics_summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
