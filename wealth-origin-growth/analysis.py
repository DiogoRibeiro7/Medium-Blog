"""Inherited and self-made billionaire wealth and subsequent growth, within countries.

    python analysis.py

Writes results/panel.csv, results/specs.csv, results/joint_test.csv, results/placebo.csv,
results/wid_check.csv, results/forbes_quality.csv and results/summary.txt.

Design
------
For every country with at least one million people in Penn World Table 11.0 and a World Bank
GDP figure, and every Forbes list year t from 2011 to 2023, billionaire wealth is split by
Forbes' self-made flag into inherited (flag false) and self-made (flag true) and expressed as a
percentage of the previous year's GDP in current dollars (the list is compiled early in year t).
Countries without billionaires enter with zeros. The outcome is the annualised growth of real
GDP per capita or TFP from t to t + h. The question is whether, within a country, a larger
inherited stock predicts slower subsequent growth than a self-made stock of the same size.

The difference is estimated directly by regressing growth on inherited wealth and on total
billionaire wealth: the coefficient on inherited wealth is then b_inherited - b_self_made.

Specifications: horizon h in {1, 3, 5} years x outcome {GDP per capita, TFP} x transform
{log(1 + % of GDP), % of GDP} x sample {all countries; countries with billionaires in at least
seven of the thirteen years; all but financial centres} x controls {none; log income level at
the start of the window} x estimator {country and year fixed effects; year effects only}:
144 fits. Standard errors are clustered by country (CR1). A null-imposed joint test across all
144 (Simonsohn, Simmons and Nelson 2020) resamples countries jointly, 1,000 draws.

Placebo: the same fixed-effects regression with past growth (t - h to t) on the left-hand side.

WID check: Conjecture 1 of the first article in its literal form. In country-years with a
source-based WID top-1% wealth share (data quality above zero), does the association between
that share and growth over the next three years depend on the inherited share of the country's
billionaire wealth? Country and year fixed effects; clusters by country.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "results"

YEARS = range(2011, 2024)          # list years with a usable self-made flag and PWT growth after
OFFSHORE = {"HKG", "SGP", "CHE", "CYP", "LUX", "MCO", "LIE", "BMU", "MAC", "IRL", "SWZ"}
PUBLISHED_LIST_SIZE = {2002: 497, 2008: 1125}  # Forbes' own published counts, for the quality check

LABEL_FIX = {
    "Czech Republic": "CZE", "Egypt": "EGY", "Eswatini (Swaziland)": "SWZ", "Swaziland": "SWZ",
    "Hong Kong": "HKG", "Macao": "MAC", "Macau": "MAC", "Russia": "RUS", "Slovakia": "SVK",
    "South Korea": "KOR", "Taiwan": "TWN", "Turkey": "TUR", "Venezuela": "VEN", "Vietnam": "VNM",
    "Guernsey": "GGY",
}


# ---------------------------------------------------------------------------------------
# data


def load_countries() -> pd.DataFrame:
    c = pd.read_csv(DATA / "wdi_countries.csv", keep_default_na=False)
    return c[c["income_group"] != "Aggregates"]


def forbes_iso3(countries: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    f = pd.read_csv(DATA / "forbes_citizenship_year.csv", keep_default_na=False, na_values=[""])
    f = f[f["year"].isin(YEARS) | (f["year"] > max(YEARS))]
    names = dict(zip(countries["name"].str.lower(), countries["iso3"]))
    f["iso3"] = [LABEL_FIX.get(c, names.get(c.lower())) for c in f["citizenship"]]
    unmapped = sorted(f.loc[f["iso3"].isna(), "citizenship"].unique())
    f = f.dropna(subset=["iso3"])
    g = f.groupby(["iso3", "year"], as_index=False)[
        ["n", "worth_m", "worth_flag_selfmade_m", "worth_flag_inherited_m", "worth_flag_missing_m"]].sum()
    return g, unmapped


def build_panel(countries: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    pwt = pd.read_csv(DATA / "pwt110_panel.csv").rename(columns={"countrycode": "iso3"})
    pwt["gdppc"] = pwt["rgdpna"] / pwt["pop"]
    pwt["level_pc"] = pwt["rgdpe"] / pwt["pop"]
    p = pwt.set_index(["iso3", "year"])
    gdp = pd.read_csv(DATA / "wdi_gdp_usd.csv").set_index(["iso3", "year"])["gdp_usd"]
    f, unmapped = forbes_iso3(countries)
    f = f.set_index(["iso3", "year"])

    big = pwt.loc[(pwt["year"] == 2011) & (pwt["pop"] >= 1.0), "iso3"]
    iso = sorted(set(big) & set(countries["iso3"]))
    rows = []
    for i in iso:
        for t in YEARS:
            g_prev = gdp.get((i, t - 1), np.nan)
            if not np.isfinite(g_prev) or (i, t) not in p.index:
                continue
            r = {"iso3": i, "year": t, "gdp_prev_usd": g_prev,
                 "ln_y": np.log(p.loc[(i, t), "level_pc"])}
            if (i, t) in f.index:
                rec = f.loc[(i, t)]
                r.update(n=rec["n"], inherited_m=rec["worth_flag_inherited_m"],
                         self_made_m=rec["worth_flag_selfmade_m"], missing_m=rec["worth_flag_missing_m"])
            else:
                r.update(n=0, inherited_m=0.0, self_made_m=0.0, missing_m=0.0)
            for var, col in (("gdppc", "gdppc"), ("tfp", "rtfpna")):
                for h in (1, 3, 5):
                    fwd = p[col].get((i, t + h), np.nan)
                    back = p[col].get((i, t - h), np.nan)
                    now = p[col].get((i, t), np.nan)
                    r[f"g_{var}_{h}"] = 100 * (np.log(fwd) - np.log(now)) / h
                    r[f"b_{var}_{h}"] = 100 * (np.log(now) - np.log(back)) / h
                r[f"ln_y_back"] = np.nan
            for h in (1, 3, 5):
                lv = p["level_pc"].get((i, t - h), np.nan)
                r[f"ln_y_m{h}"] = np.log(lv) if np.isfinite(lv) else np.nan
            rows.append(r)
    d = pd.DataFrame(rows).drop(columns=["ln_y_back"])
    d["inherited_pct"] = 100 * d["inherited_m"] * 1e6 / d["gdp_prev_usd"]
    d["self_made_pct"] = 100 * d["self_made_m"] * 1e6 / d["gdp_prev_usd"]
    d["total_pct"] = d["inherited_pct"] + d["self_made_pct"]
    years_with = d.groupby("iso3")["n"].transform(lambda s: int((s > 0).sum()))
    d["billionaire_country"] = years_with >= 7
    return d, unmapped


# ---------------------------------------------------------------------------------------
# estimation


def ols(y: np.ndarray, X: np.ndarray, groups: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and standard errors: CR1 by ``groups`` if given, else HC1."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    sc = X * u[:, None]
    if groups is None:
        meat = sc.T @ sc * n / (n - k)
    else:
        codes = pd.factorize(groups)[0]
        G = int(codes.max()) + 1
        S = np.zeros((G, k))
        np.add.at(S, codes, sc)
        meat = S.T @ S * G / (G - 1) * (n - 1) / (n - k)
    V = XtX_inv @ meat @ XtX_inv
    return b, np.sqrt(np.diag(V))


def transform(v: pd.Series, how: str) -> pd.Series:
    return np.log1p(v) if how == "log1p" else v


def design(d: pd.DataFrame, y: str, regs: list[str], controls: list[str], estimator: str) -> dict | None:
    """Outcome, design matrix and clusters after the estimator's transformation.

    ``regs`` come first in the design, so regs[0] has index 0 (fe) or 1 (pooled).
    """
    d = d.dropna(subset=[y] + regs + controls).copy()
    if estimator == "fe":
        d = d[d.groupby("iso3")["year"].transform("size") >= 2]
    if len(d) < 30 or d["iso3"].nunique() < 10:
        return None
    years = pd.get_dummies(d["year"], prefix="y", drop_first=True, dtype=float)
    cols = regs + controls + list(years.columns)
    d = pd.concat([d.reset_index(drop=True), years.reset_index(drop=True)], axis=1)
    if estimator == "fe":
        d[cols + [y]] = d[cols + [y]] - d.groupby("iso3")[cols + [y]].transform("mean")
        X, j0 = d[cols].to_numpy(), 0
    else:
        X, j0 = np.column_stack([np.ones(len(d)), d[cols].to_numpy()]), 1
    return dict(y=d[y].to_numpy(), X=X, j0=j0, groups=d["iso3"].to_numpy(),
                n_obs=len(d), n_countries=d["iso3"].nunique())


def sample(d: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "billionaire_countries":
        return d[d["billionaire_country"]]
    if name == "no_offshore":
        return d[~d["iso3"].isin(OFFSHORE)]
    return d


def prep(d: pd.DataFrame, how: str) -> pd.DataFrame:
    d = d.copy()
    d["x_inh"] = transform(d["inherited_pct"], how)
    d["x_self"] = transform(d["self_made_pct"], how)
    d["x_tot"] = transform(d["total_pct"], how)
    # for the difference: inherited and (inherited + self-made) on the same scale
    d["x_sum"] = d["x_inh"] + d["x_self"]
    return d


def specifications(panel: pd.DataFrame, designs: list | None = None) -> pd.DataFrame:
    rows = []
    for h, var, how, smp, ctrl, est in itertools.product(
            (1, 3, 5), ("gdppc", "tfp"), ("log1p", "raw"),
            ("all", "billionaire_countries", "no_offshore"), ("none", "income"), ("fe", "pooled")):
        d = prep(sample(panel, smp), how)
        y = f"g_{var}_{h}"
        controls = ["ln_y"] if ctrl == "income" else []
        # separate coefficients
        des = design(d, y, ["x_inh", "x_self"], controls, est)
        if des is None:
            continue
        b, se = ols(des["y"], des["X"], des["groups"])
        b_inh, s_inh = b[des["j0"]], se[des["j0"]]
        b_self, s_self = b[des["j0"] + 1], se[des["j0"] + 1]
        # the difference, estimated directly: y on x_inh and x_sum -> coefficient on x_inh
        dd = design(d, y, ["x_inh", "x_sum"], controls, est)
        bd, sd = ols(dd["y"], dd["X"], dd["groups"])
        diff, s_diff = bd[dd["j0"]], sd[dd["j0"]]
        # total billionaire wealth alone, as in the first article
        dt = design(d, y, ["x_tot"], controls, est)
        bt, st = ols(dt["y"], dt["X"], dt["groups"])
        if designs is not None:
            dd["b"] = diff
            designs.append(dd)
        rows.append(dict(horizon=h, outcome=var, transform=how, sample=smp, controls=ctrl, estimator=est,
                         n_obs=des["n_obs"], n_countries=des["n_countries"],
                         b_inh=b_inh, se_inh=s_inh, b_self=b_self, se_self=s_self,
                         diff=diff, se_diff=s_diff, lo_diff=diff - 1.96 * s_diff, hi_diff=diff + 1.96 * s_diff,
                         b_total=bt[dt["j0"]], se_total=st[dt["j0"]]))
    out = pd.DataFrame(rows)
    out["sign"] = np.where(out["lo_diff"] > 0, "positive", np.where(out["hi_diff"] < 0, "negative", "n.s."))
    return out


def _fit_fast(y, X, j, codes):
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    u = y - X @ b
    sc = X * u[:, None]
    G = int(codes.max()) + 1
    S = np.zeros((G, k))
    np.add.at(S, codes, sc)
    meat = S.T @ S * G / (G - 1) * (n - 1) / (n - k)
    v = XtX_inv[j] @ meat @ XtX_inv[:, j]
    return float(b[j]), float(np.sqrt(max(v, 0.0)))


SUBSETS = {
    "all": lambda s: np.ones(len(s), bool),
    "fixed_effects": lambda s: (s["estimator"] == "fe").to_numpy(),
    "year_effects_only": lambda s: (s["estimator"] == "pooled").to_numpy(),
    "horizon_1": lambda s: (s["horizon"] == 1).to_numpy(),
    "horizon_3": lambda s: (s["horizon"] == 3).to_numpy(),
    "horizon_5": lambda s: (s["horizon"] == 5).to_numpy(),
    "gdp_per_capita": lambda s: (s["outcome"] == "gdppc").to_numpy(),
    "tfp": lambda s: (s["outcome"] == "tfp").to_numpy(),
}


def _stats(b, s):
    z = b / s
    return {"median": float(np.median(b)), "n_neg": int(np.sum(b + 1.96 * s < 0)),
            "n_pos": int(np.sum(b - 1.96 * s > 0)), "stouffer": float(np.sum(z) / np.sqrt(len(z)))}


def joint_test(specs: pd.DataFrame, designs: list[dict], draws: int = 1000, seed: int = 20261112) -> pd.DataFrame:
    """Null-imposed joint test for the difference (inherited minus self-made) across all fits."""
    rng = np.random.default_rng(seed)
    nulls, by_c = [], []
    for d in designs:
        nulls.append(d["y"] - d["b"] * d["X"][:, d["j0"]])
        by_c.append(pd.Series(np.arange(len(d["groups"]))).groupby(d["groups"]).apply(np.array).to_dict())
    countries = np.array(sorted(set().union(*[set(g) for g in by_c])))
    masks = {k: f(specs) for k, f in SUBSETS.items()}
    obs = {k: _stats(specs["diff"].to_numpy()[m], specs["se_diff"].to_numpy()[m]) for k, m in masks.items()}
    null = {k: [] for k in masks}
    for _ in range(draws):
        pick = rng.choice(countries, size=len(countries), replace=True)
        b = np.full(len(designs), np.nan)
        s = np.full(len(designs), np.nan)
        for i, (d, y0, g) in enumerate(zip(designs, nulls, by_c)):
            parts = [g[c] for c in pick if c in g]
            if len(parts) < 3:
                continue
            rows = np.concatenate(parts)
            codes = np.concatenate([np.full(len(p), k) for k, p in enumerate(parts)])
            b[i], s[i] = _fit_fast(y0[rows], d["X"][rows], d["j0"], codes)
        ok = np.isfinite(b) & np.isfinite(s) & (s > 0)
        for k, m in masks.items():
            null[k].append(_stats(b[m & ok], s[m & ok]))
    out = []
    for k in masks:
        ns, o = pd.DataFrame(null[k]), obs[k]
        lower = o["median"] < 0
        out.append({"subset": k, "n_specs": int(masks[k].sum()), "draws": draws, "median": o["median"],
                    "p_median": float(np.mean(ns["median"] <= o["median"]) if lower else np.mean(ns["median"] >= o["median"])),
                    "n_neg": o["n_neg"], "null_n_neg_p95": float(np.percentile(ns["n_neg"], 95)),
                    "p_n_neg": float(np.mean(ns["n_neg"] >= o["n_neg"])),
                    "n_pos": o["n_pos"], "null_n_pos_p95": float(np.percentile(ns["n_pos"], 95)),
                    "p_n_pos": float(np.mean(ns["n_pos"] >= o["n_pos"])),
                    "stouffer": o["stouffer"],
                    "p_stouffer": float(np.mean(ns["stouffer"] <= o["stouffer"]) if o["stouffer"] < 0
                                        else np.mean(ns["stouffer"] >= o["stouffer"]))})
    return pd.DataFrame(out)


def placebo(panel: pd.DataFrame) -> pd.DataFrame:
    """Difference coefficient for past (t-h to t) and future (t to t+h) growth, fixed effects."""
    rows = []
    for var, how, smp in itertools.product(("gdppc", "tfp"), ("log1p", "raw"), ("all", "billionaire_countries")):
        d = prep(sample(panel, smp), how)
        for h in (5, 3, 1):
            dd = design(d, f"b_{var}_{h}", ["x_inh", "x_sum"], [f"ln_y_m{h}"], "fe")
            b, s = ols(dd["y"], dd["X"], dd["groups"])
            rows.append(dict(outcome=var, transform=how, sample=smp, window=f"t-{h} to t", offset=-h,
                             diff=b[0], se=s[0], n_obs=dd["n_obs"]))
        for h in (1, 3, 5):
            dd = design(d, f"g_{var}_{h}", ["x_inh", "x_sum"], ["ln_y"], "fe")
            b, s = ols(dd["y"], dd["X"], dd["groups"])
            rows.append(dict(outcome=var, transform=how, sample=smp, window=f"t to t+{h}", offset=h,
                             diff=b[0], se=s[0], n_obs=dd["n_obs"]))
    out = pd.DataFrame(rows)
    out["lo"], out["hi"] = out["diff"] - 1.96 * out["se"], out["diff"] + 1.96 * out["se"]
    return out


def pre_post_test(panel: pd.DataFrame, draws: int = 1000, seed: int = 20261113) -> pd.DataFrame:
    """Forward (t to t+h) minus backward (t-h to t) difference coefficient, by country bootstrap.

    If composition merely tracked growth that had already happened, the forward coefficient would
    look like the backward one; the test asks whether they differ.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for var, how, smp, h in itertools.product(("gdppc", "tfp"), ("log1p", "raw"),
                                              ("all", "billionaire_countries"), (3, 5)):
        d = prep(sample(panel, smp), how)
        fwd = design(d, f"g_{var}_{h}", ["x_inh", "x_sum"], ["ln_y"], "fe")
        bwd = design(d, f"b_{var}_{h}", ["x_inh", "x_sum"], [f"ln_y_m{h}"], "fe")
        bf, _ = ols(fwd["y"], fwd["X"], fwd["groups"])
        bb, _ = ols(bwd["y"], bwd["X"], bwd["groups"])
        idx_f = pd.Series(np.arange(len(fwd["groups"]))).groupby(fwd["groups"]).apply(np.array).to_dict()
        idx_b = pd.Series(np.arange(len(bwd["groups"]))).groupby(bwd["groups"]).apply(np.array).to_dict()
        cs = np.array(sorted(set(idx_f) | set(idx_b)))
        diffs = []
        for _ in range(draws):
            pick = rng.choice(cs, size=len(cs), replace=True)
            rf = np.concatenate([idx_f[c] for c in pick if c in idx_f])
            rb = np.concatenate([idx_b[c] for c in pick if c in idx_b])
            cf = np.linalg.lstsq(fwd["X"][rf], fwd["y"][rf], rcond=None)[0][0]
            cb = np.linalg.lstsq(bwd["X"][rb], bwd["y"][rb], rcond=None)[0][0]
            diffs.append(cf - cb)
        diffs = np.array(diffs)
        rows.append(dict(outcome=var, transform=how, sample=smp, horizon=h, forward=bf[0], backward=bb[0],
                         forward_minus_backward=bf[0] - bb[0], lo=np.percentile(diffs, 2.5),
                         hi=np.percentile(diffs, 97.5), p_ge_0=float(np.mean(diffs >= 0))))
    return pd.DataFrame(rows)


def wid_check(panel: pd.DataFrame, countries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = pd.read_csv(DATA / "wid_shares.csv", keep_default_na=False)
    iso = dict(zip(countries["iso2"], countries["iso3"]))
    s = s[(s["variable"] == "shweal992j") & (s["percentile"] == "p99p100") & (s["data_quality"] > 0)]
    s = s.assign(iso3=s["country"].map(iso)).dropna(subset=["iso3"])
    top = s.set_index(["iso3", "year"])["value"]
    d = panel.copy()
    d["top1"] = [100 * top.get((i, t), np.nan) for i, t in zip(d["iso3"], d["year"])]
    d = d[(d["n"] > 0) & d["top1"].notna()].copy()
    d["inh_share"] = d["inherited_pct"] / d["total_pct"]
    d["inh_c"] = d["inh_share"] - d["inh_share"].mean()
    d["top1_x_inh"] = d["top1"] * d["inh_c"]
    rows = []
    for var, h, est in itertools.product(("gdppc", "tfp"), (1, 3), ("fe", "pooled")):
        dd = design(d, f"g_{var}_{h}", ["top1_x_inh", "top1", "inh_c"], ["ln_y"], est)
        if dd is None:
            continue
        b, se = ols(dd["y"], dd["X"], dd["groups"])
        j = dd["j0"]
        rows.append(dict(outcome=var, horizon=h, estimator=est, n_obs=dd["n_obs"], n_countries=dd["n_countries"],
                         interaction=b[j], se_interaction=se[j], top1=b[j + 1], se_top1=se[j + 1],
                         years=f"{int(d['year'].min())}-{int(d.dropna(subset=[f'g_{var}_{h}'])['year'].max())}"))
    out = pd.DataFrame(rows)
    out["lo"], out["hi"] = out["interaction"] - 1.96 * out["se_interaction"], out["interaction"] + 1.96 * out["se_interaction"]
    return out, d


# ---------------------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(exist_ok=True)
    countries = load_countries()
    panel, unmapped = build_panel(countries)
    panel.to_csv(OUT / "panel.csv", index=False, float_format="%.6f")
    cov = pd.read_csv(DATA / "forbes_coverage.csv").drop(columns=["keys"])
    cov["published_list_size"] = cov["year"].map(PUBLISHED_LIST_SIZE)
    cov.to_csv(OUT / "forbes_quality.csv", index=False, float_format="%.4f")

    designs: list[dict] = []
    specs = specifications(panel, designs)
    specs.to_csv(OUT / "specs.csv", index=False, float_format="%.6f")
    joint = joint_test(specs, designs)
    joint.to_csv(OUT / "joint_test.csv", index=False, float_format="%.6f")
    plc = placebo(panel)
    plc.to_csv(OUT / "placebo.csv", index=False, float_format="%.6f")
    prepost = pre_post_test(panel)
    prepost.to_csv(OUT / "pre_post_test.csv", index=False, float_format="%.6f")
    wid, wid_sample = wid_check(panel, countries)
    wid.to_csv(OUT / "wid_check.csv", index=False, float_format="%.6f")

    L = []
    w = L.append
    w("# Results (every number quoted in the article)")
    w("")
    w("## Forbes list quality")
    for _, r in cov.iterrows():
        if pd.to_numeric(r["status"], errors="coerce") != 200:
            w(f"{int(r['year'])}: status {r['status']}")
            continue
        extra = f", published list size {int(r['published_list_size'])}" if pd.notna(r["published_list_size"]) else ""
        w(f"{int(r['year'])}: {int(r['n_persons'])} people{extra}; self-made flag filled {r['share_selfmade_flag']:.3f}, "
          f"flag true {r['share_selfmade_true']:.3f}; self-made score filled {r['share_selfmade_rank']:.3f}; "
          f"total net worth ${r['total_worth_bn']:.0f}bn")
    w(f"unmapped Forbes citizenship labels (dropped): {', '.join(unmapped) if unmapped else 'none'}")
    f_iso, _ = forbes_iso3(countries)
    f_iso = f_iso[f_iso["year"].isin(YEARS)]
    absent = f_iso[~f_iso["iso3"].isin(panel["iso3"])]
    w(f"billionaire citizenships outside the panel (no PWT or WDI data, or under one million people): "
      f"{', '.join(sorted(absent['iso3'].unique()))}; their share of listed wealth 2011-2023: "
      f"{absent['worth_m'].sum() / f_iso['worth_m'].sum():.4f}")
    w("")
    w("## Panel")
    w(f"country-years {len(panel)}, countries {panel['iso3'].nunique()}, years {panel['year'].min()}-{panel['year'].max()}")
    wb = panel[panel["n"] > 0]
    w(f"country-years with at least one billionaire {len(wb)} in {wb['iso3'].nunique()} countries; "
      f"'billionaire countries' (>= 7 of 13 years): {panel.loc[panel['billionaire_country'], 'iso3'].nunique()}")
    share = wb["inherited_m"].sum() / (wb["inherited_m"].sum() + wb["self_made_m"].sum())
    w(f"inherited share of billionaire wealth, all country-years pooled: {share:.4f}; "
      f"wealth with no flag: {wb['missing_m'].sum():.1f} million")
    for y in (2011, 2017, 2023):
        sub = wb[wb["year"] == y]
        w(f"  {y}: inherited share {sub['inherited_m'].sum() / (sub['inherited_m'].sum() + sub['self_made_m'].sum()):.4f}, "
          f"billionaire countries {len(sub)}")
    cm = wb.groupby("iso3").agg(years=("year", "size"), inh=("inherited_m", "sum"), sm=("self_made_m", "sum"),
                                tot=("total_pct", "mean"))
    cm["inh_share"] = cm["inh"] / (cm["inh"] + cm["sm"])
    big = cm[cm["years"] >= 7]
    w(f"country means (>= 7 years): inherited share median {big['inh_share'].median():.3f}, "
      f"10th-90th pct {big['inh_share'].quantile(0.1):.3f}-{big['inh_share'].quantile(0.9):.3f}; "
      f"billionaire wealth/GDP median {big['tot'].median():.2f}%")
    for i in ("USA", "CHN", "IND", "DEU", "FRA", "ITA", "ESP", "SWE", "RUS", "BRA", "MEX", "KOR", "JPN", "GBR", "HKG", "CHE"):
        if i in cm.index:
            r = cm.loc[i]
            w(f"  {i}: years {int(r.years)}, inherited share {r.inh_share:.3f}, mean billionaire wealth {r.tot:.2f}% of GDP")
    within = wb[wb["billionaire_country"]].assign(sh=lambda x: x["inherited_m"] / (x["inherited_m"] + x["self_made_m"]))
    w(f"within-country sd of the inherited share (billionaire countries), median over countries: "
      f"{within.groupby('iso3')['sh'].std().median():.4f}")
    xs = prep(panel, "log1p")
    w(f"share of variance of log(1+inherited %) that is within-country: "
      f"{1 - xs.groupby('iso3')['x_inh'].transform('mean').var() / xs['x_inh'].var():.4f}")
    w("")
    w("## Specifications (diff = inherited minus self-made coefficient, pp of annual growth per unit of the transform)")
    c = specs["sign"].value_counts()
    w(f"fits: {len(specs)}; diff < 0 in {(specs['diff'] < 0).sum()} ({100 * (specs['diff'] < 0).mean():.1f}%); "
      f"significantly negative {c.get('negative', 0)}, significantly positive {c.get('positive', 0)}; "
      f"median diff {specs['diff'].median():+.4f}")
    for col in ("estimator", "horizon", "outcome", "transform", "sample", "controls"):
        w(f"by {col}:")
        for k, sub in specs.groupby(col):
            cc = sub["sign"].value_counts()
            w(f"  {k}: n={len(sub)}, median diff {sub['diff'].median():+.4f}, share<0 {100 * (sub['diff'] < 0).mean():.0f}%, "
              f"sig- {cc.get('negative', 0)}, sig+ {cc.get('positive', 0)}; median b_inh {sub['b_inh'].median():+.4f}, "
              f"median b_self {sub['b_self'].median():+.4f}, median b_total {sub['b_total'].median():+.4f}")
    base = specs[(specs.horizon == 3) & (specs.outcome == "gdppc") & (specs["transform"] == "log1p")
                 & (specs["sample"] == "all") & (specs.controls == "income")]
    for _, r in base.iterrows():
        w(f"baseline ({r['estimator']}, h=3, GDP pc, log1p, all, income control; n={r['n_obs']}, countries={r['n_countries']}): "
          f"inherited {r['b_inh']:+.4f} (se {r['se_inh']:.4f}), self-made {r['b_self']:+.4f} (se {r['se_self']:.4f}), "
          f"difference {r['diff']:+.4f} [{r['lo_diff']:+.4f}, {r['hi_diff']:+.4f}], total {r['b_total']:+.4f} (se {r['se_total']:.4f})")
    w("joint test (null imposed on the difference, countries resampled jointly; one-sided p in the observed direction):")
    for _, r in joint.iterrows():
        w(f"  {r['subset']}: {r['n_specs']} fits, median {r['median']:+.4f} (p={r['p_median']:.3f}); "
          f"sig- {r['n_neg']} (null 95th pct {r['null_n_neg_p95']:.0f}, p={r['p_n_neg']:.3f}); "
          f"sig+ {r['n_pos']} (null 95th pct {r['null_n_pos_p95']:.0f}, p={r['p_n_pos']:.3f}); "
          f"Stouffer Z {r['stouffer']:+.2f} (p={r['p_stouffer']:.3f})")
    w("")
    w("## Placebo: difference coefficient for past and future growth (fixed effects, start-of-window income control)")
    for (o, t, s), sub in plc.groupby(["outcome", "transform", "sample"]):
        parts = [f"{r['window']}: {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]" for _, r in sub.sort_values("offset").iterrows()]
        w(f"{o} / {t} / {s}: " + "; ".join(parts))
    w("forward minus backward (country bootstrap, 1,000 draws):")
    for _, r in prepost.iterrows():
        w(f"  {r['outcome']} / {r['transform']} / {r['sample']} / h={r['horizon']}: forward {r['forward']:+.4f}, "
          f"backward {r['backward']:+.4f}, difference {r['forward_minus_backward']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}], "
          f"P(>=0) {r['p_ge_0']:.3f}")
    w("")
    w("## WID check (source-based top-1% wealth share x inherited share of billionaire wealth)")
    w(f"country-years: {len(wid_sample)}, countries {wid_sample['iso3'].nunique()}, years "
      f"{int(wid_sample['year'].min())}-{int(wid_sample['year'].max())}")
    for _, r in wid.iterrows():
        w(f"{r['estimator']} {r['outcome']} h={r['horizon']} (n={r['n_obs']}, countries={r['n_countries']}, years {r['years']}): "
          f"interaction {r['interaction']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]; top1 {r['top1']:+.4f} (se {r['se_top1']:.4f})")
    text = "\n".join(L) + "\n"
    (OUT / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
