"""Cross-country evidence. Writes results/panel_*.csv, results/billionaire_*.csv and
results/analysis_summary.txt.

    python analysis.py

Part A, specification multiverse. Annualised growth of real GDP per capita or of TFP (Penn
World Table 11.0) over the following window, regressed on a top share at the start of the
window (WID), across every combination of

    measure    top 1%, top 10%, bottom 50% (sign flipped) of net personal wealth;
               top 1%, top 10% of pre-tax national income
    outcome    GDP per capita growth, TFP growth
    window     5 years (1980-2019), 10 years (1980-2019)
    controls   none; initial income; initial income + human capital + investment share
    estimator  pooled with period effects; country and period fixed effects; between
               (country means)
    sample     every WID area matched to PWT; only starts backed by source data (WID data
               quality > 0 within two years); high-income countries (World Bank group)

Coefficients are percentage points of annual growth per one standard deviation more
concentration in the estimation sample; 95% intervals are country-clustered (CR1) for pooled
and fixed-effects fits and HC1 for between fits.

Part B, composition. Forbes billionaire wealth in 1996, 2001 and 2014 (Billionaire
Characteristics Database, Freund and Oliver 2016) split by origin, as a share of GDP in
current dollars, and growth over the next five years.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "results"
RNG = np.random.default_rng(20261111)

# ---------------------------------------------------------------------------------------
# loading


def load_pwt() -> pd.DataFrame:
    p = pd.read_csv(DATA / "pwt110_panel.csv")
    p = p.rename(columns={"countrycode": "iso3"})
    p["gdppc"] = p["rgdpna"] / p["pop"]
    p["level_pc"] = p["rgdpe"] / p["pop"]
    return p


def load_countries() -> pd.DataFrame:
    c = pd.read_csv(DATA / "wdi_countries.csv", keep_default_na=False)
    return c[c["income_group"] != "Aggregates"]


def load_wid(countries: pd.DataFrame) -> pd.DataFrame:
    s = pd.read_csv(DATA / "wid_shares.csv", keep_default_na=False)
    iso = dict(zip(countries["iso2"], countries["iso3"]))
    iso.update({"KS": "XKX", "TW": "TWN"})
    s["iso3"] = s["country"].map(iso)
    s = s.dropna(subset=["iso3"])
    s["measure"] = s["variable"].str[:6] + "_" + s["percentile"]
    return s[["iso3", "year", "measure", "value", "data_quality"]]


MEASURES = {
    "wealth_top1": "shweal_p99p100",
    "wealth_top10": "shweal_p90p100",
    "wealth_bottom50": "shweal_p0p50",
    "income_top1": "sptinc_p99p100",
    "income_top10": "sptinc_p90p100",
}

# ---------------------------------------------------------------------------------------
# regression helpers


def ols(y: np.ndarray, X: np.ndarray, groups: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and standard errors: CR1 by ``groups`` if given, else HC1."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    if groups is None:
        meat = (X * u[:, None]).T @ (X * u[:, None]) * n / (n - k)
    else:
        meat = np.zeros((k, k))
        uniq = np.unique(groups)
        for g in uniq:
            m = groups == g
            s = X[m].T @ u[m]
            meat += np.outer(s, s)
        G = len(uniq)
        meat *= G / (G - 1) * (n - 1) / (n - k)
    V = XtX_inv @ meat @ XtX_inv
    return b, np.sqrt(np.diag(V))


def demean(df: pd.DataFrame, cols: list[str], by: str) -> pd.DataFrame:
    out = df.copy()
    out[cols] = df[cols] - df.groupby(by)[cols].transform("mean")
    return out


# ---------------------------------------------------------------------------------------
# Part A


def windows(length: int) -> list[tuple[int, int]]:
    if length == 5:
        starts = range(1980, 2016, 5)
    else:
        starts = range(1980, 2011, 10)
    return [(s, min(s + length, 2019)) for s in starts]


def build_panel(pwt: pd.DataFrame, wid: pd.DataFrame, countries: pd.DataFrame,
                length: int) -> pd.DataFrame:
    p = pwt.set_index(["iso3", "year"])
    rows = []
    for s, e in windows(length):
        a = p.xs(s, level="year")
        b = p.xs(e, level="year")
        span = p.loc[(slice(None), slice(s, e - 1)), :].groupby(level="iso3")["csh_i"].mean()
        d = pd.DataFrame({
            "start": s, "end": e,
            "g_gdppc": 100 * (np.log(b["gdppc"]) - np.log(a["gdppc"])) / (e - s),
            "g_tfp": 100 * (np.log(b["rtfpna"]) - np.log(a["rtfpna"])) / (e - s),
            "ln_y0": np.log(a["level_pc"]),
            "hc0": a["hc"],
            "inv": span,
        })
        rows.append(d.reset_index())
    panel = pd.concat(rows, ignore_index=True)
    # inequality at the start of each window, and whether a source-based value lies within 2 years
    for name, code in MEASURES.items():
        m = wid[wid["measure"] == code]
        val = m.set_index(["iso3", "year"])["value"]
        panel[name] = [val.get((i, s), np.nan) for i, s in zip(panel["iso3"], panel["start"])]
        obs = m[m["data_quality"] > 0]
        obs_years = obs.groupby("iso3")["year"].apply(np.array).to_dict()
        obs_val = obs.set_index(["iso3", "year"])["value"]
        o_v = []
        for i, s in zip(panel["iso3"], panel["start"]):
            ys = obs_years.get(i)
            if ys is None:
                o_v.append(np.nan)
                continue
            d = np.abs(ys - s)
            j = int(np.argmin(d))
            o_v.append(obs_val[(i, ys[j])] if d[j] <= 2 else np.nan)
        panel[name + "_obs"] = o_v
    panel["high_income"] = panel["iso3"].isin(countries.loc[countries["income_group"] == "High income", "iso3"])
    panel["window"] = length
    return panel


def fit_spec(df: pd.DataFrame, x: str, y: str, controls: list[str], estimator: str) -> dict | None:
    d = df.dropna(subset=[x, y] + controls).copy()
    if estimator == "within":
        counts = d.groupby("iso3")["start"].transform("size")
        d = d[counts >= 2]
    if len(d) < 20 or d["iso3"].nunique() < 8:
        return None
    sd = float(d[x].std())
    if not np.isfinite(sd) or sd == 0:
        return None
    if estimator == "between":
        m = d.groupby("iso3")[[x, y] + controls].mean()
        if len(m) < 8:
            return None
        X = np.column_stack([np.ones(len(m)), m[x].to_numpy()] + [m[c].to_numpy() for c in controls])
        b, se = ols(m[y].to_numpy(), X)
        n_obs, n_c = len(m), len(m)
    else:
        periods = pd.get_dummies(d["start"], prefix="p", drop_first=True, dtype=float)
        cols = [x] + controls + list(periods.columns)
        d = pd.concat([d, periods], axis=1)
        if estimator == "within":
            d = demean(d, cols + [y], "iso3")
            X = d[cols].to_numpy()
        else:
            X = np.column_stack([np.ones(len(d)), d[cols].to_numpy()])
        b, se = ols(d[y].to_numpy(), X, d["iso3"].to_numpy())
        if estimator == "within":
            b, se = np.r_[0.0, b], np.r_[0.0, se]
        n_obs, n_c = len(d), d["iso3"].nunique()
    beta, s = b[1] * sd, se[1] * sd
    return dict(beta=beta, se=s, lo=beta - 1.96 * s, hi=beta + 1.96 * s, n_obs=n_obs,
                n_countries=n_c, sd_x=sd)


def multiverse(panels: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    control_sets = {"none": [], "income": ["ln_y0"], "full": ["ln_y0", "hc0", "inv"]}
    for (length, panel), measure, outcome, (cname, controls), est, sample in itertools.product(
            panels.items(), MEASURES, ["g_gdppc", "g_tfp"], control_sets.items(),
            ["pooled", "within", "between"], ["all", "observed", "high_income"]):
        df = panel.copy()
        x = measure
        if sample == "observed":
            df[x] = df[measure + "_obs"]
        elif sample == "high_income":
            df = df[df["high_income"]]
        if measure == "wealth_bottom50":  # more concentration = lower bottom-50 share
            df[x] = -df[x]
        r = fit_spec(df, x, outcome, controls, est)
        if r is None:
            continue
        r.update(window=length, measure=measure, outcome=outcome, controls=cname,
                 estimator=est, sample=sample)
        rows.append(r)
    out = pd.DataFrame(rows)
    out["sign"] = np.where(out["lo"] > 0, "positive", np.where(out["hi"] < 0, "negative", "n.s."))
    return out


def imputation_diagnostic(panel: pd.DataFrame, wid: pd.DataFrame) -> dict[str, float]:
    """How closely do WID wealth shares track income shares, imputed vs source-based years?"""
    w = wid[wid["measure"] == "shweal_p99p100"].set_index(["iso3", "year"])
    i = wid[wid["measure"] == "sptinc_p99p100"].set_index(["iso3", "year"])["value"]
    j = w.join(i.rename("inc"), how="inner").reset_index()
    j = j[(j["year"] >= 1995) & (j["year"] <= 2019)]
    out = {}
    for label, m in [("imputed", j["data_quality"] == 0), ("observed", j["data_quality"] > 0)]:
        sub = j[m]
        out[f"corr_{label}"] = float(np.corrcoef(sub["value"], sub["inc"])[0, 1])
        # share of within-country variance of the wealth share explained by the income share
        dm = sub.assign(v=sub["value"] - sub.groupby("iso3")["value"].transform("mean"),
                        x=sub["inc"] - sub.groupby("iso3")["inc"].transform("mean"))
        dm = dm[dm.groupby("iso3")["year"].transform("size") >= 3]
        bb = np.polyfit(dm["x"], dm["v"], 1)
        resid = dm["v"] - np.polyval(bb, dm["x"])
        out[f"within_r2_{label}"] = float(1 - resid.var() / dm["v"].var())
        out[f"n_{label}"] = int(len(sub))
        out[f"countries_{label}"] = int(sub["iso3"].nunique())
    return out


# ---------------------------------------------------------------------------------------
# Part B


WAVES = (1996, 2001, 2014)
GROUPS = {
    "inherited": ["inherited"],
    "self_made": ["founder non-finance", "self-made finance", "executive"],
    "resources": ["privatized and resources"],
}


def billionaire_panel(pwt: pd.DataFrame, countries: pd.DataFrame, horizon: int) -> pd.DataFrame:
    b = pd.read_csv(DATA / "billionaires_piie.csv")
    b["iso3"] = b["location.country code"].replace({"DEN": "DNK", "Taiwan": "TWN"})
    b["worth"] = b["wealth.worth in billions"] * 1e9
    gdp = pd.read_csv(DATA / "wdi_gdp_usd.csv").set_index(["iso3", "year"])["gdp_usd"]
    agg = {}
    for g, types in GROUPS.items():
        sub = b[b["wealth.type"].isin(types)]
        agg[g] = sub.groupby(["iso3", "year"])["worth"].sum()
    agg["count"] = b.groupby(["iso3", "year"]).size()
    bw = pd.DataFrame(agg).fillna(0.0)

    p = pwt.set_index(["iso3", "year"])
    rows = []
    for w in WAVES:
        e = min(w + horizon, 2023)
        a, z = p.xs(w, level="year"), p.xs(e, level="year")
        span = p.loc[(slice(None), slice(w, e - 1)), :].groupby(level="iso3")["csh_i"].mean()
        d = pd.DataFrame({
            "wave": w, "end": e,
            "g_gdppc": 100 * (np.log(z["gdppc"]) - np.log(a["gdppc"])) / (e - w),
            "g_tfp": 100 * (np.log(z["rtfpna"]) - np.log(a["rtfpna"])) / (e - w),
            "ln_y0": np.log(a["level_pc"]), "hc0": a["hc"], "inv": span, "pop": a["pop"],
        }).reset_index()
        d["gdp_usd"] = [gdp.get((i, w), np.nan) for i in d["iso3"]]
        for g in list(GROUPS) + ["count"]:
            d[g] = [bw[g].get((i, w), 0.0) if (i, w) in bw.index else 0.0 for i in d["iso3"]]
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out = out[(out["pop"] >= 1.0) & out["iso3"].isin(countries["iso3"])]  # PWT pop in millions
    out = out.dropna(subset=["gdp_usd", "g_gdppc", "ln_y0", "hc0", "inv"])
    for g in GROUPS:
        out[g + "_pct"] = 100 * out[g] / out["gdp_usd"]
    out["total_pct"] = out[[g + "_pct" for g in GROUPS]].sum(axis=1)
    return out


def fit_billionaires(d: pd.DataFrame, y: str, regs: list[str], transform: str,
                     boot: int = 2000) -> dict:
    d = d.dropna(subset=[y]).copy()
    X_parts = []
    for r in regs:
        v = d[r].to_numpy()
        X_parts.append(np.log1p(v) if transform == "log1p" else v)
    waves = pd.get_dummies(d["wave"], prefix="w", drop_first=True, dtype=float).to_numpy()
    controls = d[["ln_y0", "hc0", "inv"]].to_numpy()
    X = np.column_stack([np.ones(len(d))] + X_parts + [controls, waves])
    yv = d[y].to_numpy()
    groups = d["iso3"].to_numpy()
    b, se = ols(yv, X, groups)
    res = {"n_obs": len(d), "n_countries": d["iso3"].nunique(),
           "n_with_billionaires": int((d["count"] > 0).sum())}
    for j, r in enumerate(regs, start=1):
        res[f"b_{r}"], res[f"se_{r}"] = b[j], se[j]
    # country-cluster bootstrap for the coefficients and for inherited - self_made
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    draws = []
    for _ in range(boot):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        bb = np.linalg.lstsq(X[rows], yv[rows], rcond=None)[0]
        draws.append(bb[1:1 + len(regs)])
    draws = np.array(draws)
    for j, r in enumerate(regs):
        res[f"lo_{r}"], res[f"hi_{r}"] = np.percentile(draws[:, j], [2.5, 97.5])
    if "inherited_pct" in regs and "self_made_pct" in regs:
        i, s = regs.index("inherited_pct"), regs.index("self_made_pct")
        diff = draws[:, i] - draws[:, s]
        res["diff_inh_minus_self"] = b[1 + i] - b[1 + s]
        res["diff_lo"], res["diff_hi"] = np.percentile(diff, [2.5, 97.5])
        res["p_diff_ge_0"] = float(np.mean(diff >= 0))
    return res


def billionaire_specs(pwt: pd.DataFrame, countries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    panels = {}
    for horizon in (5, 9):
        panels[horizon] = billionaire_panel(pwt, countries, horizon)
    for horizon, y, transform, sample, model in itertools.product(
            (5, 9), ("g_gdppc", "g_tfp"), ("log1p", "raw"), ("all", "with_billionaires", "no_offshore"),
            ("total", "composition")):
        d = panels[horizon]
        if sample == "with_billionaires":
            d = d[d["count"] > 0]
        elif sample == "no_offshore":
            d = d[~d["iso3"].isin(["HKG", "SGP", "CHE", "CYP", "LUX", "MCO", "LIE", "BMU", "MAC", "IRL"])]
        regs = ["total_pct"] if model == "total" else ["inherited_pct", "self_made_pct", "resources_pct"]
        r = fit_billionaires(d, y, regs, transform)
        r.update(horizon=horizon, outcome=y, transform=transform, sample=sample, model=model)
        rows.append(r)
    return pd.DataFrame(rows), panels[5]


# ---------------------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(exist_ok=True)
    pwt = load_pwt()
    countries = load_countries()
    wid = load_wid(countries)
    panels = {L: build_panel(pwt, wid, countries, L) for L in (5, 10)}
    pd.concat(panels.values()).to_csv(OUT / "panel_data.csv", index=False, float_format="%.6f")
    specs = multiverse(panels)
    specs.to_csv(OUT / "panel_specs.csv", index=False, float_format="%.6f")
    diag = imputation_diagnostic(panels[5], wid)
    bspecs, bpanel = billionaire_specs(pwt, countries)
    bspecs.to_csv(OUT / "billionaire_specs.csv", index=False, float_format="%.6f")
    bpanel.to_csv(OUT / "billionaire_panel.csv", index=False, float_format="%.6f")

    L = []
    w = L.append
    w("# Cross-country results (every number quoted in the article)")
    w("")
    w("## Data coverage")
    ws = wid[wid["measure"] == "shweal_p99p100"]
    ws = ws[(ws["year"] >= 1980) & (ws["year"] <= 2024)]
    w(f"WID wealth top-1% series: {ws['iso3'].nunique()} countries matched, {len(ws)} country-years, "
      f"{(ws['data_quality'] > 0).sum()} with data_quality > 0 "
      f"({100 * (ws['data_quality'] > 0).mean():.1f}%), in {ws.loc[ws['data_quality'] > 0, 'iso3'].nunique()} countries")
    for k, v in diag.items():
        w(f"imputation diagnostic {k}: {v:.4f}" if isinstance(v, float) else f"imputation diagnostic {k}: {v}")
    for L_, p in panels.items():
        w(f"{L_}-year panel: {len(p.dropna(subset=['wealth_top1', 'g_gdppc']))} country-windows with wealth "
          f"top-1% and growth; {p['wealth_top1_obs'].notna().sum()} with a source-based start value "
          f"in {p.loc[p['wealth_top1_obs'].notna(), 'iso3'].nunique()} countries")
    w("")
    w("## Part A, multiverse")
    w(f"specifications estimated: {len(specs)}")
    for label, sub in [("all", specs), ("wealth measures", specs[specs.measure.str.startswith("wealth")]),
                       ("income measures", specs[specs.measure.str.startswith("income")])]:
        c = sub["sign"].value_counts()
        w(f"{label}: n={len(sub)}, point estimate < 0 in {100 * (sub.beta < 0).mean():.1f}%, "
          f"significant negative {c.get('negative', 0)} ({100 * c.get('negative', 0) / len(sub):.1f}%), "
          f"significant positive {c.get('positive', 0)} ({100 * c.get('positive', 0) / len(sub):.1f}%), "
          f"median {sub.beta.median():+.3f}, range [{sub.beta.min():+.3f}, {sub.beta.max():+.3f}]")
    for col in ("estimator", "sample", "measure", "outcome", "controls", "window"):
        w(f"by {col}:")
        for k, sub in specs.groupby(col):
            c = sub["sign"].value_counts()
            w(f"  {k}: n={len(sub)}, median {sub.beta.median():+.3f}, share<0 {100 * (sub.beta < 0).mean():.1f}%, "
              f"sig- {c.get('negative', 0)}, sig+ {c.get('positive', 0)}, median n_obs {sub.n_obs.median():.0f}")
    w("by estimator x sample (wealth measures, median beta / share<0 / sig- / sig+):")
    wm = specs[specs.measure.str.startswith("wealth")]
    for (e, s), sub in wm.groupby(["estimator", "sample"]):
        c = sub["sign"].value_counts()
        w(f"  {e} / {s}: {sub.beta.median():+.3f} / {100 * (sub.beta < 0).mean():.0f}% / "
          f"{c.get('negative', 0)} / {c.get('positive', 0)} (n={len(sub)})")
    w("")
    w("## Part B, billionaire wealth by origin (5-year growth after 1996, 2001, 2014)")
    bp = bpanel
    w(f"panel: {len(bp)} country-waves, {bp['iso3'].nunique()} countries, {(bp['count'] > 0).sum()} "
      f"country-waves with at least one billionaire")
    for g in list(GROUPS) + ["total"]:
        w(f"  {g}_pct among countries with billionaires: median {bp.loc[bp['count'] > 0, g + '_pct'].median():.3f}, "
          f"mean {bp.loc[bp['count'] > 0, g + '_pct'].mean():.3f}, max {bp[g + '_pct'].max():.2f}")
    for _, r in bspecs.iterrows():
        tag = f"h={r['horizon']} {r['outcome']} {r['transform']} {r['sample']} {r['model']} (n={r['n_obs']}, countries={r['n_countries']})"
        if r.model == "total":
            w(f"{tag}: total {r.b_total_pct:+.4f} [{r.lo_total_pct:+.4f}, {r.hi_total_pct:+.4f}]")
        else:
            w(f"{tag}: inherited {r.b_inherited_pct:+.4f} [{r.lo_inherited_pct:+.4f}, {r.hi_inherited_pct:+.4f}], "
              f"self-made {r.b_self_made_pct:+.4f} [{r.lo_self_made_pct:+.4f}, {r.hi_self_made_pct:+.4f}], "
              f"resources {r.b_resources_pct:+.4f} [{r.lo_resources_pct:+.4f}, {r.hi_resources_pct:+.4f}]; "
              f"inherited - self-made {r.diff_inh_minus_self:+.4f} [{r.diff_lo:+.4f}, {r.diff_hi:+.4f}], "
              f"P(diff >= 0) {r.p_diff_ge_0:.3f}")
    comp = bspecs[bspecs["model"] == "composition"]
    w(f"composition specs: {len(comp)}; inherited - self-made < 0 in {(comp.diff_inh_minus_self < 0).sum()}; "
      f"95% interval excludes 0 in {((comp.diff_hi < 0) | (comp.diff_lo > 0)).sum()}; "
      f"inherited coefficient < 0 in {(comp.b_inherited_pct < 0).sum()}; self-made coefficient > 0 in "
      f"{(comp.b_self_made_pct > 0).sum()}; resources coefficient < 0 in {(comp.b_resources_pct < 0).sum()}; "
      f"P(diff >= 0) range [{comp.p_diff_ge_0.min():.3f}, {comp.p_diff_ge_0.max():.3f}]")
    tot = bspecs[bspecs["model"] == "total"]
    w(f"total specs: {len(tot)}; total coefficient < 0 in {(tot.b_total_pct < 0).sum()}; 95% interval excludes 0 in "
      f"{((tot.hi_total_pct < 0) | (tot.lo_total_pct > 0)).sum()}")
    text = "\n".join(L) + "\n"
    (OUT / "analysis_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
