"""Is the inherited-wealth / TFP association a sector effect?

    python analysis.py

Writes results/*.csv and results/summary.txt.

The previous project (wealth-origin-growth) found that, within countries, inherited billionaire
wealth predicted slower TFP growth than self-made wealth, but that the association rested on
slow change across 2011-2023 and could reflect the sectors in which fortunes sit (technology
fortunes are mostly self-made, retail and diversified holdings mostly inherited). Here the
Forbes lists are aggregated by industry as well, which allows two tests.

1. Sector controls. The previous fixed-effects regression of growth on inherited wealth and on
   total billionaire wealth (both log(1 + % of GDP)) is re-estimated with the shares of the
   country's billionaire wealth in six sector groups as controls.

2. Shift-share decomposition. A country's inherited share of billionaire wealth is split into
   the part predicted by its sector mix, sum_s w(c,t,s) * rho_s(-c), where rho_s(-c) is the
   inherited share of sector s among all other countries' billionaires, and the remainder,
   which measures whether a country's heirs hold more or less of each sector than heirs
   elsewhere. Growth is regressed on both parts, controlling for the size of billionaire
   wealth. A sector effect loads on the first part; an origin effect on the second.

Inference: country-clustered standard errors (CR1); a null-imposed joint test across the
specifications for the within-sector coefficient; a placebo on past growth; the same split of
the period as before (windows ending by 2019 against windows including 2020 or later).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "results"

YEARS = range(2011, 2024)
OFFSHORE = {"HKG", "SGP", "CHE", "CYP", "LUX", "MCO", "LIE", "BMU", "MAC", "IRL", "SWZ"}

LABEL_FIX = {
    "Czech Republic": "CZE", "Egypt": "EGY", "Eswatini (Swaziland)": "SWZ", "Swaziland": "SWZ",
    "Hong Kong": "HKG", "Macao": "MAC", "Macau": "MAC", "Russia": "RUS", "Slovakia": "SVK",
    "South Korea": "KOR", "Taiwan": "TWN", "Turkey": "TUR", "Venezuela": "VEN", "Vietnam": "VNM",
    "Guernsey": "GGY",
}

# Forbes industry labels drift across years; map every label seen to the 18 current categories
INDUSTRY_MAP = {
    "Technology": "Technology", "Telecom": "Telecom", "Fashion & Retail": "Fashion & Retail",
    "Fashion and Retail": "Fashion & Retail", "Retail": "Fashion & Retail",
    "Diversified": "Diversified", "Real Estate": "Real Estate", "Manufacturing": "Manufacturing",
    "Finance & Investments": "Finance & Investments", "Finance and Investments": "Finance & Investments",
    "Investments": "Finance & Investments", "Finance": "Finance & Investments",
    "Energy": "Energy", "Oil": "Energy", "Healthcare": "Healthcare", "Health care": "Healthcare",
    "Medicine": "Healthcare", "Food & Beverage": "Food & Beverage", "Food and Beverage": "Food & Beverage",
    "Metals & Mining": "Metals & Mining", "Media & Entertainment": "Media & Entertainment",
    "Media": "Media & Entertainment", "Automotive": "Automotive", "Service": "Service",
    "Construction & Engineering": "Construction & Engineering", "Logistics": "Logistics",
    "Gambling & Casinos": "Gambling & Casinos", "Gaming": "Gambling & Casinos", "Sports": "Sports",
}
GROUPS = {
    "tech": ["Technology", "Telecom"],
    "finance": ["Finance & Investments"],
    "consumer": ["Fashion & Retail", "Food & Beverage"],
    "property": ["Real Estate", "Construction & Engineering"],
    "resources": ["Energy", "Metals & Mining"],
    "industry": ["Manufacturing", "Automotive", "Logistics"],
}  # everything else (diversified, healthcare, media, service, sports, gambling, unclassified) is the base


# ---------------------------------------------------------------------------------------
# data


def load_countries() -> pd.DataFrame:
    c = pd.read_csv(DATA / "wdi_countries.csv", keep_default_na=False)
    return c[c["income_group"] != "Aggregates"]


def load_forbes(countries: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    f = pd.read_csv(DATA / "forbes_country_industry_year.csv", keep_default_na=False, na_values=[""])
    f = f[f["year"].isin(YEARS)].copy()
    names = dict(zip(countries["name"].str.lower(), countries["iso3"]))
    f["iso3"] = [LABEL_FIX.get(c, names.get(c.lower())) for c in f["citizenship"]]
    unmapped_c = sorted(f.loc[f["iso3"].isna(), "citizenship"].unique())
    f["sector"] = f["industry"].map(INDUSTRY_MAP).fillna("Unclassified")
    unmapped_i = sorted(f.loc[f["sector"] == "Unclassified", "industry"].unique())
    return f.dropna(subset=["iso3"]), unmapped_c, unmapped_i


def sector_rho(f: pd.DataFrame) -> pd.DataFrame:
    """Inherited share of each sector's billionaire wealth among all *other* countries."""
    known = f[f["origin"].isin(["inherited", "self_made"])]
    by = known.pivot_table(index=["iso3", "sector"], columns="origin", values="worth_m",
                           aggfunc="sum", fill_value=0.0).reset_index()
    tot = by.groupby("sector")[["inherited", "self_made"]].sum()
    rows = []
    for _, r in by.iterrows():
        inh = tot.loc[r["sector"], "inherited"] - r["inherited"]
        sm = tot.loc[r["sector"], "self_made"] - r["self_made"]
        rows.append({"iso3": r["iso3"], "sector": r["sector"],
                     "rho_loo": inh / (inh + sm) if inh + sm > 0 else np.nan})
    out = pd.DataFrame(rows)
    glob = (tot["inherited"] / (tot["inherited"] + tot["self_made"])).rename("rho_global")
    return out.merge(glob, left_on="sector", right_index=True)


def build_panel(countries: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    f, unmapped_c, unmapped_i = load_forbes(countries)
    rho = sector_rho(f)
    f = f.merge(rho[["iso3", "sector", "rho_loo"]], on=["iso3", "sector"], how="left")
    known = f[f["origin"].isin(["inherited", "self_made"])].copy()
    known["pred_inh_m"] = known["worth_m"] * known["rho_loo"].fillna(0.0)
    # every sector except the base (Diversified) gets its own share column for the robustness check
    sectors = [s for s in sorted(known["sector"].unique()) if s != "Diversified"] + ["Diversified"]
    cy = known.groupby(["iso3", "year"]).apply(lambda g: pd.Series({
        "n": g["n"].sum(),
        "inherited_m": g.loc[g["origin"] == "inherited", "worth_m"].sum(),
        "self_made_m": g.loc[g["origin"] == "self_made", "worth_m"].sum(),
        "pred_inh_m": g["pred_inh_m"].sum(),
        **{f"w_{k}": g.loc[g["sector"].isin(v), "worth_m"].sum() for k, v in GROUPS.items()},
        **{f"wz_{k}": g.loc[g["sector"] == s, "worth_m"].sum() for k, s in enumerate(sectors[:-1])},
    }), include_groups=False)

    pwt = pd.read_csv(DATA / "pwt110_panel.csv").rename(columns={"countrycode": "iso3"})
    pwt["gdppc"] = pwt["rgdpna"] / pwt["pop"]
    pwt["level_pc"] = pwt["rgdpe"] / pwt["pop"]
    p = pwt.set_index(["iso3", "year"])
    gdp = pd.read_csv(DATA / "wdi_gdp_usd.csv").set_index(["iso3", "year"])["gdp_usd"]
    big = pwt.loc[(pwt["year"] == 2011) & (pwt["pop"] >= 1.0), "iso3"]
    iso = sorted(set(big) & set(countries["iso3"]))
    rows = []
    for i in iso:
        for t in YEARS:
            g_prev = gdp.get((i, t - 1), np.nan)
            if not np.isfinite(g_prev) or (i, t) not in p.index:
                continue
            r = {"iso3": i, "year": t, "gdp_prev_usd": g_prev, "ln_y": np.log(p.loc[(i, t), "level_pc"])}
            if (i, t) in cy.index:
                r.update(cy.loc[(i, t)].to_dict())
            else:
                r.update({k: 0.0 for k in cy.columns})
            for var, col in (("gdppc", "gdppc"), ("tfp", "rtfpna")):
                for h in (1, 3, 5):
                    fwd, back, now = (p[col].get((i, t + h), np.nan), p[col].get((i, t - h), np.nan),
                                      p[col].get((i, t), np.nan))
                    r[f"g_{var}_{h}"] = 100 * (np.log(fwd) - np.log(now)) / h
                    r[f"b_{var}_{h}"] = 100 * (np.log(now) - np.log(back)) / h
            for h in (1, 3, 5):
                lv = p["level_pc"].get((i, t - h), np.nan)
                r[f"ln_y_m{h}"] = np.log(lv) if np.isfinite(lv) else np.nan
            rows.append(r)
    d = pd.DataFrame(rows)
    scale = 100 * 1e6 / d["gdp_prev_usd"]
    d["inherited_pct"] = d["inherited_m"] * scale
    d["self_made_pct"] = d["self_made_m"] * scale
    d["total_pct"] = d["inherited_pct"] + d["self_made_pct"]
    tot_m = d["inherited_m"] + d["self_made_m"]
    with np.errstate(invalid="ignore", divide="ignore"):
        d["inh_share"] = np.where(tot_m > 0, d["inherited_m"] / tot_m, np.nan)
        d["pred_share"] = np.where(tot_m > 0, d["pred_inh_m"] / tot_m, np.nan)
        for k in GROUPS:
            d[f"s_{k}"] = np.where(tot_m > 0, d[f"w_{k}"] / tot_m, 0.0)
        for k in range(len(sectors) - 1):
            d[f"z_{k}"] = np.where(tot_m > 0, d[f"wz_{k}"] / tot_m, 0.0)
    d["resid_share"] = d["inh_share"] - d["pred_share"]
    years_with = d.groupby("iso3")["n"].transform(lambda s: int((s > 0).sum()))
    d["billionaire_country"] = years_with >= 7
    meta = {"sectors": sectors, "unmapped_citizenship": unmapped_c, "unclassified_industry": unmapped_i, "rho": rho, "forbes": f}
    return d, meta


# ---------------------------------------------------------------------------------------
# estimation


def ols(y: np.ndarray, X: np.ndarray, groups: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
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


def design(d: pd.DataFrame, y: str, regs: list[str], controls: list[str], estimator: str) -> dict | None:
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
    return dict(y=d[y].to_numpy(), X=X, j0=j0, groups=d["iso3"].to_numpy(), n_obs=len(d),
                n_countries=d["iso3"].nunique(), cols=cols)


def prep(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["x_inh"] = np.log1p(d["inherited_pct"])
    d["x_sum"] = np.log1p(d["inherited_pct"]) + np.log1p(d["self_made_pct"])
    d["x_tot"] = np.log1p(d["total_pct"])
    return d


def sample(d: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "billionaire_countries":
        return d[d["billionaire_country"]]
    if name == "with_billionaires":
        return d[d["n"] > 0]
    if name == "no_offshore":
        return d[~d["iso3"].isin(OFFSHORE)]
    return d


SECTOR_CTRL = [f"s_{k}" for k in GROUPS]


def fit(d: pd.DataFrame, y: str, regs: list[str], controls: list[str], est: str) -> dict | None:
    des = design(d, y, regs, controls, est)
    if des is None:
        return None
    b, s = ols(des["y"], des["X"], des["groups"])
    out = {"n_obs": des["n_obs"], "n_countries": des["n_countries"], "_des": des}
    for k, r in enumerate(regs):
        out[f"b_{r}"], out[f"se_{r}"] = b[des["j0"] + k], s[des["j0"] + k]
    return out


def main_fits(panel: pd.DataFrame, keep: dict | None = None) -> pd.DataFrame:
    rows = []
    for var, h, smp, est in itertools.product(("gdppc", "tfp"), (1, 3, 5),
                                              ("billionaire_countries", "with_billionaires"), ("fe", "pooled")):
        d = prep(sample(panel, smp))
        y = f"g_{var}_{h}"
        base = dict(outcome=var, horizon=h, sample=smp, estimator=est)
        # A: the #144 regression on this sample, without and with sector controls
        a0 = fit(d, y, ["x_inh", "x_sum"], ["ln_y"], est)
        a1 = fit(d, y, ["x_inh", "x_sum"], ["ln_y"] + SECTOR_CTRL, est)
        # B: shift-share decomposition of the inherited share, size of billionaire wealth held fixed
        b1 = fit(d, y, ["resid_share", "pred_share", "x_tot"], ["ln_y"], est)
        b0 = fit(d, y, ["inh_share", "x_tot"], ["ln_y"], est)
        if None in (a0, a1, b1, b0):
            continue
        if keep is not None:
            keep["resid"].append({**b1["_des"], "b": b1["b_resid_share"]})
            keep["diff_sector_ctrl"].append({**a1["_des"], "b": a1["b_x_inh"]})
        rows.append({**base, "n_obs": b1["n_obs"], "n_countries": b1["n_countries"],
                     "diff_no_sector": a0["b_x_inh"], "se_diff_no_sector": a0["se_x_inh"],
                     "diff_sector_ctrl": a1["b_x_inh"], "se_diff_sector_ctrl": a1["se_x_inh"],
                     "inh_share": b0["b_inh_share"], "se_inh_share": b0["se_inh_share"],
                     "resid": b1["b_resid_share"], "se_resid": b1["se_resid_share"],
                     "pred": b1["b_pred_share"], "se_pred": b1["se_pred_share"]})
    return pd.DataFrame(rows)


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


def joint_test(fits: pd.DataFrame, designs: list[dict], col: str, draws: int = 1000,
               seed: int = 20261114) -> pd.DataFrame:
    """Null-imposed joint test for one coefficient across all fits (countries resampled jointly)."""
    rng = np.random.default_rng(seed)
    nulls, by_c = [], []
    for d in designs:
        nulls.append(d["y"] - d["b"] * d["X"][:, d["j0"]])
        by_c.append(pd.Series(np.arange(len(d["groups"]))).groupby(d["groups"]).apply(np.array).to_dict())
    countries = np.array(sorted(set().union(*[set(g) for g in by_c])))
    subsets = {"all": np.ones(len(fits), bool), "tfp": (fits["outcome"] == "tfp").to_numpy(),
               "gdp_per_capita": (fits["outcome"] == "gdppc").to_numpy(),
               "tfp_fixed_effects": ((fits["outcome"] == "tfp") & (fits["estimator"] == "fe")).to_numpy(),
               "tfp_pooled": ((fits["outcome"] == "tfp") & (fits["estimator"] == "pooled")).to_numpy()}

    def stats(b, s):
        z = b / s
        return {"median": float(np.median(b)), "n_neg": int(np.sum(b + 1.96 * s < 0)),
                "n_pos": int(np.sum(b - 1.96 * s > 0)), "stouffer": float(np.sum(z) / np.sqrt(len(z)))}

    obs = {k: stats(fits[col].to_numpy()[m], fits["se_" + col].to_numpy()[m]) for k, m in subsets.items()}
    null = {k: [] for k in subsets}
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
        for k, m in subsets.items():
            null[k].append(stats(b[m & ok], s[m & ok]))
    out = []
    for k in subsets:
        ns, o = pd.DataFrame(null[k]), obs[k]
        lower = o["median"] < 0
        out.append({"coefficient": col, "subset": k, "n_fits": int(subsets[k].sum()), "median": o["median"],
                    "p_median": float(np.mean(ns["median"] <= o["median"]) if lower else np.mean(ns["median"] >= o["median"])),
                    "n_neg": o["n_neg"], "null_n_neg_p95": float(np.percentile(ns["n_neg"], 95)),
                    "p_n_neg": float(np.mean(ns["n_neg"] >= o["n_neg"])),
                    "n_pos": o["n_pos"], "null_n_pos_p95": float(np.percentile(ns["n_pos"], 95)),
                    "p_n_pos": float(np.mean(ns["n_pos"] >= o["n_pos"])),
                    "stouffer": o["stouffer"],
                    "p_stouffer": float(np.mean(ns["stouffer"] <= o["stouffer"]) if o["stouffer"] < 0
                                        else np.mean(ns["stouffer"] >= o["stouffer"]))})
    return pd.DataFrame(out)


def placebo_and_split(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decomposition coefficients for past vs future growth, and by period (fixed effects)."""
    prow, srow = [], []
    for var, smp in itertools.product(("gdppc", "tfp"), ("billionaire_countries", "with_billionaires")):
        d = prep(sample(panel, smp))
        for h in (5, 3, 1):
            r = fit(d, f"b_{var}_{h}", ["resid_share", "pred_share", "x_tot"], [f"ln_y_m{h}"], "fe")
            prow.append(dict(outcome=var, sample=smp, offset=-h, window=f"t-{h} to t",
                             resid=r["b_resid_share"], se_resid=r["se_resid_share"],
                             pred=r["b_pred_share"], se_pred=r["se_pred_share"]))
        for h in (1, 3, 5):
            r = fit(d, f"g_{var}_{h}", ["resid_share", "pred_share", "x_tot"], ["ln_y"], "fe")
            prow.append(dict(outcome=var, sample=smp, offset=h, window=f"t to t+{h}",
                             resid=r["b_resid_share"], se_resid=r["se_resid_share"],
                             pred=r["b_pred_share"], se_pred=r["se_pred_share"]))
            for label, part in (("ends by 2019", d[d["year"] + h <= 2019]), ("includes 2020", d[d["year"] + h >= 2020])):
                q = fit(part, f"g_{var}_{h}", ["resid_share", "pred_share", "x_tot"], ["ln_y"], "fe")
                a = fit(part, f"g_{var}_{h}", ["x_inh", "x_sum"], ["ln_y"], "fe")
                s6 = fit(part, f"g_{var}_{h}", ["x_inh", "x_sum"], ["ln_y"] + SECTOR_CTRL, "fe")
                if q is None or a is None or s6 is None:
                    continue
                yrs = part.dropna(subset=[f"g_{var}_{h}"])["year"]
                srow.append(dict(outcome=var, sample=smp, horizon=h, windows=label,
                                 list_years=f"{int(yrs.min())}-{int(yrs.max())}", n_obs=q["n_obs"],
                                 resid=q["b_resid_share"], se_resid=q["se_resid_share"],
                                 pred=q["b_pred_share"], se_pred=q["se_pred_share"],
                                 diff=a["b_x_inh"], se_diff=a["se_x_inh"],
                                 diff_sector=s6["b_x_inh"], se_diff_sector=s6["se_x_inh"]))
    return pd.DataFrame(prow), pd.DataFrame(srow)


MODELS = {
    "within_sector": (["resid_share", "pred_share", "x_tot"], "resid_share"),
    "diff_sector_ctrl": (["x_inh", "x_sum"], "x_inh"),
}


def _ctrl(model: str, back_h: int | None) -> list[str]:
    inc = ["ln_y"] if back_h is None else [f"ln_y_m{back_h}"]
    return inc + (SECTOR_CTRL if model == "diff_sector_ctrl" else [])


def pre_post_test(panel: pd.DataFrame, draws: int = 1000, seed: int = 20261115) -> pd.DataFrame:
    """Forward (t to t+h) minus backward (t-h to t) coefficient, fixed effects, country bootstrap."""
    rng = np.random.default_rng(seed)
    rows = []
    for model, var, smp, h in itertools.product(MODELS, ("gdppc", "tfp"),
                                                ("billionaire_countries", "with_billionaires"), (3, 5)):
        regs, _ = MODELS[model]
        d = prep(sample(panel, smp))
        fwd = design(d, f"g_{var}_{h}", regs, _ctrl(model, None), "fe")
        bwd = design(d, f"b_{var}_{h}", regs, _ctrl(model, h), "fe")
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
        rows.append(dict(model=model, outcome=var, sample=smp, horizon=h, forward=bf[0], backward=bb[0],
                         forward_minus_backward=bf[0] - bb[0], lo=np.percentile(diffs, 2.5),
                         hi=np.percentile(diffs, 97.5), p_ge_0=float(np.mean(diffs >= 0))))
    return pd.DataFrame(rows)


def past_growth_control(panel: pd.DataFrame) -> pd.DataFrame:
    """Forward coefficient with growth over the preceding h years as a control (fixed effects).

    If composition only predicted slower growth because it rises after fast growth and growth
    reverts, holding past growth fixed should remove the forward association.
    """
    rows = []
    for model, var, smp, h in itertools.product(MODELS, ("gdppc", "tfp"),
                                                ("billionaire_countries", "with_billionaires"), (3, 5)):
        regs, _ = MODELS[model]
        d = prep(sample(panel, smp))
        y = f"g_{var}_{h}"
        d = d.dropna(subset=[f"b_{var}_{h}"])
        base = fit(d, y, regs, _ctrl(model, None), "fe")
        ctrl = fit(d, y, regs + [f"b_{var}_{h}"], _ctrl(model, None), "fe")
        rows.append(dict(model=model, outcome=var, sample=smp, horizon=h, n_obs=ctrl["n_obs"],
                         b_same_sample=base["b_" + regs[0]], se_same_sample=base["se_" + regs[0]],
                         b_past_ctrl=ctrl["b_" + regs[0]], se_past_ctrl=ctrl["se_" + regs[0]],
                         b_past=ctrl[f"b_b_{var}_{h}"], se_past=ctrl[f"se_b_{var}_{h}"]))
    return pd.DataFrame(rows)


def leave_one_out(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, smp, h in itertools.product(MODELS, ("billionaire_countries",), (3, 5)):
        regs, _ = MODELS[model]
        d = prep(sample(panel, smp))
        y = f"g_tfp_{h}"
        full = fit(d, y, regs, _ctrl(model, None), "fe")
        for c in sorted(set(full["_des"]["groups"]) & set(d.loc[d["n"] > 0, "iso3"])):
            r = fit(d[d["iso3"] != c], y, regs, _ctrl(model, None), "fe")
            rows.append(dict(model=model, sample=smp, horizon=h, dropped=c, b=r["b_" + regs[0]],
                             se=r["se_" + regs[0]], full_b=full["b_" + regs[0]], full_se=full["se_" + regs[0]]))
    out = pd.DataFrame(rows)
    out["lo"], out["hi"] = out["b"] - 1.96 * out["se"], out["b"] + 1.96 * out["se"]
    return out


def leave_one_year_out(panel: pd.DataFrame) -> pd.DataFrame:
    """Sector-controlled difference (TFP, fixed effects) with one Forbes list year dropped at a time."""
    rows = []
    for smp, h in itertools.product(("billionaire_countries", "with_billionaires"), (3, 5)):
        d = prep(sample(panel, smp))
        y = f"g_tfp_{h}"
        yrs = sorted(d.dropna(subset=[y])["year"].unique())
        full = fit(d, y, ["x_inh", "x_sum"], _ctrl("diff_sector_ctrl", None), "fe")
        for t in yrs:
            r = fit(d[d["year"] != t], y, ["x_inh", "x_sum"], _ctrl("diff_sector_ctrl", None), "fe")
            rows.append(dict(sample=smp, horizon=h, dropped_year=int(t), b=r["b_x_inh"], se=r["se_x_inh"],
                             full_b=full["b_x_inh"], full_se=full["se_x_inh"]))
    out = pd.DataFrame(rows)
    out["lo"], out["hi"] = out["b"] - 1.96 * out["se"], out["b"] + 1.96 * out["se"]
    return out


def all_sector_controls(panel: pd.DataFrame, sectors: list[str]) -> pd.DataFrame:
    """The #144 difference with the share of every sector (one omitted) as controls."""
    rows = []
    cols = [f"z_{k}" for k in range(len(sectors) - 1)]
    for var, h, smp in itertools.product(("gdppc", "tfp"), (1, 3, 5), ("billionaire_countries", "with_billionaires")):
        d = prep(sample(panel, smp))
        r6 = fit(d, f"g_{var}_{h}", ["x_inh", "x_sum"], ["ln_y"] + SECTOR_CTRL, "fe")
        rA = fit(d, f"g_{var}_{h}", ["x_inh", "x_sum"], ["ln_y"] + cols, "fe")
        r0 = fit(d, f"g_{var}_{h}", ["x_inh", "x_sum"], ["ln_y"], "fe")
        rows.append(dict(outcome=var, horizon=h, sample=smp, none=r0["b_x_inh"], se_none=r0["se_x_inh"],
                         six=r6["b_x_inh"], se_six=r6["se_x_inh"], all=rA["b_x_inh"], se_all=rA["se_x_inh"]))
    return pd.DataFrame(rows)


def variance_split(panel: pd.DataFrame) -> dict:
    d = panel[panel["n"] > 0].dropna(subset=["inh_share", "pred_share", "resid_share"])
    bc = d[d["billionaire_country"]]
    out = {}
    for name, x in (("all with billionaires", d), ("billionaire countries", bc)):
        v = x["inh_share"].var()
        dm = lambda col: x[col] - x.groupby("iso3")[col].transform("mean")  # noqa: E731
        vw = dm("inh_share").var()
        out[name] = {
            "n": len(x), "countries": x["iso3"].nunique(),
            "var_pred_over_var_inh": x["pred_share"].var() / v,
            "var_resid_over_var_inh": x["resid_share"].var() / v,
            "corr_pred_inh": x[["pred_share", "inh_share"]].corr().iloc[0, 1],
            "within_var_share_inh": vw / v,
            "within_pred_over_within_inh": dm("pred_share").var() / vw,
            "within_resid_over_within_inh": dm("resid_share").var() / vw,
        }
    return out


# ---------------------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(exist_ok=True)
    countries = load_countries()
    panel, meta = build_panel(countries)
    keep_cols = [c for c in panel.columns if not c.startswith("_")]
    panel[keep_cols].to_csv(OUT / "panel.csv", index=False, float_format="%.6f")
    meta["rho"].to_csv(OUT / "sector_rho_leave_country_out.csv", index=False, float_format="%.6f")
    f = meta["forbes"]

    designs: dict = {"resid": [], "diff_sector_ctrl": []}
    fits = main_fits(panel, designs)
    fits.to_csv(OUT / "fits.csv", index=False, float_format="%.6f")
    joint = pd.concat([joint_test(fits, designs[c], c) for c in designs], ignore_index=True)
    joint.to_csv(OUT / "joint_test.csv", index=False, float_format="%.6f")
    plc, split = placebo_and_split(panel)
    plc.to_csv(OUT / "placebo.csv", index=False, float_format="%.6f")
    split.to_csv(OUT / "period_split.csv", index=False, float_format="%.6f")
    pp = pre_post_test(panel)
    pp.to_csv(OUT / "pre_post.csv", index=False, float_format="%.6f")
    pgc = past_growth_control(panel)
    pgc.to_csv(OUT / "past_growth_control.csv", index=False, float_format="%.6f")
    loo = leave_one_out(panel)
    loo.to_csv(OUT / "leave_one_out.csv", index=False, float_format="%.6f")
    loyo = leave_one_year_out(panel)
    loyo.to_csv(OUT / "leave_one_year_out.csv", index=False, float_format="%.6f")
    allsec = all_sector_controls(panel, meta["sectors"])
    allsec.to_csv(OUT / "all_sector_controls.csv", index=False, float_format="%.6f")
    vs = variance_split(panel)

    sec = (f[f["origin"].isin(["inherited", "self_made"])]
           .pivot_table(index="sector", columns="origin", values="worth_m", aggfunc="sum", fill_value=0.0))
    sec["worth_share"] = (sec["inherited"] + sec["self_made"]) / (sec["inherited"] + sec["self_made"]).sum()
    sec["inh_share"] = sec["inherited"] / (sec["inherited"] + sec["self_made"])
    sec = sec.sort_values("inh_share")
    sec.to_csv(OUT / "sector_inheritance.csv", float_format="%.6f")

    L = []
    w = L.append
    w("# Results (every number quoted in the article)")
    w("")
    w("## Data")
    cov = pd.read_csv(DATA / "forbes_coverage.csv")
    ok = cov[pd.to_numeric(cov["status"], errors="coerce") == 200]
    w(f"Forbes lists fetched: {len(ok)} of {len(cov)} ({', '.join(str(y) for y in cov.loc[~cov.index.isin(ok.index), 'year'])} failed); "
      f"industry filled: min {ok['share_industry'].min():.4f}; several industries per person: max {ok['share_multi_industry'].max():.4f}; "
      f"distinct raw labels per year {int(ok['n_distinct_industries'].min())}-{int(ok['n_distinct_industries'].max())}")
    raw_labels = sorted(f["industry"].unique())
    w(f"raw industry labels 2011-2023: {len(raw_labels)}; mapped to {f['sector'].nunique()} sectors "
      f"(unclassified labels: {', '.join(meta['unclassified_industry'])}; their share of wealth "
      f"{f.loc[f['sector'] == 'Unclassified', 'worth_m'].sum() / f['worth_m'].sum():.4f})")
    w(f"unmapped citizenships: {', '.join(meta['unmapped_citizenship']) or 'none'}")
    w(f"wealth with no self-made flag 2011-2023: {f.loc[f['origin'] == 'missing', 'worth_m'].sum():.1f} million")
    w("")
    w("## Inherited share by sector (pooled 2011-2023, all countries)")
    for s_, r in sec.iterrows():
        w(f"  {s_}: inherited share {r['inh_share']:.3f}, share of billionaire wealth {r['worth_share']:.3f}")
    w("")
    w("## Panel and decomposition")
    w(f"country-years {len(panel)}, countries {panel['iso3'].nunique()}; with billionaires {int((panel['n'] > 0).sum())} "
      f"in {panel.loc[panel['n'] > 0, 'iso3'].nunique()} countries; billionaire countries {panel.loc[panel['billionaire_country'], 'iso3'].nunique()}")
    for k, v in vs.items():
        w(f"{k} (n={v['n']}, countries={v['countries']}): var(pred)/var(inh) {v['var_pred_over_var_inh']:.3f}, "
          f"var(resid)/var(inh) {v['var_resid_over_var_inh']:.3f}, corr(pred, inh) {v['corr_pred_inh']:.3f}; "
          f"within-country share of var(inh) {v['within_var_share_inh']:.3f}; within: var(pred)/var(inh) "
          f"{v['within_pred_over_within_inh']:.3f}, var(resid)/var(inh) {v['within_resid_over_within_inh']:.3f}")
    for i in ("USA", "CHN", "IND", "DEU", "FRA", "SWE", "RUS", "JPN", "GBR", "KOR", "ITA", "BRA"):
        sub = panel[(panel["iso3"] == i) & (panel["n"] > 0)]
        if len(sub):
            w(f"  {i}: inherited share {sub['inh_share'].mean():.3f}, sector-predicted {sub['pred_share'].mean():.3f}, "
              f"within-sector remainder {sub['resid_share'].mean():+.3f}; tech share of billionaire wealth {sub['s_tech'].mean():.3f}")
    w("")
    w("## Main fits (pp of annual growth; diff = log-point coefficient as in #144; shares in units of 1.0 = 100 pp)")
    for _, r in fits.iterrows():
        def ci(b, s):
            return f"{b:+.4f} [{b - 1.96 * s:+.4f}, {b + 1.96 * s:+.4f}]"
        w(f"{r['outcome']} h={r['horizon']} {r['sample']} {r['estimator']} (n={r['n_obs']}, countries={r['n_countries']}): "
          f"diff no sector {ci(r['diff_no_sector'], r['se_diff_no_sector'])}; diff with sector shares "
          f"{ci(r['diff_sector_ctrl'], r['se_diff_sector_ctrl'])}; inherited share {ci(r['inh_share'], r['se_inh_share'])}; "
          f"within-sector {ci(r['resid'], r['se_resid'])}; sector-predicted {ci(r['pred'], r['se_pred'])}")
    for col, lab in (("diff_no_sector", "diff without sector controls"), ("diff_sector_ctrl", "diff with sector controls"),
                     ("resid", "within-sector coefficient"), ("pred", "sector-predicted coefficient"),
                     ("inh_share", "inherited-share coefficient")):
        se = fits["se_" + col]
        sig_neg = int(((fits[col] + 1.96 * se) < 0).sum())
        sig_pos = int(((fits[col] - 1.96 * se) > 0).sum())
        for o in ("tfp", "gdppc"):
            m = fits["outcome"] == o
            w(f"  {lab} / {o}: negative in {int((fits.loc[m, col] < 0).sum())} of {int(m.sum())}, sig- "
              f"{int(((fits.loc[m, col] + 1.96 * se[m]) < 0).sum())}, sig+ {int(((fits.loc[m, col] - 1.96 * se[m]) > 0).sum())}, "
              f"median {fits.loc[m, col].median():+.4f}")
        w(f"  {lab} / all: sig- {sig_neg}, sig+ {sig_pos}")
    ratio = (fits["diff_sector_ctrl"] / fits["diff_no_sector"])
    tf = fits[(fits["outcome"] == "tfp") & (fits["estimator"] == "fe")]
    w(f"fixed-effects TFP fits: median ratio diff with / without sector controls "
      f"{(tf['diff_sector_ctrl'] / tf['diff_no_sector']).median():.3f}; all fits median ratio {ratio.median():.3f}")
    w("joint tests (null imposed, countries resampled jointly):")
    for _, r in joint.iterrows():
        w(f"  {r['coefficient']} / {r['subset']}: {r['n_fits']} fits, median {r['median']:+.4f} (p={r['p_median']:.3f}); sig- {r['n_neg']} "
          f"(null 95th pct {r['null_n_neg_p95']:.0f}, p={r['p_n_neg']:.3f}); sig+ {r['n_pos']} (null 95th pct "
          f"{r['null_n_pos_p95']:.0f}, p={r['p_n_pos']:.3f}); Stouffer Z {r['stouffer']:+.2f} (p={r['p_stouffer']:.3f})")
    w("")
    w("## Placebo: within-sector and sector-predicted coefficients for past and future growth (fixed effects)")
    for (o, smp), sub in plc.groupby(["outcome", "sample"]):
        sub = sub.sort_values("offset")
        w(f"{o} / {smp}: within-sector " + "; ".join(
            f"{r['window']} {r['resid']:+.3f} [{r['resid'] - 1.96 * r['se_resid']:+.3f}, {r['resid'] + 1.96 * r['se_resid']:+.3f}]"
            for _, r in sub.iterrows()))
        w(f"{o} / {smp}: sector-predicted " + "; ".join(
            f"{r['window']} {r['pred']:+.3f} [{r['pred'] - 1.96 * r['se_pred']:+.3f}, {r['pred'] + 1.96 * r['se_pred']:+.3f}]"
            for _, r in sub.iterrows()))
    w("")
    w("## Period split (fixed effects)")
    for _, r in split.iterrows():
        w(f"{r['outcome']} / {r['sample']} / h={r['horizon']} / {r['windows']} (list years {r['list_years']}, n={r['n_obs']}): "
          f"within-sector {r['resid']:+.3f} [{r['resid'] - 1.96 * r['se_resid']:+.3f}, {r['resid'] + 1.96 * r['se_resid']:+.3f}]; "
          f"sector-predicted {r['pred']:+.3f} [{r['pred'] - 1.96 * r['se_pred']:+.3f}, {r['pred'] + 1.96 * r['se_pred']:+.3f}]; "
          f"#144 diff {r['diff']:+.3f} [{r['diff'] - 1.96 * r['se_diff']:+.3f}, {r['diff'] + 1.96 * r['se_diff']:+.3f}]; "
          f"with sector shares {r['diff_sector']:+.3f} [{r['diff_sector'] - 1.96 * r['se_diff_sector']:+.3f}, "
          f"{r['diff_sector'] + 1.96 * r['se_diff_sector']:+.3f}]")
    w("")
    w("## Forward minus backward (fixed effects, country bootstrap, 1,000 draws)")
    for _, r in pp.iterrows():
        w(f"{r['model']} / {r['outcome']} / {r['sample']} / h={r['horizon']}: forward {r['forward']:+.3f}, backward "
          f"{r['backward']:+.3f}, forward-backward {r['forward_minus_backward']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}], "
          f"share of draws >= 0 {r['p_ge_0']:.3f}")
    w("")
    w("## Forward coefficient holding past growth fixed (fixed effects, same sample: windows with both past and future growth)")
    for _, r in pgc.iterrows():
        def ci(b, s):
            return f"{b:+.3f} [{b - 1.96 * s:+.3f}, {b + 1.96 * s:+.3f}]"
        w(f"{r['model']} / {r['outcome']} / {r['sample']} / h={r['horizon']} (n={r['n_obs']}): without {ci(r['b_same_sample'], r['se_same_sample'])}; "
          f"with past growth {ci(r['b_past_ctrl'], r['se_past_ctrl'])}; past-growth coefficient {ci(r['b_past'], r['se_past'])}")
    w("")
    w("## Leave one country out (TFP, fixed effects, billionaire countries)")
    for (m, h), sub in loo.groupby(["model", "horizon"]):
        lo, hi = sub.loc[sub["b"].idxmin()], sub.loc[sub["b"].idxmax()]
        w(f"{m} h={h}: full {sub['full_b'].iloc[0]:+.3f} [{sub['full_b'].iloc[0] - 1.96 * sub['full_se'].iloc[0]:+.3f}, "
          f"{sub['full_b'].iloc[0] + 1.96 * sub['full_se'].iloc[0]:+.3f}]; {len(sub)} drops; range {lo['b']:+.3f} "
          f"(without {lo['dropped']}) to {hi['b']:+.3f} (without {hi['dropped']}); CI excludes zero in "
          f"{int((sub['hi'] < 0).sum())} of {len(sub)}; estimate negative in {int((sub['b'] < 0).sum())}")
    w("")
    w("## Leave one list year out (sector-controlled difference, TFP, fixed effects)")
    for (smp, h), sub in loyo.groupby(["sample", "horizon"]):
        w(f"{smp} h={h}: full {sub['full_b'].iloc[0]:+.3f}; " + "; ".join(
            f"without {r['dropped_year']} {r['b']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]" for _, r in sub.iterrows()))
    w("")
    w(f"## Every sector share as a control ({len(meta['sectors']) - 1} shares, Diversified omitted; fixed effects)")
    for _, r in allsec.iterrows():
        w(f"{r['outcome']} h={r['horizon']} {r['sample']}: none {r['none']:+.3f} [{r['none'] - 1.96 * r['se_none']:+.3f}, "
          f"{r['none'] + 1.96 * r['se_none']:+.3f}]; six groups {r['six']:+.3f} [{r['six'] - 1.96 * r['se_six']:+.3f}, "
          f"{r['six'] + 1.96 * r['se_six']:+.3f}]; all sectors {r['all']:+.3f} [{r['all'] - 1.96 * r['se_all']:+.3f}, "
          f"{r['all'] + 1.96 * r['se_all']:+.3f}]")
    text = "\n".join(L) + "\n"
    (OUT / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
