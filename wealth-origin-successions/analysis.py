"""Do founder-to-heir successions change productivity growth? A country-level event study.

    python analysis.py

Writes results/*.csv and results/summary.txt.

The previous projects found that, within countries, inherited billionaire wealth predicts slower TFP
growth than self-made wealth, a result that survives controls for sector, past growth and pandemic
exposure but remains an association. A succession, when a self-made founder leaves the Forbes list
and heirs with the same source of wealth appear, changes the origin of a fortune without changing
its firm, sector or (at first) its size. Its timing, especially when the founder is old, is closer
to an accident than any change in composition used so far.

Design: annual TFP growth on leads and lags of a succession shock, the wealth passing to heirs as
a share of the previous year's GDP, with country and year effects and lagged income. Successions
are dated by the first Forbes list on which the founder is missing; the death or transfer happened
in the twelve months before that list. Leads are placebos. Inference: country-clustered standard
errors and a permutation test that reshuffles each country's shock series across years.

The succession aggregates come from fetch.py (matched in memory, no person-level data stored); the
composition table, growth and GDP data are copied from earlier snapshots (data/PROVENANCE.txt).
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






# ---------------------------------------------------------------------------------------
# successions

LEADS_LAGS = [-3, -2, -1, 0, 1, 2]  # k = calendar year of growth minus list year that first omits the founder
SHOCKS = {
    "strict": "worth_succ_strict",            # heirs matched to the founder appear on the next list
    "window": "worth_succ_window",            # ... or on the list after
    "old founders": "worth_succ_strict_old",  # strict, founder aged 70 or more on the last list with them
    "heirs' wealth": "heirs_worth_strict",    # strict, measured by the heirs' wealth on the next list
    "old exits, any": "worth_old_exit_any",   # every exit of a self-made founder aged 70+, heirs found or not
    "old exits without heirs": "worth_exit_unmatched_old",
    "exits without heirs": "worth_exit_unmatched",
    "heirs without an exit": "worth_heir_unmatched",
}


def load_successions(countries: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    s = pd.read_csv(DATA / "succession_country_year.csv", keep_default_na=False, na_values=[""])
    names = dict(zip(countries["name"].str.lower(), countries["iso3"]))
    s["iso3"] = [LABEL_FIX.get(c, names.get(str(c).lower())) for c in s["citizenship"]]
    unmapped = sorted(str(c) if isinstance(c, str) else "(empty)" for c in s.loc[s["iso3"].isna(), "citizenship"].unique())
    s = s.dropna(subset=["iso3"]).fillna(0.0)
    s["worth_old_exit_any"] = s["worth_succ_strict_old"] + s["worth_exit_unmatched_old"]
    num = [c for c in s.columns if c not in ("citizenship", "iso3", "year")]
    return s.groupby(["iso3", "year"])[num].sum().reset_index(), unmapped


def annual_panel(panel: pd.DataFrame, succ: pd.DataFrame) -> pd.DataFrame:
    """Country x calendar year: annual growth, lagged income, shocks dated by list year, and leads/lags."""
    pwt = pd.read_csv(DATA / "pwt110_panel.csv").rename(columns={"countrycode": "iso3"})
    pwt["gdppc"] = pwt["rgdpna"] / pwt["pop"]
    pwt["level_pc"] = pwt["rgdpe"] / pwt["pop"]
    p = pwt.set_index(["iso3", "year"]).sort_index()
    gdp = pd.read_csv(DATA / "wdi_gdp_usd.csv").set_index(["iso3", "year"])["gdp_usd"]
    iso = sorted(panel["iso3"].unique())
    bc = set(panel.loc[panel["billionaire_country"], "iso3"])
    ever = set(panel.loc[panel["n"] > 0, "iso3"])
    sv = succ.set_index(["iso3", "year"])
    shock_years = range(2012, 2027)
    D = {}
    for name, col in SHOCKS.items():
        for i in iso:
            for t in shock_years:
                g = gdp.get((i, t - 1), np.nan)
                w = sv[col].get((i, t), 0.0) if col in sv.columns else 0.0
                D[(name, i, t)] = np.log1p(100 * w * 1e6 / g) if np.isfinite(g) and g > 0 else np.nan
    tot = panel.set_index(["iso3", "year"])["total_pct"]
    dval = {(i, t): np.log1p(tot.get((i, t), np.nan)) - np.log1p(tot.get((i, t - 1), np.nan)) for i in iso for t in shock_years}
    rows = []
    for i in iso:
        for s in range(2012, 2024):
            r = {"iso3": i, "year": s, "billionaire_country": i in bc, "ever_billionaire": i in ever}
            for var, col in (("tfp", "rtfpna"), ("gdppc", "gdppc")):
                a, b = p[col].get((i, s), np.nan), p[col].get((i, s - 1), np.nan)
                c0 = p[col].get((i, s - 2), np.nan)
                r[f"g_{var}"] = 100 * (np.log(a) - np.log(b))
                r[f"g_{var}_lag"] = 100 * (np.log(b) - np.log(c0))
            lv = p["level_pc"].get((i, s - 1), np.nan)
            r["ln_y_prev"] = np.log(lv) if np.isfinite(lv) else np.nan
            for name in SHOCKS:
                key = name.replace(" ", "_")
                for k in LEADS_LAGS:
                    r[f"{key}_k{k}"] = D.get((name, i, s - k), np.nan)
            for k in LEADS_LAGS:
                r[f"dval_k{k}"] = dval.get((i, s - k), np.nan)
            for name in ("strict", "old founders"):
                key = name.replace(" ", "_")
                for k in LEADS_LAGS:
                    v = D.get((name, i, s - k), np.nan)
                    r[f"{key}_big_k{k}"] = float(np.expm1(v) > 0.1) if np.isfinite(v) else np.nan
            rows.append(r)
    a = pd.DataFrame(rows)
    shocks = pd.DataFrame([{"shock": n, "iso3": i, "year": t, "d": v} for (n, i, t), v in D.items()])
    return a, shocks


def es_regs(shock: str) -> list[str]:
    key = shock.replace(" ", "_")
    return [f"{key}_k{k}" for k in LEADS_LAGS]


def event_study(a: pd.DataFrame, shock: str, outcome: str, smp: str, est: str = "fe",
                extra: list[str] | None = None, regs: list[str] | None = None) -> dict | None:
    d = a if smp == "all countries" else a[a["billionaire_country"]] if smp == "billionaire countries" else a[a["ever_billionaire"]]
    regs = regs or es_regs(shock)
    des = design(d, f"g_{outcome}", regs, ["ln_y_prev"] + (extra or []), est)
    if des is None:
        return None
    b, se = ols(des["y"], des["X"], des["groups"])
    j0 = des["j0"]
    V = None
    # covariance of the coefficients for the cumulative sums
    n, kx = des["X"].shape
    XtX_inv = np.linalg.pinv(des["X"].T @ des["X"])
    u = des["y"] - des["X"] @ b
    sc = des["X"] * u[:, None]
    codes = pd.factorize(des["groups"])[0]
    G = int(codes.max()) + 1
    S = np.zeros((G, kx))
    np.add.at(S, codes, sc)
    V = XtX_inv @ (S.T @ S * G / (G - 1) * (n - 1) / (n - kx)) @ XtX_inv
    idx = {k: j0 + m for m, k in enumerate(LEADS_LAGS)}
    post = [idx[k] for k in (0, 1, 2)]
    pre = [idx[k] for k in (-3, -2)]
    cum = lambda ids: (float(b[ids].sum()), float(np.sqrt(V[np.ix_(ids, ids)].sum())))  # noqa: E731
    out = {"shock": shock, "outcome": outcome, "sample": smp, "estimator": est, "n_obs": des["n_obs"],
           "n_countries": des["n_countries"]}
    for k in LEADS_LAGS:
        out[f"b_k{k}"], out[f"se_k{k}"] = float(b[idx[k]]), float(se[idx[k]])
    out["post"], out["se_post"] = cum(post)
    out["pre"], out["se_pre"] = cum(pre)
    out["_des"] = des
    return out


def fill_leads_lags(b: pd.DataFrame, P: np.ndarray, rows: dict, cols: dict, key: str) -> None:
    """Write the leads and lags of a country x list-year shock matrix into the annual panel, in place."""
    ri = np.array([rows.get(i, -1) for i in b["iso3"]])
    for k in LEADS_LAGS:
        pos = np.array([cols.get(s - k, -1) for s in b["year"]])
        good = (ri >= 0) & (pos >= 0)
        col = np.full(len(b), np.nan)
        col[good] = P[ri[good], pos[good]]
        b[f"{key}_k{k}"] = col


def permutation(a: pd.DataFrame, shocks: pd.DataFrame, shock: str, outcome: str, smp: str,
                draws: int = 1000, seed: int = 20261119) -> dict:
    """Reshuffle each country's shock series across list years; re-estimate the cumulative post effect."""
    rng = np.random.default_rng(seed)
    obs = event_study(a, shock, outcome, smp)
    sh = shocks[shocks["shock"] == shock].pivot(index="iso3", columns="year", values="d")
    M = sh.to_numpy()
    rows = {i: r for r, i in enumerate(sh.index)}
    cols = {t: c for c, t in enumerate(sh.columns)}
    key = shock.replace(" ", "_")
    b = a.copy()
    stats = []
    for _ in range(draws):
        P = M.copy()
        for r in range(P.shape[0]):
            ok = np.isfinite(P[r])
            vals = P[r, ok]
            rng.shuffle(vals)
            P[r, ok] = vals
        fill_leads_lags(b, P, rows, cols, key)
        stats.append(event_study(b, shock, outcome, smp)["post"])
    stats = np.array(stats)
    return {"shock": shock, "outcome": outcome, "sample": smp, "observed_post": obs["post"],
            "p_le": float(np.mean(stats <= obs["post"])), "null_p5": float(np.percentile(stats, 5)),
            "null_p50": float(np.percentile(stats, 50)), "null_p95": float(np.percentile(stats, 95)), "_draws": stats}


EXPOSURE = ["stringency", "deaths_pm", "tourism_gdp", "contact_emp"]
EXPO_YEAR: list[str] = []


def add_exposure_years(a: pd.DataFrame) -> pd.DataFrame:
    """Standardised pandemic-exposure measures interacted with calendar-year dummies (as in wealth-origin-pandemic)."""
    e = pd.read_csv(DATA / "exposure_by_country.csv").set_index("iso3")[EXPOSURE]
    base = e.loc[e.index.isin(a.loc[a["billionaire_country"], "iso3"])]
    z = (e - base.mean()) / base.std()
    a = a.merge(z.add_prefix("z_"), left_on="iso3", right_index=True, how="left")
    years = sorted(a["year"].unique())
    cols = {}
    for k in EXPOSURE:
        for t in years[1:]:
            cols[f"z_{k}_y{t}"] = a[f"z_{k}"] * (a["year"] == t)
    a = pd.concat([a, pd.DataFrame(cols, index=a.index)], axis=1)
    EXPO_YEAR[:] = list(cols)
    return a


def robustness(a: pd.DataFrame) -> pd.DataFrame:
    """Checks on the strict and old-founder event studies (TFP, billionaire countries)."""
    rows = []
    for shock in ("strict", "old founders"):
        key = shock.replace(" ", "_")
        variants = {
            "baseline": dict(d=a),
            "previous year's growth as a control": dict(d=a, extra=["g_tfp_lag"]),
            "change in billionaire wealth, same leads and lags, as controls": dict(d=a, extra=[f"dval_k{k}" for k in LEADS_LAGS]),
            "growth years 2020 and 2021 left out": dict(d=a[~a["year"].isin([2020, 2021])]),
            "United States left out": dict(d=a[a["iso3"] != "USA"]),
            "pandemic exposure x year as controls": dict(d=a, extra=EXPO_YEAR),
            "indicator: successions above 0.1% of GDP": dict(d=a, regs=[f"{key}_big_k{k}" for k in LEADS_LAGS]),
        }
        for name, v in variants.items():
            r = event_study(v["d"], shock, "tfp", "billionaire countries", extra=v.get("extra"), regs=v.get("regs"))
            rows.append(dict(shock=shock, variant=name, n_obs=r["n_obs"], post=r["post"], se_post=r["se_post"],
                             pre=r["pre"], se_pre=r["se_pre"], k_m1=r["b_k-1"], se_k_m1=r["se_k-1"]))
        base = event_study(a, shock, "tfp", "billionaire countries")
        drops = []
        ever = a.loc[a[f"{key}_k0"] > 0, "iso3"].unique()
        for i in sorted(set(ever) & set(a.loc[a["billionaire_country"], "iso3"])):
            r = event_study(a[a["iso3"] != i], shock, "tfp", "billionaire countries")
            drops.append((i, r["post"], r["se_post"]))
        lo = min(drops, key=lambda x: abs(x[1]))
        hi = max(drops, key=lambda x: abs(x[1]))
        rows.append(dict(shock=shock, variant="leave one country with a succession out", n_obs=len(drops),
                         post=base["post"], se_post=base["se_post"], pre=np.nan, se_pre=np.nan, k_m1=np.nan, se_k_m1=np.nan,
                         loo_closest=lo[1], loo_closest_se=lo[2], loo_closest_country=lo[0], loo_furthest=hi[1],
                         loo_furthest_country=hi[0],
                         loo_ci_excl_zero=int(sum(1 for _, b, se in drops if b + 1.96 * se < 0))))
    return pd.DataFrame(rows)


def predicted_from_panel(panel: pd.DataFrame, a: pd.DataFrame, succ: pd.DataFrame) -> dict:
    """What the within-country composition coefficient would predict for the observed successions."""
    d = prep(sample(panel, "billionaire_countries"))
    r = fit(d, "g_tfp_3", ["x_inh", "x_sum"], ["ln_y"] + SECTOR_CTRL, "fe")
    c1, c2 = r["b_x_inh"], r["b_x_sum"]
    gdp = pd.read_csv(DATA / "wdi_gdp_usd.csv").set_index(["iso3", "year"])["gdp_usd"]
    comp = panel.set_index(["iso3", "year"])
    rows = []
    for _, e in succ[succ["worth_succ_strict"] > 0].iterrows():
        i, t = e["iso3"], int(e["year"])
        if (i, t - 1) not in comp.index or not comp.loc[(i, t - 1), "billionaire_country"]:
            continue
        g = gdp.get((i, t - 1), np.nan)
        if not np.isfinite(g):
            continue
        inh, sm = comp.loc[(i, t - 1), "inherited_pct"], comp.loc[(i, t - 1), "self_made_pct"]
        w = 100 * e["worth_succ_strict"] * 1e6 / g
        dx_i = np.log1p(inh + w) - np.log1p(inh)
        dx_s = np.log1p(max(sm - w, 0.0)) - np.log1p(sm)
        rows.append(dict(iso3=i, year=t, w_pct=w, d=np.log1p(w), predicted=c1 * dx_i + c2 * (dx_i + dx_s)))
    p = pd.DataFrame(rows)
    return {"c_inh_minus_self": c1, "c_sum": c2, "events": p}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    countries = load_countries()
    panel, meta = build_panel(countries)
    succ, unmapped = load_successions(countries)
    a, shocks = annual_panel(panel, succ)
    a = add_exposure_years(a)
    a[[c for c in a.columns if "_y20" not in c]].to_csv(OUT / "annual_panel.csv", index=False, float_format="%.6f")
    cov = pd.read_csv(DATA / "forbes_coverage.csv")
    diag = pd.read_csv(DATA / "succession_diagnostics.csv")

    es_rows, designs = [], []
    for shock, outcome, smp in itertools.product(SHOCKS, ("tfp", "gdppc"),
                                                 ("billionaire countries", "ever billionaire", "all countries")):
        r = event_study(a, shock, outcome, smp)
        if r is None:
            continue
        designs.append(r.pop("_des"))
        es_rows.append(r)
    es = pd.DataFrame(es_rows)
    es.to_csv(OUT / "event_studies.csv", index=False, float_format="%.6f")
    perm_res = [permutation(a, shocks, sh, "tfp", smp)
                for sh, smp in itertools.product(("strict", "old founders", "old exits, any"),
                                                 ("billionaire countries", "ever billionaire"))]
    pd.DataFrame({f"{r['shock']} | {r['sample']}": r.pop("_draws") for r in perm_res}).to_csv(
        OUT / "permutation_draws.csv", index=False, float_format="%.5f")
    perms = pd.DataFrame(perm_res)
    rob = robustness(a)
    rob.to_csv(OUT / "robustness.csv", index=False, float_format="%.6f")
    gdp_rows = []
    for shock in ("strict", "old founders"):
        for name, d in (("baseline", a), ("growth years 2020 and 2021 left out", a[~a["year"].isin([2020, 2021])])):
            r = event_study(d, shock, "gdppc", "billionaire countries")
            gdp_rows.append(dict(shock=shock, variant=name, n_obs=r["n_obs"], post=r["post"], se_post=r["se_post"],
                                 pre=r["pre"], se_pre=r["se_pre"]))
    gdp_rob = pd.DataFrame(gdp_rows)
    perms.to_csv(OUT / "permutation.csv", index=False, float_format="%.6f")
    bc = set(a.loc[a["billionaire_country"], "iso3"])
    ev_tab = shocks[shocks["shock"].isin(["strict", "old founders"]) & shocks["iso3"].isin(bc) & (shocks["d"] > 0)].copy()
    ev_tab["pct_gdp"] = np.expm1(ev_tab["d"])
    ev_tab.drop(columns="d").to_csv(OUT / "succession_shocks.csv", index=False, float_format="%.5f")
    pred = predicted_from_panel(panel, a, succ)
    pred["events"].to_csv(OUT / "predicted_events.csv", index=False, float_format="%.6f")

    L: list[str] = []
    w = L.append
    w("# Results (every number quoted in the article)")
    w("")
    w("## Forbes lists and matching")
    ok = cov[pd.to_numeric(cov["status"], errors="coerce") == 200]
    w(f"lists fetched: {len(ok)} of {len(cov)} ({', '.join(str(y) for y in cov.loc[~cov.index.isin(ok.index), 'year']) or 'none failed'}); "
      f"people per list {int(ok['n_persons'].min())}-{int(ok['n_persons'].max())}; source filled: min {ok['share_source'].min():.4f}; "
      f"age known: min {ok['share_age'].min():.4f}; last name filled: min {ok['share_last_name'].min():.4f}")
    w(f"share of people on one list found on the next, by identifier: uri {cov['persistence_uri'].iloc[0]:.3f}, naturalId "
      f"{cov['persistence_naturalId'].iloc[0]:.3f}, name {cov['persistence_name'].iloc[0]:.3f}; identifier used: {cov['key_used'].iloc[0]}")
    dd = diag.fillna(0)
    tot = dd[["exits", "matched_strict", "matched_window_only", "unmatched", "exits_old", "matched_strict_old",
              "matched_window_only_old", "unmatched_old", "new_heirs", "new_heirs_matched_strict",
              "new_heirs_next_list_matched", "new_heirs_generic_source"]].sum()
    w(f"list pairs {int(dd['from_list'].min())}-{int(dd['to_list'].max())}: self-made exits {int(tot['exits'])}; matched to new heirs "
      f"on the next list {int(tot['matched_strict'])} ({tot['matched_strict'] / tot['exits']:.3f}); only on the list after "
      f"{int(tot['matched_window_only'])}; unmatched {int(tot['unmatched'])}")
    w(f"exits aged {70}+: {int(tot['exits_old'])} ({tot['exits_old'] / tot['exits']:.3f} of exits); of them matched on the next list "
      f"{int(tot['matched_strict_old'])} ({tot['matched_strict_old'] / tot['exits_old']:.3f}); exits under 70 or of unknown age matched "
      f"{int(tot['matched_strict'] - tot['matched_strict_old'])} of {int(tot['exits'] - tot['exits_old'])} "
      f"({(tot['matched_strict'] - tot['matched_strict_old']) / (tot['exits'] - tot['exits_old']):.3f})")
    by = dd[[c for c in ("strict_by_name_only", "strict_by_source_only", "strict_by_both") if c in dd.columns]].sum()
    w("strict matches by rule: " + ", ".join(f"{k.replace('strict_by_', '').replace('_', ' ')} {int(v)}" for k, v in by.items()))
    w(f"new inherited entries {int(tot['new_heirs'])}; matched to a founder's exit on the same pair {int(tot['new_heirs_matched_strict'])} "
      f"({tot['new_heirs_matched_strict'] / tot['new_heirs']:.3f}); matched a list later {int(tot['new_heirs_next_list_matched'])}; "
      f"with a generic source {int(tot['new_heirs_generic_source'])}")
    for _, r in dd.iterrows():
        w(f"  {int(r['from_list'])}->{int(r['to_list'])}: exits {int(r['exits'])}, strict {int(r.get('matched_strict', 0))}, "
          f"window only {int(r.get('matched_window_only', 0))}, new heirs {int(r['new_heirs'])}, heirs matched "
          f"{int(r['new_heirs_matched_strict'])}, generic sources {int(r['generic_sources'])}")
    w(f"unmapped citizenships: {', '.join(unmapped) or 'none'}")
    sv = succ[succ["worth_succ_strict"] > 0]
    w(f"country-years with a strict succession: {len(sv)} in {sv['iso3'].nunique()} countries; successions {int(sv['n_succ_strict'].sum())}; "
      f"founders' wealth {sv['worth_succ_strict'].sum() / 1e3:.1f} bn; heirs' wealth on the next list {sv['heirs_worth_strict'].sum() / 1e3:.1f} bn")
    tt = succ[[c for c in succ.columns if c.startswith(("n_", "worth_", "heirs_"))]].sum()
    w(f"mean founder wealth on the last list ($ bn): matched successions, founders 70+ "
      f"{tt['worth_succ_strict_old'] / tt['n_succ_strict_old'] / 1e3:.2f} (n={int(tt['n_succ_strict_old'])}); "
      f"matched, younger founders {(tt['worth_succ_strict'] - tt['worth_succ_strict_old']) / (tt['n_succ_strict'] - tt['n_succ_strict_old']) / 1e3:.2f} "
      f"(n={int(tt['n_succ_strict'] - tt['n_succ_strict_old'])}); unmatched exits, founders 70+ "
      f"{tt['worth_exit_unmatched_old'] / tt['n_exit_unmatched_old'] / 1e3:.2f} (n={int(tt['n_exit_unmatched_old'])}); unmatched exits, "
      f"younger {(tt['worth_exit_unmatched'] - tt['worth_exit_unmatched_old']) / (tt['n_exit_unmatched'] - tt['n_exit_unmatched_old']) / 1e3:.2f} "
      f"(n={int(tt['n_exit_unmatched'] - tt['n_exit_unmatched_old'])})")
    w(f"share of departing wealth of founders aged 70+ attributed to matched successions: "
      f"{tt['worth_succ_strict_old'] / (tt['worth_succ_strict_old'] + tt['worth_exit_unmatched_old']):.3f}; "
      f"heirs per matched succession {tt['n_heirs_strict'] / tt['n_succ_strict']:.2f}; heirs' wealth on the next list as a share of the "
      f"founders' {tt['heirs_worth_strict'] / tt['worth_succ_strict']:.3f}")
    top = sv.groupby("iso3")["n_succ_strict"].sum().sort_values(ascending=False).head(10)
    w("successions by country (largest ten): " + ", ".join(f"{i} {int(v)}" for i, v in top.items()))
    st = shocks[(shocks["shock"] == "strict") & shocks["iso3"].isin(set(a.loc[a["billionaire_country"], "iso3"]))]
    pos = st[st["d"] > 0]
    pct = np.expm1(pos["d"])
    w(f"strict shock in billionaire countries, list years 2012-2026: {len(pos)} country-years > 0 of {int(st['d'].notna().sum())}; "
      f"wealth passing to heirs, % of GDP: median {pct.median():.3f}, 75th pct {pct.quantile(0.75):.3f}, 90th pct {pct.quantile(0.9):.3f}, max {pct.max():.3f}")
    for thr in (0.1, 0.5, 1.0):
        w(f"  country-years with successions above {thr}% of GDP: {int((pct > thr).sum())} in {pos.loc[pct > thr, 'iso3'].nunique()} countries")
    w("")
    w("## Event studies (annual growth in pp; shock = log(1 + wealth passing to heirs as % of GDP); k = growth year minus list year)")
    for _, r in es.iterrows():
        def ci(b, s):
            return f"{b:+.3f} [{b - 1.96 * s:+.3f}, {b + 1.96 * s:+.3f}]"
        w(f"{r['shock']} / {r['outcome']} / {r['sample']} (n={r['n_obs']}, countries={r['n_countries']}): "
          + "; ".join(f"k={k} {ci(r[f'b_k{k}'], r[f'se_k{k}'])}" for k in LEADS_LAGS)
          + f"; post (k=0..2) {ci(r['post'], r['se_post'])}; pre (k=-3..-2) {ci(r['pre'], r['se_pre'])}")
    w("")
    w("## Permutation of shock timing within country (1,000 draws; cumulative post effect, TFP)")
    for _, r in perms.iterrows():
        w(f"{r['shock']} / {r['sample']}: observed {r['observed_post']:+.3f}; null 5th-50th-95th pct {r['null_p5']:+.3f}, "
          f"{r['null_p50']:+.3f}, {r['null_p95']:+.3f}; share of draws at or below observed {r['p_le']:.3f}")
    w("")
    w("## Robustness (TFP, billionaire countries; post = sum of k=0..2; k=-1 = growth in the year before the list year)")
    for _, r in rob.iterrows():
        if r["variant"].startswith("leave one"):
            w(f"{r['shock']} / {r['variant']}: {int(r['n_obs'])} countries; full {r['post']:+.3f}; closest to zero {r['loo_closest']:+.3f} "
              f"[{r['loo_closest'] - 1.96 * r['loo_closest_se']:+.3f}, {r['loo_closest'] + 1.96 * r['loo_closest_se']:+.3f}] without "
              f"{r['loo_closest_country']}; furthest {r['loo_furthest']:+.3f} without {r['loo_furthest_country']}; "
              f"intervals excluding zero {int(r['loo_ci_excl_zero'])} of {int(r['n_obs'])}")
        else:
            w(f"{r['shock']} / {r['variant']} (n={int(r['n_obs'])}): post {r['post']:+.3f} [{r['post'] - 1.96 * r['se_post']:+.3f}, "
              f"{r['post'] + 1.96 * r['se_post']:+.3f}]; pre {r['pre']:+.3f} [{r['pre'] - 1.96 * r['se_pre']:+.3f}, "
              f"{r['pre'] + 1.96 * r['se_pre']:+.3f}]; k=-1 {r['k_m1']:+.3f} [{r['k_m1'] - 1.96 * r['se_k_m1']:+.3f}, "
              f"{r['k_m1'] + 1.96 * r['se_k_m1']:+.3f}]")
    w("GDP per capita (billionaire countries):")
    for _, r in gdp_rob.iterrows():
        w(f"  {r['shock']} / {r['variant']} (n={int(r['n_obs'])}): post {r['post']:+.3f} [{r['post'] - 1.96 * r['se_post']:+.3f}, "
          f"{r['post'] + 1.96 * r['se_post']:+.3f}]; pre {r['pre']:+.3f} [{r['pre'] - 1.96 * r['se_pre']:+.3f}, {r['pre'] + 1.96 * r['se_pre']:+.3f}]")
    w("")
    ev = pred["events"]
    w("## What the within-country composition coefficient predicts for these successions")
    w(f"headline fit (TFP, 3 years, billionaire countries, FE, sector shares): inherited minus self-made {pred['c_inh_minus_self']:+.3f}, "
      f"coefficient on the sum {pred['c_sum']:+.3f}")
    w(f"successions in billionaire countries with composition data: {len(ev)}; wealth passing to heirs median {ev['w_pct'].median():.3f}% of GDP; "
      f"predicted change in annual TFP growth over the next three years: median {ev['predicted'].median():+.4f} pp, mean {ev['predicted'].mean():+.4f} pp, "
      f"10th pct {ev['predicted'].quantile(0.1):+.4f}")
    strict_bc = es[(es["shock"] == "strict") & (es["outcome"] == "tfp") & (es["sample"] == "billionaire countries")].iloc[0]
    w(f"event-study implied change for the median succession (post sum / 3 x median shock): "
      f"{strict_bc['post'] / 3 * ev['d'].median():+.4f} pp a year; for the mean shock {strict_bc['post'] / 3 * ev['d'].mean():+.4f}")
    text = "\n".join(L) + "\n"
    (OUT / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
