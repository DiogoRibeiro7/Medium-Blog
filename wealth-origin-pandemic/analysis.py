"""Does the inherited-wealth / TFP association survive controls for pandemic exposure?

    python analysis.py

Writes results/*.csv and results/summary.txt.

The previous project (wealth-origin-sectors) showed that the within-country association
between inherited billionaire wealth and slower TFP growth survives controls for the sector mix
of billionaire wealth, but still rests on the 2019 and 2020 Forbes lists, whose growth windows
contain the 2020 collapse and the recovery. This project asks whether that part of the
association reflects each economy's exposure to the pandemic shock rather than the origin of its
fortunes.

Exposure is measured before or independently of billionaire composition: lockdown stringency
(OxCGRT, mean of 2020-2021), cumulative COVID deaths per million at the end of 2021 (OxCGRT and
PWT population), tourism receipts as a share of GDP (WDI, mean of 2015-2019) and the share of
employment in contact-intensive activities (ILOSTAT, ISIC Rev. 4 sections G, H, I, R and S, latest
year 2015-2019). Each enters interacted with list-year dummies, so that every exposure measure
gets its own growth path over time; with country fixed effects, that absorbs any differential
pandemic shock that is linear in exposure.

The billionaire data, the panel and the sector controls are those of wealth-origin-sectors,
copied unchanged (data/PROVENANCE.txt).
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
# exposure


EXPOSURES = {
    "stringency": "lockdown stringency, mean 2020-2021 (OxCGRT index, 0-100)",
    "deaths_pm": "confirmed COVID deaths per million by end-2021 (OxCGRT, PWT population)",
    "tourism_gdp": "travel credits in the balance of payments, % of GDP, mean 2015-2019 (WDI; UNWTO receipts where missing)",
    "contact_emp": "employment share in trade, transport and communication, accommodation and food, and other services, 2019 (ILO modelled estimates)",
}
MODELLED_CONTACT = ["ECO_ISIC4_G", "ECO_ISIC4_HJ", "ECO_ISIC4_I", "ECO_ISIC4_RSTU"]
SURVEY_CONTACT = ["ECO_ISIC4_G", "ECO_ISIC4_H", "ECO_ISIC4_I", "ECO_ISIC4_R", "ECO_ISIC4_S"]


def load_exposure() -> pd.DataFrame:
    ox = pd.read_csv(DATA / "oxcgrt_country_year.csv")
    pwt = pd.read_csv(DATA / "pwt110_panel.csv").rename(columns={"countrycode": "iso3"})
    pop19 = pwt[pwt["year"] == 2019].set_index("iso3")["pop"]  # millions
    e = pd.DataFrame(index=sorted(set(ox["iso3"]) | set(pop19.index)))
    e["stringency"] = ox[ox["year"].isin([2020, 2021])].groupby("iso3")["StringencyIndex_Average"].mean()
    deaths21 = ox[ox["year"] == 2021].set_index("iso3")["deaths_cum_year_end"]
    e["deaths_pm"] = deaths21 / pop19  # deaths per million people

    w = pd.read_csv(DATA / "wdi_exposure.csv")
    w = w[w["year"].between(2015, 2019)].pivot_table(index=["iso3", "year"], columns="indicator", values="value")
    gdp = pd.read_csv(DATA / "wdi_gdp_usd.csv").set_index(["iso3", "year"])["gdp_usd"]
    w = w.join(gdp, how="left")
    bop = 100 * w["travel_pct_service_exports"] / 100 * w["service_exports_usd"] / w["gdp_usd"]
    unwto = 100 * w["tourism_receipts_usd"] / w["gdp_usd"]
    e["tourism_bop"] = bop.groupby("iso3").mean()
    e["tourism_unwto"] = unwto.groupby("iso3").mean()
    e["tourism_gdp"] = e["tourism_bop"].fillna(e["tourism_unwto"])
    e["tourism_source"] = np.where(e["tourism_bop"].notna(), "bop", np.where(e["tourism_unwto"].notna(), "unwto", ""))

    m = pd.read_csv(DATA / "ilostat_emp_modelled.csv")
    m = m[m["time"] == 2019].pivot_table(index="iso3", columns="classif1", values="obs_value", aggfunc="sum")
    e["contact_emp"] = 100 * m[MODELLED_CONTACT].sum(axis=1) / m["ECO_ISIC4_TOTAL"]
    e["contact_core_modelled"] = 100 * m[["ECO_ISIC4_G", "ECO_ISIC4_I"]].sum(axis=1) / m["ECO_ISIC4_TOTAL"]

    il = pd.read_csv(DATA / "ilostat_emp_isic4.csv")
    il = il[il["time"].between(2014, 2019)]
    piv = il.pivot_table(index=["iso3", "time", "source"], columns="classif1", values="obs_value", aggfunc="sum")
    piv = piv.dropna(subset=SURVEY_CONTACT + ["ECO_ISIC4_TOTAL"]).reset_index()
    piv = piv.sort_values(["iso3", "time", "ECO_ISIC4_TOTAL"]).groupby("iso3").tail(1).set_index("iso3")
    e["contact_survey"] = 100 * piv[SURVEY_CONTACT].sum(axis=1) / piv["ECO_ISIC4_TOTAL"]
    e["contact_survey_year"] = piv["time"]
    e.index.name = "iso3"
    return e


def add_exposure(panel: pd.DataFrame, expo: pd.DataFrame) -> pd.DataFrame:
    p = panel.merge(expo[list(EXPOSURES)], left_on="iso3", right_index=True, how="left")
    # standardise over the countries that ever have a billionaire, so units are comparable
    base = p[p["billionaire_country"]].drop_duplicates("iso3")
    for k in EXPOSURES:
        p[f"z_{k}"] = (p[k] - base[k].mean()) / base[k].std()
    return p


EXPO_SETS = {
    "none": [],
    "stringency": ["z_stringency"],
    "deaths": ["z_deaths_pm"],
    "tourism": ["z_tourism_gdp"],
    "contact": ["z_contact_emp"],
    "all four": [f"z_{k}" for k in EXPOSURES],
}
COMMON = [f"z_{k}" for k in EXPOSURES]  # every fit is run on country-years with all four measures


def fit_expo(d: pd.DataFrame, y: str, regs: list[str], controls: list[str], expo: list[str],
             est: str, common: list[str] = COMMON) -> dict | None:
    """Fit with each exposure measure interacted with list-year dummies (plus its level if pooled)."""
    d = d.dropna(subset=[y] + regs + controls + common).copy()
    years = sorted(d["year"].unique())
    extra = []
    for e in expo:
        if est == "pooled":
            extra.append(e)
        for t in years[1:]:
            col = f"{e}_y{t}"
            d[col] = d[e] * (d["year"] == t)
            extra.append(col)
    return fit(d, y, regs, controls + extra, est)


BASE_CTRL = ["ln_y"] + SECTOR_CTRL


def exposure_fits(panel: pd.DataFrame, keep: list | None = None) -> pd.DataFrame:
    """For every exposure set: the coefficient without and with it, on the same country-years."""
    rows = []
    for var, h, smp, est in itertools.product(("gdppc", "tfp"), (1, 3, 5),
                                              ("billionaire_countries", "with_billionaires"), ("fe", "pooled")):
        d = prep(sample(panel, smp))
        y = f"g_{var}_{h}"
        row = dict(outcome=var, horizon=h, sample=smp, estimator=est)
        r0 = fit(d, y, ["x_inh", "x_sum"], BASE_CTRL, est)  # the previous article's estimate and sample
        row.update(b_prev=r0["b_x_inh"], se_prev=r0["se_x_inh"], n_prev=r0["n_obs"], c_prev=r0["n_countries"])
        for name, ex in EXPO_SETS.items():
            if name == "none":
                continue
            key = name.replace(" ", "_")
            rb = fit_expo(d, y, ["x_inh", "x_sum"], BASE_CTRL, [], est, common=ex)
            rx = fit_expo(d, y, ["x_inh", "x_sum"], BASE_CTRL, ex, est, common=ex)
            row.update({f"b0_{key}": rb["b_x_inh"], f"se0_{key}": rb["se_x_inh"],
                        f"b_{key}": rx["b_x_inh"], f"se_{key}": rx["se_x_inh"],
                        f"n_{key}": rx["n_obs"], f"c_{key}": rx["n_countries"]})
            if name == "all four" and keep is not None:
                keep.append({**rx["_des"], "b": rx["b_x_inh"]})
        rows.append(row)
    return pd.DataFrame(rows)


def average_fits(panel: pd.DataFrame, keep: list) -> pd.DataFrame:
    """All 24 specifications with composition averaged over lists t-1 and t, without and with all exposures."""
    rows = []
    regs = MEASURES_AVG
    for var, h, smp, est in itertools.product(("gdppc", "tfp"), (1, 3, 5),
                                              ("billionaire_countries", "with_billionaires"), ("fe", "pooled")):
        d = prep(sample(panel, smp))
        y = f"g_{var}_{h}"
        rb = fit_expo(d, y, regs, BASE_CTRL, [], est)
        rx = fit_expo(d, y, regs, BASE_CTRL, COMMON, est)
        keep.append({**rx["_des"], "b": rx["b_" + regs[0]]})
        rows.append(dict(outcome=var, horizon=h, sample=smp, estimator=est, n_obs=rx["n_obs"], n_countries=rx["n_countries"],
                         avg_base=rb["b_" + regs[0]], se_avg_base=rb["se_" + regs[0]],
                         avg=rx["b_" + regs[0]], se_avg=rx["se_" + regs[0]]))
    return pd.DataFrame(rows)


MEASURES_AVG = ["x_inh_avg2", "x_sum_avg2"]


def exposure_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    """Country-level: how exposure relates to the inherited share and to its change around 2020."""
    p = panel[panel["billionaire_country"]].copy()
    p = prep(p)
    p["gap"] = p["x_inh"] - np.log1p(p["self_made_pct"])
    c = p.groupby("iso3").agg(inh_share=("inh_share", "mean"), gap=("gap", "mean"),
                              **{k: (k, "first") for k in EXPOSURES})
    for t0, t1 in ((2016, 2019), (2019, 2020), (2017, 2020)):
        a = p[p["year"] == t0].set_index("iso3")["gap"]
        b = p[p["year"] == t1].set_index("iso3")["gap"]
        c[f"dgap_{t0}_{t1}"] = b - a
    rows = []
    for k in EXPOSURES:
        for v in ("inh_share", "gap", "dgap_2016_2019", "dgap_2019_2020", "dgap_2017_2020"):
            x = c[[k, v]].dropna()
            rho = x.corr(method="spearman").iloc[0, 1]
            rows.append(dict(exposure=k, variable=v, n=len(x), spearman=rho))
    return pd.DataFrame(rows), c


def dip_and_rebound(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sections around 2020: composition on the 2019 (or 2020) list against TFP growth after it."""
    pwt = pd.read_csv(DATA / "pwt110_panel.csv").rename(columns={"countrycode": "iso3"}).set_index(["iso3", "year"])
    tfp = pwt["rtfpna"]
    windows = {"dip 2019-2020": (2019, 2019, 2020, 1), "rebound 2020-2023": (2020, 2020, 2023, 3),
               "net 2019-2022": (2019, 2019, 2022, 3), "net 2019-2023": (2019, 2019, 2023, 4),
               "pre 2016-2019": (2016, 2016, 2019, 3)}
    rows = []
    p = prep(panel[panel["billionaire_country"]])
    for (label, (list_year, t0, t1, n)), ex in itertools.product(windows.items(), ("none", "all four")):
        d = p[p["year"] == list_year].copy()
        d["y"] = [100 * (np.log(tfp.get((i, t1), np.nan)) - np.log(tfp.get((i, t0), np.nan))) / n for i in d["iso3"]]
        cols = ["x_inh", "x_sum", "ln_y"] + SECTOR_CTRL + EXPO_SETS[ex]
        d = d.dropna(subset=["y"] + cols + COMMON)
        X = np.column_stack([np.ones(len(d)), d[cols].to_numpy()])
        b, se = ols(d["y"].to_numpy(), X)
        rows.append(dict(window=label, exposure=ex, n=len(d), b=b[1], se=se[1],
                         mean_y=d["y"].mean(), sd_y=d["y"].std()))
    out = pd.DataFrame(rows)
    out["lo"], out["hi"] = out["b"] - 1.96 * out["se"], out["b"] + 1.96 * out["se"]
    return out


def split_and_years(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Period split and leave-one-list-year-out for TFP, fixed effects, without and with all exposures."""
    srows, yrows = [], []
    for smp, h in itertools.product(("billionaire_countries", "with_billionaires"), (1, 3, 5)):
        d = prep(sample(panel, smp))
        y = f"g_tfp_{h}"
        for label, part in (("ends by 2019", d[d["year"] + h <= 2019]), ("includes 2020", d[d["year"] + h >= 2020])):
            for ex in ("none", "all four"):
                r = fit_expo(part, y, ["x_inh", "x_sum"], BASE_CTRL, EXPO_SETS[ex], "fe")
                if r is None:
                    continue
                yrs = part.dropna(subset=[y])["year"]
                srows.append(dict(sample=smp, horizon=h, windows=label, exposure=ex,
                                  list_years=f"{int(yrs.min())}-{int(yrs.max())}", n_obs=r["n_obs"],
                                  b=r["b_x_inh"], se=r["se_x_inh"]))
        if h in (3, 5):
            yrs = sorted(d.dropna(subset=[y])["year"].unique())
            for ex in ("none", "all four"):
                full = fit_expo(d, y, ["x_inh", "x_sum"], BASE_CTRL, EXPO_SETS[ex], "fe")
                for t in yrs:
                    r = fit_expo(d[d["year"] != t], y, ["x_inh", "x_sum"], BASE_CTRL, EXPO_SETS[ex], "fe")
                    yrows.append(dict(sample=smp, horizon=h, exposure=ex, dropped_year=int(t), b=r["b_x_inh"],
                                      se=r["se_x_inh"], full_b=full["b_x_inh"], full_se=full["se_x_inh"]))
    s, yy = pd.DataFrame(srows), pd.DataFrame(yrows)
    for t in (s, yy):
        t["lo"], t["hi"] = t["b"] - 1.96 * t["se"], t["b"] + 1.96 * t["se"]
    return s, yy


def pre_post(panel: pd.DataFrame, draws: int = 1000, seed: int = 20261116) -> pd.DataFrame:
    """Forward minus backward coefficient with all exposures, fixed effects, country bootstrap."""
    rng = np.random.default_rng(seed)
    rows = []
    for var, smp, h in itertools.product(("gdppc", "tfp"), ("billionaire_countries", "with_billionaires"), (3, 5)):
        d = prep(sample(panel, smp))
        res = {}
        for side, y, inc in (("fwd", f"g_{var}_{h}", "ln_y"), ("bwd", f"b_{var}_{h}", f"ln_y_m{h}")):
            r = fit_expo(d, y, ["x_inh", "x_sum"], [inc] + SECTOR_CTRL, COMMON, "fe")
            res[side] = r
        idx = {s: pd.Series(np.arange(len(r["_des"]["groups"]))).groupby(r["_des"]["groups"]).apply(np.array).to_dict()
               for s, r in res.items()}
        cs = np.array(sorted(set(idx["fwd"]) | set(idx["bwd"])))
        diffs = []
        for _ in range(draws):
            pick = rng.choice(cs, size=len(cs), replace=True)
            est = {}
            for s in ("fwd", "bwd"):
                rows_ = np.concatenate([idx[s][c] for c in pick if c in idx[s]])
                des = res[s]["_des"]
                est[s] = np.linalg.lstsq(des["X"][rows_], des["y"][rows_], rcond=None)[0][0]
            diffs.append(est["fwd"] - est["bwd"])
        diffs = np.array(diffs)
        f, b = res["fwd"]["b_x_inh"], res["bwd"]["b_x_inh"]
        rows.append(dict(outcome=var, sample=smp, horizon=h, forward=f, backward=b, fwd_minus_bwd=f - b,
                         lo=np.percentile(diffs, 2.5), hi=np.percentile(diffs, 97.5), p_ge_0=float(np.mean(diffs >= 0))))
    return pd.DataFrame(rows)


def add_counts(panel: pd.DataFrame, forbes: pd.DataFrame) -> pd.DataFrame:
    """Headcounts by origin per 10 million people, and composition averaged over lists t-1 and t."""
    k = forbes[forbes["origin"].isin(["inherited", "self_made"])]
    cnt = k.pivot_table(index=["iso3", "year"], columns="origin", values="n", aggfunc="sum", fill_value=0)
    cnt.columns = [f"n_{c}" for c in cnt.columns]
    pwt = pd.read_csv(DATA / "pwt110_panel.csv").rename(columns={"countrycode": "iso3"}).set_index(["iso3", "year"])["pop"]
    p = panel.merge(cnt, left_on=["iso3", "year"], right_index=True, how="left")
    p[["n_inherited", "n_self_made"]] = p[["n_inherited", "n_self_made"]].fillna(0)
    pop = np.array([pwt.get((i, t), np.nan) for i, t in zip(p["iso3"], p["year"])])
    p["xn_inh"] = np.log1p(10 * p["n_inherited"] / pop)
    p["xn_sum"] = p["xn_inh"] + np.log1p(10 * p["n_self_made"] / pop)
    q = prep(p).sort_values(["iso3", "year"])
    for c in ("x_inh", "x_sum"):
        prev = q.groupby("iso3")[c].shift(1)
        gap = q["year"] - q.groupby("iso3")["year"].shift(1)
        p.loc[q.index, f"{c}_avg2"] = np.where(gap == 1, (q[c] + prev) / 2, np.nan)
    return p


def measurement_checks(panel: pd.DataFrame) -> pd.DataFrame:
    """The headline with composition measured by headcount, or averaged over two lists (TFP, FE, all exposures)."""
    rows = []
    for smp, h in itertools.product(("billionaire_countries", "with_billionaires"), (3, 5)):
        d = prep(sample(panel, smp))
        y = f"g_tfp_{h}"
        variants = {
            "wealth, as in the main fits": (d, ["x_inh", "x_sum"]),
            "headcount per 10 million people": (d, ["xn_inh", "xn_sum"]),
            "wealth, lists 2012 onwards (same sample as the two-list average)": (d[d["year"] >= 2012], ["x_inh", "x_sum"]),
            "wealth averaged over lists t-1 and t": (d, ["x_inh_avg2", "x_sum_avg2"]),
        }
        for name, (dd, regs) in variants.items():
            for drop2020 in (False, True):
                part = dd[dd["year"] != 2020] if drop2020 else dd
                r = fit_expo(part, y, regs, BASE_CTRL, COMMON, "fe")
                rows.append(dict(sample=smp, horizon=h, measure=name, without_2020_list=drop2020,
                                 n_obs=r["n_obs"], b=r["b_" + regs[0]], se=r["se_" + regs[0]]))
    out = pd.DataFrame(rows)
    out["lo"], out["hi"] = out["b"] - 1.96 * out["se"], out["b"] + 1.96 * out["se"]
    return out


MEASURES = {"wealth": ["x_inh", "x_sum"], "headcount": ["xn_inh", "xn_sum"], "two-list average": ["x_inh_avg2", "x_sum_avg2"]}


def measures_by_year(panel: pd.DataFrame, draws: int = 1000, seed: int = 20261117) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each composition measure (billionaire countries, TFP, FE, all exposures): leave one list year
    out, and the forward-minus-backward bootstrap."""
    rng = np.random.default_rng(seed)
    d = prep(sample(panel, "billionaire_countries"))
    yrows, prows = [], []
    for (m, regs), h in itertools.product(MEASURES.items(), (3, 5)):
        y = f"g_tfp_{h}"
        full = fit_expo(d, y, regs, BASE_CTRL, COMMON, "fe")
        for t in sorted(d.dropna(subset=[y] + regs)["year"].unique()):
            r = fit_expo(d[d["year"] != t], y, regs, BASE_CTRL, COMMON, "fe")
            yrows.append(dict(measure=m, horizon=h, dropped_year=int(t), b=r["b_" + regs[0]], se=r["se_" + regs[0]],
                              full_b=full["b_" + regs[0]], full_se=full["se_" + regs[0]]))
        for label, part in (("ends by 2019", d[d["year"] + h <= 2019]), ("includes 2020", d[d["year"] + h >= 2020])):
            r = fit_expo(part, y, regs, BASE_CTRL, COMMON, "fe")
            yrs = part.dropna(subset=[y] + regs)["year"]
            yrows.append(dict(measure=m, horizon=h, dropped_year=-1, split=label,
                              list_years=f"{int(yrs.min())}-{int(yrs.max())}", n_obs=r["n_obs"],
                              b=r["b_" + regs[0]], se=r["se_" + regs[0]], full_b=full["b_" + regs[0]],
                              full_se=full["se_" + regs[0]]))
        bwd = fit_expo(d, f"b_tfp_{h}", regs, [f"ln_y_m{h}"] + SECTOR_CTRL, COMMON, "fe")
        res = {"fwd": full, "bwd": bwd}
        idx = {k: pd.Series(np.arange(len(r["_des"]["groups"]))).groupby(r["_des"]["groups"]).apply(np.array).to_dict()
               for k, r in res.items()}
        cs = np.array(sorted(set(idx["fwd"]) | set(idx["bwd"])))
        diffs = []
        for _ in range(draws):
            pick = rng.choice(cs, size=len(cs), replace=True)
            est = {}
            for k in ("fwd", "bwd"):
                rr = np.concatenate([idx[k][c] for c in pick if c in idx[k]])
                des = res[k]["_des"]
                est[k] = np.linalg.lstsq(des["X"][rr], des["y"][rr], rcond=None)[0][0]
            diffs.append(est["fwd"] - est["bwd"])
        diffs = np.array(diffs)
        f_, b_ = full["b_" + regs[0]], bwd["b_" + regs[0]]
        prows.append(dict(measure=m, horizon=h, forward=f_, se_forward=full["se_" + regs[0]], backward=b_,
                          se_backward=bwd["se_" + regs[0]], fwd_minus_bwd=f_ - b_, lo=np.percentile(diffs, 2.5),
                          hi=np.percentile(diffs, 97.5), p_ge_0=float(np.mean(diffs >= 0))))
    yy = pd.DataFrame(yrows)
    yy["lo"], yy["hi"] = yy["b"] - 1.96 * yy["se"], yy["b"] + 1.96 * yy["se"]
    return yy, pd.DataFrame(prows)


def exposure_paths(panel: pd.DataFrame) -> pd.DataFrame:
    """Coefficients on exposure x list year in the headline fit: does exposure track the 2020 shock?"""
    d = prep(sample(panel, "billionaire_countries"))
    rows = []
    for h in (1, 3):
        y = f"g_tfp_{h}"
        dd = d.dropna(subset=[y, "x_inh", "x_sum"] + BASE_CTRL + COMMON).copy()
        years = sorted(dd["year"].unique())
        for e in COMMON:
            extra = []
            for t in years[1:]:
                dd[f"{e}_y{t}"] = dd[e] * (dd["year"] == t)
                extra.append(f"{e}_y{t}")
            r = fit(dd, y, ["x_inh", "x_sum"] + extra, BASE_CTRL, "fe")
            for t, col in zip(years[1:], extra):
                rows.append(dict(horizon=h, exposure=e[2:], year=int(t), b=r[f"b_{col}"], se=r[f"se_{col}"]))
    out = pd.DataFrame(rows)
    out["lo"], out["hi"] = out["b"] - 1.96 * out["se"], out["b"] + 1.96 * out["se"]
    return out


# ---------------------------------------------------------------------------------------


def ci(b: float, s: float, k: int = 3) -> str:
    return f"{b:+.{k}f} [{b - 1.96 * s:+.{k}f}, {b + 1.96 * s:+.{k}f}]"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    countries = load_countries()
    panel, meta = build_panel(countries)
    panel = add_counts(panel, meta["forbes"])
    expo = load_exposure()
    expo.to_csv(OUT / "exposure_by_country.csv", float_format="%.4f")
    panel = add_exposure(panel, expo)
    panel[[c for c in panel.columns if not c.startswith("_")]].to_csv(OUT / "panel.csv", index=False, float_format="%.6f")

    designs: list[dict] = []
    fits = exposure_fits(panel, designs)
    fits.to_csv(OUT / "fits.csv", index=False, float_format="%.6f")
    fits_for_joint = fits.rename(columns={"b_all_four": "all_four"})
    joint = joint_test(fits_for_joint, designs, "all_four")
    joint.to_csv(OUT / "joint_test.csv", index=False, float_format="%.6f")
    adesigns: list[dict] = []
    afits = average_fits(panel, adesigns)
    afits.to_csv(OUT / "average_fits.csv", index=False, float_format="%.6f")
    ajoint = joint_test(afits, adesigns, "avg", seed=20261118)
    ajoint.to_csv(OUT / "average_joint_test.csv", index=False, float_format="%.6f")
    corr, cmeans = exposure_correlations(panel)
    corr.to_csv(OUT / "exposure_correlations.csv", index=False, float_format="%.6f")
    cmeans.to_csv(OUT / "country_means.csv", float_format="%.6f")
    dr = dip_and_rebound(panel)
    dr.to_csv(OUT / "dip_and_rebound.csv", index=False, float_format="%.6f")
    split, loyo = split_and_years(panel)
    split.to_csv(OUT / "period_split.csv", index=False, float_format="%.6f")
    loyo.to_csv(OUT / "leave_one_year_out.csv", index=False, float_format="%.6f")
    pp = pre_post(panel)
    pp.to_csv(OUT / "pre_post.csv", index=False, float_format="%.6f")
    paths = exposure_paths(panel)
    paths.to_csv(OUT / "exposure_paths.csv", index=False, float_format="%.6f")
    meas = measurement_checks(panel)
    meas.to_csv(OUT / "measurement_checks.csv", index=False, float_format="%.6f")
    myy, mpp = measures_by_year(panel)
    myy.to_csv(OUT / "measures_leave_one_year_out.csv", index=False, float_format="%.6f")
    mpp.to_csv(OUT / "measures_pre_post.csv", index=False, float_format="%.6f")

    L: list[str] = []
    w = L.append
    w("# Results (every number quoted in the article)")
    w("")
    w("## Exposure measures")
    bc = panel[panel["billionaire_country"]].drop_duplicates("iso3")
    for k, lab in EXPOSURES.items():
        s = bc[k].dropna()
        w(f"{k}: {lab}; billionaire countries with a value {len(s)} of {len(bc)}; median {s.median():.2f}, "
          f"10th-90th pct {s.quantile(0.1):.2f}-{s.quantile(0.9):.2f}")
    all4 = bc.dropna(subset=list(EXPOSURES))
    w(f"billionaire countries with all four measures: {len(all4)}; missing at least one: "
      f"{', '.join(sorted(set(bc['iso3']) - set(all4['iso3'])))}")
    bci = bc["iso3"]
    src = expo.loc[expo.index.isin(bci), "tourism_source"]
    w(f"tourism source among billionaire countries: balance of payments {int((src == 'bop').sum())}, UNWTO receipts {int((src == 'unwto').sum())}, none {int((src == '').sum())}")
    both = expo.loc[expo.index.isin(bci)].dropna(subset=["tourism_bop", "tourism_unwto"])
    w(f"BoP travel vs UNWTO receipts where both exist (n={len(both)}): Spearman {both[['tourism_bop', 'tourism_unwto']].corr(method='spearman').iloc[0, 1]:.3f}")
    cs = expo.loc[expo.index.isin(bci)].dropna(subset=["contact_emp", "contact_survey"])
    w(f"modelled vs survey contact-intensive employment where both exist (n={len(cs)}): Spearman "
      f"{cs[['contact_emp', 'contact_survey']].corr(method='spearman').iloc[0, 1]:.3f}; median difference "
      f"{(cs['contact_emp'] - cs['contact_survey']).median():+.2f} pp")
    cm = bc.set_index("iso3")[list(EXPOSURES)].corr(method="spearman")
    w("Spearman correlations between exposure measures (billionaire countries): " + "; ".join(
        f"{a}-{b} {cm.loc[a, b]:+.2f}" for a, b in itertools.combinations(EXPOSURES, 2)))
    inh_mean = panel[panel["n"] > 0].groupby("iso3")["inh_share"].mean()
    for i in ("USA", "CHN", "DEU", "FRA", "ITA", "ESP", "GBR", "IND", "THA", "RUS", "SWE", "KOR", "JPN"):
        if i in expo.index:
            r = expo.loc[i]
            w(f"  {i}: stringency {r['stringency']:.1f}, deaths per million {r['deaths_pm']:.0f}, tourism {r['tourism_gdp']:.2f}% of GDP, "
              f"contact-intensive employment {r['contact_emp']:.1f}%; inherited share of billionaire wealth (mean of lists) {inh_mean.get(i, np.nan):.3f}")
    alt = expo.loc[expo.index.isin(bci)]
    w(f"alternative sources among the 63 billionaire countries: UN Tourism receipts for 2015-2019 available for "
      f"{int(alt['tourism_unwto'].notna().sum())} (missing: {', '.join(sorted(alt.index[alt['tourism_unwto'].isna()]))}); "
      f"survey-based contact-intensive employment for 2014-2019 available for {int(alt['contact_survey'].notna().sum())} "
      f"(missing: {', '.join(sorted(alt.index[alt['contact_survey'].isna()]))}); both available for "
      f"{int((alt['tourism_unwto'].notna() & alt['contact_survey'].notna()).sum())}")
    w("")
    w("## How exposure relates to billionaire composition (country level, billionaire countries; Spearman)")
    for _, r in corr.iterrows():
        w(f"{r['exposure']} vs {r['variable']}: {r['spearman']:+.3f} (n={r['n']})")
    w("")
    w("## Main fits: inherited-minus-self-made coefficient, sector controls as in wealth-origin-sectors; "
      "each exposure set without (base) and with exposure x list-year, on the same country-years")
    names = [k.replace(" ", "_") for k in EXPO_SETS if k != "none"]
    for _, r in fits.iterrows():
        w(f"{r['outcome']} h={r['horizon']} {r['sample']} {r['estimator']}: previous sample (n={r['n_prev']}, "
          f"countries={r['c_prev']}) {ci(r['b_prev'], r['se_prev'])}; " + "; ".join(
              f"{k} (countries={r['c_' + k]}) base {ci(r['b0_' + k], r['se0_' + k])} with {ci(r['b_' + k], r['se_' + k])}"
              for k in names))
    for o in ("tfp", "gdppc"):
        m = fits["outcome"] == o
        fe = m & (fits["estimator"] == "fe")
        for k in names:
            b, s_ = fits.loc[m, "b_" + k], fits.loc[m, "se_" + k]
            b0, s0 = fits.loc[m, "b0_" + k], fits.loc[m, "se0_" + k]
            w(f"  {o} / {k}: with exposure negative {int((b < 0).sum())} of {int(m.sum())}, sig- {int((b + 1.96 * s_ < 0).sum())} "
              f"(base sig- {int((b0 + 1.96 * s0 < 0).sum())}), sig+ {int((b - 1.96 * s_ > 0).sum())}; median {b.median():+.4f} "
              f"(base {b0.median():+.4f}); fixed-effects median ratio with/base "
              f"{(fits.loc[fe, 'b_' + k] / fits.loc[fe, 'b0_' + k]).median():.3f}")
    w("joint test for the coefficient with all four exposures (null imposed, countries resampled jointly):")
    for _, r in joint.iterrows():
        w(f"  {r['subset']}: {r['n_fits']} fits, median {r['median']:+.4f} (p={r['p_median']:.3f}); sig- {r['n_neg']} "
          f"(null 95th pct {r['null_n_neg_p95']:.0f}, p={r['p_n_neg']:.3f}); sig+ {r['n_pos']} (null 95th pct "
          f"{r['null_n_pos_p95']:.0f}, p={r['p_n_pos']:.3f}); Stouffer Z {r['stouffer']:+.2f} (p={r['p_stouffer']:.3f})")
    w("")
    w("## Exposure x list-year coefficients in the headline fit (TFP, fixed effects, billionaire countries; pp per SD, relative to 2011)")
    for (h, e), sub in paths.groupby(["horizon", "exposure"]):
        w(f"h={h} {e}: " + "; ".join(f"{r['year']} {r['b']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]" for _, r in sub.iterrows()))
    w("")
    w("## Cross-sections around 2020 (billionaire countries; HC1; TFP growth in pp a year; sector shares and income controlled)")
    for _, r in dr.iterrows():
        w(f"{r['window']} / exposure {r['exposure']} (n={r['n']}; mean growth {r['mean_y']:+.2f}, SD {r['sd_y']:.2f}): "
          f"inherited minus self-made {ci(r['b'], r['se'])}")
    w("")
    w("## Period split (TFP, fixed effects)")
    for _, r in split.iterrows():
        w(f"{r['sample']} h={r['horizon']} {r['windows']} (lists {r['list_years']}, n={r['n_obs']}) exposure {r['exposure']}: {ci(r['b'], r['se'])}")
    w("")
    w("## Leave one list year out (TFP, fixed effects)")
    for (smp, h, ex), sub in loyo.groupby(["sample", "horizon", "exposure"]):
        w(f"{smp} h={h} exposure {ex}: full {ci(sub['full_b'].iloc[0], sub['full_se'].iloc[0])}; " + "; ".join(
            f"without {r['dropped_year']} {r['b']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]" for _, r in sub.iterrows()))
    w("")
    w("## Forward minus backward with all four exposures (fixed effects, country bootstrap, 1,000 draws)")
    for _, r in pp.iterrows():
        w(f"{r['outcome']} / {r['sample']} / h={r['horizon']}: forward {r['forward']:+.3f}, backward {r['backward']:+.3f}, "
          f"difference {r['fwd_minus_bwd']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}], share of draws >= 0 {r['p_ge_0']:.3f}")
    w("")
    w("## How composition is measured (TFP, fixed effects, sector shares and all four exposures controlled)")
    for _, r in meas.iterrows():
        if r["horizon"] == 5 and r["without_2020_list"]:
            continue  # five-year windows from the 2020 list end in 2025, after the data
        w(f"{r['sample']} h={r['horizon']} {r['measure']}{' / without the 2020 list' if r['without_2020_list'] else ''} "
          f"(n={r['n_obs']}): {ci(r['b'], r['se'])}")
    w("all 24 specifications with the two-list average (sector shares controlled; base = no exposures, with = all four):")
    for _, r in afits.iterrows():
        w(f"  {r['outcome']} h={r['horizon']} {r['sample']} {r['estimator']} (n={r['n_obs']}, countries={r['n_countries']}): "
          f"base {ci(r['avg_base'], r['se_avg_base'])}; with {ci(r['avg'], r['se_avg'])}")
    for o in ("tfp", "gdppc"):
        m = afits["outcome"] == o
        b, se_ = afits.loc[m, "avg"], afits.loc[m, "se_avg"]
        w(f"  {o}: negative {int((b < 0).sum())} of {int(m.sum())}; sig- {int((b + 1.96 * se_ < 0).sum())}; sig+ "
          f"{int((b - 1.96 * se_ > 0).sum())}; median {b.median():+.4f}")
    w("  joint test, two-list average with all four exposures:")
    for _, r in ajoint.iterrows():
        w(f"    {r['subset']}: {r['n_fits']} fits, median {r['median']:+.4f} (p={r['p_median']:.3f}); sig- {r['n_neg']} "
          f"(null 95th pct {r['null_n_neg_p95']:.0f}, p={r['p_n_neg']:.3f}); sig+ {r['n_pos']} (p={r['p_n_pos']:.3f}); "
          f"Stouffer Z {r['stouffer']:+.2f} (p={r['p_stouffer']:.3f})")
    w("period split, by measure (billionaire countries):")
    for _, r in myy[myy["dropped_year"] == -1].iterrows():
        w(f"  {r['measure']} h={r['horizon']} {r['split']} (lists {r['list_years']}, n={int(r['n_obs'])}): {ci(r['b'], r['se'])}")
    w("leave one list year out, by measure (billionaire countries):")
    for (m, h), sub in myy[myy["dropped_year"] != -1].groupby(["measure", "horizon"], sort=False):
        lo, hi = sub.loc[sub["b"].idxmax()], sub.loc[sub["b"].idxmin()]
        w(f"  {m} h={h}: full {ci(sub['full_b'].iloc[0], sub['full_se'].iloc[0])}; closest to zero {lo['b']:+.3f} without {lo['dropped_year']} "
          f"[{lo['lo']:+.3f}, {lo['hi']:+.3f}]; furthest {hi['b']:+.3f} without {hi['dropped_year']}; intervals excluding zero "
          f"{int((sub['hi'] < 0).sum())} of {len(sub)}; " + "; ".join(f"{r['dropped_year']} {r['b']:+.3f}" for _, r in sub.iterrows()))
    w("forward minus backward, by measure (billionaire countries, country bootstrap):")
    for _, r in mpp.iterrows():
        w(f"  {r['measure']} h={r['horizon']}: forward {ci(r['forward'], r['se_forward'])}; backward {ci(r['backward'], r['se_backward'])}; "
          f"difference {r['fwd_minus_bwd']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}], share of draws >= 0 {r['p_ge_0']:.3f}")
    text = "\n".join(L) + "\n"
    (OUT / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
