"""Loaders and estimators for the UK innovation-ownership study.

The question: is UK innovation owned abroad more than the UK's size and language predict? The
loaders turn the six cached tables (see fetch_data.py) into tidy frames; the estimators are a
fractional logit fitted across countries, a hold-out prediction for one country, a jackknife+
interval for that prediction (Barber, Candes, Ramdas and Tibshirani 2021) and leave-one-out
prediction errors for every country, so the UK's gap can be ranked against the others'.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"

EU27 = [
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "IRL",
    "ITA", "LVA", "LTU", "LUX", "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE",
]  # fmt: skip
# Comparators outside the EU kept in the cross-country model. The US is left out: it is the
# largest foreign owner of everyone else's inventions, so its own foreign share answers a
# different question.
OTHERS = ["GBR", "CAN", "AUS", "JPN", "KOR", "CHE", "NOR", "ISR", "NZL"]
MODEL_COUNTRIES = EU27 + OTHERS
# Post-1989 transition economies, whose industry was largely privatised to foreign buyers.
CEE = ["BGR", "HRV", "CZE", "EST", "HUN", "LVA", "LTU", "POL", "ROU", "SVK", "SVN"]
# English as an official and main business language.
ENGLISH = ["GBR", "IRL", "MLT", "CAN", "AUS", "NZL"]
LARGE_EU = ["DEU", "FRA", "ITA", "ESP", "NLD", "SWE"]

ISO2_TO_ISO3 = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "HR": "HRV", "CY": "CYP", "CZ": "CZE", "DK": "DNK",
    "EE": "EST", "FI": "FIN", "FR": "FRA", "DE": "DEU", "EL": "GRC", "HU": "HUN", "IE": "IRL",
    "IT": "ITA", "LV": "LVA", "LT": "LTU", "LU": "LUX", "MT": "MLT", "NL": "NLD", "PL": "POL",
    "PT": "PRT", "RO": "ROU", "SK": "SVK", "SI": "SVN", "ES": "ESP", "SE": "SWE", "UK": "GBR",
    "NO": "NOR", "CH": "CHE", "IS": "ISL", "BA": "BIH", "RS": "SRB", "TR": "TUR", "MK": "MKD",
    "ME": "MNE", "AL": "ALB",
}  # fmt: skip

OFFICES = {"PCT": "9P50_1", "EPO": "6F0"}
COVARIATES = ["log_inventions", "english", "cee"]


# --------------------------------------------------------------------------- loaders


def load_patents(office="PCT", path=None):
    """Country-year counts: inventions (W), owned abroad (WXD), owned in the US (USA)."""
    d = pd.read_csv(path or CACHE / "patents_foreign.csv")
    d = d[d["PATENT_AUTHORITIES"] == OFFICES[office]]
    wide = d.pivot_table(index=["REF_AREA", "TIME_PERIOD"], columns="PARTNER_AREA", values="OBS_VALUE")
    wide = wide.rename(columns={"W": "inventions", "WXD": "foreign", "USA": "us"}).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"REF_AREA": "iso3", "TIME_PERIOD": "year"})
    return wide[["iso3", "year", "inventions", "foreign", "us"]]


def load_owners(path=None):
    """Owner-country counts for the six inventor countries in patents_owners.csv (PCT)."""
    d = pd.read_csv(path or CACHE / "patents_owners.csv")
    d = d.rename(columns={"REF_AREA": "iso3", "PARTNER_AREA": "owner", "TIME_PERIOD": "year", "OBS_VALUE": "n"})
    return d[["iso3", "owner", "year", "n"]]


def load_vc(path=None):
    """Venture capital investment, % of GDP, by stage: SEED, START, LATER and total (_T)."""
    d = pd.read_csv(path or CACHE / "venture_capital.csv")
    d = d[d["UNIT_MEASURE"] == "PT_B1GQ"]
    wide = d.pivot_table(index=["REF_AREA", "TIME_PERIOD"], columns="BUSINESS_DEVELOPMENT_STAGE", values="OBS_VALUE")
    wide = wide.rename(columns={"_T": "total", "SEED": "seed", "START": "start", "LATER": "later"}).reset_index()
    wide.columns.name = None
    return wide.rename(columns={"REF_AREA": "iso3", "TIME_PERIOD": "year"})


def load_high_growth(path=None):
    """Share (%) of employer enterprises with 10+ employees growing 10%+ a year, by industry."""
    d = pd.read_csv(path or CACHE / "high_growth.csv")
    d = d.rename(columns={"REF_AREA": "iso3", "ACTIVITY": "industry", "TIME_PERIOD": "year", "OBS_VALUE": "rate"})
    return d[["iso3", "industry", "year", "rate"]]


def load_pensions(path=None):
    """Pension investment as % of GDP (all vehicles) and private-equity share of pension-fund assets."""
    d = pd.read_csv(path or CACHE / "pensions.csv")
    d["MEASURE"] = d["MEASURE"].astype(str)
    assets = d[(d["MEASURE"] == "1000") & (d["UNIT_MEASURE"] == "PT_B1GQ") & (d["VEHICLE_TYPE"] == "_T")]
    pe = d[(d["MEASURE"] == "1286") & (d["UNIT_MEASURE"] == "PT_AS") & (d["VEHICLE_TYPE"] == "PF")]
    a = assets.rename(columns={"OBS_VALUE": "assets_gdp"})[["REF_AREA", "TIME_PERIOD", "assets_gdp"]]
    p = pe.rename(columns={"OBS_VALUE": "pe_share"})[["REF_AREA", "TIME_PERIOD", "pe_share"]]
    out = a.merge(p, on=["REF_AREA", "TIME_PERIOD"], how="outer")
    return out.rename(columns={"REF_AREA": "iso3", "TIME_PERIOD": "year"}).sort_values(["iso3", "year"])


def load_fats(path=None):
    """Persons employed in all, foreign-controlled and US-controlled enterprises, by industry."""
    d = pd.read_csv(path or CACHE / "fats_ict.csv")
    wide = d.pivot_table(index=["geo", "nace_r2", "TIME_PERIOD"], columns="c_ctrl", values="OBS_VALUE").reset_index()
    wide.columns.name = None
    wide = wide.rename(
        columns={"nace_r2": "industry", "TIME_PERIOD": "year", "WORLD": "all", "WRL_X_REP": "foreign", "US": "us"}
    )
    wide["iso3"] = wide["geo"].map(ISO2_TO_ISO3)
    return wide.dropna(subset=["iso3"])[["iso3", "industry", "year", "all", "foreign", "us"]]


# --------------------------------------------------------------------------- shares


def pooled_shares(panel, first, last, countries=None, min_inventions=0):
    """Pool counts over [first, last] per country and return shares owned abroad, by the US and by others.

    Pooling counts (not averaging yearly shares) weights each year by its number of inventions, so
    a thin early year cannot swing a small country's share.
    """
    p = panel[(panel["year"] >= first) & (panel["year"] <= last)]
    if countries is not None:
        p = p[p["iso3"].isin(countries)]
    g = p.groupby("iso3").agg(
        inventions=("inventions", "sum"), foreign=("foreign", "sum"), us=("us", "sum"), years=("year", "nunique")
    )
    g = g[g["inventions"] >= max(min_inventions, 1)].copy()
    g["foreign_share"] = g["foreign"] / g["inventions"]
    g["us_share"] = g["us"] / g["inventions"]
    g["other_share"] = (g["foreign"] - g["us"]) / g["inventions"]
    g["log_inventions"] = np.log(g["inventions"] / g["years"])
    return g.reset_index()


def add_covariates(shares, pensions=None, first=None, last=None):
    """English-language and transition-economy indicators, and log(1 + pension assets % GDP)."""
    out = shares.copy()
    out["english"] = out["iso3"].isin(ENGLISH).astype(float)
    out["cee"] = out["iso3"].isin(CEE).astype(float)
    if pensions is not None:
        p = pensions[(pensions["year"] >= first) & (pensions["year"] <= last)]
        mean_assets = p.groupby("iso3")["assets_gdp"].mean()
        out["pension_assets"] = out["iso3"].map(mean_assets)
        out["log_pensions"] = np.log1p(out["pension_assets"])
    return out


# --------------------------------------------------------------------------- estimators


def design(frame, covariates):
    """Constant plus covariates, always with the constant column."""
    return sm.add_constant(frame[list(covariates)].astype(float), has_constant="add")


def fit_fractional_logit(frame, outcome, covariates, groups=None):
    """Quasi-binomial GLM with a logit link (Papke and Wooldridge 1996) for a share in (0, 1).

    Standard errors are heteroskedasticity-robust (HC1) or, with groups, clustered.
    """
    y = frame[outcome].astype(float).to_numpy()
    if np.any((y < 0) | (y > 1)):
        raise ValueError(f"{outcome} must lie in [0, 1]")
    model = sm.GLM(y, design(frame, covariates), family=sm.families.Binomial())
    if groups is None:
        return model.fit(cov_type="HC1")
    return model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(frame[groups])[0]})


def predict(result, frame, covariates):
    """Predicted share for each row of frame."""
    return np.asarray(result.predict(design(frame, covariates)))


def _clip(p, eps=1e-6):
    return np.clip(p, eps, 1 - eps)


def holdout(frame, outcome, covariates, target):
    """Fit without the target country; return (actual, predicted) for the target."""
    train = frame[frame["iso3"] != target]
    test = frame[frame["iso3"] == target]
    if len(test) != 1:
        raise ValueError(f"{target} must appear exactly once")
    res = fit_fractional_logit(train, outcome, covariates)
    return float(test[outcome].iloc[0]), float(predict(res, test, covariates)[0])


def jackknife_plus(frame, outcome, covariates, target, alpha=0.1):
    """Jackknife+ prediction interval for the target country's share.

    For each training country i: fit without i (and without the target), take the absolute
    leave-one-out residual on the logit scale R_i, and the prediction for the target m_i. The
    interval runs from the floor(alpha (n+1))-th smallest m_i - R_i to the ceil((1-alpha)(n+1))-th
    smallest m_i + R_i, mapped back to shares. It covers with probability at least 1 - 2 alpha for
    exchangeable countries, whatever the model (Barber et al. 2021).
    """
    train = frame[frame["iso3"] != target].reset_index(drop=True)
    test = frame[frame["iso3"] == target]
    n = len(train)
    lower, upper = np.empty(n), np.empty(n)
    for i in range(n):
        res = fit_fractional_logit(train.drop(index=i), outcome, covariates)
        m_i = logit(_clip(predict(res, train.iloc[[i]], covariates)[0]))
        r_i = abs(logit(_clip(train[outcome].iloc[i])) - m_i)
        m_t = logit(_clip(predict(res, test, covariates)[0]))
        lower[i], upper[i] = m_t - r_i, m_t + r_i
    k_hi = int(np.ceil((1 - alpha) * (n + 1)))
    k_lo = int(np.floor(alpha * (n + 1)))
    lo = -np.inf if k_lo < 1 else np.sort(lower)[k_lo - 1]
    hi = np.inf if k_hi > n else np.sort(upper)[k_hi - 1]
    return float(expit(lo)), float(expit(hi))


def leave_one_out(frame, outcome, covariates):
    """Every country's actual share, its prediction from a fit without it, and the gap in points."""
    rows = []
    for iso in frame["iso3"]:
        actual, pred = holdout(frame, outcome, covariates, iso)
        rows.append({"iso3": iso, "actual": actual, "predicted": pred, "gap": actual - pred})
    out = pd.DataFrame(rows)
    out["logit_gap"] = logit(_clip(out["actual"])) - logit(_clip(out["predicted"]))
    out["rank"] = out["logit_gap"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("logit_gap", ascending=False).reset_index(drop=True)


def panel_frame(panel, pensions, first, last, countries, min_inventions=0):
    """Country-year rows with shares and covariates, for clustered-SE inference on coefficients."""
    p = panel[(panel["year"] >= first) & (panel["year"] <= last) & panel["iso3"].isin(countries)].copy()
    totals = p.groupby("iso3")["inventions"].sum()
    p = p[p["iso3"].map(totals) >= max(min_inventions, 1)]
    p = p[p["inventions"] > 0].copy()
    p["foreign_share"] = p["foreign"] / p["inventions"]
    p["us_share"] = p["us"] / p["inventions"]
    p["other_share"] = (p["foreign"] - p["us"]) / p["inventions"]
    p["log_inventions"] = np.log(p["iso3"].map(p.groupby("iso3")["inventions"].mean()))
    p["english"] = p["iso3"].isin(ENGLISH).astype(float)
    p["cee"] = p["iso3"].isin(CEE).astype(float)
    p = p.merge(pensions[["iso3", "year", "assets_gdp"]], on=["iso3", "year"], how="left")
    p["log_pensions"] = np.log1p(p["assets_gdp"])
    return p
