"""Part 3: billionaire density and poverty within countries, 2010-2024.

Every number quoted in the Part 3 article is computed here and written to
``results/part3/numbers.json``; ``--check`` recomputes everything and fails if
any number differs. Four figures go to ``figures/part3/``.

Inputs (all hash-checked against their manifests):

* Forbes annual lists 2010-2025, country totals by citizenship
  (``data/forbes_history_counts.csv.gz``, from ``fetch_forbes_history.py``);
* World Bank PIP national survey estimates 2010-2024 and headcounts at twenty
  lines (``data/pip_*.csv.gz``, from ``fetch_pip.py``);
* WDI GDP, GNI and household consumption per person (PPP, constant 2021
  international dollars), population and price levels (``data/wdi_*``);
* WID top wealth shares with quality grades (``data/wid_*``, Part 2).

Usage::

    python part3_within_country.py           # print, write results/part3 and figures/part3
    python part3_within_country.py --check   # fail if any number differs from numbers.json

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
import statsmodels.formula.api as smf  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

from part1_poverty_lines import (BASELINE, BLUE, INK, MUTED, ORANGE, SECONDARY, SURFACE,  # noqa: E402
                                 _compare, _frame, _style, fmt)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results" / "part3"
FIGURES = HERE / "figures" / "part3"
FIRST_YEAR, LAST_SURVEY, LAST_LIST = 2010, 2024, 2025
LINE = 30.0
FIRST_ARTICLE_COEF = -7.91  # first article: billionaire coefficient given GDP, 54 countries, $30
SOURCE = ("Data: Forbes annual billionaire lists 2010-2025 (citizenship); World Bank PIP national surveys 2010-2024 "
          "(2021 PPP); WDI GDP per person (PPP). Descriptive, not causal.")

FORBES_ISO3 = {
    "Albania": "ALB", "Algeria": "DZA", "Andorra": "AND", "Angola": "AGO", "Argentina": "ARG", "Armenia": "ARM",
    "Australia": "AUS", "Austria": "AUT", "Bahamas": "BHS", "Bahrain": "BHR", "Bangladesh": "BGD",
    "Barbados": "BRB", "Belgium": "BEL", "Belize": "BLZ", "Bermuda": "BMU", "Brazil": "BRA",
    "British Virgin Islands": "VGB", "Bulgaria": "BGR", "Cambodia": "KHM", "Canada": "CAN",
    "Cayman Islands": "CYM", "Chile": "CHL", "China": "CHN", "Colombia": "COL", "Croatia": "HRV",
    "Cyprus": "CYP", "Czech Republic": "CZE", "Czechia": "CZE", "Denmark": "DNK", "Dominican Republic": "DOM",
    "Egypt": "EGY", "Estonia": "EST", "Eswatini (Swaziland)": "SWZ", "Swaziland": "SWZ", "Finland": "FIN",
    "France": "FRA", "Georgia": "GEO", "Germany": "DEU", "Greece": "GRC", "Guatemala": "GTM", "Guernsey": "GGY",
    "Hong Kong": "HKG", "Hong Kong SAR": "HKG", "Hungary": "HUN", "Iceland": "ISL", "India": "IND",
    "Indonesia": "IDN", "Ireland": "IRL", "Israel": "ISR", "Italy": "ITA", "Japan": "JPN", "Kazakhstan": "KAZ",
    "Kuwait": "KWT", "Latvia": "LVA", "Lebanon": "LBN", "Liechtenstein": "LIE", "Lithuania": "LTU",
    "Luxembourg": "LUX", "Macao": "MAC", "Macau": "MAC", "Malaysia": "MYS", "Mexico": "MEX", "Monaco": "MCO",
    "Morocco": "MAR", "Nepal": "NPL", "Netherlands": "NLD", "New Zealand": "NZL", "Nigeria": "NGA",
    "Norway": "NOR", "Oman": "OMN", "Pakistan": "PAK", "Panama": "PAN", "Peru": "PER", "Philippines": "PHL",
    "Poland": "POL", "Portugal": "PRT", "Qatar": "QAT", "Romania": "ROU", "Russia": "RUS",
    "Saudi Arabia": "SAU", "Scotland": "GBR", "Seychelles": "SYC", "Singapore": "SGP", "Slovakia": "SVK",
    "South Africa": "ZAF", "South Korea": "KOR", "Spain": "ESP", "St. Kitts and Nevis": "KNA", "Sweden": "SWE",
    "Switzerland": "CHE", "Taiwan": "TWN", "Tanzania": "TZA", "Thailand": "THA",
    "Turks and Caicos Islands": "TCA", "Turkey": "TUR", "U.S. Virgin Islands": "VIR", "Uganda": "UGA",
    "Ukraine": "UKR", "United Arab Emirates": "ARE", "United Kingdom": "GBR", "United States": "USA",
    "Uruguay": "URY", "Uzbekistan": "UZB", "Venezuela": "VEN", "Vietnam": "VNM", "Zimbabwe": "ZWE",
}
WID_ISO3 = {
    "IN": "IND", "US": "USA", "NO": "NOR", "IS": "ISL", "NL": "NLD", "GB": "GBR", "LU": "LUX", "GR": "GRC",
    "MT": "MLT", "AT": "AUT", "PT": "PRT", "CH": "CHE", "BE": "BEL", "FI": "FIN", "CY": "CYP", "SK": "SVK",
    "SI": "SVN", "ZA": "ZAF", "UY": "URY", "ES": "ESP", "CN": "CHN", "RU": "RUS", "IE": "IRL", "EE": "EST",
    "ID": "IDN", "FR": "FRA", "LV": "LVA", "HU": "HUN", "IT": "ITA", "KR": "KOR", "SE": "SWE", "DK": "DNK",
    "PL": "POL", "CA": "CAN", "DE": "DEU", "HR": "HRV", "LT": "LTU", "MX": "MEX", "BR": "BRA", "JP": "JPN",
}
WDI_NAMES = {"NY.GDP.PCAP.PP.KD": "gdp", "NY.GNP.PCAP.PP.KD": "gni", "NE.CON.PRVT.PP.KD": "hhc",
             "SP.POP.TOTL": "pop", "PA.NUS.PPP": "ppp_factor", "PA.NUS.FCRF": "fx", "CM.MKT.LCAP.GD.ZS": "mcap"}
SHOWCASE = {"CHN": "China", "IND": "India", "RUS": "Russia", "BRA": "Brazil", "USA": "United States",
            "DEU": "Germany"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checked(manifest: str, files: dict[str, str]) -> dict[str, Any]:
    m = json.loads((DATA / manifest).read_text(encoding="utf-8"))
    for name, key in files.items():
        if _sha(DATA / name) != m[key]:
            raise ValueError(f"{name} does not match {manifest}")
    return m


def load_inputs() -> dict[str, Any]:
    forbes_m = _checked("forbes_history_manifest.json", {"forbes_history_counts.csv.gz": "counts_sha256"})
    wdi_m = _checked("wdi_manifest.json", {"wdi_series.csv.gz": "wdi_series_sha256"})
    pip_m = _checked("pip_manifest.json", {"pip_surveys.csv.gz": "pip_surveys_sha256",
                                           "pip_headcounts.csv.gz": "pip_headcounts_sha256"})
    wid_m = _checked("wid_manifest.json", {"wid_series.csv.gz": "wid_series_sha256"})
    return {
        "forbes": pd.read_csv(DATA / "forbes_history_counts.csv.gz", keep_default_na=False),
        "wdi": pd.read_csv(DATA / "wdi_series.csv.gz"),
        "surveys": pd.read_csv(DATA / "pip_surveys.csv.gz"),
        "heads": pd.read_csv(DATA / "pip_headcounts.csv.gz"),
        "wid": pd.read_csv(DATA / "wid_series.csv.gz"),
        "manifests": {"forbes": forbes_m, "wdi": wdi_m, "pip": pip_m, "wid": wid_m},
    }


def forbes_counts(forbes: pd.DataFrame) -> pd.DataFrame:
    """Citizenship counts and net worth by ISO3 and year, 2010-2025."""
    f = forbes[(forbes.year >= FIRST_YEAR) & (forbes.year <= LAST_LIST)].copy()
    unmapped = sorted(set(f.country_label) - set(FORBES_ISO3))
    if unmapped:
        raise ValueError(f"unmapped Forbes labels: {unmapped}")
    f["iso3"] = f.country_label.map(FORBES_ISO3)
    cit = f[f.field == "citizenship"].groupby(["iso3", "year"])[["n", "worth_musd"]].sum()
    return cit


def wdi_wide(wdi: pd.DataFrame) -> pd.DataFrame:
    w = wdi.pivot_table(index=["iso3", "year"], columns="indicator", values="value").rename(columns=WDI_NAMES)
    w["hhc_pc"] = w["hhc"] / w["pop"]
    w["plr"] = w["ppp_factor"] / w["fx"]
    return w


def density(n: Any, pop: Any) -> Any:
    """log10 billionaires per ten million people, with half a billionaire added so zero counts are defined."""
    return np.log10((np.asarray(n, float) + 0.5) / np.asarray(pop, float) * 1e7)


def build_panel(inp: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """National PIP surveys 2010-2024 in countries that had a billionaire in any list 2010-2025."""
    cit = forbes_counts(inp["forbes"])
    w = wdi_wide(inp["wdi"])
    s, h = inp["surveys"], inp["heads"]
    key = ["country_code", "reporting_year", "welfare_type", "reporting_level"]
    nat = s[(s.reporting_level == "national") & s.reporting_year.between(FIRST_YEAR, LAST_SURVEY)].copy()
    wide = h[(h.reporting_level == "national")].pivot_table(index=key, columns="poverty_line",
                                                           values="headcount").mul(100)
    wide.columns = [f"p_{c:g}" for c in wide.columns]
    nat = nat.merge(wide.reset_index(), on=key, how="left")
    ever = cit.groupby(level="iso3").n.max()
    nat = nat[nat.country_code.isin(ever[ever > 0].index)].copy()
    nat["year"] = nat.reporting_year.astype(int)
    nat = nat.join(w, on=["country_code", "year"])
    nat["n_cit"] = [float(cit.n.get((c, y), 0)) for c, y in zip(nat.country_code, nat.year)]
    nat["B"] = density(nat.n_cit, nat["pop"])
    nat["G"] = np.log10(nat.gdp)
    nat["GNI"] = np.log10(nat.gni)
    nat["HHC"] = np.log10(nat.hhc_pc)
    nat["M"] = np.log10(nat["mean"])
    nat["PLR"] = np.log10(nat.plr)
    nat["spell"] = nat.country_code + "|" + nat.welfare_type + "|" + nat.survey_comparability.astype(str)
    panel = nat.dropna(subset=["G", "B", "p_30", "M", "pop"]).sort_values(["spell", "year"]).reset_index(drop=True)
    dropped = sorted(set(nat.country_code) - set(panel.country_code))
    panel.attrs["dropped_no_gdp"] = dropped
    return panel, cit.reset_index(), w


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def cl(formula: str, data: pd.DataFrame, group: str = "country_code") -> Any:
    return smf.ols(formula, data).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(data[group])[0]})


def coef(fit: Any, name: str) -> dict[str, Any]:
    lo, hi = fit.conf_int().loc[name]
    return {"coef": float(fit.params[name]), "ci95": [float(lo), float(hi)], "p": float(fit.pvalues[name]),
            "n": int(fit.nobs)}


def describe(panel: pd.DataFrame, cit: pd.DataFrame, w: pd.DataFrame, manifests: dict[str, Any]) -> dict[str, Any]:
    fm = manifests["forbes"]["years"]
    totals = {int(y): int(e["n_persons"]) for y, e in fm.items() if e.get("n_persons") and int(y) >= FIRST_YEAR}
    dm = panel.groupby("spell")[["B", "G", "p_30", "M"]].transform("mean")
    spells = panel.groupby("spell").year.agg(["count", "min", "max"])
    show = {}
    for iso, name in SHOWCASE.items():
        c = cit[cit.iso3 == iso].set_index("year").n
        show[iso] = {"name": name, "n": {str(int(y)): int(v) for y, v in c.items()},
                     "max_year": int(c.idxmax()), "max": int(c.max())}
    return {
        "forbes_totals": {str(k): v for k, v in totals.items()},
        "forbes_years_partial": {str(int(y)): int(e["n_persons"]) for y, e in fm.items()
                                 if e.get("n_persons") and int(y) < FIRST_YEAR and int(y) >= 2005},
        "forbes_overlap_with_latest": {str(int(y)): float(e["overlap_with_latest"]) for y, e in fm.items()
                                       if e.get("overlap_with_latest") is not None and int(y) >= FIRST_YEAR},
        "latest_list_fetched": int(manifests["forbes"].get("latest_year")),
        "n_obs": int(len(panel)), "n_countries": int(panel.country_code.nunique()),
        "n_spells": int(panel.spell.nunique()),
        "n_spells_2plus": int((spells["count"] >= 2).sum()),
        "dropped_no_gdp": panel.attrs.get("dropped_no_gdp", []),
        "years": [int(panel.year.min()), int(panel.year.max())],
        "obs_by_year": {str(int(k)): int(v) for k, v in panel.groupby("year").size().items()},
        "n_welfare": {k: int(v) for k, v in panel.groupby("welfare_type").size().items()},
        "n_region": {k: int(v) for k, v in panel.groupby("region_code").size().items()},
        "countries_region": {k: int(v) for k, v in panel.groupby("region_code").country_code.nunique().items()},
        "surveys_per_country": {"median": float(panel.groupby("country_code").size().median()),
                                "max": int(panel.groupby("country_code").size().max())},
        "zero_billionaire_obs": int((panel.n_cit == 0).sum()),
        "sd_total": {v: float(panel[v].std()) for v in ("B", "G", "p_30", "M")},
        "sd_within": {v: float((panel[v] - dm[v]).std()) for v in ("B", "G", "p_30", "M")},
        "corr_within_B_G": float(np.corrcoef(panel.B - dm.B, panel.G - dm.G)[0, 1]),
        "showcase": show,
    }


def annual_dynamics(cit: pd.DataFrame, w: pd.DataFrame) -> dict[str, Any]:
    """What moves billionaire density within countries, on the annual lists."""
    ever = cit.groupby("iso3").n.max()
    iso = sorted(ever[ever > 0].index)
    full = pd.MultiIndex.from_product([iso, range(FIRST_YEAR, LAST_LIST + 1)], names=["iso3", "year"])
    a = cit.set_index(["iso3", "year"])[["n"]].reindex(full, fill_value=0).reset_index().join(w, on=["iso3", "year"])
    a["B"], a["G"], a["PLR"] = density(a.n, a["pop"]), np.log10(a.gdp), np.log10(a.plr)
    a = a.sort_values(["iso3", "year"])
    for v in ("B", "G", "PLR"):
        a["d" + v] = a.groupby("iso3")[v].diff()
        a["d" + v + "_l1"] = a.groupby("iso3")["d" + v].shift(1)
        a["d" + v + "_f1"] = a.groupby("iso3")["d" + v].shift(-1)
    x = a.dropna(subset=["B", "G"])
    lev = cl("B ~ G + C(iso3) + C(year)", x, "iso3")
    lev_plr = cl("B ~ G + PLR + C(iso3) + C(year)", x.dropna(subset=["PLR"]), "iso3")
    t = x.dropna(subset=["dB", "dG", "dG_l1", "dG_f1"])
    timing = cl("dB ~ dG + dG_l1 + dG_f1 + C(year)", t, "iso3")
    g = x.dropna(subset=["dG", "dB_l1", "dG_l1"])
    gr = cl("dG ~ dB_l1 + dG_l1 + C(year)", g, "iso3")
    ar = cl("dB ~ dB_l1 + dG_l1 + C(year)", g.dropna(subset=["dB"]), "iso3")
    v = x.dropna(subset=["dB", "dG", "dPLR"])
    r_year = smf.ols("dB ~ C(year)", v).fit().rsquared
    r_all = smf.ols("dB ~ C(year) + dG + dPLR", v).fit().rsquared
    sd_dB = float(x.dB.std())
    return {
        "n": int(len(x)), "n_countries": int(x.iso3.nunique()),
        "levels_G": coef(lev, "G"), "levels_PLR": coef(lev_plr, "PLR"),
        "dB_on_dG": coef(timing, "dG"), "dB_on_dG_lag": coef(timing, "dG_l1"), "dB_on_dG_lead": coef(timing, "dG_f1"),
        "dG_on_dB_lag": coef(gr, "dB_l1"), "dB_on_dB_lag": coef(ar, "dB_l1"),
        "r2_year": float(r_year), "r2_year_growth_price": float(r_all), "sd_dB": sd_dB,
        "share_years_falling": float((x.dB < 0).mean()),
    }


def estimators(panel: pd.DataFrame) -> dict[str, Any]:
    """The billionaire coefficient on poverty at $30 under each design."""
    d = panel.copy()
    out: dict[str, Any] = {}
    out["pooled_raw"] = coef(cl("p_30 ~ B", d), "B")
    out["pooled_gdp"] = coef(cl("p_30 ~ B + G", d), "B")
    out["pooled_gdp_year"] = coef(cl("p_30 ~ B + G + C(year)", d), "B")
    for v in ("B", "G"):
        d[v + "_m"] = d.groupby("spell")[v].transform("mean")
    mund = cl("p_30 ~ B + G + B_m + G_m + C(year)", d)
    between = mund.params["B"] + mund.params["B_m"]
    cov = mund.cov_params()
    se_b = float(np.sqrt(cov.loc["B", "B"] + cov.loc["B_m", "B_m"] + 2 * cov.loc["B", "B_m"]))
    out["between"] = {"coef": float(between), "ci95": [float(between - 1.96 * se_b), float(between + 1.96 * se_b)],
                      "p_difference": float(mund.pvalues["B_m"])}
    out["within"] = coef(cl("p_30 ~ B + G + C(spell) + C(year)", d), "B")
    out["within_gdp_coef"] = coef(cl("p_30 ~ B + G + C(spell) + C(year)", d), "G")
    out["within_no_gdp"] = coef(cl("p_30 ~ B + C(spell) + C(year)", d), "B")
    for key, ctrl in (("within_gni", "GNI"), ("within_hhc", "HHC"), ("within_plr", "G + PLR")):
        sub = d.dropna(subset=[c.strip() for c in ctrl.split("+")])
        fit = cl(f"p_30 ~ B + {ctrl} + C(spell) + C(year)", sub)
        out[key] = coef(fit, "B") | {"n_countries": int(sub.country_code.nunique())}
    out["within_mean"] = coef(cl("p_30 ~ B + M + G + C(spell) + C(year)", d), "B")
    for wt in ("income", "consumption"):
        sub = d[d.welfare_type == wt]
        out[f"within_{wt}"] = coef(cl("p_30 ~ B + G + C(spell) + C(year)", sub), "B") | {
            "n_countries": int(sub.country_code.nunique())}
    out_eca = d[d.region_code != "ECS"]
    out["within_outside_eca"] = coef(cl("p_30 ~ B + G + C(spell) + C(year)", out_eca), "B") | {
        "n_countries": int(out_eca.country_code.nunique())}
    wts = 1.0 / d.groupby("country_code").year.transform("size")
    wls = smf.wls("p_30 ~ B + G + C(spell) + C(year)", d, weights=wts).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.country_code)[0]})
    out["within_country_weighted"] = coef(wls, "B")
    drops = {}
    for c in sorted(d.country_code.unique()):
        drops[c] = float(cl("p_30 ~ B + G + C(spell) + C(year)", d[d.country_code != c]).params["B"])
    lo_c, hi_c = min(drops, key=drops.get), max(drops, key=drops.get)
    out["drop_one"] = {"min": drops[lo_c], "min_country": lo_c, "max": drops[hi_c], "max_country": hi_c}
    pos = d[d.n_cit > 0]
    out["within_positive_only"] = coef(cl("p_30 ~ B + G + C(spell) + C(year)", pos), "B") | {
        "n_countries": int(pos.country_code.nunique())}
    # Own trends: remove each spell's linear trend (spells with three or more surveys), then OLS.
    cnt = d.groupby("spell").year.transform("count")
    t3 = d[cnt >= 3].copy()
    t3["t"] = t3.year - 2017
    for v in ("p_30", "B", "G"):
        t3[v + "_r"] = t3.groupby("spell")[[v, "t"]].apply(
            lambda g: g[v] - np.polyval(np.polyfit(g["t"], g[v], 1), g["t"])).reset_index(level=0, drop=True)
    tr = smf.ols("p_30_r ~ B_r + G_r - 1", t3).fit(cov_type="cluster",
                                                   cov_kwds={"groups": pd.factorize(t3.country_code)[0]})
    out["own_trends"] = coef(tr, "B_r") | {"n_spells": int(t3.spell.nunique()),
                                            "n_countries": int(t3.country_code.nunique())}
    # Fractional logit with spell and year dummies.
    X = pd.get_dummies(d[["spell", "year"]].astype(str), drop_first=True).astype(float)
    X["B"], X["G"] = d.B, d.G
    X = sm.add_constant(X)
    glm = sm.GLM(d.p_30 / 100, X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.country_code)[0]})
    p = glm.predict(X)
    scale = float((p * (1 - p)).mean() * 100)
    lo, hi = glm.conf_int().loc["B"]
    out["within_fractional"] = {"ame": float(scale * glm.params["B"]), "ci95": [scale * float(lo), scale * float(hi)],
                                "p": float(glm.pvalues["B"])}
    return out


def long_differences(panel: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    for sp, g in panel.groupby("spell"):
        if len(g) >= 2 and g.year.iloc[-1] - g.year.iloc[0] >= 5:
            a, b = g.iloc[0], g.iloc[-1]
            rows.append({"spell": sp, "iso3": a.country_code, "y0": int(a.year), "y1": int(b.year),
                         "dp": b.p_30 - a.p_30, "dB": b.B - a.B, "dG": b.G - a.G, "dM": b.M - a.M,
                         "dgini": b.gini - a.gini, "dp8": b["p_8.3"] - a["p_8.3"]})
    ld = pd.DataFrame(rows)
    out: dict[str, Any] = {"n": int(len(ld)), "n_countries": int(ld.iso3.nunique()),
                           "median_gap": float((ld.y1 - ld.y0).median()),
                           "corr_dB_dG": float(ld[["dB", "dG"]].corr().iloc[0, 1]),
                           "share_dB_positive": float((ld.dB > 0).mean())}
    fit = lambda f: cl(f, ld, "iso3")  # noqa: E731
    out["dp_raw"] = coef(fit("dp ~ dB"), "dB")
    out["dp"] = coef(fit("dp ~ dB + dG"), "dB")
    out["dp_dG"] = coef(fit("dp ~ dB + dG"), "dG")
    out["dM"] = coef(fit("dM ~ dB + dG"), "dB")
    out["dM_dG"] = coef(fit("dM ~ dB + dG"), "dG")
    out["dgini"] = coef(fit("dgini ~ dB + dG"), "dB")
    out["dp8"] = coef(fit("dp8 ~ dB + dG"), "dB")
    resid = smf.ols("dp ~ dG", ld).fit().resid
    ld = ld.assign(dp_resid=resid)
    return out, ld


def level_or_spread(panel: pd.DataFrame) -> dict[str, Any]:
    d = panel.copy()
    out = {"pooled_mean": coef(cl("M ~ B + G + C(year)", d), "B"),
           "within_mean": coef(cl("M ~ B + G + C(spell) + C(year)", d), "B"),
           "within_mean_no_gdp": coef(cl("M ~ B + C(spell) + C(year)", d), "B"),
           "within_gini": coef(cl("gini ~ B + G + C(spell) + C(year)", d), "B"),
           "pooled_gini": coef(cl("gini ~ B + G + C(year)", d), "B")}
    for v in ("B", "G"):
        d[v + "_m"] = d.groupby("spell")[v].transform("mean")
    m = cl("M ~ B + G + B_m + G_m + C(year)", d)
    out["mundlak_mean"] = {"within": float(m.params["B"]), "between": float(m.params["B"] + m.params["B_m"]),
                           "p_difference": float(m.pvalues["B_m"])}
    return out


def by_line(panel: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for col in sorted([c for c in panel.columns if c.startswith("p_")], key=lambda c: float(c[2:])):
        fit = cl(f"Q('{col}') ~ B + G + C(spell) + C(year)", panel)
        rows.append({"line": float(col[2:]), **coef(fit, "B")})
    return rows


def timing(panel: pd.DataFrame, cit: pd.DataFrame, w: pd.DataFrame) -> dict[str, Any]:
    """Leads and lags in levels with fixed effects, and the same question in changes."""
    n_at = cit.set_index(["iso3", "year"]).n
    d = panel.copy()

    def b_at(k: int) -> np.ndarray:
        yrs = d.year + k
        n = [float(n_at.get((c, y), 0)) if FIRST_YEAR <= y <= LAST_LIST else np.nan for c, y in zip(d.country_code, yrs)]
        pop = [w["pop"].get((c, y), np.nan) for c, y in zip(d.country_code, yrs)]
        return density(n, pop)

    def g_at(k: int) -> np.ndarray:
        return np.log10([w["gdp"].get((c, y), np.nan) for c, y in zip(d.country_code, d.year + k)])

    for k in range(-3, 4):
        d[f"B{k + 3}"], d[f"G{k + 3}"] = b_at(k), g_at(k)
    cs = d.dropna(subset=[f"B{i}" for i in range(7)] + [f"G{i}" for i in range(7)])
    levels = []
    for k in range(-3, 4):
        levels.append({"k": k, **coef(cl(f"p_30 ~ B{k + 3} + G + C(spell) + C(year)", cs), f"B{k + 3}"),
                       "same_year_gdp": coef(cl(f"p_30 ~ B{k + 3} + G{k + 3} + C(spell) + C(year)", cs), f"B{k + 3}")})
    cs2 = cs.assign(dB_around=cs.B6 - cs.B0)
    around = coef(cl("p_30 ~ dB_around + G + C(spell) + C(year)", cs2), "dB_around")
    # Changes between consecutive surveys in the same spell.
    def b1(c: str, y: int) -> float:
        if not FIRST_YEAR <= y <= LAST_LIST:
            return np.nan
        return float(density(n_at.get((c, y), 0), w["pop"].get((c, y), np.nan)))

    def g1(c: str, y: int) -> float:
        return float(np.log10(w["gdp"].get((c, y), np.nan)))

    rows = []
    for sp, g in panel.groupby("spell"):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            a, b = g.iloc[i - 1], g.iloc[i]
            c, t0, t1 = a.country_code, int(a.year), int(b.year)
            L = t1 - t0
            rows.append({"iso3": c, "t1": t1, "gap": L, "dp": b.p_30 - a.p_30,
                         "dB": b1(c, t1) - b1(c, t0), "dG": g1(c, t1) - g1(c, t0),
                         "dB_before": b1(c, t0) - b1(c, t0 - L), "dB_after": b1(c, t1 + L) - b1(c, t1)})
    fd = pd.DataFrame(rows)
    f = lambda form, cols: cl(form, fd.dropna(subset=cols), "iso3")  # noqa: E731
    changes = {
        "n": int(len(fd)), "n_countries": int(fd.iso3.nunique()), "median_gap": float(fd.gap.median()),
        "same": coef(f("dp ~ dB + dG + C(t1)", ["dB", "dG"]), "dB"),
        "before": coef(f("dp ~ dB_before + dG + C(t1)", ["dB_before", "dG"]), "dB_before"),
        "after": coef(f("dB_after ~ dp + dG + C(t1)", ["dB_after", "dp", "dG"]), "dp"),
    }
    return {"common_n": int(len(cs)), "common_countries": int(cs.country_code.nunique()),
            "common_years": [int(cs.year.min()), int(cs.year.max())], "levels": levels, "around": around,
            "changes": changes}


def loco(panel: pd.DataFrame) -> dict[str, Any]:
    """Leave-one-country-out error of within-spell deviations, with and without billionaire density."""
    years = sorted(panel.year.unique())

    def rmse(cols: list[str]) -> float:
        errs: list[float] = []
        formula = "p_30 ~ " + " + ".join(cols) + " + C(spell) + C(year)"
        for c in panel.country_code.unique():
            tr, te = panel[panel.country_code != c], panel[panel.country_code == c]
            m = smf.ols(formula, tr).fit()
            X = te[cols] - te.groupby("spell")[cols].transform("mean")
            y = te.p_30 - te.groupby("spell").p_30.transform("mean")
            ye = np.array([m.params.get(f"C(year)[T.{yy}]", 0.0) for yy in te.year])
            ye = ye - pd.Series(ye, index=te.index).groupby(te.spell).transform("mean").to_numpy()
            pred = X.to_numpy() @ np.array([m.params[k] for k in cols]) + ye
            errs += list(y.to_numpy() - pred)
        return float(np.sqrt(np.mean(np.square(errs))))

    del years
    return {"gdp": rmse(["G"]), "gdp_b": rmse(["G", "B"])}


def wid_observed(inp: dict[str, Any], cit: pd.DataFrame, w: pd.DataFrame) -> dict[str, Any]:
    s = inp["wid"]
    n_at = cit.set_index(["iso3", "year"]).n

    def share(p: str) -> pd.Series:
        x = s[(s.variable == "shweal992j") & (s.percentile == p) & (s.year >= FIRST_YEAR) & (s.data_quality > 0)
              & s.country.isin(WID_ISO3)]
        return x.assign(iso3=x.country.map(WID_ISO3)).set_index(["iso3", "year"]).value * 100

    df = pd.DataFrame({"t1": share("p99p100"), "t10": share("p90p100"), "t001": share("p99.99p100")}).reset_index()
    df["n"] = [float(n_at.get((c, y), 0)) for c, y in zip(df.iso3, df.year)]
    df = df.join(w[["pop", "gdp"]], on=["iso3", "year"])
    df["B"], df["G"] = density(df.n, df["pop"]), np.log10(df.gdp)
    df = df.dropna(subset=["B", "G", "t1"])
    out: dict[str, Any] = {"n": int(len(df)), "n_countries": int(df.iso3.nunique()),
                           "years": [int(df.year.min()), int(df.year.max())]}
    for y in ("t1", "t10", "t001"):
        out[y] = {"across": coef(cl(f"{y} ~ B + G + C(year)", df, "iso3"), "B"),
                  "within": coef(cl(f"{y} ~ B + G + C(iso3) + C(year)", df, "iso3"), "B")}
    return out


def compute(inp: dict[str, Any]) -> dict[str, Any]:
    panel, cit, w = build_panel(inp)
    ld, _ = long_differences(panel)
    est = estimators(panel)
    return {
        "describe": describe(panel, cit, w, inp["manifests"]),
        "annual": annual_dynamics(cit, w),
        "estimators": est,
        "long_differences": ld,
        "level_or_spread": level_or_spread(panel),
        "by_line": by_line(panel),
        "timing": timing(panel, cit, w),
        "loco": loco(panel),
        "wid_observed": wid_observed(inp, cit, w),
        "first_article_coef": FIRST_ARTICLE_COEF,
        "per_doubling": {k: {"coef": v["coef"] * np.log10(2), "ci95": [c * np.log10(2) for c in v["ci95"]]}
                         for k, v in (("within", est["within"]), ("pooled_gdp_year", est["pooled_gdp_year"]))}
        | {"first_article": FIRST_ARTICLE_COEF * np.log10(2)},
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def frame(fig: plt.Figure, title: str, subtitle: str, **kw: Any) -> None:
    _frame(fig, title, subtitle, source=SOURCE, **kw)


def fig_trajectories(cit: pd.DataFrame, w: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharex=True, sharey=True)
    years = np.arange(FIRST_YEAR, LAST_LIST + 1)
    for ax, (iso, name) in zip(axes.flat, SHOWCASE.items()):
        n = cit[cit.iso3 == iso].set_index("year").n.reindex(years, fill_value=0)
        pop = np.array([w["pop"].get((iso, y), np.nan) for y in years])
        gdp = np.array([w["gdp"].get((iso, y), np.nan) for y in years])
        dens = (n.to_numpy() + 0.5) / pop * 1e7
        ax.plot(years, dens / dens[0], color=BLUE, lw=2.2, label="Billionaires per person")
        ok = ~np.isnan(gdp)
        ax.plot(years[ok], gdp[ok] / gdp[0], color=ORANGE, lw=2.2, label="GDP per person")
        ax.axhline(1, color=BASELINE, lw=1)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4, 8]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
        ax.set_ylim(0.4, 11)
        ax.set_title(f"{name}: {int(n.iloc[0])} → {int(n.iloc[-1])} billionaires", fontsize=10.5, color=INK, loc="left")
        ax.xaxis.set_major_locator(FixedLocator([2010, 2015, 2020, 2025]))
    axes[0, 0].legend(loc="upper left", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Relative to 2010 (log scale)")
    frame(fig, "Within a country, billionaire density swings far more than GDP",
          "Billionaires per person and GDP per person, each relative to its 2010 value, Forbes lists 2010-2025.",
          top=0.83, bottom=0.12, left=0.09)
    fig.subplots_adjust(hspace=0.32, wspace=0.08)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_estimators(n: dict[str, Any], out: Path) -> None:
    e, ld = n["estimators"], n["long_differences"]
    rows = [
        ("Pooled, given GDP and year", e["pooled_gdp_year"], BLUE),
        ("Between countries (spell means)", e["between"], BLUE),
        ("Within: spell and year effects", e["within"], ORANGE),
        ("Within, each spell's own trend removed", e["own_trends"], ORANGE),
        ("Within, survey mean also held fixed", e["within_mean"], ORANGE),
        ("Within, GNI instead of GDP", e["within_gni"], ORANGE),
        ("Within, household consumption instead of GDP", e["within_hhc"], ORANGE),
        ("Long differences, about ten years", ld["dp"], ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.9))
    for i, (label, r, colour) in enumerate(rows):
        ax.plot(r["ci95"], [i, i], color=colour, lw=2.4, solid_capstyle="round", zorder=2)
        ax.scatter([r["coef"]], [i], s=64, color=colour, edgecolors=SURFACE, linewidths=1.4, zorder=3)
        ax.annotate(fmt(r["coef"], 1, sign=True), (r["ci95"][1], i), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=9, color=SECONDARY)
    ax.axvline(0, color=BASELINE, lw=1.2, zorder=1)
    ax.axvline(n["first_article_coef"], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate("first article, 54 countries", (n["first_article_coef"], len(rows) - 0.45), xytext=(-4, 0),
                textcoords="offset points", ha="right", fontsize=9, color=MUTED)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.3, -0.7)
    ax.set_xlim(-25, 16)
    ax.set_xlabel("Points of poverty at $30 a day per tenfold increase in billionaire density")
    ax.grid(axis="y", visible=False)
    frame(fig, "Between countries the estimate is negative; within countries it is near zero",
          "Billionaire coefficient given GDP per person (or the stated alternative), 95% intervals clustered by country.\n"
          "Blue: comparisons between countries. Orange: changes within a country's comparable surveys.",
          top=0.84, bottom=0.15, left=0.37)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_long_differences(ld: pd.DataFrame, n: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(ld.dB, ld.dp_resid, s=46, color=BLUE, edgecolors=SURFACE, linewidths=1, zorder=3)
    xs = np.linspace(ld.dB.min(), ld.dB.max(), 50)
    fit = smf.ols("dp_resid ~ dB", ld).fit()
    ax.plot(xs, fit.params["Intercept"] + fit.params["dB"] * xs, color=INK, lw=1.6, zorder=2)
    ax.axhline(0, color=BASELINE, lw=1.1)
    ax.axvline(0, color=BASELINE, lw=1.1)
    label = {"ROU|income|3": ("Romania", (7, -3)), "EST|income|3": ("Estonia", (7, -3)),
             "POL|income|3": ("Poland", (7, -3)), "VNM|consumption|2": ("Vietnam", (-7, 6)),
             "CHN|consumption|2": ("China", (7, -9)), "GEO|consumption|2": ("Georgia", (-7, 4)),
             "PAK|consumption|1": ("Pakistan", (7, 3)), "USA|income|1": ("United States", (7, -9)),
             "BRA|income|1": ("Brazil", (7, -5)), "GRC|income|2": ("Greece", (-7, 6))}
    for _, r in ld.iterrows():
        if r.spell in label:
            name, off = label[r.spell]
            ax.annotate(name, (r.dB, r.dp_resid), xytext=off, textcoords="offset points", fontsize=9,
                        color=SECONDARY, ha="right" if off[0] < 0 else "left", va="center")
    ax.xaxis.set_major_locator(FixedLocator([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v, 2, sign=True) if v else "0"))
    ax.set_xlabel("Change in log₁₀ billionaires per ten million people")
    ax.set_ylabel("Change in poverty at $30 not explained by GDP (points)")
    r = n["long_differences"]
    frame(fig, "Over a decade, changes in billionaire density did not track poverty",
          f"{r['n']} comparable survey spells in {r['n_countries']} countries, first and last survey at least five years apart "
          f"(median {r['median_gap']:.0f}).\nSlope {fmt(r['dp']['coef'], 1, sign=True)} points per tenfold increase "
          f"(95% interval {fmt(r['dp']['ci95'][0], 1)} to {fmt(r['dp']['ci95'][1], 1)}).",
          top=0.84, bottom=0.14, left=0.1)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_timing(n: dict[str, Any], out: Path) -> None:
    t = n["timing"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ks = [r["k"] for r in t["levels"]]
    for r in t["levels"]:
        ax.plot([r["k"], r["k"]], r["ci95"], color=BLUE, lw=2.2, solid_capstyle="round")
        ax.scatter([r["k"]], [r["coef"]], s=56, color=BLUE, edgecolors=SURFACE, linewidths=1.2, zorder=3)
    ax.axhline(0, color=BASELINE, lw=1.2)
    ax.set_xticks(ks)
    ax.set_xticklabels([fmt(k, 0, sign=True) if k else "0" for k in ks])
    ax.set_xlabel("Year of the Forbes list relative to the survey year")
    ax.set_ylabel("Points of poverty at $30 per tenfold density")
    ax.set_title("Levels, with spell and year effects", fontsize=10.5, color=INK, loc="left")
    ax.grid(axis="x", visible=False)
    c = t["changes"]
    rows = [("Density change before the\nsurvey interval", c["before"], "dp per tenfold"),
            ("Density change over the\nsame interval", c["same"], "dp per tenfold")]
    for i, (lab, r, _) in enumerate(rows):
        ax2.plot(r["ci95"], [i, i], color=ORANGE, lw=2.4, solid_capstyle="round")
        ax2.scatter([r["coef"]], [i], s=60, color=ORANGE, edgecolors=SURFACE, linewidths=1.2, zorder=3)
    ax2.axvline(0, color=BASELINE, lw=1.2)
    ax2.set_yticks(range(len(rows)))
    ax2.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax2.set_ylim(len(rows) - 0.4, -0.6)
    ax2.set_xlim(-12, 12)
    ax2.set_xlabel("Change in poverty per tenfold change")
    ax2.set_title(f"Changes between consecutive surveys ({c['n']})", fontsize=10.5, color=INK, loc="left")
    ax2.grid(axis="y", visible=False)
    frame(fig, "A timing pattern in levels that disappears in changes",
          "Left: poverty at $30 regressed on billionaire density k years from the survey, given GDP, spell and year effects.\n"
          "Right: the change in poverty between consecutive surveys against the change in density. 95% intervals.",
          top=0.83, bottom=0.17, left=0.08)
    fig.subplots_adjust(wspace=0.55)
    fig.savefig(out, dpi=250)
    plt.close(fig)


FIGURE_FILES = ("fig1_trajectories.png", "fig2_estimators.png", "fig3_long_differences.png", "fig4_timing.png")


def render(inp: dict[str, Any], n: dict[str, Any], outdir: Path) -> None:
    _style()
    outdir.mkdir(parents=True, exist_ok=True)
    panel, cit, w = build_panel(inp)
    _, ld = long_differences(panel)
    fig_trajectories(cit, w, outdir / FIGURE_FILES[0])
    fig_estimators(n, outdir / FIGURE_FILES[1])
    fig_long_differences(ld, n, outdir / FIGURE_FILES[2])
    fig_timing(n, outdir / FIGURE_FILES[3])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _c(r: dict[str, Any], k: int = 2) -> str:
    return f"{fmt(r['coef'], k, sign=True)} [{fmt(r['ci95'][0], k)}, {fmt(r['ci95'][1], k)}] p {r.get('p', float('nan')):.3f}"


def summary_lines(n: dict[str, Any]) -> list[str]:
    d, a, e, ld, ls, t, lo, wo = (n[k] for k in ("describe", "annual", "estimators", "long_differences",
                                                  "level_or_spread", "timing", "loco", "wid_observed"))
    out = [f"Forbes totals {d['forbes_totals']}; partial earlier years {d['forbes_years_partial']}; latest fetched "
           f"{d['latest_list_fetched']}",
           f"Panel: {d['n_obs']} surveys, {d['n_countries']} countries, {d['n_spells']} spells "
           f"({d['n_spells_2plus']} with 2+), years {d['years']}, by welfare {d['n_welfare']}, zero-billionaire obs "
           f"{d['zero_billionaire_obs']}, dropped (no GDP) {d['dropped_no_gdp']}",
           f"SD total {({k: round(v, 3) for k, v in d['sd_total'].items()})}; within {({k: round(v, 3) for k, v in d['sd_within'].items()})}; "
           f"within corr B,G {d['corr_within_B_G']:.3f}",
           "Showcase: " + "; ".join(f"{v['name']} {v['n'][str(FIRST_YEAR)]}->{v['n'][str(LAST_LIST)]} (max {v['max']} in {v['max_year']})"
                                    for v in d["showcase"].values()),
           f"Annual ({a['n']} country-years, {a['n_countries']} countries): levels B~G {_c(a['levels_G'])}; PLR {_c(a['levels_PLR'])}; "
           f"dB~dG {_c(a['dB_on_dG'])}; lag {_c(a['dB_on_dG_lag'])}; lead {_c(a['dB_on_dG_lead'])}; dG~dB lag {_c(a['dG_on_dB_lag'], 4)}; "
           f"dB~dB lag {_c(a['dB_on_dB_lag'])}; R2 year {a['r2_year']:.3f} +growth+price {a['r2_year_growth_price']:.3f}; "
           f"SD dB {a['sd_dB']:.3f}; share falling {a['share_years_falling']:.3f}"]
    for k, v in e.items():
        if k == "drop_one":
            out.append(f"  drop one country: min {v['min']:+.2f} ({v['min_country']}), max {v['max']:+.2f} ({v['max_country']})")
            continue
        if k == "between":
            out.append(f"  {k}: {fmt(v['coef'], 2, sign=True)} [{fmt(v['ci95'][0], 2)}, {fmt(v['ci95'][1], 2)}] "
                       f"(p difference from within {v['p_difference']:.3f})")
        elif k == "within_fractional":
            out.append(f"  {k}: AME {fmt(v['ame'], 2, sign=True)} [{fmt(v['ci95'][0], 2)}, {fmt(v['ci95'][1], 2)}] p {v['p']:.3f}")
        else:
            out.append(f"  {k}: {_c(v)} n {v['n']}" + "".join(f" {x} {v[x]}" for x in ("n_countries", "n_spells") if x in v))
    out.append(f"Long differences: n {ld['n']} countries {ld['n_countries']} median gap {ld['median_gap']} corr dB,dG "
               f"{ld['corr_dB_dG']:.3f} share dB>0 {ld['share_dB_positive']:.3f}")
    for k in ("dp_raw", "dp", "dp_dG", "dM", "dM_dG", "dgini", "dp8"):
        out.append(f"  {k}: {_c(ld[k], 3)}")
    for k, v in ls.items():
        out.append(f"Level/spread {k}: " + (_c(v, 4) if "coef" in v else str({x: round(y, 4) for x, y in v.items()})))
    out.append("By line (within): " + "; ".join(f"${r['line']:g} {fmt(r['coef'], 2, sign=True)} p {r['p']:.2f}" for r in n["by_line"]))
    out.append(f"Timing levels (common {t['common_n']}, {t['common_countries']} countries, {t['common_years']}): " +
               "; ".join(f"k{r['k']:+d} {_c(r)} | same-year GDP {fmt(r['same_year_gdp']['coef'], 2, sign=True)} p {r['same_year_gdp']['p']:.3f}"
                         for r in t["levels"]))
    out.append(f"  around (B t+3 - B t-3): {_c(t['around'])}")
    c = t["changes"]
    out.append(f"Timing changes ({c['n']} pairs, {c['n_countries']} countries, median gap {c['median_gap']}): same {_c(c['same'])}; "
               f"before {_c(c['before'])}; after (dB_after on dp) {_c(c['after'], 4)}")
    out.append(f"LOCO within RMSE: GDP {lo['gdp']:.3f}; GDP+B {lo['gdp_b']:.3f}")
    out.append(f"Regions: surveys {d['n_region']}; countries {d['countries_region']}; surveys per country {d['surveys_per_country']}")
    out.append(f"Per doubling: {n['per_doubling']}")
    out.append(f"WID observed ({wo['n']} country-years, {wo['n_countries']} countries, {wo['years']}): " +
               "; ".join(f"{y} across {_c(wo[y]['across'])} within {_c(wo[y]['within'])}" for y in ("t1", "t10", "t001")))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--figures", type=Path, default=FIGURES)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    inp = load_inputs()
    numbers = json.loads(json.dumps(compute(inp)))
    lines_out = summary_lines(numbers)
    print("\n".join(lines_out))
    target = RESULTS / "numbers.json"
    if args.check:
        problems: list[str] = []
        _compare(numbers, json.loads(target.read_text(encoding="utf-8")), "numbers", problems)
        if problems:
            print("\nREPRODUCTION FAILED:\n" + "\n".join(problems[:50]), file=sys.stderr)
            return 1
        print("\nREPRODUCED: every number matches results/part3/numbers.json")
    else:
        RESULTS.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (RESULTS / "summary.txt").write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    if not args.no_figures:
        render(inp, numbers, args.figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
