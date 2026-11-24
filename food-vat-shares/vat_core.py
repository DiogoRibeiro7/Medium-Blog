"""Functions behind the food-VAT incidence numbers. Everything here is pure: tables in, numbers out.

Notation used in the docstrings: for income fifth q (1 = poorest, 5 = richest), s_q is the share of food in
the household budget, c_q is consumption spending per adult equivalent, and f_q = s_q * c_q is food spending.
A cut that lowers food prices by a fraction p, fully passed on, saves p * f_q a year. In euros the saving is
p * f_q; as a share of the budget it is p * s_q. The first rises with income whenever food spending rises
with income; the second falls with income whenever the food share falls with income (Engel's law).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FIFTHS = ["QU1", "QU2", "QU3", "QU4", "QU5"]
DECILE_CUTS = [f"D{i}" for i in range(1, 10)]

EU27 = [
    "BE", "BG", "CZ", "DK", "DE", "EE", "IE", "EL", "ES", "FR", "HR", "IT", "CY", "LV",
    "LT", "LU", "HU", "MT", "NL", "AT", "PL", "PT", "RO", "SI", "SK", "FI", "SE",
]  # fmt: skip
NAMES = {
    "BE": "Belgium", "BG": "Bulgaria", "CZ": "Czechia", "DK": "Denmark", "DE": "Germany", "EE": "Estonia",
    "IE": "Ireland", "EL": "Greece", "ES": "Spain", "FR": "France", "HR": "Croatia", "IT": "Italy",
    "CY": "Cyprus", "LV": "Latvia", "LT": "Lithuania", "LU": "Luxembourg", "HU": "Hungary", "MT": "Malta",
    "NL": "Netherlands", "AT": "Austria", "PL": "Poland", "PT": "Portugal", "RO": "Romania", "SI": "Slovenia",
    "SK": "Slovakia", "FI": "Finland", "SE": "Sweden", "EU27_2020": "EU27",
}  # fmt: skip


def price_fall(rate_before: float, rate_after: float) -> float:
    """Fall in the consumer price when VAT goes from rate_before to rate_after and is fully passed on.

    The price is net price times (1 + rate), so the proportional change is (1 + after) / (1 + before) - 1.
    A cut from 6% to 0% lowers the price by 6/106 = 5.66%, not by 6%.
    """
    return 1.0 - (1.0 + rate_after) / (1.0 + rate_before)


def budget_table(structure: pd.DataFrame, spending: pd.DataFrame, wave: int, cash: bool = True) -> pd.DataFrame:
    """One row per (country, fifth) for one survey wave.

    structure: hbs_str_t223 in long form (per mille of consumption by COICOP); spending: hbs_exp_t133
    (PPS per adult equivalent and per household). With cash=True the denominator excludes imputed rents of
    owner-occupiers (CP042), which are not paid by anyone and are treated differently across countries; where
    CP042 is not published for a wave, the published total is used and `cash_basis` is False.

    Columns: food, food_drink, restaurants (shares of the budget, as fractions); spend_ae and spend_hh (PPS a
    year on the chosen basis); cash_basis.
    """
    s = structure[(structure["time"] == wave) & structure["quant_inc"].isin(FIFTHS)]
    s = s.pivot_table(index=["geo", "quant_inc"], columns="coicop", values="value") / 1000.0
    e = spending[(spending["time"] == wave) & spending["quant_inc"].isin(FIFTHS)]
    e = e.pivot_table(index=["geo", "quant_inc"], columns="unit", values="value")
    t = s.join(e, how="inner")
    t = t[t["CP01"].notna() & t["PPS_AE"].notna()]
    imputed = t["CP042"] if "CP042" in t else pd.Series(np.nan, index=t.index)
    use_cash = cash & imputed.notna()
    keep = np.where(use_cash, 1.0 - imputed.fillna(0.0), 1.0)
    out = pd.DataFrame(index=t.index)
    out["food_drink"] = t["CP01"] / keep
    out["food"] = (t["CP011"] / keep) if "CP011" in t else np.nan
    out["restaurants"] = (t["CP111"] / keep) if "CP111" in t else np.nan
    out["spend_ae"] = t["PPS_AE"] * keep
    out["spend_hh"] = t["PPS_HH"] * keep
    out["cash_basis"] = use_cash
    return out


def complete_countries(table: pd.DataFrame, column: str) -> list[str]:
    """Countries with all five fifths present for `column`, in the order of EU27 (aggregates excluded)."""
    ok = table[column].notna().groupby(level="geo").sum() == 5
    return [g for g in EU27 if g in ok.index and ok[g]]


def fifth_profile(table: pd.DataFrame, geo: str, column: str, p: float = 0.01) -> pd.DataFrame:
    """Saving from a fall p in the price of `column`, by fifth, for one country.

    euros_ae: saving per adult equivalent a year (PPS); euros_hh: per household; pct_budget: saving as a
    percentage of the budget (= 100 * p * share); share_of_total: each fifth's share of the summed saving,
    treating the five groups as equal in size; spend_share_of_total: the same for total spending.
    """
    t = table.loc[geo].reindex(FIFTHS)
    share = t[column]
    out = pd.DataFrame(index=FIFTHS)
    out["share"] = share
    out["euros_ae"] = p * share * t["spend_ae"]
    out["euros_hh"] = p * share * t["spend_hh"]
    out["pct_budget"] = 100.0 * p * share
    out["share_of_total"] = out["euros_hh"] / out["euros_hh"].sum()
    out["spend_share_of_total"] = t["spend_hh"] / t["spend_hh"].sum()
    return out


def concentration_coefficient(amounts) -> float:
    """Concentration coefficient of an amount across five equal groups ranked by income, poorest first.

    The concentration curve joins (0, 0), (0.2, L1), ..., (1, 1), where L_k is the cumulative share of the
    amount held by the k poorest groups. The coefficient is 1 - 2 * (area under the curve), computed with the
    trapezoid rule: 0 when every group holds the same amount, positive when the amount rises with income,
    negative when it falls. It is a Gini coefficient when the groups are ranked by the amount itself.
    """
    a = np.asarray(amounts, dtype=float)
    if a.ndim != 1 or len(a) < 2 or np.any(~np.isfinite(a)) or a.sum() <= 0:
        return float("nan")
    n = len(a)
    cum = np.concatenate([[0.0], np.cumsum(a) / a.sum()])
    area = np.sum((cum[1:] + cum[:-1]) / 2.0) / n
    return float(1.0 - 2.0 * area)


def kakwani(saving, spending) -> float:
    """Progressivity of a saving relative to spending: C(saving) - C(spending), groups ranked by income.

    Negative: the saving is spread more equally than spending, so as a share of spending it falls with income
    (progressive in the public-finance sense, even if richer groups receive more euros). Positive: regressive.
    Zero: proportional to spending.
    """
    return concentration_coefficient(saving) - concentration_coefficient(spending)


def top_bottom_ratio_change(share, p: float) -> float:
    """Proportional change in the ratio of richest to poorest real spending after a price fall p on a good.

    share holds the good's budget share by fifth, poorest first. Real spending of fifth q becomes
    c_q * (1 + p * s_q) to first order (the saving can buy more of everything). The ratio c5/c1 changes by
    (1 + p * s5) / (1 + p * s1) - 1, which does not depend on c and is negative exactly when s1 > s5: the
    inequality of real spending falls when, and only when, the cut is progressive in shares.
    """
    share = np.asarray(share, dtype=float)
    return float((1.0 + p * share[-1]) / (1.0 + p * share[0]) - 1.0)


def percentile_from_cuts(income: float, cuts) -> float:
    """Position of an income (percent of people below it), by linear interpolation between decile cut-offs.

    cuts are the top cut-offs of deciles 1 to 9 (ilc_di01). Below the first cut-off the position is
    interpolated from zero income; above the ninth it is reported as 90 or more and returned as NaN.
    """
    c = np.asarray(cuts, dtype=float)
    if len(c) != 9 or np.any(~np.isfinite(c)) or np.any(np.diff(c) <= 0):
        raise ValueError("need nine increasing decile cut-offs")
    if income >= c[-1]:
        return float("nan")
    xs = np.concatenate([[0.0], c])
    ps = np.arange(0, 100, 10, dtype=float)
    return float(np.interp(income, xs, ps))


def equivalised(income: float, adults: int, children: int) -> float:
    """Household disposable income per equivalent adult, modified OECD scale (1, 0.5, 0.3), as EU-SILC uses.

    Children are members under 14.
    """
    if adults < 1 or children < 0:
        raise ValueError("a household has at least one adult")
    return income / (1.0 + 0.5 * (adults - 1) + 0.3 * children)
