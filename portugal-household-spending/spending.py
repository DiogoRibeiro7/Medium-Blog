"""Portuguese household spending on basic needs, saving, and basic-needs inflation.

Four questions, all from the Eurostat snapshot in ``data/``:

1. What share of household consumption goes to basic needs each year, 1995-2022
   (national accounts by COICOP purpose), with and without imputed rent?
2. How does that share differ by income quintile, 1988-2020 (household budget
   survey waves)?
3. How much do households save: the national-accounts saving rate (1995-2025)
   and the experimental median saving rate by income quintile (2010, 2015, 2020)?
4. Have basic needs become dearer than the average basket, and does each
   quintile face a different inflation rate (HICP by COICOP division weighted
   by each quintile's budget shares)?

Definitions
-----------
Core basic needs (cash): food and non-alcoholic beverages (CP01), actual rent
(CP041), maintenance and repair of the dwelling (CP043), water and other
dwelling services (CP044), electricity, gas and other fuels (CP045).
Wider basic needs: core plus health (CP06), operation of personal transport
equipment (CP072, mostly fuel) and transport services (CP073).
Imputed rent (CP042), the rent owner-occupiers notionally pay themselves, is not
a cash outlay; "excluding imputed rent" removes it from both the numerator and
the total.

Run from this folder::

    python spending.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"

CORE = ["CP01", "CP041", "CP043", "CP044", "CP045"]
WIDER_EXTRA = ["CP06", "CP072", "CP073"]
DIVISIONS = [f"CP{i:02d}" for i in range(1, 13)]
QUINTILES = ["QU1", "QU2", "QU3", "QU4", "QU5"]
NA_END = 2022          # last year of COICOP detail for Portugal (2021-22 provisional)
HICP_END = 2025


# ---------------------------------------------------------------- loading ---

def national_accounts() -> pd.DataFrame:
    """Household consumption by COICOP, million euro, wide by code."""
    n = pd.read_csv(DATA / "nama_10_co3_p3.csv")
    return n.pivot_table(index="year", columns="coicop", values="value").loc[:NA_END]


def budget_survey_quintiles() -> pd.DataFrame:
    """Budget-survey structure by quintile, per mille, index (quintile, year)."""
    h = pd.read_csv(DATA / "hbs_str_t223.csv")
    return h.pivot_table(index=["quant_inc", "year"], columns="coicop", values="value")


def budget_survey_total() -> pd.DataFrame:
    h = pd.read_csv(DATA / "hbs_str_t211.csv")
    return h.pivot_table(index="year", columns="coicop", values="value")


def saving_rate() -> pd.Series:
    k = pd.read_csv(DATA / "nasa_10_ki.csv")
    k = k[(k["na_item"] == "SRG_S14_S15") & (k["sector"] == "S14_S15")]
    return k.set_index("year")["value"]


def saving_by_quintile() -> pd.DataFrame:
    s = pd.read_csv(DATA / "icw_sr_03.csv")
    return s.pivot_table(index="quant_inc", columns="year", values="value")


def hicp() -> pd.DataFrame:
    p = pd.read_csv(DATA / "prc_hicp_aind.csv")
    return p.pivot_table(index="year", columns="coicop", values="value").loc[:HICP_END]


# --------------------------------------------------------------- measures ---

def shares(frame: pd.DataFrame, total: pd.Series, imputed: pd.Series | None) -> pd.DataFrame:
    """Food, housing, energy, core and wider basic-needs shares of ``total``.

    With ``imputed`` given, imputed rent is removed from the total.
    """
    denom = total - imputed if imputed is not None else total
    out = pd.DataFrame(index=frame.index)
    out["food"] = frame["CP01"] / denom
    out["rent_actual"] = frame["CP041"] / denom
    out["housing_other"] = (frame["CP043"] + frame["CP044"]) / denom
    out["energy"] = frame["CP045"] / denom
    out["core"] = frame[CORE].sum(axis=1) / denom
    if all(c in frame for c in WIDER_EXTRA):
        out["wider"] = (frame[CORE].sum(axis=1) + frame[WIDER_EXTRA].sum(axis=1)) / denom
    return out


def na_shares() -> pd.DataFrame:
    w = national_accounts()
    ex = shares(w, w["TOTAL"], w["CP042"])
    ex.columns = [f"{c}_ex_imputed" for c in ex.columns]
    inc = pd.DataFrame({
        "food_plus_housing_incl_imputed": (w["CP01"] + w["CP04"]) / w["TOTAL"],
        "imputed_rent": w["CP042"] / w["TOTAL"],
    })
    return ex.join(inc)


def hbs_shares() -> pd.DataFrame:
    """Shares by quintile and survey wave; excluding imputed rent where CP042 exists."""
    q = budget_survey_quintiles() / 1000
    rows = []
    for (quint, year), r in q.iterrows():
        rec = {"quintile": quint, "year": year,
               "food": r["CP01"], "housing_incl_imputed": r["CP04"],
               "food_plus_housing_incl_imputed": r["CP01"] + r["CP04"],
               "health": r.get("CP06", np.nan), "transport": r.get("CP07", np.nan)}
        if pd.notna(r.get("CP042")):
            denom = 1 - r["CP042"]
            rec["imputed_rent"] = r["CP042"]
            rec["core_ex_imputed"] = r[CORE].sum() / denom
            rec["wider_ex_imputed"] = (r[CORE].sum() + r[WIDER_EXTRA].sum()) / denom
            rec["food_ex_imputed"] = r["CP01"] / denom
            rec["rent_actual_ex_imputed"] = r["CP041"] / denom
            rec["energy_ex_imputed"] = r["CP045"] / denom
        rows.append(rec)
    return pd.DataFrame(rows).set_index(["quintile", "year"]).sort_index()


def quintile_inflation(start: int, end: int, wave: int) -> pd.DataFrame:
    """Cumulative inflation by quintile, ``start`` to ``end``.

    Weights: each quintile's division shares in budget-survey ``wave`` with
    imputed rent removed (HICP does not cover owner-occupied housing), applied
    as a fixed-weight (Laspeyres) index of the twelve HICP divisions rebased to
    ``start``.
    """
    q = budget_survey_quintiles() / 1000
    t = budget_survey_total() / 1000
    p = hicp()
    rel = p[DIVISIONS].loc[end] / p[DIVISIONS].loc[start]
    rows = []
    for quint in QUINTILES + ["TOTAL"]:
        r = t.loc[wave] if quint == "TOTAL" else q.loc[(quint, wave)]
        w = r[DIVISIONS].copy()
        w["CP04"] = w["CP04"] - r["CP042"]
        w = w / w.sum()
        rows.append({"quintile": quint, "inflation": float((w * rel).sum() - 1),
                     "basic_weight": float((r[CORE].sum()) / (1 - r["CP042"]))})
    out = pd.DataFrame(rows).set_index("quintile")
    out.loc["HICP all items", "inflation"] = p.loc[end, "CP00"] / p.loc[start, "CP00"] - 1
    return out


def basic_needs_price(start: int, end: int, wave: int) -> pd.Series:
    """Fixed-weight price index of core basic needs vs all items, rebased to ``start``."""
    t = budget_survey_total().loc[wave] / 1000
    w = t[CORE] / t[CORE].sum()
    p = hicp()
    idx = (p[CORE].loc[start:end].div(p[CORE].loc[start]) * w).sum(axis=1)
    allitems = p["CP00"].loc[start:end] / p.loc[start, "CP00"]
    return pd.DataFrame({"basic_needs": idx, "all_items": allitems, "relative": idx / allitems})


# ------------------------------------------------------------------- main ---

def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    lines: list[str] = []
    w = lines.append

    na = na_shares()
    na.round(4).to_csv(RESULTS / "na_basic_shares.csv")
    s = saving_rate()
    w("Household consumption by purpose (national accounts), share of consumption excluding imputed rent")
    for y in (1995, 2000, 2005, 2010, 2015, 2019, 2020, 2021, NA_END):
        r = na.loc[y]
        w(f"  {y}: food {100 * r['food_ex_imputed']:.1f}%, actual rent {100 * r['rent_actual_ex_imputed']:.1f}%, "
          f"energy {100 * r['energy_ex_imputed']:.1f}%, core {100 * r['core_ex_imputed']:.1f}%, "
          f"wider {100 * r['wider_ex_imputed']:.1f}%; imputed rent {100 * r['imputed_rent']:.1f}% of total; "
          f"saving rate {s.get(y, np.nan):.1f}%")
    w(f"  core share: min {100 * na['core_ex_imputed'].min():.1f}% ({na['core_ex_imputed'].idxmin()}), "
      f"max {100 * na['core_ex_imputed'].max():.1f}% ({na['core_ex_imputed'].idxmax()})")
    flags = pd.read_csv(DATA / "nama_10_co3_p3.csv").query("flag == 'p'")["year"].unique()
    w(f"  provisional years: {', '.join(str(int(y)) for y in sorted(flags))}")
    w("")

    w("Household saving rate, gross, % of disposable income (S14_S15)")
    w("  " + ", ".join(f"{y} {s[y]:.1f}" for y in (1995, 2000, 2005, 2008, 2009, 2010, 2014, 2017, 2019, 2020,
                                                  2021, 2022, 2023, 2024, 2025)))
    w(f"  lowest {s.min():.2f} in {s.idxmin()}; mean 1995-2007 {s.loc[1995:2007].mean():.1f}, "
      f"2008-2019 {s.loc[2008:2019].mean():.1f}, 2022-2025 {s.loc[2022:2025].mean():.1f}")
    s.to_csv(RESULTS / "saving_rate.csv", header=["saving_rate"])
    w("")

    hb = hbs_shares()
    hb.round(4).to_csv(RESULTS / "hbs_shares_by_quintile.csv")
    w("Household budget survey by income quintile (share of consumption)")
    for col, label in (("food", "food"), ("housing_incl_imputed", "housing incl. imputed rent"),
                       ("core_ex_imputed", "core basic needs excl. imputed rent"),
                       ("wider_ex_imputed", "wider basic needs excl. imputed rent"),
                       ("rent_actual_ex_imputed", "actual rent excl. imputed rent"),
                       ("energy_ex_imputed", "energy excl. imputed rent")):
        tab = hb[col].unstack("year")
        w(f"  {label}:")
        for quint in QUINTILES:
            w("    " + quint + " " + "  ".join(
                f"{y} {100 * v:5.1f}" for y, v in tab.loc[quint].items() if pd.notna(v)))
    w("")

    sq = saving_by_quintile()
    sq.to_csv(RESULTS / "saving_by_quintile.csv")
    w("Median saving rate by income quintile, % of disposable income (experimental, Eurostat icw_sr_03)")
    for quint in QUINTILES + ["TOTAL"]:
        w(f"  {quint}: " + ", ".join(f"{y} {v:+.1f}" for y, v in sq.loc[quint].items()))
    w("")

    w("Inflation by quintile: HICP divisions weighted by each quintile's budget shares (imputed rent removed)")
    rows = []
    for start, end, wave in ((2005, 2015, 2005), (2015, 2025, 2015), (2020, 2025, 2015), (2021, 2023, 2015)):
        qi = quintile_inflation(start, end, wave)
        for quint, r in qi.iterrows():
            rows.append({"start": start, "end": end, "wave": wave, "quintile": quint, **r.to_dict()})
        w(f"  {start}-{end} (weights {wave}): " + ", ".join(
            f"{qq} {100 * v:.1f}%" for qq, v in qi["inflation"].items()))
        gap = qi.loc["QU1", "inflation"] - qi.loc["QU5", "inflation"]
        w(f"    QU1 minus QU5: {100 * gap:+.1f} points")
    pd.DataFrame(rows).round(5).to_csv(RESULTS / "inflation_by_quintile.csv", index=False)
    w("")

    w("Core basic needs price index against all items (weights: budget survey total, imputed rent removed)")
    rel_rows = []
    for start, end, wave in ((1996, 2005, 1999), (2005, 2015, 2005), (2015, 2025, 2015), (2020, 2025, 2015)):
        bp = basic_needs_price(start, end, wave)
        rel_rows.append(bp.assign(start=start, wave=wave))
        w(f"  {start}-{end} (weights {wave}): basic needs +{100 * (bp['basic_needs'].iloc[-1] - 1):.1f}%, "
          f"all items +{100 * (bp['all_items'].iloc[-1] - 1):.1f}%, relative {bp['relative'].iloc[-1]:.3f}")
    pd.concat(rel_rows).round(5).to_csv(RESULTS / "basic_needs_price.csv")
    p = hicp()
    w("  components 2015-2025: " + ", ".join(
        f"{c} +{100 * (p.loc[2025, c] / p.loc[2015, c] - 1):.1f}%" for c in CORE + ["CP00"]))
    w("  components 2020-2025: " + ", ".join(
        f"{c} +{100 * (p.loc[2025, c] / p.loc[2020, c] - 1):.1f}%" for c in CORE + ["CP00"]))
    w("")

    (RESULTS / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
