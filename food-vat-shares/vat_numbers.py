"""Every number in the article, printed in the order the article uses them, and the tables in results/.

python vat_numbers.py > output.txt

Reads only the registry snapshot in data/raw/ (run fetch.py first).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import vat_core as vc
from jsonstat import read_jsonstat

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
RESULTS = HERE / "results"

MAIN_WAVE = 2015  # every member state but Italy, collected 2012-2017, before the pandemic
CHECK_WAVES = (2010, 2020)
COPIED_2020 = {"CY", "FR", "MT"}  # Eurostat's 2020 figures for these are their 2015 survey, repriced
P = 0.01  # a 1% fall in food prices, fully passed on
CUT_6_TO_0 = vc.price_fall(0.06, 0.0)
RISE_6_TO_23 = -vc.price_fall(0.06, 0.23)


def load(name):
    return read_jsonstat(RAW / f"{name}.json")


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def country_rows(table, column, countries):
    rows = []
    for g in countries:
        prof = vc.fifth_profile(table, g, column, P)
        s1, s5 = prof["share"].iloc[0], prof["share"].iloc[-1]
        e1, e5 = prof["euros_ae"].iloc[0], prof["euros_ae"].iloc[-1]
        rows.append(
            {
                "geo": g,
                "country": vc.NAMES[g],
                "share_q1": s1,
                "share_q5": s5,
                "share_ratio_q1_q5": s1 / s5,
                "euros_ratio_q5_q1": e5 / e1,
                "spend_ratio_q5_q1": table.loc[(g, "QU5"), "spend_ae"] / table.loc[(g, "QU1"), "spend_ae"],
                "saving_share_q1": prof["share_of_total"].iloc[0],
                "saving_share_q5": prof["share_of_total"].iloc[-1],
                "spend_share_q1": prof["spend_share_of_total"].iloc[0],
                "spend_share_q5": prof["spend_share_of_total"].iloc[-1],
                "kakwani": vc.kakwani(prof["euros_hh"], table.loc[g].reindex(vc.FIFTHS)["spend_hh"]),
                "monotone_down": bool(np.all(np.diff(prof["share"].to_numpy()) < 0)),
                "ratio_change_1pc": vc.top_bottom_ratio_change(prof["share"], P),
                "ratio_change_6_to_0": vc.top_bottom_ratio_change(prof["share"], CUT_6_TO_0),
            }
        )
    return pd.DataFrame(rows).set_index("geo")


def main():
    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    structure = load("hbs_str_t223")
    spending = load("hbs_exp_t133")
    icw = load("icw_aff_01")
    propensity = load("icw_sr_10")
    income = load("ilc_di01")
    totals = load("hbs_exp_t111")
    hicp = load("prc_hicp_aind")

    section("1. Price arithmetic")
    print(f"VAT 6% -> 0%, fully passed on: price falls {pct(CUT_6_TO_0, 2)}")
    print(
        f"VAT 4% -> 0%: {pct(vc.price_fall(0.04, 0.0), 2)};  5% -> 0%: {pct(vc.price_fall(0.05, 0.0), 2)};  12% -> 6%: {pct(vc.price_fall(0.12, 0.06), 2)}"
    )
    print(f"VAT 6% -> 23% (food at the standard rate): price rises {pct(RISE_6_TO_23, 2)}")

    section("2. Coverage of the budget survey (hbs_str_t223, hbs_exp_t133), cash basis (imputed rents excluded)")
    tables = {w: vc.budget_table(structure, spending, w, cash=True) for w in (MAIN_WAVE, *CHECK_WAVES)}
    for w, t in tables.items():
        cs = vc.complete_countries(t, "food_drink")
        cash = t.loc[cs, "cash_basis"].groupby(level="geo").all()
        print(f"wave {w}: {len(cs)} member states with food by fifth: {' '.join(cs)}")
        print(f"   published total used instead of cash basis (no CP042): {' '.join(g for g in cs if not cash[g]) or 'none'}")
    main_t = tables[MAIN_WAVE]
    countries = vc.complete_countries(main_t, "food_drink")

    section(f"3. Food and non-alcoholic drink as a share of the budget, poorest and richest fifth, wave {MAIN_WAVE}")
    food = country_rows(main_t, "food_drink", countries)
    food.to_csv(RESULTS / f"food_by_country_{MAIN_WAVE}.csv")
    show = food[["share_q1", "share_q5", "share_ratio_q1_q5", "euros_ratio_q5_q1", "spend_ratio_q5_q1", "monotone_down"]]
    print((show.assign(share_q1=100 * show.share_q1, share_q5=100 * show.share_q5)).round(2).sort_values("share_ratio_q1_q5").to_string())
    n = len(food)
    print(f"\nshare of the poorest fifth above the richest: {(food.share_ratio_q1_q5 > 1).sum()} of {n}")
    print(f"share falls at every step from poorest to richest: {food.monotone_down.sum()} of {n}")
    print(
        f"ratio share_q1/share_q5: median {food.share_ratio_q1_q5.median():.2f}, min {food.share_ratio_q1_q5.min():.2f} ({food.share_ratio_q1_q5.idxmin()}), max {food.share_ratio_q1_q5.max():.2f} ({food.share_ratio_q1_q5.idxmax()})"
    )
    eu = country_rows(main_t, "food_drink", ["EU27_2020"]) if "EU27_2020" in main_t.index.get_level_values(0) else None
    if eu is not None:
        r = eu.iloc[0]
        print(
            f"EU27 aggregate: poorest {pct(r.share_q1)}, richest {pct(r.share_q5)}, ratio {r.share_ratio_q1_q5:.2f}; euros ratio {r.euros_ratio_q5_q1:.2f}; spending ratio {r.spend_ratio_q5_q1:.2f}"
        )

    section("4. Euros and shares at once: a 1% fall in food prices")
    both = (food.share_ratio_q1_q5 > 1) & (food.euros_ratio_q5_q1 > 1)
    print(f"richest fifth saves more euros per adult equivalent than the poorest: {(food.euros_ratio_q5_q1 > 1).sum()} of {n}")
    print(f"... and the poorest saves more as a share of its budget: {both.sum()} of {n}")
    print(
        f"euros ratio (richest/poorest): median {food.euros_ratio_q5_q1.median():.2f}, range {food.euros_ratio_q5_q1.min():.2f} ({food.euros_ratio_q5_q1.idxmin()}) to {food.euros_ratio_q5_q1.max():.2f} ({food.euros_ratio_q5_q1.idxmax()})"
    )
    print(
        f"spending ratio (richest/poorest): median {food.spend_ratio_q5_q1.median():.2f}, range {food.spend_ratio_q5_q1.min():.2f} to {food.spend_ratio_q5_q1.max():.2f}"
    )
    print(f"countries where the euros ratio is below the spending ratio: {(food.euros_ratio_q5_q1 < food.spend_ratio_q5_q1).sum()} of {n}")
    print(
        f"richest fifth's share of the total saving: median {pct(food.saving_share_q5.median())} (range {pct(food.saving_share_q5.min())} to {pct(food.saving_share_q5.max())}); of total spending: median {pct(food.spend_share_q5.median())}"
    )
    print(
        f"poorest fifth's share of the total saving: median {pct(food.saving_share_q1.median())} (range {pct(food.saving_share_q1.min())} to {pct(food.saving_share_q1.max())}); of total spending: median {pct(food.spend_share_q1.median())}"
    )
    print(
        f"Kakwani index of the saving relative to spending: median {food.kakwani.median():.3f}; negative (progressive) in {(food.kakwani < 0).sum()} of {n}"
    )
    print(f"most progressive: {', '.join(food.kakwani.nsmallest(3).index)}; least: {', '.join(food.kakwani.nlargest(3).index)}")
    if eu is not None:
        r = eu.iloc[0]
        print(
            f"EU27: richest fifth gets {pct(r.saving_share_q5)} of the saving and {pct(r.spend_share_q5)} of spending; poorest {pct(r.saving_share_q1)} and {pct(r.spend_share_q1)}; Kakwani {r.kakwani:.3f}"
        )
        vc.fifth_profile(main_t, "EU27_2020", "food_drink", P).to_csv(RESULTS / f"eu27_profile_{MAIN_WAVE}.csv")

    section("5. Inequality of real spending: change in the richest/poorest ratio")
    print(f"per 1% fall in food prices: median {pct(food.ratio_change_1pc.median(), 3)}; falls in {(food.ratio_change_1pc < 0).sum()} of {n}")
    print(
        f"for a 6% -> 0% cut on all food and drink ({pct(CUT_6_TO_0, 2)}): median {pct(food.ratio_change_6_to_0.median(), 2)}, range {pct(food.ratio_change_6_to_0.min(), 2)} ({food.ratio_change_6_to_0.idxmin()}) to {pct(food.ratio_change_6_to_0.max(), 2)} ({food.ratio_change_6_to_0.idxmax()})"
    )

    section(f"6. Restaurants and cafés (CP111) against food, wave {MAIN_WAVE}")
    rest_cs = vc.complete_countries(main_t, "restaurants")
    rest = country_rows(main_t, "restaurants", rest_cs)
    rest.to_csv(RESULTS / f"restaurants_by_country_{MAIN_WAVE}.csv")
    m = len(rest)
    print(f"countries: {m}")
    print(f"richest fifth spends a larger share on restaurants than the poorest: {(rest.share_ratio_q1_q5 < 1).sum()} of {m}")
    print(f"share ratio poorest/richest: median {rest.share_ratio_q1_q5.median():.2f}; food median {food.share_ratio_q1_q5.median():.2f}")
    print(f"euros ratio richest/poorest: restaurants median {rest.euros_ratio_q5_q1.median():.2f}; food {food.euros_ratio_q5_q1.median():.2f}")
    print(
        f"Kakwani: restaurants median {rest.kakwani.median():.3f}, positive (regressive) in {(rest.kakwani > 0).sum()} of {m}; food median {food.kakwani.median():.3f}"
    )
    print(f"richest fifth's share of a restaurant saving: median {pct(rest.saving_share_q5.median())}; poorest {pct(rest.saving_share_q1.median())}")
    for g in ("DE", "IE", "PT", "ES", "SE"):
        if g in rest.index:
            r, f = rest.loc[g], food.loc[g]
            print(
                f"  {g}: restaurants {pct(r.share_q1)} -> {pct(r.share_q5)} (Kakwani {r.kakwani:+.3f}); food {pct(f.share_q1)} -> {pct(f.share_q5)} (Kakwani {f.kakwani:+.3f})"
            )
    if "EU27_2020" in main_t.index.get_level_values(0) and main_t.loc["EU27_2020", "restaurants"].notna().all():
        r = country_rows(main_t, "restaurants", ["EU27_2020"]).iloc[0]
        print(f"EU27 restaurants: {pct(r.share_q1)} -> {pct(r.share_q5)}, Kakwani {r.kakwani:+.3f}")

    section("7. The other waves")
    for w in CHECK_WAVES:
        t = tables[w]
        cs = [g for g in vc.complete_countries(t, "food_drink") if not (w == 2020 and g in COPIED_2020)]
        f = country_rows(t, "food_drink", cs)
        f.to_csv(RESULTS / f"food_by_country_{w}.csv")
        print(f"wave {w}: {len(cs)} countries (2020 excludes the repriced CY, FR, MT)")
        print(
            f"   poorest share above richest: {(f.share_ratio_q1_q5 > 1).sum()} of {len(f)}; richest saves more euros: {(f.euros_ratio_q5_q1 > 1).sum()}; both: {((f.share_ratio_q1_q5 > 1) & (f.euros_ratio_q5_q1 > 1)).sum()}"
        )
        print(
            f"   share ratio median {f.share_ratio_q1_q5.median():.2f}; Kakwani median {f.kakwani.median():.3f}, negative in {(f.kakwani < 0).sum()}"
        )
        if w == 2020:
            same = [g for g in cs if g in food.index]
            d = pd.DataFrame({"r2015": food.loc[same, "share_ratio_q1_q5"], "r2020": f.loc[same, "share_ratio_q1_q5"]})
            print(f"   share ratio fell from 2015 to 2020 in {(d.r2020 < d.r2015).sum()} of {len(d)} countries")
            not_prog = f.index[f.share_ratio_q1_q5 <= 1].tolist()
            print(f"   not progressive in shares in 2020: {', '.join(not_prog) or 'none'}")
            r20 = [g for g in vc.complete_countries(t, "restaurants") if g not in COPIED_2020 and g in rest.index]
            drop = pd.DataFrame(
                {
                    "rest_q5_2015": [rest.loc[g, "share_q5"] for g in r20],
                    "rest_q5_2020": [t.loc[(g, "QU5"), "restaurants"] for g in r20],
                    "rest_q1_2015": [rest.loc[g, "share_q1"] for g in r20],
                    "rest_q1_2020": [t.loc[(g, "QU1"), "restaurants"] for g in r20],
                },
                index=r20,
            )
            drop["d5"] = drop.rest_q5_2020 - drop.rest_q5_2015
            drop["d1"] = drop.rest_q1_2020 - drop.rest_q1_2015
            print(
                f"   restaurants share of the richest fifth fell 2015->2020 in {(drop.d5 < 0).sum()} of {len(drop)}; fell more than the poorest's in {(drop.d5 < drop.d1).sum()}"
            )
            for g in ("BE", "ES", "NL", "PL", "DK", "FI"):
                if g in drop.index and g in d.index:
                    print(
                        f"   {g}: food ratio {d.loc[g, 'r2015']:.2f} -> {d.loc[g, 'r2020']:.2f}; richest restaurants share {pct(drop.loc[g, 'rest_q5_2015'])} -> {pct(drop.loc[g, 'rest_q5_2020'])}"
                    )

    section("8. Food as a share of disposable income (ICW experimental statistics, icw_aff_01)")
    a = icw[icw["coicop"] == "CP01"].pivot_table(index=["time", "geo"], columns="quant_inc", values="value")
    c = icw[icw["coicop"] == "CP00"].pivot_table(index=["time", "geo"], columns="quant_inc", values="value")
    for w in (2015, 2020):
        aw = a.loc[w]
        aw = aw.loc[[g for g in vc.EU27 if g in aw.index and aw.loc[g, ["QU1", "QU5", "D1", "D10"]].notna().all()]]
        r15 = aw.QU1 / aw.QU5
        r110 = aw.D1 / aw.D10
        print(
            f"{w}: {len(aw)} countries; poorest/richest fifth ratio median {r15.median():.2f} (min {r15.min():.2f} {r15.idxmin()}, max {r15.max():.2f} {r15.idxmax()}); D1/D10 median {r110.median():.2f}"
        )
        print(
            f"   poorest fifth spends a larger share of income on food than the richest in {(r15 > 1).sum()} of {len(aw)}; at every step: {(np.diff(aw[vc.FIFTHS].to_numpy(), axis=1) < 0).all(axis=1).sum()}"
        )
        if (w, "EU27_2020") in a.index:
            e = a.loc[(w, "EU27_2020")]
            ce = c.loc[(w, "EU27_2020")]
            print(
                f"   EU27: food {e.QU1:.1f}% of income in the poorest fifth, {e.QU5:.1f}% in the richest (D1 {e.D1:.1f}%, D10 {e.D10:.1f}%); consumption {ce.QU1:.0f}% and {ce.QU5:.0f}% of income"
            )
        aw.to_csv(RESULTS / f"food_pct_income_{w}.csv")
    both = pd.DataFrame(
        {
            "spend_ratio": food["share_ratio_q1_q5"],
            "income_ratio": (a.loc[2015].QU1 / a.loc[2015].QU5).reindex(food.index),
        }
    ).dropna()
    print(
        f"2015, same countries: the income-based ratio exceeds the spending-based ratio in {(both.income_ratio > both.spend_ratio).sum()} of {len(both)}"
    )
    sp = propensity.pivot_table(index=["time", "geo"], columns="quant_inc", values="value")
    if (2020, "EU27_2020") in sp.index:
        s = sp.loc[(2020, "EU27_2020")]
        print(f"aggregate propensity to consume, EU27 2020: poorest fifth {s.QU1:.0f}%, richest {s.QU5:.0f}% of disposable income")

    section("9. Portugal")
    t20 = tables[2020]
    pt = vc.fifth_profile(t20, "PT", "food_drink", P)
    pt15 = vc.fifth_profile(main_t, "PT", "food_drink", P)
    print("wave 2020 (Portugal's survey of 2022/23), published total as denominator (CP042 not published):")
    tot = totals[(totals.geo == "PT") & (totals.time == 2020)].set_index("unit").value
    ppp_pt = tot["EUR_HH"] / tot["PPS_HH"]
    print(f"mean spending per household, 2020 wave: {tot['EUR_HH']:,.0f} euros a year ({tot['EUR_HH'] / 12:,.0f} a month)")
    h = hicp[(hicp.geo == "PT") & (hicp.coicop == "CP01")].set_index("time").value
    h_all = hicp[(hicp.geo == "PT") & (hicp.coicop == "CP00")].set_index("time").value
    last = int(h.index.max())
    uprate = h.loc[last] / h.loc[2020]
    print(
        f"euros per PPS implied by hbs_exp_t111 (mean spending per household, EUR over PPS), 2020: {ppp_pt:.4f}; HICP food {last}/2020 = {uprate:.4f} (all items {h_all.loc[last] / h_all.loc[2020]:.4f})"
    )
    spend_hh_eur = t20.loc["PT"].reindex(vc.FIFTHS)["spend_hh"] * ppp_pt
    food_hh_eur = pt["share"] * spend_hh_eur
    rows = pd.DataFrame(
        {
            "share_pct": 100 * pt["share"],
            "spend_hh_eur_month": spend_hh_eur / 12,
            "food_hh_eur_month_2020": food_hh_eur / 12,
            "food_hh_eur_month_now": food_hh_eur / 12 * uprate,
            "saving_1pc_eur_month_now": P * food_hh_eur / 12 * uprate,
            "saving_6to0_eur_month_now": CUT_6_TO_0 * food_hh_eur / 12 * uprate,
            "saving_6to0_pct_budget": 100 * CUT_6_TO_0 * pt["share"],
            "cost_6to23_eur_month_now": RISE_6_TO_23 * food_hh_eur / 12 * uprate,
            "cost_6to23_pct_budget": 100 * RISE_6_TO_23 * pt["share"],
            "saving_share_of_total_pct": 100 * pt["share_of_total"],
            "spend_share_of_total_pct": 100 * pt["spend_share_of_total"],
        }
    )
    rows.to_csv(RESULTS / "portugal_2020.csv")
    print(rows.round(2).to_string())
    print(
        f"euros ratio richest/poorest (per household): {rows.food_hh_eur_month_2020.iloc[-1] / rows.food_hh_eur_month_2020.iloc[0]:.2f}; per adult equivalent: {pt.euros_ae.iloc[-1] / pt.euros_ae.iloc[0]:.2f}"
    )
    print(
        f"spending ratio richest/poorest per adult equivalent: {t20.loc[('PT', 'QU5'), 'spend_ae'] / t20.loc[('PT', 'QU1'), 'spend_ae']:.2f}; per household: {spend_hh_eur.iloc[-1] / spend_hh_eur.iloc[0]:.2f}"
    )
    print(
        f"share ratio poorest/richest: {pt.share.iloc[0] / pt.share.iloc[-1]:.2f}; Kakwani {vc.kakwani(pt.euros_hh, t20.loc['PT'].reindex(vc.FIFTHS)['spend_hh']):+.3f}"
    )
    print(
        f"richest/poorest spending ratio after a 6%->0% cut on all food and drink: {pct(vc.top_bottom_ratio_change(pt.share, CUT_6_TO_0), 2)}; after 6%->23%: {pct(vc.top_bottom_ratio_change(pt.share, -RISE_6_TO_23), 2)}"
    )
    print(
        f"wave 2015, cash basis: shares {100 * pt15.share.iloc[0]:.1f}% and {100 * pt15.share.iloc[-1]:.1f}%; euros ratio {pt15.euros_ae.iloc[-1] / pt15.euros_ae.iloc[0]:.2f}"
    )
    if (2020, "PT") in a.index:
        ap = a.loc[(2020, "PT")]
        cp = c.loc[(2020, "PT")]
        print(
            f"ICW 2020: food {ap.QU1:.1f}% of disposable income in the poorest fifth, {ap.QU5:.1f}% in the richest; D1 {ap.D1:.1f}%, D10 {ap.D10:.1f}%"
        )
        print(f"   ratio fifths {ap.QU1 / ap.QU5:.2f}, deciles {ap.D1 / ap.D10:.2f}; consumption {cp.QU1:.0f}% and {cp.QU5:.0f}% of income")
        print(
            f"   a 6%->0% cut on all food and drink: {100 * CUT_6_TO_0 * ap.QU1 / 100:.2f}% of disposable income for the poorest fifth, {100 * CUT_6_TO_0 * ap.QU5 / 100:.2f}% for the richest"
        )

    section("10. Who is in the richest fifth: EU-SILC equivalised disposable income (ilc_di01, top cut-offs)")
    inc = income[(income.unit == "EUR") & income.quant_inc.isin(vc.DECILE_CUTS)]
    year = int(inc[inc.geo == "PT"].time.max())
    cuts = inc[inc.time == year].pivot_table(index="geo", columns="quant_inc", values="value")[vc.DECILE_CUTS]
    cuts = cuts.loc[[g for g in [*vc.EU27, "EU27_2020"] if g in cuts.index and cuts.loc[g].notna().all()]]
    print(f"survey year {year} (incomes of {year - 1}); {len(cuts) - 1} member states")
    households = {
        "single, 1,500 x 12": vc.equivalised(18000, 1, 0),
        "single, 1,500 x 14": vc.equivalised(21000, 1, 0),
        "couple, 2 x 1,500 x 14": vc.equivalised(42000, 2, 0),
        "couple, 2 x 1,500 x 14, two children under 14": vc.equivalised(42000, 2, 2),
    }
    pos = pd.DataFrame({k: [vc.percentile_from_cuts(v, cuts.loc[g].to_numpy()) for g in cuts.index] for k, v in households.items()}, index=cuts.index)
    pos["top_fifth_from_eur_year"] = cuts["D8"]
    pos["top_fifth_from_eur_month_12"] = cuts["D8"] / 12
    pos["top_fifth_from_eur_month_14"] = cuts["D8"] / 14
    pos["median_eur_year"] = cuts["D5"]
    pos.to_csv(RESULTS / "who_is_rich.csv")
    for g in ("PT", "ES", "PL", "DE", "SE", "EU27_2020"):
        if g in pos.index:
            r = pos.loc[g]
            cells = "; ".join(f"{k}: {'90+' if np.isnan(r[k]) else f'{r[k]:.0f}'}" for k in households)
            print(
                f"{g}: top fifth starts at {r.top_fifth_from_eur_year:,.0f} a year ({r.top_fifth_from_eur_month_12:,.0f} x 12, {r.top_fifth_from_eur_month_14:,.0f} x 14); median {r.median_eur_year:,.0f}; percentile — {cells}"
            )
    ms = pos.drop(index="EU27_2020", errors="ignore")
    single = ms["single, 1,500 x 12"]
    print(
        f"single person with 1,500 a month (x12) is in the richest fifth (>= P80) in {((single >= 80) | single.isna()).sum()} of {len(ms)} member states: {', '.join(single.index[(single >= 80) | single.isna()])}"
    )
    print(f"... below the median in {(single < 50).sum()} of {len(ms)}")
    couple = ms["couple, 2 x 1,500 x 14"]
    print(f"childless couple, each 1,500 x 14, in the richest fifth in {((couple >= 80) | couple.isna()).sum()} of {len(ms)}")


if __name__ == "__main__":
    main()
