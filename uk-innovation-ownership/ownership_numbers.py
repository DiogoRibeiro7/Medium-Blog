"""Every number in the article, printed in order. Run after fetch_data.py:

    python ownership_numbers.py > output.txt

Sections: A growth of ICT firms, B venture capital, C domestic pension pools, D who owns the
inventions and the ICT jobs, E the cross-country model with the UK held out, F robustness.
No random number is drawn anywhere, so the output is identical on every run.
"""

import warnings

import numpy as np
import pandas as pd

import ownership_core as oc

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 160)

MAIN = dict(office="PCT", first=2008, last=2021, min_inventions=1000)
OUTCOMES = {"foreign_share": "owned abroad", "us_share": "owned in the US", "other_share": "owned abroad, not US"}
ALPHA = 0.1


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def header(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def eu_median(series):
    s = series[series.index.isin(oc.EU27)].dropna()
    return s.median(), len(s)


def section_growth():
    header("A. Growth: medium- and high-growth enterprises (10%+ a year), % of firms with 10+ employees")
    hg = oc.load_high_growth()
    hg = hg[(hg["year"] >= 2012) & (hg["year"] <= 2018)]
    names = {"J": "ICT (J)", "J62": "programming (J62)", "J63": "information services (J63)", "M72": "R&D (M72)"}
    for ind, label in names.items():
        m = hg[hg["industry"] == ind].groupby("iso3")["rate"].mean()
        m = m[m.index.isin(oc.EU27 + ["GBR"])].sort_values(ascending=False)
        med, n = eu_median(m)
        rank = int(m.rank(ascending=False)["GBR"])
        print(f"{label:28s} UK {m['GBR']:5.2f}  EU median {med:5.2f} ({n} countries)  UK rank {rank} of {len(m)}")
        if ind == "J":
            print("   " + "  ".join(f"{k} {v:.2f}" for k, v in m.items()))
    uk_years = hg[(hg["iso3"] == "GBR") & (hg["industry"] == "J")]["year"]
    print(f"UK years in the window: {uk_years.min()}-{uk_years.max()} ({uk_years.nunique()})")


def section_vc():
    header("B. Venture capital invested in companies located in each country, % of GDP")
    vc = oc.load_vc()
    vc["later_share"] = vc["later"] / vc["total"]
    for first, last in [(2007, 2014), (2015, 2019), (2020, 2025)]:
        w = vc[(vc["year"] >= first) & (vc["year"] <= last)].groupby("iso3")[["total", "later_share"]].mean()
        med_t, n = eu_median(w["total"])
        med_l, _ = eu_median(w["later_share"])
        uk = w.loc["GBR"]
        print(
            f"{first}-{last}: UK {uk['total']:.3f}% of GDP (later stage {pct(uk['later_share'])}); "
            f"EU median {med_t:.3f}% ({pct(med_l)}), {n} countries; UK/EU median {uk['total'] / med_t:.1f}x; "
            f"US {w.loc['USA', 'total']:.3f}%"
        )
        if first == 2020:
            big = w.loc[[c for c in oc.LARGE_EU if c in w.index], "total"]
            print("   large EU: " + "  ".join(f"{k} {v:.3f}%" for k, v in big.items()))


def section_pensions():
    header("C. Domestic pools: asset-backed pension investment, % of GDP")
    pen = oc.load_pensions()
    w = pen[(pen["year"] >= 2015) & (pen["year"] <= 2024)].groupby("iso3")["assets_gdp"].mean()
    med, n = eu_median(w)
    print(f"mean 2015-2024: UK {w['GBR']:.1f}%  EU median {med:.1f}% ({n} countries)")
    show = ["GBR", "DNK", "NLD", "SWE", "FIN", "DEU", "FRA", "ITA", "ESP", "CHE", "CAN", "AUS", "USA"]
    print("   " + "  ".join(f"{k} {w[k]:.1f}" for k in show if k in w.index))
    latest = pen.dropna(subset=["assets_gdp"]).sort_values("year").groupby("iso3").tail(1).set_index("iso3")
    print(f"latest UK value: {latest.loc['GBR', 'assets_gdp']:.1f}% in {int(latest.loc['GBR', 'year'])}")
    pe = pen.dropna(subset=["pe_share"])
    pe = pe[(pe["year"] >= 2019) & (pe["year"] <= 2025) & pe["iso3"].isin(oc.MODEL_COUNTRIES)]
    pe = pe.groupby("iso3")["pe_share"].agg(["mean", "count"])
    pe = pe.sort_values("mean", ascending=False)
    print("private-equity funds, % of pension-fund assets, mean 2019-2025 (model countries reporting):")
    print("   " + "  ".join(f"{k} {r['mean']:.1f}" for k, r in pe.iterrows()))
    pe_gdp = {k: pe.loc[k, "mean"] / 100 * w[k] for k in ["GBR", "NLD", "DEU", "CHE"] if k in pe.index and k in w}
    print(
        "   private equity held by pension funds, % of GDP (share x assets): "
        + "  ".join(f"{k} {v:.2f}" for k, v in pe_gdp.items())
    )


def section_ownership():
    header("D. Who owns the inventions: PCT applications with an inventor in the country, by owner")
    panel = oc.load_patents("PCT")
    s = oc.pooled_shares(panel, 2008, 2021, countries=oc.MODEL_COUNTRIES + ["USA"]).set_index("iso3")
    med_f, n = eu_median(s["foreign_share"])
    med_u, _ = eu_median(s["us_share"])
    print(
        f"pooled 2008-2021: UK owned abroad {pct(s.loc['GBR', 'foreign_share'])}, by US applicants "
        f"{pct(s.loc['GBR', 'us_share'])}; EU median {pct(med_f)} and {pct(med_u)} ({n} countries)"
    )
    print(f"UK inventions a year, mean: {s.loc['GBR', 'inventions'] / s.loc['GBR', 'years']:.0f}")
    for k in oc.LARGE_EU + ["CAN", "IRL", "CHE", "USA"]:
        r = s.loc[k]
        print(
            f"   {k}: abroad {pct(r['foreign_share'])}  US {pct(r['us_share'])}  "
            f"inventions/yr {r['inventions'] / r['years']:.0f}"
        )
    big = s[s["inventions"] / s["years"] >= 3000].sort_values("foreign_share", ascending=False)
    print(
        "countries with 3,000+ inventions a year, by share owned abroad: "
        + "  ".join(f"{k} {pct(v)}" for k, v in big["foreign_share"].items())
    )

    print("\nfive-year windows, share owned abroad (US-owned in brackets):")
    keep = ["GBR", "DEU", "FRA", "ITA", "NLD", "SWE"]
    rows = []
    for first in range(1980, 2021, 5):
        last = min(first + 4, 2021)
        w = oc.pooled_shares(panel, first, last, countries=keep).set_index("iso3")
        rows.append((f"{first}-{last}", w))
    print("   window      " + "  ".join(f"{k:>14s}" for k in keep))
    for label, w in rows:
        print(
            f"   {label:10s}  "
            + "  ".join(f"{pct(w.loc[k, 'foreign_share']):>6s} ({pct(w.loc[k, 'us_share']):>5s})" for k in keep)
        )

    print("\nowners of UK and German inventions, pooled 2008-2021, share of all inventions:")
    own = oc.load_owners()
    own = own[(own["year"] >= 2008) & (own["year"] <= 2021)]
    for iso in ["GBR", "DEU"]:
        o = own[own["iso3"] == iso].groupby("owner")["n"].sum()
        total = o["W"]
        top = o.drop(index=["W", "WXD", "EU27_2020"], errors="ignore").sort_values(ascending=False).head(10)
        print(f"   {iso}: " + "  ".join(f"{k} {pct(v / total)}" for k, v in top.items()))

    header("D2. Who employs the ICT workers: persons employed in foreign- and US-controlled enterprises")
    f = oc.load_fats()
    f = f[(f["year"] >= 2014) & (f["year"] <= 2019)]
    for ind in ["J", "J62", "J63"]:
        g = f[f["industry"] == ind].dropna(subset=["all", "foreign"]).groupby("iso3")[["all", "foreign", "us"]].sum()
        g["foreign_share"] = g["foreign"] / g["all"]
        g["us_share"] = g["us"] / g["all"]
        med_f, n = eu_median(g["foreign_share"])
        med_u, _ = eu_median(g["us_share"])
        years = f[(f["industry"] == ind) & (f["iso3"] == "GBR")].dropna(subset=["foreign"])["year"]
        print(
            f"{ind}: UK foreign-controlled {pct(g.loc['GBR', 'foreign_share'])}, US-controlled "
            f"{pct(g.loc['GBR', 'us_share'])} (UK years {years.min()}-{years.max()}); EU median "
            f"{pct(med_f)} and {pct(med_u)} ({n} countries)"
        )
        print(
            "   "
            + "  ".join(
                f"{k} {pct(g.loc[k, 'foreign_share'])}/{pct(g.loc[k, 'us_share'])}" for k in oc.LARGE_EU if k in g.index
            )
        )


def model_frame(office, first, last, min_inventions):
    panel = oc.load_patents(office)
    s = oc.pooled_shares(panel, first, last, countries=oc.MODEL_COUNTRIES, min_inventions=min_inventions)
    return oc.add_covariates(s, oc.load_pensions(), first, last)


def section_model():
    header("E. Cross-country model, UK held out (PCT, priority years 2008-2021, 1,000+ inventions)")
    frame = model_frame(**MAIN)
    print(f"countries: {len(frame)}  ({', '.join(frame['iso3'])})")
    train = frame[frame["iso3"] != "GBR"]
    for outcome, label in OUTCOMES.items():
        res = oc.fit_fractional_logit(train, outcome, oc.COVARIATES)
        coefs = "  ".join(f"{k} {res.params[k]:+.3f} (se {res.bse[k]:.3f})" for k in ["const"] + oc.COVARIATES)
        print(f"\n{label}: fractional logit without the UK, HC1 se")
        print(f"   {coefs}")
        actual, pred = oc.holdout(frame, outcome, oc.COVARIATES, "GBR")
        lo, hi = oc.jackknife_plus(frame, outcome, oc.COVARIATES, "GBR", alpha=ALPHA)
        print(
            f"   UK actual {pct(actual)}  predicted {pct(pred)}  gap {100 * (actual - pred):+.1f} points "
            f"(ratio {actual / pred:.2f})  jackknife+ interval (nominal 90%, guaranteed 80%) "
            f"{pct(lo)}-{pct(hi)}"
        )
        loo = oc.leave_one_out(frame, outcome, oc.COVARIATES)
        uk = loo[loo["iso3"] == "GBR"].iloc[0]
        print(
            f"   leave-one-out: UK gap ranks {uk['rank']} of {len(loo)} on the logit scale "
            f"(rank/n = {uk['rank'] / len(loo):.3f}); largest, points (logit): "
            + "  ".join(f"{r.iso3} {100 * r.gap:+.1f} ({r.logit_gap:+.2f})" for r in loo.head(6).itertuples())
        )
        print(
            f"   median absolute leave-one-out gap: {100 * loo['gap'].abs().median():.1f} points; "
            f"UK {100 * abs(uk['gap']):.1f}"
        )

    print("\nsize alone (log inventions a year):")
    for outcome, label in OUTCOMES.items():
        actual, pred = oc.holdout(frame, outcome, ["log_inventions"], "GBR")
        print(f"   {label}: UK actual {pct(actual)}  predicted {pct(pred)}  gap {100 * (actual - pred):+.1f} points")

    header("E2. Do larger domestic pension pools keep inventions at home? (log(1 + pension assets % GDP))")
    with_p = oc.COVARIATES + ["log_pensions"]
    pf = frame.dropna(subset=["log_pensions"])
    print(f"countries with pension data: {len(pf)}")
    for outcome, label in OUTCOMES.items():
        res = oc.fit_fractional_logit(pf[pf["iso3"] != "GBR"], outcome, with_p)
        actual, pred = oc.holdout(pf, outcome, with_p, "GBR")
        b, se = res.params["log_pensions"], res.bse["log_pensions"]
        # Effect of moving from the large-EU pension level (about 10% of GDP) to the UK's (about 90%).
        shift = b * (np.log1p(90) - np.log1p(10))
        print(
            f"   {label}: log_pensions {b:+.3f} (se {se:.3f}, z {b / se:+.2f}); 10% -> 90% of GDP moves the "
            f"log-odds by {shift:+.2f}; UK predicted {pct(pred)} (actual {pct(actual)})"
        )

    panel = oc.load_patents("PCT")
    pfr = oc.panel_frame(panel, oc.load_pensions(), 2008, 2021, oc.MODEL_COUNTRIES, min_inventions=1000)
    pfr = pfr[pfr["iso3"] != "GBR"].dropna(subset=["log_pensions"])
    print(
        f"\ncountry-year panel without the UK: {len(pfr)} rows, {pfr['iso3'].nunique()} countries; "
        "standard errors clustered by country"
    )
    for outcome, label in OUTCOMES.items():
        res = oc.fit_fractional_logit(pfr, outcome, with_p, groups="iso3")
        print(f"   {label}: " + "  ".join(f"{k} {res.params[k]:+.3f} (se {res.bse[k]:.3f})" for k in with_p))


def section_robustness():
    header("F. Robustness: UK gap (points), jackknife+ interval and leave-one-out rank (#), by specification")
    specs = []
    for office in ["PCT", "EPO"]:
        for first, last in [(1995, 2007), (2008, 2021)]:
            for min_inv in [1000, 0]:
                specs.append((office, first, last, min_inv, None))
    specs.append(("PCT", 2008, 2021, 1000, ["CAN", "IRL"]))
    specs.append(("PCT", 2008, 2021, 1000, oc.CEE))
    for office, first, last, min_inv, drop in specs:
        frame = model_frame(office, first, last, min_inv)
        if drop:
            frame = frame[~frame["iso3"].isin(drop)].reset_index(drop=True)
        covs = [c for c in oc.COVARIATES if frame[c].nunique() > 1]
        cells = []
        for outcome in OUTCOMES:
            actual, pred = oc.holdout(frame, outcome, covs, "GBR")
            lo, hi = oc.jackknife_plus(frame, outcome, covs, "GBR", alpha=ALPHA)
            loo = oc.leave_one_out(frame, outcome, covs)
            rank = int(loo.loc[loo["iso3"] == "GBR", "rank"].iloc[0])
            cells.append(f"{100 * (actual - pred):+5.1f} [{pct(lo, 0)}-{pct(hi, 0)}] #{rank}")
        tag = f"{office} {first}-{last} {'1000+' if min_inv else 'all'}"
        if drop:
            tag += " drop " + ("CEE" if drop == oc.CEE else "+".join(drop))
        print(f"{tag:34s} n={len(frame):2d}  abroad {cells[0]}  US {cells[1]}  not US {cells[2]}")


def main():
    print("UK innovation ownership: numbers for the article (data retrieved 24 November 2026)")
    section_growth()
    section_vc()
    section_pensions()
    section_ownership()
    section_model()
    section_robustness()


if __name__ == "__main__":
    main()
