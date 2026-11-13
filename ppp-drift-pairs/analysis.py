"""Test whether the price drift arises where the European PPP comparison meets the non-European one.

    python analysis.py

Writes results/analysis_summary.txt (every number cited in the article) and the CSV files behind the figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import pairs as pr

RES = Path(__file__).parent / "results"
RES.mkdir(exist_ok=True)
LINES: list[str] = []
A, B = 2005, 2019


def say(label: str, value) -> None:
    LINES.append(f"{label}: {value}")
    print(f"{label}: {value}")


def section(title: str) -> None:
    LINES.append("")
    LINES.append(f"== {title} ==")
    print(f"\n== {title} ==")


def main() -> None:
    d = pr.load()
    cds = {agg: pr.drift_vs(d, agg) for agg in pr.AGGREGATES}

    # ------------------------------------------------------------------ coverage and transitivity
    section("Economies, groups and transitivity")
    scores = {agg: pr.programme_scores(cds[agg], A, B) for agg in pr.AGGREGATES}
    for agg, s in scores.items():
        say(f"{agg}: economies with data in {A} and {B} (Eurostat group, OECD group)",
            f"{len(s)} ({(s.group == 'Eurostat').sum()}, {(s.group == 'OECD').sum()})")
    s = scores["HFCE"]
    say("HFCE economies, Eurostat group", " ".join(s[s.group == "Eurostat"].index))
    say("HFCE economies, OECD group", " ".join(s[s.group == "OECD"].index))
    missing = [c for c in pr.OECD_GROUP if c not in s.index]
    say("OECD-group members without household-consumption PPPs for both years", " ".join(missing) or "none")
    # transitivity: a pair computed directly equals the difference of the two scores against the US
    worst = 0.0
    for i, j in (("DEU", "JPN"), ("FRA", "KOR"), ("POL", "CAN"), ("ITA", "ESP"), ("SWE", "NOR")):
        direct = pr.pair_drift(d, "HFCE", i, j)
        via = cds["HFCE"][i] - cds["HFCE"][j]
        worst = max(worst, float((direct - via).abs().max()))
    say("Largest gap between a pair computed directly and through the US (log points x 100)", f"{worst:.1e}")
    n = len(s)
    say("HFCE: pairs; independent numbers behind them", f"{n * (n - 1) // 2}; {n - 1}")

    # ------------------------------------------------------------------ pairs
    section(f"Pairs, drift {A}-{B} (log points x 100)")
    allpairs = []
    for agg in pr.AGGREGATES:
        pt = pr.pairs_table(scores[agg])
        pt["measure"] = agg
        allpairs.append(pt)
        for kind in ("within Eurostat", "within OECD", "across"):
            x = pt[pt.kind == kind].drift.abs()
            say(f"{agg} {kind}: pairs, mean |drift|, median |drift|, share above 10 points",
                f"{len(x)}, {x.mean():.1f}, {x.median():.1f}, {(x > 10).mean():.2f}")
        a, w = pt[pt.kind == "across"].drift.abs().mean(), pt[pt.kind != "across"].drift.abs().mean()
        say(f"{agg}: mean |drift| across groups / within groups", f"{a / w:.2f}")
    pd.concat(allpairs).to_csv(RES / "pairs_2005_2019.csv", index=False, float_format="%.3f")

    # ------------------------------------------------------------------ block model
    section("Block model: country drift on Eurostat-group membership")
    pli = {agg: np.log(100 * (d["PPP"][agg] / d["EXC"]).loc[A]) for agg in pr.AGGREGATES}
    for agg in pr.AGGREGATES:
        sc = scores[agg]
        m = pr.block_model(sc)
        say(f"{agg}: gap Eurostat minus OECD group (points); R2; n",
            f"{m['beta'][1]:.1f}; {m['r2']:.2f}; {m['n']}")
        sd = sc.groupby("group").drift.std()
        mean = sc.groupby("group").drift.mean()
        say(f"{agg}: group mean (Eurostat, OECD); within-group SD (Eurostat, OECD)",
            f"{mean['Eurostat']:.1f}, {mean['OECD']:.1f}; {sd['Eurostat']:.1f}, {sd['OECD']:.1f}")
        say(f"{agg}: permutation p-value of the gap (10,000 relabellings)", f"{pr.permutation_p(sc):.4f}")
        mc = pr.block_model(sc, control=pli[agg])
        say(f"{agg}: with ln price level 2005 as control: gap; slope per log point of price level; R2",
            f"{mc['beta'][1]:.1f}; {mc['beta'][2]:.1f}; {mc['r2']:.2f}")
        e = sc[sc.group == "Eurostat"]
        r = e.drift.corr(pli[agg].reindex(e.index))
        say(f"{agg}: correlation of drift with ln price level 2005 inside the Eurostat group (n, r)", f"{len(e)}, {r:.2f}")
    sc = scores["HFCE"].copy()
    sc["ln_pli_2005"] = pli["HFCE"].reindex(sc.index)
    sc["pli_2005"] = np.exp(sc["ln_pli_2005"])
    sc.to_csv(RES / "hfce_scores.csv", float_format="%.4f")
    e = sc[sc.group == "Eurostat"]
    lo, hi = e.pli_2005.nsmallest(8).index, e.pli_2005.nlargest(8).index
    say("HFCE, Eurostat group: mean drift of the 8 cheapest / 8 dearest economies in 2005 (points)",
        f"{e.loc[lo, 'drift'].mean():.1f} / {e.loc[hi, 'drift'].mean():.1f}")
    say("Eight cheapest; eight dearest (price level of household consumption, US = 100, 2005)",
        ", ".join(f"{c} {e.loc[c, 'pli_2005']:.0f}" for c in lo) + "; " +
        ", ".join(f"{c} {e.loc[c, 'pli_2005']:.0f}" for c in hi))

    # ------------------------------------------------------------------ the gap is the same in all three aggregates
    section("Same gap in three aggregates")
    eg = [c for c in scores["GDP"].index if scores["GDP"].loc[c, "group"] == "Eurostat"]
    share = (d["V"]["HFCE"] / d["V"]["GDP"]).loc[A:B, eg].mean().mean()
    gaps = {agg: pr.block_model(scores[agg])["beta"][1] for agg in pr.AGGREGATES}
    say("Household consumption as share of GDP, Eurostat group, mean 2005-2019", f"{share:.2f}")
    say("GDP gap expected if the gap came only from consumer deflators; observed GDP gap (points)",
        f"{share * gaps['HFCE']:.1f}; {gaps['GDP']:.1f}")
    common = sorted(set(scores["HFCE"].index) & set(scores["GDP"].index))
    r = scores["HFCE"].loc[common, "drift"].corr(scores["GDP"].loc[common, "drift"])
    say("Correlation across economies between HFCE and GDP drift 2005-2019 (n, r)", f"{len(common)}, {r:.2f}")

    # ------------------------------------------------------------------ construction signature
    section("Annual steps against the US: how the PPP series were built")
    steps = cds["HFCE"].diff()
    oe = [c for c in pr.OECD_GROUP if c != "USA" and c in steps and steps[c].loc[1996:2015].notna().all()]
    eu = [c for c in pr.EUROSTAT_GROUP if c in steps and steps[c].loc[1996:2015].notna().all()]
    quiet = pd.DataFrame({"OECD group": (steps[oe].abs() < 0.3).mean(axis=1),
                          "Eurostat group": (steps[eu].abs() < 0.3).mean(axis=1)}).loc[1996:pr.LAST]
    quiet.to_csv(RES / "quiet_share.csv", float_format="%.3f")
    st_long = steps.loc[1996:pr.LAST, oe + eu].stack().rename("step").reset_index()
    st_long.columns = ["year", "economy", "step"]
    st_long["group"] = st_long.economy.map(pr.group_of)
    st_long.to_csv(RES / "annual_steps_hfce.csv", index=False, float_format="%.3f")
    say("OECD-group economies with HFCE PPPs for every year 1995-2015 (excluding the US)", f"{len(oe)}: " + " ".join(oe))
    say("Eurostat-group economies with the same coverage", f"{len(eu)}")
    q = quiet.loc[1996:2012]
    say("Years 1996-2012 in which at least two-thirds of OECD-group steps were below 0.3 points",
        ", ".join(str(y) for y in q.index[q["OECD group"] >= 2 / 3]))
    say("Share of steps below 0.3 points in those years: OECD group; Eurostat group",
        f"{q.loc[q['OECD group'] >= 2 / 3, 'OECD group'].mean():.2f}; {q.loc[q['OECD group'] >= 2 / 3, 'Eurostat group'].mean():.2f}")
    say("Share of steps below 0.3 points, all years 1996-2012: OECD group; Eurostat group",
        f"{(steps.loc[1996:2012, oe].abs() < 0.3).values.mean():.2f}; {(steps.loc[1996:2012, eu].abs() < 0.3).values.mean():.2f}")
    say("Share of steps below 0.3 points, 2016-2024: OECD group; Eurostat group",
        f"{(steps.loc[2016:pr.LAST, oe].abs() < 0.3).values.mean():.2f}; {(steps.loc[2016:pr.LAST, eu].abs() < 0.3).values.mean():.2f}")
    # the step in the year after a benchmark and the step in the next benchmark year are equal
    pairs_bb = [(2000, 2002), (2003, 2005), (2006, 2008), (2009, 2011)]
    diffs, sizes = [], []
    for y1, y2 in pairs_bb:
        diffs += list((steps.loc[y1, oe] - steps.loc[y2, oe]).abs())
        sizes += list((steps.loc[y1, oe].abs() + steps.loc[y2, oe].abs()) / 2)
    say("OECD group, steps in the year after a benchmark vs in the next benchmark year (2000/02, 2003/05, 2006/08, "
        "2009/11): median absolute difference; median size (points)", f"{np.median(diffs):.2f}; {np.median(sizes):.2f}")
    say("Same comparison, Eurostat group (median absolute difference; median size)",
        f"{np.median([abs(steps.loc[a, c] - steps.loc[b, c]) for a, b in pairs_bb for c in eu]):.2f}; "
        f"{np.median([(abs(steps.loc[a, c]) + abs(steps.loc[b, c])) / 2 for a, b in pairs_bb for c in eu]):.2f}")
    say("Japan HFCE steps 1999-2008 (points)", ", ".join(f"{y}: {steps.loc[y, 'JPN']:.1f}" for y in range(2000, 2009)))

    # ------------------------------------------------------------------ the gap over time
    section("The gap between the groups over time (group means of drift against the US, points, 2020 = 0)")
    gp = {}
    for agg in pr.AGGREGATES:
        cd = cds[agg]
        e = [c for c in pr.EUROSTAT_GROUP if c in cd and cd[c].loc[1995:pr.LAST].notna().all()]
        o = [c for c in pr.OECD_GROUP if c in cd and cd[c].loc[1995:pr.LAST].notna().all()]
        gp[agg] = cd[e].mean(axis=1) - cd[o].mean(axis=1)
        say(f"{agg}: economies with every year 1995-2024 (Eurostat, OECD)", f"{len(e)}, {len(o)}")
    gp = pd.DataFrame(gp).loc[1995:pr.LAST]
    gp.to_csv(RES / "group_gap_paths.csv", float_format="%.3f")
    for agg in pr.AGGREGATES:
        g = gp[agg]
        say(f"{agg} gap: 1995, 2005, 2019, 2024; change 1995-2005, 2005-2019, 2019-2024",
            f"{g[1995]:.1f}, {g[2005]:.1f}, {g[2019]:.1f}, {g[2024]:.1f}; {g[2005] - g[1995]:.1f}, "
            f"{g[2019] - g[2005]:.1f}, {g[2024] - g[2019]:.1f}")
        st = g.diff().loc[2006:2019]
        bench = [y for y in pr.BENCHMARKS if 2006 <= y <= 2019]
        say(f"{agg} gap steps 2006-2019: in benchmark years {bench}; in other years (points; share)",
            f"{st.loc[bench].sum():.1f}; {st.drop(bench).sum():.1f} ({st.loc[bench].sum() / st.sum():.2f})")
        top = st.sort_values(ascending=False).head(4)
        say(f"{agg} gap: largest steps 2006-2019", "; ".join(f"{y}: {v:.1f}" for y, v in top.items()))
        say(f"{agg} gap per year: 2005-2015 (triennial OECD calculation); 2015-2019; 2019-2024 (points a year)",
            f"{(g[2015] - g[2005]) / 10:.2f}; {(g[2019] - g[2015]) / 4:.2f}; {(g[2024] - g[2019]) / 5:.2f}")

    # ------------------------------------------------------------------ the United Kingdom
    section("The United Kingdom")
    for agg in ("HFCE", "GDP"):
        cd = cds[agg]
        e = [c for c in pr.EUROSTAT_GROUP if c != "GBR" and c in cd and cd[c].loc[1995:pr.LAST].notna().all()]
        o = [c for c in pr.OECD_GROUP if c in cd and cd[c].loc[1995:pr.LAST].notna().all()]
        ue, uo = cd["GBR"] - cd[e].mean(axis=1), cd["GBR"] - cd[o].mean(axis=1)
        for a, b in ((2005, 2019), (2019, 2024)):
            say(f"{agg} UK drift {a}-{b}: against the Eurostat-group mean; against the OECD-group mean (points)",
                f"{ue[b] - ue[a]:.1f}; {uo[b] - uo[a]:.1f}")
    h = scores["HFCE"]
    say("HFCE drift 2005-2019 against the US: UK; Eurostat-group mean; OECD-group mean",
        f"{h.loc['GBR', 'drift']:.1f}; {h[h.group == 'Eurostat'].drift.mean():.1f}; {h[h.group == 'OECD'].drift.mean():.1f}")

    # ------------------------------------------------------------------ Eurostat cross-check
    section("Cross-check: Eurostat's own PPPs give the same European pairs")
    es = pd.read_csv(pr.DATA / "eurostat_ppp.csv")
    es = es[(es.na_item == "PPP_EU27_2020") & (es.ppp_cat == "E011")]
    es["time"] = es.time.astype(int)
    geo = {"DE": "DEU", "FR": "FRA", "PL": "POL", "SE": "SWE", "US": "USA", "JP": "JPN", "IT": "ITA"}
    p = es[es.geo.isin(geo)].pivot_table(index="time", columns="geo", values="value").rename(columns=geo)
    o = d["PPP"]["HFCE"]
    worst = max(abs(np.log((p[c] / p["DEU"]).loc[B] / (p[c] / p["DEU"]).loc[A]) -
                    np.log((o[c] / o["DEU"]).loc[B] / (o[c] / o["DEU"]).loc[A])) for c in ["FRA", "POL", "SWE", "USA", "JPN", "ITA"])
    say("Largest difference 2005-2019 in PPP change against Germany, Eurostat vs OECD (log points x 100)", f"{100 * worst:.2f}")

    (RES / "analysis_summary.txt").write_text("\n".join(LINES).lstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
