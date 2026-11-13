"""Test whether the price-level slope of the PPP drift inside Europe sits in services or in goods.

    python analysis.py

Writes ``results/analysis_summary.txt`` (every number cited in the article) and the CSV tables the
figures read. Drifts are in log points x 100 over 2005-2019 against the EU27 unless stated; slopes are
points of drift per log point of the 2005 household-consumption price level (EU27 = 100).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import catdrift as C

OUT = Path("results")


def f1(x: float) -> str:
    return "nan" if pd.isna(x) else f"{x:.1f}"


def f2(x: float) -> str:
    return "nan" if pd.isna(x) else f"{x:.2f}"


def pct(x: float) -> str:
    return "nan" if pd.isna(x) else f"{100 * x:.0f}%"


def fit_line(name: str, o: dict) -> str:
    return (f"{name}: slope {f1(o['slope'])} (HC1 se {f1(o['se'])}, t {f1(o['t'])}), r {f2(o['r'])}, n {o['n']}")


def kind_sums(T: pd.DataFrame) -> dict[str, pd.Series]:
    kinds = {}
    for k in ("goods", "services", "mixed", "public"):
        cols = [co for _, co, _, kk in C.DIVISIONS if kk == k]
        kinds[k] = T[cols].sum(axis=1)
    kinds["aggregation"] = T[["A_deflator", "B_ppp", "C_reference"]].sum(axis=1)
    return kinds


def main() -> None:
    OUT.mkdir(exist_ok=True)
    d = C.load("data")
    x = C.price_level(d)
    L: list[str] = ["# PPP drift by category inside Europe: analysis summary",
                    f"# drift over {C.START}-{C.END} against the EU27, log points x 100; regressor ln PLI 2005 (EU27=100)",
                    ""]

    # ------------------------------------------------------------------ sample and replication
    L.append("## Sample and the household-consumption slope")
    L.append(f"Economies: {len(C.EUROPE)} ({', '.join(C.EUROPE)})")
    pli = np.exp(x)
    L.append(f"2005 price level (EU27=100): min {pli.min():.0f} ({pli.idxmin()}), max {pli.max():.0f} ({pli.idxmax()}); "
             f"range of ln PLI {x.max() - x.min():.2f}")
    h = C.drift_parts(d, "E011", "TOTAL")
    oh = C.ols(h.drift, x)
    L.append(fit_line("Household consumption drift", oh))
    L.append(f"Mean drift {f1(h.drift.mean())}, sd {f1(h.drift.std())}")
    order = pli.sort_values()
    cheap, dear = order.index[:8], order.index[-8:]
    L.append(f"Eight cheapest in 2005 ({', '.join(f'{g} {pli[g]:.0f}' for g in cheap)}): mean drift {f1(h.drift[cheap].mean())}")
    L.append(f"Eight dearest in 2005 ({', '.join(f'{g} {pli[g]:.0f}' for g in dear)}): mean drift {f1(h.drift[dear].mean())}")
    L.append(f"Parts of the household slope: deflator relative to EU27 {f1(C.ols(h.deflator, x)['slope'])}, "
             f"PPP {f1(C.ols(h.ppp, x)['slope'])}")
    xr = pd.Series({g: 100 * (C.log_change(C.ppp(d, g, "E011"), C.START, C.END)
                              - C.log_change(C.ppp(d, g, "E011", "PLI_EU27_2020"), C.START, C.END)) for g in C.EUROPE})
    L.append(f"Exchange-rate change against the euro, slope {f1(C.ols(xr, x)['slope'])}; price-level convergence "
             f"measured by deflators {f1(C.ols(h.deflator - xr, x)['slope'])}, by PPPs {f1(C.ols(h.ppp - xr, x)['slope'])}")
    L.append("Reference from follow-up Part 4 (OECD data, against the US, 35 Eurostat-group economies): slope -7.5, r -0.49")
    L.append("")

    # ------------------------------------------------------------------ goods and services
    L.append("## Eurostat's goods and services split")
    gs_rows = []
    for cat, (label, _) in C.DURABILITY.items():
        p = C.drift_parts(d, cat)
        o = C.ols(p.drift, x)
        L.append(fit_line(label, o) + f"; parts: deflator {f1(C.ols(p.deflator, x)['slope'])}, PPP {f1(C.ols(p.ppp, x)['slope'])}")
        gs_rows.append({"cat": cat, "label": label, **o})
    gs = pd.DataFrame(gs_rows).set_index("cat")
    L.append(f"Services slope / goods slope: {gs.loc['P0201', 'slope'] / gs.loc['P0101', 'slope']:.2f}")
    L.append("")

    # ------------------------------------------------------------------ divisions and groups
    L.append("## COICOP divisions and groups")
    rows = []
    for level, items in (("division", C.DIVISIONS), ("group", C.GROUPS)):
        for cat, co, label, kind in items:
            p = C.drift_parts(d, cat, co)
            o = C.ols(p.drift, x)
            pli_c = pd.Series({g: 100 * C.log_change(C.ppp(d, g, cat, "PLI_EU27_2020"), C.START, C.END) for g in C.EUROPE})
            rows.append({"level": level, "cat": cat, "coicop": co, "label": label, "kind": kind,
                         "slope": o["slope"], "se": o["se"], "t": o["t"], "r": o["r"], "n": o["n"],
                         "mean": p.drift.mean(), "sd": p.drift.std(),
                         "deflator_slope": C.ols(p.deflator, x)["slope"], "ppp_slope": C.ols(p.ppp, x)["slope"],
                         "conv_ppp": C.ols(pli_c, x)["slope"], "conv_deflator": C.ols(p.deflator - xr, x)["slope"]})
    hx = C.housing_ex_energy(d)
    o = C.ols(hx.drift, x)
    rows.append({"level": "derived", "cat": "A0104X", "coicop": "CP04-CP045", "label": "Housing without energy",
                 "kind": "services", "slope": o["slope"], "se": o["se"], "t": o["t"], "r": o["r"], "n": o["n"],
                 "mean": hx.drift.mean(), "sd": hx.drift.std(), "deflator_slope": C.ols(hx.deflator, x)["slope"],
                 "ppp_slope": C.ols(hx.ppp, x)["slope"], "conv_ppp": np.nan, "conv_deflator": np.nan})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "category_slopes.csv", index=False)
    for _, r in tab.iterrows():
        L.append(f"{r.label} [{r.kind}]: slope {f1(r.slope)} (se {f1(r.se)}, t {f1(r.t)}), r {f2(r.r)}, n {r.n}; "
                 f"mean {f1(r['mean'])}, sd {f1(r.sd)}; deflator {f1(r.deflator_slope)}, PPP {f1(r.ppp_slope)}")
    L.append(f"Housing without energy: average energy share {pct(hx.energy_share.mean())}")
    for co, label in (("CP041", "actual rents"), ("CP042", "imputed rents")):
        ref = C.log_change(C.deflator(d, C.REF, co), C.START, C.END)
        dp = 100 * (pd.Series({g: C.log_change(C.deflator(d, g, co), C.START, C.END) for g in C.EUROPE}) - ref)
        L.append(f"Deflator of {label} relative to EU27: slope {f1(C.ols(dp, x)['slope'])}, n {C.ols(dp, x)['n']}")
    L.append("")

    # ------------------------------------------------------------------ decomposition
    L.append("## Decomposition of the household-consumption slope")
    T = C.decompose(d)
    shares = T[[c for c in T.columns if c.startswith("share_")]].astype(float)
    T = T[[c for c in T.columns if not c.startswith("share_")]].astype(float)
    err = (T.drift - T.drop(columns="drift").sum(axis=1)).abs().max()
    L.append(f"Identity check, largest absolute error: {err:.1e}")
    total = C.ols(T.drift, x)["slope"]
    contrib = []
    for col in T.columns.drop("drift"):
        o = C.ols(T[col], x)
        label = {co: lab for _, co, lab, _ in C.DIVISIONS}.get(col, col)
        kind = {co: k for _, co, _, k in C.DIVISIONS}.get(col, "aggregation")
        contrib.append({"term": col, "label": label, "kind": kind, "slope": o["slope"], "se": o["se"],
                        "share_of_total": o["slope"] / total, "mean_weight": shares.get(f"share_{col}", pd.Series(np.nan)).mean()})
        L.append(f"{label}: {f2(o['slope'])} (se {f2(o['se'])}), {pct(o['slope'] / total)} of {f2(total)}")
    pd.DataFrame(contrib).to_csv(OUT / "decomposition.csv", index=False)
    for g in ("BG", "PT", "DE", "DK"):
        kk = {k: v[g] for k, v in kind_sums(T).items()}
        L.append(f"Economy {g} (2005 price level {pli[g]:.0f}): drift {f1(T.drift[g])} = goods {f1(kk['goods'])} + services "
                 f"{f1(kk['services'])} (housing {f1(T.CP04[g])}, communication {f1(T.CP08[g])}) + mixed {f1(kk['mixed'])} "
                 f"+ health and education {f1(kk['public'])} + aggregation {f1(kk['aggregation'])} (PPP term {f1(T.B_ppp[g])})")
    ks = kind_sums(T)
    for k, s in ks.items():
        o = C.ols(s, x)
        L.append(f"By kind, {k}: {f2(o['slope'])} (se {f2(o['se'])}), {pct(o['slope'] / total)}")
    L.append(f"Average expenditure share of the service divisions {pct(shares[[f'share_{co}' for _, co, _, k in C.DIVISIONS if k == 'services']].sum(axis=1).mean())}, "
             f"goods divisions {pct(shares[[f'share_{co}' for _, co, _, k in C.DIVISIONS if k == 'goods']].sum(axis=1).mean())}")
    L.append("")

    # ------------------------------------------------------------------ composition
    L.append("## The aggregation term and spending composition")
    K = C.composition_term(d)
    oK = C.ols(K, x)
    L.append(fit_line("Composition term (share shifts x relative price levels)", oK))
    L.append(f"Correlation of the PPP aggregation term with the composition term: {f2(np.corrcoef(T.B_ppp.loc[K.index], K)[0, 1])}; "
             f"slope of the PPP term on the composition term {f2(C.ols(T.B_ppp, K)['slope'])} (se {f2(C.ols(T.B_ppp, K)['se'])})")
    L.append(f"Composition term slope as a share of the PPP aggregation slope: {pct(oK['slope'] / C.ols(T.B_ppp, x)['slope'])}")
    Bs = T.B_ppp.sort_values()
    L.append(f"PPP aggregation term, lowest: {', '.join(f'{g} {f1(v)}' for g, v in Bs.head(2).items())}; highest: "
             f"{', '.join(f'{g} {f1(v)}' for g, v in Bs.tail(2).items())}")
    keep = [g for g in T.index if g not in Bs.index[[0, -1]]]
    o_b = C.ols(T.B_ppp.loc[keep], x.loc[keep])
    L.append(f"PPP aggregation slope without {Bs.index[0]} and {Bs.index[-1]}: {f2(o_b['slope'])} (se {f2(o_b['se'])}, t {f1(o_b['t'])})")
    spread = C.relative_price_spread(d)
    L.append(fit_line("Spread of division price levels within each economy, 2005 (log points)", C.ols(spread, x)))
    L.append(f"Spread: smallest {spread.idxmin()} {f1(spread.min())}, largest {spread.idxmax()} {f1(spread.max())}; "
             f"median of the eight cheapest {f1(spread[cheap].median())}, of the eight dearest {f1(spread[dear].median())}")
    sp_years = {yr: C.relative_price_spread(d, yr) for yr in (2005, 2010, 2019)}
    L.append("Median spread of the eight cheapest of 2005: " + ", ".join(f"{yr} {f1(v[cheap].median())}" for yr, v in sp_years.items())
             + "; of the eight dearest: " + ", ".join(f"{yr} {f1(v[dear].median())}" for yr, v in sp_years.items()))
    sv = shares[[f"share_{co}" for _, co, _, k in C.DIVISIONS if k in ("services", "public")]].sum(axis=1)
    dsv = 100 * pd.Series({g: sum(C.nominal(d, g, co)[C.END] for _, co, _, k in C.DIVISIONS if k in ("services", "public"))
                           / sum(C.nominal(d, g, co)[C.END] for _, co, _, _ in C.DIVISIONS)
                           - sum(C.nominal(d, g, co)[C.START] for _, co, _, k in C.DIVISIONS if k in ("services", "public"))
                           / sum(C.nominal(d, g, co)[C.START] for _, co, _, _ in C.DIVISIONS) for g in C.EUROPE})
    L.append(f"Change in the share of service and public divisions 2005-2019, percentage points: mean {f1(dsv.mean())}, "
             f"slope on ln PLI {f1(C.ols(dsv, x)['slope'])} (se {f1(C.ols(dsv, x)['se'])})")
    pd.DataFrame({"ln_pli": x, "drift": h.drift, "B_ppp": T.B_ppp, "composition": K, "spread": spread,
                  "service_share_change": dsv}).to_csv(OUT / "economies.csv")
    L.append("")

    # ------------------------------------------------------------------ convergence
    L.append("## Convergence and drift across the twelve divisions")
    div = tab[tab.level == "division"].set_index("label")
    avg = (div.conv_ppp + div.conv_deflator) / 2
    L.append(f"Spearman correlation of the drift slope with convergence measured by PPPs {f2(C.spearman(div.slope, div.conv_ppp))}, "
             f"by deflators {f2(C.spearman(div.slope, div.conv_deflator))}, by their average {f2(C.spearman(div.slope, avg))}")
    for lab, r in div.sort_values("conv_deflator").iterrows():
        L.append(f"  {lab}: convergence by deflators {f1(r.conv_deflator)}, by PPPs {f1(r.conv_ppp)}, drift slope {f1(r.slope)}")
    L.append("")

    # ------------------------------------------------------------------ robustness
    L.append("## Robustness (slope of household drift; contributions by kind; aggregate goods and services slopes)")
    variants = {
        "base 2005-2019": (C.START, C.END, C.EUROPE),
        "2005-2022": (2005, 2022, C.EUROPE),
        "2005-2015": (2005, 2015, C.EUROPE),
        "2010-2019": (2010, 2019, C.EUROPE),
        "EU27 only": (C.START, C.END, C.EU27),
        "without Serbia, Albania, Bosnia": (C.START, C.END, tuple(g for g in C.EUROPE if g not in C.BALKANS)),
    }
    rob = []
    for name, (a, b, geos) in variants.items():
        xx = C.price_level(d, a, geos=geos)
        TT = C.decompose(d, a, b, geos)
        TT = TT[[c for c in TT.columns if not c.startswith("share_")]].astype(float)
        kk = kind_sums(TT)
        row = {"variant": name, "n": len(TT), "household": C.ols(TT.drift, xx)["slope"], "household_t": C.ols(TT.drift, xx)["t"]}
        for k, s in kk.items():
            row[k] = C.ols(s, xx)["slope"]
        row["ppp_aggregation"] = C.ols(TT.B_ppp, xx)["slope"]
        for co in ("CP04", "CP11", "CP12"):
            row[co] = C.ols(TT[co], xx)["slope"]
        for cat in ("P0101", "P0201"):
            o = C.ols(C.drift_parts(d, cat, a=a, b=b, geos=geos).drift, xx)
            row[cat], row[f"{cat}_t"] = o["slope"], o["t"]
        rob.append(row)
        L.append(f"{name} (n {row['n']}): household {f1(row['household'])} (t {f1(row['household_t'])}); goods {f1(row['goods'])}, "
                 f"services {f1(row['services'])}, mixed {f1(row['mixed'])}, public {f1(row['public'])}, aggregation "
                 f"{f1(row['aggregation'])} (PPP term {f1(row['ppp_aggregation'])}); housing {f1(row['CP04'])}, restaurants "
                 f"{f1(row['CP11'])}, misc {f1(row['CP12'])}; consumer goods {f1(row['P0101'])} (t {f1(row['P0101_t'])}), "
                 f"consumer services {f1(row['P0201'])} (t {f1(row['P0201_t'])})")
    rob = pd.DataFrame(rob)
    rob.to_csv(OUT / "robustness.csv", index=False)
    L.append(f"Across variants: services contribution {f1(rob.services.min())} to {f1(rob.services.max())}; goods "
             f"{f1(rob.goods.min())} to {f1(rob.goods.max())}; aggregation {f1(rob.aggregation.min())} to {f1(rob.aggregation.max())}")
    own = []
    for cat, label in (("P0101", "goods"), ("P0201", "services")):
        xc = C.price_level(d, cat=cat)
        o = C.ols(C.drift_parts(d, cat).drift, xc)
        own.append(f"{label} {f1(o['slope'])} (t {f1(o['t'])})")
    L.append("With each aggregate's own 2005 price level as regressor: " + ", ".join(own))
    L.append("")

    # ------------------------------------------------------------------ criteria
    L.append("## Criteria")
    g_, s_ = gs.loc["P0101"], gs.loc["P0201"]
    kd = {k: C.ols(s, x)["slope"] for k, s in ks.items()}
    named = div.loc[["Housing, water, electricity, gas", "Restaurants and hotels", "Miscellaneous goods and services"]]
    crit = [
        ("consumer services slope negative with |t| > 2", s_.slope < 0 and abs(s_.t) > 2),
        ("consumer goods slope flat: |t| < 2 and under a third of the services slope", abs(g_.t) < 2 and abs(g_.slope) < abs(s_.slope) / 3),
        ("service divisions carry more than half of the household slope", kd["services"] / total > 0.5),
        ("goods divisions carry less than a sixth of it in absolute value", abs(kd["goods"] / total) < 1 / 6),
        ("each named service division (housing, restaurants and hotels, miscellaneous) has slope < 0 with t < -1",
         bool(((named.slope < 0) & (named.t < -1)).all())),
        ("drift slope rises with convergence speed across divisions (Spearman > 0.5, average of both measures)",
         C.spearman(div.slope, avg) > 0.5),
    ]
    for text, ok in crit:
        L.append(f"{'PASS' if ok else 'FAIL'}: {text}")

    (OUT / "analysis_summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
