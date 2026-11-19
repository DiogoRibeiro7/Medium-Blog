"""Every number in the article, from the Eurostat snapshot in data/.

python felt_numbers.py > output.txt

Also writes the tables behind the figures to results/.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import felt_core as fc

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
N_BOOT = 999


def head(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def pct(x, d=1):
    return f"{x:+.{d}f}%"


def ci(draws):
    lo, hi = fc.interval(draws)
    return f"[{lo:.2f}, {hi:.2f}]"


def main():
    RES.mkdir(exist_ok=True)
    P = fc.read_wide("hicp_index_v1")
    W = fc.read_wide("hicp_weights_v1")
    I2 = fc.read_wide("hicp_index_v2")
    W2 = fc.read_wide("hicp_weights_v2")
    survey = fc.read_wide("perceived_prices")

    # ------------------------------------------------------------------------------------------------ 1
    head("1. Data")
    p = fc.country_panel(P, W, survey)
    print(f"countries: {p.geo.nunique()} ({' '.join(sorted(p.geo.unique()))})")
    print(f"months: {p.month.min()} to {p.month.max()}; country-months in the panel: {len(p)}")
    print("prices: Eurostat HICP, ECOICOP ver.1 to 2025-12 (rebuilt baskets); ECOICOP ver.2 aggregates to 2026-10")
    print("perceptions: EC consumer survey, price trends over the last 12 months, balance, seasonally adjusted")

    # ------------------------------------------------------------------------------------------------ 2
    head("2. Rebuilding Eurostat's frequent out-of-pocket purchases (FROOPP) index: euro area check")
    ea = fc.rebuild(P["EA20"], W["EA20"], 2001)
    wf = pd.Series({y: sum(fc.frequent_items(y, P["EA20"], W["EA20"]).values()) for y in range(2001, 2026)})
    wp = W["EA20"]["FROOPP"].reindex([str(y) for y in range(2001, 2026)])
    wp.index = wf.index
    pre, post = wf.loc[:2016] - wp.loc[:2016], wf.loc[2017:] - wp.loc[2017:]
    print(f"weight, rebuilt minus published, 2001-2016: largest gap {pre.abs().max():.2f} per mille")
    print(f"weight, rebuilt minus published, 2017-2025: {post.min():.1f} to {post.max():.1f} per mille")
    print(f"published FROOPP weight 2017-2025: {wp.loc[2017:].min():.1f} to {wp.loc[2017:].max():.1f} per mille")
    d = (fc.rate(ea.F) - fc.rate(P["EA20"]["FROOPP"].reindex(ea.index))).dropna()
    print(f"12-month rate, rebuilt minus published, 2002-2016: largest gap {d[:'2016-12'].abs().max():.3f} points")
    a = d["2017-01":].abs()
    print(f"12-month rate, rebuilt minus published, 2017-2025: mean gap {a.mean():.3f}, largest {a.max():.3f} points")
    d = (fc.rate(ea.X) - fc.rate(P["EA20"]["TOT_X_FROOPP"].reindex(ea.index))).dropna()
    print(f"everything else, 12-month rate gap 2002-2016: largest {d[:'2016-12'].abs().max():.3f} points")

    # ------------------------------------------------------------------------------------------------ 3
    head("3. Euro area (20 countries), two baskets, ECOICOP ver.2, to October 2026")
    e = I2["EA20"]
    F, X, T = e["FROOPP"], e["TOT_X_FROOPP"], e["TOTAL"]
    w26 = W2["EA20"].at["2026", "FROOPP"] / 10
    print(f"weight of frequent purchases in 2026: {w26:.1f}% of spending")
    for a_, b_ in [("2001-12", "2020-12"), ("2001-12", "2025-12"), ("2001-12", "2026-10")]:
        print(
            f"{a_} to {b_}, a year: frequent {fc.annualised(F, a_, b_):.2f}%,"
            f" all items {fc.annualised(T, a_, b_):.2f}%,"
            f" everything else {fc.annualised(X, a_, b_):.2f}%"
        )
    for a_, b_ in [("2001-12", "2026-10"), ("2020-12", "2025-12"), ("2020-12", "2026-10"), ("2025-12", "2026-10")]:
        print(
            f"{a_} to {b_}, total: frequent {pct(fc.cumulative(F, a_, b_))}, all items {pct(fc.cumulative(T, a_, b_))},"
            f" everything else {pct(fc.cumulative(X, a_, b_))}"
        )
    rF, rX, rT = fc.rate(F), fc.rate(X), fc.rate(T)
    gap = (rF - rX).loc["2002-01":"2026-10"]
    print(
        f"months 2002-01 to 2026-10 with frequent purchases rising faster than everything else:"
        f" {(gap > 0).sum()} of {len(gap)}"
    )
    years = pd.DataFrame({"fq": rF, "ee": rX, "ai": rT}).loc["2002-01":"2025-12"]
    years["year"] = years.index.str[:4]
    print("average 12-month rate by year (frequent / all items / everything else):")
    for y, g in years.groupby("year"):
        print(f"  {y}: {fc.mean(g.fq):5.2f}  {fc.mean(g.ai):5.2f}  {fc.mean(g.ee):5.2f}")
    print("12-month rates in 2026 (frequent / all items / everything else / fuel / household electricity and gas):")
    for mo in [f"2026-{m:02d}" for m in range(1, 11)]:
        print(
            f"  {mo}: {rF[mo]:5.2f}  {rT[mo]:5.2f}  {rX[mo]:5.2f}  {fc.rate(e['FUEL'])[mo]:6.2f}"
            f"  {fc.rate(e['ELC_GAS'])[mo]:6.2f}"
        )
    pd.DataFrame({"frequent": rF, "all_items": rT, "everything_else": rX}).loc["2000-12":"2026-10"].round(3).to_csv(
        RES / "ea_two_baskets.csv", index_label="month"
    )

    # ------------------------------------------------------------------------------------------------ 4
    head("4. Item by item, euro area, December 2001 to December 2025 (ECOICOP ver.1)")
    rows = []
    for code, (label, kind) in fc.SHOWCASE.items():
        s = P["EA20"][code]
        rows.append((code, label, kind, fc.cumulative(s, "2001-12", "2025-12")))
    show = pd.DataFrame(rows, columns=["code", "label", "kind", "change"]).sort_values("change", ascending=False)
    for r in show.itertuples():
        print(f"  {r.label:30s} {r.kind:9s} {pct(r.change)}")
    for kind, g in show.groupby("kind", sort=False):
        print(f"median, {kind}: {pct(g.change.median())} ({len(g)} items)")
    show.round(2).to_csv(RES / "showcase.csv", index=False)

    # ------------------------------------------------------------------------------------------------ 5
    head("5. Country by country: rebuilt baskets, December 2020 to December 2025")
    rows = []
    for geo in fc.EU27:
        r = fc.rebuild(P[geo], W[geo])
        rows.append(
            {
                "geo": geo,
                "frequent": fc.cumulative(r.F, "2020-12", "2025-12"),
                "all_items": fc.cumulative(r["T"], "2020-12", "2025-12"),
                "everything_else": fc.cumulative(r.X, "2020-12", "2025-12"),
                "share_2025": r.share_F["2025-12"] * 100,
                "long_F": fc.annualised(r.F, "2001-12", "2020-12"),
                "long_X": fc.annualised(r.X, "2001-12", "2020-12"),
            }
        )
    c = pd.DataFrame(rows).set_index("geo")
    c["gap"] = c.frequent - c.all_items
    c = c.sort_values("gap", ascending=False)
    for g, r in c.iterrows():
        print(
            f"  {g}: frequent {pct(r.frequent)}, all items {pct(r.all_items)},"
            f" everything else {pct(r.everything_else)},"
            f" gap {r.gap:+.1f} points; frequent share {r.share_2025:.1f}%"
        )
    print(f"countries where frequent purchases rose more than all items: {(c.gap > 0).sum()} of {len(c)}")
    print(
        f"median gap: {c.gap.median():.1f} points; frequent share of spending in 2025:"
        f" {c.share_2025.min():.1f}% to {c.share_2025.max():.1f}%"
    )
    lr = c.long_F - c.long_X
    print(
        f"2001-12 to 2020-12, a year, frequent minus everything else: median {lr.median():.2f} points,"
        f" positive in {(lr > 0).sum()} of {len(lr)}; negative in {', '.join(lr[lr < 0].index)}"
    )
    c.round(3).to_csv(RES / "countries.csv")

    # ------------------------------------------------------------------------------------------------ 6
    head("6. What people say: perceived price trends against the two baskets, 27 countries, 2002-2025")
    cor = p.groupby("geo").apply(
        lambda d: pd.Series({k: fc.corr(d.B, d[k]) for k in ["F12", "T12", "X12"]}), include_groups=False
    )
    print(
        f"correlation with the balance, median over countries: frequent {cor.F12.median():.2f},"
        f" all items {cor.T12.median():.2f}, everything else {cor.X12.median():.2f}"
    )
    print(f"frequent correlates more than everything else in {(cor.F12 > cor.X12).sum()} of {len(cor)} countries")
    print(f"frequent correlates more than all items in {(cor.F12 > cor.T12).sum()} of {len(cor)} countries")
    cor.round(4).to_csv(RES / "correlations.csv")
    gaps = (cor.F12 - cor.X12).sort_values(ascending=False)
    print(
        "largest gaps (frequent minus everything else): " + ", ".join(f"{g} {v:.2f}" for g, v in gaps.head(5).items())
    )
    print("everything else correlates more in: " + ", ".join(f"{g} {v:.2f}" for g, v in gaps[gaps < 0].items()))
    for x in ["T12", "F12", "X12"]:
        b, r2 = fc.fe_fit(p, "B", [x])
        print(f"one basket, country fixed effects: {x} slope {b[0]:.2f}, within R2 {r2:.3f}")

    def lam(q, two_way=False):
        b, _ = fc.fe_fit(q, "B", ["F12", "X12"], two_way)
        return b[0] / (b[0] + b[1])

    b, r2 = fc.fe_fit(p, "B", ["F12", "X12"])
    share = fc.mean(p.groupby("geo").share_F.apply(fc.mean))
    draws = fc.bootstrap(p, lam, N_BOOT)
    lam_hat = lam(p)
    print(
        f"both baskets, country fixed effects: slopes frequent {b[0]:.2f}, everything else {b[1]:.2f},"
        f" within R2 {r2:.3f}"
    )
    print(
        f"weight the answers put on frequent purchases: {lam_hat:.2f}, 95% interval {ci(draws)} (countries resampled)"
    )
    lam_lo, lam_hi = fc.interval(draws)
    print(f"weight frequent purchases have in spending (average over countries): {share:.2f}")
    draws2 = fc.bootstrap(p, lambda q: lam(q, True), 199)
    print(f"with month fixed effects as well: {lam(p, True):.2f}, 95% interval {ci(draws2)} (199 resamples)")
    print("the same weight in other samples:")
    yr = p.year
    for name, q in [
        ("2002-2019", p[yr <= 2019]),
        ("without 2021-2023", p[(yr < 2021) | (yr > 2023)]),
        ("without 2002", p[yr != 2002]),
        ("euro countries", p[p.geo.isin(fc.EURO_2025)]),
        ("countries outside the euro", p[~p.geo.isin(fc.EURO_2025)]),
    ]:
        print(
            f"  {name:28s} {lam(q):.2f} (spending share {fc.mean(q.groupby('geo').share_F.apply(fc.mean)):.2f},"
            f" {len(q)} rows)"
        )

    # ------------------------------------------------------------------------------------------------ 7
    head("7. Groceries, bills and the rest")
    names = {
        "groceries": "groceries, drinks, tobacco",
        "fuel": "fuel for cars",
        "other_frequent": "other frequent purchases",
        "energy": "household energy",
        "bills": "rent, water, phone, insurance",
        "rest": "everything else",
    }
    b, r2 = fc.fe_fit(p, "B", fc.COMPONENTS)
    bd = fc.bootstrap(p, lambda q: fc.implied_weights(fc.fe_fit(q, "B", fc.COMPONENTS)[0]), N_BOOT)
    iw = fc.implied_weights(b)
    lo, hi = fc.interval(bd)
    print(f"six baskets, country fixed effects, within R2 {r2:.3f}")
    rows = []
    for i, k in enumerate(fc.COMPONENTS):
        sh = fc.mean(p["w_" + k])
        print(
            f"  {names[k]:30s} slope {b[i]:5.2f}; weight in answers {iw[i]:.2f} [{lo[i]:.2f}, {hi[i]:.2f}];"
            f" in spending {sh:.2f}"
        )
        rows.append(("six", k, names[k], iw[i], lo[i], hi[i], sh))
    three = ["frequent", "billed", "rest"]
    b, r2 = fc.fe_fit(p, "B", three)
    bd = fc.bootstrap(p, lambda q: fc.implied_weights(fc.fe_fit(q, "B", three)[0]), N_BOOT)
    iw = fc.implied_weights(b)
    lo, hi = fc.interval(bd)
    print(f"three baskets, within R2 {r2:.3f}")
    labels3 = {
        "frequent": "frequent purchases",
        "billed": "bills (energy, rent, water, phone, insurance)",
        "rest": "everything else",
    }
    for i, k in enumerate(three):
        sh = fc.mean(p["w_" + k])
        print(
            f"  {labels3[k]:46s} slope {b[i]:5.2f}; weight in answers {iw[i]:.2f} [{lo[i]:.2f}, {hi[i]:.2f}];"
            f" in spending {sh:.2f}"
        )
        rows.append(("three", k, labels3[k], iw[i], lo[i], hi[i], sh))
    pd.DataFrame(rows, columns=["model", "basket", "label", "weight", "lo", "hi", "spending"]).round(4).to_csv(
        RES / "weights.csv", index=False
    )
    print("three baskets in other samples (weights in answers: frequent / bills / everything else):")
    for name, q in [
        ("with month fixed effects", None),
        ("2002-2019", p[yr <= 2019]),
        ("without 2021-2023", p[(yr < 2021) | (yr > 2023)]),
        ("euro countries", p[p.geo.isin(fc.EURO_2025)]),
    ]:
        bb = fc.fe_fit(p, "B", three, True)[0] if q is None else fc.fe_fit(q, "B", three)[0]
        w3 = fc.implied_weights(bb)
        print(f"  {name:26s} {w3[0]:.2f} / {w3[1]:.2f} / {w3[2]:.2f}")

    # ------------------------------------------------------------------------------------------------ 8
    head("8. Memory: the last 12 months, or longer?")
    q = p.dropna(subset=["F48", "X48", "T48"])
    print(f"sample with 48 months of history: {len(q)} rows, {q.month.min()} to {q.month.max()}")
    for h in (12, 24, 36, 48):
        bt, rt = fc.fe_fit(q, "B", [f"T{h}"])
        bb, rb = fc.fe_fit(q, "B", [f"F{h}", f"X{h}"])
        print(
            f"  change over {h} months (a year): all items within R2 {rt:.3f}; both baskets {rb:.3f},"
            f" weight on frequent {bb[0] / (bb[0] + bb[1]):.2f}"
        )
    q = p.dropna(subset=["F36", "X36"]).copy()
    short = ["F12", "X12"]
    long = ["F12", "X12", "F36", "X36"]
    bs, rs = fc.fe_fit(q, "B", short)
    bl, rl = fc.fe_fit(q, "B", long)
    print(f"sample with 36 months of history: {len(q)} rows")
    print(f"  12-month rates only: slopes {np.round(bs, 2).tolist()}, within R2 {rs:.3f}")
    print(f"  12- and 36-month rates: slopes {np.round(bl, 2).tolist()} (F12, X12, F36, X36), within R2 {rl:.3f}")
    for name, xs, beta in [("short", short, bs), ("long", long, bl)]:
        M = fc.within(q, "B", xs)
        q["res_" + name] = M[:, 0] - sum(beta[i] * M[:, i + 1] for i in range(len(xs)))
    byyear = q.groupby("year")[["res_short", "res_long"]].agg(fc.mean)
    print("average gap between answers and the model, by year (balance points; 12 months only / with 36 months):")
    for y, r in byyear.iterrows():
        print(f"  {y}: {r.res_short:6.1f}  {r.res_long:6.1f}")
    byyear.round(3).to_csv(RES / "memory_residuals.csv", index_label="year")
    # The cash changeover: the twelve countries that switched in January 2002, from 1998.
    c12 = fc.country_panel(P, W, survey, t0="1998-01")
    c12 = c12[c12.geo.isin(fc.EURO_2002)].copy()
    bc, _ = fc.fe_fit(c12, "B", short)
    M = fc.within(c12, "B", short)
    c12["res"] = M[:, 0] - bc[0] * M[:, 1] - bc[1] * M[:, 2]
    yrs = c12[c12.year <= 2006].groupby("year").res.agg(fc.mean)
    print("the twelve countries of the 2002 cash changeover, 1998-2025 fit, average gap by year:")
    for y, v in yrs.items():
        print(f"  {y}: {v:6.1f}")
    before = c12[c12.year < 2002].groupby("geo").size()
    print(f"  countries with answers before 2002: {len(before)}, {before.min()} to {before.max()} months each")

    # ------------------------------------------------------------------------------------------------ 9
    head("9. Out of sample: fit on 2002-2019, predict 2020-2025")
    train, test = p[p.year <= 2019], p[p.year >= 2020].copy()
    scores = {}
    for name, xs in [("all items", ["T12"]), ("two baskets", ["F12", "X12"]), ("three baskets", three)]:
        pred, beta = fc.fe_predict(train, test, "B", xs)
        test["pred_" + name] = pred
        scores[name] = test.assign(e=test.B - pred).groupby("geo").e.apply(lambda e: fc.rmse(e, 0 * e))
        print(
            f"  {name:14s} slopes {np.round(beta, 2).tolist()};"
            f" RMSE 2020-2025 {fc.rmse(test.B, pred):.1f} balance points"
        )
    for name in ["two baskets", "three baskets"]:
        print(
            f"  {name} beats all items in {(scores[name] < scores['all items']).sum()} of {len(scores[name])} countries"
        )
    print("  average error by year (answers minus prediction; all items / two baskets):")
    for y, g in test.groupby("year"):
        print(f"    {y}: {fc.mean(g.B - g['pred_all items']):6.1f}  {fc.mean(g.B - g['pred_two baskets']):6.1f}")
    # The euro area as a whole, 2026: the new shock, with a model fitted on 2002-2025.
    e21 = I2["EA21"]
    agg = pd.DataFrame(
        {
            "B": survey["EA21"]["BS-PT-LY"],
            "F12": fc.rate(e21["FROOPP"]),
            "X12": fc.rate(e21["TOT_X_FROOPP"]),
            "T12": fc.rate(e21["TOTAL"]),
        }
    ).dropna()
    tr, te = agg.loc["2002-01":"2025-12"], agg.loc["2026-01":"2026-10"]
    early = agg.B.loc["2001-01":"2020-12"]
    print(
        f"euro area balance: highest 2001-2020 {early.max():.1f} ({early.idxmax()});"
        f" October 2026 {agg.B['2026-10']:.1f}"
    )
    preds = {}
    print("euro area (21 countries), fitted on 2002-2025, 2026 answers minus prediction:")
    for name, xs in [("all items", ["T12"]), ("two baskets", ["F12", "X12"])]:
        coef = fc.ols(tr, "B", xs)
        pr = coef[0] + sum(coef[i + 1] * te[x] for i, x in enumerate(xs))
        preds[name] = pr
        print(
            f"  {name:12s} mean {fc.mean(te.B - pr):5.1f}, RMSE {fc.rmse(te.B, pr):5.1f};"
            f" balance in 2026 {te.B.min():.1f} to {te.B.max():.1f}"
        )
    print("  month: answers / predicted from all items / predicted from two baskets")
    for mo in te.index:
        print(f"  {mo}: {te.B[mo]:5.1f}  {preds['all items'][mo]:5.1f}  {preds['two baskets'][mo]:5.1f}")

    # ----------------------------------------------------------------------------------------------- 10
    head("10. A felt index for the euro area (20 countries)")
    wy = {int(y): W2["EA20"].at[y, "FROOPP"] / 1000 for y in W2["EA20"].index if pd.notna(W2["EA20"].at[y, "FROOPP"])}
    F, X = F.loc["2000-12":], X.loc["2000-12":]
    official = fc.mix(F, X, wy)
    print(
        f"two-basket index with the published weights against all items, 2020-12 to 2026-10:"
        f" {pct(fc.cumulative(official, '2020-12', '2026-10'), 2)} vs {pct(fc.cumulative(T, '2020-12', '2026-10'), 2)}"
    )
    print(
        f"published weight of frequent purchases, 2021-2026:"
        f" {min(wy[y] for y in range(2021, 2027)):.3f} to {max(wy[y] for y in range(2021, 2027)):.3f}"
    )
    rows = []
    for w in np.round(np.linspace(0, 1, 101), 2):
        s = fc.mix(F, X, float(w))
        rows.append((w, fc.cumulative(s, "2020-12", "2026-10"), fc.cumulative(s, "2001-12", "2026-10")))
    curve = pd.DataFrame(rows, columns=["w", "since_2020", "since_2001"])
    curve.round(4).to_csv(RES / "felt_curve.csv", index=False)
    for w in [round(wy[2026], 3), 0.5, round(lam_hat, 2), 1.0]:
        s = fc.mix(F, X, w)
        r = fc.rate(s)
        print(
            f"  weight {w:.3f}: 2020-12 to 2026-10 {pct(fc.cumulative(s, '2020-12', '2026-10'))},"
            f" 2001-12 to 2026-10 {pct(fc.cumulative(s, '2001-12', '2026-10'))};"
            f" 12-month rate 2022-12 {r['2022-12']:.1f}%, 2026-10 {r['2026-10']:.1f}%"
        )
    print(f"published all items: 12-month rate 2022-12 {rT['2022-12']:.1f}%, 2026-10 {rT['2026-10']:.1f}%")
    key = {
        "weight_official_2026": wy[2026],
        "weight_answers": lam_hat,
        "weight_answers_lo": lam_lo,
        "weight_answers_hi": lam_hi,
    }
    pd.Series(key).round(4).to_csv(RES / "key_numbers.csv", header=["value"], index_label="name")


if __name__ == "__main__":
    main()
