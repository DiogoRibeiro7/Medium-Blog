"""Diagnose the Commission's dependency criteria on the HS 6-digit panel, 2002-2025.

    python panel.py

Uses the data snapshot of the companion analysis in ``../europe-deindustrialisation/data``.

1. Concentration: the Herfindahl index never exceeds the largest supplier's share, and is at least
   its square. So "HHI > 0.4" flags only products whose largest outside supplier has more than 40%
   of extra-EU imports, and flags every product whose largest supplier has more than 63%. The
   products in between are flagged or not depending on how the rest of the imports are spread.
2. Threshold sensitivity: how many products each version of the screen flags as each threshold
   moves.
3. Stability: of the products flagged in one year, how many are still flagged in later years, for
   the Commission screen and for a single-supplier screen.

Writes ``results/panel_*.csv`` and ``results/panel_summary.txt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "europe-deindustrialisation" / "data"
RES = HERE / "results"

HHI_MIN, TOP_MIN, IMPORTANCE_MIN, SUBST_MIN = 0.4, 0.4, 0.5, 1.0


def load() -> pd.DataFrame:
    t = pd.read_csv(SRC / "hs6_totals.csv", dtype={"product": str})
    s = pd.read_csv(SRC / "hs6_suppliers.csv", dtype={"product": str}, keep_default_na=False, na_values=[""])
    d = t.merge(s, on=["product", "year"], how="inner")
    for c in ("imports_extra", "imports_intra", "exports_extra", "exports_intra"):
        d[c] = d[c].fillna(0.0)
    d["importance"] = d.imports_extra / (d.imports_extra + d.imports_intra)
    exports = d.exports_extra + d.exports_intra
    d["substitutability"] = np.where(exports > 0, d.imports_extra / exports.where(exports > 0, 1), np.inf)
    d["share_of_all_imports"] = d.top_share * d.importance
    d["commission"] = (d.hhi > HHI_MIN) & (d.importance > IMPORTANCE_MIN) & (d.substitutability > SUBST_MIN)
    d["single"] = (d.top_share > TOP_MIN) & (d.importance > IMPORTANCE_MIN) & (d.substitutability > SUBST_MIN)
    return d


def persistence(d: pd.DataFrame, col: str) -> pd.DataFrame:
    """Share of products flagged in year a that are flagged again in year b (products traded in both)."""
    years = sorted(d.year.unique())
    rows = []
    for i, a in enumerate(years):
        A = d[d.year == a]
        for b in years[i + 1 :]:
            B = d[d.year == b]
            common = set(A["product"]) & set(B["product"])
            fa = set(A.loc[A[col], "product"]) & common
            fb = set(B.loc[B[col], "product"]) & common
            if fa:
                rows.append({"screen": col, "from": int(a), "to": int(b), "gap": int(b - a), "flagged": len(fa), "still": len(fa & fb), "share": len(fa & fb) / len(fa)})
    return pd.DataFrame(rows)


def main() -> None:
    RES.mkdir(exist_ok=True)
    d = load()
    L: list[str] = []
    say = L.append
    say(f"HS 6-digit panel: {d['product'].nunique()} products, years {', '.join(str(y) for y in sorted(d.year.unique()))}")

    # 1. concentration: the bounds and the gap
    assert (d.hhi <= d.top_share + 1e-5).all() and (d.hhi >= d.top_share**2 - 1e-5).all()  # values stored to six significant digits
    say("\n1. Concentration criterion. HHI <= largest share and HHI >= largest share squared hold for every product-year")
    rows = []
    for y, g in d.groupby("year"):
        gap = g[(g.top_share > TOP_MIN) & (g.hhi <= HHI_MIN)]
        rows.append(
            {
                "year": int(y),
                "hhi_above": int((g.hhi > HHI_MIN).sum()),
                "top_above": int((g.top_share > TOP_MIN).sum()),
                "gap_products": len(gap),
                "gap_value": float(gap.imports_extra.sum()),
                "commission": int(g.commission.sum()),
                "single": int(g.single.sum()),
                "commission_value": float(g[g.commission].imports_extra.sum()),
                "single_value": float(g[g.single].imports_extra.sum()),
                "extra_imports": float(g.imports_extra.sum()),
            }
        )
    C = pd.DataFrame(rows)
    C.to_csv(RES / "panel_concentration.csv", index=False, float_format="%.6g")
    for _, r in C.iterrows():
        say(
            f"  {int(r.year)}: HHI > 0.4 {int(r.hhi_above)}; largest > 40% {int(r.top_above)}; largest > 40% but HHI <= 0.4: {int(r.gap_products)} products, EUR {r.gap_value / 1e9:,.0f} bn; "
            f"full screens: Commission {int(r.commission)} (EUR {r.commission_value / 1e9:,.0f} bn), single-supplier {int(r.single)} (EUR {r.single_value / 1e9:,.0f} bn)"
        )
    last = int(d.year.max())
    g = d[d.year == last]
    zone = g[(g.top_share > 0.4) & (g.top_share <= 0.632)]
    say(f"  {last}: products with largest share in (0.40, 0.63]: {len(zone)}; of these HHI > 0.4: {int((zone.hhi > 0.4).sum())} ({(zone.hhi > 0.4).mean():.0%})")
    big = d[d.single & ~d.commission].sort_values("imports_extra", ascending=False)
    say("  largest products flagged by the single-supplier screen but not the Commission screen, any year:")
    for _, r in big.drop_duplicates("product").head(10).iterrows():
        say(f"    {r['product']} {int(r.year)}: EUR {r.imports_extra / 1e9:,.1f} bn; largest {r.top_partner} {r.top_share:.0%}; HHI {r.hhi:.2f}")

    # 2. threshold sensitivity, last year
    say(f"\n2. Threshold sensitivity, {last} (other two criteria held at their values)")
    rows = []
    base = g[(g.importance > IMPORTANCE_MIN) & (g.substitutability > SUBST_MIN)]
    for thr in np.round(np.arange(0.2, 0.81, 0.05), 2):
        rows.append({"threshold": thr, "hhi": int((base.hhi > thr).sum()), "top_share": int((base.top_share > thr).sum())})
    for thr in np.round(np.arange(0.3, 0.71, 0.05), 2):
        rows.append({"threshold_importance": thr, "commission_importance": int(((g.hhi > HHI_MIN) & (g.importance > thr) & (g.substitutability > SUBST_MIN)).sum())})
    for thr in (0.5, 0.75, 1.0, 1.5, 2.0):
        rows.append({"threshold_subst": thr, "commission_subst": int(((g.hhi > HHI_MIN) & (g.importance > IMPORTANCE_MIN) & (g.substitutability > thr)).sum())})
    T = pd.DataFrame(rows)
    T.to_csv(RES / "panel_thresholds.csv", index=False, float_format="%.6g")
    t1 = T.dropna(subset=["threshold"])
    say("  first criterion: " + "; ".join(f"{r.threshold:.2f}: HHI {int(r.hhi)}, largest share {int(r.top_share)}" for r in t1.itertuples()))
    near = g[(g.importance > IMPORTANCE_MIN) & (g.substitutability > SUBST_MIN) & (g.hhi.between(0.35, 0.45))]
    say(f"  products passing criteria 2 and 3 with HHI within 0.05 of the threshold: {len(near)}")
    for name, col, thr in (("HHI", "hhi", HHI_MIN), ("importance", "importance", IMPORTANCE_MIN)):
        within = g[(g[col] - thr).abs() <= 0.05]
        say(f"  products within 0.05 of the {name} threshold: {len(within)} of {len(g)}")
    within = g[(np.log(g.substitutability.replace(np.inf, np.nan)) - np.log(SUBST_MIN)).abs() <= np.log(1.25)]
    say(f"  products within a factor 1.25 of the substitutability threshold: {len(within)} of {len(g)}")

    # 3. stability
    P = pd.concat([persistence(d, "commission"), persistence(d, "single")], ignore_index=True)
    P.to_csv(RES / "panel_persistence.csv", index=False, float_format="%.6g")
    say("\n3. Stability: share of flagged products still flagged later (products traded in both years)")
    for col in ("commission", "single"):
        p = P[P.screen == col]
        for a, b in ((2019, 2021), (2021, 2022), (2023, 2024), (2024, 2025), (2019, 2025), (2008, 2019), (2002, 2025)):
            r = p[(p["from"] == a) & (p["to"] == b)]
            if len(r):
                r = r.iloc[0]
                say(f"  {col}: {a} -> {b}: {int(r.still)} of {int(r.flagged)} ({r.share:.0%})")
        consecutive = p[p.gap.isin([1, 2, 3])]
        say(f"  {col}: mean persistence between adjacent observed years (gap of 1-3 years): {consecutive.share.mean():.0%}")
    # rank stability of continuous scores
    say("\n  rank correlation of continuous scores between adjacent years (Spearman), products traded in both")
    years = sorted(d.year.unique())
    rows = []
    for a, b in zip(years[:-1], years[1:]):
        A = d[d.year == a].set_index("product")
        B = d[d.year == b].set_index("product")
        common = A.index.intersection(B.index)
        for col in ("hhi", "top_share", "share_of_all_imports"):
            rows.append({"from": int(a), "to": int(b), "score": col, "spearman": float(A.loc[common, col].rank().corr(B.loc[common, col].rank()))})
    R = pd.DataFrame(rows)
    R.to_csv(RES / "panel_rank_stability.csv", index=False, float_format="%.6g")
    for col in ("hhi", "top_share", "share_of_all_imports"):
        say(f"    {col}: mean {R[R.score == col].spearman.mean():.3f}")

    (RES / "panel_summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
