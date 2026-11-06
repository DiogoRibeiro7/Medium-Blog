"""The Commission's dependency screen, applied to every HS 6-digit product and year.

    python screen.py

A product is counted as dependent on suppliers outside the EU in a year when all three criteria of
the Commission's 2021 review of strategic dependencies hold:

1. concentration: the Herfindahl-Hirschman index of extra-EU suppliers is above 0.4;
2. importance: extra-EU imports are more than half of all EU imports of the product (from outside
   and from other member states);
3. substitutability: extra-EU imports exceed total EU exports of the product (to outside and to
   other member states), a sign that EU production could not readily replace them.

The Commission applied the screen to products in sensitive industrial ecosystems in one period;
here it is applied to every product in eleven years from 2002 to 2025, so the count is larger and
the point is the trend and who the main suppliers are. A further measure, the share of all EU
imports of a product that comes from its single largest outside supplier, asks how much of the
EU's supply one country controls.

Writes ``results/screen_*.csv`` and ``results/screen_summary.txt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RES = HERE / "results"

HHI_MIN, EXTRA_SHARE_MIN, IMPORT_EXPORT_MIN = 0.4, 0.5, 1.0
MANUFACTURED = {f"{i:02d}" for i in range(28, 97)} - {"71", "93"}
GROUPS = {"energy and ores (HS 25-27)": {"25", "26", "27"}, "chemicals and pharmaceuticals (HS 28-38)": {f"{i}" for i in range(28, 39)}, "other manufactures (HS 39-96)": {f"{i}" for i in range(39, 97)} - {"71", "93"}}
NAMES = {"CN": "China", "US": "United States", "RU": "Russia", "GB": "United Kingdom", "CH": "Switzerland", "NO": "Norway", "TR": "Türkiye", "JP": "Japan", "KR": "South Korea", "IN": "India", "TW": "Taiwan", "VN": "Vietnam", "BR": "Brazil", "UA": "Ukraine", "BY": "Belarus"}


def criteria(t: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """Merge totals and supplier summaries and evaluate the three criteria."""
    d = t.merge(s, on=["product", "year"], how="inner")
    for c in ("imports_extra", "imports_intra", "exports_extra", "exports_intra"):
        d[c] = d.get(c, 0.0)
        d[c] = d[c].fillna(0.0)
    d["extra_share"] = d.imports_extra / (d.imports_extra + d.imports_intra)
    exports = d.exports_extra + d.exports_intra
    d["import_export_ratio"] = np.where(exports > 0, d.imports_extra / exports.where(exports > 0, 1), np.inf)
    d["c_concentration"] = d.hhi > HHI_MIN
    d["c_importance"] = d.extra_share > EXTRA_SHARE_MIN
    d["c_substitutability"] = d.import_export_ratio > IMPORT_EXPORT_MIN
    d["dependent"] = d.c_concentration & d.c_importance & d.c_substitutability
    # share of all EU imports of the product supplied by its largest outside supplier
    d["top_share_of_all_imports"] = d.top_share * d.extra_share
    d["chapter"] = d["product"].str[:2]
    return d


def main() -> None:
    RES.mkdir(exist_ok=True)
    t = pd.read_csv(DATA / "hs6_totals.csv", dtype={"product": str})
    s = pd.read_csv(DATA / "hs6_suppliers.csv", dtype={"product": str}, keep_default_na=False, na_values=[""])
    d = criteria(t, s)
    d.to_csv(RES / "screen_products.csv", index=False, float_format="%.6g")
    L = []
    say = L.append
    say("Dependency screen (Commission 2021 criteria: HHI > 0.4, extra-EU share of EU imports > 0.5, extra-EU imports > EU exports)")
    say(f"HS 6-digit products with extra-EU imports and complete totals: {d['product'].nunique()}; years {', '.join(str(y) for y in sorted(d.year.unique()))}")

    rows = []
    for y, g in d.groupby("year"):
        dep = g[g.dependent]
        row = {
            "year": int(y),
            "products": len(g),
            "dependent": len(dep),
            "dependent_value_share": float(dep.imports_extra.sum() / g.imports_extra.sum()),
            "dependent_value_eur": float(dep.imports_extra.sum()),
            "single_supplier_over_half": int((g.top_share_of_all_imports > 0.5).sum()),
        }
        for p in ("CN", "US", "RU", "GB", "CH", "NO", "TR", "JP", "KR", "IN", "TW", "VN"):
            row[f"dependent_top_{p}"] = int((dep.top_partner == p).sum())
        row["dependent_top_other"] = int(len(dep) - sum(row[f"dependent_top_{p}"] for p in ("CN", "US", "RU", "GB", "CH", "NO", "TR", "JP", "KR", "IN", "TW", "VN")))
        for name, chs in GROUPS.items():
            row[f"dependent_{name}"] = int(dep.chapter.isin(chs).sum())
        rows.append(row)
    Y = pd.DataFrame(rows)
    Y.to_csv(RES / "screen_by_year.csv", index=False, float_format="%.6g")

    say("\nDependent products by year: count, share of extra-EU import value, and largest supplier")
    for _, r in Y.iterrows():
        say(
            f"  {r.year}: {r.dependent} of {r.products} products, {r.dependent_value_share:.1%} of extra-EU import value (EUR {r.dependent_value_eur / 1e9:,.0f} bn); "
            f"largest supplier China {r.dependent_top_CN}, United States {r.dependent_top_US}, Russia {r.dependent_top_RU}, other {r.dependent - r.dependent_top_CN - r.dependent_top_US - r.dependent_top_RU}; "
            f"products where one outside country supplies over half of all EU imports: {r.single_supplier_over_half}"
        )
    say("\nDependent products by product group")
    for _, r in Y.iterrows():
        say(f"  {r.year}: " + "; ".join(f"{name} {int(r[f'dependent_{name}'])}" for name in GROUPS))

    # which criterion binds
    last = int(d.year.max())
    g = d[d.year == last]
    say(f"\nIn {last}: products meeting the concentration criterion {int(g.c_concentration.sum())}, importance {int(g.c_importance.sum())}, substitutability {int(g.c_substitutability.sum())}; concentration and importance {int((g.c_concentration & g.c_importance).sum())}; all three {int(g.dependent.sum())}")

    # persistence between selected years
    def dep_set(y: int) -> set[str]:
        return set(d[(d.year == y) & d.dependent]["product"])

    for a, b in ((2008, 2019), (2019, last), (2021, last)):
        A, B = dep_set(a), dep_set(b)
        common = set(d[d.year == a]["product"]) & set(d[d.year == b]["product"])
        A, B = A & common, B & common
        say(f"  {a} -> {b}: dependent in {a} {len(A)}, of which still dependent in {b} {len(A & B)} ({len(A & B) / max(len(A), 1):.0%}); newly dependent in {b} {len(B - A)}")

    # by supplier: value of EU imports in dependent products where that supplier is the largest
    say("\nExtra-EU import value of dependent products by largest supplier (EUR bn)")
    by = d[d.dependent].groupby(["year", "top_partner"]).imports_extra.sum().unstack(fill_value=0) / 1e9
    by.to_csv(RES / "screen_value_by_supplier.csv", float_format="%.6g")
    for y, r in by.iterrows():
        top = r.sort_values(ascending=False).head(5)
        say(f"  {y}: " + "; ".join(f"{NAMES.get(p, p)} {v:,.1f}" for p, v in top.items()))

    # Russia
    say("\nRussia as largest outside supplier of dependent products")
    for y in sorted(d.year.unique()):
        r = d[(d.year == y) & d.dependent & (d.top_partner == "RU")]
        say(f"  {y}: {len(r)} products, EUR {r.imports_extra.sum() / 1e9:,.1f} bn" + (f"; largest: {', '.join(r.sort_values('imports_extra', ascending=False)['product'].head(5))}" if len(r) else ""))

    # the products where one country supplies the largest share of all EU imports, last year
    say(f"\nDependent products in {last} with the highest share of all EU imports from one outside country (extra-EU imports above EUR 100 m)")
    top = g[g.dependent & (g.imports_extra > 1e8)].sort_values("top_share_of_all_imports", ascending=False).head(25)
    for _, r in top.iterrows():
        say(f"  {r['product']}: {NAMES.get(r.top_partner, r.top_partner)} supplies {r.top_share_of_all_imports:.0%} of all EU imports ({r.top_share:.0%} of extra-EU), EUR {r.imports_extra / 1e9:,.2f} bn")
    top.to_csv(RES / "screen_top_products.csv", index=False, float_format="%.6g")

    text = "\n".join(L) + "\n"
    (RES / "screen_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
