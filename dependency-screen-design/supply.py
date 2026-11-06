"""A supply-based dependency screen for 2025, compared with the Commission's criteria.

    python supply.py

Production (Prodcom 2025, sold production in euros) and trade (Comext 2025 at the CN 8-digit level,
summed to Prodcom codes with the CN-Prodcom correspondence) give, for each Prodcom product:

    EU supply                S = EU production + extra-EU imports
    outside share of supply  m = extra-EU imports / S
    single-supplier exposure E = imports from the largest outside supplier / S

E is the share of everything available in the EU, made inside or brought in from outside, that one
outside country provides: the share that would disappear if that country stopped, before any
exports are redirected to the home market. The proposed screen flags a product when E exceeds 25%.

Apparent consumption (production + imports - exports) is also computed, because it is the textbook
denominator, but it is fragile: where exports are close to production plus imports, as with goods
that are imported, processed and re-exported, it is close to zero or negative.

The Commission's three criteria are computed on the same products for comparison: HHI of outside
suppliers > 0.4, extra-EU imports > 50% of all EU imports, extra-EU imports > total EU exports.

Writes ``results/supply_*.csv`` and ``results/supply_summary.txt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SRC = HERE.parent / "europe-deindustrialisation" / "data"
RES = HERE / "results"

E_MIN = 0.25
HHI_MIN, IMPORTANCE_MIN, SUBST_MIN = 0.4, 0.5, 1.0
EU27 = {"AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "GR", "ES", "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK"}
NAMES = {"BD": "Bangladesh", "MY": "Malaysia", "PK": "Pakistan", "BW": "Botswana", "GH": "Ghana", "CI": "Côte d'Ivoire", "CN": "China", "US": "United States", "RU": "Russia", "GB": "United Kingdom", "CH": "Switzerland", "NO": "Norway", "TR": "Türkiye", "JP": "Japan", "KR": "South Korea", "IN": "India", "TW": "Taiwan", "VN": "Vietnam", "BR": "Brazil", "UA": "Ukraine", "ZA": "South Africa", "CL": "Chile", "AU": "Australia", "CA": "Canada", "MA": "Morocco", "ID": "Indonesia"}


def extra_eu(p: pd.Series) -> pd.Series:
    return (p.str.len() == 2) & ~p.isin(EU27) & ~p.str.startswith("Q")


def correspondence() -> tuple[pd.DataFrame, int]:
    c = pd.read_csv(DATA / "cn_prodcom_2025_raw.csv", header=None, dtype=str)
    c.columns = c.iloc[0]
    c = c.iloc[1:][["Prodcom code", "Commodity code"]].dropna()
    c["prodcom"] = c["Prodcom code"].str.replace(".", "", regex=False).str.strip()
    c["cn8"] = c["Commodity code"].str.replace(r"\D", "", regex=True)
    c = c[c.cn8.str.len() == 8].drop_duplicates(["prodcom", "cn8"])
    # a CN code linked to several Prodcom codes cannot be allocated; drop those Prodcom codes entirely
    multi = c.groupby("cn8").prodcom.nunique()
    shared = set(c[c.cn8.isin(multi[multi > 1].index)].prodcom)
    return c[~c.prodcom.isin(shared)][["prodcom", "cn8"]], len(shared)


def build() -> tuple[pd.DataFrame, dict]:
    info: dict = {}
    corr, info["prodcom_dropped_shared_cn"] = correspondence()
    p = pd.read_csv(DATA / "prodcom_2025.csv", dtype={"product": str})
    pw = p.pivot_table(index="product", columns="indicators", values="value").rename_axis("prodcom").reset_index()
    labels = p.drop_duplicates("product").set_index("product").label
    t = pd.read_csv(DATA / "cn8_totals_2025.csv", dtype={"product": str}).rename(columns={"product": "cn8"})
    for c in ("imports_extra", "imports_intra", "exports_extra", "exports_intra"):
        t[c] = t.get(c, 0.0)
        t[c] = t[c].fillna(0.0)
    imp = pd.read_csv(DATA / "cn8_imports_by_partner_2025.csv.gz", dtype={"product": str, "partner": str}, keep_default_na=False, na_values=[""])
    imp = imp[extra_eu(imp.partner) & (imp.value > 0)].rename(columns={"product": "cn8"})

    tr = corr.merge(t, on="cn8", how="left").fillna({"imports_extra": 0.0, "imports_intra": 0.0, "exports_extra": 0.0, "exports_intra": 0.0})
    agg = tr.groupby("prodcom")[["imports_extra", "imports_intra", "exports_extra", "exports_intra"]].sum()
    by_partner = corr.merge(imp, on="cn8").groupby(["prodcom", "partner"]).value.sum()
    shares = by_partner / by_partner.groupby(level=0).transform("sum")
    sup = pd.DataFrame(
        {
            "hhi": (shares**2).groupby(level=0).sum(),
            "top_partner": shares.groupby(level=0).idxmax().str[1],
            "top_share": shares.groupby(level=0).max(),
            "second_share": shares.groupby(level=0).apply(lambda s: float(s.nlargest(2).iloc[-1]) if len(s) > 1 else 0.0),
            "partner_imports": by_partner.groupby(level=0).sum(),
        }
    )
    d = pw.merge(agg.reset_index(), on="prodcom", how="inner").merge(sup.reset_index(), on="prodcom", how="left")
    d["label"] = d.prodcom.map(labels)
    info["prodcom_codes"] = len(pw)
    info["linked"] = len(d)
    ok = d.PRODVAL.notna() & d.top_share.notna() & (d.imports_extra > 0)
    info["no_production"] = int(d.PRODVAL.isna().sum())
    info["no_extra_imports"] = int((d.imports_extra <= 0).sum())
    d = d[ok].copy()
    d["supply"] = d.PRODVAL + d.imports_extra
    d["outside_share"] = d.imports_extra / d.supply
    d["eu_share"] = d.PRODVAL / d.supply
    d["exposure"] = d.top_share * d.imports_extra / d.supply
    d["exposure_second"] = d.second_share * d.imports_extra / d.supply
    d["consumption"] = d.PRODVAL + d.imports_extra - d.exports_extra
    d["exposure_consumption"] = np.where(d.consumption > 0, d.top_share * d.imports_extra / d.consumption.where(d.consumption > 0, 1), np.nan)
    d["importance"] = d.imports_extra / (d.imports_extra + d.imports_intra)
    exports = d.exports_extra + d.exports_intra
    d["substitutability"] = np.where(exports > 0, d.imports_extra / exports.where(exports > 0, 1), np.inf)
    d["exposure_trade"] = d.top_share * d.importance  # trade-only stand-in: largest supplier's share of all EU imports
    d["commission"] = (d.hhi > HHI_MIN) & (d.importance > IMPORTANCE_MIN) & (d.substitutability > SUBST_MIN)
    d["proposed"] = d.exposure > E_MIN
    # Prodcom's own extra-EU import figure, where published, against the CN aggregation
    d["prodcom_impval_ratio"] = d.IMPVAL / d.imports_extra
    return d, info


def main() -> None:
    RES.mkdir(exist_ok=True)
    d, info = build()
    L: list[str] = []
    say = L.append
    say("Supply-based screen, 2025 (Prodcom 2025 production, Comext 2025 trade summed from CN 8-digit codes)")
    say(f"  Prodcom codes with data: {info['prodcom_codes']}; linked to CN trade: {info['linked']}; dropped because a CN code is shared between Prodcom codes: {info['prodcom_dropped_shared_cn']}")
    say(f"  excluded: production not published {info['no_production']}, no extra-EU imports {info['no_extra_imports']}")
    say(f"  analysed: {len(d)} products; EU production EUR {d.PRODVAL.sum() / 1e9:,.0f} bn, extra-EU imports EUR {d.imports_extra.sum() / 1e9:,.0f} bn, extra-EU exports EUR {d.exports_extra.sum() / 1e9:,.0f} bn")
    v = d.dropna(subset=["prodcom_impval_ratio"])
    v = v[np.isfinite(v.prodcom_impval_ratio)]
    say(f"  check: Prodcom's published extra-EU imports against the CN aggregation, {len(v)} products: median ratio {v.prodcom_impval_ratio.median():.3f}, within 5% for {(v.prodcom_impval_ratio.sub(1).abs() <= 0.05).mean():.0%}")
    cons = d.consumption
    say(f"  apparent consumption is zero or negative for {int((cons <= 0).sum())} products and below a tenth of supply for {int(((cons > 0) & (cons < 0.1 * d.supply)).sum())}; consumption-based exposure exceeds 100% for {int((d.exposure_consumption > 1).sum())}")

    say("\nImportance criterion (extra-EU share of all EU imports) against the outside share of EU supply")
    say(f"  Spearman {d.importance.rank().corr(d.outside_share.rank()):.3f}; median importance {d.importance.median():.2f}, median outside share of supply {d.outside_share.median():.2f}")
    a, b = d.importance > IMPORTANCE_MIN, d.outside_share > 0.5
    say(f"  importance > 0.5: {int(a.sum())}; outside share of supply > 0.5: {int(b.sum())}; both {int((a & b).sum())}; importance only {int((a & ~b).sum())}; supply only {int((~a & b).sum())}")
    intra_vs_prod = d.imports_intra / d.PRODVAL
    say(f"  intra-EU imports exceed EU production for {int((intra_vs_prod > 1).sum())} products ({(intra_vs_prod > 1).mean():.0%}); median intra-EU imports / production {intra_vs_prod.median():.2f}")
    say("\nSubstitutability criterion (extra-EU imports / total EU exports) against EU production relative to extra-EU imports")
    s = pd.Series(d.substitutability).replace(np.inf, np.nan)
    say(f"  Spearman of the ratio with the EU share of supply {s.rank().corr(d.eu_share.rank()):.3f}")
    a, b = d.substitutability > SUBST_MIN, d.eu_share < 0.5
    say(f"  ratio > 1: {int(a.sum())}; EU production below extra-EU imports: {int(b.sum())}; both {int((a & b).sum())}; ratio only {int((a & ~b).sum())}; production only {int((~a & b).sum())}")
    total_exports = d.exports_extra + d.exports_intra
    say(f"  total EU exports (extra + intra) exceed EU production for {int((total_exports > d.PRODVAL).sum())} products ({(total_exports > d.PRODVAL).mean():.0%})")

    say(f"\nScreens on the same {len(d)} products")
    C, P = d.commission, d.proposed
    d["sup_imports"] = d.top_share * d.imports_extra
    for name, m in (("Commission", C), (f"proposed (E > {E_MIN:.0%})", P), ("both", C & P), ("Commission only", C & ~P), ("proposed only", ~C & P)):
        say(f"  {name}: {int(m.sum())} products, extra-EU imports EUR {d.loc[m, 'imports_extra'].sum() / 1e9:,.1f} bn, imports from the largest supplier EUR {d.loc[m, 'sup_imports'].sum() / 1e9:,.1f} bn")
    co = d[C & ~P]
    say(f"  Commission-only products: median exposure {co.exposure.median():.2f}; median EU share of supply {co.eu_share.median():.2f}")
    po = d[~C & P]
    say(f"  proposed-only products: failing HHI {int((po.hhi <= HHI_MIN).sum())}, importance {int((po.importance <= IMPORTANCE_MIN).sum())}, substitutability {int((po.substitutability <= SUBST_MIN).sum())} (of {len(po)})")
    for name, m in (("proposed", P), ("Commission", C)):
        g = d[m].groupby("top_partner").agg(n=("prodcom", "size"), v=("sup_imports", "sum")).sort_values("v", ascending=False)
        g.to_csv(RES / f"supply_{name.lower()}_by_supplier.csv", float_format="%.6g")
        say(f"  {name} screen by largest supplier (products; EUR bn from that supplier): " + "; ".join(f"{NAMES.get(k, k)} {int(r.n)} ({r.v / 1e9:,.1f})" for k, r in g.head(8).iterrows()))

    say("\nSensitivity of the proposed screen to its threshold")
    rows = []
    for thr in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60):
        m = d.exposure > thr
        rows.append({"threshold": thr, "products": int(m.sum()), "supplier_imports_eur": float(d.loc[m, "sup_imports"].sum())})
    S = pd.DataFrame(rows)
    S.to_csv(RES / "supply_thresholds.csv", index=False, float_format="%.6g")
    say("  " + "; ".join(f"{r.threshold:.0%}: {int(r.products)} products (EUR {r.supplier_imports_eur / 1e9:,.0f} bn)" for r in S.itertuples()))

    say("\nThe trade-only stand-in (largest supplier's share of all EU imports) against supply-based exposure")
    tp = d.exposure_trade > E_MIN
    say(f"  Spearman {d.exposure_trade.rank().corr(d.exposure.rank()):.3f}; E > 25%: {int(P.sum())}; trade-only > 25%: {int(tp.sum())}; both {int((P & tp).sum())}; E only {int((P & ~tp).sum())}; trade-only only {int((~P & tp).sum())}")

    say("\nWhere trade circulates inside the single market: by NACE division (first two digits of the Prodcom code), divisions with at least 40 products")
    d["division"] = d.prodcom.str[:2]
    dv = d.groupby("division").agg(n=("prodcom", "size"), intra_over_prod=("imports_intra", "sum"), prod=("PRODVAL", "sum"), exports_over_prod_share=("prodcom", "size"))
    rows = []
    for div, g2 in d.groupby("division"):
        if len(g2) < 40:
            continue
        rows.append({"division": div, "products": len(g2), "median_intra_over_production": float((g2.imports_intra / g2.PRODVAL).median()), "share_exports_above_production": float(((g2.exports_extra + g2.exports_intra) > g2.PRODVAL).mean()), "flagged_proposed": int(g2.proposed.sum()), "flagged_commission": int(g2.commission.sum())})
    DV = pd.DataFrame(rows).sort_values("share_exports_above_production", ascending=False)
    DV.to_csv(RES / "supply_divisions.csv", index=False, float_format="%.6g")
    for r in DV.itertuples():
        say(f"  {r.division}: {r.products} products; median intra-EU imports / production {r.median_intra_over_production:.2f}; total exports above production {r.share_exports_above_production:.0%}; flagged: supply-based {r.flagged_proposed}, Commission {r.flagged_commission}")

    say("\nHow concentrated the value at stake is (imports from the largest supplier, products flagged by the supply-based screen)")
    vs = d[P].sup_imports.sort_values(ascending=False)
    for k in (10, 20, 50, 100, 200):
        say(f"  largest {k}: {vs.head(k).sum() / vs.sum():.0%} of EUR {vs.sum() / 1e9:,.0f} bn")

    say("\nA second outside source (second-largest outside supplier's share of EU supply), products flagged by the supply-based screen")
    f = d[P]
    say(f"  second supplier below 5% of EU supply: {int((f.exposure_second < 0.05).sum())} of {len(f)} ({(f.exposure_second < 0.05).mean():.0%}); below 10%: {int((f.exposure_second < 0.10).sum())}; EU production below 10% of supply as well: {int(((f.exposure_second < 0.05) & (f.eu_share < 0.10)).sum())}")
    sp = f[(f.exposure_second < 0.05) & (f.eu_share < 0.10)].sort_values("sup_imports", ascending=False)
    sp.to_csv(RES / "supply_single_points.csv", index=False, float_format="%.6g")
    say("  largest by value among those with neither a second outside source above 5% nor EU production above 10%: " + "; ".join(f"{str(r.label)[:40]} ({NAMES.get(r.top_partner, r.top_partner)} {r.exposure:.0%}, EUR {r.sup_imports / 1e9:,.1f} bn)" for _, r in sp.head(8).iterrows()))

    say("\nWorked examples")
    co_big = d[C & ~P].sort_values("sup_imports", ascending=False).head(1).prodcom.tolist()
    for code in ["26201100", "26302210", "27202430", "30313410", "26112230"] + co_big:
        r = d[d.prodcom == code]
        if not len(r):
            continue
        r = r.iloc[0]
        say(
            f"  {code} {str(r.label)[:60]}: production EUR {r.PRODVAL / 1e9:,.2f} bn; extra-EU imports EUR {r.imports_extra / 1e9:,.2f} bn ({NAMES.get(r.top_partner, r.top_partner)} {r.top_share:.0%}, second {r.second_share:.0%}); "
            f"intra-EU imports EUR {r.imports_intra / 1e9:,.2f} bn; exports extra EUR {r.exports_extra / 1e9:,.2f} bn, intra EUR {r.exports_intra / 1e9:,.2f} bn; "
            f"HHI {r.hhi:.2f}, importance {r.importance:.2f}, ratio {r.substitutability:.2f} -> Commission {'yes' if r.commission else 'no'}; "
            f"outside share of supply {r.outside_share:.0%}, exposure {r.exposure:.0%} -> supply-based {'yes' if r.proposed else 'no'}"
        )

    say("\nHighest exposure, products with extra-EU imports above EUR 200 m")
    top = d[d.imports_extra > 2e8].sort_values("exposure", ascending=False).head(25)
    top.to_csv(RES / "supply_top.csv", index=False, float_format="%.6g")
    for _, r in top.iterrows():
        say(f"  {r.prodcom} {str(r.label)[:70]}: {NAMES.get(r.top_partner, r.top_partner)} {r.exposure:.0%} of EU supply (outside share {r.outside_share:.0%}, HHI {r.hhi:.2f}); imports EUR {r.imports_extra / 1e9:,.2f} bn; Commission {'yes' if r.commission else 'no'}")
    say("\nLargest products by imports from the main supplier among those flagged by the proposed screen")
    big = d[P].sort_values("sup_imports", ascending=False).head(20)
    big.to_csv(RES / "supply_largest.csv", index=False, float_format="%.6g")
    for _, r in big.iterrows():
        say(f"  {r.prodcom} {str(r.label)[:70]}: {NAMES.get(r.top_partner, r.top_partner)} EUR {r.sup_imports / 1e9:,.2f} bn, {r.exposure:.0%} of EU supply; Commission {'yes' if r.commission else 'no'} (HHI {r.hhi:.2f}, importance {r.importance:.2f}, ratio {r.substitutability:.2f})")
    d.to_csv(RES / "supply_products.csv", index=False, float_format="%.6g")

    (RES / "supply_summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
