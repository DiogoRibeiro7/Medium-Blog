"""Part 3 of the taxation-efficiency-drug-prices series: how Portugal's SNS sets the price of a medicine.

Every number quoted in the article is printed by this script. Run, after part3/fetch.py has put the
public files in part3/data (the GitHub Actions workflow does this and commits them):

    python reference_numbers.py > output.txt

Portugal does not publish a price list of the medicines it pays for, so the script rebuilds the rule
that sets those prices (erp.py) and feeds it the public price lists of the four reference countries:
Spain's nomenclator, Italy's class A list, Belgium's SAM database and France's public medicines
database (for France the manufacturer price is backed out of the retail price). France's 2024 sales
(Open Medic) weight the medicines; Portugal does not publish sales by presentation either.

Sections
1. Data and coverage: each country's file, the basket of medicines (centrally authorised brands with
   no generic competition in France, tablets and capsules sold in pharmacies) and how much of it each
   reference country prices.
2. The rule applied: the Portuguese ceiling against each reference country and against France, with
   a bootstrap over molecules, the spread across countries and the largest medicines.
3. The choice of countries: the ceiling under other rules (median, lowest, highest) and other sets
   of reference countries.
4. From the ceiling to the pharmacy: retail prices, margins, and the share of the basket that the
   2026 annual revision leaves alone (retail price up to 30 euros).
5. A check against Portuguese prices: the original brands in Infarmed's list of homogeneous groups,
   the only Portuguese prices published presentation by presentation, against the rebuilt ceiling.
6. Exceptional revisions: the cases where the Ministry of Health set prices above the rule, 2025-2026.
7. The ambulatory market: retail value, packs and the average price per pack, from the SNS
   transparency portal.
"""

from __future__ import annotations

import gzip
import itertools
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import countries as C  # noqa: E402
import erp  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
DATA = HERE / "data"
SEED = 20261112
THRESHOLD = 0.05          # generic share of boxes in France below which a molecule counts as protected
REF = ("FR", "ES", "IT", "BE")
NAMES = {"FR": "France", "ES": "Spain", "IT": "Italy", "BE": "Belgium"}

pd.set_option("display.width", 200)


def section(title: str) -> None:
    print()
    print("=" * 94)
    print(title)
    print("=" * 94)


def index(d: pd.DataFrame, col: str, base: str = "FR", w: str = "q") -> float:
    """Ratio of what the French quantities cost at `col` prices to what they cost at `base` prices,
    over the presentations that have both."""
    s = d[d[col].notna() & d[base].notna()]
    return float((s[w] * s[col]).sum() / (s[w] * s[base]).sum())


def boot(d: pd.DataFrame, col: str, rng, reps: int = 2000, base: str = "FR") -> tuple[float, float]:
    """95 percent interval of index() resampling molecules (all presentations of a molecule together)."""
    s = d[d[col].notna() & d[base].notna()]
    mols = s["molecule"].unique()
    num = s.groupby("molecule").apply(lambda g: (g["q"] * g[col]).sum(), include_groups=False)
    den = s.groupby("molecule").apply(lambda g: (g["q"] * g[base]).sum(), include_groups=False)
    out = []
    for _ in range(reps):
        pick = rng.choice(mols, size=len(mols), replace=True)
        out.append(num.loc[pick].sum() / den.loc[pick].sum())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def strength_text(s: tuple) -> str:
    return " + ".join(f"{x:g} mg" for x in s)


# ------------------------------------------------------------------------------------------ basket
def build_basket(fr: pd.DataFrame) -> pd.DataFrame:
    cis = pd.read_csv(C.PART2 / "bdpm_cis.csv.gz", dtype=str)[["cis", "type_procedure"]]
    cip = pd.read_csv(C.PART2 / "bdpm_cis_cip.csv.gz", dtype=str)[["cis", "cip13"]]
    fr = fr.merge(cip, on="cip13", how="left").merge(cis, on="cis", how="left")
    gen = fr.groupby("molecule").apply(
        lambda d: d.loc[d["segment"] == "generic", "boites"].sum() / d["boites"].sum(), include_groups=False)
    protected = set(gen[gen < THRESHOLD].index)
    fr["protected"] = fr["molecule"].isin(protected)
    fr["central"] = fr["type_procedure"].eq("Procédure centralisée")
    n_sub = fr["molecule"].str.count(r"\+") + 1
    fr["clean_strength"] = fr["strengths"].map(len) == n_sub
    return fr


def apply_rule(targets: pd.DataFrame, tables: dict) -> pd.DataFrame:
    rows = []
    for _, t in targets.iterrows():
        rec = {"cip13": t["cip13"], "brand": t["brand"], "molecule": t["molecule"], "strengths": t["strengths"],
               "units": int(t["units"]), "q": float(t["boites"]), "FR": float(t["pva"]),
               "FR_kind": "identical", "label": t["label"]}
        for k in ("ES", "IT", "BE"):
            m = erp.reference_price(tables[k], t["brand"], t["strengths"], int(t["units"]))
            rec[k] = m[0] if m else np.nan
            rec[f"{k}_kind"] = m[1] if m else ""
        rows.append(rec)
    r = pd.DataFrame(rows)
    r["n_ref"] = r[list(REF)].notna().sum(axis=1)
    r["cap"] = r.apply(lambda x: erp.pt_cap([x[k] for k in REF]), axis=1)
    r["cap_min"] = r[list(REF)].min(axis=1)
    r["cap_pvp"] = r["cap"].map(erp.pt_pvp)
    r["fr_pvp_equiv"] = r["FR"].map(erp.pt_pvp)
    return r


def main() -> None:
    rng = np.random.default_rng(SEED)
    figure: dict = {}

    # ---------------------------------------------------------------------------------------- 1
    section("1. Data and coverage")
    fr_all = C.france()
    tables = {"ES": C.spain(), "IT": C.italy(), "BE": C.belgium()}
    print("Reference-country files, tablets and capsules with a price (one row per brand, strength and pack):")
    print(f"  France   BDPM presentations sold in 2024 (Open Medic): {len(fr_all):6d}, brands {fr_all['brand'].nunique():5d}")
    for k, t in tables.items():
        print(f"  {NAMES[k]:8s} {len(t):6d} presentations, brands {t['brand'].nunique():5d}")

    # France: how well the published retail prices invert to whole-cent manufacturer prices.
    exact = np.isclose(erp.fr_ppttc(fr_all["pva"].to_numpy()), fr_all["prix_1"].to_numpy(), atol=1e-9)
    cents = np.isclose(fr_all["pva"] * 100, np.round(fr_all["pva"] * 100), atol=1e-6)
    orig = (fr_all["segment"] == "originator").to_numpy()
    print("\nFrance, manufacturer price (PFHT) backed out of the retail price before the dispensing fee:")
    print(f"  wholesale margin {erp.FR_WHOLESALE[0]:.2%} (min {erp.FR_WHOLESALE[1]:.2f}, max {erp.FR_WHOLESALE[2]:.2f} euros),"
          f" pharmacy margin by band {[(b, r) for _, b, r in erp.FR_MDL]}, VAT {erp.FR_VAT - 1:.1%}")
    print(f"  presentations whose price comes back exactly from a whole-cent PFHT: originals "
          f"{(cents & exact)[orig].mean():.3f}, all {(cents & exact).mean():.3f}")
    # The same test with the wholesale margin in force until the 2010s (6.68 percent, cap 30 euros).
    saved = erp.FR_WHOLESALE
    erp.FR_WHOLESALE = (0.0668, 0.30, 30.0)
    alt = erp.fr_pfht(fr_all["prix_1"].to_numpy())
    exact_alt = np.isclose(erp.fr_ppttc(alt), fr_all["prix_1"].to_numpy(), atol=1e-9) & \
        np.isclose(alt * 100, np.round(alt * 100), atol=1e-6)
    erp.FR_WHOLESALE = saved
    print(f"  the same with the older wholesale margin (6.68 percent, cap 30 euros): originals {exact_alt[orig].mean():.3f}")

    fr = build_basket(fr_all)
    oral_value = float((fr["boites"] * fr["pva"]).sum())
    prot = fr[fr["protected"]]
    targets = fr[fr["protected"] & fr["central"] & (fr["segment"] == "originator") & (fr["boites"] > 0)
                 & fr["clean_strength"]].copy()
    print("\nBasket: originator brands authorised by the European Commission (centralised procedure), whose")
    print(f"molecule has generics on less than {THRESHOLD:.0%} of French boxes, tablets and capsules sold in French pharmacies in 2024.")
    print(f"  presentations {len(targets)}, brands {targets['brand'].nunique()}, molecules {targets['molecule'].nunique()}")
    tv = float((targets["boites"] * targets["pva"]).sum())
    print(f"  value at manufacturer prices {tv / 1e9:.2f} bn euros: {tv / oral_value:.3f} of French oral-solid sales,"
          f" {tv / float((prot['boites'] * prot['pva']).sum()):.3f} of the protected molecules")

    r = apply_rule(targets, tables)
    r["v"] = r["q"] * r["FR"]
    print("\nCoverage of the basket by each reference country (same brand; share of presentations and of value):")
    for k in ("ES", "IT", "BE"):
        kinds = r[f"{k}_kind"].value_counts()
        print(f"  {NAMES[k]:8s} presentations {r[k].notna().mean():.3f}  value {r.loc[r[k].notna(), 'v'].sum() / r['v'].sum():.3f}"
              f"   identical pack {kinds.get('identical', 0):3d}, other pack {kinds.get('same strength', 0):3d},"
              f" other strength {kinds.get('other strength', 0):3d}")
    print("  number of reference countries pricing each presentation (France included), share of value:")
    for n in (1, 2, 3, 4):
        s = r[r["n_ref"] == n]
        print(f"    {n}: presentations {len(s):3d}, value {s['v'].sum() / r['v'].sum():.3f}")
    only_fr = r[r["n_ref"] == 1].groupby("brand")["v"].sum().sort_values(ascending=False)
    print("  largest brands that only France prices among the four (dispensed by hospitals elsewhere, or not sold):")
    print("    " + ", ".join(only_fr.index[:12]))

    # ---------------------------------------------------------------------------------------- 2
    section("2. The rule applied: the Portuguese ceiling against the reference countries")
    # Spain's and Italy's lists cover medicines dispensed by community pharmacies. A brand that
    # neither lists is, in those countries and in Portugal, mostly dispensed by hospital pharmacies
    # (the HIV combinations, the JAK inhibitors), where Portugal's rule is the lowest reference price
    # and the Spanish and Italian prices are not public; those are left out of the ambulatory rule.
    m = r[r["ES"].notna() | r["IT"].notna()].copy()
    four = r[r["n_ref"] == 4].copy()
    hosp = r[r["ES"].isna() & r["IT"].isna()]
    print(f"Pharmacy medicines (priced by France and by Spain or Italy): {len(m)} presentations "
          f"({m['molecule'].nunique()} molecules, {m['v'].sum() / r['v'].sum():.3f} of basket value)")
    print(f"  of which priced by all four: {len(four)} ({four['molecule'].nunique()} molecules, {four['v'].sum() / r['v'].sum():.3f} of value)")
    print(f"Priced by neither Spain nor Italy: {len(hosp)} presentations, {hosp['v'].sum() / r['v'].sum():.3f} of value;"
          f" largest: {', '.join(hosp.groupby('brand')['v'].sum().sort_values(ascending=False).index[:8])}")
    print("\nPrice level relative to France at French quantities (95% interval resampling molecules):")
    res = {}
    for label, d in (("pharmacy medicines", m), ("all four countries", four)):
        print(f"  {label}:")
        for col in ("ES", "IT", "BE", "cap", "cap_min"):
            lo, hi = boot(d, col, rng)
            res[(label, col)] = index(d, col)
            name = {"cap": "Portuguese ceiling (average)", "cap_min": "lowest of the countries"}.get(col, NAMES.get(col))
            print(f"    {name:30s} {index(d, col):6.3f}  [{lo:5.3f}, {hi:5.3f}]  n {int(d[col].notna().sum()):3d}")
    ratio = (m["cap"] / m["FR"])
    print(f"\nPresentation by presentation (pharmacy medicines): median ceiling/France {ratio.median():.3f}; "
          f"ceiling above France {(ratio > 1.005).mean():.3f}, within half a percent {(ratio.between(0.995, 1.005)).mean():.3f},"
          f" below {(ratio < 0.995).mean():.3f}")
    rel = four[list(REF)].div(four["FR"], axis=0)
    lowest = rel.idxmin(axis=1).value_counts(normalize=True)
    highest = rel.idxmax(axis=1).value_counts(normalize=True)
    print("Which country is cheapest and dearest, all-four presentations (share of presentations):")
    for k in REF:
        print(f"  {NAMES[k]:8s} cheapest {lowest.get(k, 0):.3f}  dearest {highest.get(k, 0):.3f}")
    spread = four[list(REF)].max(axis=1) / four[list(REF)].min(axis=1)
    print(f"Highest over lowest reference price, same presentation: median {spread.median():.2f}, "
          f"quartiles {spread.quantile(.25):.2f}-{spread.quantile(.75):.2f}, max {spread.max():.2f}")
    print(f"Ceiling over the lowest reference price: value-weighted {index(four, 'cap', base='cap_min'):.3f}")

    # Largest molecules by French value: pack prices for the most sold presentation of each.
    print("\nLargest molecules in the basket by French value (most sold presentation; ex-factory euros per pack):")
    top = (m.sort_values("q", ascending=False).drop_duplicates("molecule")
             .assign(mv=lambda d: d["molecule"].map(m.groupby("molecule")["v"].sum()))
             .sort_values("mv", ascending=False).head(15))
    for _, x in top.iterrows():
        vals = "  ".join(f"{k} {x[k]:8.2f}" if np.isfinite(x[k]) else f"{k}      n/a" for k in REF)
        print(f"  {x['brand']:12s} {strength_text(x['strengths']):18s} x{x['units']:<4d} {vals}   ceiling {x['cap']:8.2f}"
              f"  ceiling/FR {x['cap'] / x['FR']:5.3f}")
    figure["molecules"] = [
        {"brand": x["brand"], "molecule": x["molecule"], "pack": f"{strength_text(x['strengths'])} x{x['units']}",
         **{k: (x[k] / x["FR"] if np.isfinite(x[k]) else None) for k in ("ES", "IT", "BE")},
         "cap": x["cap"] / x["FR"]}
        for _, x in four.sort_values("q", ascending=False).drop_duplicates("molecule")
        .assign(mv=lambda d: d["molecule"].map(four.groupby("molecule")["v"].sum()))
        .sort_values("mv", ascending=False).iterrows()]

    # ---------------------------------------------------------------------------------------- 3
    section("3. The choice of countries")
    print(f"All-four presentations (n {len(four)}), French quantities. Ceiling under other rules, relative to France")
    print("and to the rule in force (average of the four):")
    base = index(four, "cap")
    rules = {}
    rules["average of the four (in force)"] = four["cap"]
    rules["median of the four"] = four[list(REF)].apply(lambda x: erp.pt_cap(x, "median"), axis=1)
    rules["lowest of the four (hospital rule)"] = four[list(REF)].min(axis=1)
    rules["highest of the four"] = four[list(REF)].max(axis=1)
    for drop in REF:
        keep = [k for k in REF if k != drop]
        rules[f"average without {NAMES[drop]}"] = four[keep].apply(lambda x: erp.pt_cap(x), axis=1)
    for pair in itertools.combinations(REF, 2):
        rules[f"average of {NAMES[pair[0]]} and {NAMES[pair[1]]}"] = four[list(pair)].apply(lambda x: erp.pt_cap(x), axis=1)
    for k in REF:
        rules[f"{NAMES[k]} alone"] = four[k]
    out = []
    for name, col in rules.items():
        d = four.assign(alt=col)
        out.append((name, index(d, "alt"), index(d, "alt") / base))
    for name, a, b in out:
        print(f"  {name:40s} vs France {a:6.3f}   vs rule in force {b:6.3f}")
    figure["rules"] = [{"rule": n, "vs_fr": a, "vs_rule": b} for n, a, b in out]
    lo = min(out, key=lambda x: x[1])
    hi = max(out, key=lambda x: x[1])
    print(f"Range across the rules and sets above: {lo[1]:.3f} ({lo[0]}) to {hi[1]:.3f} ({hi[0]})")

    # ---------------------------------------------------------------------------------------- 4
    section("4. From the ceiling to the pharmacy")
    m["margin_share"] = 1 - m["cap"] / m["cap_pvp"]
    w = m["q"] * m["cap_pvp"]
    print(f"Portuguese retail price at the ceiling for the basket ({len(m)} presentations):")
    print(f"  median retail price {m['cap_pvp'].median():.2f} euros; share of the retail price that is margins,"
          f" Infarmed's fee and VAT: median {m['margin_share'].median():.3f}, value-weighted {1 - (m['q'] * m['cap']).sum() / w.sum():.3f}")
    small = m["cap_pvp"] <= erp.PT_RAP_EXEMPT_PVP
    print(f"  the 2026 annual revision: retail prices up to {erp.PT_RAP_EXEMPT_PVP:.0f} euros are left alone and the others"
          f" are cut by at most {erp.PT_RAP_MAX_CUT:.0%} (Portaria 394/2025/1)")
    print(f"  retail price at or below {erp.PT_RAP_EXEMPT_PVP:.0f} euros (left alone by the 2026 annual revision): "
          f"presentations {small.mean():.3f}, value {(m.loc[small, 'q'] * m.loc[small, 'cap']).sum() / (m['q'] * m['cap']).sum():.3f}")
    for pva in (5, 20, 50, 100, 500):
        print(f"  ex-factory {pva:4d} euros -> retail {erp.pt_pvp(pva):8.2f} (margins, fee and VAT {1 - pva / erp.pt_pvp(pva):.3f} of it)")

    # ---------------------------------------------------------------------------------------- 5
    section("5. A check against Portuguese prices")
    pt = C.portugal_gh()
    print(f"Infarmed, homogeneous groups, third quarter of 2026: {len(pt)} presentations, {int(pt['oral_solid'].sum())} tablets"
          f" or capsules, {int(pt['parallel'].sum())} parallel imports")
    stems = set()
    for t in (*tables.values(), fr_all):
        stems |= {x.split(" ")[0] for x in t["molecule"].dropna() if x}
    orig_brands = set(fr_all.loc[fr_all["segment"] == "originator", "brand"])
    p = pt[pt["oral_solid"] & ~pt["parallel"] & pt["brand"].isin(orig_brands)
           & ~pt["brand"].map(lambda b: erp.stem_word(b) in stems or b in stems)].copy()
    p = p.drop_duplicates(["brand", "strengths", "units", "pvp_max"])
    ref_all = {"FR": fr_all, **tables}
    for k, t in ref_all.items():
        p[k] = [(lambda mm: mm[0] if mm else np.nan)(erp.reference_price(t, b, s, int(u)))
                for b, s, u in zip(p["brand"], p["strengths"], p["units"])]
    p["n_ref"] = p[list(REF)].notna().sum(axis=1)
    p["cap"] = p.apply(lambda x: erp.pt_cap([x[k] for k in REF]), axis=1)
    p["pva_max"] = p["pvp_max"].map(erp.pt_pva)
    p["ratio"] = p["pva_max"] / p["cap"]
    v = p[p["n_ref"] >= 3].copy()
    print(f"Original brands (a brand that France sells as an original, not a generic name), tablets and capsules: {len(p)}")
    print(f"  priced by at least three reference countries: {len(v)} presentations, {v['brand'].nunique()} brands")
    at = v["ratio"].between(0.995, 1.005)
    exact_all = [all(erp.reference_price(ref_all[k], b, s, int(u)) is None
                     or erp.reference_price(ref_all[k], b, s, int(u))[1] == "identical" for k in REF)
                 for b, s, u in zip(v["brand"], v["strengths"], v["units"])]
    v["identical"] = exact_all
    print(f"  Portuguese maximum ex-factory price over the rebuilt ceiling: median {v['ratio'].median():.3f}")
    print(f"    within half a percent of the ceiling {at.mean():.3f}; below {(v['ratio'] < 0.995).mean():.3f};"
          f" above {(v['ratio'] > 1.005).mean():.3f}")
    for label, d in (("retail above 30 euros", v[v["pvp_max"] > erp.PT_RAP_EXEMPT_PVP]),
                     ("retail 30 euros or less", v[v["pvp_max"] <= erp.PT_RAP_EXEMPT_PVP])):
        print(f"    {label:24s} n {len(d):3d}: at the ceiling {d['ratio'].between(0.995, 1.005).mean():.3f},"
              f" below {(d['ratio'] < 0.995).mean():.3f}, above {(d['ratio'] > 1.005).mean():.3f}, median {d['ratio'].median():.3f}")
    d = v[v["identical"] & (v["pvp_max"] > erp.PT_RAP_EXEMPT_PVP)]
    print(f"    retail above 30 euros, the same pack in every country that sells it, n {len(d)}: at the ceiling"
          f" {d['ratio'].between(0.995, 1.005).mean():.3f}, within 2 percent {d['ratio'].between(0.98, 1.02).mean():.3f}")
    over = v[(v["ratio"] > 1.005) & (v["pvp_max"] > erp.PT_RAP_EXEMPT_PVP)]
    print(f"  above the ceiling with retail above 30 euros: {', '.join(sorted(over['brand'].unique()))}")
    # The ceiling uses reference prices of late 2026; Portugal's prices were revised on 1 January 2026.
    # A generic that reached a reference market in between pulls the original's price there down.
    es = tables["ES"]
    first_gen = es[es["generic"]].groupby("molecule")["alta"].min()
    mol_of = fr_all.drop_duplicates("brand").set_index("brand")["molecule"]
    print("    first generic of the molecule in Spain's list, and the brand's ceiling/Portugal gap:")
    for b, g in over.groupby("brand"):
        mol = mol_of.get(b, "")
        when = first_gen.get(mol)
        print(f"      {b:10s} {mol:26s} first Spanish generic {str(when.date()) if when is not None and pd.notna(when) else 'none':10s}"
              f"  Portugal/ceiling median {g['ratio'].median():.2f}")
    under = v[(v["ratio"] < 0.995) & (v["pvp_max"] > erp.PT_RAP_EXEMPT_PVP)]
    print(f"  below the ceiling with retail above 30 euros: {', '.join(sorted(under['brand'].unique()))}")
    print("  examples within half a percent of the ceiling (Portugal's maximum retail price, the ex-factory price")
    print("  it implies, and the rebuilt ceiling):")
    for _, x in v[at].sort_values("pvp_max", ascending=False).drop_duplicates("brand").head(10).iterrows():
        print(f"    {x['name']:28s} {x['dose']:24s} x{x['units']:<4d} retail {x['pvp_max']:7.2f}  ex-factory {x['pva_max']:7.2f}"
              f"  ceiling {x['cap']:7.2f}")
    figure["check"] = [{"brand": x["brand"], "pva": x["pva_max"], "cap": x["cap"], "pvp": x["pvp_max"]}
                       for _, x in v.iterrows()]

    # ---------------------------------------------------------------------------------------- 6
    section("6. Exceptional price revisions decided by the Ministry of Health")
    t25 = gzip.decompress((DATA / "pt_rep_2025.txt.gz").read_bytes()).decode("utf-8")
    dec = re.findall(r"(Deferido|Indeferido)\s+(\d{2}/\d{2}/\d{4})\s+(\d+)\s*$", t25, flags=re.M)
    days = np.array([int(x[2]) for x in dec])
    print(f"2025 table: {len(dec)} decisions, granted {sum(x[0] == 'Deferido' for x in dec)}, refused "
          f"{sum(x[0] == 'Indeferido' for x in dec)}; days from request to decision: median {np.median(days):.0f},"
          f" quartiles {np.percentile(days, 25):.0f}-{np.percentile(days, 75):.0f}, max {days.max()}")
    t26 = gzip.decompress((DATA / "pt_rep_2026.txt.gz").read_bytes()).decode("utf-8")
    inc = re.findall(r"^(\d{7})\s+(.+?)\s{2,}(\S+).*?(Deferido|Indeferido)\s+\S+\s+([\d.,]+)\s*€\s+([\d.,]+)\s*€", t26, flags=re.M)
    print(f"2026 ambulatory table: {len(inc)} decisions")
    for reg, name, dci, d, a, b in inc:
        a, b = float(a.replace(",", ".")), float(b.replace(",", "."))
        print(f"  {name.strip():16s} {dci:12s} {d:9s} retail {a:6.2f} -> {b:6.2f} euros ({b / a - 1:+.0%})")

    # ---------------------------------------------------------------------------------------- 7
    section("7. The Portuguese ambulatory market (SNS transparency portal)")
    pm = pd.read_csv(DATA / "pt_transparencia_preco_medio_por_embalagem.csv.gz")
    ds = pd.read_csv(DATA / "pt_transparencia_despesa_com_medicamentos_no_ambulatorio_sns.csv.gz")
    for year in (2024, 2025):
        a = pm[pm["tempo"].str.startswith(str(year))]
        b = ds[ds["tempo"].str.startswith(str(year))]
        val, packs = a["valor_pvp_ambulatorio"].sum(), a["n_o_de_embalagens_ambulatorio"].sum()
        gv, gp = a["valor_pvp_genericos"].sum(), a["n_o_de_embalagens_genericos"].sum()
        print(f"  {year}: retail value {val / 1e9:.2f} bn euros, {packs / 1e6:.0f} million packs, {val / packs:.2f} euros per pack;"
              f" generics {gp / packs:.3f} of packs and {gv / val:.3f} of value ({gv / gp:.2f} euros per pack, others"
              f" {(val - gv) / (packs - gp):.2f}); SNS share of retail value {b['encargos_sns_ambulatorio'].sum() / b['valor_pvp_ambulatorio'].sum():.3f}")

    with open(HERE / "figure_data.json", "w") as fh:
        json.dump(figure, fh, indent=0)


if __name__ == "__main__":
    main()
