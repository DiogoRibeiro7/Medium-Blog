"""Part 2 of the taxation-efficiency-drug-prices series: how much more the US pays for medicines.

Every number quoted in the article is printed by this script. Run, after part2/fetch.py has put the
public files in part2/data (the GitHub Actions workflow does this and commits them):

    python prices_numbers.py > output.txt

The comparison is between Medicare Part D in the United States and the French national health
insurance (Assurance Maladie) in 2024, for tablets and capsules dispensed by retail pharmacies,
molecule by molecule. France stands in for the European single buyer because it publishes, at the
level of each presentation, how many boxes were reimbursed and at what price (Open Medic), which
Portugal's SNS does not.

Sections
1. Data and coverage: what each file contains and how much of each market the matched basket holds.
2. The naive comparison: brand against brand and generic against generic, and why it misleads.
3. Molecule baskets: single-source, multi-source and mixed molecules; four index formulas; how much
   the index depends on which molecules are in the basket; the largest molecules; what the US basket
   would have cost at French prices.
4. Sensitivity: the French dispensing fee, the exchange rate, the basket threshold, the oral-solid
   classification, outliers, the largest molecules, acquisition prices (NADAC) instead of
   reimbursement amounts, and confidential rebates on both sides.
5. Composition: an all-drug index as a function of the generic share of units.
6. Medicare negotiation: the negotiated prices of the oral drugs selected for 2026 and 2027 against
   2024 acquisition prices and against current French prices.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import drugprices as dp  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SEED = 20261112
THRESHOLD = 0.05            # generic share of units that separates protected from competed molecules

# Rebate scenarios, as a share of gross spending on brand-name drugs.
# US: CBO (2018, Options for Reducing the Deficit, option 54735): manufacturers' rebates were about 22
# percent of gross Part D spending on all brand-name drugs in 2015; CBO (2022, Prescription Drugs:
# Spending, Use, and Prices): rebates and discounts rose from 15 to 35 percent of Part D spending on
# brand-name drugs at retail prices between 2009 and 2018; RAND (Mulcahy et al. 2024) adjusts US
# retail brand prices down by 37.2 percent.
# France: DREES (Les Dossiers de la DREES 139, 2026): 41.07 billion euros of gross spending on
# reimbursable medicines in 2024 and 32.03 billion after product rebates, 22 percent over all drugs
# and therefore more on the patented ones, which receive almost all of them. 0.35 is a high case.
REBATES_US = {"none": 0.0, "CBO 2015": 0.22, "CBO 2018": 0.35, "RAND 2024": 0.372}
REBATES_FR = {"none": 0.0, "DREES 2024 all drugs": 0.22, "high": 0.35}

pd.set_option("display.width", 200)


def section(title: str) -> None:
    print()
    print("=" * 94)
    print(title)
    print("=" * 94)


def fx(year: int) -> float:
    e = pd.read_csv(DATA / "ecb_eurusd.csv.gz", dtype={"TIME_PERIOD": str})
    return float(e.loc[e["TIME_PERIOD"] == str(year), "OBS_VALUE"].iloc[0])


def fmt_idx(d: dict) -> str:
    return (f"n {d['n']:4d}  Laspeyres {d['L']:6.3f}  Paasche {d['P']:6.3f}  Fisher {d['F']:6.3f}  "
            f"Jevons {d['J']:6.3f}  median {d['median']:6.3f}")


def bn(x: float) -> str:
    return f"{x / 1e9:7.2f}"


def baskets_of(us_rows, fr_rows, usd, threshold=THRESHOLD, include_fee=True):
    return dp.molecule_baskets(dp.cells_us(us_rows), dp.cells_fr(fr_rows, include_fee=include_fee), usd, threshold)


def main() -> None:
    print("Part 2: how much more the US pays for medicines, and how the answer depends on the method.")
    print(f"pandas {pd.__version__}, NumPy {np.__version__}")
    figure: dict = {}
    usd24, usd25 = fx(2024), fx(2025)

    # ------------------------------------------------------------------------------------- 1
    section("1. Data and coverage")
    us = dp.us_tables(DATA)
    fr = dp.fr_tables(DATA)
    u = us.rows
    print("Medicare Part D Spending by Drug, 2024 (CMS). Gross spending: ingredient cost, dispensing fees and")
    print("sales tax, paid by plans and beneficiaries, before manufacturer rebates.")
    print(f"  brand-name rows (manufacturer 'Overall') {us.totals['rows']}; spending ${us.totals['spend'] / 1e9:.1f} bn; "
          f"dosage units {us.totals['units'] / 1e9:.1f} bn; claims {us.totals['claims'] / 1e6:.0f} m")
    share = u.groupby("fda_match")["Tot_Spndng_2024"].sum() / us.totals["spend"]
    print("  FDA NDC lookup, share of spending matched by brand name {:.3f}, by molecule {:.3f}, unmatched {:.3f}".format(
        share.get("brand name", 0), share.get("molecule", 0), share.get("none", 0)))
    su = u.groupby("segment")[["Tot_Dsg_Unts_2024", "Tot_Spndng_2024"]].sum()
    su = su / su.sum()
    for seg in ("originator", "generic", "other"):
        print(f"  {seg:10s} share of units {su.loc[seg, 'Tot_Dsg_Unts_2024']:.3f}  share of spending "
              f"{su.loc[seg, 'Tot_Spndng_2024']:.3f}")
    print("  (RAND, Mulcahy et al. 2024, all US channels in 2022: brand-name originators 7 percent of volume and")
    print("   87 percent of spending; unbranded generics 90 percent of volume.)")
    oral = u[u["oral_solid"]]
    print(f"  oral solid rows (FDA share of tablet or capsule listings >= 0.75): spending "
          f"${oral['Tot_Spndng_2024'].sum() / 1e9:.1f} bn ({oral['Tot_Spndng_2024'].sum() / us.totals['spend']:.3f}), "
          f"units {oral['Tot_Dsg_Unts_2024'].sum() / us.totals['units']:.3f} of all units")

    f = fr.rows
    print()
    print("Open Medic 2024 (Assurance Maladie), every reimbursed presentation dispensed by retail pharmacies,")
    print("joined to the public drug database (BDPM) for composition, form and the dispensing fee per box.")
    print(f"  presentations {fr.totals['presentations']}; boxes {fr.totals['boxes'] / 1e9:.3f} bn; reimbursement base "
          f"(BSE) EUR {fr.totals['bse'] / 1e9:.2f} bn; amount reimbursed EUR {fr.totals['rem'] / 1e9:.2f} bn")
    print(f"  joined to BDPM: boxes {f['boites'].sum() / fr.totals['boxes']:.3f}, BSE {f['bse'].sum() / fr.totals['bse']:.3f}")
    fo = f[f["oral_solid"] & (f["units"] > 0)]
    print(f"  oral solid with a known unit count: boxes {fo['boites'].sum() / fr.totals['boxes']:.3f}, BSE "
          f"{fo['bse'].sum() / fr.totals['bse']:.3f}, EUR {fo['bse'].sum() / 1e9:.2f} bn")
    seg = fo.assign(q=fo["boites"] * fo["units"]).groupby("segment")[["q", "bse"]].sum()
    seg = seg / seg.sum()
    print(f"  generic share of oral solid units {seg.loc['generic', 'q']:.3f}, of BSE {seg.loc['generic', 'bse']:.3f}")
    # The BSE is the price before the dispensing fee: per box it sits on the BDPM price without the fee.
    chk = f[(f["boites"] > 5000) & f["prix_1"].notna()]
    d = chk["bse"] / chk["boites"] - chk["prix_1"]
    print(f"  check, presentations with more than 5000 boxes ({len(chk)}): BSE per box minus current price without")
    print(f"  fee, median EUR {d.median():.3f}; minus the price with the fee, median EUR "
          f"{(chk['bse'] / chk['boites'] - chk['prix_2']).median():.3f}; the BSE excludes the fee.")
    print(f"  dispensing fee per box (BDPM third price field): median EUR {fo['fee_box'].median():.2f}")

    m = baskets_of(u, f, usd24)
    print()
    print(f"Exchange rate: ECB reference rate, annual average, 2024 USD {usd24:.5f} per EUR; 2025 {usd25:.5f}.")
    print(f"Matched molecules (oral solid in both, after adding brands and generics): {len(m)}")
    print(f"  US spending covered ${m['v_us'].sum() / 1e9:.1f} bn: {m['v_us'].sum() / oral['Tot_Spndng_2024'].sum():.3f} "
          f"of oral solid spending, {m['v_us'].sum() / us.totals['spend']:.3f} of all Part D spending")
    print(f"  French spending covered EUR {m['v_fr'].sum() / 1e9:.2f} bn with fees "
          f"(BSE EUR {m['v_fr_bse'].sum() / 1e9:.2f} bn): {m['v_fr_bse'].sum() / fo['bse'].sum():.3f} of oral solid BSE")
    print(f"  generic share of matched units: US {(m['gen_us'] * m['q_us']).sum() / m['q_us'].sum():.3f}, "
          f"France {(m['gen_fr'] * m['q_fr']).sum() / m['q_fr'].sum():.3f}")

    # ------------------------------------------------------------------------------------- 2
    section("2. The naive comparison: brand with brand, generic with generic")
    ms = dp.match(dp.cells_us(u), dp.cells_fr(f), usd24)
    figure["naive"] = {}
    for s_ in ("originator", "generic"):
        x = ms[ms["segment"] == s_]
        figure["naive"][s_] = dp.indices(x)
        print(f"  {s_:10s} {fmt_idx(figure['naive'][s_])}")
    figure["naive"]["all"] = dp.indices(ms)
    print(f"  all        {fmt_idx(figure['naive']['all'])}")
    x = ms[ms["segment"] == "originator"].copy()
    x["paasche_num"] = x["q_fr"] * x["p_us"]
    tot = x["paasche_num"].sum()
    print("  Largest contributions to the numerator of the originator Paasche index (French volumes at US prices):")
    brands = (u[(u["segment"] == "originator") & u["oral_solid"]].sort_values("Tot_Spndng_2024", ascending=False)
              .groupby("molecule")["Brnd_Name"].first())
    for _, r in x.sort_values("paasche_num", ascending=False).head(6).iterrows():
        gen = m.loc[m["molecule"] == r["molecule"], "gen_us"]
        g = f"{gen.iloc[0]:.3f}" if len(gen) else "  n/a"
        print(f"    {r['molecule']:14s} ({brands.get(r['molecule'], '?'):10s}) share {r['paasche_num'] / tot:.3f}  US ${r['p_us']:8.3f}"
              f"  FR ${r['p_fr_usd']:7.3f}  ratio {r['ratio']:7.1f}  US units {r['q_us'] / 1e6:6.2f} m  FR units "
              f"{r['q_fr'] / 1e6:7.1f} m  US generic share of the molecule {g}")
    print("  A brand that lost its patent long ago sells a few units in the US at a high price, and many units")
    print("  in France at a low one; weighting it by French volumes multiplies the old US brand price by them.")

    # ------------------------------------------------------------------------------------- 3
    section("3. Molecule baskets")
    print(f"Basket by the generic share of units in each country (threshold {THRESHOLD:.2f}):")
    rng = np.random.default_rng(SEED)
    figure["baskets"] = {}
    for b in dp.BASKETS:
        x = m[m["basket"] == b]
        ix = dp.indices(x)
        ci = dp.bootstrap(x, rng)
        print(f"  {b:14s} {fmt_idx(ix)}")
        print(f"  {'':14s} US ${bn(x['v_us'].sum())} bn ({x['v_us'].sum() / m['v_us'].sum():.3f}), France EUR "
              f"{bn(x['v_fr'].sum())} bn ({x['v_fr'].sum() / m['v_fr'].sum():.3f}); 95% bootstrap over molecules: "
              f"L {ci['L'][0]:.2f}-{ci['L'][1]:.2f}, P {ci['P'][0]:.2f}-{ci['P'][1]:.2f}, F {ci['F'][0]:.2f}-{ci['F'][1]:.2f}")
        figure["baskets"][b] = {**ix, "ci": ci, "v_us": float(x["v_us"].sum())}
    for side in ("generic in US only", "generic in France only"):
        x = m[m["mixed_side"] == side]
        figure["baskets"][f"mixed, {side}"] = {**dp.indices(x), "ci": dp.bootstrap(x, rng), "v_us": float(x["v_us"].sum())}
        print(f"  mixed, {side:22s} {fmt_idx(dp.indices(x))}  US ${bn(x['v_us'].sum())} bn")
    allm = dp.indices(m)
    figure["baskets"]["all"] = {**allm, "ci": dp.bootstrap(m, rng), "v_us": float(m["v_us"].sum())}
    print(f"  all matched    {fmt_idx(allm)}")

    print()
    print("Largest molecules by US spending (price per tablet or capsule; France with the dispensing fee):")
    for b, k in (("single-source", 15), ("multi-source", 10), ("mixed", 10)):
        print(f"  {b}")
        x = m[m["basket"] == b].sort_values("v_us", ascending=False).head(k)
        for _, r in x.iterrows():
            print(f"    {r['molecule']:28s} US ${r['v_us'] / 1e9:5.2f} bn  US ${r['p_us']:8.3f}  FR ${r['p_fr_usd']:8.3f}  "
                  f"ratio {r['ratio']:6.2f}  generic share US {r['gen_us']:.2f} FR {r['gen_fr']:.2f}")
    print("  Most extreme ratios in the single-source basket, with US units:")
    x = m[m["basket"] == "single-source"]
    for _, r in pd.concat([x.nlargest(4, "ratio"), x.nsmallest(3, "ratio")]).iterrows():
        print(f"    {r['molecule']:28s} ratio {r['ratio']:7.1f}  US units {r['q_us'] / 1e3:9.0f} k  FR units {r['q_fr'] / 1e3:9.0f} k")

    print("  Per unit and per day of treatment, tafamidis (20 mg capsules are taken four a day, 61 mg one a day):")
    tu = u[u["molecule"] == "tafamidis"].set_index("Brnd_Name")
    us20, us61 = tu.loc["Vyndaqel"], tu.loc["Vyndamax"]
    tf = f[f["molecule"] == "tafamidis"].assign(q=lambda d: d["boites"] * d["units"])
    f61 = tf[tf["denomination"].str.contains("61 mg")]
    f20 = tf[tf["denomination"].str.contains("20 mg")]
    share20_us = us20["Tot_Dsg_Unts_2024"] / (us20["Tot_Dsg_Unts_2024"] + us61["Tot_Dsg_Unts_2024"])
    share20_fr = f20["q"].sum() / tf["q"].sum()
    days_us = us20["Tot_Dsg_Unts_2024"] / 4 + us61["Tot_Dsg_Unts_2024"]
    days_fr = f20["q"].sum() / 4 + f61["q"].sum()
    fee = (tf["fee_box"] * tf["boites"]).sum()
    per_day_us = (us20["Tot_Spndng_2024"] + us61["Tot_Spndng_2024"]) / days_us
    per_day_fr = (tf["bse"].sum() + fee) / days_fr * usd24
    print(f"    share of capsules that are 20 mg: US {share20_us:.3f}, France {share20_fr:.3f}")
    print(f"    per day: US ${per_day_us:.2f}, France ${per_day_fr:.2f}, ratio {per_day_us / per_day_fr:.2f} "
          f"(per capsule: {m.loc[m['molecule'] == 'tafamidis', 'ratio'].iloc[0]:.2f})")

    print()
    print("The US basket at French prices (2024 Part D volumes, gross prices on both sides):")
    figure["counterfactual"] = {}
    for b in (*dp.BASKETS, "all"):
        x = m if b == "all" else m[m["basket"] == b]
        at_fr = float((x["q_us"] * x["p_fr_usd"]).sum())
        print(f"  {b:14s} actual ${bn(x['v_us'].sum())} bn, at French prices ${bn(at_fr)} bn, difference "
              f"${bn(x['v_us'].sum() - at_fr)} bn")
        figure["counterfactual"][b] = [float(x["v_us"].sum()), at_fr]

    keep = ["molecule", "basket", "mixed_side", "q_us", "v_us", "p_us", "q_fr", "p_fr_usd", "ratio", "gen_us", "gen_fr"]
    figure["molecules"] = m[keep].to_dict(orient="list")

    # ------------------------------------------------------------------------------------- 4
    section("4. Sensitivity")
    rows = []

    def show(label, mm, **kw):
        out = {}
        for b in ("single-source", "multi-source"):
            x = mm[mm["basket"] == b]
            out[b] = dp.indices(x, **kw)
        print(f"  {label:44s} single L {out['single-source']['L']:6.2f} P {out['single-source']['P']:6.2f}"
              f" (n {out['single-source']['n']:3d})   multi L {out['multi-source']['L']:5.2f} P "
              f"{out['multi-source']['P']:5.2f} (n {out['multi-source']['n']:3d})")
        rows.append({"label": label, **{f"{b}_{k}": v[k] for b, v in out.items() for k in ("L", "P", "F")}})

    show("main: France with fee, 2024 exchange rate", m)
    show("France without the dispensing fee", baskets_of(u, f, usd24, include_fee=False))
    show("2025 exchange rate", baskets_of(u, f, usd25))
    for t in (0.02, 0.10, 0.20):
        show(f"basket threshold {t:.2f}", baskets_of(u, f, usd24, threshold=t))
    for ot in (0.50, 0.95):
        show(f"US oral-solid threshold {ot:.2f}", baskets_of(dp.us_tables(DATA, oral_threshold=ot).rows, f, usd24))
    show("molecules with no other form in the FDA file", m[m["mol_oral_share"] >= 0.999])
    lo, hi = m["ratio"].quantile([0.025, 0.975])
    show("ratios in the 2.5-97.5 percentile range", m[m["ratio"].between(lo, hi)])
    for b in ("single-source", "multi-source"):
        top = m[m["basket"] == b].nlargest(3, "v_us")["molecule"].tolist()
        print(f"  three largest {b} molecules by US spending: {', '.join(top)}")
    top3 = m[m["basket"] == "single-source"].nlargest(3, "v_us")["molecule"].tolist()
    show("without the three largest single-source molecules", m[~m["molecule"].isin(top3)])
    figure["sensitivity"] = rows

    print()
    print("Acquisition prices: 2024 NADAC per unit (median across package codes; brand class for the")
    print("single-source basket, brand and generic mixed by Part D unit shares for the multi-source basket)")
    print("against the French price without the dispensing fee.")
    nd = dp.nadac_molecules(DATA)
    mn = baskets_of(u, f, usd24, include_fee=False).merge(nd, left_on="molecule", right_index=True, how="left")
    mn["p_nadac"] = np.where(mn["basket"] == "single-source", mn["nadac_b"].fillna(mn["nadac_g"]),
                             mn["gen_us"] * mn["nadac_g"].fillna(mn["nadac_b"])
                             + (1 - mn["gen_us"]) * mn["nadac_b"].fillna(mn["nadac_g"]))
    figure["nadac"] = {}
    for b in ("single-source", "multi-source"):
        x = mn[(mn["basket"] == b) & mn["p_nadac"].notna()]
        g = dp.indices(x)
        a = dp.indices(x, p_us="p_nadac")
        print(f"  {b:14s} molecules with a NADAC {len(x):3d} of {int((mn['basket'] == b).sum())}, US spending share "
              f"{x['v_us'].sum() / mn.loc[mn['basket'] == b, 'v_us'].sum():.3f}")
        print(f"  {'':14s} Part D gross:  L {g['L']:6.2f}  P {g['P']:6.2f}  F {g['F']:6.2f}")
        print(f"  {'':14s} NADAC:         L {a['L']:6.2f}  P {a['P']:6.2f}  F {a['F']:6.2f}   NADAC over Part D gross, "
              f"US-weighted {(x['q_us'] * x['p_nadac']).sum() / (x['q_us'] * x['p_us']).sum():.3f}")
        figure["nadac"][b] = {"gross": g, "nadac": a}
    print("  NADAC excludes manufacturer rebates and the dispensing fee; the French price without the fee still")
    print("  includes the wholesale and pharmacy margins and 2.1 percent VAT, so this ratio understates the gap")
    print("  between manufacturer prices.")

    print()
    print("Confidential rebates (single-source basket, Fisher index scaled by (1 - r_US) / (1 - r_FR)):")
    single_f = figure["baskets"]["single-source"]["F"]
    fr_only = dp.indices(m[m["mixed_side"] == "generic in France only"])["F"]
    grid = {}
    print("  " + " " * 22 + "".join(f"{k:>24s}" for k in REBATES_FR))
    for ku, ru in REBATES_US.items():
        vals = [single_f * (1 - ru) / (1 - rf) for rf in REBATES_FR.values()]
        grid[ku] = vals
        print(f"  US {ku:10s} r={ru:5.3f}" + "".join(f"{v:24.2f}" for v in vals))
    print(f"  (Fisher index of the single-source basket before rebates: {single_f:.3f}; of the molecules with")
    print(f"   generics in France only: {fr_only:.3f}.)")
    print("  Rebates that would make the single-source basket cost the same in both countries, given French")
    for rf in REBATES_FR.values():
        print(f"   rebates of {rf:.2f}: US rebate of {1 - (1 - rf) / single_f:.3f}")
    figure["rebates"] = {"us": REBATES_US, "fr": REBATES_FR, "grid": grid, "F": single_f}

    # ------------------------------------------------------------------------------------- 5
    section("5. Composition: an all-drug index and the generic share of units")
    sm = m[m["basket"].isin(["single-source", "multi-source"])]
    share_us = sm.loc[sm["basket"] == "multi-source", "q_us"].sum() / sm["q_us"].sum()
    share_fr = sm.loc[sm["basket"] == "multi-source", "q_fr"].sum() / sm["q_fr"].sum()
    print(f"Single-source and multi-source molecules only. Multi-source share of units: US {share_us:.4f}, "
          f"France {share_fr:.4f}")
    def mix(share_single, w):
        return dp.mixed_index(sm, 1.0 - share_single, "multi-source", "single-source", col="basket", weights=w)

    for single in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.0):
        print(f"  single-source share of units {single:4.2f}: index with US weights within baskets {mix(single, 'us'):6.3f},"
              f" with French weights {mix(single, 'fr'):6.3f}")
    for lab, s in (("US", share_us), ("France", share_fr)):
        print(f"  at the {lab} mix ({1 - s:.4f} of units single-source): US weights {mix(1 - s, 'us'):6.3f},"
              f" French weights {mix(1 - s, 'fr'):6.3f}")
    grid_single = np.geomspace(0.001, 1.0, 121)
    curve = {w: [mix(g, w) for g in grid_single] for w in ("us", "fr")}
    ix = dp.indices(sm)
    print(f"  the same molecules with their own weights: Laspeyres {ix['L']:.3f}, Paasche {ix['P']:.3f}")
    vs = sm.groupby("basket")[["v_us", "v_fr"]].sum()
    vs = vs / vs.sum()
    print(f"  single-source share of spending: US {vs.loc['single-source', 'v_us']:.3f}, France "
          f"{vs.loc['single-source', 'v_fr']:.3f}")
    print("  An all-drug index is an average of two very different indices, and its value is set by how many")
    print("  units fall in each: moving the generic share moves the headline without any price changing.")
    figure["mixing"] = {"single": grid_single.tolist(), **curve, "share_us": share_us, "share_fr": share_fr,
                        "L": ix["L"], "P": ix["P"]}

    # ------------------------------------------------------------------------------------- 6
    section("6. Medicare negotiation: oral drugs with a negotiated price for 2026 or 2027")
    mfp = dp.mfp_table(DATA)
    cu = dp.cells_us(u)
    orig_us = cu[cu["segment"] == "originator"].set_index("molecule")
    cfx = dp.cells_fr(f, include_fee=False)
    orig_fr = cfx[cfx["segment"] == "originator"].set_index("molecule")
    now_orig = dp.fr_current_unit_price(f, "originator")
    now_all = dp.fr_current_unit_price(f)
    mfp["p_partd"] = mfp["molecule"].map(orig_us["p_us"])
    mfp["q_partd"] = mfp["molecule"].map(orig_us["q_us"])
    mfp["v_partd"] = mfp["molecule"].map(orig_us["v_us"])
    mfp["fr24"] = mfp["molecule"].map(orig_fr["p_fr"]) * usd24
    mfp["fr_now_listed"] = mfp["molecule"].map(now_orig).notna()
    # Some specialty drugs carry no price in the current BDPM file; their 2024 reimbursement base
    # stands in, converted at the 2025 rate (marked with a dagger).
    mfp["fr_now"] = mfp["molecule"].map(now_orig).fillna(mfp["molecule"].map(orig_fr["p_fr"])) * usd25
    mfp["fr_now_all"] = mfp["molecule"].map(now_all) * usd25
    mfp["before"] = mfp["nadac_unit"].fillna(mfp["p_partd"])
    mfp["r_before"] = mfp["before"] / mfp["fr24"]
    mfp["r_after"] = mfp["mfp_unit"] / mfp["fr_now"]
    print("Per tablet or capsule. 'before': 2024 NADAC, or Part D gross where the drug has no NADAC (specialty")
    print("drugs, marked *); France 2024: reimbursement base without the fee, originator presentations.")
    print("'after': negotiated price (first year) against the current French price of the originator, at the")
    print("2025 exchange rate (+: no current French price listed; 2024 reimbursement base instead).")
    print("Negotiated price in force from 1 January of the year shown.")
    figure["mfp"] = []
    for _, r in mfp.iterrows():
        star = " " if pd.notna(r["nadac_unit"]) else "*"
        mol_ok = pd.notna(r["fr24"]) and pd.notna(r["p_partd"])
        if not mol_ok:
            print(f"  {r['ipay']} {r['drug']:12s} {r['molecule']:24s} not matched in both countries")
            continue
        dag = " " if r["fr_now_listed"] else "+"
        share_n = f"{r['mfp_over_nadac']:.2f} of NADAC" if pd.notna(r["mfp_over_nadac"]) else "no NADAC    "
        print(f"  {r['ipay']} {r['drug']:12s} before ${r['before']:8.2f}{star} FR24 ${r['fr24']:7.2f} ratio "
              f"{r['r_before']:5.1f}  | negotiated ${r['mfp_unit']:8.2f} ({share_n}) "
              f"FR now ${r['fr_now']:7.2f}{dag} ratio {r['r_after']:5.1f}"
              + (f"  [French generics: molecule ${r['fr_now_all']:.2f}]" if r['fr_now_all'] < 0.9 * r['fr_now'] else ""))
        figure["mfp"].append({k: (float(r[k]) if isinstance(r[k], (int, float, np.floating, np.integer)) else r[k])
                              for k in ("ipay", "drug", "molecule", "before", "fr24", "r_before", "mfp_unit",
                                        "fr_now", "r_after", "v_partd", "nadac_unit")})
    ok = mfp.dropna(subset=["fr24", "fr_now", "p_partd"])
    w = ok["q_partd"]
    retail = ok[ok["nadac_unit"].notna()]
    wr = retail["q_partd"]
    print(f"Matched drugs {len(ok)} of {len(mfp)}; their 2024 Part D gross spending ${ok['v_partd'].sum() / 1e9:.1f} bn.")
    print(f"  median ratio to France: before {ok['r_before'].median():.2f}, after {ok['r_after'].median():.2f}")
    print(f"  Part D-weighted (2024 units): before {(w * ok['before']).sum() / (w * ok['fr24']).sum():.2f}, after "
          f"{(w * ok['mfp_unit']).sum() / (w * ok['fr_now']).sum():.2f}")
    print(f"  retail drugs with a NADAC ({len(retail)}): Part D-weighted before {(wr * retail['before']).sum() / (wr * retail['fr24']).sum():.2f},"
          f" after {(wr * retail['mfp_unit']).sum() / (wr * retail['fr_now']).sum():.2f}; specialty drugs ({len(ok) - len(retail)}): median"
          f" before {ok.loc[ok['nadac_unit'].isna(), 'r_before'].median():.2f}, after {ok.loc[ok['nadac_unit'].isna(), 'r_after'].median():.2f}")
    print(f"  median negotiated price as a share of 2024 NADAC (drugs with a NADAC): {ok['mfp_over_nadac'].median():.3f}")
    for ipay in (2026, 2027):
        o = ok[ok["ipay"] == ipay]
        print(f"  {ipay}: {len(o)} drugs, median ratio before {o['r_before'].median():.2f}, after {o['r_after'].median():.2f}")
    single = m[m["basket"] == "single-source"]
    neg = set(ok["molecule"])
    cov = single.loc[single["molecule"].isin(neg), "v_us"].sum() / single["v_us"].sum()
    print(f"  share of single-source basket spending in these molecules: {cov:.3f}")
    print("  CMS: the 2026 prices would have cut net spending on the ten drugs by 22 percent in 2023; the 2027")
    print("  prices by 44 percent in 2024 (36 percent counting the manufacturer discount programme).")

    with open(HERE / "figure_data.json", "w") as fh:
        json.dump(figure, fh)


if __name__ == "__main__":
    main()
