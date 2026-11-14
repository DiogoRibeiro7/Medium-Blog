"""Follow-up 3 to the taxation-efficiency-drug-prices series: a test of the second conjecture of follow-up 2.

Every number quoted in the article is printed by this script. Run, after fetch.py has put the Medicare files in
data/ (the GitHub Actions workflow does this and commits them):

    python us_side_numbers.py > output.txt

The conjecture (follow-up 2). The fall in the US share of established medicines comes mainly from the United
States, through falling American net prices and earlier American generic and biosimilar entry for the largest
medicines, not from growth abroad. Two predictions:

  P1  The fall should be concentrated in medicines whose US revenue fell in absolute terms, while their revenue
      outside the United States held steady.
  P2  Medicines whose US revenue kept growing should have kept their US shares.

The revenue of each medicine in the United States and in total is rebuilt from the XBRL tags of the companies'
annual reports exactly as in follow-up 2, from that follow-up's snapshot. A medicine's US share is s = U / (U + F),
with U its US revenue and F its revenue elsewhere. Each change in s is split exactly into the part due to the
change in U and the part due to the change in F (the two-factor Shapley split, which averages the two orders of
change), and each part is weighted as in the within term of follow-up 2's decomposition.

Sections
1. Data: the medicines of follow-up 2.
2. The fall to explain: the within term of established medicines.
3. Where each change comes from: the United States or abroad.
4. P1: which medicines carry the fall.
5. P2: medicines whose US revenue kept growing.
6. The same medicines, 2016 against 2025.
7. Which market fell first.
8. What the approval dates of generics and biosimilars say.
9. Volume or price: Medicare, 2020-2024.
10. Currencies and other checks.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ROOT = HERE.parent
FU2_DATA = ROOT / "followup2" / "data"

spec = importlib.util.spec_from_file_location("fu2", ROOT / "followup2" / "portfolio_numbers.py")
fu2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fu2)

SEED = 20261114
REPS = 5000
NEW = fu2.NEW          # a medicine is new in year t if first approved in the US in years t-4 to t
START, END = 2016, 2025
STEADY = 0.10          # revenue abroad "held steady": changed by less than 10 percent in either direction


def section(title: str) -> None:
    print()
    print("=" * 94)
    print(title)
    print("=" * 94)


def share(u, f):
    u, f = np.asarray(u, float), np.asarray(f, float)
    t = u + f
    return np.divide(u, t, out=np.full_like(t, np.nan), where=t > 0)


def shapley(u0, f0, u1, f1):
    """Exact split of s(u1, f1) - s(u0, f0), s = u / (u + f), into the part due to u and the part due to f."""
    s00, s11 = share(u0, f0), share(u1, f1)
    s10, s01 = share(u1, f0), share(u0, f1)
    # a cross term is undefined only if u1 = f0 = 0 or u0 = f1 = 0; then the other order alone is used
    us = np.where(np.isnan(s10), s11 - s01, np.where(np.isnan(s01), s10 - s00, ((s10 - s00) + (s11 - s01)) / 2))
    ab = (s11 - s00) - us
    return us, ab


def medicine_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The medicine-year panel of follow-up 2 (US and total revenue in dollars, first US approval)."""
    f = pd.read_csv(FU2_DATA / "sec_revenue_facts.csv.gz")
    f["dims"] = f["dims"].fillna("")
    f["py"] = f["end"].map(fu2.fiscal_year)
    f["lag"] = f["fiscal_year"] - f["py"]
    f = f[f["lag"] >= 0]
    dd = f["dims"].map(lambda d: dict(x.split("=", 1) for x in d.split(";") if x))
    f["n_dims"] = dd.map(len)
    f["product"] = dd.map(lambda d: next((v for k, v in d.items() if k in fu2.PRODUCT_AXES), ""))
    f["n_geo"] = dd.map(lambda d: sum(k in fu2.GEO_AXES for k in d))
    f["geo"] = dd.map(lambda d: next((v for k, v in d.items() if k in fu2.GEO_AXES), ""))
    fx = pd.read_csv(ROOT / "part6" / "data" / "ecb_fx.csv.gz").pivot(index="time_period", columns="currency",
                                                                       values="obs_value")
    panel = fu2.product_panel(f)
    brands = fu2.load_brands()
    match = fu2.matcher(brands)
    keymap = {m: match(m) for m in panel["product"].drop_duplicates()}
    panel["key"] = panel["product"].map(keymap)
    panel = panel.join(brands[["date", "source"]], on="key")
    panel["approval"] = panel["date"].dt.year
    med = (panel[panel["key"].notna() & panel["us"].notna() & panel["total"].notna() & (panel["total"] > 0)]
           .sort_values("total").groupby(["company", "py", "key"]).tail(1).copy())
    med["us"] = med["us"].clip(upper=med["total"])

    def usd(v, cur, year):
        if cur == "USD":
            return v
        eur = v if cur == "EUR" else v / fx.loc[year, cur]
        return eur * fx.loc[year, "USD"]

    ccy = f.groupby("company")["currency"].agg(lambda x: x.mode().iloc[0])
    med["cur"] = med["company"].map(ccy)
    med["U"] = [usd(v, c, y) for v, c, y in zip(med["us"], med["cur"], med["py"])]
    med["R"] = [usd(v, c, y) for v, c, y in zip(med["total"], med["cur"], med["py"])]
    med["F"] = med["R"] - med["U"]
    med["approval"] = med["approval"].astype(int)
    med["unit"] = med["company"] + "|" + med["key"]
    return med.reset_index(drop=True), fx


def sample(med: pd.DataFrame, y0: int, y1: int) -> list[str]:
    yrs = set(range(y0, y1 + 1))
    return sorted(co for co, g in med.groupby("company") if yrs <= set(g["py"])
                  and g.groupby("py")["R"].sum().min() > 1e9)


def pairs(med: pd.DataFrame, cos: list[str] | None, y0: int, y1: int) -> pd.DataFrame:
    """Every medicine reported in two consecutive years, with the weights of the within term: the average of its
    two revenue weights in the pool of the companies' medicines."""
    out = []
    for y in range(y0 + 1, y1 + 1):
        m0 = med[med["py"] == y - 1]
        m1 = med[med["py"] == y]
        if cos is not None:
            m0, m1 = m0[m0["company"].isin(cos)], m1[m1["company"].isin(cos)]
        else:  # every company reported in both years
            both = set(m0["company"]) & set(m1["company"])
            m0, m1 = m0[m0["company"].isin(both)], m1[m1["company"].isin(both)]
        a = m0.assign(w=m0["R"] / m0["R"].sum())[["unit", "company", "key", "approval", "U", "F", "R", "w"]]
        b = m1.assign(w=m1["R"] / m1["R"].sum())[["unit", "U", "F", "R", "w"]]
        p = a.merge(b, on="unit", suffixes=("0", "1"))
        p["year"] = y
        out.append(p)
    p = pd.concat(out, ignore_index=True)
    p["age"] = p["year"] - p["approval"]
    p["group"] = np.where(p["age"] <= NEW, "new", "established")
    p["band"] = np.select([p["age"] <= NEW, p["age"] <= 9], ["0-4", "5-9"], "10+")
    p["wbar"] = (p["w0"] + p["w1"]) / 2
    p["s0"], p["s1"] = share(p["U0"], p["F0"]), share(p["U1"], p["F1"])
    p["ds"] = p["s1"] - p["s0"]
    p["us_part"], p["ab_part"] = shapley(p["U0"], p["F0"], p["U1"], p["F1"])
    p["within"] = p["wbar"] * p["ds"]
    p["w_us"] = p["wbar"] * p["us_part"]
    p["w_ab"] = p["wbar"] * p["ab_part"]
    p["gU"] = p["U1"] / p["U0"] - 1
    p["gF"] = np.divide(p["F1"] - p["F0"], p["F0"], out=np.where(p["F1"] > 0, np.inf, 0.0), where=p["F0"] > 0)
    return p


def boot(df: pd.DataFrame, fn, rng, reps: int = REPS) -> tuple[float, float]:
    """90 percent interval of fn(df) when medicines are resampled."""
    units = df["unit"].unique()
    idx = {u: np.flatnonzero(df["unit"].to_numpy() == u) for u in units}
    vals = np.empty(reps)
    for r in range(reps):
        pick = rng.choice(units, size=len(units), replace=True)
        rows = np.concatenate([idx[u] for u in pick])
        vals[r] = fn(df.iloc[rows])
    return float(np.nanpercentile(vals, 5)), float(np.nanpercentile(vals, 95))


def pts(x: float) -> str:
    return f"{100 * x:+.2f}"


SALTS = {"HYDROCHLORIDE", "HCL", "SODIUM", "DISODIUM", "POTASSIUM", "CALCIUM", "MAGNESIUM", "ACETATE", "MALEATE",
         "PHOSPHATE", "MESYLATE", "TARTRATE", "BITARTRATE", "SUCCINATE", "FUMARATE", "HYDROBROMIDE", "SULFATE",
         "CITRATE", "BESYLATE", "DIHYDRATE", "MONOHYDRATE", "HEMIPENTAHYDRATE", "HEPTAHYDRATE", "TRIHYDRATE",
         "HEMIHYDRATE", "ANHYDROUS", "DIMEGLUMINE", "TROMETHAMINE", "LYSINE", "MEGLUMINE"}


def ingredients(s) -> str:
    """The active moieties of a product: its ingredients without salt and hydrate words, sorted."""
    out = []
    for x in str(s).split(";"):
        w = [t for t in x.strip().upper().split() if t not in SALTS]
        if w:
            out.append(" ".join(w))
    return ";".join(sorted(set(out)))


def first_entry_dates(keys: set[str]) -> pd.DataFrame:
    """For each medicine: the date of its first US generic approval (an ANDA with the same active ingredients) or
    first US biosimilar licence (a 351(k) licence naming it as reference product), and the date of the first
    biosimilar authorised by the European Commission for the same substance (EU generics are mostly authorised
    nationally and are not in the EMA list)."""
    d = pd.read_csv(FU2_DATA / "fda_drugsfda.csv.gz", low_memory=False)
    d["ing"] = d["active_ingredients"].map(ingredients)
    d["kind"] = d["application_number"].str.extract(r"^([A-Z]+)", expand=False)
    d["date"] = pd.to_datetime(d["first_approval"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    d["nb"] = d["brand_name"].map(fu2.norm)
    pb = pd.read_csv(FU2_DATA / "fda_purplebook.csv.gz", low_memory=False)
    pb["date"] = pd.to_datetime(pb["Approval Date"], format="%d-%b-%y", errors="coerce")
    pb["nb"] = pb["Proprietary Name"].map(fu2.norm)
    pb["ref"] = pb["Ref. Product Proprietary Name"].map(lambda x: fu2.norm(x) if isinstance(x, str) else "")
    ema = pd.read_csv(ROOT / "part4" / "data" / "ema_medicines.csv.gz", low_memory=False)
    ema["date"] = pd.to_datetime(ema["Marketing authorisation date"], dayfirst=True, errors="coerce")
    ema["inn"] = ema["International non-proprietary name (INN) / common name"].astype(str).str.lower().str.strip()
    ema_bs = ema[ema["Biosimilar"] == "Yes"]
    andas = d[d["kind"] == "ANDA"].groupby("ing")["date"].min()
    rows = []
    for k in sorted(keys):
        own = d[(d["nb"] == k) & d["kind"].isin(["NDA", "BLA"])]
        ing = own["ing"].mode().iloc[0] if len(own) else ""
        gen = andas.get(ing, pd.NaT) if ing else pd.NaT
        bio = pb.loc[pb["BLA Type"].astype(str).str.startswith("351(k)") & (pb["ref"] == k), "date"].min()
        proper = pb.loc[(pb["nb"] == k) & pb["BLA Type"].astype(str).str.startswith("351(a)"), "Proper Name"]
        proper = proper.iloc[0].lower().split("-")[0].strip() if len(proper) else ""
        eu = ema_bs.loc[ema_bs["inn"].str.startswith(proper), "date"].min() if proper else pd.NaT
        us = min([x for x in (gen, bio) if pd.notna(x)], default=pd.NaT)
        rows.append({"key": k, "ingredients": ing or proper, "us_generic": gen, "us_biosimilar": bio, "us_first": us,
                     "eu_biosimilar": eu})
    return pd.DataFrame(rows).set_index("key")


def medicare(med: pd.DataFrame) -> pd.DataFrame:
    """Medicare Part D and Part B rows assigned to the medicines: the longest medicine name that a row's brand name
    begins with; failing that, the medicine whose XBRL member name contains the brand name or begins it (members
    such as Johnson & Johnson's PREZISTAPREZCOBIXREZOLSTASYMTUZA report several brands as one line); Part B rows
    named only by the generic name are assigned through the Purple Book proper name."""
    keys = set(med["key"])
    cores = {}
    for prod, k in med[["product", "key"]].drop_duplicates().itertuples(index=False):
        core = fu2.norm(prod.split(":")[-1].removesuffix("Member").removesuffix("ProductMember"))
        if len(core) >= 6:
            cores.setdefault(core, k)
    pd_ = pd.read_csv(DATA / "cms_partd_by_drug.csv.gz")
    pd_["program"], pd_["code"] = "D", pd_["Brnd_Name"] + "|" + pd_["Gnrc_Name"]
    pb_ = pd.read_csv(DATA / "cms_partb_by_drug.csv.gz")
    pb_["program"], pb_["code"] = "B", pb_["HCPCS_Cd"]
    m = pd.concat([pd_, pb_], ignore_index=True)
    m["nb"] = m["Brnd_Name"].map(fu2.norm)
    m["ng"] = m["Gnrc_Name"].map(fu2.norm)
    ks = sorted((k for k in keys if len(k) >= 4), key=len, reverse=True)
    def assign(b: str):
        k = next((k for k in ks if b.startswith(k)), None)
        if k is None and len(b) >= 6:
            k = next((v for c, v in cores.items() if b in c or b.startswith(c)), None)
        return k

    m["key"] = m["nb"].map(assign)
    pb = pd.read_csv(FU2_DATA / "fda_purplebook.csv.gz", low_memory=False)
    orig = pb[pb["BLA Type"].astype(str).str.startswith("351(a)")]
    proper = {}
    for r in orig.itertuples():
        k = fu2.norm(r[orig.columns.get_loc("Proprietary Name") + 1])
        if k in keys:
            proper.setdefault(fu2.norm(str(r[orig.columns.get_loc("Proper Name") + 1]).split("-")[0]), k)
    gen_only = m["key"].isna() & (m["program"] == "B") & (m["nb"] == m["ng"])
    m.loc[gen_only, "key"] = m.loc[gen_only, "ng"].map(proper)
    return m[m["key"].notna()].copy()


def main() -> None:
    rng = np.random.default_rng(SEED)
    figure: dict = {}

    # ------------------------------------------------------------------ 1
    section("1. Data: the medicines of follow-up 2")
    med, fx = medicine_panel()
    print(f"Medicine-years with US and total revenue: {len(med):,}; distinct medicines {med['unit'].nunique()} "
          f"(follow-up 2: 1,694 and 277)")
    main_cos = sample(med, START, END)
    print(f"Main sample, companies with dated medicines in every year {START}-{END}: {len(main_cos)}: {', '.join(main_cos)}")
    p = pairs(med, main_cos, START, END)
    est = p[p["group"] == "established"].copy()
    print(f"Medicine pairs of consecutive years in the main sample: {len(p):,} of {p['unit'].nunique()} medicines")
    for b in ["0-4", "5-9", "10+"]:
        g = p[p["band"] == b]
        print(f"  {b:4s} years since first US approval: {len(g):4d} pairs of {g['unit'].nunique():3d} medicines, "
              f"{g['R1'].sum() / p['R1'].sum():.0%} of the revenue in the pairs")

    # ------------------------------------------------------------------ 2
    section("2. The fall to explain: the within term of established medicines")
    print(f"Within term (each medicine's change in its own US share, weighted), main sample {START}-{END}, points:")
    print(f"  new (0-4 years)          {pts(p.loc[p['group'] == 'new', 'within'].sum())}   (follow-up 2: -5.65)")
    print(f"  established (5 and more) {pts(est['within'].sum())}   (follow-up 2: -5.03)")
    for b in ["5-9", "10+"]:
        print(f"    of which {b:4s} years      {pts(p.loc[p['band'] == b, 'within'].sum())}")

    # ------------------------------------------------------------------ 3
    section("3. Where each change comes from: the United States or abroad")
    print("Each change in a medicine's US share split exactly into the part due to its US revenue and the part due to "
          "its revenue abroad (points of the within term; 90% intervals, medicines resampled):")
    split = {}
    for lab, g in [("0-4", p[p["band"] == "0-4"]), ("5 and more", est), ("5-9", p[p["band"] == "5-9"]),
                   ("10+", p[p["band"] == "10+"])]:
        a, b = g["w_us"].sum(), g["w_ab"].sum()
        la, ha = boot(g, lambda d: d["w_us"].sum(), rng)
        lb, hb = boot(g, lambda d: d["w_ab"].sum(), rng)
        split[lab] = {"us": 100 * a, "abroad": 100 * b, "us_ci": [100 * la, 100 * ha], "ab_ci": [100 * lb, 100 * hb]}
        print(f"  {lab:10s} within {pts(a + b)} = from US revenue {pts(a)} ({100 * la:+.2f} to {100 * ha:+.2f}) + from "
              f"revenue abroad {pts(b)} ({100 * lb:+.2f} to {100 * hb:+.2f})")
    figure["split"] = split
    print("Where the line for an old medicine is drawn (pairs in which the medicine is at least that old):")
    figure["line"] = {int(L): [100 * p.loc[p["age"] >= L, c].sum() for c in ("within", "w_us", "w_ab")]
                      for L in range(3, 21)}
    for line in (5, 6, 7, 8, 9, 10, 12, 15, 20):
        g = p[p["age"] >= line]
        print(f"  {line:2d} years or more: {len(g):4d} pairs, within {pts(g['within'].sum())} = US {pts(g['w_us'].sum())} "
              f"+ abroad {pts(g['w_ab'].sum())}")
    for lab, g in [("5-9", p[p["band"] == "5-9"]), ("10+", p[p["band"] == "10+"])]:
        u0, u1, f0, f1 = (g[c].sum() for c in ("U0", "U1", "F0", "F1"))
        print(f"  {lab:4s} summed over all pairs: US revenue {u1 / u0 - 1:+.1%} from one year to the next, revenue "
              f"abroad {f1 / f0 - 1:+.1%}")
    pa = pairs(med, None, START, END)
    print("All companies with dated medicines, every pair of consecutive years in which a company is reported:")
    for lab, g in [("5-9", pa[pa["band"] == "5-9"]), ("10+", pa[pa["band"] == "10+"])]:
        print(f"  {lab:4s} {len(g):4d} pairs: within {pts(g['within'].sum())} = US {pts(g['w_us'].sum())} + abroad "
              f"{pts(g['w_ab'].sum())}")
    old = p[p["band"] == "10+"]
    print("Ten years and more, by year (within; from US, from abroad):")
    for y, r in old.groupby("year")[["within", "w_us", "w_ab"]].sum().iterrows():
        print(f"    {y}  {pts(r['within'])}  ({pts(r['w_us'])}, {pts(r['w_ab'])})")
    figure["by_year_old"] = {int(y): [100 * r["w_us"], 100 * r["w_ab"]]
                             for y, r in old.groupby("year")[["w_us", "w_ab"]].sum().iterrows()}

    # ------------------------------------------------------------------ 4
    section("4. P1: which medicines carry the fall")
    order = ["US fell, abroad steady", "US fell, abroad grew", "US fell, abroad fell", "US grew, abroad grew",
             "US grew, abroad fell"]
    quads = {}
    for lab in ["5-9", "10+"]:
        g = p[p["band"] == lab].copy()
        fell = g["U1"] < g["U0"]
        g["quad"] = np.select(
            [fell & (g["gF"].abs() < STEADY), fell & (g["gF"] >= STEADY), fell & (g["gF"] <= -STEADY),
             ~fell & (g["F1"] >= g["F0"]), ~fell & (g["F1"] < g["F0"])], order, "other")
        q = g.groupby("quad").agg(n=("unit", "size"), rev=("R1", "sum"), within=("within", "sum"),
                                  us=("w_us", "sum"), ab=("w_ab", "sum")).reindex(order).fillna(0)
        print(f"{lab} years since approval ('steady' = revenue abroad changed by less than {STEADY:.0%}):")
        for k, r in q.iterrows():
            print(f"  {k:24s} {int(r['n']):4d} pairs ({r['rev'] / q['rev'].sum():5.1%} of revenue): within "
                  f"{pts(r['within'])} (from US {pts(r['us'])}, abroad {pts(r['ab'])})")
        lo, hi = boot(g, lambda d: d.loc[d["U1"] < d["U0"], "within"].sum() / d["within"].sum(), rng)
        st = fell & (g["gF"].abs() < STEADY)
        print(f"  pairs in which US revenue fell: {fell.mean():.0%} of pairs, {g.loc[fell, 'R1'].sum() / g['R1'].sum():.0%} of "
              f"revenue, {g.loc[fell, 'within'].sum() / g['within'].sum():.0%} of the fall (90% interval {lo:.0%} to "
              f"{hi:.0%}); with revenue abroad steady {g.loc[st, 'within'].sum() / g['within'].sum():.0%}")
        wt = g.loc[fell, "R0"]
        print(f"  when US revenue fell, revenue-weighted mean change of US revenue {np.average(g.loc[fell, 'gU'], weights=wt):+.1%}"
              f", of revenue abroad {np.average(g.loc[fell, 'gF'].clip(upper=1.0), weights=wt):+.1%} (capped at +100%)")
        quads[lab] = [{"label": k, "within": 100 * float(r["within"]), "rev": float(r["rev"] / q["rev"].sum()),
                       "us": 100 * float(r["us"]), "ab": 100 * float(r["ab"])} for k, r in q.iterrows()]
    figure["quadrants"] = quads

    # ------------------------------------------------------------------ 5
    section("5. P2: medicines whose US revenue kept growing")
    for lab in ["5-9", "10+"]:
        g = p[p["band"] == lab]
        up, dn = g[g["U1"] >= g["U0"]], g[g["U1"] < g["U0"]]
        mw = np.average(up["ds"], weights=up["wbar"])
        lo, hi = boot(up, lambda d: np.average(d["ds"], weights=d["wbar"]), rng)
        print(f"{lab}: pairs with US revenue up {len(up)} of {len(g)}; revenue-weighted mean change in own US share "
              f"{100 * mw:+.2f} points a year (90% interval {100 * lo:+.2f} to {100 * hi:+.2f}); share fell in "
              f"{(up['ds'] < 0).mean():.0%}; within {pts(up['within'].sum())} = US {pts(up['w_us'].sum())} + abroad "
              f"{pts(up['w_ab'].sum())}")
        print(f"  pairs with US revenue down {len(dn)}: mean change in own US share "
              f"{100 * np.average(dn['ds'], weights=dn['wbar']):+.2f} points a year")
    for lab, g in [("5-9", p[p["band"] == "5-9"]), ("10+", p[p["band"] == "10+"])]:
        span = g.sort_values("year").groupby("unit").agg(first=("year", "min"), last=("year", "max"),
                                                         U0=("U0", "first"), F0=("F0", "first"), U1=("U1", "last"),
                                                         F1=("F1", "last"), within=("within", "sum"))
        span = span[span["last"] - span["first"] >= 2]
        span["ds"] = share(span["U1"], span["F1"]) - share(span["U0"], span["F0"])
        for name, h in [("US revenue higher at the end", span[span["U1"] >= span["U0"]]),
                        ("US revenue lower at the end", span[span["U1"] < span["U0"]])]:
            print(f"  {lab} over each medicine's span in the band (three pairs or more), {name}: {len(h)} medicines, "
                  f"median change in US share {100 * h['ds'].median():+.1f} points, lower for {(h['ds'] < 0).mean():.0%}")

    # ------------------------------------------------------------------ 6
    section("6. The same medicines, 2016 against 2025")
    for lim, lab in [(START - 10, "ten years or more on the market in 2016"), (START, "approved by 2016 (follow-up 2)")]:
        a = med[(med["py"] == START) & med["company"].isin(main_cos) & (med["approval"] <= lim)]
        b = med[(med["py"] == END) & med["company"].isin(main_cos)]
        c = a.merge(b, on="unit", suffixes=("0", "1"))
        c["w1"] = c["R1"] / c["R1"].sum()
        c["s0"], c["s1"] = share(c["U0"], c["F0"]), share(c["U1"], c["F1"])
        c["us_part"], c["ab_part"] = shapley(c["U0"], c["F0"], c["U1"], c["F1"])
        c["contrib"], c["c_us"], c["c_ab"] = c["w1"] * (c["s1"] - c["s0"]), c["w1"] * c["us_part"], c["w1"] * c["ab_part"]
        gone = a[~a["unit"].isin(b["unit"])]
        print(f"Medicines {lab}, reported in {START} and {END}: {len(c)} (of {len(a)} reported in {START}); US share "
              f"{c['U0'].sum() / c['R0'].sum():.1%} to {c['U1'].sum() / c['R1'].sum():.1%}; own-share effect "
              f"{pts(c['contrib'].sum())} = US {pts(c['c_us'].sum())} + abroad {pts(c['c_ab'].sum())}")
        print(f"  US revenue {c['U0'].sum() / 1e9:.1f} -> {c['U1'].sum() / 1e9:.1f} bn ({c['U1'].sum() / c['U0'].sum() - 1:+.0%}); "
              f"abroad {c['F0'].sum() / 1e9:.1f} -> {c['F1'].sum() / 1e9:.1f} bn ({c['F1'].sum() / c['F0'].sum() - 1:+.0%}); "
              f"no longer reported in {END}: {len(gone)}, {gone['R'].sum() / 1e9:.1f} bn in {START}, US share "
              f"{gone['U'].sum() / gone['R'].sum():.1%}")
        if lim == START - 10:
            print("  each medicine (US revenue, revenue abroad, bn dollars; US share):")
            for r in c.sort_values("R0", ascending=False).itertuples():
                print(f"    {r.key0:16s} {r.company0:18s} appr {r.approval0}  US {r.U0 / 1e9:5.1f} -> {r.U1 / 1e9:5.1f}; abroad "
                      f"{r.F0 / 1e9:5.1f} -> {r.F1 / 1e9:5.1f}; share {r.s0:5.1%} -> {r.s1:5.1%}")
            figure["old_constant"] = [{"key": r.key0, "company": r.company0, "U0": r.U0, "U1": r.U1, "F0": r.F0,
                                       "F1": r.F1} for r in c.itertuples()]
    top = old.groupby(["company", "key", "approval"])[["within", "w_us", "w_ab"]].sum().sort_values("within")
    print("Largest contributions, ten years and more, chained pairs (points: total; from US, from abroad):")
    for (co, k, a), r in list(top.head(10).iterrows()) + list(top.tail(4).iterrows()):
        print(f"  {k:18s} {co:18s} approved {a}  {pts(r['within'])} ({pts(r['w_us'])}, {pts(r['w_ab'])})")
    neg = top["w_us"].clip(upper=0)
    print(f"  the ten largest negative US parts carry {neg.nsmallest(10).sum() / neg.sum():.0%} of all negative US parts "
          f"({neg.sum() * 100:+.2f}); positive US parts sum to {100 * top['w_us'].clip(lower=0).sum():+.2f}")
    top59 = p[p["band"] == "5-9"].groupby(["company", "key", "approval"])[["within", "w_us", "w_ab"]].sum()
    print("Largest abroad parts, five to nine years (points: total; from US, from abroad):")
    for (co, k, a), r in top59.sort_values("w_ab").head(8).iterrows():
        print(f"  {k:18s} {co:18s} approved {a}  {pts(r['within'])} ({pts(r['w_us'])}, {pts(r['w_ab'])})")

    # ------------------------------------------------------------------ 7
    section("7. Which market fell first")
    obs = med[med["company"].isin(main_cos) & med["py"].between(START, END) & (med["U"] > 0) & (med["F"] > 0)]
    rows = []
    for u, g in obs.groupby("unit"):
        g = g.sort_values("py")
        if len(g) < 6:
            continue
        pu, pf = g.loc[g["U"].idxmax()], g.loc[g["F"].idxmax()]
        rows.append({"unit": u, "key": g["key"].iloc[0], "company": g["company"].iloc[0], "approval": g["approval"].iloc[0],
                     "first": int(g["py"].iloc[0]), "last": int(g["py"].iloc[-1]), "pu": int(pu["py"]), "pf": int(pf["py"]),
                     "du": g["U"].iloc[-1] / g["U"].max() - 1, "df": g["F"].iloc[-1] / g["F"].max() - 1,
                     "Rmax": g["R"].max()})
    pk = pd.DataFrame(rows)
    both = pk[(pk["du"] <= -0.3) & (pk["df"] <= -0.3)].copy()
    both["order"] = np.select([(both["pu"] == both["first"]) & (both["pf"] == both["first"]), both["pu"] < both["pf"],
                               both["pu"] > both["pf"]], ["both falling from the first year", "US first", "abroad first"],
                              "same year")
    print(f"Medicines with six years or more and sales in both markets, both at least 30% below their peaks by the last "
          f"year: {len(both)}")
    for lab in ["US first", "same year", "abroad first", "both falling from the first year"]:
        m = both["order"] == lab
        print(f"  {lab:32s} {int(m.sum()):3d} medicines, peak revenue {both.loc[m, 'Rmax'].sum() / 1e9:5.1f} bn")
    for r in both[both["order"] != "both falling from the first year"].sort_values("Rmax", ascending=False).head(16).itertuples():
        print(f"    {r.key:16s} {r.company:18s} US peak {r.pu} (then {r.du:+.0%}), abroad peak {r.pf} (then {r.df:+.0%}): {r.order}")
    us_only = pk[(pk["du"] <= -0.3) & (pk["df"] > -0.3)]
    ab_only = pk[(pk["du"] > -0.3) & (pk["df"] <= -0.3)]
    print(f"  US revenue 30% or more below its peak, abroad not: {len(us_only)} (peak revenue {us_only['Rmax'].sum() / 1e9:.1f} bn); "
          f"abroad, US not: {len(ab_only)} ({ab_only['Rmax'].sum() / 1e9:.1f} bn)")
    for lab, d in [("US only", us_only), ("abroad only", ab_only)]:
        print(f"    {lab}: " + ", ".join(f"{r.key} ({r.company}; {r.du:+.0%} US, {r.df:+.0%} abroad)"
                                         for r in d.sort_values("Rmax", ascending=False).head(8).itertuples()))
    figure["peaks"] = [{"key": r.key, "company": r.company, "pu": r.pu, "pf": r.pf, "du": r.du, "df": r.df,
                        "Rmax": r.Rmax, "order": r.order} for r in both.itertuples()]

    # ------------------------------------------------------------------ 8
    section("8. What the approval dates of generics and biosimilars say")
    dates = first_entry_dates(set(pk["key"]))
    fallen = pk[pk["du"] <= -0.3].join(dates, on="key")
    fallen = fallen[fallen["pu"] > fallen["first"]]  # the US peak is observed
    fallen["lead"] = fallen["pu"] - fallen["us_first"].dt.year
    has = fallen["us_first"].notna()
    print(f"Medicines whose US revenue peaked after {START} and then fell by 30% or more: {len(fallen)}; with a US generic "
          f"approval or biosimilar licence on file {int(has.sum())}, without {int((~has).sum())}")
    print(f"  years from the first US generic or biosimilar approval to the peak of US revenue: median "
          f"{fallen.loc[has, 'lead'].median():.0f}; approval two or more years before the peak for "
          f"{int((fallen.loc[has, 'lead'] >= 2).sum())}, within a year of it for {int(fallen.loc[has, 'lead'].abs().le(1).sum())}, "
          f"after the peak for {int((fallen.loc[has, 'lead'] < -1).sum())}")
    for r in fallen.sort_values("Rmax", ascending=False).head(18).itertuples():
        g = "" if pd.isna(r.us_generic) else f"generic {r.us_generic.year}"
        bsm = "" if pd.isna(r.us_biosimilar) else f"biosimilar {r.us_biosimilar.year}"
        eu = "" if pd.isna(r.eu_biosimilar) else f"; EU biosimilar {r.eu_biosimilar.year}"
        print(f"    {r.key:16s} {r.company:18s} approved {r.approval}; US peak {r.pu} ({r.du:+.0%} by {r.last}); first US "
              f"{(g or bsm) or 'none on file'}{(', ' + bsm) if g and bsm else ''}{eu}; abroad peak {r.pf}")
    bio = pk.join(dates, on="key")
    bio = bio[bio["us_biosimilar"].notna() & bio["eu_biosimilar"].notna()]
    print("Biologics with a biosimilar on both sides (first approval; peak of revenue in each market):")
    for r in bio.sort_values("Rmax", ascending=False).itertuples():
        print(f"    {r.key:16s} {r.company:18s} US biosimilar {r.us_biosimilar.year}, EU {r.eu_biosimilar.year}; US revenue "
              f"peak {r.pu} ({r.du:+.0%}), abroad {r.pf} ({r.df:+.0%})")

    # ------------------------------------------------------------------ 9
    section("9. Volume or price: Medicare, 2020-2024")
    mc = medicare(med)
    print(f"Medicare rows assigned to the medicines: Part D {int((mc['program'] == 'D').sum())}, Part B "
          f"{int((mc['program'] == 'B').sum())}; medicines {mc['key'].nunique()}")
    y0, y1 = 2020, 2024
    rows = []
    for k, g in mc.groupby("key"):
        a = g[g["year"] == y0].set_index(["program", "code"])
        b = g[g["year"] == y1].set_index(["program", "code"])
        j = a[["spend", "units"]].join(b[["spend", "units"]], lsuffix="0", rsuffix="1", how="inner")
        j = j[(j[["spend0", "spend1", "units0", "units1"]] > 0).all(axis=1)]
        if j.empty:
            continue
        wts = (j["spend0"] / j["spend0"].sum() + j["spend1"] / j["spend1"].sum()) / 2
        vol = float((wts * np.log(j["units1"] / j["units0"])).sum())
        rows.append({"key": k, "m_spend0": a["spend"].sum(), "m_spend1": b["spend"].sum(), "vol": vol,
                     "spend": float(np.log(j["spend1"].sum() / j["spend0"].sum())),
                     "programs": "+".join(sorted(set(j.index.get_level_values(0))))})
    mv = pd.DataFrame(rows).set_index("key")
    m0 = med[(med["py"] == y0) & (med["U"] > 0)][["unit", "company", "key", "approval", "U"]]
    m1 = med[(med["py"] == y1)][["unit", "U"]]
    mm = m0.merge(m1, on="unit", suffixes=("0", "1")).join(mv, on="key", how="inner")
    mm = mm[(mm["approval"] <= y1 - 10) & (mm["U0"] >= 5e8) & (mm["m_spend0"] >= 1e8) & (mm["U1"] > 0)]
    # a medicine reported by two companies (Imbruvica: AbbVie sells it in the US, Johnson & Johnson books a share of
    # the US profit) is counted once, with the company that sells it
    mm = mm.sort_values("U0", ascending=False).drop_duplicates("key")
    mm["net"] = np.log(mm["U1"] / mm["U0"])
    mm["gross_price"] = mm["spend"] - mm["vol"]
    mm["rest"] = mm["net"] - mm["spend"]
    print(f"Medicines ten years or more since approval in {y1}, US revenue of at least 0.5 bn dollars in {y0} and Medicare "
          f"spending of at least 0.1 bn: {len(mm)}")
    print("  log changes 2020-2024 (x100): company US net revenue = Medicare volume + Medicare gross price per unit + "
          "the rest (rebates, discounts and the other payers)")
    for lab, h in [("US revenue fell", mm[mm["net"] < 0]), ("US revenue grew", mm[mm["net"] >= 0])]:
        w = h["U0"]
        print(f"  {lab:16s} {len(h):2d} medicines: net {100 * np.average(h['net'], weights=w):+.1f} = volume "
              f"{100 * np.average(h['vol'], weights=w):+.1f} + gross price {100 * np.average(h['gross_price'], weights=w):+.1f} "
              f"+ rest {100 * np.average(h['rest'], weights=w):+.1f}  (weights: US revenue in {y0})")
    for r in mm.sort_values("U0", ascending=False).itertuples():
        print(f"    {r.key:16s} {r.company:18s} {r.programs:3s} US net {100 * r.net:+6.1f}; Medicare volume {100 * r.vol:+6.1f}, "
              f"gross price {100 * r.gross_price:+6.1f}; rest {100 * r.rest:+6.1f}  (US {r.U0 / 1e9:4.1f} -> {r.U1 / 1e9:4.1f} bn; "
              f"Medicare {r.m_spend0 / 1e9:4.1f} -> {r.m_spend1 / 1e9:4.1f} bn)")
    for lab, h in [("US revenue fell", mm[mm["net"] < 0]), ("US revenue fell, without humira", mm[(mm["net"] < 0) & (mm["key"] != "humira")])]:
        print(f"  {lab}: medians net {100 * h['net'].median():+.1f}, volume {100 * h['vol'].median():+.1f}, gross price "
              f"{100 * h['gross_price'].median():+.1f}, rest {100 * h['rest'].median():+.1f}; weighted net "
              f"{100 * np.average(h['net'], weights=h['U0']):+.1f} = volume {100 * np.average(h['vol'], weights=h['U0']):+.1f} + "
              f"gross price {100 * np.average(h['gross_price'], weights=h['U0']):+.1f} + rest "
              f"{100 * np.average(h['rest'], weights=h['U0']):+.1f}")
    fellm = mm[mm["net"] < 0]
    print(f"  of the medicines whose US revenue fell, Medicare volume fell by more than 20 log points for "
          f"{int((fellm['vol'] < -0.2).sum())}, held within 20 points for {int((fellm['vol'].abs() <= 0.2).sum())}, grew more for "
          f"{int((fellm['vol'] > 0.2).sum())}")
    figure["medicare"] = [{"key": r.key, "company": r.company, "net": 100 * r.net, "vol": 100 * r.vol,
                           "price": 100 * r.gross_price, "rest": 100 * r.rest, "U0": r.U0} for r in mm.itertuples()]

    # ------------------------------------------------------------------ 10
    section("10. Currencies and other checks")
    e = fx["USD"]
    j = fx["JPY"] / fx["USD"]
    print(f"USD per EUR {e.loc[START]:.3f} ({START}) to {e.loc[END]:.3f} ({END}); JPY per USD {j.loc[START]:.1f} to {j.loc[END]:.1f}")
    for a_yen in (0.0, 0.1, 0.2):
        def cc(v, y):  # revenue abroad at the exchange rates of START
            return v * ((1 - a_yen) * e.loc[START] / e.loc[y] + a_yen * j.loc[y] / j.loc[START])
        q = p.copy()
        q["F0"] = [cc(v, y - 1) for v, y in zip(q["F0"], q["year"])]
        q["F1"] = [cc(v, y) for v, y in zip(q["F1"], q["year"])]
        q["us_part"], q["ab_part"] = shapley(q["U0"], q["F0"], q["U1"], q["F1"])
        out = []
        for b in ["5-9", "10+"]:
            g = q[q["band"] == b]
            out.append(f"{b} US {pts((g['wbar'] * g['us_part']).sum())}, abroad {pts((g['wbar'] * g['ab_part']).sum())}")
        print(f"  abroad revenue at {START} exchange rates, a {a_yen:.0%} share of it in yen and the rest in euros: " + "; ".join(out))
    print("Leaving out each of the largest contributors, ten years and more (within = US + abroad):")
    for k in ["humira", "gardasil", "zetia", "crestor", "truvada"]:
        g = old[old["key"] != k]
        print(f"  without {k:10s} {pts(g['within'].sum())} = {pts(g['w_us'].sum())} + {pts(g['w_ab'].sum())}")
    g = old[~old["key"].isin(["humira", "gardasil"])]
    print(f"  without humira and gardasil {pts(g['within'].sum())} = {pts(g['w_us'].sum())} + {pts(g['w_ab'].sum())}")

    (HERE / "figure_data.json").write_text(json.dumps(figure, indent=1, default=float) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
