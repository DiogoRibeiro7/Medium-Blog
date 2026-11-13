"""Follow-up 2 to the taxation-efficiency-drug-prices series: a test of the second conjecture of Part 6.

Every number quoted in the article is printed by this script. Run, after fetch.py has put the files in data/ (the
GitHub Actions workflow does this and commits them):

    python portfolio_numbers.py > output.txt

The conjecture (Part 6). The US share of the research-based industry's revenue keeps rising because new medicines
are launched first, and priced highest, in the United States: the share is driven by the composition of the
portfolio, in which the newest medicines dominate, more than by the price gap on existing ones. Two predictions:

  P1  The rise in the US share should be concentrated in medicines launched in the period.
  P2  The US share should be higher for the companies whose revenue comes more from medicines less than five years
      old.

The companies' annual reports tag revenue by product and by country in XBRL. This script rebuilds, for each company
and year, the revenue of each medicine in the United States and in total; dates each medicine by its first US
approval (Drugs@FDA and the Purple Book); and splits the change in the pooled US share into the parts that come from
medicines launched in the period, from older medicines, and from everything else.

Sections
1. Data and coverage.
2. From XBRL members to dated medicines.
3. The US share to explain.
4. Where the change comes from: an exact decomposition, product by product.
5. The same by launch cohort, between the end years.
6. The life cycle of a medicine's US share.
7. P2: companies with more revenue from new medicines.
8. Currencies, acquisitions and other checks.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ROOT = HERE.parent

SEED = 20261113
REPS = 5000
NEW = 4            # a medicine is "new" in year t if first approved in the US in years t-4 to t (less than five years)
START, END = 2016, 2025

PRODUCT_AXES = {"ProductOrServiceAxis", "ProductsAndServicesAxis"}
GEO_AXES = {"StatementGeographicalAxis", "GeographicalAreasAxis", "MarketsOfCustomersAxis", "GeographicalRegionsAxis"}
# axes that split a product's revenue into parts (type of arrangement, partner, line of the income statement): a fact
# carrying one of them is not the product's revenue
PART_AXES = re.compile(r"(TypeOfArrangement|Alliance|Counterpart|MajorCustomer|Derivative|IncomeStatementLocation|"
                       r"Reclassification|EquityComponents|Hedging|Adjustments|InitialApplication|Capitalized|"
                       r"ResearchAndDevelopment|SalesMilestone|Disposal|BusinessAcquisition|Nonrecurring|Scenario|"
                       r"SignificantInvestment|TypesOfAgreement|PrincipalAlliances|LocationAxis)")
US_MEMBER = re.compile(r"^(country:US|[\w-]+:(Ifrs)?(Country)?(US|Us|USA|UnitedStates|UnitedStatesOfAmerica)(Member|Segment)?)$")
STANDARD = {"Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet",
            "Revenue", "RevenueFromSaleOfGoods", "RevenueFromContractWithCustomerIncludingAssessedTax"}
# members that name a group of products, a business or a kind of revenue, not a medicine
GROUP = re.compile(r"^(other|total|collaboration|royalt|product$|products$|productand|revenue|contract|livestock|"
                   r"companion)", re.I)
# names used outside the US, or misspelt in a filing, for medicines approved in the US under another name
ALIASES = {"utomiris": "ultomiris", "forxiga": "farxiga", "brilique": "brilinta", "novorapid": "novolog"}


def section(title: str) -> None:
    print()
    print("=" * 94)
    print(title)
    print("=" * 94)


def norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def fiscal_year(end: str) -> int:
    d = date.fromisoformat(end)
    return d.year if d.month >= 7 else d.year - 1


def tokens(member: str) -> list[str]:
    name = re.sub(r"Member$", "", member.split(":")[-1])
    name = re.sub(r"Member$", "", name)
    return re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", name)


def load_brands() -> pd.DataFrame:
    """First US approval of each brand name: the product-level licence date of the Purple Book for biologics (it dates
    a new brand licensed under an older application correctly), else the original approval of the NDA or BLA."""
    fda = pd.read_csv(DATA / "fda_drugsfda.csv.gz")
    fda = fda[fda["application_number"].str.startswith(("NDA", "BLA"))].copy()
    fda["key"] = fda["brand_name"].map(norm)
    fda["date"] = pd.to_datetime(fda["first_approval"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    pb = pd.read_csv(DATA / "fda_purplebook.csv.gz", dtype=str)
    pb["key"] = pb["Proprietary Name"].map(norm)
    pb["date"] = pd.to_datetime(pb["Approval Date"], format="mixed", errors="coerce")
    a = fda.groupby("key").agg(date_fda=("date", "min"), sponsor_fda=("sponsor", "first"))
    b = pb.groupby("key").agg(date_pb=("date", "min"), center=("Center", "first"), sponsor_pb=("Applicant", "first"))
    br = a.join(b, how="outer")
    br["date"] = br["date_pb"].fillna(br["date_fda"])
    br["source"] = np.where(br["date_pb"].notna(), "Purple Book " + br["center"].fillna(""), "Drugs@FDA")
    return br[br["date"].notna() & (br.index.str.len() >= 5)]


def matcher(brands: pd.DataFrame):
    keys = set(brands.index)
    dates = brands["date"].to_dict()
    by_len = sorted(keys, key=len, reverse=True)

    def match(member: str) -> str | None:
        toks = tokens(member)
        if not toks:
            return None
        joined = norm("".join(toks))
        if GROUP.search(joined) and not any(norm("".join(toks[i:])).startswith(k) for i in range(1, len(toks))
                                            for k in keys if len(k) >= 6):
            return None
        for i in range(len(toks)):                      # leftmost word that begins a brand name
            rest = norm("".join(toks[i:]))
            rest = ALIASES.get(rest, rest)
            if rest in keys:                           # the member is a brand name
                return rest
            if len(rest) >= 6:                         # the member begins a longer brand name ("Breztri Aerosphere")
                cand = sorted((k for k in keys if k.startswith(rest)), key=lambda k: (dates[k], k))
                if cand:                               # the earliest approved of them
                    return cand[0]
            for k in by_len:                           # the longest brand name the member begins with
                if rest.startswith(k):
                    return k
        return None

    return match


def company_totals(f: pd.DataFrame) -> pd.DataFrame:
    """Total revenue and US revenue of each company and fiscal year: the revenue concept with a US member on a
    geographic axis and the largest total, as reported in the earliest filing that has it (as in Part 6)."""
    rows = []
    for (co, py), g in f.groupby(["company", "py"]):
        nod = g[(g["dims"] == "") & g["concept"].isin(STANDARD)]
        geo = g[g["n_dims"].eq(1) & g["geo"].map(lambda m: bool(US_MEMBER.match(m))) & g["concept"].isin(STANDARD)]
        best = None
        for lag in sorted(g["lag"].unique()):
            n = nod[nod["lag"] == lag].groupby("concept")["value"].max()
            u = geo[geo["lag"] == lag].groupby("concept")["value"].max()
            both = [c for c in n.index if c in u.index]
            if both:
                c = max(both, key=lambda c: n[c])
                best = (n[c], u[c], c, lag, g["currency"].iloc[0])
                break
        if best:
            rows.append({"company": co, "py": py, "total": best[0], "us": best[1], "concept": best[2], "lag": best[3],
                         "currency": best[4]})
    return pd.DataFrame(rows)


def product_panel(f: pd.DataFrame) -> pd.DataFrame:
    """Revenue of each product member in the US and in total, by company and fiscal year, from the earliest filing
    that tags the product with a US member."""
    p = f[f["product"] != ""].copy()
    p = p[~p["dims"].map(lambda d: bool(PART_AXES.search(d)))]
    p = p[~p["dims"].str.contains("ConsolidationItemsAxis=") | p["dims"].str.contains("OperatingSegmentsMember")]
    p = p[p["n_geo"] <= 1]
    p["is_us"] = p["geo"].map(lambda m: bool(US_MEMBER.match(m)))
    first = p[p["is_us"]].groupby(["company", "py"])["lag"].min().rename("blag")
    p = p.join(first, on=["company", "py"], how="inner")
    p = p[p["lag"] == p["blag"]]
    # one concept per company and year: the one with the most product facts on a US member
    conc = (p[p["is_us"]].groupby(["company", "py", "concept"]).size().rename("n").reset_index()
            .sort_values(["n", "concept"]).groupby(["company", "py"]).tail(1))
    p = p.merge(conc[["company", "py", "concept"]], on=["company", "py", "concept"])
    tot = p[p["geo"] == ""].groupby(["company", "py", "product"])["value"].max().rename("total")
    us = p[p["is_us"]].groupby(["company", "py", "product"])["value"].max().rename("us")
    comp = re.compile(r"(NonUs|International|RestOfWorld|ROW|Outside|Foreign)", re.I)
    non = (p[(p["geo"] != "") & ~p["is_us"] & p["geo"].map(lambda m: bool(comp.search(m.split(":")[-1])))]
           .groupby(["company", "py", "product"])["value"].max().rename("nonus"))
    out = pd.concat([tot, us, non], axis=1).reset_index()
    out["total"] = out["total"].fillna(out["us"] + out["nonus"])
    return out


def gr_terms(prev: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    """Exact Griliches-Regev split of the change in S = sum(w s) between two years, by unit. Units in both years:
    within = mean(w) * change(s), between = (mean(s) - mean(S)) * change(w). Units only in the later year (entry):
    w (s - mean(S)); units only in the earlier year (exit): -w (s - mean(S))."""
    s0 = (prev["w"] * prev["s"]).sum()
    s1 = (cur["w"] * cur["s"]).sum()
    sb = (s0 + s1) / 2
    m = prev[["unit", "w", "s"]].merge(cur[["unit", "w", "s"]], on="unit", how="outer", suffixes=("0", "1"))
    both = m["w0"].notna() & m["w1"].notna()
    m["within"] = np.where(both, (m["w0"] + m["w1"]) / 2 * (m["s1"] - m["s0"]), 0.0)
    m["between"] = np.where(both, ((m["s0"] + m["s1"]) / 2 - sb) * (m["w1"] - m["w0"]), 0.0)
    m["entry"] = np.where(m["w0"].isna(), m["w1"] * (m["s1"] - sb), 0.0)
    m["exit"] = np.where(m["w1"].isna(), -m["w0"] * (m["s0"] - sb), 0.0)
    m["total"] = m[["within", "between", "entry", "exit"]].sum(axis=1)
    check = m["total"].sum() - (s1 - s0)
    if abs(check) > 1e-12:
        raise SystemExit(f"decomposition does not add up: {check}")
    return m


def main() -> None:
    rng = np.random.default_rng(SEED)
    figure: dict = {}

    # ------------------------------------------------------------------ 1
    section("1. Data and coverage")
    f = pd.read_csv(DATA / "sec_revenue_facts.csv.gz")
    filings = pd.read_csv(DATA / "sec_filings.csv.gz")
    f["dims"] = f["dims"].fillna("")
    f["py"] = f["end"].map(fiscal_year)
    f["lag"] = f["fiscal_year"] - f["py"]
    f = f[f["lag"] >= 0]
    dd = f["dims"].map(lambda d: dict(x.split("=", 1) for x in d.split(";") if x))
    f["n_dims"] = dd.map(len)
    f["product"] = dd.map(lambda d: next((v for k, v in d.items() if k in PRODUCT_AXES), ""))
    f["n_geo"] = dd.map(lambda d: sum(k in GEO_AXES for k in d))
    f["geo"] = dd.map(lambda d: next((v for k, v in d.items() if k in GEO_AXES), ""))
    print(f"SEC annual reports read: {len(filings)} ({filings['form'].value_counts().to_dict()}), "
          f"{filings['company'].nunique()} companies, fiscal years {filings['fiscal_year'].min()}-"
          f"{filings['fiscal_year'].max()}; annual revenue facts kept: {len(f):,}")
    fx = pd.read_csv(ROOT / "part6" / "data" / "ecb_fx.csv.gz").pivot(index="time_period", columns="currency",
                                                                       values="obs_value")
    comp = company_totals(f)
    panel = product_panel(f)
    with_us = panel[panel["us"].notna()]
    span = with_us.groupby("company")["py"].agg(["min", "max", "nunique"])
    print("Companies whose filings tag revenue by product and by country (US member), and the fiscal years covered:")
    for co, r in span.iterrows():
        print(f"  {co:20s} {r['min']}-{r['max']} ({r['nunique']} years)")
    never = sorted(set(filings["company"]) - set(span.index))
    print(f"  never by product and country: {', '.join(never)}")

    # ------------------------------------------------------------------ 2
    section("2. From XBRL members to dated medicines")
    brands = load_brands()
    print(f"Brand names with a first US approval date: {len(brands):,} (Purple Book {brands['source'].str.startswith('Purple').sum():,}, "
          f"of which CBER {(brands['source'] == 'Purple Book CBER').sum()}; Drugs@FDA {(brands['source'] == 'Drugs@FDA').sum():,})")
    match = matcher(brands)
    members = panel["product"].drop_duplicates()
    keymap = {m: match(m) for m in members}
    panel["key"] = panel["product"].map(keymap)
    panel = panel.join(brands[["date", "source"]], on="key")
    panel["approval"] = panel["date"].dt.year
    n_m = sum(v is not None for v in keymap.values())
    print(f"Product members: {len(members)}; matched to a dated US brand {n_m}; left as other revenue {len(members) - n_m}")
    # several members of one company and year can name one medicine (renamed members, a product and its total):
    # keep the largest
    med = (panel[panel["key"].notna() & panel["us"].notna() & panel["total"].notna() & (panel["total"] > 0)]
           .sort_values("total").groupby(["company", "py", "key"]).tail(1).copy())
    med["us"] = med["us"].clip(upper=med["total"])
    print(f"Medicine-years with US and total revenue: {len(med):,}; distinct medicines {med[['company', 'key']].drop_duplicates().shape[0]}")
    big = med[med["py"] == END].sort_values("total", ascending=False).head(12)
    print(f"Largest medicines in {END} (company, approval year, US share):")
    for r in big.itertuples():
        print(f"  {r.key:18s} {r.company:18s} approved {int(r.approval)}  US {r.us / r.total:5.1%}  ({r.source})")

    # ------------------------------------------------------------------ 3
    section("3. The US share to explain")

    def usd(v, cur, year):
        if cur == "USD":
            return v
        eur = v if cur == "EUR" else v / fx.loc[year, cur]
        return eur * fx.loc[year, "USD"]

    ccy = f.groupby("company")["currency"].agg(lambda x: x.mode().iloc[0])
    med["cur"] = med["company"].map(ccy)
    med["total_usd"] = [usd(v, c, y) for v, c, y in zip(med["total"], med["cur"], med["py"])]
    med["us_usd"] = [usd(v, c, y) for v, c, y in zip(med["us"], med["cur"], med["py"])]
    comp["total_usd"] = [usd(v, c, y) for v, c, y in zip(comp["total"], comp["currency"], comp["py"])]
    comp["us_usd"] = [usd(v, c, y) for v, c, y in zip(comp["us"], comp["currency"], comp["py"])]
    print(f"Currencies: " + ", ".join(f"{c} {k}" for c, k in ccy.items() if k != "USD") + "; the rest report in US "
          f"dollars. Converted at ECB annual averages (Part 6).")
    print("The universe is the medicines each company reports one by one, in the US and in total, and that carry a US "
          "brand name with an approval date. Share of each company's revenue they cover (where the company's own total "
          "and US revenue are tagged):")
    cy = med.groupby(["company", "py"])[["total_usd", "us_usd"]].sum()
    cv = cy.join(comp.set_index(["company", "py"])[["total_usd", "us_usd"]], rsuffix="_co", how="inner")
    cv["coverage"] = cv["total_usd"] / cv["total_usd_co"]
    for co, g in cv.reset_index().groupby("company"):
        g = g.sort_values("py")
        print(f"  {co:20s} {g['py'].iloc[0]} {g['coverage'].iloc[0]:5.1%}   {g['py'].iloc[-1]} {g['coverage'].iloc[-1]:5.1%}")

    def sample(y0: int, y1: int) -> list[str]:
        yrs = set(range(y0, y1 + 1))
        return sorted(co for co, g in med.groupby("company") if yrs <= set(g["py"])
                      and g.groupby("py")["total_usd"].sum().min() > 1e9)

    windows = {"main": (START, END), "2018": (2018, END), "2021": (2021, END)}
    samples = {k: sample(*v) for k, v in windows.items()}
    for k, (y0, y1) in windows.items():
        print(f"Companies with dated medicines in every year {y0}-{y1}: {len(samples[k])}: {', '.join(samples[k])}")
    main_cos = samples["main"]
    sm = med[med["company"].isin(main_cos) & med["py"].between(START, END)].groupby("py")[["total_usd", "us_usd"]].sum()
    sm["s"] = sm["us_usd"] / sm["total_usd"]
    print(f"US share of the dated medicines, main sample: " + ", ".join(f"{y} {r.s:.1%}" for y, r in sm.iterrows()))
    print(f"  change {START}-{END}: {100 * (sm['s'].iloc[-1] - sm['s'].iloc[0]):+.1f} points; revenue of these medicines "
          f"$ {sm['total_usd'].iloc[0] / 1e9:,.0f} bn to $ {sm['total_usd'].iloc[-1] / 1e9:,.0f} bn")
    rep = comp[comp["company"].isin(main_cos)]
    full = [c for c in main_cos if set(range(START, END + 1)) <= set(rep.loc[rep["company"] == c, "py"])]
    ag = rep[rep["company"].isin(full) & rep["py"].between(START, END)].groupby("py")[["total_usd", "us_usd"]].sum()
    ag["s"] = ag["us_usd"] / ag["total_usd"]
    smf = med[med["company"].isin(full) & med["py"].between(START, END)].groupby("py")[["total_usd", "us_usd"]].sum()
    smf["s"] = smf["us_usd"] / smf["total_usd"]
    print(f"Companies of the main sample whose own total and US revenue are tagged every year ({', '.join(full)}):")
    print(f"  US share of all their revenue: " + ", ".join(f"{y} {r.s:.1%}" for y, r in ag.iterrows()))
    print(f"  US share of their dated medicines: " + ", ".join(f"{y} {r.s:.1%}" for y, r in smf.iterrows()))
    print(f"  change: all revenue {100 * (ag['s'].iloc[-1] - ag['s'].iloc[0]):+.1f} points, dated medicines "
          f"{100 * (smf['s'].iloc[-1] - smf['s'].iloc[0]):+.1f}; correlation of the yearly changes "
          f"{np.corrcoef(ag['s'].diff().dropna(), smf['s'].diff().dropna())[0, 1]:+.2f}")
    p6 = (ROOT / "part6" / "output.txt").read_text(encoding="utf-8")
    m7 = re.search(r"the 7 companies tagged in all eleven years.*?\n  2015 ([\d.]+)%.*?2025 ([\d.]+)%", p6, re.S)
    print(f"  Part 6, seven US companies with the US tagged in every year 2015-2025: {m7.group(1)}% to {m7.group(2)}%")
    figure["pooled"] = {"years": [int(y) for y in sm.index], "medicines": sm["s"].round(5).tolist(),
                        "reported_years": [int(y) for y in ag.index], "reported": ag["s"].round(5).tolist()}

    # ------------------------------------------------------------------ 4
    section("4. Where the change comes from: an exact decomposition, medicine by medicine")

    def units(year: int, cos: list[str]) -> pd.DataFrame:
        m = med[(med["py"] == year) & med["company"].isin(cos)]
        u = pd.DataFrame({"unit": m["company"] + "|" + m["key"], "company": m["company"], "key": m["key"],
                          "approval": m["approval"], "rev": m["total_usd"], "usrev": m["us_usd"]})
        u["w"] = u["rev"] / u["rev"].sum()
        u["s"] = u["usrev"] / u["rev"]
        return u.reset_index(drop=True)

    def chained(cos: list[str], y0: int, y1: int):
        parts = []
        for y in range(y0 + 1, y1 + 1):
            prev, cur_ = units(y - 1, cos), units(y, cos)
            t = gr_terms(prev, cur_)
            info = pd.concat([prev, cur_]).drop_duplicates("unit", keep="last").set_index("unit")
            for c in ("key", "approval", "company"):
                t[c] = t["unit"].map(info[c])
            t["cohort"] = np.where(t["approval"] > y0, "launched in the period", "launched before")
            t["age_group"] = np.where(y - t["approval"] <= NEW, "new (under five years)", "established")
            t["year"] = y
            t["first_seen"] = t["unit"].map(med[med["company"].isin(cos)].groupby(med["company"] + "|" + med["key"])["py"].min())
            parts.append(t)
        allt = pd.concat(parts)
        s0 = units(y0, cos).eval("w * s").sum()
        s1 = units(y1, cos).eval("w * s").sum()
        return allt, s0, s1

    allt, s0, s1 = chained(main_cos, START, END)
    d = s1 - s0
    print(f"Main sample, {len(main_cos)} companies, {START}-{END}: US share of dated medicines {s0:.1%} to {s1:.1%}, "
          f"change {100 * d:+.2f} points (sum of the yearly terms {100 * allt['total'].sum():+.2f})")
    print("By launch cohort (each medicine's within, between, entry and exit terms, summed over the years):")
    by_c = allt.groupby("cohort")[["within", "between", "entry", "exit", "total"]].sum()
    for c in ["launched in the period", "launched before"]:
        r = by_c.loc[c]
        print(f"  {c:24s} total {100 * r['total']:+6.2f} points ({r['total'] / d:6.1%} of the change): within "
              f"{100 * r['within']:+.2f}, between {100 * r['between']:+.2f}, entry {100 * r['entry']:+.2f}, exit "
              f"{100 * r['exit']:+.2f}")
    print("By age in the year of each term (new = first US approval less than five years before):")
    by_a = allt.groupby("age_group")[["within", "between", "entry", "exit", "total"]].sum()
    for c in ["new (under five years)", "established"]:
        r = by_a.loc[c]
        print(f"  {c:24s} total {100 * r['total']:+6.2f} points ({r['total'] / d:6.1%}): within {100 * r['within']:+.2f}, "
              f"between {100 * r['between']:+.2f}, entry {100 * r['entry']:+.2f}, exit {100 * r['exit']:+.2f}")
    ent = allt[allt["entry"] != 0].copy()
    ent["age_at_entry"] = ent["year"] - ent["approval"]
    launch = ent["age_at_entry"] <= 2
    print(f"Entries into the reported set: {len(ent)}; within two years of US approval {int(launch.sum())} "
          f"({100 * ent.loc[launch, 'entry'].sum():+.2f} points); older medicines (acquired, newly reported) "
          f"{int((~launch).sum())} ({100 * ent.loc[~launch, 'entry'].sum():+.2f} points)")
    old_in = ent[~launch].sort_values("entry")
    print("  largest entries of older medicines: " + "; ".join(
        f"{r.key} ({r.company}, {int(r.year)}, approved {int(r.approval)}) {100 * r.entry:+.2f}"
        for r in old_in.iloc[::-1].head(5).itertuples()))
    top = allt.groupby(["company", "key", "approval"])["total"].sum().sort_values()
    print("Largest contributions by medicine (points of US share, all terms):")
    for (co, k, a), v in list(top.tail(10)[::-1].items()):
        print(f"  {k:18s} {co:18s} approved {int(a)}  {100 * v:+.2f}")
    print("  ...")
    for (co, k, a), v in list(top.head(6).items()):
        print(f"  {k:18s} {co:18s} approved {int(a)}  {100 * v:+.2f}")
    mm = med[med["company"].isin(main_cos) & med["py"].between(START, END)]
    print("  the same medicines, first and last year reported in the window (US share, revenue $ bn):")
    for (co, k, a), v in list(top.tail(10)[::-1].items()) + list(top.head(6).items()):
        g = mm[(mm["company"] == co) & (mm["key"] == k)].sort_values("py")
        f0, f1 = g.iloc[0], g.iloc[-1]
        print(f"    {k:18s} {int(f0.py)} US {f0.us_usd / f0.total_usd:5.1%} $ {f0.total_usd / 1e9:5.1f}   {int(f1.py)} US "
              f"{f1.us_usd / f1.total_usd:5.1%} $ {f1.total_usd / 1e9:5.1f}")
    byco = allt.groupby("company")["total"].sum().sort_values(ascending=False)
    print("By company (points): " + ", ".join(f"{c} {100 * v:+.2f}" for c, v in byco.items()))
    figure["decomposition"] = {
        "change": d, "s0": s0, "s1": s1,
        "cohort": {c: {k: float(by_c.loc[c, k]) for k in ["within", "between", "entry", "exit"]} for c in by_c.index},
        "age": {c: {k: float(by_a.loc[c, k]) for k in ["within", "between", "entry", "exit"]} for c in by_a.index},
        "top": [{"key": k, "company": co, "approval": int(a), "points": float(v)}
                for (co, k, a), v in list(top.tail(10)[::-1].items()) + list(top.head(5).items())]}
    print("Robustness (same decomposition, other windows and samples):")
    for label, cos, y0, y1 in [("2018-2025", samples["2018"], 2018, END), ("2021-2025", samples["2021"], 2021, END),
                               (f"{START}-{END} without Lilly", [c for c in main_cos if c != "Eli Lilly"], START, END),
                               (f"{START}-{END} without AbbVie", [c for c in main_cos if c != "AbbVie"], START, END),
                               (f"{START}-{END} without Merck", [c for c in main_cos if c != "Merck & Co"], START, END)]:
        t, a0, a1 = chained(cos, y0, y1)
        bc = t.groupby("cohort")["total"].sum()
        ba = t.groupby("age_group")["total"].sum()
        dd_ = a1 - a0
        print(f"  {label:26s} {len(cos):2d} companies: {a0:.1%} to {a1:.1%} ({100 * dd_:+.2f}); launched in the period "
              f"{100 * bc.get('launched in the period', 0):+.2f}, before {100 * bc.get('launched before', 0):+.2f}; "
              f"new {100 * ba.get('new (under five years)', 0):+.2f}, established {100 * ba.get('established', 0):+.2f}")

    print("How much of the change the age split gives to young medicines, by where the line between young and "
          "established is drawn:")
    t21, a21, b21 = chained(samples["2021"], 2021, END)
    for k in (4, 7, 9):
        r_main = allt.loc[(allt["year"] - allt["approval"]) <= k, "total"].sum()
        r_21 = t21.loc[(t21["year"] - t21["approval"]) <= k, "total"].sum()
        print(f"  under {k + 1} years old: {START}-{END} {100 * r_main:+.2f} of {100 * d:+.2f} ({r_main / d:.0%}); "
              f"2021-{END} {100 * r_21:+.2f} of {100 * (b21 - a21):+.2f} ({r_21 / (b21 - a21):.0%})")

    # ------------------------------------------------------------------ 5
    section("5. The same by launch cohort, between the end years")
    u0, u1 = units(START, main_cos), units(END, main_cos)
    for u in (u0, u1):
        u["cohort"] = np.where(u["approval"] > START, "launched in the period", "launched before")
    g0 = u0.groupby("cohort").agg(W=("w", "sum"), rev=("rev", "sum"), usrev=("usrev", "sum"))
    g1 = u1.groupby("cohort").agg(W=("w", "sum"), rev=("rev", "sum"), usrev=("usrev", "sum"))
    print(f"Revenue weight and US share by cohort, {START} and {END}:")
    for c in ["launched before", "launched in the period"]:
        a = g0.loc[c] if c in g0.index else pd.Series({"W": 0.0, "rev": 0.0, "usrev": 0.0})
        b = g1.loc[c]
        sa = a["usrev"] / a["rev"] if a["rev"] else float("nan")
        print(f"  {c:24s} {START}: weight {a['W']:5.1%}, US share {sa:5.1%}, $ {a['rev'] / 1e9:5.0f} bn   {END}: "
              f"weight {b['W']:5.1%}, US share {b['usrev'] / b['rev']:5.1%}, $ {b['rev'] / 1e9:5.0f} bn")
    sb_ = (s0 + s1) / 2
    newc = g1.loc["launched in the period"]
    print(f"  contribution of the new cohort between the end years, W x (s - mean S): "
          f"{100 * newc['W'] * (newc['usrev'] / newc['rev'] - sb_):+.2f} points of {100 * d:+.2f}")
    cont = u0[u0["approval"] <= START].merge(u1[["unit", "rev", "usrev"]], on="unit", suffixes=("0", "1"))
    s_c0 = cont["usrev0"].sum() / cont["rev0"].sum()
    s_c1 = cont["usrev1"].sum() / cont["rev1"].sum()
    print(f"Medicines launched by {START} and reported in both years ({len(cont)}): US share {s_c0:.1%} in {START}, "
          f"{s_c1:.1%} in {END}; revenue $ {cont['rev0'].sum() / 1e9:,.0f} bn to $ {cont['rev1'].sum() / 1e9:,.0f} bn")
    s_fix = (cont["rev1"] * (cont["usrev0"] / cont["rev0"])).sum() / cont["rev1"].sum()
    print(f"  at their {START} US shares and {END} revenue the group's share would be {s_fix:.1%}: change of mix inside "
          f"the group {100 * (s_fix - s_c0):+.1f} points, change of each medicine's own share {100 * (s_c1 - s_fix):+.1f}")
    cont["ds"] = cont["usrev1"] / cont["rev1"] - cont["usrev0"] / cont["rev0"]
    print(f"  revenue-weighted change in a medicine's own US share: {100 * np.average(cont['ds'], weights=cont['rev0']):+.1f} "
          f"points; median {100 * cont['ds'].median():+.1f}; rose for {(cont['ds'] > 0).mean():.0%} of them")
    gone = u0[(u0["approval"] <= START) & ~u0["unit"].isin(u1["unit"])]
    print(f"  reported in {START} but not in {END}: {len(gone)} medicines, {gone['w'].sum():.1%} of {START} revenue, "
          f"US share {gone['usrev'].sum() / gone['rev'].sum():.1%}")
    figure["cohort"] = {"g0": g0.to_dict("index"), "g1": g1.to_dict("index"), "cont": [s_c0, s_fix, s_c1]}

    # ------------------------------------------------------------------ 6
    section("6. The life cycle of a medicine's US share")
    lc = med[med["approval"].notna()].copy()
    lc["age"] = lc["py"] - lc["approval"]
    lc["s"] = lc["us_usd"] / lc["total_usd"]
    lc = lc[lc["total_usd"] >= 50e6]
    young = lc[lc["approval"] >= START - 2]
    print(f"Medicine-years with revenue of at least $ 50 million: {len(lc):,}; medicines first approved from "
          f"{START - 2}: {young[['company', 'key']].drop_duplicates().shape[0]} (those approved from {START} observed from "
          f"their launch, earlier ones from {START})")
    prof = young[young["age"].between(0, 8)].groupby("age").agg(
        n=("s", "size"), med=("s", "median"), q1=("s", lambda x: x.quantile(.25)), q3=("s", lambda x: x.quantile(.75)),
        us=("us_usd", "sum"), rev=("total_usd", "sum"))
    print(f"US share by years since first US approval, medicines approved from {START - 2}:")
    for a, r in prof.iterrows():
        print(f"  age {int(a)}: n {int(r.n):3d}  median {r.med:5.1%} [{r.q1:5.1%}, {r.q3:5.1%}]  revenue-weighted "
              f"{r.us / r.rev:5.1%}")
    x = lc.copy()
    x["s_dm"] = x["s"] - x.groupby(["company", "key"])["s"].transform("mean")
    x["a_dm"] = x["age"] - x.groupby(["company", "key"])["age"].transform("mean")
    slopes_lc = {}
    for lab, sub in (("all medicines", x), ("first 8 years", x[x["age"].between(0, 8)]),
                     ("10 years and older", x[x["age"] >= 10])):
        sub = sub.copy()
        sub["s_dm"] = sub["s"] - sub.groupby(["company", "key"])["s"].transform("mean")
        sub["a_dm"] = sub["age"] - sub.groupby(["company", "key"])["age"].transform("mean")
        sub = sub[sub.groupby(["company", "key"])["age"].transform("size") >= 3]
        b = (sub["s_dm"] * sub["a_dm"]).sum() / (sub["a_dm"] ** 2).sum()
        grp = [g for _, g in sub.groupby(["company", "key"])]
        num = np.array([(g["s_dm"] * g["a_dm"]).sum() for g in grp])
        den = np.array([(g["a_dm"] ** 2).sum() for g in grp])
        idx = rng.integers(0, len(grp), (REPS, len(grp)))
        bs = num[idx].sum(axis=1) / den[idx].sum(axis=1)
        lo, hi = np.percentile(bs, [5, 95])
        slopes_lc[lab] = b
        print(f"  within-medicine change in US share per year of age, {lab}: {100 * b:+.2f} points [{100 * lo:+.2f}, "
              f"{100 * hi:+.2f}] ({len(grp)} medicines, {len(sub)} medicine-years)")
    figure["lifecycle"] = {"age": [int(a) for a in prof.index], "median": prof["med"].tolist(), "q1": prof["q1"].tolist(),
                           "q3": prof["q3"].tolist(), "weighted": (prof["us"] / prof["rev"]).tolist(),
                           "n": prof["n"].astype(int).tolist(), "companies": int(young["company"].nunique()),
                           "first": START - 2}
    print(f"  companies in the life-cycle sample: {young['company'].nunique()}")

    # ------------------------------------------------------------------ 7
    section("7. P2: companies with more revenue from new medicines")
    m2 = med[med["py"].between(START, END)].assign(new=lambda d: (d["py"] - d["approval"]) <= NEW)
    cyr = m2.groupby(["company", "py"]).apply(lambda g: pd.Series({
        "s": g["us_usd"].sum() / g["total_usd"].sum(),
        "n": g.loc[g["new"], "total_usd"].sum() / g["total_usd"].sum()}), include_groups=False).reset_index()
    cyr = cyr[cyr["company"].isin(samples["2021"] + [c for c in main_cos if c not in samples["2021"]])]
    print(f"Company-years: {len(cyr)}, companies {cyr['company'].nunique()} (US share and new-medicine share both "
          f"measured on the company's dated medicines)")
    last = cyr[cyr["py"] == END].sort_values("n", ascending=False)
    print(f"{END}: share of dated-medicine revenue from medicines under five years old, and their US share:")
    for r in last.itertuples():
        print(f"  {r.company:20s} new {r.n:5.1%}  US {r.s:5.1%}")
    first = cyr.sort_values("py").groupby("company").head(1).set_index("company")
    print("  first year in the data: " + "; ".join(f"{c} {int(r.py)} new {r.n:.1%} US {r.s:.1%}" for c, r in first.iterrows()))
    rk = last[["n", "s"]].rank()
    rho = np.corrcoef(rk["n"], rk["s"])[0, 1]
    print(f"  rank correlation across companies in {END}: {rho:+.2f} (n={len(last)})")

    def slopes(df: pd.DataFrame) -> tuple[float, float]:
        b_pool = np.polyfit(df["n"], df["s"], 1)[0]
        dm = df.assign(s_=df["s"] - df.groupby("company")["s"].transform("mean"),
                       n_=df["n"] - df.groupby("company")["n"].transform("mean"))
        dm = dm.assign(s_=dm["s_"] - dm.groupby("py")["s_"].transform("mean"),
                       n_=dm["n_"] - dm.groupby("py")["n_"].transform("mean"))
        return b_pool, (dm["s_"] * dm["n_"]).sum() / (dm["n_"] ** 2).sum()

    b_pool, b_fe = slopes(cyr)
    cos = cyr["company"].unique()
    bs = np.array([slopes(pd.concat([cyr[cyr["company"] == c].assign(company=f"{c}{i}")
                                     for i, c in enumerate(rng.choice(cos, len(cos), replace=True))]))
                   for _ in range(1000)])
    lo_p, hi_p = np.percentile(bs[:, 0], [5, 95])
    lo_f, hi_f = np.percentile(bs[:, 1], [5, 95])
    print(f"  pooled slope: {b_pool:+.3f} [{lo_p:+.3f}, {hi_p:+.3f}] (change in US share per unit of new-medicine "
          f"share; 90% interval, companies resampled 1000 times)")
    print(f"  within companies, net of year effects: {b_fe:+.3f} [{lo_f:+.3f}, {hi_f:+.3f}]")
    figure["companies"] = cyr.to_dict("records")
    figure["p2"] = {"b_fe": b_fe, "rho": rho}

    # ------------------------------------------------------------------ 8
    section("8. Currencies")
    usd_per_eur = fx.loc[[START, END], "USD"]
    jpy = fx.loc[[START, END], "JPY"] / fx.loc[[START, END], "USD"]
    print(f"Dollars per euro: {usd_per_eur.iloc[0]:.3f} in {START}, {usd_per_eur.iloc[1]:.3f} in {END}; yen per dollar: "
          f"{jpy.iloc[0]:.1f} and {jpy.iloc[1]:.1f}")
    s1_ = sm["s"].iloc[-1]
    for lab, fall in (("all revenue outside the US moving with the euro", usd_per_eur.iloc[1] / usd_per_eur.iloc[0] - 1),
                      ("a tenth with the yen, the rest with the euro",
                       0.9 * (usd_per_eur.iloc[1] / usd_per_eur.iloc[0] - 1) + 0.1 * (jpy.iloc[0] / jpy.iloc[1] - 1)),
                      ("a fifth with the yen, the rest with the euro",
                       0.8 * (usd_per_eur.iloc[1] / usd_per_eur.iloc[0] - 1) + 0.2 * (jpy.iloc[0] / jpy.iloc[1] - 1))):
        adj = s1_ / (s1_ + (1 - s1_) / (1 + fall))
        print(f"  {END} US share at {START} exchange rates, {lab}: {adj:.1%} (at {END} rates {s1_:.1%})")
    with open(HERE / "figure_data.json", "w", encoding="utf-8") as fh:
        json.dump(figure, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
