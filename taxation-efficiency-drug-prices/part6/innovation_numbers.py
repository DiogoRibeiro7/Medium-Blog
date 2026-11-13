"""Part 6 of the taxation-efficiency-drug-prices series: who pays for pharmaceutical innovation.

Every number quoted in the article is printed by this script. Run, after part6/fetch.py has put the
public files in part6/data (the GitHub Actions workflow does this and commits them):

    python innovation_numbers.py > output.txt

The argument examined is that countries which pay low prices for medicines free-ride on the United
States, whose high prices finance the research that produces new medicines for everyone. The script
measures the parts of that argument that public data can reach.

Sections
1. Data and coverage.
2. Where the revenue of seventeen research-based companies comes from: the US share of revenue in
   their annual reports, by company and over time.
3. The US weight in world income and population, for comparison with its weight in revenue.
4. From revenue to margins: the US share of revenue above the cost of sales, under the price gaps that
   the RAND 2024 study estimates, as an illustration of how revenue shares become profit shares.
5. Whether revenue earned outside the United States would cover the companies' R&D.
6. Europe country by country: market, industry R&D located in the country, business R&D in the
   pharmaceutical industry and public budgets for health research, each as a share of GDP.
7. What the literature's elasticities imply for the number of new medicines if Portugal, or the
   largest European markets, paid more.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PART2 = HERE.parent / "part2"

SEED = 20261112
REPS = 5000

# The United States member of a geographic axis, and nothing else.
US_MEMBER = re.compile(r"^(country:US|[\w-]+:(Ifrs)?(Country)?(US|Us|USA|UnitedStates|UnitedStatesOfAmerica)(Member|Segment)?)$")
EUROPE_MEMBER = re.compile(r"(^srt:EuropeMember$|^us-gaap:EuropeMember$|IfrsEuropeMember$|DevelopedEuropeMember$|^[\w-]+:EuropeMember$)")

EFPIA_ISO = {
    "Austria": "AT", "Belgium": "BE", "Bulgaria": "BG", "Croatia": "HR", "Cyprus": "CY", "Czech Rep.": "CZ",
    "Czech Republic": "CZ", "Denmark": "DK", "Estonia": "EE", "Finland": "FI", "France": "FR", "Germany": "DE",
    "Greece": "EL", "Hungary": "HU", "Iceland": "IS", "Ireland": "IE", "Italy": "IT", "Latvia": "LV",
    "Lithuania": "LT", "Luxembourg": "LU", "Malta": "MT", "Netherlands": "NL", "Norway": "NO", "Poland": "PL",
    "Portugal": "PT", "Romania": "RO", "Serbia": "RS", "Slovakia": "SK", "Slovenia": "SI", "Spain": "ES",
    "Sweden": "SE", "Switzerland": "CH", "Turkey": "TR", "U.K.": "UK", "United Kingdom": "UK", "Ukraine": "UA",
}
ISO2_TO_3 = {"AT": "AUT", "BE": "BEL", "BG": "BGR", "HR": "HRV", "CY": "CYP", "CZ": "CZE", "DK": "DNK", "EE": "EST",
             "FI": "FIN", "FR": "FRA", "DE": "DEU", "EL": "GRC", "HU": "HUN", "IS": "ISL", "IE": "IRL", "IT": "ITA",
             "LV": "LVA", "LT": "LTU", "LU": "LUX", "MT": "MLT", "NL": "NLD", "NO": "NOR", "PL": "POL", "PT": "PRT",
             "RO": "ROU", "RS": "SRB", "SK": "SVK", "SI": "SVN", "ES": "ESP", "SE": "SWE", "CH": "CHE", "TR": "TUR",
             "UK": "GBR", "UA": "UKR", "US": "USA", "JP": "JPN"}
NAMES3 = {"AUT": "Austria", "BEL": "Belgium", "CHE": "Switzerland", "DEU": "Germany", "DNK": "Denmark", "ESP": "Spain",
          "FIN": "Finland", "FRA": "France", "GBR": "United Kingdom", "GRC": "Greece", "IRL": "Ireland", "ITA": "Italy",
          "JPN": "Japan", "NLD": "Netherlands", "NOR": "Norway", "POL": "Poland", "PRT": "Portugal", "SWE": "Sweden",
          "USA": "United States", "CZE": "Czechia", "HUN": "Hungary", "KOR": "Korea", "CAN": "Canada", "SVN": "Slovenia"}


def section(title: str) -> None:
    print()
    print("=" * 94)
    print(title)
    print("=" * 94)


def text(name: str) -> str:
    return gzip.open(DATA / f"{name}.txt.gz", "rt", encoding="utf-8").read().replace("\t", " ")


def spearman(x, y) -> float:
    return float(pd.Series(np.asarray(x, float)).rank().corr(pd.Series(np.asarray(y, float)).rank()))


def boot_corr(x, y, rng, fn=spearman) -> tuple[float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    vals = []
    for _ in range(REPS):
        i = rng.integers(0, n, n)
        if np.unique(x[i]).size > 1 and np.unique(y[i]).size > 1:
            vals.append(fn(x[i], y[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def efpia_table(t: str, head: str) -> tuple[dict[str, float], float]:
    i = t.find(head)
    j = t.find("TOTAL", i)
    block = t[i:j + 300]
    rows = re.findall(r"([A-Z][A-Za-z\.\-]+(?: [A-Za-z\.\-]+)*)\s{2,}(n\.a|\d[\d,]*)", block)
    out = {}
    for name, v in rows:
        if name in EFPIA_ISO and v != "n.a":
            out[EFPIA_ISO[name]] = float(v.replace(",", ""))
    total = float(re.search(r"TOTAL\s+([\d,]+)", block).group(1).replace(",", ""))
    return out, total


def must(t: str, *pieces: str) -> None:
    for p in pieces:
        if p not in t:
            raise SystemExit(f"EFPIA text changed: {p!r} not found")


def main() -> None:
    rng = np.random.default_rng(SEED)
    figure: dict = {}

    # ------------------------------------------------------------------------------------ 1
    section("1. Data and coverage")
    panel = pd.read_csv(DATA / "sec_panel.csv.gz")
    geo = pd.read_csv(DATA / "sec_geography.csv.gz")
    filings = pd.read_csv(DATA / "sec_filings.csv.gz")
    fx = pd.read_csv(DATA / "ecb_fx.csv.gz").pivot(index="time_period", columns="currency", values="obs_value")
    print(f"SEC annual reports read: {len(filings)} ({filings['form'].value_counts().to_dict()}), companies "
          f"{filings['company'].nunique()}, fiscal years {filings['fiscal_year'].min()}-{filings['fiscal_year'].max()}")
    # US revenue: the United States member of a geographic axis, in the revenue concept whose total is largest
    geo["is_us"] = geo["member"].map(lambda m: bool(US_MEMBER.match(str(m))))
    us = (geo[geo["is_us"] & geo["concept_total"].notna()]
          .sort_values(["concept_total", "value"], ascending=False)
          .drop_duplicates(["company", "fiscal_year"]))
    panel = panel.drop(columns=[c for c in panel.columns if c.startswith("us_")]).merge(
        us[["company", "fiscal_year", "value", "concept_total", "concept"]].rename(
            columns={"value": "us_rev", "concept_total": "geo_total", "concept": "geo_concept"}),
        on=["company", "fiscal_year"], how="left")
    panel["us_share"] = panel["us_rev"] / panel["geo_total"]
    eu = (geo[geo["member"].map(lambda m: bool(EUROPE_MEMBER.search(str(m)))) & geo["concept_total"].notna()]
          .sort_values(["concept_total", "value"], ascending=False).drop_duplicates(["company", "fiscal_year"]))
    panel = panel.merge(eu[["company", "fiscal_year", "value", "member"]].rename(
        columns={"value": "eu_rev", "member": "eu_member"}), on=["company", "fiscal_year"], how="left")
    panel["eu_share"] = panel["eu_rev"] / panel["geo_total"]

    # convert to US dollars at ECB annual average rates (currency per euro)
    def to_usd(row, col):
        v = row[col]
        if pd.isna(v):
            return np.nan
        cur = row["currency"]
        if cur == "USD":
            return v
        year = int(row["fiscal_year"])  # Takeda: April to March, converted at the rate of the year it starts
        eur = v if cur == "EUR" else v / fx.loc[year, cur]
        return eur * fx.loc[year, "USD"]

    if "sga" not in panel.columns:
        panel["sga"] = np.nan
    for c in ("revenue", "rd", "cost_of_sales", "net_income", "us_rev", "geo_total", "eu_rev", "sga"):
        panel[c + "_usd"] = panel.apply(lambda r, c=c: to_usd(r, c), axis=1)
    cov = panel.groupby("fiscal_year").agg(firms=("company", "size"), with_us=("us_share", "count"))
    print("firms with an annual report, and with the US revenue tagged, by fiscal year:")
    print("  " + "; ".join(f"{y}: {r.firms}/{r.with_us}" for y, r in cov.iterrows()))
    never = sorted(set(panel["company"]) - set(panel.loc[panel["us_share"].notna(), "company"]))
    print(f"  never tagged: {', '.join(never) or 'none'}")
    print("  US member found: " + "; ".join(f"{k} {v}" for k, v in us["member"].value_counts().items()))

    # ------------------------------------------------------------------------------------ 2
    section("2. Where the revenue comes from: the US share in the companies' own reports")
    last = panel[(panel["fiscal_year"] == 2025) & panel["us_share"].notna()].sort_values("us_share", ascending=False)
    print(f"Fiscal year 2025, {len(last)} companies with the US revenue tagged:")
    for r in last.itertuples():
        eu_txt = f", Europe {r.eu_share:.0%} ({r.eu_member.split(':')[-1]})" if pd.notna(r.eu_share) else ""
        print(f"  {r.company:22s} HQ {r.hq}  US {r.us_share:5.1%}{eu_txt}; revenue $ {r.revenue_usd / 1e9:6.1f} bn, "
              f"R&D {r.rd_usd / r.revenue_usd:5.1%} of revenue")
    pooled = last["us_rev_usd"].sum() / last["geo_total_usd"].sum()
    for hq, g in last.groupby(last["hq"].eq("US")):
        lab = "US-headquartered" if hq else "headquartered elsewhere"
        print(f"  pooled, {lab}: US share {g['us_rev_usd'].sum() / g['geo_total_usd'].sum():.1%} "
              f"(n={len(g)}, revenue $ {g['geo_total_usd'].sum() / 1e9:,.0f} bn)")
    print(f"  pooled, all {len(last)}: US share {pooled:.1%}; revenue $ {last['geo_total_usd'].sum() / 1e9:,.0f} bn, "
          f"of which US $ {last['us_rev_usd'].sum() / 1e9:,.0f} bn; median company {last['us_share'].median():.1%}")
    eu_last = last.dropna(subset=["eu_share"])
    print(f"  Europe, where reported as one area ({len(eu_last)} companies): pooled "
          f"{eu_last['eu_rev_usd'].sum() / eu_last['geo_total_usd'].sum():.1%}; "
          + ", ".join(f"{r.company} {r.eu_share:.0%}" for r in eu_last.itertuples()))
    # trend for the firms tagged in every year
    full = [c for c, g in panel.groupby("company") if g["us_share"].notna().sum() == 11]
    tr = panel[panel["company"].isin(full)].groupby("fiscal_year").apply(
        lambda g: g["us_rev_usd"].sum() / g["geo_total_usd"].sum(), include_groups=False)
    print(f"Pooled US share, the {len(full)} companies tagged in all eleven years ({', '.join(full)}):")
    print("  " + ", ".join(f"{y} {v:.1%}" for y, v in tr.items()))
    full18 = [c for c, g in panel[panel["fiscal_year"] >= 2018].groupby("company") if g["us_share"].notna().sum() == 8]
    tr18 = panel[panel["company"].isin(full18) & (panel["fiscal_year"] >= 2018)].groupby("fiscal_year").apply(
        lambda g: g["us_rev_usd"].sum() / g["geo_total_usd"].sum(), include_groups=False)
    print(f"  the {len(full18)} companies tagged in every year 2018-2025: " + ", ".join(f"{y} {v:.1%}" for y, v in tr18.items()))
    xp = panel[panel["company"].isin(full) & (panel["company"] != "Pfizer")].groupby("fiscal_year").apply(
        lambda g: g["us_rev_usd"].sum() / g["geo_total_usd"].sum(), include_groups=False)
    print("  the same without Pfizer (COVID-19 vaccine and antiviral sales abroad in 2021-2022): "
          + ", ".join(f"{y} {v:.1%}" for y, v in xp.items()))
    figure["companies_2025"] = [{"company": r.company, "hq": r.hq, "us_share": round(float(r.us_share), 4)}
                                for r in last.itertuples()]
    figure["trend"] = {"years": [int(y) for y in tr.index], "us_share": [round(float(v), 4) for v in tr.values],
                       "firms": full}

    # ------------------------------------------------------------------------------------ 3
    section("3. The US weight in world income and population")
    wb = pd.read_csv(DATA / "worldbank.csv.gz")
    wb["iso3"] = wb["iso3"].fillna(wb["country"].map({"High income": "HIC"}))  # the API leaves aggregates' codes blank
    gdp = wb[wb["indicator"] == "NY.GDP.MKTP.CD"].pivot_table(index="year", columns="iso3", values="value")
    pop = wb[wb["indicator"] == "SP.POP.TOTL"].pivot_table(index="year", columns="iso3", values="value")
    for y in (2015, 2025):
        print(f"{y}: US GDP {gdp.loc[y, 'USA'] / gdp.loc[y, 'WLD']:.1%} of world GDP, {gdp.loc[y, 'USA'] / gdp.loc[y, 'HIC']:.1%} "
              f"of high-income countries' GDP, {gdp.loc[y, 'USA'] / gdp.loc[y, 'OED']:.1%} of OECD GDP; "
              f"population {pop.loc[y, 'USA'] / pop.loc[y, 'WLD']:.1%} of world, {pop.loc[y, 'USA'] / pop.loc[y, 'HIC']:.1%} of high-income")
    us_w = gdp.loc[2025, "USA"] / gdp.loc[2025, "WLD"]
    us_h = gdp.loc[2025, "USA"] / gdp.loc[2025, "HIC"]
    print(f"Revenue per dollar of GDP, US against the rest, if the companies' non-US revenue came from the rest of the world "
          f"(2025): {(pooled / us_w) / ((1 - pooled) / (1 - us_w)):.1f} times; against the rest of the high-income world only: "
          f"{(pooled / us_h) / ((1 - pooled) / (1 - us_h)):.1f} times")
    figure["us_gdp_share"] = {int(y): round(float(gdp.loc[y, "USA"] / gdp.loc[y, "WLD"]), 4) for y in range(2015, 2026)}

    # ------------------------------------------------------------------------------------ 4
    section("4. From revenue to margins: how a revenue share becomes a profit share")
    b = last.dropna(subset=["cost_of_sales_usd"])
    s = b["us_rev_usd"].sum() / b["geo_total_usd"].sum()
    c = b["cost_of_sales_usd"].sum() / b["revenue_usd"].sum()
    rd = b["rd_usd"].sum() / b["revenue_usd"].sum()
    ni = b["net_income_usd"].sum() / b["revenue_usd"].sum()
    print(f"Pooled FY2025 ({len(b)} companies with cost of sales): US share of revenue {s:.1%}; cost of sales {c:.1%} of revenue; "
          f"R&D {rd:.1%}; net income {ni:.1%}")
    print("Share of revenue above the cost of sales that comes from the US, if a US unit sells at k times the price "
          "of a unit elsewhere and the cost of sales follows units (RAND 2024: k = 4.22 gross, 3.08 net of US rebates):")
    rents = {}
    for k in (1.0, 2.0, 3.08, 4.22):
        v_us = (s / k) / (s / k + (1 - s))
        rent = (s - c * v_us) / (1 - c)
        rents[k] = rent
        print(f"  k = {k:4.2f}: US share of units {v_us:5.1%}; US share of revenue above cost of sales {rent:5.1%}")
    figure["rents"] = {str(k): round(v, 4) for k, v in rents.items()}

    # ------------------------------------------------------------------------------------ 5
    section("5. Would the revenue earned outside the United States cover the companies' R&D?")
    non_us = b["geo_total_usd"].sum() - b["us_rev_usd"].sum()
    rd_sum = b["rd_usd"].sum()
    print(f"Pooled FY2025: R&D $ {rd_sum / 1e9:,.0f} bn; revenue outside the US $ {non_us / 1e9:,.0f} bn "
          f"({non_us / rd_sum:.2f} times R&D); US revenue $ {b['us_rev_usd'].sum() / 1e9:,.0f} bn ({b['us_rev_usd'].sum() / rd_sum:.2f} times R&D)")
    for k in (1.0, 3.08):
        v_us = (s / k) / (s / k + (1 - s))
        margin_non_us = non_us - c * b["revenue_usd"].sum() * (1 - v_us)
        print(f"  revenue outside the US above its cost of sales (k = {k:.2f}): $ {margin_non_us / 1e9:,.0f} bn, "
              f"{margin_non_us / rd_sum:.2f} times R&D")
    bs = b.dropna(subset=["sga_usd"])
    if len(bs):
        s2 = bs["us_rev_usd"].sum() / bs["geo_total_usd"].sum()
        c2 = bs["cost_of_sales_usd"].sum() / bs["revenue_usd"].sum()
        g2 = bs["sga_usd"].sum() / bs["revenue_usd"].sum()
        nu2 = bs["geo_total_usd"].sum() - bs["us_rev_usd"].sum()
        rd2 = bs["rd_usd"].sum()
        print(f"  {len(bs)} companies with selling, general and administrative costs: SG&A {g2:.1%} of revenue, "
              f"cost of sales {c2:.1%}, R&D {rd2 / bs['revenue_usd'].sum():.1%}, US share {s2:.1%}")
        for k in (1.0, 3.08):
            v_us = (s2 / k) / (s2 / k + (1 - s2))
            left = nu2 - c2 * bs["revenue_usd"].sum() * (1 - v_us) - g2 * nu2
            print(f"  revenue outside the US above its cost of sales and its share of SG&A (SG&A in proportion to revenue, "
                  f"k = {k:.2f}): $ {left / 1e9:,.0f} bn, {left / rd2:.2f} times R&D")
            left_us = bs["us_rev_usd"].sum() - c2 * bs["revenue_usd"].sum() * v_us - g2 * bs["us_rev_usd"].sum()
            print(f"    the same for US revenue: $ {left_us / 1e9:,.0f} bn, {left_us / rd2:.2f} times R&D; "
                  f"US share of what is left after cost of sales and SG&A {left_us / (left_us + left):.1%}")
    cover = (b["geo_total_usd"] - b["us_rev_usd"]) / b["rd_usd"]
    print("  by company, revenue outside the US divided by R&D: " + ", ".join(
        f"{r.company} {x:.1f}" for r, x in sorted(zip(b.itertuples(), cover), key=lambda t: t[1])))

    # ------------------------------------------------------------------------------------ 6
    section("6. Europe country by country: market, R&D located in the country, public health research")
    t = text("efpia_2026")
    market, m_total = efpia_table(t, "PHARMACEUTICAL MARKET VALUE")
    rdc, rd_total = efpia_table(t, "PHARMACEUTICAL INDUSTRY RESEARCH AND DEVELOPMENT IN EUROPE")
    prod, p_total = efpia_table(t, "PHARMACEUTICAL PRODUCTION\n")
    must(t, "1,475,815", "1,667,671", "54.7%", "23.7 %", "6.5%", "3.5%", "74.1%", "15.6 %", "104 new molecules",
         "46 originated", "28 and 16")
    world_eur, world_usd = 1_475_815, 1_667_671
    print(f"EFPIA Key Data 2026: world prescription market 2025 EUR {world_eur:,} million ($ {world_usd:,} million) at "
          f"ex-manufacturer prices; North America 54.7%, Europe 23.7%, China 6.5%, Japan 3.5% (IQVIA MIDAS)")
    print("  sales of new medicines launched 2020-2024: United States 74.1%, the five largest European markets 15.6%")
    print("  new molecules first launched in 2025: 104, of which 46 from companies headquartered in China, 28 in the US, 16 in Europe")
    print(f"  European tables, 2024 (EUR million): market at ex-factory prices {m_total:,.0f} ({len(market)} countries), "
          f"industry R&D located in Europe {rd_total:,.0f} ({len(rdc)}), production {p_total:,.0f}")
    egdp = pd.read_csv(DATA / "eurostat_gdp.csv.gz")
    egdp = egdp[egdp["time"] == 2024].set_index("geo")["value"]
    gdp_eur = {k: float(egdp[k]) for k in market if k in egdp.index}
    gdp_eur["UK"] = gdp.loc[2024, "GBR"] / fx.loc[2024, "USD"] / 1e6
    epop = pop.loc[2024]
    rows = []
    for k, v in market.items():
        g = gdp_eur.get(k)
        p3 = ISO2_TO_3.get(k)
        rows.append({"iso2": k, "market": v, "rd": rdc.get(k), "gdp": g,
                     "pop": float(epop[p3]) if p3 in epop.index else np.nan})
    eu_t = pd.DataFrame(rows).set_index("iso2")
    eu_t["market_gdp"] = eu_t["market"] / eu_t["gdp"]
    eu_t["rd_gdp"] = eu_t["rd"] / eu_t["gdp"]
    eu_t["rd_market"] = eu_t["rd"] / eu_t["market"]
    eu_t["market_pc"] = eu_t["market"] * 1e6 / eu_t["pop"]
    z = eu_t.dropna(subset=["market_gdp", "rd_gdp"])
    print(f"Countries with market, R&D and GDP: {len(z)}")
    print("  country | market % GDP | market per person (EUR) | industry R&D % GDP | R&D per euro of market")
    for k, r in z.sort_values("rd_market", ascending=False).iterrows():
        print(f"    {k}  {100 * r.market_gdp:5.2f}  {r.market_pc:6.0f}  {100 * r.rd_gdp:5.2f}  {r.rd_market:5.2f}")
    pt = z.loc["PT"]
    print(f"  Portugal: market {100 * pt.market_gdp:.2f}% of GDP (rank {int((z['market_gdp'] > pt.market_gdp).sum()) + 1} of {len(z)}), "
          f"EUR {pt.market_pc:.0f} per person; industry R&D {100 * pt.rd_gdp:.3f}% of GDP "
          f"(rank {int((z['rd_gdp'] > pt.rd_gdp).sum()) + 1} of {len(z)}); R&D per euro of market {pt.rd_market:.3f}")
    eur_mkt = z["market"].sum() / z["gdp"].sum()
    eur_rd = z["rd"].sum() / z["gdp"].sum()
    print(f"  all {len(z)} together: market {100 * eur_mkt:.2f}% of GDP, industry R&D {100 * eur_rd:.2f}% of GDP, "
          f"R&D per euro of market {z['rd'].sum() / z['market'].sum():.3f}")
    lo, hi = boot_corr(z["market_gdp"], z["rd_gdp"], rng)
    print(f"  across countries, market against industry R&D (both % of GDP): Spearman {spearman(z['market_gdp'], z['rd_gdp']):+.2f} "
          f"[{lo:+.2f}, {hi:+.2f}]")
    na_eur = 0.547 * world_eur
    na_gdp = (gdp.loc[2025, "USA"] + gdp.loc[2025, "CAN"]) / fx.loc[2025, "USD"] / 1e6
    print(f"  North America 2025: market EUR {na_eur:,.0f} million, {100 * na_eur / na_gdp:.2f}% of US and Canadian GDP; "
          f"per person EUR {na_eur * 1e6 / (pop.loc[2025, 'USA'] + pop.loc[2025, 'CAN']):,.0f}")
    pt_world = market["PT"] / world_eur
    print(f"  Portugal's market (2024) as a share of the world market (2025): {pt_world:.2%}")
    figure["europe"] = [{"iso2": k, "market_gdp": round(float(r.market_gdp), 5), "rd_gdp": round(float(r.rd_gdp), 6),
                         "rd_market": round(float(r.rd_market), 4)} for k, r in z.iterrows()]

    # OECD: business R&D in the pharmaceutical industry and public budgets for health R&D, % of GDP
    msti = pd.read_csv(DATA / "oecd_msti.csv.gz")
    bd = msti[(msti["MEASURE"] == "B_DRUG") & (msti["UNIT_MEASURE"] == "USD_PPP")].set_index(["REF_AREA", "TIME_PERIOD"])["OBS_VALUE"]
    g_ppp = msti[(msti["MEASURE"] == "GDP") & (msti["UNIT_MEASURE"] == "USD_PPP")].set_index(["REF_AREA", "TIME_PERIOD"])["OBS_VALUE"]
    gb = pd.read_csv(DATA / "oecd_gbard.csv.gz")
    gbh = gb[(gb["SEO"] == "NABS07") & (gb["UNIT_MEASURE"] == "USD_PPP") & (gb["PRICE_BASE"] == "V")].set_index(["REF_AREA", "TIME_PERIOD"])["OBS_VALUE"]
    gbt = gb[(gb["SEO"] == "_T") & (gb["UNIT_MEASURE"] == "USD_PPP") & (gb["PRICE_BASE"] == "V")].set_index(["REF_AREA", "TIME_PERIOD"])["OBS_VALUE"]
    comp = []
    for area in sorted(set(bd.index.get_level_values(0)) | set(gbh.index.get_level_values(0))):
        yrs_b = [y for (a, y) in bd.index if a == area and (a, y) in g_ppp.index]
        yrs_g = [y for (a, y) in gbh.index if a == area and (a, y) in g_ppp.index]
        rec = {"area": area}
        if yrs_b:
            y = max(yrs_b)
            rec.update(berd_year=y, berd_pharma_gdp=float(bd[(area, y)] / g_ppp[(area, y)]), berd_pharma_usd=float(bd[(area, y)]))
        if yrs_g:
            y = max(yrs_g)
            rec.update(gbard_year=y, gbard_health_gdp=float(gbh[(area, y)] / g_ppp[(area, y)]), gbard_health_usd=float(gbh[(area, y)]),
                       health_share_gbard=float(gbh[(area, y)] / gbt[(area, y)]) if (area, y) in gbt.index else np.nan)
        comp.append(rec)
    comp = pd.DataFrame(comp).set_index("area")
    show = ["USA", "CHE", "BEL", "DNK", "GBR", "DEU", "SWE", "FRA", "NLD", "ITA", "ESP", "PRT", "JPN", "KOR", "CAN", "POL", "GRC"]
    print("OECD (MSTI and GBARD by objective), latest year with GDP:")
    print("  area | business R&D in pharmaceuticals % GDP (year, $ PPP million) | public budget for health R&D % GDP (year) | health share of public R&D budget")
    for a in show:
        if a not in comp.index:
            continue
        r = comp.loc[a]
        btxt = f"{100 * r.berd_pharma_gdp:.3f} ({int(r.berd_year)}, {r.berd_pharma_usd:,.0f})" if pd.notna(r.get("berd_pharma_gdp")) else "n/a"
        gtxt = f"{100 * r.gbard_health_gdp:.3f} ({int(r.gbard_year)})" if pd.notna(r.get("gbard_health_gdp")) else "n/a"
        htxt = f"{r.health_share_gbard:.0%}" if pd.notna(r.get("health_share_gbard")) else "n/a"
        print(f"    {NAMES3.get(a, a):15s} {btxt:28s} {gtxt:16s} {htxt}")
    gf = pd.read_csv(DATA / "oecd_gerd_field.csv.gz", low_memory=False)
    med = gf[(gf["FORD"] == "FORD3") & (gf["PRICE_BASE"] == "V") & gf["SECT_PERF"].isin(["HES", "GOV"])]
    med = med.groupby(["REF_AREA", "TIME_PERIOD", "SECT_PERF"])["OBS_VALUE"].sum().unstack("SECT_PERF")
    acad = []
    for area in sorted(set(med.index.get_level_values(0))):
        yrs = [y for (a, y) in med.index if a == area and (a, y) in g_ppp.index and med.loc[(a, y)].notna().all()]
        if yrs:
            y = max(yrs)
            acad.append({"area": area, "year": y, "gdp_share": float(med.loc[(area, y)].sum() / g_ppp[(area, y)])})
    acad = pd.DataFrame(acad).set_index("area")
    print("R&D in medical and health sciences performed by universities and government institutes (OECD GERD by field), "
          "% of GDP, latest year with both sectors; the US does not report this breakdown:")
    print("  " + ", ".join(f"{NAMES3.get(a, a)} {100 * r.gdp_share:.3f} ({int(r.year)})" for a, r in acad.sort_values("gdp_share", ascending=False).iterrows()
                           if a in NAMES3))
    if "PRT" in acad.index:
        print(f"  Portugal {100 * acad.loc['PRT', 'gdp_share']:.3f}% of GDP; median of the {len(acad)} reporting countries "
              f"{100 * acad['gdp_share'].median():.3f}%")
    figure["academic"] = {a: round(float(r.gdp_share), 6) for a, r in acad.iterrows()}
    if "USA" in comp.index and "PRT" in comp.index:
        print(f"  US business R&D in pharmaceuticals ({int(comp.loc['USA', 'berd_year'])}) $ {comp.loc['USA', 'berd_pharma_usd'] / 1e3:,.1f} bn PPP; "
              f"public budget for health R&D ({int(comp.loc['USA', 'gbard_year'])}) $ {comp.loc['USA', 'gbard_health_usd'] / 1e3:,.1f} bn PPP")
        ratio = comp.loc["USA", "gbard_health_gdp"] / comp.loc["PRT", "gbard_health_gdp"]
        print(f"  public budget for health R&D, % of GDP: United States {ratio:.1f} times Portugal")
    eu_area = [a for a in ("AUT", "BEL", "DNK", "FIN", "FRA", "DEU", "GRC", "IRL", "ITA", "NLD", "PRT", "ESP", "SWE", "POL", "CZE", "HUN", "SVN")
               if a in comp.index and pd.notna(comp.loc[a, "gbard_health_gdp"])]
    w = comp.loc[eu_area]
    gdp_w = np.array([g_ppp[(a, int(w.loc[a, "gbard_year"]))] for a in eu_area])
    print(f"  {len(eu_area)} EU countries together: public budget for health R&D {100 * (w['gbard_health_usd'].sum() / gdp_w.sum()):.3f}% of GDP")
    figure["public"] = [{"area": a, "name": NAMES3.get(a, a), "gbard_health_gdp": round(float(comp.loc[a, "gbard_health_gdp"]), 6)}
                        for a in show if a in comp.index and pd.notna(comp.loc[a, "gbard_health_gdp"])]
    figure["public_eu"] = round(float(w["gbard_health_usd"].sum() / gdp_w.sum()), 6)

    # ------------------------------------------------------------------------------------ 7
    section("7. What the literature's elasticities imply for the number of new medicines")
    new_mol = 104
    print(f"Baseline: {new_mol} new molecules first launched worldwide in 2025 (EFPIA). Elasticity of the number of new "
          f"medicines to the size of their potential market: 0.25 (Dubois, de Mouzon, Scott Morton and Seabright 2015) and "
          f"4 (Acemoglu and Linn 2004, new non-generic drugs). Changes in market size are taken as shares of the world market.")
    print("  Portugal's share of the market for new medicines is taken to be its share of the whole market (an upper bound, "
          "since new medicines reach Portugal later); the five largest European markets' share is EFPIA's 15.6%.")
    scen = [
        ("Portugal spends 25% more on all medicines", pt_world * 0.25),
        ("Portugal doubles what it spends on all medicines", pt_world * 1.0),
        ("the five largest European markets pay 25% more for new medicines", 0.156 * 0.25),
        ("the five largest European markets pay US net prices for new medicines (x 3.08)", 0.156 * 2.08),
    ]
    for lab, dm in scen:
        a, b_ = 0.25 * dm, 4.0 * dm
        print(f"  {lab}: market for new medicines +{dm:.2%}; new medicines +{a:.2%} ({a * new_mol:.2f} a year) "
              f"to +{b_:.1%} ({b_ * new_mol:.1f} a year)")
    print(f"  one more new medicine a year needs a world market larger by {1 / (0.25 * new_mol):.1%} at elasticity 0.25 and "
          f"{1 / (4 * new_mol):.2%} at elasticity 4; Portugal's whole market is {pt_world:.2%} of the world's")

    # Part 2's index, for reference
    p2 = (PART2 / "output.txt").read_text(encoding="utf-8")
    m = re.search(r"single-source\s+n\s+(\d+)\s+Laspeyres\s+([\d.]+)\s+Paasche\s+([\d.]+)\s+Fisher\s+([\d.]+)", p2)
    print(f"Part 2: US against French prices, single-source molecules (n={m.group(1)}), Fisher index {m.group(4)} before rebates")

    (HERE / "figure_data.json").write_text(json.dumps(figure, indent=1, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
