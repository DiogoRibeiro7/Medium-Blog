"""Download and reduce the public files used in Part 6 (who pays for pharmaceutical innovation).

Run by .github/workflows/taxation-efficiency-drug-prices-data.yml, which commits the reduced files in
data/ together with data/fetch_log.txt. Each source is fetched independently; a failure is logged and
the others still run.

Sources
  sec        SEC EDGAR: the annual reports (10-K, 20-F) of seventeen large research-based drug companies,
             fiscal years 2015-2025, read from the XBRL instance of each filing: total revenue, revenue in
             the United States and in the other geographic areas each company reports, research and
             development expense, acquired in-process R&D, cost of sales, selling, general and
             administrative costs, and net income.
  ecb        ECB reference exchange rates, annual averages (USD, GBP, DKK, JPY, CHF per euro).
  efpia      EFPIA, The Pharmaceutical Industry in Figures, Key Data 2026, kept as layout text (country
             tables of market value, R&D, production; world market shares; origin of new molecules).
  worldbank  World Bank: GDP in current US dollars and population, 2015-2025.
  oecd       OECD: government budget allocations for R&D by socio-economic objective (health and total),
             R&D in medical and health sciences by sector of performance (GERD by field), and business R&D
             performed in the pharmaceutical industry with GDP in PPP dollars (MSTI), all reporting countries.
  eurostat   Eurostat: GBARD for health (gba_nabsfin07) and business R&D in pharmaceuticals
             (rd_e_berdindr2, NACE C21), EU and EFTA countries.

    python fetch.py --out data
"""

from __future__ import annotations

import argparse
import gzip
import io
import itertools
import json
import re
import subprocess
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import pandas as pd
import requests

SEC_UA = {"User-Agent": "Diogo Ribeiro dfr@esmad.ipp.pt", "Accept-Encoding": "gzip, deflate"}
UA = {"User-Agent": "Mozilla/5.0 (research script; contact dfr@esmad.ipp.pt)"}

COMPANIES = {  # name: (CIK, headquarters)
    "Pfizer": (78003, "US"), "Merck & Co": (310158, "US"), "Johnson & Johnson": (200406, "US"),
    "AbbVie": (1551152, "US"), "Bristol-Myers Squibb": (14272, "US"), "Eli Lilly": (59478, "US"),
    "Amgen": (318154, "US"), "Gilead": (882095, "US"), "Regeneron": (872589, "US"), "Vertex": (875320, "US"),
    "Biogen": (875045, "US"),
    "Novartis": (1114448, "CH"), "AstraZeneca": (901832, "GB"), "GSK": (1131399, "GB"), "Sanofi": (1121404, "FR"),
    "Novo Nordisk": (353278, "DK"), "Takeda": (1395064, "JP"),
}
FIRST_FY, LAST_FY = 2015, 2025

REVENUE = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
           "SalesRevenueGoodsNet", "Revenue", "RevenueFromSaleOfGoods",
           "RevenueFromContractWithCustomerIncludingAssessedTax"]
RD = ["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"]
IPRD = ["ResearchAndDevelopmentInProcess", "ResearchAndDevelopmentAssetAcquiredOtherThanThroughBusinessCombinationWrittenOff",
        "AcquiredInProcessResearchAndDevelopment", "AssetAcquisitionInProcessResearchAndDevelopment"]
COST = ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold", "CostOfSales"]
PROFIT = ["NetIncomeLoss", "ProfitLoss", "ProfitLossAttributableToOwnersOfParent"]
SGA = ["SellingGeneralAndAdministrativeExpense"]
SELLING = ["SellingAndMarketingExpense", "SellingExpense", "DistributionCosts", "SalesAndDistributionCosts",
           "SellingGeneralAndMarketingExpense"]
ADMIN = ["GeneralAndAdministrativeExpense", "AdministrativeExpense"]
CONCEPTS = set(REVENUE + RD + IPRD + COST + PROFIT + SGA + SELLING + ADMIN)

URLS = {
    "ecb": "https://data-api.ecb.europa.eu/service/data/EXR/A.USD+GBP+DKK+JPY+CHF.EUR.SP00.A?format=csvdata&startPeriod=2015",
    "efpia_2026": "https://www.efpia.eu/media/owqczcqz/the-pharmaceutical-industry-in-figures-2026.pdf",
    "wb": "https://api.worldbank.org/v2/country/all/indicator/{ind}?format=json&per_page=20000&date=2015:2025",
    "oecd_flows": "https://sdmx.oecd.org/public/rest/dataflow/all",
    "oecd_structure": "https://sdmx.oecd.org/public/rest/dataflow/{agency}/{fid}/{ver}?references=datastructure",
    "oecd_data": "https://sdmx.oecd.org/public/rest/data/{agency},{fid},{ver}/{key}?startPeriod=2015&format=csvfilewithlabels",
    "eurostat": "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{ds}?format=JSON&lang=EN&sinceTimePeriod=2015{flt}",
}

LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def get(url: str, headers=None, tries: int = 4, timeout: int = 180, wait: float = 5.0) -> requests.Response:
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=headers or UA, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"status {r.status_code}")
            r.raise_for_status()
            return r
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(wait * (k + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def sec(url: str) -> requests.Response:
    time.sleep(0.12)  # SEC asks for at most ten requests a second
    return get(url, headers=SEC_UA, wait=2.0)


def save(df: pd.DataFrame, out: Path, name: str) -> None:
    path = out / f"{name}.csv.gz"
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(df.to_csv(index=False).encode("utf-8"))
    log(f"[{name}] wrote {path.name}: {len(df)} rows, columns {list(df.columns)}")


def save_text(text: str, out: Path, name: str) -> None:
    path = out / f"{name}.txt.gz"
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(text.encode("utf-8"))
    log(f"[{name}] wrote {path.name}: {len(text)} characters")


def pdf_text(data: bytes) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "in.pdf"
        pdf.write_bytes(data)
        subprocess.run(["pdftotext", "-layout", str(pdf), str(pdf.with_suffix(".txt"))], check=True)
        return pdf.with_suffix(".txt").read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------------------------- SEC
def local(tag: str) -> str:
    return tag.split("}")[-1]


def parse_instance(xml: bytes):
    """Contexts (period and explicit dimensions), units (currency) and the numeric facts of the
    concepts in CONCEPTS."""
    root = ET.fromstring(xml)
    ctx, units = {}, {}
    for c in root.iter():
        t = local(c.tag)
        if t == "context":
            start = end = None
            dims = []
            for e in c.iter():
                lt = local(e.tag)
                if lt == "startDate":
                    start = (e.text or "").strip()
                elif lt == "endDate":
                    end = (e.text or "").strip()
                elif lt == "explicitMember":
                    dims.append(((e.get("dimension") or "").split(":")[-1], (e.text or "").strip()))
                elif lt == "typedMember":
                    dims.append(((e.get("dimension") or "").split(":")[-1], "typed"))
            ctx[c.get("id")] = (start, end, tuple(dims))
        elif t == "unit":
            ms = [(m.text or "").strip() for m in c.iter() if local(m.tag) == "measure"]
            units[c.get("id")] = "/".join(ms)
    facts = []
    for e in root:
        name = local(e.tag)
        if name not in CONCEPTS or e.get("contextRef") is None or not (e.text or "").strip():
            continue
        try:
            value = float(e.text.strip())
        except ValueError:
            continue
        start, end, dims = ctx.get(e.get("contextRef"), (None, None, ()))
        facts.append({"concept": name, "start": start, "end": end, "dims": dims, "value": value,
                      "unit": units.get(e.get("unitRef"), e.get("unitRef") or "")})
    return facts


def fiscal_year(end: str) -> int:
    d = date.fromisoformat(end)
    return d.year if d.month >= 7 else d.year - 1


# the member's local name must be the United States and nothing else ("country:US", "IfrsCountryUsMember",
# "UnitedStatesMember"); a brand such as OCREVUS must not match
US_MEMBER = re.compile(r"^(country:US|[\w-]+:(Ifrs)?(Country)?(US|Us|USA|UnitedStates|UnitedStatesOfAmerica)(Member|Segment)?)$")
NOT_US = re.compile(r"(Non|Excl|Other|And|Outside|International|Rest)", re.I)


def annual_filings(cik: int) -> list[dict]:
    sub = sec(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    rec = sub["filings"]["recent"]
    rows = [dict(zip(rec.keys(), vals)) for vals in zip(*rec.values())]
    for pg in sub["filings"].get("files", []):
        more = sec(f"https://data.sec.gov/submissions/{pg['name']}").json()
        rows += [dict(zip(more.keys(), vals)) for vals in zip(*more.values())]
    out = {}
    for r in rows:
        if r["form"] not in ("10-K", "20-F") or not r.get("reportDate"):
            continue
        fy = fiscal_year(r["reportDate"])
        if FIRST_FY <= fy <= LAST_FY:
            # keep the first filing for each fiscal year (amendments use other form types)
            if fy not in out or r["filingDate"] < out[fy]["filingDate"]:
                out[fy] = r
    return [out[k] for k in sorted(out)]


def instance_name(files: list[str]) -> str | None:
    cand = [x for x in files if x.endswith("_htm.xml")]
    if not cand:
        cand = [x for x in files if x.endswith(".xml")
                and not re.search(r"(_cal|_def|_lab|_pre|FilingSummary|MetaLinks|^R\d+)", x)]
    return cand[0] if cand else None


def year_facts(facts: list[dict], end: str) -> list[dict]:
    """Facts for the fiscal year ending at `end`: duration contexts of 330 to 380 days."""
    out = []
    e = date.fromisoformat(end)
    for f in facts:
        if not f["start"] or not f["end"]:
            continue
        fe, fs = date.fromisoformat(f["end"]), date.fromisoformat(f["start"])
        if abs((fe - e).days) <= 7 and 330 <= (fe - fs).days <= 380:
            out.append(f)
    return out


def fetch_sec(out: Path) -> None:
    panel, geo, used = [], [], []
    for name, (cik, hq) in COMPANIES.items():
        try:
            filings = annual_filings(cik)
        except Exception as exc:  # noqa: BLE001
            log(f"[sec] {name}: submissions failed: {exc}")
            continue
        for f in filings:
            fy = fiscal_year(f["reportDate"])
            acc = f["accessionNumber"].replace("-", "")
            try:
                idx = sec(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/index.json").json()
                files = [i["name"] for i in idx["directory"]["item"]]
                inst = instance_name(files)
                if inst is None:
                    log(f"[sec] {name} FY{fy} {f['form']}: no XBRL instance ({len(files)} files)")
                    continue
                facts = year_facts(parse_instance(sec(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{inst}").content),
                                   f["reportDate"])
            except Exception as exc:  # noqa: BLE001
                log(f"[sec] {name} FY{fy}: failed: {exc}")
                continue
            used.append({"company": name, "fiscal_year": fy, "form": f["form"], "period_end": f["reportDate"],
                         "accession": f["accessionNumber"], "instance": inst})
            totals = {}
            for x in facts:
                if not x["dims"]:
                    totals.setdefault(x["concept"], x)
            rec = {"company": name, "hq": hq, "fiscal_year": fy, "period_end": f["reportDate"], "form": f["form"]}
            rev = [(c, totals[c]["value"], totals[c]["unit"]) for c in REVENUE if c in totals]
            if rev:
                c, v, u = max(rev, key=lambda t: t[1])
                rec.update(revenue=v, revenue_concept=c, currency=u.split(":")[-1].upper())
            rds = [(totals[c]["value"], c) for c in RD if c in totals]
            if rds:  # some filers tag a small sub-item as ResearchAndDevelopmentExpense; the largest is the line item
                rec["rd"], rec["rd_concept"] = max(rds)
            for group, key in ((IPRD, "iprd"), (COST, "cost_of_sales"), (PROFIT, "net_income")):
                for c in group:
                    if c in totals:
                        rec[key] = totals[c]["value"]
                        rec[key + "_concept"] = c
                        break
            # selling, general and administrative costs: one line, or selling plus administrative
            if any(c in totals for c in SGA):
                rec["sga"] = next(totals[c]["value"] for c in SGA if c in totals)
            else:
                sell = next((totals[c]["value"] for c in SELLING if c in totals), None)
                adm = next((totals[c]["value"] for c in ADMIN if c in totals), None)
                if sell is not None and adm is not None:
                    rec["sga"] = sell + adm
            if "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost" in totals:
                rec["rd_ex_iprd"] = totals["ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"]["value"]
            # geographic revenue: one dimension, on a geographic axis
            seen = set()
            for x in facts:
                if x["concept"] not in REVENUE or len(x["dims"]) != 1:
                    continue
                axis, member = x["dims"][0]
                if not (re.search(r"Geograph|Area|Region|Countr|Market", axis)
                        or (US_MEMBER.search(member) and not NOT_US.search(member.split(":")[-1]))):
                    continue
                key = (x["concept"], member, x["value"])
                if key in seen:
                    continue
                seen.add(key)
                geo.append({"company": name, "fiscal_year": fy, "concept": x["concept"], "axis": axis,
                            "member": member, "value": x["value"],
                            "concept_total": totals.get(x["concept"], {}).get("value"),
                            "currency": x["unit"].split(":")[-1].upper()})
            us = [g for g in geo if g["company"] == name and g["fiscal_year"] == fy
                  and US_MEMBER.search(g["member"]) and not NOT_US.search(g["member"].split(":")[-1])]
            if us:
                best = max(us, key=lambda g: (g["concept_total"] or 0, g["value"]))
                rec.update(us_revenue=best["value"], us_concept=best["concept"], us_member=best["member"],
                           us_concept_total=best["concept_total"])
            if not us:
                axes = sorted({(x["concept"], tuple(d for d, _ in x["dims"])) for x in facts
                               if x["concept"] in REVENUE and x["dims"]})
                log(f"[sec] {name} FY{fy}: no US revenue; revenue axes {axes[:12]}")
            panel.append(rec)
            log(f"[sec] {name} FY{fy} {f['form']}: revenue {rec.get('revenue')} {rec.get('currency')}, "
                f"US {rec.get('us_revenue')} ({rec.get('us_member')}), R&D {rec.get('rd')}")
    save(pd.DataFrame(panel), out, "sec_panel")
    save(pd.DataFrame(geo), out, "sec_geography")
    save(pd.DataFrame(used), out, "sec_filings")


# ---------------------------------------------------------------------------------------------- other
def fetch_ecb(out: Path) -> None:
    df = pd.read_csv(io.StringIO(get(URLS["ecb"]).text))
    save(df[["CURRENCY", "TIME_PERIOD", "OBS_VALUE"]].rename(columns=str.lower), out, "ecb_fx")


def fetch_efpia(out: Path) -> None:
    save_text(pdf_text(get(URLS["efpia_2026"]).content), out, "efpia_2026")


def fetch_worldbank(out: Path) -> None:
    rows = []
    for ind in ("NY.GDP.MKTP.CD", "SP.POP.TOTL", "PA.NUS.PPP"):
        js = get(URLS["wb"].format(ind=ind), timeout=300).json()
        for r in js[1]:
            rows.append({"indicator": ind, "iso3": r["countryiso3code"], "country": r["country"]["value"],
                         "year": int(r["date"]), "value": r["value"]})
    save(pd.DataFrame(rows).dropna(subset=["value"]), out, "worldbank")


def oecd_flow(fid_pattern: str) -> tuple[str, str, str] | None:
    js = get(URLS["oecd_flows"], headers={**UA, "Accept": "application/vnd.sdmx.structure+json;version=1.0"}).json()
    for f in js["data"]["dataflows"]:
        if re.fullmatch(fid_pattern, f["id"]):
            return f["agencyID"], f["id"], f["version"]
    return None


def oecd_query(agency: str, fid: str, ver: str, filters: dict[str, str]) -> pd.DataFrame:
    js = get(URLS["oecd_structure"].format(agency=agency, fid=fid, ver=ver),
             headers={**UA, "Accept": "application/vnd.sdmx.structure+json;version=1.0"}).json()
    dims = js["data"]["dataStructures"][0]["dataStructureComponents"]["dimensionList"]["dimensions"]
    order = [d["id"] for d in sorted(dims, key=lambda d: d.get("position", 0))]
    key = ".".join(filters.get(d, "") for d in order)
    log(f"[oecd] {fid}: dimensions {order}; key {key}")
    r = get(URLS["oecd_data"].format(agency=agency, fid=fid, ver=ver, key=key), timeout=600, tries=3, wait=30)
    return pd.read_csv(io.StringIO(r.text), low_memory=False)


def oecd_query_by_name(agency: str, fid: str, ver: str, filters: dict[str, str], by_name: dict[str, str]) -> pd.DataFrame:
    """Like oecd_query, but some dimensions are filtered by code names (regular expressions), looked up in
    the codelists of the data structure."""
    js = get(URLS["oecd_structure"].format(agency=agency, fid=fid, ver=ver).replace("datastructure", "all"),
             headers={**UA, "Accept": "application/vnd.sdmx.structure+json;version=1.0"}).json()["data"]
    dims = js["dataStructures"][0]["dataStructureComponents"]["dimensionList"]["dimensions"]
    lists = {cl["id"]: cl.get("codes", []) for cl in js.get("codelists", [])}
    order, key = [], []
    for d in sorted(dims, key=lambda d: d.get("position", 0)):
        order.append(d["id"])
        val = filters.get(d["id"], "")
        enum = d.get("localRepresentation", {}).get("enumeration", "")
        cl_id = re.sub(r"\(.*\)$", "", enum).split(".")[-1].split(":")[-1]
        for pat_dim, pat_name in by_name.items():
            if re.fullmatch(pat_dim, d["id"]):
                codes = [c["id"] for c in lists.get(cl_id, [])
                         if re.fullmatch(pat_name, c.get("name") or c.get("names", {}).get("en", ""))]
                val = "+".join(codes)
                log(f"[oecd] {fid}: {d['id']} codes {codes}")
        key.append(val)
    log(f"[oecd] {fid}: dimensions {order}; key {'.'.join(key)}")
    r = get(URLS["oecd_data"].format(agency=agency, fid=fid, ver=ver, key=".".join(key)), timeout=600, tries=3, wait=30)
    return pd.read_csv(io.StringIO(r.text), low_memory=False)


def fetch_oecd(out: Path) -> None:
    gbard = ("OECD.STI.STP", "DSD_RDS_GOV@DF_GBARD_NABS07", "1.0")
    df = oecd_query(*gbard, {"FREQ": "A", "SEO": "NABS07+_T"})
    keep = [c for c in df.columns if c in ("REF_AREA", "Reference area", "MEASURE", "SEO", "FUNDMODE", "TRANSCOORD",
                                           "UNIT_MEASURE", "Unit of measure", "PRICE_BASE", "TIME_PERIOD", "OBS_VALUE",
                                           "UNIT_MULT")]
    save(df[keep], out, "oecd_gbard")
    try:
        ford = oecd_flow(r"DSD_RDS_GERD@DF_GERD_FORD")
        df = oecd_query_by_name(*ford, {"FREQ": "A"},
                                {r"FORD|FIELD.*": r"(Medical and health sciences|Total.*)",
                                 r"SECT_PERF": r"(Total|Higher education.*|Government.*|Business enterprise.*)",
                                 r"UNIT_MEASURE": r"(US dollars, PPP converted|Percentage of GDP)"})
        keep = [c for c in df.columns if not c.startswith("Unnamed")]
        save(df[keep], out, "oecd_gerd_field")
    except Exception as exc:  # noqa: BLE001
        log(f"[oecd] GERD by field failed: {exc}")
    msti = oecd_flow(r"DSD_MSTI@DF_MSTI")
    log(f"[oecd] MSTI flow: {msti}")
    if msti:
        df = oecd_query(*msti, {"MEASURE": "B_DRUG+B+GDP"})
        keep = [c for c in df.columns if c in ("REF_AREA", "Reference area", "MEASURE", "Measure", "UNIT_MEASURE",
                                               "Unit of measure", "PRICE_BASE", "TIME_PERIOD", "OBS_VALUE", "UNIT_MULT")]
        save(df[keep], out, "oecd_msti")


def jsonstat(js: dict) -> pd.DataFrame:
    ids, size = js["id"], js["size"]
    cats = [list(js["dimension"][d]["category"]["index"].keys()) for d in ids]
    vals = js.get("value", {})
    rows = []
    for flat, combo in enumerate(itertools.product(*cats)):
        v = vals.get(str(flat)) if isinstance(vals, dict) else vals[flat]
        if v is not None:
            rows.append(dict(zip(ids, combo), value=v))
    assert len(cats) == len(size)
    return pd.DataFrame(rows)


def fetch_eurostat(out: Path) -> None:
    for name, ds, flt in (("eurostat_gbard_health", "gba_nabsfin07", "&nabs07=NABS07&nabs07=TOTAL"),
                          ("eurostat_berd_pharma", "rd_e_berdindr2", "&nace_r2=C21&nace_r2=TOTAL"),
                          ("eurostat_gdp", "nama_10_gdp", "&na_item=B1GQ&unit=CP_MEUR")):
        save(jsonstat(get(URLS["eurostat"].format(ds=ds, flt=flt)).json()), out, name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, fn in (("ecb", fetch_ecb), ("efpia", fetch_efpia), ("worldbank", fetch_worldbank),
                     ("eurostat", fetch_eurostat), ("oecd", fetch_oecd), ("sec", fetch_sec)):
        t0 = time.time()
        try:
            fn(out)
            log(f"== {name} done in {time.time() - t0:.0f} s")
        except Exception:  # noqa: BLE001
            failed.append(name)
            log(f"== {name} FAILED\n{traceback.format_exc()}")
    log(f"failed: {failed}")
    (out / "fetch_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
