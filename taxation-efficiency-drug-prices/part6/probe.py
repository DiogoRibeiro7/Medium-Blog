"""Survey the sources for Part 6 (who pays for pharmaceutical innovation). Temporary: removed once
fetch.py is written.

    python probe.py --out data
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

SEC_UA = {"User-Agent": "Diogo Ribeiro dfr@esmad.ipp.pt", "Accept-Encoding": "gzip, deflate"}
UA = {"User-Agent": "Mozilla/5.0 (research script; contact dfr@esmad.ipp.pt)"}
REPORT: list[str] = []

COMPANIES = {  # name: CIK
    "Pfizer": 78003, "Merck": 310158, "Johnson & Johnson": 200406, "AbbVie": 1551152,
    "Bristol-Myers Squibb": 14272, "Eli Lilly": 59478, "Amgen": 318154, "Gilead": 882095,
    "Regeneron": 872589, "Vertex": 875320, "Biogen": 875045,
    "Novartis": 1114448, "AstraZeneca": 901832, "GSK": 1131399, "Sanofi": 1121404,
    "Novo Nordisk": 353278, "Takeda": 1395064,
}


def say(msg: str) -> None:
    print(msg, flush=True)
    REPORT.append(msg)


def get(url: str, headers=None, tries: int = 3, timeout: int = 120) -> requests.Response:
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
            time.sleep(3 * (k + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def sec(url: str) -> requests.Response:
    time.sleep(0.15)
    return get(url, headers=SEC_UA)


def gz(path: Path, data: bytes) -> None:
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as g:
        g.write(data)


def pdf_text(data: bytes) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "in.pdf"
        pdf.write_bytes(data)
        subprocess.run(["pdftotext", "-layout", str(pdf), str(pdf.with_suffix(".txt"))], check=True)
        return pdf.with_suffix(".txt").read_text(encoding="utf-8", errors="replace")


def local(tag: str) -> str:
    return tag.split("}")[-1]


def parse_instance(xml: bytes) -> tuple[dict, list[dict]]:
    root = ET.fromstring(xml)
    ctx = {}
    for c in root.iter():
        if local(c.tag) != "context":
            continue
        cid = c.get("id")
        start = end = instant = None
        dims = []
        for e in c.iter():
            t = local(e.tag)
            if t == "startDate":
                start = e.text
            elif t == "endDate":
                end = e.text
            elif t == "instant":
                instant = e.text
            elif t == "explicitMember":
                dims.append((e.get("dimension"), (e.text or "").strip()))
            elif t == "typedMember":
                dims.append((e.get("dimension"), "typed"))
        ctx[cid] = {"start": start, "end": end, "instant": instant, "dims": tuple(dims)}
    facts = []
    for e in root:
        name = local(e.tag)
        cref = e.get("contextRef")
        if not cref or e.text is None:
            continue
        prefix = e.tag.split("}")[0].strip("{")
        facts.append({"ns": prefix, "name": name, "ctx": cref, "value": e.text.strip(),
                      "decimals": e.get("decimals"), "unit": e.get("unitRef")})
    return ctx, facts


REV = re.compile(r"^(Revenues?|RevenueFromContractWithCustomer\w*|SalesRevenueNet|SalesRevenueGoodsNet|"
                 r"Revenue|RevenueFromSaleOfGoods|NetSales\w*|.*ProductSales\w*|.*TotalRevenues?)$", re.I)
RND = re.compile(r"ResearchAndDevelopment\w*", re.I)


def probe_sec(out: Path) -> None:
    say("== SEC EDGAR")
    filings = {}
    for name, cik in COMPANIES.items():
        try:
            sub = sec(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
            rec = sub["filings"]["recent"]
            rows = [dict(zip(rec.keys(), vals)) for vals in zip(*rec.values())]
            ann = [r for r in rows if r["form"] in ("10-K", "20-F")]
            say(f"{name}: {sub.get('name')} annual filings in recent list {len(ann)}; "
                + ", ".join(f"{r['form']} {r['reportDate']} {r['accessionNumber']} {r['primaryDocument']}" for r in ann[:3]))
            filings[name] = (cik, ann)
        except Exception as exc:  # noqa: BLE001
            say(f"{name}: submissions failed {exc}")
    (out / "sec_filings.json").write_text(json.dumps({k: [cik, v[:12]] for k, (cik, v) in filings.items()}, indent=1))

    for name in ("Pfizer", "Novartis", "AstraZeneca", "Johnson & Johnson", "Novo Nordisk", "Sanofi"):
        if name not in filings or not filings[name][1]:
            continue
        cik, ann = filings[name]
        for f in ann[:1] + ann[8:9]:
            acc = f["accessionNumber"].replace("-", "")
            try:
                idx = sec(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/index.json").json()
                files = [i["name"] for i in idx["directory"]["item"]]
                inst = [x for x in files if x.endswith("_htm.xml")] or \
                       [x for x in files if x.endswith(".xml") and not re.search(r"(_cal|_def|_lab|_pre|FilingSummary|MetaLinks)", x)]
                say(f"  {name} {f['form']} {f['reportDate']}: {len(files)} files; instance candidates {inst[:4]}")
                if not inst:
                    continue
                xml = sec(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{inst[0]}").content
                gz(out / f"sec_{re.sub(r'[^a-z]', '', name.lower())}_{f['reportDate']}.xml.gz", xml)
                ctx, facts = parse_instance(xml)
                yr_end = f["reportDate"]
                axes = {}
                for fa in facts:
                    if not (REV.match(fa["name"]) or RND.match(fa["name"])):
                        continue
                    c = ctx.get(fa["ctx"], {})
                    if c.get("end") != yr_end or not c.get("start"):
                        continue
                    key = (fa["ns"].split("/")[-2] if "/" in fa["ns"] else fa["ns"], fa["name"],
                           tuple(d for d, _ in c["dims"]))
                    axes.setdefault(key, []).append(((tuple(m for _, m in c["dims"])), fa["value"], c["start"]))
                for (ns, nm, dims), vals in sorted(axes.items(), key=lambda kv: (kv[0][1], len(kv[0][2]))):
                    if len(dims) > 1:
                        continue
                    shown = "; ".join(f"{'/'.join(m) or 'total'}={v}" for m, v, _ in vals[:12])
                    say(f"    {ns}:{nm} dims={dims}: {shown}")
            except Exception as exc:  # noqa: BLE001
                say(f"  {name}: instance failed {exc}")

    for name in ("Pfizer", "Novartis"):
        cik = COMPANIES[name]
        try:
            cf = sec(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json").json()
            for tax, concepts in cf["facts"].items():
                hits = [c for c in concepts if REV.match(c) or RND.match(c)]
                say(f"  companyfacts {name} {tax}: {hits[:12]}")
        except Exception as exc:  # noqa: BLE001
            say(f"  companyfacts {name} failed {exc}")


def probe_oecd(out: Path) -> None:
    say("== OECD SDMX")
    try:
        r = get("https://sdmx.oecd.org/public/rest/dataflow/all", headers={**UA, "Accept": "application/vnd.sdmx.structure+json;version=1.0"})
        js = r.json()
        flows = js["data"]["dataflows"]
        keep = []
        for f in flows:
            nm = f.get("name") or (f.get("names") or {}).get("en", "")
            if re.search(r"BERD|ANBERD|GBARD|budget allocations|health expenditure|SHA|pharmaceutical|R&D", f["id"] + " " + str(nm), re.I):
                keep.append(f"{f['agencyID']},{f['id']},{f['version']} | {nm}")
        say(f"dataflows {len(flows)}; matching {len(keep)}")
        for k in keep[:80]:
            say("  " + k)
        (out / "oecd_flows.txt").write_text("\n".join(keep))
    except Exception as exc:  # noqa: BLE001
        say(f"dataflow list failed {exc}")
    tests = {
        "anberd": "https://sdmx.oecd.org/public/rest/data/OECD.STI.STP,DSD_ANBERD@DF_ANBERD,/all?startPeriod=2019&dimensionAtObservation=AllDimensions&format=csvfilewithlabels",
        "sha": "https://sdmx.oecd.org/public/rest/data/OECD.ELS.HD,DSD_SHA@DF_SHA,/all?startPeriod=2021&format=csvfilewithlabels",
    }
    for k, u in tests.items():
        try:
            r = get(u, timeout=300)
            txt = r.text
            gz(out / f"oecd_{k}_sample.csv.gz", txt[:3_000_000].encode())
            lines = txt.splitlines()
            say(f"{k}: {len(lines)} lines; header {lines[0][:600]}")
            for ln in lines[1:4]:
                say("   " + ln[:400])
        except Exception as exc:  # noqa: BLE001
            say(f"{k}: failed {exc}")


def probe_eurostat(out: Path) -> None:
    say("== Eurostat")
    base = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{}?format=JSON&lang=EN&sinceTimePeriod=2018"
    for ds, flt in (("rd_e_berdindr2", "&nace_r2=C21"), ("gba_nabsfin07", "&nabs07=NABS07"), ("hlth_sha11_hc", "&icha11_hc=HC51")):
        try:
            js = get(base.format(ds) + flt).json()
            dims = {d: list(js["dimension"][d]["category"]["index"].keys())[:15] for d in js["id"]}
            say(f"{ds}: size {js['size']}; dims {dims}; values {len(js.get('value', {}))}")
        except Exception as exc:  # noqa: BLE001
            say(f"{ds}: failed {exc}")


def probe_worldbank(out: Path) -> None:
    say("== World Bank")
    for ind in ("NY.GDP.MKTP.CD", "NY.GDP.MKTP.PP.CD", "SP.POP.TOTL"):
        try:
            js = get(f"https://api.worldbank.org/v2/country/all/indicator/{ind}?format=json&per_page=20000&date=2015:2025").json()
            say(f"{ind}: {js[0]}; rows {len(js[1])}; sample {js[1][0]}")
        except Exception as exc:  # noqa: BLE001
            say(f"{ind}: failed {exc}")


def probe_reports(out: Path) -> None:
    say("== Reports")
    urls = {
        "efpia_2025": "https://efpia.eu/media/uj0popel/the-pharmaceutical-industry-in-figures.pdf",
        "efpia_2026": "https://www.efpia.eu/media/owqczcqz/the-pharmaceutical-industry-in-figures-2026.pdf",
        "frech_2023": "https://www.nber.org/system/files/working_papers/w31272/w31272.pdf",
        "schaeffer_2018": "https://schaeffer.usc.edu/wp-content/uploads/2024/10/01.2018_Global20Burden20of20Medical20Innovation.pdf",
    }
    for k, u in urls.items():
        try:
            data = get(u, timeout=180).content
            txt = pdf_text(data)
            gz(out / f"{k}.txt.gz", txt.encode())
            say(f"{k}: {len(data)} bytes, {len(txt)} characters")
        except Exception as exc:  # noqa: BLE001
            say(f"{k}: failed {exc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    out = Path(ap.parse_args().out) / "probe"
    out.mkdir(parents=True, exist_ok=True)
    for fn in (probe_sec, probe_oecd, probe_eurostat, probe_worldbank, probe_reports):
        try:
            fn(out)
        except Exception:  # noqa: BLE001
            say(traceback.format_exc())
    (out.parent / "probe_report.txt").write_text("\n".join(REPORT) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
