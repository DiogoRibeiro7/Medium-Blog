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


REPORT_PATH: Path | None = None


def say(msg: str) -> None:
    print(msg, flush=True)
    REPORT.append(msg)
    if REPORT_PATH is not None:
        REPORT_PATH.write_text("\n".join(REPORT) + "\n", encoding="utf-8")


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
    say("== SEC EDGAR: older filings and submission pages")
    for name in ("Pfizer", "Novartis", "GSK", "Takeda", "AbbVie", "Merck"):
        cik = COMPANIES[name]
        try:
            sub = sec(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
            pages = [f["name"] for f in sub["filings"].get("files", [])]
            rec = sub["filings"]["recent"]
            rows = [dict(zip(rec.keys(), vals)) for vals in zip(*rec.values())]
            for pg in pages:
                more = sec(f"https://data.sec.gov/submissions/{pg}").json()
                rows += [dict(zip(more.keys(), vals)) for vals in zip(*more.values())]
            ann = sorted([r for r in rows if r["form"] in ("10-K", "20-F")], key=lambda r: r["reportDate"])
            say(f"{name}: pages {pages}; annual filings {len(ann)}: " + ", ".join(f"{r['form']} {r['reportDate']}" for r in ann[-14:]))
            for f in [r for r in ann if r["reportDate"][:4] in ("2015", "2018")]:
                acc = f["accessionNumber"].replace("-", "")
                idx = sec(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/index.json").json()
                files = [i["name"] for i in idx["directory"]["item"]]
                inst = [x for x in files if x.endswith("_htm.xml")] or \
                       [x for x in files if x.endswith(".xml") and not re.search(r"(_cal|_def|_lab|_pre|FilingSummary|MetaLinks)", x)]
                say(f"  {f['form']} {f['reportDate']}: instance candidates {inst[:4]} of {len(files)} files")
                if not inst:
                    continue
                xml = sec(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{inst[0]}").content
                ctx, facts = parse_instance(xml)
                end = f["reportDate"]
                axes = {}
                for fa in facts:
                    if not (REV.match(fa["name"]) or RND.match(fa["name"])) or len(fa["value"]) > 30:
                        continue
                    c = ctx.get(fa["ctx"], {})
                    if c.get("end") != end or not c.get("start") or c["start"][:4] not in (end[:4], str(int(end[:4]) - 1)):
                        continue
                    dims = tuple(d for d, _ in c["dims"])
                    if len(dims) > 1 or (dims and not re.search("Geograph|Area", dims[0])):
                        continue
                    axes.setdefault((fa["name"], dims, fa["unit"]), []).append(("/".join(m for _, m in c["dims"]) or "total", fa["value"]))
                for (nm, dims, unit), vals in sorted(axes.items()):
                    say(f"    {nm} {dims} {unit}: " + "; ".join(f"{m}={v}" for m, v in vals[:12]))
        except Exception as exc:  # noqa: BLE001
            say(f"{name}: failed {exc}")


def probe_oecd(out: Path) -> None:
    say("== OECD SDMX structures")
    flows = ["OECD.ELS.HD,HEALTH_PHMC@DF_PHMC_SALES,1.1", "OECD.ELS.HD,HEALTH_PHMC@DF_KEY_INDIC,1.1",
             "OECD.ELS.HD,DSD_SHA@DF_SHA,1.1", "OECD.STI.STP,DSD_RDS_BERD@DF_BERD_INDU,1.0",
             "OECD.STI.STP,DSD_ANBERD@DF_ANBERDi4,1.0", "OECD.STI.STP,DSD_RDS_GOV@DF_GBARD_NABS07,1.0",
             "OECD.STI.STP,DSD_RDS_BERD@DF_BERD_MA_SOF,1.0"]
    for fl in flows:
        agency, fid, ver = fl.split(",")
        try:
            r = get(f"https://sdmx.oecd.org/public/rest/dataflow/{agency}/{fid}/{ver}?references=all",
                    headers={**UA, "Accept": "application/vnd.sdmx.structure+json;version=1.0"}, timeout=120)
            js = r.json()["data"]
            dsd = js["dataStructures"][0]
            dims = dsd["dataStructureComponents"]["dimensionList"]["dimensions"]
            cls = {c["id"]: c for c in js.get("codelists", [])}
            say(f"{fl}: dimensions in order:")
            for d in dims:
                ref = d.get("localRepresentation", {}).get("enumeration", "")
                clid = ref.split("(")[0].split(".")[-1].split(":")[-1] if ref else ""
                codes = []
                for cid, cl in cls.items():
                    if cid == clid or (clid and clid in cid):
                        codes = [(c["id"], c.get("name") or c.get("names", {}).get("en", "")) for c in cl.get("codes", [])]
                        break
                show = codes if len(codes) <= 40 else [c for c in codes if re.search(r"PRT|USA|DEU|_T|C21|PHARM|HC51|HEALTH|NABS07|EUR|USD|PPP|GDP|CAP|SALES|TOTAL|^R|^S", c[0] + str(c[1]), re.I)][:60]
                say(f"   {d['id']} ({len(codes)} codes): " + "; ".join(f"{a}={b}" for a, b in show))
        except Exception as exc:  # noqa: BLE001
            say(f"{fl}: failed {exc}")


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


def probe_ecb(out: Path) -> None:
    say("== ECB exchange rates")
    try:
        r = get("https://data-api.ecb.europa.eu/service/data/EXR/A.USD+GBP+DKK+JPY+CHF.EUR.SP00.A?format=csvdata&startPeriod=2015")
        lines = r.text.splitlines()
        say(f"rows {len(lines)}; header {lines[0][:300]}; first {lines[1][:300]}")
    except Exception as exc:  # noqa: BLE001
        say(f"ECB failed {exc}")


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
    global REPORT_PATH
    out = Path(ap.parse_args().out) / "probe"
    out.mkdir(parents=True, exist_ok=True)
    REPORT_PATH = out.parent / "probe_report.txt"
    for fn in (probe_oecd, probe_eurostat, probe_worldbank, probe_ecb, probe_reports, probe_sec):
        try:
            fn(out)
        except Exception:  # noqa: BLE001
            say(traceback.format_exc())
    (out.parent / "probe_report.txt").write_text("\n".join(REPORT) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
