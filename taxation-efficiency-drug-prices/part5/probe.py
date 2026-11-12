"""Survey the public sources on parallel export and export restrictions, for Part 5 (round 2: Infarmed's
public lists of shortages and discontinuations, the current export-control lists, their history in the
Wayback Machine, and EMA's parallel distribution register).

Run by .github/workflows/taxation-efficiency-drug-prices-data.yml, which commits data/probe_report.txt
and, under data/raw/, the text of the documents (PDFs as layout text, gzipped) and small spreadsheets.
Nothing is analysed here: the report is used to write fetch.py.

    python probe.py --out data
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import re
import subprocess
import tempfile
import time
from html import unescape
from pathlib import Path
from urllib.parse import urljoin

import requests

UA = {"User-Agent": "Mozilla/5.0 (research script; contact dfr@esmad.ipp.pt)",
      "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.8"}

PAGES = {
    "siats_ruturas": "https://extranet.infarmed.pt/siats/Publico/Rupturas.aspx",
    "siats_cessacoes": "https://extranet.infarmed.pt/siats/Publico/CessaoComerc.aspx",
    "dr_regulamento_2025": "https://diariodarepublica.pt/dr/detalhe/deliberacao/234-2025-907742765",
}

INF = "https://www.infarmed.pt"
SAVE = {
    "pt_disponibilidade_2025": INF + "/documents/15786/10901776/Relat%C3%B3rio2025/a2d40b5d-60ea-4297-6bc8-fa146f1a5c70",
    "pt_lnp_oct_2026": INF + "/documents/15786/9423561/LNP+-+outubro+2026+%28II%29/d9dcc03e-a5d6-d90e-0787-91a72518f57a",
    "pt_proibicoes_oct_2026": INF + "/documents/15786/11927579/Lista+de+proibi%C3%A7%C3%B5es+-+17+de+outubro+de+2026/2aa0a5bb-d463-65e5-9a35-91b6d72e0e7f",
    "pt_suspensa_nov_2026": INF + "/documents/15786/9423565/Lista+de+exporta%C3%A7%C3%A3o+temporariamente+suspensa+-+Novembro+2026/90058a23-ee1d-91e0-df78-c406a5a2a60e",
    "pt_delib_110_2026": INF + "/documents/15786/9423565/Delibera%C3%A7%C3%A3o+110+-+exporta%C3%A7%C3%A3o+temporariamente+suspensa+-+novembro+2026/77984131-b2d5-5e3a-3f84-d8067b8c3469",
}

FILE_LINK = re.compile(r"\.(csv|xls|xlsx|zip|ods|xml|json|txt|pdf)(\?|$|#)|/documents/|download|export|lista|relat",
                       re.IGNORECASE)
HREF = re.compile(r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
TAGS = re.compile(r"<[^>]+>")


def get(url: str, tries: int = 3, timeout: int = 120, session=None, **kw):
    last = None
    for i in range(tries):
        try:
            return (session or requests).get(url, headers=UA, timeout=timeout, **kw)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * (i + 1))
    raise last


def pdf_text(data: bytes) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "in.pdf"
        p.write_bytes(data)
        subprocess.run(["pdftotext", "-layout", str(p), str(p.with_suffix(".txt"))], check=True)
        return p.with_suffix(".txt").read_text(encoding="utf-8", errors="replace")


def survey_page(name: str, url: str, out: list[str], out_dir: Path) -> str:
    out.append(f"===== {name}\nURL {url}")
    try:
        r = get(url)
    except Exception as e:  # noqa: BLE001
        out.append(f"ERROR {type(e).__name__}: {e}")
        return ""
    ctype = r.headers.get("content-type", "")
    out.append(f"status {r.status_code}  final {r.url}  type {ctype}  bytes {len(r.content)}")
    if "html" not in ctype:
        out.append(r.content[:400].decode("utf-8", "replace"))
        return ""
    html = r.text
    raw = out_dir / "raw"
    raw.mkdir(exist_ok=True)
    (raw / f"{name}.html.gz").write_bytes(gzip.compress(html.encode("utf-8"), mtime=0))
    title = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    out.append("title " + (unescape(title.group(1)).strip()[:200] if title else ""))
    seen = set()
    n = 0
    for href, text in HREF.findall(html):
        href = unescape(href).strip()
        if not FILE_LINK.search(href) or href.startswith("javascript"):
            continue
        full = urljoin(r.url, href)
        if full in seen:
            continue
        seen.add(full)
        label = re.sub(r"\s+", " ", unescape(TAGS.sub(" ", text))).strip()[:160]
        out.append(f"  LINK {label} -> {full}")
        n += 1
        if n >= 300:
            out.append("  (more links not listed)")
            break
    if n == 0:
        out.append("  no file links; first 2000 characters of text:")
        out.append(re.sub(r"\s+", " ", unescape(TAGS.sub(" ", html)))[:2000])
    return html


def save(name: str, url: str, out_dir: Path, out: list[str], session=None) -> None:
    out.append(f"===== save {name}\nURL {url}")
    try:
        r = get(url, timeout=300, session=session)
    except Exception as e:  # noqa: BLE001
        out.append(f"ERROR {type(e).__name__}: {e}")
        return
    ctype = r.headers.get("content-type", "")
    out.append(f"status {r.status_code}  final {r.url}  type {ctype}  bytes {len(r.content)}")
    if r.status_code != 200:
        return
    raw = out_dir / "raw"
    raw.mkdir(exist_ok=True)
    if r.content[:4] == b"%PDF":
        text = pdf_text(r.content)
        (raw / f"{name}.txt.gz").write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
        out.append(f"saved raw/{name}.txt.gz ({text.count(chr(12)) + 1} pages)")
        out.append("  first lines: " + re.sub(r"\s+", " ", text[:600]))
    elif r.content[:2] == b"PK" and len(r.content) < 20_000_000:
        (raw / f"{name}.xlsx").write_bytes(r.content)
        out.append(f"saved raw/{name}.xlsx")
    elif r.content[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        (raw / f"{name}.xls").write_bytes(r.content)
        out.append(f"saved raw/{name}.xls")
    else:
        out.append(r.content[:600].decode("utf-8", "replace"))


def iris(out: list[str], out_dir: Path) -> None:
    """EMA's register of parallel distribution notices is a Power Pages portal. Ask its grid data service
    for records, trying the header combinations the portal's own script sends."""
    out.append("===== iris register (entity grid)")
    s = requests.Session()
    page = "https://iris.ema.europa.eu/registerpd/"
    r = get(page, session=s)
    html = r.text
    layouts = re.findall(r"data-view-layouts=\"([^\"]+)\"", html)
    get_url = re.findall(r"data-get-url=\"([^\"]+)\"", html)
    lay = json.loads(base64.b64decode(unescape(layouts[0])))
    conf = lay[0]["Base64SecureConfiguration"]
    url = urljoin(page, unescape(get_url[0]))
    tok_html = get("https://iris.ema.europa.eu/_layout/tokenhtml", session=s).text
    m = re.search(r"value=\"([^\"]+)\"", tok_html)
    token = m.group(1) if m else ""
    out.append(f"  token {'found' if token else 'missing'}; cookies {list(s.cookies.keys())}")
    # Member-state filter options (checkbox values) with their labels.
    opts = re.findall(r'<input type="checkbox" name="(\d+)" value="(\{[^}]+\})"\s*/>\s*</?[^>]*>?\s*([^<]*)', html)
    labels = re.findall(r'value="(\{[0-9a-f-]+\})"[^>]*/>[^<]*<[^>]*>([^<]+)<', html)
    out.append(f"  checkbox options {len(opts)}; labelled {len(labels)}; sample {labels[:5]}")
    i = html.find("Member State(s) of Origin")
    out.append("  origin filter html: " + re.sub(r"\s+", " ", html[i:i + 1500]))
    base = {"base64SecureConfiguration": conf, "sortExpression": "ema_dateofnotification DESC", "search": None,
            "page": 1, "pageSize": 26, "pagingCookie": "", "filter": None, "metaFilter": None, "nlSearchFilter": None,
            "timezoneOffset": 0, "customParameters": [], "entityName": None, "entityId": None}
    variants = {
        "xhr": {"X-Requested-With": "XMLHttpRequest", "Referer": page, "Origin": "https://iris.ema.europa.eu"},
        "xhr_noorigin": {"X-Requested-With": "XMLHttpRequest", "Referer": page},
        "plain": {},
    }
    raw = out_dir / "raw"
    raw.mkdir(exist_ok=True)
    for vname, extra in variants.items():
        for size in (26, 500):
            body = dict(base, pageSize=size)
            hdr = {**UA, "Content-Type": "application/json; charset=utf-8", "Accept": "application/json, text/javascript, */*; q=0.01",
                   "__RequestVerificationToken": token, **extra}
            try:
                rr = s.post(url, data=json.dumps(body), headers=hdr, timeout=120, allow_redirects=False)
            except Exception as e:  # noqa: BLE001
                out.append(f"  {vname}/{size}: ERROR {e}")
                continue
            ct = rr.headers.get("content-type", "")
            out.append(f"  {vname}/{size}: {rr.status_code} {ct} bytes {len(rr.content)} loc {rr.headers.get('location')}")
            if "json" in ct:
                js = rr.json()
                out.append("    keys " + str(list(js.keys())))
                out.append("    ItemCount " + str(js.get("ItemCount")) + " PageCount " + str(js.get("PageCount"))
                           + " MoreRecords " + str(js.get("MoreRecords")))
                recs = js.get("Records") or []
                if recs:
                    out.append("    first record " + json.dumps(recs[0])[:2500])
                (raw / f"iris_{vname}_{size}.json.gz").write_bytes(gzip.compress(rr.content, mtime=0))
                return
            out.append("    " + re.sub(r"\s+", " ", rr.text[:300]))


def wayback(out: list[str], out_dir: Path) -> None:
    """Snapshots of Infarmed's availability page in the Wayback Machine, to recover the links to past
    monthly lists of medicines whose export was suspended and past prior-notification lists."""
    out.append("===== wayback snapshots of the availability page")
    cdx = ("https://web.archive.org/cdx/search/cdx?url=infarmed.pt/web/infarmed/gestao-da-disponibilidade-do-medicamento"
           "&output=json&from=2019&filter=statuscode:200&collapse=timestamp:6")
    try:
        r = get(cdx, timeout=180)
        rows = r.json()
    except Exception as e:  # noqa: BLE001
        out.append(f"  CDX ERROR {e}")
        return
    out.append(f"  snapshots (one per month): {len(rows) - 1}")
    links: dict[str, str] = {}
    for row in rows[1:]:
        ts = row[1]
        snap = f"https://web.archive.org/web/{ts}id_/https://www.infarmed.pt/web/infarmed/gestao-da-disponibilidade-do-medicamento"
        try:
            h = get(snap, timeout=120).text
        except Exception as e:  # noqa: BLE001
            out.append(f"  {ts} ERROR {e}")
            continue
        found = []
        for href, text in HREF.findall(h):
            label = re.sub(r"\s+", " ", unescape(TAGS.sub(" ", text))).strip()
            href = unescape(href)
            if "/documents/15786/" in href and re.search(r"suspens|LNP|notifica|proibi|Lista", href + label, re.I):
                full = re.sub(r"^https?://web\.archive\.org/web/\d+[a-z_]*/", "", urljoin("https://www.infarmed.pt/", href))
                if full not in links:
                    links[full] = f"{ts} {label}"
                    found.append(label)
        out.append(f"  {ts}: {len(found)} new links {found[:6]}")
        time.sleep(1.0)
    out.append(f"  distinct list documents: {len(links)}")
    for u, v in links.items():
        out.append(f"  DOC {v} -> {u}")
    (out_dir / "raw").mkdir(exist_ok=True)
    (out_dir / "raw" / "wayback_links.json").write_text(json.dumps(links, indent=1, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    for name, url in PAGES.items():
        survey_page(name, url, out, out_dir)
    for name, url in SAVE.items():
        save(name, url, out_dir, out)
    try:
        iris(out, out_dir)
    except Exception as e:  # noqa: BLE001
        out.append(f"iris ERROR {type(e).__name__}: {e}")
    wayback(out, out_dir)
    (out_dir / "probe_report.txt").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
