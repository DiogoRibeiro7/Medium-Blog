"""Download and reduce the public price and volume files used in Part 2.

Run by .github/workflows/taxation-efficiency-drug-prices-data.yml, which commits the reduced files in
data/ together with data/fetch_log.txt. Each source is fetched independently; a failure is logged and
the others still run.

Sources
  US
    cms_partd_2024      Medicare Part D Spending by Drug, data years 2020-2024 (CMS, released June 2026).
                        Gross spending, dosage units and claims by brand name and generic name.
    nadac_2024          National Average Drug Acquisition Cost, weekly file for 2024 (CMS / Medicaid).
                        Average invoice price paid by retail pharmacies per unit, by NDC.
    fda_ndc             FDA National Drug Code directory (product and package files): dosage form,
                        strength and ingredient for each NDC.
    cms_mfp             Medicare Drug Price Negotiation Program: selected drugs and maximum fair prices,
                        per 30-day supply and per NDC unit.
  France
    bdpm                Base de donnees publique des medicaments: specialities, presentations (with the
                        number of units in each box), compositions and generic groups.
    open_medic_2024     Open Medic 2024 (Assurance Maladie): boxes dispensed in community pharmacies and
                        the reimbursement base, by CIP13 presentation.
  Exchange rate
    ecb_eurusd          ECB reference rate, US dollars per euro, annual averages.

    python fetch.py --out data
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import time
import traceback
import zipfile
from pathlib import Path

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (research script; contact dfr@esmad.ipp.pt)"}

URLS = {
    "cms_partd_2024": "https://data.cms.gov/sites/default/files/2026-06/98218f98-166c-4723-8438-c344a4ef96a6/"
                      "DSD_PTD_RY26_P04_V10_DY24_BGM.csv",
    "nadac_2024": "https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost-12-25-2024.csv",
    "fda_ndc": "https://www.accessdata.fda.gov/cder/ndctext.zip",
    "cms_mfp": "https://www.cms.gov/files/zip/selected-drug-list-negotiated-prices-also-known-maximum-fair-prices-statutezip.zip",
    "bdpm_base": "https://base-donnees-publique.medicaments.gouv.fr/download/file/",
    "open_medic_dataset": "https://www.data.gouv.fr/api/1/datasets/566e964188ee3875beaf0bf5/",
    "ecb_eurusd": "https://data-api.ecb.europa.eu/service/data/EXR/A.USD.EUR.SP00.A?format=csvdata&startPeriod=2019",
}

LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def get(url: str, tries: int = 4, timeout: int = 300) -> bytes:
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def describe(name: str, df: pd.DataFrame) -> None:
    log(f"[{name}] shape {df.shape}")
    log(f"[{name}] columns {list(df.columns)}")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60):
        log(df.head(5).to_string())


def save(df: pd.DataFrame, out: Path, name: str) -> None:
    path = out / f"{name}.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        df.to_csv(fh, index=False)
    log(f"[{name}] wrote {path.name}: {len(df)} rows, {path.stat().st_size / 1e6:.2f} MB")


def num(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.astype(str).str.replace(" ", "", regex=False).str.replace(" ", "", regex=False)
        s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


# ----------------------------------------------------------------------------------------------- US
def fetch_partd(out: Path) -> None:
    raw = get(URLS["cms_partd_2024"])
    df = pd.read_csv(io.BytesIO(raw), dtype=str, encoding="latin-1")
    describe("cms_partd_2024", df)
    keep = [c for c in df.columns if c in ("Brnd_Name", "Gnrc_Name", "Tot_Mftr", "Mftr_Name")
            or c.endswith("_2023") or c.endswith("_2024")]
    df = df[keep]
    save(df, out, "cms_partd_2024")


def fetch_nadac(out: Path) -> None:
    raw = get(URLS["nadac_2024"], timeout=900)
    log(f"[nadac_2024] downloaded {len(raw) / 1e6:.1f} MB")
    df = pd.read_csv(io.BytesIO(raw), dtype=str)
    describe("nadac_2024", df)
    cols = {c.lower().strip(): c for c in df.columns}
    c_ndc = cols.get("ndc")
    c_desc = cols.get("ndc description")
    c_price = cols.get("nadac per unit") or cols.get("nadac_per_unit")
    c_unit = cols.get("pricing unit")
    c_cls = cols.get("classification for rate setting")
    c_otc = cols.get("otc")
    c_asof = cols.get("as of date")
    c_eff = cols.get("effective date")
    df["_price"] = num(df[c_price])
    df["_asof"] = pd.to_datetime(df[c_asof], errors="coerce")
    df = df[df["_asof"].dt.year == 2024]
    log(f"[nadac_2024] rows with an as-of date in 2024: {len(df)}; distinct as-of dates {df['_asof'].nunique()}")
    g = df.groupby(c_ndc)
    red = pd.DataFrame({
        "ndc": g.size().index,
        "description": g[c_desc].last().values,
        "pricing_unit": g[c_unit].last().values,
        "classification": g[c_cls].last().values,
        "otc": g[c_otc].last().values,
        "nadac_mean_2024": g["_price"].mean().values,
        "nadac_last_2024": g["_price"].last().values,
        "weeks": g.size().values,
        "last_effective_date": g[c_eff].last().values,
    })
    save(red, out, "nadac_2024")


def fetch_fda(out: Path) -> None:
    raw = get(URLS["fda_ndc"])
    zf = zipfile.ZipFile(io.BytesIO(raw))
    log(f"[fda_ndc] members {zf.namelist()}")
    def read(member):
        with zf.open(member) as fh:
            return pd.read_csv(fh, sep="\t", dtype=str, encoding="latin-1", on_bad_lines="skip")
    prod = read([m for m in zf.namelist() if m.lower().startswith("product")][0])
    pack = read([m for m in zf.namelist() if m.lower().startswith("package")][0])
    describe("fda_product", prod)
    describe("fda_package", pack)
    pcols = [c for c in ("PRODUCTID", "PRODUCTNDC", "PRODUCTTYPENAME", "PROPRIETARYNAME", "NONPROPRIETARYNAME",
                         "DOSAGEFORMNAME", "ROUTENAME", "SUBSTANCENAME", "ACTIVE_NUMERATOR_STRENGTH",
                         "ACTIVE_INGRED_UNIT", "MARKETINGCATEGORYNAME", "APPLICATIONNUMBER") if c in prod.columns]
    kcols = [c for c in ("PRODUCTID", "PRODUCTNDC", "NDCPACKAGECODE", "PACKAGEDESCRIPTION") if c in pack.columns]
    m = pack[kcols].merge(prod[pcols], on=[c for c in ("PRODUCTID", "PRODUCTNDC") if c in kcols and c in pcols],
                          how="left")

    def ndc11(code: str) -> str | None:
        try:
            a, b, c = str(code).split("-")
        except ValueError:
            return None
        return a.zfill(5) + b.zfill(4) + c.zfill(2)

    m["ndc11"] = m["NDCPACKAGECODE"].map(ndc11)
    m = m.drop(columns=[c for c in ("PRODUCTID",) if c in m.columns])
    save(m, out, "fda_ndc")


def fetch_mfp(out: Path) -> None:
    raw = get(URLS["cms_mfp"])
    zf = zipfile.ZipFile(io.BytesIO(raw))
    log(f"[cms_mfp] members {zf.namelist()}")
    for member in zf.namelist():
        if not member.lower().endswith((".csv", ".xlsx", ".xls")):
            continue
        data = zf.read(member)
        if member.lower().endswith(".csv"):
            for enc in ("utf-8-sig", "latin-1"):
                try:
                    df = pd.read_csv(io.BytesIO(data), dtype=str, encoding=enc)
                    break
                except Exception:  # noqa: BLE001
                    continue
        else:
            df = pd.read_excel(io.BytesIO(data), dtype=str)
        describe(f"cms_mfp:{member}", df)
        stem = Path(member).stem.lower().replace(" ", "_")[:60]
        save(df, out, f"cms_mfp_{stem}")


# ------------------------------------------------------------------------------------------- France
BDPM_FILES = {
    "CIS_bdpm.txt": ["cis", "denomination", "forme", "voies", "statut_amm", "type_procedure", "etat_commercialisation",
                     "date_amm", "statut_bdm", "autorisation_europeenne", "titulaires", "surveillance"],
    "CIS_CIP_bdpm.txt": ["cis", "cip7", "libelle_presentation", "statut_admin", "etat_commercialisation",
                         "date_declaration", "cip13", "agrement_collectivites", "taux_remboursement", "prix_1",
                         "prix_2", "prix_3", "indications_remboursement"],
    "CIS_COMPO_bdpm.txt": ["cis", "element_pharmaceutique", "code_substance", "substance", "dosage", "reference_dosage",
                           "nature_composant", "numero_liaison"],
    "CIS_GENER_bdpm.txt": ["groupe_id", "groupe_libelle", "cis", "type_generique", "numero_tri"],
}


def fetch_bdpm(out: Path) -> None:
    for fname, names in BDPM_FILES.items():
        raw = get(URLS["bdpm_base"] + fname)
        text = None
        for enc in ("utf-8", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        rows = [line.split("\t") for line in text.splitlines() if line.strip()]
        width = max(len(r) for r in rows)
        cols = names + [f"extra_{i}" for i in range(max(0, width - len(names)))]
        rows = [r + [""] * (width - len(r)) for r in rows]
        df = pd.DataFrame(rows, columns=cols[:width])
        name = "bdpm_" + fname.split("_bdpm")[0].lower()
        describe(name, df)
        save(df, out, name)


def _follow_links(url: str, depth: int = 0) -> tuple[str, bytes]:
    """Download url; if it is an HTML listing, follow the first data-file link it contains."""
    import re
    from urllib.parse import urljoin

    raw = get(url, timeout=1800)
    head = raw[:4000].lower()
    is_html = b"<html" in head or b"<!doctype" in head or b"<a " in head
    if not is_html or depth > 2:
        return url, raw
    text = raw.decode("latin-1", errors="replace")
    log(f"[open_medic] {url} is an HTML page ({len(raw)} bytes); first 1500 chars:\n{text[:1500]}")
    links = re.findall(r'href=["\']([^"\']+)["\']', text, flags=re.I)
    links = [urljoin(url, h) for h in links]
    log(f"[open_medic] links found: {links[:60]}")
    data_links = [h for h in links if re.search(r"\.(zip|csv|gz|txt)(\?|$)", h, flags=re.I)
                  or "download" in h.lower() and h != url]
    data_links = [h for h in data_links if "2024" in h] or data_links
    if not data_links:
        raise RuntimeError(f"no data link on {url}")
    return _follow_links(data_links[0], depth + 1)


def _read_table(raw: bytes) -> pd.DataFrame:
    if raw[:2] == b"PK":
        zf = zipfile.ZipFile(io.BytesIO(raw))
        log(f"[open_medic] zip members {zf.namelist()}")
        member = [m for m in zf.namelist() if m.lower().endswith((".csv", ".txt"))][0]
        raw = zf.read(member)
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    head = raw[:3000].decode("latin-1")
    log(f"[open_medic] table head:\n{head[:800]}")
    sep = ";" if head.count(";") > head.count(",") else ","
    return pd.read_csv(io.BytesIO(raw), sep=sep, dtype=str, encoding="latin-1")


def fetch_open_medic(out: Path) -> None:
    meta = json.loads(get(URLS["open_medic_dataset"]))
    res = meta.get("resources", [])
    by_title = {(r.get("title") or "").strip(): r.get("url") for r in res}
    # Prefer the file already aggregated by CIP13 presentation; fall back on the complete base.
    order = ["2024 CIP13", "Open Medic 2024"]
    last_err = None
    for title in order:
        url = by_title.get(title)
        if not url:
            continue
        try:
            final_url, raw = _follow_links(url)
            log(f"[open_medic] '{title}' -> {final_url}, {len(raw) / 1e6:.1f} MB")
            df = _read_table(raw)
            describe(f"open_medic_2024[{title}]", df)
            cols = {c.upper().strip(): c for c in df.columns}
            need = [k for k in ("BOITES", "REM", "BSE") if k in cols]
            cip = cols.get("CIP13")
            if cip is None or "BOITES" not in need:
                raise RuntimeError(f"unexpected columns {list(df.columns)}")
            for k in need:
                df[cols[k]] = num(df[cols[k]])
            keys = [cip] + [cols[k] for k in ("L_CIP13", "TOP_GEN", "GEN_NUM", "ATC5", "L_ATC5") if k in cols]
            agg = df.groupby(keys, dropna=False)[[cols[k] for k in need]].sum().reset_index()
            agg.columns = [c.lower().strip() for c in agg.columns]
            save(agg, out, "open_medic_2024")
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            log(f"[open_medic] '{title}' failed: {exc}")
    raise RuntimeError(f"Open Medic 2024 not obtained: {last_err}")


# ------------------------------------------------------------------------------------- exchange rate
def fetch_ecb(out: Path) -> None:
    raw = get(URLS["ecb_eurusd"])
    df = pd.read_csv(io.BytesIO(raw), dtype=str)
    describe("ecb_eurusd", df)
    keep = [c for c in ("TIME_PERIOD", "OBS_VALUE") if c in df.columns]
    save(df[keep], out, "ecb_eurusd")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = {
        "cms_partd": fetch_partd, "nadac": fetch_nadac, "fda": fetch_fda, "mfp": fetch_mfp,
        "bdpm": fetch_bdpm, "open_medic": fetch_open_medic, "ecb": fetch_ecb,
    }
    failed = []
    for name, fn in jobs.items():
        if args.only and name not in args.only:
            continue
        log(f"===== {name}")
        try:
            fn(out)
        except Exception:  # noqa: BLE001
            failed.append(name)
            log(f"[{name}] FAILED\n{traceback.format_exc()}")
    log(f"failed: {failed}")
    (out / "fetch_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
