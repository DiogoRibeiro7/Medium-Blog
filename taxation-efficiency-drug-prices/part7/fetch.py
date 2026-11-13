"""Download and reduce the public files used in Part 7 (a ledger of what Portugal's public buyer of
medicines saves, and where the saving comes from).

Run by .github/workflows/taxation-efficiency-drug-prices-data.yml, which commits the reduced files in
data/ together with data/fetch_log.txt. Each source is fetched independently; a failure is logged and
the others still run. Part 7 also reads the committed results of Parts 2 to 6 (their figure_data.json
and data/ folders), which are not downloaded again.

Sources
  sns        SNS transparency portal: monthly SNS charges for medicines in hospitals, and in community
             pharmacies (SNS charges, patients' charges, retail value), by region, 2011 onwards.
  infarmed   Infarmed press note of 15 November 2025 on medicines spending in 2025, kept as text (the
             estimated annual total of industry returns to the SNS).
  apifarma   APIFARMA, Boletim de Conjuntura, December 2025, kept as text (value of the pharmacy market
             at ex-factory prices and the share of it with generic competition), and the 2025 agreement
             between the Portuguese State and the industry.
  oecd       OECD Health Statistics, System of Health Accounts (DSD_SHA@DF_SHA): current health spending,
             spending on pharmaceuticals and on governance and administration, by financing scheme, all
             reporting countries, 2015 onwards.

    python fetch.py --out data
"""

from __future__ import annotations

import argparse
import gzip
import io
import re
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (research script; contact dfr@esmad.ipp.pt)"}
URLS = {
    "sns": "https://transparencia.sns.gov.pt/api/explore/v2.1/catalog/datasets/{id}/exports/csv?delimiter=%3B",
    "infarmed_note_2025": "https://www.infarmed.pt/documents/15786/10634545/Nota+Informativa+-+Refor%C3%A7o+do+acesso+dos+cidad%C3%A3os+e+impacto+da+inova%C3%A7%C3%A3o%C2%A0/f7f43298-6e88-1b69-b292-8ba9edb51f5f",
    "apifarma_boletim_2025_12": "https://apifarma.pt/wp-content/uploads/2026/01/Boletim_Conjuntura-Dez-2025.pdf",
    "apifarma_agreement_2025": "https://apifarma.pt/wp-content/uploads/2025/07/Acordo-Governo-APIFARMA-2025.pdf",
    "oecd_structure": "https://sdmx.oecd.org/public/rest/dataflow/OECD.ELS.HD/DSD_SHA@DF_SHA/1.1?references=datastructure",
    "oecd_data": "https://sdmx.oecd.org/public/rest/data/OECD.ELS.HD,DSD_SHA@DF_SHA,1.1/{key}?startPeriod=2015&format=csvfilewithlabels",
}
SNS_DATASETS = ("despesa-com-medicamentos-nos-hospitais-do-sns", "despesa-com-medicamentos-no-ambulatorio-sns")

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


def fetch_sns(out: Path) -> None:
    for ds in SNS_DATASETS:
        df = pd.read_csv(io.StringIO(get(URLS["sns"].format(id=ds)).text), sep=";")
        df = df.sort_values(list(df.columns[:2])).reset_index(drop=True)
        save(df, out, "sns_" + ds.replace("-", "_"))


def fetch_documents(out: Path) -> None:
    checks = {
        "infarmed_note_2025": r"anual\s+estimado\s+de\s+500-600\s+milh",
        "apifarma_boletim_2025_12": r"M\.\s*Concorrencial\s+48,2%",
        "apifarma_agreement_2025": r"Despesa\s+L.quida",
    }
    for name, pattern in checks.items():
        text = pdf_text(get(URLS[name]).content)
        flat = re.sub(r"\s+", " ", text)
        if not re.search(pattern, flat):
            raise RuntimeError(f"{name}: expected text not found ({pattern})")
        save_text(text, out, name)


def fetch_oecd(out: Path) -> None:
    js = get(URLS["oecd_structure"], headers={**UA, "Accept": "application/vnd.sdmx.structure+json;version=1.0"}).json()
    dims = js["data"]["dataStructures"][0]["dataStructureComponents"]["dimensionList"]["dimensions"]
    order = [d["id"] for d in sorted(dims, key=lambda d: d.get("position", 0))]
    filters = {"FREQ": "A", "MEASURE": "EXP_HEALTH", "FINANCING_SCHEME": "_T+HF1+HF3",
               "FUNCTION": "_T+HC51+HC511+HC7", "MODE_PROVISION": "_T", "PROVIDER": "_T", "PRICE_BASE": "V"}
    key = ".".join(filters.get(d, "") for d in order)
    log(f"[oecd] SHA dimensions {order}; key {key}")
    r = get(URLS["oecd_data"].format(key=key), timeout=600, tries=3, wait=30)
    df = pd.read_csv(io.StringIO(r.text), low_memory=False)
    keep = [c for c in df.columns if c in ("REF_AREA", "Reference area", "UNIT_MEASURE", "Unit of measure",
                                           "FINANCING_SCHEME", "FUNCTION", "TIME_PERIOD", "OBS_VALUE", "UNIT_MULT")]
    df = df[keep].sort_values(["REF_AREA", "FUNCTION", "FINANCING_SCHEME", "UNIT_MEASURE", "TIME_PERIOD"])
    log(f"[oecd] units: {sorted(df['Unit of measure'].dropna().unique().tolist())}")
    save(df.reset_index(drop=True), out, "oecd_sha")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, fn in (("sns", fetch_sns), ("documents", fetch_documents), ("oecd", fetch_oecd)):
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
