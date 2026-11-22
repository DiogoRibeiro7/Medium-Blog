"""Build one bid-level table of BOEM Gulf lease sales from March 2014 to August 2026.

    python build_dataset.py      # writes data/bids.csv and results/build_check.txt

Two inputs, both downloaded and checked by fetch.py:

- ``data/case_study/data/bids.csv``: sales 249 (August 2017) to BBG3 (August 2026), parsed
  and checked against BOEM's totals in the case study of the game- and auction-theory series.
- ``data/boem/<n>/``: sales 231, 233, 235, 238, 241, 246, 247 and 248 (August 2013 to March
  2017), as BOEM publishes them: an archive of fixed-format ASCII files (``PREBID``, ``BID``,
  ``MAPS``) read with the record layout of the case study's parser, the sale statistics and
  the final bid recap. The recap gives the sale date and marks rejected high bids with an
  ``R``; the statistics give the totals that every sale is checked against.

One row per bid (a joint bid is one bid). The minimum bid is $25 per acre, or fraction of an
acre, in water shallower than 400 m and $100 per acre deeper; BOEM set the deep-water figure
in Sale 218 (December 2011), before every sale here.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber

HERE = Path(__file__).parent
DATA = HERE / "data"
OLD_SALES = ["231", "233", "235", "238", "241", "246", "247", "248"]

DEPTH_BAND = {
    "RS20": "0-200 m", "RS22": "0-200 m", "RS21": "200-400 m", "RS27": "200-400 m",
    "R21": "400-800 m", "R27": "400-800 m", "R22": "800-1,600 m", "R28": "800-1,600 m",
    "R23": "1,600 m+", "R29": "1,600 m+",
}
SHALLOW = {"0-200 m", "200-400 m"}
# Sale 233 has no statistics file on its page; its totals are those of BOEM's press release
# "Western Gulf of Mexico Lease Sale Yields $102.4 Million in High Bids on More Than 300,000
# Acres", 28 August 2013.
PRESS_RELEASE = {"233": {"tracts": 53, "bids": 61, "high": 102_351_712.0, "source": "28 August 2013"}}
MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def pdf_text(path: Path, pages: int | None = None) -> str:
    with pdfplumber.open(path) as pdf:
        ps = pdf.pages if pages is None else pdf.pages[:pages]
        return "\n".join((p.extract_text() or "") for p in ps)


def _member(z: zipfile.ZipFile, prefix: str) -> str:
    name = next(n for n in z.namelist() if n.upper().startswith(prefix))
    return z.read(name).decode("latin-1")


def read_ascii_sale(path: Path, sale: str) -> pd.DataFrame:
    """Tracts and bids of one sale; the same column positions as the case study's parser."""
    with zipfile.ZipFile(path) as z:
        prebid, bid, maps = (_member(z, p) for p in ("PREBID.", "BID.", "MAPS."))
    areas = {ln[54:61].strip(): ln[4:54].strip() for ln in maps.splitlines() if ln.strip()}
    tracts = pd.DataFrame([{
        "lease": "G" + ln[0:5].strip(), "protraction": ln[6:13].strip(), "block": ln[15:21].strip(),
        "acres": float(ln[24:31]) / 1000.0, "bid_type": ln[32:44].strip(),
        "term_years": int(ln[52:54]), "n_bids": int(ln[63:65]),
    } for ln in prebid.splitlines() if ln.strip()])
    tracts["area"] = tracts.protraction.map(areas)
    rows = pd.DataFrame([{
        "lease": ln[8:15].strip(), "amount": float(ln[16:26]), "company": ln[27:32].strip(),
        "share": float(ln[35:43]),
    } for ln in bid.splitlines() if ln.strip()])
    bids = (rows.groupby(["lease", "amount"], sort=False)
                .agg(companies=("company", ";".join), share_total=("share", "sum"),
                     n_partners=("company", "size")).reset_index())
    if not (bids.share_total.round(3) == 100).all():
        raise ValueError(f"sale {sale}: partner shares do not add to 100")
    bids["lead_company"] = bids.companies.str.split(";").str[0]
    out = bids.merge(tracts, on="lease", how="left", validate="many_to_one")
    out.insert(0, "sale", sale)
    out["source"] = f"boem/{sale}/data.zip"
    return out.drop(columns="share_total")


def sale_facts(sale: str) -> dict:
    """Date and totals from the statistics; date and rejected leases from the final recap."""
    d = DATA / "boem" / sale
    facts: dict = {"sale": sale}
    stats = d / "stats.html"
    if stats.exists() and stats.read_bytes()[:4] == b"%PDF":
        t = pdf_text(stats, pages=2)
        m = re.search(r"Sale Date:?\s*(\d{2})-([A-Z]{3})-(\d{4})", t)
        facts["date"] = f"{m[3]}-{MONTHS[m[2]]:02d}-{m[1]}"
        num = lambda pat: float(re.search(pat, t)[1].replace(",", ""))
        facts["stat_high"] = num(r"Sum of High Bids:?\s*\$([\d,]+\.\d\d)")
        facts["stat_all"] = num(r"Sum of All Bids:?\s*\$([\d,]+\.\d\d)")
        facts["stat_bids"] = int(num(r"Total Number of Bids:?\s*([\d,]+)"))
        facts["stat_tracts"] = int(num(r"Tracts/Blocks Receiving Bids:?\s*([\d,]+)"))
    recap = d / "final_bid_recap.pdf"
    if recap.exists():
        t = pdf_text(recap)
        if "date" not in facts:
            m = re.search(r"Sale Date\s*:\s*(\d{2})-([A-Z]{3})-(\d{4})", t)
            facts["date"] = f"{m[3]}-{MONTHS[m[2]]:02d}-{m[1]}"
        facts["rejected"] = set(re.findall(r"^R\s*(?:\d+ \d/\d%|\d+%)\s+RS?\d+\s*(G\d{5})", t, re.M))
        m = re.search(r"\(A\) Rejected Tracts\s+(\d+)", t)
        facts["recap_rejected_count"] = int(m[1]) if m else None
    return facts


def build() -> tuple[pd.DataFrame, list[str]]:
    log = []
    cs = pd.read_csv(DATA / "case_study" / "data" / "bids.csv", dtype={"sale": str, "lead_company": str,
                                                                      "companies": str})
    cs["source"] = "case_study/" + cs.source.astype(str)
    parts = []
    for sale in OLD_SALES:
        b = read_ascii_sale(DATA / "boem" / sale / "data.zip", sale)
        f = sale_facts(sale)
        b["sale_date"] = f["date"]
        if "rejected" in f:
            missing = f["rejected"] - set(b.lease)
            if missing or (f["recap_rejected_count"] is not None and len(f["rejected"]) != f["recap_rejected_count"]):
                raise ValueError(f"sale {sale}: rejected leases {sorted(f['rejected'])} do not match the recap")
            b["tract_rejected"] = b.lease.isin(f["rejected"])
            b["review_known"] = True
        else:
            b["tract_rejected"] = False
            b["review_known"] = False
        high = b.groupby("lease").amount.max()
        msg = (f"sale {sale} ({f['date']}): tracts {b.lease.nunique()}, bids {len(b)}, "
               f"high bids ${high.sum():,.2f}, all bids ${b.amount.sum():,.2f}")
        if "stat_high" in f:
            ok = (b.lease.nunique() == f["stat_tracts"] and len(b) == f["stat_bids"]
                  and abs(high.sum() - f["stat_high"]) < 0.5 and abs(b.amount.sum() - f["stat_all"]) < 0.5)
            msg += (f"; BOEM statistics: tracts {f['stat_tracts']}, bids {f['stat_bids']}, high ${f['stat_high']:,.2f}, "
                    f"all ${f['stat_all']:,.2f} -> {'match' if ok else 'MISMATCH'}")
            if not ok:
                raise ValueError(msg)
        elif sale in PRESS_RELEASE:
            pr = PRESS_RELEASE[sale]
            ok = (b.lease.nunique() == pr["tracts"] and len(b) == pr["bids"] and abs(high.sum() - pr["high"]) < 0.5)
            msg += (f"; BOEM press release ({pr['source']}): tracts {pr['tracts']}, bids {pr['bids']}, "
                    f"high ${pr['high']:,.2f} -> {'match' if ok else 'MISMATCH'}")
            if not ok:
                raise ValueError(msg)
        if "rejected" in f:
            msg += f"; rejected high bids in the final recap: {len(f['rejected'])}"
        else:
            msg += "; no final recap, review outcome unknown"
        log.append(msg)
        parts.append(b)
    old = pd.concat(parts, ignore_index=True)
    old["bid_code"] = old.bid_type.str.split().str[-1]
    old["royalty"] = old.bid_type.str.extract(r"^(\d+(?: \d/\d)?%)")[0]
    old["depth_band"] = old.bid_code.map(DEPTH_BAND)
    if old.depth_band.isna().any():
        raise ValueError(f"unknown bid codes {sorted(old[old.depth_band.isna()].bid_code.unique())}")
    old["shallow"] = old.depth_band.isin(SHALLOW)
    old["min_per_acre"] = old.shallow.map({True: 25.0, False: 100.0})
    old["minimum_bid"] = old.min_per_acre * np.ceil(old.acres)
    old["per_acre"] = old.amount / old.acres
    old = old.sort_values(["sale", "lease", "amount"], ascending=[True, True, False])
    old["rank"] = old.groupby(["sale", "lease"]).cumcount() + 1
    cols = list(cs.columns)
    df = pd.concat([old[cols], cs[cols]], ignore_index=True)
    df["sale_date"] = pd.to_datetime(df.sale_date)
    df["block_id"] = df.protraction.astype(str).str.strip() + " " + df.block.astype(str).str.strip()
    df = df.sort_values(["sale_date", "sale", "lease", "rank"]).reset_index(drop=True)
    below = df[df.amount < df.minimum_bid - 0.5]
    log.append(f"bids below the minimum bid: {len(below)}"
               + (" (" + "; ".join(f"sale {r.sale} {r.lease} ${r.amount:,.0f} vs ${r.minimum_bid:,.0f}, rejected {r.tract_rejected}"
                                   for r in below.itertuples()) + ")" if len(below) else ""))
    log.append(f"all sales: {df.sale.nunique()}, tracts {df.groupby(['sale', 'lease']).ngroups:,}, bids {len(df):,}")
    return df, log


if __name__ == "__main__":
    df, log = build()
    df.to_csv(DATA / "bids.csv", index=False)
    # the table is rebuilt at every run, so its checksum is what CI compares with the committed run
    log.append(f"sha256 of data/bids.csv: {hashlib.sha256((DATA / 'bids.csv').read_bytes()).hexdigest()}")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "build_check.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))
