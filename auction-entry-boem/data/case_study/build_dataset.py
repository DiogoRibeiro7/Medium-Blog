"""Build one bid-level table from the files BOEM publishes for each lease sale.

Inputs, all in data/raw/ (downloaded from boem.gov, see data/README.md):
  sale-249.zip ... sale-261.zip   fixed-format ASCII files of tracts, bids and companies
  sale-bbg1-final-bid-recap.pdf   bid recaps, for the three sales without the ASCII files
  sale-bbg2-final-bid-recap.pdf
  sale-bbg3-preliminary-bid-recap.pdf
and data/rejected_tracts.csv, the high bids BOEM rejected in sales 249 to 261,
transcribed from each sale's Final Bid Recap.

Output: data/bids.csv, one row per bid (joint bids are one bid), and a printed
check of every sale's totals.

    python build_dataset.py

Requires pandas and pdfplumber.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import pdfplumber

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"

SALE_DATES = {
    "249": "2017-08-16", "250": "2018-03-21", "251": "2018-08-15",
    "252": "2019-03-20", "253": "2019-08-21", "254": "2020-03-18",
    "256": "2020-11-18", "257": "2021-11-17", "259": "2023-03-29",
    "261": "2023-12-20", "BBG1": "2025-12-10", "BBG2": "2026-03-11",
    "BBG3": "2026-08-12",
}

# Water-depth band of each bid-type code, from the "Explanation of terms used"
# in each sale's bid recap. Sales 259 and 261 call the shallowest band RS20,
# the BBG sales renumbered every code.
DEPTH_BAND = {
    "RS20": "0-200 m", "RS22": "0-200 m", "RS21": "200-400 m", "RS27": "200-400 m",
    "R21": "400-800 m", "R27": "400-800 m", "R22": "800-1,600 m", "R28": "800-1,600 m",
    "R23": "1,600 m+", "R29": "1,600 m+",
}
SHALLOW = {"0-200 m", "200-400 m"}


# ---------------------------------------------------------------------------
# Sales 249 to 261: fixed-format files
# ---------------------------------------------------------------------------

def _member(z: zipfile.ZipFile, prefix: str) -> str:
    name = next(n for n in z.namelist() if n.upper().startswith(prefix))
    return z.read(name).decode("latin-1")


def read_ascii_sale(path: Path) -> pd.DataFrame:
    """Tracts and bids of one sale, following the record layouts in FORMATS2.txt."""
    sale = re.search(r"sale-(\d+)", path.name).group(1)
    with zipfile.ZipFile(path) as z:
        prebid, bid, maps = (_member(z, p) for p in ("PREBID.", "BID.", "MAPS."))

    areas = {ln[54:61].strip(): ln[4:54].strip() for ln in maps.splitlines() if ln.strip()}
    tracts = pd.DataFrame([{
        "lease": "G" + ln[0:5].strip(),
        "protraction": ln[6:13].strip(),
        "block": ln[15:21].strip(),
        "acres": float(ln[24:31]) / 1000.0,
        "bid_type": ln[32:44].strip(),
        "term_years": int(ln[52:54]),
        "n_bids": int(ln[63:65]),
    } for ln in prebid.splitlines() if ln.strip()])
    tracts["area"] = tracts.protraction.map(areas)

    rows = pd.DataFrame([{
        "lease": ln[8:15].strip(),
        "amount": float(ln[16:26]),
        "company": ln[27:32].strip(),
        "share": float(ln[35:43]),
    } for ln in bid.splitlines() if ln.strip()])
    # A joint bid appears once per partner with the same amount; the shares add to 100.
    bids = (rows.groupby(["lease", "amount"], sort=False)
                .agg(companies=("company", ";".join), share_total=("share", "sum"),
                     n_partners=("company", "size"))
                .reset_index())
    if not (bids.share_total.round(3) == 100).all():
        raise ValueError(f"sale {sale}: partner shares do not add to 100")
    bids["lead_company"] = bids.companies.str.split(";").str[0]
    out = bids.merge(tracts, on="lease", how="left", validate="many_to_one")
    out.insert(0, "sale", sale)
    out["source"] = path.name
    return out.drop(columns="share_total")


# ---------------------------------------------------------------------------
# Sales BBG1 to BBG3: bid recap PDFs
# ---------------------------------------------------------------------------

PER_ACRE = r"\$(?P<per_acre>[\d,]+\.\d\d)"
TOTAL = r"\$(?P<total>[\d,]+\.\d\d)"
FIRST_BID = re.compile(
    r"^(?P<rejected>R\s+)?(?P<royalty>\d+(?: \d/\d)?%)\s+(?P<code>RS?\d+)\s*(?P<lease>G\d{5})"
    r"(?:\s+[A-Z])?\s+(?P<block>\d+)\s+(?P<acres>[\d,]+\.\d+)A\s+(?P<company>\d{5})\s+"
    r"(?P<share>[\d.]+)\s+.*?\s+" + PER_ACRE + r"\s+" + TOTAL + r"\s+(?P<rank>\d+)$")
NEXT_BID = re.compile(
    r"^(?P<company>\d{5})\s+(?P<share>[\d.]+)\s+.*?\s+" + PER_ACRE + r"\s+" + TOTAL
    + r"\s+(?P<rank>\d+)$")
PARTNER = re.compile(r"^(?P<company>\d{5})\s+(?P<share>[\d.]+)\s+\D.*$")
AREA = re.compile(r"^Area:?\s+(?P<code>\S+)\s+Area Name:?\s+(?P<name>.+)$")


def read_recap_sale(path: Path) -> pd.DataFrame:
    sale = re.search(r"sale-(bbg\d)", path.name).group(1).upper()
    with pdfplumber.open(path) as pdf:
        lines = [ln.strip() for page in pdf.pages
                 for ln in (page.extract_text() or "").splitlines()]
    bids, tract, area, current = [], None, (None, None), None
    for ln in lines:
        if m := AREA.match(ln):
            area = (m["code"], m["name"].strip())
        elif m := FIRST_BID.match(ln):
            tract = {"lease": m["lease"], "protraction": area[0], "area": area[1],
                     "block": m["block"], "acres": float(m["acres"].replace(",", "")),
                     "bid_type": f"{m['royalty']} {m['code']}",
                     "rejected_flag": bool(m["rejected"])}
            current = {**tract, "amount": float(m["total"].replace(",", "")),
                       "rank": int(m["rank"]), "companies": [m["company"]],
                       "shares": [float(m["share"])]}
            bids.append(current)
        elif tract and (m := NEXT_BID.match(ln)):
            current = {**tract, "amount": float(m["total"].replace(",", "")),
                       "rank": int(m["rank"]), "companies": [m["company"]],
                       "shares": [float(m["share"])]}
            bids.append(current)
        elif current and (m := PARTNER.match(ln)):
            current["companies"].append(m["company"])
            current["shares"].append(float(m["share"]))
    out = pd.DataFrame(bids)
    if not (out.shares.map(sum).round(3) == 100).all():
        raise ValueError(f"sale {sale}: partner shares do not add to 100")
    out["n_partners"] = out.companies.map(len)
    out["lead_company"] = out.companies.str[0]
    out["companies"] = out.companies.map(";".join)
    out["n_bids"] = out.groupby("lease").lease.transform("size")
    out["term_years"] = pd.NA
    out.insert(0, "sale", sale)
    out["source"] = path.name
    return out.drop(columns=["shares", "rank"])


# ---------------------------------------------------------------------------
# Assemble and check
# ---------------------------------------------------------------------------

def build() -> pd.DataFrame:
    parts = [read_ascii_sale(p) for p in sorted(RAW.glob("sale-2*.zip"))]
    parts += [read_recap_sale(p) for p in sorted(RAW.glob("sale-bbg*.pdf"))]
    df = pd.concat(parts, ignore_index=True)

    df["sale_date"] = pd.to_datetime(df.sale.map(SALE_DATES))
    df["area"] = df.area.str.replace("DeSoto Canyon", "De Soto Canyon", regex=False)
    df["bid_code"] = df.bid_type.str.split().str[-1]
    df["royalty"] = df.bid_type.str.extract(r"^(\d+(?: \d/\d)?%)")[0]
    df["depth_band"] = df.bid_code.map(DEPTH_BAND)
    if df.depth_band.isna().any():
        raise ValueError(f"unknown bid codes: {sorted(df[df.depth_band.isna()].bid_code.unique())}")
    df["shallow"] = df.depth_band.isin(SHALLOW)
    # Minimum bid: $25 per acre (or fraction of an acre) in water shallower than
    # 400 m and $100 per acre deeper, in every sale here (see the article).
    df["min_per_acre"] = df.shallow.map({True: 25.0, False: 100.0})
    df["minimum_bid"] = df.min_per_acre * df.acres.apply(lambda a: float(-(-a // 1)))
    df["per_acre"] = df.amount / df.acres

    # Rank within tract, high bid first, and the outcome of BOEM's review.
    df = df.sort_values(["sale", "lease", "amount"], ascending=[True, True, False])
    df["rank"] = df.groupby(["sale", "lease"]).cumcount() + 1
    rejected = pd.read_csv(HERE / "data" / "rejected_tracts.csv", dtype=str)
    known = set(zip(rejected.sale, rejected.lease))
    from_pdf = set(zip(df.sale[df.rejected_flag.fillna(False).astype(bool)],
                       df.lease[df.rejected_flag.fillna(False).astype(bool)]))
    df["tract_rejected"] = [(s, l) in known or (s, l) in from_pdf
                            for s, l in zip(df.sale, df.lease)]
    df["review_known"] = df.sale != "BBG3"   # BBG3's final recap was not yet published

    # Every rejected tract listed in the recaps must exist with the same block number.
    check = rejected.merge(df[df["rank"] == 1][["sale", "lease", "block", "area"]],
                           on=["sale", "lease"], how="left", suffixes=("_recap", "_data"))
    bad = check[check.block_recap != check.block_data]
    if len(bad):
        raise ValueError(f"rejected tracts that do not match the data:\n{bad}")

    cols = ["sale", "sale_date", "lease", "area", "protraction", "block", "acres",
            "bid_type", "royalty", "bid_code", "depth_band", "shallow", "term_years",
            "n_bids", "rank", "amount", "per_acre", "min_per_acre", "minimum_bid",
            "lead_company", "companies", "n_partners", "tract_rejected", "review_known",
            "source"]
    return df[cols].reset_index(drop=True)


# Totals published in each sale's Final Bid Recap (BBG3: Preliminary Bid Recap),
# "Final Sale Summary": tracts receiving bids, bids, sum of all bids, sum of high
# bids, rejected tracts.
PUBLISHED = {
    "249": (90, 99, 137_006_181, 121_143_055, 7),
    "250": (148, 159, 139_122_383, 124_763_581, 9),
    "251": (144, 171, 202_667_923, 178_069_406, 3),
    "252": (227, 257, 283_782_480, 244_299_344, 15),
    "253": (151, 165, 174_922_200, 159_386_761, 4),
    "254": (71, 84, 108_587_185, 93_083_453, 8),
    "256": (93, 105, 135_558_336, 120_868_274, 7),
    "257": (308, 317, 198_511_834, 191_688_984, 1),
    "259": (313, 353, 309_798_397, 263_801_783, 14),
    "261": (311, 352, 441_896_332, 382_168_507, 12),
    "BBG1": (181, 219, 371_881_093, 300_425_222, 3),
    "BBG2": (25, 38, 69_838_782, 46_976_423, 2),
    "BBG3": (59, 69, 99_476_285, 82_689_756, None),
}


def check_totals(df: pd.DataFrame) -> list[str]:
    lines = []
    lines.append(f"{'sale':>5} {'tracts':>7} {'bids':>5} {'sum of all bids':>16} "
          f"{'sum of high bids':>17} {'rejected':>9}  matches BOEM")
    print(lines[-1])
    for sale, g in df.groupby("sale", sort=False):
        high = g[g["rank"] == 1]
        mine = (g.lease.nunique(), len(g), round(g.amount.sum()), round(high.amount.sum()),
                int(high.tract_rejected.sum()) if g.review_known.iloc[0] else None)
        ok = all(p is None or p == m for p, m in zip(PUBLISHED[sale], mine))
        lines.append(f"{sale:>5} {mine[0]:7d} {mine[1]:5d} {mine[2]:16,d} {mine[3]:17,d} "
                     f"{'' if mine[4] is None else mine[4]:>9}  {'yes' if ok else 'NO'}")
        print(lines[-1])
        if not ok:
            raise ValueError(f"sale {sale}: {mine} against published {PUBLISHED[sale]}")
    return lines


if __name__ == "__main__":
    bids = build()
    report = check_totals(bids)
    (HERE / "data" / "build_check.txt").write_text(
        "BOEM's published totals (BBG3: rejections not yet published) against the rebuilt table\n"
        + "\n".join(report) + "\n")
    bids.to_csv(HERE / "data" / "bids.csv", index=False)
    print(f"\nwrote data/bids.csv: {len(bids)} bids on "
          f"{bids.groupby(['sale', 'lease']).ngroups} tracts in {bids.sale.nunique()} sales")
