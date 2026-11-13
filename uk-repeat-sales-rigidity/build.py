"""Turn HM Land Registry Price Paid Data into repeat-sale aggregates.

    python build.py --ppd /scratch/pp-complete.csv --cpi data/uk_cpi_monthly.csv --out data

A repeat-sale pair is two consecutive standard sales (PPD category A) of the same address:
postcode, primary and secondary addressable object name and street. A pair is kept when

- property type (detached, semi-detached, terraced, flat) and tenure are the same at both sales,
- the second sale is not flagged as a new build,
- the two sales are at least 183 days apart,
- both prices are at least GBP 10,000,
- the absolute log price change is below log(10) and, per year held, below 0.5.

Pairs whose first sale was a new build are kept apart (new-build premium) and excluded from the
histograms. Only aggregates are written; no address or individual transaction leaves this
script. Outputs (all counts):

- ``pairs_hist.csv``  resale year x region x holding band x kind x 0.5% bin of the log price
  change, -40% to +40% plus two tail bins; kind is nominal, real, or real without the resales at
  exactly the old price (for the real-zero placebo);
- ``pairs_cells.csv`` resale year x region x holding band x first-sale-new-build flag: counts of
  nominal and real losses, hidden losses, exact resales at the old price, medians;
- ``pairs_diffs.csv`` resale year x region x holding band: resales whose price differs from the
  old price by exactly -10,000 ... +10,000 GBP (the exact-price test and its placebos);
- ``pairs_monthly.csv`` resale month x region: pairs, nominal and real losses, exact resales;
- ``sales_monthly.csv`` month x region: all standard sales (transaction volume).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

COLS = ["tuid", "price", "date", "postcode", "ptype", "oldnew", "duration", "paon", "saon",
        "street", "locality", "town", "district", "county", "ppd_cat", "rec_status"]
BIN = 0.005
EDGE = 0.40
NBIN = int(round(EDGE / BIN))
HOLD_BANDS = [(0.5, 2, "0.5-2y"), (2, 5, "2-5y"), (5, 10, "5-10y"), (10, 20, "10-20y"),
              (20, 99, "20y+")]
DIFFS = [-10000, -5000, -2500, -1000, 0, 1000, 2500, 5000, 10000]
LONDON_AREAS = ["E", "EC", "N", "NW", "SE", "SW", "W", "WC"]
SOUTH_AREAS = ["AL", "BA", "BH", "BN", "BR", "BS", "CB", "CM", "CO", "CR", "CT", "DA", "DT", "EN",
               "EX", "GL", "GU", "HA", "HP", "IG", "IP", "KT", "LU", "ME", "MK", "NR", "OX", "PE",
               "PL", "PO", "RG", "RH", "RM", "SG", "SL", "SM", "SN", "SO", "SP", "SS", "TA", "TN",
               "TQ", "TR", "TW", "UB", "WD"]
WALES_AREAS = ["CF", "LL", "LD", "NP", "SA"]
WALES_COUNTIES = ["BLAENAU GWENT", "BRIDGEND", "CAERPHILLY", "CARDIFF", "CARMARTHENSHIRE",
                  "CEREDIGION", "CONWY", "DENBIGHSHIRE", "FLINTSHIRE", "GWYNEDD", "ISLE OF ANGLESEY",
                  "MERTHYR TYDFIL", "MONMOUTHSHIRE", "NEATH PORT TALBOT", "NEWPORT", "PEMBROKESHIRE",
                  "POWYS", "RHONDDA CYNON TAFF", "SWANSEA", "THE VALE OF GLAMORGAN",
                  "VALE OF GLAMORGAN", "TORFAEN", "WREXHAM"]


def _sql_list(xs: list[str]) -> str:
    return ", ".join("'" + x.replace("'", "''") + "'" for x in xs)


def region_sql() -> str:
    return f"""CASE
        WHEN county = 'GREATER LONDON' THEN 'London'
        WHEN county IN ({_sql_list(WALES_COUNTIES)}) OR area IN ({_sql_list(WALES_AREAS)}) THEN 'Wales'
        WHEN area IN ({_sql_list(SOUTH_AREAS + LONDON_AREAS)}) THEN 'South'
        ELSE 'North and Midlands' END"""


def hold_sql(col: str = "hold_y") -> str:
    parts = " ".join(f"WHEN {col} >= {lo} AND {col} < {hi} THEN '{lab}'" for lo, hi, lab in HOLD_BANDS)
    return f"CASE {parts} END"


def bin_sql(col: str) -> str:
    return (f"CASE WHEN {col} < -{EDGE} THEN -999 WHEN {col} >= {EDGE} THEN 999 "
            f"ELSE CAST(floor({col} / {BIN}) AS INTEGER) END")


def build(ppd: str, cpi: str, out: Path, memory_limit: str = "10GB", threads: int | None = None,
          temp: str | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    if temp:
        con.execute(f"SET temp_directory='{temp}'")
    con.execute("SET preserve_insertion_order=false")
    if threads:
        con.execute(f"SET threads={threads}")
    cols = "{" + ", ".join(f"'{c}': 'VARCHAR'" for c in COLS) + "}"
    con.execute(f"CREATE TABLE cpi AS SELECT CAST(month AS DATE) AS m, CAST(cpi AS DOUBLE) AS cpi "
                f"FROM read_csv('{cpi}', header=true)")
    con.execute(f"""
        CREATE TABLE s AS
        SELECT tuid, p, d, pc, ptype, oldnew, duration, paon, saon, street, area, {region_sql()} AS region
        FROM (
          SELECT tuid,
                 TRY_CAST(price AS BIGINT) AS p,
                 TRY_CAST(substr(date, 1, 10) AS DATE) AS d,
                 upper(trim(postcode)) AS pc,
                 regexp_extract(upper(trim(postcode)), '^[A-Z]+') AS area,
                 ptype, oldnew, duration,
                 upper(trim(paon)) AS paon,
                 upper(trim(coalesce(saon, ''))) AS saon,
                 upper(trim(coalesce(street, ''))) AS street,
                 upper(trim(coalesce(county, ''))) AS county
          FROM read_csv('{ppd}', header=false, columns={cols}, quote='"', escape='"',
                        delim=',', parallel=true)
          WHERE ppd_cat = 'A' AND coalesce(rec_status, 'A') <> 'D'
            AND ptype IN ('D', 'S', 'T', 'F')
            AND postcode IS NOT NULL AND trim(postcode) <> ''
            AND paon IS NOT NULL AND trim(paon) <> ''
        ) WHERE p IS NOT NULL AND d IS NOT NULL
    """)
    n_sales = con.execute("SELECT count(*) FROM s").fetchone()[0]
    con.execute(f"""
        COPY (SELECT CAST(date_trunc('month', d) AS DATE) AS month, region, count(*) AS sales
              FROM s GROUP BY ALL ORDER BY month, region)
        TO '{out / "sales_monthly.csv"}' (HEADER)
    """)
    con.execute("""
        CREATE TABLE pairs AS
        SELECT * FROM (
          SELECT region, ptype, duration, oldnew, p, d,
                 lag(p) OVER w AS p0, lag(d) OVER w AS d0, lag(ptype) OVER w AS ptype0,
                 lag(duration) OVER w AS duration0, lag(oldnew) OVER w AS new0
          FROM s
          WINDOW w AS (PARTITION BY pc, paon, saon, street ORDER BY d, tuid)
        ) WHERE p0 IS NOT NULL
    """)
    n_raw_pairs = con.execute("SELECT count(*) FROM pairs").fetchone()[0]
    con.execute("DROP TABLE s")
    con.execute(f"""
        CREATE TABLE clean AS
        SELECT region, year(d) AS y, CAST(date_trunc('month', d) AS DATE) AS month,
               p, p0, new0, hold_y, lr, lr - ln(c1.cpi / c0.cpi) AS lrr,
               ln(c1.cpi / c0.cpi) / hold_y AS infl_ann, {hold_sql()} AS hold
        FROM (
          SELECT *, date_diff('day', d0, d) / 365.25 AS hold_y, ln(CAST(p AS DOUBLE) / p0) AS lr
          FROM pairs
          WHERE ptype = ptype0 AND duration = duration0 AND oldnew = 'N'
            AND date_diff('day', d0, d) >= 183 AND p >= 10000 AND p0 >= 10000
        ) x
        JOIN cpi c0 ON c0.m = CAST(date_trunc('month', x.d0) AS DATE)
        JOIN cpi c1 ON c1.m = CAST(date_trunc('month', x.d) AS DATE)
        WHERE abs(lr) < ln(10) AND abs(lr) / hold_y <= 0.5
    """)
    n_clean = con.execute("SELECT count(*) FROM clean").fetchone()[0]
    con.execute(f"""
        COPY (
          SELECT y, region, hold, 'nominal' AS kind, {bin_sql('lr')} AS bin, count(*) AS n
          FROM clean WHERE new0 = 'N' GROUP BY ALL
          UNION ALL
          SELECT y, region, hold, 'real' AS kind, {bin_sql('lrr')} AS bin, count(*) AS n
          FROM clean WHERE new0 = 'N' GROUP BY ALL
          UNION ALL
          SELECT y, region, hold, 'real_excl_exact' AS kind, {bin_sql('lrr')} AS bin, count(*) AS n
          FROM clean WHERE new0 = 'N' AND p <> p0 GROUP BY ALL
          ORDER BY y, region, hold, kind, bin
        ) TO '{out / "pairs_hist.csv"}' (HEADER)
    """)
    con.execute(f"""
        COPY (
          SELECT y, region, hold, new0 AS first_sale_new,
                 count(*) AS n,
                 count(*) FILTER (WHERE lr < 0) AS nominal_loss,
                 count(*) FILTER (WHERE lrr < 0) AS real_loss,
                 count(*) FILTER (WHERE lrr < 0 AND lr >= 0) AS hidden_loss,
                 count(*) FILTER (WHERE p = p0) AS exact_old_price,
                 count(*) FILTER (WHERE lr >= 0 AND lr < 0.02) AS within_2pct_above,
                 count(*) FILTER (WHERE lr < 0 AND lr >= -0.02) AS within_2pct_below,
                 median(lr) AS median_lr, median(lrr) AS median_lrr,
                 avg(infl_ann) AS mean_infl_ann, median(hold_y) AS median_hold_y
          FROM clean GROUP BY ALL ORDER BY y, region, hold, first_sale_new
        ) TO '{out / "pairs_cells.csv"}' (HEADER)
    """)
    diff_cols = ", ".join(f"count(*) FILTER (WHERE p - p0 = {k}) AS \"d{k}\"" for k in DIFFS)
    con.execute(f"""
        COPY (
          SELECT y, region, hold, count(*) AS n, {diff_cols}
          FROM clean WHERE new0 = 'N' GROUP BY ALL ORDER BY y, region, hold
        ) TO '{out / "pairs_diffs.csv"}' (HEADER)
    """)
    con.execute(f"""
        COPY (
          SELECT month, region, count(*) AS n,
                 count(*) FILTER (WHERE lr < 0) AS nominal_loss,
                 count(*) FILTER (WHERE lrr < 0) AS real_loss,
                 count(*) FILTER (WHERE p = p0) AS exact_old_price,
                 median(lr) AS median_lr, median(lrr) AS median_lrr
          FROM clean WHERE new0 = 'N' GROUP BY ALL ORDER BY month, region
        ) TO '{out / "pairs_monthly.csv"}' (HEADER)
    """)
    first = con.execute("SELECT min(month), max(month) FROM clean").fetchone()
    stats = {"standard_sales": n_sales, "raw_pairs": n_raw_pairs, "clean_pairs": n_clean,
             "first_resale_month": str(first[0]), "last_resale_month": str(first[1])}
    (out / "build_log.txt").write_text("\n".join(f"{k}\t{v}" for k, v in stats.items()) + "\n")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppd", required=True)
    ap.add_argument("--cpi", default="data/uk_cpi_monthly.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--memory-limit", default="10GB")
    ap.add_argument("--temp", default=None, help="DuckDB spill directory (outside the repository)")
    args = ap.parse_args()
    print(build(args.ppd, args.cpi, Path(args.out), args.memory_limit, temp=args.temp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
