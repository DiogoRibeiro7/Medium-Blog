"""Turn HM Land Registry Price Paid Data into repeat-sale pairs and an area price index.

    python build.py --ppd /scratch/pp-complete.csv --pairs /scratch/pairs.parquet --out data

A repeat-sale pair is two consecutive standard sales (PPD category A) of the same address:
postcode, primary and secondary addressable object name and street. The filters are the ones
used in ``uk-repeat-sales-rigidity``:

- property type and tenure are the same at both sales,
- the second sale is not flagged as a new build,
- the two sales are at least 183 days apart,
- both prices are at least GBP 10,000,
- the absolute log price change is below log(10) and, per year held, below 0.5.

Pairs whose first sale was a new build are kept, with a flag, because a valuation model has to
value those homes too. They are left out of the index.

Outputs
-------
- ``--pairs`` (scratch, never committed): one row per pair with the postcode area, district and
  sector, type, tenure, the two prices and dates. It carries no street address, but it is still
  individual transactions, so it stays on the runner.
- ``data/area_index.csv``: a monthly repeat-sales log price index for each postcode area, from
  pairs whose first sale was not a new build (see ``area_index``).
- ``data/area_year_median.csv``: median log price of all standard sales by postcode area and
  year (the price level of the area, used to place a home within its local market).
- ``data/build_log.txt``: counts at each step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

COLS = ["tuid", "price", "date", "postcode", "ptype", "oldnew", "duration", "paon", "saon",
        "street", "locality", "town", "district", "county", "ppd_cat", "rec_status"]
START_YEAR = 1995
LAMBDA = 300.0          # smoothness of the area index (second differences); see area_index
MIN_AREA_PAIRS = 1000   # postcode areas with fewer index pairs are dropped
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


def month_index(col: str) -> str:
    """Months since January 1995 (0 = 1995-01)."""
    return f"((year({col}) - {START_YEAR}) * 12 + month({col}) - 1)"


def area_index(agg: pd.DataFrame, n_months: int, lam: float = LAMBDA) -> pd.DataFrame:
    """Monthly repeat-sales log index per area from aggregated pairs.

    ``agg`` has one row per (area, t0, t1) with ``n`` pairs and ``s`` = sum of their log price
    changes. For each area the index I minimises

        sum over pairs (lr - I[t1] + I[t0])^2  +  lam * sum over t (I[t+1] - 2 I[t] + I[t-1])^2

    with I[0] = 0. The first term is the Bailey-Muth-Nourse repeat-sales regression; the second
    is a smoothness prior on the monthly growth rate (lam = 300 corresponds to a prior standard
    deviation of about one percentage point a month for the change in monthly growth, given a
    pair-level residual of about 0.17 log points). The normal equations are T x T, so they are
    assembled from the aggregates and solved directly.
    """
    T = n_months
    d2 = np.zeros((T - 2, T))
    idx = np.arange(T - 2)
    d2[idx, idx], d2[idx, idx + 1], d2[idx, idx + 2] = 1.0, -2.0, 1.0
    pen = lam * d2.T @ d2
    out = []
    for area, g in agg.groupby("area", sort=True):
        t0 = g.t0.to_numpy(int)
        t1 = g.t1.to_numpy(int)
        n = g.n.to_numpy(float)
        s = g.s.to_numpy(float)
        ata = pen.copy()
        np.add.at(ata, (t1, t1), n)
        np.add.at(ata, (t0, t0), n)
        np.add.at(ata, (t0, t1), -n)
        np.add.at(ata, (t1, t0), -n)
        atb = np.zeros(T)
        np.add.at(atb, t1, s)
        np.add.at(atb, t0, -s)
        sol = np.linalg.solve(ata[1:, 1:], atb[1:])
        level = np.concatenate([[0.0], sol])
        touch = np.zeros(T)
        np.add.at(touch, t1, n)
        np.add.at(touch, t0, n)
        out.append(pd.DataFrame({"area": area, "t": np.arange(T), "log_index": level,
                                 "pairs_touching": touch.astype(int)}))
    res = pd.concat(out, ignore_index=True)
    res["month"] = pd.to_datetime({"year": START_YEAR + res.t // 12, "month": res.t % 12 + 1,
                                   "day": 1}).dt.strftime("%Y-%m-%d")
    return res[["area", "month", "t", "log_index", "pairs_touching"]]


def build(ppd: str, pairs_out: str, out: Path, memory_limit: str = "10GB",
          threads: int | None = None, temp: str | None = None, lam: float = LAMBDA,
          min_area_pairs: int = MIN_AREA_PAIRS) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    if temp:
        con.execute(f"SET temp_directory='{temp}'")
    con.execute("SET preserve_insertion_order=false")
    if threads:
        con.execute(f"SET threads={threads}")
    cols = "{" + ", ".join(f"'{c}': 'VARCHAR'" for c in COLS) + "}"
    con.execute(f"""
        CREATE TABLE s AS
        SELECT tuid, p, d, pc, ptype, oldnew, duration, paon, saon, street, area, outward, sector,
               {region_sql()} AS region
        FROM (
          SELECT tuid,
                 TRY_CAST(price AS BIGINT) AS p,
                 TRY_CAST(substr(date, 1, 10) AS DATE) AS d,
                 upper(trim(postcode)) AS pc,
                 regexp_extract(upper(trim(postcode)), '^[A-Z]+') AS area,
                 split_part(upper(trim(postcode)), ' ', 1) AS outward,
                 split_part(upper(trim(postcode)), ' ', 1) || ' ' ||
                   substr(split_part(upper(trim(postcode)), ' ', 2), 1, 1) AS sector,
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
        ) WHERE p IS NOT NULL AND d IS NOT NULL AND year(d) >= {START_YEAR}
    """)
    n_sales = con.execute("SELECT count(*) FROM s").fetchall()[0][0]
    con.execute(f"""
        COPY (SELECT area, year(d) AS year, count(*) AS n, median(ln(p)) AS median_lp
              FROM s WHERE p >= 10000 GROUP BY ALL ORDER BY area, year)
        TO '{out / "area_year_median.csv"}' (HEADER)
    """)
    con.execute("""
        CREATE TABLE pairs AS
        SELECT * FROM (
          SELECT area, outward, sector, region, ptype, duration, oldnew, p, d,
                 lag(p) OVER w AS p0, lag(d) OVER w AS d0, lag(ptype) OVER w AS ptype0,
                 lag(duration) OVER w AS duration0, lag(oldnew) OVER w AS new0,
                 row_number() OVER w - 1 AS n_prior
          FROM s
          WINDOW w AS (PARTITION BY pc, paon, saon, street ORDER BY d, tuid)
        ) WHERE p0 IS NOT NULL
    """)
    n_raw_pairs = con.execute("SELECT count(*) FROM pairs").fetchall()[0][0]
    con.execute("DROP TABLE s")
    con.execute(f"""
        CREATE TABLE clean AS
        SELECT area, outward, sector, region, ptype, duration AS tenure,
               (new0 = 'Y') AS new0, p0, p, {month_index('d0')} AS t0, {month_index('d')} AS t1,
               year(d) AS year, hold_y, lr, n_prior,
               (p = p0) AS exact
        FROM (
          SELECT *, date_diff('day', d0, d) / 365.25 AS hold_y, ln(CAST(p AS DOUBLE) / p0) AS lr
          FROM pairs
          WHERE ptype = ptype0 AND duration = duration0 AND oldnew = 'N'
            AND date_diff('day', d0, d) >= 183 AND p >= 10000 AND p0 >= 10000
        ) x
        WHERE abs(lr) < ln(10) AND abs(lr) / hold_y <= 0.5
    """)
    n_clean = con.execute("SELECT count(*) FROM clean").fetchall()[0][0]
    agg = con.execute("""
        SELECT area, t0, t1, count(*) AS n, sum(lr) AS s FROM clean WHERE NOT new0 GROUP BY ALL
    """).df()
    keep = agg.groupby("area").n.sum()
    keep = keep[keep >= min_area_pairs].index
    n_months = int(con.execute("SELECT max(t1) FROM clean").fetchall()[0][0]) + 1
    idx = area_index(agg[agg.area.isin(keep)], n_months, lam)
    idx.to_csv(out / "area_index.csv", index=False, float_format="%.8g")
    con.register("keep_areas", pd.DataFrame({"area": list(keep)}))
    con.execute(f"""
        COPY (SELECT c.* FROM clean c JOIN keep_areas k USING (area))
        TO '{pairs_out}' (FORMAT parquet, COMPRESSION zstd)
    """)
    n_kept = con.execute("SELECT count(*) FROM clean c JOIN keep_areas k USING (area)").fetchall()[0][0]
    span = con.execute("SELECT min(t1), max(t1) FROM clean").fetchall()[0]
    stats = {"standard_sales": n_sales, "raw_pairs": n_raw_pairs, "clean_pairs": n_clean,
             "areas_in_index": len(keep), "pairs_in_indexed_areas": n_kept,
             "index_months": n_months, "first_resale_t": span[0], "last_resale_t": span[1],
             "index_lambda": lam}
    (out / "build_log.txt").write_text("\n".join(f"{k}\t{v}" for k, v in stats.items()) + "\n")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppd", required=True)
    ap.add_argument("--pairs", required=True, help="scratch parquet file for the pairs")
    ap.add_argument("--out", default="data")
    ap.add_argument("--memory-limit", default="10GB")
    ap.add_argument("--temp", default=None, help="DuckDB spill directory (outside the repository)")
    args = ap.parse_args()
    print(build(args.ppd, args.pairs, Path(args.out), args.memory_limit, temp=args.temp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
