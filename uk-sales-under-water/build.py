"""Owners, their position against the price they paid, and whether they sold: aggregates from the Price Paid Data.

    python build.py --ppd /scratch/pp-complete.csv --out data --temp /scratch/duck

Every standard sale (PPD category A) of a house or flat starts an ownership spell at that
address (postcode, primary and secondary addressable object name, street). The spell ends at
the next standard sale of the same address, or is still running at the end of the data. On
1 January of each year from 2000 to 2025, the stock is every spell that started before that
date and had not ended, and a resale is a spell in the stock that ends during the year. The
owner's position is m, the change in the log repeat-sales index of the home's postcode area
between the month of purchase and the December before the year starts: the nominal gain the
index implies.

Only owners who bought from 1995 on are in the stock, because the data start then. Sales in
PPD category B (from October 2013: repossessions, buy-to-let purchases identified by a
mortgage, sales to companies) neither start nor end a spell, so the definition is the same in
every year.

Outputs (all aggregates; no address or single transaction):

- ``data/area_index.csv``: monthly log repeat-sales index for each postcode area, built as in
  ``uk-valuation-purchase-price`` (pairs whose first sale was not a new build).
- ``data/hazard_cells.csv.gz``: stock and resales by postcode area, year, whole years since
  purchase (capped at 25), 2% band of m, flat or house, and new build or not.
- ``data/district_year.csv.gz``: for each postcode district and year, the stock, resales,
  their split by band of m and by years since purchase, the share of flats and of new builds
  in the stock, and all standard sales in the district that year.
- ``data/national_monthly.csv``: standard sales by month.
- ``data/monthly_position.csv``: for each month from January 2005, region, band of m and band of
  years since purchase: the stock at the start of the month and the resales during it, with m
  measured at the end of the previous month. Built from purchase cohorts (area x month of
  purchase), so it needs no expansion of individual spells by month.
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
YEARS = range(2000, 2026)
LAMBDA = 300.0
MIN_AREA_PAIRS = 1000
M_BIN = 0.02
M_BIN_CAP = 50      # m of +100% and more pooled
DUR_CAP = 25
MONTHLY_FROM = 2005
M_BANDS = [(-9.0, -0.10, "below_m10"), (-0.10, 0.0, "m10_0"), (0.0, 0.10, "0_10"),
           (0.10, 0.25, "10_25"), (0.25, 0.50, "25_50"), (0.50, 9.0, "above_50")]
DUR_BANDS = [(0, 2, "d0_2"), (2, 5, "d2_5"), (5, 10, "d5_10"), (10, 99, "d10p")]
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
    return f"((year({col}) - {START_YEAR}) * 12 + month({col}) - 1)"


def area_index(agg: pd.DataFrame, n_months: int, lam: float = LAMBDA) -> pd.DataFrame:
    """Monthly repeat-sales log index per area, with a smoothness penalty on growth changes.

    Minimises sum over pairs (lr - I[t1] + I[t0])^2 + lam * sum (I[t+1] - 2 I[t] + I[t-1])^2
    with I[0] = 0, from (area, t0, t1, n, s) aggregates (s = sum of log price changes).
    """
    T = n_months
    d2 = np.zeros((T - 2, T))
    i = np.arange(T - 2)
    d2[i, i], d2[i, i + 1], d2[i, i + 2] = 1.0, -2.0, 1.0
    pen = lam * d2.T @ d2
    out = []
    for area, g in agg.groupby("area", sort=True):
        t0, t1 = g.t0.to_numpy(int), g.t1.to_numpy(int)
        n, s = g.n.to_numpy(float), g.s.to_numpy(float)
        ata = pen.copy()
        np.add.at(ata, (t1, t1), n)
        np.add.at(ata, (t0, t0), n)
        np.add.at(ata, (t0, t1), -n)
        np.add.at(ata, (t1, t0), -n)
        atb = np.zeros(T)
        np.add.at(atb, t1, s)
        np.add.at(atb, t0, -s)
        level = np.concatenate([[0.0], np.linalg.solve(ata[1:, 1:], atb[1:])])
        out.append(pd.DataFrame({"area": area, "t": np.arange(T), "log_index": level}))
    res = pd.concat(out, ignore_index=True)
    res["month"] = pd.to_datetime({"year": START_YEAR + res.t // 12, "month": res.t % 12 + 1,
                                   "day": 1}).dt.strftime("%Y-%m-%d")
    return res[["area", "month", "t", "log_index"]]


def band_case(col: str, bands, numeric: bool = True) -> str:
    parts = " ".join(f"WHEN {col} >= {lo} AND {col} < {hi} THEN '{lab}'" for lo, hi, lab in bands)
    return f"CASE {parts} END"


def build(ppd: str, out: Path, memory_limit: str = "10GB", threads: int | None = None,
          temp: str | None = None, years=YEARS, min_area_pairs: int = MIN_AREA_PAIRS) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute("SET preserve_insertion_order=false")
    if temp:
        con.execute(f"SET temp_directory='{temp}'")
    if threads:
        con.execute(f"SET threads={threads}")
    cols = "{" + ", ".join(f"'{c}': 'VARCHAR'" for c in COLS) + "}"
    con.execute(f"""
        CREATE TABLE s AS
        SELECT tuid, p, d, {month_index('d')} AS t, pc, ptype, oldnew, duration, paon, saon, street,
               area, outward, {region_sql()} AS region
        FROM (
          SELECT tuid, TRY_CAST(price AS BIGINT) AS p, TRY_CAST(substr(date, 1, 10) AS DATE) AS d,
                 upper(trim(postcode)) AS pc,
                 regexp_extract(upper(trim(postcode)), '^[A-Z]+') AS area,
                 split_part(upper(trim(postcode)), ' ', 1) AS outward,
                 ptype, oldnew, duration, upper(trim(paon)) AS paon,
                 upper(trim(coalesce(saon, ''))) AS saon, upper(trim(coalesce(street, ''))) AS street,
                 upper(trim(coalesce(county, ''))) AS county
          FROM read_csv('{ppd}', header=false, columns={cols}, quote='"', escape='"', delim=',',
                        parallel=true)
          WHERE ppd_cat = 'A' AND coalesce(rec_status, 'A') <> 'D' AND ptype IN ('D', 'S', 'T', 'F')
            AND postcode IS NOT NULL AND trim(postcode) <> '' AND paon IS NOT NULL AND trim(paon) <> ''
        ) WHERE p IS NOT NULL AND d IS NOT NULL AND year(d) >= {START_YEAR}
    """)
    n_sales = con.execute("SELECT count(*) FROM s").fetchall()[0][0]
    last_t = con.execute("SELECT max(t) FROM s").fetchall()[0][0]
    con.execute(f"""
        COPY (SELECT year(d) AS year, month(d) AS month, count(*) AS sales,
                     sum((oldnew = 'Y')::INT) AS new_builds
              FROM s GROUP BY ALL ORDER BY year, month) TO '{out / "national_monthly.csv"}' (HEADER)
    """)
    # ------------------------------------------------------------------ area index
    con.execute("""
        CREATE TABLE spells AS
        SELECT area, outward, region, ptype, oldnew, duration, p, t AS tbuy,
               lead(t) OVER w AS tnext, lead(p) OVER w AS pnext, lead(ptype) OVER w AS ptype_next,
               lead(duration) OVER w AS duration_next, lead(oldnew) OVER w AS oldnew_next,
               date_diff('day', d, lead(d) OVER w) AS days
        FROM s WINDOW w AS (PARTITION BY pc, paon, saon, street ORDER BY d, tuid)
    """)
    agg = con.execute("""
        SELECT area, tbuy AS t0, tnext AS t1, count(*) AS n, sum(ln(CAST(pnext AS DOUBLE) / p)) AS s
        FROM spells
        WHERE tnext IS NOT NULL AND ptype = ptype_next AND duration = duration_next
          AND oldnew = 'N' AND oldnew_next = 'N' AND days >= 183 AND p >= 10000 AND pnext >= 10000
          AND abs(ln(CAST(pnext AS DOUBLE) / p)) < ln(10)
          AND abs(ln(CAST(pnext AS DOUBLE) / p)) / (days / 365.25) <= 0.5
        GROUP BY ALL
    """).df()
    keep = agg.groupby("area").n.sum()
    keep = sorted(keep[keep >= min_area_pairs].index)
    idx = area_index(agg[agg.area.isin(keep)], int(last_t) + 1)
    idx.to_csv(out / "area_index.csv", index=False, float_format="%.8g")
    con.register("idx", idx[["area", "t", "log_index"]])
    # ------------------------------------------------------------------ stock by year
    con.register("years", pd.DataFrame({"Y": list(years),
                                        "ystart": [(y - START_YEAR) * 12 for y in years]}))
    con.execute(f"""
        CREATE TABLE sy AS
        SELECT sp.area, sp.outward AS district, sp.region, y.Y, y.ystart, sp.tbuy,
               (sp.ptype = 'F') AS flat, (sp.oldnew = 'Y') AS new,
               count(*) AS stock,
               sum(CASE WHEN sp.tnext IS NOT NULL AND sp.tnext < y.ystart + 12 THEN 1 ELSE 0 END) AS resales
        FROM spells sp JOIN years y ON sp.tbuy < y.ystart
                                   AND (sp.tnext IS NULL OR sp.tnext >= y.ystart)
        WHERE sp.area IN (SELECT DISTINCT area FROM idx)
        GROUP BY ALL
    """)
    con.execute(f"""
        CREATE TABLE sym AS
        SELECT sy.*, (sy.ystart - sy.tbuy) / 12.0 AS dur,
               ie.log_index - ib.log_index AS m
        FROM sy JOIN idx ie ON ie.area = sy.area AND ie.t = sy.ystart - 1
                JOIN idx ib ON ib.area = sy.area AND ib.t = sy.tbuy
    """)
    con.execute(f"""
        COPY (SELECT area, Y AS year, least(floor(dur), {DUR_CAP})::INT AS dur_year,
                     least(floor(m / {M_BIN}), {M_BIN_CAP})::INT AS m_bin, flat, new,
                     sum(stock) AS stock, sum(resales) AS resales, sum(m * stock) AS sum_m
              FROM sym GROUP BY ALL ORDER BY area, year, dur_year, m_bin, flat, new)
        TO '{out / "hazard_cells.csv.gz"}' (HEADER, COMPRESSION gzip)
    """)
    m_case = band_case("m", M_BANDS)
    d_case = band_case("dur", DUR_BANDS)
    band_cols_m = ", ".join(
        f"sum(CASE WHEN mb = '{lab}' THEN stock ELSE 0 END) AS stock_{lab}, "
        f"sum(CASE WHEN mb = '{lab}' THEN resales ELSE 0 END) AS resales_{lab}" for _, _, lab in M_BANDS)
    band_cols_d = ", ".join(
        f"sum(CASE WHEN db = '{lab}' THEN stock ELSE 0 END) AS stock_{lab}, "
        f"sum(CASE WHEN db = '{lab}' THEN resales ELSE 0 END) AS resales_{lab}" for _, _, lab in DUR_BANDS)
    con.execute(f"""
        CREATE TABLE dy AS
        SELECT district, Y AS year, mode(area) AS area, mode(region) AS region,
               sum(stock) AS stock, sum(resales) AS resales,
               sum(m * stock) / sum(stock) AS mean_m, sum(dur * stock) / sum(stock) AS mean_dur,
               sum(CASE WHEN flat THEN stock ELSE 0 END) AS stock_flat,
               sum(CASE WHEN new THEN stock ELSE 0 END) AS stock_new,
               {band_cols_m}, {band_cols_d}
        FROM (SELECT *, {m_case} AS mb, {d_case} AS db FROM sym)
        GROUP BY district, Y
    """)
    con.execute("""
        CREATE TABLE ds AS
        SELECT outward AS district, year(d) AS year, count(*) AS sales_all,
               sum((oldnew = 'Y')::INT) AS sales_new_build
        FROM s GROUP BY ALL
    """)
    con.execute(f"""
        COPY (SELECT dy.*, coalesce(ds.sales_all, 0) AS sales_all,
                     coalesce(ds.sales_new_build, 0) AS sales_new_build
              FROM dy LEFT JOIN ds USING (district, year) ORDER BY district, year)
        TO '{out / "district_year.csv.gz"}' (HEADER, COMPRESSION gzip)
    """)
    # ------------------------------------------------------------------ monthly, by cohort
    area_region = f"""CASE WHEN area IN ({_sql_list(LONDON_AREAS)}) THEN 'London'
        WHEN area IN ({_sql_list(WALES_AREAS)}) THEN 'Wales'
        WHEN area IN ({_sql_list(SOUTH_AREAS)}) THEN 'South' ELSE 'North and Midlands' END"""
    t0 = (MONTHLY_FROM - START_YEAR) * 12
    con.register("months", pd.DataFrame({"t": list(range(t0, int(last_t) + 1))}))
    con.execute("""
        CREATE TABLE coh AS SELECT area, tbuy, count(*) AS n FROM spells
        WHERE area IN (SELECT DISTINCT area FROM idx) GROUP BY ALL
    """)
    con.execute("""
        CREATE TABLE exits AS
        SELECT area, tbuy, tnext, n, sum(n) OVER (PARTITION BY area, tbuy ORDER BY tnext) AS cum
        FROM (SELECT area, tbuy, tnext, count(*) AS n FROM spells
              WHERE tnext IS NOT NULL AND area IN (SELECT DISTINCT area FROM idx) GROUP BY ALL)
    """)
    con.execute("""
        CREATE TABLE grid AS
        SELECT c.area, c.tbuy, c.n AS bought, m.t FROM coh c JOIN months m ON m.t > c.tbuy
    """)
    con.execute(f"""
        CREATE TABLE cm AS
        SELECT g.area, g.tbuy, g.t, g.bought - coalesce(e.cum, 0) AS stock,
               coalesce(x.n, 0) AS resales, (g.t - g.tbuy) / 12.0 AS dur,
               ie.log_index - ib.log_index AS m
        FROM grid g
        ASOF LEFT JOIN exits e ON e.area = g.area AND e.tbuy = g.tbuy AND e.tnext < g.t
        LEFT JOIN exits x ON x.area = g.area AND x.tbuy = g.tbuy AND x.tnext = g.t
        JOIN idx ie ON ie.area = g.area AND ie.t = g.t - 1
        JOIN idx ib ON ib.area = g.area AND ib.t = g.tbuy
    """)
    con.execute(f"""
        COPY (SELECT (t // 12 + {START_YEAR})::INT AS year, (t % 12 + 1)::INT AS month, region, mb AS m_band,
                     db AS dur_band, sum(stock) AS stock, sum(resales) AS resales, sum(m * stock) AS sum_m
              FROM (SELECT *, {area_region} AS region, {m_case} AS mb, {d_case} AS db FROM cm WHERE stock > 0)
              GROUP BY ALL ORDER BY year, month, region, m_band, dur_band)
        TO '{out / "monthly_position.csv"}' (HEADER)
    """)
    totals = con.execute("SELECT count(*), sum(stock), sum(resales) FROM sy").fetchall()[0]
    n_spells = con.execute("SELECT count(*) FROM spells").fetchall()[0][0]
    n_dist = con.execute("SELECT count(DISTINCT district) FROM dy").fetchall()[0][0]
    stats = {"standard_sales": n_sales, "spells": n_spells, "index_pairs": int(agg.n.sum()),
             "areas_in_index": len(keep), "last_month_t": last_t, "years": f"{min(years)}-{max(years)}",
             "stock_year_cells": totals[0], "owner_years": int(totals[1]), "resales": int(totals[2]),
             "districts": n_dist}
    (out / "build_log.txt").write_text("\n".join(f"{k}\t{v}" for k, v in stats.items()) + "\n")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppd", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--memory-limit", default="10GB")
    ap.add_argument("--temp", default=None)
    args = ap.parse_args()
    print(build(args.ppd, Path(args.out), args.memory_limit, temp=args.temp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
