"""Turn the geolocated DVF files into repeat-sale aggregates for France.

    python build.py --dvf /scratch/dvf --cpi data/fr_cpi_monthly.csv --out data --temp /scratch/duck

DVF has one row per combination of mutation (a sale deed), cadastral parcel, building unit
("local") and land-use type, and the sale price (``valeur_fonciere``) is repeated on every row of
the mutation. The script first collapses rows to mutations and keeps those that transfer exactly
one dwelling (one house or one flat, with any number of outbuildings, cellars or parking spaces,
and no commercial unit), under a single disposition and a single price. Each such mutation gets
a property key:

- a house: the sorted set of cadastral parcels sold;
- a flat: its parcel and the sorted set of co-ownership lot numbers sold (the flat and its
  annexes), when the deed lists all of them (five lots or fewer).

Consecutive mutations with the same key form a pair. The chain of sales includes every nature of
mutation, so an exchange or an auction between two sales breaks the pair instead of being skipped.
A pair enters the main sample when

- both mutations are ordinary sales ("Vente"); sales off plan (VEFA, new builds) as the first sale
  and auction sales ("Adjudication") as either sale are kept apart;
- the dwelling type and the number of main rooms are unchanged, the built surface differs by
  less than 10% and, for houses, the land area by less than 10%;
- the two sales are at least 183 days apart and both prices are at least EUR 10,000;
- the absolute log price change is below log(10) and, per year held, below 0.5.

Only aggregates are written; no address, parcel, lot or individual sale leaves this script.
Outputs (all counts):

- ``pairs_hist.csv``   resale year x region x dwelling type x holding band x kind x 0.5% bin of
  the log price change (-40% to +40%, two tail bins). Kind: nominal; real; real without resales at
  exactly the old price; and the same restricted to pairs whose cumulative CPI inflation is at
  least 10%, so that the real zero sits well away from the nominal zero (``real_far_excl_exact``).
- ``pairs_cells.csv``  resale year x region x type x holding band x nature of each sale x
  structure flag: counts of nominal and real losses, hidden losses, resales at exactly the old
  price, medians.
- ``pairs_diffs.csv``  resale year x region x type x holding band: resales whose price differs
  from the old one by exactly -10,000 ... +10,000 euros.
- ``dep_hist.csv``     departement x resale year x type x 0.5% bin, nominal change within +-10%
  (tails pooled), and ``dep_cells.csv`` with the counts, exact resales and medians by departement,
  year and type.
- ``pairs_qq.csv``     region x type x quarter of purchase x quarter of resale: pairs and the sum
  of log changes (for a repeat-sales index).
- ``pairs_monthly.csv``, ``sales_monthly.csv``  monthly counts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

BIN = 0.005
EDGE = 0.40
DEP_EDGE = 0.10
HOLD_BANDS = [(0.5, 1, "0.5-1y"), (1, 2, "1-2y"), (2, 3, "2-3y"), (3, 4, "3-4y"), (4, 99, "4y+")]
DIFFS = [-10000, -5000, -2500, -1000, 0, 1000, 2500, 5000, 10000]
FAR_INFLATION = 0.10           # pairs with at least this cumulative log CPI change: real_far kind
STRUCT_TOL = 0.10              # surface and land tolerance (log ratio below log 1.1)
Q0_YEAR = 2014
BIG_CITY_DEPS = ["06", "13", "31", "33", "34", "35", "44", "59", "69"]
INNER_RING = ["92", "93", "94"]
OUTER_IDF = ["77", "78", "91", "95"]
VENTE = "Vente"
VEFA = "Vente en l'état futur d'achèvement"
AUCTION = "Adjudication"
COLS = ["id_mutation", "date_mutation", "numero_disposition", "nature_mutation", "valeur_fonciere",
        "code_commune", "code_departement", "id_parcelle", "lot1_numero", "lot2_numero",
        "lot3_numero", "lot4_numero", "lot5_numero", "nombre_lots", "code_type_local",
        "surface_reelle_bati", "nombre_pieces_principales", "code_nature_culture", "surface_terrain"]


def _q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _sql_list(xs: list[str]) -> str:
    return ", ".join(_q(x) for x in xs)


def region_sql(col: str = "dep") -> str:
    return f"""CASE
        WHEN {col} = '75' THEN 'Paris'
        WHEN {col} IN ({_sql_list(INNER_RING)}) THEN 'Inner ring'
        WHEN {col} IN ({_sql_list(OUTER_IDF)}) THEN 'Outer Ile-de-France'
        WHEN {col} IN ({_sql_list(BIG_CITY_DEPS)}) THEN 'Big cities'
        WHEN {col} LIKE '97%' THEN 'Overseas'
        ELSE 'Rest of France' END"""


def hold_sql(col: str = "hold_y") -> str:
    parts = " ".join(f"WHEN {col} >= {lo} AND {col} < {hi} THEN '{lab}'" for lo, hi, lab in HOLD_BANDS)
    return f"CASE {parts} END"


def bin_sql(col: str, edge: float = EDGE) -> str:
    return (f"CASE WHEN {col} < -{edge} THEN -999 WHEN {col} >= {edge} THEN 999 "
            f"ELSE CAST(floor({col} / {BIN}) AS INTEGER) END")


def nature_kind(col: str) -> str:
    return (f"CASE WHEN {col} = {_q(VENTE)} THEN 'sale' WHEN {col} = {_q(VEFA)} THEN 'off-plan' "
            f"WHEN {col} = {_q(AUCTION)} THEN 'auction' ELSE 'other' END")


def _one(con: duckdb.DuckDBPyConnection, sql: str):
    return con.execute(sql).fetchall()[0][0]


def build(dvf: list[str], cpi: str, out: Path, memory_limit: str = "10GB", threads: int | None = None,
          temp: str | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    db = ":memory:"
    if temp:
        Path(temp).mkdir(parents=True, exist_ok=True)
        db = str(Path(temp) / "dvf.duckdb")
        Path(db).unlink(missing_ok=True)
    con = duckdb.connect(db)
    con.execute(f"SET memory_limit='{memory_limit}'")
    if temp:
        con.execute(f"SET temp_directory='{temp}'")
    con.execute("SET preserve_insertion_order=false")
    if threads:
        con.execute(f"SET threads={threads}")
    files = "[" + ", ".join(_q(f) for f in dvf) + "]"
    stats: dict = {}
    con.execute(f"CREATE TABLE cpi AS SELECT CAST(month AS DATE) AS m, CAST(cpi AS DOUBLE) AS cpi "
                f"FROM read_csv({_q(cpi)}, header=true)")
    cols = ", ".join(COLS)
    con.execute(f"""
        CREATE TABLE r AS
        SELECT id_mutation AS mid,
               TRY_CAST(date_mutation AS DATE) AS d,
               nature_mutation AS nat,
               TRY_CAST(valeur_fonciere AS DOUBLE) AS v,
               numero_disposition AS disp,
               code_commune AS com,
               code_departement AS dep,
               id_parcelle AS parc,
               list_transform(
                 list_filter([lot1_numero, lot2_numero, lot3_numero, lot4_numero, lot5_numero],
                             lambda x: x IS NOT NULL AND trim(x) <> ''),
                 lambda x: coalesce(nullif(regexp_replace(upper(trim(x)), '^0+', ''), ''), '0')) AS lots,
               TRY_CAST(nombre_lots AS INTEGER) AS nlots,
               TRY_CAST(code_type_local AS INTEGER) AS tl,
               TRY_CAST(surface_reelle_bati AS DOUBLE) AS sb,
               TRY_CAST(nombre_pieces_principales AS INTEGER) AS np,
               code_nature_culture AS cult,
               TRY_CAST(surface_terrain AS DOUBLE) AS st
        FROM (SELECT {cols} FROM read_csv({files}, header=true, all_varchar=true,
                                          union_by_name=true, delim=','))
    """)
    stats["rows"] = _one(con, "SELECT count(*) FROM r")
    stats["first_date"] = str(_one(con, "SELECT min(d) FROM r"))
    stats["last_date"] = str(_one(con, "SELECT max(d) FROM r"))
    for nat, n in con.execute("SELECT nat, count(DISTINCT mid) FROM r GROUP BY nat ORDER BY 2 DESC").fetchall():
        stats[f"mutations_{nat}"] = n
    stats["mutations"] = _one(con, "SELECT count(DISTINCT mid) FROM r")
    con.execute("""
        CREATE TABLE mut AS
        SELECT mid,
               min(d) AS d, count(DISTINCT d) AS n_d,
               any_value(nat) AS nat, count(DISTINCT nat) AS n_nat,
               max(v) AS v, count(DISTINCT v) AS n_v,
               count(DISTINCT disp) AS n_disp,
               any_value(dep) AS dep, count(DISTINCT dep) AS n_dep,
               count(DISTINCT parc) AS n_parc,
               string_agg(DISTINCT parc, ',' ORDER BY parc) AS parcs,
               max(nlots) AS max_nlots,
               count(DISTINCT CASE WHEN tl IN (1, 2) THEN concat_ws('|', parc, tl, coalesce(sb, -1),
                                                                      coalesce(np, -1)) END) AS n_main,
               count(DISTINCT CASE WHEN tl = 4 THEN concat_ws('|', parc, coalesce(sb, -1)) END) AS n_comm,
               max(CASE WHEN tl IN (1, 2) THEN tl END) AS tl,
               max(CASE WHEN tl IN (1, 2) THEN sb END) AS sb,
               max(CASE WHEN tl IN (1, 2) THEN np END) AS np
        FROM r GROUP BY mid
    """)
    con.execute("""
        CREATE TABLE mlots AS
        SELECT mid, string_agg(DISTINCT lot, ',' ORDER BY lot) AS lotset, count(DISTINCT lot) AS n_lots
        FROM (SELECT mid, unnest(lots) AS lot FROM r) GROUP BY mid
    """)
    con.execute("""
        CREATE TABLE mland AS
        SELECT mid, sum(st) AS land
        FROM (SELECT DISTINCT mid, parc, cult, st FROM r WHERE st IS NOT NULL AND st > 0) GROUP BY mid
    """)
    con.execute("DROP TABLE r")
    stats["single_dwelling_mutations"] = _one(con, """
        SELECT count(*) FROM mut
        WHERE n_main = 1 AND n_comm = 0 AND n_nat = 1 AND n_v = 1 AND n_disp = 1 AND n_dep = 1
          AND n_d = 1 AND v IS NOT NULL AND d IS NOT NULL""")
    con.execute(f"""
        CREATE TABLE s AS
        SELECT m.mid, m.d, m.nat, m.v, m.dep, {region_sql('m.dep')} AS region,
               CASE WHEN m.tl = 1 THEN 'house' ELSE 'flat' END AS ptype, m.sb, m.np,
               coalesce(ld.land, 0) AS land,
               CASE WHEN m.tl = 1 THEN 'M|' || m.parcs
                    ELSE 'A|' || m.parcs || '|' || l.lotset END AS pkey
        FROM mut m
        LEFT JOIN mlots l USING (mid)
        LEFT JOIN mland ld USING (mid)
        WHERE m.n_main = 1 AND m.n_comm = 0 AND m.n_nat = 1 AND m.n_v = 1 AND m.n_disp = 1
          AND m.n_dep = 1 AND m.n_d = 1 AND m.v IS NOT NULL AND m.d IS NOT NULL
          AND (m.tl = 1 OR (m.n_parc = 1 AND l.lotset IS NOT NULL
                            AND coalesce(m.max_nlots, l.n_lots) <= 5 AND l.n_lots <= 5))
    """)
    stats["keyed_sales"] = _one(con, "SELECT count(*) FROM s")
    stats["keyed_houses"] = _one(con, "SELECT count(*) FROM s WHERE ptype = 'house'")
    stats["keyed_flats"] = _one(con, "SELECT count(*) FROM s WHERE ptype = 'flat'")
    # a key sold twice on the same day is ambiguous: drop every sale of that key on that day
    con.execute("""
        CREATE TABLE s2 AS
        SELECT * EXCLUDE (n_same_day)
        FROM (SELECT *, count(*) OVER (PARTITION BY pkey, d) AS n_same_day FROM s)
        WHERE n_same_day = 1
    """)
    stats["same_day_duplicates_dropped"] = stats["keyed_sales"] - _one(con, "SELECT count(*) FROM s2")
    con.execute("DROP TABLE s")
    con.execute(f"""
        COPY (SELECT CAST(date_trunc('month', d) AS DATE) AS month, region, ptype,
                     {nature_kind('nat')} AS nature, count(*) AS sales
              FROM s2 GROUP BY ALL ORDER BY month, region, ptype, nature)
        TO {_q(str(out / 'sales_monthly.csv'))} (HEADER)
    """)
    con.execute(f"""
        CREATE TABLE pairs AS
        SELECT * FROM (
          SELECT pkey, region, dep, ptype, d, v, sb, np, land, {nature_kind('nat')} AS nat1,
                 lag(d) OVER w AS d0, lag(v) OVER w AS v0, lag(sb) OVER w AS sb0,
                 lag(np) OVER w AS np0, lag(land) OVER w AS land0, lag(ptype) OVER w AS ptype0,
                 {nature_kind('lag(nat) OVER w')} AS nat0
          FROM s2
          WINDOW w AS (PARTITION BY pkey ORDER BY d, mid)
        ) WHERE d0 IS NOT NULL
    """)
    stats["raw_pairs"] = _one(con, "SELECT count(*) FROM pairs")
    tol = f"ln({1 + STRUCT_TOL})"
    con.execute(f"""
        CREATE TABLE clean AS
        SELECT region, dep, ptype, year(d) AS y, CAST(date_trunc('month', d) AS DATE) AS month,
               ({year_q('d')}) AS q1, ({year_q('d0')}) AS q0,
               v, v0, nat0, nat1, same_struct, hold_y, lr, lr - ln(c1.cpi / c0.cpi) AS lrr,
               ln(c1.cpi / c0.cpi) AS cum_infl, {hold_sql()} AS hold
        FROM (
          SELECT *, date_diff('day', d0, d) / 365.25 AS hold_y, ln(v / v0) AS lr,
                 (ptype = ptype0 AND np IS NOT DISTINCT FROM np0
                  AND sb > 0 AND sb0 > 0 AND abs(ln(sb / sb0)) < {tol}
                  AND (ptype = 'flat'
                       OR (land = 0 AND land0 = 0)
                       OR (land > 0 AND land0 > 0 AND abs(ln(land / land0)) < {tol}))) AS same_struct
          FROM pairs
          WHERE date_diff('day', d0, d) >= 183 AND v >= 10000 AND v0 >= 10000
        ) x
        JOIN cpi c0 ON c0.m = CAST(date_trunc('month', x.d0) AS DATE)
        JOIN cpi c1 ON c1.m = CAST(date_trunc('month', x.d) AS DATE)
        WHERE abs(lr) < ln(10) AND abs(lr) / hold_y <= 0.5
    """)
    stats["pairs_after_price_and_time_filters"] = _one(con, "SELECT count(*) FROM clean")
    stats["pairs_structure_changed"] = _one(con, "SELECT count(*) FROM clean WHERE NOT same_struct")
    for k0, k1, n in con.execute("SELECT nat0, nat1, count(*) FROM clean WHERE same_struct "
                                 "GROUP BY ALL ORDER BY 3 DESC").fetchall():
        stats[f"pairs_{k0}_then_{k1}"] = n
    con.execute("CREATE VIEW main AS SELECT * FROM clean WHERE same_struct AND nat0 = 'sale' AND nat1 = 'sale'")
    stats["main_pairs"] = _one(con, "SELECT count(*) FROM main")
    con.execute(f"""
        COPY (
          SELECT y, region, ptype, hold, 'nominal' AS kind, {bin_sql('lr')} AS bin, count(*) AS n
          FROM main GROUP BY ALL
          UNION ALL
          SELECT y, region, ptype, hold, 'real' AS kind, {bin_sql('lrr')} AS bin, count(*) AS n
          FROM main GROUP BY ALL
          UNION ALL
          SELECT y, region, ptype, hold, 'real_excl_exact' AS kind, {bin_sql('lrr')} AS bin, count(*) AS n
          FROM main WHERE v <> v0 GROUP BY ALL
          UNION ALL
          SELECT y, region, ptype, hold, 'real_far_excl_exact' AS kind, {bin_sql('lrr')} AS bin, count(*) AS n
          FROM main WHERE v <> v0 AND cum_infl >= {FAR_INFLATION} GROUP BY ALL
          ORDER BY y, region, ptype, hold, kind, bin
        ) TO {_q(str(out / 'pairs_hist.csv'))} (HEADER)
    """)
    con.execute(f"""
        COPY (
          SELECT y, region, ptype, hold, nat0 AS first_sale, nat1 AS second_sale, same_struct,
                 count(*) AS n,
                 count(*) FILTER (WHERE lr < 0) AS nominal_loss,
                 count(*) FILTER (WHERE lrr < 0) AS real_loss,
                 count(*) FILTER (WHERE lrr < 0 AND lr >= 0) AS hidden_loss,
                 count(*) FILTER (WHERE v = v0) AS exact_old_price,
                 count(*) FILTER (WHERE lr >= 0 AND lr < 0.02) AS within_2pct_above,
                 count(*) FILTER (WHERE lr < 0 AND lr >= -0.02) AS within_2pct_below,
                 count(*) FILTER (WHERE cum_infl >= {FAR_INFLATION}) AS far_inflation,
                 median(lr) AS median_lr, median(lrr) AS median_lrr,
                 avg(cum_infl) AS mean_cum_infl, median(hold_y) AS median_hold_y,
                 median(v0) AS median_price0
          FROM clean GROUP BY ALL ORDER BY y, region, ptype, hold, first_sale, second_sale, same_struct
        ) TO {_q(str(out / 'pairs_cells.csv'))} (HEADER)
    """)
    diff_cols = ", ".join(f"count(*) FILTER (WHERE v - v0 = {k}) AS \"d{k}\"" for k in DIFFS)
    con.execute(f"""
        COPY (
          SELECT y, region, ptype, hold, count(*) AS n, {diff_cols},
                 count(*) FILTER (WHERE v0 % 1000 = 0) AS round_k0,
                 count(*) FILTER (WHERE v % 1000 = 0) AS round_k1
          FROM main GROUP BY ALL ORDER BY y, region, ptype, hold
        ) TO {_q(str(out / 'pairs_diffs.csv'))} (HEADER)
    """)
    con.execute(f"""
        COPY (
          SELECT dep, y, ptype, {bin_sql('lr', DEP_EDGE)} AS bin, count(*) AS n
          FROM main GROUP BY ALL ORDER BY dep, y, ptype, bin
        ) TO {_q(str(out / 'dep_hist.csv'))} (HEADER)
    """)
    con.execute(f"""
        COPY (
          SELECT dep, region, y, ptype, count(*) AS n,
                 count(*) FILTER (WHERE v = v0) AS exact_old_price,
                 count(*) FILTER (WHERE lr < 0) AS nominal_loss,
                 median(lr) AS median_lr,
                 median(lr) FILTER (WHERE hold_y >= 1 AND hold_y < 3) AS median_lr_1_3y,
                 count(*) FILTER (WHERE hold_y >= 1 AND hold_y < 3) AS n_1_3y,
                 median(hold_y) AS median_hold_y
          FROM main GROUP BY ALL ORDER BY dep, y, ptype
        ) TO {_q(str(out / 'dep_cells.csv'))} (HEADER)
    """)
    con.execute(f"""
        COPY (
          SELECT region, ptype, q0, q1, count(*) AS n, sum(lr) AS sum_lr, sum(lr * lr) AS sum_lr2
          FROM main GROUP BY ALL ORDER BY region, ptype, q0, q1
        ) TO {_q(str(out / 'pairs_qq.csv'))} (HEADER)
    """)
    con.execute(f"""
        COPY (
          SELECT month, region, ptype, count(*) AS n,
                 count(*) FILTER (WHERE lr < 0) AS nominal_loss,
                 count(*) FILTER (WHERE lrr < 0) AS real_loss,
                 count(*) FILTER (WHERE v = v0) AS exact_old_price,
                 median(lr) AS median_lr
          FROM main GROUP BY ALL ORDER BY month, region, ptype
        ) TO {_q(str(out / 'pairs_monthly.csv'))} (HEADER)
    """)
    first, last = con.execute("SELECT min(month), max(month) FROM main").fetchall()[0]
    stats["first_resale_month"] = str(first)
    stats["last_resale_month"] = str(last)
    (out / "build_log.txt").write_text("\n".join(f"{k}\t{v}" for k, v in stats.items()) + "\n")
    con.close()
    if db != ":memory:":
        Path(db).unlink(missing_ok=True)
    return stats


def year_q(col: str) -> str:
    return f"(year({col}) - {Q0_YEAR}) * 4 + quarter({col}) - 1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dvf", required=True, help="directory with dvf_<year>.csv.gz files")
    ap.add_argument("--cpi", default="data/fr_cpi_monthly.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--memory-limit", default="10GB")
    ap.add_argument("--temp", default=None, help="DuckDB spill directory (outside the repository)")
    args = ap.parse_args()
    files = sorted(str(p) for p in Path(args.dvf).glob("dvf_*.csv*"))
    if not files:
        print(f"no dvf_*.csv files in {args.dvf}", file=sys.stderr)
        return 1
    print(build(files, args.cpi, Path(args.out), args.memory_limit, temp=args.temp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
