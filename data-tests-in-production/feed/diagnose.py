"""Follow the alarms on the real deliveries back to the rows.

    python -m feed.diagnose --months 2019-01,2019-02 --out results/diagnosis.txt --cache /tmp/tlc

A change check fires on a number in a profile. Whether the alarm is a broken delivery, a change
in the world, or a change in how the data are recorded can only be settled on the rows. This
script answers the questions the alarms in ``results/summary.txt`` raised. The months were
chosen after those alarms were seen, so it is a follow-up and plays no part in the evaluation.

For each month:

- by vendor: rows, share with payment type 0, share whose total does not reconcile with its
  components, share with drop-off before pick-up, share with a negative fare;
- among rows that do not reconcile, the most common gaps (total minus the sum of components),
  and how many rows each alternative reading of the total would explain;
- days with fewer than half the month's median trips, split by vendor;
- negative fares and exact duplicate rows, where they come from;
- rows with payment type 0: which columns are empty;
- every column not in the TLC dictionary, with its values.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import duckdb
import numpy as np

from feed.checks import CONTRACT
from feed.profile import COMPONENTS, _cols, _q, download

NULLABLE = ("passenger_count", "ratecodeid", "store_and_fwd_flag", "congestion_surcharge", "airport_fee")


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.2f}%"


def report(path: Path, month: str) -> list[str]:
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{path}')")
    cols = _cols(con, "t")
    c = {k: _q(v[0]) for k, v in cols.items()}

    def num(k: str) -> str:
        return f"CAST({c[k]} AS DOUBLE)"

    present = [k for k in COMPONENTS if k in c]
    parts = " + ".join(f"coalesce({num(k)}, 0)" for k in present)
    gap = f"round({num('total_amount')} - ({parts}), 2)"
    vendor, pay = c["vendorid"], f"CAST({c['payment_type']} AS VARCHAR)"
    pu, do = c["tpep_pickup_datetime"], c["tpep_dropoff_datetime"]
    first = f"DATE '{month}-01'"
    nxt = f"({first} + INTERVAL 1 MONTH)"
    n = con.execute("SELECT count(*) FROM t").fetchone()[0]
    L = [f"=== {month}: {n:,} rows; columns {', '.join(v[0] for v in cols.values())}"]

    # by vendor
    L.append("  by vendor: rows | payment type 0 | total does not reconcile | drop-off before pick-up "
             "(median gap, min) | negative fare")
    rows = con.execute(
        f"SELECT {vendor}, count(*), avg(CASE WHEN {pay} = '0' THEN 1.0 ELSE 0 END), "
        f"avg(CASE WHEN abs({gap}) > 0.011 THEN 1.0 ELSE 0 END), avg(CASE WHEN {do} < {pu} THEN 1.0 ELSE 0 END), "
        f"quantile_cont(CASE WHEN {do} < {pu} THEN (epoch({pu}) - epoch({do})) / 60 END, 0.5), "
        f"avg(CASE WHEN {num('fare_amount')} < 0 THEN 1.0 ELSE 0 END) FROM t GROUP BY 1 ORDER BY 2 DESC").fetchall()
    for v, k, p0, rec, dp, dpm, neg in rows:
        gap_txt = f" ({dpm:.0f})" if dpm is not None else ""
        L.append(f"    vendor {v}: {k:,} | {pct(p0)} | {pct(rec)} | {pct(dp)}{gap_txt} | {pct(neg)}")

    # reconciliation: the gaps, and what each alternative reading of the total explains
    bad = con.execute(f"SELECT count(*) FROM t WHERE abs({gap}) > 0.011").fetchone()[0]
    if bad:
        L.append(f"  rows whose total does not reconcile: {bad:,} ({pct(bad / n)}); most common gaps (vendor, gap, "
                 "share of those rows):")
        for v, g, k in con.execute(f"SELECT {vendor}, {gap} AS g, count(*) FROM t WHERE abs({gap}) > 0.011 "
                                   "GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 8").fetchall():
            L.append(f"    vendor {v}: {g:+.2f} on {pct(k / bad)}")
        alts = {f"total excludes {k}": num(k) for k in present if k != "fare_amount"}
        if {"congestion_surcharge", "airport_fee"} <= set(present):
            alts["total excludes congestion_surcharge and airport_fee"] = (
                f"coalesce({num('congestion_surcharge')}, 0) + coalesce({num('airport_fee')}, 0)")
        L.append("  rows that do not reconcile, explained if the total left out one component:")
        for label, e in alts.items():
            k = con.execute(f"SELECT count(*) FROM t WHERE abs({gap}) > 0.011 "
                            f"AND abs({gap} + coalesce({e}, 0)) <= 0.011").fetchone()[0]
            if k / bad >= 0.005:
                L.append(f"    {label}: {pct(k / bad)}")
        k = con.execute(f"SELECT count(*) FROM t WHERE abs({gap}) > 0.011 AND {num('extra')} >= 2.5").fetchone()[0]
        L.append(f"    share of them with extra >= 2.50: {pct(k / bad)}")

    # thin days
    rows = con.execute(f"SELECT CAST({pu} AS DATE), {vendor}, count(*) FROM t WHERE {pu} >= {first} "
                       f"AND {pu} < {nxt} GROUP BY 1, 2").fetchall()
    days: dict = {}
    for d, v, k in rows:
        days.setdefault(d, {})[v] = k
    totals = {d: sum(x.values()) for d, x in days.items()}
    med = float(np.median(list(totals.values()))) if totals else 0.0
    thin = sorted(d for d, k in totals.items() if k < 0.5 * med)
    if thin:
        L.append(f"  days with fewer than half the median day ({med:,.0f} trips), by vendor:")
        for d in thin:
            L.append(f"    {d}: {totals[d]:,} = " + ", ".join(f"vendor {v} {k:,}" for v, k in sorted(days[d].items())))

    # negative fares
    neg = con.execute(f"SELECT count(*) FROM t WHERE {num('fare_amount')} < 0").fetchone()[0]
    if neg / n > 0.005:
        L.append(f"  negative fares: {neg:,} ({pct(neg / n)}); by payment type (share of them, median fare):")
        for p, k, m in con.execute(f"SELECT {pay}, count(*), median({num('fare_amount')}) FROM t "
                                   f"WHERE {num('fare_amount')} < 0 GROUP BY 1 ORDER BY 2 DESC LIMIT 5").fetchall():
            L.append(f"    payment type {p}: {pct(k / neg)}, median {m:.2f}")
        k = con.execute(f"SELECT count(*) FROM t a WHERE {num('fare_amount')} < 0 AND EXISTS (SELECT 1 FROM t b "
                        f"WHERE b.{vendor} = a.{vendor} AND b.{pu} = a.{pu} AND b.{do} = a.{do} "
                        f"AND CAST(b.{c['fare_amount']} AS DOUBLE) = -CAST(a.{c['fare_amount']} AS DOUBLE))"
                        ).fetchone()[0]
        L.append(f"    with a positive twin (same vendor and times, opposite fare): {pct(k / neg)}")

    # payment type 0
    p0 = con.execute(f"SELECT count(*) FROM t WHERE {pay} = '0'").fetchone()[0]
    if p0 / n > 0.005:
        empty = ", ".join(f"avg(CASE WHEN {c[k]} IS NULL THEN 1.0 ELSE 0 END)" for k in NULLABLE if k in c)
        shares = con.execute(f"SELECT {empty} FROM t WHERE {pay} = '0'").fetchone()
        nulls = [f"{k} {pct(s)}" for k, s in zip([k for k in NULLABLE if k in c], shares, strict=True)]
        other = con.execute(f"SELECT avg(CASE WHEN {c['passenger_count']} IS NULL THEN 1.0 ELSE 0 END) FROM t "
                            f"WHERE {pay} <> '0'").fetchone()[0]
        L.append(f"  payment type 0: {p0:,} rows ({pct(p0 / n)}); empty among them: {', '.join(nulls)}; "
                 f"passenger_count empty on other rows: {pct(other)}")
        for label, cond in (("payment type 0", f"{pay} = '0'"), ("other payment types", f"{pay} <> '0'")):
            r = con.execute(f"SELECT median({num('fare_amount')}), median({num('total_amount')}), "
                            f"avg(CASE WHEN {num('fare_amount')} < 0 THEN 1.0 ELSE 0 END), "
                            f"avg(CASE WHEN abs({gap}) > 0.011 THEN 1.0 ELSE 0 END) FROM t WHERE {cond}").fetchone()
            L.append(f"    {label}: median fare {r[0]:.2f}, median total {r[1]:.2f}, negative fare {pct(r[2])}, "
                     f"total does not reconcile {pct(r[3])}")
        L.append("    payment type 0 by vendor: " + ", ".join(
            f"vendor {v} {pct(k / p0)}" for v, k in con.execute(
                f"SELECT {vendor}, count(*) FROM t WHERE {pay} = '0' GROUP BY 1 ORDER BY 2 DESC").fetchall()))

    # exact duplicates
    dup = con.execute("SELECT count(*) - (SELECT count(*) FROM (SELECT DISTINCT * FROM t)) FROM t").fetchone()[0]
    if dup:
        L.append(f"  exact duplicate rows: {dup:,}; by pick-up date and vendor (extra copies):")
        for d, v, k in con.execute(f"SELECT CAST({pu} AS DATE), {vendor}, sum(k - 1) FROM (SELECT *, count(*) AS k "
                                   "FROM t GROUP BY ALL) WHERE k > 1 GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 6").fetchall():
            L.append(f"    {d} vendor {v}: {int(k):,}")

    # columns the dictionary does not list
    known = {k.lower() for k in CONTRACT}
    for k, (name, typ) in cols.items():
        if k in known:
            continue
        L.append(f"  column not in the dictionary: {name} ({typ}); values:")
        for v, m in con.execute(f"SELECT CAST({c[k]} AS VARCHAR), count(*) FROM t GROUP BY 1 ORDER BY 2 DESC "
                                "LIMIT 8").fetchall():
            L.append(f"    {v}: {pct(m / n)}")
        if "ratecodeid" in c:
            r99 = con.execute(f"SELECT CAST({c[k]} AS VARCHAR), avg(CASE WHEN CAST({c['ratecodeid']} AS DOUBLE) = 99 "
                              f"THEN 1.0 ELSE 0 END), avg(CASE WHEN {pay} = '0' THEN 1.0 ELSE 0 END) FROM t "
                              "GROUP BY 1 ORDER BY 1").fetchall()
            L.append("    by value: RatecodeID 99 | payment type 0: "
                     + "; ".join(f"{v} {pct(a)} | {pct(b)}" for v, a, b in r99))
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", required=True, help="comma-separated YYYY-MM")
    ap.add_argument("--out", default="results/diagnosis.txt")
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    cache = Path(a.cache) if a.cache else Path(tempfile.mkdtemp())
    out = ["Follow-up on the alarms in summary.txt: questions answered on the rows of the months they pointed at.",
           "Months chosen after the alarms were seen; not part of the evaluation.", ""]
    for month in a.months.split(","):
        path = download(month, cache)
        out += report(path, month) + [""]
        path.unlink()
        print(f"{month} done", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
