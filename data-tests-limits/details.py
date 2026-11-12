"""Descriptive numbers quoted in the article that evaluate.py does not print.

    python details.py --profiles profiles --sources sources --results results --out results

Not part of the pre-registered analysis. Field-level shares from the profiles, the file-versus-report
difference month by month, the noise each detector has to see through, the outside-count alarms on the
real deliveries, and two consistency checks on yellow taxis that use Part 3's profiles
(../data-tests-in-production/profiles/real.jsonl.gz) when they are present. Writes results/details.txt.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np

from evaluate import read_profiles
from feed import counts as C

PART3 = Path(__file__).resolve().parent.parent / "data-tests-in-production" / "profiles" / "real.jsonl.gz"


def share(p: dict, key: str) -> float:
    return 100 * p["counts"].get(key, 0) / p["n_rows"]


def rows_section(P: dict, L: list[str]) -> None:
    L.append("Rows per monthly file, from " + C.FIRST_EVAL)
    for f in C.REPORT_CLASSES:
        ns = [p["n_rows"] for m, p in P[f].items() if m >= C.FIRST_EVAL]
        L.append(f"  {f:6s} {min(ns):>12,} to {max(ns):>12,} ({len(P[f])} files in all)")
    L.append("")


def rules_section(results: Path, L: list[str]) -> None:
    strict, tolerant, alarms = defaultdict(int), defaultdict(int), defaultdict(int)
    keys = set()
    with gzip.open(results / "rules_by_delivery.csv.gz", "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            k = (r["feed"], r["month"])
            keys.add(k)
            if r["fired"] == "1":
                alarms[k] += 1
                (strict if r["variant"] == "strict" else tolerant)[k] += 1
    L.append("E1. Rules failed per delivery")
    for f in ("yellow", "fhv", "fhvhv"):
        ks = sorted(k for k in keys if k[0] == f)
        L.append(
            f"  {f}: fewest strict failures on a delivery {min(strict[k] for k in ks)}, "
            f"fewest tolerant {min(tolerant[k] for k in ks)}"
        )
        by: dict[str, list[int]] = defaultdict(list)
        for k in ks:
            by[k[1][:4]].append(alarms[k])
        L.append("    mean alarms by year: " + "; ".join(f"{y} {mean(v):.2f}" for y, v in sorted(by.items())))
    L.append("")


def fhv_section(P: dict, L: list[str]) -> None:
    F = P["fhv"]
    L.append("FHV fields, % of rows, by year (median over the year's months; ranges are min to max)")
    for y in sorted({m[:4] for m in F}):
        ps = [F[m] for m in sorted(F) if m.startswith(y)]
        pu = median(share(p, "null|PUlocationID") for p in ps)
        do = [share(p, "null|DOlocationID") for p in ps]
        af = median(share(p, "null|Affiliated_base_number") for p in ps)
        du = [share(p, "unique|duplicates") for p in ps]
        L.append(
            f"  {y}: pick-up zone empty {pu:5.1f}; drop-off zone empty {min(do):4.1f} to {max(do):4.1f}; "
            f"affiliated base empty {af:5.2f}; exact duplicates {min(du):.3f} to {max(du):.3f}"
        )
    L.append("  types of PUlocationID, DOlocationID and SR_Flag, from the month each combination starts:")
    prev = None
    for m in sorted(F):
        t = dict(F[m]["schema"])
        k = (t.get("PUlocationID"), t.get("DOlocationID"), t.get("SR_Flag"))
        if k != prev:
            L.append(f"    {m}: {', '.join(str(x) for x in k)}")
            prev = k
    L.append("")


def fhvhv_section(P: dict, L: list[str]) -> None:
    H = P["fhvhv"]
    L.append(
        "High-volume FHV by month, % of rows: request after pick-up; access_a_ride_flag and wav_match_flag "
        "outside Y/N; originating base empty; negative base fare; airport fee outside {0, 2.5, 5}, "
        "with its most common values (rows)"
    )
    for m in sorted(H):
        p = H[m]
        off = p.get("offending", {}).get("airport_fee", [])
        top = ", ".join(f"{v} x {n:,}" for v, n in off[:3])
        L.append(
            f"  {m}: {share(p, 'order|request_datetime<=pickup_datetime'):5.2f} "
            f"{share(p, 'code|access_a_ride_flag'):5.1f} {share(p, 'code|wav_match_flag'):5.1f} "
            f"{share(p, 'null|originating_base_num'):5.1f} {share(p, 'neg|base_passenger_fare'):6.3f} "
            f"{share(p, 'code|airport_fee'):5.2f} [{top}]"
        )
    L.append("")


def report_section(P: dict, report: dict, L: list[str]) -> None:
    rows = C.rows_from_profiles(P)
    L.append("E2. File rows / report count - 1, %, every month")
    for f in C.REPORT_CLASSES:
        ms = [m for m in sorted(rows[f]) if m in report.get(f, {})]
        cells = [f"{m} {100 * (rows[f][m] / report[f][m] - 1):+.2f}" for m in ms]
        L.append(f"  {f}:")
        for i in range(0, len(cells), 6):
            L.append("    " + "; ".join(cells[i : i + 6]))
        # The months worth naming: flagged by the outside count against its history, or more than 1.5% off
        # in a feed whose file and report usually agree (median |difference| under 0.5% from FIRST_EVAL).
        flagged = {m for m, v in C.scores(rows[f], report[f])["outside count"].items() if v > C.K}
        typical = median(abs(rows[f][m] / report[f][m] - 1) for m in ms if m >= C.FIRST_EVAL)
        if typical < 0.005:
            flagged |= {m for m in ms if m >= C.FIRST_EVAL and abs(rows[f][m] / report[f][m] - 1) > 0.015}
        L.append(f"    median |difference| from {C.FIRST_EVAL}: {100 * typical:.3f}%")
        for m, n, r in [(m, rows[f][m], report[f][m]) for m in sorted(flagged)]:
            L.append(f"    {m}: file {n:,} vs report {r:,.0f}, difference {n - r:+,.0f}")
    L.append("")


def noise_section(P: dict, report: dict, L: list[str]) -> None:
    rows = C.rows_from_profiles(P)
    L.append(
        "Noise each history-based detector sees on clean deliveries from "
        + C.FIRST_EVAL
        + ": median robust scale (1.4826 x MAD of the 12 previous values, no floor) and the 95th "
        "percentile of |value - median of the 12 previous|, both in % (log units x 100)"
    )
    for f in C.REPORT_CLASSES:
        s = C.series(rows[f], report[f])
        parts = []
        for name, x in s.items():
            ms = sorted(x)
            scale, dev = [], []
            for i, m in enumerate(ms):
                if m < C.FIRST_EVAL or i < C.WINDOW:
                    continue
                h = np.asarray([x[ms[j]] for j in range(i - C.WINDOW, i)])
                med = float(np.median(h))
                scale.append(1.4826 * float(np.median(np.abs(h - med))))
                dev.append(abs(x[m] - med))
            parts.append(f"{name} {100 * median(scale):.3f} / {100 * float(np.percentile(dev, 95)):.3f}")
        L.append(f"  {f}: " + "; ".join(parts))
    L.append("")


def alarms_section(P: dict, report: dict, L: list[str]) -> None:
    rows = C.rows_from_profiles(P)
    L.append(f"Real deliveries flagged from {C.FIRST_EVAL} (no loss planted)")
    for f in C.REPORT_CLASSES:
        sc = C.scores(rows[f], report[f])
        for name in C.DETECTORS:
            hits = [f"{m} ({v:.2f})" for m, v in sorted(sc[name].items()) if v > C.threshold(name)]
            L.append(f"  {f}, {name}: " + (", ".join(hits) if hits else "none"))
    L.append("  (|z| for the history-based detectors, firing above 5; the fixed check is |log ratio| / log 1.01)")
    L.append("")


def part3_section(report: dict, L: list[str]) -> None:
    if not PART3.exists():
        L.append("Part 3 profiles not found; yellow consistency checks skipped.")
        return
    prof = {}
    with gzip.open(PART3, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            prof[r["month"]] = r["profile"]
    L.append("Yellow consistency checks with Part 3's profiles")
    for m in ("2022-10", "2022-11", "2022-12"):
        v = prof[m]["categories"]["vendorid"]
        L.append(f"  {m}: vendor 1 {100 * v['1'] / prof[m]['n_rows']:.2f}% of {prof[m]['n_rows']:,} rows")
    m = "2022-11"
    n, r = prof[m]["n_rows"], report["yellow"][m]
    v1 = prof[m]["categories"]["vendorid"]["1"]
    L.append(
        f"  {m}: report {r:,.0f}, gap {r - n:,.0f}; vendor 1 share if every missing trip were vendor 1's "
        f"{100 * (v1 + r - n) / r:.2f}%"
    )
    m = "2023-09"
    days = {d: c for d, c in prof[m]["by_day"].items() if d.startswith(m)}
    med = median(days.values())
    short = {d: c for d, c in days.items() if c < 0.5 * med}
    lack = sum(med - c for c in short.values())
    n, r = prof[m]["n_rows"], report["yellow"][m]
    L.append(
        f"  {m}: median day {med:,.0f}; days under half of it: "
        + ", ".join(f"{d[-2:]} {c:,}" for d, c in sorted(short.items()))
        + f"; they lack {lack:,.0f} trips against a median day; report gap {r - n:,.0f} "
        f"({100 * (r - n) / lack:.0f}% of what they lack)"
    )
    L.append("")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profiles", type=Path, default=Path("profiles"))
    ap.add_argument("--sources", type=Path, default=Path("sources"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    P = read_profiles(a.profiles)
    report = C.read_report(a.sources / "data_reports_monthly.csv")
    L = ["Descriptive numbers quoted in the article (not part of the pre-registered analysis).", ""]
    rows_section(P, L)
    rules_section(a.results, L)
    fhv_section(P, L)
    fhvhv_section(P, L)
    report_section(P, report, L)
    noise_section(P, report, L)
    alarms_section(P, report, L)
    part3_section(report, L)
    a.out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(L).rstrip() + "\n"
    (a.out / "details.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
