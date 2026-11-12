"""Test the prediction of PLAN.md on four new pairs of counts, and recompute it on Part 4's four pairs.

    python evaluate.py --counts counts --sources sources --part4 ../data-tests-limits --out results

New pairs (the test): Citi Bike trip files against the operator's monthly reports and against NYC DOT's
monthly table; Chicago rideshare trips against the trips summed over the vehicle records and over the driver
records. Part 4's pairs (calibration, already seen): the four NYC TLC feeds against the TLC monthly report.

Writes results/summary.txt, results/curves.csv and results/ratios.csv.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import gzip
import json
import math
from pathlib import Path

from feed import detect as D

NEW = ("citibike / operator reports", "citibike / DOT table", "chicago / vehicles", "chicago / drivers")
TLC_CLASSES = {
    "yellow": ("Yellow",),
    "green": ("Green",),
    "fhv": ("FHV - Black Car", "FHV - Livery", "FHV - Lux Limo"),
    "fhvhv": ("FHV - High Volume",),
}
TLC_FIRST_EVAL = "2021-05"  # Part 4's evaluation window
CITIBIKE_FIRST = "2013-06"  # the system opened on 27 May 2013


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def citibike_delivered(counts: Path) -> tuple[dict[str, int], dict]:
    """Rows by month of trip start. For a year with monthly files only those are used, else its archive."""
    by: dict[str, int] = {}
    info = {"unparsed": 0, "years": {}}
    for f in sorted((counts / "citibike").glob("*.json")):
        y = json.loads(f.read_text(encoding="utf-8"))
        keys = y["keys"]
        monthly = {k: v for k, v in keys.items() if v["monthly"]}
        use = monthly if monthly else keys
        info["years"][y["year"]] = {"keys": sorted(use), "left_out": sorted(set(keys) - set(use))}
        for v in use.values():
            info["unparsed"] += v["unparsed"]
            for m, n in v["months"].items():
                by[m] = by.get(m, 0) + n
    return {m: n for m, n in sorted(by.items()) if m >= CITIBIKE_FIRST}, info


def outside_from(rows: list[dict], col: str = "trips") -> dict[str, float]:
    return {r["month"]: float(r[col]) for r in rows if r.get(col) not in (None, "")}


def tlc_pairs(part4: Path) -> dict[str, tuple[dict[str, int], dict[str, float]]]:
    rows: dict[str, dict[str, int]] = {}
    for f in sorted((part4 / "profiles").glob("*.jsonl.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                p = json.loads(line)
                rows.setdefault(p["feed"], {})[p["month"]] = int(p["n_rows"])
    by: dict[tuple[str, str], float] = {}
    for r in read_csv(part4 / "sources" / "data_reports_monthly.csv"):
        m = r["Year/Month"][:7]
        tpd = (r.get("Trips Per Day") or "").replace(",", "").strip()
        if tpd:
            by[(r["License Class"].strip(), m)] = float(tpd) * calendar.monthrange(*map(int, m.split("-")))[1]
    out = {}
    for feed, classes in TLC_CLASSES.items():
        rep = {}
        for m in {m for (_, m) in by}:
            vals = [by.get((c, m)) for c in classes]
            if all(v is not None for v in vals):
                rep[m] = sum(vals)
        out[f"TLC {feed} / monthly report"] = (rows.get(feed, {}), rep)
    return out


def analyse(r: dict[str, float], first: str | None) -> dict:
    cur = D.curve(r, first)
    obs = D.observed_l90(cur)
    pred = D.predicted(r, first)
    clean = D.scores(r, 0.0, first)
    return {
        "months": len(clean),
        "first": min(clean) if clean else None,
        "last": max(clean) if clean else None,
        "p95_dev": D.deviation_p95(r, first),
        "median_scale": D.median_scale(r, first),
        "pred": pred,
        "obs": obs,
        "curve": cur,
        "threshold": D.matched_threshold(clean),
        "h1": D.within_factor_two(pred["p95_rule"], obs),
        "err_p95": D.log_error(pred["p95_rule"], obs),
        "err_gauss": D.log_error(pred["gaussian_rule"], obs),
    }


def pct(x: float | None) -> str:
    return "not reached by 50%" if x is None else f"{100 * x:.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--counts", type=Path, default=Path("counts"))
    ap.add_argument("--sources", type=Path, default=Path("sources"))
    ap.add_argument("--part4", type=Path, default=Path("../data-tests-limits"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    cb, cb_info = citibike_delivered(a.counts)
    ops = read_csv(a.sources / "citibike_operator_reports.csv")
    chi = {r["month"]: int(r["trips"]) for r in read_csv(a.sources / "chicago_trips.csv")}
    pairs: dict[str, tuple[dict[str, int], dict[str, float], str | None]] = {
        "citibike / operator reports": (cb, outside_from(ops), None),
        "citibike / DOT table": (cb, outside_from(read_csv(a.sources / "citibike_dot.csv")), None),
        "chicago / vehicles": (chi, outside_from(read_csv(a.sources / "chicago_vehicles.csv")), None),
        "chicago / drivers": (chi, outside_from(read_csv(a.sources / "chicago_drivers.csv")), None),
    }
    for name, (d, o) in tlc_pairs(a.part4).items():
        pairs[name] = (d, o, TLC_FIRST_EVAL)

    res, ratio_rows, curve_rows = {}, [], []
    for name, (d, o, first) in pairs.items():
        r = D.ratio_series(d, o)
        res[name] = x = analyse(r, first)
        clean = D.scores(r, 0.0, first)
        for m in sorted(r):
            ratio_rows.append([name, m, d[m], round(o[m]), f"{r[m]:.6f}", f"{clean[m]:.3f}" if m in clean else ""])
        for p, v in x["curve"].items():
            curve_rows.append([name, p, f"{v:.4f}"])

    L = [
        "Does the agreement of two counts on clean months predict the smallest random loss they can catch?",
        "Settings and hypotheses: PLAN.md.",
        "",
    ]
    L.append(f"Citi Bike rows whose start time has no recognisable date (left out): {cb_info['unparsed']:,}")
    failed = [r["month"] for r in ops if not r.get("trips")]
    L.append(f"Operator reports without a usable total (left out): {len(failed)} {' '.join(failed)}")
    L.append("")
    for name, x in res.items():
        tag = "TEST" if name in NEW else "calibration (Part 4, already seen)"
        L.append(f"{name}  [{tag}]")
        L.append(
            f"  evaluated months {x['months']} ({x['first']} to {x['last']}); "
            f"matched threshold |z| > {x['threshold']:.3f}"
        )
        L.append(
            f"  95th percentile of |r - median of previous 12|: {100 * x['p95_dev']:.3f}% (log x 100); "
            f"median robust scale {100 * x['median_scale']:.3f}%"
        )
        L.append(
            f"  predicted smallest loss caught 90%: p95 rule {pct(x['pred']['p95_rule'])}; "
            f"Gaussian rule {pct(x['pred']['gaussian_rule'])}"
        )
        L.append(f"  observed: {pct(x['obs'])}; within a factor of two of the p95 rule: {'yes' if x['h1'] else 'no'}")
        L.append(f"  |log(observed / predicted)|: p95 rule {x['err_p95']:.3f}; Gaussian rule {x['err_gauss']:.3f}")
        L.append("  detection: " + "; ".join(f"{100 * p:.2g}% {100 * v:.0f}" for p, v in x["curve"].items()))
        L.append("")
    h1 = all(res[n]["h1"] for n in NEW)
    better = sum(1 for n in NEW if res[n]["err_p95"] < res[n]["err_gauss"])
    L.append(
        "H1 (observed within a factor of two of the p95 rule in all four new pairs): "
        + ("supported" if h1 else "not supported")
    )
    L.append(f"    pairs within: {sum(res[n]['h1'] for n in NEW)} of 4")
    L.append(
        f"H2 (the p95 rule closer than the Gaussian rule in at least three of four new pairs): "
        f"{'supported' if better >= 3 else 'not supported'} ({better} of 4)"
    )
    text = "\n".join(L) + "\n"
    (a.out / "summary.txt").write_text(text, encoding="utf-8")
    with open(a.out / "ratios.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "month", "delivered", "outside", "log_ratio", "clean_abs_z"])
        w.writerows(ratio_rows)
    with open(a.out / "curves.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "loss", "detected"])
        w.writerows(curve_rows)
    (a.out / "citibike_keys_used.json").write_text(json.dumps(cb_info, indent=1), encoding="utf-8")
    print(text)
    if not math.isfinite(sum(x["p95_dev"] for x in res.values())):
        raise SystemExit("a pair has no evaluated months")


if __name__ == "__main__":
    main()
