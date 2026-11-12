"""The checks, written against profiles (``feed/profile.py``).

Two families, each check returning a ``Result`` (fired or not, and the number it fired on).

**Table checks** look at one delivery on its own, against rules written in advance from the
TLC data dictionary: the schema, required columns, ranges, code lists, the month window,
uniqueness, reconciliation of the total, a volume floor. Each comes in a *strict* variant (any
violation fails) and a *tolerant* one (a violation share above a fixed limit fails), because
that is the choice every team writing such rules makes.

**Change checks** compare the delivery with the same measurements on the deliveries before it:
for each metric, a robust z-score against the trailing ``window`` months (median and MAD, with a
floor on the MAD so that a perfectly stable metric does not turn every wiggle into an alarm).
A check fires when its largest |z| exceeds ``k``; ``k`` is the false-alarm budget.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- the contract (TLC dictionary, 18 March 2025)

INT, FLOAT, TS, STR = "integer", "float", "timestamp", "string"
CONTRACT = {
    "VendorID": INT, "tpep_pickup_datetime": TS, "tpep_dropoff_datetime": TS, "passenger_count": INT,
    "trip_distance": FLOAT, "RatecodeID": INT, "store_and_fwd_flag": STR, "PULocationID": INT, "DOLocationID": INT,
    "payment_type": INT, "fare_amount": FLOAT, "extra": FLOAT, "mta_tax": FLOAT, "tip_amount": FLOAT,
    "tolls_amount": FLOAT, "improvement_surcharge": FLOAT, "total_amount": FLOAT, "congestion_surcharge": FLOAT,
    "airport_fee": FLOAT, "cbd_congestion_fee": FLOAT,
}
OPTIONAL_BEFORE = {"cbd_congestion_fee": "2025-01"}   # column exists from this month on
REQUIRED_NOT_NULL = ("vendorid", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
                     "trip_distance", "ratecodeid", "pulocationid", "dolocationid", "payment_type", "fare_amount",
                     "total_amount")
DOMAINS = {"vendorid": {"1", "2", "6", "7"}, "ratecodeid": {"1", "2", "3", "4", "5", "6", "99"},
           "payment_type": {"0", "1", "2", "3", "4", "5", "6"}, "store_and_fwd_flag": {"Y", "N"}}
NONNEG = ("trip_distance", "fare_amount", "tip_amount", "tolls_amount", "total_amount")
TOLERANCE = 0.01            # tolerant variants: more than 1% of rows violating
VOLUME_FLOOR = 1_000_000    # a static floor of the kind written once and forgotten


def kind(duck_type: str) -> str:
    t = duck_type.upper()
    if t.startswith("TIMESTAMP") or t == "DATE":
        return TS
    if t in ("VARCHAR", "BLOB") or t.startswith("VARCHAR"):
        return STR
    if t in ("DOUBLE", "FLOAT", "REAL") or t.startswith("DECIMAL"):
        return FLOAT
    if t in ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT"):
        return INT
    return t.lower()


@dataclass(frozen=True)
class Result:
    fired: bool
    value: float
    detail: str = ""


def _null_share(p, col):
    n = p["n_rows"] or 1
    if col in p["numeric"]:
        return p["numeric"][col]["null"] / n
    if col in p.get("categories", {}):
        return p["categories"][col].get("null", 0) / n
    if col.startswith("tpep_pickup"):
        return p["time"]["null_pickup"] / n
    return 0.0


def _present(p) -> dict[str, str]:
    return {name.lower(): typ for name, typ in p["schema"]}


# ---------------------------------------------------------------- table checks


def t_schema_strict(p) -> Result:
    have = {name: kind(t) for name, t in p["schema"]}
    want = {k: v for k, v in CONTRACT.items() if p["month"] >= OPTIONAL_BEFORE.get(k, "0000")}
    bad = [k for k, v in want.items() if have.get(k) != v] + [k for k in have if k not in CONTRACT]
    return Result(bool(bad), len(bad), ", ".join(sorted(bad)))


def t_schema_tolerant(p) -> Result:
    have = {k: kind(t) for k, t in _present(p).items()}
    numeric = {INT, FLOAT}
    bad = []
    for k, v in CONTRACT.items():
        if p["month"] < OPTIONAL_BEFORE.get(k, "0000"):
            continue
        h = have.get(k.lower())
        if h is None or (h != v and not (h in numeric and v in numeric)):
            bad.append(k)
    return Result(bool(bad), len(bad), ", ".join(bad))


def t_not_null(strict: bool):
    def check(p) -> Result:
        shares = {c: _null_share(p, c) for c in REQUIRED_NOT_NULL if c in _present(p)}
        worst = max(shares, key=shares.get)
        lim = 0.0 if strict else TOLERANCE
        return Result(shares[worst] > lim, shares[worst], worst)
    return check


def _violations(p) -> dict[str, float]:
    n = p["n_rows"] or 1
    v = {f"{c}<0": p["numeric"][c]["neg"] / n for c in NONNEG if c in p["numeric"]}
    pc = p["categories"].get("passenger_count", {})
    v["passenger_count>9"] = sum(cnt for val, cnt in pc.items() if val != "null" and int(val) > 9) / n
    v["dropoff<pickup"] = p["time"]["dropoff_before_pickup"] / n
    v["duration>24h"] = p["time"]["over_24h"] / n
    return v


def t_ranges(strict: bool):
    def check(p) -> Result:
        v = _violations(p)
        worst = max(v, key=v.get)
        lim = 0.0 if strict else TOLERANCE
        return Result(v[worst] > lim, v[worst], worst)
    return check


def t_domains(strict: bool):
    def check(p) -> Result:
        n = p["n_rows"] or 1
        out = {}
        for col, ok in DOMAINS.items():
            cats = p["categories"].get(col, {})
            out[col] = sum(cnt for val, cnt in cats.items() if val != "null" and val not in ok) / n
        worst = max(out, key=out.get)
        lim = 0.0 if strict else TOLERANCE
        return Result(out[worst] > lim, out[worst], worst)
    return check


def t_window(strict: bool):
    def check(p) -> Result:
        s = (p["time"]["before_month"] + p["time"]["after_month"]) / (p["n_rows"] or 1)
        return Result(s > (0.0 if strict else TOLERANCE), s)
    return check


def t_unique(strict: bool):
    def check(p) -> Result:
        s = p["duplicates"] / (p["n_rows"] or 1)
        return Result(s > (0.0 if strict else TOLERANCE), s)
    return check


def t_reconcile(strict: bool):
    def check(p) -> Result:
        s = p.get("recon_mismatch", 0) / (p["n_rows"] or 1)
        return Result(s > (0.0 if strict else TOLERANCE), s)
    return check


def t_volume(p) -> Result:
    return Result(p["n_rows"] < VOLUME_FLOOR, p["n_rows"])


TABLE: dict[str, Callable] = {
    "schema (strict)": t_schema_strict, "schema (tolerant)": t_schema_tolerant,
    "not null (strict)": t_not_null(True), "not null (1%)": t_not_null(False),
    "ranges (strict)": t_ranges(True), "ranges (1%)": t_ranges(False),
    "code lists (strict)": t_domains(True), "code lists (1%)": t_domains(False),
    "month window (strict)": t_window(True), "month window (1%)": t_window(False),
    "unique rows (strict)": t_unique(True), "unique rows (1%)": t_unique(False),
    "reconciliation (strict)": t_reconcile(True), "reconciliation (1%)": t_reconcile(False),
    "volume floor": t_volume,
}

# ---------------------------------------------------------------- metrics for the change checks


def _share(d: dict, n: int) -> dict[str, float]:
    return {k: v / n for k, v in d.items()} if n else {}


def _tv(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def days_in_month(month: str) -> list[str]:
    y, m = map(int, month.split("-"))
    leap = y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
    nd = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
    return [f"{month}-{d:02d}" for d in range(1, nd + 1)]


def scalar_metrics(p) -> dict[str, dict[str, float]]:
    """Metric groups: group -> {metric name: value}. Every value is a number per delivery."""
    n = p["n_rows"] or 1
    num = p["numeric"]
    g: dict[str, dict[str, float]] = {}
    g["volume"] = {"log rows": float(np.log(max(p["n_rows"], 1)))}
    counts = np.array([p.get("by_day", {}).get(d, 0) for d in days_in_month(p["month"])], float)
    med = np.median(counts) if counts.size else 1.0
    g["daily completeness"] = {"lowest day / median day": float(counts.min() / med) if med else 0.0,
                               "highest day / median day": float(counts.max() / med) if med else 0.0}
    g["null shares"] = {c: _null_share(p, c) for c in ("vendorid", "passenger_count", "ratecodeid", "payment_type",
                                                       "trip_distance", "fare_amount", "tip_amount", "total_amount",
                                                       "congestion_surcharge", "airport_fee", "pulocationid")
                        if c in _present(p)}
    viol = _violations(p)
    viol["outside month"] = (p["time"]["before_month"] + p["time"]["after_month"]) / n
    viol["zero distance"] = num["trip_distance"]["zero"] / n if "trip_distance" in num else 0.0
    viol["zero passengers"] = p["categories"].get("passenger_count", {}).get("0", 0) / n
    g["rule-violation rates"] = viol
    d = {}
    for col, qs in (("trip_distance", ("0.25", "0.5", "0.75", "0.95")), ("fare_amount", ("0.5", "0.95")),
                    ("total_amount", ("0.5",)), ("tip_amount", ("0.5", "0.95"))):
        if col in num:
            for q in qs:
                d[f"{col} q{q}"] = num[col]["q"][q]
    if p.get("fare_per_mile"):
        d["fare per mile q0.5"] = p["fare_per_mile"]["q50"]
    d["duration q0.5"] = p["time"]["dur_q50"]
    if p.get("card_tips"):
        d["card tip share"] = p["card_tips"]["mean_share"]
        d["card trips tipped"] = p["card_tips"]["share_tipped"]
    g["value distributions"] = {k: float(v) for k, v in d.items() if v is not None}
    g["duplicate rows"] = {"duplicate share": p["duplicates"] / n}
    g["reconciliation"] = {"mismatch share": p.get("recon_mismatch", 0) / n}
    return g


def mixes(p) -> dict[str, dict[str, float]]:
    """Distributions compared as wholes: category mixes, hour of day, pickup zones."""
    n = p["n_rows"] or 1
    out = {f"mix: {c}": _share(p["categories"].get(c, {}), n)
           for c in ("payment_type", "vendorid", "ratecodeid", "passenger_count")}
    hours = p.get("by_hour", {})
    out["hour of day"] = _share(hours, sum(hours.values()) or 1)
    zones = p.get("pu_zone", {})
    out["pickup zones"] = _share(zones, sum(zones.values()) or 1)
    return out


# ---------------------------------------------------------------- change checks


def robust_z(x: float, hist: list[float], rel_floor: float = 0.02, abs_floor: float = 1e-4) -> float:
    h = np.asarray(hist, float)
    med = float(np.median(h))
    mad = 1.4826 * float(np.median(np.abs(h - med)))
    scale = max(mad, rel_floor * abs(med), abs_floor)
    return (x - med) / scale


# floors on the scale of each metric group (relative to the median, absolute): a metric that has
# not moved in a year must still change by this much to count; fixed before the real data were seen
FLOORS = {"volume": (0.0, 0.01), "daily completeness": (0.0, 0.02), "null shares": (0.05, 2e-4),
          "rule-violation rates": (0.05, 2e-4), "value distributions": (0.01, 1e-3), "duplicate rows": (0.05, 1e-5),
          "reconciliation": (0.05, 2e-4)}


def change_scores(p, history: list[dict]) -> dict[str, tuple[float, str]]:
    """Largest |z| of each change check, and the metric it came from."""
    out: dict[str, tuple[float, str]] = {}
    cur = scalar_metrics(p)
    past = [scalar_metrics(h) for h in history]
    for group, metrics in cur.items():
        best = (0.0, "")
        rel, ab = FLOORS[group]
        for name, x in metrics.items():
            hist = [h[group][name] for h in past if group in h and name in h[group]]
            if len(hist) < 6:
                continue
            z = abs(robust_z(x, hist, rel_floor=rel, abs_floor=ab))
            if z > best[0]:
                best = (z, name)
        out[group] = best
    # whole distributions: distance of this month from the median of the history, against the
    # same distance for each past month (leave-one-out), as a robust z
    cm, pm = mixes(p), [mixes(h) for h in history]
    for name, cur_mix in cm.items():
        hist = [m[name] for m in pm if m.get(name)]
        if len(hist) < 6 or not cur_mix:
            continue
        keys = set(cur_mix).union(*hist)
        center = {k: float(np.median([h.get(k, 0.0) for h in hist])) for k in keys}
        d_now = _tv(cur_mix, center)
        d_hist = []
        for i, h in enumerate(hist):
            others = hist[:i] + hist[i + 1:]
            c_i = {k: float(np.median([o.get(k, 0.0) for o in others])) for k in keys}
            d_hist.append(_tv(h, c_i))
        out[name] = (abs(robust_z(d_now, d_hist, rel_floor=0.25, abs_floor=0.002)), f"TV {d_now:.3f}")
    # schema: did names or types change since the previous delivery?
    if history:
        prev = {n: kind(t) for n, t in history[-1]["schema"]}
        now = {n: kind(t) for n, t in p["schema"]}
        diff = sorted(set(prev.items()) ^ set(now.items()))
        out["schema change"] = (float("inf") if diff else 0.0, "; ".join(f"{a}:{b}" for a, b in diff))
    return out


CHANGE = ("volume", "daily completeness", "null shares", "rule-violation rates", "value distributions",
          "duplicate rows", "reconciliation", "mix: payment_type", "mix: vendorid", "mix: ratecodeid",
          "mix: passenger_count", "hour of day", "pickup zones", "schema change")


def run_table(p) -> dict[str, Result]:
    return {name: f(p) for name, f in TABLE.items()}


def run_change(p, history: list[dict], k: float) -> dict[str, Result]:
    s = change_scores(p, history)
    return {name: Result(s.get(name, (0.0, ""))[0] > k, s.get(name, (0.0, ""))[0], s.get(name, (0.0, ""))[1])
            for name in CHANGE}
