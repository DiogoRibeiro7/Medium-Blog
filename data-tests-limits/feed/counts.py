"""Random losses of rows, and what can see them: the delivery's own history, or a count from outside it.

The delivered count of a month is the number of rows in its file (the profile's count, which is the
row count in the file's Parquet footer). The outside count is the
TLC monthly data report's average trips per day, for the licence classes of the feed, times the days of the
month. A random loss of a fraction p of the rows multiplies the delivered count by (1 - p) and leaves every
share, mix and quantile unchanged in expectation, so counts are all that carry it; the binomial noise of the
loss is ignored (relative sd sqrt(p / ((1 - p) n)), under 0.3% even for 50,000 rows and a 20% loss).

Detectors, each a robust z-score against the 12 previous clean deliveries (median and 1.4826 x median
absolute deviation, scale floored at FLOOR), firing at |z| > K:

- volume: log rows (the Part 3 check);
- seasonal: log rows minus a seasonal forecast, log rows 12 months earlier plus the median year-on-year
  change of the 3 previous months;
- outside count: log rows minus log of the report's count;

and one table check with an outside number, no history: the file and the report differ by more than 1%.
"""

from __future__ import annotations

import calendar
import csv
import io
import math
from pathlib import Path

import numpy as np

from feed.contracts import CONTRACTS, GREEN_REPORT

WINDOW, K, FLOOR, FIXED_TOL = 12, 5.0, 0.005, 0.01
LOSSES = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20)
BUDGET = 0.05  # matched comparison: the threshold at which a detector fires on at most 5% of clean deliveries
FIRST_EVAL = "2021-05"
DETECTORS = ("volume", "seasonal", "outside count", "outside count, fixed 1%")
REPORT_CLASSES = {**{f: c.report_classes for f, c in CONTRACTS.items()}, "green": GREEN_REPORT}


def days(month: str) -> int:
    y, m = map(int, month.split("-"))
    return calendar.monthrange(y, m)[1]


def rows_from_profiles(profiles: dict[str, dict[str, dict]]) -> dict[str, dict[str, int]]:
    """Delivered rows by feed and month: count(*) over the file, the same number as its Parquet footer."""
    return {feed: {m: int(p["n_rows"]) for m, p in ps.items()} for feed, ps in profiles.items()}


def num(x: str) -> float:
    x = (x or "").replace(",", "").replace("$", "").strip()
    return float(x) if x else math.nan


def read_report(path: Path) -> dict[str, dict[str, float]]:
    """Trips per month by feed: the sum over the feed's licence classes of trips per day times days."""
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
    by: dict[tuple[str, str], float] = {}
    for r in rows:
        month = r["Year/Month"][:7]
        by[(r["License Class"].strip(), month)] = num(r["Trips Per Day"]) * days(month)
    out: dict[str, dict[str, float]] = {}
    for feed, classes in REPORT_CLASSES.items():
        months = {m for (cl, m) in by if cl in classes}
        for m in months:
            vals = [by.get((cl, m), math.nan) for cl in classes]
            if all(not math.isnan(v) for v in vals):
                out.setdefault(feed, {})[m] = float(sum(vals))
    return out


def robust_z(x: float, hist: list[float]) -> float:
    h = np.asarray(hist, float)
    med = float(np.median(h))
    scale = max(1.4826 * float(np.median(np.abs(h - med))), FLOOR)
    return (x - med) / scale


def series(rows: dict[str, int], report: dict[str, float]) -> dict[str, dict[str, float]]:
    """The quantity each history-based detector watches, by month (clean deliveries)."""
    ms = sorted(rows)
    lr = {m: math.log(rows[m]) for m in ms}
    seasonal = {}
    for i, m in enumerate(ms):
        if i < 15 or ms[i - 15] != shift(m, -15):
            continue
        yoy = [lr[ms[i - j]] - lr[ms[i - j - 12]] for j in (1, 2, 3)]
        seasonal[m] = lr[m] - (lr[ms[i - 12]] + float(np.median(yoy)))
    outside = {m: lr[m] - math.log(report[m]) for m in ms if m in report and report[m] > 0}
    return {"volume": lr, "seasonal": seasonal, "outside count": outside}


def shift(month: str, k: int) -> str:
    y, m = map(int, month.split("-"))
    t = y * 12 + (m - 1) + k
    return f"{t // 12:04d}-{t % 12 + 1:02d}"


def scores(rows: dict[str, int], report: dict[str, float], loss: float = 0.0) -> dict[str, dict[str, float]]:
    """|z| of each detector on each evaluated delivery, with a loss of ``loss`` applied to that delivery only.

    The fixed check returns |log rows - log report| / log(1.01), so it fires above 1, not above K."""
    s = series(rows, report)
    d = math.log(1 - loss)
    out: dict[str, dict[str, float]] = {k: {} for k in DETECTORS}
    for name in ("volume", "seasonal", "outside count"):
        x = s[name]
        ms = sorted(x)
        for i, m in enumerate(ms):
            if m < FIRST_EVAL or i < WINDOW:
                continue
            hist = [x[ms[j]] for j in range(i - WINDOW, i)]
            out[name][m] = abs(robust_z(x[m] + d, hist))
    for m, v in s["outside count"].items():
        if m >= FIRST_EVAL:
            out["outside count, fixed 1%"][m] = abs(v + d) / math.log(1 + FIXED_TOL)
    return out


def threshold(name: str) -> float:
    return 1.0 if name == "outside count, fixed 1%" else K


def matched_threshold(clean: dict[str, float], budget: float = BUDGET) -> float:
    """Smallest threshold at which the detector fires on at most ``budget`` of the clean deliveries."""
    v = np.sort(np.asarray(list(clean.values()), float))
    if len(v) == 0:
        return math.inf
    allowed = int(math.floor(budget * len(v)))
    return float(v[len(v) - allowed - 1]) if allowed < len(v) else 0.0
