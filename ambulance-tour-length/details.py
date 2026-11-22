"""Descriptive analyses added after the registered results (21 November 2026); not part of the registered test.

python details.py --sources sources --results results --out results/details.txt

Uses the binding hours saved by evaluate.py (results/binding.npz) and re-solves a few rosters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from staff import comb, data, schedule

WEEK = data.WEEK


def comb_value_path(need: np.ndarray, L: int) -> float:
    """Largest sum of need over hours of a path (not a cycle) pairwise at least L apart."""
    need = np.asarray(need, dtype=np.float64)
    dp = np.zeros(need.size)
    for i in range(need.size):
        take = need[i] + (dp[i - L] if i >= L else 0.0)
        dp[i] = max(dp[i - 1] if i >= 1 else 0.0, take)
    return float(dp[-1])


def weekly_phase(mask_week: np.ndarray, L: int) -> int:
    """The phase phi (clock hour mod L) holding most of a week's binding hours."""
    hours = np.flatnonzero(mask_week) % 24
    counts = np.bincount(hours % L, minlength=L)
    return int(np.argmax(counts))


def circ(a: int, b: int, L: int) -> int:
    d = abs(a - b) % L
    return min(d, L - d)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources", type=Path, default=Path("sources"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/details.txt"))
    a = ap.parse_args()
    needs = data.requirements(a.sources)
    masks = np.load(a.results / "binding.npz")
    cells = pd.read_csv(a.results / "cells.csv")
    out: list[str] = []
    js: dict = {}

    out.append(
        "1. Week-by-week phase of the binding chain, L in 6, 8, 12: share of weeks at the comb-rule phase of the"
    )
    out.append("   mean profile, and within one hour of it; with the dataset's mean requirement (pool size)")
    js["weekly_phase"] = {}
    for d, nd in needs.items():
        m = nd.reshape(-1, 24).mean(axis=0)
        parts = []
        for L in data.DIVISORS:
            mk = masks[f"{d}__{L}"]
            target = comb.phase_by_comb(m, L)
            ph = np.array([weekly_phase(mk[w], L) for w in range(data.WEEKS)])
            at = float((ph == target).mean())
            near = float(np.mean([circ(p, target, L) <= 1 for p in ph]))
            js["weekly_phase"][f"{d}__{L}"] = {"at": at, "within_1h": near}
            parts.append(f"L {L}: at {at:.2f}, within 1 h {near:.2f}")
        out.append(f"   {d:18s} mean need {nd.mean():6.1f}; " + "; ".join(parts))
    out.append("")

    out.append("2. Comb sums of the mean daily profile by phase, as a share of the best phase (L = 8 and 12)")
    for d in ("ems_city", "nyc311"):
        m = needs[d].reshape(-1, 24).mean(axis=0)
        for L in (8, 12):
            sums = np.array([m[phi::L].sum() for phi in range(L)])
            out.append(
                f"   {d:10s} L {L:2d}: " + ", ".join(f"{phi}: {s / sums.max():.3f}" for phi, s in enumerate(sums))
            )
    out.append("")

    out.append("3. Harmonic view: peak hours of the mean profile's harmonic with period L (n = 24 / L), and its teeth")
    for d in ("ems_city", "nyc311"):
        m = needs[d].reshape(-1, 24).mean(axis=0)
        F = np.fft.rfft(m) / 24
        for L in data.DIVISORS:
            n = 24 // L
            peak = (-np.angle(F[n]) / (2 * np.pi) * 24 / n) % (24 / n)
            teeth = sorted((peak + k * L) % 24 for k in range(n))
            amp = 2 * abs(F[n]) / m.mean()
            out.append(
                f"   {d:10s} L {L:2d}: harmonic {n}, amplitude {amp:.3f} of the mean, peaks at "
                + ", ".join(f"{t:.1f}" for t in teeth)
                + f"; comb-rule phase {comb.phase_by_comb(m, L)}"
            )
    out.append("")

    out.append(
        "4. Where the idle hours sit, EMS citywide and 311: mean ambulances (agents) on duty above the requirement,"
    )
    out.append("   by clock hour, over the 52 weeks, for 8- and 12-hour tours")
    js["idle_by_hour"] = {}
    for d in ("ems_city", "nyc311"):
        for L in (8, 12):
            idle = np.zeros(WEEK)
            for w in range(data.WEEKS):
                need = needs[d][w].astype(np.float64)
                _, x = schedule.roster(need, L)
                idle += schedule.cover(L) @ x - need
            by_hour = (idle / data.WEEKS).reshape(7, 24).mean(axis=0)
            js["idle_by_hour"][f"{d}__{L}"] = by_hour.tolist()
            out.append(f"   {d:10s} L {L:2d}: " + " ".join(f"{v:.1f}" for v in by_hour))
    out.append("")

    out.append("5. Scale check, EMS citywide: tours per day in the cheapest 8-hour roster")
    c = cells[(cells["dataset"] == "ems_city") & (cells["L"] == 8)]
    tours = c["ip"].sum() / 8 / (data.WEEKS * 7)
    out.append(f"   {tours:.0f} eight-hour tours per day on average; mean requirement {needs['ems_city'].mean():.1f}")
    js["tours_per_day_8h"] = float(tours)
    out.append("")

    out.append("6. Paid hours relative to 8-hour tours, by dataset and L (from cells.csv)")
    for d in needs:
        cd = cells[cells["dataset"] == d].groupby("L")["ip"].sum()
        out.append(f"   {d:18s} " + ", ".join(f"{L}: {cd[L] / cd[8]:.3f}" for L in data.LENGTHS))
    out.append("")

    out.append("7. Binding hours per day on weekdays and at weekends, L = 8 (mean)")
    days = np.arange(data.WEEKS * 7)
    wd = (days % 7) < 5
    for d in needs:
        per_day = masks[f"{d}__8"].reshape(-1, 24).sum(axis=1)
        out.append(f"   {d:18s} weekdays {per_day[wd].mean():.2f}, weekends {per_day[~wd].mean():.2f}")

    out.append("")
    out.append("8. A worked week: EMS citywide, the week with the median total requirement, 8- and 12-hour tours")
    tot = needs["ems_city"].sum(axis=1)
    w = int(np.argsort(tot)[len(tot) // 2])
    need = needs["ems_city"][w].astype(np.float64)
    start = data.grid()[w * WEEK]
    out.append(f"   week {w} starting {start.date()}: total requirement {need.sum():.0f} ambulance-hours")
    js["worked_week"] = {"week": w, "start": str(start.date()), "need": float(need.sum())}
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for L in (8, 12):
        ip, x = schedule.roster(need, L)
        teeth = comb.best_comb(need, L)
        v = need[teeth].sum()
        b = np.flatnonzero(masks[f"ems_city__{L}"][w])
        out.append(
            f"   L {L}: cheapest roster {ip:.0f} paid hours ({ip / L:.0f} tours); comb bound {L * v:.0f}; "
            f"{len(teeth)} teeth, {len(b)} binding hours"
        )
        out.append(
            "     teeth: "
            + ", ".join(f"{days[t // 24]} {t % 24:02d}:00 ({need[t]:.0f})" for t in teeth[:9])
            + (" ..." if len(teeth) > 9 else "")
        )
        extra = [h for h in b if h not in teeth]
        out.append(
            "     binding hours that are not teeth of this comb: "
            + (", ".join(f"{days[h // 24]} {h % 24:02d}:00" for h in extra) if extra else "none")
        )
        js["worked_week"][f"L{L}"] = {"ip": ip, "comb_bound": float(L * v), "teeth": len(teeth), "binding": len(b)}
    out.append("")
    out.append(
        "9. Signal and noise of the comb's position, L in 6, 8, 12: weekly comb sums of the best and the second-best"
    )
    out.append(
        "   phase of the mean profile; mean weekly difference, its standard deviation over weeks, and their ratio"
    )
    js["comb_margin"] = {}
    for d, nd in needs.items():
        m = nd.reshape(-1, 24).mean(axis=0)
        parts = []
        for L in data.DIVISORS:
            sums = np.array([m[phi::L].sum() for phi in range(L)])
            first, second = np.argsort(sums)[::-1][:2]
            weekly = nd.reshape(data.WEEKS, 7, 24)
            diff = weekly[:, :, first::L].sum(axis=(1, 2)) - weekly[:, :, second::L].sum(axis=(1, 2))
            ratio = diff.mean() / diff.std(ddof=1)
            near = js["weekly_phase"][f"{d}__{L}"]["at"]
            js["comb_margin"][f"{d}__{L}"] = {
                "mean": float(diff.mean()),
                "sd": float(diff.std(ddof=1)),
                "ratio": float(ratio),
                "weeks_at_phase": near,
            }
            parts.append(
                f"L {L}: {diff.mean():.0f} +/- {diff.std(ddof=1):.0f} (ratio {ratio:.2f}; at phase {near:.2f})"
            )
        out.append(f"   {d:18s} " + "; ".join(parts))
    cm = js["comb_margin"].values()
    rho = spearmanr([v["ratio"] for v in cm], [v["weeks_at_phase"] for v in cm]).statistic
    js["comb_margin_spearman"] = float(rho)
    out.append(f"   Spearman correlation of the ratio with the share of weeks at the phase, 21 cases: {rho:.2f}")
    out.append("")
    out.append(
        "10. The year as one continuous roster: comb bound L x V over all 8,736 hours in sequence (tours may run"
    )
    out.append(
        "    across week boundaries; at most one tooth of leftover a year), relative to 8-hour tours, against the"
    )
    out.append("    registered ratio of the 52 cyclic weeks")
    js["continuous_year"] = {}
    for d, nd in needs.items():
        series = nd.ravel().astype(np.float64)
        bound = {L: L * comb_value_path(series, L) for L in data.LENGTHS}
        cd = cells[cells["dataset"] == d].groupby("L")["ip"].sum()
        js["continuous_year"][d] = {int(L): float(bound[L] / bound[8]) for L in data.LENGTHS}
        out.append(
            f"   {d:18s} "
            + ", ".join(f"{L}: {bound[L] / bound[8]:.3f} (weeks {cd[L] / cd[8]:.3f})" for L in data.LENGTHS)
            + f"; idle at 8 h {bound[8] / series.sum() - 1:.3f}, at 12 h {bound[12] / series.sum() - 1:.3f}"
        )
    a.out.write_text("\n".join(out) + "\n")
    (a.out.parent / "details.json").write_text(json.dumps(js, indent=1) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
