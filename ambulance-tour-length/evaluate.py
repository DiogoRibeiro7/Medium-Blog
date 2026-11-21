"""The registered test (PLAN.md): rosters of L-hour tours, combs, binding hours, H1 to H3.

python evaluate.py --sources sources --out results

Writes results/summary.txt (every registered number), results/cells.csv, results/binding_share.csv,
results/profiles.csv, results/short_tours.csv, results/sensitivity.txt and results/verdicts.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from staff import comb, data, schedule

WEEK = data.WEEK


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _cell_job(args: tuple[str, int, np.ndarray]) -> tuple[str, int, list[dict], np.ndarray, np.ndarray]:
    d, L, nd = args
    rows = []
    bind = np.zeros((data.WEEKS, WEEK), dtype=bool)
    combh = np.zeros((data.WEEKS, WEEK), dtype=bool)
    for w in range(data.WEEKS):
        need = nd[w].astype(np.float64)
        ip, _ = schedule.roster(need, L)
        lp = schedule.relaxation(need, L)
        v = comb.comb_value(need, L)
        bind[w] = schedule.binding_by_resolve(need, L, ip)
        combh[w] = comb.binding(need, L)
        rows.append(
            {
                "dataset": d,
                "week": w,
                "L": L,
                "need": float(need.sum()),
                "ip": ip,
                "lp": lp,
                "comb": v,
                "L_comb": L * v,
                "binding_hours": int(bind[w].sum()),
                "comb_hours": int(combh[w].sum()),
                "binding_in_comb": int((bind[w] & combh[w]).sum()),
                "teeth": len(comb.best_comb(need, L)),
            }
        )
    return d, L, rows, bind, combh


def solve_all(needs: dict[str, np.ndarray], workers: int = 4) -> tuple[pd.DataFrame, dict, dict]:
    """Every (dataset, week, L) cell: integer optimum, relaxation, comb value, binding hours, comb hours."""
    jobs = [(d, L, nd) for d, nd in needs.items() for L in data.LENGTHS]
    rows, masks, combs = [], {}, {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for d, L, r, bind, combh in ex.map(_cell_job, jobs):
            rows += r
            masks[(d, L)] = bind
            combs[(d, L)] = combh
            log(f"solved {d} L {L}")
    return pd.DataFrame(rows), masks, combs


def profile(nd: np.ndarray) -> np.ndarray:
    """Mean requirement at each clock hour over the 364 test days."""
    return nd.reshape(-1, 24).mean(axis=0)


def share_by_hour(mask: np.ndarray, days: np.ndarray | None = None) -> np.ndarray:
    """f(k): share of days on which clock hour k is a binding hour."""
    by_day = mask.reshape(-1, 24)
    if days is not None:
        by_day = by_day[days]
    return by_day.mean(axis=0)


def rules(m: np.ndarray, L: int) -> dict[str, int]:
    climb = int(np.argmax(m - np.roll(m, 1)))
    return {
        "comb": comb.phase_by_comb(m, L),
        "climb": climb % L,
        "peak": int(np.argmax(m)) % L,
        "climb_hour": climb,
        "peak_hour": int(np.argmax(m)),
    }


def hypotheses(needs: dict[str, np.ndarray], cells: pd.DataFrame, masks: dict) -> tuple[dict, list]:
    lines: list[str] = []
    v: dict = {}

    # H1 (L in 6, 8, 12; 9 and 10 reported)
    for L in data.LENGTHS:
        c = cells[cells["L"] == L]
        g = c["ip"] - c["L_comb"]
        lines.append(
            f"  L {L:2d}: comb bound tight in {(g.abs() < 1e-6).mean():.4f}; largest shortfall "
            f"{(g / c['ip']).max():.4%}; binding hours that are comb hours "
            f"{c['binding_in_comb'].sum() / c['binding_hours'].sum():.3f}; comb hours that are binding "
            f"{c['binding_in_comb'].sum() / c['comb_hours'].sum():.3f}; relaxation = integer optimum in "
            f"{((c['ip'] - c['lp']).abs() < 1e-6).mean():.3f}"
        )
    c = cells[cells["L"].isin(data.DIVISORS)]
    gap = c["ip"] - c["L_comb"]
    tight = float((gap.abs() < 1e-6).mean())
    rel = float((gap / c["ip"]).max())
    in_comb = float(c["binding_in_comb"].sum() / c["binding_hours"].sum())
    comb_binding = float(c["binding_in_comb"].sum() / c["comb_hours"].sum())
    lines.insert(
        0,
        f"H1 (L in 6, 8, 12; {len(c)} cells): comb bound equals the integer optimum in {tight:.4f}; "
        f"largest shortfall {rel:.4%}; binding hours that are comb hours {in_comb:.4f}; comb hours that are "
        f"binding {comb_binding:.4f}",
    )
    h1 = tight >= 0.80 and rel <= 0.01 and in_comb >= 0.75 and comb_binding >= 0.90
    v["H1"] = {
        "cells": len(c),
        "tight_share": tight,
        "largest_shortfall": rel,
        "binding_in_comb": in_comb,
        "comb_binding": comb_binding,
        "supported": bool(h1),
    }
    lines.append(f"  H1 supported: {h1}")
    lines.append("")

    # H2
    matches = {"comb": 0, "climb": 0, "peak": 0}
    rows = []
    for d, nd in needs.items():
        m = profile(nd)
        for L in data.DIVISORS:
            f = share_by_hour(masks[(d, L)])
            obs = comb.phase_of_share(f, L)
            r = rules(m, L)
            for k in matches:
                matches[k] += int(r[k] == obs)
            rows.append({"dataset": d, "L": L, "observed": obs, **r})
            lines.append(
                f"H2 {d:18s} L {L:2d}: observed phase {obs}; comb {r['comb']}, climb {r['climb']} "
                f"(hour {r['climb_hour']}), peak {r['peak']} (hour {r['peak_hour']})"
            )
    n_cases = len(rows)  # 21 in the registered test: at least 19, i.e. all but two
    h2 = matches["comb"] >= n_cases - 2 and matches["comb"] > matches["climb"] and matches["comb"] > matches["peak"]
    lines.append(
        f"H2: matches out of {n_cases}: comb {matches['comb']}, steepest climb {matches['climb']}, "
        f"peak {matches['peak']}; supported: {h2}"
    )
    v["H2"] = {"matches": matches, "cases": rows, "supported": bool(h2)}
    lines.append("")

    # H3
    ok_by_L = {}
    for L in data.LENGTHS:
        tops = {d: float(share_by_hour(masks[(d, L)]).max()) for d in needs}
        good = sum((t > 0.5) if L in data.DIVISORS else (t < 0.5) for t in tops.values())
        ok_by_L[L] = good
        lines.append(
            f"H3 L {L:2d}: largest binding share by dataset: "
            + ", ".join(f"{d} {t:.2f}" for d, t in tops.items())
            + f"; datasets on the predicted side of 0.5: {good} of {len(tops)}"
        )
    h3 = all(g >= len(needs) - 1 for g in ok_by_L.values())
    lines.append(f"H3 supported: {h3}")
    v["H3"] = {"datasets_on_predicted_side": ok_by_L, "supported": bool(h3)}
    return v, lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources", type=Path, default=Path("sources"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    needs = data.requirements(a.sources)
    lines = ["Tour length and the hours that size a roster: EMS and 311, test weeks 31 Dec 2018 to 29 Dec 2019", ""]
    rev = json.loads((a.sources / "REVISION").read_text()) if (a.sources / "REVISION").exists() else {}
    if rev:
        lines.append("EMS download: " + json.dumps(rev))
    for d, nd in needs.items():
        lines.append(
            f"{d:18s} mean requirement {nd.mean():8.1f}, min {nd.min():4d}, max {nd.max():4d}, "
            f"hours with zero requirement {int((nd == 0).sum())}"
        )
    lines.append("")

    workers = min(4, os.cpu_count() or 1)
    cells, masks, _ = solve_all(needs, workers)
    cells.to_csv(a.out / "cells.csv", index=False)
    v, hl = hypotheses(needs, cells, masks)
    lines += hl
    lines.append("")

    # Also reported
    lines.append("Idle share (paid hours over required hours, minus one) and paid hours relative to 8-hour tours")
    for d in needs:
        c = cells[cells["dataset"] == d].groupby("L")[["ip", "need"]].sum()
        idle = c["ip"] / c["need"] - 1
        rel = c["ip"] / c.loc[8, "ip"]
        lines.append(f"  {d:18s} " + "; ".join(f"L {L}: idle {idle[L]:.3f}, x{rel[L]:.3f}" for L in data.LENGTHS))
    lines.append("Binding hours per week (mean) against the most a comb can hold, floor(168 / L)")
    for d in needs:
        c = cells[cells["dataset"] == d].groupby("L")[["binding_hours", "teeth"]].mean()  # mean over weeks
        lines.append(
            f"  {d:18s} "
            + "; ".join(
                f"L {L}: binding {c.loc[L, 'binding_hours']:.1f}, teeth {c.loc[L, 'teeth']:.1f} of {WEEK // L}"
                for L in data.LENGTHS
            )
        )
    lines.append("Harmonics of the mean daily profile: amplitude (share of the mean) and peak hour, harmonics 1 to 6")
    prof_rows = []
    for d, nd in needs.items():
        m = profile(nd)
        for k in range(24):
            prof_rows.append({"dataset": d, "hour": k, "mean_need": m[k]})
        F = np.fft.rfft(m) / 24
        parts = []
        for n in range(1, 7):
            amp = 2 * abs(F[n]) / m.mean()
            peak = (-np.angle(F[n]) / (2 * np.pi) * 24 / n) % (24 / n)
            parts.append(f"{n}: {amp:.3f} at {peak:.1f}h")
        lines.append(f"  {d:18s} " + "; ".join(parts))
    pd.DataFrame(prof_rows).to_csv(a.out / "profiles.csv", index=False)

    days = np.arange(data.WEEKS * 7)
    weekday = (days % 7) < 5
    share_rows = []
    for (d, L), mk in masks.items():
        f_all = share_by_hour(mk)
        f_wd = share_by_hour(mk, weekday)
        f_we = share_by_hour(mk, ~weekday)
        for k in range(24):
            share_rows.append(
                {"dataset": d, "L": L, "hour": k, "all": f_all[k], "weekday": f_wd[k], "weekend": f_we[k]}
            )
    pd.DataFrame(share_rows).to_csv(a.out / "binding_share.csv", index=False)
    lines.append("Binding share by clock hour, three most frequent hours, all days | weekdays | weekends")
    for (d, L), mk in masks.items():
        parts = []
        for sel in (None, weekday, ~weekday):
            f = share_by_hour(mk, sel)
            top = np.argsort(f)[::-1][:3]
            parts.append(", ".join(f"{k:02d}:00 {f[k]:.2f}" for k in top))
        lines.append(f"  {d:18s} L {L:2d}: " + " | ".join(parts))

    lines.append("Short tours: share of 4-hour tours covering at least one binding hour of the L-only roster, vs 4 / L")
    st_rows = []
    for d in ("ems_city", "nyc311"):
        for L in (8, 12):
            hit = tot = 0
            for w in range(data.WEEKS):
                need = needs[d][w].astype(np.float64)
                _, y = schedule.roster_with_short(need, L)
                B = masks[(d, L)][w]
                for j in np.flatnonzero(y > 0):
                    cov = B[(j + np.arange(schedule.SHORT)) % WEEK].any()
                    hit += int(y[j]) * int(cov)
                    tot += int(y[j])
            share = hit / tot if tot else float("nan")
            st_rows.append(
                {"dataset": d, "L": L, "short_tours": tot, "covering_binding": hit, "share": share, "random": 4 / L}
            )
            lines.append(
                f"  {d:18s} L {L:2d}: {tot} short tours, {share:.3f} cover a binding hour (random {4 / L:.3f})"
            )
        log(f"short tours {d}")
    pd.DataFrame(st_rows).to_csv(a.out / "short_tours.csv", index=False)

    (a.out / "summary.txt").write_text("\n".join(lines) + "\n")
    (a.out / "verdicts.json").write_text(json.dumps(v, indent=1, default=str) + "\n")
    log("registered test written")

    # Sensitivity of the EMS requirement (registered as also reported)
    sens = []
    for label, alpha in (("2% chance of no free ambulance", 0.02), ("load rounded up, no margin", None)):
        nds = data.requirements(a.sources, alpha=alpha)
        nds = {k: x for k, x in nds.items() if k.startswith("ems")}
        c2, m2, _ = solve_all(nds, workers)
        v2, l2 = hypotheses(nds, c2, m2)
        sens.append(f"== {label}")
        sens += l2
        sens.append("")
        c2.to_csv(a.out / f"cells_{'a02' if alpha else 'ceil'}.csv", index=False)
    (a.out / "sensitivity.txt").write_text("\n".join(sens) + "\n")
    print("\n".join(lines))
    log("done")


if __name__ == "__main__":
    main()
