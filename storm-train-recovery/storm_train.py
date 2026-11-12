"""Recovery capacity under clustered storms: Portugal's 2026 storm train.

Reproducibility companion for the article "Recovery Capacity under Clustered
Storms: A Queueing View of Portugal's 2026 Storm Train".

The script does four things and prints every number the article quotes:

1. Reads the E-Redes outage counts published during the January-February 2026
   storms, fits a two-phase restoration curve to the reports that measure the
   Kristin backlog alone, and measures how fast that backlog drained between
   the later storms.
2. Puts the Kristin restoration next to Florida Power & Light's restorations
   after the 2004-05 and 2022-24 hurricanes and Entergy's after Katrina, and
   compares the size of the restoration workforce with the size of the damage.
3. Simulates storm seasons with the same mean number of storms arriving either
   independently (Poisson) or in trains (a self-exciting Hawkes process), and
   pushes them through a repair queue with finite crews, a delayed surge of
   outside crews, and damage that compounds when a storm hits a network that
   is still being repaired.
4. Compares preparedness levers (hardening, more own crews, a faster surge,
   keeping surge crews after a storm) under the two arrival processes.

Everything is deterministic: the simulations use a fixed seed and the script
downloads nothing. Figures go to figures/, displayed equations to
figures/equations/.

Usage:
    python storm_train.py > results/output.txt
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
EQUATIONS = FIGURES / "equations"

SEED = 20260128

# --- chart style (light surface, fixed categorical order) -------------------
SURFACE = "#ffffff"
INK = "#292929"
INK_2 = "#52514e"
GRID = "#e6e5e1"
BAND = "#f0efec"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": GRID,
    "axes.linewidth": 1.0,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "axes.axisbelow": True,
})
DPI = 160


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def parse_time(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M")


def days_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 86400.0


def read_csv(name: str) -> list[dict[str, str]]:
    with open(DATA / name, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt_int(x: float) -> str:
    return f"{x:,.0f}"


def loglinear(t_pts: np.ndarray, y_pts: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Interpolate a positive series linearly in log space."""
    return np.exp(np.interp(t, t_pts, np.log(y_pts)))


def time_to_fraction(t_pts, f_pts, target: float) -> float | None:
    """First time a positive fraction series crosses `target`, interpolating in logs.

    Returns None when the crossing is not bracketed by two positive observations.
    """
    t_pts = np.asarray(t_pts, float)
    f_pts = np.asarray(f_pts, float)
    for i in range(len(t_pts) - 1):
        f0, f1 = f_pts[i], f_pts[i + 1]
        if f0 > 0 and f1 > 0 and f0 >= target > f1:
            w = (math.log(target) - math.log(f0)) / (math.log(f1) - math.log(f0))
            return float(t_pts[i] + w * (t_pts[i + 1] - t_pts[i]))
    return None


# =============================================================================
# 1. The Portuguese storm train
# =============================================================================

KRISTIN_PEAK_TIME = parse_time("2026-01-28 06:00")
KRISTIN_PEAK = 1_000_000.0
MAINLAND_LV_CUSTOMERS = 6_595_634  # ERSE, low-voltage customers in mainland Portugal


def load_eredes():
    rows = read_csv("eredes_customers_out_2026.csv")
    out = []
    for r in rows:
        when = parse_time(r["time_local"])
        out.append({
            "time": when,
            "t": days_between(KRISTIN_PEAK_TIME, when),
            "n": float(r["customers_out"]),
            "scope": r["scope"],
            "fit": r["fit_baseline"] == "1",
            "note": r["note"],
        })
    return out


def biexp(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    a, tau1, tau2 = params
    return KRISTIN_PEAK * (a * np.exp(-t / tau1) + (1.0 - a) * np.exp(-t / tau2))


def fit_biexp(t: np.ndarray, n: np.ndarray, start=(0.6, 0.3, 5.0)) -> np.ndarray:
    def resid(p):
        return np.log(biexp(p, t)) - np.log(n)

    res = least_squares(resid, x0=np.array(start), bounds=([0.01, 0.01, 0.3], [0.99, 3.0, 60.0]))
    return res.x


def fit_single(t: np.ndarray, n: np.ndarray) -> float:
    def resid(p):
        return np.log(KRISTIN_PEAK * np.exp(-t / p[0])) - np.log(n)

    return float(least_squares(resid, x0=[3.0], bounds=([0.01], [60.0])).x[0])


def kristin_analysis(rows):
    section("1. The Portuguese storm train (E-Redes customers without power)")
    storms = read_csv("storms_2026.csv")
    onsets = [(s["storm"], parse_time(s["start_local"])) for s in storms]
    print("Storm onsets used for the gaps (local time = UTC in winter):")
    for (name, when) in onsets:
        print(f"  {name:<9} {when:%d %b %H:%M}")
    gaps = [days_between(onsets[i][1], onsets[i + 1][1]) for i in range(len(onsets) - 1)]
    for i, g in enumerate(gaps):
        print(f"  gap {onsets[i][0]} -> {onsets[i + 1][0]}: {g:.1f} days")
    span = days_between(onsets[0][1], onsets[-1][1])
    print(f"Five named storms with onsets spread over {span:.1f} days; mean gap {np.mean(gaps):.1f} days")

    print(f"Peak: about {fmt_int(KRISTIN_PEAK)} customers at 06:00 on 28 January, "
          f"{100 * KRISTIN_PEAK / MAINLAND_LV_CUSTOMERS:.1f}% of the {fmt_int(MAINLAND_LV_CUSTOMERS)} "
          "low-voltage customers in mainland Portugal")
    print(f"Published counts used: {len(rows)} "
          f"({sum(r['scope'] == 'total' for r in rows)} for the whole mainland, "
          f"{sum(r['scope'] == 'kristin' for r in rows)} for the Kristin-affected areas)")

    # restoration within the first hours: the switching phase
    early = [r for r in rows if r["scope"] == "total" and 0 <= r["t"] <= 0.4]
    for r in early:
        print(f"  {r['time']:%d %b %H:%M}: {fmt_int(r['n'])} out "
              f"({100 * (1 - r['n'] / KRISTIN_PEAK):.0f}% of the peak restored)")

    fit_rows = [r for r in rows if r["fit"]]
    t = np.array([r["t"] for r in fit_rows])
    n = np.array([r["n"] for r in fit_rows])
    params = fit_biexp(t, n)
    a, tau1, tau2 = params
    resid = np.log(biexp(params, t)) - np.log(n)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    tau_single = fit_single(t, n)
    rms_single = float(np.sqrt(np.mean((np.log(KRISTIN_PEAK * np.exp(-t / tau_single)) - np.log(n)) ** 2)))

    rng = np.random.default_rng(SEED)
    fitted_log = np.log(biexp(params, t))
    boot = []
    for _ in range(1000):
        y = np.exp(fitted_log + rng.choice(resid, size=resid.size, replace=True))
        boot.append(fit_biexp(t, y, start=params))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [5, 95], axis=0)

    print()
    print(f"Two-phase fit to the {len(fit_rows)} reports that measure the Kristin backlog alone:")
    print(f"  fast (switching) share a = {a:.3f}  [90% bootstrap {lo[0]:.3f}-{hi[0]:.3f}]")
    print(f"  fast time constant tau1 = {24 * tau1:.1f} h  [{24 * lo[1]:.1f}-{24 * hi[1]:.1f} h]")
    print(f"  slow time constant tau2 = {tau2:.2f} days  [{lo[2]:.2f}-{hi[2]:.2f}]; "
          f"slow half-life {tau2 * math.log(2):.2f} days [{lo[2] * math.log(2):.2f}-{hi[2] * math.log(2):.2f}]")
    print(f"  RMS error in logs: {rms:.3f} (a single exponential: tau = {tau_single:.2f} days, RMS {rms_single:.3f})")
    for frac in (0.5, 0.25, 0.10, 0.05, 0.01):
        tt = brentq_fraction(params, frac)
        print(f"  fitted curve reaches {100 * frac:>4.0f}% of the peak after {tt:.2f} days")

    # backlog left by Kristin when the next storms began, on the fitted curve
    for name, when in onsets:
        tt = days_between(KRISTIN_PEAK_TIME, when)
        if tt > 0:
            print(f"  fitted Kristin backlog at the onset of {name} (day {tt:.1f}): "
                  f"{fmt_int(biexp(params, np.array([tt]))[0])} customers")

    # drain rate of the Kristin backlog between storms
    print()
    print("Drain rate of the Kristin-attributed backlog, window by window:")
    anchors = [
        ("2026-01-30 15:00", "total", "2026-02-04 12:00", "kristin", "new faults on 2 Feb"),
        ("2026-02-04 12:00", "kristin", "2026-02-09 08:00", "kristin", "Leonardo and Marta"),
        ("2026-02-09 08:00", "kristin", "2026-02-12 08:00", "kristin", "no new storm"),
        ("2026-02-12 08:00", "kristin", "2026-02-16 08:00", "kristin", "new faults and flooding on 13 Feb"),
        ("2026-02-16 08:00", "kristin", "2026-02-22 17:00", "kristin", "no new storm"),
    ]
    lookup = {(r["time"], r["scope"]): r["n"] for r in rows}
    windows = []
    for s_time, s_scope, e_time, e_scope, label in anchors:
        t0, t1 = parse_time(s_time), parse_time(e_time)
        n0, n1 = lookup[(t0, s_scope)], lookup[(t1, e_scope)]
        dt = days_between(t0, t1)
        rate = math.log(n0 / n1) / dt
        windows.append((t0, t1, n0, n1, rate, label))
        print(f"  {t0:%d %b %H:%M} -> {t1:%d %b %H:%M} ({dt:.1f} d): {fmt_int(n0)} -> {fmt_int(n1)}, "
              f"rate {rate:.3f}/day, half-life {math.log(2) / rate:.1f} days  [{label}]")
    quiet = [w[4] for w in windows if w[5] == "no new storm"]
    stormy = [w[4] for w in windows if w[5] == "Leonardo and Marta"][0]
    print(f"  mean rate in the windows without a new storm: {np.mean(quiet):.3f}/day; "
          f"during Leonardo and Marta: {stormy:.3f}/day ({100 * stormy / np.mean(quiet):.0f}% of it)")

    # customer-days: what the train added on top of a Kristin-only restoration
    totals = sorted([r for r in rows if r["scope"] == "total" and r["t"] >= 0], key=lambda r: r["t"])
    tt = np.array([r["t"] for r in totals])
    nn = np.array([r["n"] for r in totals])
    grid = np.linspace(0, tt[-1], 20001)
    observed = loglinear(tt, nn, grid)
    baseline = biexp(params, grid)
    cd_obs = np.trapezoid(observed, grid)
    cd_base = np.trapezoid(baseline, grid)
    after = grid >= days_between(KRISTIN_PEAK_TIME, parse_time("2026-02-02 00:00"))
    cd_obs_after = np.trapezoid(observed[after], grid[after])
    cd_base_after = np.trapezoid(baseline[after], grid[after])
    print()
    print(f"Customer-days without power, 28 Jan 06:00 to {totals[-1]['time']:%d %b %H:%M} "
          "(whole mainland, log-linear between reports):")
    print(f"  observed {cd_obs / 1e6:.2f} million; Kristin-only fitted curve {cd_base / 1e6:.2f} million; "
          f"excess {100 * (cd_obs / cd_base - 1):.0f}%")
    print(f"  from 2 February on: observed {cd_obs_after / 1e6:.2f} million vs {cd_base_after / 1e6:.2f} million "
          f"({cd_obs_after / cd_base_after:.2f} times the Kristin-only curve)")

    # one percent of the peak: fitted vs observed
    first_1pc = min(r["t"] for r in rows if r["scope"] == "kristin" and r["n"] <= 0.01 * KRISTIN_PEAK)
    print(f"  the Kristin backlog first reported at or below 1% of the peak on day {first_1pc:.1f}; "
          f"the fitted curve gets there on day {brentq_fraction(params, 0.01):.1f}")

    workforce = read_csv("eredes_workforce_2026.csv")
    print()
    print("E-Redes operational staff in the field:")
    for w in workforce:
        print(f"  {parse_time(w['time_local']):%d %b}: {fmt_int(float(w['operational_staff']))}")
    return params, windows, onsets


def brentq_fraction(params, frac: float) -> float:
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if biexp(params, np.array([mid]))[0] / KRISTIN_PEAK > frac:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def figure_storm_train(rows, onsets):
    storms = read_csv("storms_2026.csv")
    workforce = read_csv("eredes_workforce_2026.csv")
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for s in storms:
        a, b = parse_time(s["start_local"]), parse_time(s["end_local"])
        ax.axvspan(a, b, color=BAND, lw=0, zorder=0)
        mid = a + (b - a) / 2
        high = s["storm"] in ("Ingrid", "Kristin", "Marta")
        ax.text(mid, 2.4e6 if high else 1.45e6, s["storm"], ha="center", va="bottom", fontsize=9.5, color=INK_2)
    pre = [r for r in rows if r["scope"] == "total" and r["t"] < 0]
    tot = sorted([r for r in rows if r["scope"] == "total" and r["t"] >= 0], key=lambda r: r["time"])
    ax.plot([r["time"] for r in pre], [r["n"] for r in pre], linestyle="none", color=BLUE, marker="o", ms=5,
            mec=SURFACE, mew=1.5, zorder=3)
    ax.annotate("27 Jan (Joseph):\n10,000, then 1,700", xy=(pre[0]["time"], pre[0]["n"]), xytext=(-92, 12),
                textcoords="offset points", fontsize=9, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    kri = sorted([r for r in rows if r["scope"] == "kristin"], key=lambda r: r["time"])
    ax.plot([r["time"] for r in tot], [r["n"] for r in tot], color=BLUE, lw=2, marker="o", ms=5,
            mec=SURFACE, mew=1.5, label="All of mainland Portugal", zorder=3)
    ax.plot([r["time"] for r in kri], [r["n"] for r in kri], color=ORANGE, lw=2, marker="s", ms=5,
            mec=SURFACE, mew=1.5, label="Areas hit by Kristin", zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(4.5e2, 4e6)
    ax.set_yticks([1e3, 1e4, 1e5, 1e6])
    ax.set_yticklabels(["1,000", "10,000", "100,000", "1,000,000"])
    ax.set_xlim(parse_time("2026-01-21 00:00"), parse_time("2026-02-24 00:00"))
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[22, 25, 28, 1, 4, 7, 10, 13, 16, 19, 22]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_ylabel("Customers without power")
    notes = [
        ("2026-02-02 08:00", 161000, "new faults\novernight", (8, 30)),
        ("2026-02-07 19:30", 167000, "Marta:\n56k → 167k", (12, 22)),
        ("2026-02-13 08:00", 45000, "new faults\nand flooding", (10, 26)),
    ]
    for when, y, text, off in notes:
        ax.annotate(text, xy=(parse_time(when), y), xytext=off, textcoords="offset points",
                    fontsize=9, color=INK_2, ha="left",
                    arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    for w in workforce:
        when = parse_time(w["time_local"])
        staff = int(float(w["operational_staff"]))
        ax.plot([when], [8.5e2], marker="^", color=INK_2, ms=6, zorder=4)
        side = {400: ("right", -3), 1200: ("left", 3)}.get(staff, ("center", 0))
        ax.annotate(f"{staff:,} staff", xy=(when, 8.5e2), xytext=(side[1], -9), textcoords="offset points",
                    fontsize=8.5, color=INK_2, ha=side[0], va="top")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.93))
    fig.tight_layout()
    fig.savefig(FIGURES / "01_storm_train.png", dpi=DPI)
    plt.close(fig)


def figure_kristin_fit(rows, params, windows):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.45, 1]})
    used = [r for r in rows if r["fit"]]
    other = [r for r in rows if (not r["fit"]) and r["t"] >= 0 and (r["scope"] == "kristin")]
    grid = np.linspace(0, 27, 1000)
    ax.plot(grid, biexp(params, grid), color=INK_2, lw=1.5, label="Two-phase fit")
    ax.plot([r["t"] for r in used], [r["n"] for r in used], "o", color=BLUE, ms=6, mec=SURFACE, mew=1.5,
            label="Reports used in the fit", zorder=3)
    ax.plot([r["t"] for r in other], [r["n"] for r in other], "o", color=ORANGE, ms=6, mec=SURFACE, mew=1.5,
            label="Within 36 h of a new storm or new faults", zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(1e3, 1.5e6)
    ax.set_yticks([1e3, 1e4, 1e5, 1e6])
    ax.set_yticklabels(["1,000", "10,000", "100,000", "1,000,000"])
    ax.set_xlim(-0.5, 27)
    ax.set_xlabel("Days since the peak (06:00, 28 January)")
    ax.set_ylabel("Customers without power in the Kristin areas")
    ax.legend(loc="upper right", fontsize=9.5)
    a, tau1, tau2 = params
    ax.text(0.6, 1.6e3, f"fast share {a:.0%}, τ₁ = {24 * tau1:.1f} h\nslow τ₂ = {tau2:.1f} days",
            fontsize=9.5, color=INK)

    labels = [f"{w[0]:%d %b}–{w[1]:%d %b}\n{w[5]}" for w in windows]
    rates = [w[4] for w in windows]
    y = np.arange(len(windows))[::-1]
    colors = [ORANGE if w[5] != "no new storm" else BLUE for w in windows]
    bx.barh(y, rates, height=0.5, color=colors)
    for yi, r in zip(y, rates):
        bx.text(r + 0.006, yi, f"{r:.2f}/day\nhalf-life {math.log(2) / r:.1f} d", va="center", fontsize=9,
                color=INK)
    bx.set_yticks(y)
    bx.set_yticklabels(labels, fontsize=9)
    bx.set_xlim(0, 0.42)
    bx.set_xlabel("Drain rate of the Kristin backlog (per day)")
    bx.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "02_kristin_backlog.png", dpi=DPI)
    plt.close(fig)


# =============================================================================
# 2. Florida, Katrina and the size of the repair force
# =============================================================================


def us_comparison(rows):
    section("2. Restoration curves and repair capacity: Portugal, Florida, Louisiana")
    us = read_csv("us_restoration.csv")
    series = {}
    for r in us:
        key = (r["utility"], r["storm"])
        land = parse_time(r["landfall_local"])
        when = parse_time(r["time_local"])
        series.setdefault(key, []).append({
            "t": days_between(land, when),
            "f": float(r["customers_out"]) / float(r["customers_affected"]),
            "kind": r["kind"],
            "affected": float(r["customers_affected"]),
        })
    kristin = sorted(
        [{"t": r["t"], "f": r["n"] / KRISTIN_PEAK, "kind": "observed"} for r in rows
         if r["t"] >= 0 and (r["scope"] == "kristin" or (r["scope"] == "total" and r["t"] < 4.0))],
        key=lambda p: p["t"])
    series[("E-Redes", "Kristin 2026")] = [{**p, "affected": KRISTIN_PEAK} for p in kristin]

    print("Share of affected customers still without power (log-linear between reports;"
          " '-' where the reports do not bracket the value):")
    print(f"  {'utility':<8} {'event':<13} {'affected':>10} {'day 3':>7} {'day 5':>7} {'day 7':>7} "
          f"{'to 10%':>7} {'to 1%':>6} {'last report':>18}")
    summary = {}
    for (utility, storm), pts in series.items():
        pts = sorted(pts, key=lambda p: p["t"])
        pos = [p for p in pts if p["f"] > 0]
        tp = np.array([0.0] + [p["t"] for p in pos]) if pos[0]["t"] > 0 else np.array([p["t"] for p in pos])
        fp = np.array([1.0] + [p["f"] for p in pos]) if pos[0]["t"] > 0 else np.array([p["f"] for p in pos])
        vals = []
        for day in (3, 5, 7):
            vals.append(f"{100 * loglinear(tp, fp, np.array([day]))[0]:.0f}%" if tp[0] <= day <= tp[-1] else "-")
        t10 = time_to_fraction(tp, fp, 0.10)
        t01 = time_to_fraction(tp, fp, 0.01)
        zero = [p["t"] for p in pts if p["f"] == 0]
        last = f"0 on day {zero[0]:.0f}" if zero else f"{100 * pos[-1]['f']:.2f}% on day {pos[-1]['t']:.1f}"
        print(f"  {utility:<8} {storm:<13} {fmt_int(pts[0]['affected']):>10} {vals[0]:>7} {vals[1]:>7} "
              f"{vals[2]:>7} {('%.1f' % t10) if t10 else '-':>7} {('%.1f' % t01) if t01 else '-':>6} {last:>18}")
        summary[(utility, storm)] = (tp, fp, zero[0] if zero else None, pts)

    cap = read_csv("capacity.csv")
    print()
    print("Restoration workforce against the size of the damage:")
    for c in cap:
        n = float(c["customers_affected"])
        w = float(c["workforce"])
        per_k = 1000 * w / n
        poles = c["damaged_poles"]
        extra = f", {w / float(poles):.2f} per damaged pole ({fmt_int(float(poles))} poles)" if poles else ""
        days = f", restored in {c['days_to_restore']} days" if c["days_to_restore"] else ""
        print(f"  {c['utility']:<8} {c['event']:<20} {fmt_int(w):>7} workers for {fmt_int(n):>9} customers: "
              f"{per_k:.1f} per 1,000{extra}{days}")
    fpl = [1000 * float(c["workforce"]) / float(c["customers_affected"]) for c in cap if c["utility"] == "FPL"]
    ered = [1000 * float(c["workforce"]) / float(c["customers_affected"]) for c in cap if c["utility"] == "E-Redes"]
    print(f"  FPL range {min(fpl):.1f}-{max(fpl):.1f} per 1,000 customers affected; E-Redes {min(ered):.1f}-{max(ered):.1f}"
          f" (FPL minimum is {min(fpl) / max(ered):.1f} times the E-Redes maximum)")
    print("  Entergy's largest outage before 2005 was 270,000 customers (Tropical Storm Cindy);"
          f" Katrina's 1.1 million was {1.1e6 / 270000:.1f} times that.")
    print(f"  REN lost 101 very-high-voltage towers to Kristin against 25 in its previous worst storm (2009): "
          f"{101 / 25:.1f} times.")
    return summary, cap


def figure_restoration(summary):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    floor = 1e-3
    groups = {
        ("E-Redes", "Kristin 2026"): (BLUE, "o"),
        ("FPL", "Charley 2004"): (ORANGE, "v"),
        ("FPL", "Frances 2004"): (ORANGE, "s"),
        ("FPL", "Jeanne 2004"): (ORANGE, "D"),
        ("FPL", "Wilma 2005"): (ORANGE, "^"),
        ("FPL", "Ian 2022"): (AQUA, "s"),
        ("FPL", "Milton 2024"): (AQUA, "D"),
        ("Entergy", "Katrina 2005"): (YELLOW, "o"),
    }
    for key, (color, marker) in groups.items():
        tp, fp, zero, pts = summary[key]
        if zero is not None:
            zero = max(zero, tp[-1] + 0.15)
        xs = list(tp) + ([zero] if zero is not None else [])
        ys = list(fp) + ([floor] if zero is not None else [])
        lw = 2.5 if key[0] == "E-Redes" else 1.6
        ls = (0, (1.5, 2.5)) if key[0] == "Entergy" else "-"
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=3 if key[0] == "E-Redes" else 2)
        obs = [(p["t"], p["f"]) for p in pts if p["f"] > 0 and p["kind"] == "observed"]
        stm = [(p["t"], p["f"]) for p in pts if p["f"] > 0 and p["kind"] == "statement"]
        if obs:
            ax.plot(*zip(*obs), linestyle="none", marker=marker, color=color, ms=6, mec=SURFACE, mew=1.2, zorder=4)
        if stm:
            ax.plot(*zip(*stm), linestyle="none", marker=marker, mfc=SURFACE, mec=color, mew=1.6, ms=6, zorder=4)
        if zero is not None:
            ax.plot([zero], [floor], marker="|", color=color, ms=12, mew=2, zorder=4)
        name = key[1].split()[0]
        if key[0] == "E-Redes":
            ax.text(25.8, 2.2e-3, "Kristin", fontsize=10, color=INK, va="bottom", fontweight="bold")
        elif zero is not None:
            ax.text(zero + 0.3, floor * 1.25, name, fontsize=8.5, color=INK_2, rotation=60, va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.8, 1.3)
    ax.set_xlim(-0.5, 44)
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1])
    ax.set_yticklabels(["all restored", "1%", "10%", "100%"])
    ax.set_xlabel("Days since landfall (US) or since the outage peak (Kristin)")
    ax.set_ylabel("Share of affected customers still without power")
    handles = [
        plt.Line2D([], [], color=BLUE, lw=2.5, marker="o", label="E-Redes, Kristin 2026"),
        plt.Line2D([], [], color=ORANGE, lw=1.6, marker="s", label="FPL, 2004–05 (Charley, Frances, Jeanne, Wilma)"),
        plt.Line2D([], [], color=AQUA, lw=1.6, marker="s", label="FPL, 2022–24 (Ian, Milton)"),
        plt.Line2D([], [], color=YELLOW, lw=1.6, marker="o", label="Entergy, Katrina 2005"),
        plt.Line2D([], [], color=INK_2, lw=0, marker="o", mfc=SURFACE, mec=INK_2, label="hollow: utility statement"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "03_restoration_comparison.png", dpi=DPI)
    plt.close(fig)


def figure_capacity(cap):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 5.0), gridspec_kw={"width_ratios": [1.6, 1]})
    color = {"E-Redes": BLUE, "FPL": ORANGE, "Entergy": AQUA}
    items = sorted(cap, key=lambda c: float(c["workforce"]) / float(c["customers_affected"]))
    y = np.arange(len(items))
    vals = [1000 * float(c["workforce"]) / float(c["customers_affected"]) for c in items]
    ax.hlines(y, 0, vals, color=GRID, lw=1)
    ax.scatter(vals, y, s=70, c=[color[c["utility"]] for c in items], edgecolors=SURFACE, linewidths=1.5, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.4, yi, f"{v:.1f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c['utility']}, {c['event']}" for c in items], fontsize=9)
    ax.set_xlabel("Restoration workers per 1,000 customers without power")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 24)
    poles = [c for c in cap if c["damaged_poles"]]
    poles = sorted(poles, key=lambda c: float(c["workforce"]) / float(c["damaged_poles"]))
    y2 = np.arange(len(poles))
    v2 = [float(c["workforce"]) / float(c["damaged_poles"]) for c in poles]
    bx.barh(y2, v2, height=0.3, color=[color[c["utility"]] for c in poles])
    for yi, v in zip(y2, v2):
        bx.text(v + 0.1, yi, f"{v:.2f}", va="center", fontsize=9, color=INK)
    bx.set_yticks(y2)
    bx.set_yticklabels([f"{c['utility']}, {c['event'].replace(' (day 9)', '')}" for c in poles], fontsize=9)
    bx.set_xlabel("Workers per damaged pole")
    bx.set_xlim(0, 8.5)
    bx.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], lw=0, marker="o", ms=8, color=color[k], label=k) for k in color]
    ax.legend(handles=handles, loc="lower right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "04_capacity.png", dpi=DPI)
    plt.close(fig)


# =============================================================================
# 3. Storm seasons: independent storms or trains
# =============================================================================

SEASON_DAYS = 120.0
MEAN_STORMS = 2.0
THETA = 3.0  # days: how long a storm raises the chance of another
N_SEASONS = 20000
DT = 0.05


def branching_for_dispersion(psi: float) -> float:
    """Branching ratio n giving a long-window dispersion psi = 1/(1-n)^2 - 1."""
    return 1.0 - 1.0 / math.sqrt(1.0 + psi)


def simulate_arrivals(rng, n_branch: float, n_seasons: int = N_SEASONS, mean_count: float = MEAN_STORMS,
                      season: float = SEASON_DAYS, theta: float = THETA, burn: float = 60.0):
    """Event times for many seasons: Poisson if n_branch == 0, else a stationary Hawkes process.

    The Hawkes process is built from its cluster representation: immigrants arrive
    as a Poisson process of rate nu, and each event has Poisson(n) offspring after
    exponential delays with mean theta. A burn-in period before the season makes it
    stationary, so the mean count in the season is exactly nu * season / (1 - n).
    """
    nu = mean_count * (1.0 - n_branch) / season
    counts = rng.poisson(nu * (season + burn), n_seasons)
    sid = np.repeat(np.arange(n_seasons), counts)
    times = rng.uniform(-burn, season, counts.sum())
    all_s, all_t = [sid], [times]
    gen_s, gen_t = sid, times
    while n_branch > 0 and gen_t.size:
        kids = rng.poisson(n_branch, gen_t.size)
        gen_s = np.repeat(gen_s, kids)
        gen_t = np.repeat(gen_t, kids) + rng.exponential(theta, kids.sum())
        keep = gen_t < season
        gen_s, gen_t = gen_s[keep], gen_t[keep]
        all_s.append(gen_s)
        all_t.append(gen_t)
    s = np.concatenate(all_s)
    t = np.concatenate(all_t)
    keep = t >= 0
    s, t = s[keep], t[keep]
    order = np.lexsort((t, s))
    return s[order], t[order]


def next_storm_probability(sid, times, horizons, season=SEASON_DAYS, h_max=30.0):
    """P(another storm within h days | a storm now), using storms early enough to see h_max days ahead."""
    same = np.r_[sid[1:] == sid[:-1], False]
    gap = np.where(same, np.r_[times[1:], np.inf] - times, np.inf)
    eligible = times <= season - h_max
    g = gap[eligible]
    return np.array([np.mean(g <= h) for h in horizons])


@dataclass(frozen=True)
class System:
    mu0: float = 0.10      # own crews: storm-units of repair work per day
    s_max: float = 0.10    # largest outside surge, same units
    t_target: float = 3.0  # surge is requested to clear the backlog in this many days
    lag: float = 4.0       # days for outside crews to arrive (time constant)
    release: float = 1.0   # days for surplus outside crews to be sent home (time constant)
    hold: float = 0.0      # days after each storm during which no outside crews are sent home
    kappa: float = 0.5     # compounding: extra damage when a storm hits a backlog
    harden: float = 1.0    # multiplies the damage of every storm


def simulate_backlog(sid, times, damage, system: System, n_seasons: int = N_SEASONS,
                     season: float = SEASON_DAYS, dt: float = DT):
    steps = int(round(season / dt))
    k = np.minimum((times / dt).astype(int), steps - 1)
    order = np.argsort(k, kind="stable")
    ks, ss, ds = k[order], sid[order], damage[order]
    bounds = np.searchsorted(ks, np.arange(steps + 1))
    q = np.zeros(n_seasons)
    s = np.zeros(n_seasons)
    last = np.full(n_seasons, -1e9)
    area = np.zeros(n_seasons)
    peak = np.zeros(n_seasons)
    surge_days = np.zeros(n_seasons)
    q_before = np.zeros(times.size)
    for step in range(steps):
        a, b = bounds[step], bounds[step + 1]
        if b > a:
            idx = ss[a:b]
            qb = q[idx]
            q_before[order[a:b]] = qb
            add = system.harden * ds[a:b] * (1.0 + system.kappa * np.minimum(1.0, qb))
            np.add.at(q, idx, add)
            last[idx] = step * dt
        target = np.clip(q / system.t_target - system.mu0, 0.0, system.s_max)
        held = (step * dt - last) < system.hold
        rate = np.where(target > s, dt / system.lag, np.where(held, 0.0, dt / system.release))
        s = s + (target - s) * rate
        q = np.maximum(q - (system.mu0 + s) * dt, 0.0)
        area += q * dt
        np.maximum(peak, q, out=peak)
        surge_days += s * dt
    return {"area": area, "peak": peak, "surge_days": surge_days, "q_before": q_before}


def describe(result, system: System) -> dict:
    area = result["area"]
    return {
        "mean": float(area.mean()),
        "p95": float(np.percentile(area, 95)),
        "p99": float(np.percentile(area, 99)),
        "peak_gt2": float(np.mean(result["peak"] > 2.0)),
        "overlap": float(np.mean(result["q_before"] > 0.25)),
        "surge": float(result["surge_days"].mean()),
        "crew_days": float(system.mu0 * SEASON_DAYS + result["surge_days"].mean()),
    }


def season_simulations():
    section("3. Storm seasons: independent storms against trains")
    rng = np.random.default_rng(SEED)
    sigma = 1.0
    print(f"Season {SEASON_DAYS:.0f} days, mean {MEAN_STORMS:.0f} damaging storms; storm damage lognormal with "
          f"mean 1 (one 'storm-unit' of repair work) and log-sd {sigma}; {N_SEASONS:,} seasons per case.")
    processes = {}
    for label, psi in (("Poisson", 0.0), ("Hawkes, psi = 0.5", 0.5), ("Hawkes, psi = 1", 1.0)):
        n_br = branching_for_dispersion(psi)
        sid, t = simulate_arrivals(rng, n_br)
        damage = rng.lognormal(-0.5 * sigma ** 2, sigma, t.size)
        counts = np.bincount(sid, minlength=N_SEASONS)
        psi_hat = counts.var() / counts.mean() - 1.0
        processes[label] = (sid, t, damage, n_br)
        print(f"  {label:<18} branching n = {n_br:.3f}: mean storms {counts.mean():.3f}, "
              f"simulated dispersion {psi_hat:.2f}, seasons with 3 or more storms {np.mean(counts >= 3):.1%}, "
              f"with 4 or more {np.mean(counts >= 4):.1%}")
    print(f"  damage: median {math.exp(-0.5 * sigma ** 2):.2f}, 95th percentile "
          f"{math.exp(-0.5 * sigma ** 2 + 1.6449 * sigma):.2f} storm-units")

    horizons = np.arange(0, 30.01, 0.25)
    lam = MEAN_STORMS / SEASON_DAYS
    print()
    print("Chance of another storm within h days of a storm:")
    probs = {}
    for label, (sid, t, _, n_br) in processes.items():
        p = next_storm_probability(sid, t, horizons)
        probs[label] = p
        sel = {h: p[np.searchsorted(horizons, h)] for h in (3, 7, 14)}
        print(f"  {label:<18} 3 d {sel[3]:.3f}, 7 d {sel[7]:.3f}, 14 d {sel[14]:.3f}")
    for h in (3, 7, 14):
        print(f"  Poisson formula 1 - exp(-lambda h), h = {h:>2}: {1 - math.exp(-lam * h):.3f}")
    n1 = processes["Hawkes, psi = 1"][3]
    print(f"  Hawkes (psi = 1): a storm adds n/theta = {n1 / THETA:.3f} storms per day to the rate, against a "
          f"mean rate of {lam:.4f}; right after a storm the rate is {(lam + n1 / THETA) / lam:.1f} times the mean")

    base = System()
    print()
    print(f"Repair system: own crews clear {base.mu0} storm-units a day; outside surge up to {base.s_max} more, "
          f"requested to clear the backlog in {base.t_target:.0f} days, arriving with a {base.lag:.0f}-day lag "
          f"and sent home with a {base.release:.0f}-day lag as the backlog shrinks; compounding kappa = {base.kappa}.")
    print("Backlog-days per season (storm-unit-days; proportional to customer-days if outages scale with the backlog):")
    tails = {}
    for label in ("Poisson", "Hawkes, psi = 1"):
        sid, t, damage, _ = processes[label]
        for kappa in (0.0, 0.5):
            sysk = replace(base, kappa=kappa)
            res = simulate_backlog(sid, t, damage, sysk)
            d = describe(res, sysk)
            tails[(label, kappa)] = res["area"]
            print(f"  {label:<16} kappa {kappa}: mean {d['mean']:.2f}, 95th pct {d['p95']:.2f}, 99th pct {d['p99']:.2f}, "
                  f"P(peak backlog > 2) {d['peak_gt2']:.3f}, storms arriving on a backlog > 0.25: {d['overlap']:.1%}")
    m = {k: float(v.mean()) for k, v in tails.items()}
    p95 = {k: float(np.percentile(v, 95)) for k, v in tails.items()}
    p99 = {k: float(np.percentile(v, 99)) for k, v in tails.items()}
    print(f"  trains instead of independent storms (kappa 0.5): mean +{100 * (m[('Hawkes, psi = 1', 0.5)] / m[('Poisson', 0.5)] - 1):.0f}%, "
          f"95th pct +{100 * (p95[('Hawkes, psi = 1', 0.5)] / p95[('Poisson', 0.5)] - 1):.0f}%, "
          f"99th pct +{100 * (p99[('Hawkes, psi = 1', 0.5)] / p99[('Poisson', 0.5)] - 1):.0f}%")
    print(f"  compounding (kappa 0 -> 0.5): Poisson mean +{100 * (m[('Poisson', 0.5)] / m[('Poisson', 0.0)] - 1):.0f}%, "
          f"trains mean +{100 * (m[('Hawkes, psi = 1', 0.5)] / m[('Hawkes, psi = 1', 0.0)] - 1):.0f}%; "
          f"Poisson 99th +{100 * (p99[('Poisson', 0.5)] / p99[('Poisson', 0.0)] - 1):.0f}%, "
          f"trains 99th +{100 * (p99[('Hawkes, psi = 1', 0.5)] / p99[('Hawkes, psi = 1', 0.0)] - 1):.0f}%")

    print()
    print("Preparedness levers, relative to the base system (kappa 0.5):")
    levers = [
        ("Base system", base),
        ("Hardening: 30% less damage per storm", replace(base, harden=0.7)),
        ("50% more own crews", replace(base, mu0=0.15)),
        ("Surge arrives in 1 day instead of 4", replace(base, lag=1.0)),
        ("Keep surge crews 10 days after each storm", replace(base, hold=10.0)),
        ("Sturdier temporary repairs: compounding halved", replace(base, kappa=0.25)),
    ]
    lever_results = {}
    for label in ("Poisson", "Hawkes, psi = 1"):
        sid, t, damage, _ = processes[label]
        ref = None
        print(f"  {label}:")
        for name, sysl in levers:
            d = describe(simulate_backlog(sid, t, damage, sysl), sysl)
            if ref is None:
                ref = d
            dm = 100 * (1 - d["mean"] / ref["mean"])
            dp = 100 * (1 - d["p95"] / ref["p95"])
            extra = d["crew_days"] - ref["crew_days"]
            per = (ref["mean"] - d["mean"]) / extra if extra > 1e-6 else float("nan")
            lever_results[(label, name)] = (dm, dp, extra, per, d)
            per_txt = f", {per:.2f} backlog-days saved per extra crew-day" if extra > 1e-6 else ""
            print(f"    {name:<48} mean {d['mean']:.2f} ({dm:.0f}% less), 95th {d['p95']:.2f} ({dp:.0f}% less), "
                  f"crew-days {d['crew_days']:.1f} ({extra:+.2f}){per_txt}")
    print()
    print("Sensitivity: the same levers with a deeper outside surge (up to 0.4 storm-units a day, four times own crews):")
    for label in ("Poisson", "Hawkes, psi = 1"):
        sid, t, damage, _ = processes[label]
        ref = None
        parts = []
        for name, sysl in levers:
            sysl = replace(sysl, s_max=0.4)
            d = describe(simulate_backlog(sid, t, damage, sysl), sysl)
            if ref is None:
                ref = d
                parts.append(f"base mean {d['mean']:.2f}")
                continue
            parts.append(f"{name.split(':')[0].split(' instead')[0].split(' after')[0]} "
                         f"{100 * (1 - d['mean'] / ref['mean']):.0f}%")
        print(f"  {label}: " + "; ".join(parts))
    c = base.mu0 + base.s_max
    print()
    print(f"Square law for one storm drained at a fixed capacity c = {c:.1f}: backlog-days = D^2 / (2c); "
          f"D = 1 gives {1 / (2 * c):.2f}, D = 0.7 gives {0.49 / (2 * c):.2f} ({100 * (1 - 0.49):.0f}% less)")
    for name in [n for n, _ in levers][1:]:
        gp = lever_results[("Poisson", name)][0]
        gh = lever_results[("Hawkes, psi = 1", name)][0]
        sp = lever_results[("Poisson", name)][3]
        sh = lever_results[("Hawkes, psi = 1", name)][3]
        print(f"  {name}: mean backlog-days {gp:.0f}% lower with independent storms, {gh:.0f}% lower with trains; "
              f"absolute saving {lever_results[('Poisson', 'Base system')][4]['mean'] - lever_results[('Poisson', name)][4]['mean']:.2f}"
              f" vs {lever_results[('Hawkes, psi = 1', 'Base system')][4]['mean'] - lever_results[('Hawkes, psi = 1', name)][4]['mean']:.2f}")
    return processes, probs, horizons, tails, levers, lever_results


def figure_next_storm(probs, horizons):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    colors = {"Poisson": BLUE, "Hawkes, psi = 0.5": ORANGE, "Hawkes, psi = 1": AQUA}
    names = {"Poisson": "Independent storms (Poisson)", "Hawkes, psi = 0.5": "Trains, dispersion ψ = 0.5",
             "Hawkes, psi = 1": "Trains, dispersion ψ = 1"}
    for label, p in probs.items():
        ax.plot(horizons, p, color=colors[label], label=names[label])
        ax.text(30.3, p[-1], f"{p[-1]:.0%}", va="center", fontsize=9.5, color=INK)
    lam = MEAN_STORMS / SEASON_DAYS
    ax.plot(horizons, 1 - np.exp(-lam * horizons), color=INK_2, lw=1, ls=(0, (2, 2)),
            label="1 − exp(−λh)")
    ax.axvline(7, color=GRID, lw=1)
    for label, p in probs.items():
        i = np.searchsorted(horizons, 7)
        ax.plot([7], [p[i]], "o", color=colors[label], ms=7, mec=SURFACE, mew=1.5, zorder=4)
        ax.text(7.4, p[i], f"{p[i]:.0%}", fontsize=9, color=INK, va="bottom")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, max(p.max() for p in probs.values()) * 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Days after a damaging storm")
    ax.set_ylabel("Chance that another damaging storm has arrived")
    ax.legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "05_next_storm.png", dpi=DPI)
    plt.close(fig)


def figure_tails(tails):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    styles = {
        ("Poisson", 0.0): (BLUE, (0, (4, 3)), "Independent storms, no compounding"),
        ("Poisson", 0.5): (BLUE, "-", "Independent storms, compounding κ = 0.5"),
        ("Hawkes, psi = 1", 0.0): (ORANGE, (0, (4, 3)), "Trains (ψ = 1), no compounding"),
        ("Hawkes, psi = 1", 0.5): (ORANGE, "-", "Trains (ψ = 1), compounding κ = 0.5"),
    }
    for key, (color, ls, name) in styles.items():
        x = np.sort(tails[key])
        surv = 1.0 - np.arange(1, x.size + 1) / x.size
        keep = surv > 0
        ax.plot(x[keep], surv[keep], color=color, ls=ls, label=name)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1)
    ax.set_xlim(0, 420)
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1])
    ax.set_yticklabels(["1 in 1,000", "1 in 100", "1 in 10", "every season"])
    ax.set_xlabel("Backlog-days in the season (storm-units of unrepaired damage × days)")
    ax.set_ylabel("Share of seasons exceeding it")
    ax.legend(loc="upper right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "06_backlog_tails.png", dpi=DPI)
    plt.close(fig)


def figure_levers(levers, lever_results):
    names = [n for n, _ in levers][1:]
    short = {
        "Hardening: 30% less damage per storm": "Harden: 30% less damage",
        "50% more own crews": "50% more own crews",
        "Surge arrives in 1 day instead of 4": "Surge in 1 day, not 4",
        "Keep surge crews 10 days after each storm": "Keep surge crews 10 days",
        "Sturdier temporary repairs: compounding halved": "Halve compounding",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, col, title in ((axes[0], 0, "Mean backlog-days"), (axes[1], 1, "95th-percentile season")):
        for off, label, color in ((0.17, "Poisson", BLUE), (-0.17, "Hawkes, psi = 1", ORANGE)):
            vals = [lever_results[(label, n)][col] for n in names]
            ax.barh(y + off, vals, height=0.3, color=color,
                    label="Independent storms" if label == "Poisson" else "Trains (ψ = 1)")
            for yi, v in zip(y + off, vals):
                ax.text(v + 0.8, yi, f"{v:.0f}%", va="center", fontsize=8.5, color=INK)
        ax.set_title(f"Reduction in {title.lower()}", fontsize=11, color=INK, loc="left")
        ax.set_xlim(0, 75)
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(100, decimals=0))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([short[n] for n in names], fontsize=9.5)
    axes[1].legend(loc="lower right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "07_levers.png", dpi=DPI)
    plt.close(fig)


# =============================================================================
# 4. Displayed equations (Medium does not render LaTeX)
# =============================================================================

EQUATION_TEX = {
    "eq01_two_phase": r"$N(t) = N_0\,\left[\,a\,e^{-t/\tau_1} + (1-a)\,e^{-t/\tau_2}\,\right]$",
    "eq02_drain_rate": r"$r = \dfrac{\ln(N_{\mathrm{start}}\,/\,N_{\mathrm{end}})}{\Delta t},\qquad t_{1/2} = \dfrac{\ln 2}{r}$",
    "eq03_hawkes": r"$\lambda(t) = \nu + \sum_{t_k < t}\ \dfrac{n}{\theta}\,e^{-(t - t_k)/\theta}$",
    "eq04_dispersion": r"$\psi = \dfrac{\mathrm{Var}(N_T)}{\mathrm{E}(N_T)} - 1 \approx \dfrac{1}{(1-n)^2} - 1$",
    "eq05_poisson_gap": r"$\Pr(G \leq h) = 1 - e^{-\bar{\lambda} h}$",
    "eq06_backlog": r"$\dfrac{dQ}{dt} = -\left(\mu_0 + S(t)\right)\quad \mathrm{while}\ \ Q(t) > 0$",
    "eq07_compounding": r"$Q(t_k^{+}) = Q(t_k^{-}) + \eta\,D_k\left[\,1 + \kappa\,\min\left(1,\ Q(t_k^{-})/\bar{D}\right)\right]$",
    "eq08_surge": r"$\dfrac{dS}{dt} = \dfrac{R(t) - S(t)}{\ell},\qquad R(t) = \min\left(S_{\max},\ \max\left(0,\ \dfrac{Q(t)}{T^{*}} - \mu_0\right)\right)$",
    "eq09_square_law": r"$\int_0^{D/c} Q(t)\,dt = \int_0^{D/c} (D - c\,t)\,dt = \dfrac{D^2}{2c}$",
}


def draw_equations() -> None:
    """Render each displayed equation on a canvas of fixed width (1600 px) so all share one scale."""
    EQUATIONS.mkdir(parents=True, exist_ok=True)
    width_in, dpi = 8.0, 200
    for name, tex in EQUATION_TEX.items():
        fig = plt.figure(figsize=(width_in, 1.0), dpi=dpi)
        fig.patch.set_facecolor(SURFACE)
        with plt.rc_context({"mathtext.fontset": "cm"}):
            text = fig.text(0.5, 0.5, tex, ha="center", va="center", fontsize=19, color=INK)
            fig.canvas.draw()
            box = text.get_window_extent()
        height_in = box.height / dpi + 0.32
        fig.set_size_inches(width_in, height_in)
        fig.savefig(EQUATIONS / f"{name}.png", dpi=dpi, facecolor=SURFACE)
        plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("Recovery capacity under clustered storms: printed results")
    print("=========================================================")
    rows = load_eredes()
    params, windows, onsets = kristin_analysis(rows)
    summary, cap = us_comparison(rows)
    processes, probs, horizons, tails, levers, lever_results = season_simulations()
    figure_storm_train(rows, onsets)
    figure_kristin_fit(rows, params, windows)
    figure_restoration(summary)
    figure_capacity(cap)
    figure_next_storm(probs, horizons)
    figure_tails(tails)
    figure_levers(levers, lever_results)
    draw_equations()
    print()
    print(f"Figures written to {FIGURES.relative_to(ROOT)}/ and equations to {EQUATIONS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
