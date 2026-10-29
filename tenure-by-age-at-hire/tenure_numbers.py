"""How long do new hires stay, by age at hire? Every number quoted in the article.

Sections
  1. The published job-duration tables (NLSY79, six releases) and their two estimands
  2. Survival from hire by age at start: piecewise-exponential hazards and restricted means
  3. Censoring near the survey date: how much can it move the published shares?
  4. Worst cases and break-even exit rates beyond the observed window
  5. Hires per seat: the renewal view of the same curves
  6. A career model, calibrated to both published estimands, and what averaging alone does
  7. Stock versus flow: what median tenure by age can and cannot say
  8. The employer's view from the calibrated model: survival, restricted means, hires per seat
  9. Sensitivity to the two assumptions (time between jobs, retirement age)

Run:  python tenure_numbers.py > output.txt   # about two minutes
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from tenure_core import (DATA, KNOT_YEARS, LT_COLUMNS, RESULTS, interval_hazards, knots_from_F, median_from_knots,
                         nlsy79_rows, pick, read_csv, renewal_hires, rmst, survival_at, value)

RESULTS.mkdir(exist_ok=True)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


# ---------------------------------------------------------------------------
# 1. The tables and the two estimands
# ---------------------------------------------------------------------------

rows = nlsy79_rows()

section("1. NLSY79 Table 2 across six releases (cumulative % of jobs ended)")
print("release     survey   phrasing         group   <1y   <2y   <5y  <10y  <15y  ongoing")
for r in rows:
    cells = "".join(f"{100 * v:6.1f}" if not np.isnan(v) else "     ." for v in r["F"])
    print(f"{r['release']}  {r['survey']}  {r['phrasing']:<15}  {r['group']:<6}{cells}  {100 * r['ongoing']:5.1f}")

print()
print("The break between the 2019 and 2021 releases (same cohort, two survey years apart):")
pairs = [("18-24", "18-24"), ("25-34", "25-34"), ("35-44", "35-44"), ("45-52", "45-54")]
for g19, g21 in pairs:
    a, b = pick(rows, "2019-08-22", g19), pick(rows, "2021-08-31", g21)
    diffs = [(100 * (a["F"][k] - b["F"][k])) for k in range(3) if not np.isnan(a["F"][k]) and not np.isnan(b["F"][k])]
    print(f"  {g19:>5} (2019, 'jobs') vs {g21:>5} (2021, 'average worker'): "
          f"<1y {100 * a['F'][0]:.1f} vs {100 * b['F'][0]:.1f};  drop in points at 1/2/5 y: "
          + ", ".join(f"{d:.1f}" for d in diffs))

print()
print("Stability within each phrasing (same cohort, consecutive releases):")
for g in ["18-24", "25-34", "35-44"]:
    seq = [pick(rows, rel, g)["F"][0] for rel in ["2017-08-24", "2019-08-22", "2021-08-31", "2023-08-22", "2025-08-26"]]
    print(f"  {g}: <1y by release 2017..2025 = " + ", ".join(f"{100 * v:.1f}" for v in seq))
seq45 = [pick(rows, "2017-08-24", "45-50"), pick(rows, "2019-08-22", "45-52"), pick(rows, "2021-08-31", "45-54"),
         pick(rows, "2023-08-22", "45-56"), pick(rows, "2025-08-26", "45-54")]
print("  45+: <1y by release 2017..2025 = " + ", ".join(f"{100 * r['F'][0]:.1f} ({r['group']})" for r in seq45))
print("  45+: <2y by release 2017..2025 = " + ", ".join(f"{100 * r['F'][1]:.1f}" for r in seq45))


# ---------------------------------------------------------------------------
# 2. Survival from hire, hazards, restricted means
# ---------------------------------------------------------------------------

HIRE = {g: pick(rows, "2019-08-22", g) for g in ["18-24", "25-34", "35-44", "45-52"]}
WORKER = {g: pick(rows, "2025-08-26", g) for g in ["18-24", "25-34", "35-44", "45-54", "55-58"]}

section("2. Survival from hire by age at start")
for label, table in [("Hire-weighted (2019 release, 'jobs')", HIRE), ("Worker-averaged (2025 release, 'average worker')", WORKER)]:
    print(f"\n{label}")
    print("group   S(1)   S(2)   S(5)  S(10)  S(15) | hazard/yr in [0,1) [1,2) [2,5) [5,10) [10,15) | median")
    for g, r in table.items():
        t, S = knots_from_F(r["F"])
        h = interval_hazards(t, S)
        Scells = "".join(f"{100 * s:6.1f}" for s in S[1:]) + "      ." * (5 - (len(S) - 1))
        hcells = "".join(f"{x:7.3f}" for x in h) + "      ." * (5 - len(h))
        med = median_from_knots(t, S)
        med_s = f"{med:5.2f} y" if not np.isnan(med) else f" >{t[-1]:.0f} y"
        print(f"{g:<6}{Scells} |              {hcells} | {med_s}")

print("\nRatios of the first-year exit share, younger group over older group:")
for label, table, old in [("hire-weighted", HIRE, "45-52"), ("worker-averaged", WORKER, "45-54")]:
    for young in ["18-24", "25-34"]:
        print(f"  {label}: {young} vs {old}: {table[young]['F'][0] / table[old]['F'][0]:.2f}x")

print("\nRestricted mean time with the employer (years) within H years of the hire:")
print("group         H=1    H=2    H=5   H=10   H=15")
for label, table in [("hire", HIRE), ("worker", WORKER)]:
    for g, r in table.items():
        t, S = knots_from_F(r["F"])
        cells = "".join(f"{rmst(H, t, S):7.2f}" if H <= t[-1] else "      ." for H in [1, 2, 5, 10, 15])
        print(f"{label:<6} {g:<6}{cells}")

print("\nHow many of 100 hires are still there at 1 and 2 years (the payback window):")
for label, table in [("hire", HIRE), ("worker", WORKER)]:
    for g, r in table.items():
        print(f"  {label:<6} {g:<6} {100 * (1 - r['F'][0]):5.1f} at 1y, {100 * (1 - r['F'][1]):5.1f} at 2y")

print("\nNLSY97 (born 1980-84, 2026 release, 'average worker'):")
for r in read_csv("nlsy97_table2_2026.csv"):
    print(f"  {r['age_group']}: <1y {r['lt1']}, <2y {r['lt2']}, <3y {r['lt3']}, <6y {r['lt6']}, <13y {r['lt13']}, ongoing {r['ongoing']}")


# ---------------------------------------------------------------------------
# 3. Censoring near the survey date
# ---------------------------------------------------------------------------

section("3. Censoring: share of starts with less follow-up than the cell needs")


def followup_shortfall(age_lo: float, age_hi: float, horizon: float, interview: tuple[float, float],
                       min_age_at_interview: float | None, n: int = 2_000_000, seed: int = 11) -> float:
    """P(follow-up < horizon) for job starts at ages [age_lo, age_hi), cohort born 1957-64.

    Births uniform over 1957-64, interview dates uniform over the window, start ages uniform in
    the band and before the interview. Only the timing is modelled, not who starts a job."""
    rng = np.random.default_rng(seed)
    birth = rng.uniform(1957.0, 1965.0, n)
    iv = rng.uniform(*interview, n)
    start_age = rng.uniform(age_lo, age_hi, n)
    start = birth + start_age
    ok = start < iv
    if min_age_at_interview is not None:
        ok &= (iv - birth) >= min_age_at_interview
    fu = iv[ok] - start[ok]
    return float(np.mean(fu < horizon))


CENSOR_CASES = [
    ("55-58 at 1 year", 55, 59, 1.0, 59.0, WORKER["55-58"]["F"][0]),
    ("55-58 at 2 years", 55, 59, 2.0, 59.0, WORKER["55-58"]["F"][1]),
    ("45-54 at 5 years", 45, 55, 5.0, None, WORKER["45-54"]["F"][2]),
    ("45-54 at 2 years", 45, 55, 2.0, None, WORKER["45-54"]["F"][1]),
]
print("cell               published  share of starts short of follow-up   upper bound on the true share")
print("                              (interview window 2022.5-2023.5 | 2022.0-2024.0)")
censor_bounds = {}
for name, lo, hi, hor, mia, Fpub in CENSOR_CASES:
    a = followup_shortfall(lo, hi, hor, (2022.5, 2023.5), mia)
    b = followup_shortfall(lo, hi, hor, (2022.0, 2024.0), mia)
    w = max(a, b)
    censor_bounds[name] = (Fpub, Fpub + w)
    print(f"{name:<18} {pct(Fpub)}      {pct(a)} | {pct(b)}                     {pct(Fpub + w)}")


# ---------------------------------------------------------------------------
# 4. Worst cases and break-even exit rates
# ---------------------------------------------------------------------------

section("4. Beyond the observed window: worst cases and break-even exit rates")


def worst_and_carry(r: dict, H: float) -> tuple[float, float, float]:
    t, S = knots_from_F(r["F"])
    worst = rmst(t[-1], t, S)  # every survivor leaves on the day the data stop
    carry = rmst(H, t, S, tail_hazard=interval_hazards(t, S)[-1])  # last observed hazard continues
    stay = worst + S[-1] * (H - t[-1])  # nobody leaves after the data stop
    return worst, carry, stay


def break_even_hazard(r: dict, H: float, target: float) -> float:
    """Constant exit rate after the last knot that brings the restricted mean down to target."""
    t, S = knots_from_F(r["F"])
    base = rmst(t[-1], t, S)
    if base >= target:
        return float("inf")
    f = lambda h: base + S[-1] * (1 - np.exp(-h * (H - t[-1]))) / h - target
    lo, hi = 1e-6, 50.0
    if f(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


for label, table, old, horizons in [("worker-averaged", WORKER, "45-54", [10, 15]),
                                     ("worker-averaged", WORKER, "55-58", [5, 10]),
                                     ("hire-weighted", HIRE, "45-52", [5, 10])]:
    r_old = table[old]
    t_old, S_old = knots_from_F(r_old["F"])
    print(f"\n{label}, {old} (data stop at {t_old[-1]:.0f} years, S = {100 * S_old[-1]:.1f}%; "
          f"last observed hazard {interval_hazards(t_old, S_old)[-1]:.3f}/yr)")
    for H in horizons:
        worst, carry, stay = worst_and_carry(r_old, H)
        line = f"  H={H:>2}: {old} worst {worst:.2f}, carry-forward {carry:.2f}, nobody-leaves {stay:.2f}"
        for young in ["25-34", "35-44"]:
            ry = table[young]
            ty, Sy = knots_from_F(ry["F"])
            ym = rmst(H, ty, Sy) if H <= ty[-1] else rmst(H, ty, Sy, tail_hazard=interval_hazards(ty, Sy)[-1])
            be = break_even_hazard(r_old, H, ym)
            be_s = "never" if np.isinf(be) else f"{be:.3f}/yr (={100 * (1 - np.exp(-be)):.0f}% of stayers per year)"
            line += f" | {young} {ym:.2f}, break-even exit rate {be_s}"
        print(line)

print("\nThe break-even exit rate for 55-58 starters against the observed exit rate in their second year:")
t55, S55 = knots_from_F(WORKER["55-58"]["F"])
h55 = interval_hazards(t55, S55)
t2534, S2534 = knots_from_F(WORKER["25-34"]["F"])
be55 = break_even_hazard(WORKER["55-58"], 5, rmst(5, t2534, S2534))
print(f"  observed hazard in [1,2): {h55[1]:.3f}/yr; break-even over 5 years vs 25-34: {be55:.3f}/yr; ratio {be55 / h55[1]:.1f}x")
# the censoring-corrected worst case for 55-58: raise the 1- and 2-year exit shares to their upper bounds
F_adj = WORKER["55-58"]["F"].copy()
F_adj[0], F_adj[1] = censor_bounds["55-58 at 1 year"][1], censor_bounds["55-58 at 2 years"][1]
r_adj = dict(WORKER["55-58"], F=F_adj)
t_a, S_a = knots_from_F(F_adj)
be55_adj = break_even_hazard(r_adj, 5, rmst(5, t2534, S2534))
print(f"  with the 1- and 2-year shares at their censoring upper bounds ({pct(F_adj[0])}, {pct(F_adj[1])}): "
      f"RMST(2) {rmst(2, t_a, S_a):.2f}, break-even {be55_adj:.3f}/yr = {be55_adj / interval_hazards(t_a, S_a)[1]:.1f}x the second-year hazard")
print(f"  RMST(2) unadjusted: 55-58 {rmst(2, t55, S55):.2f} vs 25-34 {rmst(2, t2534, S2534):.2f} vs 35-44 "
      f"{rmst(2, *knots_from_F(WORKER['35-44']['F'])):.2f} vs 45-54 {rmst(2, *knots_from_F(WORKER['45-54']['F'])):.2f}")


# ---------------------------------------------------------------------------
# 5. Hires per seat
# ---------------------------------------------------------------------------

section("5. Hires needed to keep one seat filled (renewal, vacancies filled at once)")
print("group          H=1    H=2    H=5   (each hire drawn from the same age-at-start curve)")
seat = {}
for label, table in [("hire", HIRE), ("worker", WORKER)]:
    for g, r in table.items():
        t, S = knots_from_F(r["F"])
        vals = [renewal_hires(H, t, S) if H <= t[-1] else float("nan") for H in [1, 2, 5]]
        seat[(label, g)] = vals
        print(f"{label:<6} {g:<6}" + "".join(f"{v:7.2f}" if not np.isnan(v) else "      ." for v in vals))


# ---------------------------------------------------------------------------
# 6. A career model with unobserved mover types
# ---------------------------------------------------------------------------

section("6. Career model: gamma frailty on the exit hazard, calibrated to both estimands")

BANDS = [(18, 25), (25, 35), (35, 45), (45, 55), (55, 59)]
BAND_NAMES = ["18-24", "25-34", "35-44", "45-54", "55-58"]
N_PERSONS = 120_000
J_MAX = 160
GAP_MEAN = 0.35        # years between jobs (assumption)
RETIRE_MEAN, RETIRE_SD = 64.0, 3.0   # labour-force exit age (assumption)


class Draws:
    """Common random numbers, so the calibration is a smooth function of its parameters."""

    def __init__(self, n: int, seed: int, gap_mean: float = GAP_MEAN, retire_mean: float = RETIRE_MEAN):
        rng = np.random.default_rng(seed)
        self.u_frailty = rng.uniform(1e-9, 1 - 1e-9, n)
        self.e_job = rng.exponential(1.0, (n, J_MAX))
        self.gap = gap_mean * rng.exponential(1.0, (n, J_MAX + 1))
        self.retire = np.clip(retire_mean + RETIRE_SD * rng.standard_normal(n), 55.0, 75.0)
        self.birth = rng.uniform(1957.0, 1965.0, n)
        self.interview = rng.uniform(2022.5, 2023.5, n)


def frailty(draws: Draws, theta: float) -> np.ndarray:
    return stats.gamma.ppf(draws.u_frailty, a=1.0 / theta, scale=theta)


def band_of(age: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(np.array([25, 35, 45, 55]), age, side="right"), 0, 4)


def invert_cumhaz(x: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Duration d with cumulative base hazard x, for one band's piecewise-constant hazards (6 intervals)."""
    widths = np.diff(KNOT_YEARS)
    cum = np.concatenate([[0.0], np.cumsum(base[:5] * widths)])
    k = np.clip(np.searchsorted(cum, x, side="right") - 1, 0, 5)
    return KNOT_YEARS[k] + (x - cum[k]) / base[k]


def simulate(base: np.ndarray, theta: float, draws: Draws, observed: bool = True) -> dict:
    """Careers from 18. observed=True stops at the NLSY79 interview age; False runs to retirement."""
    n = len(draws.birth)
    Z = frailty(draws, theta)
    stop = np.minimum(draws.retire, draws.interview - draws.birth) if observed else draws.retire.copy()
    obs_end = (draws.interview - draws.birth) if observed else np.full(n, np.inf)
    s = 18.0 + draws.gap[:, 0]
    active = s < stop
    starts, ends, who, ongoing = [], [], [], []
    for j in range(J_MAX):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        sj = s[idx]
        bj = band_of(sj)
        x = draws.e_job[idx, j] / Z[idx]
        d = np.empty(idx.size)
        for b in range(5):
            m = bj == b
            if m.any():
                d[m] = invert_cumhaz(x[m], base[b])
        end = np.minimum(sj + d, draws.retire[idx])
        starts.append(sj)
        ends.append(end)
        who.append(idx)
        ongoing.append(end > obs_end[idx])
        s[idx] = end + draws.gap[idx, j + 1]
        active[idx] = s[idx] < stop[idx]
    capped = int(active.sum())
    return {"start": np.concatenate(starts), "end": np.concatenate(ends), "who": np.concatenate(who),
            "ongoing": np.concatenate(ongoing), "n": n, "capped": capped, "Z": Z}


def band_stats(sim: dict, draws: Draws, b: int, horizons=(1, 2, 5, 10, 15)) -> dict:
    lo, hi = BANDS[b]
    m = (sim["start"] >= lo) & (sim["start"] < hi)
    if b == 0:
        m &= draws.birth[sim["who"]] >= 1960.0   # BLS: excludes those who turned 18 before 1978
    if b == 4:
        m &= (draws.interview - draws.birth)[sim["who"]] >= 59.0   # BLS: excludes those not yet 59
    who = sim["who"][m]
    dur = sim["end"][m] - sim["start"][m]
    done = ~sim["ongoing"][m]
    n_jobs = np.bincount(who, minlength=sim["n"])
    has = n_jobs > 0
    out = {"pooled": [], "person": [], "jobs": int(m.sum()), "people": int(has.sum())}
    for H in horizons:
        ended = done & (dur < H)
        out["pooled"].append(ended.mean())
        per = np.bincount(who, weights=ended.astype(float), minlength=sim["n"])[has] / n_jobs[has]
        out["person"].append(per.mean())
    per_on = np.bincount(who, weights=(~done).astype(float), minlength=sim["n"])[has] / n_jobs[has]
    out["ongoing_pooled"] = (~done).mean()
    out["ongoing_person"] = per_on.mean()
    return out


# Calibration targets: (band, interval index k, estimand, S at knot k, S at knot k+1)
def S_of(r: dict, k: int) -> float:
    return 1.0 if k == 0 else 1.0 - r["F"][k - 1]


TARGETS = []
for b, g in [(0, "18-24"), (1, "25-34")]:
    for k in range(5):
        TARGETS.append((b, k, "pooled", S_of(HIRE[g], k), S_of(HIRE[g], k + 1)))
for k in range(3):
    TARGETS.append((2, k, "pooled", S_of(HIRE["35-44"], k), S_of(HIRE["35-44"], k + 1)))
for k in (3, 4):   # beyond 5 years only the worker-averaged cells exist
    TARGETS.append((2, k, "person", S_of(WORKER["35-44"], k), S_of(WORKER["35-44"], k + 1)))
for k in range(2):
    TARGETS.append((3, k, "pooled", S_of(HIRE["45-52"], k), S_of(HIRE["45-52"], k + 1)))
TARGETS.append((3, 2, "person", S_of(WORKER["45-54"], 2), S_of(WORKER["45-54"], 3)))
for k in range(2):
    TARGETS.append((4, k, "person", S_of(WORKER["55-58"], k), S_of(WORKER["55-58"], k + 1)))
# Past the last published knot the only information is the share of jobs still ongoing at the survey.
# One constant hazard per band (45-54 from 5 years, 55-58 from 2 years) is set to reproduce it.
ONGOING_TARGETS = [(3, 3, WORKER["45-54"]["ongoing"]), (4, 2, WORKER["55-58"]["ongoing"])]
THETA_TARGET = 1.0 - WORKER["25-34"]["F"][0]   # worker-averaged S(1) for 25-34


def fill_uncalibrated(base: np.ndarray) -> np.ndarray:
    base = base.copy()
    for b in range(3):
        base[b, 5] = base[b, 4]     # the 10-15 year hazard continues
    base[3, 4:] = base[3, 3]
    base[4, 3:] = base[4, 2]
    return base


def sim_S(st: dict, kind: str, k: int) -> float:
    return 1.0 if k == 0 else 1.0 - st[kind][k - 1]


def calibrate_base(theta: float, draws: Draws, base0: np.ndarray, iters: int = 14) -> tuple[np.ndarray, float]:
    base = fill_uncalibrated(base0)
    worst = np.inf
    for _ in range(iters):
        sim = simulate(base, theta, draws)
        st = {b: band_stats(sim, draws, b) for b in range(5)}
        worst = 0.0
        for b, k, kind, Sa, Sb in TARGETS:
            target_h = np.log(Sa / Sb) / (KNOT_YEARS[k + 1] - KNOT_YEARS[k])
            sa, sb = sim_S(st[b], kind, k), sim_S(st[b], kind, k + 1)
            sim_h = np.log(sa / sb) / (KNOT_YEARS[k + 1] - KNOT_YEARS[k])
            base[b, k] *= (target_h / sim_h) ** 0.9
            worst = max(worst, abs(sb - Sb))
        for b, k, target_on in ONGOING_TARGETS:
            on = st[b]["ongoing_person"]
            base[b, k] = max(base[b, k] * (on / target_on) ** 1.5, 1e-4)
            worst = max(worst, abs(on - target_on))
        base = fill_uncalibrated(base)
    return base, worst


def initial_base() -> np.ndarray:
    base = np.zeros((5, 6))
    for b, g in enumerate(["18-24", "25-34", "35-44", "45-52"]):
        t, S = knots_from_F(HIRE[g]["F"])
        h = interval_hazards(t, S)
        base[b, : len(h)] = h
    base[2, 3:5] = interval_hazards(*knots_from_F(WORKER["35-44"]["F"]))[3:5]
    t, S = knots_from_F(WORKER["55-58"]["F"])
    base[4, :2] = interval_hazards(t, S)
    base[3, 2] = interval_hazards(*knots_from_F(WORKER["45-54"]["F"]))[2]
    base[3, 3] = 0.7 * base[3, 2]
    base[4, 2] = base[3, 2]
    return fill_uncalibrated(base)


def fit_theta(d: Draws, base0: np.ndarray, steps: int, iters: int) -> tuple[float, np.ndarray, list]:
    """Bisection on the frailty variance so that the worker-averaged S(1) of 25-34 starters matches."""
    lo, hi = 0.001, 3.0
    base = base0
    trace = []
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        base, _ = calibrate_base(mid, d, base, iters=iters)
        s1 = 1.0 - band_stats(simulate(base, mid, d), d, 1)["person"][0]
        trace.append((mid, s1))
        if s1 < THETA_TARGET:
            lo = mid   # averaged survival too low: not enough heterogeneity between workers
        else:
            hi = mid
    return 0.5 * (lo + hi), base, trace


draws = Draws(N_PERSONS, seed=2026)
theta_hat, base_fit, trace = fit_theta(draws, initial_base(), steps=12, iters=6)
print("bisection on theta (frailty variance) -> worker-averaged S(1) for 25-34:")
for th, s1 in trace:
    print(f"  theta {th:.4f}: S(1) {100 * s1:.2f}%  (target {100 * THETA_TARGET:.1f}%)")

base_hat, cal_err = calibrate_base(theta_hat, draws, base_fit, iters=24)
sim_obs = simulate(base_hat, theta_hat, draws)
st_obs = {b: band_stats(sim_obs, draws, b) for b in range(5)}
print(f"\ntheta = {theta_hat:.3f}; largest calibration error {100 * cal_err:.2f} points; persons at job cap: {sim_obs['capped']}")
print("base hazards (per year, before frailty) by band and tenure interval [0,1) [1,2) [2,5) [5,10) [10,15) [15,inf):")
for b in range(5):
    print(f"  {BAND_NAMES[b]}: " + " ".join(f"{h:6.3f}" for h in base_hat[b]))

base_zero, _ = calibrate_base(0.001, draws, base_hat, iters=16)
st_zero = {b: band_stats(simulate(base_zero, 0.001, draws), draws, b) for b in range(5)}
print("\nThe averaging rule alone: identical workers (theta = 0.001), recalibrated to the same targets")
print("band    pooled <1y  worker-avg <1y (model)  worker-avg <1y (published 2025)")
for b, g in enumerate(BAND_NAMES):
    print(f"{g:<6}   {100 * st_zero[b]['pooled'][0]:6.1f}      {100 * st_zero[b]['person'][0]:6.1f}                 {100 * WORKER[g]['F'][0]:6.1f}")

print("\nModel vs published, both estimands (cumulative % ended; * = calibration target):")
targ_set = {(b, k + 1, kind) for b, k, kind, _, _ in TARGETS} | {(1, 1, "person")}   # + the theta target
print("band   estimand  knot   model  published")
check_rows = []
for b, g in enumerate(BAND_NAMES):
    for kind, table, gname in [("pooled", HIRE, {"45-54": "45-52"}.get(g, g)), ("person", WORKER, g)]:
        if gname not in table:
            continue
        F = table[gname]["F"]
        for k, H in enumerate([1, 2, 5, 10, 15]):
            if np.isnan(F[k]):
                continue
            star = "*" if (b, k + 1, kind) in targ_set else " "
            mod = st_obs[b][kind][k]
            check_rows.append((b, kind, H, mod, F[k], star == "*"))
            print(f"{g:<6} {kind:<8} {H:>3}y  {100 * mod:6.1f}  {100 * F[k]:6.1f} {star}")
oos = [(m, p) for _, _, _, m, p, is_t in check_rows if not is_t]
err = np.array([100 * (m - p) for m, p in oos])
print(f"\nOut-of-sample cells: {len(oos)}; mean absolute error {np.mean(np.abs(err)):.2f} points; largest {np.max(np.abs(err)):.2f}")
print("ongoing at the survey (model person-avg vs published 2025):")
for b, g in enumerate(BAND_NAMES):
    print(f"  {g}: {100 * st_obs[b]['ongoing_person']:.1f} vs {100 * WORKER[g]['ongoing']:.1f}")

print("\nJobs held per person in each age band (model vs BLS Table 1, 2025: 5.6, 4.5, 2.9, 2.2, 1.3):")
held = []
for b, (lo, hi) in enumerate(BANDS):
    overlap = (sim_obs["start"] < hi) & (sim_obs["end"] > lo)
    elig = np.ones(sim_obs["n"], bool)
    if b == 0:
        elig = draws.birth >= 1960.0
    if b == 4:
        elig = (draws.interview - draws.birth) >= 59.0
    cnt = np.bincount(sim_obs["who"][overlap], minlength=sim_obs["n"])[elig]
    held.append(cnt.mean())
print("  model: " + ", ".join(f"{h:.1f}" for h in held))

print("\nWhy the two estimands differ: jobs per person within the band, among people with a start there")
for b, g in enumerate(BAND_NAMES):
    st = st_obs[b]
    print(f"  {g}: {st['jobs'] / st['people']:.2f} starts per person; pooled vs worker-avg <1y "
          f"{100 * st['pooled'][0]:.1f} vs {100 * st['person'][0]:.1f}")

SCENARIOS = [(0.20, 64.0), (0.60, 64.0), (0.35, 62.0), (0.35, 66.0)]


# ---------------------------------------------------------------------------
# 7. Stock versus flow
# ---------------------------------------------------------------------------

section("7. Stock (median tenure of current jobs) versus flow (duration of new hires)")

cps = read_csv("cps_median_tenure.csv")
draws_life = Draws(N_PERSONS, seed=99)


def stock_medians(base: np.ndarray, theta: float, d: Draws, ages_per_person: int = 12, seed: int = 5) -> dict:
    sim = simulate(base, theta, d, observed=False)
    rng = np.random.default_rng(seed)
    order = np.lexsort((sim["start"], sim["who"]))
    who, st, en = sim["who"][order], sim["start"][order], sim["end"][order]
    key = who * 1000.0 + st          # sorted, because ages are below 1000
    out = {}
    groups = [(20, 25), (25, 35), (35, 45), (45, 55), (55, 65), (65, 75)]
    ages = rng.uniform(20.0, 75.0, (sim["n"], ages_per_person))
    tenures = {gr: [] for gr in groups}
    persons = np.arange(sim["n"])
    for p in range(ages_per_person):
        a = ages[:, p]
        # the last job that started at or before age a, for each person
        pos = np.searchsorted(key, persons * 1000.0 + a, side="right") - 1
        safe = np.clip(pos, 0, None)
        valid = (pos >= 0) & (who[safe] == persons)
        emp = valid & (en[safe] > a)
        ten = a - st[safe]
        for gr in groups:
            m = emp & (a >= gr[0]) & (a < gr[1])
            tenures[gr].append(ten[m])
    for gr in groups:
        out[gr] = float(np.median(np.concatenate(tenures[gr])))
    flows = {}
    for b, (lo, hi) in enumerate(BANDS):
        m = (sim["start"] >= lo) & (sim["start"] < hi)
        flows[BAND_NAMES[b]] = float(np.median(sim["end"][m] - sim["start"][m]))
    return {"stock": out, "flow": flows, "sim": sim}


life = stock_medians(base_hat, theta_hat, draws_life)
flat = base_hat.copy()
for b in range(5):
    flat[b] = base_hat[1]
life_flat = stock_medians(flat, theta_hat, draws_life)

print("age group   CPS Jan 2026   model (age-specific flows)   model (25-34 flows at every age)")
labels = {(20, 25): "20-24", (25, 35): "25-34", (35, 45): "35-44", (45, 55): "45-54", (55, 65): "55-64", (65, 75): "65+"}
cps26 = {r["age_group"]: float(r["jan2026"]) for r in cps}
for gr, name in labels.items():
    print(f"{name:<10}  {cps26[name]:8.1f}        {life['stock'][gr]:8.1f}                     {life_flat['stock'][gr]:8.1f}")

print("\nMedian duration of a new hire by age at start (model, followed to retirement):")
for g in BAND_NAMES:
    print(f"  {g}: {life['flow'][g]:.2f} years (flat-flow counterfactual {life_flat['flow'][g]:.2f})")
print("Data-only medians (piecewise-exponential through the knots):")
for label, table in [("hire", HIRE), ("worker", WORKER)]:
    for g, r in table.items():
        t, S = knots_from_F(r["F"])
        med = median_from_knots(t, S)
        print(f"  {label:<6} {g}: " + (f"{med:.2f} years" if not np.isnan(med) else f"more than {t[-1]:.0f} years"))


# ---------------------------------------------------------------------------
# 8. Decision quantities from the calibrated model
# ---------------------------------------------------------------------------

section("8. The employer's view from the calibrated model (hire-weighted, followed to retirement)")


def employer_view(sim: dict, seed: int = 31) -> dict:
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 10, 1001)
    out = {}
    for b, (lo, hi) in enumerate(BANDS):
        m = (sim["start"] >= lo) & (sim["start"] < hi)
        dur = np.sort(sim["end"][m] - sim["start"][m])
        Sg = 1.0 - np.searchsorted(dur, grid, side="right") / dur.size
        rm = {H: float(np.trapezoid(Sg[grid <= H], grid[grid <= H])) for H in (1, 2, 5, 10)}
        cum = np.cumsum(rng.choice(dur, size=(100_000, 80)), axis=1)
        seat_grid = np.arange(0.5, 10.01, 0.5)
        seat_curve = np.array([np.mean(1 + np.sum(cum < H, axis=1)) for H in seat_grid])
        seats = {H: float(seat_curve[np.searchsorted(seat_grid, H)]) for H in (2, 5, 10)}
        S_at = {x: float(Sg[np.searchsorted(grid, x)]) for x in (1, 2, 5, 10)}
        out[BAND_NAMES[b]] = {"grid": grid, "S": Sg, "S_at": S_at, "rmst": rm, "seats": seats,
                              "seat_grid": seat_grid, "seat_curve": seat_curve, "median": float(np.median(dur))}
    return out


view = employer_view(life["sim"])
print("band    S(1)   S(2)   S(5)  S(10) | RMST H=2  H=5  H=10 | hires per seat H=2  H=5  H=10")
for g, v in view.items():
    print(f"{g:<6}" + "".join(f"{100 * v['S_at'][x]:6.1f}" for x in (1, 2, 5, 10))
          + f" |      {v['rmst'][2]:.2f} {v['rmst'][5]:5.2f} {v['rmst'][10]:5.2f} |"
          + f"              {v['seats'][2]:.2f} {v['seats'][5]:5.2f} {v['seats'][10]:5.2f}")
for H in (2, 5, 10):
    print(f"  seats over {H:>2} years, 25-34 / 45-54: {view['25-34']['seats'][H] / view['45-54']['seats'][H]:.2f}; "
          f"25-34 / 55-58: {view['25-34']['seats'][H] / view['55-58']['seats'][H]:.2f}")


# ---------------------------------------------------------------------------
# 9. How much the conclusions depend on the two assumptions
# ---------------------------------------------------------------------------

section("9. Sensitivity: time between jobs and retirement age (everything re-calibrated)")
print("gap  retire  theta  | 45-54: S(5)  S(10)  RMST10  seats10 | 55-58: S(5)  S(10)  RMST10  seats10 | 25-34 RMST10 seats10")
sens_rows = []
for gm, rm_ in SCENARIOS:
    d2 = Draws(N_PERSONS // 2, seed=7, gap_mean=gm, retire_mean=rm_)
    th2, b2, _ = fit_theta(d2, base_hat, steps=8, iters=6)
    b2, err2 = calibrate_base(th2, d2, b2, iters=12)
    on2 = {b: band_stats(simulate(b2, th2, d2), d2, b)["ongoing_person"] for b in (3, 4)}
    v2 = employer_view(simulate(b2, th2, Draws(N_PERSONS // 2, seed=8, gap_mean=gm, retire_mean=rm_), observed=False))
    sens_rows.append((gm, rm_, th2, v2))
    o, s5, y = v2["45-54"], v2["55-58"], v2["25-34"]
    print(f"{gm:.2f}  {rm_:5.0f}  {th2:.3f}  |       {100 * o['S_at'][5]:5.1f}  {100 * o['S_at'][10]:5.1f}   {o['rmst'][10]:5.2f}    {o['seats'][10]:5.2f}"
          f" |       {100 * s5['S_at'][5]:5.1f}  {100 * s5['S_at'][10]:5.1f}   {s5['rmst'][10]:5.2f}    {s5['seats'][10]:5.2f}"
          f" |   {y['rmst'][10]:5.2f}   {y['seats'][10]:5.2f}   (cal. err {100 * err2:.2f}; ongoing 45-54 {100 * on2[3]:.1f}, 55-58 {100 * on2[4]:.1f}; post-knot base hazards {b2[3, 3]:.3f}, {b2[4, 2]:.4f})")

np.savez(RESULTS / "model.npz", base=base_hat, theta=theta_hat, grid=view["25-34"]["grid"],
         **{f"curve_{g}": v["S"] for g, v in view.items()},
         **{f"seats_{g}": v["seat_curve"] for g, v in view.items()}, seat_grid=view["25-34"]["seat_grid"],
         stock=np.array([life["stock"][gr] for gr in labels]),
         stock_flat=np.array([life_flat["stock"][gr] for gr in labels]),
         flow=np.array([life["flow"][g] for g in BAND_NAMES]),
         check=np.array([(b, 0 if kind == "pooled" else 1, H, m, p, t) for b, kind, H, m, p, t in check_rows]),
         zero_pooled=np.array([st_zero[b]["pooled"][0] for b in range(5)]),
         zero_person=np.array([st_zero[b]["person"][0] for b in range(5)]))
print("\nSaved results/model.npz for figures.py")
