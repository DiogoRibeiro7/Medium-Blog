"""Print every number the article uses, and write the tables behind the figures to results/.

python hours_numbers.py > output.txt
"""

import csv
import math
from pathlib import Path

import numpy as np

import hours_core as hc

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RES = HERE / "results"
END = hc.q(hc.LAST)
Y0, Y1 = 2019, 2025  # the comparison window: last full pre-pandemic year and last full year


def load(name, **keep):
    return hc.load(DATA / f"{name}.csv", **keep)


def write(name, header, rows):
    RES.mkdir(exist_ok=True)
    with open(RES / name, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def fmt(x, d=2, sign=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


GVA, _ = load("gva")
H_NA, _ = load("hours_na")
H_LFS, H_LFS_FLAGS = load("hours_lfs", age="Y15-74")
H_LFS_2064, _ = load("hours_lfs", age="Y20-64")
N_NA, _ = load("employment_na", na_item="EMP_DC")
N_NC, _ = load("employment_na", na_item="EMP_NC")
N_LFS, _ = load("employment_lfs")
ABS_LFS, _ = load("absences_lfs")
POP_LFS, _ = load("population_lfs", citizen="TOTAL", wstatus="POP")
EMP_LFS_ALL, _ = load("population_lfs", citizen="TOTAL", wstatus="EMP")
EMP_LFS_FOR, _ = load("population_lfs", citizen="FOR", wstatus="EMP")
PROD_ES, _ = load("productivity_eurostat")
DEMO_GROUPS, _ = {}, {}
with open(DATA / "population_demo.csv", newline="") as f:
    for r in csv.DictReader(f):
        DEMO_GROUPS.setdefault(r["country"], {}).setdefault(int(r["period"]), {})[r["age"]] = float(r["value"])
DEMO = {
    c: {y: math.fsum(g.values()) for y, g in ys.items() if len(g) == len(hc.DEMO_GROUPS)}
    for c, ys in DEMO_GROUPS.items()
}
ALL = (hc.EU, *hc.MEMBERS)


def productivity(c):
    """(output per hour with national-accounts hours, with LFS hours), both indexed 2009 = 100."""
    na = hc.index_to(hc.ratio(GVA[c], H_NA[c]))
    lfs = hc.index_to(hc.ratio(GVA[c], H_LFS[c])) if c in H_LFS else {}
    return na, lfs


def pgrowth(c, y0, y1, lfs=False, hours=None):
    """Growth of annual output per hour, % a year, with NA hours, LFS hours or a supplied hours series."""
    h = hours if hours is not None else (H_LFS[c] if lfs else H_NA[c])
    return hc.growth_ratio(GVA[c], h, y0, y1)


def log_gap(c, lfs=None):
    """ln(LFS hours / NA hours), both indexed 2009 = 100: zero in 2009 by construction."""
    lfs = H_LFS[c] if lfs is None else lfs
    a, b = hc.index_to(lfs), hc.index_to(H_NA[c])
    return {k: math.log(a[k] / b[k]) for k in a if k in b}


# ---------------------------------------------------------------------------------------------------------------
print("1. Data")
print(
    f"   national accounts: {hc.NA_FIRST} to {hc.LAST}; LFS: {hc.LFS_FIRST} to {hc.LAST}; snapshot of 21 November 2026"
)
for c in ALL:
    g, h, lf = GVA.get(c, {}), H_NA.get(c, {}), H_LFS.get(c, {})
    breaks = sorted(hc.label(k) for k, v in H_LFS_FLAGS.get(c, {}).items() if "b" in v)
    missing = [hc.label(k) for k in range(hc.q(hc.LFS_FIRST), END + 1) if k not in lf]
    print(
        f"   {c:<9} {hc.NAMES[c]:<12} GVA {hc.label(min(g))}..{hc.label(max(g))}  hours NA {hc.label(min(h))}.."
        f"{hc.label(max(h))}  LFS {len(lf)} quarters"
        + (f", missing {', '.join(missing)}" if missing else "")
        + (f", breaks {', '.join(breaks)}" if breaks else "")
    )

# ---------------------------------------------------------------------------------------------------------------
print("\n2. EU-27: output per hour under the two denominators (real GVA per hour, 2009 = 100)")
na, lfs = productivity(hc.EU)
trend_g, trend = hc.log_trend(na, hc.q(hc.TREND[0]), hc.q(hc.TREND[1]))
trend_idx = {k: trend(k) for k in na if k >= hc.q("1997-Q1")}
print(f"   level in {hc.LAST}: national accounts {na[END]:.1f}, LFS {lfs[END]:.1f}, 1997-2007 trend {trend(END):.1f}")
print(f"   trend growth 1997-Q1..2007-Q4 (national accounts): {trend_g:.2f}% a year")
for y0, y1 in ((1997, 2007), (2009, 2019), (Y0, Y1), (2009, Y1)):
    print(
        f"   growth {y0}-{y1}: national accounts {fmt(pgrowth(hc.EU, y0, y1))}% a year, "
        f"LFS {fmt(pgrowth(hc.EU, y0, y1, lfs=True)) if y0 >= 2009 else 'n/a'}% a year"
    )
print(
    f"   last two years (four quarters to {hc.LAST} vs four quarters to {hc.label(END - 8)}): national accounts "
    f"{fmt(hc.growth_last_years(na, END))}% a year, LFS {fmt(hc.growth_last_years(lfs, END))}% a year"
)
print(
    f"   distance to the 1997-2007 trend in {Y1} (annual means): national accounts "
    f"{100 * (hc.annual_mean(na, Y1) / hc.annual_mean(trend_idx, Y1) - 1):+.1f}%, LFS "
    f"{100 * (hc.annual_mean(lfs, Y1) / hc.annual_mean(trend_idx, Y1) - 1):+.1f}%"
)
g_gva = 100 * (hc.annual_mean(GVA[hc.EU], Y1) / hc.annual_mean(GVA[hc.EU], Y0) - 1)
g_hna = 100 * (hc.annual_mean(H_NA[hc.EU], Y1) / hc.annual_mean(H_NA[hc.EU], Y0) - 1)
g_hlf = 100 * (hc.annual_mean(H_LFS[hc.EU], Y1) / hc.annual_mean(H_LFS[hc.EU], Y0) - 1)
g_hna_09 = 100 * (hc.annual_mean(H_NA[hc.EU], Y1) / hc.annual_mean(H_NA[hc.EU], 2009) - 1)
g_hlf_09 = 100 * (hc.annual_mean(H_LFS[hc.EU], Y1) / hc.annual_mean(H_LFS[hc.EU], 2009) - 1)
print(f"   cumulative {Y0}-{Y1}: real GVA {g_gva:+.1f}%, hours NA {g_hna:+.1f}%, hours LFS {g_hlf:+.1f}%")
print(f"   cumulative 2009-{Y1}: hours NA {g_hna_09:+.1f}%, hours LFS {g_hlf_09:+.1f}%")
d_prod = (Y1 - Y0) * (math.log(1 + pgrowth(hc.EU, Y0, Y1, lfs=True) / 100) - math.log(1 + pgrowth(hc.EU, Y0, Y1) / 100))
d_hours = math.log(hc.annual_mean(H_NA[hc.EU], Y1) / hc.annual_mean(H_NA[hc.EU], Y0)) - math.log(
    hc.annual_mean(H_LFS[hc.EU], Y1) / hc.annual_mean(H_LFS[hc.EU], Y0)
)
print(f"   identity check: log productivity gap {100 * d_prod:.4f} = log hours gap {100 * d_hours:.4f} (log points)")
r_es = hc.index_to(PROD_ES[hc.EU])
print(
    f"   Eurostat's own series (real GDP per hour, namq_10_lp_ulc): 2009-2019 {fmt(hc.growth(r_es, 2009, 2019))}%, "
    f"{Y0}-{Y1} {fmt(hc.growth(r_es, Y0, Y1))}% a year"
)
print(
    f"   LFS hours for ages 20-64 instead of 15-74: {Y0}-{Y1} growth "
    f"{fmt(pgrowth(hc.EU, Y0, Y1, hours=H_LFS_2064[hc.EU]))}% a year"
)
write(
    "eu27_paths.csv",
    ["quarter", "prod_na", "prod_lfs", "trend", "hours_na", "hours_lfs"],
    [
        [
            hc.label(k),
            f"{na[k]:.3f}",
            f"{lfs[k]:.3f}" if k in lfs else "",
            f"{trend_idx[k]:.3f}",
            f"{hc.index_to(H_NA[hc.EU])[k]:.3f}",
            f"{hc.index_to(H_LFS[hc.EU])[k]:.3f}" if k in H_LFS[hc.EU] else "",
        ]
        for k in sorted(trend_idx)
        if k <= END
    ],
)

# ---------------------------------------------------------------------------------------------------------------
print(f"\n3. Countries: growth of output per hour {Y0}-{Y1}, % a year")
rows, gna, glf = [], {}, {}
for c in ALL:
    na, lfs = productivity(c)
    a, b = pgrowth(c, Y0, Y1), pgrowth(c, Y0, Y1, lfs=True)
    hn = 100 * (hc.annual_mean(H_NA[c], Y1) / hc.annual_mean(H_NA[c], Y0) - 1)
    hl = 100 * (hc.annual_mean(H_LFS[c], Y1) / hc.annual_mean(H_LFS[c], Y0) - 1)
    t, _ = hc.log_trend(na, hc.q(hc.TREND[0]), hc.q(hc.TREND[1]))
    n0919 = pgrowth(c, 2009, 2019)
    l0919 = pgrowth(c, 2009, 2019, lfs=True)
    rows.append([c, f"{a:.3f}", f"{b:.3f}", f"{b - a:.3f}", f"{hn:.2f}", f"{hl:.2f}", f"{t:.3f}"])
    if c != hc.EU:
        gna[c], glf[c] = a, b
    print(
        f"   {c:<9} NA {a:+.2f}  LFS {b:+.2f}  gap {b - a:+.2f} | hours NA {hn:+.1f}%  LFS {hl:+.1f}% | "
        f"2009-2019 NA {n0919:+.2f} LFS {l0919:+.2f} | trend 1997-2007 {t:+.2f}"
    )
write(
    "countries_2019_2025.csv",
    ["country", "prod_na", "prod_lfs", "gap", "hours_na_pct", "hours_lfs_pct", "trend_1997_2007"],
    rows,
)
gaps = {c: glf[c] - gna[c] for c in gna}
print(f"   LFS growth above national accounts growth in {sum(v > 0 for v in gaps.values())} of {len(gaps)} countries")
print(f"   median gap {np.median(list(gaps.values())):+.2f} points a year")
big = sorted(gaps, key=lambda c: -gaps[c])
print("   largest gaps: " + ", ".join(f"{c} {gaps[c]:+.2f}" for c in big[:6]))
print("   gaps below zero: " + ", ".join(f"{c} {gaps[c]:+.2f}" for c in big if gaps[c] < 0))
stag_na = {c for c in gna if gna[c] < hc.STAGNATION}
stag_lf = {c for c in glf if glf[c] < hc.STAGNATION}
print(
    f"   below {hc.STAGNATION}% a year: {len(stag_na)} countries with NA hours ({', '.join(sorted(stag_na))}), "
    f"{len(stag_lf)} with LFS hours ({', '.join(sorted(stag_lf))})"
)
print(f"   classified differently: {', '.join(sorted(stag_na ^ stag_lf))}")
neg_na = {c for c in gna if gna[c] < 0}
neg_lf = {c for c in glf if glf[c] < 0}
print(f"   negative growth: NA {sorted(neg_na)}, LFS {sorted(neg_lf)}")
keys = sorted(gna)
rho = hc.spearman([gna[c] for c in keys], [glf[c] for c in keys])
r_na, r_lf = hc.ranks(gna), hc.ranks(glf)
moves = sorted(keys, key=lambda c: -abs(r_na[c] - r_lf[c]))
print(f"   Spearman rank correlation NA vs LFS ranking: {rho:.3f}")
print("   largest rank moves (NA rank -> LFS rank): " + ", ".join(f"{c} {r_na[c]}->{r_lf[c]}" for c in moves[:8]))
print(f"   countries moving five or more places: {sum(abs(r_na[c] - r_lf[c]) >= 5 for c in keys)}")
write("ranks.csv", ["country", "rank_na", "rank_lfs"], [[c, r_na[c], r_lf[c]] for c in keys])

# ---------------------------------------------------------------------------------------------------------------
print("\n4. Drift in ln(LFS hours / NA hours), per cent a year, Newey-West standard errors with 4 lags")
print("   pre-pandemic window 2009-Q1..2019-Q4; post-break window 2021-Q2..2026-Q2")
drift_rows = {}
for c in ALL:
    g = log_gap(c)
    out = []
    for a, b in ((hc.q("2009-Q1"), hc.q("2019-Q4")), (hc.q("2021-Q2"), END)):
        ks = [k for k in sorted(g) if a <= k <= b]
        s, se = hc.hac_slope([k / 4 for k in ks], [100 * g[k] for k in ks])
        out.append((s, se, len(ks)))
    drift_rows[c] = out
    (s1, e1, n1), (s2, e2, n2) = out
    print(
        f"   {c:<9} 2009-2019 {s1:+.2f} (se {e1:.2f}, t {s1 / e1:+.1f}, n {n1})   "
        f"2021-2026 {s2:+.2f} (se {e2:.2f}, t {s2 / e2:+.1f}, n {n2})"
    )
g_eu = log_gap(hc.EU)
print(
    f"   EU-27 log ratio, annual mean: 2019 {100 * hc.annual_mean(g_eu, 2019):+.2f}, "
    f"{Y1} {100 * hc.annual_mean(g_eu, Y1):+.2f}"
)
neg_pre = [c for c in hc.MEMBERS if drift_rows[c][0][0] / drift_rows[c][0][1] < -2]
neg_post = [c for c in hc.MEMBERS if drift_rows[c][1][0] / drift_rows[c][1][1] < -2]
pos_pre = [c for c in hc.MEMBERS if drift_rows[c][0][0] / drift_rows[c][0][1] > 2]
pos_post = [c for c in hc.MEMBERS if drift_rows[c][1][0] / drift_rows[c][1][1] > 2]
print(
    f"   t below -2 (LFS hours falling behind): 2009-2019 {len(neg_pre)} {neg_pre}; "
    f"2021-2026 {len(neg_post)} {neg_post}"
)
print(
    f"   t above +2 (LFS hours pulling ahead): 2009-2019 {len(pos_pre)} {pos_pre}; 2021-2026 {len(pos_post)} {pos_post}"
)
write(
    "drift.csv",
    ["country", "pre_slope", "pre_se", "post_slope", "post_se"],
    [[c, *(f"{v:.4f}" for v in (*drift_rows[c][0][:2], *drift_rows[c][1][:2]))] for c in ALL],
)

# ---------------------------------------------------------------------------------------------------------------
print(f"\n5. The {hc.BREAK} break: step in ln(LFS hours / NA hours) and what removing it does to {Y0}-{Y1}")
print("   step = change from the last observed quarter before the break to the break quarter;")
print("   noise = standard deviation of quarterly changes 2009-Q2..2019-Q4")
br = hc.q(hc.BREAK)
adj_rows, adj_growth = [], {}
for c in ALL:
    g = log_gap(c)
    prev = max(k for k in g if k < br)
    step = g[br] - g[prev]
    diffs = [g[k] - g[k - 1] for k in range(hc.q("2009-Q2"), hc.q("2019-Q4") + 1) if k in g and k - 1 in g]
    sd = float(np.std(diffs, ddof=1))
    # remove the whole step: scale every LFS quarter from the break on by exp(-step)
    lfs_adj = {k: (v * math.exp(-step) if k >= br else v) for k, v in H_LFS[c].items()}
    a = pgrowth(c, Y0, Y1)
    b = pgrowth(c, Y0, Y1, lfs=True)
    b_adj = pgrowth(c, Y0, Y1, hours=lfs_adj)
    adj_growth[c] = b_adj
    adj_rows.append([c, f"{100 * step:.3f}", f"{100 * sd:.3f}", f"{b_adj:.3f}"])
    print(
        f"   {c:<9} step {100 * step:+.2f} log points from {hc.label(prev)} (noise sd {100 * sd:.2f}, ratio "
        f"{step / sd:+.1f}) | {Y0}-{Y1}: NA {a:+.2f}, LFS {b:+.2f}, LFS without the step {b_adj:+.2f}"
    )
write("break_adjusted.csv", ["country", "step_logpts", "noise_sd_logpts", "prod_lfs_adjusted"], adj_rows)
still = [c for c in hc.MEMBERS if adj_growth[c] > gna[c]]
print(f"   with the step removed, LFS growth still above NA growth in {len(still)} of {len(hc.MEMBERS)} countries")
print(f"   EU-27 without the step: LFS {adj_growth[hc.EU]:+.2f}% a year against NA {pgrowth(hc.EU, Y0, Y1):+.2f}%")

# ---------------------------------------------------------------------------------------------------------------
print(f"\n6. Decomposing the change in ln(LFS hours / NA hours) from {Y0} to {Y1} (annual means, log points)")
print("   persons part: ln(LFS employment / NA employment); hours-per-worker part: the rest")
dec_rows = {}
for c in ALL:

    def am(s, y):
        return hc.annual_mean(s, y)

    if c not in N_NA:
        continue
    total = math.log(am(H_LFS[c], Y1) / am(H_LFS[c], Y0)) - math.log(am(H_NA[c], Y1) / am(H_NA[c], Y0))
    persons = math.log(am(N_LFS[c], Y1) / am(N_LFS[c], Y0)) - math.log(am(N_NA[c], Y1) / am(N_NA[c], Y0))
    border = (
        math.log(am(N_NC[c], Y1) / am(N_NC[c], Y0)) - math.log(am(N_NA[c], Y1) / am(N_NA[c], Y0))
        if c in N_NC and c in N_NA
        else float("nan")
    )
    dec_rows[c] = (100 * total, 100 * persons, 100 * (total - persons), 100 * border)
    print(
        f"   {c:<9} total {100 * total:+6.2f} = persons {100 * persons:+6.2f} + hours per worker "
        f"{100 * (total - persons):+6.2f} | national minus domestic employment {100 * border:+5.2f}"
    )
print("\n   the same change by source: growth from 2019 to 2025 in log points")
src_rows = []
for c in ALL:
    am = hc.annual_mean
    nn = 100 * math.log(am(N_NA[c], Y1) / am(N_NA[c], Y0))
    nl = 100 * math.log(am(N_LFS[c], Y1) / am(N_LFS[c], Y0))
    hn = 100 * math.log(am(H_NA[c], Y1) / am(H_NA[c], Y0))
    hl = 100 * math.log(am(H_LFS[c], Y1) / am(H_LFS[c], Y0))
    src_rows.append([c, *(f"{v:.3f}" for v in (nn, nl, hn - nn, hl - nl))])
    print(
        f"   {c:<9} employed persons NA {nn:+6.2f} LFS {nl:+6.2f} | hours per worker NA {hn - nn:+6.2f} "
        f"LFS {hl - nl:+6.2f}"
    )
write("by_source.csv", ["country", "persons_na", "persons_lfs", "per_worker_na", "per_worker_lfs"], src_rows)
write(
    "decomposition.csv",
    ["country", "total", "persons", "hours_per_worker", "national_minus_domestic"],
    [[c, *(f"{v:.3f}" for v in dec_rows[c])] for c in ALL if c in dec_rows],
)
members = [c for c in hc.MEMBERS]
share_persons = [c for c in members if abs(dec_rows[c][1]) > abs(dec_rows[c][2])]
print(f"   persons part larger in absolute value than hours-per-worker part in {len(share_persons)} of {len(members)}")
tot = np.array([dec_rows[c][0] for c in members])
per = np.array([dec_rows[c][1] for c in members])
print(f"   correlation across countries between the total and the persons part: {np.corrcoef(tot, per)[0, 1]:.2f}")
print(
    "   national minus domestic employment, largest: "
    + ", ".join(f"{c} {dec_rows[c][3]:+.2f}" for c in sorted(members, key=lambda c: -abs(dec_rows[c][3]))[:4])
)

# ---------------------------------------------------------------------------------------------------------------
print(f"\n7. Inside the survey: LFS hours per worker = (1 - share absent) x hours per person at work, {Y0} to {Y1}")
abs_rows = {}
for c in ALL:
    if c not in ABS_LFS:
        continue
    am = hc.annual_mean
    a0, a1 = am(ABS_LFS[c], Y0), am(ABS_LFS[c], Y1)
    if a0 is None or a1 is None:
        continue
    per_worker = 100 * (math.log(am(H_LFS[c], Y1) / am(H_LFS[c], Y0)) - math.log(am(N_LFS[c], Y1) / am(N_LFS[c], Y0)))
    absence = 100 * math.log((1 - a1 / 100) / (1 - a0 / 100))
    abs_rows[c] = (a0, a1, per_worker, absence, per_worker - absence)
    print(
        f"   {c:<9} absent {a0:4.1f}% -> {a1:4.1f}% | hours per worker {per_worker:+6.2f} = absence {absence:+5.2f} "
        f"+ hours at work {per_worker - absence:+6.2f}"
    )
write(
    "absences.csv",
    ["country", "absent_y0", "absent_y1", "per_worker", "absence_part", "at_work_part"],
    [[c, *(f"{v:.3f}" for v in abs_rows[c])] for c in abs_rows],
)
up = [c for c in hc.MEMBERS if c in abs_rows and abs_rows[c][1] > abs_rows[c][0]]
print(f"   share absent higher in {Y1} than in {Y0} in {len(up)} of {sum(c in abs_rows for c in hc.MEMBERS)} countries")

# ---------------------------------------------------------------------------------------------------------------
print("\n8. Population behind the survey: LFS population aged 15-74 against Eurostat's demographic estimate")
print(f"   LFS: first-quarter population in private households; demographic: population on 1 January; {Y0} to {Y1}")
pop_rows = {}
for c in ALL:
    if c not in POP_LFS or c not in DEMO:
        continue
    a0, a1 = POP_LFS[c].get(4 * Y0), POP_LFS[c].get(4 * Y1)
    d0, d1 = DEMO[c].get(Y0), DEMO[c].get(Y1)
    if None in (a0, a1, d0, d1):
        continue
    lfs_g, demo_g = 100 * math.log(a1 / a0), 100 * math.log(d1 / d0)
    f0 = EMP_LFS_FOR[c].get(4 * Y0) if c in EMP_LFS_FOR else None
    f1 = EMP_LFS_FOR[c].get(4 * Y1) if c in EMP_LFS_FOR else None
    e0, e1 = EMP_LFS_ALL[c].get(4 * Y0), EMP_LFS_ALL[c].get(4 * Y1)
    sh0 = 100 * f0 / e0 if f0 and e0 else None
    sh1 = 100 * f1 / e1 if f1 and e1 else None
    pop_rows[c] = (lfs_g, demo_g, lfs_g - demo_g, sh0, sh1)
    print(
        f"   {c:<9} LFS {lfs_g:+5.2f}  demographic {demo_g:+5.2f}  difference {lfs_g - demo_g:+5.2f} log points | "
        f"foreign citizens in LFS employment {fmt(sh0, 1, False)}% -> {fmt(sh1, 1, False)}%"
    )
write(
    "population.csv",
    ["country", "lfs_pop_logpts", "demo_pop_logpts", "difference", "foreign_share_y0", "foreign_share_y1"],
    [[c, *("" if v is None else f"{v:.3f}" for v in pop_rows[c])] for c in pop_rows],
)
pc = [c for c in hc.MEMBERS if c in pop_rows]
x = np.array([pop_rows[c][2] for c in pc])
y = np.array([dec_rows[c][1] for c in pc])
print(f"   countries with both: {len(pc)}; LFS population below demographic in {int((x < 0).sum())}")
print(f"   correlation between the population difference and the persons part: {np.corrcoef(x, y)[0, 1]:.2f}")
slope, intercept = np.polyfit(x, y, 1)
print(f"   least-squares slope of persons part on population difference: {slope:.2f} (intercept {intercept:+.2f})")

# ---------------------------------------------------------------------------------------------------------------
print("\n9. Check against Eurostat's published productivity per hour (real GDP per hour), % a year")
for c in (hc.EU, "DE", "FR", "IT", "ES", "NL", "PL", "PT", "CZ"):
    theirs = hc.index_to(PROD_ES[c])
    print(
        f"   {c:<9} ours (GVA) 2009-2019 {pgrowth(c, 2009, 2019):+.2f}, {Y0}-{Y1} {pgrowth(c, Y0, Y1):+.2f} | "
        f"Eurostat (GDP) {hc.growth(theirs, 2009, 2019):+.2f}, {hc.growth(theirs, Y0, Y1):+.2f}"
    )

# ---------------------------------------------------------------------------------------------------------------
print("\n10. Index levels quoted for Portugal and Czechia (annual means of quarterly GVA per hour, 2009 = 100)")
for c in ("PT", "CZ"):
    na, lfs = productivity(c)
    print(
        f"   {c}: national accounts "
        + ", ".join(f"{y} {hc.annual_mean(na, y):.1f}" for y in (2014, 2015, 2016, 2019, 2021, Y1))
        + " | LFS "
        + ", ".join(f"{y} {hc.annual_mean(lfs, y):.1f}" for y in (2019, 2021, Y1))
    )
