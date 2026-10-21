"""How many runs does an agent comparison need? Variance components from 6,428 released tau-bench runs.

Prints every number the article quotes and draws its four figures into figures/.
Reads data/runs.csv (built by fetch_data.py) and calls no model. All randomness comes
from one seeded generator, so a rerun with the pinned versions reproduces results/output.txt.

Sections
  1  The data, and a check that it reproduces the published pass^k leaderboard
  2  Where the run-to-run noise sits: variance components and the U-shaped pass counts
  3  Two questions, three standard errors, for 20 paired model comparisons
  4  Checks on held-out runs: the task-level standard error and the simulator
  5  Three runs and a bootstrap: coverage of five interval methods
  6  How many runs, and how many dollars
"""
import collections
import csv
import itertools
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(20261021)
Z95, Z80 = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
K = Z95 + Z80                                            # 2.80: a difference this many standard errors wide is found 80% of the time

# chart ink and the first three categorical slots of the reference palette
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "axes.titlecolor": INK, "axes.titleweight": "bold", "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight"})


def section(title):
    print(f"\n== {title} ==")


# ---------------------------------------------------------------------------------------------------------------
# Load: one 0/1 matrix per (benchmark, domain, model), tasks in rows and repeated runs in columns
# ---------------------------------------------------------------------------------------------------------------
cells, costs = collections.defaultdict(lambda: collections.defaultdict(dict)), collections.defaultdict(list)
with open(HERE / "data" / "runs.csv") as f:
    for r in csv.DictReader(f):
        key = (r["benchmark"], r["domain"], r["model"])
        cells[key][r["task_id"]][int(r["trial"])] = int(r["reward"])
        if r["agent_cost"]:
            costs[key].append(float(r["agent_cost"]) + float(r["user_cost"]))

Y, TASKS = {}, {}
for key, tasks in cells.items():
    ids = sorted(tasks, key=lambda t: (0, int(t), "") if t.isdigit() else (1, 0, t))
    TASKS[key] = ids
    Y[key] = np.array([[tasks[t][j] for j in sorted(tasks[t])] for t in ids], dtype=float)
COST = {k: float(np.mean(v)) for k, v in costs.items()}  # dollars per run, agent plus simulated user
CELLS = sorted(Y)
DOMAINS = sorted({k[:2] for k in CELLS})
PAIRS = [(b, d, a, c) for (b, d) in DOMAINS for a, c in itertools.combinations(sorted(k[2] for k in CELLS if k[:2] == (b, d)), 2)]
SHORT = {"claude-3.5-sonnet": "Sonnet 3.5", "gpt-4o": "GPT-4o", "claude-3.7-sonnet": "Sonnet 3.7", "gpt-4.1": "GPT-4.1",
         "gpt-4.1-mini": "GPT-4.1-mini", "o4-mini": "o4-mini"}


def within(y):
    """Unbiased run-to-run variance of each task, p(1 - p), from its repeated 0/1 outcomes."""
    return y.var(axis=1, ddof=1)


def components(y):
    """Mean pass rate, average within-task variance and between-task variance of the true pass rates."""
    n, r = y.shape
    sw = within(y).mean()
    sb = y.mean(1).var(ddof=1) - sw / r                  # the spread of task means, less the part run noise explains
    return y.mean(), sw, sb


def pair_stats(y_a, y_b):
    """Paired comparison of two models on the same tasks: the difference and its three standard errors."""
    n, ra = y_a.shape
    rb = y_b.shape[1]
    d_i = y_a.mean(1) - y_b.mean(1)                      # per-task difference in pass rate
    noise = within(y_a) / ra + within(y_b) / rb          # run noise in each task's difference
    se_fixed = np.sqrt(noise.sum()) / n                  # question 1: better on these exact tasks
    se_pop = d_i.std(ddof=1) / np.sqrt(n)                # question 2: better on tasks drawn like these
    pa, pb = y_a.mean(), y_b.mean()
    se_naive = np.sqrt(pa * (1 - pa) / (n * ra) + pb * (1 - pb) / (n * rb))   # every run an independent coin flip
    w = (within(y_a) + within(y_b)).mean()               # W: run variance of one run of each model, averaged over tasks
    s2_delta = max(d_i.var(ddof=1) - noise.mean(), 0.0)  # sigma_Delta^2: how much the true per-task difference varies
    return dict(n=n, d=d_i.mean(), se_fixed=se_fixed, se_pop=se_pop, se_naive=se_naive, w=w, s2_delta=s2_delta)


# ---------------------------------------------------------------------------------------------------------------
section("1. The data")
# ---------------------------------------------------------------------------------------------------------------
print(f"{sum(y.size for y in Y.values())} runs, {sum(y.shape[0] for k, y in Y.items())} task-model cells over "
      f"{len(CELLS)} model-domain cells and {len(PAIRS)} paired comparisons")
tau2_dollars = sum(sum(v) for v in costs.values())
print(f"tau2-bench runs: {sum(len(v) for v in costs.values())}, released at a recorded cost of ${tau2_dollars:,.0f} (agent plus simulated user)")
for k in CELLS:
    y = Y[k]
    p = y.mean(1)
    print(f"{k[0]:<10} {k[1]:<8} {k[2]:<18} tasks {y.shape[0]:>3}  runs/task {y.shape[1]}  pass rate {y.mean():.3f}  "
          f"always pass {np.mean(p == 1):.2f}  always fail {np.mean(p == 0):.2f}  mixed {np.mean((p > 0) & (p < 1)):.2f}"
          + (f"  ${COST[k]:.3f}/run" if k in COST else ""))


def pass_hat_k(y, k):
    """Unbiased pass^k from tau-bench: the chance that k fresh runs of a task all pass, averaged over tasks."""
    r, c = y.shape[1], y.sum(1).astype(int)
    return np.mean([comb(ci, k) / comb(r, k) for ci in c])


print("check against the tau-bench README leaderboard (pass^1..pass^4, four decimals):")
for k in [c for c in CELLS if c[0] == "tau-bench"]:
    print(f"  {k[1]:<8} {k[2]:<18} " + "  ".join(f"{pass_hat_k(Y[k], j):.4f}" for j in range(1, 5)))

# ---------------------------------------------------------------------------------------------------------------
section("2. Where the run-to-run noise sits")
# ---------------------------------------------------------------------------------------------------------------
icc = []
for k in CELLS:
    m, sw, sb = components(Y[k])
    icc.append(sb / (sb + sw))
    print(f"{k[0]:<10} {k[1]:<8} {k[2]:<18} within-task var {sw:.3f}  between-task var {sb:.3f}  share from the task {sb / (sb + sw):.2f}")
print(f"share of one run's variance that belongs to the task: {min(icc):.2f} to {max(icc):.2f}, median {np.median(icc):.2f}")

# pooled over the 12 tau2-bench cells (4 runs each): passes out of 4, observed and under independent runs at the cell's rate
obs, exp = np.zeros(5), np.zeros(5)
for k in [c for c in CELLS if c[0] == "tau2-bench"]:
    y = Y[k]
    obs += np.bincount(y.sum(1).astype(int), minlength=5)
    exp += y.shape[0] * stats.binom.pmf(np.arange(5), 4, y.mean())
obs, exp = obs / obs.sum(), exp / exp.sum()
print("passes out of 4, share of task-model cells (tau2-bench, 1,112 cells):")
for j in range(5):
    print(f"  {j}/4  observed {obs[j]:.3f}   if every run were an independent coin flip at the model's rate {exp[j]:.3f}")
print(f"all or nothing (0/4 or 4/4): observed {obs[0] + obs[4]:.2f}, independent-runs model {exp[0] + exp[4]:.2f}")

fig, ax = plt.subplots(figsize=(7.2, 3.9))
x = np.arange(5)
ax.bar(x - 0.2, obs, 0.38, color=BLUE, label="Observed", zorder=3)
ax.bar(x + 0.2, exp, 0.38, color="#c3c2b7", label="If runs were independent coin flips", zorder=3)
for xi, (o, e) in enumerate(zip(obs, exp)):
    ax.text(xi - 0.2, o + 0.008, f"{o:.0%}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.text(xi + 0.2, e + 0.008, f"{e:.0%}", ha="center", va="bottom", fontsize=8.5, color=INK2)
ax.set_xticks(x, [f"{j} of 4" for j in range(5)])
ax.set_xlabel("Runs that passed, out of four repeats of the same task")
ax.set_ylabel("Share of task-model cells")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=9)
ax.set_ylim(0, max(obs.max(), exp.max()) * 1.22)
ax.set_title("Most tasks pass every time or fail every time", loc="left")
fig.text(0.0, -0.04, "Data: tau2-bench released runs (Sierra Research, MIT), 4 models x 3 domains, 1,112 task-model cells. Figure: Diogo Ribeiro.",
         fontsize=7.5, color=MUTED)
fig.savefig(FIG / "fig1_pass_counts.png")
plt.close(fig)

# ---------------------------------------------------------------------------------------------------------------
section("3. Two questions, three standard errors")
# ---------------------------------------------------------------------------------------------------------------
ROWS = []
for b, d, a, c in PAIRS:
    s = pair_stats(Y[(b, d, a)], Y[(b, d, c)])
    ROWS.append(((b, d, a, c), s))
    print(f"{b:<10} {d:<8} {SHORT[a]:>12} - {SHORT[c]:<12} diff {s['d']:+.3f}  "
          f"SE these tasks {s['se_fixed']:.3f} (z {s['d'] / s['se_fixed']:+.1f})  "
          f"SE binomial {s['se_naive']:.3f} (z {s['d'] / s['se_naive']:+.1f})  "
          f"SE tasks like these {s['se_pop']:.3f} (z {s['d'] / s['se_pop']:+.1f})  sd of per-task difference {np.sqrt(s['s2_delta']):.2f}")
sig = {m: sum(abs(s["d"] / s[m]) > Z95 for _, s in ROWS) for m in ("se_fixed", "se_naive", "se_pop")}
for (b, d, a, c), s in ROWS:
    if (abs(s["d"] / s["se_fixed"]) > Z95) != (abs(s["d"] / s["se_pop"]) > Z95):
        print(f"verdict changes with the question: {b} {d} {SHORT[a]} - {SHORT[c]} (z {s['d'] / s['se_fixed']:+.1f} versus {s['d'] / s['se_pop']:+.1f})")
print(f"comparisons with |z| > 1.96, of {len(ROWS)}: these tasks {sig['se_fixed']}, binomial {sig['se_naive']}, tasks like these {sig['se_pop']}")
between = sum(s["se_fixed"] < s["se_naive"] < s["se_pop"] for _, s in ROWS)
print(f"binomial SE lies strictly between the two task-level SEs in {between} of {len(ROWS)} comparisons")
print(f"ratio of the two task-level SEs (tasks like these / these tasks): {min(s['se_pop'] / s['se_fixed'] for _, s in ROWS):.2f} "
      f"to {max(s['se_pop'] / s['se_fixed'] for _, s in ROWS):.2f}, median {np.median([s['se_pop'] / s['se_fixed'] for _, s in ROWS]):.2f}")
print(f"sd of the true per-task difference: {min(np.sqrt(s['s2_delta']) for _, s in ROWS):.2f} to "
      f"{max(np.sqrt(s['s2_delta']) for _, s in ROWS):.2f}")
wins = [(np.mean(Y[(b, d, a)].mean(1) > Y[(b, d, c)].mean(1)), np.mean(Y[(b, d, a)].mean(1) < Y[(b, d, c)].mean(1)))
        for (b, d, a, c), _ in ROWS]
print("tasks where the first model does better / worse (averaged over its runs), median over comparisons: "
      f"{np.median([w[0] for w in wins]):.2f} / {np.median([w[1] for w in wins]):.2f}")

fig, ax = plt.subplots(figsize=(7.6, 8.4))
labels = []
for i, ((b, d, a, c), s) in enumerate(ROWS):
    yv = len(ROWS) - 1 - i
    for off, se, col, lw in [(0.22, s["se_fixed"], BLUE, 3.0), (0.0, s["se_naive"], MUTED, 1.6), (-0.22, s["se_pop"], ORANGE, 1.6)]:
        ax.plot([s["d"] - Z95 * se, s["d"] + Z95 * se], [yv + off] * 2, color=col, lw=lw, solid_capstyle="round")
    ax.plot(s["d"], yv, "o", ms=5, color=INK, mec="white", mew=1.2, zorder=5)
    labels.append(f"{'τ' if b == 'tau-bench' else 'τ²'} {d}: {SHORT[a]} − {SHORT[c]}")
ax.axvline(0, color="#c3c2b7", lw=1, zorder=0)
ax.set_yticks(range(len(ROWS))[::-1], labels, fontsize=8.5)
for yv in range(len(ROWS)):
    ax.axhline(yv - 0.5, color=GRID, lw=0.5, zorder=0)
ax.set_ylim(-0.6, len(ROWS) - 0.4)
ax.set_xlabel("Difference in pass rate (first model minus second), with 95% intervals")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.2f}" if v else "0"))
h = [matplotlib.lines.Line2D([], [], color=BLUE, lw=3), matplotlib.lines.Line2D([], [], color=MUTED, lw=1.6),
     matplotlib.lines.Line2D([], [], color=ORANGE, lw=1.6)]
ax.legend(h, ["These exact tasks (run noise only)", "Binomial: every run independent", "Tasks like these (task-level paired)"],
          frameon=False, loc="upper center", bbox_to_anchor=(0.45, 1.065), ncol=2, fontsize=8.5)
ax.set_title("The same runs answer two different questions", loc="left", pad=34)
fig.text(0.0, 0.0, "Data: tau-bench and tau2-bench released runs (Sierra Research, MIT). Figure: Diogo Ribeiro.", fontsize=7.5, color=MUTED)
fig.savefig(FIG / "fig2_two_questions.png")
plt.close(fig)

# ---------------------------------------------------------------------------------------------------------------
section("4. Checks on the assumptions")
# ---------------------------------------------------------------------------------------------------------------
# (a) Model-free: if runs are independent once the task is fixed, the pass rate of a whole run of the benchmark varies
#     across runs by sum_i p_i (1 - p_i) / n^2, which the task-level variances predict without looking at whole runs.
chi2, dof = 0.0, 0
for k in CELLS:
    y = Y[k]
    n, r = y.shape
    whole = y.mean(0)                                     # pass rate of each whole run
    predicted = within(y).sum() / n ** 2
    chi2 += (r - 1) * whole.var(ddof=1) / predicted
    dof += r - 1
    print(f"{k[0]:<10} {k[1]:<8} {k[2]:<18} whole-run pass rates {' '.join(f'{v:.3f}' for v in whole)}  "
          f"spread observed / predicted {whole.var(ddof=1) / predicted:.2f}")
p_two = 2 * min(stats.chi2.cdf(chi2, dof), stats.chi2.sf(chi2, dof))
print(f"pooled over the {len(CELLS)} cells: chi-square {chi2:.1f} on {dof} df, observed / predicted {chi2 / dof:.2f}, two-sided p {p_two:.2f}")


# (b) The simulator used in sections 5 and 6: each task's true pass rate is drawn from its posterior under a Beta prior
#     fitted to the model-domain cell by moments. Fit on runs 1-2, predict how runs 3-4 split, compare with what they did.
def beta_prior(y):
    m, sw, sb = components(y)
    s = max(m * (1 - m) / max(sb, 1e-3) - 1, 0.05)       # alpha + beta
    return m * s, (1 - m) * s


def draw_rates(y, a0, b0, size, idx=None):
    """size x n draws of each task's true pass rate given its observed runs; idx resamples tasks."""
    passes = y.sum(1) if idx is None else y.sum(1)[idx]
    fails = y.shape[1] - passes
    return rng.beta(a0 + passes, b0 + fails, size=(size,) + passes.shape)


pred, seen = np.zeros(3), np.zeros(3)
for k in [c for c in CELLS if c[0] == "tau2-bench"]:
    y_fit, y_new = Y[k][:, :2], Y[k][:, 2:]
    a0, b0 = beta_prior(y_fit)
    p = draw_rates(y_fit, a0, b0, 4000)
    pred += np.stack([(1 - p) ** 2, 2 * p * (1 - p), p ** 2]).mean(axis=1).sum(axis=1)
    seen += np.bincount(y_new.sum(1).astype(int), minlength=3)
pred, seen = pred / pred.sum(), seen / seen.sum()
print("simulator fitted on runs 1-2, predicting runs 3-4 (share of task-model cells):")
for j, name in enumerate(["both fail", "one passes", "both pass"]):
    print(f"  {name:<11} predicted {pred[j]:.3f}  observed {seen[j]:.3f}")

# ---------------------------------------------------------------------------------------------------------------
section("5. Three runs and a bootstrap")
# ---------------------------------------------------------------------------------------------------------------
print("a percentile bootstrap over three runs: the 2.5% point is the smallest run, because all three resamples")
print(f"  equal the smallest with chance 1/27 = {1 / 27:.3f} > 0.025, so the 95% interval is the range of the three runs,")
print(f"  which holds a symmetric quantity's centre with chance 1 - 2(1/2)^3 = {1 - 2 * 0.5 ** 3:.2f}")
print(f"  the honest t multiplier with three runs is {stats.t.ppf(0.975, 2):.2f}, not {Z95:.2f}")

METHODS = ["Bootstrap over whole runs", "t-interval over whole runs", "Binomial, runs independent",
           "Task-level, these tasks", "Task-level paired, tasks like these"]


def intervals(ya, yb, boot=2000):
    """95% intervals from five methods for one simulated comparison (rows = tasks, columns = runs)."""
    n, r = ya.shape
    ma, mb = ya.mean(0), yb.mean(0)                      # one pass rate per whole run of the benchmark
    w = rng.multinomial(r, np.full(r, 1 / r), size=(2, boot))
    bd = w[0] @ ma / r - w[1] @ mb / r
    out = [tuple(np.quantile(bd, [0.025, 0.975]))]
    va, vb = ma.var(ddof=1) / r, mb.var(ddof=1) / r      # Welch on the whole-run pass rates
    df = (va + vb) ** 2 / (va ** 2 / (r - 1) + vb ** 2 / (r - 1)) if va + vb > 0 else r - 1
    h = stats.t.ppf(0.975, df) * np.sqrt(va + vb)
    d = ma.mean() - mb.mean()
    out.append((d - h, d + h))
    s = pair_stats(ya, yb)
    out.append((d - Z95 * s["se_naive"], d + Z95 * s["se_naive"]))
    out.append((d - Z95 * s["se_fixed"], d + Z95 * s["se_fixed"]))
    h = stats.t.ppf(0.975, n - 1) * s["se_pop"]
    out.append((d - h, d + h))
    return np.array(out)


def simulate(r, reps, pairs):
    """Coverage of the fixed-task and the population difference, and median width, for each method."""
    hit_fixed, hit_pop, width = np.zeros(len(METHODS)), np.zeros(len(METHODS)), []
    single_hits = 0
    for b, d, a, c in pairs:
        ya, yb = Y[(b, d, a)], Y[(b, d, c)]
        (aa, ba), (ac, bc) = beta_prior(ya), beta_prior(yb)
        n = ya.shape[0]
        truth_pop = np.mean((aa + ya.sum(1)) / (aa + ba + ya.shape[1]) - (ac + yb.sum(1)) / (ac + bc + yb.shape[1]))
        for _ in range(reps):
            idx = rng.integers(0, n, n)                  # a fresh draw of tasks from the same population
            pa, pb = draw_rates(ya, aa, ba, 1, idx)[0], draw_rates(yb, ac, bc, 1, idx)[0]
            sa = (rng.random((n, r)) < pa[:, None]).astype(float)
            sb = (rng.random((n, r)) < pb[:, None]).astype(float)
            ci = intervals(sa, sb)
            truth_fixed = np.mean(pa - pb)
            hit_fixed += (ci[:, 0] <= truth_fixed) & (truth_fixed <= ci[:, 1])
            hit_pop += (ci[:, 0] <= truth_pop) & (truth_pop <= ci[:, 1])
            width.append(ci[:, 1] - ci[:, 0])
            m = sa.mean(0)                               # one model alone: bootstrap of its mean over whole runs
            bm = rng.multinomial(r, np.full(r, 1 / r), size=2000) @ m / r
            lo, hi = np.quantile(bm, [0.025, 0.975])
            single_hits += lo <= pa.mean() <= hi
    total = reps * len(pairs)
    return hit_fixed / total, hit_pop / total, np.median(width, axis=0), single_hits / total


T2PAIRS = [p for p in PAIRS if p[0] == "tau2-bench"]
cov_fixed, cov_pop, wid, single = simulate(3, 300, T2PAIRS)
print(f"simulation: 18 tau2 comparisons x 300 replicates, three runs per task per model")
print(f"one model's mean, percentile bootstrap over its three runs: coverage {single:.3f}")
for m, cf, cp, w in zip(METHODS, cov_fixed, cov_pop, wid):
    print(f"  {m:<38} covers these-tasks difference {cf:.3f}   tasks-like-these difference {cp:.3f}   median width {w:.3f}")
for r in (5, 10, 20):
    cf, _, _, s1 = simulate(r, 60, T2PAIRS)
    print(f"  with {r:>2} runs per task: bootstrap over whole runs covers the these-tasks difference {cf[0]:.3f} "
          f"(one model's mean {s1:.3f}); task-level these-tasks interval {cf[3]:.3f}")

fig, ax = plt.subplots(figsize=(7.6, 3.9))
yy = np.arange(len(METHODS))[::-1]
for yv, cf, cp in zip(yy, cov_fixed, cov_pop):
    ax.plot([min(cf, cp), max(cf, cp)], [yv, yv], color=GRID, lw=2, zorder=1)
ax.plot(cov_fixed, yy, "o", ms=9, color=BLUE, mec="white", mew=1.5, zorder=3, label="Target: difference on these tasks")
ax.plot(cov_pop, yy, "o", ms=9, color=ORANGE, mec="white", mew=1.5, zorder=3, label="Target: difference on tasks like these")
for yv, cf, cp in zip(yy, cov_fixed, cov_pop):
    pct = lambda v: f"{v:.1%}" if v >= 0.995 else f"{v:.0%}"     # 99.8% must not print as 100%
    ax.text(cf, yv + 0.24, pct(cf), ha="center", fontsize=8, color=INK)
    ax.text(cp, yv - 0.42, pct(cp), ha="center", fontsize=8, color=INK)
ax.axvline(0.95, color=INK2, lw=1, ls=(0, (3, 3)), zorder=0)
ax.text(0.947, len(METHODS) - 0.38, "95% promised", ha="right", fontsize=8, color=INK2)
ax.set_yticks(yy, METHODS, fontsize=9)
ax.set_xlim(0.45, 1.02)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
ax.set_xlabel("How often the 95% interval contains the true difference (three runs per task)")
ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
ax.set_ylim(-0.8, len(METHODS) - 0.2)
ax.legend(frameon=False, loc="lower left", fontsize=8.5)
ax.set_title("Which interval keeps its promise depends on the question", loc="left")
fig.text(0.0, -0.04, "Simulation from posterior task pass rates fitted to tau2-bench runs; 18 comparisons x 300 replicates. Figure: Diogo Ribeiro.",
         fontsize=7.5, color=MUTED)
fig.savefig(FIG / "fig3_coverage.png")
plt.close(fig)

# ---------------------------------------------------------------------------------------------------------------
section("6. How many runs, and how many dollars")
# ---------------------------------------------------------------------------------------------------------------
print("minimum detectable difference at 80% power, two-sided 5%: MDD = 2.80 x SE, with")
print("  SE(these tasks) = sqrt(W / (n r))   and   SE(tasks like these) = sqrt(sigma_Delta^2 / n + W / (n r))")
fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.7), sharey=True)
rgrid = np.unique(np.round(np.logspace(0, np.log10(64), 60)).astype(int))
for ax, (b, d) in zip(axes, [x for x in DOMAINS if x[0] == "tau2-bench"]):
    rows = [(p, s) for p, s in ROWS if p[:2] == (b, d)]
    n = rows[0][1]["n"]
    w, s2 = np.median([s["w"] for _, s in rows]), np.median([s["s2_delta"] for _, s in rows])
    mdd_fixed = K * np.sqrt(w / (n * rgrid))
    mdd_pop = K * np.sqrt(s2 / n + w / (n * rgrid))
    floor = K * np.sqrt(s2 / n)
    print(f"{d:<8} n = {n:>3} tasks  median W {w:.3f}  median sigma_Delta {np.sqrt(s2):.3f}  "
          f"MDD at 4 runs: these tasks {K * np.sqrt(w / (n * 4)):.3f}, tasks like these {K * np.sqrt(s2 / n + w / (n * 4)):.3f}; "
          f"floor with unlimited runs {floor:.3f}")
    for delta in (0.02, 0.05, 0.10):
        r_need = (K / delta) ** 2 * w / n
        dollars = [n * r_need * (COST[(b, d, p[2])] + COST[(b, d, p[3])]) for p, _ in rows]
        n_need = (K / delta) ** 2 * (s2 + w / 4)
        print(f"   detect {delta:.2f} on these tasks: {r_need:6.1f} runs per task per model, "
              f"${min(dollars):,.0f} to ${max(dollars):,.0f} for one comparison  |  on tasks like these at 4 runs: "
              f"{n_need:,.0f} tasks; reachable with these {n} tasks and unlimited runs: {delta > floor}")
    for p, s in rows:
        need = (K / abs(s["d"])) ** 2 * s["w"] / n if s["d"] else np.inf
        print(f"   {SHORT[p[2]]:>12} - {SHORT[p[3]]:<12} observed |diff| {abs(s['d']):.3f}: runs per task to resolve it on these tasks "
              f"{need:,.1f}; resolvable for tasks like these with unlimited runs: {abs(s['d']) > K * np.sqrt(s['s2_delta'] / n)}")
    ax.plot(rgrid, mdd_fixed, color=BLUE, lw=2, label="These exact tasks")
    ax.plot(rgrid, mdd_pop, color=ORANGE, lw=2, label="Tasks like these")
    ax.axhline(floor, color=ORANGE, lw=1, ls=(0, (3, 3)))
    ax.text(64, floor + 0.006, f"floor {floor:.3f}", ha="right", va="bottom", fontsize=8, color=INK2)
    for _, s in rows:
        ax.plot(4, abs(s["d"]), "o", ms=5, color=INK, mec="white", mew=1, zorder=4, alpha=0.85)
    ax.axvline(4, color=GRID, lw=1, zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"])
    ax.set_title(f"{d} ({n} tasks)", loc="left")
    ax.set_xlabel("Runs per task, per model")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
axes[0].set_ylabel("Smallest difference found 80% of the time")
axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
axes[0].legend(frameon=False, loc="upper right", fontsize=8.5)
axes[0].set_ylim(0, 0.24)
fig.suptitle("More runs shrink one error and leave the other at its floor", x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK, y=1.02)
fig.text(0.0, -0.06, "Dots: observed gaps between the six model pairs in each domain, at the four runs released. Medians of W and sigma_Delta over the pairs. "
         "Data: tau2-bench (Sierra Research, MIT). Figure: Diogo Ribeiro.", fontsize=7.5, color=MUTED)
fig.savefig(FIG / "fig4_runs_needed.png")
plt.close(fig)
print("\nfigures written:", ", ".join(sorted(p.name for p in FIG.glob("*.png"))))
