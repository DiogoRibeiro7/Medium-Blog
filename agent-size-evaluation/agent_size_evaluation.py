"""
Simulations for "Evaluating Over-Engineering in LLM Coding Agents: Metrics, Variance and Design".

Every number quoted in the article is printed by this script, and every figure is written by it.
Nothing here runs a language model: the agent behaviour is simulated under stated assumptions,
so the results describe the evaluation designs, not any particular agent.

Run:  python agent_size_evaluation.py > results/output.txt   (about a minute)
"""
import ast
import difflib
import io
import itertools
import math
import tokenize
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(20261015)

# ---------------------------------------------------------------- figure style
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.linewidth": 1, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "grid.linestyle": "-",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.fontsize": 9.5, "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def finish(fig, path, title, subtitle):
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.012, 0.935, subtitle, ha="left", va="top", fontsize=10, color=INK2)
    fig.text(0.012, 0.012, "Simulation and figure: Diogo Ribeiro", ha="left", va="bottom", fontsize=8.5, color=MUTED)
    fig.savefig(FIG / path, dpi=200)
    plt.close(fig)


def header(text):
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


# ============================================================================
# 1. Size measured in code tokens, not lines
# ============================================================================
DROP = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
        tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER}


def docstring_starts(src):
    """(line, column) where each docstring starts: module, class and function docstrings."""
    starts = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                starts.add((first.lineno, first.col_offset))
    return starts


def code_tokens(src):
    """The tokens Python executes: no comments, docstrings, newlines or indentation."""
    skip = docstring_starts(src)
    return [t.string for t in tokenize.generate_tokens(io.StringIO(src).readline)
            if t.type not in DROP and t.start not in skip]


def churn(a, b):
    """(added, removed) between two sequences, from a minimal edit script."""
    added = removed = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
    return added, removed


def token_churn(before, after):
    return churn(code_tokens(before), code_tokens(after))


def line_churn(before, after):
    return churn(before.splitlines(), after.splitlines())


SEED = '''def parse_rows(lines, sep=","):
    rows = []
    for line in lines:
        rows.append(line.rstrip("\\n").split(sep))
    return rows
'''

REFORMATTED = '''def parse_rows(lines, sep=","):
    """Split every line of a delimited file into its fields."""
    rows = []
    for line in lines:
        # drop the newline before splitting
        rows.append(
            line.rstrip("\\n").split(sep)
        )
    return rows
'''

# Ticket: "files exported from the Portuguese locale use semicolons; accept them."
NARROW = '''def _separator(line, candidates=(",", ";")):
    """The candidate that occurs most often in the line; the first one on a tie."""
    return max(candidates, key=line.count)


def parse_rows(lines, sep=None):
    lines = list(lines)
    if sep is None:
        sep = _separator(lines[0]) if lines else ","
    rows = []
    for line in lines:
        rows.append(line.rstrip("\\n").split(sep))
    return rows
'''

WIDE = '''def parse_rows(lines, sep=","):
    rows = []
    for line in lines:
        fields = [f.strip() for f in line.rstrip("\\n").replace(";", sep).split(sep)]
        if any(fields):
            rows.append(fields)
    return rows
'''


def load(src):
    namespace = {}
    exec(compile(src, "<patch>", "exec"), namespace)
    return namespace["parse_rows"]


def call(fn, lines):
    try:
        return ("ok", fn(list(lines)))
    except Exception as e:  # an exception is a behaviour too
        return ("raised", type(e).__name__)


FIELDS = np.array(["a", "b", "42", "x y", " padded", "trailing ", "", "a;b"])
FIELD_P = np.array([0.20, 0.15, 0.15, 0.10, 0.10, 0.10, 0.12, 0.08])


def random_file(sep, allow_semicolon_values):
    pool = FIELDS if allow_semicolon_values else FIELDS[:-1]
    p = FIELD_P if allow_semicolon_values else FIELD_P[:-1] / FIELD_P[:-1].sum()
    width = rng.integers(1, 5)
    return [sep.join(rng.choice(pool, size=width, p=p)) + "\n" for _ in range(rng.integers(1, 6))]


header("1. Size in code tokens, and behavioural footprint outside the ticket")
la, lr = line_churn(SEED, REFORMATTED)
ta, tr = token_churn(SEED, REFORMATTED)
print(f"reformatting only : lines +{la} -{lr} | code tokens +{ta} -{tr}")

seed_fn, narrow_fn, wide_fn = load(SEED), load(NARROW), load(WIDE)
N_PROBES = 5000
out_of_scope = [random_file(",", True) for _ in range(N_PROBES)]  # what the seed already handled
in_scope = [random_file(";", False) for _ in range(N_PROBES)]    # what the ticket asks for
example = ["a;b;c\n", "1;2;3\n"]                                  # the single acceptance example
expected_example = [["a", "b", "c"], ["1", "2", "3"]]

for name, src, fn in (("narrow patch", NARROW, narrow_fn), ("wide patch", WIDE, wide_fn)):
    la, lr = line_churn(SEED, src)
    ta, tr = token_churn(SEED, src)
    changed = np.mean([call(fn, f) != call(seed_fn, f) for f in out_of_scope])
    correct = np.mean([call(fn, f) == ("ok", [ln.rstrip("\n").split(";") for ln in f]) for f in in_scope])
    passes = call(fn, example) == ("ok", expected_example)
    print(f"{name:<13}: lines +{la} -{lr} | code tokens +{ta} -{tr} | acceptance example passes: {passes} | "
          f"in-scope probes correct: {correct:.1%} | out-of-scope probes changed: {changed:.1%}")

# ============================================================================
# 2. Zero failures, non-inferiority, and intervals that collapse
# ============================================================================
header("2. Safety at the ceiling: what zero failures can and cannot show")


def zero_failure_upper(n, alpha=0.05):
    """Exact one-sided (1 - alpha) upper bound on a failure rate after 0 failures in n runs."""
    return 1 - alpha ** (1 / n)


def wilson(k, n, z):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def newcombe_upper(k1, n1, k0, n0, z=1.959964):
    """Upper end of Newcombe's hybrid score interval for p1 - p0."""
    p1, p0 = k1 / n1, k0 / n0
    _, u1 = wilson(k1, n1, z)
    l0, _ = wilson(k0, n0, z)
    return (p1 - p0) + np.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)


for n in (10, 20, 40, 60, 100):
    print(f"0 failures in {n:>3} runs: exact one-sided 95% upper bound {zero_failure_upper(n):.1%} "
          f"(rule of three: {3 / n:.1%})")
n_exact = math.ceil(math.log(0.05) / math.log(0.95))
z2 = 1.959964 ** 2
n_newcombe = math.ceil(z2 * (1 / 0.05 - 1))
print(f"runs needed before 0/n bounds the failure rate below 5% (one-sided exact): {n_exact}")
print(f"runs per arm needed before 0/n vs 0/n gives a Newcombe 95% upper bound <= 5 points: {n_newcombe}")

MARGIN, R_SAFE, B = 0.05, 2, 1000
runs_grid = np.arange(10, 202, 8)
claims = {("bootstrap", 0.0): [], ("bootstrap", 0.1): [], ("newcombe", 0.0): [], ("newcombe", 0.1): []}
for n in runs_grid:
    T = n // R_SAFE
    M = np.stack([np.bincount(rng.integers(0, T, T), minlength=T) for _ in range(B)])  # resample multiplicities
    for p_arm in (0.0, 0.1):
        k = rng.binomial(R_SAFE, p_arm, size=(2000, T))  # baseline fails nowhere, so k is the paired difference
        diffs = k / R_SAFE
        boot = diffs @ M.T / T
        upper = np.percentile(boot, 97.5, axis=1)
        claims[("bootstrap", p_arm)].append(np.mean(upper <= MARGIN))
        nc = newcombe_upper(k.sum(axis=1), T * R_SAFE, 0, T * R_SAFE)
        claims[("newcombe", p_arm)].append(np.mean(nc <= MARGIN))
for n_show in (10, 18, 34, 74, 106):
    i = int(np.argmin(np.abs(runs_grid - n_show)))
    print(f"runs per arm {runs_grid[i]:>3}: P(claim non-inferiority at 5 points) | harmless arm: "
          f"ticket bootstrap {claims[('bootstrap', 0.0)][i]:.0%}, Newcombe {claims[('newcombe', 0.0)][i]:.0%} | "
          f"arm failing 10% of runs: ticket bootstrap {claims[('bootstrap', 0.1)][i]:.0%}, "
          f"Newcombe {claims[('newcombe', 0.1)][i]:.0%}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.17, wspace=0.28)
n_line = np.arange(5, 201)
ax1.plot(n_line, 100 * zero_failure_upper(n_line), color=BLUE)
ax1.axhline(5, color=MUTED, lw=1)
for n_mark in (20, n_exact):
    ax1.plot(n_mark, 100 * zero_failure_upper(n_mark), "o", ms=8, color=BLUE, mec=SURFACE, mew=2, zorder=3)
    ax1.annotate(f"{n_mark} runs: {zero_failure_upper(n_mark):.1%}", (n_mark, 100 * zero_failure_upper(n_mark)),
                 xytext=(10, 6), textcoords="offset points", color=INK2, fontsize=9.5)
ax1.text(200, 5.6, "5% failure rate", ha="right", color=MUTED, fontsize=9)
ax1.set_title("Upper bound after zero failures")
ax1.set_xlabel("runs observed, all safe")
ax1.set_ylabel("one-sided 95% upper bound (%)")
ax1.set_ylim(0, 32)
for (method, p_arm), color, ls in ((("bootstrap", 0.0), ORANGE, "-"), (("bootstrap", 0.1), ORANGE, "--"),
                                   (("newcombe", 0.0), BLUE, "-"), (("newcombe", 0.1), BLUE, "--")):
    label = ("ticket bootstrap" if method == "bootstrap" else "Newcombe interval") + \
            (", harmless arm" if p_arm == 0 else ", arm failing 10%")
    ax2.plot(runs_grid, 100 * np.array(claims[(method, p_arm)]), color=color, ls=ls, label=label)
ax2.set_title("How often non-inferiority is claimed")
ax2.set_xlabel("runs per arm (2 per ticket); baseline never fails")
ax2.set_ylabel("claims at a 5-point margin (%)")
ax2.set_ylim(-3, 103)
ax2.annotate(f"{n_newcombe} runs per arm", (n_newcombe, 75), xytext=(-8, 0), textcoords="offset points", color=INK2, fontsize=9.5, va="center", ha="right")
ax2.legend(loc="center right", bbox_to_anchor=(1.0, 0.45))
finish(fig, "fig3_zero_failures.png", "Zero failures are weak evidence of safety",
       "Left: exact bound on the failure rate. Right: a percentile bootstrap over tickets collapses to a point "
       "when nothing fails.")

# ============================================================================
# 3. Variance components: tickets, replicates and what to buy
# ============================================================================
header("3. Variance components and the allocation of tickets and replicates")
RHO, TAU, SIGMA_W = 0.85, 0.30, 0.50          # true ratio, effect heterogeneity, run-to-run log SD
SIZE_MU, SIZE_SD = math.log(60), 0.90         # tokens the ticket needs, across tickets
BLOAT_MU, BLOAT_SD = math.log(1.5), 0.30      # baseline excess over need, across tickets


def draw_tickets(T, size=None):
    shape = (T,) if size is None else (size, T)
    need = np.exp(rng.normal(SIZE_MU, SIZE_SD, shape))
    m0 = need * np.exp(rng.normal(BLOAT_MU, BLOAT_SD, shape))
    m1 = m0 * np.exp(rng.normal(math.log(RHO), TAU, shape))
    return m0, m1


def draw_runs(m, R):
    """R runs per ticket: Poisson counts around a log-normal run-level mean."""
    lam = m[..., None] * np.exp(rng.normal(-SIGMA_W ** 2 / 2, SIGMA_W, m.shape + (R,)))
    return rng.poisson(lam)


m0, m1 = draw_tickets(200_000)
v0 = np.exp(SIGMA_W ** 2) - 1 + np.mean(1 / m0)    # within-ticket squared CV of one run, baseline arm
v1 = np.exp(SIGMA_W ** 2) - 1 + np.mean(1 / m1)
v = (v0 + v1) / 2
between = np.var(np.log(m0))
print(f"assumptions: ratio {RHO}, effect heterogeneity tau {TAU}, run log-SD {SIGMA_W}")
print(f"between-ticket variance of log baseline size: {between:.3f} (SD {math.sqrt(between):.2f}); "
      f"within-ticket squared CV of a run: {v:.3f} (CV {math.sqrt(v):.2f})")
print("in a paired log-ratio, the between-ticket size variance cancels; tau^2 and the run noise remain")
pair = draw_runs(m0[:100_000], 2)
pair = pair[(pair > 0).all(axis=1)]
print(f"two runs of the same arm on the same ticket differ by a factor of 1.5 or more: "
      f"{np.mean(np.abs(np.log(pair[:, 0] / pair[:, 1])) >= math.log(1.5)):.0%}")


def se_formula(T, R):
    return np.sqrt((TAU ** 2 + 2 * v / R) / T)


def r_star(cost_ratio):
    return math.sqrt(v) / TAU * math.sqrt(cost_ratio)


z_a, z_b = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
for R in (1, 2, 4, 6, 10):
    T_need = (z_a + z_b) ** 2 * (TAU ** 2 + 2 * v / R) / math.log(RHO) ** 2
    print(f"R = {R:>2} replicates per arm: tickets for 80% power at ratio {RHO}: {math.ceil(T_need)}")
print(f"floor as R grows: {math.ceil((z_a + z_b) ** 2 * TAU ** 2 / math.log(RHO) ** 2)} tickets")

BUDGET = 3000  # in episode-equivalents
costs = (2, 10, 40)
R_grid = np.arange(1, 31)
fig, ax = plt.subplots(figsize=(10, 4.8))
fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.15)
for c_T, color in zip(costs, (AQUA, BLUE, ORANGE)):
    T_c = BUDGET / (c_T + 2 * R_grid)
    se = se_formula(T_c, R_grid)
    ax.plot(R_grid, se, color=color, label=f"a ticket costs {c_T} episodes to write")
    rs = r_star(c_T)
    T_opt = BUDGET / (c_T + 2 * rs)
    ax.plot(rs, se_formula(T_opt, rs), "o", ms=9, color=color, mec=SURFACE, mew=2, zorder=3)
    print(f"ticket cost {c_T:>2} episodes: optimal replicates R* = {rs:.1f}, tickets at budget {BUDGET}: "
          f"{T_opt:.0f}, SE of log ratio {se_formula(T_opt, rs):.3f} "
          f"(vs R = 2: T = {BUDGET / (c_T + 4):.0f}, SE {se_formula(BUDGET / (c_T + 4), 2):.3f})")

# check the formula against the estimator it describes, for the middle cost
sim_R, sim_se = [], []
for R in (1, 2, 4, 6, 10, 15, 20, 30):
    T = int(round(BUDGET / (10 + 2 * R)))
    a0, a1 = draw_tickets(T, size=3000)
    y0, y1 = draw_runs(a0, R).mean(axis=-1), draw_runs(a1, R).mean(axis=-1)
    est = np.mean(np.log(np.maximum(y1, 0.5)) - np.log(np.maximum(y0, 0.5)), axis=1)
    sim_R.append(R)
    sim_se.append(est.std())
    print(f"check, ticket cost 10, R = {R:>2}, T = {T:>3}: simulated SE {est.std():.3f}, "
          f"formula {se_formula(T, R):.3f}")
ax.plot(sim_R, sim_se, "s", ms=7, color=INK2, mfc=SURFACE, mew=1.5, label="simulated, ticket cost 10")
ax.set_xlabel("replicate runs per ticket and arm (R)")
ax.set_ylabel("standard error of the log ratio")
ax.set_ylim(0, None)
ax.legend(loc="upper right")
finish(fig, "fig2_allocation.png", "Fixed budget: how many tickets, how many runs each",
       f"Budget of {BUDGET} episode-equivalents. Dots mark R* = (CV / tau) * sqrt(ticket cost / episode cost).")

# ============================================================================
# 4. Three estimators, three estimands
# ============================================================================
header("4. Estimators of the size ratio: median of ratios, geometric mean, ratio of totals")
R_EST, SIMS, B_EST = 2, 1000, 1000
T_grid = (10, 25, 50, 75, 100, 150, 200)
w_target = float(np.sum(m1) / np.sum(m0))  # population ratio of totals under these assumptions
print(f"targets: median and geometric-mean ratio = {RHO}; size-weighted ratio of totals = {w_target:.3f}")


def estimate_all(y0, y1):
    """Point estimates and 95% intervals for the three estimators. y0, y1: (sims, T) per-ticket arm means."""
    sims, T = y0.shape
    # median of per-ticket (y1 + 1) / (y0 + 1), percentile bootstrap over tickets
    r = (y1 + 1) / (y0 + 1)
    med = np.median(r, axis=1)
    lo_m, hi_m = np.empty(sims), np.empty(sims)
    for s0 in range(0, sims, 50):
        idx = rng.integers(0, T, (B_EST, T))
        bm = np.median(r[s0:s0 + 50][:, idx], axis=2)
        lo_m[s0:s0 + 50], hi_m[s0:s0 + 50] = np.percentile(bm, [2.5, 97.5], axis=1)
    # geometric mean of per-ticket ratios, t interval on the log scale
    g = np.log(np.maximum(y1, 0.5)) - np.log(np.maximum(y0, 0.5))
    gm, gs = g.mean(axis=1), g.std(axis=1, ddof=1) / np.sqrt(T)
    tq = stats.t.ppf(0.975, T - 1)
    # ratio of totals = Poisson pseudo-likelihood with ticket fixed effects; cluster-robust delta method
    S0, S1 = y0.sum(axis=1), y1.sum(axis=1)
    u = y1 / S1[:, None] - y0 / S0[:, None]
    ts = np.sqrt(T / (T - 1) * (u ** 2).sum(axis=1))
    lt = np.log(S1 / S0)
    return {"median": (med, lo_m, hi_m),
            "geomean": (np.exp(gm), np.exp(gm - tq * gs), np.exp(gm + tq * gs)),
            "totals": (np.exp(lt), np.exp(lt - tq * ts), np.exp(lt + tq * ts))}


power = {k: [] for k in ("median", "geomean", "totals")}
for T in T_grid:
    a0, a1 = draw_tickets(T, size=SIMS)
    y0, y1 = draw_runs(a0, R_EST).mean(axis=-1), draw_runs(a1, R_EST).mean(axis=-1)
    res = estimate_all(y0, y1)
    line = []
    for k, (est, lo, hi) in res.items():
        target = w_target if k == "totals" else RHO
        power[k].append(np.mean(hi < 1))
        line.append(f"{k}: power {np.mean(hi < 1):.0%}, coverage {np.mean((lo <= target) & (target <= hi)):.0%}, "
                    f"mean estimate {np.mean(est):.3f}")
    print(f"T = {T:>3}, R = {R_EST}: " + " | ".join(line))

fig, ax = plt.subplots(figsize=(10, 4.8))
fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.15)
labels = {"geomean": "geometric mean of ticket ratios (t interval)",
          "median": "median of ticket ratios (bootstrap)",
          "totals": "ratio of totals (Poisson, ticket effects, cluster-robust)"}
for k, color in (("geomean", BLUE), ("median", ORANGE), ("totals", AQUA)):
    ax.plot(T_grid, 100 * np.array(power[k]), color=color, marker="o", ms=8, mec=SURFACE, mew=2, label=labels[k])
ax.axhline(80, color=MUTED, lw=1)
ax.text(12, 81.5, "80% power", ha="left", color=MUTED, fontsize=9)
ax.set_xlabel("tickets (2 runs per ticket and arm)")
ax.set_ylabel("power to detect the reduction (%)")
ax.set_ylim(0, 100)
ax.legend(handles=[Line2D([0], [0], color=c, lw=2, label=labels[k]) for k, c in (("geomean", BLUE), ("median", ORANGE), ("totals", AQUA))], loc="lower right")
finish(fig, "fig1_estimators.png", "Same data, three estimators",
       f"True ratio {RHO}; run-to-run log SD {SIGMA_W}; effect heterogeneity across tickets {TAU}.")

# ============================================================================
# 5. Ablation: one at a time against a fractional factorial
# ============================================================================
header("5. Ablation: one-factor-at-a-time against a 2^(8-4) resolution IV fractional factorial")
base4 = np.array(list(itertools.product([-1, 1], repeat=4)))
A_, B_, C_, D_ = base4.T
DESIGN = np.column_stack([A_, B_, C_, D_, B_ * C_ * D_, A_ * C_ * D_, A_ * B_ * C_, A_ * B_ * D_])  # 16 x 8
K = DESIGN.shape[1]
pairs_ok = all(abs(DESIGN[:, i] @ DESIGN[:, j]) < 16 for i, j in itertools.combinations(range(K), 2))
triples_ok = all(len(set(DESIGN[:, i] * DESIGN[:, j] * DESIGN[:, k])) > 1
                 for i, j, k in itertools.combinations(range(K), 3))
print(f"design 16 x {K}: no aliased main effects: {pairs_ok}; no main effect aliased with a two-factor "
      f"interaction (no words of length 3): {triples_ok}")

MAIN = np.array([-0.10, -0.08, -0.06, 0.0, 0.0, 0.0, 0.0, 0.0])  # effect of switching a component on (log size)
SUBST = 0.12   # components 4 and 5 are substitutes: the penalty appears only when both are off
HET = 0.05     # ticket-level heterogeneity of each non-zero effect


def log_means(configs, T, R, n_sims):
    """Per (sim, config, ticket) log of the mean size over R runs."""
    on = (configs + 1) / 2                                         # (C, K) in {0, 1}
    base = rng.normal(math.log(90), 0.95, (n_sims, 1, T))
    het = rng.normal(0, HET, (n_sims, K, T)) * (MAIN != 0)[None, :, None]
    eff = np.einsum("ck,skt->sct", on, MAIN[None, :, None] + het)
    pen_het = rng.normal(0, HET, (n_sims, 1, T))
    pen = ((1 - on[:, 3]) * (1 - on[:, 4]))[None, :, None] * (SUBST + pen_het)
    m = np.exp(base + eff + pen)
    y = draw_runs(m, R).mean(axis=-1)
    return np.log(np.maximum(y, 0.5))


T_ABL = 30
FULL = np.ones((1, K), dtype=int)
OFAT = np.vstack([FULL] + [np.where(np.arange(K) == k, -1, 1)[None, :] for k in range(K)])  # 9 configs
R_OFAT, R_FF = 5, 3
print(f"budget: one-at-a-time {OFAT.shape[0]} configs x {T_ABL} tickets x {R_OFAT} runs = "
      f"{OFAT.shape[0] * T_ABL * R_OFAT} episodes; factorial 16 x {T_ABL} x {R_FF} = {16 * T_ABL * R_FF}")


def ofat_effects(lm):
    """(sims, K, T): log size with everything on minus log size with component k off."""
    return lm[:, :1, :] - lm[:, 1:, :]


def factorial_effects(lm):
    """(sims, K, T): mean log size over the 8 configs with k on, minus the 8 with k off."""
    return np.einsum("sct,ck->skt", lm, DESIGN) / 8


def summarise(eff, level=0.95):
    T = eff.shape[-1]
    mean = eff.mean(axis=-1)
    se = eff.std(axis=-1, ddof=1) / np.sqrt(T)
    q = stats.t.ppf(0.5 + level / 2, T - 1)
    return mean, mean - q * se, mean + q * se


SIMS_ABL = 2000
ofat = ofat_effects(log_means(OFAT, T_ABL, R_OFAT, SIMS_ABL))
ff = factorial_effects(log_means(DESIGN, T_ABL, R_FF, SIMS_ABL))
truth_ofat = MAIN.copy()                    # conditional on everything else on: substitutes look useless
truth_ff = MAIN.copy()
truth_ff[3] = truth_ff[4] = -SUBST / 2      # averaged over the other component being on or off
EQ = 0.05                                    # equivalence margin on the log scale for "no effect"
for name, eff, truth in (("one-at-a-time", ofat, truth_ofat), ("factorial", ff, truth_ff)):
    mean, lo, hi = summarise(eff)
    _, lo90, hi90 = summarise(eff, 0.90)
    print(f"{name}:")
    for k in range(K):
        print(f"  component {k + 1}: design's own target {truth[k]:+.3f} | mean estimate {mean[:, k].mean():+.3f} | "
              f"CI excludes 0: {np.mean((hi[:, k] < 0) | (lo[:, k] > 0)):.0%} | "
              f"shown equivalent to 0 within +-{EQ} (TOST): {np.mean((lo90[:, k] > -EQ) & (hi90[:, k] < EQ)):.0%} | "
              f"mean CI half-width {np.mean(hi[:, k] - lo[:, k]) / 2:.3f}")

# one illustrative replicate for the figure
one_ofat = summarise(ofat_effects(log_means(OFAT, T_ABL, R_OFAT, 1)))
one_ff = summarise(factorial_effects(log_means(DESIGN, T_ABL, R_FF, 1)))
fig, ax = plt.subplots(figsize=(10, 5.4))
fig.subplots_adjust(left=0.13, right=0.97, top=0.80, bottom=0.14)
ypos = np.arange(K)[::-1]
for (mean, lo, hi), color, dy, label in ((one_ofat, ORANGE, 0.17, "one at a time"),
                                         (one_ff, BLUE, -0.17, "fractional factorial")):
    ax.hlines(ypos + dy, lo[0], hi[0], color=color, lw=2)
    ax.plot(mean[0], ypos + dy, "o", ms=8, color=color, mec=SURFACE, mew=2, label=label, zorder=3)
ax.plot(truth_ofat, ypos + 0.17, "|", ms=12, mew=2, color=INK2, zorder=4)
ax.plot(truth_ff, ypos - 0.17, "|", ms=12, mew=2, color=INK2, zorder=4, label="true target of each design")
ax.axvline(0, color=AXIS, lw=1)
ax.set_yticks(ypos, [f"component {k + 1}" for k in range(K)])
ax.grid(axis="y", visible=False)
ax.set_xlabel("effect of switching the component on (change in log code size; negative = smaller)")
fig.legend(loc="upper left", bbox_to_anchor=(0.125, 0.895), ncol=3, handletextpad=0.4, columnspacing=1.6)
finish(fig, "fig4_ablation.png", "Removing one piece at a time cannot see redundancy",
       "Components 4 and 5 do the same job. 95% intervals from one simulated study of each design, equal budgets.")

# ============================================================================
# 6. Selecting the smallest of k candidates
# ============================================================================
header("6. Best-of-k by size: the reduction and the selection hazard")
Q_UNSAFE, SHRINK = 0.10, 0.70   # share of candidates that drop a guard, and how much smaller they are
N_SEL = 200_000
k_grid = np.arange(1, 11)
sel = {d: {"size": [], "unsafe": [], "none": []} for d in (0.0, 0.8, 0.95)}
single = np.exp(rng.normal(-SIGMA_W ** 2 / 2, SIGMA_W, N_SEL))
unsafe_single = rng.random(N_SEL) < Q_UNSAFE
single = np.where(unsafe_single, SHRINK * single, single)
ref_size = single.mean()
for d in sel:
    for k in k_grid:
        size = np.exp(rng.normal(-SIGMA_W ** 2 / 2, SIGMA_W, (N_SEL, k)))
        unsafe = rng.random((N_SEL, k)) < Q_UNSAFE
        size = np.where(unsafe, SHRINK * size, size)
        flagged = unsafe & (rng.random((N_SEL, k)) < d)              # caught by the safety probes
        size = np.where(flagged, np.inf, size)
        pick = np.argmin(size, axis=1)
        chosen = size[np.arange(N_SEL), pick]
        none = ~np.isfinite(chosen)
        sel[d]["none"].append(none.mean())
        sel[d]["unsafe"].append(np.mean(unsafe[np.arange(N_SEL), pick] & ~none))
        sel[d]["size"].append(chosen[~none].mean() / ref_size)
for d in sel:
    for k in (1, 3, 5, 10):
        i = k - 1
        print(f"probe detection {d:.0%}, k = {k:>2}: size vs one unselected run {sel[d]['size'][i]:.2f} | "
              f"unsafe answer delivered {sel[d]['unsafe'][i]:.1%} | no answer {sel[d]['none'][i]:.2%}")
print(f"one unselected run: unsafe {np.mean(unsafe_single):.1%}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.15, wspace=0.28)
for d, color in ((0.0, ORANGE), (0.8, BLUE), (0.95, AQUA)):
    lab = "no safety probes" if d == 0 else f"probes catch {d:.0%} of unsafe candidates"
    ax1.plot(k_grid, sel[d]["size"], color=color, marker="o", ms=7, mec=SURFACE, mew=2, label=lab)
    ax2.plot(k_grid, 100 * np.array(sel[d]["unsafe"]), color=color, marker="o", ms=7, mec=SURFACE, mew=2, label=lab)
ax1.set_title("Size of the selected patch")
ax1.set_xlabel("candidates sampled (k)")
ax1.set_ylabel("relative to one unselected run")
ax1.set_ylim(0, 1.05)
ax2.set_title("Unsafe patch delivered")
ax2.set_xlabel("candidates sampled (k)")
ax2.set_ylabel("share of tickets (%)")
ax2.set_ylim(0, None)
ax2.legend(handles=[Line2D([0], [0], color=c, lw=2, label=l) for c, l in ((ORANGE, "no safety probes"), (BLUE, "probes catch 80% of unsafe candidates"), (AQUA, "probes catch 95% of unsafe candidates"))], loc="center right")
finish(fig, "fig5_selection.png", "Choosing the smallest patch also chooses the patch that skipped a check",
       f"Assumed: {Q_UNSAFE:.0%} of candidates drop a guard and are {1 - SHRINK:.0%} smaller on average; "
       f"run-to-run log SD {SIGMA_W}.")

print(f"\nfigures written to {FIG.name}/")
