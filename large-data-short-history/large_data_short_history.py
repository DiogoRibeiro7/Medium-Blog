"""
Simulations for "Selection, Dependence, Historical Coverage and Measurement: Four Limits on What a
Large Dataset Can Estimate".

Every number quoted in the article is printed by this script, and every figure is written by it.
All data are simulated under stated assumptions; the published figures the article cites
(Meng 2018; Bradley et al. 2021; Lazer et al. 2014; Ginsberg et al. 2009) enter only as inputs
to arithmetic checks and are labelled as such.

Run:  python large_data_short_history.py > results/output.txt   (under a minute)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import signal, stats

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(20261017)
Z95 = stats.norm.ppf(0.975)

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


def dot(ax, x, y, color, **kw):
    ax.plot(x, y, "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=3, **kw)


# ============================================================================
# 1. Selection: Meng's identity and a sample that converges to the wrong number
# ============================================================================
header("1. Selection bias does not shrink with n")

# --- 1a. The identity holds exactly on any finite population ----------------
N = 1_000_000
p_pop = 0.5
r_prop = 1.10  # inclusion propensity of a unit with Y = 1, relative to Y = 0
Y = (rng.random(N) < p_pop).astype(float)
f_check = 0.05
w = np.where(Y == 1, r_prop, 1.0)
R = (rng.random(N) < f_check * w / w.mean()).astype(float)
n_check = int(R.sum())
ybar_n, ybar_N = Y[R == 1].mean(), Y.mean()
sigma_Y = Y.std()                            # population SD (ddof = 0)
rho = np.corrcoef(R, Y)[0, 1]                # finite-population correlation
rhs = rho * np.sqrt((N - n_check) / n_check) * sigma_Y
print(f"Population N = {N:,}, population mean {ybar_N:.5f}, sigma_Y = {sigma_Y:.5f}")
print(f"Selected sample: n = {n_check:,}, propensity ratio r = {r_prop}")
print(f"  sample mean - population mean      = {ybar_n - ybar_N:+.10f}")
print(f"  rho * sqrt((N-n)/n) * sigma_Y      = {rhs:+.10f}")
print(f"  data defect correlation rho        = {rho:.6f}")
print(f"  absolute difference (identity)     = {abs((ybar_n - ybar_N) - rhs):.2e}")

# --- 1b. Expected bias for a propensity ratio --------------------------------
p_sel = r_prop * p_pop / (r_prop * p_pop + (1 - p_pop))
bias = p_sel - p_pop
sigma = np.sqrt(p_pop * (1 - p_pop))
n_star = (Z95 * sigma / bias) ** 2
print(f"\nWith r = {r_prop} and p = {p_pop}: mean among the selected = {p_sel:.5f}, "
      f"bias = {100 * bias:.2f} percentage points")
print(f"Nominal 95% half-width falls below the bias once n > (1.96 sigma / bias)^2 = {n_star:,.0f}")

# --- 1c. Monte Carlo across sample sizes (exact Poisson sampling) ------------
N1 = int(Y.sum())
N0 = N - N1
truth = N1 / N
n_grid = np.array([100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000])
reps = 4000
res = {"random": {"rmse": [], "cover": [], "half": []}, "selected": {"rmse": [], "cover": [], "half": []}}
for n in n_grid:
    for design, ratio in (("random", 1.0), ("selected", r_prop)):
        scale = n / (ratio * N1 + N0)
        k1 = rng.binomial(N1, ratio * scale, size=reps)
        k0 = rng.binomial(N0, scale, size=reps)
        nn = k1 + k0
        phat = k1 / nn
        half = Z95 * np.sqrt(phat * (1 - phat) / nn * (1 - nn / N))  # with finite-population correction
        res[design]["rmse"].append(np.sqrt(np.mean((phat - truth) ** 2)))
        res[design]["cover"].append(np.mean(np.abs(phat - truth) <= half))
        res[design]["half"].append(np.mean(half))
print(f"\nMonte Carlo, {reps} samples per size, population mean {truth:.5f}")
print(f"{'n':>9} | {'RMSE random':>12} {'RMSE selected':>14} | {'95% half-width':>15} | "
      f"{'coverage random':>16} {'coverage selected':>18}")
for i, n in enumerate(n_grid):
    print(f"{n:>9,} | {100 * res['random']['rmse'][i]:>10.3f}pp {100 * res['selected']['rmse'][i]:>12.3f}pp | "
          f"{100 * res['selected']['half'][i]:>13.3f}pp | {100 * res['random']['cover'][i]:>15.1f}% "
          f"{100 * res['selected']['cover'][i]:>17.1f}%")

# --- 1d. Bias-adjusted effective sample size ---------------------------------
def n_eff_mse(bias_, sigma_, n_, N_):
    mse = bias_ ** 2 + sigma_ ** 2 * (1 - n_ / N_) / n_
    return sigma_ ** 2 / mse


N_us, n_fb = 255_000_000, 250_000
print("\nBias-adjusted effective sample size n_eff = sigma^2 / MSE (simple-random-sample equivalent)")
print(f"  N = {N_us:,} adults, n = {n_fb:,}, bias {100 * bias:.2f}pp (r = {r_prop}): "
      f"n_eff = {n_eff_mse(bias, sigma, n_fb, N_us):,.0f}")
f_cces, rho_cces = 0.01, 0.005
print(f"  Check against Meng (2018): f = {f_cces}, rho = {rho_cces} -> "
      f"n_eff = f/(1-f)/rho^2 = {f_cces / (1 - f_cces) / rho_cces ** 2:,.0f}  (Meng reports about 400)")
for bench in (0.5, 0.6):
    s = np.sqrt(bench * (1 - bench))
    print(f"  Check against Bradley et al. (2021): 17pp error, benchmark {bench:.0%} -> "
          f"n_eff = {n_eff_mse(0.17, s, n_fb, N_us):.1f}  (they report fewer than 10)")
f_fb = n_fb / N_us
rho10 = np.sqrt(f_fb / (1 - f_fb) / 10)
print(f"  |rho| needed for n_eff = 10 at n = {n_fb:,} of {N_us:,}: {rho10:.4f}")
rho_r11 = bias / (np.sqrt((N_us - n_fb) / n_fb) * sigma)
print(f"  for comparison, rho implied by r = {r_prop} at that sampling fraction: {rho_r11:.5f}")
bias10 = rho10 * np.sqrt((N_us - n_fb) / n_fb) * sigma
ps10 = p_pop + bias10
r10 = ps10 * (1 - p_pop) / (p_pop * (1 - ps10))
print(f"  rho = {rho10:.4f} at that fraction means a bias of {100 * bias10:.1f}pp for a 50/50 outcome, "
      f"i.e. units with Y = 1 recorded {r10:.2f} times as often as units with Y = 0")

# --- Figure 1 ---------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.9))
fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.15, wspace=0.28)
ax1.plot(n_grid, 100 * np.array(res["random"]["rmse"]), color=BLUE, marker="o", ms=7, mec=SURFACE, mew=2)
ax1.plot(n_grid, 100 * np.array(res["selected"]["rmse"]), color=ORANGE, marker="o", ms=7, mec=SURFACE, mew=2)
ax1.plot(n_grid, 100 * np.array(res["selected"]["half"]), color=INK2, ls="--", lw=1.5)
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel("observations, n"); ax1.set_ylabel("percentage points")
ax1.set_title("Root mean squared error of the sample mean")
ax1.text(n_grid[-1], 100 * res["random"]["rmse"][-1] * 1.25, "random sample", color=INK2, ha="right", fontsize=9.5)
ax1.text(n_grid[-1], 100 * res["selected"]["rmse"][-1] * 1.25, "selected sample (r = 1.1)", color=INK2,
         ha="right", fontsize=9.5)
ax1.text(n_grid[4], 100 * res["selected"]["half"][4] * 1.3, "reported 95% margin", color=INK2, fontsize=9.5)
ax1.axhline(100 * bias, color=MUTED, lw=1, ls=":")
ax1.text(n_grid[0], 100 * bias * 0.78, "bias 2.38", color=MUTED, fontsize=9)
ax1.axvline(n_star, color=AXIS, lw=1)
ax1.text(n_star * 1.12, 0.08, f"margin = bias\nat n ≈ {n_star:,.0f}", color=MUTED, fontsize=9)
ax2.plot(n_grid, 100 * np.array(res["random"]["cover"]), color=BLUE, marker="o", ms=7, mec=SURFACE, mew=2,
         label="random sample")
ax2.plot(n_grid, 100 * np.array(res["selected"]["cover"]), color=ORANGE, marker="o", ms=7, mec=SURFACE, mew=2,
         label="selected sample (r = 1.1)")
ax2.axhline(95, color=MUTED, lw=1)
ax2.text(n_grid[0], 97, "nominal 95%", color=MUTED, fontsize=9)
ax2.set_xscale("log"); ax2.set_ylim(-3, 103)
ax2.set_xlabel("observations, n"); ax2.set_ylabel("% of intervals containing the truth")
ax2.set_title("Coverage of the reported 95% interval")
ax2.legend(loc="lower left")
finish(fig, "fig1_selection.png", "More observations, same bias, narrower intervals",
       "Population of 1,000,000, half with Y = 1. Units with Y = 1 are 10% more likely to be recorded. "
       f"{reps:,} samples per size.")

# ============================================================================
# 2. Dependence: autocorrelation and clustering
# ============================================================================
header("2. Correlated observations: effective sample size")


def ar1_neff_exact(n, phi):
    k = np.arange(1, n)
    return n / (1 + 2 * np.sum((1 - k / n) * phi ** k))


n_ar = 10_000
phis = np.array([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
reps_ar = 4000
neff_mc, cover_mc = [], []
for phi in phis:
    xbar, sd = [], []
    for _ in range(reps_ar // 500):
        e = rng.standard_normal((500, n_ar))
        x0 = rng.standard_normal(500) / np.sqrt(1 - phi ** 2)
        x = signal.lfilter([1.0], [1.0, -phi], e, axis=1, zi=(phi * x0)[:, None])[0]
        xbar.append(x.mean(axis=1)); sd.append(x.std(axis=1, ddof=1))
    xbar, sd = np.concatenate(xbar), np.concatenate(sd)
    var_marg = 1 / (1 - phi ** 2)
    neff_mc.append(var_marg / np.mean(xbar ** 2))
    cover_mc.append(np.mean(np.abs(xbar) <= Z95 * sd / np.sqrt(n_ar)))
print(f"AR(1), n = {n_ar:,}, {reps_ar} series per phi")
print(f"{'phi':>6} | {'n(1-phi)/(1+phi)':>17} {'exact':>10} {'Monte Carlo':>12} | {'naive 95% CI coverage':>22}")
for phi, nm, cv in zip(phis, neff_mc, cover_mc):
    print(f"{phi:>6.2f} | {n_ar * (1 - phi) / (1 + phi):>17,.1f} {ar1_neff_exact(n_ar, phi):>10,.1f} "
          f"{nm:>12,.1f} | {100 * cv:>21.1f}%")
phi9 = 0.9
ne9 = ar1_neff_exact(n_ar, phi9)
print(f"Theoretical naive coverage at phi = {phi9}: 2*Phi(1.96*sqrt(n_eff/n)) - 1 = "
      f"{100 * (2 * stats.norm.cdf(Z95 * np.sqrt(ne9 / n_ar)) - 1):.1f}%")

# --- Clustering --------------------------------------------------------------
n_big = 1_000_000
print(f"\nClustered data, n = {n_big:,}: design effect 1 + (m - 1) * ICC (Kish)")
for m, icc in ((1000, 0.01), (1000, 0.05), (100, 0.05), (10, 0.05)):
    deff = 1 + (m - 1) * icc
    print(f"  clusters of m = {m:>5,}, ICC = {icc:.2f}: design effect {deff:6.2f}, n_eff = {n_big / deff:,.0f}")
G, m_sim, icc_sim, reps_cl = 100, 200, 0.05, 2000
gm = np.empty(reps_cl)
for i in range(reps_cl):
    u = rng.normal(0, np.sqrt(icc_sim), G)
    e = rng.normal(0, np.sqrt(1 - icc_sim), (G, m_sim))
    gm[i] = (u[:, None] + e).mean()
deff_mc = np.var(gm) * G * m_sim
print(f"  Monte Carlo check, {G} clusters of {m_sim}, ICC {icc_sim}: design effect {deff_mc:.2f} "
      f"(formula {1 + (m_sim - 1) * icc_sim:.2f})")

# --- Figure 2 ---------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.9))
fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.15, wspace=0.3)
phi_line = np.linspace(0, 0.99, 200)
ax1.plot(phi_line, [ar1_neff_exact(n_ar, p) for p in phi_line], color=BLUE)
for phi, nm in zip(phis, neff_mc):
    ax1.plot(phi, nm, "s", ms=7, color=INK2, mfc=SURFACE, mew=1.5, zorder=3)
ax1.set_yscale("log"); ax1.set_ylim(40, 15000)
ax1.set_xlabel("lag-one autocorrelation, φ"); ax1.set_ylabel("effective sample size")
ax1.set_title(f"{n_ar:,} observations of an AR(1) process")
dot(ax1, phi9, ne9, BLUE)
ax1.annotate(f"φ = 0.9: {ne9:,.0f}", (phi9, ne9), xytext=(-20, -4), textcoords="offset points",
             ha="right", color=INK2, fontsize=9.5)
ax1.legend(handles=[Line2D([0], [0], color=BLUE, lw=2, label="exact formula"),
                    Line2D([0], [0], marker="s", ls="", ms=7, color=INK2, mfc=SURFACE, mew=1.5,
                           label="Monte Carlo")], loc="lower left")
icc_line = np.logspace(-3, -0.5, 200)
for m, color in ((10, AQUA), (100, BLUE), (1000, ORANGE)):
    ax2.plot(icc_line, n_big / (1 + (m - 1) * icc_line), color=color, label=f"clusters of {m:,}")
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel("intra-cluster correlation"); ax2.set_ylabel("effective sample size")
ax2.set_title(f"{n_big:,} observations in clusters")
ax2.axvline(0.05, color=AXIS, lw=1)
dot(ax2, 0.05, n_big / (1 + 999 * 0.05), ORANGE)
ax2.annotate(f"{n_big / (1 + 999 * 0.05):,.0f}", (0.05, n_big / (1 + 999 * 0.05)), xytext=(8, 6),
             textcoords="offset points", color=INK2, fontsize=9.5)
ax2.legend(loc="lower left")
finish(fig, "fig2_dependence.png", "Rows are not independent observations",
       "Effective sample size: the number of independent observations that would estimate the mean "
       "as precisely.")

# ============================================================================
# 3. Sampling frequency versus observation horizon (Brownian motion with drift)
# ============================================================================
header("3. Drift and volatility: frequency versus horizon")
mu, sig = 0.06, 0.20
T_list = [5, 20, 80]
m_list = np.array([1, 4, 12, 52, 252, 252 * 78])
m_names = ["annual", "quarterly", "monthly", "weekly", "daily", "5-minute"]
reps_bm = 20_000
sd_mu, sd_sig = {}, {}
for T in T_list:
    sd_mu[T], sd_sig[T] = [], []
    for m in m_list:
        n = m * T
        # Increments are i.i.d. N(mu*dt, sig^2*dt); the sum and the centred sum of squares are
        # independent, so drawing them directly is exact, not an approximation.
        mu_hat = mu + sig * rng.standard_normal(reps_bm) / np.sqrt(T)
        sig_hat = sig * np.sqrt(rng.chisquare(n - 1, reps_bm) / n) if n > 1 else np.full(reps_bm, np.nan)
        sd_mu[T].append(mu_hat.std()); sd_sig[T].append(np.nanstd(sig_hat) if n > 1 else np.nan)
print(f"mu = {mu}, sigma = {sig} per year; {reps_bm:,} draws per cell; SDs in percentage points")
print(f"{'horizon':>8} {'sampling':>10} {'obs':>10} | {'SD(mu_hat)':>10} {'sig/sqrt(T)':>12} | "
      f"{'SD(sig_hat)':>11} {'sig/sqrt(2n)':>13}")
for T in T_list:
    for j, m in enumerate(m_list):
        n = m * T
        s_sig = f"{100 * sd_sig[T][j]:>11.3f}" if n > 1 else f"{'n/a':>11}"
        print(f"{T:>6} y {m_names[j]:>10} {n:>10,} | {100 * sd_mu[T][j]:>10.3f} {100 * sig / np.sqrt(T):>12.3f} | "
              f"{s_sig} {100 * sig / np.sqrt(2 * n):>13.3f}")
# Direct path simulation: one five-minute path per replication, aggregated to every coarser grid,
# so all grids estimate from the same realised path.
T_chk, reps_path, chunk_p = 20, 300, 25
m_fine = int(m_list[-1])
mu_by_grid = {int(m): [] for m in m_list}
sig_by_grid = {int(m): [] for m in m_list}
for _ in range(reps_path // chunk_p):
    inc = mu / m_fine + sig / np.sqrt(m_fine) * rng.standard_normal((chunk_p, T_chk * m_fine))
    for m in m_list:
        m = int(m)
        agg = inc.reshape(chunk_p, T_chk * m, m_fine // m).sum(axis=2)
        mu_by_grid[m].append(agg.sum(axis=1) / T_chk)
        if T_chk * m > 1:
            sig_by_grid[m].append(np.sqrt(((agg - agg.mean(axis=1, keepdims=True)) ** 2).sum(axis=1) * m / (T_chk * m)))
mu_by_grid = {m: np.concatenate(v) for m, v in mu_by_grid.items()}
sig_by_grid = {m: np.concatenate(v) for m, v in sig_by_grid.items()}
max_diff = max(np.max(np.abs(mu_by_grid[int(m)] - mu_by_grid[1])) for m in m_list)
print(f"\nDirect path simulation: {reps_path} paths of {T_chk} years at five-minute resolution, "
      f"each aggregated to every coarser grid")
print(f"  largest difference between drift estimates from different grids of the same path: {max_diff:.1e}")
print(f"  SD of the drift estimate across paths: {100 * mu_by_grid[1].std():.3f}pp (formula {100 * sig / np.sqrt(T_chk):.3f})")
for j, m in enumerate(m_list):
    print(f"  {m_names[j]:>10}: SD(sig_hat) = {100 * sig_by_grid[int(m)].std():.3f}pp "
          f"(formula {100 * sig / np.sqrt(2 * T_chk * m):.3f})")
print(f"Years of data for a 95% interval on mu to exclude zero: (1.96 sigma / mu)^2 = {(Z95 * sig / mu) ** 2:.1f}")
print(f"t-statistic for mu after 20 years at any sampling frequency: mu / (sigma/sqrt(20)) = "
      f"{mu / (sig / np.sqrt(20)):.2f}")

# --- Figure 3 ---------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.9))
fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.17, wspace=0.28)
colors = {5: AQUA, 20: BLUE, 80: ORANGE}
for T in T_list:
    ax1.plot(m_list, 100 * np.array(sd_mu[T]), color=colors[T], marker="o", ms=7, mec=SURFACE, mew=2,
             label=f"{T} years of data")
    ok = m_list * T > 1
    ax2.plot(m_list[ok], 100 * np.array(sd_sig[T])[ok], color=colors[T], marker="o", ms=7, mec=SURFACE, mew=2,
             label=f"{T} years of data")
for ax in (ax1, ax2):
    ax.set_xscale("log")
    ax.set_xticks(m_list, ["year", "qtr", "month", "week", "day", "5 min"])
    ax.minorticks_off()
    ax.set_xlabel("sampling interval")
ax2.set_yscale("log")
ax1.set_ylim(0, 10.5)
ax1.set_ylabel("standard deviation, percentage points")
ax1.set_title("Estimating the drift μ = 6% a year")
ax2.set_title("Estimating the volatility σ = 20% a year")
ax1.axhline(100 * mu, color=MUTED, lw=1)
ax1.text(m_list[-1], 100 * mu + 0.25, "size of the drift", ha="right", color=MUTED, fontsize=9)
for T in T_list:
    ax1.text(m_list[0], 100 * sd_mu[T][0] + 0.35, f"{T} years of data", color=INK2, fontsize=9.5)
ax2.legend(loc="lower left")
finish(fig, "fig3_frequency_horizon.png", "Sampling more often sharpens volatility, not drift",
       "Brownian motion with drift. Precision of the drift depends on the length of the record only.")

# ============================================================================
# 4. Overlapping windows: how many decades are in a century?
# ============================================================================
header("4. Overlapping windows")
years, window = 100, 10
mu_a, sig_a = 0.06, 0.20
p_true = stats.norm.cdf(-mu_a * window / (sig_a * np.sqrt(window)))
reps_w = 20_000
annual = mu_a + sig_a * rng.standard_normal((reps_w, years))
csum = np.concatenate([np.zeros((reps_w, 1)), np.cumsum(annual, axis=1)], axis=1)
roll = csum[:, window:] - csum[:, :-window]          # 91 overlapping decade returns
blocks = annual.reshape(reps_w, years // window, window).sum(axis=2)  # 10 non-overlapping decades
p_over = (roll < 0).mean(axis=1)
p_block = (blocks < 0).mean(axis=1)
n_over = roll.shape[1]
print(f"{years} years of i.i.d. annual returns, mu = {mu_a}, sigma = {sig_a}; "
      f"true P(losing decade) = {p_true:.4f}")
print(f"  overlapping windows: {n_over}; SD of estimate {p_over.std():.4f}; "
      f"binomial SD if they were independent {np.sqrt(p_true * (1 - p_true) / n_over):.4f}")
print(f"  non-overlapping decades: {years // window}; SD of estimate {p_block.std():.4f}; "
      f"binomial SD {np.sqrt(p_true * (1 - p_true) / (years // window)):.4f}")
print(f"  effective number of independent decades in the overlapping estimate: "
      f"{p_true * (1 - p_true) / p_over.var():.1f}")
naive_cover = np.mean(np.abs(p_over - p_true) <= Z95 * np.sqrt(p_over * (1 - p_over) / n_over))
print(f"  coverage of a naive 95% interval that treats the {n_over} windows as independent: {100 * naive_cover:.1f}%")

# ============================================================================
# 5. Screening many candidates against a short, smooth record
# ============================================================================
header("5. Screening candidates against a short record")
n_weeks, K, chunk = 128, 1_000_000, 200_000
for phi_s in (0.0, 0.95):
    x0 = rng.standard_normal() / np.sqrt(1 - phi_s ** 2)
    target = signal.lfilter([1.0], [1.0, -phi_s], rng.standard_normal(n_weeks), zi=[phi_s * x0])[0]
    tz = (target - target.mean()) / target.std()
    r_all = []
    for _ in range(K // chunk):
        x0 = rng.standard_normal(chunk) / np.sqrt(1 - phi_s ** 2)
        e = rng.standard_normal((chunk, n_weeks))
        cand = signal.lfilter([1.0], [1.0, -phi_s], e, axis=1, zi=(phi_s * x0)[:, None])[0]
        cz = (cand - cand.mean(axis=1, keepdims=True)) / cand.std(axis=1, keepdims=True)
        r_all.append(cz @ tz / n_weeks)
    r_all = np.abs(np.concatenate(r_all))
    n_eff_r = n_weeks * (1 - phi_s ** 2) / (1 + phi_s ** 2)
    print(f"phi = {phi_s:.2f}: {K:,} candidate series unrelated to the target by construction, {n_weeks} weeks")
    print(f"  Bartlett effective n for a correlation n(1-phi^2)/(1+phi^2) = {n_eff_r:.1f}")
    print(f"  effective n from the simulated correlations, 1/Var(r) = {1 / np.mean(r_all ** 2):.1f}")
    print(f"  largest |r| = {r_all.max():.3f}; 99.9th percentile |r| = {np.quantile(r_all, 0.999):.3f}; "
          f"candidates with |r| > 0.7: {np.sum(r_all > 0.7):,}")
print(f"\nGinsberg et al. (2009) input for comparison: {n_weeks} weekly points in each of 9 regions = "
      f"{n_weeks * 9:,} data points, screened against 50 million queries")
