"""Experiment 3 - which test of a coefficient keeps its level: pretests, HC types, wild bootstrap.

Model: y = 1 + 1 * x + sigma(x) * eps, eps ~ N(0, 1), test H0: slope = 1 at 5%.

Designs for x (n draws per sample):
  uniform     U(0.2, 2): no high-leverage points
  lognormal   exp(N(0, 1)): a long right tail, a few points with large leverage
  income      the incomes of single-person households in wooldridge::k401ksubs
              (2,017 values, $1,000s), drawn with replacement
Variance: sigma(x) proportional to x^gamma, the textbook power form, with
gamma in {0, 0.5, 1, 1.5} (gamma = 1: standard deviation proportional to x),
scaled so that E sigma^2 = 1 in the design's population.
n in {25, 50, 100, 200, 500}; 2,000 samples per cell.

Methods: t tests with the classical, HC0, HC1, HC2, HC3, HC4, HC4m and HC5
covariance and t(n - 2) critical values; two pretest rules (Koenker's
Breusch-Pagan test or White's test at 5%: HC3 if it rejects, classical
otherwise); restricted wild bootstrap with Rademacher and Webb weights and
unrestricted wild bootstrap with Rademacher weights (t with HC1, B = 399).
Also recorded: standard-error ratios to the classical one, and power: the
rejection rate of the false null slope = 1 - delta, with delta = 2.5 /
sqrt(n var(x)), 2.5 standard errors of the slope under homoscedasticity.

Run:  python 03_simulation_inference.py            (about 10 minutes on 2 cores)
      python 03_simulation_inference.py --reps 200 (quick run)
      python 03_simulation_inference.py --figures-only (digest and figures from the saved summary)
"""

from __future__ import annotations

import argparse
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from common import AQUA, BLUE, FIG, INK, INK2, ORANGE, RES, VIOLET, YELLOW, fetch, style, write
from het import bp_test, ols, white_test, wild_test

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=2000)
ap.add_argument("--figures-only", action="store_true", help="redraw the figures from results/03_inference_summary.csv")
ARGS, _ = ap.parse_known_args()

k = fetch("wooldridge", "k401ksubs")
INCOME = k[k.fsize == 1].inc.values
DESIGNS = ("uniform", "lognormal", "income")
GAMMAS = (0.0, 0.5, 1.0, 1.5)
NS = (25, 50, 100, 200, 500)
B = 399
HCS = ("const", "HC0", "HC1", "HC2", "HC3", "HC4", "HC4m", "HC5")
METHODS = ["Classical", "HC0", "HC1", "HC2", "HC3", "HC4", "HC4m", "HC5", "Pretest BP -> HC3",
           "Pretest White -> HC3", "Wild restricted Rademacher", "Wild restricted Webb",
           "Wild unrestricted Rademacher"]
POWER_METHODS = METHODS[:11]

# population moments of each design, for standardising x inside sigma(x)
_pop_rng = np.random.default_rng(0)
POP = {"uniform": _pop_rng.uniform(0.2, 2, size=1_000_000), "lognormal": np.exp(_pop_rng.normal(size=1_000_000)),
       "income": INCOME}
MOM = {d: (v.mean(), v.std()) for d, v in POP.items()}
SCALE = {(d, g): np.mean(POP[d] ** (2 * g)) for d in DESIGNS for g in GAMMAS}


def draw_x(design, n, rng):
    if design == "uniform":
        return rng.uniform(0.2, 2, size=n)
    if design == "lognormal":
        return np.exp(rng.normal(size=n))
    return rng.choice(INCOME, n, replace=True)


def one(task):
    d, g, n, rep = task
    rng = np.random.default_rng([3, DESIGNS.index(d), GAMMAS.index(g), NS.index(n), rep])
    x = draw_x(d, n, rng)
    sig = x ** g / np.sqrt(SCALE[(d, g)])
    X = np.column_stack([np.ones(n), x])
    y = 1 + x + sig * rng.standard_normal(n)
    f = ols(X, y)
    tcrit = stats.t.ppf(0.975, n - 2)
    rej = {}
    se = {t: f.se(t)[1] for t in HCS}
    for t, m in zip(HCS, METHODS[:8]):
        rej[m] = abs(f.coef[1] - 1) / se[t] > tcrit
    p_bp = bp_test(f.resid, x)[1]
    p_wh = white_test(f.resid, x)[1]
    rej["Pretest BP -> HC3"] = rej["HC3"] if p_bp < 0.05 else rej["Classical"]
    rej["Pretest White -> HC3"] = rej["HC3"] if p_wh < 0.05 else rej["Classical"]
    rej["Wild restricted Rademacher"] = wild_test(X, y, 1, 1.0, B=B, kind="rademacher", rng=rng) < 0.05
    rej["Wild restricted Webb"] = wild_test(X, y, 1, 1.0, B=B, kind="webb", rng=rng) < 0.05
    rej["Wild unrestricted Rademacher"] = wild_test(X, y, 1, 1.0, B=B, kind="rademacher", restricted=False,
                                                    rng=rng) < 0.05
    # power against slope = 1 - delta
    b_alt = 1 - 2.5 / np.sqrt(n * MOM[d][1] ** 2)
    pw = {}
    for t, m in zip(HCS, METHODS[:8]):
        pw[m] = abs(f.coef[1] - b_alt) / se[t] > tcrit
    pw["Pretest BP -> HC3"] = pw["HC3"] if p_bp < 0.05 else pw["Classical"]
    pw["Pretest White -> HC3"] = pw["HC3"] if p_wh < 0.05 else pw["Classical"]
    pw["Wild restricted Rademacher"] = wild_test(X, y, 1, b_alt, B=B, kind="rademacher", rng=rng) < 0.05
    return (d, g, n, rep, p_bp, p_wh, se["HC3"] / se["const"], se["HC4"] / se["const"],
            se["HC5"] / se["const"], *[bool(rej[m]) for m in METHODS],
            *[bool(pw[m]) for m in POWER_METHODS])


def summarise(df):
    rows = []
    out = [f"Rejection rates of a true null at 5% ({df.rep.nunique()} samples per cell; Monte Carlo SE at 5% "
           f"~ {np.sqrt(.05 * .95 / df.rep.nunique()):.4f})"]
    for (d, g, n), s in df.groupby(["design", "gamma", "n"], sort=False):
        r = {m: s[m].mean() for m in METHODS}
        pwr = {f"power {m}": s[f"power {m}"].mean() for m in POWER_METHODS}
        rows.append({"design": d, "gamma": g, "n": n, **r, **pwr,
                     "bp_reject": (s.p_bp < 0.05).mean(), "white_reject": (s.p_white < 0.05).mean(),
                     "se_ratio_hc3": s.se_ratio_hc3.median(), "se_ratio_hc4": s.se_ratio_hc4.median(),
                     "se_ratio_hc5": s.se_ratio_hc5.median(),
                     "classical_rej_when_bp_accepts": s.loc[s.p_bp >= 0.05, "Classical"].mean(),
                     "classical_rej_when_bp_rejects": s.loc[s.p_bp < 0.05, "Classical"].mean(),
                     "hc3_rej_when_bp_accepts": s.loc[s.p_bp >= 0.05, "HC3"].mean(),
                     "hc3_rej_when_bp_rejects": s.loc[s.p_bp < 0.05, "HC3"].mean()})
        out.append(f"-- {d}, gamma {g}, n {n}: BP rejects {(s.p_bp < 0.05).mean():.3f}, White rejects "
                   f"{(s.p_white < 0.05).mean():.3f}; median SE ratio HC3/classical {s.se_ratio_hc3.median():.3f}, "
                   f"HC4 {s.se_ratio_hc4.median():.3f}, HC5 {s.se_ratio_hc5.median():.3f}")
        out.append("   size: " + "; ".join(f"{m} {r[m]:.4f}" for m in METHODS))
        out.append("   power: " + "; ".join(f"{m} {pwr['power ' + m]:.4f}" for m in POWER_METHODS))
        acc = s.p_bp >= 0.05
        out.append(f"   classical test rejects {s.loc[acc, 'Classical'].mean():.4f} when BP accepts "
                   f"({acc.sum()} samples) and {s.loc[~acc, 'Classical'].mean():.4f} when BP rejects; HC3 "
                   f"{s.loc[acc, 'HC3'].mean():.4f} and {s.loc[~acc, 'HC3'].mean():.4f}")
    pd.DataFrame(rows).to_csv(RES / "03_inference_summary.csv", index=False)
    return "\n".join(out)


def digest(S, reps):
    """Summaries of the saved cell means, written to results/03_inference_digest.txt."""
    band = 1.96 * np.sqrt(.05 * .95 / reps)
    out = [f"Digest of {len(S)} cells. A rejection rate within 5% +/- {band:.4f} (1.96 Monte Carlo SE) counts as "
           f"'at level'."]
    for m in METHODS:
        v = S[m]
        line = (f"   {m:28s} at level in {np.sum(np.abs(v - .05) <= band)} of {len(S)} cells; range "
                f"{v.min():.4f}-{v.max():.4f}")
        for d in DESIGNS:
            w = S.loc[S.design == d, m]
            line += f"; {d} {w.min():.4f}-{w.max():.4f}"
        out.append(line)
    het = S[S.gamma > 0]
    for rule in ("Pretest BP -> HC3", "Pretest White -> HC3"):
        diff = het[rule] - het["HC3"]
        out.append(f"{rule} minus always-HC3 in the {len(het)} heteroscedastic cells: higher in {np.sum(diff > 0)}, "
                   f"equal in {np.sum(diff == 0)}, lower in {np.sum(diff < 0)}; largest excess {diff.max():.4f}")
    for d in DESIGNS:
        for lab_, sel in (("n <= 50", het.n <= 50), ("n >= 200", het.n >= 200)):
            dd = (het["Pretest BP -> HC3"] - het["HC3"])[(het.design == d) & sel]
            out.append(f"   {d}, {lab_}: BP pretest minus always-HC3 {dd.min():+.4f} to {dd.max():+.4f} "
                       f"({np.sum(dd > 0)} of {len(dd)} cells higher)")
    ok = (1 - het.bp_reject) * reps >= 100
    sub = het[ok]
    out.append(f"Classical test in the samples where Breusch-Pagan accepts (cells with at least 100 such samples, "
               f"{len(sub)} of {len(het)}): rejection {sub.classical_rej_when_bp_accepts.min():.4f}-"
               f"{sub.classical_rej_when_bp_accepts.max():.4f}; higher than the unconditional classical rate in "
               f"{np.sum(sub.classical_rej_when_bp_accepts > sub.Classical)} of {len(sub)}")
    homo = S[S.gamma == 0]
    out.append("Power when the variance is constant (classical exact), relative to the classical test, mean over n: "
               + "; ".join(f"{d}: " + ", ".join(
                   f"{m} {np.mean(homo.loc[homo.design == d, f'power {m}'] / homo.loc[homo.design == d, 'power Classical']):.3f}"
                   for m in ("HC1", "HC3", "HC4", "HC5", "Wild restricted Rademacher")) for d in DESIGNS))
    # the leverage that drives HC4 and HC5: n * max(h) / p in fresh draws of each design
    rng = np.random.default_rng(99)
    for d in DESIGNS:
        vals = []
        for n in NS:
            r = []
            for _ in range(2000):
                x = draw_x(d, n, rng)
                h = 1 / n + (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)
                r.append(n * h.max() / 2)
            vals.append(f"n {n}: {np.median(r):.1f}")
        out.append(f"Median of n max(h) / p, {d} design (HC4 caps its exponent at 4; HC5 at max(4, 0.7 n max(h) / p)): "
                   + "; ".join(vals))
    write("03_inference_digest.txt", "\n".join(out))


def figures(S):
    style()
    pct = FuncFormatter(lambda v, _: f"{v:.0%}")
    # size by n: three designs, no heteroscedasticity and gamma = 1
    show = [("Classical", INK, "o", "-"), ("HC1", YELLOW, "v", "-"), ("HC3", BLUE, "s", "-"),
            ("HC4", AQUA, "^", "-"), ("HC5", VIOLET, "D", "-"), ("Wild restricted Webb", ORANGE, "P", "--")]
    lab = {"Wild restricted Webb": "Wild bootstrap (restricted, Webb)"}
    dname = {"uniform": "x uniform (no leverage)", "lognormal": "x lognormal (long tail)",
             "income": "x = household income"}
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.6), sharex=True, gridspec_kw={"hspace": 0.32, "wspace": 0.22})
    for r, g in enumerate((0.0, 1.0)):
        for c, d in enumerate(DESIGNS):
            ax = axes[r, c]
            s = S[(S.design == d) & (S.gamma == g)]
            ax.axhspan(0.05 - 1.96 * np.sqrt(.05 * .95 / ARGS.reps), 0.05 + 1.96 * np.sqrt(.05 * .95 / ARGS.reps),
                       color="#d9d8d3", lw=0, zorder=0)
            off = []
            for m, col, mk, ls in show:
                v = s[m].to_numpy()
                ax.plot(range(len(NS)), np.where(v > 0.30, np.nan, v), marker=mk, color=col, ls=ls, lw=1.6, ms=5,
                        label=lab.get(m, m))
                hi = v > 0.30
                if hi.all():
                    off.append(f"{lab.get(m, m)} {v.min():.0%}-{v.max():.0%}")
                elif hi.any():
                    idx = np.flatnonzero(hi)
                    where = f"n = {NS[idx[0]]}" if len(idx) == 1 else f"n = {NS[idx[0]]}-{NS[idx[-1]]}"
                    rng_ = f"{v[idx].min():.0%}" if len(idx) == 1 else f"{v[idx].min():.0%}-{v[idx].max():.0%}"
                    off.append(f"{lab.get(m, m)} {rng_} at {where}")
            if off:
                ax.annotate("Above the scale:\n" + "\n".join(off), (0.98, 0.97), xycoords="axes fraction", ha="right",
                            va="top", fontsize=8.5, color=INK)
            ax.set_ylim(0, 0.31)
            ax.yaxis.set_major_formatter(pct)
            ax.set_xticks(range(len(NS)), [str(n) for n in NS])
            if r == 0:
                ax.set_title(dname[d], fontsize=10.5)
            if c == 0:
                ax.set_ylabel(("Constant variance" if g == 0 else "SD proportional to x") +
                              "\nrejection rate of a true null")
            if r == 1:
                ax.set_xlabel("Sample size n")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8.5, bbox_to_anchor=(0.5, -0.03), frameon=False)
    fig.suptitle("Nominal 5% tests of a slope: the grey band is 5% ± Monte Carlo error", x=0.07, ha="left",
                 fontsize=11.5, fontweight="bold", y=0.99)
    fig.savefig(FIG / "fig4_size_simulation.png"); plt.close(fig)

    # the pretest: what happens to the classical test in the samples where the test for heteroscedasticity passes
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, d in zip(axes, ("lognormal", "income")):
        s = S[(S.design == d) & (S.gamma == 0.5)].reset_index(drop=True)
        xs = np.arange(len(NS))
        ax.plot(xs, s["Classical"], "o-", color=INK, lw=1.6, ms=5, label="Always classical")
        ax.plot(xs, s["Pretest BP -> HC3"], "D-", color=ORANGE, lw=1.6, ms=5,
                label="Pretest: Breusch-Pagan at 5%, HC3 if it rejects")
        ax.plot(xs, s["HC3"], "s-", color=BLUE, lw=1.6, ms=5, label="Always HC3")
        ok = (1 - s.bp_reject) * ARGS.reps >= 100
        ax.plot(xs[ok], s.loc[ok, "classical_rej_when_bp_accepts"], "o", mfc="white", mec=INK, mew=1.4, ms=8,
                ls="none", label="Classical, in the samples where the pretest passed")
        for i in xs:
            ax.annotate(f"{s.bp_reject[i]:.0%}", (i, 0.0), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=INK2)
        ax.axhline(0.05, color=INK2, lw=0.8, ls=":")
        ax.set_xticks(xs, [str(n) for n in NS])
        ax.set_xlabel("Sample size n")
        ax.set_title(dname[d] + ", variance proportional to x", fontsize=10.5)
        ax.yaxis.set_major_formatter(pct)
    axes[0].set_ylabel("Rejection rate of a true null (nominal 5%)")
    axes[0].set_ylim(0, 0.6)
    axes[1].legend(fontsize=8.5, loc="upper right")
    axes[1].text(0.98, 0.50, "Grey numbers above the axis: share of samples\nin which the Breusch-Pagan test rejects",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
    fig.savefig(FIG / "fig3_pretest.png"); plt.close(fig)


if __name__ == "__main__":
    if ARGS.figures_only:
        digest(pd.read_csv(RES / "03_inference_summary.csv"), ARGS.reps)
        figures(pd.read_csv(RES / "03_inference_summary.csv"))
        raise SystemExit
    tasks = [(d, g, n, r) for d in DESIGNS for g in GAMMAS for n in NS for r in range(ARGS.reps)]
    with Pool(2) as pool:
        res = pool.map(one, tasks, chunksize=50)
    df = pd.DataFrame(res, columns=["design", "gamma", "n", "rep", "p_bp", "p_white", "se_ratio_hc3",
                                    "se_ratio_hc4", "se_ratio_hc5", *METHODS,
                                    *[f"power {m}" for m in POWER_METHODS]])
    df.to_csv(RES / "03_inference_replications.csv.gz", index=False)
    write("03_simulation_inference.txt", summarise(df))
    digest(pd.read_csv(RES / "03_inference_summary.csv"), ARGS.reps)
    figures(pd.read_csv(RES / "03_inference_summary.csv"))
