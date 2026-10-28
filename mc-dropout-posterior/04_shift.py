"""Uncertainty where the training data stop: a gap and a tail in temperature.

UCI combined-cycle power plant data (9,568 hourly averages; inputs ambient
temperature AT, exhaust vacuum V, ambient pressure AP, relative humidity RH;
target net electrical output PE in MW). The rows are cut by ambient
temperature quantiles of the whole data set:

- gap: 45th to 60th percentile of AT, removed from training;
- tail: above the 85th percentile, removed from training;
- the rest is the pool; 15% of it, at random, is the in-distribution test set
  and the other 85% the training set.

Methods, each fitted to the same training rows:
- MC dropout, one hidden layer of 50 units, rate and tau tuned as in the
  reference protocol (grid p in {0.005, 0.01, 0.05, 0.1}, tau in the power
  plant grid {0.05, 0.1, 0.15}, 80/20 inner split, best MC log-likelihood);
- the same with two hidden layers of 50 units;
- a deep ensemble of 5 networks (one hidden layer, no dropout); noise variance
  from the inner validation rows (mean squared error of the ensemble mean
  minus the mean ensemble variance);
- an exact Gaussian process (ARD squared exponential + noise, hyperparameters
  by marginal likelihood on 1,500 training rows).
Networks train for 400 epochs (10 times the listed 40). Three seeds for the
networks (different inner splits are not used: the inner split is the
reference one, the last 20% of the training rows; seeds change
initialisation, minibatches and masks).

Reported per region: RMSE, mean epistemic sd, mean predictive sd, coverage
and width of the central 90% predictive interval (Gaussian with the
predictive mean and variance), and the AUROC with which the epistemic sd
separates gap or tail rows from in-distribution rows.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcd import (AQUA, BLUE, FIGURES, GRID, INK, INK2, ORANGE, QUICK, RESULTS, Nets, Report,
                 auroc, fit_gp_hypers, gal_weight_decay, gp_posterior, load_uci,
                 mc_loglik, steps_for, style_axes)

SEEDS = [0, 1, 2]
GRID_P = [0.005, 0.01, 0.05, 0.1]
TAU_GRID = [0.05, 0.1, 0.15]
EPOCHS = 400
T = 500
Z90 = 1.6448536269514722
if QUICK:
    SEEDS, EPOCHS, T = [0], 4, 50


def tuned_mc(X, y, tr, hidden, seed, tests):
    n_in = int(0.8 * len(tr))
    inner, val = tr[:n_in], tr[n_in:]
    grid = [(p, t) for p in GRID_P for t in TAU_GRID]
    drop = np.array([g[0] for g in grid])
    tau = np.array([g[1] for g in grid])
    nets = Nets(X.shape[1], hidden, drop, gal_weight_decay(drop, n_in, tau), seed=seed)
    nets.fit(X, y, [inner] * len(grid), steps=steps_for(n_in, EPOCHS))
    vs = nets.predict(X[val], T=T)
    ll = [mc_loglik(y[val], vs[i], tau[i]).mean() for i in range(len(grid))]
    b = int(np.argmax(ll))
    final = Nets(X.shape[1], hidden, [drop[b]], [gal_weight_decay(drop[b], len(tr), tau[b])],
                 seed=seed + 7)
    final.fit(X, y, [tr], steps=steps_for(len(tr), EPOCHS))
    out = {}
    for name, idx in tests.items():
        s = final.predict(X[idx], T=T)[0]
        out[name] = (s.mean(0), s.std(0), 1.0 / tau[b])
    return out, (drop[b], tau[b])


def ensemble(X, y, tr, seed, tests, m=5):
    n_in = int(0.8 * len(tr))
    inner, val = tr[:n_in], tr[n_in:]
    a = Nets(X.shape[1], [50], np.zeros(m), np.zeros(m), seed=seed + 11)
    a.fit(X, y, [inner] * m, steps=steps_for(n_in, EPOCHS))
    pv = a.predict(X[val], mc=False)[:, 0, :]
    noise = max(np.mean((y[val] - pv.mean(0)) ** 2) - np.mean(pv.var(0, ddof=1)), 1e-6)
    f = Nets(X.shape[1], [50], np.zeros(m), np.zeros(m), seed=seed + 13)
    f.fit(X, y, [tr] * m, steps=steps_for(len(tr), EPOCHS))
    out = {}
    for name, idx in tests.items():
        p = f.predict(X[idx], mc=False)[:, 0, :]
        out[name] = (p.mean(0), p.std(0, ddof=1), noise)
    return out


def main():
    D = load_uci("power-plant")
    X, y = D["X"], D["y"]
    at = X[:, 0]
    q45, q60, q85 = np.quantile(at, [0.45, 0.60, 0.85])
    gap = np.flatnonzero((at > q45) & (at <= q60))
    tail = np.flatnonzero(at > q85)
    pool = np.flatnonzero((at <= q45) | ((at > q60) & (at <= q85)))
    rng = np.random.default_rng(2026)
    pool = rng.permutation(pool)
    n_test = int(round(0.15 * len(pool)))
    test_in, tr = pool[:n_test], pool[n_test:]
    tests = {"in": test_in, "gap": gap, "tail": tail}

    rep = Report("04_shift")
    rep("UCI power plant: training rows exclude a gap and a tail in ambient temperature")
    rep(f"AT quantiles: 45% {q45:.2f} C, 60% {q60:.2f} C, 85% {q85:.2f} C")
    rep(f"training rows {len(tr)}; test rows: in-distribution {len(test_in)}, gap {len(gap)}, tail {len(tail)}")
    rep(f"PE sd in training {y[tr].std():.2f} MW; mean PE in training {y[tr].mean():.2f}, gap {y[gap].mean():.2f}, tail {y[tail].mean():.2f}")
    rep("")

    records, curves = [], {}
    for seed in SEEDS:
        for label, hidden in (("MC dropout, 1 hidden layer", [50]), ("MC dropout, 2 hidden layers", [50, 50])):
            out, (p, t) = tuned_mc(X, y, tr, hidden, 1000 * seed + len(hidden), tests)
            records.append((label, seed, out, f"p={p:g}, tau={t:g}"))
        records.append(("Deep ensemble (5)", seed, ensemble(X, y, tr, 1000 * seed, tests), ""))
        print(f"seed {seed} done", flush=True)
    hyp = fit_gp_hypers(X[tr], y[tr], n_sub=1500, seed=0)
    gout = {}
    for name, idx in tests.items():
        m, s, noise = gp_posterior(X[tr], y[tr], X[idx], hyp)
        gout[name] = (m, s, noise)
    records.append(("Gaussian process", 0, gout, f"noise sd {np.sqrt(gout['in'][2]):.3f}"))

    rows = []
    for label, seed, out, info in records:
        for region, (m, s, noise) in out.items():
            idx = tests[region]
            v = s ** 2 + noise
            err = y[idx] - m
            rows.append({"method": label, "seed": seed, "region": region, "info": info,
                         "rmse": float(np.sqrt(np.mean(err ** 2))),
                         "epi_sd": float(np.mean(s)),
                         "pred_sd": float(np.mean(np.sqrt(v))),
                         "noise_sd": float(np.sqrt(noise)),
                         "cover90": float(np.mean(np.abs(err) <= Z90 * np.sqrt(v))),
                         "width90": float(np.mean(2 * Z90 * np.sqrt(v)))})
        rows[-1]["auroc_tail"] = float(auroc(out["tail"][1], out["in"][1]))
        rows[-2]["auroc_gap"] = float(auroc(out["gap"][1], out["in"][1]))
        if seed == 0:
            curves[label] = out
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "04_shift.csv", index=False, float_format="%.6g")

    methods = ["MC dropout, 1 hidden layer", "MC dropout, 2 hidden layers", "Deep ensemble (5)", "Gaussian process"]
    for mth in methods:
        g = df[df.method == mth]
        rep(f"{mth}  ({'; '.join(sorted(set(g['info'])))})")
        for region in ("in", "gap", "tail"):
            gg = g[g.region == region]
            rep(f"  {region:4s}: RMSE {gg.rmse.mean():.3f}  epistemic sd {gg.epi_sd.mean():.3f}  "
                f"predictive sd {gg.pred_sd.mean():.3f}  90% coverage {gg.cover90.mean():.3f} "
                f"[{gg.cover90.min():.3f}, {gg.cover90.max():.3f}]  width {gg.width90.mean():.2f}")
        rat_gap = g[g.region == "gap"].epi_sd.values / g[g.region == "in"].epi_sd.values
        rat_tail = g[g.region == "tail"].epi_sd.values / g[g.region == "in"].epi_sd.values
        rep(f"  epistemic sd ratio gap/in {rat_gap.mean():.2f}, tail/in {rat_tail.mean():.2f}")
        rep(f"  AUROC of epistemic sd, gap vs in {g.auroc_gap.dropna().mean():.3f}, "
            f"tail vs in {g.auroc_tail.dropna().mean():.3f}")
    rep.save()

    # Figure: epistemic sd against temperature, seed 0
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), facecolor="white", sharex=True)
    regcol = {"in": BLUE, "gap": ORANGE, "tail": AQUA}
    for ax, mth in zip(axes.ravel(), methods):
        style_axes(ax)
        out = curves[mth]
        ax.axvspan(q45, q60, color=GRID, alpha=0.6, lw=0)
        ax.axvspan(q85, at.max() + 0.5, color=GRID, alpha=0.6, lw=0)
        for region in ("in", "gap", "tail"):
            idx = tests[region]
            ax.scatter(at[idx], out[region][1], s=5, color=regcol[region], alpha=0.5, lw=0)
        ax.set_title(mth, loc="left", fontsize=10.5, color=INK)
        ax.set_ylabel("epistemic sd, MW", color=INK2)
        ax.set_ylim(bottom=0)
    for ax in axes[1]:
        ax.set_xlabel("ambient temperature AT, C", color=INK2)
    ax0 = axes[0, 0]
    yl = ax0.get_ylim()[1]
    ax0.text((q45 + q60) / 2, yl * 0.95, "gap", ha="center", va="top", fontsize=9, color=INK)
    ax0.text((q85 + at.max()) / 2, yl * 0.95, "tail", ha="center", va="top", fontsize=9, color=INK)
    fig.text(0.01, 0.01, "Test rows of the UCI power-plant data; shaded bands were removed from training. "
             "Blue: in-distribution test rows; orange: gap; green: tail. One seed per network method.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig5_shift.png", dpi=170)
    print("figure written")


if __name__ == "__main__":
    main()
