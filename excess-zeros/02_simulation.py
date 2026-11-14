"""Simulation with known truth - what excess zeros do to model choice and to decisions.

Five data-generating processes, each with two independent standard normal
regressors x and z and a count component with log mu = 0.5 + 0.5 x (on average
exp(0.625), about 1.87):

  NB                      NB2 with theta = 1; no structural zeros
  NB, dispersion varies   NB2 with log(theta) = z: z changes the share of zeros
                          but not the mean; no structural zeros
  Poisson-lognormal       Poisson with a lognormal multiplier (sd 1, mean 1);
                          overdispersed, not NB-shaped; no structural zeros
  Zero-inflated NB        structural zero with probability 1 / (1 + exp(1.2 - z)),
                          otherwise NB2 with theta = 2
  Hurdle NB               y > 0 with probability 1 / (1 + exp(-(0.4 + 0.5 x + z))),
                          then a zero-truncated NB2 with theta = 1

Each sample (n = 1,000 or 4,000; 500 samples per cell) is fitted with Poisson,
NB, NB with log(theta) linear in (1, x, z), zero-inflated Poisson and NB, and a
hurdle NB, all with (1, x, z) in every equation, as an analyst who does not
know which variable does what would. Recorded per sample:

  model choice   Vuong's test of ZIP against Poisson and of ZINB against NB
                 (one-sided 5%, as printed by pscl::vuong), with and without the
                 AIC and BIC corrections; the model with the lowest AIC and BIC
                 among NB, NB with varying dispersion, ZINB and hurdle NB
  decisions      the average effect of a unit increase in x on E[Y]; the average
                 effect of a unit increase in z on P(Y > 0); the mean absolute
                 error of each person's P(Y = 0); and the ZINB's estimated share
                 of structural zeros, against the truth in that sample

Outputs: results/02_simulation.txt (every cell), 02_simulation_digest.txt,
02_simulation_summary.csv, results/02_simulation_replications.csv.gz (not
committed), figures fig6_model_choice.png, fig5_structural_share.png and
fig7_decision_errors.png.

Run:  python 02_simulation.py            (about 15 minutes on 2 cores)
      python 02_simulation.py --reps 50  (quick run)
      python 02_simulation.py --figures-only
"""

from __future__ import annotations

import argparse
import warnings
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, gammaln

from common import AQUA, BLUE, FIG, INK, INK2, ORANGE, RES, VIOLET, YELLOW, style, write
from counts import hurdle, negbin, poisson, vuong, zeroinfl

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=500)
ap.add_argument("--figures-only", action="store_true")
ARGS = ap.parse_args()

DGPS = ["NB", "NB, dispersion varies", "Poisson-lognormal", "Zero-inflated NB", "Hurdle NB"]
NS = [1000, 4000]
MODELS = ["Poisson", "NB", "NB, dispersion varies", "ZIP", "ZINB", "Hurdle-NB"]
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(60)          # for the Poisson-lognormal P(Y = 0)
GH_W = GH_W / GH_W.sum()
SIG = 1.0


def _nb_logf0(mu, th):
    return th * (np.log(th) - np.log(th + mu))


def truth(dgp, x, z):
    """True E[Y | x, z] and P(Y = 0 | x, z)."""
    mu = np.exp(0.5 + 0.5 * x)
    if dgp == "NB":
        return mu, np.exp(_nb_logf0(mu, 1.0))
    if dgp == "NB, dispersion varies":
        return mu, np.exp(_nb_logf0(mu, np.exp(z)))
    if dgp == "Poisson-lognormal":
        lam = mu[:, None] * np.exp(SIG * GH_X[None, :] - SIG ** 2 / 2)
        return mu, np.exp(-lam) @ GH_W
    if dgp == "Zero-inflated NB":
        pi = expit(-1.2 + z)
        return (1 - pi) * mu, pi + (1 - pi) * np.exp(_nb_logf0(mu, 2.0))
    if dgp == "Hurdle NB":
        p = expit(0.4 + 0.5 * x + z)
        f0 = np.exp(_nb_logf0(mu, 1.0))
        return p * mu / (1 - f0), 1 - p
    raise ValueError(dgp)


def draw(dgp, x, z, rng):
    n = len(x)
    mu = np.exp(0.5 + 0.5 * x)
    if dgp == "NB":
        return rng.negative_binomial(1.0, 1.0 / (1.0 + mu))
    if dgp == "NB, dispersion varies":
        th = np.exp(z)
        return rng.negative_binomial(th, th / (th + mu))
    if dgp == "Poisson-lognormal":
        return rng.poisson(mu * np.exp(SIG * rng.standard_normal(n) - SIG ** 2 / 2))
    if dgp == "Zero-inflated NB":
        y = rng.negative_binomial(2.0, 2.0 / (2.0 + mu))
        return y * (rng.random(n) >= expit(-1.2 + z))
    if dgp == "Hurdle NB":
        pos = rng.random(n) < expit(0.4 + 0.5 * x + z)
        y = np.zeros(n, int)
        m = pos.sum()
        draws = rng.negative_binomial(1.0, 1.0 / (1.0 + mu[pos]))
        bad = draws == 0
        while bad.any():                                     # zero-truncation by rejection
            draws[bad] = rng.negative_binomial(1.0, 1.0 / (1.0 + mu[pos][bad]))
            bad = draws == 0
        y[pos] = draws
        assert m == (y > 0).sum()
        return y
    raise ValueError(dgp)


def one(task):
    dgp, n, rep = task
    rng = np.random.default_rng([7, DGPS.index(dgp), NS.index(n), rep])
    x, z = rng.standard_normal(n), rng.standard_normal(n)
    y = draw(dgp, x, z, rng).astype(float)
    X = np.column_stack([np.ones(n), x, z])
    Xx, Xz = X.copy(), X.copy()
    Xx[:, 1] += 1
    Xz[:, 2] += 1
    fits = {"Poisson": poisson(X, y), "NB": negbin(X, y), "NB, dispersion varies": negbin(X, y, X),
            "ZIP": zeroinfl(X, y, X, "poisson"), "ZINB": zeroinfl(X, y, X), "Hurdle-NB": hurdle(X, y, X)}
    m0, p0 = truth(dgp, x, z)
    ame_x = np.mean(truth(dgp, x + 1, z)[0] - m0)
    eff_z = np.mean(p0 - truth(dgp, x, z + 1)[1])
    row = {"dgp": dgp, "n": n, "rep": rep, "zeros": float(np.mean(y == 0)), "true_ame_x": ame_x, "true_eff_z": eff_z,
           "true_share": float(np.mean(expit(-1.2 + z))) if dgp == "Zero-inflated NB" else 0.0}
    for k, f in fits.items():
        key = {"NB, dispersion varies": "W", "ZIP": "Z", "ZINB": "Z", "Hurdle-NB": "Z"}.get(k)
        kw = (lambda M: {key: M}) if key else (lambda M: {})
        row[f"{k}|ame_x"] = float(np.mean(f.mean(X=Xx, **kw(Xx)) - f.mean(X=X, **kw(X))))
        row[f"{k}|eff_z"] = float(np.mean(f.pmf(0, X=X, **kw(X)) - f.pmf(0, X=Xz, **kw(Xz))))
        row[f"{k}|p0_mae"] = float(np.mean(np.abs(f.pmf(0) - p0)))
        row[f"{k}|aic"] = f.aic
        row[f"{k}|bic"] = f.bic(n)
        row[f"{k}|conv"] = f.converged
    infl = fits["ZINB"].extra["inflation"]()
    row["ZINB|share"] = float(np.mean(infl))
    row["ZINB|maxpi"] = float(np.max(infl))
    for lab, (a, b) in {"zip_pois": ("ZIP", "Poisson"), "zinb_nb": ("ZINB", "NB")}.items():
        for c in (None, "aic", "bic"):
            v, p = vuong(fits[a], fits[b], c)
            row[f"vuong_{lab}_{c or 'raw'}"] = v
    cand = ["NB", "NB, dispersion varies", "ZINB", "Hurdle-NB"]
    row["aic_choice"] = min(cand, key=lambda k: fits[k].aic)
    row["bic_choice"] = min(cand, key=lambda k: fits[k].bic(n))
    return row


def summarise(R):
    rows = []
    for (dgp, n), u in R.groupby(["dgp", "n"], sort=False):
        r = {"dgp": dgp, "n": n, "reps": len(u), "zeros": u.zeros.mean(), "true_ame_x": u.true_ame_x.mean(),
             "true_eff_z": u.true_eff_z.mean(), "true_share": u.true_share.mean()}
        for k in MODELS:
            e1 = u[f"{k}|ame_x"] - u.true_ame_x
            e2 = u[f"{k}|eff_z"] - u.true_eff_z
            r[f"{k}|ame_x_bias"], r[f"{k}|ame_x_rmse"] = e1.mean(), np.sqrt((e1 ** 2).mean())
            r[f"{k}|eff_z_bias"], r[f"{k}|eff_z_rmse"] = e2.mean(), np.sqrt((e2 ** 2).mean())
            r[f"{k}|p0_mae"] = u[f"{k}|p0_mae"].mean()
            r[f"{k}|conv"] = u[f"{k}|conv"].mean()
        r["share_median"] = u["ZINB|share"].median()
        r["share_q05"], r["share_q95"] = u["ZINB|share"].quantile(0.05), u["ZINB|share"].quantile(0.95)
        r["share_bias"] = (u["ZINB|share"] - u.true_share).mean()
        sep = u["ZINB|maxpi"] > 0.99
        r["separated"] = sep.mean()
        r["vuong_raw_separated"] = (sep & (u.vuong_zinb_nb_raw > stats.norm.ppf(0.95))).mean()
        r["zinb_unexplained_nonconv"] = (~u["ZINB|conv"].astype(bool) & ~sep & (u["ZINB|share"] >= 1e-4)).mean()
        r["other_nonconv"] = max(1 - u[f"{k}|conv"].astype(bool).mean() for k in MODELS if k != "ZINB")
        for c in ("raw", "aic", "bic"):
            r[f"zip_pois_{c}"] = (u[f"vuong_zip_pois_{c}"] > stats.norm.ppf(0.95)).mean()
            r[f"zinb_nb_{c}"] = (u[f"vuong_zinb_nb_{c}"] > stats.norm.ppf(0.95)).mean()
        for crit in ("aic", "bic"):
            for k in ("NB", "NB, dispersion varies", "ZINB", "Hurdle-NB"):
                r[f"{crit}_picks|{k}"] = (u[f"{crit}_choice"] == k).mean()
        rows.append(r)
    return pd.DataFrame(rows)


def report(S):
    out = []
    for r in S.to_dict("records"):
        out.append(f"-- {r['dgp']}, n = {r['n']} ({r['reps']} samples): share of zeros {r['zeros']:.3f}; true average "
                   f"effect of x on E[Y] {r['true_ame_x']:.4f}; true effect of z on P(Y > 0) {r['true_eff_z']:+.4f}; true "
                   f"share of structural zeros {r['true_share']:.3f}")
        out.append(f"   Vuong ZIP > Poisson at 5%: raw {r['zip_pois_raw']:.3f}, AIC {r['zip_pois_aic']:.3f}, BIC "
                   f"{r['zip_pois_bic']:.3f}; Vuong ZINB > NB at 5%: raw {r['zinb_nb_raw']:.3f}, AIC {r['zinb_nb_aic']:.3f}, "
                   f"BIC {r['zinb_nb_bic']:.3f}")
        out.append("   lowest AIC: " + "; ".join(f"{k} {r[f'aic_picks|{k}']:.3f}" for k in
                                                  ("NB", "NB, dispersion varies", "ZINB", "Hurdle-NB")))
        out.append("   lowest BIC: " + "; ".join(f"{k} {r[f'bic_picks|{k}']:.3f}" for k in
                                                  ("NB", "NB, dispersion varies", "ZINB", "Hurdle-NB")))
        out.append(f"   ZINB share of structural zeros: median {r['share_median']:.4f}, 5%-95% {r['share_q05']:.4f}-"
                   f"{r['share_q95']:.4f}; mean error {r['share_bias']:+.4f}; samples in which some inflation "
                   f"probability exceeds 0.99 (zeros separated by the inflation equation) {r['separated']:.3f}, and in "
                   f"which those also make the raw Vuong test favour the ZINB {r['vuong_raw_separated']:.3f}")
        for k in MODELS:
            out.append(f"   {k:22s} effect of x on E[Y]: bias {r[f'{k}|ame_x_bias']:+.4f}, RMSE {r[f'{k}|ame_x_rmse']:.4f}; "
                       f"effect of z on P(Y > 0): bias {r[f'{k}|eff_z_bias']:+.4f}, RMSE {r[f'{k}|eff_z_rmse']:.4f}; "
                       f"mean |error| of P(Y = 0) {r[f'{k}|p0_mae']:.4f}" +
                       ("" if k == "ZINB" else f"; converged {r[f'{k}|conv']:.3f}"))
    return "\n".join(out)


def digest(S):
    out = ["Digest", f"Count component: log mean 0.5 + 0.5 x with x standard normal; its average mean is exp(0.625) = "
                     f"{np.exp(0.625):.4f}"]
    no_inflation = S[S.dgp.isin(["NB", "NB, dispersion varies", "Poisson-lognormal", "Hurdle NB"])]
    out.append(f"Vuong ZIP > Poisson (raw) at 5%, all designs: {S.zip_pois_raw.min():.3f}-{S.zip_pois_raw.max():.3f}")
    for d in DGPS:
        u = S[S.dgp == d]
        out.append(f"{d}: zeros {u.zeros.min():.3f}-{u.zeros.max():.3f}; Vuong ZINB > NB raw {u.zinb_nb_raw.min():.3f}-"
                   f"{u.zinb_nb_raw.max():.3f}, AIC-corrected {u.zinb_nb_aic.min():.3f}-{u.zinb_nb_aic.max():.3f}, "
                   f"BIC-corrected {u.zinb_nb_bic.min():.3f}-{u.zinb_nb_bic.max():.3f}; AIC picks ZINB or hurdle "
                   f"{(u['aic_picks|ZINB'] + u['aic_picks|Hurdle-NB']).min():.3f}-"
                   f"{(u['aic_picks|ZINB'] + u['aic_picks|Hurdle-NB']).max():.3f}; BIC picks ZINB or hurdle "
                   f"{(u['bic_picks|ZINB'] + u['bic_picks|Hurdle-NB']).min():.3f}-"
                   f"{(u['bic_picks|ZINB'] + u['bic_picks|Hurdle-NB']).max():.3f}; ZINB median share of structural "
                   f"zeros {u.share_median.min():.4f}-{u.share_median.max():.4f} (truth {u.true_share.mean():.3f})")
    for d in DGPS:
        for n in NS:
            r = S[(S.dgp == d) & (S.n == n)].iloc[0]
            best_x = min(MODELS, key=lambda k: r[f"{k}|ame_x_rmse"])
            best_z = min(MODELS, key=lambda k: r[f"{k}|eff_z_rmse"])
            best_p = min(MODELS, key=lambda k: r[f"{k}|p0_mae"])
            out.append(f"{d}, n = {n}: smallest RMSE for the effect of x on E[Y] {best_x} ({r[f'{best_x}|ame_x_rmse']:.4f}; "
                       f"NB {r['NB|ame_x_rmse']:.4f}, Poisson {r['Poisson|ame_x_rmse']:.4f}, ZINB "
                       f"{r['ZINB|ame_x_rmse']:.4f}); for the effect of z on P(Y > 0) {best_z} "
                       f"({r[f'{best_z}|eff_z_rmse']:.4f}; NB {r['NB|eff_z_rmse']:.4f}, ZINB {r['ZINB|eff_z_rmse']:.4f}, "
                       f"hurdle {r['Hurdle-NB|eff_z_rmse']:.4f}); for P(Y = 0) {best_p} ({r[f'{best_p}|p0_mae']:.4f}; NB "
                       f"{r['NB|p0_mae']:.4f}, ZINB {r['ZINB|p0_mae']:.4f})")
    out.append(f"Fits that did not converge: models other than the ZINB, largest share in any cell "
               f"{S.other_nonconv.max():.3f}; ZINB fits that did not converge without either separating some zeros "
               f"(an inflation probability above 0.99) or reaching the boundary (a share of structural zeros below "
               f"0.0001), largest share in any cell {S.zinb_unexplained_nonconv.max():.3f}")
    for d in ("NB", "NB, dispersion varies", "Poisson-lognormal", "Hurdle NB"):
        u = S[S.dgp == d]
        out.append(f"{d}: samples with zeros separated by the ZINB's inflation equation {u.separated.min():.3f}-"
                   f"{u.separated.max():.3f}; raw Vuong favours ZINB {u.zinb_nb_raw.min():.3f}-{u.zinb_nb_raw.max():.3f}, "
                   f"of which with separated zeros {u.vuong_raw_separated.min():.3f}-{u.vuong_raw_separated.max():.3f}")
    out.append(f"Designs without structural zeros: ZINB median share of structural zeros "
               f"{no_inflation.share_median.min():.4f}-{no_inflation.share_median.max():.4f}")
    return "\n".join(out)


COL = {"Poisson": INK2, "NB": BLUE, "NB, dispersion varies": AQUA, "ZIP": ORANGE, "ZINB": YELLOW, "Hurdle-NB": VIOLET}
SHORT = {"NB": "NB", "NB, dispersion varies": "NB,\ndispersion\nvaries", "Poisson-lognormal": "Poisson-\nlognormal",
         "Zero-inflated NB": "Zero-\ninflated NB", "Hurdle NB": "Hurdle NB"}


def figures(S, R):
    style()
    # figure 6: model choice
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3), gridspec_kw={"wspace": 0.25})
    xs = np.arange(len(DGPS))
    ax = axes[0]
    w = 0.2
    for i, (c, lab, col) in enumerate((("zip_pois_raw", "ZIP vs Poisson", ORANGE), ("zinb_nb_raw", "ZINB vs NB", YELLOW),
                                       ("zinb_nb_aic", "ZINB vs NB, AIC-corrected", VIOLET),
                                       ("zinb_nb_bic", "ZINB vs NB, BIC-corrected", BLUE))):
        v = [S[(S.dgp == d) & (S.n == 4000)][c].iloc[0] for d in DGPS]
        ax.bar(xs + (i - 1.5) * w, v, width=w, color=col, label=lab)
    ax.axhline(0.05, color=INK, lw=0.8, ls=":")
    ax.set_xticks(xs, [SHORT[d] for d in DGPS], fontsize=9)
    ax.set_ylabel("Share of samples in which the test\nfavours the zero-inflated model")
    ax.set_ylim(0, 1.05)
    ax.set_title("Vuong's test at 5%, n = 4,000")
    ax.legend(fontsize=8.2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax = axes[1]
    bottom = np.zeros(len(DGPS))
    for k, col in (("NB", BLUE), ("NB, dispersion varies", AQUA), ("ZINB", YELLOW), ("Hurdle-NB", VIOLET)):
        v = np.array([S[(S.dgp == d) & (S.n == 4000)][f"aic_picks|{k}"].iloc[0] for d in DGPS])
        ax.bar(xs, v, bottom=bottom, color=col, width=0.6, label=k)
        bottom += v
    ax.set_xticks(xs, [SHORT[d] for d in DGPS], fontsize=9)
    ax.set_ylabel("Share of samples")
    ax.set_ylim(0, 1)
    ax.set_title("Model with the lowest AIC, n = 4,000")
    ax.legend(fontsize=8.2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.savefig(FIG / "fig6_model_choice.png")
    plt.close(fig)

    # figure 5: the ZINB's share of structural zeros
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    data, pos, cols = [], [], []
    for i, d in enumerate(DGPS):
        for j, (n, col) in enumerate(((1000, AQUA), (4000, BLUE))):
            data.append(R[(R.dgp == d) & (R.n == n)]["ZINB|share"].values)
            pos.append(i + (j - 0.5) * 0.32)
            cols.append(col)
    vp = ax.violinplot(data, positions=pos, widths=0.28, showextrema=False, showmedians=True)
    for b, c in zip(vp["bodies"], cols):
        b.set_facecolor(c)
        b.set_alpha(0.55)
        b.set_edgecolor("none")
    vp["cmedians"].set_color(INK)
    for i, d in enumerate(DGPS):
        t = S[S.dgp == d].true_share.mean()
        ax.plot([i - 0.45, i + 0.45], [t, t], color=ORANGE, lw=2)
    ax.set_xticks(range(len(DGPS)), [SHORT[d] for d in DGPS], fontsize=9)
    ax.set_ylabel("Estimated share of structural zeros\n(mean of the ZINB's inflation probability)")
    ax.set_title("What a zero-inflated NB reports as structural zeros")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(color=AQUA, alpha=0.55, label="n = 1,000"), Patch(color=BLUE, alpha=0.55, label="n = 4,000"),
                       Line2D([0], [0], color=ORANGE, lw=2, label="True share")], fontsize=8.5, loc="upper left")
    fig.savefig(FIG / "fig5_structural_share.png")
    plt.close(fig)

    # figure 7: decision quantities
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6), sharey=True, gridspec_kw={"wspace": 0.08})
    specs = (("ame_x_rmse", "RMSE of the effect of x on E[Y]"), ("eff_z_rmse", "RMSE of the effect of z on P(Y > 0)"),
             ("p0_mae", "Mean |error| of each person's P(Y = 0)"))
    rows = {d: i for i, d in enumerate(DGPS)}
    for ax, (key, title) in zip(axes, specs):
        for j, k in enumerate(MODELS):
            v = [S[(S.dgp == d) & (S.n == 4000)][f"{k}|{key}"].iloc[0] for d in DGPS]
            ax.plot(v, np.arange(len(DGPS)) + (j - 2.5) * 0.12, "o", color=COL[k], ms=6, label=k)
        ax.set_xscale("log")
        allv = [S[(S.dgp == d) & (S.n == 4000)][f"{k}|{key}"].iloc[0] for d in DGPS for k in MODELS]
        lo_, hi_ = min(allv) * 0.9, max(allv) * 1.1
        ticks = [c for c in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2) if lo_ <= c <= hi_]
        if len(ticks) < 3:
            ticks = [c for c in (0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2) if lo_ <= c <= hi_]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("Error, n = 4,000 (log scale)")
    axes[0].set_yticks(range(len(DGPS)), [d.replace("Zero-inflated", "Zero-\ninflated").replace(", dispersion", ",\ndispersion")
                                          for d in DGPS], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("True process")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=6)
    fig.savefig(FIG / "fig7_decision_errors.png")
    plt.close(fig)


if __name__ == "__main__":
    if ARGS.figures_only:
        S = pd.read_csv(RES / "02_simulation_summary.csv")
        R = pd.read_csv(RES / "02_simulation_replications.csv.gz")
    else:
        tasks = [(d, n, r) for d in DGPS for n in NS for r in range(ARGS.reps)]
        with Pool(2) as pool:
            rows = pool.map(one, tasks, chunksize=10)
        R = pd.DataFrame(rows)
        R.to_csv(RES / "02_simulation_replications.csv.gz", index=False)
        S = summarise(R)
        S.to_csv(RES / "02_simulation_summary.csv", index=False, float_format="%.6g")
    write("02_simulation.txt", report(S))
    write("02_simulation_digest.txt", digest(S))
    figures(S, R)
