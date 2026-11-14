"""Mechanism - why overdispersion produces zeros.

A count whose Poisson rate varies between people (unobserved heterogeneity)
has more zeros than a Poisson count with the same mean, because exp(-lambda)
is convex: E[exp(-lambda)] >= exp(-E[lambda]) (Jensen's inequality). With a
gamma-distributed rate the count is negative binomial, and
P(Y = 0) = (theta / (theta + mu))^theta.

The script tabulates P(Y = 0) against the mean for the Poisson and for NB2 with
theta = 0.5, 1, 2 and 5, and for a Poisson-lognormal count with the same mean
(lognormal multiplier with sd 1), and draws figure 1: the curves, and for the
three data sets of 01_three_datasets.py the observed share of zeros against the
shares expected under the Poisson, NB, zero-inflated NB and hurdle NB.

Run:  python 03_mechanism.py   (after 01_three_datasets.py)
"""

import re

import matplotlib.pyplot as plt
import numpy as np

from common import AQUA, BLUE, FIG, INK, INK2, ORANGE, RES, VIOLET, YELLOW, style, write

style()
gx, gw = np.polynomial.hermite_e.hermegauss(80)
gw = gw / gw.sum()


def p0_nb(mu, th):
    return (th / (th + mu)) ** th


def p0_pln(mu, s=1.0):
    return np.exp(-np.outer(mu, np.exp(s * gx - s ** 2 / 2))) @ gw


out = ["P(Y = 0) by mean: Poisson; NB2 with theta 0.5, 1, 2, 5; Poisson-lognormal (sd 1)"]
for mu in (0.5, 1.0, 2.0, 5.0):
    out.append(f"   mean {mu:.1f}: Poisson {np.exp(-mu):.4f}; NB theta 0.5 {p0_nb(mu, 0.5):.4f}, 1 {p0_nb(mu, 1):.4f}, "
               f"2 {p0_nb(mu, 2):.4f}, 5 {p0_nb(mu, 5):.4f}; Poisson-lognormal {p0_pln(np.array([mu]))[0]:.4f}")

# observed and expected zeros from 01_three_datasets.txt
txt = (RES / "01_three_datasets.txt").read_text()
blocks = re.split(r"^== ", txt, flags=re.M)[1:]
DATA = {}
for b in blocks:
    name = b.split(":")[0]
    n = int(re.search(r"n (\d+);", b).group(1))
    obs = int(re.search(r"zeros (\d+) ", b).group(1))
    exp0 = {m: float(re.search(rf"^   {re.escape(m)}\s+parameters.*?expected zeros\s+([\d.]+)", b, flags=re.M).group(1))
            for m in ("Poisson", "NB", "ZINB", "Hurdle-NB")}
    DATA[name] = (n, obs, exp0)
    out.append(f"{name}: share of zeros observed {obs / n:.4f}; expected under Poisson {exp0['Poisson'] / n:.4f}, NB "
               f"{exp0['NB'] / n:.4f}, ZINB {exp0['ZINB'] / n:.4f}, hurdle NB {exp0['Hurdle-NB'] / n:.4f}")
write("03_mechanism.txt", "\n".join(out))

fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), gridspec_kw={"wspace": 0.28, "width_ratios": [1.15, 1]})
ax = axes[0]
mu = np.linspace(0.01, 6, 300)
ax.plot(mu, np.exp(-mu), color=ORANGE, lw=2.4, label="Poisson")
for th, ls in ((5, (0, (1, 1.6))), (2, "-."), (1, "--"), (0.5, "-")):
    ax.plot(mu, p0_nb(mu, th), color=BLUE, lw=1.8, ls=ls, label=f"NB, θ = {th:g}")
ax.plot(mu, p0_pln(mu), color=AQUA, lw=1.8, label="Poisson-lognormal (sd 1)")
ax.set_xlabel("Mean count")
ax.set_ylabel("P(Y = 0)")
ax.set_ylim(0, 1)
ax.set_xlim(0, 6)
ax.set_title("At the same mean, overdispersion means more zeros")
ax.legend(fontsize=8.5, loc="upper right")
ax = axes[1]
models = [("Poisson", ORANGE, "o"), ("NB", BLUE, "o"), ("ZINB", YELLOW, "D")]
names = list(DATA)
for i, name in enumerate(names):
    n, obs, exp0 = DATA[name]
    ax.plot([obs / n] * 2, [i - 0.3, i + 0.3], color=INK, lw=2.2, label="Observed (the hurdle NB matches it\nby construction)"
            if i == 0 else None, zorder=3)
    for j, (m, col, mk) in enumerate(models):
        ax.scatter(exp0[m] / n, i + (j - 1) * 0.17, marker=mk, s=40, color=col, zorder=4, label=m if i == 0 else None)
ax.set_yticks(range(len(names)), [nm.replace(", ", "\n") for nm in names], fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 0.9)
ax.set_xlabel("Share of zeros")
ax.set_title("Observed and expected zeros")
ax.legend(fontsize=8.5, loc="lower left")
fig.savefig(FIG / "fig1_overdispersion_zeros.png")
plt.close(fig)
