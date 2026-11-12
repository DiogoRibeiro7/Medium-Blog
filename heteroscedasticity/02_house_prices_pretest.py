"""Case 2 - a pretest that passes: house prices from Wooldridge's textbook data.

wooldridge::hprice1: 88 house sales (Boston area, 1990), price on lot size,
square footage and bedrooms. In levels the Breusch-Pagan test rejects; in logs
it does not, and the usual routine (test, then robust only if the test
rejects) would keep the classical standard errors. The robust standard errors
tell a different story.

Run:  python 02_house_prices_pretest.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from common import AQUA, BLUE, FIG, INK2, ORANGE, fetch, style, write
from het import HC_TYPES, bp_test, ols, white_test

warnings.filterwarnings("ignore")
style()
h = fetch("wooldridge", "hprice1")
out = [f"hprice1: n = {len(h)}"]
store = {}
for label, ycol, xcols in (("levels", "price", ["lotsize", "sqrft", "bdrms"]),
                           ("logs", "lprice", ["llotsize", "lsqrft", "bdrms"])):
    X = np.column_stack([np.ones(len(h))] + [h[c].values for c in xcols])
    y = h[ycol].values
    f = ols(X, y)
    lm, p = bp_test(f.resid, X[:, 1:])
    lw, pw = white_test(f.resid, X[:, 1:])
    out.append(f"-- {label}: {ycol} ~ {' + '.join(xcols)}; Koenker BP LM {lm:.2f} (p {p:.4f}); White LM {lw:.2f} "
               f"(p {pw:.4f}); largest leverage {f.hat.max():.3f} (mean {X.shape[1] / len(h):.3f})")
    se_c = f.se("const")
    store[label] = (xcols, {t: f.se(t)[1:] / se_c[1:] for t in HC_TYPES[1:]}, p, pw, f.hat.max())
    for j, c in enumerate(xcols, start=1):
        out.append(f"  {c:9s} coef {f.coef[j]:.4f}; SE classical {se_c[j]:.4f}; " + "; ".join(
            f"{t} {f.se(t)[j]:.4f} (x{f.se(t)[j] / se_c[j]:.2f})" for t in HC_TYPES[1:]))
write("02_house_prices_pretest.txt", "\n".join(out))

# --------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), sharey=True, gridspec_kw={"wspace": 0.08})
cols = (BLUE, ORANGE, AQUA)
mks = ("o", "s", "D")
types = list(HC_TYPES[1:])
for ax, label in zip(axes, ("levels", "logs")):
    xcols, ratios, p, pw, hmax = store[label]
    for j, (c, col, mk) in enumerate(zip(xcols, cols, mks)):
        ax.plot([ratios[t][j] for t in types], np.arange(len(types)) + (j - 1) * 0.18, mk, color=col, ms=6,
                ls="none", label=c)
    ax.axvline(1, color=INK2, lw=1)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 100, 1000] if label == "levels" else [1, 1.25, 1.5, 2]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"×{v:g}"))
    ax.set_xlim(*((0.8, 1500) if label == "levels" else (0.95, 2.1)))
    ax.set_title(f"Price in {label}: Breusch-Pagan p = {p:.4f}\nWhite p = {pw:.4f}, largest leverage {hmax:.2f}",
                 fontsize=10.5)
    ax.set_xlabel("Robust standard error relative to the classical one (log scale)")
    ax.legend(fontsize=8.5, loc="upper right", title="coefficient of", title_fontsize=8.5)
axes[0].set_yticks(range(len(types)), types)
axes[0].invert_yaxis()
fig.savefig(FIG / "fig2_hprice_multipliers.png"); plt.close(fig)
