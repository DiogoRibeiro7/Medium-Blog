"""Case 1 - one high-leverage state: US public-school spending and income.

sandwich::PublicSchools (Greene's textbook data): per-capita spending on
public schools and per-capita income for the 50 states and Washington DC in
1979 (Wisconsin has no spending figure). The textbook model is quadratic in
income, and Alaska, rich and a big spender, has a leverage of 0.65. Cribari-
Neto (2004) used it to show how far the HC estimators disagree.

Run:  python 01_public_schools.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

from common import BLUE, FIG, INK2, ORANGE, AQUA, VIOLET, fetch, style, write
from het import HC_TYPES, bp_test, ols, t_pvalue, white_test, wild_test

warnings.filterwarnings("ignore")
style()
ps = fetch("sandwich", "PublicSchools").rename(columns={"rownames": "state"})
missing = ps[ps.Expenditure.isna()].state.tolist()
ps = ps.dropna().reset_index(drop=True)
out = [f"PublicSchools: {len(ps) + len(missing)} rows, dropped for missing spending: {', '.join(missing)}; n = {len(ps)}"]


def analyse(d, label):
    inc = d.Income.values * 1e-4
    X = np.column_stack([np.ones(len(d)), inc, inc ** 2])
    y = d.Expenditure.values
    f = ols(X, y)
    lines = [f"-- {label}: n = {len(d)}; coefficients {np.round(f.coef, 2)}; residual SD "
             f"{np.sqrt(np.sum(f.resid ** 2) / f.df):.2f}"]
    top = np.argsort(-f.hat)[:3]
    lines.append("  largest leverages: " + "; ".join(f"{d.state[i]} {f.hat[i]:.3f}" for i in top)
                 + f" (mean leverage p/n = {3 / len(d):.3f})")
    lm, p = bp_test(f.resid, X[:, 1:])
    lw, pw = white_test(f.resid, X[:, 1:])
    lines.append(f"  Koenker Breusch-Pagan LM {lm:.2f} (p {p:.4f}); White LM {lw:.2f} (p {pw:.4f})")
    res = {}
    for t in HC_TYPES:
        se = f.se(t)
        pz = t_pvalue(f.coef[2], se[2])
        res[t] = (se[2], pz)
        lines.append(f"  {t:5s} SE of Income^2 {se[2]:9.2f} (x{se[2] / f.se('const')[2]:.2f} classical); "
                     f"z p-value {pz:.4f}; SE of Income {se[1]:9.2f}")
    for kind in ("rademacher", "mammen", "webb"):
        p_r = wild_test(X, y, 2, 0.0, B=9999, kind=kind, restricted=True, hc="HC1", rng=7)
        res[f"WR {kind}"] = (np.nan, p_r)
        lines.append(f"  wild bootstrap, restricted, {kind:10s} (t with HC1, B = 9999): p {p_r:.4f}")
    p_u = wild_test(X, y, 2, 0.0, B=9999, kind="rademacher", restricted=False, hc="HC1", rng=7)
    res["WU rademacher"] = (np.nan, p_u)
    lines.append(f"  wild bootstrap, unrestricted, rademacher (t with HC1, B = 9999): p {p_u:.4f}")
    return f, res, lines


f_all, res_all, l1 = analyse(ps, "all states")
f_noak, res_noak, l2 = analyse(ps[ps.state != "Alaska"].reset_index(drop=True), "without Alaska")
out += l1 + l2
ak = int(np.flatnonzero(ps.state == "Alaska")[0])
e, h = f_all.resid[ak], f_all.hat[ak]
out.append(f"Alaska: per-capita income {ps.Income[ak]:,.0f} (rank {int((ps.Income > ps.Income[ak]).sum()) + 1} of {len(ps)}), "
           f"spending {ps.Expenditure[ak]:,.0f} (rank {int((ps.Expenditure > ps.Expenditure[ak]).sum()) + 1} of {len(ps)})")
out.append(f"Alaska: residual {e:.2f}, leverage {h:.3f}; its squared residual is {e ** 2:.0f}; HC3 inflates it by "
           f"1/(1-h)^2 = {1 / (1 - h) ** 2:.2f}, HC4 by {1 / (1 - h) ** min(4, 50 * h / 3):.2f}")
# how much of the gap between HC3 and HC4 is Alaska's factor alone: HC3 for every state, HC4's factor for Alaska
X1 = np.column_stack([np.ones(len(ps)), ps.Income.values * 1e-4, (ps.Income.values * 1e-4) ** 2])
w = f_all.resid ** 2 / (1 - f_all.hat) ** 2
w[ak] = e ** 2 / (1 - h) ** min(4, len(ps) * h / 3)
V = f_all.XtX_inv @ (X1.T * w) @ X1 @ f_all.XtX_inv
out.append(f"HC3 with Alaska's factor replaced by HC4's: SE of Income^2 {np.sqrt(V[2, 2]):.2f}, z p-value "
           f"{t_pvalue(f_all.coef[2], np.sqrt(V[2, 2])):.4f} (HC3 {res_all['HC3'][1]:.4f}, HC4 {res_all['HC4'][1]:.4f})")
write("01_public_schools.txt", "\n".join(out))

# --------------------------------------------------------------- figure 1
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.42})
ax = axes[0]
inc = ps.Income.values * 1e-4
sc = ax.scatter(ps.Income / 1000, ps.Expenditure, c=f_all.hat, cmap="viridis_r", s=36, edgecolor="white", lw=0.5)
g = np.linspace(inc.min(), inc.max(), 200)
ax.plot(g * 10, f_all.coef[0] + f_all.coef[1] * g + f_all.coef[2] * g ** 2, color=ORANGE, lw=2,
        label="Quadratic fit, all states")
ax.plot(g * 10, f_noak.coef[0] + f_noak.coef[1] * g + f_noak.coef[2] * g ** 2, color=INK2, lw=1.5, ls="--",
        label="Quadratic fit without Alaska")
ax.annotate("Alaska (leverage 0.65)", (ps.Income[ak] / 1000, ps.Expenditure[ak]), xytext=(-10, -4),
            textcoords="offset points", ha="right", fontsize=9)
ax.set_xlabel("Per-capita income, 1979 (thousand dollars)")
ax.set_ylabel("Per-capita spending on public schools ($)")
ax.set_title("One state decides the curvature")
ax.legend(fontsize=8.5, loc="upper left")
cax = ax.inset_axes([0.68, 0.1, 0.28, 0.03])
cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
cax.set_title("Leverage", fontsize=8.5, loc="left", fontweight="normal")
cb.ax.tick_params(labelsize=8)
ax = axes[1]
names = list(HC_TYPES) + ["WR rademacher", "WR mammen", "WR webb", "WU rademacher"]
labels = ["Classical", "HC0", "HC1", "HC2", "HC3", "HC4", "HC4m", "HC5",
          "Wild, restricted,\nRademacher", "Wild, restricted,\nMammen", "Wild, restricted,\nWebb",
          "Wild, unrestricted,\nRademacher"]
for i, nm in enumerate(names):
    ax.plot(res_all[nm][1], i, "o", color=BLUE, ms=6, label="All states" if i == 0 else None)
    ax.plot(res_noak[nm][1], i, "s", color=AQUA, ms=5, label="Without Alaska" if i == 0 else None)
ax.axvline(0.05, color=ORANGE, lw=1, ls=":")
ax.text(0.055, -0.6, "5%", color=ORANGE, fontsize=9)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(labels, fontsize=8.5)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlim(1e-4, 1.2)
from matplotlib.ticker import FuncFormatter
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlabel("p-value for the Income² coefficient (log scale)")
ax.set_title("Same model, twelve answers")
ax.legend(fontsize=8.5, loc="lower left")
fig.savefig(FIG / "fig1_public_schools.png"); plt.close(fig)
