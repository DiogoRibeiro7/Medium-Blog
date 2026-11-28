"""Five figures for the buybacks / CEO wealth article.
Usage: python3 -I figures.py RESULTS_DIR OUT_DIR
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from series import BUYBACKS, PAY_RATIO

res, out = sys.argv[1:3]
BLUE, RED, GREY, LIGHT, INK, MUTED, GREEN = "#257de0", "#c4452d", "#9c9c9c", "#dddddd", "#191919", "#666666", "#1c9b7a"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#cccccc", "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
grid = dict(color="#eeeeee", lw=0.8, zorder=0)
yrs = sorted(BUYBACKS)

# Fig 1: buybacks by year
fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
vals = [BUYBACKS[y] for y in yrs]
ax.bar(yrs, vals, color=[RED if y in (2009,) else BLUE for y in yrs], width=0.72, zorder=2)
for y in (2007, 2009, 2018, 2022, 2024):
    ax.annotate(f"{BUYBACKS[y]:.0f}", (y, BUYBACKS[y]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color=INK)
ax.set_ylabel("USD billion"); ax.set_xticks(yrs[::2])
ax.set_title("S&P 500 buybacks fell by three quarters from 2007 to 2009, then reached 942 bn in 2024", loc="left", fontsize=11, color=INK)
ax.grid(axis="y", **grid); fig.tight_layout(); fig.savefig(f"{out}/fig1_buybacks_by_year.png"); plt.close(fig)

# Fig 2: pay ratios 1965-2024 with events
ally = sorted(PAY_RATIO)
fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=160)
ax.plot(ally, [PAY_RATIO[y][0] for y in ally], color=BLUE, lw=2, label="Realized (stock actually received)")
ax.plot(ally, [PAY_RATIO[y][1] for y in ally], color=RED, lw=2, label="Granted (value when awarded)")
for x, lab, dy in [(1982, "1982\nRule 10b-18", 215), (1997, "1997\nbuybacks pass\ndividends", 120), (2023, "2023\n1% buyback\ntax", 120)]:
    ax.axvline(x, color="#bbbbbb", lw=1, ls=":")
    ax.text(x - 0.6, 440 if x != 2023 else 478, lab, ha="right", va="top", fontsize=8.5, color=MUTED)
ax.set_ylabel("CEO pay as a multiple of typical worker pay"); ax.set_ylim(0, 480)
ax.set_title("The pay ratio rose most before buybacks became large, and is lower in 2024 than in 2000", loc="left", fontsize=11, color=INK)
ax.legend(frameon=False, loc="upper left", fontsize=9, bbox_to_anchor=(0.02, 0.74))
ax.grid(**grid); fig.tight_layout(); fig.savefig(f"{out}/fig2_pay_ratio_long_run.png"); plt.close(fig)

# Fig 3: co-movement of annual log changes
y1 = np.array(yrs[1:])
bb = np.array([BUYBACKS[y] for y in yrs]); db = np.diff(np.log(bb)) * 100
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), dpi=160, sharex=True)
for ax, (idx, name, col) in zip(axs, [(1, "Granted pay ratio", RED), (0, "Realized pay ratio", BLUE)]):
    s = np.array([PAY_RATIO[y][idx] for y in yrs]); ds = np.diff(np.log(s)) * 100
    r = stats.pearsonr(db, ds)[0]; b = np.polyfit(db, ds, 1)
    xs = np.linspace(db.min(), db.max(), 50)
    ax.plot(xs, np.polyval(b, xs), color=col, lw=1.5, alpha=0.6, zorder=2)
    ax.scatter(db, ds, color=col, s=26, zorder=3)
    for yv, xv, zv in zip(y1, db, ds):
        if yv in (2008, 2009, 2010, 2012, 2018, 2020, 2021, 2022, 2023):
            ax.annotate(str(yv), (xv, zv), xytext=(4, 3), textcoords="offset points", fontsize=8, color=MUTED)
    ax.axhline(0, color="#cccccc", lw=0.8); ax.axvline(0, color="#cccccc", lw=0.8)
    ax.set_xlabel("Change in S&P 500 buybacks, log points x 100"); ax.set_ylabel(f"Change in {name.lower()}, log points x 100")
    ax.set_title(f"{name}: r = {r:.2f}, n = {len(db)}", loc="left", fontsize=11, color=INK); ax.grid(**grid)
fig.suptitle("Year-to-year changes, 2008-2024: buybacks and CEO pay ratios rise and fall together", x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(f"{out}/fig3_comovement.png"); plt.close(fig)

# Fig 4: robustness of the correlations
rb = json.load(open(f"{res}/robustness.json"))
labels = [("all", "All years (2008-2024)"), ("drop_2008_2009", "Drop 2008, 2009"), ("drop_2008_2009_2010", "Drop 2008-2010"),
          ("drop_2020_2021", "Drop 2020, 2021"), ("drop_2008_2009_2020", "Drop 2008, 2009, 2020"),
          ("last_decade_2016_2024", "2016-2024 only"), ("first_period_2008_2015", "2008-2015 only")]
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
for i, (k, lab) in enumerate(labels):
    g, r_ = rb[k]["granted"], rb[k]["realized"]
    ax.scatter(g[0], -i, color=RED, s=40, zorder=3, label="Granted ratio" if i == 0 else None)
    ax.scatter(r_[0], -i, color=BLUE, s=40, zorder=3, label="Realized ratio" if i == 0 else None)
    ax.annotate(f"n={g[2]}", (1.02, -i), fontsize=8, color=MUTED, va="center", annotation_clip=False)
ax.axvline(0, color="#bbbbbb", lw=1)
ax.set_yticks([-i for i in range(len(labels))]); ax.set_yticklabels([l for _, l in labels])
ax.set_xlim(-0.1, 1.05); ax.set_xlabel("Pearson correlation of annual changes with buyback growth")
ax.set_title("The granted-pay link holds across sample changes; the realized-pay link is fragile", loc="left", fontsize=11, color=INK)
ax.legend(frameon=False, loc="lower left", fontsize=9, bbox_to_anchor=(0.0, 0.0)); ax.grid(axis="x", **grid)
fig.tight_layout(); fig.savefig(f"{out}/fig4_robustness.png"); plt.close(fig)

# Fig 5: transfer arithmetic
tm = json.load(open(f"{res}/transfer_model.json"))
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), dpi=160)
ax = axs[0]
ratio = np.linspace(0.7, 1.5, 100); f = 0.05
ax.plot(ratio, 100 * f * (1 - ratio) / (1 - f), color=BLUE, lw=2)
ax.axhline(0, color="#bbbbbb", lw=1); ax.axvline(1, color="#bbbbbb", lw=1, ls=":")
ax.set_xlabel("Buyback price divided by intrinsic value"); ax.set_ylabel("Change in value of a share you keep, %")
ax.set_title("A 5% buyback: holders who stay gain only if price is below value", loc="left", fontsize=10.5, color=INK); ax.grid(**grid)
ax = axs[1]
B = tm["transfer_2024_volume"]["buybacks_usd_bn"]; m = np.linspace(-0.1, 0.2, 100)
ax.plot(100 * m, B * m / (1 + m), color=RED, lw=2)
ax.axhline(0, color="#bbbbbb", lw=1)
ax.set_xlabel("Average price premium over intrinsic value, %"); ax.set_ylabel("Wealth moved to sellers, USD billion")
ax.set_title(f"At 2024 volume ({B:.0f} bn), each point of premium moves about 9 bn", loc="left", fontsize=10.5, color=INK); ax.grid(**grid)
fig.tight_layout(); fig.savefig(f"{out}/fig5_transfer_arithmetic.png"); plt.close(fig)
print("figures written")
