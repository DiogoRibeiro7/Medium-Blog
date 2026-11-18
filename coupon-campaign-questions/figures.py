"""Draw the five figures of the article from the tables coupon_numbers.py writes to results/.

    python figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

# chart ink and the first three categorical slots of the reference palette
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SHADE = "#f0efec"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "axes.titlecolor": INK, "axes.titleweight": "bold", "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False})


def grid_x(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def grid_y(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def frame(fig, title, subtitle=None, note=None, top_in=0.75, bottom_in=0.75):
    """Title and subtitle above the plots, a note and the data credit below, with room reserved in inches."""
    h = fig.get_figheight()
    fig.subplots_adjust(top=1 - top_in / h, bottom=bottom_in / h)
    fig.text(0.01, 1 - 0.08 / h, title, fontsize=11, fontweight="bold", color=INK, ha="left", va="top")
    if subtitle:
        fig.text(0.01, 1 - 0.34 / h, subtitle, fontsize=8.5, color=INK2, ha="left", va="top")
    y = 0.06 / h
    fig.text(0.01, y, "Data: 84.51° Complete Journey (completejourney R package, CC0). Figure: Diogo Ribeiro.",
             fontsize=7.5, color=MUTED, ha="left", va="bottom")
    if note:
        fig.text(0.01, y + 0.2 / h, note, fontsize=8, color=INK2, ha="left", va="bottom")


def fig1_groups():
    """Weekly spend before and during the targeted campaigns, by what the dashboard groups people into."""
    g = pd.read_csv(RES / "naive_groups.csv").set_index("group")
    order = ["not sent the campaign", "recipients, no redemption", "redeemers"]
    labels = {"not sent the campaign": "Not sent the campaign", "recipients, no redemption": "Sent it, did not redeem",
              "redeemers": "Sent it, redeemed a coupon"}
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for i, name in enumerate(order):
        r = g.loc[name]
        ax.plot([r.pre, r.during], [i, i], color=GRID, linewidth=2, zorder=1)
        ax.scatter(r.pre, i, s=110, facecolor="white", edgecolor=BLUE, linewidth=2, zorder=3)
        ax.scatter(r.during, i, s=45, color=BLUE, edgecolor="white", linewidth=1.5, zorder=4)
        ax.annotate(f"${r.pre:.0f}", (r.pre, i), xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=8.5, color=INK2)
        ax.annotate(f"${r.during:.0f}", (r.during, i), xytext=(0, -15), textcoords="offset points", ha="center",
                    fontsize=8.5, color=INK)
    ax.set_yticks(range(3), [labels[n] for n in order])
    ax.set_ylim(-0.7, 2.7)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Mean spend per household per week (dollars)")
    grid_x(ax)
    ax.scatter([], [], s=60, facecolor="white", edgecolor=BLUE, linewidth=2, label="Eight weeks before")
    ax.scatter([], [], s=60, color=BLUE, label="During the campaign")
    ax.legend(loc="lower right", fontsize=8.5, handletextpad=0.3)
    frame(fig, "The redeemers were the big spenders before anyone redeemed anything",
          "Targeted campaigns 8, 13 and 18 pooled; households with a receipt in the eight weeks before",
          bottom_in=0.95)
    fig.subplots_adjust(left=0.27)
    fig.savefig(FIG / "fig1_redeemers_before.png")
    plt.close(fig)


def fig2_event_study():
    """Recipients minus controls by week, for three ways of choosing the controls."""
    ev = pd.read_csv(RES / "event_study.csv")
    labs = list(ev.position.drop_duplicates())
    x = np.arange(len(labs), dtype=float)
    # leave a visual gap between the campaign weeks and the post weeks (post weeks count from the calendar end)
    x[[i for i, l in enumerate(labs) if l[0] in "dp"]] += 0.8
    x[[i for i, l in enumerate(labs) if l.startswith("p")]] += 1.2
    lines = [("matched on weeks -8..-1", BLUE, "Matched on weeks −8 to −1"),
             ("matched on weeks -16..-9", ORANGE, "Matched on weeks −16 to −9"),
             ("no matching", AQUA, "No matching")]
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.4), sharex=True)
    specs = [("known", 100, "Baskets with a covered product,\nper 100 recipients per week",
              "Known coupons (11 campaigns of Types B and C)"),
             ("targeted", 1, "Total spend per recipient\nper week (dollars)", "Targeted coupons (campaigns 13 and 18)")]
    d_idx = [i for i, l in enumerate(labs) if l.startswith("d")]
    for ax, (prog, scale, ylab, title) in zip(axes, specs):
        ax.axvspan(x[d_idx[0]] - 0.5, x[d_idx[-1]] + 0.5, color=SHADE, zorder=0)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        for key, col, name in lines:
            s = ev[(ev.programme == prog) & (ev.line == key)].set_index("position").reindex(labs)
            for j, part in enumerate("wdp"):           # draw before, during and after as separate segments
                m = (~s.effect.isna()).to_numpy() & np.array([l[0] == part for l in labs])
                ax.fill_between(x[m], s.lo[m] * scale, s.hi[m] * scale, color=col, alpha=0.10, linewidth=0)
                ax.plot(x[m], s.effect[m] * scale, color=col, linewidth=2, label=name if j == 0 else None)
        # mark the window each matched line was matched on
        y0, y1 = ax.get_ylim()
        y = y0 - 0.02 * (y1 - y0)
        ax.set_ylim(y0 - 0.06 * (y1 - y0), y1)
        for (lo, hi), col in [((-8, -1), BLUE), ((-16, -9), ORANGE)]:
            a, b = labs.index(f"w{lo}"), labs.index(f"w{hi}")
            ax.plot([x[a] - 0.4, x[b] + 0.4], [y, y], color=col, linewidth=4, solid_capstyle="butt")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=10)
        grid_y(ax)
        top = ax.get_ylim()[1]
        ax.text(x[d_idx[0]] - 0.3, top, "campaign weeks", fontsize=8, color=INK2, va="top")
    ticks = [i for i, l in enumerate(labs) if l in ("w-24", "w-16", "w-8", "w-1", "d1", "d5", "p1", "p8")]
    names = {"w-24": "−24", "w-16": "−16", "w-8": "−8", "w-1": "−1", "d1": "1", "d5": "5", "p1": "+1", "p8": "+8"}
    axes[1].set_xticks(x[ticks], [names[labs[i]] for i in ticks])
    axes[1].set_xlabel("Weeks before the campaign (−24 to −1), campaign weeks (1 to 5), weeks after its calendar end (+1 to +8)")
    axes[0].legend(loc="lower left", fontsize=8.5, ncol=3, bbox_to_anchor=(0.0, 1.10))
    frame(fig, "Each matched line jumps where its matching window ends, not where the coupons start",
          "Recipients minus controls, each line measured from its own baseline window (bars on the axis); "
          "bands are 95% household-bootstrap intervals", top_in=1.15, bottom_in=0.95)
    fig.subplots_adjust(hspace=0.35)
    fig.savefig(FIG / "fig2_where_the_jump_is.png")
    plt.close(fig)


def fig3_ledger():
    """Each design's estimate next to what the same design reports in a window with no coupons."""
    led = pd.read_csv(RES / "design_ledger.csv")
    names = {"matched": "Matched on the\neight weeks before", "plain": "All households,\nchange from the\neight weeks before",
             "trend": "All households,\neach against its\nown trend"}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.8))
    specs = [("known", 100, "Baskets with a covered product\nper 100 recipient-weeks", "Known coupons"),
             ("targeted", 1, "Dollars of total spend\nper recipient-week", "Targeted coupons")]
    for ax, (prog, scale, xlab, title) in zip(axes, specs):
        s = led[led.programme == prog]
        for i, d in enumerate(["matched", "plain", "trend"]):
            for kind, off, col, mk in [("estimate", 0.14, BLUE, "o"), ("placebo", -0.14, ORANGE, "D")]:
                r = s[(s.design == d) & (s.kind == kind)].iloc[0]
                y = 2 - i + off
                ax.plot([r.lo * scale, r.hi * scale], [y, y], color=col, linewidth=2, solid_capstyle="round")
                ax.scatter(r.value * scale, y, s=46, marker=mk, color=col if kind == "estimate" else "white",
                           edgecolor=col, linewidth=2, zorder=3)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_yticks([2, 1, 0], [names[d] for d in ["matched", "plain", "trend"]] if ax is axes[0] else [])
        ax.set_ylim(-0.6, 2.6)
        ax.set_xlabel(xlab)
        ax.set_title(title, loc="left", fontsize=10)
        grid_x(ax)
    axes[0].scatter([], [], s=40, color=BLUE, label="Estimate in the campaign weeks")
    axes[0].scatter([], [], s=40, marker="D", facecolor="white", edgecolor=ORANGE, linewidth=2,
                    label="Same design, eight weeks earlier (no coupons)")
    axes[0].legend(loc="lower left", fontsize=8.5, ncol=2, bbox_to_anchor=(0.0, 1.10))
    frame(fig, "A design is only as good as what it says when nothing happened",
          "Each design in the campaign weeks, and the same design eight weeks earlier, before this campaign's coupons existed",
          note="95% household-bootstrap intervals. A placebo away from zero means the design manufactures effects of that size.",
          top_in=1.2, bottom_in=1.2)
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(FIG / "fig3_designs_and_placebos.png")
    plt.close(fig)


def fig4_per_campaign():
    """One z-statistic per campaign under each design: how many campaigns 'worked'."""
    per = pd.read_csv(RES / "campaigns_by_design.csv", dtype={"campaign": str})
    rows = [("matched16", "Matched on the eight\nweeks before"), ("plain16", "All households, change\nfrom the eight weeks before"),
            ("trend16", "All households, each\nagainst its own trend")]
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.axvspan(-1.96, 1.96, color=SHADE, zorder=0)
    for i, (d, name) in enumerate(rows):
        s = per[per.design == d].copy()
        z = np.sign(s.estimate) * norm.ppf(1 - s.p / 2)
        rank = np.argsort(np.argsort(z.to_numpy()))
        y = 2 - i + ((rank % 3) - 1) * 0.13                 # three lanes, so neighbouring dots do not hide each other
        sig = (s.p < 0.05).to_numpy()
        ax.scatter(z[sig], y[sig], s=46, color=BLUE, edgecolor=BLUE, linewidth=1.5, zorder=3)
        ax.scatter(z[~sig], y[~sig], s=46, facecolor="white", edgecolor=BLUE, linewidth=1.5, zorder=3)
        ax.text(9.6, 2 - i, f"{sig.sum()} of {len(s)}", va="center", ha="right", fontsize=9, color=INK)
    ax.set_yticks([2, 1, 0], [n for _, n in rows])
    ax.set_xlim(-4.5, 9.8)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel("Estimate divided by its standard error, one dot per campaign (16 campaigns)")
    ax.text(0, 2.52, "|z| < 1.96", ha="center", fontsize=8, color=INK2)
    ax.text(9.6, 2.52, "p < 0.05", ha="right", fontsize=8, color=INK2)
    grid_x(ax)
    frame(fig, "The same sixteen campaigns, three designs, three different lists of winners",
          "Filled: p < 0.05 for that campaign alone. Shaded: |z| < 1.96.", bottom_in=0.95)
    fig.subplots_adjust(left=0.27)
    fig.savefig(FIG / "fig4_winners_by_design.png")
    plt.close(fig)


def fig5_break_even():
    """Incremental share a coupon needs to pay, against the contribution margin, with what the data allow."""
    sh = pd.read_csv(RES / "incremental_shares.csv").set_index("design")
    pw = pd.read_csv(RES / "power_inputs.csv").iloc[0]
    be = pd.read_csv(RES / "break_even.csv")
    depth = float(be.break_even_share.iloc[0] * be.margin.iloc[0])
    m = np.linspace(0.10, 0.80, 200)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    r = sh.loc["plain"]
    ax.axhspan(max(r.lo, 0), r.hi, color=AQUA, alpha=0.14, linewidth=0)
    ax.axhline(r.share, color=AQUA, linewidth=2)
    ax.plot(m, depth / m, color=BLUE, linewidth=2)
    ax.fill_between(m, depth / m, 1.6, color=BLUE, alpha=0.05, linewidth=0)
    ax.set_xlim(0.10, 0.80)
    ax.set_ylim(0, 1.6)
    ax.set_xticks(np.arange(0.1, 0.81, 0.1), [f"{v:.0%}" for v in np.arange(0.1, 0.81, 0.1)])
    ax.set_yticks(np.arange(0, 1.61, 0.2), [f"{v:.0%}" for v in np.arange(0, 1.61, 0.2)])
    ax.set_xlabel("Manufacturer's contribution margin on the shelf value of the couponed items")
    ax.set_ylabel("Share of redemptions that are\nextra purchases")
    ax.text(0.155, 1.48, "coupon pays", fontsize=9, color=BLUE)
    ax.text(0.305, depth / 0.30 + 0.05, f"break-even = {depth:.0%} coupon depth ÷ margin", fontsize=8.5,
            color=BLUE, ha="left", va="bottom")
    ax.text(0.115, r.share + 0.025, f"estimate {r.share:.2f}", fontsize=8.5, color=INK, ha="left", va="bottom")
    ax.text(0.235, r.hi + 0.025, f"upper end of the 95% interval, {r.hi:.2f} (lower end {r.lo:.2f})", fontsize=8.5,
            color=INK2, ha="left", va="bottom")
    ax.axvline(depth / r.share, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.text(depth / r.share + 0.01, 1.55, f"break-even margin\nat the estimate: {depth / r.share:.0%}", fontsize=8,
            color=INK2, ha="left", va="top")
    grid_y(ax)
    frame(fig, f"At the estimate the coupons pay only above a {depth / r.share:.0%} margin; "
               f"the interval reaches break-even above {depth / r.hi:.0%}",
          f"Known-coupon campaigns; share from the plain difference in differences (redemption rate {pw.r * 100:.2f} "
          f"per 100 recipient-weeks)", bottom_in=1.0)
    fig.savefig(FIG / "fig5_break_even.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig1_groups()
    fig2_event_study()
    fig3_ledger()
    fig4_per_campaign()
    fig5_break_even()


if __name__ == "__main__":
    main()
