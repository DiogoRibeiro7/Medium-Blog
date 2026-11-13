"""Figures for the closing article. Reads results/, writes figures/fig*.png.

    python figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

# reference palette (light mode); blue / orange validated for CVD separation on this surface
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
})

SOURCE = ("Data: results of the five companion projects (Forbes lists 2011-2026, Penn World Table 11.0, "
          "World Bank WDI, OxCGRT, ILOSTAT, WID). Intervals: 95%, clustered by country.")

STUDY_ORDER = ["audit", "panel", "sectors", "pandemic", "successions"]
STUDY_LABEL = {
    "audit": "Simulation and\ncross-country audit",
    "panel": "Within-country\npanel",
    "sectors": "Sector\ntest",
    "pandemic": "Pandemic\nexposure",
    "successions": "Succession\nevents",
}

GROUPS = [
    ("The claim", [
        "Top-1% wealth share predicts slower growth within countries",
        "Inherited control goes with slower TFP growth",
        "Holds across the whole specification set",
        "Also shows in GDP per capita growth",
    ]),
    ("Confounders", [
        "Not driven by one country",
        "Not explained by the sector mix",
        "Not explained by pandemic exposure",
    ]),
    ("Measurement", [
        "Not an artefact of one list's valuations",
        "Survives without the 2020 list",
    ]),
    ("Timing and cause", [
        "Not a reflection of earlier growth",
        "Beats randomly reshuffled timing",
        "Founder exits without heirs show no drop",
        "Event study and panel agree on size",
    ]),
]

VERDICT_STYLE = {
    "passed": dict(marker="o", ms=10, mfc=BLUE, mec=BLUE, mew=1.5, label="passed"),
    "open": dict(marker="o", ms=11, mfc=SURFACE, mec=INK2, mew=1.8, label="open (test could not tell)"),
    "failed": dict(marker="X", ms=12, mfc=ORANGE, mec=ORANGE, mew=1.0, label="failed"),
}


def minus(s: str) -> str:
    """Typographic minus signs for printed numbers."""
    return s.replace("-", "−")


def save(fig, name: str, y: float = 0.005) -> None:
    fig.text(0.01, y, SOURCE, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1_estimates(est: pd.DataFrame) -> None:
    comp_rows = [
        ("panel", "panel_pre2020", "Within-country panel,\nincome controlled"),
        ("sectors6", "sectors6_no2020", "+ six sector-group shares"),
        ("sectors18", None, "+ all 18 sector shares"),
        ("exposure", "exposure_no2020", "+ four pandemic exposures"),
        ("headcount", "headcount_no2020", "Headcount measure,\nexposures controlled"),
        ("twolist", "twolist_no2020", "Two-list average,\nexposures controlled"),
    ]
    event_rows = [
        ("succ_all", "succ_all_no2020", "All matched successions"),
        ("succ_old", "succ_old_no2020", "Founders aged 70 or more"),
        ("succ_exposure", None, "All successions,\npandemic exposure controlled"),
        ("succ_old_any", None, "Any exit of a founder\naged 70 or more"),
        ("succ_old_noheir", None, "Exit aged 70 or more,\nno heir on the list"),
    ]
    E = est.set_index("key")
    power = json.loads((RES / "power.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.85})
    off = 0.17

    def draw(ax, rows, xlabel, title, step):
        n = len(rows)
        for i, (full, drop, label) in enumerate(rows):
            y = n - 1 - i
            f = E.loc[full]
            ax.plot([f.lo, f.hi], [y + off, y + off], color=BLUE, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(f.b, y + off, "o", ms=8, color=BLUE, mec=SURFACE, mew=1.5, zorder=3)
            if drop is not None:
                d = E.loc[drop]
                ax.plot([d.lo, d.hi], [y - off, y - off], color=ORANGE, lw=2, solid_capstyle="round", zorder=2)
                ax.plot(d.b, y - off, "o", ms=8, mfc=SURFACE, mec=ORANGE, mew=2, zorder=3)
                if drop == "panel_pre2020":
                    ax.annotate("lists 2011–2016 only", (d.hi, y - off), xytext=(6, 0),
                                textcoords="offset points", va="center", fontsize=8, color=INK2)
        ax.axvline(0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_yticks(range(n))
        ax.set_yticklabels([r[2] for r in rows][::-1], fontsize=9.5, color=INK)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.7, n - 0.3)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(xlabel, fontsize=9.5)
        ax.set_title(title, fontsize=11)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(step))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: minus(f"{v:+g}") if abs(v) > 1e-9 else "0"))

    draw(axes[0], comp_rows,
         "Inherited minus self-made coefficient\n(pp of annual TFP growth per log point)",
         "Composition coefficient, six designs", 1)
    draw(axes[1], event_rows,
         "Change in TFP growth over the event year\nand the two after (pp per log point of the shock)",
         "Succession event study", 2)
    target = power["panel_implied_post"]
    ax = axes[1]
    ax.axvline(target, color=INK, lw=1.2, ls=(0, (1, 2)), zorder=1)
    ax.annotate("size the panel\nimplies", (target, len(event_rows) - 0.45), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=INK2, va="top")

    handles = [
        Line2D([], [], color=BLUE, marker="o", ms=8, mec=SURFACE, mew=1.5, lw=2, label="all list years"),
        Line2D([], [], color=ORANGE, marker="o", ms=8, mfc=SURFACE, mec=ORANGE, mew=2, lw=2,
               label="2020 list dropped (event study: growth years 2020–21 dropped)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 1.0), ncol=2, fontsize=9.5,
               handlelength=2.2)
    fig.suptitle("The coefficient is stable across controls and moves with the 2020 list",
                 x=0.01, y=1.05, ha="left", fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.86)
    save(fig, "fig1_estimates", y=-0.06)


def fig2_ledger(led: pd.DataFrame) -> None:
    rows: list[tuple[str, str | None]] = []
    for group, items in GROUPS:
        rows.append((group, None))
        rows.extend((group, it) for it in items)
    known = {it for _, items in GROUPS for it in items}
    missing = set(led["objection"]) - known
    if missing:
        raise ValueError(f"ledger rows without a group: {missing}")

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(13.5, 0.62 * n_rows + 1.6))
    ax.set_xlim(-3.55, len(STUDY_ORDER) - 0.45)
    ax.set_ylim(n_rows - 0.4, -1.25)
    ax.axis("off")

    for j, s in enumerate(STUDY_ORDER):
        ax.text(j, -0.95, STUDY_LABEL[s], ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)

    cell = {(r.objection, r.study): r for r in led.itertuples()}
    for i, (group, item) in enumerate(rows):
        if item is None:
            ax.text(-3.5, i + 0.1, group.upper(), ha="left", va="center", fontsize=8.5, color=MUTED,
                    fontweight="bold")
            ax.plot([-3.5, len(STUDY_ORDER) - 0.5], [i - 0.42, i - 0.42], color=GRID, lw=1, clip_on=False)
            continue
        ax.text(-3.4, i, item, ha="left", va="center", fontsize=9.5, color=INK)
        for j, s in enumerate(STUDY_ORDER):
            r = cell.get((item, s))
            if r is None:
                ax.plot(j, i, marker="o", ms=2.5, color=GRID, ls="none")
                continue
            st = {k: v for k, v in VERDICT_STYLE[r.verdict].items() if k != "label"}
            ax.plot(j, i - 0.17, ls="none", **st)
            ax.text(j, i + 0.27, minus(r.cell), ha="center", va="center", fontsize=8, color=INK2)

    handles = [Line2D([], [], ls="none", **{k: v for k, v in st.items()}) for st in VERDICT_STYLE.values()]
    handles.append(Line2D([], [], ls="none", marker="o", ms=2.5, color=GRID, label="not tested in that study"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.015), ncol=4, fontsize=9.5,
               handletextpad=0.4, columnspacing=1.6)
    counts = led["verdict"].value_counts()
    fig.suptitle(f"{len(led)} tests of one association: what passed, what failed, what stayed open",
                 x=0.01, y=0.995, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.955,
             f"{int(counts.get('passed', 0))} passed, {int(counts.get('open', 0))} open, "
             f"{int(counts.get('failed', 0))} failed. Each cell shows the statistic the verdict rests on "
             "(estimate and 95% interval, a p-value, or a count).",
             fontsize=9.5, color=INK2, ha="left")
    fig.subplots_adjust(top=0.92, bottom=0.08)
    save(fig, "fig2_ledger")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    est = pd.read_csv(RES / "estimates.csv")
    led = pd.read_csv(RES / "ledger.csv")
    fig1_estimates(est)
    fig2_ledger(led)


if __name__ == "__main__":
    main()
