"""Figures for the article. Reads results/ and data/wdi_countries.csv, writes figures/fig*.png.

    python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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

SOURCE = "Data: Forbes person-list API (annual lists 2011-2023), Penn World Table 11.0, World Bank WDI."


def save(fig, name: str, source: str = SOURCE) -> None:
    fig.text(0.01, 0.005, source, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1_sector_inheritance() -> None:
    s = pd.read_csv(RES / "sector_inheritance.csv").set_index("sector").drop("Unclassified", errors="ignore")
    s = s.sort_values("inh_share")
    overall = s["inherited"].sum() / (s["inherited"] + s["self_made"]).sum()
    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    y = np.arange(len(s))
    ax.hlines(y, 0, s["inh_share"], color=GRID, lw=1.2, zorder=1)
    ax.scatter(s["inh_share"], y, s=2400 * s["worth_share"] + 12, color=BLUE, edgecolor=SURFACE,
               linewidth=2, zorder=3, alpha=0.95)
    for yi, (name, r) in zip(y, s.iterrows()):
        ax.annotate(f"{r['inh_share']:.0%}  ({r['worth_share']:.0%} of wealth)", (r["inh_share"], yi),
                    textcoords="offset points", xytext=(14, -3), fontsize=8.5, color=INK2)
    ax.axvline(overall, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=0)
    ax.text(overall + 0.006, len(s) - 0.4, f"all sectors: {overall:.0%}", fontsize=8.5, color=INK2, va="bottom")
    ax.set_yticks(y, s.index)
    ax.set_xlim(0, 0.84)
    ax.set_ylim(-0.8, len(s) + 0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Share of the sector's billionaire wealth held by heirs (pooled 2011-2023 lists)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Heirs hold 6% of technology fortunes and 58% of diversified holdings\n"
                 "(dot area: sector's share of all billionaire wealth)")
    fig.tight_layout()
    save(fig, "fig1_sector_inheritance")


def fig2_observed_vs_predicted() -> None:
    p = pd.read_csv(RES / "panel.csv")
    names = pd.read_csv(HERE / "data" / "wdi_countries.csv", keep_default_na=False).set_index("iso3")["name"]
    names = names.to_dict()
    names.update({"KOR": "South Korea", "RUS": "Russia", "HKG": "Hong Kong", "TWN": "Taiwan", "TUR": "Turkey",
                  "CZE": "Czechia", "EGY": "Egypt"})
    w = p[p["n"] > 0].groupby("iso3")[["inherited_m", "self_made_m", "pred_inh_m"]].sum()
    w["total"] = w["inherited_m"] + w["self_made_m"]
    w = w.sort_values("total", ascending=False).head(22)
    w["obs"] = w["inherited_m"] / w["total"]
    w["pred"] = w["pred_inh_m"] / w["total"]
    w = w.sort_values("obs")
    fig, ax = plt.subplots(figsize=(9.0, 7.4))
    y = np.arange(len(w))
    for yi, (_, r) in zip(y, w.iterrows()):
        ax.plot([r["pred"], r["obs"]], [yi, yi], color=GRID, lw=3, zorder=1, solid_capstyle="butt")
    ax.scatter(w["pred"], y, s=70, facecolor=SURFACE, edgecolor=ORANGE, linewidth=2, zorder=3,
               label="predicted from the country's sector mix")
    ax.scatter(w["obs"], y, s=70, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=4,
               label="observed")
    ax.set_yticks(y, [names.get(i, i) for i in w.index])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(-0.02, 0.9)
    ax.set_xlabel("Inherited share of the country's billionaire wealth (pooled 2011-2023 lists)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"Sector mix predicts inherited shares between {w['pred'].min():.0%} and {w['pred'].max():.0%};\n"
                 f"observed shares run from {w['obs'].min():.1%} to {w['obs'].max():.0%}")
    fig.tight_layout()
    save(fig, "fig2_observed_vs_predicted",
         SOURCE + " 22 countries with the most billionaire wealth; prediction applies other countries' inherited share in each sector to the country's sector mix.")


def fig3_with_without() -> None:
    f = pd.read_csv(RES / "fits.csv")
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    lim = (-1.5, 0.55)
    ax.plot(lim, lim, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    for var, col, lab in (("tfp", BLUE, "TFP growth"), ("gdppc", ORANGE, "GDP per capita growth")):
        for est, filled in (("fe", True), ("pooled", False)):
            sub = f[(f["outcome"] == var) & (f["estimator"] == est)]
            ax.scatter(sub["diff_no_sector"], sub["diff_sector_ctrl"], s=64, zorder=3,
                       facecolor=col if filled else SURFACE, edgecolor=col if not filled else SURFACE,
                       linewidth=2 if not filled else 1.5,
                       label=f"{lab}, {'country fixed effects' if est == 'fe' else 'pooled'}")
    head = f[(f["outcome"] == "tfp") & (f["horizon"] == 3) & (f["sample"] == "billionaire_countries") & (f["estimator"] == "fe")].iloc[0]
    ax.annotate(f"previous headline estimate:\n{head['diff_no_sector']:+.2f} without, {head['diff_sector_ctrl']:+.2f} with sector shares",
                (head["diff_no_sector"], head["diff_sector_ctrl"]), textcoords="data", xytext=(-1.08, -1.44),
                fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.text(0.5, 0.02, "dashed line: same estimate\nwith and without", fontsize=8.5, color=MUTED, ha="right", va="top")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Inherited minus self-made coefficient, no sector controls")
    ax.set_ylabel("Same coefficient, controlling for six sector-group shares")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Holding the sector mix fixed leaves the 24 estimates\nclose to where they were")
    fig.tight_layout()
    save(fig, "fig3_with_without", SOURCE + " Percentage points of annual growth per log point.")


def fig4_list_years() -> None:
    d = pd.read_csv(RES / "leave_one_year_out.csv")
    d = d[d["sample"] == "billionaire_countries"]
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    for h, col, dx in ((3, BLUE, -0.13), (5, ORANGE, 0.13)):
        sub = d[d["horizon"] == h]
        ax.vlines(sub["dropped_year"] + dx, sub["lo"], sub["hi"], color=col, lw=2, alpha=0.55)
        ax.scatter(sub["dropped_year"] + dx, sub["b"], s=46, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   label=f"{h}-year growth (full sample {sub['full_b'].iloc[0]:+.2f})")
        ax.hlines(sub["full_b"].iloc[0], sub["dropped_year"].min() - 0.4, sub["dropped_year"].max() + 0.4,
                  color=col, lw=1, ls=(0, (4, 3)))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(2011, 2021))
    ax.set_xlabel("Forbes list year left out of the estimation")
    ax.set_ylabel("Inherited minus self-made\n(pp of TFP growth per log point)")
    ax.set_ylim(-2.75, 0.3)
    ax.legend(loc="lower center", ncol=2, fontsize=9)
    ax.set_title("Leaving out the 2019 or the 2020 list pulls the three-year estimate toward zero;\n"
                 "no single list year moves the five-year estimate much")
    fig.tight_layout()
    save(fig, "fig4_list_years", SOURCE + " Country fixed effects, sector-group shares controlled, 95% intervals clustered by country.")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig1_sector_inheritance()
    fig2_observed_vs_predicted()
    fig3_with_without()
    fig4_list_years()


if __name__ == "__main__":
    main()
