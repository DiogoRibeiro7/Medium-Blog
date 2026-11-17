"""Figures for the article. Run after votes_seats.py, causal.py and counterfactual_2026.py."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
DEM = "#2a78d6"   # categorical slot 1 (blue)
REP = "#e34948"   # categorical slot 8 (red)
NEU = "#8a8984"   # neutral: neither party drew the map
CREDIT = "Diogo Ribeiro · data: MIT Election Lab, ICPSR 6311, Wikipedia (2026, unofficial)"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
})


def _credit(fig, y=0.005):
    fig.text(0.995, y, CREDIT, ha="right", va="bottom", fontsize=7.5, color=INK2)


def fig1_post_states():
    d = pd.read_csv(RES / "post_check_2026.csv")
    d["n2"] = d["seats"]
    d["sym_pct"] = 100 * d["expected_dem_seats_symmetric"] / d["n2"]
    d = d.sort_values("vote_pct")
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ys = np.arange(len(d))
    for y, r in zip(ys, d.itertuples()):
        ax.plot([r.seat_pct, r.sym_pct], [y, y], color=GRID, lw=6, solid_capstyle="round", zorder=1)
        ax.scatter(r.vote_pct, y, s=90, facecolor=SURFACE, edgecolor=INK, lw=1.6, zorder=3)
        ax.scatter(r.sym_pct, y, s=80, marker="D", color=NEU, edgecolor=SURFACE, lw=1.5, zorder=3)
        ax.scatter(r.seat_pct, y, s=110, color=DEM, edgecolor=SURFACE, lw=1.5, zorder=4)
        gap = r.dem_seats - r.expected_dem_seats_symmetric
        ax.text(101, y, f"{r.dem_seats} of {r.seats} seats  ({gap:+.1f})", va="center", fontsize=9, color=INK2)
    ax.set_yticks(ys, d["state_po"])
    ax.set_xlim(-2, 100)
    ax.set_xlabel("Per cent")
    ax.axvline(50, color=INK2, lw=0.8, ls=":")
    handles = [
        Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=INK, markersize=9,
               label="Democratic share of the two-party House vote"),
        Line2D([], [], marker="o", ls="", color=DEM, markersize=10, label="Democratic share of seats won"),
        Line2D([], [], marker="D", ls="", color=NEU, markersize=8,
               label="Seat share a symmetric (cube-law) map gives at that vote"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, 1.2), ncol=1, frameon=False,
              fontsize=9)
    ax.set_title("The eight states in the post, 2026: votes, seats and a fair benchmark", loc="left", pad=70)
    fig.subplots_adjust(left=0.07, right=0.8, top=0.78, bottom=0.11)
    _credit(fig)
    fig.savefig(FIG / "fig1_post_states_2026.png", dpi=200)
    plt.close(fig)


def fig2_every_election():
    panel = pd.read_csv(RES / "seat_bonus_state_year.csv")
    curves = pd.read_csv(RES / "curves_by_year.csv").set_index("year")
    rho = json.loads((RES / "summary_votes_seats.json").read_text())["rho_pooled"]
    years = sorted(panel.year.unique())
    fig, axes = plt.subplots(4, 7, figsize=(14, 8.6), sharex=True, sharey=True)
    xs = np.linspace(0.2, 0.85, 200)
    colors = {"R": REP, "D": DEM, "Court": NEU, "Commission": NEU, "Split": NEU}
    for ax, yr in zip(axes.flat, years):
        g = panel[(panel.year == yr) & (panel.seats >= 2) & panel.v.notna()]
        c = [colors.get(x, "#c3c2b7") for x in g["drawer"]]
        ax.scatter(g["v"], g["s"], s=4 + 1.6 * g["seats"], c=c, alpha=0.75, edgecolor=SURFACE, lw=0.5, zorder=3)
        ax.plot(xs, expit(rho * logit(xs)), color=INK2, lw=1, ls="--", zorder=2)
        lam, r = curves.loc[yr, "lam"], curves.loc[yr, "rho"]
        ax.plot(xs, expit(lam + r * logit(xs)), color=INK, lw=1.6, zorder=4)
        b = 100 * curves.loc[yr, "bias_at_50"]
        ax.set_title(f"{yr}   {50 + b:.0f}% at 50/50", fontsize=9, loc="left", fontweight="bold" if yr == 2026 else "normal")
        ax.set_xlim(0.2, 0.85)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xticks([0.3, 0.5, 0.7], ["30", "50", "70"])
        ax.set_yticks([0, 0.5, 1], ["0", "50", "100"])
        ax.tick_params(labelsize=8)
    fig.supxlabel("Democratic share of the two-party House vote in the state (%)", fontsize=10, color=INK2)
    fig.supylabel("Democratic share of the state's seats (%)", fontsize=10, color=INK2)
    handles = [
        Line2D([], [], color=INK, lw=1.6, label="Curve fitted to that year's states"),
        Line2D([], [], color=INK2, lw=1, ls="--", label="Symmetric cube-law curve (no bias)"),
        Line2D([], [], marker="o", ls="", color=REP, label="Map drawn under Republican control"),
        Line2D([], [], marker="o", ls="", color=DEM, label="Map drawn under Democratic control"),
        Line2D([], [], marker="o", ls="", color=NEU, label="Court, commission or split government"),
        Line2D([], [], marker="o", ls="", color="#c3c2b7", label="Small state, not coded"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Votes and seats by state in every House election, 1972-2026", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.07, hspace=0.38, wspace=0.08)
    _credit(fig)
    fig.savefig(FIG / "fig2_votes_seats_every_election.png", dpi=200)
    plt.close(fig)


def fig3_national_bias():
    n = pd.read_csv(RES / "national_bias.csv")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    col = np.where(n.bias_seats >= 0, DEM, REP)
    ax.bar(n.year, n.bias_seats, width=1.4, color=col, edgecolor=SURFACE, lw=1)
    ax.axhline(0, color=INK2, lw=0.8)
    for yr in [1972, 1982, 1992, 2002, 2012, 2022]:
        ax.axvline(yr - 1, color=GRID, lw=1, zorder=0)
    lab = n.set_index("year")
    for yr in [1990, 2012, 2024]:
        v = lab.loc[yr, "bias_seats"]
        ax.text(yr, v + (1.5 if v >= 0 else -1.5), f"{v:+.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9, color=INK2)
    ax.set_ylabel("Seats above or below 218 at a 50/50 national vote")
    ax.set_xticks([1972, 1982, 1992, 2002, 2012, 2022])
    ax.text(1972.5, 33, "Democratic advantage", color=DEM, fontsize=9.5)
    ax.text(1972.5, -24, "Republican advantage", color=REP, fontsize=9.5)
    ax.set_ylim(-28, 38)
    ax.set_title("National partisan bias of the House map, 1972-2024 (uniform swing)", loc="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _credit(fig)
    fig.savefig(FIG / "fig3_national_bias.png", dpi=200)
    plt.close(fig)


def fig4_bias_by_drawer():
    p = pd.read_csv(RES / "seat_bonus_state_year.csv")
    p = p[(p.seats >= 2) & (p.year <= 2024)].copy()
    p["grp"] = p.drawer.map({"R": "R", "D": "D", "Court": "N", "Commission": "N", "Split": "N"})
    p["bseats"] = p.bias_us * p.seats
    t = p.pivot_table(index="year", columns="grp", values="bseats", aggfunc="sum")
    # no coded plan of a group in force that year -> gap, not zero
    t = t.where(p.pivot_table(index="year", columns="grp", values="seats", aggfunc="count") > 0)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    spec = [("D", DEM, "Maps drawn under Democratic control"),
            ("N", NEU, "Court, commission or split government"),
            ("R", REP, "Maps drawn under Republican control")]
    for k, c, lab in spec:
        ax.plot(t.index, t[k], color=c, marker="o", markersize=5, label=lab)
        ax.text(2024.6, t[k].iloc[-1], lab.split(" under ")[-1].replace("Court, commission or split government", "neutral"),
                color=c, va="center", fontsize=9)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("Seats, summed over states")
    ax.set_xlim(1971, 2031)
    ax.set_xticks([1972, 1982, 1992, 2002, 2012, 2022])
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("Where the bias sits: each state's map judged at a tied statewide vote", loc="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _credit(fig)
    fig.savefig(FIG / "fig4_bias_by_drawer.png", dpi=200)
    plt.close(fig)


def fig5_event_study():
    a = pd.read_csv(RES / "event_study_coefs.csv")
    b = pd.read_csv(RES / "event_study_coefs_post1992.csv")
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for df, c, off, lab in [(a, INK2, -0.08, "Every redistricting, 1972-2024"),
                            (b, DEM, 0.08, "Redistrictings from 1992 on")]:
        df = pd.concat([df, pd.DataFrame([dict(k_rel=-1, coef=0, lo=0, hi=0)])]).sort_values("k_rel")
        x = df.k_rel + off
        ax.errorbar(x, df.coef, yerr=[df.coef - df.lo, df.hi - df.coef], fmt="o-", color=c,
                    ecolor=c, elinewidth=1.4, capsize=3, markersize=7, lw=1.6, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(-0.5, color=INK2, lw=0.8, ls=":")
    ax.text(-0.45, -4.3, "new map takes effect", fontsize=9, color=INK2)
    ax.set_xticks(range(-2, 5), ["-2", "-1\n(base)", "0", "+1", "+2", "+3", "+4"])
    ax.set_xlabel("Elections relative to the first election under the new map")
    ax.set_ylabel("Change in partisan bias (points of the delegation)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_ylim(-5, 14)
    ax.set_title("What one step of control toward a party does to the map's bias", loc="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _credit(fig)
    fig.savefig(FIG / "fig5_event_study.png", dpi=200)
    plt.close(fig)


def fig6_old_map_2026():
    d = pd.read_csv(RES / "old_map_projection_2026.csv")
    d["col"] = np.where(d.drawer_new == "R", REP, np.where(d.drawer_new == "D", DEM,
                        np.where(d.drawer_new == "Court", INK, NEU)))
    d = d.sort_values(["gap", "seats"], ascending=[True, False]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    ys = np.arange(len(d))[::-1]
    for y, r in zip(ys, d.itertuples()):
        ax.plot([0, r.gap], [y, y], color=GRID, lw=3, zorder=1)
        ax.scatter(r.gap, y, s=30 + 3 * r.seats, color=r.col, edgecolor=SURFACE, lw=1.2, zorder=3)
        ax.text(-3.6, y, f"{r.state_po} ({r.seats})", va="center", fontsize=8.5,
                color=INK if r.status == "redrawn" else INK2,
                fontweight="bold" if r.status == "redrawn" else "normal")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(-3.8, 2.8)
    ax.set_yticks([])
    ax.set_xlabel("Seats Democrats won in 2026 minus seats the 2024 map gives at the 2026 swing")
    handles = [Line2D([], [], marker="o", ls="", color=REP, markersize=8, label="New map, Republican control"),
               Line2D([], [], marker="o", ls="", color=DEM, markersize=8, label="New map, Democratic control (CA)"),
               Line2D([], [], marker="o", ls="", color=INK, markersize=8, label="New map, court (UT)"),
               Line2D([], [], marker="o", ls="", color=NEU, markersize=8, label="Same map as 2024")]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper right")
    ax.set_title("2026: did the new maps beat the old ones?", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _credit(fig)
    fig.savefig(FIG / "fig6_old_map_projection_2026.png", dpi=200)
    plt.close(fig)


def fig7_symmetry_2024():
    s = pd.read_csv(RES / "national_symmetry.csv")
    s = s[s.year == 2024]
    fig, ax = plt.subplots(figsize=(8.4, 5))
    x = 100 * s.dem_vote
    ax.plot(x, s.dem_seats, color=DEM, label="Democratic seats when Democrats get this share")
    ax.plot(x, s.rep_seats_if_rep_had_this_vote, color=REP,
            label="Republican seats when Republicans get this share")
    ax.axhline(217.5, color=INK2, lw=0.8, ls=":")
    ax.axvline(50, color=INK2, lw=0.8, ls=":")
    v26 = 100 * 59825068 / (59825068 + 46629496)
    ax.axvline(v26, color=INK, lw=1, ls="--")
    ax.text(v26 + 0.2, 160, f"2026 two-party\nDemocratic share,\npartial count ({v26:.1f}%)", fontsize=8.5, color=INK2)
    for xv in [50, 56]:
        r = s[np.isclose(s.dem_vote, xv / 100)].iloc[0]
        ax.annotate(f"{int(r.dem_seats)}", (xv, r.dem_seats), xytext=(-26, 6), textcoords="offset points",
                    color=DEM, fontsize=9)
        ax.annotate(f"{int(r.rep_seats_if_rep_had_this_vote)}", (xv, r.rep_seats_if_rep_had_this_vote),
                    xytext=(4, -14 if xv == 50 else 6), textcoords="offset points", color=REP, fontsize=9)
    ax.set_xlabel("Party's share of the two-party national House vote (%)")
    ax.set_ylabel("Seats (of 435)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("The 2024 map and its mirror: equal votes, unequal seats away from 50/50", loc="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _credit(fig)
    fig.savefig(FIG / "fig7_symmetry_2024.png", dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig7_symmetry_2024()
    fig1_post_states()
    fig2_every_election()
    fig3_national_bias()
    fig4_bias_by_drawer()
    fig5_event_study()
    fig6_old_map_2026()


if __name__ == "__main__":
    main()
