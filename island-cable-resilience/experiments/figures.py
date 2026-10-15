"""Every figure in the article, from the saved results."""
import sys

from common import *  # noqa: F401,F403

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
LAND, SEA = "#ebeae4", SURFACE
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200,
})
CREDIT = "Figure by the author. Cable data: TeleGeography Submarine Cable Map (CC BY-NC-SA 3.0); land: Natural Earth."


def finish(fig, name, y=0.005):
    fig.text(0.995, y, CREDIT, ha="right", va="bottom", fontsize=7, color=MUTED)
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


FLOOR = 1e-6   # hours per year shown as "below this"


def fig_map():
    from cableresilience.data import load
    net = load(DATA, extra=("new-cam-ring", "nuvem"))
    polys = net.meta["polygons"]
    planned = {"new-cam-ring", "nuvem"}
    views = [("The Portuguese Atlantic cables in the map data", (-32.0, -8.0, 27.0, 42.0)),
             ("São Miguel", (-26.05, -25.05, 37.55, 38.05)),
             ("Madeira", (-17.35, -16.55, 32.5, 32.95))]
    fig = plt.figure(figsize=(11, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1, 1], wspace=0.08, hspace=0.18)
    axes = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
    for ax, (title, (x0, x1, y0, y1)) in zip(axes, views):
        lat_mid = (y0 + y1) / 2
        for poly in polys:
            bx0, by0, bx1, by1 = poly.bounds
            if bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1:
                continue
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, color=LAND, lw=0.4, ec=AXIS, zorder=1)
        for e in net.edges:
            lat, lng = e.coords[:, 0], e.coords[:, 1]
            if lng.max() < x0 - 2 or lng.min() > x1 + 2 or lat.max() < y0 - 2 or lat.min() > y1 + 2:
                continue
            col = C[1] if e.cable in planned else C[0]
            ax.plot(lng, lat, color=col, lw=1.6 if ax is axes[0] else 2.0, alpha=0.9, zorder=2,
                    solid_capstyle="round")
        for lid, lp in net.landing_points.items():
            if x0 <= lp["lng"] <= x1 and y0 <= lp["lat"] <= y1:
                ax.plot(lp["lng"], lp["lat"], "o", ms=5 if ax is axes[0] else 7, color=INK, mec=SURFACE,
                        mew=1.5, zorder=4)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect(1 / np.cos(np.radians(lat_mid)))
        ax.set_title(title, fontsize=10.5)
        ax.grid(False)
        ax.tick_params(labelsize=8)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(AXIS)
    labels0 = {"Ponta Delgada, Portugal": (6, 10), "Funchal, Portugal": (-8, -14), "Carcavelos, Portugal": (-10, 8),
               "Faial, Portugal": (6, 10)}
    for name, off in labels0.items():
        lp = next((v for v in net.landing_points.values() if v["name"] == name), None)
        if lp:
            axes[0].annotate(name.split(",")[0], (lp["lng"], lp["lat"]), xytext=off, textcoords="offset points",
                             fontsize=8, color=INK2, ha="left" if off[0] > 0 else "right")
    # zoom panels: distance rings around the main landing
    for ax, name in ((axes[1], "Ponta Delgada, Portugal"), (axes[2], "Funchal, Portugal")):
        lp = next(v for v in net.landing_points.values() if v["name"] == name)
        for r_km, ls in ((10, "-"),):
            dlat = r_km / 111.0
            dlng = r_km / (111.0 * np.cos(np.radians(lp["lat"])))
            t = np.linspace(0, 2 * np.pi, 200)
            ax.plot(lp["lng"] + dlng * np.cos(t), lp["lat"] + dlat * np.sin(t), color=INK2, lw=1, ls=ls, zorder=3)
        ax.annotate("10 km", (lp["lng"] + dlng * 0.71, lp["lat"] - dlat * 0.71), xytext=(4, -10),
                    textcoords="offset points", fontsize=8, color=INK2)
    for ax, names in ((axes[1], ["Ponta Delgada, Portugal"]), (axes[2], ["Funchal, Portugal", "Machico, Portugal"])):
        for name in names:
            lp = next(v for v in net.landing_points.values() if v["name"] == name)
            ax.annotate(name.split(",")[0], (lp["lng"], lp["lat"]), xytext=(8, 6), textcoords="offset points",
                        fontsize=8.5, color=INK2)
    handles = [Line2D([], [], color=C[0], lw=2, label="In service"),
               Line2D([], [], color=C[1], lw=2, label="Planned (New CAM Ring, Nuvem)"),
               Line2D([], [], marker="o", color=INK, lw=0, ms=6, mec=SURFACE, label="Landing point")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.45, 0.0))
    fig.subplots_adjust(bottom=0.09)
    finish(fig, "fig1_portugal_map.png", y=-0.01)


def fig_k_vs_n():
    d = load_json("exp1_structure.json")
    rows = [r for r in d["rows"] if r["k"] >= 2]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3), sharey=True)
    cap = 10
    for ax, r_km in zip(axes, (1, 10, 50)):
        cnt = {}
        for r in rows:
            k = min(r["k"], cap)
            n = min(r[f"N_{r_km}km"], cap)
            cnt[(k, n)] = cnt.get((k, n), 0) + 1
        ks = np.array([k for k, _ in cnt]); ns = np.array([n for _, n in cnt]); cs = np.array(list(cnt.values()))
        ax.plot([0.5, cap + 0.5], [0.5, cap + 0.5], color=AXIS, lw=1, zorder=1)
        ax.scatter(ks, ns, s=30 * cs ** 0.7, color=C[0], alpha=0.9, edgecolor=SURFACE, lw=1.2, zorder=3)
        for k, n, c in zip(ks, ns, cs):
            if c >= 10:
                ax.annotate(str(c), (k, n), ha="center", va="center", fontsize=7.5, color="white", zorder=4,
                            fontweight="bold")
        ax.set_xlim(1.4, cap + 0.6)
        ax.set_ylim(0.4, cap + 0.6)
        ax.set_xticks(range(2, cap + 1))
        ax.set_xticklabels([str(i) if i < cap else f"{cap}+" for i in range(2, cap + 1)])
        ax.set_yticks(range(1, cap + 1))
        ax.set_yticklabels([str(i) if i < cap else f"{cap}+" for i in range(1, cap + 1)])
        ax.set_title(f"Incidents of radius {r_km} km")
        ax.set_xlabel("Cable systems landing on the island")
    axes[0].set_ylabel("Fewest incidents that isolate it")
    fig.suptitle("Each bubble counts islands; 176 islands with two or more cable systems", x=0.01, y=1.02,
                 ha="left", fontsize=9.5, color=INK2)
    axes[2].annotate("islands where the two\nnumbers agree lie on\nthe diagonal", (7.3, 7.0), xytext=(5.6, 9.0),
                     fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    finish(fig, "fig2_cables_vs_incidents.png", y=-0.04)


def fig_by_k():
    d = load_json("exp2_risk.json")
    rows = d["all_islands"]
    groups = [("2", lambda k: k == 2), ("3", lambda k: k == 3), ("4", lambda k: k == 4),
              ("5 to 7", lambda k: 5 <= k <= 7), ("8 or more", lambda k: k >= 8)]
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    floor_y = FLOOR / 4
    ax.axhspan(FLOOR / 60, FLOOR, color=GRID, alpha=0.6, lw=0, zorder=0)
    for gi, (name, f) in enumerate(groups):
        sel = [r for r in rows if f(r["k"])]
        for j, (key, col, dx) in enumerate((("independent_h", C[0], -0.17), ("incident_h", C[1], 0.17))):
            v = np.array([r[key] for r in sel])
            shown = np.where(v >= FLOOR, v, floor_y * np.exp(rng.uniform(-0.6, 0.6, len(v))))
            x = gi + dx + rng.uniform(-0.09, 0.09, len(v))
            ax.scatter(x, shown, s=16, color=col, alpha=0.75, edgecolor="none", zorder=3)
            med = np.median(v)
            ax.plot([gi + dx - 0.14, gi + dx + 0.14], [max(med, floor_y)] * 2, color=INK, lw=2.2, zorder=4)
        ax.annotate(f"{len(sel)} islands", (gi, 60), ha="center", fontsize=8, color=MUTED)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR / 60, 150)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_xlabel("Cable systems landing on the island")
    ax.set_ylabel("Expected hours a year with no cable\nto a continent (log scale)")
    ax.grid(axis="x", visible=False)
    ax.text(-0.42, FLOOR / 30, "Below 10\u207b\u2076 hours a year, or (independent faults) no cut of four or fewer segments",
            ha="left", va="center", fontsize=8, color=INK2)
    ax.set_title("The same faults, counted two ways: more cables help only if they fail separately")
    handles = [Line2D([], [], marker="o", color=C[0], lw=0, ms=6, label="Faults independent"),
               Line2D([], [], marker="o", color=C[1], lw=0, ms=6, label="Faults arriving as incidents"),
               Line2D([], [], color=INK, lw=2.2, label="Median")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.27), ncol=3)
    finish(fig, "fig3_by_cable_count.png", y=-0.1)


def fig_sensitivity():
    d = load_json("exp2_risk.json")
    pick = ["São Miguel (Azores)", "Madeira", "Malta", "Maldives (Malé)", "Bermuda", "Cyprus"]
    rows = [r for l in pick for r in d["cases"] if r["label"] == l]
    settings = [k for k in rows[0] if isinstance(rows[0][k], dict) and "incident" in rows[0][k]]
    nice = {"baseline": "Baseline", "no regional incidents": "No regional incidents",
            "regional share 2%": "Regional share 2% (baseline 10%)",
            "regional share 25%": "Regional share 25%",
            "regional radius 5 km": "Regional median radius 5 km (baseline 15)",
            "regional radius 40 km": "Regional median radius 40 km",
            "fault rate x3": "Fault rate three times the global average",
            "repairs start after 10 days": "Median wait for a repair ship 10 days (baseline 21)",
            "repairs start after 50 days": "Median wait for a repair ship 50 days"}
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ys = np.arange(len(settings))[::-1]
    zero_x = 6e-3
    marks = ["o", "s", "^", "D", "v", "P"]
    for i, (r, mk, col) in enumerate(zip(rows, marks, C + [INK2])):
        for y, s_ in zip(ys, settings):
            v = r[s_]["incident"]["hours_per_year"]
            x = v if v >= zero_x * 2 else zero_x
            ax.plot([x], [y + (i - 2.5) * 0.09], marker=mk, ms=7, color=col, mec=SURFACE, mew=1, lw=0,
                    label=r["label"] if y == ys[0] else None, zorder=3)
    ax.axvspan(zero_x / 1.6, zero_x * 1.6, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.text(zero_x, ys[0] + 0.75, "none\nsampled", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(zero_x / 1.8, 100)
    ax.set_xticks([1e-2, 1e-1, 1, 10, 100])
    ax.set_ylim(-0.6, len(settings) - 0.1 + 0.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([nice[s_] for s_ in settings], fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Expected hours a year isolated, incident model (log scale)")
    ax.set_title("How the incident model responds to its assumptions")
    ax.legend(loc="lower center", bbox_to_anchor=(0.35, -0.3), ncol=3, fontsize=8.5)
    finish(fig, "fig4_sensitivity.png", y=-0.12)


if __name__ == "__main__":
    for w in sys.argv[1:] or ["map", "k_vs_n", "by_k", "sensitivity"]:
        globals()[f"fig_{w}"]()
