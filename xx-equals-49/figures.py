"""Figures for the x**x = 49 article. Run after solve_xx.py is importable."""
from __future__ import annotations

import contextlib
import io
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

import solve_xx as s

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3de"
SURFACE = "#ffffff"
# categorical slots, fixed order; colour follows the method across figures
COL = {
    "bisection": "#2a78d6",
    "fixed": "#eb6834",
    "newton_raw": "#1baf7a",
    "newton_log": "#eda100",
    "halley": "#e87ba4",
}

family = "Inter" if any("Inter" == f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({
    "font.family": family,
    "font.size": 12,
    "mathtext.fontset": "dejavusans",
    "axes.edgecolor": GRID,
    "axes.linewidth": 1.0,
    "axes.labelcolor": INK2,
    "axes.titlesize": 12.5,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "grid.color": GRID,
    "grid.linewidth": 1.0,
    "axes.grid": True,
    "axes.axisbelow": True,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def dot(ax, x, y, color, size=8, zorder=5):
    ax.plot([x], [y], "o", ms=size, mfc=color, mec=SURFACE, mew=2, zorder=zorder)


def footer(fig, text="Computed with solve_xx.py · figure by the author"):
    fig.text(0.012, 0.012, text, fontsize=9, color=MUTED, ha="left", va="bottom")


# ------------------------------------------------------------------ figure 1
def fig_shape():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.7), gridspec_kw={"wspace": 0.24})
    c = COL["bisection"]

    x = np.linspace(1e-4, 1.25, 800)
    a1.plot(x, x ** x, color=c)
    a1.axhline(0.8, color=MUTED, lw=1)
    xm, ym = 1 / math.e, math.exp(-1 / math.e)
    lo, hi = 0.09464971086492492, 0.7395336500107104
    for xr in (lo, hi):
        dot(a1, xr, 0.8, c)
    dot(a1, xm, ym, INK2, size=7)
    a1.annotate("minimum at $x = 1/e$\nvalue $e^{-1/e} \\approx 0.692$", (xm, ym), xytext=(0.43, 0.652),
                fontsize=10.5, color=INK2, ha="left", va="center")
    a1.annotate("$x \\approx 0.0946$", (lo, 0.8), xytext=(lo + 0.02, 0.822), fontsize=10.5, color=INK)
    a1.annotate("$x \\approx 0.7395$", (hi, 0.8), xytext=(hi - 0.02, 0.822), fontsize=10.5, color=INK, ha="right")
    a1.text(1.25, 0.806, "$k = 0.8$", fontsize=10.5, color=INK2, ha="right", va="bottom")
    a1.set_xlim(0, 1.25)
    a1.set_ylim(0.6, 1.1)
    a1.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    a1.set_xlabel("$x$")
    a1.set_title("Below 1: the curve dips, and a level is hit twice")

    x = np.linspace(1e-4, 4.2, 900)
    a2.plot(x, x ** x, color=c)
    a2.set_yscale("log")
    a2.axhline(49, color=MUTED, lw=1)
    for xb, lab, off in ((3, "$3^3 = 27$", (0.12, 0.62)), (4, "$4^4 = 256$", (-0.12, 1.25))):
        a2.plot([xb], [xb ** xb], "o", ms=7, mfc=SURFACE, mec=INK2, mew=1.5, zorder=4)
        a2.text(xb + off[0], xb ** xb * off[1], lab, fontsize=10.5, color=INK2,
                ha="left" if off[0] > 0 else "right", va="center")
    dot(a2, s.X_STAR, 49, c)
    a2.annotate("$x \\approx 3.2780$", (s.X_STAR, 49), xytext=(s.X_STAR - 0.1, 64), fontsize=10.5, color=INK, ha="right")
    a2.text(0.05, 56, "$k = 49$", fontsize=10.5, color=INK2, ha="left", va="bottom")
    a2.set_xlim(0, 4.2)
    a2.set_ylim(0.5, 600)
    a2.set_yticks([1, 10, 100])
    a2.set_yticklabels(["1", "10", "100"])
    a2.set_xlabel("$x$")
    a2.set_title("Above 1: one crossing (log scale)")

    fig.suptitle("The function $x^x$ on the positive axis", x=0.012, y=0.985, ha="left", fontsize=14.5,
                 fontweight="semibold", color=INK)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.83, bottom=0.17)
    footer(fig)
    fig.savefig("fig1_shape_of_xx.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ figure 2
def fig_convergence(out):
    def errs(h):
        return [max(s.err(v), 1e-18) for v in h if math.isfinite(v)]

    series = [
        ("bisection", "Bisection on [3, 4]", errs(out["Bisection"])),
        ("fixed", "Fixed point  $x \\leftarrow \\ln 49 / \\ln x$", errs(out["Fixed point x = ln49/ln x"])),
        ("newton_raw", "Newton on  $x^x - 49$  (raw form)", errs(out["Newton on x^x - 49"])),
        ("newton_log", "Newton on  $x \\ln x - \\ln 49$  (log form)", errs(out["Newton on x ln x - ln 49"])),
        ("halley", "Halley on  $w e^w - \\ln 49$  (Lambert W)", errs(out["Halley on w e^w (Lambert W)"])),
    ]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5.2), gridspec_kw={"wspace": 0.16, "width_ratios": [1.1, 1]})
    half_ulp = math.ulp(s.X_STAR) / 2

    for ax in (a1, a2):
        ax.set_yscale("log")
        ax.set_ylim(1e-17, 10)
        ax.set_yticks([1e0, 1e-4, 1e-8, 1e-12, 1e-16])
        ax.axhspan(1e-17, half_ulp, color="#f1f0ec", zorder=0)
        ax.set_xlabel("iteration")

    handles = []
    for key, lab, e in series:
        (ln,) = a1.plot(range(len(e)), e, color=COL[key], label=lab)
        handles.append(ln)
    a1.set_xlim(0, 62)
    a1.set_ylabel("absolute error in $x$")
    a1.set_title("All five, first 60 iterations")
    a1.text(11, 3.2e-17, "below double-precision resolution", fontsize=9.5, color=INK2,
            ha="left", va="center")
    eb = series[0][2]
    a1.annotate("one bit\nper step", (30, eb[30]), xytext=(33, eb[30] * 3e2), fontsize=10.5, color=INK, ha="left", va="center")
    ef = series[1][2]
    a1.annotate("error shrinks by\n0.842 per step", (44, ef[44]), xytext=(42, ef[44] * 40), fontsize=10.5,
                color=INK, ha="center", va="bottom")

    for key, lab, e in series[2:]:
        a2.plot(range(len(e)), e, color=COL[key], marker="o", ms=7, mec=SURFACE, mew=1.6)
    a2.set_xlim(-0.2, 8.6)
    a2.set_xticks(range(0, 9))
    a2.set_yticklabels([])
    a2.set_title("The three fast ones, enlarged")
    a2.text(5.3, 3e-5, "Newton,\nraw form", fontsize=10.5, color=INK, ha="left", va="center")
    a2.text(4.42, 1.2e-11, "Newton,\nlog form", fontsize=10.5, color=INK, ha="center", va="center")
    a2.text(2.42, 1.2e-11, "Halley\n(Lambert W)", fontsize=10.5, color=INK, ha="right", va="center")

    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.006, 0.925), ncol=3, frameon=False,
               fontsize=10.5, handlelength=1.6, columnspacing=1.6, labelcolor=INK2)
    fig.suptitle("Error against a 60-digit reference, by iteration", x=0.012, y=0.985, ha="left", fontsize=14.5,
                 fontweight="semibold", color=INK)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.74, bottom=0.16)
    footer(fig)
    fig.savefig("fig4_convergence.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ figure 3
def fig_newton_from_one(out):
    raw = out["Newton from 1 on x^x - 49"]
    log = out["Newton from 1 on x ln x - ln 49"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.axhline(s.X_STAR, color=MUTED, lw=1)
    ax.plot(range(len(raw)), raw, color=COL["newton_raw"], label="Newton on  $x^x - 49$  (raw form)")
    ax.plot(range(len(log)), log, color=COL["newton_log"], label="Newton on  $x \\ln x - \\ln 49$  (log form)")
    dot(ax, len(raw) - 1, raw[-1], COL["newton_raw"])
    dot(ax, len(log) - 1, log[-1], COL["newton_log"])
    ax.annotate("first step lands on $x = 49$", (1, 49), xytext=(6, 49), fontsize=10.5, color=INK, va="center")
    ax.annotate("then about 0.2 per step,\nfor the next 190 steps", (90, raw[90]), xytext=(96, raw[90] + 5.5),
                fontsize=10.5, color=INK, ha="left", va="center")
    ax.annotate(f"converged after {len(raw) - 1} iterations", (len(raw) - 1, raw[-1]), xytext=(203, 21),
                fontsize=10.5, color=INK, ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1, shrinkA=2, shrinkB=6))
    ax.annotate(f"converged after {len(log) - 1} iterations", (len(log) - 1, log[-1]), xytext=(17, 14.5),
                fontsize=10.5, color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1, shrinkA=2, shrinkB=6))
    ax.text(203, s.X_STAR + 1.2, "root", fontsize=10.5, color=INK2, ha="right", va="bottom")
    ax.set_xlim(-2, 204)
    ax.set_ylim(0, 54)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel("iteration")
    ax.set_ylabel("current iterate $x_n$")
    ax.legend(loc="upper right", frameon=False, fontsize=10.5, handlelength=1.6, labelcolor=INK2)
    fig.suptitle("Newton's method started at $x_0 = 1$, on two forms of the same equation", x=0.012, y=0.985,
                 ha="left", fontsize=14.5, fontweight="semibold", color=INK)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.88, bottom=0.17)
    footer(fig)
    fig.savefig("fig2_newton_from_one.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ figure 4
def fig_bifurcation():
    import xx_theory as th

    ee = math.e ** math.e
    k_lo = np.linspace(2.0, ee, 300)
    r_lo = np.exp(np.real(th.lambertw(np.log(k_lo))))
    k_hi = np.linspace(ee, 60.0, 300)
    r_hi = np.exp(np.real(th.lambertw(np.log(k_hi))))
    t = np.exp(np.linspace(math.log(0.03), math.log(4000.0), 4000))
    a = (1 + 1 / t) ** t
    b = (1 + 1 / t) ** (t + 1)
    kk = a ** b
    keep = kk <= 60.0
    a, b, kk = a[keep], b[keep], kk[keep]

    fig, ax = plt.subplots(figsize=(10, 5.9))
    c_fix, c_cyc = COL["bisection"], COL["fixed"]
    ax.axvline(ee, color=MUTED, lw=1)
    ax.plot(k_lo, r_lo, color=c_fix, label="fixed point $r(k)$, attracting")
    ax.plot(k_hi, r_hi, color=c_fix, ls=(0, (1.2, 2.2)), label="fixed point $r(k)$, repelling")
    (l2,) = ax.plot(kk, a, color=c_cyc, label="two-cycle $(a, b)$ with $a^b = b^a = k$")
    ax.plot(kk, b, color=c_cyc)
    ks = np.array([4.0, 8.0, 12.0, 16.0, 22.0, 30.0, 40.0, 49.0, 58.0])
    lows, highs, _ = th.bifurcation_data(ks)
    for kv, lo_, hi_ in zip(ks, lows, highs):
        ax.plot([kv, kv], [lo_, hi_], "o", ms=6.5, mfc=INK, mec=SURFACE, mew=1.6, zorder=6, ls="none")
    ax.plot([], [], "o", ms=6.5, mfc=INK, mec=SURFACE, mew=1.6, ls="none", label="where the iteration actually settles")
    ax.set_yscale("log")
    ax.set_ylim(0.9, 60)
    ax.set_yticks([1, 2, 4, 10, 30])
    ax.set_yticklabels(["1", "2", "4", "10", "30"])
    ax.set_xlim(2, 60)
    ax.set_xticks([10, 20, 30, 40, 50, 60])
    ax.set_xlabel("$k$")
    ax.set_ylabel("value of $x$")
    ax.text(ee + 0.6, 1.0, "$k = e^e \\approx 15.15$", fontsize=10.5, color=INK2, ha="left", va="center")
    ax.text(14.5, 4.0, "the cycle $(2,\\ 4)$ at $k = 16$", fontsize=10.5, color=INK, ha="right", va="center")
    ax.annotate("30.93", (49, 30.927527), xytext=(49, 39), fontsize=10.5, color=INK, ha="center", va="bottom")
    ax.annotate("1.134", (49, 1.134097), xytext=(49, 1.3), fontsize=10.5, color=INK, ha="center", va="bottom")
    ax.annotate("$r = 3.278$ at $k = 49$", (49, s.X_STAR), xytext=(49, 4.1), fontsize=10.5, color=INK, ha="center",
                va="bottom")
    ax.plot([49], [s.X_STAR], "o", ms=6.5, mfc=SURFACE, mec=c_fix, mew=1.8, zorder=6)
    fig.legend(loc="upper left", bbox_to_anchor=(0.006, 0.925), ncol=2, frameon=False, fontsize=10.5,
               handlelength=2.2, columnspacing=1.8, labelcolor=INK2)
    fig.suptitle("Where the iteration $x \\leftarrow k^{1/x}$ ends up, as $k$ varies", x=0.012, y=0.985, ha="left",
                 fontsize=14.5, fontweight="semibold", color=INK)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.77, bottom=0.15)
    footer(fig, "Computed with xx_theory.py · figure by the author")
    fig.savefig("fig3_bifurcation.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ figure 5
def fig_complex():
    import mpmath as mp

    import xx_theory as th

    x = np.linspace(-4.0, 10.0, 1401)
    y = np.linspace(-9.6, 9.6, 1921)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    with np.errstate(divide="ignore", invalid="ignore"):
        G = Z * np.log(Z)
    cut = (X <= 0) & (np.abs(Y) < 0.03)
    re = np.where(cut, np.nan, G.real)
    im = np.where(cut, np.nan, G.imag) / (2 * math.pi)

    fig, ax = plt.subplots(figsize=(8.2, 9.6))
    ax.set_aspect("equal")
    ax.grid(False)
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.axvline(0, color=GRID, lw=1, zorder=0)
    ax.plot([-4, 0], [0, 0], color=MUTED, lw=2.4, solid_capstyle="butt", zorder=1)
    ax.contour(X, Y, im, levels=np.arange(-12, 13), colors=[COL["fixed"]], linewidths=1.5, linestyles="solid")
    ax.contour(X, Y, re, levels=[s.C], colors=[COL["bisection"]], linewidths=2.2)
    mp.mp.dps = 30
    sols = [complex(th.z_solution(m)) for m in range(-4, 5)]
    for z in sols:
        ax.plot([z.real], [z.imag], "o", ms=8.5, mfc=INK, mec=SURFACE, mew=2, zorder=6)
    lab = {0: "$z_0 = 3.278$", 1: "$z_1 = 3.672 + 2.644\\,i$", 2: "$z_2 = 4.273 + 4.811\\,i$",
           3: "$z_3 = 4.852 + 6.736\\,i$", 4: "$z_4 = 5.396 + 8.524\\,i$"}
    for m, text in lab.items():
        z = complex(th.z_solution(m))
        ax.annotate(text, (z.real, z.imag), xytext=(z.real + 0.4, z.imag + (0.5 if m else -0.75)), fontsize=10.5,
                    color=INK, ha="left", va="center",
                    bbox=dict(boxstyle="round,pad=0.18", fc=SURFACE, ec="none", alpha=0.9))
    ax.text(-1.05, -0.2, "branch cut", fontsize=10, color=INK2, ha="center", va="top")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COL["bisection"], lw=2.2, label="$|z^z| = 49$"),
               Line2D([], [], color=COL["fixed"], lw=1.5, label="$z^z$ real and positive"),
               Line2D([], [], color=INK, marker="o", ms=8.5, mec=SURFACE, mew=2, ls="none", label="solutions of $z^z = 49$")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.008, 0.955), ncol=3, frameon=False, fontsize=10.5,
               handlelength=1.6, columnspacing=1.6, labelcolor=INK2)
    ax.set_xlim(-4, 10)
    ax.set_ylim(-9.6, 9.6)
    ax.set_xlabel("Re $z$")
    ax.set_ylabel("Im $z$")
    ax.set_xticks([-4, -2, 0, 2, 4, 6, 8, 10])
    ax.set_yticks([-8, -4, 0, 4, 8])
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    fig.suptitle("The solutions of $z^z = 49$ in the complex plane", x=0.012, y=0.988, ha="left", fontsize=14.5,
                 fontweight="semibold", color=INK)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.9, bottom=0.085)
    footer(fig, "Computed with xx_theory.py · figure by the author")
    fig.savefig("fig5_complex_solutions.png", dpi=200)
    plt.close(fig)
    mp.mp.dps = 60


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        results = s.run()
    fig_shape()
    fig_convergence(results)
    fig_newton_from_one(results)
    fig_bifurcation()
    fig_complex()
    print("wrote fig1_shape_of_xx.png, fig4_convergence.png, fig2_newton_from_one.png,",
          "fig3_bifurcation.png, fig5_complex_solutions.png")
