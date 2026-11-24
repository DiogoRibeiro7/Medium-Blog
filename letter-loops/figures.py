"""The five figures of the article, from results/ (run letters_numbers.py first).

python figures.py

Matplotlib 3.11 shapes Hebrew and Arabic-script text and writes it right to left by itself; each
right-to-left run is wrapped in Unicode isolates so that counts beside it stay in place. The glyphs
come from Noto Sans Arabic and Noto Sans Hebrew (Ubuntu: fonts-noto-core).
"""

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle  # noqa: E402

import letters_core as L  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9890", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RTL = {"Hebrew", "Arabic"}
CHOICES = [("standard", "text"), ("folded", "text"), ("standard", "dictionary"), ("folded", "dictionary")]
CHOICE_LABELS = [
    "Standard letters,\nrunning text\n(the post's choice)",
    "Accents and final\nforms folded,\nrunning text",
    "Standard letters,\ndictionary\n(each word once)",
    "Accents and final\nforms folded,\ndictionary",
]

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans", "Noto Sans Arabic", "Noto Sans Hebrew"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.edgecolor": MUTED,
    }
)


def visual(text, code):
    """Text in a right-to-left script wrapped in a first-strong isolate, so that it is laid out on its own."""
    return f"\u2068{text}\u2069" if L.LANGUAGES[code][2] in RTL else text


def load():
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    with open(RESULTS / "counts.pkl", "rb") as fh:
        counts = pickle.load(fh)
    return summary, counts


def describe(code, ch, boundary):
    """Attractors of one chain as short text, largest basin first; sizes shown when there are several."""
    sym, f, basin = ch["sym"], np.array(ch["f"]), np.array(ch["basin"])
    loops = [tuple(x) for x in ch["loops"]]
    att, ends = [], []
    sizes = np.bincount(basin)
    for b in np.argsort(-sizes, kind="stable"):
        members = np.flatnonzero(basin == b)
        lp = next((x for x in loops if x[0] in members), None)
        if lp is None:  # a dead end
            ends.append(visual(sym[next(i for i in members if f[i] < 0)], code))
        elif len(lp) == 1:
            att.append((visual(sym[lp[0]] * 2, code), sizes[b]))
        elif boundary and lp[0] == 0:
            att.append(("“" + visual("".join(sym[i] for i in lp[1:]), code) + "”", sizes[b]))
        else:
            att.append((visual("".join(sym[i] for i in lp), code), sizes[b]))
    parts = [t for t, _ in att] if len(att) + len(ends) == 1 else [f"{t} ({n})" for t, n in att]
    if ends:
        parts.append(" ".join(ends) + (" (dead ends)" if len(ends) > 1 else " (dead end)"))
    return "  ·  ".join(parts)


# ---------------------------------------------------------------------------------------------------


def radial_positions(f, depth, loop):
    """Loop on an inner circle; every tree hanging off it fanned out by depth.

    Each loop letter gets a slice of the circle in proportion to the letters that drain into it,
    with a floor so that loop letters never sit on top of each other.
    """
    n = len(f)
    children = {v: [] for v in range(n)}
    for u in range(n):
        if depth[u] > 0:
            children[f[u]].append(u)

    def leaves(v):
        return 1 if not children[v] else sum(leaves(c) for c in children[v])

    raw = [sum(leaves(c) for c in children[v]) for v in loop]
    floor = 0.12 * sum(raw) + 0.5
    weights = [max(w, floor) for w in raw]
    total = sum(weights)
    r0, dr = (0.55, 1.0) if len(loop) == 2 else (1.0, 1.15)
    pos, start = {}, np.pi / 2

    def place(v, a0, a1, d):
        kids = sorted(children[v], key=leaves, reverse=True)
        tot = sum(leaves(c) for c in kids) or 1
        a = a0
        for c in kids:
            w = (a1 - a0) * leaves(c) / tot
            ang = a + w / 2
            pos[c] = ((r0 + d * dr) * np.cos(ang), (r0 + d * dr) * np.sin(ang))
            place(c, a, a + w, d + 1)
            a += w

    for v, w in zip(loop, weights, strict=True):
        span = 2 * np.pi * w / total
        mid = start - span / 2
        pos[v] = (r0 * np.cos(mid), r0 * np.sin(mid))
        place(v, start - span, start, 1)
        start -= span
    return pos


def draw_map(ax, sym, f, depth, loop, probs, title):
    pos = radial_positions(f, depth, loop)
    rad = 0.17
    centre = np.mean([pos[v] for v in loop], axis=0)
    for u in range(len(sym)):
        v = int(f[u])
        (x0, y0), (x1, y1) = pos[u], pos[v]
        in_loop = u in loop
        sign = 1.0
        if in_loop:
            chord = np.hypot(x1 - x0, y1 - y0)
            nx, ny = (y1 - y0) / chord, -(x1 - x0) / chord  # side the arc bulges to when rad > 0
            mid = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            if len(loop) > 2 and np.dot(mid - centre, (nx, ny)) < 0:
                sign = -1.0  # bend every arc of a longer loop outwards
        arrow = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=9,
            lw=1.6 if in_loop else 0.9,
            color=BLUE if in_loop else MUTED,
            shrinkA=10,
            shrinkB=10,
            connectionstyle=f"arc3,rad={0.35 * sign}" if in_loop else "arc3,rad=0",
            zorder=1,
        )
        ax.add_patch(arrow)
        if in_loop:
            off = sign * (0.175 * chord + 0.22)
            ax.text(mid[0] + off * nx, mid[1] + off * ny, f"{probs[u]:.0%}", ha="center", va="center", fontsize=8,
                    color=INK2)  # fmt: skip
    for u, (x, y) in pos.items():
        in_loop = u in loop
        ax.add_patch(Circle((x, y), rad, facecolor=BLUE if in_loop else SURFACE, edgecolor=BLUE if in_loop else MUTED,
                            lw=1.2, zorder=2))  # fmt: skip
        label = "␣" if sym[u] == L.BOUNDARY else sym[u]
        ax.text(x, y - 0.005, label, ha="center", va="center", zorder=3, fontsize=10,
                color="white" if in_loop else INK, fontweight="bold" if in_loop else "normal")  # fmt: skip
    r = max(np.hypot(*p) for p in pos.values()) + 0.35
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, color=INK, loc="left")


def fig1(summary, counts):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4))
    for ax, boundary, title in (
        (axes[0], False, "Letter pairs inside words, as in Norvig's table:\nevery letter ends in the loop e → r → e"),
        (axes[1], True, "With the end of a word as a symbol (␣):\nevery letter ends in “the”"),
    ):
        sym, m, f, s, order = L.chain(counts["en"], "en", "standard", "text", boundary)
        p = m / m.sum(axis=1, keepdims=True)
        probs = {u: p[u, f[u]] for u in range(len(sym))}
        draw_map(ax, sym, f, s.depth, s.loops[0], probs, title)
    note = (
        "Each arrow points to the letter most likely to follow. Percentages: share of the letter's "
        "successors on the loop. English word frequencies from wordfreq 3.1.1."
    )
    fig.text(0.01, 0.015, note, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(FIGURES / "fig1_english_maps.png", dpi=200)
    plt.close(fig)


def atlas(summary, boundary, path, title, extra=None):
    codes = list(L.LANGUAGES)
    n = len(codes)
    fig = plt.figure(figsize=(12.5, 0.33 * n + 2.0))
    ax = fig.add_axes((0.01, 0.01, 0.98, 0.98))
    ax.set_xlim(0, 100)
    ax.set_ylim(n + 3.6, -0.6)
    ax.axis("off")
    x_name, x_cols, width = 0.5, [17, 38, 59, 80], 20.5
    ax.text(0.5, -0.2, title, fontsize=12, va="top", color=INK)
    for x, lab in zip(x_cols, CHOICE_LABELS, strict=True):
        ax.text(x + 0.3, 2.7, lab, fontsize=8.5, color=INK2, va="bottom")
    prev_script = None
    for r, code in enumerate(codes):
        y = r + 3.3
        name, _, script = L.LANGUAGES[code]
        if script != prev_script:
            ax.plot([0.5, 100], [y - 0.5, y - 0.5], color=GRID, lw=0.8)
            prev_script = script
        ax.text(x_name, y, name, fontsize=9, va="center", color=INK)
        base = None
        for k, (a, w) in enumerate(CHOICES):
            ch = summary["chains"][f"{code}|{a}|{w}|{int(boundary)}"]
            text = describe(code, ch, boundary)
            if k == 0:
                base = text
                if extra:
                    text = text.replace("”", "”" + extra(code), 1)
            elif text != base:
                ax.add_patch(Rectangle((x_cols[k] - 0.2, y - 0.46), width, 0.92, facecolor=ORANGE, alpha=0.14, lw=0))
            ax.text(x_cols[k] + 0.3, y, text, fontsize=8.5, va="center", color=INK)
    fig.savefig(FIGURES / path, dpi=200)
    plt.close(fig)


def fig2(summary):
    atlas(
        summary,
        False,
        "fig2_atlas_inside_words.png",
        "Where every letter ends, counting letter pairs inside words only. Shaded: different from the first column.\n"
        "A loop is written as the letters it visits (er = e → r → e); a doubled letter (ll) points at itself; "
        "(n) = letters ending there.",
    )


def fig4(summary):
    words = summary["words"]
    atlas(
        summary,
        True,
        "fig4_atlas_word_boundary.png",
        "Where every letter ends once the end of a word is a symbol. Quoted: a loop through the word boundary, "
        "read as a word.\nShaded: different from the first column. In the first column, # is the word's rank "
        "by frequency in its language.",
        extra=lambda c: f" #{words[c]['rank']:,}" if words[c]["rank"] else "",
    )


def fig3(summary):
    sh = summary["shuffle"]
    codes = list(L.LANGUAGES)
    uniform = sum(sh[f"{c}|0"]["p_uniform"] for c in codes)
    shuffled = sum(sh[f"{c}|0"]["p_one"] for c in codes)
    noself = sum(sh[f"{c}|0"]["p_one_noself"] for c in codes)
    observed = sum(sh[f"{c}|0"]["observed"] == 1 for c in codes)
    rows = [
        ("Arrows pointing anywhere at random", uniform, MUTED),
        ("Each language's own arrows, shuffled\n(each letter keeps its number of incoming arrows)", shuffled, MUTED),
        ("Shuffled as above, but no letter\nmay point at itself", noself, MUTED),
        ("Observed in the 36 languages", observed, BLUE),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, (_lab, v, col) in enumerate(rows):
        ax.barh(i, v, height=0.55, color=col, edgecolor=SURFACE, lw=2)
        ax.text(v + 0.4, i, f"{v:.1f}" if col == MUTED else f"{v:d}", va="center", fontsize=10, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 36)
    ax.set_xlabel("Languages (of 36) in which every letter ends in the same loop\n(letter pairs inside words)")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_baselines.png", dpi=200)
    plt.close(fig)


def fig5(summary, counts):
    temps = np.geomspace(1.0, 0.03, 40)
    fig, ax = plt.subplots(figsize=(9, 5))
    curves = {}
    for code in L.LANGUAGES:
        sym, m, f, s, order = L.chain(counts[code], code, "standard", "text", True)
        loop = list(L.main_attractor(s)[1])
        mass = []
        for t in temps:
            pi = np.clip(L.stationary(L.tempered(m, t))[0], 0, None)
            mass.append(pi[loop].sum() / pi.sum())
        curves[code] = np.array(mass)
    for code, y in curves.items():
        if code != "en":
            ax.plot(temps, 100 * y, color=MUTED, lw=0.8, alpha=0.7)
    ax.plot(temps, 100 * curves["en"], color=BLUE, lw=2.2)
    box = {"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5}
    i = int(np.argmin(np.abs(temps - 0.2)))
    ax.annotate("English: share of the time spent writing “the”", xy=(temps[i], 100 * curves["en"][i]),
                xytext=(0.62, 93), fontsize=9, color=INK, bbox=box,
                arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 0.8})  # fmt: skip
    lo = min(curves, key=lambda c: curves[c][0])
    ax.text(0.97, 100 * curves[lo][0] - 4.5, f"{L.LANGUAGES[lo][0]}", color=INK2, fontsize=8, ha="left", bbox=box)
    ax.set_xscale("log")
    ax.set_xlim(1.0, 0.03)
    ax.set_ylim(0, 102)
    ax.set_xticks([1, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03], ["1\nsample the\ncounts", "0.5", "0.3", "0.2", "0.1", "0.05",
                   "0.03\nalmost always\nthe likeliest"])  # fmt: skip
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Share of the time on the word loop (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.99, 0.04, "Grey: the other 35 languages", transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_temperature.png", dpi=200)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    summary, counts = load()
    fig1(summary, counts)
    fig2(summary)
    fig3(summary)
    fig4(summary)
    fig5(summary, counts)
    print("figures/fig1 to fig5 written")


if __name__ == "__main__":
    main()
