"""Figures for the article, drawn from results/bigram_loop.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
LANG_NAMES = {"en": "English", "fi": "Finnish", "fr": "French",
              "de": "German", "pt": "Portuguese", "sv": "Swedish"}
CREDIT = "Diogo Ribeiro · own simulation on the NLTK Genesis corpus"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def _style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def _credit(fig):
    fig.text(0.01, 0.01, CREDIT, fontsize=7.5, color=MUTED, ha="left", va="bottom")


def _mean(results, name, key, idx=None):
    runs = results[name]["runs"]
    arr = []
    for r in runs:
        v = [rec[key] if idx is None else rec[key][idx] for rec in r]
        arr.append(v)
    arr = np.asarray(arr, dtype=float)
    return arr.mean(axis=0), arr.min(axis=0), arr.max(axis=0)


def _label_ends(ax, xs, ys, labels, colors=None, min_gap=0.035, x_pad=0.6):
    """Direct labels at line ends, nudged apart so they do not collide.

    A short stroke in the series colour sits before each label, so identity
    never rests on position alone; the label text stays in ink.
    """
    order = np.argsort(ys)
    lo, hi = ax.get_ylim()
    x0, x1 = ax.get_xlim()
    span = x1 - x0
    gap = min_gap * (hi - lo)
    placed = []
    for i in order:
        y = ys[i]
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        if colors is not None:
            ax.plot([xs + 0.012 * span, xs + 0.045 * span], [y, y], color=colors[i], lw=2.5,
                    clip_on=False, solid_capstyle="round")
            ax.text(xs + 0.06 * span, y, labels[i], va="center", ha="left", fontsize=9, color=INK2)
        else:
            ax.text(xs + x_pad, y, labels[i], va="center", ha="left", fontsize=9, color=INK2)


def fig_tail(meta, res, out):
    langs = meta["langs"]
    T = meta["T"]
    t = np.arange(T + 1)
    t0 = np.asarray(meta["types0"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ends, handles = [], []
    for li, code in enumerate(langs):
        m, lo, hi = _mean(res, "replace", "types_lang", li)
        h, = ax.plot(t, m / t0[li], color=SERIES[li], lw=2,
                     label=f"{LANG_NAMES[code]}  {m[-1] / t0[li]:.0%}")
        handles.append(h)
        ends.append(m[-1] / t0[li])
    _style(ax)
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("generation")
    ax.set_ylabel("word types the model can still produce")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    order = np.argsort(ends)[::-1]
    leg = ax.legend([handles[i] for i in order], [handles[i].get_label() for i in order],
                    loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9,
                    title="retained at\ngeneration 50", title_fontsize=8.5, handlelength=1.6)
    leg._legend_box.align = "left"
    for txt in leg.get_texts():
        txt.set_color(INK2)
    leg.get_title().set_color(MUTED)
    ax.set_title("Retrained on its own text, the model forgets Finnish words fastest",
                 loc="left", fontsize=11.5, pad=12)
    fig.subplots_adjust(left=0.1, right=0.76, top=0.88, bottom=0.16)
    _credit(fig)
    fig.savefig(out / "fig1_vocabulary_by_language.png", dpi=200)
    plt.close(fig)


def fig_metrics(meta, res, out):
    T = meta["T"]
    t = np.arange(T + 1)
    series = [
        ("gini_simpson", "Gini–Simpson index of words"),
        ("shannon_bits", "Shannon entropy of words"),
        ("cond_gini_simpson", "next-word Gini–Simpson, per context"),
        ("types", "word types"),
        ("bigram_types", "word-pair types"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ends = []
    for k, (key, lab) in enumerate(series):
        m, _, _ = _mean(res, "replace", key)
        y = m / m[0]
        ax.plot(t, y, color=SERIES[k], lw=2)
        ends.append(y[-1])
    _style(ax)
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("generation")
    ax.set_ylabel("value relative to generation 0")
    _label_ends(ax, T, np.asarray(ends), [s[1] for s in series], colors=SERIES, min_gap=0.045)
    ax.set_title("The summary that barely moves and the counts that collapse",
                 loc="left", fontsize=11.5, pad=12)
    fig.subplots_adjust(left=0.09, right=0.62, top=0.88, bottom=0.16)
    _credit(fig)
    fig.savefig(out / "fig2_metrics_replace.png", dpi=200)
    plt.close(fig)


def fig_policies(meta, res, out):
    T = meta["T"]
    t = np.arange(T + 1)
    pol = [("replace", "replace"), ("accumulate", "accumulate"),
           ("anchor10", "anchor, 10% fresh real text")]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ends = []
    for k, (name, lab) in enumerate(pol):
        m, lo, hi = _mean(res, name, "missing_mass")
        ax.fill_between(t, lo, hi, color=SERIES[k], alpha=0.15, lw=0)
        ax.plot(t, m, color=SERIES[k], lw=2)
        ends.append(m[-1])
    _style(ax)
    ax.set_xlim(0, T)
    ax.set_ylim(0, max(0.35, ax.get_ylim()[1]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("generation")
    ax.set_ylabel("held-out words the model can no longer write")
    _label_ends(ax, T, np.asarray(ends), [p[1] for p in pol], colors=SERIES)
    ax.set_title("What the next model is trained on decides how much it forgets",
                 loc="left", fontsize=11.5, pad=12)
    fig.subplots_adjust(left=0.1, right=0.68, top=0.88, bottom=0.16)
    _credit(fig)
    fig.savefig(out / "fig3_policies_missing_mass.png", dpi=200)
    plt.close(fig)


def fig_babel(meta, res, out, name="replace_tau08", T_show=25):
    langs = meta["langs"]
    t = np.arange(T_show + 1)
    shares = []
    for li in range(len(langs)):
        m, _, _ = _mean(res, name, "lang_share_output", li)
        shares.append(m[: T_show + 1])
    shares = np.asarray(shares)
    shares = shares / shares.sum(axis=0, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.stackplot(t, shares, colors=SERIES[: len(langs)], edgecolor=SURFACE, linewidth=1.0)
    _style(ax)
    ax.grid(False)
    ax.set_xlim(0, T_show)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("generation")
    ax.set_ylabel("share of the words the model writes")
    # labels at generation 0, inside each band
    cum = np.cumsum(shares[:, 0])
    for li, code in enumerate(langs):
        mid = cum[li] - shares[li, 0] / 2
        ax.text(0.2, mid, LANG_NAMES[code], va="center", ha="left", fontsize=8, color=INK)
    ax.set_title("Sampling at temperature 0.8: six languages become one",
                 loc="left", fontsize=11.5, pad=12)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.16)
    _credit(fig)
    fig.savefig(out / "fig4_languages_tau08.png", dpi=200)
    plt.close(fig)


def fig_bias(meta, res, out):
    T = meta["T"]
    t = np.arange(T + 1)
    en = meta["langs"].index("en")
    pol = [("replace_tau08", "replace"), ("accumulate_tau08", "accumulate"),
           ("anchor10_tau08", "anchor, 10% fresh real text")]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ends = []
    for k, (name, lab) in enumerate(pol):
        m, lo, hi = _mean(res, name, "lang_share_output", en)
        ax.fill_between(t, lo, hi, color=SERIES[k], alpha=0.15, lw=0)
        ax.plot(t, m, color=SERIES[k], lw=2)
        ends.append(m[-1])
    m0, _, _ = _mean(res, "replace", "lang_share_output", en)
    ax.axhline(m0.mean(), color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(T * 0.5, m0.mean() + 0.025, "English share when sampling at temperature 1",
            fontsize=8.5, color=INK2, ha="center")
    _style(ax)
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("generation")
    ax.set_ylabel("English share of the words the model writes")
    _label_ends(ax, T, np.asarray(ends), [p[1] for p in pol], colors=SERIES, min_gap=0.05)
    ax.set_title("Keeping the original data does not undo a biased sampler",
                 loc="left", fontsize=11.5, pad=12)
    fig.subplots_adjust(left=0.1, right=0.68, top=0.88, bottom=0.16)
    _credit(fig)
    fig.savefig(out / "fig5_bias_by_policy.png", dpi=200)
    plt.close(fig)


def main(results_dir: Path, out_dir: Path):
    d = json.loads((results_dir / "bigram_loop.json").read_text())
    meta, res = d["meta"], d["results"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_tail(meta, res, out_dir)
    fig_metrics(meta, res, out_dir)
    fig_policies(meta, res, out_dir)
    fig_babel(meta, res, out_dir)
    fig_bias(meta, res, out_dir)


if __name__ == "__main__":
    main(Path("results"), Path("figures"))
