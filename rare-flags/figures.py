"""Draw the article's figures from the tables flag_numbers.py writes to results/.

python figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

import flags_core as fc  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
LIGHT_ORANGE, LIGHT_GREY = "#f8c9b2", "#e9e8e2"
RAMP = ["#86b6ef", "#4f94e6", "#256abf", "#0d366b"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    }
)
CODE = "Code: Medium-Blog/rare-flags. Figure: Diogo Ribeiro."


def frame(fig, title, subtitle, credit, note=None, top_in=0.75, bottom_in=0.7):
    h = fig.get_figheight()
    fig.subplots_adjust(top=1 - top_in / h, bottom=bottom_in / h)
    fig.text(0.01, 1 - 0.08 / h, title, fontsize=11, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.01, 1 - 0.34 / h, subtitle, fontsize=8.5, color=INK2, ha="left", va="top")
    y = 0.06 / h
    fig.text(0.01, y, credit, fontsize=7.2, color=MUTED, ha="left", va="bottom")
    if note:
        fig.text(0.01, y + 0.2 / h, note, fontsize=8, color=INK2, ha="left", va="bottom")


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def whole(x):
    return int(np.floor(x + 0.5))


def fig1_thousand_essays():
    f = pd.read_csv(RES / "thousand_essays.csv").set_index("group")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), gridspec_kw={"wspace": 0.08})
    cols, rows = 40, 25
    for ax, group, title in zip(
        axes,
        ["native", "non-native"],
        ["A class of native English writers", "A class of non-native writers"],
        strict=True,
    ):
        r = f.loc[group]
        ai, ai_flag = whole(r.positive), whole(r.true_flags)
        human_flag = whole(r.false_flags)
        kinds = [ORANGE] * ai_flag + [LIGHT_ORANGE] * (ai - ai_flag) + [BLUE] * human_flag
        kinds += [LIGHT_GREY] * (cols * rows - len(kinds))
        for i, c in enumerate(kinds):
            x, y = i % cols, rows - 1 - i // cols
            ax.add_patch(Rectangle((x + 0.08, y + 0.08), 0.84, 0.84, color=c, lw=0))
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            f"{title}\n{ai_flag + human_flag} flags, of which {ai_flag} are right",
            fontsize=9.5,
            color=INK2,
            loc="left",
        )
    handles = [
        Patch(color=ORANGE, label="AI-written, flagged"),
        Patch(color=LIGHT_ORANGE, label="AI-written, missed"),
        Patch(color=BLUE, label="Written by the student, flagged"),
        Patch(color=LIGHT_GREY, label="Written by the student, not flagged"),
    ]
    fig.legend(handles=handles, loc="lower center", ncols=4, fontsize=8.5, bbox_to_anchor=(0.5, 0.11))
    frame(
        fig,
        "Figure 1. One thousand essays, 30 of them AI-written, checked by the same detector",
        "Average flag rates of seven detectors tested by Liang et al. (2023): 48% of AI essays, 5% of essays by US "
        "eighth-graders, 61% of TOEFL essays.",
        "Data: Liang et al. (2023), Patterns. " + CODE,
        note="Counts rounded to whole essays.",
        bottom_in=1.05,
    )
    fig.savefig(FIG / "fig1_thousand_essays.png")
    plt.close(fig)


def fig2_ppv_curves():
    cases = pd.read_csv(RES / "cases.csv")
    b = np.logspace(-5, np.log10(0.5), 400)
    fig, ax = plt.subplots(figsize=(8, 5.4))
    for fpr, col in zip([0.1, 0.01, 0.001, 0.0001], RAMP, strict=True):
        ax.semilogx(b, fc.ppv(b, 0.9, fpr) * 100, color=col, lw=1.8)
        half = fpr / (0.9 + fpr)  # base rate at which half the flags are right
        ax.text(half * 0.85, 50, f"{fpr * 100:g}% false alarms", fontsize=7.8, color=col, ha="right", va="center")
    pts = [
        ("Mammography", 0.01, 0.90, 0.09, (-30, 8)),
        ("OpenAI classifier, 3% AI", 0.03, 0.26, 0.09, (6, -4)),
        ("OpenAI classifier, 11% AI", 0.11, 0.26, 0.09, (6, -4)),
        ("Card alert at the EEA fraud rate,\n0.1% false alarms", fc_eu(cases), 0.9, 0.001, (6, 2)),
    ]
    for name, base, sens, fpr, off in pts:
        v = float(fc.ppv(base, sens, fpr)) * 100
        ax.plot(base, v, "o", color=ORANGE, ms=6, zorder=5)
        ax.annotate(name, (base, v), xytext=off, textcoords="offset points", fontsize=7.8, color=INK2)
    ax.set_ylim(0, 100)
    ax.set_xlim(1e-5, 0.5)
    ax.set_xlabel("Base rate: share of cases that are really positive")
    ax.set_ylabel("Share of flags that are right (%)")
    grid(ax)
    frame(
        fig,
        "Figure 2. The same detector, flags worth very different amounts",
        "Share of flags that are right, for a detector that catches 90% of true cases, by base rate and "
        "false-alarm rate. Dots: real cases.",
        "Sources: Gigerenzer et al. (2007); OpenAI (2023); Turnitin (2024); EBA and ECB (2024). " + CODE,
        note="Dots use their own detection rates: 90% for mammography, 26% for OpenAI's classifier.",
        bottom_in=0.85,
    )
    fig.savefig(FIG / "fig2_ppv_curves.png")
    plt.close(fig)


def fc_eu(cases):
    return float(cases[cases.case.str.startswith("Card alert")].base.iloc[0])


def fig3_detectors():
    d = pd.read_csv(RES / "detectors.csv")
    order = fc.DETECTORS[::-1]
    y = np.arange(len(order))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, groups, title in (
        (
            a1,
            [("native", BLUE, "US eighth-graders"), ("non-native", ORANGE, "TOEFL, non-native")],
            "Essays written by students: flagged as AI",
        ),
        (
            a2,
            [("ai", BLUE, "Plain ChatGPT output"), ("ai_edited", ORANGE, "After a self-edit prompt")],
            "Essays written by ChatGPT: flagged as AI",
        ),
    ):
        for k, (g, col, lab) in enumerate(groups):
            s = d[d.group == g].set_index("detector").loc[order]
            yy = y + (0.15 if k == 0 else -0.15)
            ax.hlines(yy, s.lo * 100, s.hi * 100, color=col, lw=2, alpha=0.45)
            ax.plot(s.rate * 100, yy, "o", color=col, ms=6, label=lab)
        ax.set_xlim(-2, 102)
        ax.set_xlabel("Share flagged (%), with 95% interval")
        ax.set_title(title, fontsize=9.5, color=INK2, loc="left")
        ax.legend(loc="lower right", fontsize=8)
        grid(ax, "x")
    a1.set_yticks(y)
    a1.set_yticklabels(order)
    frame(
        fig,
        "Figure 3. Seven detectors, on student essays and on ChatGPT's",
        "A score above 0.5 counts as a flag. 88 essays by US eighth-graders, 91 TOEFL essays, 176 ChatGPT essays "
        "before and after a self-edit prompt.",
        "Data: Liang et al. (2023), github.com/Weixin-Liang/ChatGPT-Detector-Bias. " + CODE,
        bottom_in=0.75,
    )
    fig.savefig(FIG / "fig3_detectors.png")
    plt.close(fig)


def fig4_fraud():
    c = pd.read_csv(RES / "fraud_curve.csv")
    r = pd.read_csv(RES / "fraud_recall.csv")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(c.recall * 100, c.precision * 100, color=BLUE, lw=1.8, label="At the data's fraud rate, 0.15%")
    ax.plot(c.recall * 100, c.precision_eu * 100, color=ORANGE, lw=1.8, label="At the EEA card-fraud rate, 0.015%")
    for row in r.itertuples():
        if row.recall in (0.5, 0.8):
            for v, col in ((row.precision, BLUE), (row.precision_eu, ORANGE)):
                ax.plot(row.recall * 100, v * 100, "o", color=col, ms=5)
                ax.annotate(
                    f"{v * 100:.0f}%",
                    (row.recall * 100, v * 100),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=col,
                )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Share of frauds caught (%)")
    ax.set_ylabel("Share of alerts that are fraud (%)")
    ax.legend(loc="upper right", fontsize=8.5)
    grid(ax)
    frame(
        fig,
        "Figure 4. One fraud model, two base rates",
        "Logistic regression trained on day 1 of 284,807 card transactions and tested on day 2; orange: the same "
        "scores if fraud were ten times rarer.",
        "Data: ULB Machine Learning Group and Worldline, September 2013 (Dal Pozzolo et al. 2015). " + CODE,
        bottom_in=0.75,
    )
    fig.savefig(FIG / "fig4_fraud.png")
    plt.close(fig)


def fig5_required():
    b = np.logspace(-4.2, np.log10(0.3), 300)
    fig, ax = plt.subplots(figsize=(8, 5.4))
    for t, col in zip([0.5, 0.9, 0.99], RAMP[1:], strict=True):
        ax.loglog(b * 100, fc.required_fpr(b, 0.9, t) * 100, color=col, lw=1.8)
    rec = pd.read_csv(RES / "fraud_recall.csv").set_index("recall")
    det = pd.read_csv(RES / "detectors.csv").pivot(index="detector", columns="group", values="rate").mean()
    pts = [
        ("average detector, non-native essays", 0.03, det["non-native"]),
        ("OpenAI's classifier", 0.03, 0.09),
        ("average detector, native essays", 0.03, det["native"]),
        ("the 1% bound stated by Turnitin", 0.03, 0.01),
        ("mammography", 0.01, 0.09),
        ("fraud model catching 80%, at the EEA fraud rate", 0.00015, rec.loc[0.8, "fpr"]),
    ]
    for k, (_, base, fpr) in enumerate(pts, start=1):
        ax.plot(base * 100, fpr * 100, "o", color=ORANGE, ms=6, zorder=5)
        ax.annotate(
            str(k),
            (base * 100, fpr * 100),
            xytext=(7, -3),
            textcoords="offset points",
            fontsize=8.5,
            color=INK,
            fontweight="bold",
        )
    items = [f"{k} {name}" for k, (name, _, _) in enumerate(pts, start=1)]
    key = "Dots: " + "; ".join(items[:3]) + ";\n" + "; ".join(items[3:]) + "."
    ax.set_xlabel("Base rate (%)")
    ax.set_ylabel("False-alarm rate (%)")
    ax.set_ylim(3e-5, 300)
    ax.set_xlim(4.5e-3, 45)
    fig.canvas.draw()
    for t, col in zip([0.5, 0.9, 0.99], RAMP[1:], strict=True):
        x1, x2 = 0.03, 0.3  # in %, along the line
        y1, y2 = (float(fc.required_fpr(x / 100, 0.9, t)) * 100 for x in (x1, x2))
        (px1, py1), (px2, py2) = ax.transData.transform([(x1, y1), (x2, y2)])
        angle = np.degrees(np.arctan2(py2 - py1, px2 - px1))
        ax.text(
            x1,
            y1 * 1.35,
            f"{t:.0%} of flags right",
            fontsize=7.8,
            color=col,
            rotation=angle,
            rotation_mode="anchor",
            ha="left",
            va="bottom",
        )
    grid(ax, "both")
    frame(
        fig,
        "Figure 5. How rare a false alarm must be for a flag to be trusted",
        "Lines: the highest false-alarm rate at which 50%, 90% or 99% of flags are right, for a detector that "
        "catches 90% of true cases. Dots: real systems.",
        "Sources as in Figures 1 to 4. " + CODE,
        note="A dot above a line means fewer than that share of its flags are right (at 90% detection).\n" + key,
        bottom_in=1.15,
    )
    fig.savefig(FIG / "fig5_required.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig1_thousand_essays()
    fig2_ppv_curves()
    fig3_detectors()
    fig4_fraud()
    fig5_required()


if __name__ == "__main__":
    main()
