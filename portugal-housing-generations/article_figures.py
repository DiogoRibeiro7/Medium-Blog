"""Two figures for the Medium article (reads results.json, results_integrated.json and affordability_panel.csv).

fig_article_payment_decomposition.png      What moved the first-payment-to-income index in each period
fig_article_completions_composition.png    New dwellings are dearer, and the completions association vanishes
                                           when only existing dwellings are priced

Needs matplotlib. Run after analysis.py, affordability.py and integrated.py:
    python article_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regional_figures import AQUA, BLUE, INK, MUTED, ORANGE

HERE = Path(__file__).parent
OUT = HERE / "figures"
PERIODS = ["2008-2015", "2015-2021", "2021-2023", "2023-2025"]
AREAS = [("PT", "Portugal"), ("1A0", "Grande\nLisboa"), ("1B0", "Península\nde Setúbal"), ("11A", "Área Metropolitana\ndo Porto")]


def payment_decomposition() -> None:
    nat = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    dec = nat["initial_payment_to_income_change_log_points"]["survey_year_alignment"]
    terms = [("price", "house prices", INK), ("financing_cost", "financing cost (mortgage rate)", ORANGE),
             ("income", "income (enters with a minus sign)", BLUE)]
    fig, ax = plt.subplots(figsize=(10.4, 4.9))
    width = 0.24
    for i, (key, label, colour) in enumerate(terms):
        values = [dec[p][key] * (-1 if key == "income" else 1) for p in PERIODS]
        xs = [j + (i - 1) * width for j in range(len(PERIODS))]
        ax.bar(xs, values, width=width - 0.02, color=colour, label=label)
    for j, p in enumerate(PERIODS):
        total = dec[p]["total"]
        ax.plot([j - 1.5 * width, j + 1.5 * width], [total, total], color=MUTED, lw=2.2, solid_capstyle="butt")
        ax.text(j, 66, f"total {total:+.1f}", fontsize=8.6, color=INK, ha="center", va="center", fontweight="bold")
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(range(len(PERIODS)), [p.replace("-", "–") for p in PERIODS])
    ax.set_xlim(-0.55, len(PERIODS) - 0.45)
    ax.set_ylim(-52, 72)
    ax.set_ylabel("contribution, log points")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", fontsize=8.2, frameon=False)
    ax.set_title("Change in the first-mortgage-payment-to-income index, split into its three terms",
                 loc="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "Grey line and bold figure: total change (price + financing − income). Survey-year alignment of EU-SILC "
             "income; 30-year annuity at the average rate on new loans.\nSources: Eurostat prc_hpi_a, ilc_di03; ECB MIR. "
             "Diogo Ribeiro.", fontsize=6.9, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "fig_article_payment_decomposition.png")
    plt.close(fig)


def completions_composition() -> None:
    with (HERE / "affordability_panel.csv").open(encoding="utf-8") as f:
        rows = {(r["region_code"], r["year"]): r for r in csv.DictReader(f)}
    check = json.loads((HERE / "results_integrated.json").read_text(encoding="utf-8"))["completions_composition_check"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.4, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    width = 0.36
    for j, (code, name) in enumerate(AREAS):
        existing = float(rows[(code, "2024")]["price_m2_existing"])
        new = float(rows[(code, "2024")]["price_m2_new"])
        a.bar(j - width / 2, existing, width=width - 0.02, color=BLUE, label="existing dwellings" if j == 0 else None)
        a.bar(j + width / 2, new, width=width - 0.02, color=ORANGE, label="new dwellings" if j == 0 else None)
        a.annotate(f"+{100 * (new / existing - 1):.0f}%", (j + width / 2, new), xytext=(0, 3), textcoords="offset points",
                   ha="center", fontsize=8.4, color=INK)
    a.set_xticks(range(len(AREAS)), [n for _, n in AREAS], fontsize=8.6)
    a.set_ylabel("median price, € per m² (2024)")
    a.grid(axis="x", visible=False)
    a.legend(loc="upper right", fontsize=8.2, frameon=False)
    a.set_title("New dwellings sell at a premium", loc="left", fontsize=10.5, color=INK)

    labels = [("all_dwellings", f"all sales\n({check['municipalities'][0]} municipalities)", INK),
              ("existing_dwellings", f"existing dwellings only\n({check['municipalities'][1]} municipalities)", AQUA)]
    for i, (key, label, colour) in enumerate(labels):
        est = 100 * check[key]["estimate"]
        lo, hi = (100 * v for v in check[key]["ci95_cr1"])
        b.plot([lo, hi], [i, i], color=colour, lw=2.4, solid_capstyle="round")
        b.plot([est], [i], "o", color=colour, ms=9, markeredgecolor="white", markeredgewidth=1.5)
        b.annotate(f"{est:+.1f}%  [{lo:+.1f}, {hi:+.1f}]", (hi, i), xytext=(8, 0), textcoords="offset points",
                   va="center", fontsize=8.6, color=INK)
    b.axvline(0, color=INK, lw=0.9)
    b.set_yticks([0, 1], [lab for _, lab, _ in labels], fontsize=8.6)
    b.set_ylim(1.6, -0.6)
    b.set_xlim(-1.5, 4.2)
    b.grid(axis="y", visible=False)
    b.set_xlabel("% price change per extra completion per 1,000 residents")
    b.set_title("Within-municipality association, 2021–2024", loc="left", fontsize=10.5, color=INK)
    fig.suptitle("The price–construction association comes from the mix of dwellings sold",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "Right: two-way fixed effects with income, municipality and year effects; 95% intervals "
             "cluster-robust by municipality. Sources: INE 0012234, 0013248, 0012712, 0012917. Diogo Ribeiro.",
             fontsize=6.9, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    fig.savefig(OUT / "fig_article_completions_composition.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    payment_decomposition()
    completions_composition()
    print("wrote fig_article_payment_decomposition.png, fig_article_completions_composition.png")


if __name__ == "__main__":
    main()
