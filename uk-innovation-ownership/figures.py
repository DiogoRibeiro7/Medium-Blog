"""Draw the five figures into figures/. Run after fetch_data.py.

python figures.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt

import ownership_core as oc
import ownership_numbers as on

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = Path(__file__).resolve().parent / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#a8a6a0"
GRID = "#e6e4df"
UK = "#eb6834"  # categorical slot 2
BLUE = "#2a78d6"  # categorical slot 1
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]  # slots 1, 3, 4, 5, 7 for the peers
SOURCE_OECD = "Source: OECD"
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "font.size": 10,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def note(fig, text):
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=8, color=INK2)


def fig1_growth_and_funding():
    hg = oc.load_high_growth()
    hg = hg[(hg["year"] >= 2012) & (hg["year"] <= 2018) & (hg["industry"] == "J")]
    m = hg.groupby("iso3")["rate"].mean()
    m = m[m.index.isin(oc.EU27 + ["GBR"])].sort_values()
    vc = oc.load_vc()
    vc = vc[vc["year"] >= 2007]
    eu = vc[vc["iso3"].isin(oc.EU27)].groupby("year")["total"].median()

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 6.2), gridspec_kw={"width_ratios": [1, 1.25]})
    colors = [UK if k == "GBR" else MUTED for k in m.index]
    a.barh(m.index, m.values, color=colors, height=0.72)
    med = m[m.index.isin(oc.EU27)].median()
    a.axvline(med, color=INK, lw=1, ls="--")
    a.text(med + 0.2, -0.9, f"EU median {med:.1f}%", color=INK, fontsize=8, va="bottom")
    a.set_title("ICT firms growing 10%+ a year, 2012–2018")
    a.set_xlabel("% of ICT firms with 10+ employees (mean of years)")
    a.tick_params(axis="y", labelsize=8)
    a.grid(axis="x", color=GRID, lw=0.8)
    a.set_axisbelow(True)

    for iso, color, label in [("GBR", UK, "UK"), ("DEU", BLUE, "Germany"), ("FRA", "#1baf7a", "France")]:
        s = vc[vc["iso3"] == iso].set_index("year")["total"]
        b.plot(s.index, s.values, color=color, lw=2, label=label)
        b.text(s.index[-1] + 0.3, s.values[-1], label, color=INK2, fontsize=9, va="center")
    b.plot(eu.index, eu.values, color=INK, lw=1.5, ls="--", label="EU median")
    b.text(eu.index[-1] + 0.3, eu.values[-1], "EU median", color=INK2, fontsize=9, va="center")
    b.set_title("Venture capital invested in each country's companies")
    b.set_ylabel("% of GDP")
    b.set_xlim(2007, 2027.5)
    b.set_xticks(range(2008, 2026, 2))
    b.set_ylim(0, None)
    b.grid(axis="y", color=GRID, lw=0.8)
    b.legend(frameon=False, loc="upper left", fontsize=9)
    note(fig, "Source: OECD, SDBS business demography (high-growth enterprises) and venture capital market statistics.")
    fig.savefig(OUT / "fig1_growth_and_funding.png")
    plt.close(fig)


def fig2_pensions_and_ownership():
    frame = on.model_frame(**on.MAIN).dropna(subset=["pension_assets"])
    fig, ax = plt.subplots(figsize=(9, 6.2))
    others = frame[frame["iso3"] != "GBR"]
    ax.scatter(others["pension_assets"] + 1, 100 * others["foreign_share"], s=60, color=MUTED, edgecolor=SURFACE, lw=2)
    uk = frame[frame["iso3"] == "GBR"].iloc[0]
    ax.scatter(uk["pension_assets"] + 1, 100 * uk["foreign_share"], s=110, color=UK, edgecolor=SURFACE, lw=2, zorder=3)
    offsets = {"IRL": (6, 6), "BEL": (-30, 4), "ITA": (-28, 4), "FRA": (6, -2), "HUN": (6, 2), "ROU": (6, 2)}
    labelled = {"GBR", "DEU", "FRA", "ITA", "NLD", "DNK", "SWE", "CHE", "CAN", "AUS", "IRL", "BEL", "ESP", "JPN"}
    labelled |= {"ROU", "HUN", "SVK", "POL", "CZE", "KOR"}
    for r in frame.itertuples():
        if r.iso3 in labelled:
            ax.annotate(
                r.iso3,
                (r.pension_assets + 1, 100 * r.foreign_share),
                xytext=offsets.get(r.iso3, (6, 4)),
                textcoords="offset points",
                fontsize=9,
                color=INK if r.iso3 == "GBR" else INK2,
                fontweight="bold" if r.iso3 == "GBR" else "normal",
            )
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 11, 21, 51, 101, 201])
    ax.set_xticklabels(["1", "4", "10", "20", "50", "100", "200"])
    ax.set_xlabel("Asset-backed pension investment, % of GDP (mean 2008–2021, log scale)")
    ax.set_ylabel("% of the country's inventions owned abroad (PCT, 2008–2021)")
    ax.set_title("Large domestic pension pools do not keep inventions at home")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    note(
        fig, "Source: OECD, Global Pension Statistics and patent statistics (foreign ownership of domestic inventions)."
    )
    fig.savefig(OUT / "fig2_pensions_and_ownership.png")
    plt.close(fig)


def fig3_ownership_trend():
    panel = oc.load_patents("PCT")
    keep = ["GBR", "DEU", "FRA", "ITA", "NLD", "SWE"]
    names = {"GBR": "UK", "DEU": "Germany", "FRA": "France", "ITA": "Italy", "NLD": "Netherlands", "SWE": "Sweden"}
    mids, shares = [], {k: [] for k in keep}
    us = {k: [] for k in keep}
    for first in range(1980, 2021, 5):
        last = min(first + 4, 2021)
        w = oc.pooled_shares(panel, first, last, countries=keep).set_index("iso3")
        mids.append((first + last) / 2)
        for k in keep:
            shares[k].append(100 * w.loc[k, "foreign_share"])
            us[k].append(100 * w.loc[k, "us_share"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    colors = dict(zip(keep, [UK] + SERIES, strict=True))
    for ax, data, title in [(a, shares, "Owned by applicants abroad"), (b, us, "Owned by US applicants")]:
        for k in keep:
            ax.plot(mids, data[k], color=colors[k], lw=2.4 if k == "GBR" else 1.6, marker="o", ms=4, label=names[k])
        ax.set_title(title)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_xticks(mids)
        ax.set_xticklabels([f"{int(m - 2)}–{str(int(m + 2))[2:]}" if m < 2020 else "2020–21" for m in mids])
        ax.tick_params(axis="x", labelsize=8, rotation=45)
    a.set_ylabel("% of inventions with an inventor in the country")
    a.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
    note(fig, "Source: OECD patent statistics, PCT applications by priority date, pooled over five-year windows.")
    fig.savefig(OUT / "fig3_ownership_trend.png")
    plt.close(fig)


def fig4_holdout():
    frame = on.model_frame(**on.MAIN)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    for ax, outcome, title in [
        (axes[0], "us_share", "Owned in the US"),
        (axes[1], "other_share", "Owned abroad, outside the US"),
    ]:
        loo = oc.leave_one_out(frame, outcome, oc.COVARIATES)
        lo, hi = oc.jackknife_plus(frame, outcome, oc.COVARIATES, "GBR", alpha=on.ALPHA)
        top = max(loo["actual"].max(), loo["predicted"].max(), hi) * 100 * 1.05
        ax.plot([0, top], [0, top], color=INK2, lw=1, ls="--")
        rest = loo[loo["iso3"] != "GBR"]
        ax.scatter(100 * rest["predicted"], 100 * rest["actual"], s=55, color=MUTED, edgecolor=SURFACE, lw=2)
        uk = loo[loo["iso3"] == "GBR"].iloc[0]
        ax.vlines(100 * uk["predicted"], 100 * lo, 100 * hi, color=UK, lw=2, alpha=0.5)
        ax.scatter(100 * uk["predicted"], 100 * uk["actual"], s=110, color=UK, edgecolor=SURFACE, lw=2, zorder=3)
        ax.annotate(
            f"UK, rank {uk['rank']} of {len(loo)}",
            (100 * uk["predicted"], 100 * uk["actual"]),
            xytext=(8, 6),
            textcoords="offset points",
            color=INK,
            fontsize=9,
            fontweight="bold",
        )
        for r in loo.head(5).itertuples():
            if r.iso3 != "GBR":
                off = {"BEL": (6, -9), "AUT": (6, 3)}.get(r.iso3, (6, 2))
                ax.annotate(
                    r.iso3, (100 * r.predicted, 100 * r.actual), xytext=off, textcoords="offset points",
                    fontsize=8, color=INK2,
                )  # fmt: skip
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("Predicted from a model fitted without the country, %")
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Actual share of inventions, %")
    note(
        fig,
        "Fractional logit on log inventions a year, English and transition-economy indicators; 29 countries, PCT "
        "2008–2021.\nThe orange bar is the jackknife+ interval for the UK (nominal 90%). "
        "Source: OECD patent statistics.",
    )
    fig.savefig(OUT / "fig4_holdout.png")
    plt.close(fig)


def fig5_ict_jobs():
    f = oc.load_fats()
    f = f[(f["year"] >= 2014) & (f["year"] <= 2019) & (f["industry"] == "J")].dropna(subset=["all", "foreign"])
    g = f.groupby("iso3")[["all", "foreign", "us"]].sum()
    g = g[g.index.isin(oc.EU27 + ["GBR"])]
    g["us_pct"] = 100 * g["us"] / g["all"]
    g["other_pct"] = 100 * (g["foreign"] - g["us"]) / g["all"]
    g = g.assign(total=g["us_pct"] + g["other_pct"]).sort_values("us_pct")
    fig, ax = plt.subplots(figsize=(9, 7.4))
    us_col = [UK if k == "GBR" else BLUE for k in g.index]
    other_col = ["#f3a582" if k == "GBR" else "#9ec5f4" for k in g.index]
    ax.barh(g.index, g["us_pct"], color=us_col, height=0.72, label="US-controlled")
    ax.barh(
        g.index, g["other_pct"], left=g["us_pct"] + 0.25, color=other_col, height=0.72, label="other foreign-controlled"
    )
    ax.set_title("Who employs ICT workers: foreign-controlled firms, 2014–2019")
    ax.set_xlabel("% of persons employed in information and communication (NACE J)")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9)
    note(
        fig,
        "Countries sorted by the US-controlled share. UK data cover 2014–2017. "
        "Source: Eurostat, inward FATS (fats_g1a_08).",
    )
    fig.savefig(OUT / "fig5_ict_jobs.png")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    fig1_growth_and_funding()
    fig2_pensions_and_ownership()
    fig3_ownership_trend()
    fig4_holdout()
    fig5_ict_jobs()


if __name__ == "__main__":
    main()
