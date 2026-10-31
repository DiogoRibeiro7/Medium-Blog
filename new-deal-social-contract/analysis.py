"""Eight reproducible New Deal figures with explicit evidence classification.

The first five figures transcribe retrospective series or institutional events
from named original sources. The last three use invented mathematical inputs,
never interpreted as policy effects or reconstructed historical observations.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURE_NAMES = (
    "figure_01_unemployment_rates.png",
    "figure_02_real_gdp_index.png",
    "figure_03_relief_worker_measurement.png",
    "figure_04_bank_suspensions.png",
    "figure_05_new_deal_timeline.png",
    "figure_06_hypothetical_bank_stress.png",
    "figure_07_hypothetical_income_smoothing.png",
    "figure_08_hypothetical_coverage.png",
)

INK, TEAL, GOLD = "#172D38", "#007F85", "#D88D3D"
RED, BLUE, PURPLE, GREY = "#B35056", "#4A78A0", "#79609A", "#6C7F89"
PAPER = "#FCFCFA"


def read_csv_numbers(filename: str, columns: Iterable[str]) -> dict[str, np.ndarray]:
    """Parse a small local source transcription; reject gaps and invalid values.

    Missing historical years are represented by **omitted rows**, not by
    invented interpolation. Exact provenance is in README.md.
    """
    with (DATA_DIR / filename).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty historical dataset: {filename}")
    values: dict[str, np.ndarray] = {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in columns
    }
    if any(not np.all(np.isfinite(array)) for array in values.values()):
        raise ValueError("All supplied source values must be finite")
    years = values.get("year")
    if years is not None and not np.all(np.diff(years) > 0):
        raise ValueError("Source years must increase strictly")
    return values


def unemployment_data() -> dict[str, np.ndarray]:
    """BLS retrospective annual estimates, civilian labor force aged 14+."""
    return read_csv_numbers(
        "bls_unemployment.csv",
        ("year", "unemployment_rate_percent", "unemployed_thousands"),
    )


def gdp_data() -> dict[str, np.ndarray]:
    """BEA NIPA Table 1.1.6, real GDP in chained 2017 dollars."""
    return read_csv_numbers(
        "bea_real_gdp.csv",
        ("year", "real_gdp_billion_chained_2017_usd"),
    )


def relief_data() -> dict[str, np.ndarray]:
    """Mathy (BLS Monthly Labor Review 2026), chart 1 citing Darby (1975)."""
    return read_csv_numbers(
        "relief_worker_classification.csv",
        ("year", "unemployed_if_relief_workers_unemployed_thousands",
         "unemployed_if_relief_workers_employed_thousands"),
    )


def bank_data() -> dict[str, np.ndarray]:
    """Congressional Record March 3, 1936: 1933 is not comparable and omitted."""
    return read_csv_numbers("bank_suspensions.csv", ("year", "number_suspended"))


def gdp_index(data: dict[str, np.ndarray], base_year: int = 1929) -> np.ndarray:
    """Index real GDP to 100 at a specified **observed** reference year."""
    years = data["year"]
    values = data["real_gdp_billion_chained_2017_usd"]
    matches = np.flatnonzero(years == base_year)
    if len(matches) != 1 or values[matches[0]] <= 0:
        raise ValueError("Requested baseline must be observed and positive")
    return 100.0 * values / values[matches[0]]


def relief_classification_gap(data: dict[str, np.ndarray]) -> np.ndarray:
    """Difference between two **published estimates**, in thousands."""
    a = data["unemployed_if_relief_workers_unemployed_thousands"]
    b = data["unemployed_if_relief_workers_employed_thousands"]
    if np.any(a < b):
        raise ValueError("First count must be at least as large as second")
    return a - b


def _fraction(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0,1]")
    return float(value)


def liquidity_gap(
    insured_share: float,
    uninsured_withdrawal_rate: float,
    liquid_assets_share: float = .2,
    insured_withdrawal_rate: float = .04,
) -> float:
    """Illustrative short-run withdrawals exceeding liquid assets.

    All parameters describe an invented bank. Values are fractions of deposits.
    No equilibrium, emergency borrowing, bank solvency or behavioral feedback
    is inferred. A larger positive gap represents *illustrative* cash stress.
    """
    q = _fraction(insured_share, "insured_share")
    p = _fraction(uninsured_withdrawal_rate, "uninsured_withdrawal_rate")
    r = _fraction(liquid_assets_share, "liquid_assets_share")
    s = _fraction(insured_withdrawal_rate, "insured_withdrawal_rate")
    withdrawals = (1 - q) * p + q * s
    return max(0.0, withdrawals - r)


def monthly_disposable_income(
    employment: np.ndarray,
    wage: float,
    replacement_rate: float,
) -> np.ndarray:
    """Hypothetical income with benefits paid only in unemployed months."""
    employed = np.asarray(employment)
    if employed.ndim != 1 or employed.size == 0:
        raise ValueError("Employment states must be a nonempty 1D array")
    if not np.all((employed == 0) | (employed == 1)):
        raise ValueError("Employment states must be binary")
    if isinstance(wage, bool) or not math.isfinite(float(wage)) or wage <= 0:
        raise ValueError("Wage must be finite and strictly positive")
    benefit = _fraction(replacement_rate, "replacement_rate")
    return float(wage) * (employed + (1 - employed) * benefit)


def realized_relief(
    baseline_harm: np.ndarray,
    legal_coverage: np.ndarray,
    access_probability: np.ndarray,
    effectiveness: float = .6,
) -> np.ndarray:
    """Invented group-level avoided harm; never an estimated policy effect."""
    harms = np.asarray(baseline_harm, dtype=float)
    covered = np.asarray(legal_coverage, dtype=float)
    access = np.asarray(access_probability, dtype=float)
    if harms.ndim != 1 or harms.size == 0 or not (
        harms.shape == covered.shape == access.shape
    ):
        raise ValueError("Group inputs must have the same nonempty 1D shape")
    if not np.all(np.isfinite(harms)) or np.any(harms < 0):
        raise ValueError("Baseline harm must be nonnegative and finite")
    if not np.all(np.isfinite(covered)) or np.any((covered < 0) | (covered > 1)):
        raise ValueError("Coverage must be finite and in [0,1]")
    if not np.all(np.isfinite(access)) or np.any((access < 0) | (access > 1)):
        raise ValueError("Access must be finite and in [0,1]")
    e = _fraction(effectiveness, "effectiveness")
    return harms * covered * access * e


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER, "savefig.dpi": 175,
        "axes.titlesize": 16, "axes.titleweight": "bold",
        "font.size": 11, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": "#B5C3C8",
    })


def _save(fig: plt.Figure, path: Path, note: str) -> None:
    fig.subplots_adjust(bottom=.20)
    fig.text(.075, .025, note, color=GREY, fontsize=8.5)
    fig.savefig(path, bbox_inches="tight", pad_inches=.20)
    plt.close(fig)


def _unemployment(path: Path) -> None:
    d = unemployment_data()
    fig, ax = plt.subplots(figsize=(10.5, 5.9))
    ax.plot(d["year"], d["unemployment_rate_percent"], color=TEAL, lw=3,
            marker="o", ms=6.5)
    ax.axvline(1933, ls="--", lw=1.2, color=GOLD)
    ax.annotate("1933: 24.9%", xy=(1933, 24.9), xytext=(1934, 26.5),
                arrowprops={"arrowstyle":"->", "color":INK}, fontsize=12, weight="bold")
    ax.axvspan(1937, 1938, color=RED, alpha=.07)
    ax.set(xlim=(1928.6, 1939.4), ylim=(0, 31), ylabel="Unemployment rate (%)",
           xlabel="Year", title="Unemployment did not vanish with the New Deal")
    ax.set_xticks([1929, 1931, 1933, 1935, 1937, 1939])
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", alpha=.15)
    _save(fig, path, "HISTORICAL ESTIMATE • BLS LFU21000100 (retrospective 1929–39; not contemporaneous CPS survey).")


def _gdp(path: Path) -> None:
    d = gdp_data()
    indexed = gdp_index(d)
    fig, ax = plt.subplots(figsize=(10.5, 5.9))
    ax.plot(d["year"], indexed, lw=3, color=BLUE, marker="o", ms=6)
    ax.axhline(100, color=GOLD, ls="--", lw=1.6, label="1929 level")
    ax.fill_between(d["year"], indexed, 100, where=indexed < 100, color=RED, alpha=.11)
    ax.annotate("1933 trough", (1933, indexed[4]), xytext=(1930, 91),
                arrowprops={"arrowstyle":"->", "color":INK})
    ax.set(xlim=(1928.6, 1940.4), ylabel="Real GDP index (1929 = 100)",
           xlabel="Year", title="Output recovered unevenly after the collapse")
    ax.set_xticks([1929, 1931, 1933, 1935, 1937, 1939])
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", alpha=.15)
    ax.legend(frameon=False, loc="upper left")
    _save(fig, path, "HISTORICAL RECONSTRUCTION • BEA NIPA Table 1.1.6, real GDP in chained 2017 dollars, indexed.")


def _relief_measurement(path: Path) -> None:
    d = relief_data()
    a = d["unemployed_if_relief_workers_unemployed_thousands"] / 1e3
    b = d["unemployed_if_relief_workers_employed_thousands"] / 1e3
    fig, ax = plt.subplots(figsize=(10.5, 5.9))
    ax.plot(d["year"], a, color=RED, lw=2.8, marker="o",
            label="Relief workers classified unemployed")
    ax.plot(d["year"], b, color=TEAL, lw=2.8, marker="s",
            label="Relief workers classified employed")
    ax.fill_between(d["year"], b, a, color=GOLD, alpha=.16)
    ax.annotate("1936 gap: 3.74m", (1936, (a[7]+b[7])/2),
                xytext=(1932.8, 5.0), fontsize=10.5,
                arrowprops={"arrowstyle":"->", "color":INK})
    ax.set(xlim=(1928.5, 1939.5), ylim=(0, 15), xlabel="Year",
           ylabel="Estimated unemployed (millions)",
           title="The measurement rule changes the unemployment count")
    ax.set_xticks([1929, 1931, 1933, 1935, 1937, 1939])
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", alpha=.15)
    _save(fig, path, "PUBLISHED ESTIMATES • Mathy (BLS MLR, 2026), chart 1 reproducing Darby (1975); not two surveys.")


def _bank_suspensions(path: Path) -> None:
    d = bank_data()
    fig, ax = plt.subplots(figsize=(10.5, 5.9))
    bars = ax.bar(d["year"], d["number_suspended"], color=RED, width=.64)
    for bar, count in zip(bars, d["number_suspended"]):
        ax.text(bar.get_x()+bar.get_width()/2, count+60, str(int(count)),
                ha="center", fontsize=10.5)
    ax.axvspan(1932.55, 1933.45, color=GOLD, alpha=.12)
    ax.text(1933, 1120, "1933 omitted:\nnon-comparable\nbanking holiday", ha="center",
            color=INK, fontsize=10.2)
    ax.set(xlim=(1928.4, 1935.6), ylim=(0, 2900),
           xlabel="Year", ylabel="Bank suspensions (count)",
           title="Bank suspensions and a critical 1933 data break")
    ax.set_xticks(range(1929, 1936))
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", alpha=.15)
    _save(fig, path, "CONTEMPORARY TABLE • Congressional Record, 3 Mar 1936; 1933 excluded as noncomparable, not zero.")


def _timeline(path: Path) -> None:
    with (DATA_DIR / "new_deal_events.csv").open("r", newline="", encoding="utf-8") as f:
        events = list(csv.DictReader(f))
    colors = {"Finance": BLUE, "Employment": TEAL, "Infrastructure": GOLD,
              "Labor": RED, "Insurance": PURPLE}
    fig, ax = plt.subplots(figsize=(12.4, 7.0))
    for i, event in enumerate(events):
        date = int(event["year"]) + (int(event["month"])-1)/12
        color = colors[event["category"]]
        ax.hlines(i, 1932.7, date, color=color, alpha=.30, lw=1.4)
        ax.scatter(date, i, s=72, color=color, zorder=3)
        ax.annotate(event["label"], (date, i), xytext=(10,0),
                    textcoords="offset points", va="center", fontsize=9.4, color=color)
    ax.set(xlim=(1932.6, 1941.4), ylim=(-.7, len(events)-.3),
           xlabel="Year", title="A sequence of institutions, not one master plan")
    ax.set_xticks([1933, 1935, 1937, 1939, 1941])
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.13)
    _save(fig, path, "SOURCED CHRONOLOGY • FDIC, NARA, SEC, NLRB, SSA, DOL; selected policy milestones only.")


def _bank_model(path: Path) -> None:
    qs = np.linspace(0, 1, 101)
    ps = np.linspace(0, 1, 101)
    intensity = np.array([[liquidity_gap(q,p) for q in qs] for p in ps])
    fig, ax = plt.subplots(figsize=(10.0, 6.2))
    im = ax.imshow(intensity, origin="lower", extent=(0,1,0,1), cmap="YlOrRd",
                   aspect="auto", vmin=0, vmax=.8)
    cs = ax.contour(qs, ps, intensity, levels=[.000001, .10, .25, .5],
                    colors=[INK, "#704440", "#704440", "#704440"], linewidths=1)
    ax.clabel(cs, inline=True, fontsize=9)
    ax.set(xlabel="Share of deposits assumed insured", 
           ylabel="Uninsured depositors' withdrawal rate",
           title="How insurance changes a hypothetical liquidity shortfall")
    cbar = fig.colorbar(im, ax=ax, fraction=.045, pad=.03)
    cbar.set_label("Cash shortfall / deposits")
    _save(fig, path, "ILLUSTRATION • Liquid assets 20% of deposits, insured withdrawal 4%; no bank is historically modeled.")


def _income_smoothing(path: Path) -> None:
    employed = np.array([1,1,1,0,0,0,0,1,1,1,1,1])
    months = np.arange(1,13)
    fig, ax = plt.subplots(figsize=(10.5,5.9))
    for replacement, color in [(0.,RED),(.35,GOLD),(.60,TEAL)]:
        income = monthly_disposable_income(employed,100.,replacement)
        ax.step(months, income, where="mid", lw=2.8, color=color,
                label=f"Unemployment replacement {replacement:.0%}")
    ax.axvspan(3.5,7.5,alpha=.08,color=RED)
    ax.set(xlim=(.7,12.3), ylim=(0,115), xlabel="Illustrative month",
           ylabel="Disposable cash income (arbitrary units)",
           title="Insurance can soften an invented unemployment shock")
    ax.set_xticks(months)
    ax.legend(frameon=False, fontsize=9.7, loc="lower right")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",alpha=.13)
    _save(fig,path,"ILLUSTRATION • Employment path, wage 100 and replacement rates are invented; no savings, taxes or prices.")


def _coverage(path: Path) -> None:
    groups=["Example A","Example B","Example C"]
    harm=np.array([100.,65.,35.])
    access=np.array([.60,.75,.90])
    narrow=realized_relief(harm,np.array([.30,.65,.95]),access)
    extended=realized_relief(harm,np.array([.85,.90,.95]),access)
    fig,ax=plt.subplots(figsize=(10.5,5.9))
    x=np.arange(3)
    ax.bar(x-.17,narrow,width=.34,label="Narrow coverage",color=GOLD)
    ax.bar(x+.17,extended,width=.34,label="Expanded coverage",color=TEAL)
    ax.set_xticks(x,groups)
    ax.set(ylabel="Avoided harm (arbitrary units)",
           title="The distribution of a safety net depends on eligibility")
    ax.legend(frameon=False)
    ax.spines[["right","top"]].set_visible(False)
    ax.grid(axis="y",alpha=.13)
    _save(fig,path,"ILLUSTRATION • Three invented groups, harms, access and eligibility; not 1935 racial coverage rates.")


def _write_csv(path: Path, fields: list[str],
               rows: list[dict[str, object]]) -> None:
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate_all(output: Path = ROOT) -> None:
    """Create eight image assets and three audit-ready result tables."""
    figures,results=output/"figures",output/"results"
    figures.mkdir(parents=True,exist_ok=True)
    results.mkdir(parents=True,exist_ok=True)
    _style()
    figure_functions=(
        _unemployment,_gdp,_relief_measurement,_bank_suspensions,
        _timeline,_bank_model,_income_smoothing,_coverage,
    )
    for func,name in zip(figure_functions,FIGURE_NAMES):
        func(figures/name)
    summary=[]
    for name,d,col,unit,source in [
        ("BLS estimated unemployment",unemployment_data(),"unemployment_rate_percent",
         "percent","BLS LFU21000100 retrospective annual"),
        ("BEA real GDP",gdp_data(),"real_gdp_billion_chained_2017_usd",
         "billions chained 2017 USD","BEA NIPA Table 1.1.6"),
        ("Bank suspensions",bank_data(),"number_suspended",
         "banks","Congressional Record 3 March 1936"),
    ]:
        summary.extend({"series":name,"year":int(year),"value":float(value),
                        "unit":unit,"source":source,"evidence_class":"historical"}
                       for year,value in zip(d["year"],d[col]))
    _write_csv(results/"source_based_summary.csv",
               ["series","year","value","unit","source","evidence_class"],summary)
    stress=[]
    for q in np.linspace(0,1,21):
        for p in np.linspace(0,1,21):
            stress.append({"insured_share":round(float(q),3),
                           "uninsured_withdrawal_rate":round(float(p),3),
                           "cash_shortfall_fraction":liquidity_gap(float(q),float(p)),
                           "evidence_class":"hypothetical"})
    _write_csv(results/"hypothetical_bank_stress.csv",
               ["insured_share","uninsured_withdrawal_rate",
                "cash_shortfall_fraction","evidence_class"],stress)
    rows=[]
    for replacement in (0.,.35,.60):
        employed=np.array([1,1,1,0,0,0,0,1,1,1,1,1])
        for month,income in enumerate(
            monthly_disposable_income(employed,100.,replacement),start=1
        ):
            rows.append({"month":month,"replacement_rate":replacement,
                         "disposable_income_arbitrary_units":income,
                         "evidence_class":"hypothetical"})
    _write_csv(results/"hypothetical_income_smoothing.csv",
               ["month","replacement_rate","disposable_income_arbitrary_units",
                "evidence_class"],rows)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT)
    args=parser.parse_args()
    generate_all(args.output.resolve())
    print(f"Created {len(FIGURE_NAMES)} figures and three CSV result tables")


if __name__=="__main__":
    main()
