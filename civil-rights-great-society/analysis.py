"""Reproduce eight civil-rights and Great Society article figures.

Figures 1–4 reproduce *descriptive* official historical series; figure 5
plots dated events from original institutional sources. Figures 6–8 are
explicitly invented mathematical illustrations, **not treatment effects**.

No network downloads, imputed years, random-number draws or API keys are used.
The complete data dictionary and original URLs appear in README.md.
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import date
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURE_NAMES = (
    "figure_01_national_poverty.png",
    "figure_02_racial_poverty_gap.png",
    "figure_03_poverty_by_age.png",
    "figure_04_medicare_enrollment.png",
    "figure_05_institutional_timeline.png",
    "figure_06_hypothetical_voter_access.png",
    "figure_07_hypothetical_benefit_cliff.png",
    "figure_08_hypothetical_poverty_measure.png",
)
INK = "#192D3B"
TEAL = "#047E81"
BLUE = "#4276A7"
GOLD = "#CA873D"
RED = "#BA5761"
PURPLE = "#7668A7"
GREY = "#5C727C"
PAPER = "#FCFCFA"


def read_numeric_csv(filename: str, columns: Iterable[str]) -> dict[str, np.ndarray]:
    """Read strictly increasing annual records; keep empty cells as NaN.

    In the Census source files some historical columns genuinely contain
    'N' (unavailable). Their transcription uses an empty CSV field and this
    reader yields NaN. It never fills values or connects a line across gaps.
    """
    with (DATA / filename).open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise ValueError(f"No data in {filename}")
    values: dict[str, np.ndarray] = {}
    for column in columns:
        try:
            values[column] = np.array([
                float(record[column]) if record[column].strip() else np.nan
                for record in records
            ], dtype=float)
        except (KeyError, ValueError, AttributeError) as exc:
            raise ValueError(f"Invalid column {column!r} in {filename}") from exc
    years = values.get("year")
    if years is None or np.any(~np.isfinite(years)) or not np.all(np.diff(years) > 0):
        raise ValueError("Years must be finite and strictly increasing")
    if np.any(years != np.floor(years)):
        raise ValueError("Years must be integers")
    for column, array in values.items():
        if column != "year" and np.any(np.isinf(array)):
            raise ValueError("Infinite source values are not permitted")
    return values


def national_poverty() -> dict[str, np.ndarray]:
    """Census historical poverty Table 2, U.S. all-person annual rate."""
    return read_numeric_csv("census_poverty_all.csv", ("year", "poverty_rate_percent"))


def racial_poverty() -> dict[str, np.ndarray]:
    """Census historical poverty Table 2; Black 1960–65 are unavailable."""
    return read_numeric_csv(
        "census_poverty_race.csv",
        ("year", "white_poverty_rate_percent", "black_poverty_rate_percent"),
    )


def age_poverty() -> dict[str, np.ndarray]:
    """Census historical poverty Table 3; age 18–64/65+ 1960–65 unavailable."""
    return read_numeric_csv(
        "census_poverty_age.csv",
        ("year", "under18_poverty_rate_percent", "age18_to64_poverty_rate_percent",
         "age65plus_poverty_rate_percent"),
    )


def medicare_enrollment() -> dict[str, np.ndarray]:
    """CMS HI enrollment as of July 1; disability category starts in 1973."""
    return read_numeric_csv(
        "cms_medicare_enrollment.csv",
        ("year", "aged_hospital_insurance_enrollees",
         "disabled_hospital_insurance_enrollees", "total_hospital_insurance_enrollees"),
    )


def historical_events() -> list[tuple[date, str, str]]:
    """Read selected legal/administrative milestones from primary sources."""
    with (DATA / "institutional_events.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    result = [(date.fromisoformat(r["date"]), r["label"], r["source"]) for r in rows]
    if not result or any(result[i][0] >= result[i + 1][0]
                         for i in range(len(result) - 1)):
        raise ValueError("Event dates must increase strictly")
    return result


def poverty_gap(white: np.ndarray, black: np.ndarray) -> np.ndarray:
    """Absolute Black–White poverty-rate gap in *percentage points*.

    Comparisons with missing values remain missing, rather than estimated.
    This arithmetic association does not identify any legislative effect.
    """
    w, b = np.asarray(white, dtype=float), np.asarray(black, dtype=float)
    if w.shape != b.shape or w.ndim != 1:
        raise ValueError("Equal-length one-dimensional inputs required")
    if np.any(np.isfinite(w) & ((w < 0) | (w > 100))):
        raise ValueError("Poverty rates must be percentages in [0, 100]")
    if np.any(np.isfinite(b) & ((b < 0) | (b > 100))):
        raise ValueError("Poverty rates must be percentages in [0, 100]")
    return b - w


def _probability(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a numeric probability")
    if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
        raise ValueError(f"{name} must lie in [0, 1]")
    return float(value)


def realized_voter_access(
    eligible_population: int,
    eligibility_share: float,
    administrative_barrier: float,
) -> float:
    """Invented friction model E * e * (1-b), not actual voter registration.

    The model excludes demand, turnout, campaigns, suppression heterogeneity,
    demographic change and legal/administrative general-equilibrium responses.
    """
    if isinstance(eligible_population, bool) or not isinstance(eligible_population, int):
        raise TypeError("eligible_population must be integer")
    if eligible_population < 0:
        raise ValueError("Population must be nonnegative")
    share = _probability(eligibility_share, "eligibility_share")
    barrier = _probability(administrative_barrier, "administrative_barrier")
    return eligible_population * share * (1 - barrier)


def benefit_schedule(
    earnings: np.ndarray,
    maximum_benefit: float = 9000,
    phaseout_start: float = 18000,
    withdrawal_rate: float = .35,
    cliff: bool = False,
) -> np.ndarray:
    """Invented annual benefit: tapered entitlement or discontinuous cutoff.

    The tapered schedule is max(0, B-r*max(0,y-T)); the cliff schedule is B
    at y<=T and 0 at y>T. Dollars are *illustrative* and not policy parameters.
    """
    y = np.asarray(earnings, dtype=float)
    if y.ndim != 1 or y.size == 0 or np.any(~np.isfinite(y)) or np.any(y < 0):
        raise ValueError("Earnings must be a nonempty finite, nonnegative 1D array")
    if not math.isfinite(maximum_benefit) or maximum_benefit < 0:
        raise ValueError("Maximum benefit must be finite and nonnegative")
    if not math.isfinite(phaseout_start) or phaseout_start < 0:
        raise ValueError("Phaseout start must be finite and nonnegative")
    rate = _probability(withdrawal_rate, "withdrawal_rate")
    if cliff:
        return np.where(y <= phaseout_start, maximum_benefit, 0.0)
    return np.maximum(0.0, maximum_benefit - rate * np.maximum(0.0, y - phaseout_start))


def poverty_classification(
    cash_income: float, noncash_value: np.ndarray, illustrative_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare illustrative cash-only test to a nonofficial resource total.

    A program's noncash benefit is *not* automatically a cash-equivalent amount.
    This second measure is neither the official nor the Supplemental Poverty
    Measure; it only exhibits a classification and valuation distinction.
    """
    if not math.isfinite(cash_income) or cash_income < 0:
        raise ValueError("Cash income must be finite and nonnegative")
    if not math.isfinite(illustrative_threshold) or illustrative_threshold <= 0:
        raise ValueError("Threshold must be finite and strictly positive")
    val = np.asarray(noncash_value, dtype=float)
    if val.ndim != 1 or val.size == 0 or np.any(~np.isfinite(val)) or np.any(val < 0):
        raise ValueError("Noncash values must be a finite nonnegative 1D array")
    cash_only = np.full(val.shape, cash_income < illustrative_threshold)
    combined = cash_income + val < illustrative_threshold
    return cash_only, combined


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER, "savefig.dpi": 180,
        "axes.titlesize": 15.5, "axes.titleweight": "bold",
        "font.size": 11, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": "#BBC6CC",
    })


def _save(fig: plt.Figure, destination: Path, note: str) -> None:
    fig.subplots_adjust(bottom=.18)
    fig.text(.07, .02, note, fontsize=8.0, color=GREY)
    fig.savefig(destination, bbox_inches="tight", pad_inches=.22)
    plt.close(fig)


def _axes(ax: plt.Axes, y_label: str = "Poverty rate (%)") -> None:
    ax.set(xlabel="Year", ylabel=y_label)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.17)
    ax.set_xticks([1959, 1963, 1965, 1968, 1971, 1975])


def _national(path: Path) -> None:
    d = national_poverty()
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    ax.plot(d["year"], d["poverty_rate_percent"], color=TEAL, lw=3,
            marker="o", ms=6)
    ax.axvspan(1964, 1965.9, alpha=.12, color=GOLD, label="1964–65 legislation")
    ax.annotate("1959: 22.4%", (1959, 22.4), (1961, 25),
                arrowprops={"arrowstyle": "->", "color": INK})
    ax.annotate("1973: 11.1%", (1973, 11.1), (1969.8, 14.7),
                arrowprops={"arrowstyle": "->", "color": INK})
    ax.set(title="Poverty fell before and after the Great Society",
           ylim=(7, 28), xlim=(1958.4, 1975.6))
    _axes(ax)
    ax.legend(frameon=False, loc="upper right")
    _save(fig, path, "HISTORICAL • Census Table 2, all people (official cash-income poverty); not a causal estimate.")


def _race(path: Path) -> None:
    d = racial_poverty()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(d["year"], d["white_poverty_rate_percent"], color=BLUE, lw=2.8,
            marker="o", label="White")
    # A NaN in every 1960–65 Black cell forces a genuine visual break.
    ax.plot(d["year"], d["black_poverty_rate_percent"], color=RED, lw=2.8,
            marker="s", label="Black")
    ax.axvspan(1959.55, 1965.45, color=RED, alpha=.035)
    ax.text(1960.4, 47, "Black annual series\nunavailable 1960–65", fontsize=10,
            color=RED, ha="left")
    ax.set(title="The racial poverty gap narrowed, but remained large",
           ylim=(0, 61), xlim=(1958.4, 1975.6))
    _axes(ax)
    ax.legend(frameon=False, loc="upper right")
    _save(fig, path, "HISTORICAL • Census Table 2; omitted Black 1960–65 observations are NOT interpolated.")


def _age(path: Path) -> None:
    d = age_poverty()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for column, color, label in (
        ("under18_poverty_rate_percent", RED, "Under 18"),
        ("age18_to64_poverty_rate_percent", BLUE, "18–64"),
        ("age65plus_poverty_rate_percent", TEAL, "65+")
    ):
        ax.plot(d["year"], d[column], color=color, lw=2.7, marker="o", ms=5,
                label=label)
    ax.text(1960.2, 33, "Adult/older-age rows\nmissing in 1960–65", fontsize=9.5,
            color=GREY)
    ax.set(title="Older-adult poverty fell; child poverty persisted",
           ylim=(0, 41), xlim=(1958.4, 1975.6))
    _axes(ax)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    _save(fig, path, "HISTORICAL • Census Table 3; 18–64 and 65+ 1960–65 values unavailable; no interpolation.")


def _medicare(path: Path) -> None:
    d = medicare_enrollment()
    aged = d["aged_hospital_insurance_enrollees"] / 1e6
    total = d["total_hospital_insurance_enrollees"] / 1e6
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(d["year"], aged, color=TEAL, marker="o", lw=2.8, label="Aged HI enrollees")
    ax.plot(d["year"], total, color=BLUE, marker="s", lw=2.8,
            label="Total HI enrollees")
    ax.fill_between(d["year"], aged, total, where=d["year"] >= 1973,
                    color=GOLD, alpha=.24, label="Disability group (1973 onward)")
    ax.axvline(1973, color=GOLD, ls="--", lw=1.2)
    ax.set(title="Medicare coverage grew after its 1966 start",
           ylabel="Hospital Insurance enrollment (millions)", xlabel="Year",
           ylim=(17, 27), xlim=(1965.5, 1975.5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.17)
    ax.set_xticks([1966, 1968, 1970, 1972, 1973, 1975])
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    _save(fig, path, "HISTORICAL • CMS hospital-insurance enrollees, July 1; disability coverage begins 1973.")


def _timeline(path: Path) -> None:
    events = historical_events()
    fig, ax = plt.subplots(figsize=(13.1, 7.7))
    # Ordinal horizontal positioning means month spacing remains exact.
    x = np.asarray([d.toordinal() for d, _, _ in events], dtype=float)
    levels = np.asarray([1.6, -1.7, 2.4, -2.5, 1.9, -1.65, 3.1, -3.0, 1.8, -1.8])
    ax.axhline(0, color=INK, lw=1.5)
    for i, ((d, label, _), xi, yi) in enumerate(zip(events, x, levels)):
        color = TEAL if i in (0, 1, 4, 7) else GOLD
        ax.scatter(xi, 0, s=65, color=color, zorder=5)
        ax.plot([xi, xi], [0, yi * .76], color=color, alpha=.7, lw=1.3)
        ax.annotate(f"{d:%b %Y}\n{label}", (xi, yi * .76), xytext=(xi, yi),
                    ha="center", va="center", fontsize=9.0, color=INK)
    years = [1963, 1965, 1967, 1969, 1971, 1973]
    ax.set_xticks([date(y, 1, 1).toordinal() for y in years], [str(y) for y in years])
    ax.set_xlim(date(1962, 8, 1).toordinal(), date(1973, 2, 1).toordinal())
    ax.set_ylim(-3.7, 3.8)
    ax.set_yticks([])
    ax.set_title("Rights, programs and enforcement arrived in stages", pad=22)
    ax.spines[["top", "bottom", "left", "right"]].set_visible(False)
    _save(fig, path, "SELECTED DOCUMENTED EVENTS • King Institute, NARA, LOC, CMS, LBJ Library, HUD and EEOC; not an effect timeline.")


def _voter_access(path: Path) -> None:
    barriers = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    for eligibility, color in ((.65, RED), (.85, TEAL), (1.0, BLUE)):
        accessed = [realized_voter_access(100000, eligibility, float(b)) / 1e3
                    for b in barriers]
        ax.plot(barriers * 100, accessed, color=color, lw=2.8,
                label=f"Eligibility share: {eligibility:.0%}")
    ax.set(title="A legal right is not identical to practical access",
           xlabel="Hypothetical administrative barrier (%)",
           ylabel="People able to access registration (thousands)",
           xlim=(0, 100), ylim=(0, 110))
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.16)
    _save(fig, path, "HYPOTHETICAL MODEL • Invented population and access probabilities; NOT historical registration data.")


def _benefit_cliff(path: Path) -> None:
    income = np.linspace(0, 60000, 601)
    cliff = income + benefit_schedule(income, cliff=True)
    taper = income + benefit_schedule(income, cliff=False)
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    ax.plot(income / 1000, cliff / 1000, color=RED, lw=2.6,
            label="Hard cutoff at $18k")
    ax.plot(income / 1000, taper / 1000, color=TEAL, lw=2.8,
            label="35% phaseout above $18k")
    ax.plot(income / 1000, income / 1000, color=GREY, ls="--", lw=1.5,
            label="Earnings alone")
    ax.axvline(18, color=GOLD, ls=":")
    ax.set(title="Eligibility rules can create a benefit cliff",
           xlabel="Illustrative annual earnings ($thousands)",
           ylabel="Earnings + illustrative cash benefit ($thousands)",
           xlim=(0, 60), ylim=(0, 68))
    ax.legend(frameon=False, fontsize=9.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.15)
    _save(fig, path, "HYPOTHETICAL MODEL • $9k benefit, $18k cutoff, illustrative 35% taper; no program calibration.")


def _measurement(path: Path) -> None:
    noncash = np.linspace(0, 12000, 121)
    cash_only, augmented = poverty_classification(18000, noncash, 25000)
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    ax.plot(noncash / 1000, np.full(noncash.shape, 18), color=RED,
            lw=3, label="Cash income (official measure concept)")
    ax.plot(noncash / 1000, (18000 + noncash) / 1000, color=TEAL,
            lw=3, label="Cash + assigned noncash value (NOT official)")
    ax.axhline(25, color=GOLD, lw=1.8, ls="--",
               label="Invented $25k poverty threshold")
    ax.axvline(7, color=GREY, ls=":")
    ax.set(title="A cash-only poverty rate cannot measure every benefit",
           xlabel="Illustrative assigned noncash benefit value ($thousands)",
           ylabel="Illustrative resources ($thousands)", xlim=(0, 12), ylim=(12, 34))
    ax.legend(frameon=False, loc="upper left", fontsize=9.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.15)
    assert cash_only.all() and augmented[0] and not augmented[-1]
    _save(fig, path, "HYPOTHETICAL MEASUREMENT • Noncash value is arbitrary; neither line estimates SPM resources.")


def _write_csv(path: Path, columns: tuple[str, ...], rows: Iterable[tuple[object, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def _results(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[object, ...]] = []
    files: tuple[tuple[str, tuple[str, ...], str], ...] = (
        ("census_poverty_all.csv", ("poverty_rate_percent",), "Census historical Table 2"),
        ("census_poverty_race.csv", ("white_poverty_rate_percent", "black_poverty_rate_percent"),
         "Census historical Table 2"),
        ("census_poverty_age.csv", ("under18_poverty_rate_percent",
         "age18_to64_poverty_rate_percent", "age65plus_poverty_rate_percent"),
         "Census historical Table 3"),
        ("cms_medicare_enrollment.csv", ("aged_hospital_insurance_enrollees",
         "disabled_hospital_insurance_enrollees", "total_hospital_insurance_enrollees"),
         "CMS HI 2010 historical enrollment table"),
    )
    for filename, cols, source in files:
        series = read_numeric_csv(filename, ("year",) + cols)
        for index, year in enumerate(series["year"]):
            for col in cols:
                val = series[col][index]
                # Preserve, but do not fabricate, missing source cells.
                rows.append((int(year), col, "" if np.isnan(val) else f"{val:.5f}",
                             "missing" if np.isnan(val) else "historical_descriptive", source))
    _write_csv(output / "source_based_summary.csv",
               ("year", "series", "value", "evidence_class", "source"), rows)

    rows_v = []
    for eligibility in (.65, .85, 1.0):
        for barrier in np.linspace(0, 1, 101):
            rows_v.append((f"{eligibility:.2f}", f"{barrier:.2f}",
                           f"{realized_voter_access(100000, eligibility, float(barrier)):.2f}",
                           "hypothetical"))
    _write_csv(output / "hypothetical_voter_access.csv",
               ("eligibility_share", "administrative_barrier", "people_with_access",
                "evidence_class"), rows_v)

    earnings = np.arange(0, 60001, 100)
    cliff = benefit_schedule(earnings, cliff=True)
    tapered = benefit_schedule(earnings, cliff=False)
    _write_csv(output / "hypothetical_benefit_schedule.csv",
               ("earnings_usd", "cutoff_benefit_usd", "tapered_benefit_usd",
                "cutoff_resources_usd", "tapered_resources_usd", "evidence_class"),
               ((int(y), f"{c:.2f}", f"{t:.2f}", f"{y+c:.2f}", f"{y+t:.2f}",
                 "hypothetical") for y, c, t in zip(earnings, cliff, tapered)))


def build_all(output: Path | None = None) -> list[Path]:
    """Generate all eight figures and audit CSVs under the given base folder."""
    base = ROOT if output is None else Path(output)
    figures = base / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _style()
    builders: tuple[Callable[[Path], None], ...] = (
        _national, _race, _age, _medicare, _timeline,
        _voter_access, _benefit_cliff, _measurement,
    )
    paths = [figures / name for name in FIGURE_NAMES]
    for draw, path in zip(builders, paths):
        draw(path)
    _results(base / "results")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT,
                        help="Destination base directory; defaults to article folder")
    args = parser.parse_args()
    for figure in build_all(args.output):
        print(figure)
