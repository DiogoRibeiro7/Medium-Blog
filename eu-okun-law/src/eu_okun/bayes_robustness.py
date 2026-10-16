"""Robustness of country differences and ranks across specifications.

Amendment 2 of ``BAYES_PROTOCOL.md`` (section 10). Everything is computed from the saved
posterior draws (``draws.npz``) and summaries (``fit.json``) of the fits:

- posterior mean, median, 90% interval and P(theta < 0) per country and fit, with a
  draw-by-draw check that ``theta = beta0 + sum_k b_k``;
- pairwise probabilities ``P(theta[i] < theta[j])`` from aligned joint draws, and the
  differences that are credible in every accepted fit of a specification set;
- rank distributions per fit and their envelope across a set;
- the four focus countries against the classical paired block bootstrap;
- convergence and posterior predictive diagnostics of every fit.

No model averaging is done, and a posterior probability is never read as causal.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun.bayes_model import AMENDMENT_1, AMENDMENT_2, PREREGISTERED
from eu_okun.bayes_report import (
    LoadedFit,
    credible_interval,
    load_fit,
    pairwise_probability,
    rank_distribution,
)

THRESHOLD = 0.95                 # "credible" pairwise probability (protocol, section 10.4)
SET_A: tuple[str, ...] = (*PREREGISTERED, *AMENDMENT_2)
SET_B: tuple[str, ...] = (*SET_A, *AMENDMENT_1)
FOCUS: tuple[str, ...] = ("ES", "LU", "NL", "PT")
CLASSICAL_ALPHA = 0.05
PREDICTIVE_STATISTICS = ("lag1_autocorrelation", "cross_country_correlation", "mean_sd")


# ----------------------------------------------------------------------------------
# One fit
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class CountrySummary:
    """Posterior summaries of one country's coefficient sum in one fit, next to OLS."""

    spec: str
    geo: str
    mean: float
    median: float
    low: float
    high: float
    p_negative: float
    rank_median: float
    rank_low: float
    rank_high: float
    ols: float
    ols_low: float
    ols_high: float


@dataclass
class FitSummary:
    """What the robustness analysis needs from one fit."""

    name: str
    accepted: bool
    countries: list[str]
    pairwise: np.ndarray                 # P(theta[i] < theta[j]), (N, N)
    ranks: np.ndarray                    # P(rank of i = r), (N, N)
    rows: list[CountrySummary]
    identity_error: float                # max |theta - beta0 - sum_k b_k| over draws
    diagnostics: dict[str, Any]
    predictive: dict[str, Any]

    @property
    def rank_interval(self) -> np.ndarray:
        """(N, 2) array of 90% rank interval bounds."""
        return np.asarray([[r.rank_low, r.rank_high] for r in self.rows])


def summarise_fit(fit: LoadedFit) -> FitSummary:
    """Country summaries, pairwise matrix and rank distribution of one fit."""
    theta = fit.draws["theta"]
    if theta.ndim != 2 or theta.shape[1] != len(fit.countries):
        raise ValueError(f"{fit.name}: theta draws do not match the countries.")
    identity = float(np.max(np.abs(theta - fit.draws["beta0"] - fit.draws["b"].sum(axis=-1))))
    ranks_per_draw = np.argsort(np.argsort(theta, axis=1), axis=1) + 1
    rows = []
    for i, geo in enumerate(fit.countries):
        low, high = credible_interval(theta[:, i])
        r_low, r_high = credible_interval(ranks_per_draw[:, i].astype(float))
        classical = fit.summary["classical_same_rows"][geo]
        rows.append(CountrySummary(
            fit.name, geo, float(theta[:, i].mean()), float(np.median(theta[:, i])), low, high,
            float(np.mean(theta[:, i] < 0)), float(np.median(ranks_per_draw[:, i])), r_low,
            r_high, float(classical["theta"]), float(classical["ci_low"]),
            float(classical["ci_high"])))
    return FitSummary(fit.name, fit.accepted, list(fit.countries), pairwise_probability(theta),
                      rank_distribution(theta), rows, identity, dict(fit.summary["diagnostics"]),
                      dict(fit.summary["predictive_check"]))


# ----------------------------------------------------------------------------------
# Across fits
# ----------------------------------------------------------------------------------


def credible_counts(fits: Sequence[FitSummary], threshold: float = THRESHOLD) -> np.ndarray:
    """Number of fits in which P(theta[i] < theta[j]) >= threshold, (N, N)."""
    return np.sum([f.pairwise >= threshold for f in fits], axis=0)


def surviving_pairs(fits: Sequence[FitSummary], threshold: float = THRESHOLD) -> np.ndarray:
    """(N, N) boolean: i credibly below j, in the same direction, in every fit (10.4)."""
    if not fits:
        raise ValueError("At least one accepted fit is needed.")
    return np.asarray(credible_counts(fits, threshold) == len(fits))


def rank_envelope(fits: Sequence[FitSummary]) -> np.ndarray:
    """(N, 2): lowest lower and highest upper 90% rank bound across fits (10.5)."""
    intervals = np.stack([f.rank_interval for f in fits])
    return np.column_stack([intervals[:, :, 0].min(axis=0), intervals[:, :, 1].max(axis=0)])


@dataclass(frozen=True)
class PairVerdict:
    """One focus pair: classical bootstrap against the Bayesian fits (10.6)."""

    country_a: str
    country_b: str
    classical_estimate: float           # a - b, pandemic excluded, block length 8
    classical_min_holm: float           # smallest Holm sign p over block lengths
    classical_apparent: bool
    full_estimate: float                # a - b, full sample, block length 8
    full_min_holm: float
    p_primary: float                    # P(theta_a < theta_b) in primary
    p_min: float                        # over accepted fits of set A
    p_max: float
    survives_a: bool
    survives_b: bool
    direction_agrees: bool | None       # classical sign vs Bayesian direction, if both apparent
    verdict: str


def classical_pairs(path: Path) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    """Rows of ``pairs_review.csv`` keyed by (sample, country_a, country_b)."""
    out: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            out.setdefault((row["sample"], row["country_a"], row["country_b"]), []).append(row)
    return out


def _classical(rows: Sequence[dict[str, str]]) -> tuple[float, float]:
    """Estimate at block length 8 and the smallest Holm p over block lengths."""
    estimate = next(float(r["estimate"]) for r in rows if r["block_length"] == "8")
    return estimate, min(float(r["holm_p"]) for r in rows)


def focus_verdicts(set_a: Sequence[FitSummary], set_b: Sequence[FitSummary],
                   classical: Mapping[tuple[str, str, str], list[dict[str, str]]],
                   primary: FitSummary) -> list[PairVerdict]:
    """Verdicts for the six pairs among PT, ES, NL and LU (protocol, section 10.6)."""
    index = {geo: i for i, geo in enumerate(primary.countries)}
    out = []
    for a, b in combinations(FOCUS, 2):
        i, j = index[a], index[b]
        estimate, holm = _classical(classical[("pandemic_excluded", a, b)])
        full_estimate, full_holm = _classical(classical[("full", a, b)])
        apparent = holm < CLASSICAL_ALPHA
        probabilities = [f.pairwise[i, j] for f in set_a]
        below_a = all(p >= THRESHOLD for p in probabilities)
        above_a = all(p <= 1 - THRESHOLD for p in probabilities)
        probabilities_b = [f.pairwise[i, j] for f in set_b]
        survives_b = (all(p >= THRESHOLD for p in probabilities_b)
                      or all(p <= 1 - THRESHOLD for p in probabilities_b))
        survives = below_a or above_a
        direction: bool | None = None
        if apparent and survives:
            direction = (estimate < 0) == below_a
        if apparent and survives and direction:
            verdict = "survives"
        elif apparent and survives:
            verdict = "contradicted (opposite direction)"
        elif apparent:
            verdict = "does not survive"
        elif survives:
            verdict = "credible only in the Bayesian fits"
        else:
            verdict = "not supported by either"
        out.append(PairVerdict(a, b, estimate, holm, apparent, full_estimate, full_holm,
                               float(primary.pairwise[i, j]), float(min(probabilities)),
                               float(max(probabilities)), survives, survives_b, direction,
                               verdict))
    return out


# ----------------------------------------------------------------------------------
# Review
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class RobustnessReplication:
    """An independent run's robustness tables (e.g. on another platform)."""

    label: str
    accepted: dict[str, bool]
    surviving_a: set[tuple[str, str]]
    focus: dict[tuple[str, str], str]


@dataclass
class RobustnessReview:
    """Everything the report is generated from."""

    fits: dict[str, FitSummary]
    set_a: list[str]
    set_b: list[str]
    surviving_a: np.ndarray
    surviving_b: np.ndarray
    counts_a: np.ndarray
    envelope_a: np.ndarray
    focus: list[PairVerdict]
    replication: RobustnessReplication | None = None
    missing: list[str] = field(default_factory=list)

    @property
    def primary(self) -> FitSummary:
        return self.fits["primary"]

    @property
    def countries(self) -> list[str]:
        return self.primary.countries


def pair_list(matrix: np.ndarray, countries: Sequence[str]) -> list[tuple[str, str]]:
    """(lower, higher) country pairs where ``matrix[i, j]`` is True."""
    return [(countries[i], countries[j]) for i, j in zip(*np.nonzero(matrix), strict=True)]


def load_replication(directory: Path) -> RobustnessReplication:
    """Read ``fits.csv``, ``surviving_pairs.csv`` and ``focus_pairs.csv`` of another run."""
    with (directory / "fits.csv").open(encoding="utf-8") as stream:
        accepted = {r["spec"]: r["accepted"] == "True" for r in csv.DictReader(stream)}
    with (directory / "surviving_pairs.csv").open(encoding="utf-8") as stream:
        surviving = {(r["lower"], r["higher"]) for r in csv.DictReader(stream)
                     if r["set"] == "A"}
    with (directory / "focus_pairs.csv").open(encoding="utf-8") as stream:
        focus = {(r["country_a"], r["country_b"]): r["verdict"] for r in csv.DictReader(stream)}
    readme = directory / "README.md"
    label = (readme.read_text(encoding="utf-8").splitlines()[0].lstrip("# ")
             if readme.exists() else directory.name)
    return RobustnessReplication(label, accepted, surviving, focus)


def build_robustness(fits_dir: Path, pairs_csv: Path,
                     replication_dir: Path | None = None) -> RobustnessReview:
    """Summarise every saved fit of set B and apply the rules of section 10."""
    fits: dict[str, FitSummary] = {}
    missing = []
    for name in SET_B:
        if (fits_dir / name / "fit.json").exists():
            fits[name] = summarise_fit(load_fit(fits_dir / name))
        else:
            missing.append(name)
    if "primary" not in fits or any(n not in fits for n in SET_A):
        raise FileNotFoundError(f"Set A is incomplete in {fits_dir}: missing "
                                f"{', '.join(n for n in SET_A if n not in fits)}.")
    countries = fits["primary"].countries
    if any(f.countries != countries for f in fits.values()):
        raise ValueError("Fits disagree on the country order.")
    set_a = [n for n in SET_A if fits[n].accepted]
    set_b = [n for n in SET_B if n in fits and fits[n].accepted]
    a_fits = [fits[n] for n in set_a]
    b_fits = [fits[n] for n in set_b]
    review = RobustnessReview(
        fits, set_a, set_b, surviving_pairs(a_fits), surviving_pairs(b_fits),
        credible_counts(a_fits), rank_envelope(a_fits),
        focus_verdicts(a_fits, b_fits, classical_pairs(pairs_csv), fits["primary"]),
        missing=missing)
    if replication_dir is not None:
        review.replication = load_replication(replication_dir)
    return review


# ----------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]], align: str = "") -> list[str]:
    align = align or "l" + "r" * (len(header) - 1)
    marks = ["---" if a == "l" else "---:" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(marks) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines + [""]


def _names(items: Sequence[str]) -> str:
    items = list(items)
    if not items:
        return "none"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _code(names: Sequence[str]) -> str:
    return _names([f"`{n}`" for n in names])


def render(review: RobustnessReview, assets: str, command: str) -> str:
    """Assemble the Markdown report."""
    countries = review.countries
    n = len(countries)
    total_pairs = n * (n - 1) // 2
    fits, primary = review.fits, review.primary
    failed = [name for name, f in fits.items() if not f.accepted]
    primary_credible = (primary.pairwise >= THRESHOLD).sum()
    lost = (primary.pairwise >= THRESHOLD) & ~review.surviving_a
    widths_primary = primary.rank_interval[:, 1] - primary.rank_interval[:, 0]
    widths_envelope = review.envelope_a[:, 1] - review.envelope_a[:, 0]
    identity = max(f.identity_error for f in fits.values())

    lines = ["# Hierarchical Bayes: robustness of country differences and ranks", "",
             "Generated by `eu-okun bayes-robustness` from the saved posterior draws of the "
             "frozen snapshot `snapshots/eurostat-2026-10-15`. The rules were registered in "
             "amendment 2 of [BAYES_PROTOCOL.md](../BAYES_PROTOCOL.md) (section 10) before any "
             "cross-specification pairwise or rank result was computed and before `lags_1` "
             "was fitted. Everything here is a descriptive conditional association: a "
             "posterior probability that one country's coefficient sum is below another's "
             "is not evidence of a causal difference in labour markets.", ""]

    # Summary
    lines += ["## Summary", "",
              f"- **Fits.** Set A (registered): {len(review.set_a)} of {len(SET_A)} "
              f"specifications accepted; set B (with quarter effects): {len(review.set_b)} "
              f"accepted. Failed: {_code(failed)}.",
              f"- **Country differences.** {primary_credible} of {total_pairs} pairs are "
              f"credible (P >= {THRESHOLD}) in `primary`; {int(review.surviving_a.sum())} "
              f"survive in every accepted fit of set A and {int(review.surviving_b.sum())} in "
              f"set B. {int(lost.sum())} pairs credible in `primary` do not survive.",
              f"- **Ranks.** The median width of the 90% rank interval is "
              f"{np.median(widths_primary):.0f} places in `primary` and "
              f"{np.median(widths_envelope):.0f} places across set A (envelope). No country's "
              "rank is established; positions are described only through surviving pairs.",
              "- **Focus countries.** " + "; ".join(
                  f"{v.country_a}-{v.country_b}: {v.verdict}" for v in review.focus) + ".",
              f"- **Coefficient sums.** `theta` equals `beta0 + sum_k b_k` draw by draw in "
              f"every fit (largest absolute deviation {identity:.1e}).", ""]
    if review.replication is not None:
        rep = review.replication
        here = set(pair_list(review.surviving_a, countries))
        lines += [f"- **Replication.** {rep.label}: {len(here & rep.surviving_a)} pairs survive "
                  f"in both runs, {len(here - rep.surviving_a)} only here and "
                  f"{len(rep.surviving_a - here)} only there; focus verdicts "
                  + ("identical." if all(rep.focus.get((v.country_a, v.country_b)) == v.verdict
                                         for v in review.focus) else "differ (section 7)."),
                  ""]

    # 1. Diagnostics
    lines += ["## 1. Specifications, convergence and predictive checks", "",
              "Acceptance (protocol, section 6): R-hat <= 1.01 (rank-normalised), bulk and tail "
              "ESS >= 400, no divergent transitions, E-BFMI >= 0.3 in every chain. Failed fits "
              "stay listed and are excluded from every robustness judgement. Posterior "
              "predictive p-values: share of replications at least as large as observed.", ""]
    rows = []
    for name, f in fits.items():
        d, pc = f.diagnostics, f.predictive
        rows.append([f"`{name}`", "A" if name in SET_A else "B",
                     "accepted" if f.accepted else "**failed**", f"{d['max_rhat']:.3f}",
                     f"{d['min_ess_bulk']:.0f}", f"{d['min_ess_tail']:.0f}", d["divergences"],
                     f"{d['min_bfmi']:.2f}", d.get("tree_depth_saturated", ""),
                     *(f"{pc['p_value'][k]:.2f}" for k in PREDICTIVE_STATISTICS),
                     f"{pc['coverage_90']:.2f}"])
    lines += _table(["Fit", "Set", "Status", "Max R-hat", "Min bulk ESS", "Min tail ESS",
                     "Divergences", "Min E-BFMI", "Tree depth hits", "p: lag-1 autocorr.",
                     "p: cross-country corr.", "p: mean SD", "One-step 90% coverage"], rows,
                    "lllrrrrrrrrrr")
    lines += [f"![Convergence and predictive diagnostics]({assets}/diagnostics.png)", ""]

    # 2. Country estimates
    lines += ["## 2. Country coefficient sums", "",
              "`primary`: posterior mean, median and 90% equal-tailed interval of each "
              "country's sum of GDP coefficients (lags 0-2), next to OLS with a 95% HAC "
              "interval on the same quarters. The last columns give the range of posterior "
              "medians across the accepted fits of set A. Alphabetical; no ranking.", ""]
    rows = []
    for i, geo in enumerate(countries):
        r = primary.rows[i]
        medians = [fits[name].rows[i].median for name in review.set_a]
        rows.append([geo, f"{r.mean:+.3f}", f"{r.median:+.3f}",
                     f"[{r.low:+.3f}, {r.high:+.3f}]", f"{r.p_negative:.3f}", f"{r.ols:+.3f}",
                     f"[{r.ols_low:+.3f}, {r.ols_high:+.3f}]", f"{min(medians):+.3f}",
                     f"{max(medians):+.3f}"])
    lines += _table(["Country", "Mean", "Median", "90% interval", "P(theta < 0)", "OLS",
                     "95% HAC interval", "Lowest median (set A)", "Highest median (set A)"],
                    rows)
    shrink = [abs(primary.rows[i].median - primary.rows[i].ols) for i in range(n)]
    toward_zero = sum(abs(primary.rows[i].median) < abs(primary.rows[i].ols) for i in range(n))
    lines += [f"Partial pooling and the error model move the medians a median "
              f"{np.median(shrink):.3f} from OLS; {toward_zero} of {n} move towards zero.", "",
              f"![Shrinkage and specification sensitivity]({assets}/estimates.png)", ""]

    # 3. Pairwise differences
    lines += ["## 3. Which country differences survive", "",
              f"A difference is credible in a fit when P(theta[i] < theta[j]) >= {THRESHOLD} "
              f"or <= {1 - THRESHOLD:.2f}, computed from aligned joint draws; it survives when "
              "it is credible in the same direction in every accepted fit of the set.", ""]
    rows = [["Set A (" + str(len(review.set_a)) + " accepted fits)",
             int(review.surviving_a.sum())],
            ["Set B (" + str(len(review.set_b)) + " accepted fits)",
             int(review.surviving_b.sum())],
            ["`primary` alone", int(primary_credible)]]
    lines += _table(["Specification set", f"Pairs credible in every fit (of {total_pairs})"],
                    rows)
    per_country = review.surviving_a.sum(axis=1) + review.surviving_a.sum(axis=0)
    rows = [[geo, int(per_country[i]),
             _names([countries[j] for j in np.nonzero(review.surviving_a[i])[0]]),
             _names([countries[j] for j in np.nonzero(review.surviving_a[:, i])[0]])]
            for i, geo in enumerate(countries)]
    lines += ["Robust differences per country (set A). 'Below' lists countries whose sum is "
              "credibly larger (less negative) in every accepted fit:", ""]
    lines += _table(["Country", "Robust differences", "Credibly below", "Credibly above"],
                    rows, "lrll")
    lines += [f"![Robustness of pairwise differences]({assets}/pairwise_robustness.png)", ""]
    lost_pairs = pair_list(lost, countries)
    lines += [f"{len(lost_pairs)} pairs credible in `primary` do not survive in set A"
              + (": " + ", ".join(f"{a}<{b}" for a, b in lost_pairs) + "." if lost_pairs else "."),
              ""]

    # 4. Ranks
    lines += ["## 4. Rank uncertainty", "",
              "Rank 1 is the most negative sum. For each country: the median rank and 90% "
              "interval in `primary`, and the envelope of the 90% intervals across the accepted "
              "fits of set A (model uncertainty included, no averaging). Alphabetical.", ""]
    rows = []
    for i, geo in enumerate(countries):
        r = primary.rows[i]
        lo, hi = review.envelope_a[i]
        rows.append([geo, f"{r.rank_median:.0f}", f"{r.rank_low:.0f}-{r.rank_high:.0f}",
                     f"{lo:.0f}-{hi:.0f}", f"{hi - lo + 1:.0f}"])
    lines += _table(["Country", "Median rank (primary)", "90% interval (primary)",
                     "Envelope (set A)", "Places spanned"], rows)
    lines += [f"![Rank intervals by specification]({assets}/rank_intervals.png)", ""]

    # 5. Focus countries
    lines += ["## 5. Portugal, Spain, the Netherlands and Luxembourg", "",
              "Classical: the review's paired block bootstrap (difference a - b at block "
              "length 8; smallest Holm-adjusted sign p over block lengths 4, 8 and 12). "
              "'Apparent' means Holm p < 0.05 in the pandemic-excluded sample, the closest to "
              "the Bayesian 2009-2019 window. Bayesian: P(theta_a < theta_b).", ""]
    rows = []
    for v in review.focus:
        rows.append([f"{v.country_a}-{v.country_b}",
                     f"{v.full_estimate:+.3f} ({v.full_min_holm:.3f})",
                     f"{v.classical_estimate:+.3f} ({v.classical_min_holm:.3f})",
                     "yes" if v.classical_apparent else "no", f"{v.p_primary:.3f}",
                     f"{v.p_min:.3f}-{v.p_max:.3f}", "yes" if v.survives_a else "no",
                     "yes" if v.survives_b else "no", f"**{v.verdict}**"])
    lines += _table(["Pair", "Classical, full (Holm p)", "Classical, excl. 2020-21 (Holm p)",
                     "Apparent", "P(a < b), primary", "Range over set A", "Survives (A)",
                     "Survives (B)", "Verdict"], rows, "lrrlrrlll")
    rows = []
    for geo in FOCUS:
        i = countries.index(geo)
        rows.append([geo, *(f"{fits[name].rows[i].rank_median:.0f} "
                            f"({fits[name].rows[i].rank_low:.0f}-"
                            f"{fits[name].rows[i].rank_high:.0f})" for name in review.set_a)])
    lines += ["Median rank (90% interval) of the focus countries in each accepted fit of set A:",
              ""]
    lines += _table(["Country", *(f"`{n}`" for n in review.set_a)], rows)
    lines += [f"![Focus pairs across specifications]({assets}/focus_pairs.png)", ""]

    number = 6
    if review.replication is not None:
        rep = review.replication
        here = set(pair_list(review.surviving_a, countries))
        only_here = ", ".join(f"{a}<{b}" for a, b in sorted(here - rep.surviving_a)) or "none"
        only_there = ", ".join(f"{a}<{b}" for a, b in sorted(rep.surviving_a - here)) or "none"
        differ = [name for name, f in fits.items()
                  if name in rep.accepted and rep.accepted[name] != f.accepted]
        lines += ["## 6. Replication on a second platform", "",
                  f"{rep.label}: the same snapshot, lock file, seeds and settings. Its tables "
                  "are in `replication-linux/`.", "",
                  f"- Acceptance differs for {_code(differ)}.",
                  f"- Pairs surviving in set A: {len(here & rep.surviving_a)} in both runs, "
                  f"{len(here - rep.surviving_a)} only here ({only_here}), "
                  f"{len(rep.surviving_a - here)} only there ({only_there}).",
                  "- Focus verdicts: " + "; ".join(
                      f"{v.country_a}-{v.country_b} {v.verdict} / "
                      f"{rep.focus.get((v.country_a, v.country_b), 'n/a')}" for v in review.focus)
                  + ".", ""]
        number += 1
    lines += [
        f"## {number}. Limitations", "",
        "- All results are conditional associations between GDP growth and unemployment "
        "changes; none identifies a causal effect or a structural shock.",
        "- Specification sets are fixed in advance, but they do not span every defensible "
        "model; a difference that survives them can still depend on assumptions shared by "
        "all of them (for example, the 2009-2019 window or the Student-t AR(1) family).",
        "- The fitted models understate the persistence of unemployment changes (posterior "
        "predictive lag-1 check), which can make intervals too narrow.",
        "- Ranks of coefficient sums are not rankings of labour markets.", "",
        f"## {number + 1}. Reproduction", "",
        "```bash",
        "poetry run eu-okun bayes-fit --data-dir snapshots/eurostat-2026-10-15 "
        "--output-dir results/bayes --specs all --retry-warmup 3000",
        command,
        "```", "",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------
# Figures and tables
# ----------------------------------------------------------------------------------


def write_figures(review: RobustnessReview, directory: Path, source: str) -> None:
    """Diagnostics, estimates, pairwise robustness, rank intervals and focus pairs."""
    from eu_okun.review_figures import AQUA, BLUE, INK2, MUTED, ORANGE, SURFACE, _finish, _pyplot

    plt = _pyplot()
    from matplotlib.lines import Line2D

    fits = review.fits
    names = list(fits)
    countries = review.countries
    n = len(countries)

    # Diagnostics: R-hat and minimum ESS per fit, then predictive p-values.
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), sharey=True)
    ys = np.arange(len(names))[::-1]
    for y, name in zip(ys, names, strict=True):
        d = fits[name].diagnostics
        colour = BLUE if fits[name].accepted else ORANGE
        axes[0].plot([d["max_rhat"]], [y], "o", ms=7, color=colour, mec=SURFACE, mew=1)
        axes[1].plot([min(d["min_ess_bulk"], d["min_ess_tail"])], [y], "o", ms=7, color=colour,
                     mec=SURFACE, mew=1)
        for k, (stat, marker) in enumerate(zip(PREDICTIVE_STATISTICS, ("o", "s", "D"),
                                               strict=True)):
            axes[2].plot([fits[name].predictive["p_value"][stat]], [y + 0.2 * (k - 1)], marker,
                         ms=5.5, color=(BLUE, AQUA, ORANGE)[k], mec=SURFACE, mew=0.8)
    axes[0].axvline(1.01, color=MUTED, lw=1, ls=(0, (3, 2)))
    axes[0].set_title("Max R-hat (threshold 1.01)")
    axes[1].axvline(400, color=MUTED, lw=1, ls=(0, (3, 2)))
    axes[1].set_title("Min bulk/tail ESS (threshold 400)")
    for x in (0.05, 0.95):
        axes[2].axvline(x, color=MUTED, lw=1, ls=(0, (3, 2)))
    axes[2].set_xlim(-0.03, 1.03)
    axes[2].set_title("Posterior predictive p-values")
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(names, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    fig.legend(handles=[
        Line2D([], [], marker="o", color=BLUE, lw=0, ms=7, label="Accepted fit"),
        Line2D([], [], marker="o", color=ORANGE, lw=0, ms=7, label="Failed fit"),
        Line2D([], [], marker="o", color=BLUE, lw=0, ms=5.5, label="p: lag-1 autocorrelation"),
        Line2D([], [], marker="s", color=AQUA, lw=0, ms=5.5,
               label="p: cross-country correlation"),
        Line2D([], [], marker="D", color=ORANGE, lw=0, ms=5.5, label="p: mean SD"),
    ], loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08), fontsize=8)
    fig.subplots_adjust(wspace=0.1, bottom=0.18)
    _finish(plt, fig, directory / "diagnostics.png", source, credit_y=-0.12)

    # Estimates: primary mean/median and interval, OLS, and the range over set A.
    order = sorted(range(n), key=lambda j: countries[j])
    ys = np.arange(n)[::-1]
    fig, ax = plt.subplots(figsize=(7.8, 9.2))
    primary = review.primary
    for y, i in zip(ys, order, strict=True):
        r = primary.rows[i]
        medians = [fits[name].rows[i].median for name in review.set_a]
        ax.plot([min(medians), max(medians)], [y + 0.25] * 2, color=AQUA, lw=4, alpha=0.5,
                solid_capstyle="round")
        ax.plot([r.ols_low, r.ols_high], [y - 0.25] * 2, color=ORANGE, lw=1.6, alpha=0.6,
                solid_capstyle="round")
        ax.plot([r.ols], [y - 0.25], "o", ms=5, color=ORANGE, mec=SURFACE, mew=1)
        ax.plot([r.low, r.high], [y] * 2, color=BLUE, lw=1.6, solid_capstyle="round")
        ax.plot([r.median], [y], "o", ms=5.5, color=BLUE, mec=SURFACE, mew=1.2)
        ax.plot([r.mean], [y], "|", ms=9, color=INK2, mew=1.4)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([countries[i] for i in order])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Sum of GDP coefficients (pp per pp), 2009-2019")
    ax.set_title("Posterior estimates, OLS and the spread across specifications")
    ax.legend(handles=[
        Line2D([], [], marker="o", color=BLUE, lw=1.6, ms=6,
               label="Primary: median and 90% interval"),
        Line2D([], [], marker="|", color=INK2, lw=0, ms=9, mew=1.4, label="Primary: mean"),
        Line2D([], [], marker="o", color=ORANGE, lw=1.6, ms=6, label="OLS, 95% HAC interval"),
        Line2D([], [], color=AQUA, lw=4, alpha=0.5,
               label="Range of medians across accepted fits of set A"),
    ], loc="upper left", bbox_to_anchor=(0.0, -0.075), fontsize=8.5)
    _finish(plt, fig, directory / "estimates.png", source, credit_y=-0.12)

    # Pairwise robustness: number of accepted set-A fits in which row < column credibly.
    k = len(review.set_a)
    fig, ax = plt.subplots(figsize=(8.6, 8.0))
    image = ax.imshow(review.counts_a, cmap="Blues", vmin=0, vmax=k, interpolation="nearest")
    rows_i, cols_j = np.nonzero(review.surviving_a)
    ax.plot(cols_j, rows_i, "s", ms=4.2, mfc="none", mec=INK2, mew=0.8)
    ax.set_xticks(range(n))
    ax.set_xticklabels(countries, fontsize=7.5, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(countries, fontsize=7.5)
    ax.grid(False)
    ax.set_title(f"Fits (of {k}) in which the row's sum is credibly below the column's; "
                 "outlined: all", fontsize=9.5)
    fig.colorbar(image, ax=ax, shrink=0.7, label="Accepted fits of set A")
    _finish(plt, fig, directory / "pairwise_robustness.png", source, credit_y=-0.02)

    # Rank intervals by specification, alphabetical.
    fig, ax = plt.subplots(figsize=(8.0, 9.4))
    offsets = np.linspace(-0.32, 0.32, len(review.set_a))
    for offset, name in zip(offsets, review.set_a, strict=True):
        colour = BLUE if name == "primary" else MUTED
        width = 2.0 if name == "primary" else 1.0
        for y, i in zip(ys, order, strict=True):
            r = fits[name].rows[i]
            ax.plot([r.rank_low, r.rank_high], [y + offset] * 2, color=colour, lw=width,
                    alpha=1.0 if name == "primary" else 0.6, solid_capstyle="round")
    ax.set_yticks(ys)
    ax.set_yticklabels([countries[i] for i in order])
    ax.set_xticks(range(1, n + 1, 2))
    ax.set_xlim(0.5, n + 0.5)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Rank (1 = most negative coefficient sum); 90% intervals")
    ax.set_title("Rank intervals are wide in every specification")
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=2, label="primary"),
                       Line2D([], [], color=MUTED, lw=1, alpha=0.6,
                              label="Other accepted fits of set A")],
              loc="upper left", bbox_to_anchor=(0.0, -0.07), fontsize=8.5)
    _finish(plt, fig, directory / "rank_intervals.png", source, credit_y=-0.1)

    # Focus pairs across specifications.
    pairs = review.focus
    fig, axes = plt.subplots(1, len(pairs), figsize=(13.0, 4.8), sharey=True)
    spec_names = review.set_b
    ys_spec = np.arange(len(spec_names))[::-1]
    index = {geo: i for i, geo in enumerate(countries)}
    for ax, v in zip(axes, pairs, strict=True):
        i, j = index[v.country_a], index[v.country_b]
        for y, name in zip(ys_spec, spec_names, strict=True):
            colour = BLUE if name in SET_A else AQUA
            ax.plot([fits[name].pairwise[i, j]], [y], "o", ms=6, color=colour, mec=SURFACE,
                    mew=1)
        for x in (0.05, 0.95):
            ax.axvline(x, color=MUTED, lw=1, ls=(0, (3, 2)))
        ax.set_xlim(-0.05, 1.05)
        ax.set_title(f"P({v.country_a} < {v.country_b})", fontsize=9.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ys_spec)
    axes[0].set_yticklabels(spec_names, fontsize=7.5)
    fig.legend(handles=[
        Line2D([], [], marker="o", color=BLUE, lw=0, ms=6, label="Set A"),
        Line2D([], [], marker="o", color=AQUA, lw=0, ms=6, label="Quarter-effect fits (set B)"),
    ], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.07))
    fig.subplots_adjust(wspace=0.08, bottom=0.16)
    _finish(plt, fig, directory / "focus_pairs.png", source, credit_y=-0.12)


def write_tables(review: RobustnessReview, directory: Path) -> None:
    """Country summaries, fits, surviving pairs, focus pairs and pairwise probabilities."""
    directory.mkdir(parents=True, exist_ok=True)

    def write(name: str, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        with (directory / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)

    summaries = [asdict(r) for f in review.fits.values() for r in f.rows]
    write("country_summaries.csv", list(summaries[0]), [list(r.values()) for r in summaries])
    write("fits.csv", ["spec", "set", "accepted", "max_rhat", "min_ess_bulk", "min_ess_tail",
                       "divergences", "min_bfmi", "identity_error"],
          [[name, "A" if name in SET_A else "B", f.accepted, f.diagnostics["max_rhat"],
            f.diagnostics["min_ess_bulk"], f.diagnostics["min_ess_tail"],
            f.diagnostics["divergences"], f.diagnostics["min_bfmi"], f.identity_error]
           for name, f in review.fits.items()])
    countries = review.countries
    write("surviving_pairs.csv", ["set", "lower", "higher"],
          [["A", a, b] for a, b in pair_list(review.surviving_a, countries)]
          + [["B", a, b] for a, b in pair_list(review.surviving_b, countries)])
    write("focus_pairs.csv", list(asdict(review.focus[0])),
          [list(asdict(v).values()) for v in review.focus])
    write("pairwise_probabilities.csv", ["spec", "row", *countries],
          [[name, geo, *(f"{p:.4f}" for p in f.pairwise[i])]
           for name, f in review.fits.items() for i, geo in enumerate(countries)])
    (directory / "rank_envelope.json").write_text(json.dumps({
        "set_a": review.set_a, "set_b": review.set_b,
        "envelope": {geo: [float(v) for v in review.envelope_a[i]]
                     for i, geo in enumerate(countries)},
    }, indent=1) + "\n", encoding="utf-8", newline="\n")


def write_robustness_report(fits_dir: Path, output_dir: Path, report: Path | None = None,
                            pairs_csv: Path = Path("docs/review/pairs_review.csv"),
                            figures: bool = True,
                            replication_dir: Path | None = None) -> RobustnessReview:
    """Build the review and write its tables, figures and Markdown report."""
    review = build_robustness(fits_dir, pairs_csv, replication_dir)
    report = report or output_dir / "BAYES_ROBUSTNESS.md"
    write_tables(review, output_dir)
    if figures:
        source = ("Source: Eurostat, snapshot 2026-10-15; hierarchical Bayesian model "
                  "(BAYES_PROTOCOL.md, amendment 2). Descriptive, not causal.")
        write_figures(review, output_dir, source)
    assets = Path(os.path.relpath(output_dir, report.parent)).as_posix()
    command = (f"poetry run eu-okun bayes-robustness --fits-dir {fits_dir.as_posix()} "
               f"--output-dir {output_dir.as_posix()} --report {report.as_posix()} "
               f"--pairs-csv {pairs_csv.as_posix()}"
               + (f" --replication-dir {replication_dir.as_posix()}"
                  if replication_dir is not None else ""))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render(review, assets, command), encoding="utf-8", newline="\n")
    return review

