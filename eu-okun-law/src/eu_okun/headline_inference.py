"""Small-sample inference for the classical headline sums (``HEADLINE_INFERENCE_PROTOCOL.md``).

The classical headline is the two-lag GDP coefficient sum of a per-country OLS on the
pandemic-excluded common sample, with a calendar-aware Newey-West t-statistic and a
normal reference. This module keeps the data, design and statistic and replaces only the
reference distribution, with the null-imposed sieve and block bootstraps of
``small_sample``. It adds bootstrap-t intervals and a Monte Carlo size study. NumPy only;
everything reads the frozen snapshot.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun.baseline import _read_panel
from eu_okun.contract import EU27, EurostatDataError
from eu_okun.small_sample import (
    Z_975,
    binomial_range,
    bootstrap_p,
    fit_sieve,
    hac_design,
    hac_t,
    null_block_test,
    null_sieve_test,
    restricted_fit,
    segment_starts,
    sieve_errors,
)
from eu_okun.stability import holm_adjust

ALPHA = 0.05
LAGS = 2
BANDWIDTH = 4
SENSITIVITY_BANDWIDTHS = (2, 8)
B_MAIN = 9999
B_SIZE = 199
R_SIZE = 200
SEED_TESTS = 20261022
SEED_INTERVALS = 20261023
SEED_SIZE = 20261024
TOLERANCE = 1e-10


@dataclass(frozen=True)
class Headline:
    """One country's headline regression on the pandemic-excluded common sample."""

    index: int
    geo: str
    x: np.ndarray
    y: np.ndarray
    periods: list[str]


def sum_contrast(width: int) -> np.ndarray:
    """Picks b0 + b1 + b2 from ``[1, g_t, g_{t-1}, g_{t-2}]``."""
    c = np.zeros(width)
    c[1: LAGS + 2] = 1.0
    return c


def headline_designs(data_dir: Path,
                     sample: str = "pandemic_excluded") -> tuple[list[Headline], str]:
    """The published headline designs, in alphabetical EU-27 order, and the last quarter.

    ``sample`` is a classical review sample; the registered headline is pandemic_excluded
    (MEASUREMENT_PROTOCOL.md repeats it on quality_screened).
    """
    from eu_okun.audit import audit_snapshot
    from eu_okun.review import SAMPLES, read_flags, sample_designs

    end = audit_snapshot(data_dir)["analysis_period"]["end"]
    if not isinstance(end, str):
        raise EurostatDataError("Invalid final quarter in audit.")
    path = data_dir / "processed" / "eu27_quarterly.csv"
    spec = next(s for s in SAMPLES if s.name == sample)
    designs = sample_designs(_read_panel(path), read_flags(path), spec, end)
    return [Headline(i, geo, *designs[(geo, LAGS, False)]) for i, geo in enumerate(EU27)], end


def check_published(heads: Sequence[Headline], grid_csv: Path,
                    sample: str = "pandemic_excluded") -> float:
    """Largest gap to the committed sums and standard errors; raises above the tolerance."""
    with grid_csv.open(encoding="utf-8") as stream:
        grid = {r["geo"]: r for r in csv.DictReader(stream)
                if r["sample"] == sample and r["gdp_lags"] == str(LAGS)
                and r["participation"] == "False"}
    gap = 0.0
    for head in heads:
        design = hac_design(head.x, head.periods, sum_contrast(head.x.shape[1]), BANDWIDTH)
        estimate, se, _ = hac_t(design, head.y)
        published = grid[head.geo]
        gap = max(gap, abs(float(estimate[0]) - float(published["gdp_sum"])),
                  abs(float(se[0]) - float(published["se"])))
    if gap > TOLERANCE:
        raise ValueError(f"The implementation does not reproduce the published headline "
                         f"(largest gap {gap:.2e}); the run stops (protocol, section 2).")
    return gap


@dataclass(frozen=True)
class HeadlineTest:
    """One headline sum under the normal and bootstrap reference distributions."""

    geo: str
    theta: float
    se: float
    t: float
    p_normal: float
    p_sieve: float
    p_block: float
    p_sieve_bw2: float
    p_sieve_bw8: float
    normal_low: float
    normal_high: float
    boot_low: float
    boot_high: float
    rho: str
    rho_unrestricted: str               # not registered: diagnostic added after the run
    p_normal_holm: float = 1.0
    p_sieve_holm: float = 1.0
    p_block_holm: float = 1.0

    @property
    def negative(self) -> bool:
        """Section 6: negative under both schemes, unadjusted."""
        return self.theta < 0 and self.p_sieve <= ALPHA and self.p_block <= ALPHA

    @property
    def negative_holm(self) -> bool:
        return self.theta < 0 and self.p_sieve_holm <= ALPHA and self.p_block_holm <= ALPHA

    @property
    def scheme_dependent(self) -> bool:
        return self.theta < 0 and (self.p_sieve <= ALPHA) != (self.p_block <= ALPHA)


def _interval(design: Any, head: Headline, theta: float, se: float,
              draws: int) -> tuple[float, float]:
    """Section 4: bootstrap-t from the unrestricted sieve."""
    beta = design.projector @ head.y
    resid = head.y - design.x @ beta
    sieve = fit_sieve(resid, segment_starts(head.periods))
    rng = np.random.default_rng([SEED_INTERVALS, head.index])
    y_star = (design.x @ beta)[:, None] + sieve_errors(sieve, resid, draws, rng)
    e_star, se_star, _ = hac_t(design, y_star)
    q_low, q_high = np.quantile((e_star - theta) / se_star, [0.025, 0.975])
    return theta - float(q_high) * se, theta - float(q_low) * se


def headline_tests(heads: Sequence[Headline], draws: int = B_MAIN) -> list[HeadlineTest]:
    """Sections 3 and 4 for every country."""
    rows = []
    for head in heads:
        width = head.x.shape[1]
        starts = segment_starts(head.periods)
        design = hac_design(head.x, head.periods, sum_contrast(width), BANDWIDTH)
        estimate, se_arr, _ = hac_t(design, head.y)
        theta, se = float(estimate[0]), float(se_arr[0])
        t_obs, t_star, sieve = null_sieve_test(
            design, head.y, starts, draws, np.random.default_rng([SEED_TESTS, head.index]))
        t_block = null_block_test(design, head.y, starts, draws,
                                  np.random.default_rng([SEED_TESTS, head.index, 100]))
        by_bandwidth = {}
        for bandwidth in SENSITIVITY_BANDWIDTHS:
            alt = hac_design(head.x, head.periods, sum_contrast(width), bandwidth)
            t_alt, t_alt_star, _ = null_sieve_test(
                alt, head.y, starts, draws,
                np.random.default_rng([SEED_TESTS, head.index, bandwidth]))
            by_bandwidth[bandwidth] = bootstrap_p(t_alt, t_alt_star)
        boot_low, boot_high = _interval(design, head, theta, se, draws)
        free = fit_sieve(head.y - design.x @ (design.projector @ head.y), starts)
        rows.append(HeadlineTest(
            head.geo, theta, se, t_obs, math.erfc(abs(t_obs) / math.sqrt(2.0)),
            bootstrap_p(t_obs, t_star), bootstrap_p(t_obs, t_block),
            by_bandwidth[2], by_bandwidth[8], theta - Z_975 * se, theta + Z_975 * se,
            boot_low, boot_high, " / ".join(f"{r:+.2f}" for r in sieve.period_rho),
            " / ".join(f"{r:+.2f}" for r in free.period_rho)))
    holm_normal = holm_adjust([r.p_normal for r in rows])
    holm_sieve = holm_adjust([r.p_sieve for r in rows])
    holm_block = holm_adjust([r.p_block for r in rows])
    return [HeadlineTest(**{**{k: getattr(r, k) for k in r.__dataclass_fields__},
                            "p_normal_holm": a, "p_sieve_holm": s, "p_block_holm": b})
            for r, a, s, b in zip(rows, holm_normal, holm_sieve, holm_block, strict=True)]


@dataclass(frozen=True)
class SizeResult:
    geo: str
    normal: float
    bootstrap: float


def size_study(heads: Sequence[Headline], replications: int = R_SIZE,
               draws: int = B_SIZE) -> list[SizeResult]:
    """Section 5: both tests on data simulated from each country's fitted null."""
    out = []
    for head in heads:
        design = hac_design(head.x, head.periods, sum_contrast(head.x.shape[1]), BANDWIDTH)
        starts = segment_starts(head.periods)
        rng = np.random.default_rng([SEED_SIZE, head.index])
        beta_r, resid = restricted_fit(design, head.y)
        sieve = fit_sieve(resid, starts)
        simulated = (design.x @ beta_r)[:, None] + sieve_errors(sieve, resid, replications,
                                                                 rng)
        t_sim = hac_t(design, simulated)[2]
        rejected = 0
        for r in range(replications):
            t_obs, t_star, _ = null_sieve_test(design, simulated[:, r], starts, draws, rng)
            rejected += bootstrap_p(t_obs, t_star) <= ALPHA
        out.append(SizeResult(head.geo, float(np.mean(np.abs(t_sim) > Z_975)),
                              rejected / replications))
    return out


@dataclass
class HeadlineReview:
    tests: list[HeadlineTest]
    size: list[SizeResult]
    check_gap: float
    quarters: int
    first: str
    last: str

    def summary(self) -> dict[str, Any]:
        t = self.tests
        size_n = len(self.size) * R_SIZE
        return {
            "countries": len(t),
            "quarters": self.quarters,
            "sample": [self.first, self.last],
            "normal_negative": sum(r.theta < 0 and r.p_normal <= ALPHA for r in t),
            "normal_negative_holm": sum(r.theta < 0 and r.p_normal_holm <= ALPHA for r in t),
            "bootstrap_negative": sum(r.negative for r in t),
            "bootstrap_negative_holm": sum(r.negative_holm for r in t),
            "scheme_dependent": [r.geo for r in t if r.scheme_dependent],
            "sieve_negative": sum(r.theta < 0 and r.p_sieve <= ALPHA for r in t),
            "block_negative": sum(r.theta < 0 and r.p_block <= ALPHA for r in t),
            "lost": [r.geo for r in t if r.theta < 0 and r.p_normal <= ALPHA and not r.negative],
            "bandwidth_negative": {
                str(bw): {
                    "unadjusted": sum(r.theta < 0 and getattr(r, f"p_sieve_bw{bw}") <= ALPHA
                                      for r in t),
                    "holm": sum(r.theta < 0 and p <= ALPHA for r, p in zip(
                        t, holm_adjust([getattr(r, f"p_sieve_bw{bw}") for r in t]),
                        strict=True)),
                } for bw in SENSITIVITY_BANDWIDTHS},
            "boot_intervals_below_zero": sum(r.boot_high < 0 for r in t),
            "size_pooled": {
                "normal": float(np.mean([s.normal for s in self.size])),
                "bootstrap": float(np.mean([s.bootstrap for s in self.size])),
            },
            "size_range_pooled": list(binomial_range(size_n)),
            "size_range_country": list(binomial_range(R_SIZE)),
            "normal_over_rejecting": sum(s.normal > binomial_range(R_SIZE)[1]
                                         for s in self.size),
            "rho_gap_pre_median": float(np.median([
                float(r.rho.split(" / ")[0]) - float(r.rho_unrestricted.split(" / ")[0])
                for r in t])),
            "point_negative": sum(r.theta < 0 for r in t),
            "check_largest_gap": self.check_gap,
            "settings": {"bandwidth": BANDWIDTH, "B": B_MAIN, "B_size": B_SIZE,
                         "R_size": R_SIZE, "seeds": [SEED_TESTS, SEED_INTERVALS, SEED_SIZE]},
        }


def build_review(data_dir: Path, grid_csv: Path, size: bool = True) -> HeadlineReview:
    heads, _ = headline_designs(data_dir)
    gap = check_published(heads, grid_csv)
    tests = headline_tests(heads)
    sizes = size_study(heads) if size else []
    return HeadlineReview(tests, sizes, gap, len(heads[0].y), heads[0].periods[0],
                          heads[0].periods[-1])


# ----------------------------------------------------------------------------------
# Tables and report
# ----------------------------------------------------------------------------------


def _fmt(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}"


def _p(x: float) -> str:
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def render(review: HeadlineReview, command: str) -> str:
    s = review.summary()
    lines = [
        "# Small-sample inference for the classical headline sums", "",
        "Generated by `eu-okun headline-inference` from the frozen snapshot, under "
        "[HEADLINE_INFERENCE_PROTOCOL.md](../HEADLINE_INFERENCE_PROTOCOL.md), which was "
        "registered before any bootstrap or simulation. Only the reference distribution "
        "of the published statistic changes. Descriptive associations, not causal "
        "effects.", "",
        "## Summary", "",
        f"- **Sample.** {s['quarters']} quarters per country, {s['sample'][0]} to "
        f"{s['sample'][1]}, 2020-2021 and their lag history removed; the published "
        f"estimates and standard errors are reproduced (largest gap "
        f"{s['check_largest_gap']:.1e}).",
        f"- **Negative sums, unadjusted.** Normal reference: {s['normal_negative']} of "
        f"{s['countries']}. Sieve and block bootstraps: {s['bootstrap_negative']}"
        + (f"; no longer negative: {', '.join(s['lost'])}" if s["lost"] else "") + ".",
        f"- **With Holm over the {s['countries']} countries.** Normal: "
        f"{s['normal_negative_holm']}. Bootstraps: {s['bootstrap_negative_holm']}.",
        f"- **Scheme-dependent.** "
        f"{', '.join(s['scheme_dependent']) if s['scheme_dependent'] else 'none'}.",
        f"- **Size of the normal test** under the fitted nulls: "
        f"{100 * s['size_pooled']['normal']:.1f}% at a nominal 5% (pooled over "
        f"{len(review.size) * R_SIZE:,} simulated tests); the bootstrap test: "
        f"{100 * s['size_pooled']['bootstrap']:.1f}% (binomial 95% range "
        f"{100 * s['size_range_pooled'][0]:.1f}-{100 * s['size_range_pooled'][1]:.1f}%). "
        f"{s['normal_over_rejecting']} of {len(review.size)} countries' normal tests "
        "over-reject.",
        f"- **Bootstrap-t intervals below zero:** {s['boot_intervals_below_zero']} "
        "(descriptive).",
        "- **Bandwidth (sieve).** "
        + "; ".join(f"bandwidth {bw}: {v['unadjusted']} unadjusted, {v['holm']} after Holm"
                    for bw, v in s["bandwidth_negative"].items()) + ".", "",
        "## Countries", "",
        "Two-lag sum, normal 95% interval, bootstrap-t 95% interval, and two-sided p-values "
        "(normal, sieve, block; Holm-adjusted in brackets). Verdict: negative when "
        "negative under both schemes (protocol, section 6).", "",
        "| Country | Sum | Normal 95% | Bootstrap-t 95% | p normal | p sieve | p block | "
        "AR(1), restricted | AR(1), unrestricted | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in review.tests:
        verdict = ("negative" if r.negative else "scheme-dependent" if r.scheme_dependent
                   else "not distinguishable from zero")
        lines.append(
            f"| {r.geo} | {_fmt(r.theta)} | [{_fmt(r.normal_low)}, {_fmt(r.normal_high)}] | "
            f"[{_fmt(r.boot_low)}, {_fmt(r.boot_high)}] | {_p(r.p_normal)} "
            f"({_p(r.p_normal_holm)}) | {_p(r.p_sieve)} ({_p(r.p_sieve_holm)}) | "
            f"{_p(r.p_block)} ({_p(r.p_block_holm)}) | {r.rho} | {r.rho_unrestricted} | "
            f"{verdict} |")
    lines += [
        "", "AR(1) by segment: before 2020 / after 2021. The unrestricted column is a "
        "diagnostic added after the registered run (protocol change log).", "",
        "## Why the null-imposed p-values are conservative for large sums (not registered)",
        "",
        "Imposing `theta = 0` moves the omitted GDP effect into the restricted residuals. "
        "GDP growth is persistent, so for a country with a large sum this raises the "
        "residual AR(1) of the null data-generating process and widens the bootstrap "
        "distribution of t. Before 2020 the restricted AR(1) exceeds the unrestricted one by "
        f"a median {s['rho_gap_pre_median']:+.2f} (Greece: "
        + next(f"{r.rho.split(' / ')[0]} against {r.rho_unrestricted.split(' / ')[0]}"
               for r in review.tests if r.geo == "EL")
        + "). The registered p-values therefore understate the evidence for large sums. The "
        "size study draws its null from the same restricted residuals, so its rejection "
        "rates are those of that more persistent null; the over-rejection of the normal "
        "reference is smaller on a less persistent one. A post-hoc repeat with errors from "
        "the unrestricted residuals is in [NULL_RESIDUALS.md](NULL_RESIDUALS.md). The "
        "bootstrap-t intervals, which use the unrestricted residuals, do not have this "
        "problem.",
    ]
    if review.size:
        lines += ["", "## Size study", "",
                  f"Rejection rates at a nominal 5%, {R_SIZE} data sets per country "
                  f"simulated from its fitted null (binomial 95% range for {R_SIZE} trials: "
                  f"{100 * s['size_range_country'][0]:.1f}-"
                  f"{100 * s['size_range_country'][1]:.1f}%).", "",
                  "| Country | Normal | Sieve bootstrap |", "| --- | ---: | ---: |"]
        lines += [f"| {z.geo} | {100 * z.normal:.1f}% | {100 * z.bootstrap:.1f}% |"
                  for z in review.size]
    lines += ["", "## Reproduction", "", "```bash", command, "```", ""]
    return "\n".join(lines)


def write_report(data_dir: Path, grid_csv: Path, output_dir: Path,
                 report: Path | None = None, size: bool = True) -> HeadlineReview:
    review = build_review(data_dir, grid_csv, size)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write(name: str, rows: Sequence[dict[str, Any]]) -> None:
        with (output_dir / name).open("w", encoding="utf-8", newline="\n") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    write("tests.csv", [{**asdict(r), "negative": r.negative,
                         "negative_holm": r.negative_holm,
                         "scheme_dependent": r.scheme_dependent} for r in review.tests])
    if review.size:
        write("size.csv", [asdict(z) for z in review.size])
    (output_dir / "summary.json").write_text(json.dumps(review.summary(), indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
    if report is not None:
        command = (f"eu-okun headline-inference --data-dir {data_dir.as_posix()} "
                   f"--grid {grid_csv.as_posix()} --output-dir {output_dir.as_posix()} "
                   f"--report {report.as_posix()}")
        report.write_text(render(review, command), encoding="utf-8", newline="\n")
    return review
