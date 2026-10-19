"""Does the quarterly unemployment rate inherit monthly smoothing? (``SMOOTHING_PROTOCOL.md``)

Q1 checks whether each country's quarterly rate is the average of its monthly rate. Two
tests compare countries whose monthly series are smoothed (trend, centred moving average,
moving annual averages) with countries that publish direct monthly survey estimates:

- P1: the lag-1 autocorrelation of quarterly unemployment changes, as in the Bayesian
  predictive check;
- P2: the share of the classical two-lag response carried by the lags.

Exact permutation tests over countries, Holm over the two. Reads only frozen snapshots.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun.contract import EU27, quarter_sequence
from eu_okun.monthly_data import UNEMPLOYMENT_METHOD
from eu_okun.stability import holm_adjust

SMOOTHED = ("DE", "NL", "EE", "PT", "HR", "LU")           # groups T, M and B
DIRECT = tuple(g for g, m in UNEMPLOYMENT_METHOD.items() if m == "D")
INTERPOLATED = tuple(g for g, m in UNEMPLOYMENT_METHOD.items() if m == "I")
ALPHA = 0.05
CHECK_FIRST, CHECK_LAST = "2009-Q1", "2026-Q2"
Z_975 = 1.959963984540054


# ----------------------------------------------------------------------------------
# Q1: is the quarterly rate the average of the monthly rate?
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Inheritance:
    geo: str
    method: str
    quarters: int
    mean_abs_diff: float
    max_abs_diff: float
    change_corr: float

    @property
    def average_of_monthly(self) -> bool:
        return self.change_corr >= 0.95 and self.mean_abs_diff <= 0.1


def _months(quarter: str) -> list[str]:
    year, q = quarter.split("-Q")
    first = 3 * (int(q) - 1) + 1
    return [f"{year}-{m:02d}" for m in range(first, first + 3)]


def inheritance(quarterly: Mapping[tuple[str, str], Mapping[str, float | None]],
                monthly: Mapping[tuple[str, str], Mapping[str, float | None]],
                geo: str) -> Inheritance:
    quarters = quarter_sequence(CHECK_FIRST, CHECK_LAST)
    u_q, m_q = [], []
    for q in quarters:
        value = quarterly.get((geo, q), {}).get("unemployment")
        months = [monthly.get((geo, m), {}).get("unemployment") for m in _months(q)]
        if value is None or any(v is None for v in months):
            u_q.append(math.nan)
            m_q.append(math.nan)
            continue
        u_q.append(float(value))
        m_q.append(float(np.mean([float(v) for v in months if v is not None])))
    u, m = np.asarray(u_q), np.asarray(m_q)
    ok = np.isfinite(u) & np.isfinite(m)
    diff = np.abs(u[ok] - m[ok])
    du, dm = np.diff(u), np.diff(m)
    pair = np.isfinite(du) & np.isfinite(dm)
    corr = (float(np.corrcoef(du[pair], dm[pair])[0, 1]) if pair.sum() > 3
            and du[pair].std() > 0 and dm[pair].std() > 0 else math.nan)
    return Inheritance(geo, UNEMPLOYMENT_METHOD.get(geo, "?"), int(ok.sum()),
                       float(diff.mean()) if diff.size else math.nan,
                       float(diff.max()) if diff.size else math.nan, corr)


# ----------------------------------------------------------------------------------
# Country statistics
# ----------------------------------------------------------------------------------


def bayes_autocorrelations(data: Any) -> dict[str, float]:
    """Lag-1 autocorrelation of y per country, exactly as ``bayes_fit._masked_stats``."""
    mask = data.mask
    pairs = mask[:, 1:] & mask[:, :-1] & ~data.segment_start[None, 1:]
    out = {}
    for i, geo in enumerate(data.countries):
        a, b = data.y[i, 1:][pairs[i]], data.y[i, :-1][pairs[i]]
        out[geo] = (float(np.corrcoef(a, b)[0, 1])
                    if a.size > 3 and a.std() > 0 and b.std() > 0 else math.nan)
    return out


def full_sample_autocorrelations(panel: Mapping[tuple[str, str], Mapping[str, float | None]],
                                 flags: Any, end: str) -> dict[str, float]:
    """Sensitivity: 2009-2026, every level outside 2020-2021 and no flagged change."""
    from eu_okun.review import flagged_change, in_pandemic

    quarters = quarter_sequence("2009-Q1", end)
    out = {}
    for geo in EU27:
        du = []
        for t, q in enumerate(quarters):
            if t == 0:
                du.append(math.nan)
                continue
            now = panel.get((geo, q), {}).get("unemployment")
            before = panel.get((geo, quarters[t - 1]), {}).get("unemployment")
            usable = (now is not None and before is not None and not in_pandemic(q)
                      and not in_pandemic(quarters[t - 1])
                      and not flagged_change(flags, geo, q, 0))
            du.append(float(now) - float(before) if usable and now is not None
                      and before is not None else math.nan)
        x = np.asarray(du)
        pair = np.isfinite(x[1:]) & np.isfinite(x[:-1])
        a, b = x[1:][pair], x[:-1][pair]
        out[geo] = (float(np.corrcoef(a, b)[0, 1])
                    if a.size > 3 and a.std() > 0 and b.std() > 0 else math.nan)
    return out


@dataclass(frozen=True)
class LagShare:
    geo: str
    theta: float
    low: float
    high: float
    share: float          # NaN unless the 95% interval lies below zero


def lag_shares(designs: Mapping[tuple[str, int, bool], Any]) -> dict[str, LagShare]:
    """P2: (b1 + b2) / theta of the classical headline regression."""
    from eu_okun.baseline import fit_hac_covariance

    out = {}
    for geo in EU27:
        x, y, dates = designs[(geo, 2, False)]
        beta, cov, _, _ = fit_hac_covariance(x, y, maxlags=4, periods=dates)
        c = np.array([0.0, 1.0, 1.0, 1.0])
        theta = float(c @ beta)
        se = math.sqrt(float(c @ cov @ c))
        low, high = theta - Z_975 * se, theta + Z_975 * se
        share = float((beta[2] + beta[3]) / theta) if high < 0 else math.nan
        out[geo] = LagShare(geo, theta, low, high, share)
    return out


# ----------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------


def permutation_test(first: Sequence[float], second: Sequence[float]) -> tuple[float, float,
                                                                                float]:
    """Mean(first) - mean(second) and its exact one-sided p-values (greater, less)."""
    pooled = list(first) + list(second)
    k, observed = len(first), float(np.mean(first) - np.mean(second))
    greater = less = total = 0
    for chosen in itertools.combinations(range(len(pooled)), k):
        a = [pooled[i] for i in chosen]
        b = [pooled[i] for i in range(len(pooled)) if i not in chosen]
        stat = float(np.mean(a) - np.mean(b))
        greater += stat >= observed - 1e-12
        less += stat <= observed + 1e-12
        total += 1
    return observed, greater / total, less / total


@dataclass(frozen=True)
class Test:
    setting: str
    key: str
    text: str
    smoothed: tuple[str, ...]
    comparison: tuple[str, ...]
    mean_smoothed: float
    mean_comparison: float
    difference: float
    p_greater: float
    p_less: float
    holm: float = 1.0
    holm_opposite: float = 1.0
    verdict: str = "not determined"


def _test(setting: str, key: str, text: str, values: Mapping[str, float],
          comparison_group: Sequence[str]) -> Test:
    s = [g for g in SMOOTHED if math.isfinite(values.get(g, math.nan))]
    c = [g for g in comparison_group if math.isfinite(values.get(g, math.nan))]
    diff, p_greater, p_less = permutation_test([values[g] for g in s], [values[g] for g in c])
    return Test(setting, key, text, tuple(s), tuple(c), float(np.mean([values[g] for g in s])),
                float(np.mean([values[g] for g in c])), diff, p_greater, p_less)


def _decide(tests: Sequence[Test]) -> list[Test]:
    holm = holm_adjust([t.p_greater for t in tests])
    opposite = holm_adjust([t.p_less for t in tests])
    out = []
    for t, h, o in zip(tests, holm, opposite, strict=True):
        verdict = ("supported" if h <= ALPHA else "contradicted" if o <= ALPHA
                   else "not determined")
        out.append(Test(**{**asdict(t), "holm": h, "holm_opposite": o, "verdict": verdict}))
    return out


@dataclass
class Study:
    inheritance: list[Inheritance]
    autocorrelation: dict[str, float]
    autocorrelation_full: dict[str, float]
    shares: dict[str, LagShare]
    primary: list[Test]
    sensitivity: list[Test]
    misfit: dict[str, float]
    luxembourg: dict[str, float]


def run(data_dir: Path, monthly_dir: Path, review_dir: Path, bayes_dir: Path) -> Study:
    from eu_okun.audit import audit_snapshot
    from eu_okun.baseline import _read_panel
    from eu_okun.bayes_data import build_bayes_data
    from eu_okun.monthly_data import audit_monthly_snapshot, load_monthly_panel
    from eu_okun.review import SAMPLES, read_flags, sample_designs

    end = str(audit_snapshot(data_dir)["analysis_period"]["end"])
    audit_monthly_snapshot(monthly_dir)
    path = data_dir / "processed" / "eu27_quarterly.csv"
    panel, flags = _read_panel(path), read_flags(path)
    monthly = load_monthly_panel(monthly_dir)
    inherit = [inheritance(panel, monthly, g) for g in EU27]
    auto = bayes_autocorrelations(build_bayes_data(panel, flags))
    auto_full = full_sample_autocorrelations(panel, flags, end)
    spec = next(s for s in SAMPLES if s.name == "pandemic_excluded")
    shares = lag_shares(sample_designs(panel, flags, spec, end))
    share_values = {g: s.share for g, s in shares.items()}
    p1 = "lag-1 autocorrelation of quarterly unemployment changes"
    p2 = "lag share of the two-lag unemployment response"
    primary = _decide([
        _test("primary", "P1", f"Smoothed countries: higher {p1}", auto, DIRECT),
        _test("primary", "P2", f"Smoothed countries: higher {p2}", share_values,
              DIRECT)])
    sensitivity = _decide([
        _test("direct and interpolated", "P1", f"Smoothed countries: higher {p1}", auto,
              DIRECT + INTERPOLATED),
        _test("direct and interpolated", "P2", f"Smoothed countries: higher {p2}",
              share_values, DIRECT + INTERPOLATED)]) + _decide([
        _test("2009-2026", "P1", f"Smoothed countries: higher {p1}", auto_full,
              DIRECT)])
    diagnostics = json.loads((bayes_dir / "diagnostics.json").read_text(encoding="utf-8"))
    replicated = diagnostics["primary"]["predictive_check"]["replicated_mean"][
        "lag1_autocorrelation"]
    finite = {g: v for g, v in auto.items() if math.isfinite(v)}
    misfit = {
        "observed_all": float(np.mean(list(finite.values()))),
        "observed_without_smoothed": float(np.mean([v for g, v in finite.items()
                                                    if g not in SMOOTHED])),
        "observed_direct": float(np.mean([v for g, v in finite.items() if g in DIRECT])),
        "observed_smoothed": float(np.mean([v for g, v in finite.items() if g in SMOOTHED])),
        "replicated_primary": float(replicated),
    }
    with (review_dir / "grid.csv").open(encoding="utf-8") as stream:
        grid = {(r["gdp_lags"]): r for r in csv.DictReader(stream)
                if r["sample"] == "pandemic_excluded" and r["geo"] == "LU"
                and r["participation"] == "False"}
    lu = next(i for i in inherit if i.geo == "LU")
    luxembourg = {"autocorrelation": auto["LU"], "change_corr": lu.change_corr,
                  "mean_abs_diff": lu.mean_abs_diff,
                  "sum_k2": float(grid["2"]["gdp_sum"]), "sum_k4": float(grid["4"]["gdp_sum"]),
                  "sum_k4_low": float(grid["4"]["ci_low"]),
                  "sum_k4_high": float(grid["4"]["ci_high"])}
    return Study(inherit, auto, auto_full, shares, primary, sensitivity, misfit, luxembourg)


# ----------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------


def _f(x: float, digits: int = 2) -> str:
    return "n/a" if not math.isfinite(x) else f"{x:+.{digits}f}"


def interpretation(study: Study) -> list[str]:
    by = {t.key: t.verdict for t in study.primary}
    nl = next(i for i in study.inheritance if i.geo == "NL")
    inherited = [i.geo for i in study.inheritance if i.geo in SMOOTHED and i.average_of_monthly]
    lines = ["- **Inheritance (Q1):** "
             + (f"{', '.join(inherited)} of the smoothed countries "
                f"{'is' if len(inherited) == 1 else 'are'} classed as the average of the "
                "monthly rate." if inherited else
                "none of the six smoothed countries' quarterly rates is classed as the "
                "average of its monthly rate, so the quarterly results do not inherit their "
                "monthly smoothing by construction.")]
    lines.append("- **Persistence (P1):** " + (
        "supported. The persistence misfit is at least partly a product of how smoothed "
        "countries produce their series, not only of labour-market dynamics."
        if by["P1"] == "supported" else
        f"{by['P1']}. The smoothing remains a documented caveat for the persistence misfit."))
    if nl.average_of_monthly and by["P2"] == "supported":
        lines.append("- **Timing (P2):** supported, and the Dutch quarterly rate is the "
                     "average of its monthly trend. The Dutch delayed response cannot be "
                     "separated from the trend method.")
    else:
        lines.append(f"- **Timing (P2):** {by['P2']}"
                     + ("" if nl.average_of_monthly else
                        "; the Dutch quarterly rate is not classed as the average of its "
                        "monthly rate")
                     + ". No evidence that smoothing drives the Dutch delayed response.")
    lines.append("- **Status:** statements about the data, not about labour markets.")
    return lines


def render(study: Study, command: str) -> str:
    lines = [
        "# Does the quarterly unemployment rate inherit monthly smoothing?", "",
        "Generated by `eu-okun smoothing` from the frozen snapshots under "
        "[SMOOTHING_PROTOCOL.md](../SMOOTHING_PROTOCOL.md), registered before any of these "
        "quantities was computed.", "",
        "## Q1: is the quarterly rate the average of the monthly rate?", "",
        "Quarterly rate (`une_rt_q`, updated 2026-09-10) against the mean of the three "
        "monthly rates (`une_rt_m`, seasonally adjusted, updated 2026-10-16), 2009-Q1 to "
        "2026-Q2. Classed as the average when the correlation of quarterly changes is at "
        "least 0.95 and the mean absolute difference at most 0.1 pp.", "",
        "| Country | Monthly method | Quarters | Mean abs. difference (pp) | Largest (pp) | "
        "Correlation of changes | Average of monthly |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for i in study.inheritance:
        lines.append(f"| {i.geo} | {i.method} | {i.quarters} | {i.mean_abs_diff:.3f} | "
                     f"{i.max_abs_diff:.2f} | {i.change_corr:.3f} | "
                     f"{'yes' if i.average_of_monthly else 'no'} |")
    lines += ["", "## Tests", "",
              "Smoothed group: DE, NL (trend), EE, PT (centred moving average), HR, LU (moving "
              "annual averages). Exact permutation tests over countries, one-sided; Holm over "
              "the two tests of each setting.", "",
              "| Setting | Test | Smoothed mean | Comparison mean | Difference | p | Holm p | "
              "Verdict |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for t in study.primary + study.sensitivity:
        lines.append(f"| {t.setting} | {t.key}: {t.text} | {_f(t.mean_smoothed)} "
                     f"({len(t.smoothed)}) | {_f(t.mean_comparison)} ({len(t.comparison)}) | "
                     f"{_f(t.difference)} | {t.p_greater:.3f} | {t.holm:.3f} | {t.verdict} |")
    lines += ["", "## Interpretation (rules fixed in the protocol)", "", *interpretation(study),
              "", "## Countries", "",
              "Lag-1 autocorrelation of quarterly unemployment changes (Bayesian sample "
              "2009-2019, flag-masked; and 2009-2026 without 2020-2021), and the lag share "
              "of the classical two-lag response (where its 95% interval lies below zero).",
              "", "| Country | Method | Autocorrelation 2009-2019 | Autocorrelation "
              "2009-2026 | Two-lag sum [95%] | Lag share |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for g in EU27:
        s = study.shares[g]
        mark = f"**{g}**" if g in SMOOTHED else g
        lines.append(f"| {mark} | {UNEMPLOYMENT_METHOD.get(g, '?')} | "
                     f"{_f(study.autocorrelation[g])} | {_f(study.autocorrelation_full[g])} | "
                     f"{_f(s.theta)} [{_f(s.low)}, {_f(s.high)}] | {_f(s.share)} |")
    m, lu = study.misfit, study.luxembourg
    lines += ["", "## Descriptive", "",
              f"- **The misfit statistic.** Mean country autocorrelation in the Bayesian "
              f"sample: all countries {m['observed_all']:+.3f}; without the smoothed group "
              f"{m['observed_without_smoothed']:+.3f}; direct-estimate countries "
              f"{m['observed_direct']:+.3f}; smoothed countries {m['observed_smoothed']:+.3f}. "
              f"The primary model's replicated mean is {m['replicated_primary']:+.3f}.",
              f"- **Luxembourg.** Autocorrelation {lu['autocorrelation']:+.2f}; correlation of "
              f"its quarterly and averaged monthly changes {lu['change_corr']:.3f}. Classical "
              f"sum with two lags {lu['sum_k2']:+.2f}, with four lags {lu['sum_k4']:+.2f} "
              f"[{lu['sum_k4_low']:+.2f}, {lu['sum_k4_high']:+.2f}] (pandemic excluded). "
              "This bears on, but does not test, whether annual-average smoothing attenuates "
              "the two-lag sum.",
              f"- **Negative autocorrelation (not registered).** Luxembourg's quarterly "
              f"unemployment changes are strongly negatively autocorrelated "
              f"({lu['autocorrelation']:+.2f}), as are Malta's "
              f"({study.autocorrelation['MT']:+.2f}). That is the signature of transitory "
              "noise in the quarterly level, such as survey sampling error in a small "
              "country, not of smoothing. Noise in the dependent variable widens the "
              "intervals of an Okun sum; it does not bias the sum towards zero.", "",
              "## Reproduction", "", "```bash", command, "```", ""]
    return "\n".join(lines)


def write_report(data_dir: Path, monthly_dir: Path, review_dir: Path, bayes_dir: Path,
                 output_dir: Path, report: Path | None = None) -> Study:
    study = run(data_dir, monthly_dir, review_dir, bayes_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write(name: str, rows: Sequence[Mapping[str, Any]]) -> None:
        with (output_dir / name).open("w", encoding="utf-8", newline="\n") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    write("inheritance.csv", [{**asdict(i), "average_of_monthly": i.average_of_monthly}
                              for i in study.inheritance])
    write("countries.csv", [{"geo": g, "method": UNEMPLOYMENT_METHOD.get(g, "?"),
                             "autocorrelation": study.autocorrelation[g],
                             "autocorrelation_full": study.autocorrelation_full[g],
                             **{k: v for k, v in asdict(study.shares[g]).items() if k != "geo"}}
                            for g in EU27])
    write("tests.csv", [{**asdict(t), "smoothed": " ".join(t.smoothed),
                         "comparison": " ".join(t.comparison)}
                        for t in study.primary + study.sensitivity])
    (output_dir / "summary.json").write_text(json.dumps(
        {"misfit": study.misfit, "luxembourg": study.luxembourg,
         "verdicts": {t.key: t.verdict for t in study.primary},
         "average_of_monthly": [i.geo for i in study.inheritance if i.average_of_monthly]},
        indent=1) + "\n", encoding="utf-8", newline="\n")
    if report is not None:
        command = (f"eu-okun smoothing --data-dir {data_dir.as_posix()} --monthly-dir "
                   f"{monthly_dir.as_posix()} --review-dir {review_dir.as_posix()} "
                   f"--bayes-dir {bayes_dir.as_posix()} --output-dir {output_dir.as_posix()} "
                   f"--report {report.as_posix()}")
        report.write_text(render(study, command), encoding="utf-8", newline="\n")
    return study
