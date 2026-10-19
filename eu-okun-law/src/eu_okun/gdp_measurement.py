"""Measurement noise or finance? Luxembourg's weak output-labour link
(``GDP_MEASUREMENT_PROTOCOL.md``).

Two rival explanations, each with predictions that can fail:

- H1, transitory noise in quarterly output: negatively autocorrelated growth, and a
  correlation of hours with output that rises more than elsewhere at the year-on-year
  horizon (P1.1, P1.2; LU, IE and MT);
- H2, financial value added: removing finance from output and hours strengthens the hours
  elasticity and the Okun sum more than elsewhere (P2.1, P2.2; LU).

Statistics are computed per country on rows whose levels all lie outside 2020-2021 and
compared with the median of the countries outside LU, IE and MT; a synchronised circular
block bootstrap over quarters gives intervals and p-values, Holm over eight predictions.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun.contract import EU27, EurostatDataError, quarter_sequence
from eu_okun.gdp_measurement_data import END, START, audit_gdp_measurement, load_gdp_measurement
from eu_okun.stability import holm_adjust

FOCUS = ("LU", "IE", "MT")
PANDEMIC = ("2020-Q1", "2021-Q4")
ROW_START, ROW_END = "2009-Q3", "2026-Q2"
B = 1999
BLOCK = 4
SEED = 20261025
ALPHA = 0.05

# Statistic -> (levels before t that the row uses, description)
WINDOWS = {"rho": 2, "c_q": 1, "c_y": 4, "reg": 3}


@dataclass(frozen=True)
class Prediction:
    key: str
    country: str
    statistic: str
    direction: int          # +1: the focus country is higher than the comparison group
    text: str


PREDICTIONS: tuple[Prediction, ...] = (
    *(Prediction(f"P1.1 {c}", c, "rho", -1,
                 f"{c}: growth autocorrelation lower than the comparison group")
      for c in FOCUS),
    *(Prediction(f"P1.2 {c}", c, "dc", +1,
                 f"{c}: year-on-year minus quarterly correlation higher than the "
                 "comparison group") for c in FOCUS),
    Prediction("P2.1 LU", "LU", "db", +1,
               "LU: non-financial minus total hours elasticity higher than the comparison "
               "group"),
    Prediction("P2.2 LU", "LU", "dtheta", -1,
               "LU: non-financial minus total Okun sum lower than the comparison group"),
)


# ----------------------------------------------------------------------------------
# Measures and rows
# ----------------------------------------------------------------------------------


def _log(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), np.nan)


def _diff(x: np.ndarray, lag: int = 1) -> np.ndarray:
    out = np.full_like(x, np.nan)
    out[lag:] = x[lag:] - x[:-lag]
    return out


def _series(panel: Mapping[tuple[str, str], Mapping[str, float | None]], geo: str,
            key: str, quarters: Sequence[str]) -> np.ndarray:
    return np.asarray([np.nan if (v := panel.get((geo, q), {}).get(key)) is None else v
                       for q in quarters], dtype=float)


def measures(panel: Mapping[tuple[str, str], Mapping[str, float | None]],
             unemployment: Mapping[tuple[str, str], float | None], geo: str,
             quarters: Sequence[str], method: str = "weights") -> dict[str, np.ndarray]:
    """Section 3: every measure of one country on the full quarter grid."""
    v, vk = _series(panel, geo, "va_clv", quarters), _series(panel, geo, "va_clv_k", quarters)
    c, ck = _series(panel, geo, "va_cp", quarters), _series(panel, geo, "va_cp_k", quarters)
    hours, hours_k = _series(panel, geo, "hours", quarters), _series(panel, geo, "hours_k",
                                                                    quarters)
    u = np.asarray([np.nan if (x := unemployment.get((geo, q))) is None else x
                    for q in quarters], dtype=float)
    g = 100 * _diff(_log(v))
    g_k = 100 * _diff(_log(vk))
    share = ck / c
    lagged = np.r_[np.nan, share[:-1]]
    if method == "weights":
        g_n = (g - lagged * g_k) / (1 - lagged)
    elif method == "clv":
        g_n = 100 * _diff(_log(v - vk))
    else:
        raise ValueError(f"Unknown method {method}.")
    return {"g": g, "g_n": g_n, "h": 100 * _diff(_log(hours)),
            "h_n": 100 * _diff(_log(hours - hours_k)),
            "G": 100 * _diff(_log(v), 4), "Y": 100 * _diff(_log(hours), 4),
            "du": _diff(u), "share_va": share, "share_hours": hours_k / hours}


def _outside(quarters: Sequence[str], t: int, levels: int) -> bool:
    return all(not (PANDEMIC[0] <= quarters[s] <= PANDEMIC[1])
               for s in range(t - levels, t + 1))


@dataclass(frozen=True)
class Rows:
    """The rows of one statistic for one country: grid positions and data."""

    position: np.ndarray        # index into the analysis grid (ROW_START..ROW_END)
    data: np.ndarray            # (n, k): the columns the statistic needs


def build_rows(m: Mapping[str, np.ndarray], quarters: Sequence[str],
               grid: Sequence[int]) -> dict[str, Rows]:
    """Rows of every statistic: finite values and levels outside 2020-2021."""
    specs: dict[str, tuple[str, list[Callable[[int], float]]]] = {
        "rho": ("rho", [lambda t: m["g"][t], lambda t: m["g"][t - 1]]),
        "c_q": ("c_q", [lambda t: m["h"][t], lambda t: m["g"][t]]),
        "c_y": ("c_y", [lambda t: m["Y"][t], lambda t: m["G"][t]]),
        "b": ("reg", [lambda t: m["h"][t], *(_lag(m["g"], k) for k in range(3))]),
        "b_n": ("reg", [lambda t: m["h_n"][t], *(_lag(m["g_n"], k) for k in range(3))]),
        "theta": ("reg", [lambda t: m["du"][t], *(_lag(m["g"], k) for k in range(3))]),
        "theta_n": ("reg", [lambda t: m["du"][t], *(_lag(m["g_n"], k) for k in range(3))]),
        "prod": ("reg", [lambda t: m["g"][t] - m["h"][t], *(_lag(m["g"], k)
                                                            for k in range(3))]),
        "prod_n": ("reg", [lambda t: m["g_n"][t] - m["h_n"][t], *(_lag(m["g_n"], k)
                                                                  for k in range(3))]),
    }
    out = {}
    for name, (window, columns) in specs.items():
        positions, data = [], []
        for p, t in enumerate(grid):
            row = [f(t) for f in columns]
            if np.all(np.isfinite(row)) and _outside(quarters, t, WINDOWS[window]):
                positions.append(p)
                data.append(row)
        out[name] = Rows(np.asarray(positions, dtype=int),
                         np.asarray(data, dtype=float).reshape(-1, len(columns)))
    return out


def _lag(x: np.ndarray, k: int) -> Callable[[int], float]:
    return lambda t: float(x[t - k])


# ----------------------------------------------------------------------------------
# Statistics with integer weights (bootstrap multiplicities)
# ----------------------------------------------------------------------------------


def weighted_corr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    total = w.sum()
    if total < 3:
        return math.nan
    ma, mb = (w @ a) / total, (w @ b) / total
    da, db = a - ma, b - mb
    denominator = math.sqrt(float(w @ (da * da)) * float(w @ (db * db)))
    return float(w @ (da * db)) / denominator if denominator > 0 else math.nan


def two_lag_sum(data: np.ndarray, w: np.ndarray) -> float:
    """OLS of column 0 on a constant and columns 1-3; the sum of their coefficients."""
    if np.count_nonzero(w) < 6:
        return math.nan
    x = np.column_stack((np.ones(len(data)), data[:, 1:4]))
    xtw = x.T * w
    try:
        beta = np.linalg.solve(xtw @ x, xtw @ data[:, 0])
    except np.linalg.LinAlgError:
        return math.nan
    return float(beta[1:4].sum())


def statistics(rows: Mapping[str, Rows], counts: np.ndarray | None = None) -> dict[str, float]:
    """Section 4 for one country; ``counts`` are bootstrap multiplicities by grid quarter."""
    def weights(name: str) -> np.ndarray:
        r = rows[name]
        return np.ones(len(r.position)) if counts is None else counts[r.position].astype(float)

    def corr(name: str) -> float:
        r = rows[name]
        return weighted_corr(r.data[:, 0], r.data[:, 1], weights(name)) if len(r.position) \
            else math.nan

    def reg(name: str) -> float:
        r = rows[name]
        return two_lag_sum(r.data, weights(name)) if len(r.position) else math.nan

    s = {"rho": corr("rho"), "c_q": corr("c_q"), "c_y": corr("c_y"), "b": reg("b"),
         "b_n": reg("b_n"), "theta": reg("theta"), "theta_n": reg("theta_n"),
         "prod": reg("prod"), "prod_n": reg("prod_n")}
    s["dc"] = s["c_y"] - s["c_q"]
    s["db"] = s["b_n"] - s["b"]
    s["dtheta"] = s["theta_n"] - s["theta"]
    return s


def comparison(values: Mapping[str, float], focus: str) -> float:
    """``s_f`` minus the median over countries outside LU, IE and MT with a value."""
    group = [v for g, v in values.items() if g not in FOCUS and math.isfinite(v)]
    if not group or not math.isfinite(values.get(focus, math.nan)):
        return math.nan
    return float(values[focus] - np.median(group))


# ----------------------------------------------------------------------------------
# The study
# ----------------------------------------------------------------------------------


@dataclass
class Panel:
    quarters: list[str]
    grid: list[int]                                  # positions of ROW_START..ROW_END
    rows: dict[str, dict[str, Rows]]                 # country -> statistic -> rows
    measures: dict[str, dict[str, np.ndarray]]


def build_panel(snapshot: Path, eurostat_dir: Path, method: str = "weights") -> Panel:
    from eu_okun.audit import audit_snapshot
    from eu_okun.baseline import _read_panel

    audit_gdp_measurement(snapshot)
    audit_snapshot(eurostat_dir)
    panel = load_gdp_measurement(snapshot)
    frozen = _read_panel(eurostat_dir / "processed" / "eu27_quarterly.csv")
    unemployment = {key: value.get("unemployment") for key, value in frozen.items()}
    quarters = quarter_sequence(START, END)
    grid = [i for i, q in enumerate(quarters) if ROW_START <= q <= ROW_END]
    rows, ms = {}, {}
    for geo in EU27:
        m = measures(panel, unemployment, geo, quarters, method)
        ms[geo] = m
        rows[geo] = build_rows(m, quarters, grid)
    if not any(len(rows[g]["b"].position) for g in FOCUS):
        raise EurostatDataError("No focus country has hours and value added.")
    return Panel(quarters, grid, rows, ms)


def block_counts(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Multiplicities of a circular block resample of ``n`` consecutive quarters."""
    count = math.ceil(n / block)
    starts = rng.integers(0, n, size=count)
    index = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
    return np.bincount(index, minlength=n)


@dataclass(frozen=True)
class Verdict:
    setting: str
    key: str
    country: str
    statistic: str
    text: str
    focus_value: float
    group_median: float
    estimate: float
    low: float
    high: float
    p_value: float
    holm: float
    verdict: str


def evaluate(panel: Panel, setting: str, block: int = BLOCK, draws: int = B,
             seed: int = SEED) -> tuple[list[Verdict], dict[str, dict[str, float]]]:
    """Observed statistics, the bootstrap of every prediction, Holm and the verdicts."""
    observed = {g: statistics(panel.rows[g]) for g in EU27}
    rng = np.random.default_rng(seed)
    n = len(panel.grid)
    boot = {p.key: np.empty(draws) for p in PREDICTIONS}
    for b in range(draws):
        counts = block_counts(n, block, rng)
        stats = {g: statistics(panel.rows[g], counts) for g in EU27}
        for p in PREDICTIONS:
            boot[p.key][b] = comparison({g: stats[g][p.statistic] for g in EU27}, p.country)
    rows = []
    for p in PREDICTIONS:
        values = {g: observed[g][p.statistic] for g in EU27}
        estimate = comparison(values, p.country)
        draws_ok = boot[p.key][np.isfinite(boot[p.key])]
        low, high = (np.quantile(draws_ok, [0.025, 0.975]) if draws_ok.size
                     else (math.nan, math.nan))
        pv = (min(1.0, 2 * min(float(np.mean(draws_ok <= 0)), float(np.mean(draws_ok >= 0))))
              if draws_ok.size else 1.0)
        group = [v for g, v in values.items() if g not in FOCUS and math.isfinite(v)]
        rows.append((p, values[p.country], float(np.median(group)) if group else math.nan,
                     estimate, float(low), float(high), pv))
    holm = holm_adjust([r[-1] for r in rows])
    out = []
    for (p, fv, med, est, low, high, pv), adj in zip(rows, holm, strict=True):
        predicted = (low > 0 if p.direction > 0 else high < 0)
        opposite = (high < 0 if p.direction > 0 else low > 0)
        verdict = ("supported" if predicted and adj <= ALPHA else
                   "contradicted" if opposite and adj <= ALPHA else "not determined")
        out.append(Verdict(setting, p.key, p.country, p.statistic, p.text, fv, med, est, low,
                           high, pv, adj, verdict))
    return out, observed


@dataclass
class Study:
    primary: list[Verdict]
    sensitivity: dict[str, list[Verdict]]
    observed: dict[str, dict[str, float]]
    descriptive: dict[str, dict[str, float]]
    rows: dict[str, dict[str, int]]
    manifest: dict[str, Any]


def descriptives(panel: Panel, gdp: Mapping[tuple[str, str], float | None]
                 ) -> dict[str, dict[str, float]]:
    """Finance shares (2009-2019 means) and the GVA-GDP growth correlation."""
    pre = [i for i, q in enumerate(panel.quarters) if "2009-Q1" <= q <= "2019-Q4"]
    out = {}
    for geo in EU27:
        m = panel.measures[geo]
        gdp_growth = np.asarray([np.nan if (x := gdp.get((geo, q))) is None else x
                                 for q in panel.quarters], dtype=float)
        pairs = [(m["g"][t], gdp_growth[t]) for t in panel.grid
                 if np.isfinite(m["g"][t]) and np.isfinite(gdp_growth[t])
                 and _outside(panel.quarters, t, 1)]
        corr = (float(np.corrcoef(np.asarray(pairs).T)[0, 1]) if len(pairs) > 2
                else math.nan)
        out[geo] = {"finance_share_va": float(np.nanmean(m["share_va"][pre])),
                    "finance_share_hours": float(np.nanmean(m["share_hours"][pre]))
                    if np.isfinite(m["share_hours"][pre]).any() else math.nan,
                    "gva_gdp_corr": corr}
    return out


def run(snapshot: Path, eurostat_dir: Path, draws: int = B) -> Study:
    from eu_okun.baseline import _read_panel

    panel = build_panel(snapshot, eurostat_dir)
    primary, observed = evaluate(panel, "primary", draws=draws)
    sensitivity = {
        "clv": evaluate(build_panel(snapshot, eurostat_dir, "clv"), "clv", draws=draws)[0],
        "block_8": evaluate(panel, "block_8", block=8, draws=draws)[0],
    }
    frozen = _read_panel(eurostat_dir / "processed" / "eu27_quarterly.csv")
    gdp = {key: value.get("gdp") for key, value in frozen.items()}
    rows = {g: {name: len(r.position) for name, r in panel.rows[g].items()} for g in EU27}
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    return Study(primary, sensitivity, observed, descriptives(panel, gdp), rows, manifest)


# ----------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------


def _f(x: float, digits: int = 2) -> str:
    return "n/a" if not math.isfinite(x) else f"{x:+.{digits}f}"


def interpretation(verdicts: Sequence[Verdict]) -> list[str]:
    """Section 6: the interpretation rules fixed in advance."""
    by = {v.key: v.verdict for v in verdicts}
    lines = []
    for c in FOCUS:
        both = by[f"P1.1 {c}"] == by[f"P1.2 {c}"] == "supported"
        lines.append(f"- **{c}, H1:** " + (
            "both predictions supported: the quarterly output data show the signature of "
            "transitory noise. Measurement noise is consistent with the data; it is not "
            "proven." if both else
            f"P1.1 {by[f'P1.1 {c}']}, P1.2 {by[f'P1.2 {c}']}: the signature of transitory "
            "noise is not established."))
    finance = [by["P2.1 LU"], by["P2.2 LU"]]
    lines.append("- **LU, H2:** " + (
        "the financial sector accounts for part of the weak link."
        if "supported" in finance else
        f"P2.1 {finance[0]}, P2.2 {finance[1]}: removing finance does not establish a "
        "stronger link."))
    if by["P1.1 LU"] != "supported" and "supported" not in finance and \
            by["P1.2 LU"] != "supported":
        lines.append("- **Overall:** both explanations fail their tests for Luxembourg, "
                     "whose weak output-labour link remains unexplained.")
    return lines


def render(study: Study, command: str) -> str:
    v = study.primary
    files = study.manifest["files"]
    lines = [
        "# Measurement noise or finance? Luxembourg's weak output-labour link", "",
        "Generated by `eu-okun gdp-measurement` from the frozen snapshots under "
        "[GDP_MEASUREMENT_PROTOCOL.md](../GDP_MEASUREMENT_PROTOCOL.md), registered before "
        "any of these quantities was computed. Patterns consistent with a mechanism, not "
        "causal effects.", "",
        "## Verdicts", "",
        "Test statistic: the focus country's value minus the median of the countries "
        "outside LU, IE and MT. 95% synchronised block-bootstrap intervals; Holm over the "
        "eight predictions.", "",
        "| Prediction | Statement | Focus | Comparison median | Difference [95%] | Holm p "
        "| Verdict |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in v:
        lines.append(f"| {r.key} | {r.text} | {_f(r.focus_value)} | {_f(r.group_median)} | "
                     f"{_f(r.estimate)} [{_f(r.low)}, {_f(r.high)}] | {r.holm:.3f} | "
                     f"{'**' + r.verdict + '**' if r.verdict != 'not determined' else r.verdict}"
                     " |")
    counts = {k: sum(r.verdict == k for r in v) for k in ("supported", "contradicted")}
    lines += ["", f"{counts['supported']} supported, {counts['contradicted']} contradicted, "
              f"{len(v) - sum(counts.values())} not determined, of {len(v)}.", "",
              "## Interpretation (rules fixed in the protocol)", "", *interpretation(v), "",
              "## Statistics by country", "",
              "rho: autocorrelation of quarterly value-added growth. c_q, c_y: correlation of "
              "hours with value added, quarter on quarter and year on year. b, b_n: two-lag "
              "hours elasticity, total and outside finance. theta, theta_n: two-lag Okun "
              "sum on value added, total and outside finance (pp per %).", "",
              "| Country | rho | c_q | c_y | dc | b | b_n | db | theta | theta_n | dtheta |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for g in EU27:
        s = study.observed[g]
        mark = f"**{g}**" if g in FOCUS else g
        lines.append(f"| {mark} | " + " | ".join(_f(s[k]) for k in (
            "rho", "c_q", "c_y", "dc", "b", "b_n", "db", "theta", "theta_n", "dtheta"))
            + " |")
    lines += ["", "## Sensitivity (each with its own Holm family)", "",
              "| Prediction | Primary | Chain-linked difference | Block length 8 |",
              "| --- | --- | --- | --- |"]
    for i, r in enumerate(v):
        lines.append(f"| {r.key} | {r.verdict} | {study.sensitivity['clv'][i].verdict} | "
                     f"{study.sensitivity['block_8'][i].verdict} |")
    d = study.descriptive
    lines += ["", "## Descriptive", "",
              "Finance (NACE K) as a share of value added and of hours, 2009-2019 means; the "
              "correlation of value-added growth with the frozen GDP growth; and the two-lag "
              "output-per-hour elasticity, total and outside finance.", "",
              "| Country | Finance share, value added | Finance share, hours | GVA-GDP "
              "correlation | Output per hour | Output per hour outside finance |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for g in EU27:
        s = study.observed[g]
        mark = f"**{g}**" if g in FOCUS else g
        share_h = d[g]["finance_share_hours"]
        lines.append(f"| {mark} | {100 * d[g]['finance_share_va']:.1f}% | "
                     f"{'n/a' if not math.isfinite(share_h) else f'{100 * share_h:.1f}%'} | "
                     f"{_f(d[g]['gva_gdp_corr'])} | {_f(s['prod'])} | {_f(s['prod_n'])} |")
    lines += ["", "## Not tested", "",
              "- Ireland's modified domestic demand (Irish Central Statistics Office, not "
              "Eurostat).",
              "- Annual national accounts (year-on-year changes of quarterly data are used).",
              "- Measurement error shared by output and hours.",
              "- Sectoral composition beyond finance.", "",
              "## Provenance and reproduction", "",
              f"Snapshot downloaded {study.manifest['downloaded_at_utc']}. Sources: "
              + "; ".join(f"`{f['dataset']}` ({f['key']}, updated {f['updated'][:10]})"
                          for f in files)
              + ". Unemployment and GDP growth from `snapshots/eurostat-2026-10-15`. "
              f"Bootstrap: B = {B}, block {BLOCK}, seed {SEED}.", "",
              "```bash", command, "```", ""]
    return "\n".join(lines)


def write_report(snapshot: Path, eurostat_dir: Path, output_dir: Path,
                 report: Path | None = None, draws: int = B) -> Study:
    study = run(snapshot, eurostat_dir, draws)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write(name: str, rows: Sequence[Mapping[str, Any]]) -> None:
        with (output_dir / name).open("w", encoding="utf-8", newline="\n") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    write("predictions.csv", [asdict(r) for r in study.primary]
          + [asdict(r) for rows in study.sensitivity.values() for r in rows])
    write("statistics.csv", [{"geo": g, **study.observed[g],
                              **{f"n_{k}": n for k, n in study.rows[g].items()}}
                             for g in EU27])
    write("descriptive.csv", [{"geo": g, **study.descriptive[g]} for g in EU27])
    summary = {
        "supported": [r.key for r in study.primary if r.verdict == "supported"],
        "contradicted": [r.key for r in study.primary if r.verdict == "contradicted"],
        "predictions": len(study.primary),
        "settings": {"B": draws, "block": BLOCK, "seed": SEED},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
    if report is not None:
        command = (f"eu-okun gdp-measurement --data-dir {eurostat_dir.as_posix()} "
                   f"--snapshot {snapshot.as_posix()} --output-dir {output_dir.as_posix()} "
                   f"--report {report.as_posix()}")
        report.write_text(render(study, command), encoding="utf-8", newline="\n")
    return study
