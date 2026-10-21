"""Null-imposed bootstraps with errors from the unrestricted residuals (post hoc, not
registered).

The registered small-sample studies (``HEADLINE_INFERENCE_PROTOCOL.md``,
``SMALL_SAMPLE_PROTOCOL.md``) impose the null on the mean and fit the bootstrap error
process to the *restricted* residuals. Those residuals absorb the omitted GDP effect, which
is persistent, so their AR(1) is inflated when the null is false. The same inflated process
drives the size studies. An external review (2026-10-20) asked how much this matters.

This module repeats both studies with one change: the error process (AR(1) sieve, moving
blocks) is fitted to, and resampled from, the *unrestricted* residuals, while the mean
still imposes the null. Data, designs, statistics, draws, seeds and size-study settings are
the registered ones. It also reports Benjamini-Hochberg adjustments, which control the
false discovery rate, next to the registered Holm adjustments. Nothing here replaces a
registered result.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun import headline_inference as hi
from eu_okun import small_sample as ss
from eu_okun.small_sample import (
    ALPHA,
    Z_975,
    HacDesign,
    block_errors,
    bootstrap_p,
    fit_sieve,
    hac_design,
    hac_t,
    null_sieve_test,
    restricted_fit,
    segment_starts,
    sieve_errors,
)
from eu_okun.stability import holm_adjust

# ----------------------------------------------------------------------------------
# The two bootstraps with unrestricted residuals
# ----------------------------------------------------------------------------------


def _null_mean_and_residuals(design: HacDesign, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta_r, _ = restricted_fit(design, y)
    resid = y - design.x @ (design.projector @ y)
    return design.x @ beta_r, resid


def sieve_test(design: HacDesign, y: np.ndarray, starts: np.ndarray, draws: int,
               rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """Observed t and t* with the null mean and an AR(1) sieve of the unrestricted residuals."""
    t_obs = float(hac_t(design, y)[2][0])
    mean, resid = _null_mean_and_residuals(design, y)
    y_star = mean[:, None] + sieve_errors(fit_sieve(resid, starts), resid, draws, rng)
    return t_obs, hac_t(design, y_star)[2]


def block_test(design: HacDesign, y: np.ndarray, starts: np.ndarray, draws: int,
               rng: np.random.Generator) -> np.ndarray:
    """t* with the null mean and moving blocks of the unrestricted residuals."""
    mean, resid = _null_mean_and_residuals(design, y)
    return hac_t(design, mean[:, None] + block_errors(resid, starts, draws, rng))[2]


def size(design: HacDesign, y: np.ndarray, starts: np.ndarray, replications: int,
         draws: int, rng: np.random.Generator,
         rng_registered: np.random.Generator) -> tuple[float, float, float]:
    """Rejection rates on data simulated under the null with the unrestricted error
    process: the normal test, the sieve test above, and the registered sieve test (errors
    from the restricted residuals of each simulated sample)."""
    mean, resid = _null_mean_and_residuals(design, y)
    simulated = mean[:, None] + sieve_errors(fit_sieve(resid, starts), resid, replications,
                                             rng)
    normal = float(np.mean(np.abs(hac_t(design, simulated)[2]) > Z_975))
    rejected, rejected_registered = 0, 0
    for r in range(replications):
        t_obs, t_star = sieve_test(design, simulated[:, r], starts, draws, rng)
        rejected += bootstrap_p(t_obs, t_star) <= ALPHA
        t_reg, t_reg_star, _ = null_sieve_test(design, simulated[:, r], starts, draws,
                                               rng_registered)
        rejected_registered += bootstrap_p(t_reg, t_reg_star) <= ALPHA
    return normal, rejected / replications, rejected_registered / replications


def bh_adjust(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg step-up adjusted p-values (false discovery rate)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, pvalues[i] * m / (rank + 1))
        adjusted[i] = running
    return adjusted


def by_adjust(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Yekutieli adjusted p-values: BH times sum(1/i), valid under any dependence."""
    factor = sum(1.0 / i for i in range(1, len(pvalues) + 1))
    return [min(1.0, factor * p) for p in bh_adjust(pvalues)]


def _count(negative: Sequence[bool], *adjusted: Sequence[float]) -> int:
    return sum(n and all(a[i] <= ALPHA for a in adjusted) for i, n in enumerate(negative))


# ----------------------------------------------------------------------------------
# The headline sums
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadlineRow:
    geo: str
    theta: float
    p_sieve_registered: float
    p_block_registered: float
    p_sieve: float
    p_block: float
    rho_restricted: float
    rho_unrestricted: float
    size_normal_registered: float
    size_normal: float
    size_bootstrap: float
    size_registered_bootstrap: float


def headline(data_dir: Path, registered_dir: Path, replications: int = hi.R_SIZE,
             draws_size: int = hi.B_SIZE, draws: int = hi.B_MAIN) -> list[HeadlineRow]:
    heads, _ = hi.headline_designs(data_dir)
    with (registered_dir / "tests.csv").open(encoding="utf-8") as stream:
        tests = {r["geo"]: r for r in csv.DictReader(stream)}
    with (registered_dir / "size.csv").open(encoding="utf-8") as stream:
        sizes = {r["geo"]: r for r in csv.DictReader(stream)}
    rows = []
    for head in heads:
        design = hac_design(head.x, head.periods, hi.sum_contrast(head.x.shape[1]),
                            hi.BANDWIDTH)
        starts = segment_starts(head.periods)
        t_obs, t_sieve = sieve_test(design, head.y, starts, draws,
                                    np.random.default_rng([hi.SEED_TESTS, head.index]))
        t_block = block_test(design, head.y, starts, draws,
                             np.random.default_rng([hi.SEED_TESTS, head.index, 100]))
        normal, boot, boot_registered = size(
            design, head.y, starts, replications, draws_size,
            np.random.default_rng([hi.SEED_SIZE, head.index]),
            np.random.default_rng([hi.SEED_SIZE, head.index, 1]))
        registered = tests[head.geo]
        rows.append(HeadlineRow(
            head.geo, float(registered["theta"]), float(registered["p_sieve"]),
            float(registered["p_block"]), bootstrap_p(t_obs, t_sieve),
            bootstrap_p(t_obs, t_block), float(registered["rho"].split(" / ")[0]),
            float(registered["rho_unrestricted"].split(" / ")[0]),
            float(sizes[head.geo]["normal"]), normal, boot, boot_registered))
    return rows


def headline_summary(rows: Sequence[HeadlineRow]) -> dict[str, Any]:
    negative = [r.theta < 0 for r in rows]
    out: dict[str, Any] = {}
    for label, sieve, block in (("registered", [r.p_sieve_registered for r in rows],
                                 [r.p_block_registered for r in rows]),
                                ("unrestricted", [r.p_sieve for r in rows],
                                 [r.p_block for r in rows])):
        hs, hb, bs, bb = holm_adjust(sieve), holm_adjust(block), bh_adjust(sieve), bh_adjust(block)
        out[label] = {
            "unadjusted_both": _count(negative, sieve, block),
            "holm_both": _count(negative, hs, hb),
            "holm_sieve": _count(negative, hs), "holm_block": _count(negative, hb),
            "bh_both": _count(negative, bs, bb),
            "bh_sieve": _count(negative, bs), "bh_block": _count(negative, bb),
            "by_both": _count(negative, by_adjust(sieve), by_adjust(block)),
            "holm_both_countries": [r.geo for r, n, a, b in zip(rows, negative, hs, hb,
                                                                 strict=True)
                                    if n and a <= ALPHA and b <= ALPHA],
        }
    out["size"] = {
        "normal_registered": float(np.mean([r.size_normal_registered for r in rows])),
        "normal": float(np.mean([r.size_normal for r in rows])),
        "bootstrap": float(np.mean([r.size_bootstrap for r in rows])),
        "registered_bootstrap": float(np.mean([r.size_registered_bootstrap for r in rows])),
        "over_rejecting": sum(r.size_normal > ss.binomial_range(hi.R_SIZE)[1] for r in rows),
        "bootstrap_range": list(ss.binomial_range(hi.R_SIZE * len(rows))),
    }
    out["rho_gap_median"] = float(np.median([r.rho_restricted - r.rho_unrestricted
                                             for r in rows]))
    out["countries"] = len(rows)
    return out


# ----------------------------------------------------------------------------------
# The post-2021 change tests
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChangeRow:
    geo: str
    participation: bool
    change: float
    published_significant: bool
    p_sieve_registered: float
    p_block_registered: float
    p_sieve: float
    p_block: float
    rho_restricted: float
    rho_unrestricted: float
    size_normal_registered: float
    size_normal: float
    size_bootstrap: float
    size_registered_bootstrap: float


def changes(data_dir: Path, registered_dir: Path, replications: int = ss.R_SIZE,
            draws_size: int = ss.B_SIZE, draws: int = ss.B_RESOLUTION) -> list[ChangeRow]:
    from eu_okun.audit import audit_snapshot
    from eu_okun.baseline import _read_panel
    from eu_okun.stability import fit_stability

    end = audit_snapshot(data_dir)["analysis_period"]["end"]
    if not isinstance(end, str):
        raise ValueError("Invalid final quarter in audit.")
    panel = _read_panel(data_dir / "processed" / "eu27_quarterly.csv")
    specs = ss.stability_specs(panel, end)
    published, _ = fit_stability(panel, end, hac_bandwidth=ss.PRIMARY_BANDWIDTH)
    summary = json.loads((registered_dir / "summary.json").read_text(encoding="utf-8"))
    p_sieve_reg, p_block_reg = (summary["resolution"]["p_sieve"],
                                summary["resolution"]["p_block"])
    with (registered_dir / "size_study.csv").open(encoding="utf-8") as stream:
        sizes = list(csv.DictReader(stream))
    rows = []
    for spec, pub, size_row in zip(specs, published, sizes, strict=True):
        width = spec.x.shape[1] // 2
        design = hac_design(spec.x, spec.periods, ss.change_contrast(width),
                            ss.PRIMARY_BANDWIDTH)
        starts = segment_starts(spec.periods)
        t_obs, t_sieve = sieve_test(design, spec.y, starts, draws,
                                    np.random.default_rng([ss.SEED_TESTS, spec.index, 200]))
        t_block = block_test(design, spec.y, starts, draws,
                             np.random.default_rng([ss.SEED_TESTS, spec.index, 201]))
        normal, boot, boot_registered = size(
            design, spec.y, starts, replications, draws_size,
            np.random.default_rng([ss.SEED_SIZE, spec.index]),
            np.random.default_rng([ss.SEED_SIZE, spec.index, 1]))
        _, resid_r = restricted_fit(design, spec.y)
        resid_u = spec.y - design.x @ (design.projector @ spec.y)
        rows.append(ChangeRow(
            spec.geo, spec.participation, float(hac_t(design, spec.y)[0][0]),
            pub.p_change_holm <= ALPHA, p_sieve_reg[spec.index], p_block_reg[spec.index],
            bootstrap_p(t_obs, t_sieve), bootstrap_p(t_obs, t_block),
            fit_sieve(resid_r, starts).period_rho[-1],
            fit_sieve(resid_u, starts).period_rho[-1],
            float(size_row["asymptotic"]), normal, boot, boot_registered))
    return rows


def changes_summary(rows: Sequence[ChangeRow]) -> dict[str, Any]:
    every = [True] * len(rows)
    out: dict[str, Any] = {"tests": len(rows),
                           "published_significant": sum(r.published_significant for r in rows)}
    for label, sieve, block in (("registered", [r.p_sieve_registered for r in rows],
                                 [r.p_block_registered for r in rows]),
                                ("unrestricted", [r.p_sieve for r in rows],
                                 [r.p_block for r in rows])):
        hs, hb, bs, bb = holm_adjust(sieve), holm_adjust(block), bh_adjust(sieve), bh_adjust(block)
        out[label] = {
            "holm_both": _count(every, hs, hb), "bh_both": _count(every, bs, bb),
            "by_both": _count(every, by_adjust(sieve), by_adjust(block)),
            "holm_both_tests": [f"{r.geo}{' (participation)' if r.participation else ''}"
                                for r, a, b in zip(rows, hs, hb, strict=True)
                                if a <= ALPHA and b <= ALPHA],
            "bh_both_tests": [f"{r.geo}{' (participation)' if r.participation else ''}"
                              for r, a, b in zip(rows, bs, bb, strict=True)
                              if a <= ALPHA and b <= ALPHA],
        }
    out["size"] = {
        "normal_registered": float(np.mean([r.size_normal_registered for r in rows])),
        "normal": float(np.mean([r.size_normal for r in rows])),
        "bootstrap": float(np.mean([r.size_bootstrap for r in rows])),
        "registered_bootstrap": float(np.mean([r.size_registered_bootstrap for r in rows])),
        "bootstrap_range": list(ss.binomial_range(ss.R_SIZE * len(rows))),
    }
    out["rho_gap_median"] = float(np.median([r.rho_restricted - r.rho_unrestricted
                                             for r in rows]))
    return out


# ----------------------------------------------------------------------------------
# Report and files
# ----------------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _holds(rate: float, exact: Sequence[float]) -> str:
    """Whether a pooled rejection rate lies in the 95% range of an exact 5% test."""
    if rate > exact[1]:
        return "over-rejects"
    if rate < exact[0]:
        return "under-rejects"
    return "holds its size"


def render(head: dict[str, Any], change: dict[str, Any], command: str) -> str:
    hr, hu = head["registered"], head["unrestricted"]
    cr, cu = change["registered"], change["unrestricted"]
    hs, cs = head["size"], change["size"]
    lines = [
        "# Null-imposed bootstraps with errors from the unrestricted residuals", "",
        "Generated by `eu-okun null-residuals`. **Post hoc, not registered.** An external "
        "review (2026-10-20) pointed out that the registered null-imposed bootstraps fit "
        "their error process to the restricted residuals. Those residuals absorb the omitted, "
        "persistent GDP effect, so the bootstrap and the size study simulate a null whose "
        "errors are more persistent than the data's. This run keeps the null on the mean and "
        "takes the error process from the unrestricted residuals. Everything else is "
        "registered: data, designs, statistics, draws, seeds and size-study settings. "
        "Benjamini-Hochberg (BH) adjustments, which control the false discovery rate, are "
        "added next to the registered Holm adjustments, which control the chance of any "
        "false rejection. Nothing here replaces a registered result.", "",
        "## Headline sums (HEADLINE_INFERENCE_PROTOCOL.md)", "",
        f"Restricted minus unrestricted AR(1) before 2020: median {head['rho_gap_median']:+.2f}.",
        "",
        "| Countries with a significantly negative sum | Registered errors | Unrestricted "
        "errors |",
        "| --- | ---: | ---: |",
        f"| Both schemes, unadjusted | {hr['unadjusted_both']} | {hu['unadjusted_both']} |",
        f"| Both schemes, Holm (registered rule) | {hr['holm_both']} | {hu['holm_both']} |",
        f"| Sieve alone, Holm | {hr['holm_sieve']} | {hu['holm_sieve']} |",
        f"| Block alone, Holm | {hr['holm_block']} | {hu['holm_block']} |",
        f"| Both schemes, BH | {hr['bh_both']} | {hu['bh_both']} |",
        f"| Both schemes, BY (any dependence) | {hr['by_both']} | {hu['by_both']} |",
        "",
        f"Holm, both schemes, unrestricted errors: {', '.join(hu['holm_both_countries'])}.", "",
        "**Size** (nominal 5%, pooled over countries, data simulated under the null):", "",
        f"- normal reference: {_pct(hs['normal_registered'])} with the registered error "
        f"process, {_pct(hs['normal'])} with the unrestricted one; {hs['over_rejecting']} "
        "countries still over-reject;",
        f"- sieve bootstrap with unrestricted errors: {_pct(hs['bootstrap'])} (95% range for "
        f"an exact test: {_pct(hs['bootstrap_range'][0])} to {_pct(hs['bootstrap_range'][1])});",
        f"- registered sieve bootstrap, on the same simulated data: "
        f"{_pct(hs['registered_bootstrap'])}.",
        "",
        "## Post-2021 changes (SMALL_SAMPLE_PROTOCOL.md)", "",
        f"{change['tests']} tests; the first classical pass found "
        f"{change['published_significant']} significant after Holm. Restricted minus "
        f"unrestricted AR(1) after 2021: median {change['rho_gap_median']:+.2f}. Both schemes "
        f"at B = {ss.B_RESOLUTION}, as in the registered resolution check.", "",
        "| Changes significant under both schemes | Registered errors | Unrestricted errors |",
        "| --- | ---: | ---: |",
        f"| Holm (registered rule) | {cr['holm_both']} | {cu['holm_both']} |",
        f"| BH | {cr['bh_both']} | {cu['bh_both']} |",
        f"| BY (any dependence) | {cr['by_both']} | {cu['by_both']} |",
        "",
        f"Holm, unrestricted errors: {', '.join(cu['holm_both_tests']) or 'none'}. BH, "
        f"unrestricted errors: {', '.join(cu['bh_both_tests']) or 'none'}.", "",
        "**Size** (nominal 5%, pooled over tests):", "",
        f"- normal reference: {_pct(cs['normal_registered'])} registered, "
        f"{_pct(cs['normal'])} with the unrestricted error process;",
        f"- sieve bootstrap with unrestricted errors: {_pct(cs['bootstrap'])} (95% range "
        f"{_pct(cs['bootstrap_range'][0])} to {_pct(cs['bootstrap_range'][1])});",
        f"- registered sieve bootstrap, on the same simulated data: "
        f"{_pct(cs['registered_bootstrap'])}.",
        "",
        "## Reading", "",
        "- Restricted residuals are the usual choice for a null-imposed bootstrap, but when "
        "the omitted effect is persistent they make the null process too persistent. That "
        "lowers power for large sums and exaggerates the size distortion of the normal "
        "reference: "
        f"{_pct(hs['normal_registered'])} against {_pct(hs['normal'])} for the headline sums.",
        "- On the less persistent null, the registered sieve bootstrap "
        + _holds(hs["registered_bootstrap"], hs["bootstrap_range"])
        + f" ({_pct(hs['registered_bootstrap'])}), and the variant with unrestricted errors "
        + _holds(hs["bootstrap"], hs["bootstrap_range"]) + f" ({_pct(hs['bootstrap'])}). A "
        "count that rests on a test that over-rejects is not a valid replacement for the "
        "registered count.",
        "- Holm controls the chance of any false rejection among the countries. BH controls "
        "the expected share of false rejections among those declared, which is the question "
        "of how many countries have a negative sum. On the registered p-values BH gives "
        f"{hr['bh_both']} countries ({hu['bh_both']} with unrestricted errors). BH is "
        "guaranteed for independent tests and for positively dependent one-sided tests; "
        "these are two-sided and dependent through common shocks. Benjamini-Yekutieli (BY), "
        f"valid under any dependence, gives {hr['by_both']} ({hu['by_both']}).",
        "", "## Reproduction", "", "```bash", command, "```", "",
    ]
    return "\n".join(lines)


def write_report(data_dir: Path, headline_dir: Path, small_sample_dir: Path,
                 output_dir: Path, report: Path | None = None) -> dict[str, Any]:
    head_rows = headline(data_dir, headline_dir)
    change_rows = changes(data_dir, small_sample_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("headline.csv", head_rows), ("changes.csv", change_rows)):
        records = [asdict(r) for r in rows]
        with (output_dir / name).open("w", encoding="utf-8", newline="\n") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    head, change = headline_summary(head_rows), changes_summary(change_rows)
    summary: dict[str, Any] = {"headline": head, "changes": change, "not_registered": True,
                               "review": "external review, 2026-10-20"}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
    if report is not None:
        command = (f"poetry run eu-okun null-residuals --data-dir {data_dir.as_posix()} "
                   f"--headline-dir {headline_dir.as_posix()} --small-sample-dir "
                   f"{small_sample_dir.as_posix()} --output-dir {output_dir.as_posix()} "
                   f"--report {report.as_posix()}")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render(head, change, command),
                          encoding="utf-8", newline="\n")
    return summary
