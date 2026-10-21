"""Post-hoc sensitivity of the identification results to the pandemic window.

Not pre-registered: added after an external review on 2026-10-19. The registered rules
exclude an observation if any period of its variables, from t-L to t+h, lies in
2020-2021 (quarterly local projections: t-4; monthly: t-12; structural VAR: t-2). The
change at the longest lag is measured from the level one period earlier, so that level
can lie in 2021 while the observation is kept. This module widens every window by one
period and re-estimates the primary specifications. The registered results are unchanged;
this is reported next to them.
"""

from __future__ import annotations

import csv
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from eu_okun.audit import audit_snapshot
from eu_okun.baseline import _read_panel
from eu_okun.external_shocks import audit_shock_snapshot, load_quarterly, monthly_series
from eu_okun.monthly_data import audit_monthly_snapshot, load_monthly_panel
from eu_okun.monthly_structural import PRIMARY as MONTHLY_PRIMARY
from eu_okun.monthly_structural import SHOCKS as MONTHLY_SHOCKS
from eu_okun.monthly_structural import estimate_shock as monthly_estimate
from eu_okun.monthly_structural import monthly_arrays
from eu_okun.sign_restrictions import (
    FEVD_HORIZON,
    audit_deflator,
    estimate,
    load_deflator,
    var_data,
)
from eu_okun.sign_restrictions import PRIMARY as SVAR_PRIMARY
from eu_okun.structural import PRIMARY as QUARTERLY_PRIMARY
from eu_okun.structural import SHOCKS as QUARTERLY_SHOCKS
from eu_okun.structural import estimate_shock as quarterly_estimate
from eu_okun.structural import panel_arrays

FIELDS = ("design", "margin", "target", "statistic", "value", "n")


def _row(design: str, margin: int, target: str, statistic: str, value: Any,
         n: int | str = "") -> dict[str, Any]:
    return {"design": design, "margin": margin, "target": target, "statistic": statistic,
            "value": value, "n": n}


def local_projections(eurostat_dir: Path, shocks_dir: Path, monthly_dir: Path,
                      margins: tuple[int, ...] = (0, 1)) -> list[dict[str, Any]]:
    """First-stage t and Anderson-Rubin set of every primary ratio, by window margin."""
    rows = []
    audit = audit_snapshot(eurostat_dir)
    audit_shock_snapshot(shocks_dir)
    panel = _read_panel(eurostat_dir / "processed" / "eu27_quarterly.csv")
    arrays = panel_arrays(panel, "2008-Q1", str(audit["analysis_period"]["end"]))
    shocks = load_quarterly(shocks_dir)
    for margin in margins:
        settings = replace(QUARTERLY_PRIMARY, pandemic_margin=margin)
        for key, spec in QUARTERLY_SHOCKS.items():
            for o in quarterly_estimate(arrays, shocks, spec, settings).okun:
                rows.append(_row("quarterly LP", margin, f"{key} h={o.h}", "first_stage_t",
                                 o.first_stage_t, o.n))
                rows.append(_row("quarterly LP", margin, f"{key} h={o.h}", "set", o.kind,
                                 o.n))
    manifest = audit_monthly_snapshot(monthly_dir)
    monthly = monthly_arrays(load_monthly_panel(monthly_dir), manifest["period"]["start"],
                             manifest["period"]["end"])
    mshocks = monthly_series(shocks_dir)
    for margin in margins:
        monthly_settings = replace(MONTHLY_PRIMARY, pandemic_margin=margin)
        for key, spec in MONTHLY_SHOCKS.items():
            for o in monthly_estimate(monthly, mshocks, spec, monthly_settings).ratios:
                rows.append(_row("monthly LP", margin, f"{key} h={o.h}", "first_stage_t",
                                 o.first_stage_t, o.n))
                rows.append(_row("monthly LP", margin, f"{key} h={o.h}", "set", o.kind, o.n))
    return rows


def shift_share(eurostat_dir: Path, snapshot: Path,
                margins: tuple[int, ...] = (0, 1)) -> list[dict[str, Any]]:
    """The feasibility statistics of every partner set, by window margin. The origins keep
    [t-4, t] clear of 2020-2021, but the lag-4 controls of the GDP residual scale use the
    level at t-5."""
    from eu_okun.contract import quarter_sequence
    from eu_okun.shift_share import (
        LAST_ORIGIN,
        PARTNER_SETS,
        SHIFT_START,
        audit_shift_share,
        decision,
        feasibility,
        instrument,
        load_inputs,
        sigma_u,
    )

    audit_shift_share(snapshot)
    audit_snapshot(eurostat_dir)
    panel = _read_panel(eurostat_dir / "processed" / "eu27_quarterly.csv")
    growth: dict[str, dict[str, float]] = {}
    for (geo, quarter), row in panel.items():
        if row.get("gdp") is not None:
            growth.setdefault(geo, {})[quarter] = float(row["gdp"])  # type: ignore[arg-type]
    inputs = load_inputs(snapshot, growth)
    quarters = quarter_sequence(SHIFT_START, LAST_ORIGIN)
    rows = []
    for margin in margins:
        sigma, n_sigma = sigma_u(panel, quarters, margin=margin)
        rows.append(_row("shift-share feasibility", margin, "GDP residual", "sigma_u", sigma,
                         n_sigma))
        results = {key: feasibility(instrument(inputs, key), sigma, key, margin=margin)
                   for key in PARTNER_SETS}
        for key, result in results.items():
            rows.append(_row("shift-share feasibility", margin, key, "expected_t",
                             result.central.expected_t, result.n))
        rows.append(_row("shift-share feasibility", margin, "P1 rule", "decision",
                         decision(results), results["P1"].n))
    return rows


def sign_restrictions(eurostat_dir: Path, deflator_dir: Path, margin: int = 1,
                      draws: int = 1000, rotations: int = 20000) -> list[dict[str, Any]]:
    """The primary sign-restricted SVAR with a widened window."""
    audit = audit_snapshot(eurostat_dir)
    audit_deflator(deflator_dir)
    panel = _read_panel(eurostat_dir / "processed" / "eu27_quarterly.csv")
    data = var_data(panel, load_deflator(deflator_dir), "2008-Q1",
                    str(audit["analysis_period"]["end"]))
    result = estimate(data, replace(SVAR_PRIMARY, pandemic_margin=margin), draws, rotations)
    rows = [_row("sign-restricted SVAR", margin, "all draws", "nonempty_share",
                 result.nonempty_share, result.n)]
    for shock in ("demand", "technology"):
        for h in (4, 8):
            s = result.summary(f"{shock}:okun:{h}")
            bounded = math.isfinite(s.region_lower) and math.isfinite(s.region_upper)
            rows.append(_row("sign-restricted SVAR", margin, f"{shock} Okun ratio h={h}",
                             "robust_region", "bounded" if bounded else "unbounded",
                             result.n))
    for shock in ("demand", "technology", "labour_supply"):
        s = result.summary(f"{shock}:fevd_du:{FEVD_HORIZON}")
        rows.append(_row("sign-restricted SVAR", margin, f"{shock} variance share of du",
                         "robust_region_upper", s.region_upper, result.n))
    return rows


def committed_svar(directory: Path) -> list[dict[str, Any]]:
    """The registered (margin 0) values of the same statistics, from the committed tables."""
    with (directory / "specifications.csv").open(encoding="utf-8") as stream:
        spec = next(r for r in csv.DictReader(stream) if r["spec"] == "primary")
    with (directory / "summaries.csv").open(encoding="utf-8") as stream:
        summaries = {r["target"]: r for r in csv.DictReader(stream) if r["spec"] == "primary"}
    rows = [_row("sign-restricted SVAR", 0, "all draws", "nonempty_share",
                 float(spec["nonempty_share"]), spec.get("n", ""))]
    for shock in ("demand", "technology"):
        for h in (4, 8):
            s = summaries[f"{shock}:okun:{h}"]
            bounded = math.isfinite(float(s["region_lower"])) and math.isfinite(
                float(s["region_upper"]))
            rows.append(_row("sign-restricted SVAR", 0, f"{shock} Okun ratio h={h}",
                             "robust_region", "bounded" if bounded else "unbounded",
                             spec.get("n", "")))
    for shock in ("demand", "technology", "labour_supply"):
        s = summaries[f"{shock}:fevd_du:{FEVD_HORIZON}"]
        rows.append(_row("sign-restricted SVAR", 0, f"{shock} variance share of du",
                         "robust_region_upper", float(s["region_upper"]), spec.get("n", "")))
    return rows


def write(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "window_sensitivity.csv"
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                             for k, v in r.items()})
    return target


def render(rows: list[dict[str, Any]], command: str) -> str:
    """A side-by-side table of the registered window (margin 0) and the widened one."""
    keys = list(dict.fromkeys((r["design"], r["target"], r["statistic"]) for r in rows))

    def cell(design: str, target: str, statistic: str, margin: int) -> str:
        found = [r for r in rows if (r["design"], r["target"], r["statistic"], r["margin"])
                 == (design, target, statistic, margin)]
        if not found:
            return "-"
        v = found[0]["value"]
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = [
        "# Pandemic-window sensitivity (post-hoc, not pre-registered)", "",
        "Generated by `eu-okun window-sensitivity`. Added after an external review on "
        "2026-10-19; the registered results are unchanged and remain primary.", "",
        "The registered rules exclude an observation if any period of its variables lies in "
        "2020-2021: quarterly local projections from t-4, monthly from t-12, the structural "
        "VAR from t-2. The change at the longest lag is measured from the level one period "
        "earlier, so that level can fall in 2021 while the observation is kept. The widened "
        "window (margin 1) also excludes those observations. The shift-share feasibility "
        "study has the same gap: its origins keep t-4 to t clear, but the lag-4 controls of "
        "the GDP residual scale use the level at t-5 (added after a second review on "
        "2026-10-20).", "",
        "| Design | Target | Statistic | Registered window | Widened by one period |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for design, target, statistic in keys:
        lines.append(f"| {design} | {target} | {statistic.replace('_', ' ')} | "
                     f"{cell(design, target, statistic, 0)} | "
                     f"{cell(design, target, statistic, 1)} |")
    lines += ["", "Rows: country-period observations used. Sets: `all` and `two rays` are "
              "unbounded Anderson-Rubin sets. Variance shares are of the change in "
              "unemployment at h = 8 (upper bound of the robust region). Shift-share: the "
              "expected first-stage t of the central calibration for each partner set, and "
              "the registered go/no-go rule (go only if P1 reaches 3.16).", "",
              "Reproduce:", "", "```bash", command, "```", ""]
    return "\n".join(lines)


def run(eurostat_dir: Path, shocks_dir: Path, monthly_dir: Path, deflator_dir: Path,
        svar_dir: Path, output_dir: Path, report: Path | None = None, draws: int = 1000,
        rotations: int = 20000,
        shift_share_dir: Path | None = None) -> list[dict[str, Any]]:
    rows = local_projections(eurostat_dir, shocks_dir, monthly_dir)
    rows += committed_svar(svar_dir)
    rows += sign_restrictions(eurostat_dir, deflator_dir, 1, draws, rotations)
    if shift_share_dir is not None:
        rows += shift_share(eurostat_dir, shift_share_dir)
    write(output_dir, rows)
    if report is not None:
        command = (f"eu-okun window-sensitivity --output-dir {output_dir.as_posix()} "
                   f"--report {report.as_posix()}")
        report.write_text(render(rows, command), encoding="utf-8", newline="\n")
    return rows
