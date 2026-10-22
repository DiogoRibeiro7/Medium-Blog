"""Tests of the country-heterogeneity case study: data decoding and snapshot audit, the
margin identity, the synchronised block bootstrap, elasticities on known data, the
prediction statistics, Holm and permutation inference, the pair rule and the report.
Synthetic data; no network."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eu_okun.contract import EU27, EurostatDataError, quarter_sequence
from eu_okun.heterogeneity import (
    FIRST,
    PRIMARY,
    Estimate,
    Rows,
    Run,
    block_counts,
    build_country,
    correlations,
    estimate,
    evaluate,
    holm,
    lag_rows,
    pair_results,
    quarters_for,
    run,
    spearman,
    two_sided_p,
    weighted_ols,
)
from eu_okun.heterogeneity_data import (
    SERIES,
    audit_heterogeneity,
    decode_available,
    fetch_heterogeneity,
    parse_epl,
)

ROOT = Path(__file__).resolve().parents[1]


def synthetic(seed: int = 0) -> tuple[list[str], dict[Any, Any], dict[Any, Any]]:
    """Base panel and extra series with known elasticities; LU has a commuting margin."""
    rng = np.random.default_rng(seed)
    quarters = quarters_for()
    base: dict[tuple[str, str], dict[str, float | None]] = {}
    extra: dict[tuple[str, str], dict[str, Any]] = {}
    for k, geo in enumerate(EU27):
        hours, emp, nc, temp, perm = 1e5, 1e3, 1e3, 100.0, 900.0
        u, p = 8.0, 60.0
        for q in quarters:
            g = float(rng.normal(0.4, 0.8))
            e = 0.4 * g + float(rng.normal(0, 0.1))
            comm = (0.3 * g if geo == "LU" else 0.0) + float(rng.normal(0, 0.05))
            emp *= math.exp(e / 100)
            nc *= math.exp((e - comm) / 100)
            hours *= math.exp((0.6 * g + float(rng.normal(0, 0.1))) / 100)
            temp *= math.exp((1.5 * g + float(rng.normal(0, 0.5))) / 100)
            perm *= math.exp((0.3 * g + float(rng.normal(0, 0.2))) / 100)
            u += -0.3 * g + float(rng.normal(0, 0.05))
            p += 0.05 * g + float(rng.normal(0, 0.05))
            base[(geo, q)] = {"gdp": g, "unemployment": u if q >= "2009-Q1" else None,
                              "participation": p if q >= "2009-Q1" else None}
            extra[(geo, q)] = {
                "emp_dc": emp, "emp_nc": nc, "emp_a10": emp, "hours": hours,
                "temp": temp, "employees": temp + perm, "temp_share": 10.0 + k * 0.5,
                "pt_share": 20.0 - k * 0.3,
                **{f"{key}_flag": ("b" if q == "2014-Q1" and key in ("temp", "employees")
                                  else "") for key in SERIES}}
    return quarters, base, extra


def test_margin_identity_commuting_definition_and_break_flags() -> None:
    quarters, base, extra = synthetic()
    cd = build_country("LU", quarters, base, extra)
    m = cd.measures
    ok = np.isfinite(m["prod"]) & np.isfinite(m["hpw"]) & np.isfinite(m["e_dom"])
    assert np.allclose((m["prod"] + m["hpw"] + m["e_dom"])[ok], cd.g[ok])
    assert np.allclose(m["comm"][ok], (m["e_dom_pe"] - m["e_res"])[ok])
    i = quarters.index("2014-Q1")
    assert all(math.isnan(cd.yy["temp"][t]) for t in range(i, i + 5))
    assert math.isfinite(cd.yy["temp"][i + 5]) and math.isnan(cd.yy["temp"][3])


def test_crisis_reports_unpublished_series_as_missing_not_zero() -> None:
    from eu_okun.heterogeneity import crisis

    quarters, base, extra = synthetic(5)
    for q in quarters:
        extra[("PT", q)]["emp_a10"] = None
    data = {g: build_country(g, quarters, base, extra) for g in ("ES", "PT")}
    rows = {r["geo"]: r for r in crisis(data, quarters)}
    assert math.isnan(rows["PT"]["e_dom"]) and math.isnan(rows["PT"]["hpw"])
    assert math.isfinite(rows["PT"]["prod"]) and math.isfinite(rows["ES"]["e_dom"])
    assert rows["ES"]["du_from"] == "2009-Q2"


def test_block_counts_and_weighted_ols_match_an_explicit_resample() -> None:
    counts = block_counts(30, 4, 50, np.random.default_rng(1))
    assert counts.shape == (50, 30) and np.all(counts.sum(axis=1) == 30)
    rng = np.random.default_rng(2)
    x = np.column_stack([np.ones(30), rng.normal(size=(30, 2))])
    y = x @ np.array([0.5, 1.0, -0.4]) + rng.normal(0, 0.2, 30)
    rows = Rows(y, x, np.arange(30))
    coef = weighted_ols(rows, counts)
    for b in range(5):
        idx = np.repeat(np.arange(30), counts[b].astype(int))
        explicit = np.linalg.lstsq(x[idx], y[idx], rcond=None)[0]
        assert np.allclose(coef[b], explicit)


def test_elasticities_recover_known_values_and_margins_sum_to_one() -> None:
    quarters, base, extra = synthetic(3)
    data = {g: build_country(g, quarters, base, extra) for g in ("ES", "PT", "NL", "LU")}
    r = run(data, quarters, PRIMARY, replicates=199)
    es = r.get("ES", "e_dom")
    assert es is not None and es.point == pytest.approx(0.4, abs=0.06)
    lo, hi = es.interval()
    assert lo < 0.4 < hi
    total = sum(r.estimates[("NL", m)].point for m in ("prod", "hpw", "e_dom"))
    assert total == pytest.approx(1.0, abs=1e-9)
    draws = sum(r.estimates[("NL", m)].draws for m in ("prod", "hpw", "e_dom"))
    assert np.allclose(draws, 1.0)
    assert r.get("LU", "comm") is not None
    assert r.estimates[("LU", "comm")].point == pytest.approx(0.3, abs=0.06)
    assert r.get("ES", "neg_du").point == pytest.approx(0.3, abs=0.05)  # type: ignore[union-attr]
    assert r.estimates[("ES", "temp")].point > r.estimates[("ES", "perm")].point


def test_rows_respect_the_pandemic_window_and_the_sample() -> None:
    quarters = quarter_sequence("2008-Q1", "2026-Q2")
    y = np.ones(len(quarters))
    rows = lag_rows(y, np.ones(len(quarters)), quarters, 2, FIRST, "2026-Q2")
    kept = [quarters[t] for t in rows.q]
    assert kept[0] == FIRST and "2022-Q1" not in kept and "2022-Q2" not in kept
    assert "2022-Q3" in kept and "2019-Q4" in kept and "2020-Q1" not in kept
    assert estimate(Rows(y[:3], np.ones((3, 4)), np.arange(3)), np.ones((2, 3)), 0) is None


def test_p_values_and_holm() -> None:
    assert two_sided_p(np.full(999, 1.0)) == pytest.approx(2 / 1000)
    assert two_sided_p(np.r_[np.full(500, -1.0), np.full(499, 1.0)]) == pytest.approx(1.0)
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def _est(point: float, sd: float, seed: int) -> Estimate:
    return Estimate(point, np.random.default_rng(seed).normal(point, sd, 999), 50)


def test_pair_rule_remains_disappears_reverses() -> None:
    est = {("ES", "du"): _est(-0.8, 0.05, 1), ("PT", "du"): _est(-0.3, 0.05, 2),
           ("ES", "e_dom"): _est(0.7, 0.05, 3), ("PT", "e_dom"): _est(0.3, 0.05, 4),
           ("ES", "h"): _est(0.5, 0.3, 5), ("PT", "h"): _est(0.5, 0.3, 6),
           ("ES", "e_res"): _est(0.1, 0.02, 7), ("PT", "e_res"): _est(0.5, 0.02, 8)}
    r = Run(PRIMARY, est, {}, 999)
    status = {(p.a, p.b, p.measure): p.status for p in pair_results(r)}
    assert status[("ES", "PT", "neg_du")] == "different"
    assert status[("ES", "PT", "e_dom")] == "remains"
    assert status[("ES", "PT", "h")] == "disappears"
    assert status[("ES", "PT", "e_res")] == "reverses"
    assert status[("ES", "NL", "e_dom")] == "not available"


def test_spearman_ties_and_permutation_inference() -> None:
    assert spearman(np.array([1.0, 2, 2, 3]), np.array([1.0, 2, 3, 4])) == pytest.approx(
        0.9486832980505138)
    theta = {g: -0.01 * k for k, g in enumerate(EU27)}
    ind = {g: {"temp_share": float(k), "pt_share": float((k * 7) % 27), "epl_r": 1.0 + k,
               "epl_t": float(k % 3), "beta_hpw": float(k), "commuting": 1.0}
           for k, g in enumerate(EU27)}
    res = {c.indicator: c for c in correlations(theta, ind, "test", permutations=999)}
    assert res["temp_share"].rho == pytest.approx(-1.0)
    assert res["temp_share"].verdict == "supported"            # predicted negative
    assert res["epl_r"].verdict == "contradicted"              # predicted positive
    assert res["beta_hpw"].verdict == "contradicted"
    assert res["pt_share"].holm > 0.05
    assert math.isnan(res["commuting"].rho) and res["commuting"].verdict == "not determined"


# ----------------------------------------------------------------------------------
# Data module
# ----------------------------------------------------------------------------------


def _payload(spec_key: str, geos: list[str]) -> dict[str, Any]:
    spec = SERIES[spec_key]
    quarters = quarter_sequence("2008-Q1", "2008-Q2")
    dims: dict[str, Any] = {k: {"category": {"index": {v: 0}}} for k, v in spec.filters.items()}
    dims["geo"] = {"category": {"index": {g: i for i, g in enumerate(geos)}}}
    dims["time"] = {"category": {"index": {q: j for j, q in enumerate(quarters)}}}
    return {"class": "dataset", "version": "2.0", "updated": "2026-10-16",
            "id": [*spec.filters, "geo", "time"],
            "size": [1] * len(spec.filters) + [len(geos), 2], "dimension": dims,
            "value": {str(i * 2 + j): 100.0 + i for i in range(len(geos)) for j in range(2)}}


EPL_CSV = ("DATAFLOW,FREQ,REF_AREA,MEASURE,UNIT_MEASURE,VERSION,TIME_PERIOD,OBS_VALUE\n"
           "X,A,ESP,EPL_R,0_TO_6,VERSION3,2008,2.4\nX,A,ESP,EPL_T,0_TO_6,VERSION3,2008,3.7\n"
           "X,A,ESP,EPL_R,0_TO_6,VERSION2,2008,9.9\nX,A,PRT,EPL_R,0_TO_6,VERSION3,2008,4.4\n")


def test_decoding_keeps_unpublished_countries_missing_and_epl_filtering() -> None:
    cells = decode_available(_payload("emp_dc", [g for g in EU27 if g != "BE"]),
                             SERIES["emp_dc"])
    assert ("BE", "2008-Q1") not in cells and cells[("AT", "2008-Q1")].value == 100.0
    assert parse_epl(EPL_CSV.encode()) == [{"geo": "ES", "epl_r": "2.4", "epl_t": "3.7"}]
    with pytest.raises(EurostatDataError):
        parse_epl(b"DATAFLOW,FREQ,REF_AREA,MEASURE,UNIT_MEASURE,VERSION,TIME_PERIOD,OBS_VALUE\n")


def test_fetch_freezes_and_the_audit_detects_tampering(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    import eu_okun.heterogeneity_data as hd

    def fake_fetch(spec: Any, start: str, end: str) -> tuple[bytes, dict[str, Any], str]:
        geos = [g for g in EU27 if not (spec.dataset == "namq_10_pe" and g == "BE")]
        key = next(k for k, s in SERIES.items() if s == spec)
        payload = _payload(key, geos)
        return json.dumps(payload).encode(), payload, f"url/{key}"

    class Response(io.BytesIO):
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    monkeypatch.setattr(hd, "fetch_dataset", fake_fetch)
    monkeypatch.setattr(hd, "urlopen", lambda request, timeout: Response(EPL_CSV.encode()))
    snap = tmp_path / "snap"
    manifest = fetch_heterogeneity(snap)
    assert len(manifest["files"]) == len(SERIES) + 1
    audit_heterogeneity(snap)
    with (snap / "processed" / "quarterly.csv").open(encoding="utf-8") as stream:
        rows = {(r["geo"], r["quarter"]): r for r in csv.DictReader(stream)}
    assert rows[("BE", "2008-Q1")]["emp_dc"] == "" and rows[("BE", "2008-Q1")]["hours"] != ""
    path = snap / "processed" / "epl.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("2.4", "2.5"), encoding="utf-8",
                    newline="")
    with pytest.raises(EurostatDataError, match="manifest hash"):
        audit_heterogeneity(snap)


# ----------------------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------------------


@pytest.mark.integration
def test_case_study_renders(tmp_path: Path) -> None:
    from eu_okun.heterogeneity_report import (
        render,
        study_from_data,
        write_figures,
        write_tables,
    )

    quarters, base, extra = synthetic(4)
    data = {g: build_country(g, quarters, base, extra) for g in EU27}
    epl = {g: {"epl_r": 2.0 + 0.1 * k, "epl_t": 1.0 + 0.05 * k} for k, g in enumerate(EU27[:20])}
    theta = {"bayes": {g: -0.3 + 0.01 * k for k, g in enumerate(EU27)},
             "classical": {g: -0.5 + 0.02 * k for k, g in enumerate(EU27)}}
    cs = study_from_data(data, quarters, epl, theta, replicates=999, permutations=99)
    assert len(cs.evaluations) == 14 and set(cs.sensitivity) == {"lags_1", "lags_4",
                                                                  "pre_2020", "block_8"}
    assert cs.evaluations[0].key == "P1.1" and cs.evaluations[0].verdict == "supported"
    write_tables(cs, tmp_path)
    write_figures(cs, tmp_path, "synthetic")
    report = render(cs, ".", "eu-okun heterogeneity")
    for text in ("## Summary", "## 3. Luxembourg", "## 4. The Netherlands",
                 "## 5. Temporary jobs", "## 6. Portugal and Spain", "## 7. Institutions",
                 "## 8. Do the country differences remain", "## 9. Sensitivity",
                 "Identified causal effects:** none", "cannot be tested reliably"):
        assert text in report, text
    for name in ("elasticities.csv", "predictions.csv", "pairs.csv", "correlations.csv",
                 "indicators.csv", "crisis.csv", "evidence.png", "margins.png",
                 "luxembourg.png", "crisis.png", "institutions.png"):
        assert (tmp_path / name).exists(), name
    assert len(evaluate(cs.primary)) == 14


@pytest.mark.reproduction
def test_committed_case_study_reproduces(tmp_path: Path) -> None:
    """The committed tables regenerate from the frozen snapshots (validation gate)."""
    from eu_okun.heterogeneity_report import write_report

    committed = ROOT / "docs" / "heterogeneity"
    snaps = sorted((ROOT / "snapshots").glob("heterogeneity-*"))
    if not (committed / "predictions.csv").exists() or not snaps:
        pytest.skip("No committed case-study results.")
    write_report(ROOT / "snapshots" / "eurostat-2026-10-15", snaps[-1], ROOT / "docs" / "review",
                 ROOT / "docs" / "bayes-robustness", tmp_path, figures=False)
    for name in ("elasticities.csv", "predictions.csv", "pairs.csv", "correlations.csv",
                 "indicators.csv", "crisis.csv"):
        with (committed / name).open(encoding="utf-8") as a, (tmp_path / name).open(
                encoding="utf-8") as b:
            old, new = list(csv.DictReader(a)), list(csv.DictReader(b))
        assert len(old) == len(new), name
        for x, y in zip(old, new, strict=True):
            for key, value in x.items():
                try:
                    va, vb = float(value), float(y[key])
                    if math.isnan(va):
                        assert math.isnan(vb), (name, key)
                    else:
                        assert va == pytest.approx(vb, rel=1e-7, abs=1e-10), (name, key)
                except ValueError:
                    assert value == y[key], (name, key)
