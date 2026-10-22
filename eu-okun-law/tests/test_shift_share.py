"""Tests of the shift-share feasibility study: OECD parsing and snapshot audit, the
instrument (leave-one-out, missing shifts), the sample rules, the residualisation and its
weights, the benchmark power formula against a Monte Carlo first stage, the GDP residual
scale, the decision rule and the report. Synthetic data; no network."""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eu_okun.contract import EU27, EurostatDataError, quarter_sequence
from eu_okun.shift_share import (
    ISO3,
    NEAR,
    P1,
    PARTNER_SETS,
    Calibration,
    Inputs,
    audit_shift_share,
    decision,
    feasibility,
    fetch_shift_share,
    instrument,
    origins,
    p_bounded,
    parse_balance,
    parse_exposure,
    parse_shifts,
    residual_maker,
    sigma_u,
)

ROOT = Path(__file__).resolve().parents[1]
AREAS = (*P1, *NEAR, *(ISO3[g] for g in EU27))


def _csv(header: list[str], rows: list[list[Any]]) -> bytes:
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode()


def fdva_bytes(missing: tuple[str, str] | None = None) -> bytes:
    header = ["DATAFLOW", "MEASURE", "VALUE_ADDED_SOURCE_AREA", "VALUE_ADDED_SOURCE_ACTIVITY",
              "FINAL_DEMAND_AREA", "FINAL_DEMAND_ACTIVITY", "UNIT_MEASURE", "FREQ",
              "TIME_PERIOD", "OBS_VALUE", "UNIT_MULT"]
    rows = []
    for k, geo in enumerate(EU27):
        src = ISO3[geo]
        for year in ("2005", "2006", "2007"):
            rows.append(["X", "FDVA", src, "_T", "W", "_T", "USD", "A", year, 1000.0 + k, 6])
            for j, area in enumerate(AREAS):
                if (src, area) == missing:
                    continue
                value = 600.0 if area == src else 1.0 + 0.1 * j + 0.01 * k + 0.5 * (year == "2006")
                rows.append(["X", "FDVA", src, "_T", area, "_T", "USD", "A", year, value, 6])
    return _csv(header, rows)


def mainlv_bytes() -> bytes:
    header = ["DATAFLOW", "MEASURE", "REF_AREA", "ACTIVITY", "COUNTERPART_AREA", "UNIT_MEASURE",
              "FREQ", "TIME_PERIOD", "OBS_VALUE", "UNIT_MULT"]
    rows = []
    for k, geo in enumerate(EU27):
        for year in ("2005", "2006", "2007"):
            rows += [["X", "VALU", ISO3[geo], "_T", "W", "USD", "A", year, 1000.0, 6],
                     ["X", "VALU", ISO3[geo], "BTE", "W", "USD", "A", year, 200.0 + k, 6],
                     ["X", "EXGR", ISO3[geo], "_T", "W", "USD", "A", year, 400.0 + 10 * k, 6]]
    return _csv(header, rows)


def qna_bytes(adjustment: str = "Y") -> bytes:
    header = ["DATAFLOW", "FREQ", "ADJUSTMENT", "REF_AREA", "SECTOR", "COUNTERPART_SECTOR",
              "TRANSACTION", "INSTR_ASSET", "ACTIVITY", "EXPENDITURE", "UNIT_MEASURE",
              "PRICE_BASE", "TRANSFORMATION", "TABLE_IDENTIFIER", "TIME_PERIOD", "OBS_VALUE"]
    rng = np.random.default_rng(0)
    rows = [["X", "Q", adjustment, p, "S1", "S1", "B1GQ", "_Z", "_Z", "_Z", "PC", "L", "G1",
             "T0102", q, round(float(rng.normal(0.5, 1.0)), 4)]
            for p in (*P1, *NEAR) for q in quarter_sequence("2010-Q1", "2026-Q2")]
    return _csv(header, rows)


# ----------------------------------------------------------------------------------
# Parsing and snapshot
# ----------------------------------------------------------------------------------


def test_parsing_shares_balance_and_shifts() -> None:
    exposure = parse_exposure(fdva_bytes())
    row = next(r for r in exposure if r["geo"] == "AT" and r["partner"] == "USA")
    j = AREAS.index("USA")
    expected = np.mean([(1.0 + 0.1 * j + 0.5 * (y == "2006")) / 1000.0
                        for y in ("2005", "2006", "2007")])
    assert float(row["share"]) == pytest.approx(expected)
    assert len(exposure) == 27 * len(AREAS)
    with pytest.raises(EurostatDataError, match="lacks AUT->USA"):
        parse_exposure(fdva_bytes(missing=("AUT", "USA")))
    balance = parse_balance(mainlv_bytes())
    assert float(balance[0]["industry_share"]) == pytest.approx(0.2)
    shifts = parse_shifts(qna_bytes())
    assert {r["partner"] for r in shifts} == set(P1) | set(NEAR)
    with pytest.raises(EurostatDataError, match="seasonally adjusted"):
        parse_shifts(qna_bytes(adjustment="N"))


def test_fetch_freezes_and_the_audit_detects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import eu_okun.shift_share as ss

    class Response(io.BytesIO):
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    def fake_urlopen(request: Any, timeout: int) -> Response:
        url = request.full_url
        assert "csv" in request.get_header("Accept")
        if "DSD_TIVA_FDVA" in url:
            return Response(fdva_bytes())
        if "DSD_TIVA_MAINLV" in url:
            return Response(mainlv_bytes())
        return Response(qna_bytes())

    monkeypatch.setattr(ss, "urlopen", fake_urlopen)
    snapshot = tmp_path / "snap"
    manifest = fetch_shift_share(snapshot)
    assert {f["key"] for f in manifest["files"]} == {"tiva_fdva", "tiva_mainlv", "oecd_qna"}
    audit_shift_share(snapshot)
    exposure = snapshot / "processed" / "exposure.csv"
    text = exposure.read_text(encoding="utf-8")
    exposure.write_text(text.replace("AT,", "AU,", 1), encoding="utf-8", newline="")
    with pytest.raises(EurostatDataError, match="manifest hash"):
        audit_shift_share(snapshot)
    exposure.write_text(text, encoding="utf-8", newline="")
    (snapshot / "raw" / "oecd_qna.csv").write_bytes(b"tampered")
    with pytest.raises(EurostatDataError, match="SHA-256"):
        audit_shift_share(snapshot)


# ----------------------------------------------------------------------------------
# Instrument and sample
# ----------------------------------------------------------------------------------


def synthetic_inputs(seed: int = 1, first: str = "2010-Q1") -> Inputs:
    rng = np.random.default_rng(seed)
    exposure = {g: {p: float(rng.uniform(0.001, 0.05)) for p in AREAS} for g in EU27}
    shifts = {p: {q: float(rng.normal(0.5, 1.0)) for q in quarter_sequence(first, "2026-Q2")}
              for p in AREAS}
    balance = {g: {"industry_share": float(rng.uniform(0.1, 0.35)),
                   "export_share": float(rng.uniform(0.3, 1.5))} for g in EU27}
    return Inputs(exposure, shifts, balance)


def test_instrument_leaves_the_country_out_and_propagates_missing_shifts() -> None:
    inputs = synthetic_inputs()
    p3 = instrument(inputs, "P3")
    at, aut = EU27.index("AT"), PARTNER_SETS["P3"].index("AUT")
    assert np.all(p3.components[at, :, aut] == 0)
    de = PARTNER_SETS["P3"].index("DEU")
    q = p3.quarters.index("2015-Q3")
    assert p3.components[at, q, de] == pytest.approx(
        inputs.exposure["AT"]["DEU"] * inputs.shifts["DEU"]["2015-Q3"])
    del inputs.shifts["CHN"]["2015-Q3"]
    p1 = instrument(inputs, "P1")
    assert math.isnan(p1.z[at, q]) and math.isfinite(p1.z[at, q - 1])


def test_origins_respect_the_first_quarter_and_the_pandemic_rule() -> None:
    quarters = quarter_sequence("2010-Q1", "2026-Q2")
    kept = [quarters[t] for t in origins(quarters)]
    assert kept[0] == "2012-Q1" and kept[-1] == "2026-Q2"
    assert "2019-Q4" in kept and "2020-Q1" not in kept
    assert "2022-Q4" not in kept and "2023-Q1" in kept


# ----------------------------------------------------------------------------------
# Residualisation, weights and power
# ----------------------------------------------------------------------------------


def test_residualisation_matches_ols_and_weights_sum_to_one() -> None:
    panel = instrument(synthetic_inputs(2), "P1")
    res = feasibility(panel, 1.0, "P1")
    assert sum(res.partner_weights.values()) == pytest.approx(1.0)
    assert sum(res.country_weights.values()) == pytest.approx(1.0)
    # Independent check of S with an explicit regression.
    z, rows = panel.z, []
    for t in origins(panel.quarters):
        for i in range(27):
            rows.append((i, t))
    x = np.column_stack([
        np.asarray([[1.0 if i == k else 0.0 for k in range(27)] for i, _ in rows]),
        np.asarray([[1.0 if t == s else 0.0 for s in sorted({t for _, t in rows})[1:]]
                    for _, t in rows]),
        np.asarray([[z[i, t - lag] for lag in range(1, 5)] for i, t in rows])])
    y = np.asarray([z[i, t] for i, t in rows])
    resid = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    assert res.s == pytest.approx(float(resid @ resid), rel=1e-8)
    m = residual_maker([0, 0, 1, 1], [0, 1, 0, 1], None)
    assert np.allclose(m @ np.ones(4), 0)


def test_expected_t_and_p_bounded_match_a_simulated_first_stage() -> None:
    panel = instrument(synthetic_inputs(3), "P1")
    sigma, beta = 0.8, 1.5
    res = feasibility(panel, sigma, "P1")
    expected = beta * math.sqrt(res.s) / sigma
    rows = [(i, t) for t in origins(panel.quarters) for i in range(27)]
    z = panel.z
    country = [i for i, _ in rows]
    time = [t for _, t in rows]
    lags = np.asarray([[z[i, t - lag] for lag in range(1, 5)] for i, t in rows])
    m = residual_maker(country, time, lags)
    zt = m @ np.asarray([z[i, t] for i, t in rows])
    rng = np.random.default_rng(4)
    ts = []
    for _ in range(400):
        g = (beta * np.asarray([z[i, t] for i, t in rows]) + 0.3 * np.asarray(country)
             - 0.2 * np.asarray(time) + rng.normal(0, sigma, len(rows)))
        gt = m @ g
        b = float(zt @ gt / (zt @ zt))
        resid = gt - b * zt
        dof = len(rows) - 27 - len(set(time)) + 1 - 4 - 1
        se = math.sqrt(float(resid @ resid) / dof / float(zt @ zt))
        ts.append(b / se)
    assert np.mean(ts) == pytest.approx(expected, rel=0.08)
    assert np.mean(np.abs(ts) > 1.645) == pytest.approx(p_bounded(expected), abs=0.06)
    assert p_bounded(0.0) == pytest.approx(0.10, abs=1e-6)


def test_sigma_u_recovers_a_known_scale_and_the_decision_rule() -> None:
    quarters = quarter_sequence("2010-Q1", "2026-Q2")
    rng = np.random.default_rng(5)
    panel: dict[tuple[str, str], dict[str, float | None]] = {}
    for k, geo in enumerate(EU27):
        u, p = 8.0, 60.0
        for j, q in enumerate(quarters):
            u += float(rng.normal(0, 0.2))
            p += float(rng.normal(0, 0.2))
            panel[(geo, q)] = {"gdp": 0.1 * k + 0.05 * j % 3 + float(rng.normal(0, 0.7)),
                               "unemployment": u, "participation": p}
    sd, n = sigma_u(panel, quarters)
    assert sd == pytest.approx(0.7, rel=0.08) and n > 1000
    panel_res = feasibility(instrument(synthetic_inputs(6), "P1"), 1.0, "P1")
    central = panel_res.central

    def with_t(value: float) -> Any:
        return type(panel_res)(**{**panel_res.__dict__, "calibrations": [
            Calibration(1.0, 1.0, "central", value, p_bounded(value))]})

    assert decision({"P1": with_t(3.2)}) == "go"
    assert decision({"P1": with_t(3.1)}) == "no go"
    assert panel_res.beta_needed == pytest.approx(3.16 / central.expected_t)


# ----------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------


def test_report_renders(tmp_path: Path) -> None:
    from eu_okun.shift_share_report import render, review_from_inputs, write_figure, write_tables

    inputs = synthetic_inputs(7)
    quarters = quarter_sequence("2008-Q1", "2026-Q2")
    rng = np.random.default_rng(8)
    eurostat = {(g, q): {"gdp": float(rng.normal(0.4, 0.8)), "unemployment": 8.0 + 0.01 * j,
                         "participation": 60.0 + 0.01 * j}
                for g in EU27 for j, q in enumerate(quarters)}
    review = review_from_inputs(inputs, eurostat)
    assert set(review.results) == {"P1", "P2", "P3"}
    write_tables(review, tmp_path)
    write_figure(review, tmp_path, "synthetic")
    report = render(review, ".", "eu-okun shift-share")
    for text in ("## Decision", "## 1. Exposure", "## 2. Identifying variation",
                 "## 3. Where the variation comes from", "## 4. What this means"):
        assert text in report, text
    assert ("**Go.**" in report) == (review.decision == "go")
    for name in ("feasibility.csv", "calibrations.csv", "partner_weights.csv",
                 "country_weights.csv", "exposure_summary.csv", "balance.csv",
                 "provenance.json", "exposure.png"):
        assert (tmp_path / name).exists(), name


@pytest.mark.reproduction
def test_committed_shift_share_results_reproduce(tmp_path: Path) -> None:
    """The committed tables regenerate from the frozen snapshots (validation gate)."""
    from eu_okun.shift_share_report import write_report

    committed = ROOT / "docs" / "shift-share"
    snapshots = sorted((ROOT / "snapshots").glob("shift-share-*"))
    if not (committed / "feasibility.csv").exists() or not snapshots:
        pytest.skip("No committed shift-share results.")
    write_report(snapshots[-1], ROOT / "snapshots" / "eurostat-2026-10-15", tmp_path,
                 figures=False)
    for name in ("feasibility.csv", "calibrations.csv", "partner_weights.csv",
                 "country_weights.csv", "exposure_summary.csv", "balance.csv"):
        with (committed / name).open(encoding="utf-8") as a, (tmp_path / name).open(
                encoding="utf-8") as b:
            old, new = list(csv.DictReader(a)), list(csv.DictReader(b))
        assert len(old) == len(new), name
        for x, y in zip(old, new, strict=True):
            for key, value in x.items():
                try:
                    assert float(value) == pytest.approx(float(y[key]), rel=1e-7, abs=1e-12)
                except ValueError:
                    assert value == y[key], (name, key)
