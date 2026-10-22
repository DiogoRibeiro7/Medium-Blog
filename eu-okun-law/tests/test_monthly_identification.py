"""Tests of the monthly identification study: the monthly snapshot (download, freezing,
audit), the country classification of the protocol, local projections with known
responses, the outcome-timing placebo, the predictability test's size with a
zero-inflated shock, and the report. Everything runs on synthetic data; no network."""

from __future__ import annotations

import csv
import io
import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eu_okun.contract import EU27, EurostatDataError
from eu_okun.external_shocks import to_monthly
from eu_okun.monthly_data import (
    MONTHLY_SERIES,
    UNEMPLOYMENT_METHOD,
    audit_monthly_snapshot,
    decode_monthly,
    load_monthly_panel,
    month_sequence,
    primary_countries,
    save_monthly_snapshot,
)
from eu_okun.monthly_structural import (
    PRIMARY,
    SHOCKS,
    MonthlyArrays,
    country_responses,
    entry_month,
    estimate_shock,
    in_euro_area,
    lp_design,
    monthly_arrays,
    placebo,
    predictability,
    sample_countries,
    touches_pandemic,
)

ROOT = Path(__file__).resolve().parents[1]
SYN = replace(SHOCKS["OIL"], key="SYN", series="s", scale=1.0)


# ----------------------------------------------------------------------------------
# Months, shocks and the classification
# ----------------------------------------------------------------------------------


def test_months_and_native_monthly_shocks() -> None:
    assert month_sequence("2019-11", "2020-02") == ["2019-11", "2019-12", "2020-01", "2020-02"]
    with pytest.raises(ValueError, match="End month"):
        month_sequence("2020-02", "2020-01")
    with pytest.raises(ValueError, match="format"):
        month_sequence("2020-13", "2021-01")
    assert to_monthly({(2020, 2): 1.5, (2019, 12): -0.5}) == {"2019-12": -0.5, "2020-02": 1.5}
    with pytest.raises(ValueError):
        to_monthly({(2020, 0): 1.0})
    assert entry_month("LT") == "2015-01" and entry_month("SE") is None
    assert not in_euro_area("LT", "2014-12") and in_euro_area("LT", "2015-01")
    assert touches_pandemic(["2019-12", "2020-01"]) and not touches_pandemic(["2022-01"])


def test_classification_covers_the_eu27_and_matches_the_protocol() -> None:
    assert set(UNEMPLOYMENT_METHOD) == set(EU27)
    assert primary_countries() == ("AT", "CZ", "EL", "FI", "HU", "IT", "RO", "SE")
    assert [g for g in primary_countries() if in_euro_area(g, "2009-01")] == ["AT", "EL", "FI",
                                                                            "IT"]
    text = (ROOT / "MONTHLY_IDENTIFICATION_PROTOCOL.md").read_text(encoding="utf-8")
    table: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^\| (?:\*\*)?([DTAMIBU])(?:\*\*)? \| [^|]+ \| ([A-Z, ]+) \|", line)
        if match:
            for geo in match.group(2).split(", "):
                table[geo.strip()] = match.group(1)
    assert table == UNEMPLOYMENT_METHOD


# ----------------------------------------------------------------------------------
# Monthly snapshot
# ----------------------------------------------------------------------------------


def _payload(spec_index: int, months: list[str], missing: set[tuple[str, str]]) -> bytes:
    spec = MONTHLY_SERIES[spec_index]
    ids = [*spec.filters, "geo", "time"]
    dimension: dict[str, Any] = {k: {"category": {"index": {v: 0}}}
                                 for k, v in spec.filters.items()}
    dimension["geo"] = {"category": {"index": {g: i for i, g in enumerate(EU27)}}}
    dimension["time"] = {"category": {"index": {m: j for j, m in enumerate(months)}}}
    values, status = {}, {}
    for i, geo in enumerate(EU27):
        for j, month in enumerate(months):
            if (geo, month) not in missing:
                values[str(i * len(months) + j)] = (6.0 + 0.1 * j if spec.name == "unemployment"
                                                    else 100.0 + i + j)
            if geo == "EL" and spec.name == "ip":
                status[str(i * len(months) + j)] = "p"
    payload = {"class": "dataset", "version": "2.0", "updated": "2026-10-16", "id": ids,
               "size": [1] * len(spec.filters) + [len(EU27), len(months)],
               "dimension": dimension, "value": values, "status": status}
    return json.dumps(payload).encode()


def test_snapshot_is_frozen_cut_at_the_last_complete_month_and_audited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import eu_okun.monthly_data as md

    months = ["2000-01", "2000-02", "2000-03", "2000-04"]
    payloads = {"une_rt_m": _payload(0, months, set()),
                "sts_inpr_m": _payload(1, months, {("MT", "2000-04")})}

    class Response(io.BytesIO):
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    def fake_urlopen(request: Any, timeout: int) -> Response:
        url = request.full_url
        assert "sinceTimePeriod=2000-01" in url
        return Response(payloads["une_rt_m" if "/une_rt_m?" in url else "sts_inpr_m"])

    monkeypatch.setattr(md, "urlopen", fake_urlopen)
    snapshot = tmp_path / "snap"
    manifest = save_monthly_snapshot(snapshot, "2000-01")
    assert manifest["period"] == {"start": "2000-01", "end": "2000-03"}
    assert manifest["country_month_rows"] == 27 * 3
    assert manifest["series"]["ip"]["flags"] == {"p": 3}
    audit_monthly_snapshot(snapshot)
    panel = load_monthly_panel(snapshot)
    assert panel[("AT", "2000-02")] == {"unemployment": 6.1, "ip": 101.0}

    processed = snapshot / "processed" / "eu27_monthly.csv"
    original = processed.read_text(encoding="utf-8")
    processed.write_text(original.replace("6.1", "6.2", 1), encoding="utf-8", newline="")
    with pytest.raises(EurostatDataError, match="manifest hash"):
        audit_monthly_snapshot(snapshot)
    processed.write_text(original, encoding="utf-8", newline="")
    audit_monthly_snapshot(snapshot)
    document = json.loads((snapshot / "snapshot_manifest.json").read_text(encoding="utf-8"))
    document["unemployment_method"]["countries"]["DE"] = "D"
    (snapshot / "snapshot_manifest.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(EurostatDataError, match="classification"):
        audit_monthly_snapshot(snapshot)
    document["unemployment_method"]["countries"]["DE"] = "T"
    (snapshot / "snapshot_manifest.json").write_text(json.dumps(document), encoding="utf-8")
    (snapshot / "raw" / "une_rt_m.json").write_bytes(b"{}")
    with pytest.raises(EurostatDataError, match="Raw monthly response changed"):
        audit_monthly_snapshot(snapshot)


def test_monthly_decoding_rejects_quarters() -> None:
    payload = json.loads(_payload(0, ["2000-01"], set()))
    payload["dimension"]["time"] = {"category": {"index": {"2000-Q1": 0}}}
    with pytest.raises(EurostatDataError, match="time period"):
        decode_monthly(payload, MONTHLY_SERIES[0])


# ----------------------------------------------------------------------------------
# Local projections with a known data-generating process
# ----------------------------------------------------------------------------------


def synthetic(months: list[str], shock: dict[str, float], seed: int = 0, impact: float = -0.8,
              kappa: float = -0.25, rho: float = 0.5,
              countries: tuple[str, ...] = EU27) -> MonthlyArrays:
    """Common shock s; dip_it = rho dip_i,t-1 + impact s_t + e; du_it = kappa dip_it + v."""
    rng = np.random.default_rng(seed)
    s = np.asarray([shock.get(m, 0.0) for m in months])
    e = rng.normal(0, 0.5, (len(countries), len(months)))
    v = rng.normal(0, 0.1, (len(countries), len(months)))
    dip = np.zeros_like(e)
    for t in range(len(months)):
        dip[:, t] = (rho * dip[:, t - 1] if t else 0.0) + impact * s[t] + e[:, t]
    lip = 400.0 + np.cumsum(dip, axis=1)
    u = 8.0 + np.cumsum(kappa * dip + v, axis=1)
    return MonthlyArrays(countries, tuple(months), u, lip)


def _shock(months: list[str], seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    return {m: float(rng.normal()) for m in months}


def test_local_projection_recovers_known_responses_and_ratio() -> None:
    months = month_sequence("1990-01", "2019-12")
    shock = _shock(months, 10)
    arrays = synthetic(months, shock)
    settings = replace(PRIMARY, first_origin="1991-02")
    results = estimate_shock(arrays, {"s": shock}, SYN, settings, horizons=(0, 6, 12))
    for r in results.responses:
        truth = sum(-0.8 * 0.5**j for j in range(r.h + 1))
        if r.outcome == "unemployment":
            truth *= -0.25
        assert abs(r.estimate - truth) < 3 * r.se, (r.outcome, r.h, r.estimate, truth)
    ratio = results.ratios[0]
    assert ratio.h == 12 and ratio.informative and ratio.lower <= -0.25 <= ratio.upper
    assert ratio.estimate == pytest.approx(-0.25, abs=0.03)
    assert abs(ratio.first_stage_t) > 5


def test_windows_membership_and_lag_alignment() -> None:
    months = month_sequence("2000-01", "2026-08")
    shock = _shock(months, 11)
    arrays = synthetic(months, shock, seed=1)
    h = 6
    design = lp_design(arrays, shock, h, replace(SHOCKS["MP"], series="s"), PRIMARY)
    geos = {arrays.countries[i] for i in np.unique(design.country)}
    assert geos == {"AT", "EL", "FI", "IT"}
    for i, t in zip(design.country, design.time, strict=True):
        assert arrays.months[t] >= "2009-01"
        assert in_euro_area(arrays.countries[i], arrays.months[t])
        assert not touches_pandemic(arrays.months[t - 12: t + h + 1])
    i, t = design.country[0], design.time[0]
    assert design.x[0, 0] == pytest.approx(shock[arrays.months[t]])
    assert design.x[0, design.names.index("du_l1")] == pytest.approx(
        arrays.u[i, t - 1] - arrays.u[i, t - 2])
    assert design.x[0, design.names.index("dip_l12")] == pytest.approx(
        arrays.lip[i, t - 12] - arrays.lip[i, t - 13])
    assert design.y["unemployment"][0] == pytest.approx(arrays.u[i, t + h] - arrays.u[i, t - 1])
    assert design.y["ip"][0] == pytest.approx(arrays.lip[i, t + h] - arrays.lip[i, t - 1])
    # Placebo: the outcome ends at t-1 and no control is dated t-1.
    pl = lp_design(arrays, shock, 0, SYN, PRIMARY, placebo=True)
    assert "du_l1" not in pl.names and "du_l2" in pl.names and "du_l13" in pl.names
    i, t = pl.country[0], pl.time[0]
    assert pl.y["unemployment"][0] == pytest.approx(arrays.u[i, t - 1] - arrays.u[i, t - 2])
    assert pl.x[0, pl.names.index("du_l2")] == pytest.approx(
        arrays.u[i, t - 2] - arrays.u[i, t - 3])
    # Samples: group D for the primary, all countries otherwise; IP-only drops du lags.
    assert [arrays.countries[i] for i in sample_countries(arrays, PRIMARY)] == list(
        primary_countries())
    ip_only = lp_design(arrays, shock, 0, SYN, replace(PRIMARY, sample="all", du_controls=False),
                        outcomes=("ip",))
    assert not any(n.startswith("du") for n in ip_only.names)
    assert len(np.unique(ip_only.country)) == 27
    long = lp_design(arrays, shock, 0, SYN, replace(PRIMARY, first_origin="2001-02"))
    assert arrays.months[int(long.time.min())] == "2001-02"


def test_placebo_is_quiet_on_clean_data_and_detects_centred_smoothing() -> None:
    months = month_sequence("1990-01", "2019-12")
    shock = _shock(months, 12)
    arrays = synthetic(months, shock, seed=2)
    settings = replace(PRIMARY, first_origin="1991-03")
    clean = {p.outcome: p for p in placebo(arrays, shock, SYN, settings, 1.0)}
    assert clean["unemployment"].p_value > 0.01 and clean["ip"].p_value > 0.01
    smoothed_u = arrays.u.copy()
    smoothed_u[:, 1:-1] = (arrays.u[:, :-2] + arrays.u[:, 1:-1] + arrays.u[:, 2:]) / 3
    smoothed = replace(arrays, u=smoothed_u)
    leaked = {p.outcome: p for p in placebo(smoothed, shock, SYN, settings, 1.0)}
    assert leaked["unemployment"].p_value < 1e-6
    assert leaked["unemployment"].estimate == pytest.approx(-0.25 * -0.8 / 3, abs=0.02)


@pytest.mark.integration
def test_predictability_keeps_its_size_with_a_zero_inflated_heavy_tailed_shock() -> None:
    months = month_sequence("2007-01", "2025-10")
    countries = primary_countries()
    rejections = 0
    for seed in range(150):
        rng = np.random.default_rng(1000 + seed)
        shock = {m: (0.0 if rng.random() < 0.25 else float(0.05 * rng.standard_t(3)))
                 for m in months}
        arrays = synthetic(months, {}, seed=seed, countries=countries)
        rejections += predictability(arrays, shock, SYN, range(len(countries))).p_value < 0.05
    assert rejections / 150 <= 0.10                  # nominal 5%; binomial sd 1.8%
    arrays = synthetic(months, {}, seed=99, countries=countries)
    mean_dip = arrays.dip.mean(axis=0)
    rng = np.random.default_rng(5)
    predictable = {m: float(mean_dip[j - 1] + 0.05 * rng.normal())
                   for j, m in enumerate(months) if j > 0}
    assert predictability(arrays, predictable, SYN, range(len(countries))).p_value < 1e-6


def test_country_responses_cover_the_truth() -> None:
    months = month_sequence("1990-01", "2019-12")
    shock = _shock(months, 13)
    arrays = synthetic(months, shock, seed=3)
    rows = country_responses(arrays, shock, SYN, 1.0, replace(PRIMARY, first_origin="1991-02"))
    ip = [r for r in rows if r.outcome == "ip"]
    assert len(ip) == 27 and {r.group for r in ip} == set(UNEMPLOYMENT_METHOD.values())
    truth = sum(-0.8 * 0.5**j for j in range(13))
    assert sum(r.low90 <= truth <= r.high90 for r in ip) >= 21


# ----------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------


def _synthetic_review_inputs() -> tuple[MonthlyArrays, dict[str, dict[str, float]]]:
    months = month_sequence("2000-01", "2026-08")
    rng = np.random.default_rng(20)
    shocks = {}
    for name in ("MP_median", "MP_pm", "pc1_hf", "CBI_median", "CBI_pm"):
        shocks[name] = {m: float(rng.normal(0, 0.05)) for m in months if m <= "2025-10"}
    for name in ("oil_surprise", "oil_news_shock"):
        shocks[name] = {m: float(rng.normal(0, 3)) for m in months if m <= "2025-12"}
    combined = {m: -20.0 * shocks["MP_median"].get(m, 0.0) - 0.1 * shocks["oil_surprise"][m]
                for m in months if m in shocks["oil_surprise"]}
    return synthetic(months, combined, seed=21, impact=1.0), shocks


@pytest.mark.integration
def test_monthly_report_renders(tmp_path: Path) -> None:
    from eu_okun.monthly_report import render, review_from_arrays, write_figures, write_tables

    arrays, shocks = _synthetic_review_inputs()
    review = review_from_arrays(arrays, shocks, quarterly_dir=ROOT / "docs" / "identification")
    assert set(review.primary) == {"MP", "OIL"}
    assert any(o.informative for o in review.primary["MP"].ratios)
    assert len(review.robustness) == 2 + 3 + 4 * 2
    write_tables(review, tmp_path)
    write_figures(review, tmp_path, "synthetic")
    report = render(review, ".", "eu-okun monthly-identification")
    for heading in ("## Summary", "## 2. Diagnostics", "## 4. Shock-conditional ratio",
                    "## 5. Robustness", "## 6. Countries", "## 7. Comparison with the quarterly",
                    "Demand against supply", "contrast only (not an identified shock)",
                    "A7 doubtful by construction"):
        assert heading in report, heading
    for name in ("responses.csv", "ratio_sets.csv", "ip_all_members.csv", "predictability.csv",
                 "placebo.csv", "country_responses.csv", "provenance.json",
                 "responses_MP.png", "responses_OIL.png", "countries.png"):
        assert (tmp_path / name).exists(), name
    with (tmp_path / "placebo.csv").open(encoding="utf-8") as stream:
        assert {(r["shock"], r["sample"]) for r in csv.DictReader(stream)} == {
            ("MP", "primary"), ("MP", "all"), ("OIL", "primary"), ("OIL", "all")}
    assert math.isfinite(review.predictability["MP"].f_statistic)


def test_monthly_arrays_from_a_panel() -> None:
    panel = {("AT", "2000-01"): {"unemployment": 5.0, "ip": 100.0},
             ("AT", "2000-02"): {"unemployment": None, "ip": 101.0}}
    arrays = monthly_arrays(panel, "2000-01", "2000-03", ("AT",))
    assert arrays.u[0, 0] == 5.0 and math.isnan(arrays.u[0, 1]) and math.isnan(arrays.lip[0, 2])
    assert arrays.dip[0, 1] == pytest.approx(100 * math.log(1.01))


@pytest.mark.reproduction
def test_committed_monthly_results_reproduce(tmp_path: Path) -> None:
    """The committed tables regenerate from the frozen snapshots (validation gate)."""
    from eu_okun.monthly_report import write_report

    committed = ROOT / "docs" / "monthly-identification"
    snapshots = sorted((ROOT / "snapshots").glob("eurostat-monthly-*"))
    if not (committed / "responses.csv").exists() or not snapshots:
        pytest.skip("No committed monthly identification results.")
    write_report(snapshots[-1], ROOT / "snapshots" / "external-shocks-2026-10-16", tmp_path,
                 quarterly_dir=ROOT / "docs" / "identification", figures=False)

    def read(path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    for name in ("responses.csv", "ratio_sets.csv", "predictability.csv", "placebo.csv",
                 "ip_all_members.csv", "country_responses.csv"):
        old, new = read(committed / name), read(tmp_path / name)
        assert len(old) == len(new), name
        for a, b in zip(old, new, strict=True):
            for key, value in a.items():
                try:
                    assert float(value) == pytest.approx(float(b[key]), rel=1e-7, abs=1e-9)
                except ValueError:
                    assert value == b[key], (name, key)
