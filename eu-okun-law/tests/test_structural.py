"""Tests of the identification study: shock files, aggregation, local projections with
known responses, the Anderson-Rubin (Fieller) set, clustering, membership and the
pandemic rule. Everything runs on synthetic data; no network access."""

from __future__ import annotations

import io
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats

from eu_okun.contract import quarter_sequence
from eu_okun.external_shocks import (
    audit_shock_snapshot,
    fetch_shocks,
    load_quarterly,
    read_monetary,
    read_oil,
    to_quarterly,
)
from eu_okun.structural import (
    SHOCKS,
    Settings,
    anderson_rubin_set,
    country_responses,
    estimate_shock,
    fit_lp,
    in_euro_area,
    long_run_meat,
    lp_design,
    panel_arrays,
    predictability,
    touches_pandemic,
)

Panel = dict[tuple[str, str], dict[str, float | None]]


# ----------------------------------------------------------------------------------
# Shock files
# ----------------------------------------------------------------------------------


def _monetary_csv() -> str:
    lines = ["year,month,pc1_hf,STOXX50_hf,MP_pm,CBI_pm,MP_median,CBI_median"]
    for month in range(1, 8):          # Jan..Jul: Q1, Q2 complete, Q3 incomplete
        lines.append(f"2020,{month},{0.01 * month},0,{0.02 * month},0,{0.03 * month},0")
    return "\n".join(lines) + "\n"


def _oil_xlsx(path: Path) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Monthly"
    ws.append(["Date", "Oil supply surprise series", "Oil supply news shock"])
    for month in range(1, 7):
        ws.append([f"2020M{month:02d}", float(month), -float(month)])
    wb.create_sheet("Daily")
    wb.save(path)


def test_parsing_and_quarterly_aggregation(tmp_path: Path) -> None:
    csv_path = tmp_path / "m.csv"
    csv_path.write_text(_monetary_csv(), encoding="utf-8")
    monetary = read_monetary(csv_path)
    quarterly = to_quarterly(monetary["MP_median"])
    assert quarterly == pytest.approx({"2020-Q1": 0.03 * 6, "2020-Q2": 0.03 * 15})
    xlsx = tmp_path / "o.xlsx"
    _oil_xlsx(xlsx)
    oil = read_oil(xlsx)
    assert to_quarterly(oil["oil_surprise"]) == {"2020-Q1": 6.0, "2020-Q2": 15.0}
    assert to_quarterly(oil["oil_news_shock"])["2020-Q2"] == -15.0
    with pytest.raises(ValueError):
        to_quarterly({(2020, 13): 1.0})


def test_fetch_records_hashes_and_the_audit_detects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import eu_okun.external_shocks as es

    xlsx = tmp_path / "fixture.xlsx"
    _oil_xlsx(xlsx)
    payloads = {"shocks_ecb_mpd_me_m.csv": _monetary_csv().encode(),
                "oilSupplyNewsShocks_2025M12.xlsx": xlsx.read_bytes()}

    class Response(io.BytesIO):
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    def fake_urlopen(url: str, timeout: int) -> Response:
        return Response(payloads[url.rsplit("/", 1)[1]])

    monkeypatch.setattr(es.urllib.request, "urlopen", fake_urlopen)
    snapshot = tmp_path / "snap"
    manifest = fetch_shocks(snapshot)
    assert {f["licence"] for f in manifest["files"]} == {"CC BY 4.0"}
    assert manifest["coverage"]["MP_median"] == ["2020-Q1", "2020-Q2"]
    audit_shock_snapshot(snapshot)
    assert load_quarterly(snapshot)["oil_surprise"]["2020-Q2"] == 15.0
    (snapshot / "raw" / "shocks_ecb_mpd_me_m.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(Exception, match="SHA-256"):
        audit_shock_snapshot(snapshot)
    with pytest.raises(Exception, match="does not match"):
        fetch_shocks(tmp_path / "again", expected_sha256={"monetary": "0" * 64})


# ----------------------------------------------------------------------------------
# Local projections with a known data-generating process
# ----------------------------------------------------------------------------------


def synthetic_panel(n: int = 15, first: str = "1980-Q1", last: str = "2009-Q4",
                    seed: int = 0, impact: float = -0.8, okun: float = -0.25,
                    ) -> tuple[Panel, dict[str, float], list[str]]:
    """Common iid shock s; g_it = 0.5 g_i,t-1 + impact*s_t + e; du = okun*g + v."""
    rng = np.random.default_rng(seed)
    quarters = quarter_sequence(first, last)
    shock = {q: float(rng.normal()) for q in quarters}
    countries = [f"C{k:02d}" for k in range(n)]
    panel: Panel = {}
    for geo in countries:
        g_prev, u, p = 0.0, 8.0, 70.0
        for q in quarters:
            g = 0.5 * g_prev + impact * shock[q] + float(rng.normal(0, 0.5))
            u = u + okun * g + float(rng.normal(0, 0.1))
            p = p + float(rng.normal(0, 0.1))
            panel[(geo, q)] = {"gdp": g, "unemployment": u, "participation": p}
            g_prev = g
    return panel, shock, countries


def test_local_projection_recovers_known_responses_and_okun_coefficient() -> None:
    panel, shock, countries = synthetic_panel()
    arrays = panel_arrays(panel, "1980-Q1", "2009-Q4", countries)
    spec = replace(SHOCKS["OIL"], key="SYN", series="s", scale=1.0)
    settings = Settings(exclude_pandemic=False)
    results = estimate_shock(arrays, {"s": shock}, spec, settings, horizons=(0, 2, 4))
    for r in results.responses:
        if r.outcome == "gdp":
            truth = sum(-0.8 * 0.5**j for j in range(r.h + 1))
            assert abs(r.estimate - truth) < 3 * r.se, (r.h, r.estimate, truth)
            assert r.low90 < r.estimate < r.high90
        if r.outcome == "unemployment":
            truth = -0.25 * sum(-0.8 * 0.5**j for j in range(r.h + 1))
            assert abs(r.estimate - truth) < 3 * r.se
    okun4 = next(o for o in results.okun if o.h == 4)
    assert okun4.informative and okun4.lower <= -0.25 <= okun4.upper
    assert okun4.estimate == pytest.approx(-0.25, abs=0.05)
    assert abs(okun4.first_stage_t) > 5


def test_anderson_rubin_set_matches_a_grid_and_is_unbounded_when_weak() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        a, b = rng.normal(0, 1), rng.normal(0, 2)
        m = rng.normal(size=(2, 2))
        v = m @ m.T * 0.1 + np.eye(2) * 0.05
        c = 1.7
        kind, lo, hi = anderson_rubin_set(a, b, v[0, 0], v[0, 1], v[1, 1], c)
        grid = np.linspace(-50, 50, 200001)
        accepted = (a - grid * b) ** 2 <= c * c * (v[0, 0] - 2 * grid * v[0, 1]
                                                   + grid * grid * v[1, 1])
        if kind == "bounded":
            inside = grid[accepted]
            assert inside.min() == pytest.approx(lo, abs=1e-3)
            assert inside.max() == pytest.approx(hi, abs=1e-3)
        elif kind == "two rays":
            assert accepted[0] and accepted[-1] and not accepted[(grid > lo) & (grid < hi)].any()
        else:
            assert accepted.all()
    kind, _, _ = anderson_rubin_set(0.3, 0.01, 0.04, 0.0, 0.04, 1.645)
    assert kind != "bounded"                         # first stage t = 0.05: weak


def test_clustered_and_driscoll_kraay_covariances() -> None:
    rng = np.random.default_rng(2)
    s = rng.normal(size=(10, 3))
    times = np.arange(10)
    np.testing.assert_allclose(long_run_meat(s, s, times, 0), s.T @ s)
    manual = s.T @ s + 0.5 * (s[1:].T @ s[:-1] + s[:-1].T @ s[1:])
    np.testing.assert_allclose(long_run_meat(s, s, times, 1), manual)
    gap = np.array([0, 1, 2, 5, 6, 7, 8, 9, 10, 11])  # calendar gap between 2 and 5
    meat = long_run_meat(s, s, gap, 1)
    pairs = [(j, j - 1) for j in range(1, 10) if gap[j] - gap[j - 1] == 1]
    expected = s.T @ s + 0.5 * sum(np.outer(s[j], s[k]) + np.outer(s[k], s[j]) for j, k in pairs)
    np.testing.assert_allclose(meat, expected)


def test_membership_pandemic_rule_and_lagged_controls() -> None:
    assert not in_euro_area("LT", "2014-Q4") and in_euro_area("LT", "2015-Q1")
    assert not in_euro_area("SE", "2025-Q1") and in_euro_area("HR", "2023-Q1")
    assert touches_pandemic(["2019-Q4", "2020-Q1"]) and not touches_pandemic(["2019-Q4"])
    quarters = quarter_sequence("2010-Q1", "2024-Q4")
    rng = np.random.default_rng(3)
    panel: Panel = {}
    for geo in ("LT", "DE"):
        for q in quarters:
            panel[(geo, q)] = {"gdp": float(rng.normal()), "unemployment": float(rng.normal()),
                               "participation": float(rng.normal())}
    shock = {q: float(rng.normal()) for q in quarters}
    arrays = panel_arrays(panel, "2010-Q1", "2024-Q4", ["LT", "DE"])
    design = lp_design(arrays, shock, 2, SHOCKS["MP"], Settings())
    rows_lt = design.time[design.country == 0]
    assert min(arrays.quarters[t] for t in rows_lt) >= "2015-Q1"
    for t in design.time:                            # window [t-4, t+2] avoids 2020-2021
        assert not touches_pandemic(arrays.quarters[t - 4: t + 3])
    # Controls are lags only: the first row's du_l1 is u[t-1] - u[t-2].
    i, t = design.country[0], design.time[0]
    du_l1 = design.x[0, design.names.index("du_l1")]
    assert du_l1 == pytest.approx(arrays.u[i, t - 1] - arrays.u[i, t - 2])
    assert design.x[0, 0] == pytest.approx(shock[arrays.quarters[t]])
    assert design.y["gdp"][0] == pytest.approx(arrays.g[i, t: t + 3].sum())


def test_predictability_and_country_responses() -> None:
    panel, shock, countries = synthetic_panel(n=8, seed=4)
    arrays = panel_arrays(panel, "1980-Q1", "2009-Q4", countries)
    spec = replace(SHOCKS["OIL"], key="SYN", series="s", scale=1.0)
    clean = predictability(arrays, shock, spec)
    assert clean.p_value > 0.01 and abs(clean.autocorrelation) < 0.3
    mean_g = arrays.g.mean(axis=0)
    rng = np.random.default_rng(5)
    predictable = {q: float(mean_g[j - 1] + 0.1 * rng.normal())
                   for j, q in enumerate(arrays.quarters) if j > 0}
    assert predictability(arrays, predictable, spec).p_value < 1e-6
    rows = country_responses(arrays, shock, spec, 1.0, Settings(exclude_pandemic=False))
    gdp = [r for r in rows if r.outcome == "gdp"]
    assert len(gdp) == 8
    truth = sum(-0.8 * 0.5**j for j in range(5))
    assert sum(r.low90 <= truth <= r.high90 for r in gdp) >= 6


def test_predictability_test_has_the_right_size_with_a_short_sample() -> None:
    spec = replace(SHOCKS["OIL"], key="S", series="s", scale=1.0)
    rejections = 0
    for seed in range(200):
        panel, shock, countries = synthetic_panel(n=4, seed=seed, first="2009-Q1",
                                                  last="2025-Q4")
        arrays = panel_arrays(panel, "2009-Q1", "2025-Q4", countries)
        rejections += predictability(arrays, shock, spec).p_value < 0.05
    assert rejections / 200 <= 0.10                   # nominal 5%; binomial sd 1.5%


def test_fit_rejects_too_few_rows() -> None:
    panel, shock, countries = synthetic_panel(n=2, first="2000-Q1", last="2002-Q4")
    arrays = panel_arrays(panel, "2000-Q1", "2002-Q4", countries)
    spec = replace(SHOCKS["OIL"], series="s", scale=1.0)
    design = lp_design(arrays, shock, 4, spec, Settings(exclude_pandemic=False))
    with pytest.raises(ValueError):
        fit_lp(design)
    assert math.isfinite(stats.t.ppf(0.95, 10))


def test_identification_report_renders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import eu_okun.structural_report as sr
    from eu_okun.contract import EU27

    rng = np.random.default_rng(6)
    quarters = quarter_sequence("2008-Q1", "2026-Q2")
    names = ("MP_median", "MP_pm", "pc1_hf", "CBI_median", "CBI_pm", "oil_surprise",
             "oil_news_shock")
    shocks = {n: {q: float(rng.normal(0, 0.05 if n.startswith(("MP", "pc1", "CBI")) else 3))
                  for q in quarters if n.startswith("oil") or q <= "2025-Q3"} for n in names}
    panel: Panel = {}
    for geo in EU27:
        g_prev, u, p = 0.0, 8.0, 70.0
        for q in quarters:
            g = (0.5 * g_prev - 4.0 * shocks["MP_median"].get(q, 0.0)
                 - 0.05 * shocks["oil_surprise"][q] + float(rng.normal(0, 0.5)))
            u, p = u - 0.3 * g + float(rng.normal(0, 0.1)), p + float(rng.normal(0, 0.1))
            panel[(geo, q)] = {"gdp": g, "unemployment": u, "participation": p}
            g_prev = g
    monkeypatch.setattr(sr, "audit_snapshot", lambda d: {
        "analysis_period": {"end": "2026-Q2"}, "snapshot_manifest_sha256": "m",
        "processed_csv_sha256": "c"})
    monkeypatch.setattr(sr, "_read_panel", lambda path: panel)
    monkeypatch.setattr(sr, "audit_shock_snapshot", lambda d: {
        "files": [{"file": "f", "sha256": "s", "commit": "c", "licence": "CC BY 4.0"}],
        "quarterly_csv_sha256": "q"})
    monkeypatch.setattr(sr, "load_quarterly", lambda d: shocks)
    review = sr.write_report(tmp_path, tmp_path, tmp_path / "out", figures=True)
    assert set(review.primary) == {"MP", "OIL"}
    assert any(o.informative for o in review.primary["MP"].okun)
    report = (tmp_path / "out" / "IDENTIFICATION_RESULTS.md").read_text(encoding="utf-8")
    for heading in ("## Summary", "## 2. Diagnostics", "## 4. Shock-conditional Okun",
                    "## 5. Robustness", "## 6. Countries", "Demand against supply"):
        assert heading in report
    assert "contrast only (not an identified shock)" in report
    for name in ("responses.csv", "okun_sets.csv", "predictability.csv",
                 "country_responses.csv", "provenance.json", "responses_MP.png",
                 "responses_OIL.png", "countries.png"):
        assert (tmp_path / "out" / name).exists()
