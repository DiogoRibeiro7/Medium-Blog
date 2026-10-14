"""Offline contract tests for filtered Eurostat JSON-stat observations."""

from __future__ import annotations

from typing import Any

import pytest

from eu_okun.eurostat import (
    SERIES,
    EurostatDataError,
    Observation,
    build_panel,
    decode_series,
    quarter_sequence,
    request_url,
)


def make_payload() -> dict[str, Any]:
    """A deliberately sparse response, indexed in geo-then-time order."""
    return {
        "class": "dataset", "version": "2.0",
        "id": ["freq", "unit", "geo", "time"],
        "size": [1, 1, 2, 2],
        "dimension": {
            "freq": {"category": {"index": {"Q": 0}}},
            "unit": {"category": {"index": {"CLV_PCH_PRE": 0}}},
            "geo": {"category": {"index": {"NL": 0, "PT": 1}}},
            "time": {"category": {"index": {"2025-Q1": 0, "2025-Q2": 1}}},
        },
        "value": {"0": 1.2, "1": 1.4, "3": -0.1},
        "status": {"0": "p", "3": "b"},
    }


def gdp_payload() -> dict[str, Any]:
    """Fill all non-geographic dimensions needed by the GDP request."""
    payload = make_payload()
    payload["id"] = ["freq", "unit", "s_adj", "na_item", "geo", "time"]
    payload["size"] = [1, 1, 1, 1, 2, 2]
    payload["dimension"]["s_adj"] = {"category": {"index": {"SCA": 0}}}
    payload["dimension"]["na_item"] = {"category": {"index": {"B1GQ": 0}}}
    return payload


def test_decoding_preserves_gaps_and_flags() -> None:
    spec = next(spec for spec in SERIES if spec.name == "gdp")
    observations = decode_series(gdp_payload(), spec, countries=("NL", "PT"))
    assert observations[("NL", "2025-Q1")] == Observation(1.2, "p")
    assert observations[("PT", "2025-Q1")] == Observation(None, "")
    assert observations[("PT", "2025-Q2")] == Observation(-0.1, "b")


def test_rejects_unfiltered_dimensions() -> None:
    spec = next(spec for spec in SERIES if spec.name == "gdp")
    with pytest.raises(EurostatDataError, match="changed its dimension set"):
        decode_series(make_payload(), spec, countries=("NL",))


def test_rejects_missing_countries() -> None:
    spec = next(spec for spec in SERIES if spec.name == "gdp")
    with pytest.raises(EurostatDataError, match="Countries not in"):
        decode_series(gdp_payload(), spec, countries=("NL", "LU"))


def test_quarters_and_invalid_ranges() -> None:
    assert quarter_sequence("2025-Q4", "2026-Q2") == [
        "2025-Q4", "2026-Q1", "2026-Q2",
    ]
    with pytest.raises(ValueError, match="End quarter"):
        quarter_sequence("2026-Q1", "2025-Q4")
    with pytest.raises(ValueError, match="format"):
        quarter_sequence("2026-Q5", "2026-Q6")


def test_panel_preserves_country_quarter_grid() -> None:
    sources = {spec.name: {} for spec in SERIES}
    sources["gdp"][("NL", "2026-Q1")] = Observation(0.5, "")
    rows = build_panel(sources, "2026-Q1", "2026-Q2", countries=("NL",))
    assert len(rows) == 2
    assert rows[0]["gdp_growth_qoq_pct"] == 0.5
    assert rows[0]["unemployment_rate_pct"] is None
    assert rows[1]["gdp_growth_qoq_pct"] is None


def test_request_url_contains_explicit_filters() -> None:
    for spec in SERIES:
        url = request_url(spec, "2008-Q1", "2026-Q2")
        assert url.startswith("https://ec.europa.eu/eurostat/api/")
        assert "sinceTimePeriod=2008-Q1" in url
        assert "untilTimePeriod=2026-Q2" in url
        for key, value in spec.filters.items():
            assert f"{key}={value}" in url
