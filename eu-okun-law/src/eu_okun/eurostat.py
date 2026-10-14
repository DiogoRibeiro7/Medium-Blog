"""Public Eurostat acquisition/validation interface."""

from eu_okun.acquisition import build_panel, fetch_dataset, save_snapshot
from eu_okun.contract import EU27, SERIES, EurostatDataError, Observation, SeriesSpec
from eu_okun.contract import quarter_sequence, request_url
from eu_okun.jsonstat import decode_series

__all__ = [
    "EU27", "SERIES", "EurostatDataError", "Observation", "SeriesSpec",
    "build_panel", "decode_series", "fetch_dataset", "quarter_sequence",
    "request_url", "save_snapshot",
]
