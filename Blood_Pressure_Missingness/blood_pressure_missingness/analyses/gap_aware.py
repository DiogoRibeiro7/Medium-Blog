"""Canonical package façade for gap-aware trend decomposition."""

from blood_pressure_missingness._compat import reexport_legacy

_legacy = reexport_legacy("gap_aware_trend_decomposition", globals())

if __name__ == "__main__":
    _legacy.main()
