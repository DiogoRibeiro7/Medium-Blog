"""Canonical package façade for temperature-covariate robustness analysis."""

from blood_pressure_missingness._compat import reexport_legacy

_legacy = reexport_legacy("temperature_covariate_sensitivity", globals())

if __name__ == "__main__":
    _legacy.main()
