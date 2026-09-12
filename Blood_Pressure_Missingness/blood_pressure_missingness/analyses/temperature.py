"""Canonical package façade for measurement-time temperature analysis."""

from blood_pressure_missingness._compat import reexport_legacy

_legacy = reexport_legacy("temperature_covariate", globals())

if __name__ == "__main__":
    _legacy.main()
