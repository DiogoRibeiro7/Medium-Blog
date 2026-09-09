"""Canonical package façade for day-influence sensitivity analysis."""

from blood_pressure_missingness._compat import reexport_legacy

_legacy = reexport_legacy("day_influence_sensitivity", globals())

if __name__ == "__main__":
    _legacy.main()
