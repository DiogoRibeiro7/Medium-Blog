"""Canonical package façade for temporal-dependence diagnostics."""

from blood_pressure_missingness._compat import reexport_legacy

_legacy = reexport_legacy("temporal_dependence_diagnostics", globals())

if __name__ == "__main__":
    _legacy.main()
