"""Canonical package façade for public-narrative validation."""

from blood_pressure_missingness._compat import reexport_legacy

_legacy = reexport_legacy("validate_current_narrative", globals())

if __name__ == "__main__":
    _legacy.main()
