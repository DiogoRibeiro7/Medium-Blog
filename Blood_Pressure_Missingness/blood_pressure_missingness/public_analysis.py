"""Canonical package façade for the public analysis layer."""

from blood_pressure_missingness._compat import reexport_legacy

_legacy = reexport_legacy("analysis", globals())

if __name__ == "__main__":
    _legacy.main()
