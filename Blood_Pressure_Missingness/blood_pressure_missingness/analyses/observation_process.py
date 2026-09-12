"""Canonical package façade for observation-process sensitivity."""

from blood_pressure_missingness._compat import reexport_legacy

_legacy = reexport_legacy("observation_process_sensitivity", globals())

if __name__ == "__main__":
    _legacy.main()
