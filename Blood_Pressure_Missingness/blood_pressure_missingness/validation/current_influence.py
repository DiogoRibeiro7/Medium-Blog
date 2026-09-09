"""Canonical package façade for influence-findings validation."""

from blood_pressure_missingness._compat import reexport_legacy

_legacy = reexport_legacy("validate_current_influence_findings", globals())

if __name__ == "__main__":
    _legacy.main()
