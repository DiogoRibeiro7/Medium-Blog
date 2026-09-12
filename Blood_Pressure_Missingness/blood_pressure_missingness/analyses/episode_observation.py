"""Canonical package façade for episode observation sensitivity."""

from blood_pressure_missingness._compat import reexport_legacy

_legacy = reexport_legacy("episode_observation_sensitivity", globals())

if __name__ == "__main__":
    _legacy.main()
