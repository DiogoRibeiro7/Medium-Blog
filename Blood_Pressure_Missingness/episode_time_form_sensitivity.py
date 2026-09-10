"""Compatibility entry point for :mod:`blood_pressure_missingness.analyses.episode_time_form`."""

from importlib import import_module as _import_module

_impl = _import_module("blood_pressure_missingness.analyses.episode_time_form")
globals().update(
    {name: value for name, value in vars(_impl).items() if not name.startswith("__")}
)

if __name__ == "__main__":
    _impl.main()
