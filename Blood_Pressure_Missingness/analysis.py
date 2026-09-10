"""Compatibility shim for :mod:`blood_pressure_missingness.public_analysis`.

The implementation now lives in the canonical package. This module remains so
existing notebook imports, tests, CLI commands, and external links continue to
work during the staged refactor.
"""

from blood_pressure_missingness.public_analysis import *  # noqa: F401,F403
from blood_pressure_missingness.public_analysis import main as _main


if __name__ == "__main__":
    _main()
