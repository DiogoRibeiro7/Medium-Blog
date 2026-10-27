"""Helpers shared by the regeneration tests (not a test module)."""

from __future__ import annotations

import math


def assert_same(committed: object, regenerated: object, path: str = "") -> None:
    """Equal structure and non-float values; floats equal to a relative 1e-7.

    The last digits of least-squares fits differ between processors and BLAS builds;
    every number the reports show is far coarser than this tolerance.
    """
    if isinstance(committed, dict):
        assert isinstance(regenerated, dict) and set(committed) == set(regenerated), path
        for key in committed:
            assert_same(committed[key], regenerated[key], f"{path}.{key}")
    elif isinstance(committed, list):
        assert isinstance(regenerated, list) and len(committed) == len(regenerated), path
        for i, (a, b) in enumerate(zip(committed, regenerated, strict=True)):
            assert_same(a, b, f"{path}[{i}]")
    elif isinstance(committed, float) and isinstance(regenerated, float):
        assert math.isclose(committed, regenerated, rel_tol=1e-7, abs_tol=1e-12), \
            (path, committed, regenerated)
    else:
        assert committed == regenerated, (path, committed, regenerated)
