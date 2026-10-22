"""Test-suite conventions shared by every module."""

from __future__ import annotations

import re

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """A test that regenerates or reproduces results must carry the reproduction marker,
    so that the fast unit-test job never runs one by accident."""
    # The function name, not the parametrised id (which can contain file names).
    unmarked = [item.nodeid for item in items
                if re.search(r"regenerat|reproduc", getattr(item, "originalname", item.name))
                and item.get_closest_marker("reproduction") is None]
    if unmarked:
        raise pytest.UsageError("Mark these tests @pytest.mark.reproduction: "
                                + ", ".join(unmarked))
