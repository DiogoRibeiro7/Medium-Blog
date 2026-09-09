"""Temporary compatibility helpers used during the staged package refactor."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any


def reexport_legacy(module_name: str, namespace: dict[str, Any]) -> ModuleType:
    """Re-export one legacy top-level module into a canonical package module.

    Private helper names are intentionally preserved because the current regression
    tests exercise a few of them directly. Stage 2 will replace this bridge with
    package-native implementations and explicit public interfaces.
    """

    module = import_module(module_name)
    namespace.update(
        name_value
        for name_value in vars(module).items()
        if name_value[0] not in {"__name__", "__package__", "__loader__", "__spec__"}
    )
    return module
