"""Exceptions raised by the package.

Every error the package raises on purpose derives from :class:`AirlineOpsError`, so a
caller can separate modelling problems from programming errors.
"""

from __future__ import annotations


class AirlineOpsError(Exception):
    """Base class for all errors raised deliberately by the package."""


class ConfigError(AirlineOpsError):
    """A configuration file is missing, malformed or inconsistent."""


class DataError(AirlineOpsError):
    """Input data failed an integrity check (checksum, schema or content)."""


class InfeasibleModelError(AirlineOpsError):
    """An optimisation model has no feasible solution, or the solver could not find one."""

    def __init__(self, model: str, detail: str) -> None:
        super().__init__(f"{model}: {detail}")
        self.model = model
        self.detail = detail


class SolverError(AirlineOpsError):
    """The solver returned a status the code does not know how to handle."""


class PlanValidationError(AirlineOpsError):
    """A plan violates at least one hard constraint."""

    def __init__(self, plan: str, violations: list[str]) -> None:
        shown = "; ".join(violations[:5])
        more = f" (+{len(violations) - 5} more)" if len(violations) > 5 else ""
        super().__init__(f"plan {plan!r} has {len(violations)} violation(s): {shown}{more}")
        self.plan = plan
        self.violations = violations
