"""Structured logging: one JSON object per line.

Each record carries the logger name, level, message and any extra fields passed with
``logger.info("...", extra={"fields": {...}})``.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Return the record as a JSON string."""
        payload: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update({str(k): v for k, v in fields.items()})
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in {"fields", "message"} and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Send JSON log lines to standard error at the given level.

    Args:
        level: A standard logging level name such as ``"INFO"`` or ``"DEBUG"``.

    Raises:
        ValueError: When ``level`` is not a known logging level.

    """
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        raise ValueError(f"unknown logging level: {level!r}")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("airline_ops")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger in the package namespace."""
    return logging.getLogger(f"airline_ops.{name}")
