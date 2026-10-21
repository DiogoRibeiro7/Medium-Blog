"""Fingerprints of generated text files that do not depend on the platform.

A rebuild on another platform changes floating-point results in their last digits: about
1e-16 relative, and up to a few 1e-12 absolute in ill-conditioned regressions.
Numerical-error measurements, such as the classical review's verification gaps (1e-16 to
1e-12), change by more than that relative to their own size. None of this is a change of
result. A fingerprint is therefore the SHA-256 of a text file with LF line endings and with
every stand-alone decimal number rounded to ``ROUND_PLACES`` decimal places. A change of a
result by more than about 1e-8 changes the fingerprint.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROUND_PLACES = 8
TEXT_SUFFIXES = {".md", ".csv", ".json", ".tex", ".toml", ".lock", ".py", ".txt", ".bib"}
# A decimal number with a point or an exponent, not part of a word, a hash or a version
# string such as 2.5.1. A full stop after it ends a sentence unless a digit follows.
_DECIMAL = re.compile(
    r"(?<![\w.])-?(?:\d+\.\d+(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)(?!\w|\.\d)")


def round_number(x: float) -> str:
    text = format(x, f".{ROUND_PLACES}f")
    return format(0.0, f".{ROUND_PLACES}f") if float(text) == 0 else text


def round_numbers(text: str) -> str:
    """Every stand-alone decimal number in ``text`` rounded; other text unchanged."""
    return _DECIMAL.sub(lambda m: round_number(float(m.group(0))), text)


def text_bytes(path: Path) -> bytes:
    """A file's bytes, with LF line endings for text files."""
    data = path.read_bytes()
    if path.suffix in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return data


def fingerprint(path: Path) -> str:
    """SHA-256 of a text file with its numbers rounded (raw SHA-256 for other files)."""
    data = text_bytes(path)
    if path.suffix in TEXT_SUFFIXES:
        data = round_numbers(data.decode("utf-8")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
