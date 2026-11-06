"""Prevent long-form editorial manuscripts from reappearing in the public repo.

The public repository retains research data, methods, evidence and concise
article-path notices. Full Medium prose belongs in the private editorial repo.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Final, Sequence

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MAX_EDITORIAL_BYTES: Final[int] = 3_000
EDITORIAL_FILENAMES: Final[frozenset[str]] = frozenset(
    {"article.md", "article.template.md", "medium_ready.md"}
)


def is_editorial_path(path: str) -> bool:
    """Return whether a tracked path is a conventional editorial manuscript."""
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    normalized = path.replace("\\", "/")
    return Path(normalized).name.casefold() in EDITORIAL_FILENAMES


def find_oversized_editorial_files(
    paths: Sequence[str],
    root: Path,
    max_bytes: int = MAX_EDITORIAL_BYTES,
) -> list[str]:
    """Report tracked manuscript paths exceeding the public-notice limit."""
    if not isinstance(root, Path):
        raise TypeError("root must be a pathlib.Path")
    if not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError("max_bytes must be a nonnegative integer")
    offenders: list[str] = []
    for path in paths:
        if not is_editorial_path(path):
            continue
        target = root / path
        if target.is_file() and target.stat().st_size > max_bytes:
            offenders.append(path)
    return sorted(set(offenders))


def tracked_paths(root: Path) -> list[str]:
    """Collect Git-tracked paths without reading ignored or untracked files."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    return [name.decode("utf-8") for name in result.stdout.split(b"\0") if name]


def main(argv: Sequence[str] | None = None) -> int:
    """Check public manuscript size, returning a nonzero exit on violation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    arguments = parser.parse_args(argv)
    failures = find_oversized_editorial_files(tracked_paths(arguments.root), arguments.root)
    if failures:
        for path in failures:
            print(f"Oversized editorial manuscript: {path}")
        print("Move full prose to private research-articles-medium; retain a public notice.")
        return 1
    print("Public editorial manuscript guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
