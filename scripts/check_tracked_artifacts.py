"""Reject disposable generated files accidentally committed to the monorepo.

Only unambiguously transient files are blocked. Historical datasets, notebook
outputs, and research results remain outside the scope of this check.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import PurePosixPath

DISPOSABLE_DIRECTORIES: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".ipynb_checkpoints",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".Rproj.user",
    }
)
DISPOSABLE_FILENAMES: frozenset[str] = frozenset(
    {".DS_Store", "Thumbs.db", ".Rhistory", ".coverage"}
)
DISPOSABLE_SUFFIXES: tuple[str, ...] = (".pyc", ".pyo", ".swp")
KNOWN_GENERATED_EXPORTS: frozenset[str] = frozenset(
    {
        "just_fun/temp.csv",
        "just_fun/AlcoholCPI.tiff",
        "just_fun/DisposableIncomes.tiff",
    }
)


def is_disposable(path: str) -> bool:
    """Return whether a repository-relative Git path is clearly transient.

    Args:
        path: A tracked file path, using Git's slash-separated convention.

    Raises:
        TypeError: When ``path`` is not a string.
    """
    if not isinstance(path, str):
        raise TypeError("path must be a string")

    normalized = path.replace("\\", "/")
    components = PurePosixPath(normalized).parts
    if not components:
        return False

    filename = components[-1]
    return (
        normalized in KNOWN_GENERATED_EXPORTS
        or any(part in DISPOSABLE_DIRECTORIES for part in components[:-1])
        or filename in DISPOSABLE_FILENAMES
        or filename.startswith(".coverage.")
        or filename.endswith(DISPOSABLE_SUFFIXES)
    )


def find_disposable_paths(paths: Iterable[str]) -> list[str]:
    """Return a sorted, deduplicated list of forbidden tracked file paths."""
    return sorted({path for path in paths if is_disposable(path)})


def git_tracked_paths() -> list[str]:
    """List tracked Git files without ambiguities from spaces or newlines."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Unable to inspect Git index: {detail}")

    # Git's NUL delimiter preserves unusual historical filenames exactly.
    return [os.fsdecode(raw) for raw in result.stdout.split(b"\x00") if raw]


def main() -> int:
    """Check the index and return a CI-friendly process exit code."""
    try:
        paths = git_tracked_paths()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    violations = find_disposable_paths(paths)
    if violations:
        print("Disposable generated files are tracked:", file=sys.stderr)
        for path in violations:
            print(f"  - {path}", file=sys.stderr)
        print("Remove the generated files from Git before merging.", file=sys.stderr)
        return 1

    print(f"Checked {len(paths)} tracked files; no disposable artifacts found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
