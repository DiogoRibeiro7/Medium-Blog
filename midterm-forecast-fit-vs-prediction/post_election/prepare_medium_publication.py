"""Prepare a Medium-compatible provisional sequel, preserving the scientific draft.

No election outcome is scored here. This mechanically converts eight relative
figure links to immutable raw PNG URLs and relative companion links to pinned
GitHub references. The article already uses prose/Unicode in place of tables
and LaTeX, so no new mathematical rasterization is required.

Usage::

    python prepare_medium_publication.py --write
    python prepare_medium_publication.py --check

The first command is an idempotent bulk publication formatting operation.
The second is read-only and intended for GitHub Actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

ROOT: Final[Path] = Path(__file__).resolve().parent
REPO_ROOT: Final[Path] = ROOT.parents[1]
SOURCE: Final[Path] = ROOT / "article.md"
PUBLICATION: Final[Path] = ROOT / "publication"
OUTPUT: Final[Path] = PUBLICATION / "medium_ready.md"
MANIFEST: Final[Path] = PUBLICATION / "figure_manifest.json"
REPOSITORY: Final[str] = "DiogoRibeiro7/Medium-Blog"
ARTICLE_DIRECTORY: Final[str] = "midterm-forecast-fit-vs-prediction/post_election"
FIGURES: Final[re.Pattern[str]] = re.compile(r"!\[([^\]]+)\]\((figures/[^()\s]+\.png)\)")
MARKDOWN_LINKS: Final[re.Pattern[str]] = re.compile(r"(?<!!)\[([^\]]+)\]\(([^()\s]+)\)")
TABLE_LINES: Final[re.Pattern[str]] = re.compile(r"(?m)^\s*\|[^\n]+\|\s*$")
DISPLAY_MATH: Final[re.Pattern[str]] = re.compile(r"(?m)^\\\[|^\$\$")
INLINE_MATH: Final[re.Pattern[str]] = re.compile(r"\\\(|\\\)")
SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_SHA: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True, slots=True)
class FigureManifest:
    """Immutable image-location and source-byte integrity declaration."""

    commit: str
    sha256: dict[str, str]


def load_figure_manifest(path: Path = MANIFEST) -> FigureManifest:
    """Load the pinned reference, refusing changes to this source class."""

    payload: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("manifest_version") != 1:
        raise ValueError("Unexpected publication figure manifest version")
    if payload.get("status") != "draft_provisional_not_final_result":
        raise ValueError("This publication package must remain provisional")
    commit: object = payload.get("immutable_figure_commit")
    checksums: object = payload.get("sha256")
    if not isinstance(commit, str) or not COMMIT_SHA.fullmatch(commit):
        raise ValueError("Figure source must be pinned to a full Git commit SHA")
    if not isinstance(checksums, dict) or len(checksums) != 8:
        raise ValueError("Exactly eight named figures must be registered")
    if not all(
        isinstance(name, str) and name.startswith("fig") and name.endswith(".png")
        and isinstance(digest, str) and SHA256.fullmatch(digest)
        for name, digest in checksums.items()
    ):
        raise ValueError("Malformed figure names or SHA-256 digests")
    return FigureManifest(commit=commit, sha256=checksums)


def validate_manuscript(source: str) -> None:
    """Reject unsupported Medium syntax or removal of the provisional status."""

    if not isinstance(source, str):
        raise TypeError("Manuscript must be UTF-8 text")
    if not source.startswith("# The Votes Are In: Testing an 89% Midterm Model"):
        raise ValueError("Unexpected sequel manuscript title")
    if "Research draft, 4 November 2026" not in source:
        raise ValueError("The provisional publication status was removed")
    if "68 House races remained uncalled" not in source:
        raise ValueError("The live-results uncertainty disclaimer was removed")
    if "48 uncalled races" not in source or "5 November" not in source:
        raise ValueError("The 5 November dated update was removed")
    if "31 October" not in source or "1 November" not in source:
        raise ValueError("The distinct information and registration dates must remain visible")
    if len(source.split()) < 8000:
        raise ValueError("The long-form research manuscript is incomplete")
    if TABLE_LINES.search(source):
        raise ValueError("Medium does not reliably import Markdown pipe tables")
    if DISPLAY_MATH.search(source) or INLINE_MATH.search(source):
        raise ValueError("Use figure images or Unicode rather than raw LaTeX for Medium")


def resolve_companion(target: str) -> tuple[str, Path]:
    """Resolve a relative manuscript link within the repository boundary."""

    if not isinstance(target, str) or not target or target.startswith("/"):
        raise ValueError("Companion links must be nonempty relative paths")
    if "?" in target or "#" in target:
        raise ValueError("Companion link query/fragment would need explicit review")
    resolved = (ROOT / target).resolve(strict=True)
    try:
        relative = resolved.relative_to(REPO_ROOT.resolve(strict=True))
    except ValueError as error:
        raise ValueError("Companion path escapes the study repository") from error
    kind: str = "tree" if resolved.is_dir() else "blob"
    if not resolved.is_dir() and not resolved.is_file():
        raise ValueError("Companion target is not a regular file or directory")
    return kind, relative


def build_medium_copy(source: str, figure_manifest: FigureManifest) -> str:
    """Convert only display links; do not alter scientific claims or forecasts."""

    validate_manuscript(source)
    discovered = FIGURES.findall(source)
    expected_figures: set[str] = set(figure_manifest.sha256)
    if len(discovered) != 8 or {Path(path).name for _, path in discovered} != expected_figures:
        raise ValueError("Manuscript figure set does not match the pinned manifest")
    if len({path for _, path in discovered}) != 8:
        raise ValueError("The same figure appears more than once")

    def embed_png(match: re.Match[str]) -> str:
        alt: str = match.group(1)
        relative: str = match.group(2)
        png: Path = ROOT / relative
        if not png.is_file():
            raise FileNotFoundError(f"Missing original evidence image: {relative}")
        digest = hashlib.sha256(png.read_bytes()).hexdigest()
        if digest != figure_manifest.sha256[png.name]:
            raise ValueError(f"Image changed since immutable pinned revision: {png.name}")
        url = (
            f"https://raw.githubusercontent.com/{REPOSITORY}/"
            f"{figure_manifest.commit}/{ARTICLE_DIRECTORY}/{relative}"
        )
        return f"![{alt}]({url})"

    converted = FIGURES.sub(embed_png, source)

    def link_companion(match: re.Match[str]) -> str:
        label: str = match.group(1)
        target: str = match.group(2)
        if target.startswith(("https://", "mailto:", "#")):
            return match.group(0)
        if "://" in target or target.startswith("http:"):
            raise ValueError("Unsupported external link scheme")
        kind, relative = resolve_companion(target)
        url = (
            f"https://github.com/{REPOSITORY}/{kind}/"
            f"{figure_manifest.commit}/{relative.as_posix()}"
        )
        return f"[{label}]({url})"

    converted = MARKDOWN_LINKS.sub(link_companion, converted)
    if len(re.findall(r"https://raw\.githubusercontent\.com/", converted)) != 8:
        raise ValueError("Exactly eight immutable source PNG URLs are required")
    if FIGURES.search(converted) or TABLE_LINES.search(converted):
        raise ValueError("Unsupported relative images or tables remain")
    # Maintain every word and heading; only URLs differ between source/copy.
    return converted.rstrip() + "\n"


def main() -> None:
    """Build or check a deterministic publication derivative from the draft."""

    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="Regenerate the Medium Markdown derivative")
    action.add_argument("--check", action="store_true", help="Verify committed derivative without modifying files")
    args = parser.parse_args()
    manifest = load_figure_manifest()
    source = SOURCE.read_text(encoding="utf-8")
    prepared = build_medium_copy(source, manifest)
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(prepared, encoding="utf-8")
    else:
        current = OUTPUT.read_text(encoding="utf-8")
        if current != prepared:
            raise ValueError("Medium copy is stale; regenerate before review")
    print(f"Provisional Medium copy: {len(prepared.split())} words, 8 pinned PNG figures, "
          f"source commit {manifest.commit[:12]}; no official outcome scored")


if __name__ == "__main__":
    main()
