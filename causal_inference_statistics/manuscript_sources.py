"""Read causal-series manuscripts without mistaking public notices for articles.

After editorial migration, point ``--manuscript-root`` at a private checkout's
``series/causal-inference-in-statistics`` directory (``part-01-*`` to
``part-20-*``). The pre-November-2026 layout (``part1`` to ``part20``) is still
accepted. Never load network credentials here.
"""

from __future__ import annotations

from pathlib import Path


NOTICE_MARKERS: tuple[str, ...] = (
    "The full editorial manuscript is now maintained",
    "This short notice preserves the historical",
)


def read_manuscript(root: Path, part: int) -> str:
    """Read a complete manuscript, rejecting missing and migrated notice paths.

    Args:
        root: Directory containing ``part-01-*`` through ``part-20-*`` folders
            (or the older ``part1`` through ``part20``).
        part: One-based article part number.

    Raises:
        ValueError: If the part is outside 1–20 or content is only a notice.
        FileNotFoundError: If the manuscript is absent.
    """
    if type(part) is not int or not 1 <= part <= 20:
        raise ValueError("part must be an integer in 1..20")
    path = root / f"part{part}" / "article.md"
    if not path.exists():
        matches = sorted(root.glob(f"part-{part:02d}-*/article.md"))
        if len(matches) == 1:
            path = matches[0]
    content = path.read_text(encoding="utf-8")
    if any(marker in content for marker in NOTICE_MARKERS):
        raise ValueError(
            f"{path} is a public compatibility notice, not an article. "
            "Pass --manuscript-root pointing to the private manuscript checkout."
        )
    if "## References" not in content or len(content.strip()) < 1_000:
        raise ValueError(f"{path} is not a complete causal-inference manuscript")
    return content
