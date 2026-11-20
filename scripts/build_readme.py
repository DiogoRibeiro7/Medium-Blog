"""Build the "Projects" part of README.md from projects.yml.

The part between <!-- projects:start --> and <!-- projects:end --> lists every top-level
folder under its section and group. A folder's line shows the first heading of its own
README (or the `title` given in projects.yml). Everything outside the markers is left alone.

    python scripts/build_readme.py             rebuild README.md
    python scripts/build_readme.py --check     fail if README.md is out of date or projects.yml is invalid
    python scripts/build_readme.py --validate  check projects.yml only; README.md is not read

A folder missing from projects.yml is a warning, not an error: it is listed under
"Not yet sorted" until someone adds its line.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
START = "<!-- projects:start -->"
END = "<!-- projects:end -->"
README_NAMES = ("README.md", "readme.md", "Readme.md")
KINDS = ("series", "themes", "tooling", "archive")

# Endings and prefixes that folder READMEs add to the article title.
_TITLE_NOISE = [
    re.compile(r"^Article\s*#?\s*\d+\s*[—:–-]\s*", re.IGNORECASE),
    re.compile(r"\s*[—–-]+\s*(public\s+)?(reproducible\s+)?(research\s+)?companion\s*$", re.IGNORECASE),
    re.compile(r"\s*:\s*reproducible\s+companion\s*$", re.IGNORECASE),
    re.compile(r"\s*[—–-]+\s*Article\s+\d+\s*$", re.IGNORECASE),
]


class ProjectsError(Exception):
    """projects.yml is malformed."""


def clean_title(title: str) -> str:
    for pattern in _TITLE_NOISE:
        title = pattern.sub("", title)
    return title.strip()


def readme_title(folder: Path) -> str | None:
    for name in README_NAMES:
        path = folder / name
        if path.is_file():
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("# "):
                    return clean_title(line[2:].strip()) or None
            return None
    return None


def top_level_folders(root: Path) -> list[str]:
    return sorted(
        (p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")),
        key=str.lower,
    )


def load_projects(path: Path) -> dict:
    import yaml  # imported here so the pure functions can be tested without PyYAML

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        keys = set()
        for key_node, _ in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in keys:
                raise ProjectsError(f"{path.name}: {key!r} is listed twice")
            keys.add(key)
        return yaml.SafeLoader.construct_mapping(loader, node, deep)

    UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    with path.open(encoding="utf-8") as handle:
        return yaml.load(handle, Loader=UniqueKeyLoader)


def normalise(data: dict, folders: list[str]) -> tuple[dict, list[str], list[str]]:
    """Return (model, errors, unsorted folders)."""
    errors: list[str] = []
    sections = data.get("sections") or []
    groups = data.get("groups") or []
    entries = data.get("folders") or {}
    section_ids = [s.get("id") for s in sections]
    group_ids = [g.get("id") for g in groups]
    for kind, ids in (("section", section_ids), ("group", group_ids)):
        seen = set()
        for i in ids:
            if not i:
                errors.append(f"a {kind} has no id")
            elif i in seen:
                errors.append(f"{kind} {i!r} is defined twice")
            seen.add(i)
    for s in sections:
        if s.get("kind") not in KINDS:
            errors.append(f"section {s.get('id')!r}: kind must be one of {', '.join(KINDS)}")
    for g in groups:
        if g.get("section") not in section_ids:
            errors.append(f"group {g.get('id')!r} names an unknown section {g.get('section')!r}")

    present = set(folders)
    listed: dict[str, dict] = {}
    for name, value in entries.items():
        entry = {"group": value} if isinstance(value, str) else dict(value or {})
        unknown_keys = set(entry) - {"group", "part", "title"}
        if unknown_keys:
            errors.append(f"{name!r}: unknown keys {sorted(unknown_keys)}")
        if entry.get("group") not in group_ids:
            errors.append(f"{name!r}: unknown group {entry.get('group')!r}")
        if "part" in entry and not isinstance(entry["part"], int):
            errors.append(f"{name!r}: part must be a whole number")
        if name not in present:
            errors.append(f"{name!r} is listed but there is no such folder")
        listed[name] = entry
    unsorted = [f for f in folders if f not in listed]
    model = {"sections": sections, "groups": groups, "folders": listed}
    return model, errors, unsorted


def link(name: str) -> str:
    return f"[`{name}/`]({quote(name)}/)"


def render(model: dict, folders: list[str], titles: dict[str, str | None], unsorted: list[str]) -> str:
    present = set(folders)
    by_group: dict[str, list[str]] = {g["id"]: [] for g in model["groups"]}
    for name, entry in model["folders"].items():
        if name in present and entry.get("group") in by_group:
            by_group[entry["group"]].append(name)

    def sort_key(name: str):
        part = model["folders"][name].get("part")
        return (part is None, part if part is not None else 0, name.lower())

    for names in by_group.values():
        names.sort(key=sort_key)

    def title_of(name: str) -> str | None:
        return model["folders"].get(name, {}).get("title") or titles.get(name)

    kind_of = {s["id"]: s.get("kind") for s in model["sections"]}
    group_kind = {g["id"]: kind_of.get(g["section"]) for g in model["groups"]}

    def count(kind: str) -> tuple[int, int]:
        groups = [g for g in model["groups"] if group_kind[g["id"]] == kind and by_group[g["id"]]]
        return sum(len(by_group[g["id"]]) for g in groups), len(groups)

    in_series, n_series = count("series")
    in_themes, n_themes = count("themes")
    tooling, _ = count("tooling")
    archived, _ = count("archive")

    lines = [
        START,
        "",
        f"{len(folders)} top-level folders: {in_series} belong to {n_series} series, "
        f"{in_themes} are standalone studies in {n_themes} themes, {tooling} hold repository tooling "
        f"and {archived} hold older material." + (f" {len(unsorted)} not yet sorted." if unsorted else ""),
        "",
    ]
    for section in model["sections"]:
        groups = [g for g in model["groups"] if g["section"] == section["id"] and by_group[g["id"]]]
        if not groups:
            continue
        lines += [f"### {section['title']}", ""]
        for group in groups:
            names = by_group[group["id"]]
            if section.get("kind") == "archive":
                if group.get("about"):
                    lines += [group["about"], ""]
                lines += [", ".join(link(n) for n in names), ""]
                continue
            if section.get("kind") in ("series", "themes"):
                lines += [f"#### {group['title']}", ""]
            if group.get("about"):
                lines += [group["about"], ""]
            for name in names:
                title = title_of(name)
                lines.append(f"- {link(name)} — {title}" if title else f"- {link(name)}")
            lines.append("")
    if unsorted:
        lines += ["### Not yet sorted", "", "Add these folders to `projects.yml`.", ""]
        for name in unsorted:
            title = titles.get(name)
            lines.append(f"- {link(name)} — {title}" if title else f"- {link(name)}")
        lines.append("")
    lines.append(END)
    return "\n".join(lines)


def splice(readme: str, block: str) -> str:
    if readme.count(START) != 1 or readme.count(END) != 1 or readme.index(START) > readme.index(END):
        raise ProjectsError(f"README.md must contain {START} once, followed by {END} once")
    head, rest = readme.split(START, 1)
    _, tail = rest.split(END, 1)
    return head + block + tail


def annotate(level: str, message: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{level} file=projects.yml::{message}")
    else:
        print(f"{level}: {message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail if README.md is out of date")
    mode.add_argument("--validate", action="store_true", help="check projects.yml only")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    root = args.root
    folders = top_level_folders(root)
    try:
        data = load_projects(root / "projects.yml")
    except ProjectsError as exc:
        annotate("error", str(exc))
        return 1
    model, errors, unsorted = normalise(data or {}, folders)
    for message in errors:
        annotate("error", message)
    for name in unsorted:
        annotate("warning", f"{name!r} is not in projects.yml; README.md lists it under 'Not yet sorted'")
    if errors:
        return 1
    if args.validate:
        print(f"projects.yml: {len(model['folders'])} folders listed, {len(unsorted)} not yet sorted.")
        return 0

    titles = {name: readme_title(root / name) for name in folders}
    readme_path = root / "README.md"
    current = readme_path.read_text(encoding="utf-8")
    try:
        updated = splice(current, render(model, folders, titles, unsorted))
    except ProjectsError as exc:
        annotate("error", str(exc))
        return 1
    if args.check:
        if updated != current:
            annotate("error", "README.md is out of date; run python scripts/build_readme.py")
            return 1
        print("README.md is up to date.")
        return 0
    if updated != current:
        readme_path.write_text(updated, encoding="utf-8")
        print("README.md rebuilt.")
    else:
        print("README.md already up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
