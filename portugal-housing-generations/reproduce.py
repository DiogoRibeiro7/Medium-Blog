"""Reproduce every result of the release from the pinned data, and say whether anything changed.

Steps (standard library only; matplotlib only for step 4):

1. check every file in ``data/`` against ``data/SHA256SUMS``;
2. run the unit tests;
3. re-run the five analysis scripts and the data inventory, and check that each
   output file is byte-identical to the one shipped with the release;
4. redraw the figures if matplotlib is installed, decode any figure whose bytes
   changed, and warn if its pixels changed too. A figure that differs only in
   PNG encoding (a different compression level or zlib build) is counted but not
   flagged; a pixel difference without any change in the data comes from a
   different matplotlib, FreeType or font build. Neither fails the run.

Works from a plain copy of the folder (no git needed).

Run:  python reproduce.py                    # exits 0 if everything reproduces
      python reproduce.py --write-manifest   # maintainers: rewrite data/SHA256SUMS
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
MANIFEST = DATA / "SHA256SUMS"
ANALYSES = {
    "analysis.py": ["results.json"],
    "regional.py": ["results_regional.json", "regional_panel.csv"],
    "affordability.py": ["results_affordability.json", "affordability_panel.csv"],
    "rental.py": ["results_rental.json", "rental_panel.csv"],
    "integrated.py": ["results_integrated.json", "integrated_panel.csv", "data_audit.csv"],
    "data_sources.py": ["DATA_SOURCES.csv"],
}
FIGURE_SCRIPTS = ["regional_figures.py", "affordability_figures.py", "rental_figures.py", "integrated_figures.py",
                  "report_figures.py", "article_figures.py"]


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_files() -> list[Path]:
    """Every pinned data file (the manifest itself excluded), in a stable order."""
    return sorted(p for p in DATA.iterdir() if p.is_file() and p.name != MANIFEST.name)


def read_manifest(path: Path = MANIFEST) -> dict[str, str]:
    """file name -> SHA-256 from a ``sha256sum``-format manifest."""
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        entries[name.lstrip("*")] = digest
    return entries


def check_manifest() -> list[str]:
    """Problems found comparing data/ with its manifest (empty list = all good)."""
    expected = read_manifest()
    present = {p.name: p for p in data_files()}
    problems = [f"missing from data/: {n}" for n in sorted(set(expected) - set(present))]
    problems += [f"not in the manifest: {n}" for n in sorted(set(present) - set(expected))]
    problems += [f"checksum differs: {n}" for n in sorted(set(expected) & set(present)) if sha256(present[n]) != expected[n]]
    return problems


def write_manifest() -> None:
    """Rewrite data/SHA256SUMS (sha256sum format, so `sha256sum -c` also works)."""
    MANIFEST.write_text("".join(f"{sha256(p)}  {p.name}\n" for p in data_files()), encoding="utf-8")


ENCODING_ONLY = "identical pixels (encoding only)"


def pixel_difference(old_png: bytes, new_png: bytes) -> str:
    """Describe how two PNGs differ once decoded: identical pixels, a size change, or the share of changed pixels."""
    import io

    import matplotlib.image as mpimg
    import numpy as np

    old, new = mpimg.imread(io.BytesIO(old_png)), mpimg.imread(io.BytesIO(new_png))
    if old.shape != new.shape:
        return f"size {old.shape[1]}x{old.shape[0]} -> {new.shape[1]}x{new.shape[0]}"
    changed = np.any(np.abs(old - new) > 1e-6, axis=-1) if old.ndim == 3 else np.abs(old - new) > 1e-6
    share = float(changed.mean())
    return ENCODING_ONLY if share == 0 else f"{100 * share:.3f}% of pixels differ"


def run(cmd: list[str], quiet: bool = False) -> int:
    """Run a command in this folder; with ``quiet``, show its output only if it fails."""
    print("$", " ".join(cmd), flush=True)
    if not quiet:
        return subprocess.run(cmd, cwd=HERE, check=False).returncode
    done = subprocess.run(cmd, cwd=HERE, check=False, capture_output=True, text=True)
    if done.returncode != 0:
        print(done.stdout[-4000:], done.stderr[-4000:], sep="\n", file=sys.stderr)
    return done.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write-manifest", action="store_true", help="rewrite data/SHA256SUMS and stop")
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest()
        print(f"wrote {MANIFEST.relative_to(HERE)} ({len(data_files())} files)")
        return 0

    failures = 0
    problems = check_manifest()
    print("1. data manifest:", "OK" if not problems else "; ".join(problems))
    failures += bool(problems)

    code = run([sys.executable, "-m", "unittest", "discover", "-p", "test_*.py"], quiet=True)
    print("2. unit tests:", "OK" if code == 0 else "FAILED")
    failures += code != 0

    changed = []
    for script, outputs in ANALYSES.items():
        before = {o: sha256(HERE / o) for o in outputs if (HERE / o).exists()}
        if run([sys.executable, script], quiet=True) != 0:
            changed.append(f"{script} failed")
            continue
        changed += [o for o in outputs if before.get(o) != sha256(HERE / o)]
    print("3. analysis outputs:", "all byte-identical" if not changed else "CHANGED: " + ", ".join(changed))
    failures += bool(changed)

    if importlib.util.find_spec("matplotlib") is None:
        print("4. figures: skipped (matplotlib not installed; pip install matplotlib==3.11.2)")
    else:
        figures = sorted((HERE / "figures").glob("*.png"))
        before_png = {f.name: f.read_bytes() for f in figures}
        for script in FIGURE_SCRIPTS:
            if run([sys.executable, script], quiet=True) != 0:
                failures += 1
        after = sorted((HERE / "figures").glob("*.png"))
        diff = {f.name: pixel_difference(before_png[f.name], f.read_bytes()) if f.name in before_png else "new file"
                for f in after if before_png.get(f.name) != f.read_bytes()}
        drift = {name: detail for name, detail in diff.items() if detail != ENCODING_ONLY}
        on_ci = os.environ.get("GITHUB_ACTIONS") == "true"  # also report as annotations readable through the API
        if not diff:
            summary = f"all {len(after)} figures byte-identical to the committed PNGs"
        elif not drift:
            summary = (f"all {len(after)} figures pixel-identical to the committed PNGs; {len(diff)} differ only in "
                       "PNG encoding")
        else:
            summary = f"{len(drift)} of {len(after)} figures differ in pixels (warning; check visually)"
        print(f"4. figures: {summary}")
        if on_ci and not drift:
            print(f"::notice title=Figures::{summary}")
        for name, detail in drift.items():
            print(f"   {name}: {detail}")
            if on_ci:
                print(f"::warning title=Figure drift::{name}: {detail}")
    print("REPRODUCED" if failures == 0 else "NOT REPRODUCED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
