"""Reproduce every result of the release from the pinned data, and say whether anything changed.

Steps (standard library only; matplotlib only for step 4):

1. check every file in ``data/`` against ``data/SHA256SUMS``;
2. run the unit tests;
3. re-run the five analysis scripts and the data inventory, and check that each
   output file is byte-identical to the one shipped with the release;
4. redraw the figures if matplotlib is installed and report any figure whose
   bytes changed, saying whether its decoded pixels changed too (a warning only:
   a different matplotlib, FreeType or zlib build can change PNG bytes, and a
   different matplotlib or FreeType can also move pixels, without any change in
   the data).

Works from a plain copy of the folder (no git needed).

Run:  python reproduce.py                    # exits 0 if everything reproduces
      python reproduce.py --write-manifest   # maintainers: rewrite data/SHA256SUMS
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
    return "identical pixels (encoding only)" if share == 0 else f"{100 * share:.3f}% of pixels differ"


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
        diff = [f for f in sorted((HERE / "figures").glob("*.png")) if before_png.get(f.name) != f.read_bytes()]
        if not diff:
            print("4. figures: all byte-identical")
        else:
            print(f"4. figures: {len(diff)} differ in bytes (warning; check visually):")
            for f in diff:
                detail = pixel_difference(before_png[f.name], f.read_bytes()) if f.name in before_png else "new file"
                print(f"   {f.name}: {detail}")
    print("REPRODUCED" if failures == 0 else "NOT REPRODUCED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
