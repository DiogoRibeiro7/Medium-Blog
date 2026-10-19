"""Reproduce the causal-inference series and compare archived CSV results.

Run:
    python causal_inference_statistics/audit_reproducibility.py --parts 6 7
    python causal_inference_statistics/audit_reproducibility.py --all

Runs each experiment in a disposable directory, with its script copied there.
No checked-in results are modified. Failures are reported and cause exit code 1.
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT: Path = Path(__file__).resolve().parent
DEFAULT_TOLERANCE: float = 1e-8


def compare_csv(reference: Path, generated: Path, tolerance: float) -> list[str]:
    """Compare CSV schema and numeric/text fields, allowing floating-point tolerance."""
    with reference.open(newline="", encoding="utf-8") as file:
        left = csv.DictReader(file)
        left_fields = left.fieldnames
        left_rows = list(left)
    with generated.open(newline="", encoding="utf-8") as file:
        right = csv.DictReader(file)
        right_fields = right.fieldnames
        right_rows = list(right)
    errors: list[str] = []
    if left_fields != right_fields:
        return [f"Headers differ: {left_fields!r} vs {right_fields!r}"]
    if len(left_rows) != len(right_rows):
        return [f"Row counts differ: {len(left_rows)} vs {len(right_rows)}"]
    for row_no, (original, rerun) in enumerate(zip(left_rows, right_rows, strict=True), start=2):
        for field in left_fields or ():
            a, b = original[field].strip(), rerun[field].strip()
            if a == b:
                continue
            try:
                av, bv = float(a), float(b)
            except ValueError:
                errors.append(f"row {row_no} {field}: {a!r} != {b!r}")
                continue
            if not (math.isfinite(av) and math.isfinite(bv)):
                if a != b:
                    errors.append(f"row {row_no} {field}: nonfinite {a!r} != {b!r}")
            elif not math.isclose(av, bv, abs_tol=tolerance, rel_tol=tolerance):
                errors.append(f"row {row_no} {field}: {a!r} != {b!r}")
            if len(errors) >= 20:
                return errors
    return errors


def audit_part(part: int, *, tolerance: float, timeout: int) -> list[str]:
    """Run one series experiment in isolation and validate its archived output."""
    folder = ROOT / f"part{part}"
    script = folder / "experiment.py"
    reference = folder / "results.csv"
    if not script.is_file():
        return ["experiment.py is missing"]
    if not reference.is_file():
        return ["results.csv missing: output cannot be compared"]
    with tempfile.TemporaryDirectory(prefix=f"causal-part-{part}-") as tmp:
        isolated = Path(tmp)
        shutil.copy2(script, isolated / "experiment.py")
        try:
            completed = subprocess.run(
                [sys.executable, str(isolated / "experiment.py")],
                cwd=isolated,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return [f"execution exceeded {timeout} seconds"]
        if completed.returncode:
            return [f"execution failed ({completed.returncode}): {completed.stderr[-1000:]}"]
        generated = isolated / "results.csv"
        if not generated.is_file():
            return ["experiment ran but produced no results.csv"]
        return compare_csv(reference, generated, tolerance)


def main() -> int:
    """Audit selected parts and return nonzero if any check fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_mutually_exclusive_group(required=True)
    choices.add_argument("--parts", type=int, nargs="+", metavar="N")
    choices.add_argument("--all", action="store_true")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--timeout", type=int, default=600, help="Maximum seconds per experiment")
    args = parser.parse_args()
    parts: list[int] = list(range(1, 21)) if args.all else args.parts
    if any(not 1 <= n <= 20 for n in parts):
        parser.error("Each part must be between 1 and 20")
    if not 0 <= args.tolerance < 1 or args.timeout <= 0:
        parser.error("Tolerance must be in [0,1); timeout must be positive")
    failures = 0
    for part in parts:
        issues = audit_part(part, tolerance=args.tolerance, timeout=args.timeout)
        if issues:
            failures += 1
            print(f"FAIL part {part}: {'; '.join(issues)}")
        else:
            print(f"PASS part {part}: regenerated CSV matches")
    print(f"Audited {len(parts)} part(s): {len(parts) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
