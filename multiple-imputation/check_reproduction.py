"""Compare every number in a fresh results/*.txt run with the committed copy.

Usage: python check_reproduction.py <dir with committed txt files>
Most results must match to 0.002 absolute or 0.5% relative. For the CPS
predictive-mean-matching Monte Carlo summaries, the rounded relative bias,
SE/SD ratio, and empirical coverage have metric-specific tolerances to absorb
small cross-run differences in donor matching. Coefficients, standard errors,
and RMSE values retain the strict baseline tolerances.
On GitHub Actions the mismatches are also reported as one annotation, so they
can be read from the checks page without downloading the log.
"""

import os
import re
import sys
from pathlib import Path

NUM = re.compile(r"-?\d+\.?\d*(?:e-?\d+)?")

# Token positions in CPS PMM summary rows: relative bias (% points), SE/SD,
# and empirical coverage. Other quantities must still match tightly.
PMM_SUMMARY_TOLERANCES: dict[int, float] = {2: 0.15, 5: 0.025, 6: 0.02}


def tolerance(filename: str, row: str, position: int, reference: float) -> float:
    """Return an absolute tolerance for one printed summary statistic.

    PMM donor selection can change at nearly tied predictions across numerical
    backends. Only the CPS Monte Carlo summary's derived ratios and coverage
    receive wider bounds; the scientific point estimates remain strict.
    """
    baseline = max(0.002, 0.005 * abs(reference))
    is_cps_pmm = filename.startswith("01_cps_mechanisms_") and row.lstrip().startswith("MI, PMM")
    if is_cps_pmm:
        return max(baseline, PMM_SUMMARY_TOLERANCES.get(position, 0.0))
    return baseline

old_dir, new_dir = Path(sys.argv[1]), Path(__file__).parent / "results"
issues = []
for old in sorted(old_dir.glob("*.txt")):
    new = new_dir / old.name
    if not new.exists():
        issues.append(f"{old.name}: not reproduced")
        continue
    a_lines, b_lines = old.read_text().splitlines(), new.read_text().splitlines()
    if len(a_lines) != len(b_lines):
        issues.append(f"{old.name}: {len(a_lines)} lines committed, {len(b_lines)} reproduced")
        continue
    for i, (la, lb) in enumerate(zip(a_lines, b_lines), 1):
        a, b = NUM.findall(la), NUM.findall(lb)
        if len(a) != len(b):
            issues.append(f"{old.name}:{i}: {len(a)} numbers committed, {len(b)} reproduced")
            continue
        for position, (x, y) in enumerate(zip(map(float, a), map(float, b))):
            if abs(x - y) > tolerance(old.name, la, position, x):
                issues.append(f"{old.name}:{i}: committed {x:g}, reproduced {y:g} | {la.strip()[:70]}")
for line in issues:
    print(line)
print("reproduction check:", f"FAILED ({len(issues)} mismatches)" if issues else "OK")
if issues and os.environ.get("GITHUB_ACTIONS"):
    shown = issues[:80] + ([f"... and {len(issues) - 80} more"] if len(issues) > 80 else [])
    body = "%0A".join(s.replace("%", "%25").replace("\r", "").replace("\n", " ") for s in shown)
    print(f"::error title=Reproduction mismatches ({len(issues)})::{body}")
sys.exit(1 if issues else 0)
