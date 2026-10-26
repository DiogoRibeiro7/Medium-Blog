"""Compare every number in a fresh results/*.txt run with the committed copy.

Usage: python check_reproduction.py <dir with committed txt files>
Numbers must match to 0.002 absolute or 0.5% relative (thread counts can move
the last printed digit of the gradient-boosting and simulation results).
"""

import re
import sys
from pathlib import Path

NUM = re.compile(r"-?\d+\.?\d*(?:e-?\d+)?")
old_dir, new_dir = Path(sys.argv[1]), Path(__file__).parent / "results"
bad = 0
for old in sorted(old_dir.glob("*.txt")):
    new = new_dir / old.name
    a, b = NUM.findall(old.read_text()), NUM.findall(new.read_text())
    if len(a) != len(b):
        print(f"{old.name}: {len(a)} numbers committed, {len(b)} reproduced")
        bad += 1
        continue
    for x, y in zip(map(float, a), map(float, b)):
        if abs(x - y) > max(0.002, 0.005 * abs(x)):
            print(f"{old.name}: committed {x}, reproduced {y}")
            bad += 1
print("reproduction check:", "FAILED" if bad else "OK")
sys.exit(1 if bad else 0)
