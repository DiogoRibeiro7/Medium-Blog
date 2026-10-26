import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE.parent / "mechanistic-vs-data-driven-generators"
                       / "src"))
