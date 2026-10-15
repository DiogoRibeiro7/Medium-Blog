#!/usr/bin/env bash
# Reproduce every result and figure. Run ./fetch_data.sh first.
set -uo pipefail
cd "$(dirname "$0")/experiments"
mkdir -p ../results
for e in exp1_structure exp2_risk exp3_portugal exp4_landing; do
  echo "== $e $(date +%T)"
  python3 "$e.py" > "../results/$e.log" 2>&1 || echo "   FAILED: see results/$e.log"
done
echo "== figures $(date +%T)"
python3 figures.py > ../results/figures.log 2>&1 || echo "   FAILED: see results/figures.log"
echo "== done $(date +%T)"
