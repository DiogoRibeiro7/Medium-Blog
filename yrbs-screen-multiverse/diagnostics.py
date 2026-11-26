"""Aggregate checks of the downloaded files: the layout found for each item and its answer counts.

    python diagnostics.py > diagnostics.txt

Prints counts only, no student-level rows. Used to confirm that each question number in
yrbs_core.ITEMS reads the intended question (answer options and their frequencies).
"""

import numpy as np

import yrbs_core as Y


def main():
    for wave, (dat, sas) in Y.FILES.items():
        layout = Y.parse_sas_input((Y.DATA / sas.split("/")[1]).read_text(encoding="latin-1"))
        names = sorted(set(Y.ITEMS[wave].values()))
        raw = Y.read_fixed(Y.DATA / dat.split("/")[1], layout, names)
        n = len(raw[names[0]])
        print(f"## {wave}: {n} rows, {len(layout)} variables in the SAS layout")
        for item, var in sorted(Y.ITEMS[wave].items()):
            a, b = layout[var]
            v = raw[var]
            if item in ("weight", "stratum", "psu"):
                fin = v[np.isfinite(v)]
                extra = f"{len(np.unique(fin))} distinct" if item != "weight" else f"sum {fin.sum():.1f}"
                print(f"  {item:9s} {var:8s} cols {a}-{b}: missing {int(np.isnan(v).sum())}, {extra}")
                continue
            vals, cnt = np.unique(v[np.isfinite(v)], return_counts=True)
            dist = ", ".join(f"{int(x) if float(x).is_integer() else x}:{c}" for x, c in zip(vals, cnt, strict=True))
            print(f"  {item:9s} {var:8s} cols {a}-{b}: missing {int(np.isnan(v).sum())}; {dist}")
        print()


if __name__ == "__main__":
    main()
