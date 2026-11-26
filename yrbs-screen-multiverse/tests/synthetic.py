"""Synthetic stand-ins for the four YRBS files, with the same layout conventions, for tests and dry runs.

    python tests/synthetic.py OUTDIR    # writes the eight files fetch_data.py would download

The data are invented: a known linear-probability effect of heavy use on each outcome lets the tests
check that the pipeline recovers it. Nothing here resembles the real survey's numbers.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yrbs_core as Y  # noqa: E402

TRUE_EFFECT = 0.06  # risk difference of the top category (5+ hours or hourly use) on every outcome


def make_wave(wave, n=3000, seed=0):
    rng = np.random.default_rng(seed + wave)
    it = Y.ITEMS[wave]
    n_strata, per = 12, 4
    stratum = rng.integers(1, n_strata + 1, n)
    psu = stratum * 1000 + rng.integers(1, per + 1, n)
    female = rng.random(n) < 0.5
    cols = {it["sex"]: np.where(female, 1, 2), it["grade"]: rng.integers(1, 5, n), it["raceeth"]: rng.integers(1, 9, n)}
    cols[it["sexid"]] = np.where(rng.random(n) < 0.8, 1, rng.integers(2, 5, n))
    top = np.zeros(n, bool)
    for m in Y.MEASURES[wave]:
        if m == "social":
            code = rng.integers(1, 9, n)
            top |= code >= 7
        elif m == "screen":
            code = rng.integers(1, 7, n)
            top |= code == 6
        else:
            code = rng.integers(1, 8, n)
            top |= code == 7
        cols[it[m]] = code
    base = 0.2 + 0.1 * female
    for o in ("sad", "consider", "plan", "attempt", "poor_mh"):
        if o not in it:
            continue
        p = np.clip(base + TRUE_EFFECT * top, 0, 1)
        yv = rng.random(n) < p
        if o == "attempt":
            cols[it[o]] = np.where(yv, rng.integers(2, 6, n), 1)
        elif o == "poor_mh":
            cols[it[o]] = np.where(yv, rng.integers(4, 6, n), rng.integers(1, 4, n))
        else:
            cols[it[o]] = np.where(yv, 1, 2)
    cols[it["sleep"]] = rng.integers(1, 8, n)
    cols[it["bullied"]] = rng.integers(1, 3, n)
    cols[it["ebullied"]] = rng.integers(1, 3, n)
    cols[it["active"]] = rng.integers(1, 9, n)
    cols[it["weight"]] = np.round(rng.uniform(0.3, 3.0, n), 4)
    cols[it["stratum"]] = stratum
    cols[it["psu"]] = psu
    # Blank out a few answers, as in the real files.
    for k in list(cols):
        if k not in (it["weight"], it["stratum"], it["psu"], it["sex"]):
            v = cols[k].astype(object)
            v[rng.random(n) < 0.03] = None
            cols[k] = v
    return cols


def write(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for wave, (dat_rel, sas_rel) in Y.FILES.items():
        cols = make_wave(wave)
        names = sorted(cols)
        widths = {k: (10 if k == "weight" else 8 if k == "psu" else 3) for k in names}
        pos, spec = 17, []
        for k in names:
            spec.append((k, pos, pos + widths[k] - 1))
            pos += widths[k]
        lines = []
        n = len(next(iter(cols.values())))
        for i in range(n):
            row = [" " * 16]
            for k, _a, _b in spec:
                v = cols[k][i]
                s = "" if v is None else (f"{v:.4f}" if k == "weight" else str(int(v)))
                row.append(s.rjust(widths[k]))
            lines.append("".join(row))
        (outdir / dat_rel.split("/")[1]).write_text("\n".join(lines) + "\n")
        body = "\n".join(f"  {k.upper()} $ {a}-{b}" if k not in ("weight",) else f"  {k} {a}-{b}" for k, a, b in spec)
        (outdir / sas_rel.split("/")[1]).write_text(
            f"DATA yrbs{wave};\nINFILE 'x.dat' LRECL=600;\nINPUT\n{body}\n;\nRUN;\n"
        )


if __name__ == "__main__":
    write(sys.argv[1])
