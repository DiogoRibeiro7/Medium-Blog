"""Rebuild the vintage panel and Portugal's current series from the AMECO files.

Outputs (not committed; rebuilt at each run):

- data/vintages.csv: one row per release, country, year and variable, for the 27 member states, from the
  seventeen archived releases and the current one. Each release keeps its own estimates, forecast years
  included, because the comparison is between what each release said and what the current one says.
- data/portugal.csv: Portugal's general government series from the current release, 1994-2025, for the
  replay of the fiscal rules.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_V = HERE / "data" / "vintages.csv"
OUT_P = HERE / "data" / "portugal.csv"

EU27 = [
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "IRL",
    "ITA", "LVA", "LTU", "LUX", "MLT", "NLD", "POL", "PRT", "ROM", "SVK", "SVN", "ESP", "SWE",
]  # fmt: skip
VINTAGE_VARS = {
    "gap": "1.0.0.0.AVGDGP",  # output gap, % of potential GDP
    "Y": "1.1.0.0.OVGD",  # real GDP
    "Ypot": "1.0.0.0.OVGDP",  # potential GDP
    "sb": "1.0.319.0.UBLGAPS",  # structural balance, % of potential GDP
    "cab": "1.0.319.0.UBLGAP",  # cyclically adjusted balance, % of potential GDP
    "capb": "1.0.319.0.UBLGBP",  # cyclically adjusted primary balance, % of potential GDP
    "bal": "1.0.319.0.UBLGE",  # headline balance, % of GDP
}
PORTUGAL_VARS = {
    "debt": "1.0.319.0.UDGG",
    "pb": "1.0.319.0.UBLGIE",
    "bal": "1.0.319.0.UBLGE",
    "interest": "1.0.319.0.UYIGE",
    "revenue": "1.0.319.0.URTG",
    "primary_exp": "1.0.319.0.UUTGI",
    "gap": "1.0.0.0.AVGDGP",
    "gdp_nom": "1.0.0.0.UVGD",
    "gdp_real": "1.1.0.0.OVGD",
    "deflator": "3.1.0.0.PVGD",
    "ltr": "1.1.0.0.ILN",
}
LAST_ACTUAL = 2025


def text_members(path: Path) -> Iterator[bytes]:
    """The AMECO text files in a release; the autumn 2021 release nests them in further zips."""
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            data = z.read(name)
            if name.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(data)) as inner:
                    for n2 in sorted(inner.namelist()):
                        if n2.upper().endswith(".TXT"):
                            yield inner.read(n2)
            elif name.upper().endswith(".TXT"):
                yield data


def read_release(path: Path, codes: dict[str, tuple[str, str]]) -> list[tuple[str, int, str, float]]:
    rows = []
    for raw in text_members(path):
        lines = raw.decode("latin-1").splitlines()
        years = lines[0].rstrip(";").split(";")[5:]
        for line in lines[1:]:
            parts = line.rstrip(";").split(";")
            key = parts[0].strip()
            if key not in codes:
                continue
            country, var = codes[key]
            for y, x in zip(years, parts[5:], strict=False):
                x = x.strip()
                if x in ("NA", ""):
                    continue
                try:
                    rows.append((country, int(y), var, float(x)))
                except ValueError:
                    continue
    return rows


def release_label(path: Path) -> tuple[int, str]:
    m = re.search(r"(spring|autumn)(\d{4})", path.name)
    if m is None:
        raise ValueError(path)
    return int(m.group(2)), m.group(1)


def main() -> None:
    OUT_V.parent.mkdir(exist_ok=True)
    codes = {f"{c}.{v}": (c, k) for c in EU27 for k, v in VINTAGE_VARS.items()}
    frames = []
    archive = json.loads((ROOT / "data_sources" / "ec-ameco-vintages-2011-2021.json").read_text(encoding="utf-8"))
    current = json.loads((ROOT / "data_sources" / "ec-ameco-autumn-2026.json").read_text(encoding="utf-8"))
    releases = [(HERE / f["path"], *release_label(HERE / f["path"])) for f in archive["files"]]
    releases.append((HERE / current["files"][0]["path"], 2026, "autumn"))
    for path, vy, season in releases:
        df = pd.DataFrame(read_release(path, codes), columns=["country", "year", "var", "value"])
        df.insert(0, "season", season)
        df.insert(0, "vintage_year", vy)
        frames.append(df.drop_duplicates(["country", "year", "var"]))
    v = pd.concat(frames, ignore_index=True)
    v.to_csv(OUT_V, index=False)

    pcodes = {f"PRT.{v_}": ("PRT", k) for k, v_ in PORTUGAL_VARS.items()}
    p = pd.DataFrame(read_release(releases[-1][0], pcodes), columns=["country", "year", "var", "value"])
    p = p.drop_duplicates(["year", "var"]).pivot_table(index="year", columns="var", values="value").reset_index()
    p = p[(p.year >= 1994) & (p.year <= LAST_ACTUAL)]
    p.to_csv(OUT_P, index=False)
    n = v.groupby(["vintage_year", "season"]).ngroups
    print(f"wrote {OUT_V.relative_to(HERE)}: {n} releases, {len(v)} rows; {OUT_P.relative_to(HERE)}: {len(p)} years")


if __name__ == "__main__":
    main()
