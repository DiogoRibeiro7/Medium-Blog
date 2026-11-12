"""Public data used by the actuarial-tariff-calibration series.

Every file is downloaded once from a pinned commit of the CASdatasets repository (Dutang and Charpentier),
checked against its SHA-256 digest and kept in ``.cache/`` next to this file. Nothing is redistributed
from this repository; CASdatasets is distributed under the GPL (>= 2).

    from data import swedish_1977, french_mtpl2, belgian_mtpl97
    cells = swedish_1977()        # dict of NumPy arrays, one entry per column
    policies, claims = french_mtpl2()
    belgian = belgian_mtpl97()
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np

CAS_COMMIT = "227fb56b8734bdb7c0327a41180e01d2ddaeaf26"  # dutangc/CASdatasets, 10 June 2026
CAS_RAW = f"https://raw.githubusercontent.com/dutangc/CASdatasets/{CAS_COMMIT}/"

FILES = {
    # name: (path in CASdatasets, SHA-256 of the file)
    "swautoins": ("data/swautoins.rda", "057594a03af696dff034363f1c5af751c51e01669a751b170e491ca4650d9ba7"),
    "freMTPL2freq": ("data/freMTPL2freq.rda", "82c8598513d9fa78226b6c7d271a9940eca4b8083e9e32320a75d377bdfe15a3"),
    "freMTPL2sev": ("data/freMTPL2sev.rda", "78e6e5016ae046603b37e88ff8ad8ca327f5d59f827c1f75d140388de09b14b3"),
    "beMTPL97": ("data/beMTPL97.rda", "955a821a7a693bf18076c425e4d5a5a99889f3c89e1fbd99eca6239c11e963a6"),
}

CACHE = Path(__file__).resolve().parent / ".cache"


def fetch(name: str) -> Path:
    """Return the local path of a CASdatasets file, downloading and checking it the first time."""
    rel, digest = FILES[name]
    path = CACHE / Path(rel).name
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        with urllib.request.urlopen(CAS_RAW + rel, timeout=120) as response:
            body = response.read()
        tmp = path.with_suffix(".part")
        tmp.write_bytes(body)
        tmp.replace(path)
    found = hashlib.sha256(path.read_bytes()).hexdigest()
    if found != digest:
        raise RuntimeError(f"{path.name}: SHA-256 {found} does not match the pinned {digest}")
    return path


def _read_rda(path: Path, obj: str):
    import rdata  # imported here so that parts without R data do not need it

    return rdata.read_rda(path)[obj]


def swedish_1977() -> dict[str, np.ndarray]:
    """Swedish third-party motor insurance, 1977 (CASdatasets ``swautoins``; 2,182 tariff cells).

    Compiled by the Swedish Committee on the Analysis of Risk Premium in Motor Insurance; studied by
    Hallin and Ingenbleek (1983). Rating factors are coded from 1: Kilometres (5 bands of annual distance),
    Zone (7 geographical zones, 1 = Stockholm, Goteborg and Malmo with surroundings, 7 = Gotland),
    Bonus (years since the last claim plus one, 1 to 7) and Make (8 common car models and 9 = all others).
    Insured is exposure in policy-years, Claims the number of claims, Payment the total paid in kronor.
    """
    df = _read_rda(fetch("swautoins"), "swautoins")
    out = {c: df[c].to_numpy(dtype=int) for c in ("Kilometres", "Zone", "Bonus", "Make")}
    out["Insured"] = df["Insured"].to_numpy(dtype=float)
    out["Claims"] = df["Claims"].to_numpy(dtype=float)
    out["Payment"] = df["Payment"].to_numpy(dtype=float)
    return out


def french_mtpl2():
    """French motor third-party liability, freMTPL2freq and freMTPL2sev (CASdatasets; 677,991 policies).

    Returns the policy table and the claim table as pandas DataFrames, exactly as stored. In the pinned version
    the two tables reconcile: every policy with claims has as many claim amounts as claims. Cleaning (exposure
    capped at one year, claim counts capped at four) is left to the caller and documented there.
    """
    freq = _read_rda(fetch("freMTPL2freq"), "freMTPL2freq")
    sev = _read_rda(fetch("freMTPL2sev"), "freMTPL2sev")
    return freq, sev


def belgian_mtpl97():
    """Belgian motor third-party liability, 1997 (CASdatasets ``beMTPL97``; 163,212 policies).

    One row per policy with its exposure (at most one year), its number of claims and the rating factors: the
    policyholder's age and sex, the bonus-malus level (0 to 22 on the Belgian scale), the vehicle's power in kW and
    age, fuel, use, fleet membership, the type of cover, the postcode and its longitude and latitude. Returned as a
    pandas DataFrame, exactly as stored.
    """
    return _read_rda(fetch("beMTPL97"), "beMTPL97")
