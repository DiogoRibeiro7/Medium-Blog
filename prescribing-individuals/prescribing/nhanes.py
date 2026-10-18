"""NHANES: download the public files and build one tidy row per examined adult.

Cycles 2011-2012 (G), 2013-2014 (H), 2015-2016 (I), 2017-March 2020 pre-pandemic
(P) and August 2021-August 2023 (L). Blood pressure was measured by auscultation
in G, H and I and by an oscillometric device in P and L, so the blood-pressure
baselines in the article use P and L only; all five cycles are used for
prescription stimulant use.

Pooled weights follow the NHANES guidance for combining cycles: each cycle's
examination weight is multiplied by its share of the combined time span.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import http
from .paths import DATA, RAW

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt"
OLD_BASE = "https://wwwn.cdc.gov/Nchs/Nhanes/{period}/{name}.XPT"


@dataclass(frozen=True)
class Cycle:
    code: str
    period: str
    year: int
    years: float          # time span covered, for pooled weights
    mode: str             # auscultatory or oscillometric
    weight: str
    files: dict           # role -> file name


CYCLES = [
    Cycle("G", "2011-2012", 2011, 2.0, "auscultatory", "WTMEC2YR",
          dict(demo="DEMO_G", bp="BPX_G", bpq="BPQ_G", mcq="MCQ_G", bmx="BMX_G", rx="RXQ_RX_G")),
    Cycle("H", "2013-2014", 2013, 2.0, "auscultatory", "WTMEC2YR",
          dict(demo="DEMO_H", bp="BPX_H", bpq="BPQ_H", mcq="MCQ_H", bmx="BMX_H", rx="RXQ_RX_H")),
    Cycle("I", "2015-2016", 2015, 2.0, "auscultatory", "WTMEC2YR",
          dict(demo="DEMO_I", bp="BPX_I", bpq="BPQ_I", mcq="MCQ_I", bmx="BMX_I", rx="RXQ_RX_I")),
    Cycle("P", "2017-2020", 2017, 3.2, "oscillometric", "WTMECPRP",
          dict(demo="P_DEMO", bp="P_BPXO", bpq="P_BPQ", mcq="P_MCQ", bmx="P_BMX", rx="P_RXQ_RX")),
    Cycle("L", "2021-2023", 2021, 2.0, "oscillometric", "WTMEC2YR",
          dict(demo="DEMO_L", bp="BPXO_L", bpq="BPQ_L", mcq="MCQ_L", bmx="BMX_L", rx="RXQ_RX_L")),
]

STIMULANT_TERMS = ["METHYLPHENIDATE", "DEXMETHYLPHENIDATE", "AMPHETAMINE", "DEXTROAMPHETAMINE",
                   "LISDEXAMFETAMINE", "METHAMPHETAMINE", "SERDEXMETHYLPHENIDATE"]


def _fetch(cycle: Cycle, name: str) -> pd.DataFrame | None:
    dest = RAW / "nhanes" / f"{name}.xpt"
    path = http.download(BASE.format(year=cycle.year, name=name), dest, ok_404=True)
    if path is None:
        path = http.download(OLD_BASE.format(period=cycle.period, name=name), dest, ok_404=True)
    if path is None:
        print(f"  NHANES {name}: not available")
        return None
    data = path.read_bytes()
    if data[:6] != b"HEADER":  # an HTML error page instead of a SAS transport file
        path.unlink()
        print(f"  NHANES {name}: not a transport file, skipped")
        return None
    df = pd.read_sas(io.BytesIO(data), format="xport", encoding="latin-1")
    df.columns = [c.upper() for c in df.columns]
    print(f"  NHANES {name}: {len(df):,} rows")
    return df


def _first(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(np.nan, index=df.index)


def _bp_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    sys_cols = [c for c in ("BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4")
                if c in df.columns]
    dia_cols = [c for c in ("BPXODI1", "BPXODI2", "BPXODI3", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4")
                if c in df.columns]
    return sys_cols, dia_cols


def build_cycle(cycle: Cycle) -> pd.DataFrame | None:
    print(f"NHANES cycle {cycle.code} ({cycle.period})")
    demo = _fetch(cycle, cycle.files["demo"])
    if demo is None:
        return None
    out = pd.DataFrame({
        "seqn": demo["SEQN"].astype(int),
        "cycle": cycle.code,
        "mode": cycle.mode,
        "age": demo["RIDAGEYR"],
        "sex": demo["RIAGENDR"].map({1: "male", 2: "female"}),
        "examined": demo.get("RIDSTATR", pd.Series(2, index=demo.index)) == 2,
        "pregnant": _first(demo, ["RIDEXPRG"]) == 1,
        "w_cycle": demo[cycle.weight],
        "stratum": cycle.code + "-" + demo["SDMVSTRA"].astype(int).astype(str),
        "psu": demo["SDMVPSU"].astype(int),
        "cycle_years": cycle.years,
    })

    bp = _fetch(cycle, cycle.files["bp"])
    if bp is not None:
        sys_cols, dia_cols = _bp_columns(bp)
        s = bp.set_index("SEQN")[sys_cols].rename(columns=lambda c: f"sbp{c[-1]}")
        d = bp.set_index("SEQN")[dia_cols].rename(columns=lambda c: f"dbp{c[-1]}")
        d = d.where(d > 0)  # a recorded diastolic of 0 is not a measurement
        out = out.join(s, on="seqn").join(d, on="seqn")

    bpq = _fetch(cycle, cycle.files["bpq"])
    if bpq is not None:
        b = bpq.set_index("SEQN")
        out = out.join(pd.DataFrame({
            "told_hbp": _first(b, ["BPQ020"]) == 1,
            "bp_med_answer": _first(b, ["BPQ050A", "BPQ150", "BPQ040A"]),
        }), on="seqn")
        out["bp_med"] = np.where(out["bp_med_answer"].isin([1, 2]), out["bp_med_answer"] == 1, False)
        out = out.drop(columns="bp_med_answer")

    mcq = _fetch(cycle, cycle.files["mcq"])
    if mcq is not None:
        m = mcq.set_index("SEQN")
        flags = [c for c in ("MCQ160B", "MCQ160C", "MCQ160E", "MCQ160F") if c in m.columns]
        out = out.join(pd.DataFrame({"cvd": (m[flags] == 1).any(axis=1)}), on="seqn")

    bmx = _fetch(cycle, cycle.files["bmx"])
    if bmx is not None and "BMXBMI" in bmx.columns:
        out = out.join(bmx.set_index("SEQN")[["BMXBMI"]].rename(columns={"BMXBMI": "bmi"}), on="seqn")

    rx = _fetch(cycle, cycle.files["rx"])
    if rx is not None and "RXDDRUG" in rx.columns:
        names = rx["RXDDRUG"].fillna("").astype(str).str.upper()
        hit = names.apply(lambda s: any(t in s for t in STIMULANT_TERMS))
        users = rx.loc[hit].groupby("SEQN")["RXDDRUG"].apply(lambda s: "; ".join(sorted(set(map(str, s)))))
        out["rx_available"] = True
        out["stimulant"] = out["seqn"].isin(users.index)
        out = out.join(users.rename("stimulant_names"), on="seqn")
    else:
        out["rx_available"] = False
        out["stimulant"] = np.nan
    return out


def build(force: bool = False) -> pd.DataFrame:
    dest = DATA / "nhanes_adults.csv.gz"
    frames = [f for f in (build_cycle(c) for c in CYCLES) if f is not None]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["examined"] & (df["age"] >= 18)].copy()
    total_years = sum(c.years for c in CYCLES if c.code in set(df["cycle"]))
    osc_years = sum(c.years for c in CYCLES if c.mode == "oscillometric" and c.code in set(df["cycle"]))
    df["w_all"] = df["w_cycle"] * df["cycle_years"] / total_years
    df["w_osc"] = np.where(df["mode"] == "oscillometric", df["w_cycle"] * df["cycle_years"] / osc_years, 0.0)
    sbp = df.filter(regex=r"^sbp\d$")
    dbp = df.filter(regex=r"^dbp\d$")
    df["sbp"] = sbp.mean(axis=1)
    df["dbp"] = dbp.mean(axis=1)
    df["n_sbp"] = sbp.notna().sum(axis=1)
    df = df.drop(columns=["examined"])
    df.to_csv(dest, index=False, compression="gzip")
    print(f"NHANES: {len(df):,} examined adults written to {dest.name}")
    return df


def load() -> pd.DataFrame:
    return pd.read_csv(DATA / "nhanes_adults.csv.gz", low_memory=False)
