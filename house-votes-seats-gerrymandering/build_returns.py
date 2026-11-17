"""Build one district-level table of U.S. House general elections, 1972-2024.

Output: data/derived/house_districts_1972_2024.csv with one row per
(year, state, district):

    year, state_po, district, dem_votes, rep_votes, total_votes,
    contested (both major parties on the ballot), winner (D/R/I),
    dem_share (two-party Democratic share; NaN when not contested), source

Rules
-----
* General elections only. Where a court ordered a special open election in
  place of the regular one (Texas 1996 and 2006) the special is used; otherwise
  special elections are dropped.
* Fusion (New York, Connecticut): votes are summed by candidate across party
  lines, and a candidate is a Democrat (Republican) if any of their lines is.
* Runoffs (Louisiana, Texas 1996, Georgia): shares come from the November
  round (all Democrats against all Republicans); the seat goes to the runoff
  winner.
* A district is "contested" when at least one Democrat and one Republican
  received votes. Same-party top-two races count as uncontested for the party
  that won.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rdata

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
OUT = HERE / "data" / "derived"

ICPSR = {
    1: "CT", 2: "ME", 3: "MA", 4: "NH", 5: "RI", 6: "VT", 11: "DE", 12: "NJ",
    13: "NY", 14: "PA", 21: "IL", 22: "IN", 23: "MI", 24: "OH", 25: "WI",
    31: "IA", 32: "KS", 33: "MN", 34: "MO", 35: "NE", 36: "ND", 37: "SD",
    40: "VA", 41: "AL", 42: "AR", 43: "FL", 44: "GA", 45: "LA", 46: "MS",
    47: "NC", 48: "SC", 49: "TX", 51: "KY", 52: "MD", 53: "OK", 54: "TN",
    55: "DC", 56: "WV", 61: "AZ", 62: "CO", 63: "ID", 64: "MT", 65: "NV",
    66: "NM", 67: "UT", 68: "WY", 71: "CA", 72: "OR", 73: "WA", 81: "AK",
    82: "HI",
}

DEM_LABELS = {
    "DEMOCRAT", "DEMOCRATIC", "DEMOCRATIC-FARM-LABOR", "DEMOCRATIC-FARMER-LABOR",
    "DEMOCRATIC-NONPARTISAN LEAGUE", "DEMOCRATIC-NPL", "FOGLIETTA (DEMOCRAT)",
    "WRITE-IN (DEMOCRATIC)",
}
REP_LABELS = {"REPUBLICAN", "INDEPENDENT-REPUBLICAN", "WRITE-IN (REPUBLICAN)"}

# Precinct files carry bookkeeping rows in the candidate column.
META_CANDIDATE = (r"^(TOTAL|CONTEST TOTAL|UNDER ?VOTES?|OVER ?VOTES?|BLANKS?|SPOILED|"
                  r"WRITE-?INS?$|OTHERS$|SCATTER|NONE OF|REGISTERED|BALLOTS|VOID|"
                  r"UNRESOLVED|NO VOTE)")

# JudgeIt leaves the share missing where no votes were printed (unopposed
# candidates in Arkansas and Florida, fusion-only races in New York) or where
# a third candidate won. Winners for those rows, checked against the official
# party totals (1972: 242 D, 192 R, 1 I; 1974: 291 D, 144 R).
JUDGEIT_WINNER_FIX = {
    (1972, "MA", 9): "I",   # Moakley, elected as an independent
    (1972, "NY", 10): "D",
    (1972, "AR", 4): "D",
    (1972, "TN", 6): "R",   # open seat won by Beard
    (1974, "LA", 6): "R",   # Moore, after the court-ordered January 1975 rerun
}


# 2024 gaps in the precinct files: unopposed candidates whose races were not
# printed on the ballot, and NJ-3, whose precinct rows carry no party labels.
MISSING_2024 = [("FL", 20, "D"), ("OK", 3, "R")]
NJ3_2024_PARTY = {"HERB CONAWAY": "D", "RAJESH MOHAN": "R"}


def load_judgeit(years=(1972, 1974)) -> pd.DataFrame:
    parsed = rdata.parser.parse_file(RAW / "JudgeIt" / "data" / "house6311.RData")
    house = rdata.conversion.convert(parsed)["house6311"]
    rows = []
    for year in years:
        d = house[str(year)].copy()
        d["VOTE"] = d["VOTE"].astype(float)
        d["TURNOUT"] = pd.to_numeric(d["TURNOUT"], errors="coerce")
        for r in d.itertuples(index=False):
            st = ICPSR[int(r.STATE)]
            if st == "DC":
                continue
            v, t = r.VOTE, r.TURNOUT
            contested = bool(np.isfinite(v) and 0.0 < v < 1.0)
            if np.isfinite(v):
                winner = "D" if v > 0.5 else "R"
            else:
                inc = int(r.INC) if pd.notna(r.INC) else 0
                winner = {1: "D", -1: "R"}.get(inc, None)
            winner = JUDGEIT_WINNER_FIX.get((year, st, int(r.DIST)), winner)
            dem = v * t if (np.isfinite(v) and np.isfinite(t)) else np.nan
            rep = (1 - v) * t if (np.isfinite(v) and np.isfinite(t)) else np.nan
            rows.append(dict(year=year, state_po=st, district=int(r.DIST),
                             dem_votes=dem, rep_votes=rep, total_votes=t,
                             contested=contested, winner=winner,
                             source="ICPSR 6311 via JudgeIt"))
    return pd.DataFrame(rows)


def _classify(party: str) -> str:
    if party in DEM_LABELS:
        return "D"
    if party in REP_LABELS:
        return "R"
    return "O"


def _collapse_race(g: pd.DataFrame) -> dict:
    """g: candidate-level rows of one race (one round). Returns votes by side."""
    by_cand = g.groupby("candidate", dropna=False).agg(votes=("votes", "sum"))
    line = g.groupby(["candidate", "side"], dropna=False)["votes"].sum().unstack(fill_value=0)
    d_line = line["D"] if "D" in line else 0 * by_cand["votes"]
    r_line = line["R"] if "R" in line else 0 * by_cand["votes"]
    has_d = g.groupby("candidate", dropna=False)["side"].apply(lambda s: (s == "D").any())
    has_r = g.groupby("candidate", dropna=False)["side"].apply(lambda s: (s == "R").any())
    # A candidate on both major-party lines (NY-34 1978: Horton, R, also on the
    # Democratic line) belongs to the line that gave them more votes.
    side = np.where(has_d & has_r, np.where(d_line >= r_line, "D", "R"),
                    np.where(has_d, "D", np.where(has_r, "R", "O")))
    by_cand = by_cand.assign(side=pd.Series(side, index=by_cand.index))
    dem = by_cand.loc[by_cand.side == "D", "votes"].sum()
    rep = by_cand.loc[by_cand.side == "R", "votes"].sum()
    tot = by_cand["votes"].sum()
    top = by_cand.sort_values("votes", ascending=False)
    winner = top.iloc[0]["side"] if len(top) else None
    winner = {"O": "I"}.get(winner, winner)
    return dict(dem_votes=float(dem), rep_votes=float(rep), total_votes=float(tot),
                winner=winner, n_dem=int((by_cand.side == "D").sum()),
                n_rep=int((by_cand.side == "R").sum()))


def load_medsl_1976_2022() -> pd.DataFrame:
    d = pd.read_csv(RAW / "medsl_house_1976_2022.csv", dtype={"district": str},
                    low_memory=False)
    tx96 = (d["year"] == 1996) & (d["state_po"] == "TX") & d["special"].fillna(False).astype(bool)
    d = d[(d["stage"] == "GEN") | tx96].copy()
    d = d[d["state_po"] != "DC"]
    d["votes"] = pd.to_numeric(d["candidatevotes"], errors="coerce").fillna(0).clip(lower=0)
    d["side"] = d["party"].fillna("").map(_classify)
    d["district"] = d["district"].astype(int)
    d["runoff"] = d["runoff"].fillna(False).astype(bool)
    d["special"] = d["special"].fillna(False).astype(bool)
    rows = []
    for (year, st, dist), g in d.groupby(["year", "state_po", "district"]):
        if year == 1996 and st == "TX" and g["special"].any():
            # Court-ordered open election in November, runoff in December.
            first, second = g[g["special"]], g[~g["special"]]
        else:
            regular = g[~g["special"]]
            g = regular if len(regular) else g  # TX 2006 court-ordered specials
            first = g[~g["runoff"]]
            second = g[g["runoff"]]
        base = _collapse_race(first if len(first) else g)
        if len(second):
            base["winner"] = _collapse_race(second)["winner"]
        base.update(year=int(year), state_po=st, district=int(dist),
                    contested=bool(base["dem_votes"] > 0 and base["rep_votes"] > 0),
                    source="MEDSL U.S. House 1976-2022")
        rows.append(base)
    return pd.DataFrame(rows)


def load_medsl_2024() -> pd.DataFrame:
    folder = RAW / "medsl2024" / "individual_states"
    rows = []
    for zf in sorted(folder.glob("*24.zip")):
        with zipfile.ZipFile(zf) as z:
            name = [n for n in z.namelist() if n.endswith(".csv")][0]
            with z.open(name) as fh:
                df = pd.read_csv(fh, dtype=str, low_memory=False,
                                 usecols=["office", "party_simplified", "party_detailed",
                                          "mode", "votes", "candidate", "district",
                                          "stage", "special", "state_po", "writein",
                                          "county_name", "precinct"])
        h = df[(df["office"] == "US HOUSE") & (df["stage"] == "GEN")].copy()
        h["is_special"] = h["special"].fillna("").str.upper() == "TRUE"
        # Keep specials only where they are the only rows for the district
        # (TX-18 2024: the regular general is flagged special in the file).
        has_regular = h["district"].isin(set(h.loc[~h["is_special"], "district"]))
        h = h[~h["is_special"] | ~has_regular]
        if h.empty:
            continue
        h = h[h["candidate"].notna() & (h["candidate"].str.strip() != "")]
        h = h[~h["candidate"].str.upper().str.match(META_CANDIDATE)]
        h["votes"] = pd.to_numeric(h["votes"], errors="coerce").fillna(0).clip(lower=0)
        # Spelling variants of one candidate ("WATSON COLEMAN"/"WATSON-COLEMAN").
        h["candidate"] = (h["candidate"].str.upper().str.replace(r"[^A-Z ]", " ", regex=True)
                          .str.replace(r"\s+", " ", regex=True).str.strip())
        if zf.name.startswith("nj"):
            # Bergen County also lists municipality totals ("Allendale") next to
            # its precincts ("Allendale 1"); drop a row whose name is a prefix
            # of another precinct name in the same county.
            def _aggregates(names: pd.Series) -> set:
                u = set(names.dropna())
                return {n for n in u if any(m.startswith(n + " ") for m in u)}
            agg = {"BERGEN": _aggregates(h.loc[h["county_name"] == "BERGEN", "precinct"])}
            drop = [n in agg.get(c, set()) for c, n in zip(h["county_name"], h["precinct"])]
            h = h[~np.array(drop)]
        # CT, DE, NJ, NY, OK and RI report a TOTAL row next to the by-mode rows
        # for some precincts; there TOTAL equals the sum of the modes (checked:
        # 100% of groups in five states, 99.8% in NJ), so keep TOTAL only.
        h["mode"] = h["mode"].fillna("NA")
        key = ["county_name", "precinct", "candidate", "district"]
        has_total = h.groupby(key, dropna=False)["mode"].transform(lambda s: (s == "TOTAL").any())
        h = h[~(has_total & (h["mode"] != "TOTAL"))]
        modes = set(h["mode"].unique())
        # A candidate's side comes from any row that labels them; Vermont's
        # Republican ran on a REPUBLICAN/LIBERTARIAN line that MEDSL simplifies
        # to OTHER, and NJ leaves the party blank in some counties.
        det = h["party_detailed"].fillna("").str.upper()
        simp = h["party_simplified"].fillna("").str.upper()
        row_side = np.where((simp == "DEMOCRAT") | det.str.split("/").map(lambda t: "DEMOCRAT" in t or "DEMOCRATIC" in t), "D",
                    np.where((simp == "REPUBLICAN") | det.str.split("/").map(lambda t: "REPUBLICAN" in t), "R", ""))
        h["row_side"] = row_side
        cand_side = (h[h["row_side"] != ""].groupby(["district", "candidate"])["row_side"]
                     .agg(lambda s: s.mode().iloc[0]))
        h["side"] = [cand_side.get((dd, cc), "O") for dd, cc in zip(h["district"], h["candidate"])]
        if zf.name.startswith("nj"):
            fix = (h["district"] == "003") & h["candidate"].isin(NJ3_2024_PARTY)
            h.loc[fix, "side"] = h.loc[fix, "candidate"].map(NJ3_2024_PARTY)
        # Unattributed write-in and over/under-vote rows carry no district.
        h = h[h["district"].notna()]
        num = pd.to_numeric(h["district"], errors="coerce")
        at_large = h["district"].str.upper().isin(["AT-LARGE", "STATEWIDE", "AT LARGE"])
        bad = h.loc[num.isna() & ~at_large, "district"].unique()
        if len(bad):
            raise ValueError(f"{zf.name}: unrecognised district labels {bad[:5]}")
        h["district"] = num.fillna(0).astype(int)
        if (h["district"] > 0).any():  # stray statewide write-ins (RI)
            h = h[h["district"] > 0]
        for (st, dist), g in h.groupby(["state_po", "district"]):
            base = _collapse_race(g)
            base.update(year=2024, state_po=st, district=int(dist),
                        contested=bool(base["dem_votes"] > 0 and base["rep_votes"] > 0),
                        source="MEDSL 2024 precinct returns, aggregated",
                        modes="|".join(sorted(modes)))
            rows.append(base)
    for st, dist, party in MISSING_2024:
        rows.append(dict(year=2024, state_po=st, district=dist, dem_votes=np.nan,
                         rep_votes=np.nan, total_votes=np.nan, winner=party,
                         contested=False, source="unopposed, not on ballot"))
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    early = load_judgeit()
    mid = load_medsl_1976_2022()
    late = load_medsl_2024()
    cols = ["year", "state_po", "district", "dem_votes", "rep_votes", "total_votes",
            "contested", "winner", "source"]
    allx = pd.concat([early[cols], mid[cols], late[cols]], ignore_index=True)
    allx["dem_share"] = np.where(allx["contested"],
                                 allx["dem_votes"] / (allx["dem_votes"] + allx["rep_votes"]),
                                 np.nan)
    allx = allx.sort_values(["year", "state_po", "district"]).reset_index(drop=True)
    allx.to_csv(OUT / "house_districts_raw_1972_2024.csv", index=False)
    print(allx.groupby("year").agg(n=("district", "size"),
                                   D=("winner", lambda s: (s == "D").sum()),
                                   R=("winner", lambda s: (s == "R").sum()),
                                   I=("winner", lambda s: (s == "I").sum()),
                                   none=("winner", lambda s: s.isna().sum()),
                                   uncont=("contested", lambda s: (~s).sum())).to_string())


if __name__ == "__main__":
    main()
