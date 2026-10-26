"""Read and clean the Anonymous Bank call-centre records (1999).

The data were collected by Ilan Guedj and Avi Mandelbaum and are published
by the Technion SEE Center:

    https://see-center.iem.technion.ac.il/databases/AnonymousBank/Data/AnonymousBank.rar

The archive holds twelve tab-separated monthly files, one row per call. This
module reads them (from the archive or from a folder of extracted files),
parses the clock times and returns one cleaned table of calls. The raw data
are not stored in the repository; ``fetch.py`` downloads them and checks the
archive's SHA-256.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVE_URL = ("https://see-center.iem.technion.ac.il/databases/"
               "AnonymousBank/Data/AnonymousBank.rar")
ARCHIVE_SHA256 = ("110527823a296e3163566f5d10d687e1"
                  "3859c60d963fd403d4623d89c7cadc4b")
CLOCKS = ("vru_entry", "vru_exit", "q_start", "q_exit", "ser_start",
          "ser_exit")
NUMBERS = ("vru_time", "q_time", "ser_time", "priority")


def _clock_seconds(s: pd.Series) -> pd.Series:
    """'h:mm:ss' to seconds after midnight; anything else becomes NaN."""
    parts = s.str.split(":", expand=True)
    if parts.shape[1] != 3:
        return pd.Series(np.nan, index=s.index)
    parts = parts.apply(pd.to_numeric, errors="coerce")
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def _read_month(text: str) -> pd.DataFrame:
    frame = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
    frame.columns = [c.strip() for c in frame.columns]
    for c in frame.columns:
        frame[c] = frame[c].str.strip()
    return frame


def read_monthly_texts(source: Path) -> list[str]:
    """Contents of the twelve monthly files, from the archive or a folder."""
    source = Path(source)
    if source.is_dir():
        files = sorted(source.glob("*1999.txt"))
        return [f.read_text(encoding="latin-1") for f in files]
    listing = subprocess.run(["bsdtar", "-tf", str(source)], check=True,
                             capture_output=True, text=True).stdout.split()
    names = sorted(n for n in listing if n.endswith("1999.txt"))
    return [subprocess.run(["bsdtar", "-xOf", str(source), n], check=True,
                           capture_output=True).stdout.decode("latin-1")
            for n in names]


def load_calls(source: Path) -> tuple[pd.DataFrame, dict]:
    """All calls of 1999, cleaned, with the counts of what was dropped.

    Columns added: ``date`` (datetime), ``dow`` (Monday = 0), the six clock
    times in seconds after midnight (suffix ``_s``), ``t_entry`` (seconds
    since 1970 at entry to the voice-response unit), and three flags:
    ``queued`` (waited in the queue for an agent), ``abandoned`` (hung up
    while queued) and ``served`` (reached an agent).
    """
    raw = pd.concat([_read_month(t) for t in read_monthly_texts(source)],
                    ignore_index=True)
    info = {"rows_read": len(raw)}
    for c in CLOCKS:
        raw[c + "_s"] = _clock_seconds(raw[c])
    for c in NUMBERS:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["date"] = pd.to_datetime("19" + raw["date"], format="%Y%m%d",
                                 errors="coerce")
    bad = raw[[c + "_s" for c in CLOCKS] + list(NUMBERS) + ["date"]] \
        .isna().any(axis=1)
    info["malformed_rows"] = int(bad.sum())
    calls = raw[~bad].copy()
    calls["dow"] = calls["date"].dt.dayofweek
    calls["t_entry"] = (calls["date"].astype("int64") // 10 ** 9
                        + calls["vru_entry_s"])
    calls["queued"] = calls["q_time"] > 0
    calls["abandoned"] = calls["queued"] & (calls["outcome"] == "HANG")
    calls["served"] = calls["outcome"] == "AGENT"
    calls["has_id"] = calls["customer_id"] != "0"
    info["calls"] = len(calls)
    info["days"] = int(calls["date"].nunique())
    return calls.reset_index(drop=True), info
