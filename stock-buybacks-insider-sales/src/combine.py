"""Concatenate shard outputs and run the analysis.  python3 -I src/combine.py shards_dir out_dir"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import analysis


def main(shards, out):
    files = sorted(glob.glob(os.path.join(shards, "**", "events.csv"), recursive=True))
    ev = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    ev = ev.drop_duplicates(["cik", "accession"]) if "accession" in ev else ev
    os.makedirs(out, exist_ok=True)
    ev.to_csv(os.path.join(out, "events.csv"), index=False)
    if len(ev):
        aud = ev[ev.kind.str.startswith("announcement")]
        aud = aud.sample(min(50, len(aud)), random_state=7) if len(aud) else aud
        aud[["ticker", "date", "kind", "accession", "snippet"]].to_csv(
            os.path.join(out, "announcement_audit_sample.csv"), index=False)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(analysis.summary(ev) if len(ev) else {}, f, indent=2)
    print("shards:", len(files), "events:", len(ev), "firms:", ev.cik.nunique() if len(ev) else 0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
