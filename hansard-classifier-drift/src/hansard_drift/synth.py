"""A small synthetic corpus in the schema of the real one, with known truth.

Used by the tests and CI only. The Chair's speech changes across eras on purpose: later
eras use phrasings the dictionary misses and transitions it wrongly counts, so the
pipeline has a known drift to recover. Rows are written out of sitting order, so the
ordering by id is exercised too.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from .config import Config

CAUGHT = [
    "Order. Order.",
    "Order. The hon. Gentleman must resume his seat.",
    "Order. There is far too much noise in the Chamber.",
    "The hon. Member must withdraw that remark.",
    "That language is unparliamentary.",
]
MISSED = [
    "Less of that, please. The hon. Gentleman will contain himself.",
    "I will not have that. Let the Minister answer.",
    "Members must stop chuntering from the Benches.",
]
FALSE_POSITIVE = [
    "Order. I call Mr. {name}.",
    "That is not a point of order; it is a matter for debate, and the hon. Gentleman has made his point.",
    "Order. Statement, the Secretary of State.",
]
NEUTRAL = [
    "I call Mr. {name}.",
    "The Question is, That the amendment be made.",
    "I have selected the amendment in the name of the Prime Minister.",
    "We now come to the motion on the Order Paper.",
]
# Per era, in config order: P(caught discipline), P(missed discipline), P(false positive).
DRIFT = [
    (0.40, 0.02, 0.04),
    (0.45, 0.03, 0.05),
    (0.38, 0.05, 0.08),
    (0.33, 0.08, 0.10),
    (0.24, 0.12, 0.16),
    (0.24, 0.10, 0.14),
]
CHAIR_NAMES = [
    "Mr. Speaker",
    "Madam Speaker",
    "Mr. Deputy Speaker",
    "The Chairman",
    "Mr. Deputy Speaker (Mr. Paul Dean)",
]
PARTIES = ["Conservative", "Labour", "Liberal Democrat", "Scottish National Party", "Labour (Co-op)"]


def write_synthetic(cfg: Config, days_per_era: int = 30, rows_per_day: int = 60, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    members = [(f"Member {i}", PARTIES[i % len(PARTIES)]) for i in range(60)]
    rows, truth = [], []
    for ei, era in enumerate(cfg.eras):
        span = (era.end - era.start).days
        offsets = np.sort(rng.choice(np.arange(1, span), size=min(days_per_era, span - 1), replace=False))
        p_caught, p_missed, p_fp = DRIFT[ei % len(DRIFT)]
        for off in offsets:
            day = era.start + dt.timedelta(days=int(off))
            col, para = 100, 0
            last_party = None
            for _ in range(rows_per_day):
                para += 1
                if para > 6:
                    col, para = col + 1, 1
                gid = f"uk.org.publicwhip/debate/{day.isoformat()}a.{col}.{para}"
                u = rng.random()
                base = {
                    "id": gid,
                    "date": pd.Timestamp(day),
                    "colnum": str(col),
                    "display_as": "",
                    "person_id": "",
                    "parliamentary_posts": [],
                }
                if u < 0.55:
                    name, party = members[rng.integers(len(members))]
                    last_party = party
                    rows.append(
                        {
                            **base,
                            "speech": "I beg to move. " * int(rng.integers(1, 40)),
                            "party": party,
                            "speech_class": "Speech",
                            "speakername": f"Mr. {name}",
                            "display_as": name,
                            "person_id": name,
                        }
                    )
                elif u < 0.60:
                    rows.append(
                        {
                            **base,
                            "speech": "Resign!",
                            "party": "",
                            "speech_class": "Speech",
                            "speakername": "Hon. Members",
                        }
                    )
                elif u < 0.63:
                    rows.append(
                        {
                            **base,
                            "speech": "Question put and agreed to.",
                            "party": "",
                            "speech_class": "Procedural",
                            "speakername": "",
                        }
                    )
                else:
                    v = rng.random()
                    opp_boost = 1.5 if last_party is not None and last_party != "Conservative" else 1.0
                    if v < p_caught:
                        text, y = CAUGHT[rng.integers(len(CAUGHT))], 1
                    elif v < p_caught + p_missed:
                        text, y = MISSED[rng.integers(len(MISSED))], 1
                    elif v < p_caught + p_missed + p_fp * opp_boost:
                        text, y = FALSE_POSITIVE[rng.integers(len(FALSE_POSITIVE))], 0
                    else:
                        text, y = NEUTRAL[rng.integers(len(NEUTRAL))], 0
                    text = text.format(name=members[rng.integers(len(members))][0])
                    chair = CHAIR_NAMES[rng.integers(len(CHAIR_NAMES))]
                    posts = []
                    if rng.random() < 0.05:
                        chair, posts = "Mr. Paul Dean", ["Deputy Speaker and First Deputy Chairman of Ways and Means"]
                    rows.append(
                        {
                            **base,
                            "speech": text,
                            "party": "Speaker" if "Speaker" in chair else "",
                            "speech_class": "Speech",
                            "speakername": chair,
                            "parliamentary_posts": posts,
                        }
                    )
                    truth.append({"id": gid, "truth_discipline": y})
    df = pd.DataFrame(rows)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    out = cfg.cache_dir / "default" / "train"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "0000.parquet", index=False)
    pd.DataFrame(truth).to_parquet(cfg.cache_dir / "truth.parquet", index=False)
    manifest = {"repo_id": "synthetic", "revision": "synthetic", "commit": "synthetic", "rows": len(df), "files": []}
    (cfg.cache_dir / "fetch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def autolabel(cfg: Config, annotator: str = "synthetic") -> int:
    """Label the drawn sample from the synthetic truth (tests only)."""
    truth = pd.read_parquet(cfg.cache_dir / "truth.parquet").set_index("id")["truth_discipline"]
    sample = pd.read_csv(cfg.labels_dir / "sample.csv", dtype=str, keep_default_na=False)
    lab = pd.DataFrame({"id": sample["id"], "discipline": sample["id"].map(truth).astype(int), "unsure": 0, "note": ""})
    lab.to_csv(cfg.labels_dir / f"labels_{annotator}.csv", index=False)
    return len(lab)
