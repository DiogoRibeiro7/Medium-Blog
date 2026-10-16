"""The validation sample: drawn once, stratified, and shown to the annotator blind.

Strata are era x dictionary tag (discipline) x side of the Member who last had the floor.
Tagged strata give the dictionary's precision; untagged strata give what it misses. The
annotator sees the speech and its neighbours but not the tag, the stratum or the era
label, so the labels cannot lean on the instrument they are meant to check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .data import iter_chunks

SIDE_KEYS = {"Government": "government", "Opposition": "opposition", "Other": "other"}


def target_size(cfg: Config, d_tag: int, side: str) -> int:
    prefix = "tagged" if d_tag else "untagged"
    return int(cfg.sample[f"{prefix}_{SIDE_KEYS[side]}"])


def draw(cfg: Config, force: bool = False) -> pd.DataFrame:
    design_path = cfg.labels_dir / "design.csv"
    if design_path.exists() and not force:
        raise FileExistsError(
            f"{design_path} exists. The validation sample is drawn once; pass --force only before any labelling."
        )
    chair = pd.read_parquet(cfg.build_dir / "chair.parquet")
    rng = np.random.default_rng(int(cfg.sample["seed"]))
    picked = []
    for era in cfg.era_names:
        for d_tag in (1, 0):
            for side in SIDE_KEYS:
                pool = chair[(chair["era"] == era) & (chair["d_tag"] == d_tag) & (chair["target_side"] == side)]
                pool = pool.sort_values("id")
                n = min(target_size(cfg, d_tag, side), len(pool))
                if n == 0:
                    continue
                take = pool.iloc[rng.choice(len(pool), size=n, replace=False)].copy()
                take["N"] = len(pool)
                take["n"] = n
                picked.append(take)
    sample = pd.concat(picked, ignore_index=True)
    sample = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)
    sample["item"] = np.arange(1, len(sample) + 1)

    ctx = _context(cfg, set(sample["prev_id"]) | set(sample["next_id"]))
    k = int(cfg.sample["context_chars"])
    before = sample["prev_id"].map(ctx["text"]).fillna("")
    after = sample["next_id"].map(ctx["text"]).fillna("")
    blind = pd.DataFrame(
        {
            "item": sample["item"],
            "id": sample["id"],
            "date": pd.to_datetime(sample["date"]).dt.date.astype(str),
            "before_speaker": sample["prev_id"].map(ctx["speakername"]).fillna(""),
            "before_text": before.map(lambda s: s if len(s) <= k else "... " + s[-k:]),
            "chair_speaker": sample["speakername"],
            "chair_text": sample["text"],
            "after_speaker": sample["next_id"].map(ctx["speakername"]).fillna(""),
            "after_text": after.map(lambda s: s if len(s) <= k else s[:k] + " ..."),
        }
    )
    design = sample[["item", "id", "era", "d_tag", "target_side", "N", "n", "h_tag"]].rename(
        columns={"target_side": "side"}
    )
    cfg.labels_dir.mkdir(parents=True, exist_ok=True)
    blind.to_csv(cfg.labels_dir / "sample.csv", index=False)
    design.to_csv(design_path, index=False)
    return design


def _context(cfg: Config, ids: set[str]) -> pd.DataFrame:
    ids = {i for i in ids if i}
    parts = []
    for chunk in iter_chunks(cfg, columns=["id", "speech", "speakername"]):
        hit = chunk[chunk["id"].isin(ids)]
        if len(hit):
            parts.append(hit)
    if not parts:
        return pd.DataFrame(columns=["text", "speakername"])
    ctx = pd.concat(parts).drop_duplicates("id").set_index("id")
    return ctx.rename(columns={"speech": "text"})[["text", "speakername"]]
