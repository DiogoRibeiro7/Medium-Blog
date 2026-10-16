"""Stage 1: roles, sitting order, who had the floor, dictionary tags and the naive rates.

Everything here runs without labels. It writes the population counts the correction
needs and the naive numbers a dictionary study would publish.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from . import __version__
from .config import Config
from .data import fetch_manifest, iter_chunks
from .dictionary import CATEGORIES, DICTIONARY_VERSION, DISCIPLINE, describe, normalise, tag_series
from .roles import (
    assign_era,
    assign_side,
    classify_roles,
    normalise_party,
    order_keys,
    roster_match,
    speaker_surnames,
)

LIGHT_COLUMNS = [
    "id",
    "date",
    "k_letter",
    "k_col",
    "k_para",
    "k_parsed",
    "role",
    "chair_kind",
    "role_source",
    "name_norm",
    "speakername",
    "display_as",
    "party_norm",
    "n_words",
    "opens_order",
    "speech_class",
]
CATEGORICAL = [
    "role",
    "chair_kind",
    "role_source",
    "name_norm",
    "speakername",
    "display_as",
    "party_norm",
    "speech_class",
]
SIDES = ["Government", "Opposition", "Other"]
LENGTH_THRESHOLDS = (5, 10, 20, 50)


def _read(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    light_parts, chair_parts = [], []
    row0 = 0
    for chunk in iter_chunks(cfg):
        chunk["date"] = pd.to_datetime(chunk["date"]).dt.normalize()
        df = classify_roles(
            chunk,
            speaker_surname=speaker_surnames(chunk["date"], cfg),
            roster_hit=roster_match(chunk["display_as"], chunk["date"], cfg),
        )
        df = df.join(order_keys(df["id"], df["colnum"]))
        text = df["speech"].fillna("").astype(str)
        df["n_words"] = text.str.count(r"\S+").astype("int32")
        # Audit only, never used to assign roles: speeches that open with a bare "Order".
        df["opens_order"] = text.str.match(r"\s*order\s*[.!,]", case=False).fillna(False).astype(bool)
        df["display_as"] = df["display_as"].fillna("").astype(str)
        df["party_norm"] = df["party"].map(normalise_party)
        light = df[LIGHT_COLUMNS].copy()
        light["row"] = np.arange(row0, row0 + len(df), dtype="int64")
        row0 += len(df)
        for c in CATEGORICAL:
            light[c] = light[c].astype("category")
        light_parts.append(light)
        is_chair = (df["role"] == "chair").to_numpy()
        chair_parts.append(pd.DataFrame({"id": df["id"].to_numpy()[is_chair], "text": text.to_numpy()[is_chair]}))
    light = pd.concat(light_parts, ignore_index=True)
    for c in CATEGORICAL:
        light[c] = light[c].astype(str).astype("category")
    return light, pd.concat(chair_parts, ignore_index=True)


def _order_and_floor(light: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    light = light.sort_values(["date", "k_letter", "k_col", "k_para", "row"], kind="stable").reset_index(drop=True)
    light["era"] = assign_era(light["date"], cfg)
    light = light[light["era"].notna()].reset_index(drop=True)
    is_member = (light["role"] == "member").to_numpy()
    side = assign_side(light["party_norm"].astype(str), light["date"], cfg)
    light["side"] = np.where(is_member, side, "")

    # Who had the floor: the most recent Member to speak that day, before this row.
    day = light["date"].to_numpy()
    idx = pd.Series(np.where(is_member, np.arange(len(light)), np.nan))
    last = idx.groupby(day).ffill().to_numpy()
    has = ~np.isnan(last)
    li = np.where(has, last, 0).astype("int64")
    light["target_side"] = np.where(has, light["side"].to_numpy()[li], "Other")
    light["target_party"] = np.where(has, light["party_norm"].astype(str).to_numpy()[li], "")
    light.loc[light["target_side"] == "", "target_side"] = "Other"

    # Neighbours, for the context shown to the annotator.
    same_prev = np.r_[False, day[1:] == day[:-1]]
    same_next = np.r_[day[:-1] == day[1:], False]
    ids = light["id"].to_numpy()
    light["prev_id"] = np.where(same_prev, np.r_[[""], ids[:-1]], "")
    light["next_id"] = np.where(same_next, np.r_[ids[1:], [""]], "")
    return light


def build(cfg: Config) -> dict:
    light, chair_text = _read(cfg)
    light = _order_and_floor(light, cfg)

    chair = light[light["role"] == "chair"].merge(chair_text, on="id", how="left")
    chair["norm"] = chair["text"].map(normalise)
    chair = pd.concat([chair, tag_series(chair["norm"])], axis=1)

    cfg.build_dir.mkdir(parents=True, exist_ok=True)
    rows = light[["id", "date", "era", "role", "speakername", "side", "prev_id", "next_id"]].copy()
    rows["speakername"] = rows["speakername"].astype(str)
    rows.to_parquet(cfg.build_dir / "rows.parquet", index=False)
    keep = [
        "id",
        "date",
        "era",
        "chair_kind",
        "role_source",
        "speakername",
        "target_side",
        "target_party",
        "prev_id",
        "next_id",
        "n_words",
        "text",
        "d_tag",
        "h_tag",
        *[f"hits_{c.name}" for c in CATEGORIES],
    ]
    out = chair[keep].copy()
    for c in ["chair_kind", "role_source", "speakername"]:
        out[c] = out[c].astype(str)
    out.to_parquet(cfg.build_dir / "chair.parquet", index=False)

    summary = write_stage1(cfg, light, chair)
    return summary


def write_stage1(cfg: Config, light: pd.DataFrame, chair: pd.DataFrame) -> dict:
    """The label-free results: population counts, denominators, naive rates, audits."""
    d = cfg.stage1_dir
    d.mkdir(parents=True, exist_ok=True)
    eras = cfg.era_names
    members = light[light["role"] == "member"]

    # Role audit: every printed name classified as the Chair, and every name that looks
    # like a title but was not.
    names = light.groupby(["name_norm", "role", "chair_kind", "role_source"], observed=True).size()
    names = names.rename("rows").reset_index()
    titled = names["name_norm"].astype(str).str.contains(r"speaker|chair", regex=True)
    audit = names[(names["role"] == "chair") | titled].sort_values("rows", ascending=False)
    audit.to_csv(d / "role_audit.csv", index=False)

    roles = light.groupby(["era", "role"], observed=True).size().unstack(fill_value=0).reindex(eras)
    roles.to_csv(d / "roles_by_era.csv")
    light.groupby(["speech_class", "role"], observed=True).size().rename("rows").reset_index().to_csv(
        d / "class_by_role.csv", index=False
    )
    write_role_diagnostics(cfg, light)

    # Population of Chair speeches by stratum (era x dictionary tag x side of the floor).
    pop = (
        chair.groupby(["era", "d_tag", "target_side"], observed=True)
        .size()
        .rename("N")
        .reset_index()
        .rename(columns={"target_side": "side"})
    )
    full = pd.MultiIndex.from_product([eras, [0, 1], SIDES], names=["era", "d_tag", "side"])
    pop = pop.set_index(["era", "d_tag", "side"]).reindex(full, fill_value=0).reset_index()
    pop.to_csv(d / "population.csv", index=False)

    den = members.groupby(["era", "side"], observed=True).size().rename("members").reset_index()
    den = den.set_index(["era", "side"]).reindex(pd.MultiIndex.from_product([eras, SIDES]), fill_value=0)
    den.index.names = ["era", "side"]
    den = den.reset_index()
    den.to_csv(d / "denominators.csv", index=False)

    # Naive rates per 1,000 Member speeches, as a dictionary study would publish them.
    rows = []
    for construct, col in (("discipline", "d_tag"), ("timekeeping", "h_tag")):
        for era in eras:
            ce = chair[chair["era"] == era]
            m_all = int((members["era"] == era).sum())
            rows.append(_rate_row(construct, era, "All", int(ce[col].sum()), m_all, len(ce)))
            for side in SIDES:
                m = int(((members["era"] == era) & (members["side"] == side)).sum())
                cs = ce[ce["target_side"] == side]
                rows.append(_rate_row(construct, era, side, int(cs[col].sum()), m, len(cs)))
    naive = pd.DataFrame(rows)
    naive.to_csv(d / "naive_rates.csv", index=False)

    by_year = pd.DataFrame(
        {
            "members": members.groupby(members["date"].dt.year).size(),
            "chair_speeches": chair.groupby(chair["date"].dt.year).size(),
            "discipline_tagged": chair.groupby(chair["date"].dt.year)["d_tag"].sum().astype("int64"),
            "timekeeping_tagged": chair.groupby(chair["date"].dt.year)["h_tag"].sum().astype("int64"),
        }
    ).fillna(0)
    by_year.index.name = "year"
    by_year["discipline_rate"] = 1000 * by_year["discipline_tagged"] / by_year["members"]
    by_year["timekeeping_rate"] = 1000 * by_year["timekeeping_tagged"] / by_year["members"]
    by_year.reset_index().to_csv(d / "naive_by_year.csv", index=False)

    # Which categories make up the discipline tag, era by era.
    cat_rows = []
    hit_cols = [f"hits_{n}" for n in DISCIPLINE]
    n_cats = (chair[hit_cols] > 0).sum(axis=1)
    for era in eras:
        ce = chair[chair["era"] == era]
        tagged = int(ce["d_tag"].sum())
        nc = n_cats[ce.index]
        for c in CATEGORIES:
            h = ce[f"hits_{c.name}"]
            cat_rows.append(
                {
                    "era": era,
                    "category": c.name,
                    "family": c.family,
                    "speeches": int((h > 0).sum()),
                    "hits": int(h.sum()),
                    "share_of_discipline_tagged": (int((h > 0).sum()) / tagged)
                    if c.family == "discipline" and tagged
                    else np.nan,
                    "only_category": int(((h > 0) & (nc == 1)).sum()) if c.family == "discipline" else np.nan,
                }
            )
    pd.DataFrame(cat_rows).to_csv(d / "categories.csv", index=False)

    # The length filter as a classifier: what dropping short speeches removes.
    len_rows = []
    for era in eras:
        ce = chair[(chair["era"] == era) & (chair["d_tag"] == 1)]
        me = members[members["era"] == era]
        for t in LENGTH_THRESHOLDS:
            len_rows.append(
                {
                    "era": era,
                    "min_words": t,
                    "discipline_tagged_removed": float((ce["n_words"] < t).mean()) if len(ce) else np.nan,
                    "member_speeches_removed": float((me["n_words"] < t).mean()) if len(me) else np.nan,
                }
            )
    pd.DataFrame(len_rows).to_csv(d / "length_filter.csv", index=False)

    from .lexical import lexical_drift

    lex = lexical_drift(chair["norm"], chair["era"].astype(str), cfg)
    lex["top"].to_csv(d / "lexical_top.csv", index=False)
    lex["jsd"].to_csv(d / "lexical_jsd.csv")

    summary = {
        "package_version": __version__,
        "dictionary_version": DICTIONARY_VERSION,
        "built_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "source": fetch_manifest(cfg),
        "rows": int(len(light)),
        "first_date": str(light["date"].min().date()),
        "last_date": str(light["date"].max().date()),
        "sitting_days": int(light["date"].nunique()),
        "ids_parsed_share": float(light["k_parsed"].mean()),
        "roles": {k: int(v) for k, v in light["role"].value_counts().items()},
        "chair_by_source": {k: int(v) for k, v in chair["role_source"].value_counts().items()},
        "order_openers_classified_as_chair": float(
            (light.loc[light["opens_order"], "role"] == "chair").mean() if light["opens_order"].any() else np.nan
        ),
        "chair_by_kind": {k: int(v) for k, v in chair["chair_kind"].value_counts().items()},
        "chair_speeches": int(len(chair)),
        "discipline_tagged": int(chair["d_tag"].sum()),
        "timekeeping_tagged": int(chair["h_tag"].sum()),
        "member_speeches": int(len(members)),
        "chair_with_no_member_before": int((chair["target_side"] == "Other").sum()),
        "dictionary": describe(),
    }
    (d / "manifest.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def _rate_row(construct: str, era: str, side: str, tagged: int, members: int, chair: int) -> dict:
    return {
        "construct": construct,
        "era": era,
        "side": side,
        "tagged": tagged,
        "chair_speeches": chair,
        "members": members,
        "rate_per_1000": 1000 * tagged / members if members else np.nan,
    }


def write_role_diagnostics(cfg: Config, light: pd.DataFrame) -> None:
    """How role detection behaves year by year.

    A speech that opens with a bare "Order" is almost always the Chair's, so the share of
    such speeches classified as the Chair is a label-free check on the role detector's
    recall, and the unclassified ones show which printed names it misses. It is a check
    only: roles are never assigned from the text.
    """
    d = cfg.stage1_dir
    year = light["date"].dt.year.rename("year")
    chair = light["role"] == "chair"
    by_year = (
        light[chair].groupby([year[chair], light.loc[chair, "role_source"].astype(str)]).size().unstack(fill_value=0)
    )
    by_year["members"] = light[light["role"] == "member"].groupby(year).size()
    op = light["opens_order"]
    by_year["order_openers"] = light[op].groupby(year).size()
    by_year["order_openers_chair"] = light[op & chair].groupby(year).size()
    by_year = by_year.fillna(0).astype("int64")
    by_year["order_openers_chair_share"] = by_year["order_openers_chair"] / by_year["order_openers"].replace(0, np.nan)
    by_year.reset_index().to_csv(d / "role_by_year.csv", index=False)

    era_share = light[op].groupby("era")["role"].apply(lambda r: float((r == "chair").mean()))
    era_share.reindex(cfg.era_names).rename("order_openers_chair_share").reset_index().to_csv(
        d / "order_openers_by_era.csv", index=False
    )

    keys = ["name_norm", "display_as", "party_norm", "role", "role_source"]
    sp = light[light["party_norm"] == "Speaker"]
    (
        sp.groupby([sp["date"].dt.year.rename("year"), *[sp[k].astype(str) for k in keys]])
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["year", "rows"], ascending=[True, False])
        .to_csv(d / "party_speaker_rows.csv", index=False)
    )

    miss = light[op & ~chair]
    top = (
        miss.groupby([miss["date"].dt.year.rename("year"), *[miss[k].astype(str) for k in keys]])
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["year", "rows"], ascending=[True, False])
    )
    top.groupby("year").head(15).to_csv(d / "unclassified_order_openers.csv", index=False)
