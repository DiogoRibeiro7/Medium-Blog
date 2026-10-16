"""Who is speaking: the Chair, a Member, the House, or nobody.

Role detection is itself a classifier. The validation sample is drawn from the speeches
this module calls the Chair's, so its errors are not measured by the hand labels; the
build writes an audit of every printed name it classified so they can be checked by eye.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import Config

_PUNCT = re.compile(r"[.,:]")
_SPACE = re.compile(r"\s+")

# Printed titles of the occupant of the Chair, after :func:`normalise_name`.
_SPEAKER = re.compile(r"^(?:mr|madam|madame|mrs|ms) speaker\b(?!\s*-?\s*elect)|^the speaker\b(?!\s*-?\s*elect)")
_DEPUTY = re.compile(
    r"^(?:mr|madam|madame|mrs|ms) (?:first |second )?deputy speaker\b|^the (?:first |second )?deputy speaker\b"
)
_COMMITTEE = re.compile(
    r"^(?:the |mr |madam |madame |mrs |ms )?(?:temporary |first deputy |second deputy |deputy )?"
    r"chair(?:man|woman|person)?\b(?! of the [a-z ]*committee)"
)
_COLLECTIVE = re.compile(
    r"^(?:(?:several|some|many|an|other|opposition|labour|conservative) )?(?:right )?hon (?:members?|friends?|gentlemen|ladies)\b"
    r"|^members\b|^several members\b"
)
_ELECT = re.compile(r"speaker\s*-?\s*elect")

_POST_SPEAKER = "speaker of the house of commons"
_POST_DEPUTY = ("deputy speaker", "chairman of ways and means", "deputy chairman of ways and means")


def normalise_name(name) -> str:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    text = str(name).lower().replace("hon.", "hon")
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def _posts_text(posts) -> str:
    if posts is None:
        return ""
    if isinstance(posts, float) and np.isnan(posts):
        return ""
    if isinstance(posts, str):
        return posts.lower()
    try:
        return " | ".join(str(p) for p in posts).lower()
    except TypeError:
        return str(posts).lower()


def classify_roles(
    df: pd.DataFrame, speaker_surname: pd.Series | None = None, roster_hit: pd.Series | None = None
) -> pd.DataFrame:
    """Add ``role``, ``chair_kind`` and ``role_source`` to a frame of raw rows.

    The Chair is recognised, in order of trust, by
      1. the printed title ("Mr. Speaker", "Madam Deputy Speaker (...)", "The Chairman");
      2. a parliamentary post held at the time (Speaker, Deputy Speaker, Ways and Means);
      3. party "Speaker" on a row whose name is the reigning Speaker's. Party alone is not
         enough: older rows carry party "Speaker" on other Members;
      4. a Deputy Speaker on the roster in the config, inside their term, for the years
         when the record prints them under their own name with no title and no post.
    The text of the speech is never used, so the frame the validation sample is drawn
    from does not depend on the dictionary being validated.

    role: chair | member | floor | procedural | other
    role_source: title | post | party | roster | none
    """
    name = df["speakername"].map(normalise_name)
    posts = df["parliamentary_posts"].map(_posts_text) if "parliamentary_posts" in df else pd.Series("", index=df.index)
    cls = df["speech_class"].fillna("").astype(str)

    is_speaker = name.str.contains(_SPEAKER)
    is_deputy = name.str.contains(_DEPUTY)
    is_committee = name.str.contains(_COMMITTEE)
    is_title = is_speaker | is_deputy | is_committee
    is_elect = name.str.contains(_ELECT)

    post_speaker = posts.str.contains(_POST_SPEAKER, regex=False)
    post_deputy = pd.Series(False, index=df.index)
    for p in _POST_DEPUTY:
        post_deputy |= posts.str.contains(p, regex=False)
    by_post = ~is_title & ~is_elect & (post_speaker | post_deputy) & (cls == "Speech")

    by_party = pd.Series(False, index=df.index)
    if speaker_surname is not None and "party" in df:
        party_speaker = df["party"].map(normalise_party) == "Speaker"
        display = df["display_as"].fillna("").astype(str).str.lower() if "display_as" in df else name
        surname = speaker_surname.reindex(df.index).fillna("").astype(str)
        has = surname != ""
        named = pd.Series(
            [bool(s) and (s in n or s in d) for s, n, d in zip(surname, name, display, strict=True)], index=df.index
        )
        by_party = ~is_title & ~is_elect & ~by_post & party_speaker & has & named & (cls == "Speech")

    by_roster = pd.Series(False, index=df.index)
    if roster_hit is not None:
        by_roster = ~is_title & ~is_elect & ~by_post & ~by_party & roster_hit.reindex(df.index).fillna(False)
        by_roster &= cls == "Speech"

    collective = name.str.contains(_COLLECTIVE)

    role = np.select(
        [
            is_title & ~is_elect,
            by_post,
            by_party,
            by_roster,
            collective,
            cls == "Procedural",
            (cls == "Speech") & (name != ""),
        ],
        ["chair", "chair", "chair", "chair", "floor", "procedural", "member"],
        default="other",
    )
    kind = np.select(
        [is_speaker & ~is_elect, is_deputy, is_committee, by_post & post_speaker, by_post, by_party, by_roster],
        ["speaker", "deputy", "committee", "speaker", "deputy", "speaker", "deputy"],
        default="",
    )
    source = np.select(
        [is_title & ~is_elect, by_post, by_party, by_roster], ["title", "post", "party", "roster"], default="none"
    )
    out = df.copy()
    out["name_norm"] = name
    out["role"] = role
    out["chair_kind"] = np.where(role == "chair", kind, "")
    out["role_source"] = source
    return out


def normalise_party(party) -> str:
    if party is None or (isinstance(party, float) and np.isnan(party)):
        return "Unknown"
    p = str(party).strip()
    low = p.lower()
    if not p or low in {"unknown", "none", "nan"}:
        return "Unknown"
    if low.startswith("labour"):
        return "Labour"
    if low.startswith("conservative"):
        return "Conservative"
    if low.startswith("liberal democrat") or low.startswith("liberal-democrat") or low == "libdem":
        return "Liberal Democrat"
    if low == "speaker" or low.startswith("speaker"):
        return "Speaker"
    return p


def assign_era(dates: pd.Series, cfg: Config) -> pd.Series:
    d = pd.to_datetime(dates).dt.date
    era = pd.Series(pd.NA, index=dates.index, dtype="object")
    for e in cfg.eras:
        era[(d >= e.start) & (d < e.end)] = e.name
    return era


def roster_match(display_as: pd.Series, dates: pd.Series, cfg: Config) -> pd.Series:
    """True where the row's person is a rostered Deputy Speaker inside their term."""
    name = display_as.fillna("").astype(str).str.strip().str.lower()
    d = pd.to_datetime(dates).dt.date
    hit = pd.Series(False, index=display_as.index)
    for r in cfg.roster:
        hit |= (name == r.name.lower()) & (d >= r.start) & (d <= r.end)
    return hit


def speaker_surnames(dates: pd.Series, cfg: Config) -> pd.Series:
    """The reigning Speaker's surname, lower case, for each date ("" outside the eras)."""
    d = pd.to_datetime(dates).dt.date
    out = pd.Series("", index=dates.index, dtype="object")
    for e in cfg.eras:
        out[(d >= e.start) & (d < e.end)] = e.speaker.split()[-1].lower()
    return out


def assign_side(party: pd.Series, dates: pd.Series, cfg: Config) -> pd.Series:
    """Government | Opposition | Other, for Members only."""
    d = pd.to_datetime(dates).dt.date
    gov = pd.Series(False, index=party.index)
    for g in cfg.governments:
        gov |= party.isin(g.parties) & (d >= g.start) & (d < g.end)
    other = party.isin(["Unknown", "Speaker"])
    return pd.Series(np.select([gov, other], ["Government", "Other"], default="Opposition"), index=party.index)


_ID = re.compile(r"/(\d{4}-\d{2}-\d{2})([a-z]?)\.(\d+)\.(\d+)$")


def order_keys(ids: pd.Series, colnum: pd.Series | None = None) -> pd.DataFrame:
    """Sitting order within a day, from the TheyWorkForYou id: sitting letter, column, paragraph."""
    parts = ids.astype(str).str.extract(_ID)
    letter = parts[1].fillna("").map(lambda s: ord(s) - ord("a") + 1 if s else 0)
    col = pd.to_numeric(parts[2], errors="coerce")
    if colnum is not None:
        col = col.fillna(pd.to_numeric(colnum, errors="coerce"))
    para = pd.to_numeric(parts[3], errors="coerce")
    return pd.DataFrame(
        {
            "k_letter": letter.astype("int64"),
            "k_col": col.fillna(-1).astype("int64"),
            "k_para": para.fillna(-1).astype("int64"),
            "k_parsed": parts[0].notna(),
        },
        index=ids.index,
    )
