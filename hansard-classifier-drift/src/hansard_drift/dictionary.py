"""The measuring instrument: a fixed dictionary of Chair interventions.

It is written the way such dictionaries usually are, from general knowledge of how the
Chair speaks, before any of the validation labels were seen. It is frozen: the study
measures how well this one instrument works in each era, so it must not be tuned to the
labels afterwards. Changing a pattern means a new instrument and a new validation sample.

Every pattern runs on text that went through :func:`normalise`: lower case, straight
quotes, single spaces and "hon." written as "hon".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

DICTIONARY_VERSION = "1.0"


@dataclass(frozen=True)
class Category:
    name: str
    family: str  # "discipline" or "timekeeping"
    patterns: tuple[str, ...]
    description: str

    @property
    def regex(self) -> str:
        return "|".join(f"(?:{p})" for p in self.patterns)


# A bare call of "Order": the word on its own as a sentence or exclamation. Each call is
# counted, so "Order. Order." is two calls; the lookbehinds do not consume the full stop
# that ends the first call, which is the bug a naive pattern has.
_ORDER = (
    r"(?:^|(?<=[.!?:;] )|(?<=[.!?:;]\" )|(?<=order, )|(?<=order - ))"
    r"order(?=\s?[.!,;—-](?: |$|\"))"
)

CATEGORIES: tuple[Category, ...] = (
    Category("ORDER", "discipline", (_ORDER,), 'a bare call of "Order"'),
    Category(
        "SEAT",
        "discipline",
        (r"\bresume (?:his|her|their|your|my) seat", r"\bsit down\b"),
        "a Member told to sit down",
    ),
    Category(
        "WITHDRAW",
        "discipline",
        (r"\bwithdraws?\b(?! (?:immediately )?from the (?:house|chamber))",),
        "a Member asked to withdraw a remark",
    ),
    Category(
        "NAME",
        "discipline",
        (r"\bi (?:must |have to |shall )?name (?:the hon|the right hon|mr|mrs|ms|miss|dr|sir|dame)\b",),
        "a Member named",
    ),
    Category(
        "SUSPEND",
        "discipline",
        (
            r"\bwithdraw (?:immediately )?from the (?:house|chamber)",
            r"\bleave the (?:house|chamber)\b",
            r"\bstanding order (?:no\.? ?)?(?:42|43|44)\b",
        ),
        "a Member ordered out",
    ),
    Category(
        "NOISE",
        "discipline",
        (
            r"\bnois(?:e|y)\b",
            r"\bquiet(?:er|ly)?\b",
            r"\bcalm\b",
            r"\bshout(?:ing|s|ed)?\b",
            r"\bbarrack(?:ing|ed)?\b",
            r"\bheckl(?:e|es|ed|ing)\b",
            r"\bsedentary\b",
            r"\b(?:cannot|can't|could not|couldn't|unable to) hear\b",
        ),
        "the House told to be quiet",
    ),
    Category(
        "ADMONISH",
        "discipline",
        (
            r"\bunparliamentary\b",
            r"\bdisorderly\b",
            r"\bout of order\b",
            r"\bnot in order\b",
            r"\bintemperate\b",
            r"\bbehav(?:e|iour)\b",
            r"\bdecorum\b",
            r"\bnot (?:acceptable|tolerable)\b",
            r"\bwill not tolerate\b",
            r"\bmoderate (?:his|her|their|your) language\b",
        ),
        "conduct or language rebuked",
    ),
    Category(
        "NOT_POO",
        "discipline",
        (r"\b(?:not|no) (?:a )?point of order\b",),
        "a point of order ruled out",
    ),
    Category(
        "HURRY",
        "timekeeping",
        (
            r"\bbe (?:brief|short|quick|succinct)\b",
            r"\bbrevity\b",
            r"\b(?:brief|short) (?:questions?|answers?|interventions?|contributions?|speech(?:es)?)\b",
            r"\bsuccinct\b",
            r"\bpithy\b",
            r"\btime is (?:short|limited|pressing|very short|against us)\b",
            r"\bas many (?:hon )?members as (?:possible|i can)\b",
            r"\b(?:great deal|lot) of interest\b",
            r"\bmany (?:hon )?members (?:wish|want|are seeking|seeking|hoping)\b",
            r"\btime limit\b",
        ),
        "the House told to hurry",
    ),
)

DISCIPLINE = tuple(c.name for c in CATEGORIES if c.family == "discipline")
TIMEKEEPING = tuple(c.name for c in CATEGORIES if c.family == "timekeeping")

_COMPILED = {c.name: re.compile(c.regex) for c in CATEGORIES}

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-"})
_HON = re.compile(r"\bhon\.")
_SPACE = re.compile(r"\s+")


def normalise(text: str | None) -> str:
    if not text:
        return ""
    text = text.translate(_QUOTES).lower()
    text = _HON.sub("hon", text)
    return _SPACE.sub(" ", text).strip()


def tag_text(text: str) -> dict[str, int]:
    """Hits per category in one already-normalised text."""
    return {name: len(rx.findall(text)) for name, rx in _COMPILED.items()}


def tag_series(norm: pd.Series) -> pd.DataFrame:
    """Hits per category for a series of normalised texts, plus the two family tags."""
    norm = norm.fillna("").astype(str)
    out = pd.DataFrame(index=norm.index)
    for c in CATEGORIES:
        out[f"hits_{c.name}"] = norm.str.count(c.regex).astype("int64")
    out["d_tag"] = (out[[f"hits_{n}" for n in DISCIPLINE]].sum(axis=1) > 0).astype("int64")
    out["h_tag"] = (out[[f"hits_{n}" for n in TIMEKEEPING]].sum(axis=1) > 0).astype("int64")
    return out


def describe() -> list[dict]:
    return [
        {"category": c.name, "family": c.family, "description": c.description, "patterns": list(c.patterns)}
        for c in CATEGORIES
    ]
