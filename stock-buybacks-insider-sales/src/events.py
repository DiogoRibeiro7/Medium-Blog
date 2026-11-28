"""Classify 8-Ks into buyback announcements and earnings placebo dates."""
import html
import re

AUTH = re.compile(r"(authoriz|approv)\w*", re.I)
BUYBACK = re.compile(r"(share|stock)\s+(repurchase|buy-?back)|repurchase\s+(program|plan|authorization)|buy-?back", re.I)
BOARD = re.compile(r"board of directors|board", re.I)
TAG = re.compile(r"<[^>]+>")


def clean(text):
    text = TAG.sub(" ", text)
    return re.sub(r"\s+", " ", html.unescape(text))


def announces_buyback(text):
    """A new authorization: buyback wording near authorization wording, board mentioned."""
    t = clean(text)
    for m in BUYBACK.finditer(t):
        window = t[max(0, m.start() - 250): m.end() + 250]
        if AUTH.search(window) and BOARD.search(window):
            return True
    return False


def classify(items, text):
    """Return 'announcement', 'earnings_placebo' or None.

    items is the EDGAR item string, e.g. '2.02,9.01'.
    An announcement filed together with earnings (2.02) is flagged separately,
    because that is the calendar confound the analysis must handle.
    """
    its = {i.strip() for i in items.split(",") if i.strip()}
    has_bb = announces_buyback(text)
    if has_bb and its & {"8.01", "7.01", "2.02"}:
        return "announcement_with_earnings" if "2.02" in its else "announcement"
    if "2.02" in its and not has_bb:
        return "earnings_placebo"
    return None
