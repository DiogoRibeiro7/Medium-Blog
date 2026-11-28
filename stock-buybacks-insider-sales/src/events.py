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


NEW = re.compile(r"\bnew\b|additional|increas|up to|replac|extend|renew|expand", re.I)
AMOUNT = re.compile(r"\$\s?[\d.,]+\s*(billion|million|bn|mm|b|m)?", re.I)
EXISTING = re.compile(r"remaining (share )?(repurchase )?authoriz|remained available|repurchased", re.I)


def find_authorization(text):
    """Return a snippet when the text announces a NEW or enlarged authorization, else None.

    Needs, in one window around buyback wording: authorization wording, a board mention,
    a dollar amount and new/additional/increase wording. Windows that only report past
    purchases or remaining capacity are rejected.
    """
    t = clean(text)
    for m in BUYBACK.finditer(t):
        window = t[max(0, m.start() - 250): m.end() + 250]
        if not (AUTH.search(window) and BOARD.search(window) and AMOUNT.search(window) and NEW.search(window)):
            continue
        if EXISTING.search(window) and not re.search(r"authorized (a|an) (new|additional)|new .{0,40}authoriz|additional \$", window, re.I):
            continue
        return window.strip()
    return None


def announces_buyback(text):
    return find_authorization(text) is not None


def classify(items, text):
    """Return (kind, snippet). kind is 'announcement', 'announcement_with_earnings',
    'earnings_placebo' or None. An announcement filed together with earnings (2.02) is kept
    apart, because that is the calendar confound the analysis must handle."""
    its = {i.strip() for i in items.split(",") if i.strip()}
    snippet = find_authorization(text)
    if snippet and its & {"8.01", "7.01", "2.02"}:
        return ("announcement_with_earnings" if "2.02" in its else "announcement"), snippet
    if "2.02" in its and not snippet:
        return "earnings_placebo", ""
    return None, ""
