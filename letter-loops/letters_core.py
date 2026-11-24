"""Greedy letter chains: start at a letter, move to the letter most likely to follow it, repeat.

The rule turns a table of letter-pair counts into a map from each symbol to one other symbol (a
functional graph). On a finite set every such map ends in loops; this module measures which loops,
how many, and how the answer moves with the choices made before any counting is done:

- the alphabet: the standard letters of each language (CLDR exemplar characters) or the same letters
  with accents and positional forms folded away;
- word boundaries: pairs inside words only, as in Norvig's table, or with a boundary symbol so that
  a chain can end a word and start the next one;
- the weighting: every word as often as it is used (running text) or every word once (a dictionary).

Word frequencies come from wordfreq 3.1.1; alphabets from CLDR 49 (see fetch_data.py).
"""

import importlib.metadata
import math
import unicodedata as ud
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLDR_DIR = HERE / "data" / "cldr"
WORDFREQ_VERSION = "3.1.1"
BOUNDARY = "_"  # the word-boundary symbol; printed as a space in generated text
TYPE_THRESHOLD = 1e-6  # a dictionary counts each word with frequency >= 1e-6 once
ALPHABETS = ("standard", "folded")
WEIGHTINGS = ("text", "dictionary")

# wordfreq code -> (English name, CLDR locale whose exemplar letters define the alphabet, script).
# Every wordfreq language written in an alphabet or an abjad: 36 of its 42. Left out: Bengali, Hindi
# and Tamil (abugidas, where vowels are marks on consonants), Chinese and Japanese (not alphabetic)
# and Korean (an alphabet written in syllable blocks; letters would first have to be decomposed).
LANGUAGES = {
    "en": ("English", "en", "Latin"),
    "fr": ("French", "fr", "Latin"),
    "de": ("German", "de", "Latin"),
    "es": ("Spanish", "es", "Latin"),
    "it": ("Italian", "it", "Latin"),
    "pt": ("Portuguese", "pt", "Latin"),
    "ca": ("Catalan", "ca", "Latin"),
    "ro": ("Romanian", "ro", "Latin"),
    "nl": ("Dutch", "nl", "Latin"),
    "da": ("Danish", "da", "Latin"),
    "nb": ("Norwegian Bokmål", "nb", "Latin"),
    "sv": ("Swedish", "sv", "Latin"),
    "is": ("Icelandic", "is", "Latin"),
    "fi": ("Finnish", "fi", "Latin"),
    "hu": ("Hungarian", "hu", "Latin"),
    "pl": ("Polish", "pl", "Latin"),
    "cs": ("Czech", "cs", "Latin"),
    "sk": ("Slovak", "sk", "Latin"),
    "sl": ("Slovenian", "sl", "Latin"),
    "sh": ("Serbo-Croatian", "hr", "Latin"),
    "lt": ("Lithuanian", "lt", "Latin"),
    "lv": ("Latvian", "lv", "Latin"),
    "tr": ("Turkish", "tr", "Latin"),
    "id": ("Indonesian", "id", "Latin"),
    "ms": ("Malay", "ms", "Latin"),
    "fil": ("Filipino", "fil", "Latin"),
    "vi": ("Vietnamese", "vi", "Latin"),
    "el": ("Greek", "el", "Greek"),
    "ru": ("Russian", "ru", "Cyrillic"),
    "uk": ("Ukrainian", "uk", "Cyrillic"),
    "bg": ("Bulgarian", "bg", "Cyrillic"),
    "mk": ("Macedonian", "mk", "Cyrillic"),
    "he": ("Hebrew", "he", "Hebrew"),
    "ar": ("Arabic", "ar", "Arabic"),
    "fa": ("Persian", "fa", "Arabic"),
    "ur": ("Urdu", "ur", "Arabic"),
}

# Folding: decompose, drop every combining mark, then merge the letters below. The point is not that
# folding is right (a Finn will say ä is not a), but that it is a choice somebody has to make.
FOLDS = {
    "ß": "ss",
    "æ": "ae",
    "œ": "oe",
    "ø": "o",
    "ł": "l",
    "đ": "d",
    "ð": "d",
    "þ": "th",
    "ı": "i",
    "ς": "σ",  # Greek final sigma
    "ך": "כ",  # Hebrew final forms
    "ם": "מ",
    "ן": "נ",
    "ף": "פ",
    "ץ": "צ",
    "ة": "ه",  # Arabic taa marbuta and alif maqsura
    "ى": "ي",
}
FOLDS_PERSO_ARABIC = {
    "ي": "ی",
    "ى": "ی",
    "ك": "ک",
}  # Persian and Urdu write yeh and kaf this way

# Norvig (2013), "English Letter Frequency Counts: Mayzner Revisited", top 50 bigrams in billions,
# counted inside words in the Google Books 1-grams (743.8 billion words). https://norvig.com/mayzner.html
NORVIG_TOP50 = {
    "th": 100.3, "he": 86.7, "in": 68.6, "er": 57.8, "an": 56.0, "re": 52.3, "on": 49.6, "at": 41.9,
    "en": 41.0, "nd": 38.1, "ti": 37.9, "es": 37.8, "or": 36.0, "te": 34.0, "of": 33.1, "ed": 32.9,
    "is": 31.8, "it": 31.7, "al": 30.7, "ar": 30.3, "st": 29.7, "to": 29.4, "nt": 29.4, "ng": 26.9,
    "se": 26.3, "ha": 26.1, "as": 24.6, "ou": 24.5, "io": 23.5, "le": 23.4, "ve": 23.3, "co": 22.4,
    "me": 22.4, "de": 21.6, "hi": 21.5, "ri": 20.5, "ro": 20.5, "ic": 19.7, "ne": 19.5, "ea": 19.4,
    "ra": 19.3, "ce": 18.4, "li": 17.6, "ch": 16.9, "ll": 16.3, "be": 16.2, "ma": 15.9, "si": 15.5,
    "om": 15.4, "ur": 15.3,
}  # fmt: skip
NORVIG_LETTERS = 3_563_505_777_820
NORVIG_WORDS = 743_842_922_321


# ---------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------


def wordfreq_version():
    return importlib.metadata.version("wordfreq")


def wordlist_path(code):
    """Path of the list wordfreq uses for get_frequency_dict(code, 'best'): large if it exists."""
    import wordfreq

    large = wordfreq.available_languages("large")
    return Path(large[code] if code in large else wordfreq.available_languages("small")[code])


def wordlist_kind(code):
    return wordlist_path(code).name.split("_")[0]


def word_frequencies(code):
    """Word -> frequency (share of word tokens) from wordfreq's best list for the language."""
    import wordfreq

    return wordfreq.get_frequency_dict(code, wordlist="best")


def parse_exemplars(text):
    """Letters of a CLDR exemplar set such as '[a ą b {ng} c]', in order, lower case, NFC.

    Multi-letter sequences in braces are skipped (each of their letters is listed on its own where
    the language uses it alone), and so are marks, escapes and anything that is not a letter.
    """
    body = text.strip()[1:-1]
    out, depth, i = [], 0, 0
    while i < len(body):
        ch = body[i]
        if ch == "\\":  # an escape such as ‎
            i += 6 if body[i + 1 : i + 2] == "u" else 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif depth == 0:
            c = ud.normalize("NFC", ch.lower())
            if len(c) == 1 and ud.category(c).startswith("L") and c not in out:
                out.append(c)
        i += 1
    return out


def standard_alphabet(code, cldr_dir=CLDR_DIR):
    """The language's standard letters: CLDR main exemplar characters for its locale."""
    import json

    loc = LANGUAGES[code][1]
    data = json.loads((Path(cldr_dir) / f"{loc}.json").read_text(encoding="utf-8"))
    return parse_exemplars(data["main"][loc]["characters"]["exemplarCharacters"])


def fold(word, code):
    """Strip accents and merge positional forms (see FOLDS)."""
    decomposed = ud.normalize("NFD", word)
    base = ud.normalize("NFC", "".join(c for c in decomposed if not ud.category(c).startswith("M")))
    table = dict(FOLDS)
    if code in ("fa", "ur"):
        table.update(FOLDS_PERSO_ARABIC)
    return "".join(table.get(c, c) for c in base)


def folded_alphabet(code, standard):
    """Image of the standard letters under fold, in first-appearance order."""
    out = []
    for c in standard:
        for d in fold(c, code):
            if d not in out:
                out.append(d)
    return out


# ---------------------------------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------------------------------


@dataclass
class Counts:
    """Letter counts for one language, alphabet and weighting. Index 0 is the boundary symbol."""

    symbols: list
    pairs: np.ndarray  # pairs[i, j]: weight of letter i followed by letter j inside a word
    initial: np.ndarray  # weight of words starting with each letter
    final: np.ndarray  # weight of words ending with each letter
    letters: np.ndarray  # weight of each letter (all positions)
    kept: float  # share of the list's token mass in words written only with the alphabet's letters
    words: int  # number of words kept


def count_language(code, freqs=None, cldr_dir=CLDR_DIR):
    """Counts for every alphabet x weighting combination, in one pass over the word list.

    Running text weights each word by its frequency; the dictionary weights each word with
    frequency >= TYPE_THRESHOLD by 1 (and drops the rest). Words containing anything outside the
    standard alphabet (digits, apostrophes, foreign letters) are dropped in every combination.
    """
    freqs = word_frequencies(code) if freqs is None else freqs
    std = standard_alphabet(code, cldr_dir)
    alph = {"standard": std, "folded": folded_alphabet(code, std)}
    index = {a: {c: k + 1 for k, c in enumerate(alph[a])} for a in ALPHABETS}
    out = {}
    for a in ALPHABETS:
        n = len(alph[a]) + 1
        for w in WEIGHTINGS:
            out[(a, w)] = Counts(
                [BOUNDARY] + alph[a],
                np.zeros((n, n)),
                np.zeros(n),
                np.zeros(n),
                np.zeros(n),
                0.0,
                0,
            )
    total = sum(freqs.values())
    std_set = set(std)
    for word, f in freqs.items():
        word = ud.normalize("NFC", word)
        if not word or any(c not in std_set for c in word):
            continue
        forms = {"standard": word, "folded": fold(word, code)}
        weights = {"text": f, "dictionary": 1.0 if f >= TYPE_THRESHOLD else 0.0}
        for a in ALPHABETS:
            ids = [index[a][c] for c in forms[a]]
            for w in WEIGHTINGS:
                wt = weights[w]
                if wt == 0.0:
                    continue
                c = out[(a, w)]
                c.initial[ids[0]] += wt
                c.final[ids[-1]] += wt
                for x, y in zip(ids, ids[1:], strict=False):
                    c.pairs[x, y] += wt
                for x in ids:
                    c.letters[x] += wt
                c.kept += f
                c.words += 1
    for key, c in out.items():
        c.kept /= total
        out[key] = drop_unseen(c)
    return out


def unseen_letters(code, counts):
    """Standard letters that never occur in the list (wordfreq case-folds ß to ss and ς to σ)."""
    seen = set(counts[("standard", "text")].symbols)
    return [s for s in standard_alphabet(code) if s not in seen]


def drop_unseen(c):
    """Remove letters with zero weight: they are in the alphabet but never in the data."""
    keep = [0] + [i for i in range(1, len(c.symbols)) if c.letters[i] > 0]
    k = np.array(keep)
    return Counts(
        [c.symbols[i] for i in keep],
        c.pairs[np.ix_(k, k)],
        c.initial[k],
        c.final[k],
        c.letters[k],
        c.kept,
        c.words,
    )


def transitions(c, boundary):
    """(symbols, matrix) of the chain: inside words only, or with the boundary symbol at index 0."""
    if boundary:
        m = c.pairs.copy()
        m[0, :] = c.initial
        m[:, 0] = c.final
        m[0, 0] = 0.0
        return list(c.symbols), m
    return list(c.symbols[1:]), c.pairs[1:, 1:].copy()


def letter_weights(c, boundary):
    """Weights used to order symbols (most frequent first in printed loops)."""
    return np.concatenate([[c.initial.sum()], c.letters[1:]]) if boundary else c.letters[1:].copy()


# ---------------------------------------------------------------------------------------------------
# The greedy map and its structure
# ---------------------------------------------------------------------------------------------------


def greedy_map(m, tiebreak=None):
    """f[i] = most likely successor of i; -1 where a row is empty.

    Ties (which happen with dictionary counts, where every word counts 1) go to the successor with
    the larger tiebreak value (running-text counts), then to the lower index.
    """
    if tiebreak is None:
        f = np.argmax(m, axis=1)
    else:
        top = m == m.max(axis=1, keepdims=True)
        f = np.argmax(np.where(top, tiebreak + 1.0, -1.0), axis=1)
    f[m.sum(axis=1) == 0] = -1
    return f


def tied_rows(m):
    """Rows whose maximum is shared by two or more successors."""
    mx = m.max(axis=1, keepdims=True)
    return np.flatnonzero(((m == mx) & (mx > 0)).sum(axis=1) > 1)


def chain(counts, code, alphabet, weighting, boundary):
    """(symbols, matrix, greedy map, structure, symbol weights) for one language and set of choices."""
    c = counts[(alphabet, weighting)]
    sym, m = transitions(c, boundary)
    tb = None
    if weighting == "dictionary":
        ct = counts[(alphabet, "text")]
        pos = {s: k for k, s in enumerate(ct.symbols if boundary else ct.symbols[1:])}
        _, mt = transitions(ct, boundary)
        k = np.array([pos[s] for s in sym])
        tb = mt[np.ix_(k, k)]
    f = greedy_map(m, tb)
    order = letter_weights(c, boundary)
    return sym, m, f, analyse(f, order, 0 if boundary else None), order


def rotate(cycle, order, anchor=None):
    """Start a cycle at the anchor (the boundary symbol) if present, else at its heaviest symbol."""
    k = cycle.index(anchor) if anchor in cycle else int(np.argmax([order[i] for i in cycle]))
    return tuple(cycle[k:] + cycle[:k])


@dataclass
class Structure:
    """Loops, dead ends and basins of a greedy map."""

    loops: list  # tuples of node indices, rotated by rotate()
    dead_ends: list  # nodes with no successor (letters never followed by another letter)
    basin: np.ndarray  # basin[i] = index of i's attractor in attractors()
    depth: np.ndarray  # steps from i to its loop or dead end
    attractors: list  # ("loop", tuple) or ("end", node)

    def basin_sizes(self):
        return np.bincount(self.basin, minlength=len(self.attractors))

    def n_attractors(self):
        return len(self.attractors)


def analyse(f, order=None, anchor=None):
    """Follow f from every node: which loop or dead end it ends in, and after how many steps.

    order (symbol weights) and anchor (index of the boundary symbol, if any) only fix where each
    loop is written to start.
    """
    n = len(f)
    order = np.zeros(n) if order is None else order
    att_index, attractors = {}, []
    basin = np.full(n, -1)
    depth = np.full(n, -1)
    for start in range(n):
        path, pos, x = [], {}, start
        while basin[x] < 0 and x not in pos and f[x] >= 0:
            pos[x] = len(path)
            path.append(x)
            x = f[x]
        if basin[x] >= 0:  # reached a node already labelled
            b, d0 = basin[x], depth[x]
        elif f[x] < 0:  # a dead end
            key = ("end", int(x))
            if key not in att_index:
                att_index[key] = len(attractors)
                attractors.append(key)
            b, d0 = att_index[key], 0
            basin[x], depth[x] = b, 0
        else:  # closed a new loop at x
            cyc = rotate([int(v) for v in path[pos[x] :]], order, anchor)
            key = ("loop", cyc)
            att_index[key] = len(attractors)
            attractors.append(key)
            b, d0 = att_index[key], 0
            for v in cyc:
                basin[v], depth[v] = b, 0
            path = path[: pos[x]]
        for k, v in enumerate(reversed(path)):
            basin[v], depth[v] = b, d0 + k + 1
    loops = [a[1] for a in attractors if a[0] == "loop"]
    ends = [a[1] for a in attractors if a[0] == "end"]
    return Structure(loops, ends, basin, depth, attractors)


def spell(nodes, symbols):
    return "".join(symbols[i] for i in nodes)


def main_attractor(s):
    """The attractor with the largest basin (ties: first found)."""
    return s.attractors[int(np.argmax(s.basin_sizes()))]


def greedy_word(f):
    """Letters visited from the boundary until the chain returns to it; None if it never does."""
    out, x, seen = [], f[0], {0}
    while x > 0 and x not in seen:
        seen.add(x)
        out.append(int(x))
        x = f[x]
    return out if x == 0 else None


# ---------------------------------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------------------------------


def p_one_loop_uniform(n):
    """Probability that a uniformly random map of n points has a single loop (Katz 1955), exact."""
    s = sum(Fraction(math.perm(n, k), n**k) for k in range(1, n + 1))
    return s / n


def shuffled_targets(f, rng, draws, allow_self=True):
    """Same targets, reassigned at random to the nodes: number of attractors in each draw.

    Nodes with no successor stay dead ends. The shuffle keeps how many arrows point at each letter
    and nothing else, so it measures how much of the structure the popularity of the targets explains.
    With allow_self=False, draws in which a letter points at itself are rejected.
    """
    src = np.flatnonzero(f >= 0)
    tgt = f[src]
    out = np.empty(draws, dtype=int)
    g = f.copy()
    d = 0
    while d < draws:
        g[src] = rng.permutation(tgt)
        if not allow_self and np.any(g[src] == src):
            continue
        out[d] = analyse(g).n_attractors()
        d += 1
    return out


def classify_attractors(s):
    """Label each attractor: 'loop' (two or more letters), 'fixed point' (a letter whose likeliest
    successor is itself) or 'dead end' (a letter never followed by another)."""
    out = []
    for kind, val in s.attractors:
        if kind == "end":
            out.append("dead end")
        else:
            out.append("fixed point" if len(val) == 1 else "loop")
    return out


# ---------------------------------------------------------------------------------------------------
# Robustness to the corpus
# ---------------------------------------------------------------------------------------------------


def margins(m):
    """Ratio of the top successor's count to the runner-up's, per row (inf if there is no runner-up)."""
    s = -np.sort(-m, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = s[:, 0] / s[:, 1]
    r[s[:, 0] == 0] = np.nan
    return r


def norvig_comparison(c):
    """Compare English counts with Norvig's top 50 bigrams (Google Books).

    Returns the shares of the 50 pairs in both sources (each normalised over the 50), the SD of the
    log ratio, and, for every letter whose arrow Norvig's table settles (its top pair beats every
    unlisted pair, all of which are at most the 50th count), the two arrows.
    """
    sym = c.symbols
    idx = {s: k for k, s in enumerate(sym)}
    keys = list(NORVIG_TOP50)
    nv = np.array([NORVIG_TOP50[k] for k in keys])
    wf = np.array([c.pairs[idx[k[0]], idx[k[1]]] for k in keys])
    nv_s, wf_s = nv / nv.sum(), wf / wf.sum()
    lr = np.log(wf_s / nv_s)
    floor = min(NORVIG_TOP50.values())
    settled = {}
    for a in "abcdefghijklmnopqrstuvwxyz":
        listed = {k[1]: v for k, v in NORVIG_TOP50.items() if k[0] == a}
        if listed:
            b, v = max(listed.items(), key=lambda kv: kv[1])
            if v > floor:
                row = c.pairs[idx[a], 1:]
                settled[a] = (b, sym[1 + int(np.argmax(row))])
    return keys, nv_s, wf_s, float(np.std(lr, ddof=1)), settled


def perturbed(m, sigma, rng, draws, order=None, anchor=None):
    """Multiply every nonzero count by exp(sigma * Z) and redo the greedy map, draws times.

    Returns, per draw, the set of loops (frozenset of rotated tuples, comparable with loop_set of
    the unperturbed structure when the same order is used) and the number of attractors.
    """
    nz = m > 0
    loops, n_att = [], np.empty(draws, dtype=int)
    for d in range(draws):
        noisy = np.where(nz, m * np.exp(sigma * rng.standard_normal(m.shape)), 0.0)
        s = analyse(greedy_map(noisy), order, anchor)
        loops.append(loop_set(s))
        n_att[d] = s.n_attractors()
    return loops, n_att


def loop_set(s):
    return frozenset(s.loops)


# ---------------------------------------------------------------------------------------------------
# Randomness instead of the maximum
# ---------------------------------------------------------------------------------------------------


def tempered(m, temperature):
    """Rows of m raised to 1/T and normalised: T = 1 samples the counts, T -> 0 is the greedy rule."""
    mx = m.max(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(mx > 0, m / mx, 0.0)
    p = r ** (1.0 / temperature)
    return p / p.sum(axis=1, keepdims=True)


def stationary(p):
    """Stationary distribution of an irreducible chain, by a linear solve; also the residual."""
    n = len(p)
    a = p.T - np.eye(n)
    a[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    pi = np.linalg.solve(a, b)
    return pi, float(np.abs(pi @ p - pi).max())


def is_irreducible(m):
    """True if every symbol can reach every other through nonzero counts."""
    n = len(m)
    reach = (m > 0) | np.eye(n, dtype=bool)
    for _ in range(int(np.ceil(np.log2(n))) + 1):
        reach = reach | ((reach.astype(int) @ reach.astype(int)) > 0)
    return bool(reach.all())


def generate(p, symbols, rng, length, start=0, greedy=False):
    """Text from the chain, starting after a boundary; the boundary is written as a space."""
    out, x = [], start
    cum = np.cumsum(p, axis=1)
    for _ in range(length):
        x = int(np.argmax(p[x])) if greedy else int(np.searchsorted(cum[x], rng.random() * cum[x, -1]))
        out.append(" " if symbols[x] == BOUNDARY else symbols[x])
    return "".join(out)


def entropy_rate(p, pi):
    """Bits per symbol of the chain in its stationary state."""
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log2(p), 0.0).sum(axis=1)
    return float(pi @ h)
