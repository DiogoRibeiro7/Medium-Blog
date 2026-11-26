"""Decoding rules and the loops they make, without any deep-learning dependency.

A language model enters here only through a small interface (see Model): give it a batch of
token sequences and it returns next-token logits; give it one token per sequence and it returns
the next logits. Everything else lives in numpy so that it can be tested on toy models:

- the prompts: passages from 18 public-domain books (the NLTK copy of Project Gutenberg);
- the decoding rules: greedy, temperature sampling, nucleus (top-p) sampling;
- the decoding loop, which records each chosen token's probability and the top-1/top-2 margin;
- loop detection: does a continuation end in a block of tokens that repeats, from where, and with
  what period;
- word n-gram chains, the finite-memory end of the comparison.
"""

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

BOOKS = [
    "austen-emma", "austen-persuasion", "austen-sense", "bible-kjv", "blake-poems", "bryant-stories",
    "burgess-busterbrown", "carroll-alice", "chesterton-ball", "chesterton-brown", "chesterton-thursday",
    "edgeworth-parents", "melville-moby_dick", "milton-paradise", "shakespeare-caesar", "shakespeare-hamlet",
    "shakespeare-macbeth", "whitman-leaves",
]  # fmt: skip
PROMPTS_PER_BOOK = 8
PROMPT_WORDS = 40
REFERENCE_WORDS = 260  # human continuation, cut to N_NEW tokens per tokenizer
N_NEW = 256
SKIP_FRONT = 0.05  # skip title pages and tables of contents

# The decoding rules compared. Temperature T rescales the logits by 1/T before the softmax; T -> 0 is
# greedy. Nucleus sampling keeps the smallest set of tokens whose probabilities add up to p.
RULES = {
    "greedy": {"kind": "greedy"},
    "T0.25": {"kind": "temperature", "T": 0.25},
    "T0.5": {"kind": "temperature", "T": 0.5},
    "T0.75": {"kind": "temperature", "T": 0.75},
    "T1": {"kind": "temperature", "T": 1.0},
    "top-p0.9": {"kind": "top_p", "p": 0.9},
}

# Loop criterion: the continuation ends in a block of `period` tokens that repeats back to back,
# covering at least MIN_REPEATS full periods and at least MIN_LOOP_TOKENS tokens.
MAX_PERIOD = 64
MIN_REPEATS = 3
MIN_LOOP_TOKENS = 24


# ---------------------------------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------------------------------


def read_book(name, data=DATA):
    """A book as one string with all whitespace collapsed to single spaces."""
    raw = (Path(data) / "gutenberg" / f"{name}.txt").read_text(encoding="latin-1")
    return re.sub(r"\s+", " ", raw).strip()


def sentence_starts(words):
    """Indices of words that start a sentence: after a word ending in . ! or ?, and capitalised."""
    return [i for i in range(1, len(words)) if re.search(r"[.!?][\"')\]]*$", words[i - 1]) and words[i][:1].isupper()]


@dataclass
class Passage:
    book: str
    index: int  # word position of the prompt's first word
    prompt: str
    reference: str  # the words that actually follow, for the human baseline


def make_prompts(data=DATA, per_book=PROMPTS_PER_BOOK):
    """Evenly spaced passages from every book, each starting at a sentence."""
    out = []
    for name in BOOKS:
        words = read_book(name, data).split(" ")
        starts = np.array(sentence_starts(words))
        usable = len(words) - PROMPT_WORDS - REFERENCE_WORDS
        for k in range(per_book):
            target = int(len(words) * (SKIP_FRONT + (1 - SKIP_FRONT) * (k + 0.5) / per_book))
            later = starts[(starts >= target) & (starts <= usable)]
            i = int(later[0]) if len(later) else int(starts[starts <= usable][-1])
            prompt = " ".join(words[i : i + PROMPT_WORDS])
            ref = " ".join(words[i + PROMPT_WORDS : i + PROMPT_WORDS + REFERENCE_WORDS])
            out.append(Passage(name, i, prompt, " " + ref))
    return out


# ---------------------------------------------------------------------------------------------------
# Decoding rules
# ---------------------------------------------------------------------------------------------------


def log_softmax(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def choose(logits, rule, rng):
    """Next token per row of a (batch, vocab) array of logits under one decoding rule."""
    logits = np.asarray(logits, dtype=np.float64)
    if rule["kind"] == "greedy":
        return logits.argmax(axis=1)
    if rule["kind"] == "temperature":
        probs = np.exp(log_softmax(logits / rule["T"]))
    elif rule["kind"] == "top_p":
        probs = np.exp(log_softmax(logits))
        order = np.argsort(-probs, axis=1, kind="stable")
        sorted_p = np.take_along_axis(probs, order, axis=1)
        keep_sorted = np.cumsum(sorted_p, axis=1) - sorted_p < rule["p"]  # keep until mass reaches p
        keep = np.zeros_like(keep_sorted)
        np.put_along_axis(keep, order, keep_sorted, axis=1)
        probs = np.where(keep, probs, 0.0)
        probs /= probs.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"unknown rule {rule}")
    u = rng.random(len(probs))
    cdf = np.cumsum(probs, axis=1)
    return np.minimum((cdf < (u * cdf[:, -1])[:, None]).sum(axis=1), probs.shape[1] - 1)


class Model:
    """Interface: start(list of token-id lists) -> (batch, vocab) logits; step(ids) -> logits."""

    def start(self, prompts):  # pragma: no cover - interface
        raise NotImplementedError

    def step(self, ids):  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class Decoded:
    tokens: np.ndarray  # (batch, n_new) chosen tokens
    logprob: np.ndarray  # log-probability of each chosen token under the model (no temperature)
    margin: np.ndarray  # log-probability gap between the model's top two tokens at each step


def decode(model, prompts, rule, n_new, rng):
    """Run one decoding rule for n_new steps on a batch of tokenised prompts."""
    logits = model.start(prompts)
    b = len(prompts)
    tokens, lp, mg = (
        np.zeros((b, n_new), dtype=np.int64),
        np.zeros((b, n_new)),
        np.zeros((b, n_new)),
    )
    for t in range(n_new):
        ls = log_softmax(np.asarray(logits, dtype=np.float64))
        nxt = choose(logits, rule, rng)
        top2 = -np.partition(-ls, 1, axis=1)[:, :2]
        tokens[:, t] = nxt
        lp[:, t] = ls[np.arange(b), nxt]
        mg[:, t] = top2[:, 0] - top2[:, 1]
        if t + 1 < n_new:
            logits = model.step(nxt)
    return Decoded(tokens, lp, mg)


class MarkovModel(Model):
    """A first-order model over a small vocabulary, for tests: logits = log of a row of counts."""

    def __init__(self, counts):
        self.logits = np.log(np.asarray(counts, dtype=np.float64) + 1e-12)

    def start(self, prompts):
        self.last = np.array([p[-1] for p in prompts])
        return self.logits[self.last]

    def step(self, ids):
        self.last = np.asarray(ids)
        return self.logits[self.last]


# ---------------------------------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------------------------------


@dataclass
class Tail:
    period: int  # length of the repeating block (0 if none)
    start: int  # index where the periodic stretch begins
    repeats: float  # how many periods the stretch covers

    @property
    def looped(self):
        return self.period > 0


def periodic_tail(x, max_period=MAX_PERIOD, min_repeats=MIN_REPEATS, min_tokens=MIN_LOOP_TOKENS):
    """The longest stretch at the end of x that repeats with some period p <= max_period.

    For each p, extend backwards while x[i] == x[i - p]. A stretch qualifies as a loop if it spans at
    least min_repeats periods and min_tokens tokens. Among qualifying periods the one covering the
    longest stretch wins; ties go to the shortest period.
    """
    x = np.asarray(x)
    n = len(x)
    best = Tail(0, n, 0.0)
    for p in range(1, min(max_period, n // 2) + 1):
        eq = x[p:] == x[:-p]  # eq[j] means x[j + p] == x[j]
        bad = np.flatnonzero(~eq)
        k = len(eq) - 1 - bad[-1] if len(bad) else len(eq)  # matching run at the end
        span = k + p  # the matching run plus the first copy of the block
        if span >= min_repeats * p and span >= min_tokens and span > n - best.start:
            best = Tail(int(p), int(n - span), float(span / p))
    return best


def repeated_ngram_share(x, n=4):
    """Share of the n-grams of x that already appeared earlier in x."""
    seen, rep, total = set(), 0, 0
    for i in range(len(x) - n + 1):
        g = tuple(x[i : i + n])
        rep += g in seen
        total += 1
        seen.add(g)
    return rep / total if total else 0.0


def common_prefix(a, b):
    """Number of leading tokens two sequences share (memorisation check)."""
    k = 0
    for u, v in zip(a, b, strict=False):
        if u != v:
            break
        k += 1
    return k


def wilson(k, n, z=1.96):
    """Wilson 95% interval for a proportion k/n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# ---------------------------------------------------------------------------------------------------
# Word n-gram chains: the finite-memory end of the comparison
# ---------------------------------------------------------------------------------------------------


def tokenize_words(text):
    return re.findall(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]", text)


class NGramChain:
    """Greedy word chain with memory of n - 1 words, backing off to shorter contexts when unseen.

    Words are coded as integers in alphabetical order; a context of k words is one integer key, so
    each table is a pair of sorted arrays (context key, most frequent next word). Ties between next
    words go to the alphabetically first.
    """

    def __init__(self, words, n):
        self.n = n
        self.vocab = sorted(set(words))
        self.code = {w: i for i, w in enumerate(self.vocab)}
        v = len(self.vocab)
        if v ** max(n - 1, 1) >= 2**62:
            raise ValueError("vocabulary too large for integer context keys")
        ids = np.array([self.code[w] for w in words], dtype=np.int64)
        self.v = v
        self.keys, self.best = [], []
        for k in range(n):
            nxt = ids[k:]
            ctx = np.zeros(len(nxt), dtype=np.int64)
            for j in range(k):
                ctx = ctx * v + ids[j : len(ids) - k + j]
            pair = ctx * v + nxt
            uniq, cnt = np.unique(pair, return_counts=True)
            c_ctx, c_next = uniq // v, uniq % v
            order = np.lexsort((c_next, -cnt, c_ctx))  # by context, then count desc, then word id
            c_ctx, c_next = c_ctx[order], c_next[order]
            first = np.r_[True, c_ctx[1:] != c_ctx[:-1]]
            self.keys.append(c_ctx[first])
            self.best.append(c_next[first])

    def key(self, ctx_ids):
        k = 0
        for w in ctx_ids:
            k = k * self.v + w
        return k

    def next_id(self, hist_ids):
        for k in range(self.n - 1, -1, -1):
            if len(hist_ids) < k:
                continue
            key = self.key(hist_ids[len(hist_ids) - k :]) if k else 0
            pos = np.searchsorted(self.keys[k], key)
            if pos < len(self.keys[k]) and self.keys[k][pos] == key:
                return int(self.best[k][pos])
        raise KeyError("empty model")

    def continue_from(self, prompt_words, steps):
        """Greedy continuation; prompt words unknown to the model are dropped from the history."""
        hist = [self.code[w] for w in prompt_words if w in self.code]
        out = []
        for _ in range(steps):
            w = self.next_id(hist)
            out.append(w)
            hist.append(w)
        return [self.vocab[i] for i in out]
