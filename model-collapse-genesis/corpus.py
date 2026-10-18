"""Load and tokenise the multilingual Genesis corpus distributed with NLTK.

Six natural-language translations are used: English (King James), Finnish,
French (Louis Segond 1910), German (Luther), Portuguese and Swedish (1917).
The World English Bible duplicate and the Lolcat parody are left out.

Tokenisation is deliberately simple: lower-case, words are maximal runs of
letters, and the six punctuation marks . , ; : ! ? are kept as tokens.
Words are tagged with their language, so "abraham" in English and "abraham"
in French are different tokens; punctuation marks are shared by all six
languages. The shared punctuation is the only place where a bigram model can
switch from one language to another, which makes the language of every
generated word known exactly.
"""

from __future__ import annotations

import html
import io
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/genesis.zip"

LANGUAGES = {
    # code: (file name, encoding)
    "en": ("english-kjv.txt", "latin-1"),
    "fi": ("finnish.txt", "latin-1"),
    "fr": ("french.txt", "latin-1"),
    "de": ("german.txt", "latin-1"),
    "pt": ("portuguese.txt", "utf-8"),
    "sv": ("swedish.txt", "cp850"),
}
LANG_NAMES = {
    "en": "English", "fi": "Finnish", "fr": "French",
    "de": "German", "pt": "Portuguese", "sv": "Swedish",
}
PUNCT = [".", ",", ";", ":", "!", "?"]
TOKEN_RE = re.compile(r"[^\W\d_]+|[.,;:!?]")


def download(data_dir: Path) -> Path:
    """Download and unpack the NLTK Genesis corpus (once)."""
    target = data_dir / "genesis"
    if (target / "english-kjv.txt").exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as r:
        payload = r.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        z.extractall(data_dir)
    return target


def clean(text: str) -> str:
    """Remove the HTML left in the German file and quote characters."""
    text = re.sub(r"<h3>.*?</h3>", " ", text, flags=re.S)  # chapter headings
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = text.replace("¢", " ")  # used as a quote mark in the Swedish file
    text = text.replace("{", " ").replace("}", " ")  # supplied words in the Portuguese file
    return text


def tokenise(text: str) -> list[str]:
    return TOKEN_RE.findall(clean(text).lower())


@dataclass
class Corpus:
    vocab: list[str]            # token strings, "xx|word" or "pun|."
    lang_of: np.ndarray         # language index per token id, -1 for punctuation
    langs: list[str]            # language codes, index -> code
    streams: dict               # code -> np.ndarray of token ids (in text order)
    punct_ids: np.ndarray

    @property
    def V(self) -> int:
        return len(self.vocab)


def load(data_dir: Path) -> Corpus:
    folder = download(data_dir)
    langs = list(LANGUAGES)
    vocab: list[str] = [f"pun|{p}" for p in PUNCT]
    index = {t: i for i, t in enumerate(vocab)}
    lang_of = [-1] * len(PUNCT)
    streams = {}
    for li, code in enumerate(langs):
        fname, enc = LANGUAGES[code]
        raw = (folder / fname).read_bytes().decode(enc)
        ids = []
        for tok in tokenise(raw):
            key = f"pun|{tok}" if tok in PUNCT else f"{code}|{tok}"
            if key not in index:
                index[key] = len(vocab)
                vocab.append(key)
                lang_of.append(li)
            ids.append(index[key])
        streams[code] = np.asarray(ids, dtype=np.int64)
    return Corpus(vocab=vocab, lang_of=np.asarray(lang_of), langs=langs,
                  streams=streams, punct_ids=np.arange(len(PUNCT)))


def split_blocks(corpus: Corpus, block: int, test_frac: float, rng: np.random.Generator):
    """Cut each language stream into contiguous blocks and hold out a random share."""
    train, test = [], []
    for code in corpus.langs:
        s = corpus.streams[code]
        blocks = [s[i:i + block] for i in range(0, len(s), block)]
        is_test = rng.random(len(blocks)) < test_frac
        for b, t in zip(blocks, is_test):
            (test if t else train).append(b)
    return train, test
