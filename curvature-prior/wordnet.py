"""WordNet 3.0 nouns from the pinned NLTK data package, verified and parsed.

The package is downloaded at run time from a fixed commit of github.com/nltk/nltk_data and
checked against the size and SHA-256 recorded in the data registry contract
(data_sources/princeton-wordnet-3-0.json). Nothing from it is committed here. The parser
reads data.noun directly: each synset, its words and its hypernym pointers (@ and @i).
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACT = HERE.parent / "data_sources" / "princeton-wordnet-3-0.json"
DATA = HERE / "data"


def contract() -> dict:
    return json.loads(CONTRACT.read_text())


def fetch(verbose: bool = False) -> Path:
    """Download the pinned file if it is missing, then verify size and SHA-256."""
    spec = contract()["files"][0]
    target = DATA / spec["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        if verbose:
            print(f"downloading {spec['url']}")
        urllib.request.urlretrieve(spec["url"], target)
    raw = target.read_bytes()
    if len(raw) != spec["bytes"] or hashlib.sha256(raw).hexdigest() != spec["sha256"]:
        raise RuntimeError(f"{target} does not match the contract (size or SHA-256)")
    return target


def read_nouns(path: Path | None = None) -> dict[str, tuple[list[str], list[str]]]:
    """offset -> (words, hypernym offsets), for every noun synset."""
    path = path or fetch()
    with zipfile.ZipFile(path) as z:
        lines = z.read("wordnet/data.noun").decode("latin-1").splitlines()
    syn = {}
    for ln in lines:
        if ln.startswith("  "):  # licence header
            continue
        body = ln.split(" | ")[0].split()
        off = body[0]
        w = int(body[3], 16)
        words = [body[4 + 2 * k] for k in range(w)]
        i = 4 + 2 * w
        p = int(body[i])
        i += 1
        hyp = []
        for _ in range(p):
            sym, toff, tpos, _st = body[i:i + 4]
            i += 4
            if sym in ("@", "@i") and tpos == "n":
                hyp.append(toff)
        syn[off] = (words, hyp)
    return syn


def first_sense(path: Path, lemma: str) -> str:
    with zipfile.ZipFile(path) as z:
        for ln in z.read("wordnet/index.noun").decode("latin-1").splitlines():
            if ln.startswith(lemma + " "):
                parts = ln.split()
                n_senses = int(parts[2])
                return parts[-n_senses]
    raise KeyError(lemma)


def subtree(syn: dict, root: str) -> tuple[list[str], list[tuple[int, int]], list[str]]:
    """All synsets below `root` (hypernym closure), the hypernym links among them as an
    undirected edge list, and a readable label (first word) for each node."""
    children = defaultdict(list)
    for o, (_w, hyp) in syn.items():
        for t in hyp:
            children[t].append(o)
    seen, stack = {root}, [root]
    while stack:
        x = stack.pop()
        for c in children[x]:
            if c not in seen:
                seen.add(c)
                stack.append(c)
    nodes = sorted(seen)
    index = {o: k for k, o in enumerate(nodes)}
    edges = sorted({(min(index[o], index[t]), max(index[o], index[t]))
                    for o in nodes for t in syn[o][1] if t in index})
    labels = [syn[o][0][0] for o in nodes]
    return nodes, edges, labels
