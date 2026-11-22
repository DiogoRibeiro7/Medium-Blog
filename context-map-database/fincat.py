"""Finitely presented categories: the schema side of the functorial data model.

A presentation is a directed multigraph (objects and generating arrows) plus path
equations. A morphism is an equivalence class of paths; here every path is reduced to a
normal form by rewriting with the equations, oriented from the longer side to the shorter
(shortlex). The rewriting is checked for local confluence (Knuth-Bendix critical pairs),
so normal forms are unique for the presentations used in this folder. Morphisms are then
enumerated by breadth-first search, which stops with an error if a hom-set grows past a
bound, so an infinite category is reported rather than silently truncated.

Paths are written in diagrammatic order: ("u", "i") means "first u, then i".
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Arrow:
    name: str
    src: str
    tgt: str


@dataclass(frozen=True)
class Path:
    src: str
    tgt: str
    word: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"id_{self.src}" if not self.word else ";".join(self.word)


def _shortlex_key(word: tuple[str, ...]) -> tuple[int, tuple[str, ...]]:
    return (len(word), word)


@dataclass
class Presentation:
    objects: tuple[str, ...]
    arrows: tuple[Arrow, ...]
    equations: tuple[tuple[Path, Path], ...] = ()
    max_hom: int = 200
    _rules: list[tuple[tuple[str, ...], tuple[str, ...]]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.by_name = {a.name: a for a in self.arrows}
        if len(self.by_name) != len(self.arrows):
            raise ValueError("arrow names must be unique")
        for a in self.arrows:
            if a.src not in self.objects or a.tgt not in self.objects:
                raise ValueError(f"arrow {a.name} has an unknown endpoint")
        rules = []
        for lhs, rhs in self.equations:
            for p in (lhs, rhs):
                self._check_path(p)
            if (lhs.src, lhs.tgt) != (rhs.src, rhs.tgt):
                raise ValueError(f"equation {lhs} = {rhs} has mismatched endpoints")
            big, small = sorted((lhs.word, rhs.word), key=_shortlex_key, reverse=True)
            if big == small:
                continue
            rules.append((big, small))
        self._rules = rules

    # -- paths -----------------------------------------------------------------
    def _check_path(self, p: Path) -> None:
        here = p.src
        for name in p.word:
            a = self.by_name[name]
            if a.src != here:
                raise ValueError(f"path {p} is not composable at {name}")
            here = a.tgt
        if here != p.tgt:
            raise ValueError(f"path {p} does not end at {p.tgt}")

    def path(self, src: str, *names: str) -> Path:
        here = src
        for name in names:
            a = self.by_name[name]
            if a.src != here:
                raise ValueError(f"{name} does not start at {here}")
            here = a.tgt
        return Path(src, here, tuple(names))

    def reduce_word(self, word: tuple[str, ...]) -> tuple[str, ...]:
        changed = True
        while changed:
            changed = False
            for big, small in self._rules:
                k = len(big)
                for i in range(len(word) - k + 1):
                    if word[i:i + k] == big:
                        word = word[:i] + small + word[i + k:]
                        changed = True
                        break
                if changed:
                    break
        return word

    def normal(self, p: Path) -> Path:
        return Path(p.src, p.tgt, self.reduce_word(p.word))

    def compose(self, p: Path, q: Path) -> Path:
        if p.tgt != q.src:
            raise ValueError("not composable")
        return self.normal(Path(p.src, q.tgt, p.word + q.word))

    def equal(self, p: Path, q: Path) -> bool:
        return (p.src, p.tgt) == (q.src, q.tgt) and self.normal(p).word == self.normal(q).word

    # -- confluence ------------------------------------------------------------
    def critical_pairs(self) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
        """Overlaps of rule left-hand sides whose two reductions do not join."""
        bad = []
        rules = self._rules
        for l1, r1 in rules:
            for l2, r2 in rules:
                # suffix of l1 equal to prefix of l2
                for k in range(1, min(len(l1), len(l2)) + (0 if l1 == l2 else 1)):
                    if l1[-k:] == l2[:k]:
                        w = l1 + l2[k:]
                        a = self.reduce_word(r1 + l2[k:])
                        b = self.reduce_word(l1[:-k] + r2)
                        if a != b:
                            bad.append((a, b))
                # l2 inside l1
                if l1 != l2 and len(l2) < len(l1):
                    for i in range(len(l1) - len(l2) + 1):
                        if l1[i:i + len(l2)] == l2:
                            a = self.reduce_word(r1)
                            b = self.reduce_word(l1[:i] + r2 + l1[i + len(l2):])
                            if a != b:
                                bad.append((a, b))
        return bad

    def is_confluent(self) -> bool:
        return not self.critical_pairs()

    # -- morphisms -------------------------------------------------------------
    def morphisms(self) -> dict[tuple[str, str], list[Path]]:
        """Every morphism, as a normal-form path, grouped by (source, target)."""
        out: dict[tuple[str, str], list[Path]] = {(a, b): [] for a in self.objects for b in self.objects}
        for start in self.objects:
            seen = {()}
            ident = Path(start, start, ())
            out[(start, start)].append(ident)
            queue = deque([ident])
            while queue:
                p = queue.popleft()
                for a in self.arrows:
                    if a.src != p.tgt:
                        continue
                    q = self.normal(Path(p.src, a.tgt, p.word + (a.name,)))
                    if q.word in seen:
                        continue
                    seen.add(q.word)
                    out[(q.src, q.tgt)].append(q)
                    if len(seen) > self.max_hom * len(self.objects):
                        raise RuntimeError(f"more than {self.max_hom} morphisms out of {start}: infinite?")
                    queue.append(q)
        return out

    def hom(self, a: str, b: str) -> list[Path]:
        return self.morphisms()[(a, b)]

    def reachable(self, a: str, b: str) -> bool:
        """Whether some path a -> b exists. Equations never create or destroy this."""
        seen, queue = {a}, deque([a])
        while queue:
            x = queue.popleft()
            if x == b:
                return True
            for arr in self.arrows:
                if arr.src == x and arr.tgt not in seen:
                    seen.add(arr.tgt)
                    queue.append(arr.tgt)
        return False

    def generating_paths(self, a: str, b: str, max_len: int = 4) -> list[Path]:
        """Simple paths of generators from a to b (no repeated object), for display."""
        found = []

        def walk(here, word, visited):
            if here == b and word:
                found.append(Path(a, b, tuple(word)))
            if len(word) >= max_len:
                return
            for arr in self.arrows:
                if arr.src == here and arr.tgt not in visited:
                    walk(arr.tgt, word + [arr.name], visited | {arr.tgt})

        walk(a, [], {a})
        return found
