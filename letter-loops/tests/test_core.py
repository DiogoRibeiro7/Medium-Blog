"""Tests of the methods on cases where the answer is known.

python -m unittest discover -s tests -v
"""

import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import letters_core as L  # noqa: E402


def brute_one_loop(n):
    """Share of all n**n maps of n points with exactly one loop, by enumeration."""
    hits = 0
    for f in itertools.product(range(n), repeat=n):
        hits += L.analyse(np.array(f)).n_attractors() == 1
    return hits / n**n


class TestRandomMaps(unittest.TestCase):
    def test_katz_formula_matches_enumeration(self):
        for n in range(1, 7):
            self.assertAlmostEqual(float(L.p_one_loop_uniform(n)), brute_one_loop(n), places=12)

    def test_katz_known_values(self):
        self.assertEqual(L.p_one_loop_uniform(2), 3 / 4)
        self.assertAlmostEqual(float(L.p_one_loop_uniform(26)), 0.2337, places=4)


class TestStructure(unittest.TestCase):
    def setUp(self):
        # 0<->1 is a loop, 2 feeds it; 3 -> 4 <-> 5 is another loop; 7 -> 6, a dead end; 8 points at itself
        self.f = np.array([1, 0, 0, 4, 5, 4, -1, 6, 8])
        self.s = L.analyse(self.f, order=np.arange(9, dtype=float))

    def test_attractors(self):
        kinds = sorted(L.classify_attractors(self.s))
        self.assertEqual(kinds, ["dead end", "fixed point", "loop", "loop"])
        self.assertEqual(sorted(self.s.basin_sizes().tolist()), [1, 2, 3, 3])

    def test_depths(self):
        self.assertEqual(self.s.depth.tolist(), [0, 0, 1, 1, 0, 0, 0, 1, 0])

    def test_rotation_starts_at_heaviest_or_boundary(self):
        self.assertIn((1, 0), self.s.loops)  # node 1 is heavier than node 0 under order = index
        self.assertEqual(L.rotate([2, 0, 1], np.zeros(3), anchor=0), (0, 1, 2))
        self.assertEqual(L.rotate([2, 0, 1], np.array([0.0, 0.0, 5.0])), (2, 0, 1))

    def test_greedy_word(self):
        # boundary 0 -> t(1) -> h(2) -> e(3) -> 0
        self.assertEqual(L.greedy_word(np.array([1, 2, 3, 0])), [1, 2, 3])
        self.assertIsNone(L.greedy_word(np.array([1, 2, 1])))

    def test_shuffle_keeps_attractor_count_in_range(self):
        rng = np.random.default_rng(0)
        f = np.array([1, 0, 0, 0, 1, 2])
        out = L.shuffled_targets(f, rng, 500, allow_self=False)
        self.assertTrue(((out >= 1) & (out <= 3)).all())


class TestGreedy(unittest.TestCase):
    def test_argmax_and_empty_rows(self):
        m = np.array([[0, 3, 1], [2, 0, 0], [0, 0, 0]], dtype=float)
        self.assertEqual(L.greedy_map(m).tolist(), [1, 0, -1])

    def test_ties_broken_by_running_text(self):
        m = np.array([[0, 2, 2], [1, 0, 0], [1, 0, 0]], dtype=float)
        tb = np.array([[0, 1, 5], [0, 0, 0], [0, 0, 0]], dtype=float)
        self.assertEqual(L.greedy_map(m, tb)[0], 2)
        self.assertEqual(L.tied_rows(m).tolist(), [0])

    def test_margins(self):
        m = np.array([[0, 4, 2], [3, 0, 0], [0, 0, 0]], dtype=float)
        r = L.margins(m)
        self.assertEqual(r[0], 2.0)
        self.assertTrue(np.isinf(r[1]))
        self.assertTrue(np.isnan(r[2]))


class TestAlphabets(unittest.TestCase):
    def test_parse_exemplars(self):
        self.assertEqual(L.parse_exemplars("[aą b {ng} c \\u200E ً]"), ["a", "ą", "b", "c"])

    def test_fold(self):
        self.assertEqual(L.fold("straße", "de"), "strasse")
        self.assertEqual(L.fold("ça", "fr"), "ca")
        self.assertEqual(L.fold("ää", "fi"), "aa")
        self.assertEqual(L.fold("λόγος", "el"), "λογοσ")
        self.assertEqual(L.fold("שלום", "he"), "שלומ")
        self.assertEqual(L.fold("أي", "ar"), "اي")
        self.assertEqual(L.fold("أي", "fa"), "ای")


class TestCounting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        payload = {
            "main": {
                "en": {"characters": {"exemplarCharacters": "[a b c d e f g h i j k l m n o p q r s t u v w x y z]"}}
            }
        }
        Path(self.tmp.name, "en.json").write_text(json.dumps(payload))
        freqs = {"the": 0.5, "he": 0.2, "x1": 0.2, "tea": 1e-7}
        self.c = L.count_language("en", freqs, cldr_dir=self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_kept_share_and_unseen_letters_dropped(self):
        c = self.c[("standard", "text")]
        self.assertAlmostEqual(c.kept, (0.5 + 0.2 + 1e-7) / (0.9 + 1e-7))
        self.assertEqual(c.symbols, ["_", "a", "e", "h", "t"])

    def test_pairs_initial_final(self):
        c = self.c[("standard", "text")]
        ix = {s: k for k, s in enumerate(c.symbols)}
        self.assertAlmostEqual(c.pairs[ix["h"], ix["e"]], 0.7)
        self.assertAlmostEqual(c.pairs[ix["t"], ix["h"]], 0.5)
        self.assertAlmostEqual(c.initial[ix["t"]], 0.5 + 1e-7)
        self.assertAlmostEqual(c.final[ix["e"]], 0.7)

    def test_dictionary_counts_each_frequent_word_once(self):
        c = self.c[("standard", "dictionary")]
        ix = {s: k for k, s in enumerate(c.symbols)}
        self.assertEqual(c.pairs[ix["h"], ix["e"]], 2.0)
        self.assertEqual(c.words, 2)

    def test_boundary_matrix(self):
        sym, m = L.transitions(self.c[("standard", "text")], boundary=True)
        f = L.greedy_map(m)
        self.assertEqual(L.spell(L.greedy_word(f), sym), "the")


class TestNorvig(unittest.TestCase):
    def test_identical_source_gives_zero_spread_and_full_agreement(self):
        sym = ["_", *"abcdefghijklmnopqrstuvwxyz"]
        ix = {s: k for k, s in enumerate(sym)}
        pairs = np.zeros((27, 27))
        for k, v in L.NORVIG_TOP50.items():
            pairs[ix[k[0]], ix[k[1]]] = v
        c = L.Counts(sym, pairs, np.zeros(27), np.zeros(27), np.ones(27), 1.0, 0)
        keys, nv, wf, sd, settled = L.norvig_comparison(c)
        self.assertAlmostEqual(sd, 0.0)
        self.assertEqual(settled["e"], ("r", "r"))
        self.assertTrue(all(a == b for a, b in settled.values()))
        self.assertEqual(L.NORVIG_LETTERS - L.NORVIG_WORDS, 2_819_662_855_499)


class TestChains(unittest.TestCase):
    def test_stationary_two_states(self):
        p = np.array([[0.9, 0.1], [0.5, 0.5]])
        pi, res = L.stationary(p)
        np.testing.assert_allclose(pi, [5 / 6, 1 / 6])
        self.assertLess(res, 1e-12)

    def test_temperature_limits(self):
        m = np.array([[0, 3, 1], [2, 0, 2], [5, 1, 0]], dtype=float)
        np.testing.assert_allclose(L.tempered(m, 1.0), m / m.sum(axis=1, keepdims=True))
        cold = L.tempered(m, 0.01)
        np.testing.assert_array_equal(cold.argmax(axis=1), [1, 0, 0])
        self.assertGreater(cold[0, 1], 0.999)
        self.assertAlmostEqual(cold[1, 0], 0.5)  # a tie stays a tie

    def test_irreducible(self):
        self.assertTrue(L.is_irreducible(np.array([[0, 1], [1, 0]])))
        self.assertFalse(L.is_irreducible(np.array([[1, 1], [0, 1]])))

    def test_entropy_rate_of_a_fair_coin(self):
        p = np.full((2, 2), 0.5)
        self.assertAlmostEqual(L.entropy_rate(p, np.array([0.5, 0.5])), 1.0)


if __name__ == "__main__":
    unittest.main()
