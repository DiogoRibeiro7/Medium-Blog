"""Tests of the methods on cases where the answer is known.

python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import loops_core as L  # noqa: E402


class TestPeriodicTail(unittest.TestCase):
    def test_pure_cycle(self):
        x = [5, 6, 7] * 20
        t = L.periodic_tail(x)
        self.assertEqual((t.period, t.start), (3, 0))
        self.assertTrue(t.looped)

    def test_lock_in_after_a_prefix(self):
        rng = np.random.default_rng(0)
        prefix = list(rng.integers(100, 200, size=37))
        x = prefix + [1, 2, 3, 4, 5] * 10
        t = L.periodic_tail(x)
        self.assertEqual((t.period, t.start), (5, 37))

    def test_no_loop_in_random_text(self):
        rng = np.random.default_rng(1)
        self.assertFalse(L.periodic_tail(rng.integers(0, 1000, size=256)).looped)

    def test_too_few_repeats_is_not_a_loop(self):
        x = list(range(100, 200)) + list(range(20)) * 2  # period 20 seen only twice
        self.assertFalse(L.periodic_tail(x).looped)

    def test_short_periods_need_enough_tokens(self):
        self.assertFalse(L.periodic_tail([9] * 10 + [8] * 20).looped)  # 20 tokens of period 1 < 24
        self.assertEqual(L.periodic_tail(list(range(50)) + [8] * 30).period, 1)

    def test_prefers_longest_stretch_then_shortest_period(self):
        x = [1, 2] * 30  # periods 2, 4, 6... all cover the whole sequence
        self.assertEqual(L.periodic_tail(x).period, 2)


class TestRepetition(unittest.TestCase):
    def test_repeated_ngram_share(self):
        self.assertEqual(L.repeated_ngram_share([1, 2, 3, 4, 5], 4), 0.0)
        self.assertAlmostEqual(L.repeated_ngram_share([1, 2] * 5, 2), 7 / 9)

    def test_counting_loop_needs_digit_masking(self):
        text = "".join(f"1:{v} And he said unto them, Go. " for v in range(10, 40))
        codes = lambda t: np.frombuffer(t.encode("utf-32-le"), dtype=np.uint32)  # noqa: E731
        self.assertFalse(L.periodic_tail(codes(text), max_period=400, min_tokens=60).looped)
        self.assertTrue(L.periodic_tail(codes(L.mask_digits(text)), max_period=400, min_tokens=60).looped)

    def test_common_prefix(self):
        self.assertEqual(L.common_prefix([1, 2, 3, 9], [1, 2, 3, 4]), 3)

    def test_wilson(self):
        lo, hi = L.wilson(0, 144)
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.0260, places=3)
        lo, hi = L.wilson(72, 144)
        self.assertAlmostEqual(lo + hi, 1.0)


class TestRules(unittest.TestCase):
    def setUp(self):
        self.logits = np.log(np.array([[0.5, 0.3, 0.15, 0.05]] * 20000))

    def test_greedy(self):
        np.testing.assert_array_equal(L.choose(self.logits[:3], L.RULES["greedy"], None), [0, 0, 0])

    def test_temperature_one_samples_the_probabilities(self):
        x = L.choose(self.logits, L.RULES["T1"], np.random.default_rng(0))
        np.testing.assert_allclose(np.bincount(x, minlength=4) / len(x), [0.5, 0.3, 0.15, 0.05], atol=0.012)

    def test_low_temperature_sharpens(self):
        x = L.choose(self.logits, L.RULES["T0.25"], np.random.default_rng(0))
        p = np.array([0.5, 0.3, 0.15, 0.05]) ** 4
        np.testing.assert_allclose(np.bincount(x, minlength=4) / len(x), p / p.sum(), atol=0.012)

    def test_top_p_keeps_the_nucleus(self):
        x = L.choose(self.logits, L.RULES["top-p0.9"], np.random.default_rng(0))
        p = np.array([0.5, 0.3, 0.15, 0.0])  # 0.5 + 0.3 < 0.9, so the third token is kept and the fourth dropped
        np.testing.assert_allclose(np.bincount(x, minlength=4) / len(x), p / p.sum(), atol=0.012)


class TestDecode(unittest.TestCase):
    def test_markov_greedy_must_loop(self):
        counts = np.array([[0, 5, 1, 1], [1, 0, 6, 2], [1, 1, 0, 9], [1, 7, 1, 0]])  # 1 -> 2 -> 3 -> 1
        d = L.decode(L.MarkovModel(counts), [[0], [3]], L.RULES["greedy"], 30, np.random.default_rng(0))
        np.testing.assert_array_equal(d.tokens[0, :6], [1, 2, 3, 1, 2, 3])
        self.assertEqual(L.periodic_tail(d.tokens[1]).period, 3)
        np.testing.assert_allclose(np.exp(d.logprob[0, 0]), 5 / 7)
        np.testing.assert_allclose(d.margin[0, 0], np.log(5))

    def test_sampling_from_the_same_model_rarely_loops(self):
        rng = np.random.default_rng(3)
        counts = rng.integers(1, 10, size=(50, 50))
        d = L.decode(L.MarkovModel(counts), [[0]] * 40, L.RULES["T1"], 200, rng)
        looped = sum(L.periodic_tail(t).looped for t in d.tokens)
        self.assertLessEqual(looped, 1)


class TestNGram(unittest.TestCase):
    def test_bigram_chain_is_finite_state(self):
        words = "the cat sat on the mat and the cat ran to the cat".split()
        chain = L.NGramChain(words, 2)
        out = chain.continue_from(["on"], 12)
        self.assertEqual(out[:6], ["the", "cat", "ran", "to", "the", "cat"])

    def test_backoff_to_unigram(self):
        chain = L.NGramChain("a b a c a".split(), 3)
        self.assertEqual(chain.continue_from(["zzz"], 1), ["a"])  # unknown context: most frequent word


class TestPrompts(unittest.TestCase):
    def test_sentence_starts(self):
        self.assertEqual(L.sentence_starts("He ran. She sat! it was. Done".split()), [2, 6])


if __name__ == "__main__":
    unittest.main()
