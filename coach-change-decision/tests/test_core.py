"""Tests on synthetic leagues, where the truth is known.

python -m unittest discover -s tests -v
"""

import sys
import unittest
from itertools import permutations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import coach_core as cc
from simulate import estimate, round_robin, synthetic


def run(m, sv, nj):
    return estimate(m, sv, nj)


class TestStandings(unittest.TestCase):
    def test_table_and_safety_line(self):
        # four clubs, one round: 0 beats 1, 2 draws 3; then 1 beats 2, 3 beats 0
        rows = []
        for g, (date, h, a, gh, ga) in enumerate(
            [
                ("2020-01-01", 0, 1, 2, 0),
                ("2020-01-01", 2, 3, 1, 1),
                ("2020-01-08", 1, 2, 1, 0),
                ("2020-01-08", 3, 0, 3, 0),
            ]
        ):
            rows += [
                ("T", 1, pd.Timestamp(date), g, h, a, "x", gh, ga, 0, 0, 0, 0, "m", 1),
                ("T", 1, pd.Timestamp(date), g, a, h, "x", ga, gh, 0, 0, 0, 0, "m", 0),
            ]
        m = cc.prepare(
            pd.DataFrame(
                rows,
                columns=[
                    "league",
                    "season",
                    "date",
                    "game_id",
                    "club_id",
                    "opp_id",
                    "club",
                    "gf",
                    "ga",
                    "shots_for",
                    "shots_against",
                    "sot_for",
                    "sot_against",
                    "manager",
                    "home",
                ],
            )
        )
        st = m.join(cc.standings(m))
        after1 = st[st.k == 1].set_index("club_id").pos.to_dict()
        self.assertEqual(after1, {0: 1, 2: 2, 3: 3, 1: 4})  # 2 and 3 tie on everything: club order decides
        after2 = st[st.k == 2].set_index("club_id")
        # points: 0 -> 3, 1 -> 3, 2 -> 1, 3 -> 4; club 3 top, then 0 (GD -1) and 1 (GD -1, GF 1) vs 0 (GF 2)
        self.assertEqual(after2.pos.to_dict(), {3: 1, 0: 2, 1: 3, 2: 4})
        # four clubs: the safety line is place n-3 = 1st
        self.assertTrue((after2.safe_pts == 4).all())

    def test_round_robin_is_complete(self):
        rounds = round_robin(6)
        games = [g for rd in rounds for g in rd]
        self.assertEqual(len(games), 30)
        self.assertEqual(set(games), set(permutations(range(6), 2)))


class TestChanges(unittest.TestCase):
    def test_change_detection_and_absence(self):
        m, sv, nj = synthetic(seasons=1, clubs=6, seed=3)
        # club 0: a stand-in for one match, then the manager is back; club 1: a real change
        m.loc[(m.club_id == 0) & (m.k == 6), "manager"] = "stand-in"
        m.loc[(m.club_id == 1) & (m.k >= 6), "manager"] = "new"
        dp = cc.decision_points(m, sv, nj)
        row0 = dp[(dp.club_id == 0) & (dp.k == 5)].iloc[0]
        row1 = dp[(dp.club_id == 1) & (dp.k == 5)].iloc[0]
        self.assertEqual(row0.change, 1)
        self.assertTrue(row0.returns)
        self.assertEqual(row1.change, 1)
        self.assertFalse(row1.returns)
        self.assertEqual(dp[(dp.club_id == 1) & (dp.k == 6)].tenure.iloc[0], 1)


class TestRegressionToTheMean(unittest.TestCase):
    def test_no_effect_is_recovered(self):
        """With no true effect, the before/after gain is large and the same-spot estimate is near zero."""
        est, before_after, n = run(*synthetic(effect=0.0, seed=11))
        self.assertGreater(n, 150)
        self.assertGreater(before_after, 3.0)
        self.assertLess(abs(est), 0.8)

    def test_a_real_effect_is_found(self):
        """A new manager worth +0.4 in strength shows up as a clear gain over the next ten."""
        est, _, n = run(*synthetic(effect=0.4, seed=12))
        self.assertGreater(n, 150)
        self.assertGreater(est, 1.5)


class TestHelpers(unittest.TestCase):
    def test_auc(self):
        self.assertAlmostEqual(cc.auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)


if __name__ == "__main__":
    unittest.main()
