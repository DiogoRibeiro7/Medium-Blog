"""Tests for official CAOP extraction and noncausal distance sensitivity."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from extract_caop import (PROVENANCE, SNAPSHOT, panel_municipality_codes, multipolygon_area_centroid,
                          parse_caop, ring_area_centroid, serialize_csv)
from porto_distance_sensitivity import (FIGURES, RESULTS, comparison_table,
                                        diagnostics, load_centroids, partial_rank_correlation, run)


class OfficialCaopGeometryTests(unittest.TestCase):
    def test_ring_area_centroid(self):
        ring = [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]
        self.assertEqual(ring_area_centroid(ring), (4, 1, 1))

    def test_ring_reversal_invariant(self):
        ring = [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]
        self.assertEqual(ring_area_centroid(ring), ring_area_centroid(ring[::-1]))

    def test_ring_rejects_unclosed(self):
        with self.assertRaises(ValueError):
            ring_area_centroid([[0, 0], [2, 0], [2, 2], [0, 2]])

    def test_ring_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            ring_area_centroid([[0, 0], [float('nan'), 0], [1, 1], [0, 0]])

    def test_ring_rejects_degenerate(self):
        with self.assertRaises(ValueError):
            ring_area_centroid([[0, 0], [1, 0], [2, 0], [0, 0]])

    def test_multipolygon_holes(self):
        exterior = [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]
        hole = [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]
        area, x, y = multipolygon_area_centroid([[exterior, hole]])
        self.assertEqual(area, 15)
        self.assertAlmostEqual(x, (16 * 2 - 1.5) / 15)
        self.assertAlmostEqual(y, (16 * 2 - 1.5) / 15)

    def test_multipolygon_rejects_excessive_hole(self):
        little = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
        big = [[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]]
        with self.assertRaises(ValueError):
            multipolygon_area_centroid([[little, big]])

    def test_invalid_source_rejected(self):
        for payload in (b'not-json', b'[]', b'{"type":"FeatureCollection","numberMatched":0,"numberReturned":0,"features":[]}'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                parse_caop(payload)

    def test_pinned_official_derivative_integrity(self):
        meta = json.loads(PROVENANCE.read_text(encoding='utf-8'))
        self.assertEqual(meta['source_epsg'], 3763)
        self.assertEqual(meta['source_nuts3_code'], '11A')
        self.assertEqual(meta['records'], 17)
        self.assertEqual(hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(), meta['derived_csv_sha256'])
        self.assertIn('CC-BY', meta['source_license'])
        self.assertEqual(meta['asserted_publication_date'], '2026-02-18')
        self.assertIn('not attributes', meta['asserted_dates_basis'])
        self.assertIn('not committed', meta['raw_response_retention'])

    def test_standalone_extractor_uses_only_python_standard_library(self):
        script = Path(__file__).resolve().parents[1] / 'extract_caop.py'
        completed = subprocess.run([sys.executable, '-S', str(script), '--help'],
                                   capture_output=True, text=True, check=True)
        self.assertIn('--download', completed.stdout)

    def test_complete_17_municipality_code_registry(self):
        self.assertEqual(len(panel_municipality_codes()), 17)
        self.assertEqual(panel_municipality_codes()['1312'], 'Porto')

    def test_oversize_stdin_rejected_explicitly(self):
        script = Path(__file__).resolve().parents[1] / 'extract_caop.py'
        completed = subprocess.run([sys.executable, str(script), '--stdin'],
                                   input=b' ' * 25_000_001, capture_output=True, check=False)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(b'CAOP stdin input exceeds bounded 25 MB', completed.stderr)

    def test_derivative_checksum_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            modified = Path(tmp) / 'bad.csv'
            modified.write_bytes(SNAPSHOT.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                load_centroids(modified)

    def test_serialization_is_stable(self):
        with SNAPSHOT.open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 17)
        self.assertEqual(serialize_csv(rows), SNAPSHOT.read_bytes())


class PortoDistanceRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = comparison_table()
        cls.stats, cls.deletions = diagnostics(cls.rows)

    def test_complete_centroids(self):
        c = load_centroids()
        self.assertEqual(len(c), 17)
        self.assertEqual(c['Porto'], (-40942.960, 165950.533))

    def test_distance_range(self):
        d = [float(r['caop2025_porto_centroid_euclidean_distance_km']) for r in self.rows]
        self.assertAlmostEqual(min(d), 7.096492)
        self.assertAlmostEqual(max(d), 43.641129)

    def test_four_windows_four_exposures(self):
        self.assertEqual(len(self.stats), 16)
        self.assertEqual(len({(r['start_year'], r['exposure']) for r in self.stats}), 16)

    def test_reject_unknown_comparator_name(self):
        mutated = [dict(row) for row in self.rows]
        mutated[0]['municipality'] = 'Outside AMP'
        with self.assertRaises(ValueError):
            diagnostics(mutated)

    def test_primary_distance_rank(self):
        r = next(r for r in self.stats if r['start_year'] == 2020 and r['exposure'] == 'centroid_distance_km')
        self.assertAlmostEqual(float(r['unadjusted_spearman_rho']), .176471, places=6)

    def test_strong_distance_commute_collinearity(self):
        r = next(r for r in self.stats if r['start_year'] == 2020 and r['exposure'] == 'per_1000_residents')
        self.assertEqual(r['exposure_vs_distance_spearman_rho'], '-0.950000')

    def test_primary_distance_partial_rank(self):
        r = next(r for r in self.stats if r['start_year'] == 2020 and r['exposure'] == 'per_1000_residents')
        self.assertAlmostEqual(float(r['partial_spearman_controlling_centroid_distance']), -.421057, places=6)

    def test_partial_rank_invariant_to_monotone_transform(self):
        x = np.array([1., 2., 5., 3., 7., 4., 6.])
        y = np.array([4., 3., 7., 2., 1., 6., 5.])
        z = np.array([7., 2., 3., 1., 6., 5., 4.])
        self.assertAlmostEqual(partial_rank_correlation(x, y, z), partial_rank_correlation(x**3, y**3, z**3))

    def test_exact_rank_collinearity_rejected(self):
        x = np.arange(6.)
        with self.assertRaisesRegex(ValueError, 'collinearity'):
            partial_rank_correlation(x, x[::-1], x)

    def test_partial_leave_one_out_complete(self):
        self.assertEqual(len(self.deletions), 16)
        self.assertAlmostEqual(min(float(r['leave_one_out_15_partial_spearman']) for r in self.deletions), -.563284, places=6)
        self.assertAlmostEqual(max(float(r['leave_one_out_15_partial_spearman']) for r in self.deletions), -.288604, places=6)

    def test_regenerates_offline_outputs_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            report = run(base)
            self.assertEqual(report['municipalities'], 16)
            self.assertEqual(report['diagnostic_rows'], 16)
            self.assertEqual(report['leave_one_out_rows'], 16)
            self.assertEqual(len(report['limits']), 9)
            self.assertIn('not drive time', ' '.join(report['limits']))
            for figure in FIGURES:
                self.assertGreater((base / 'figures' / figure).stat().st_size, 10000)
            for result in RESULTS:
                self.assertTrue((base / 'results' / result).is_file())
            written = json.loads((base / 'results' / RESULTS[-1]).read_text(encoding='utf-8'))
            self.assertEqual(written['primary_per_resident_partial_spearman'], '-0.421057')


if __name__ == '__main__':
    unittest.main()
