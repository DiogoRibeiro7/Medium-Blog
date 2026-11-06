"""Scientific publication-asset contracts, independent of private manuscripts."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from prepare_medium_publication import (
    ROOT,
    build_release_markdown,
    check_scientific_assets,
    extract_equations,
    main,
    normalize_public_companion_links,
)


class ScientificPublicationAssetTests(unittest.TestCase):
    """Guard public assets and safe editorial CLI behavior."""

    def test_all_figure_and_equation_pngs(self) -> None:
        self.assertEqual(check_scientific_assets(ROOT), (17, 15))

    def test_missing_public_assets_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="midterm-public-assets-") as temporary:
            with self.assertRaisesRegex(ValueError, "Missing or undersized scientific PNG"):
                check_scientific_assets(Path(temporary))

    def test_public_scientific_cli_needs_no_private_checkout(self) -> None:
        with patch.object(sys, "argv", ["prepare_medium_publication.py", "--check-scientific"]):
            self.assertEqual(main(), 0)

    def test_editorial_cli_requires_explicit_paths(self) -> None:
        with patch.object(sys, "argv", ["prepare_medium_publication.py", "--check"]):
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    main()
        self.assertEqual(caught.exception.code, 2)

    def test_editorial_cli_rejects_public_notices_as_input(self) -> None:
        args: list[str] = [
            "prepare_medium_publication.py",
            "--build",
            "--source",
            str(ROOT / "article.md"),
            "--output",
            str(ROOT / "publication" / "medium_ready.md"),
        ]
        with patch.object(sys, "argv", args):
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    main()
        self.assertEqual(caught.exception.code, 2)

    def test_private_archive_links_remain_renderable(self) -> None:
        source: str = (
            "![Figure](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/"
            "6e6f84c8593198fbd70811fd8d1915def98373a8/"
            "midterm-forecast-fit-vs-prediction/figures/fig1_historical_fit.png) "
            "[Data](https://github.com/DiogoRibeiro7/Medium-Blog/blob/"
            "6e6f84c8593198fbd70811fd8d1915def98373a8/"
            "midterm-forecast-fit-vs-prediction/results/summary.json)"
        )
        normalized: str = normalize_public_companion_links(source)
        self.assertIn("](figures/fig1_historical_fit.png)", normalized)
        self.assertIn("](results/summary.json)", normalized)
        self.assertNotIn("raw.githubusercontent.com", normalized)

    def test_editorial_renderer_rejects_unvalidated_content(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unexpected article title"):
            build_release_markdown("# Unrelated manuscript")
        with self.assertRaisesRegex(ValueError, "Expected 15 display equations"):
            extract_equations("# No equations here")


if __name__ == "__main__":
    unittest.main()
