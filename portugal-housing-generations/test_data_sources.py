"""The data inventory lists every pinned file with its provider terms, and blocks unlicensed files from archives."""

from __future__ import annotations

import csv
import re
import unittest
from pathlib import Path

import data_sources as D
import reproduce

HERE = Path(__file__).parent
WORKFLOW = HERE.parent / ".github" / "workflows" / "portugal-housing-generations.yml"


class InventoryTests(unittest.TestCase):
    def test_committed_inventory_matches_the_retrieval_records(self) -> None:
        with D.OUT.open(encoding="utf-8", newline="") as fh:
            committed = list(csv.DictReader(fh))
        self.assertEqual(committed, D.inventory())

    def test_every_pinned_file_is_listed_once(self) -> None:
        listed = [r["file"] for r in D.inventory()]
        self.assertEqual(len(listed), len(set(listed)))
        data_files = {name for name in reproduce.read_manifest() if not name.endswith(".meta.json")}
        self.assertEqual(set(listed), data_files)
        for name in reproduce.read_manifest():
            if name.endswith(".meta.json"):
                with self.subTest(record=name):
                    self.assertTrue(any(name.startswith(f.rsplit(".", 1)[0]) for f in listed))

    def test_every_row_names_a_source_terms_and_attribution(self) -> None:
        for row in D.inventory():
            with self.subTest(file=row["file"]):
                for field in ("provider", "dataset", "title", "source_url", "retrieved_utc", "terms", "attribution"):
                    self.assertTrue(row[field].strip(), field)
                self.assertTrue(row["source_url"].startswith("https://"))
                self.assertIn(row["redistribution"], {D.PERMITTED, D.NOT_ESTABLISHED})

    def test_provider_terms(self) -> None:
        rows = D.inventory()
        for row in rows:
            with self.subTest(file=row["file"]):
                if row["provider"] == "INE":
                    self.assertIn("CC BY 4.0", row["terms"])
                    self.assertIn("Instituto Nacional de Estatística", row["attribution"])
                    self.assertRegex(row["dataset"], r"^\d{7}$")
                elif row["provider"] == "Eurostat":
                    self.assertEqual(row["doi"], f"https://doi.org/10.2908/{row['dataset'].upper()}")
                    self.assertIn("/data/" + row["dataset"] + "?", row["source_url"])
                elif row["provider"] == "ECB":
                    self.assertIn("quoted", row["terms"])
                    self.assertTrue(row["source_url"].startswith("https://data-api.ecb.europa.eu/"))
        self.assertEqual({r["provider"] for r in rows}, {"INE", "Eurostat", "ECB", "PORDATA"})

    def test_pordata_file_is_the_only_one_not_cleared(self) -> None:
        self.assertEqual(D.blocked(D.inventory()), ["ine_pordata_completed_new_dwellings_PT_2026-10-29.csv"])

    @unittest.skipUnless(WORKFLOW.exists(), "workflow is outside this folder (e.g. in the release archive)")
    def test_archive_task_checks_redistribution_first(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        archive_job = workflow.split("release-archive:\n", 1)[1]
        check = archive_job.index("python data_sources.py --check-redistribution")
        self.assertLess(check, archive_job.index("git archive"))

    def test_documents_describe_the_blocker(self) -> None:
        doc = (HERE / "DATA_SOURCES.md").read_text(encoding="utf-8")
        for snippet in ("Blocker: the PORDATA transcription", "Routes to clear it", "pordata@ffms.pt",
                        "https://www.pordata.pt/Termos+Legais", "https://ec.europa.eu/eurostat/help/copyright-notice",
                        "usage_policy.en.html", "INST=3585857"):
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, doc)
        notes = re.sub(r"\s+", " ", (HERE / "RELEASE_NOTES_v1.0.0.md").read_text(encoding="utf-8"))
        self.assertRegex(notes, re.compile(r"PORDATA.*not been established|not established.*PORDATA", re.DOTALL))


if __name__ == "__main__":
    unittest.main()
