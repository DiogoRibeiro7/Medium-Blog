"""Guards on two scientific statements corrected after an independent review.

1. The HICP actual-rentals index (COICOP 04.1) is a price index for a sample of rented dwellings, covering new and
   existing contracts and private, public and social units (Eurostat HICP Methodological Manual 2024, section
   12.4). It is not a measure of what "sitting tenants" pay, and dividing it by median income does not show a
   tenant's rent burden.
2. The near-zero completions coefficient for existing dwellings is consistent with a change in the mix of
   dwellings sold, but it does not identify that mechanism.

The tests fail if an overstatement reappears in the documents, figure scripts or result-file text, or if the
qualified statements, methodology references and source identifiers are removed.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

HERE = Path(__file__).parent
MANUAL_URL = "https://ec.europa.eu/eurostat/documents/3859598/18594110/KS-GQ-24-003-EN-N.pdf"

# Overstatements removed in the correction (case-insensitive; whitespace is normalised first).
FORBIDDEN = [
    r"rents? (paid by|of) sitting tenants",
    r"(index|indices) (of|for) sitting tenants",
    r"sitting tenants'? rents?",
    r"rent index of sitting",
    r"sitting tenants:",
    r"for the same dwelling over time",
    r"already had a lease",
    r"smaller share of (their )?income",
    r"excludes that replacement",
    r"which the quality-adjusted index excludes",
    r"rents for comparable ongoing tenancies grew more slowly than income",
    r"sale-mix artefact",
    r"artefact of (the sale mix|which dwellings)",
    r"lives entirely in the composition",
    r"comes from the mix of dwellings",
    r"is a change in the mix of dwellings",
    r"which shows it to be a change",
    r"reflects new dwellings entering",
    r"the data show it directly",
    r"existing dwellings: sale mix",
]
# "sitting tenants" may appear only where the text explains what the index cannot measure.
ALLOWED_SITTING = [r"fixed (group|cohort) of sitting tenants"]


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("**", "").replace("*", "")).strip()


def release_notes_findings() -> str:
    """The findings part of the release notes (the changes section quotes the corrected wording on purpose)."""
    text = (HERE / "RELEASE_NOTES_v1.0.0.md").read_text(encoding="utf-8")
    return text.split("## Changes in this release")[0]


def scanned_texts() -> dict[str, str]:
    """Every document, script and result file whose wording is guarded."""
    names = ["README.md", "REPORT.md", "article.md", "rental.py", "integrated.py", "results_rental.json",
             "results_integrated.json", *sorted(p.name for p in HERE.glob("*_figures.py"))]
    texts = {name: (HERE / name).read_text(encoding="utf-8") for name in names}
    texts["RELEASE_NOTES_v1.0.0.md (findings)"] = release_notes_findings()
    return texts


class OverstatementTests(unittest.TestCase):
    def test_corrected_overstatements_do_not_reappear(self) -> None:
        for name, raw in scanned_texts().items():
            text = normalise(raw)
            for pattern in FORBIDDEN:
                with self.subTest(file=name, pattern=pattern):
                    self.assertIsNone(re.search(pattern, text, flags=re.IGNORECASE))

    def test_sitting_tenants_only_in_explanations_of_what_the_index_cannot_measure(self) -> None:
        for name, raw in scanned_texts().items():
            text = normalise(raw)
            for allowed in ALLOWED_SITTING:
                text = re.sub(allowed, "", text, flags=re.IGNORECASE)
            with self.subTest(file=name):
                self.assertNotRegex(text.lower(), r"sitting tenant")

    def test_patterns_catch_the_original_wording(self) -> None:
        originals = [
            "Rents of sitting tenants are what the consumer price index follows.",
            "For a household that already had a lease, rent became a smaller share of income over the decade.",
            "The quality-adjusted index, by design, excludes that replacement.",
            "The positive within-municipality association with completions is an artefact of the sale mix.",
            "The association lives entirely in the composition of what is sold.",
            "The price–construction association comes from the mix of dwellings sold",
        ]
        for line in originals:
            with self.subTest(line=line):
                self.assertTrue(any(re.search(p, normalise(line), flags=re.IGNORECASE) for p in FORBIDDEN))


class QualifiedStatementTests(unittest.TestCase):
    REQUIRED = {
        "REPORT.md": [
            "new and existing contracts",
            "private, public and social rental units (Eurostat, 2024, Section 12.4)",
            "Why the index cannot describe a fixed group of sitting tenants.",
            "it does not show that any tenant's rent fell relative to that tenant's income",
            "Comparing the index with new-lease medians.",
            "Index versus level:",
            "The pattern is consistent with a composition effect.",
            "The data do not identify that mechanism",
            "compares indices of change; it does not show rent burdens.",
            "the completions verdict does not identify a price effect",
            "KS-GQ-24-003",
            "DiNardo, J., Fortin, N. M. and Lemieux, T. (1996)",
            "Inquérito às Rendas de Habitação",
        ],
        "article.md": [
            "under new and existing contracts, for private, public and social rental units (section 12.4 of the 2024 edition)",
            "The index cannot support that reading",
            "a comparison of indices of change, not of rent burdens",
            "which the data support but cannot prove",
            "It is still a reading, not an identification",
            "KS-GQ-24-003",
        ],
        "README.md": [
            "HICP Methodological Manual 2024, §12.4",
            "It cannot describe the rents or rent burden of a fixed group of",
            "Completions: consistent with the sale mix, mechanism not identified.",
        ],
    }

    def test_qualified_statements_are_present(self) -> None:
        for name, snippets in self.REQUIRED.items():
            text = normalise((HERE / name).read_text(encoding="utf-8"))
            for snippet in snippets:
                with self.subTest(file=name, snippet=snippet):
                    self.assertIn(normalise(snippet), text)

    def test_result_file_text_is_qualified(self) -> None:
        rental = json.loads((HERE / "results_rental.json").read_text(encoding="utf-8"))
        definition = rental["definitions"]["hicp_actual_rentals"]
        self.assertIn("new and existing contracts", definition)
        self.assertIn("section 12.4", definition)
        self.assertIn("not the rent path of a fixed group of tenants", definition)
        integrated = json.loads((HERE / "results_integrated.json").read_text(encoding="utf-8"))
        self.assertIn("does not identify that mechanism", integrated["completions_composition_check"]["note"])

    def test_methodology_references_and_source_identifiers(self) -> None:
        for name in ("REPORT.md", "article.md"):
            text = (HERE / name).read_text(encoding="utf-8")
            with self.subTest(file=name):
                self.assertIn(MANUAL_URL, text)
                self.assertIn("HPI-HICP", text)
        meta = json.loads((HERE / "data" / "eurostat_prc_hicp_aind_PT_rents_2026-10-29.meta.json").read_text(encoding="utf-8"))
        url = meta["url"]
        self.assertIn("/data/prc_hicp_aind?", url)
        self.assertIn("coicop=CP041", url)
        self.assertIn("unit=INX_A_AVG", url)
        for name in ("README.md", "REPORT.md"):
            text = (HERE / name).read_text(encoding="utf-8")
            with self.subTest(file=name):
                self.assertIn("prc_hicp_aind", text)
                self.assertIn("04.1", text)


if __name__ == "__main__":
    unittest.main()
