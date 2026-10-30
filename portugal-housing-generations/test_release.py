"""Release integrity: data checksums, citation and archive metadata, and the release documents agree."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import reproduce

HERE = Path(__file__).parent
VERSION = "1.0.0"
ORCID = "0009-0001-2022-7072"
# Portaria n.º 332-A/2026/1 (11 August 2026): the Politécnico do Porto became the Universidade Técnica do Porto,
# with the Faculdade de Media Artes e Design (FMAD) among its units.
AFFILIATION = "Faculdade de Media Artes e Design, Universidade Técnica do Porto"


def cff_field(text: str, key: str) -> str:
    """Value of a top-level scalar key in CITATION.cff (no YAML parser needed for these keys)."""
    match = re.search(rf"^{key}:\s*\"?([^\"\n]+)\"?\s*$", text, flags=re.MULTILINE)
    if match is None:
        raise KeyError(key)
    return match.group(1).strip()


class DataManifestTests(unittest.TestCase):
    def test_every_pinned_file_matches_its_checksum(self) -> None:
        self.assertEqual(reproduce.check_manifest(), [])
        self.assertEqual(len(reproduce.read_manifest()), 63)

    def test_a_changed_or_extra_file_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            shutil.copytree(reproduce.DATA, data)
            first = sorted(p for p in data.iterdir() if p.name != "SHA256SUMS")[0]
            first.write_bytes(first.read_bytes() + b"\n")
            (data / "extra.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(reproduce, "DATA", data), mock.patch.object(reproduce, "MANIFEST", data / "SHA256SUMS"):
                problems = reproduce.check_manifest()
        self.assertIn(f"checksum differs: {first.name}", problems)
        self.assertIn("not in the manifest: extra.json", problems)

    def test_manifest_is_in_sha256sum_format(self) -> None:
        for line in (HERE / "data" / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            self.assertRegex(line, r"^[0-9a-f]{64}  [^/\s]+$")


class MetadataTests(unittest.TestCase):
    def test_citation_and_zenodo_metadata_agree(self) -> None:
        cff = (HERE / "CITATION.cff").read_text(encoding="utf-8")
        zenodo = json.loads((HERE / ".zenodo.json").read_text(encoding="utf-8"))
        self.assertEqual(cff_field(cff, "cff-version"), "1.2.0")
        self.assertEqual(cff_field(cff, "version"), VERSION)
        self.assertEqual(zenodo["version"], VERSION)
        self.assertEqual(cff_field(cff, "title"), zenodo["title"])
        self.assertEqual(cff_field(cff, "license"), zenodo["license"])
        self.assertIn(f"https://orcid.org/{ORCID}", cff)
        self.assertEqual([c["orcid"] for c in zenodo["creators"]], [ORCID])
        self.assertIn(f"-v{VERSION}", zenodo["related_identifiers"][0]["identifier"])
        self.assertEqual(cff.count(f'affiliation: "{AFFILIATION}"'), 2)
        self.assertEqual([c["affiliation"] for c in zenodo["creators"]], [AFFILIATION])
        self.assertIn("email: dfr@esmad.ipp.pt", cff)

    def test_no_release_date_or_doi_before_the_release_exists(self) -> None:
        """The date and DOI are set when the tag and the Zenodo deposit exist; none may be invented before."""
        cff = (HERE / "CITATION.cff").read_text(encoding="utf-8")
        zenodo = json.loads((HERE / ".zenodo.json").read_text(encoding="utf-8"))
        self.assertNotIn("date-released", cff)
        self.assertNotIn("doi", cff.lower().replace("doi.org/10.2908", ""))
        self.assertNotIn("publication_date", zenodo)
        self.assertNotIn("doi", zenodo)
        for name in ("README.md", "REPORT.md", "RELEASE_NOTES_v1.0.0.md", "RELEASE_CHECKLIST_v1.0.0.md"):
            with self.subTest(file=name):
                self.assertNotRegex((HERE / name).read_text(encoding="utf-8"), r"10\.5281/zenodo")

    def test_licence_statements_do_not_relicense_third_party_data(self) -> None:
        zenodo = json.loads((HERE / ".zenodo.json").read_text(encoding="utf-8"))
        self.assertIn("Third-party data are not relicensed", zenodo["description"])
        readme = (HERE / "README.md").read_text(encoding="utf-8")
        self.assertIn("are **not** relicensed under Apache-2.0", readme)
        self.assertIn("are **not** relicensed under Apache-2.0", (HERE / "DATA_SOURCES.md").read_text(encoding="utf-8"))

    def test_release_documents_name_the_version(self) -> None:
        header = "\n".join((HERE / "REPORT.md").read_text(encoding="utf-8").splitlines()[:10])
        self.assertIn(f"Research release **v{VERSION}**", header)
        self.assertIn(AFFILIATION, header.replace("\n", " "))
        notes = HERE / f"RELEASE_NOTES_v{VERSION}.md"
        self.assertTrue(notes.exists())
        self.assertIn(f"v{VERSION}", notes.read_text(encoding="utf-8"))
        self.assertTrue((HERE / "REVIEW.md").exists())

    def test_every_figure_is_used_and_every_figure_script_is_reproduced(self) -> None:
        docs = "".join((HERE / name).read_text(encoding="utf-8") for name in ("README.md", "REPORT.md", "article.md"))
        figures = sorted((HERE / "figures").glob("*.png"))
        self.assertEqual(len(figures), 21)
        for fig in figures:
            with self.subTest(figure=fig.name):
                self.assertIn(fig.name, docs)
        scripts = sorted(p.name for p in HERE.glob("*_figures.py"))
        self.assertEqual(sorted(reproduce.FIGURE_SCRIPTS), sorted(scripts))


if __name__ == "__main__":
    unittest.main()
