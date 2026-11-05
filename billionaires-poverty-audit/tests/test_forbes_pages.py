"""Synthetic tests for Forbes page fallback; no external requests are made."""

from __future__ import annotations

import json
import unittest
from urllib.parse import parse_qs, urlsplit

from forbes_pages import ForbesSourceError, get_people, page_url, retrieve_annual_list


URL = "https://www.forbes.com/forbesapi/person/billionaires/2026/position/true.json?limit=5000"


def encode_people(start: int, n: int, *, array_key: str = "persons") -> bytes:
    """Create explicitly fictional person-list records with unique identifiers."""
    return json.dumps({
        "personList": {array_key: [
            {"uri": f"fictional-{i}", "countryOfCitizenship": "Example"}
            for i in range(start, start + n)
        ]}
    }).encode("utf-8")


class PageFallbackTests(unittest.TestCase):
    """Ensure errors cannot turn partial feeds into plausible full results."""

    def test_build_page_query_preserves_annual_path(self) -> None:
        url = page_url(URL, start=25, limit=10)
        self.assertEqual(parse_qs(urlsplit(url).query)["start"], ["25"])
        self.assertEqual(parse_qs(urlsplit(url).query)["limit"], ["10"])
        self.assertIn("/billionaires/2026/", urlsplit(url).path)

    def test_single_response_works_without_pagination(self) -> None:
        calls: list[str] = []

        def retrieve(url: str, **kwargs: object) -> bytes:
            calls.append(url)
            return encode_people(0, 6)

        data, ledger = retrieve_annual_list(
            URL, expected_count=6, fetcher=retrieve
        )
        self.assertEqual(ledger["mode"], "single_response")
        self.assertEqual(len(json.loads(data)["personList"]["persons"]), 6)
        self.assertEqual(calls, [URL])

    def test_plural_persons_lists_is_explicitly_supported(self) -> None:
        """Forbes has also used `personList.personsLists` for person arrays."""
        def retrieve(url: str, **kwargs: object) -> bytes:
            return encode_people(0, 6, array_key="personsLists")

        data, ledger = retrieve_annual_list(
            URL, expected_count=6, fetcher=retrieve
        )
        self.assertEqual(ledger["mode"], "single_response")
        self.assertEqual(len(get_people(data)), 6)

    def test_plural_paginated_response_preserves_uniqueness_checks(self) -> None:
        """The paginated fallback must accept the same explicit array path."""
        def retrieve(url: str, **kwargs: object) -> bytes:
            if url == URL:
                raise RuntimeError("HTTP 503 first byte timeout")
            parameters = parse_qs(urlsplit(url).query)
            return encode_people(
                int(parameters["start"][0]),
                int(parameters["limit"][0]),
                array_key="personsLists",
            )

        data, ledger = retrieve_annual_list(
            URL, expected_count=7, page_limit=3, fetcher=retrieve
        )
        self.assertEqual(ledger["page_count"], 3)
        self.assertEqual(len(get_people(data)), 7)

    def test_ambiguous_array_keys_fail_closed(self) -> None:
        """Do not silently choose a person list if two plausible ones exist."""
        document = json.loads(encode_people(0, 6))
        document["personList"]["personsLists"] = document["personList"]["persons"]
        with self.assertRaisesRegex(ForbesSourceError, "ambiguous person arrays"):
            get_people(json.dumps(document).encode("utf-8"))

    def test_transient_503_falls_back_to_pages(self) -> None:
        requested: list[str] = []

        def retrieve(url: str, **kwargs: object) -> bytes:
            requested.append(url)
            if url == URL:
                raise RuntimeError("HTTP 503 first byte timeout")
            args = parse_qs(urlsplit(url).query)
            return encode_people(int(args["start"][0]), int(args["limit"][0]))

        data, ledger = retrieve_annual_list(
            URL, expected_count=7, page_limit=3, fetcher=retrieve
        )
        self.assertEqual(len(json.loads(data)["personList"]["persons"]), 7)
        self.assertEqual(ledger["page_count"], 3)
        self.assertEqual(ledger["mode"], "derived_from_paginated_responses")
        self.assertEqual(len(ledger["pages"]), 3)
        self.assertEqual(len(requested), 4)

    def test_unequal_page_length_fails(self) -> None:
        def retrieve(url: str, **kwargs: object) -> bytes:
            if url == URL:
                raise RuntimeError("HTTP 503")
            return encode_people(0, 1)

        with self.assertRaisesRegex(ForbesSourceError, "returned 1, expected 3"):
            retrieve_annual_list(URL, expected_count=6, page_limit=3, fetcher=retrieve)

    def test_duplicate_ids_across_pages_fail(self) -> None:
        def retrieve(url: str, **kwargs: object) -> bytes:
            if url == URL:
                raise RuntimeError("HTTP 503")
            return encode_people(0, 2)

        with self.assertRaisesRegex(ForbesSourceError, "duplicate identifiers"):
            retrieve_annual_list(URL, expected_count=4, page_limit=2, fetcher=retrieve)

    def test_unexpected_single_response_count_fails_closed(self) -> None:
        def retrieve(url: str, **kwargs: object) -> bytes:
            return encode_people(0, 3)

        with self.assertRaisesRegex(ForbesSourceError, "expected 6"):
            retrieve_annual_list(URL, expected_count=6, fetcher=retrieve)

    def test_invalid_json_does_not_fall_back(self) -> None:
        def retrieve(url: str, **kwargs: object) -> bytes:
            return b"<html>blocked</html>"

        with self.assertRaisesRegex(ForbesSourceError, "not JSON"):
            retrieve_annual_list(URL, expected_count=6, fetcher=retrieve)


if __name__ == "__main__":
    unittest.main()
