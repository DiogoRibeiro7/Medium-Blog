"""Deterministic tests of transient and permanent public-source failures."""

from __future__ import annotations

import socket
import unittest
from email.message import Message
from http.client import IncompleteRead
from urllib.error import HTTPError
from urllib.request import Request

from resilient_fetch import fetch_bytes


class FakeResponse:
    """Context-managed response with a test-controlled body and headers."""

    def __init__(
        self, payload: bytes, *, expected_length: int | None = None,
        content_type: str = "text/csv",
    ) -> None:
        self.payload = payload
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        if expected_length is not None:
            self.headers["Content-Length"] = str(expected_length)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class RetryTests(unittest.TestCase):
    """Validate exact returned bytes without calling public networks."""

    def setUp(self) -> None:
        self.data = b"Entity,Code,Year,value\n" + b"Country,AAA,2024,40\n" * 8
        self.url = "https://example.org/research.csv"

    def test_first_attempt_returns_exact_bytes(self) -> None:
        attempts: list[float] = []

        def opener(request: Request, *, timeout: float) -> FakeResponse:
            self.assertEqual(request.full_url, self.url)
            attempts.append(timeout)
            return FakeResponse(self.data, expected_length=len(self.data))

        value = fetch_bytes(self.url, name="source.csv", opener=opener)
        self.assertEqual(value, self.data)
        self.assertEqual(attempts, [120.0])

    def test_timeout_then_success_retries_once(self) -> None:
        count = 0
        delays: list[float] = []

        def opener(_request: Request, *, timeout: float) -> FakeResponse:
            nonlocal count
            count += 1
            if count == 1:
                raise socket.timeout("fixture timeout")
            return FakeResponse(self.data)

        result = fetch_bytes(
            self.url, name="source.csv", opener=opener, sleeper=delays.append
        )
        self.assertEqual(result, self.data)
        self.assertEqual(count, 2)
        self.assertEqual(delays, [3.0])

    def test_permanent_404_does_not_retry(self) -> None:
        count = 0

        def opener(_request: Request, *, timeout: float) -> FakeResponse:
            nonlocal count
            count += 1
            raise HTTPError(self.url, 404, "Not Found", {}, None)

        with self.assertRaisesRegex(ValueError, "HTTP 404"):
            fetch_bytes(self.url, name="source.csv", opener=opener)
        self.assertEqual(count, 1)

    def test_transient_503_retry(self) -> None:
        count = 0

        def opener(_request: Request, *, timeout: float) -> FakeResponse:
            nonlocal count
            count += 1
            if count == 1:
                raise HTTPError(self.url, 503, "Busy", {}, None)
            return FakeResponse(self.data)

        self.assertEqual(
            fetch_bytes(self.url, name="source.csv", opener=opener, sleeper=lambda _: None),
            self.data,
        )
        self.assertEqual(count, 2)

    def test_exhausted_timeout_raises_actionable_error(self) -> None:
        count = 0

        def opener(_request: Request, *, timeout: float) -> FakeResponse:
            nonlocal count
            count += 1
            raise TimeoutError("delayed")

        with self.assertRaisesRegex(RuntimeError, "after 2 attempts"):
            fetch_bytes(
                self.url, name="source.csv", opener=opener, sleeper=lambda _: None
            )
        self.assertEqual(count, 2)

    def test_content_length_mismatch_triggers_retry(self) -> None:
        count = 0

        def opener(_request: Request, *, timeout: float) -> FakeResponse:
            nonlocal count
            count += 1
            expected = len(self.data) + 7 if count == 1 else len(self.data)
            return FakeResponse(self.data, expected_length=expected)

        result = fetch_bytes(
            self.url, name="source.csv", opener=opener, sleeper=lambda _: None
        )
        self.assertEqual(result, self.data)
        self.assertEqual(count, 2)

    def test_html_error_page_rejected(self) -> None:
        html = b"<!doctype html>" + b"no results" * 30

        def opener(_request: Request, *, timeout: float) -> FakeResponse:
            return FakeResponse(html, content_type="text/html")

        with self.assertRaisesRegex(ValueError, "HTML"):
            fetch_bytes(self.url, name="source.csv", opener=opener)

    def test_short_body_rejected(self) -> None:
        def opener(_request: Request, *, timeout: float) -> FakeResponse:
            return FakeResponse(b"bad")

        with self.assertRaisesRegex(ValueError, "short"):
            fetch_bytes(self.url, name="source.csv", opener=opener)

    def test_invalid_attempts_rejected(self) -> None:
        for attempts in (0, -1, True):
            with self.subTest(attempts=attempts), self.assertRaises(ValueError):
                fetch_bytes(self.url, name="source.csv", max_attempts=attempts)

    def test_invalid_url_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "HTTP"):
            fetch_bytes("file:///tmp/source.csv", name="source.csv")


if __name__ == "__main__":
    unittest.main()
