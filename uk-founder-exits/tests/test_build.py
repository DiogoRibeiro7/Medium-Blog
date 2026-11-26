"""The crawl, the PSC extract and the cohort build on synthetic Companies House records."""

import io
import json
import sys
import tempfile
import unittest
import urllib.error
import zipfile
from datetime import date
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ch_api  # noqa: E402
import ch_build as cb  # noqa: E402
import ch_bulk  # noqa: E402

CENSOR = date(2026, 11, 25)


def individual(nationality, notified="2018-01-10", residence="England", ceased=None):
    return {
        "kind": "individual-person-with-significant-control",
        "notified_on": notified,
        "ceased_on": ceased,
        "natures": ["ownership-of-shares-25-to-50-percent"],
        "nationality": nationality,
        "residence": residence,
    }


def corporate(name, country, notified, number=None, ceased=None, natures=None):
    return {
        "kind": "corporate-entity-person-with-significant-control",
        "notified_on": notified,
        "ceased_on": ceased,
        "natures": natures or ["ownership-of-shares-75-to-100-percent"],
        "name": name,
        "country_registered": country,
        "place_registered": None,
        "legal_authority": None,
        "registration_number": number,
    }


def search_item(name="Acme Robotics Ltd", inc="2018-01-05", status="active", ceased=None, postcode="EC1A 1BB"):
    return {
        "company_number": "11111111",
        "company_name": name,
        "company_status": status,
        "company_type": "ltd",
        "date_of_creation": inc,
        "date_of_cessation": ceased,
        "sic_codes": ["62012"],
        "postal_code": postcode,
    }


def sh01(d):
    return {"type": "SH01", "category": "capital", "date": d, "action_date": d}


class Names(unittest.TestCase):
    def test_similar_names(self):
        self.assertTrue(cb.similar_names("Acme Robotics Ltd", "ACME ROBOTICS INC."))
        self.assertTrue(cb.similar_names("Acme Robotics Ltd", "Acme Robotics Holdings Limited"))
        self.assertTrue(cb.similar_names("Acme Ltd", "Acme Robotics Group Inc"))
        self.assertFalse(cb.similar_names("Acme Robotics Ltd", "Globex Corporation"))
        self.assertFalse(cb.similar_names("Holdings Ltd", "Globex Ltd"))

    def test_nationality_groups(self):
        cases = {
            "British": "British",
            "English": "British",
            "Welsh": "British",
            "British,Irish": "British",
            "Northern Irish": "British",
            "Irish": "EU and EEA",
            "French": "EU and EEA",
            "American": "North America",
            "Indian": "India",
            "Chinese": "China",
            "Nigerian": "other",
            "": "unknown",
        }
        for nat, group in cases.items():
            self.assertEqual(cb.nationality_group(nat), group, nat)

    def test_founder_group(self):
        self.assertEqual(cb.founder_group(["British", "English"]), ("British", "British"))
        self.assertEqual(cb.founder_group(["British", "French"]), ("mixed", "mixed"))
        self.assertEqual(cb.founder_group(["French", "German"]), ("non-British", "EU and EEA"))
        self.assertEqual(cb.founder_group(["French", "Indian"]), ("non-British", "several"))
        self.assertEqual(cb.founder_group(["British", None]), ("British", "British"))
        self.assertEqual(cb.founder_group([None]), (None, None))

    def test_region(self):
        self.assertEqual(cb.region("EC1A 1BB"), "London")
        self.assertEqual(cb.region("SW1A 2AA"), "London")
        self.assertEqual(cb.region("EN1 1AA"), "rest of UK")
        self.assertEqual(cb.region("CB2 1TN"), "Oxford and Cambridge")
        self.assertEqual(cb.region(None), "unknown")


class Registration(unittest.TestCase):
    def test_uk_numbers_are_normalised(self):
        self.assertEqual(ch_bulk.uk_registration_number(corporate("X", "England", None, "1234567")), "01234567")
        self.assertEqual(ch_bulk.uk_registration_number(corporate("X", "Scotland", None, "SC 123456")), "SC123456")
        self.assertIsNone(ch_bulk.uk_registration_number(corporate("X", "Delaware", None, "1234567")))

    def test_slim_drops_names_of_individuals(self):
        rec = {
            "kind": "individual-person-with-significant-control",
            "name": "Jane Doe",
            "name_elements": {"forename": "Jane"},
            "date_of_birth": {"month": 1, "year": 1980},
            "nationality": "British",
            "country_of_residence": "England",
            "natures_of_control": ["ownership-of-shares-25-to-50-percent"],
            "notified_on": "2018-01-10",
        }
        out = ch_bulk.slim(rec)
        self.assertNotIn("name", out)
        self.assertNotIn("date_of_birth", out)
        self.assertEqual(out["nationality"], "British")

    def test_extract_reads_the_zip(self):
        lines = [
            {"company_number": "11111111", "data": {"kind": "individual-person-with-significant-control"}},
            {"company_number": "22222222", "data": {"kind": "individual-person-with-significant-control"}},
            {"company_number": "11111111", "data": {"kind": "corporate-entity-person-with-significant-control"}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(tmp) / "psc.zip"
            with zipfile.ZipFile(zpath, "w") as z:
                z.writestr("psc.txt", "\n".join(json.dumps(x) for x in lines) + "\nnot json\n")
            out = ch_bulk.extract(zpath, {"11111111"})
        self.assertEqual(list(out), ["11111111"])
        self.assertEqual(len(out["11111111"]), 2)


class Classify(unittest.TestCase):
    def test_foreign_owner_with_another_name(self):
        r = corporate("Globex Corporation", "Delaware", "2021-03-01")
        self.assertEqual(cb.classify("Acme Robotics Ltd", date(2021, 3, 1), r, {})[0], "foreign_acquisition")

    def test_foreign_parent_with_the_same_name_is_a_flip(self):
        r = corporate("Acme Robotics Inc.", "Delaware", "2021-03-01")
        self.assertEqual(cb.classify("Acme Robotics Ltd", date(2021, 3, 1), r, {})[0], "flip")

    def test_offshore(self):
        r = corporate("Acme Robotics Holdings", "Jersey", "2021-03-01")
        self.assertEqual(cb.classify("Acme Robotics Ltd", date(2021, 3, 1), r, {})[0], "offshore")

    def test_uk_owner_with_and_without_a_foreign_parent(self):
        r = corporate("Initech Ltd", "England And Wales", "2021-03-01", "5555555")
        self.assertEqual(cb.classify("Acme Robotics Ltd", date(2021, 3, 1), r, {})[0], "uk_acquisition")
        lookup = {"05555555": [corporate("Initech Global Inc", "California", "2015-01-01")]}
        self.assertEqual(cb.classify("Acme Robotics Ltd", date(2021, 3, 1), r, lookup)[0], "foreign_acquisition")
        # A foreign parent two UK levels up still counts.
        lookup = {
            "05555555": [corporate("Initech Midco Ltd", "England", "2015-01-01", "6666666")],
            "06666666": [corporate("Initech Global Inc", "California", "2015-01-01")],
        }
        self.assertEqual(cb.classify("Acme Robotics Ltd", date(2021, 3, 1), r, lookup)[0], "foreign_acquisition")

    def test_uk_holding_company_is_followed(self):
        holdco = corporate("Acme Robotics Holdings Ltd", "England", "2020-01-01", "7777777")
        lookup = {
            "07777777": [
                individual("British", notified="2019-12-01"),
                corporate("Globex Corporation", "Delaware", "2022-05-01"),
            ]
        }
        kind, when = cb.classify("Acme Robotics Ltd", date(2020, 1, 1), holdco, lookup)
        self.assertEqual((kind, when), ("foreign_acquisition", date(2022, 5, 1)))
        self.assertEqual(cb.classify("Acme Robotics Ltd", date(2020, 1, 1), holdco, {})[0], "reorganisation")


class BuildRow(unittest.TestCase):
    def test_a_company_sold_abroad(self):
        records = [individual("British"), individual("English"), corporate("Globex Corp", "Delaware", "2022-05-01")]
        items = [sh01("2018-06-01"), sh01("2018-09-01"), sh01("2020-02-01"), {"type": "RES01", "category": "resolution",
                 "date": "2018-06-03"}]  # fmt: skip
        row, why = cb.build_row("11111111", search_item(), items, records, {}, CENSOR)
        self.assertIsNone(why)
        self.assertEqual(row["entry"], "2018-06-01")
        self.assertEqual(row["round_entry"], "2018-06-01")
        self.assertEqual(row["foreign_acquisition"], "2022-05-01")
        self.assertEqual(row["founder_group"], "British")
        self.assertEqual(row["founders"], 2)
        self.assertEqual(row["allotments_first_year"], 2)
        self.assertEqual(row["next_allotment"], "2020-02-01")
        self.assertEqual(row["region"], "London")

    def test_closure_takes_the_first_insolvency_filing(self):
        items = [sh01("2018-06-01"), {"type": "LIQ02", "category": "insolvency", "date": "2021-04-01"}]
        s = search_item(status="dissolved", ceased="2023-01-10")
        row, _ = cb.build_row("11111111", s, items, [individual("Indian", residence="India")], {}, CENSOR)
        self.assertEqual(row["closure"], "2021-04-01")
        founder = (row["founder_group"], row["founder_origin"], row["founder_abroad"])
        self.assertEqual(founder, ("non-British", "India", 1))

    def test_exclusions(self):
        f = [individual("British")]
        cases = [
            (search_item(), [], f, "no allotment of shares within three years"),
            (search_item(), [sh01("2022-01-01")], f, "no allotment of shares within three years"),
            (search_item(), [sh01("2018-06-01")], [], "no individual founder recorded"),
            (search_item(), [sh01("2018-06-01")], [individual("")], "founder nationality not recorded"),
            (search_item(), [sh01("2018-06-01")], f + [corporate("P Ltd", "England", "2018-01-10", "1")],
             "majority corporate controller at incorporation"),
        ]  # fmt: skip
        for s, items, records, reason in cases:
            self.assertEqual(cb.build_row("11111111", s, items, records, {}, CENSOR), (None, reason))


class FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class Crawl(unittest.TestCase):
    def test_months_cover_the_window(self):
        m = list(ch_api.months(ch_api.COHORT_FROM, ch_api.COHORT_TO))
        self.assertEqual(len(m), 54)
        self.assertEqual(m[0], (date(2016, 7, 1), date(2016, 7, 31)))
        self.assertEqual(m[-1], (date(2020, 12, 1), date(2020, 12, 31)))

    def test_client_treats_404_as_an_answer_and_retries_5xx(self):
        calls = []

        def opener(req, timeout):
            calls.append(req.full_url)
            if len(calls) == 1:
                raise urllib.error.HTTPError(req.full_url, 502, "bad gateway", {}, None)
            if "missing" in req.full_url:
                raise urllib.error.HTTPError(req.full_url, 404, "not found", {}, None)
            return FakeResponse(b'{"ok": 1}')

        client = ch_api.Client("key", min_interval=0, opener=opener)
        with mock.patch.object(ch_api.time, "sleep"):
            self.assertEqual(client.get("/x"), (200, {"ok": 1}))
            self.assertEqual(client.get("/missing"), (404, None))
        self.assertEqual(len(calls), 3)
        self.assertTrue(client.auth.startswith("Basic "))

    def test_search_and_filings_resume(self):
        class Fake:
            calls = 0

            def get(self, path, params=None):
                self.calls += 1
                if path.startswith("/advanced-search"):
                    if params["sic_codes"] != "62012" or params["incorporated_from"] != "2018-01-01":
                        return 404, None
                    start = params["start_index"]
                    items = [dict(search_item(), company_number=f"{i:08d}") for i in range(3)]
                    return 200, {"hits": 3, "items": items[start : start + 2]}
                return 200, {"total_count": 1, "items": [sh01("2018-06-01")]}

        with tempfile.TemporaryDirectory() as tmp:
            db = ch_api.connect(str(Path(tmp) / "c.sqlite"))
            fake = Fake()
            self.assertTrue(ch_api.crawl_search(db, fake, deadline=float("inf")))
            self.assertEqual(sorted(ch_api.cohort_companies(db)), ["00000000", "00000001", "00000002"])
            self.assertFalse(ch_api.crawl_filings(db, fake, deadline=float("-inf")))
            self.assertTrue(ch_api.crawl_filings(db, fake, deadline=float("inf")))
            before = fake.calls
            self.assertTrue(ch_api.crawl_search(db, fake, deadline=float("inf")))
            self.assertTrue(ch_api.crawl_filings(db, fake, deadline=float("inf")))
            self.assertEqual(fake.calls, before)  # nothing is fetched twice
            self.assertTrue(ch_api.progress(db)["crawl_complete"])


class EndToEnd(unittest.TestCase):
    def test_build_reports_exclusions(self):
        search = {"11111111": search_item(), "22222222": dict(search_item(), company_number="22222222")}
        filings = {"11111111": [sh01("2018-06-01")]}
        psc = {"snapshot": "x-2026-11-25.zip", "cohort": {"11111111": [individual("British")]}, "parents": {}}
        rows, why = cb.build(search, filings, psc, CENSOR)
        self.assertEqual(len(rows), 1)
        self.assertEqual(dict(why), {"filing history not fetched": 1})


if __name__ == "__main__":
    unittest.main()
