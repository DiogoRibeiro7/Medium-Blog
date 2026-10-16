import pandas as pd

from hansard_drift.roles import assign_era, assign_side, classify_roles, normalise_party, order_keys


def frame(rows):
    return pd.DataFrame(rows, columns=["speakername", "speech_class", "parliamentary_posts"])


def test_chair_titles_and_posts():
    df = classify_roles(
        frame(
            [
                ("Mr. Speaker", "Speech", []),
                ("Madam Deputy Speaker (Dame Eleanor Laing)", "Speech", []),
                ("The Temporary Chairman (Mr. Jimmy Hood)", "Speech", []),
                ("The First Deputy Chairman", "Speech", []),
                ("Mr. Speaker-Elect", "Speech", []),
                ("Mr. Paul Dean", "Speech", ["Deputy Speaker and First Deputy Chairman of Ways and Means"]),
                ("Mr. Dennis Skinner", "Speech", []),
                ("Hon. Members", "Speech", []),
                ("Several hon. Members", "Speech", []),
                ("", "Procedural", []),
            ]
        )
    )
    assert df["role"].tolist() == [
        "chair",
        "chair",
        "chair",
        "chair",
        "member",
        "chair",
        "member",
        "floor",
        "floor",
        "procedural",
    ]
    assert df["chair_kind"].tolist()[:6] == ["speaker", "deputy", "committee", "committee", "", "deputy"]
    assert df["role_source"].tolist()[5] == "post"


def test_parties_and_sides(cfg):
    party = pd.Series(["Labour (Co-op)", "Conservative", "Liberal Democrat", "Scottish National Party", "", "Speaker"])
    norm = party.map(normalise_party)
    assert norm.tolist()[:3] == ["Labour", "Conservative", "Liberal Democrat"]
    dates = pd.Series(pd.to_datetime(["2012-01-10"] * 6))
    assert assign_side(norm, dates, cfg).tolist() == [
        "Opposition",
        "Government",
        "Government",
        "Opposition",
        "Other",
        "Other",
    ]
    later = pd.Series(pd.to_datetime(["2016-01-10"] * 6))
    assert assign_side(norm, later, cfg).tolist()[2] == "Opposition"


def test_eras(cfg):
    d = pd.Series(pd.to_datetime(["1979-05-09", "1983-06-14", "1983-06-15", "2009-06-22", "2021-04-29", "2021-05-01"]))
    assert assign_era(d, cfg).tolist()[:5] == ["Thomas", "Thomas", "Weatherill", "Bercow", "Hoyle"]
    assert pd.isna(assign_era(d, cfg).iloc[5])


def test_order_keys_sort_by_sitting_column_paragraph():
    ids = pd.Series(
        [
            "uk.org.publicwhip/debate/1990-01-10a.12.3",
            "uk.org.publicwhip/debate/1990-01-10a.9.7",
            "uk.org.publicwhip/debate/1990-01-10b.1.1",
            "uk.org.publicwhip/debate/1990-01-10a.12.10",
        ]
    )
    k = order_keys(ids)
    order = k.sort_values(["k_letter", "k_col", "k_para"]).index.tolist()
    assert order == [1, 0, 3, 2]
    assert k["k_parsed"].all()


def test_party_speaker_counts_only_for_the_reigning_speaker(cfg):
    from hansard_drift.roles import speaker_surnames

    df = pd.DataFrame(
        {
            "speakername": ["John Bercow", "Mr. Denzil Davies", "Lindsay Hoyle", "Lindsay Hoyle"],
            "display_as": ["John Bercow", "Denzil Davies", "Lindsay Hoyle", "Lindsay Hoyle"],
            "party": ["Speaker", "Speaker", "Labour", "Speaker"],
            "speech_class": ["Speech"] * 4,
            "parliamentary_posts": [[], [], [], []],
            "date": pd.to_datetime(["2017-03-01", "1985-03-01", "2017-03-01", "2020-03-01"]),
        }
    )
    out = classify_roles(df, speaker_surname=speaker_surnames(df["date"], cfg))
    assert out["role"].tolist() == ["chair", "member", "member", "chair"]
    assert out["role_source"].tolist() == ["party", "none", "none", "party"]


def test_rostered_deputy_counts_only_inside_the_term(cfg):
    from hansard_drift.roles import roster_match

    df = pd.DataFrame(
        {
            "speakername": ["Dawn Primarolo", "Dawn Primarolo", "Eleanor Laing", "Eleanor Laing"],
            "display_as": ["Dawn Primarolo", "Dawn Primarolo", "Eleanor Laing", "Eleanor Laing"],
            "party": ["Labour", "Labour", "Conservative", "Conservative"],
            "speech_class": ["Speech"] * 4,
            "parliamentary_posts": [[], [], [], []],
            "date": pd.to_datetime(["2012-05-01", "2008-05-01", "2021-02-01", "2011-02-01"]),
        }
    )
    out = classify_roles(df, roster_hit=roster_match(df["display_as"], df["date"], cfg))
    assert out["role"].tolist() == ["chair", "member", "chair", "member"]
    assert out["role_source"].tolist() == ["roster", "none", "roster", "none"]
