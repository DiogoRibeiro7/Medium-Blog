import pytest

from hansard_drift.dictionary import normalise, tag_text


def hits(text):
    return {k: v for k, v in tag_text(normalise(text)).items() if v}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Order. Order.", {"ORDER": 2}),
        ("Order, order.", {"ORDER": 2}),
        ("Order! Order! Order!", {"ORDER": 3}),
        ("Order - order!", {"ORDER": 2}),
        ("Order. I call Mr. Smith.", {"ORDER": 1}),
    ],
)
def test_each_call_of_order_is_counted(text, expected):
    assert hits(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Standing Order No. 24.",
        "On a point of order, Mr. Speaker.",
        "We now come to the motion on the Order Paper.",
        "Orders of the Day",
        "I call it in order.",
        "I call Mr. Smith.",
    ],
)
def test_procedural_uses_of_order_are_not_calls(text):
    assert hits(text) == {}


def test_categories():
    assert hits("Order. The hon. Gentleman must resume his seat.") == {"ORDER": 1, "SEAT": 1}
    assert hits("I name the hon. Member for Bolsover, Mr. Dennis Skinner.") == {"NAME": 1}
    assert hits("I ask the hon. Gentleman to withdraw that remark.") == {"WITHDRAW": 1}
    assert hits("I order the hon. Member to withdraw immediately from the House.") == {"SUSPEND": 1}
    assert hits("That is not a point of order.") == {"NOT_POO": 1}
    assert hits("That language is unparliamentary.") == {"ADMONISH": 1}
    assert hits("There is far too much noise.") == {"NOISE": 1}
    assert hits("Short questions and short answers, please.") == {"HURRY": 2}


def test_normalise():
    assert normalise("  The  Hon. Gentleman ’s ") == "the hon gentleman 's"
    assert normalise(None) == ""
