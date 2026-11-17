import pytest

from clinical_lora.parsing import parse_extraction

GOOD = '{"chemicals": ["aspirin"], "diseases": ["stroke"]}'


def test_valid_json_is_strict_valid():
    r = parse_extraction(GOOD)
    assert r.valid and not r.repaired and r.parsed
    assert r.extraction.as_dict() == {"chemicals": ["aspirin"], "diseases": ["stroke"]}


@pytest.mark.parametrize(
    "text",
    [
        "Sure! Here you go:\n```json\n" + GOOD + "\n```\nHope that helps.",
        "Answer: " + GOOD + " trailing prose",
    ],
)
def test_extracts_json_from_prose_and_fences(text):
    r = parse_extraction(text)
    assert r.valid and r.extraction.chemicals == ["aspirin"]


@pytest.mark.parametrize(
    "text, chem, dis",
    [
        ('{"chemicals": ["a", "b",], "diseases": ["x"],}', ["a", "b"], ["x"]),  # trailing commas
        ("{'chemicals': ['a'], 'diseases': []}", ["a"], []),  # single quotes
        ('{"chemicals": ["a", "b"], "diseases": ["x", "y', ["a", "b"], ["x", "y"]),  # truncated
        ('{"chemicals": ["a"], "diseases":', ["a"], []),  # dangling key
        ('{"chemicals": ["a"]', ["a"], []),  # missing brace; missing key -> []
    ],
)
def test_repair(text, chem, dis):
    r = parse_extraction(text)
    assert r.repaired and not r.valid and r.parsed, r.error
    assert (r.extraction.chemicals, r.extraction.diseases) == (chem, dis)


@pytest.mark.parametrize(
    "text", ["", "no json here", "[1, 2, 3]", '{"foo": 1}', '{"chemicals": "a"}']
)
def test_unparseable(text):
    r = parse_extraction(text)
    assert not r.parsed and not r.valid and r.error


def test_entities_deduped_and_whitespace_normalised():
    r = parse_extraction('{"chemicals": ["a  b", "a b", ""], "diseases": []}')
    assert r.extraction.chemicals == ["a b"]
