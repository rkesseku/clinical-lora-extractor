import json

import pytest

from clinical_lora.data import (
    Example,
    bio_to_entities,
    rows_to_examples,
    spans_to_entities,
    subsample,
    target_json,
)
from clinical_lora.prompts import build_messages, select_shots


def test_spans_to_entities_orders_dedupes_and_ignores_unknown():
    text = "Naloxone and clonidine; naloxone again. Hypertension, hypertension."

    def span(word, nth=0):
        i = -1
        for _ in range(nth + 1):
            i = text.index(word, i + 1)
        return i, i + len(word)

    spans = [
        (*span("naloxone"), "chemical"),  # lowercase mention -> distinct surface form
        (*span("Naloxone"), "Chemical"),
        (*span("clonidine"), "Chemical"),
        (*span("Hypertension"), "Disease"),
        (*span("hypertension"), "Disease"),
        (*span("Naloxone"), "Gene"),
    ]
    ents = spans_to_entities(text, spans)
    assert ents["chemicals"] == ["Naloxone", "clonidine", "naloxone"]
    assert ents["diseases"] == ["Hypertension", "hypertension"]


def test_spans_out_of_range_raises():
    with pytest.raises(ValueError):
        spans_to_entities("abc", [(0, 10, "Chemical")])


def test_bio_to_entities_multiword_and_adjacent():
    toks = ["Cisplatin", "caused", "acute", "renal", "failure", "and", "ototoxicity"]
    tags = ["B-Chemical", "O", "B-Disease", "I-Disease", "I-Disease", "O", "B-Disease"]
    ents = bio_to_entities(toks, tags)
    assert ents == {"chemicals": ["Cisplatin"], "diseases": ["acute renal failure", "ototoxicity"]}


def test_target_json_is_canonical_and_round_trips():
    s = target_json({"chemicals": ["a"], "diseases": []})
    assert json.loads(s) == {"chemicals": ["a"], "diseases": []}
    assert list(json.loads(s)) == ["chemicals", "diseases"]


def test_rows_to_examples_bigbio_and_tner():
    big = {
        "passages": [
            {
                "text": ["Aspirin study"],
                "offsets": [[0, 13]],
                "entities": [{"type": "Chemical", "text": ["Aspirin"]}],
            },
            {
                "text": ["Caused stroke."],
                "offsets": [[14, 28]],
                "entities": [{"type": "Disease", "text": ["stroke"]}],
            },
        ]
    }
    tner = {"tokens": ["Lidocaine", "caused", "seizures", "."], "tags": [1, 0, 2, 0]}
    a, b = rows_to_examples([big])[0], rows_to_examples([tner])[0]
    assert a.text == "Aspirin study Caused stroke."
    assert a.entities == {"chemicals": ["Aspirin"], "diseases": ["stroke"]}
    assert b.text == "Lidocaine caused seizures."
    assert b.entities == {"chemicals": ["Lidocaine"], "diseases": ["seizures"]}
    with pytest.raises(ValueError):
        rows_to_examples([{"foo": 1}])


def test_subsample_deterministic():
    ex = [Example(str(i), {}) for i in range(50)]
    a, b = subsample(ex, 10, seed=1), subsample(ex, 10, seed=1)
    assert [e.text for e in a] == [e.text for e in b] and len(a) == 10
    assert len(subsample(ex, None)) == 50


def test_prompt_messages_with_shots():
    shots = [Example("s", {"chemicals": ["x"], "diseases": []})]
    msgs = build_messages("doc", select_shots(shots, 1))
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert json.loads(msgs[2]["content"]) == {"chemicals": ["x"], "diseases": []}
