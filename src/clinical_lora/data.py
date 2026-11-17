"""BC5CDR loading and span -> JSON conversion.

BC5CDR is biomedical *literature* (PubMed abstracts, chemical + disease mentions). It is a public
proxy for clinical text: it contains no PHI and is not clinical-note data.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from .schema import ENTITY_TYPES, TYPE_TO_KEY

# Tried in order. ``datasets`` >= 4 no longer runs Hub loading scripts (``bc5cdr.py``), so the
# primary sources are plain data files that need no script: the raw JSON-lines files in the
# tner/bc5cdr repo, then the Hub's auto-converted parquet copy of the same data. The bigbio
# script dataset is kept last for environments that still pin ``datasets`` < 4.
_HF = "https://huggingface.co/datasets"
_TNER_RAW = f"{_HF}/tner/bc5cdr/resolve/main/dataset"
_TNER_PARQUET = f"{_HF}/tner/bc5cdr/resolve/refs%2Fconvert%2Fparquet/bc5cdr"

DATASET_CANDIDATES: tuple[tuple[str, str | None, dict[str, str] | None], ...] = (
    (
        "json",
        None,
        {
            "train": f"{_TNER_RAW}/train.json",
            "validation": f"{_TNER_RAW}/valid.json",
            "test": f"{_TNER_RAW}/test.json",
        },
    ),
    (
        "parquet",
        None,
        {s: f"{_TNER_PARQUET}/{s}/0000.parquet" for s in ("train", "validation", "test")},
    ),
    ("bigbio/bc5cdr", "bc5cdr_bigbio_kb", None),
)

TNER_LABELS = ["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"]


@dataclass
class Example:
    """One document: raw text plus gold entities (deduplicated, in order of first appearance)."""

    text: str
    entities: dict[str, list[str]] = field(default_factory=dict)

    def target_json(self) -> str:
        return target_json(self.entities)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        it = " ".join(it.split())
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


def spans_to_entities(text: str, spans: list[tuple[int, int, str]]) -> dict[str, list[str]]:
    """Convert character spans ``(start, end, type)`` to a ``{"chemicals", "diseases"}`` dict.

    Mentions are sliced from ``text``, ordered by start offset, whitespace-normalised and
    deduplicated per type. Unknown types are ignored; out-of-range spans raise ``ValueError``.
    """
    out: dict[str, list[tuple[int, str]]] = {k: [] for k in ENTITY_TYPES}
    for start, end, etype in spans:
        key = TYPE_TO_KEY.get(etype.strip().lower())
        if key is None:
            continue
        if not (0 <= start < end <= len(text)):
            raise ValueError(f"span ({start}, {end}) outside text of length {len(text)}")
        out[key].append((start, text[start:end]))
    return {k: _dedupe([m for _, m in sorted(v, key=lambda t: t[0])]) for k, v in out.items()}


def bio_to_entities(tokens: list[str], tags: list[str]) -> dict[str, list[str]]:
    """Convert a BIO-tagged token sequence (e.g. ``B-Chemical``/``I-Disease``) to entity strings."""
    found: dict[str, list[tuple[int, str]]] = {k: [] for k in ENTITY_TYPES}
    cur: list[str] = []
    cur_type: str | None = None
    cur_start = 0

    def flush():
        nonlocal cur, cur_type
        if cur and cur_type:
            found[cur_type].append((cur_start, " ".join(cur)))
        cur, cur_type = [], None

    for i, (tok, tag) in enumerate(zip(tokens, tags, strict=True)):
        if tag == "O" or "-" not in tag:
            flush()
            continue
        prefix, etype = tag.split("-", 1)
        key = TYPE_TO_KEY.get(etype.lower())
        if key is None:
            flush()
            continue
        if prefix == "B" or key != cur_type:
            flush()
            cur, cur_type, cur_start = [tok], key, i
        else:
            cur.append(tok)
    flush()
    return {k: _dedupe([m for _, m in v]) for k, v in found.items()}


def target_json(entities: dict[str, list[str]]) -> str:
    """Canonical JSON string used as the training target."""
    return json.dumps({k: list(entities.get(k, [])) for k in ENTITY_TYPES}, ensure_ascii=False)


def _detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    for p in (" .", " ,", " ;", " :", " )", " ?", " !"):
        text = text.replace(p, p.strip())
    return text.replace("( ", "(")


def _bigbio_row(row: dict) -> Example:
    passages = row["passages"]
    text = " ".join(
        " ".join(p["text"]) if isinstance(p["text"], list) else p["text"] for p in passages
    )
    mentions: dict[str, list[tuple[int, str]]] = {k: [] for k in ENTITY_TYPES}
    for p in passages:
        for ent in p.get("entities", []):
            key = TYPE_TO_KEY.get(str(ent["type"]).lower())
            if key is None or not ent.get("text"):
                continue
            surface = ent["text"][0] if isinstance(ent["text"], list) else ent["text"]
            pos = text.find(surface)
            mentions[key].append((pos if pos >= 0 else len(text), surface))
    ents = {k: _dedupe([m for _, m in sorted(v, key=lambda t: t[0])]) for k, v in mentions.items()}
    return Example(text=text, entities=ents)


def _tner_row(row: dict, id2label: dict[int, str]) -> Example:
    tokens = list(row["tokens"])
    tags = [id2label[t] if isinstance(t, int) else t for t in row["tags"]]
    return Example(text=_detokenize(tokens), entities=bio_to_entities(tokens, tags))


def rows_to_examples(rows, label_names: list[str] | None = None) -> list[Example]:
    """Convert dataset rows from either supported schema into :class:`Example` objects."""
    out: list[Example] = []
    for row in rows:
        if "passages" in row:
            ex = _bigbio_row(row)
        elif "tokens" in row and ("tags" in row or "ner_tags" in row):
            row = dict(row)
            row.setdefault("tags", row.get("ner_tags"))
            names = label_names or TNER_LABELS
            ex = _tner_row(row, dict(enumerate(names)))
        else:
            raise ValueError(f"Unrecognised dataset row schema: keys={sorted(row)}")
        if ex.text.strip():
            out.append(ex)
    return out


def load_bc5cdr(
    candidates=DATASET_CANDIDATES, cache_dir: str | None = None
) -> dict[str, list[Example]]:
    """Load BC5CDR via Hugging Face ``datasets``, trying several public sources.

    Each candidate is ``(builder_or_id, config, data_files)``: with ``data_files`` the generic
    ``json``/``parquet`` builder reads those files directly (no Hub script involved).

    Returns ``{"train": [...], "validation": [...], "test": [...]}``. Raises ``RuntimeError`` with a
    clear message listing every failed candidate if none can be loaded.
    """
    from datasets import load_dataset  # lazy: heavy, and not needed for unit tests

    errors: list[str] = []
    for cand in candidates:
        name, config, data_files = (tuple(cand) + (None, None))[:3]
        try:
            kwargs: dict = {"cache_dir": cache_dir}
            if data_files:
                kwargs["data_files"] = data_files
            ds = load_dataset(name, config, **kwargs) if config else load_dataset(name, **kwargs)
            label_names = None
            try:
                feat = ds["train"].features.get("tags")
                label_names = feat.feature.names if feat is not None else None
            except Exception:  # noqa: BLE001 - label metadata is optional
                label_names = None
            splits = {s: rows_to_examples(ds[s], label_names) for s in ds}
            if "train" not in splits or "test" not in splits:
                raise ValueError(f"missing train/test splits (got {sorted(splits)})")
            return splits
        except Exception as exc:  # noqa: BLE001 - we report every failure to the user
            source = next(iter(data_files.values())) if data_files else f"{name} ({config})"
            errors.append(f"  - {source}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "Could not load BC5CDR from any known Hugging Face dataset id. Tried:\n"
        + "\n".join(errors)
        + "\nCheck your internet connection / `datasets` version, or pass your own candidates."
    )


def subsample(examples: list[Example], n: int | None, seed: int = 0) -> list[Example]:
    """Deterministic subsample (``n=None`` or ``n >= len`` returns a copy of everything)."""
    if n is None or n >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(examples)), n))
    return [examples[i] for i in idx]
