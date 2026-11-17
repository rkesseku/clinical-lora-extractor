"""Robust parsing and repair of model output into a validated :class:`Extraction`."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from .schema import Extraction

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


@dataclass
class ParseResult:
    extraction: Extraction | None
    valid: bool  # parsed as-is: strict JSON + schema-valid, no repair needed
    repaired: bool  # recovered only after repair heuristics
    error: str | None = None

    @property
    def parsed(self) -> bool:
        """True if an extraction was recovered, with or without repair."""
        return self.extraction is not None


def _extract_candidate(text: str) -> str:
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start < 0:
        return text.strip()
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]  # unbalanced / truncated: leave to the repair step


def _close_open_structures(s: str) -> str:
    """Close an unterminated string and any open brackets (typical of truncated generations)."""
    stack: list[str] = []
    in_str, esc = False, False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    if in_str:
        s += '"'
    s = s.rstrip()
    s = re.sub(r",\s*$", "", s)
    # A dangling key with no value (e.g. '"diseases":') cannot be completed; drop it.
    s = re.sub(r',?\s*"[^"]*"\s*:\s*$', "", s)
    return s + "".join(reversed(stack))


def _repair(candidate: str) -> str:
    s = candidate.strip()
    # Python-style / single-quoted output
    if "'" in s and '"' not in s:
        s = s.replace("'", '"')
    s = _TRAILING_COMMA.sub(r"\1", s)
    s = _close_open_structures(s)
    return _TRAILING_COMMA.sub(r"\1", s)


def _validate(obj) -> Extraction:
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value must be an object")
    if not ({"chemicals", "diseases"} & set(obj)):
        raise ValueError("object has neither 'chemicals' nor 'diseases'")
    return Extraction.model_validate({k: obj.get(k) for k in ("chemicals", "diseases")})


def parse_extraction(text: str) -> ParseResult:
    """Parse generated text. ``valid`` means no repair was needed; ``repaired`` means recovered."""
    candidate = _extract_candidate(text or "")
    try:
        return ParseResult(_validate(json.loads(candidate)), valid=True, repaired=False)
    except (json.JSONDecodeError, ValueError, ValidationError) as first_err:
        err = str(first_err)
    try:
        return ParseResult(_validate(json.loads(_repair(candidate))), valid=False, repaired=True)
    except (json.JSONDecodeError, ValueError, ValidationError) as second_err:
        return ParseResult(None, valid=False, repaired=False, error=f"{err} | repair: {second_err}")
