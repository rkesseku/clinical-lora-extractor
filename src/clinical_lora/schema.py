"""Output schema shared by training targets, parsing, evaluation and serving."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

ENTITY_TYPES: tuple[str, ...] = ("chemicals", "diseases")

# BC5CDR entity-type name -> JSON key
TYPE_TO_KEY = {"chemical": "chemicals", "disease": "diseases"}


class Extraction(BaseModel):
    """Structured extraction result: unique entity mentions per type."""

    chemicals: list[str] = Field(default_factory=list)
    diseases: list[str] = Field(default_factory=list)

    @field_validator("chemicals", "diseases", mode="before")
    @classmethod
    def _clean(cls, v):
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("must be a list of strings")
        out: list[str] = []
        for item in v:
            if not isinstance(item, str):
                raise ValueError("entities must be strings")
            item = " ".join(item.split())
            if item and item not in out:
                out.append(item)
        return out

    def as_dict(self) -> dict[str, list[str]]:
        return {k: list(getattr(self, k)) for k in ENTITY_TYPES}
