"""The Note — the atomic unit of the corpus.

A Note is what "understanding a paper" cashes out as (rungs 1-3): a faithful
summary whose center of gravity is a list of **checkable claims**, each with an
evidence pointer and a strength. Notes are stored as flat YAML, one file per
paper, so the corpus is a folder you can read, diff, and query by globbing.

Validation is deliberately lenient: model output is messy, so we coerce rather
than reject where we safely can (unknown strengths fall back, missing claim ids
are filled). Structural problems (not a dict, no claims) still raise so the read
pipeline can retry.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

STRENGTHS = {"strong", "moderate", "anecdotal", "asserted"}


class NoteError(Exception):
    """The model output could not be coerced into a valid Note."""


class Claim(BaseModel):
    """A single checkable proposition the paper makes."""

    model_config = ConfigDict(extra="ignore")

    id: str = ""
    statement: str
    evidence: str = ""
    strength: str = "asserted"

    @field_validator("strength", mode="before")
    @classmethod
    def _norm_strength(cls, v):
        v = str(v).strip().lower()
        return v if v in STRENGTHS else "asserted"


class Note(BaseModel):
    """One paper, understood — rungs 1-3. Rungs 4-5 (connections, assumptions) are
    added later by corpus ops and are intentionally absent here."""

    model_config = ConfigDict(extra="ignore")

    id: str = ""
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    source: str = "text"
    read_on: str = ""
    confidence: float = 0.5

    one_liner: str = ""
    problem: str = ""
    key_idea: str = ""
    method: list[str] = Field(default_factory=list)

    claims: list[Claim] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    # rung 4 — filled by corpus ops (link/contradictions), not the read pipeline.
    # e.g. {"related": [...ids], "extends": [...ids], "contradicts": [...ids]}
    connections: dict = Field(default_factory=dict)

    def fill_defaults(self, *, fallback_id: str) -> "Note":
        """Assign an id / read_on if the model omitted them, and number claim ids."""
        if not self.id:
            self.id = fallback_id
        if not self.read_on:
            self.read_on = date.today().isoformat()
        for i, c in enumerate(self.claims, start=1):
            if not c.id:
                c.id = f"c{i}"
        return self


# ---- coercion + storage ----

def coerce_note(raw, *, fallback_id: str) -> Note:
    """Turn a model's FINAL() output (a dict, or a JSON string) into a valid Note."""
    import json

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise NoteError(f"FINAL() was a string but not JSON: {e}") from None
    if not isinstance(raw, dict):
        raise NoteError(f"expected a note dict, got {type(raw).__name__}")
    note = Note.model_validate(raw)
    if not note.claims:
        raise NoteError("note has no claims — the atom of a note is the claim")
    return note.fill_defaults(fallback_id=fallback_id)


def save_note(note: Note, corpus_dir: str | Path = "corpus") -> Path:
    d = Path(corpus_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{note.id}.yaml"
    path.write_text(
        yaml.safe_dump(note.model_dump(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def load_note(path: str | Path) -> Note:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Note.model_validate(data)
