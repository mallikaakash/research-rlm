"""Corpus operations — what turns a folder of Notes into a brain.

The Note (rung 1-3) is the atom. These ops build the layers *around* the notes:
an index for fast lookup, edges between notes (rung 4), and a goal queue that
feeds proactivity (rung 6). Everything here is a pure function over the corpus
directory, so it needs no network and is fully testable offline.

  load_corpus / build_index / save_index  — read the folder, summarize it
  related_notes / link                    — rung 4: heuristic edges between notes
  contradictions                          — model-driven: which claims conflict
  pick_goal / write_goals                 — the proactive seed (open questions + gaps)

`link` and `pick_goal` are heuristic (no model) so they run anywhere; only
`contradictions` needs a backend, because judging whether two claims *conflict*
is a reasoning task — and it's an RLM call over the claims-as-oversized-input.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rlm import Budget, LocalSandbox, run

from .note import Note, load_note, save_note


# ---- load / index ----

def load_corpus(corpus_dir: str | Path = "corpus") -> list[Note]:
    d = Path(corpus_dir)
    if not d.exists():
        return []
    notes = []
    for p in sorted(d.glob("*.yaml")):
        if p.name.startswith("_"):
            continue
        try:
            notes.append(load_note(p))
        except Exception:  # noqa: BLE001 — skip a malformed note, don't crash the corpus
            continue
    return notes


def build_index(notes: list[Note]) -> list[dict]:
    return [
        {
            "id": n.id,
            "title": n.title,
            "tags": n.tags,
            "n_claims": len(n.claims),
            "n_open_questions": len(n.open_questions),
            "degree": len(n.connections.get("related", [])) if n.connections else 0,
        }
        for n in notes
    ]


def save_index(corpus_dir: str | Path, notes: list[Note] | None = None) -> Path:
    d = Path(corpus_dir)
    notes = notes if notes is not None else load_corpus(d)
    path = d / "_index.json"
    path.write_text(json.dumps(build_index(notes), indent=2), encoding="utf-8")
    return path


def all_claims(notes: list[Note]) -> list[dict]:
    """Flatten every claim across the corpus, each tagged with its source note."""
    out = []
    for n in notes:
        for c in n.claims:
            out.append(
                {"ref": f"{n.id}#{c.id}", "note": n.id, "statement": c.statement, "strength": c.strength}
            )
    return out


# ---- rung 4: edges ----

def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{4,}", (text or "").lower()))


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _profile(n: Note) -> tuple[set, set]:
    tags = set(t.lower() for t in n.tags)
    terms = _terms(" ".join([n.title, n.key_idea, *(c.statement for c in n.claims)]))
    return tags, terms


def related_notes(note: Note, notes: list[Note], k: int = 5) -> list[tuple[str, float]]:
    """Heuristic relatedness: shared tags (weighted) + overlap of claim/idea terms."""
    base_tags, base_terms = _profile(note)
    scored = []
    for other in notes:
        if other.id == note.id:
            continue
        o_tags, o_terms = _profile(other)
        score = 2.0 * _jaccard(base_tags, o_tags) + _jaccard(base_terms, o_terms)
        if score > 0:
            scored.append((other.id, round(score, 3)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def link(corpus_dir: str | Path = "corpus", notes: list[Note] | None = None, k: int = 5,
         threshold: float = 0.05, save: bool = True) -> list[Note]:
    """Populate each note's `connections['related']` with its nearest neighbours."""
    notes = notes if notes is not None else load_corpus(corpus_dir)
    for n in notes:
        rel = [rid for rid, s in related_notes(n, notes, k) if s >= threshold]
        n.connections = {**(n.connections or {}), "related": rel}
        if save:
            save_note(n, corpus_dir)
    return notes


# ---- model-driven: contradictions ----

_CONTRA_INSTRUCTION = (
    "PROMPT is a list of research claims (one per line as `ref: statement [note]`) "
    "drawn from many papers. Find pairs of claims that genuinely CONTRADICT each "
    "other — opposing assertions about the same thing, not merely different topics. "
    "Return the result with FINAL(pairs) where pairs is a list of dicts: "
    "{'a': <ref>, 'b': <ref>, 'reason': <short why>}. If there are none, FINAL([])."
)


def contradictions(notes: list[Note], backend, *, model: str | None = None,
                   sandbox_factory=LocalSandbox, max_steps: int = 8, max_depth: int = 3) -> list[dict]:
    """Ask the engine which claims across the corpus conflict (an RLM call over the
    claims-as-oversized-input). Returns a list of {a, b, reason}."""
    claims = all_claims(notes)
    if len(claims) < 2:
        return []
    blob = "\n".join(f"{c['ref']}: {c['statement']} [{c['note']}]" for c in claims)
    result = run(
        blob, backend, instruction=_CONTRA_INSTRUCTION, model=model,
        budget=Budget(max_depth=max_depth), max_steps=max_steps, sandbox_factory=sandbox_factory,
    )
    out = result.output
    if isinstance(out, str):
        try:
            out = json.loads(out)
        except Exception:  # noqa: BLE001
            return []
    return [p for p in out if isinstance(p, dict)] if isinstance(out, list) else []


# ---- rung 6: the proactive seed ----

def pick_goal(notes: list[Note]) -> dict | None:
    """Rank what to research next: open questions (from richer notes first), then
    under-connected notes worth expanding. Returns the top goal, or None if empty."""
    goals: list[dict] = []
    for n in notes:
        for q in n.open_questions:
            goals.append({"kind": "open_question", "note": n.id, "query": q, "weight": 2 + len(n.claims)})
    for n in notes:
        degree = len(n.connections.get("related", [])) if n.connections else 0
        if degree == 0:
            goals.append({"kind": "expand", "note": n.id,
                          "query": f"find work related to: {n.title}", "weight": 1})
    if not goals:
        return None
    goals.sort(key=lambda g: g["weight"], reverse=True)
    return goals[0]


def write_goals(corpus_dir: str | Path, notes: list[Note] | None = None) -> Path:
    """Dump the full ranked goal queue to _goals.jsonl (newest ranking each run)."""
    d = Path(corpus_dir)
    notes = notes if notes is not None else load_corpus(d)
    goals = []
    for n in notes:
        for q in n.open_questions:
            goals.append({"kind": "open_question", "note": n.id, "query": q, "weight": 2 + len(n.claims)})
    goals.sort(key=lambda g: g["weight"], reverse=True)
    path = d / "_goals.jsonl"
    path.write_text("\n".join(json.dumps(g) for g in goals), encoding="utf-8")
    return path
