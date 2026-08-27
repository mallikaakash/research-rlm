"""Corpus-ops test — offline, deterministic.

Builds a small synthetic corpus of Notes on disk and exercises the pure ops
(load, index, related, link, pick_goal) with no model, plus contradictions() with
a MockBackend. No network needed.

    python tests/test_corpus.py     # prints "ok"
    pytest tests/test_corpus.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.corpus import (  # noqa: E402
    all_claims,
    build_index,
    contradictions,
    link,
    load_corpus,
    pick_goal,
    related_notes,
)
from rlm import MockBackend  # noqa: E402
from research.note import Claim, Note, save_note  # noqa: E402


def _note(id, title, key_idea, tags, claims, oq=()):
    return Note(
        id=id, title=title, key_idea=key_idea, tags=list(tags),
        claims=[Claim(id=f"c{i+1}", statement=s, evidence="§x", strength="moderate")
                for i, s in enumerate(claims)],
        open_questions=list(oq),
    )


def _seed(d):
    notes = [
        _note("grpo", "GRPO for reasoning", "Group-relative RL for LLM reasoning.",
              ["rl", "post-training", "reasoning"],
              ["GRPO removes the value function and beats PPO on reasoning."],
              oq=["Does GRPO scale to 100B+ models?"]),
        _note("ppo", "PPO for RLHF", "Proximal policy optimization for alignment.",
              ["rl", "post-training", "alignment"],
              ["PPO with a value function is more stable than value-free RL for reasoning."]),
        _note("gemma", "Gemma 2 report", "A small open model via distillation.",
              ["frontier-report", "distillation"],
              ["Knowledge distillation beats next-token training at small scale."]),
    ]
    for n in notes:
        save_note(n, d)
    return notes


def test_corpus_ops():
    with tempfile.TemporaryDirectory() as d:
        _seed(d)

        notes = load_corpus(d)
        assert len(notes) == 3, [n.id for n in notes]

        # index
        idx = {r["id"]: r for r in build_index(notes)}
        assert idx["grpo"]["n_claims"] == 1 and idx["grpo"]["n_open_questions"] == 1

        # all_claims flattens with refs
        claims = all_claims(notes)
        assert len(claims) == 3 and all("#" in c["ref"] for c in claims)

        # related: the two RL notes should be each other's top match (shared tags/terms),
        # ahead of the unrelated frontier-report note.
        rel = related_notes(next(n for n in notes if n.id == "grpo"), notes)
        assert rel and rel[0][0] == "ppo", rel

        # link writes connections back to disk
        link(d, notes)
        reloaded = {n.id: n for n in load_corpus(d)}
        assert "ppo" in reloaded["grpo"].connections.get("related", []), reloaded["grpo"].connections

        # pick_goal prefers the open question from the richer note
        goal = pick_goal(notes)
        assert goal and goal["kind"] == "open_question" and goal["note"] == "grpo", goal


def test_contradictions_with_mock():
    with tempfile.TemporaryDirectory() as d:
        notes = _seed(d)
        # The model "notices" grpo#c1 (value-free beats PPO) vs ppo#c1 (PPO more stable).
        final = ("```python\n"
                 "FINAL([{'a': 'grpo#c1', 'b': 'ppo#c1', 'reason': 'value-free vs value-based RL for reasoning'}])\n"
                 "```")
        pairs = contradictions(notes, MockBackend([final]))
        assert len(pairs) == 1 and pairs[0]["a"] == "grpo#c1", pairs


if __name__ == "__main__":
    test_corpus_ops()
    test_contradictions_with_mock()
    print("ok")
