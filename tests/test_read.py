"""Read-pipeline test — offline, deterministic.

Uses MockBackend (no network) but the real engine + sandbox: the mock model plays
a single cell that inspects PROMPT and FINAL()s a note dict. Verifies the pipeline
coerces it into a valid Note, fills defaults, and saves/loads YAML.

    python tests/test_read.py     # prints "ok"
    pytest tests/test_read.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rrl.engine import MockBackend  # noqa: E402
from rrl.note import NoteError, coerce_note, load_note  # noqa: E402
from rrl.read import read_paper  # noqa: E402

# The model's single turn: build a note dict from PROMPT and FINAL it. Note that
# `strength: "high"` is invalid and must be coerced to "asserted"; claim c2 omits
# its id and must be auto-filled.
_FINAL_CELL = """```python
note = {
    "title": "Recursive Language Models",
    "authors": ["Zhang", "Kraska"],
    "one_liner": "Reason over unbounded context via a REPL variable.",
    "problem": "LLMs degrade past their context window.",
    "key_idea": "Prompt-as-variable; recurse on sub-chunks.",
    "method": ["Python REPL", "recursive llm_query"],
    "claims": [
        {"id": "c1", "statement": "Handles ~100x the window.",
         "evidence": "Table 2", "strength": "strong"},
        {"statement": "Mini in an RLM beats the full model on long QA.",
         "evidence": "Fig 3", "strength": "high"},
    ],
    "open_questions": ["Does it help reasoning, not just retrieval?"],
    "tags": ["long-context", "inference"],
}
FINAL(note)
```"""


def test_read_pipeline():
    backend = MockBackend([_FINAL_CELL])
    with tempfile.TemporaryDirectory() as d:
        note = read_paper("some paper text stands in for a real paper", backend, corpus_dir=d)

        assert note.title == "Recursive Language Models"
        assert len(note.claims) == 2
        assert note.claims[1].id == "c2"                 # auto-filled
        assert note.claims[1].strength == "asserted"     # "high" coerced -> fallback
        assert note.id and note.read_on                  # defaults filled

        # round-trips through YAML on disk
        reloaded = load_note(os.path.join(d, f"{note.id}.yaml"))
        assert reloaded.claims[0].statement == "Handles ~100x the window."


def test_note_with_no_claims_is_rejected():
    try:
        coerce_note({"title": "x", "claims": []}, fallback_id="x")
    except NoteError:
        return
    raise AssertionError("expected NoteError for a claimless note")


if __name__ == "__main__":
    test_read_pipeline()
    test_note_with_no_claims_is_rejected()
    print("ok")
