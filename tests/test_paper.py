"""paper harness tests — fully offline (no network, no API key).

    python tests/test_paper.py     # prints "ok"
    pytest tests/test_paper.py
"""

import gzip
import io
import os
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paper import EXPLAIN_PROMPT, explain, parse_arxiv_id  # noqa: E402
from paper.fetch import extract_tex  # noqa: E402
from rlm import Backend, MockBackend, Usage  # noqa: E402


def test_parse_arxiv_id():
    assert parse_arxiv_id("2512.24601") == "2512.24601"
    assert parse_arxiv_id("2401.02385v2") == "2401.02385v2"
    assert parse_arxiv_id("https://arxiv.org/abs/2512.24601") == "2512.24601"
    assert parse_arxiv_id("arxiv.org/pdf/2401.02385v2") == "2401.02385v2"
    assert parse_arxiv_id("http://arxiv.org/pdf/2401.02385.pdf") == "2401.02385"
    assert parse_arxiv_id("hep-th/9901001") == "hep-th/9901001"
    for bad in ["", "not a paper", "/home/user/paper.tex", "https://example.com/x"]:
        try:
            parse_arxiv_id(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"should have rejected {bad!r}")


def test_extract_tex_from_targz():
    # Build the shape arXiv actually returns: a gzipped tar of .tex (+ noise).
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as tf:
        for name, body in [("main.tex", b"\\section{Intro}\nhello"), ("appendix.tex", b"\\section{App}\nbye")]:
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tf.addfile(info, io.BytesIO(body))
        note = b"not tex"  # a non-.tex member must be ignored
        info = tarfile.TarInfo("README")
        info.size = len(note)
        tf.addfile(info, io.BytesIO(note))
    text = extract_tex(raw.getvalue())
    assert "\\section{Intro}" in text and "\\section{App}" in text, text
    assert "not tex" not in text, "non-.tex members should be skipped"


def test_extract_tex_single_gzip():
    raw = gzip.compress(b"\\documentclass{article}\n\\section{Solo}")
    assert "\\section{Solo}" in extract_tex(raw)


def test_explain_happy_path():
    # The agent explores and FINAL()s a report — no fallback needed.
    script = [
        "```python\nprint(len(PROMPT))\n```",
        "```python\nreport = '# Breakdown\\nThesis: X.'\nFINAL(report)\n```",
    ]
    exp = explain(text="\\section{Intro}\nA paper about X." * 20, backend=MockBackend(script))
    assert not exp.used_fallback, "should have converged via FINAL()"
    assert "Breakdown" in exp.output, exp.output


def test_explain_fallback_when_no_final():
    # A backend that never emits a FINAL -> engine returns None -> fallback synthesizes.
    class NeverFinals(Backend):
        def complete(self, messages, *, model=None):
            return "no code, just prose", Usage(calls=1, prompt_tokens=1, completion_tokens=1)

    exp = explain(text="body " * 4000, backend=NeverFinals(), max_steps=3)
    assert exp.used_fallback, "should have fallen back"
    assert exp.output.strip(), "fallback must produce non-empty output"
    assert EXPLAIN_PROMPT  # sanity: the instruction exists


if __name__ == "__main__":
    test_parse_arxiv_id()
    test_extract_tex_from_targz()
    test_extract_tex_single_gzip()
    test_explain_happy_path()
    test_explain_fallback_when_no_final()
    print("ok")
