"""The read pipeline — one paper in, one structured Note out.

This is the first piece of the *brain*: it drives the RLM engine over a paper and
turns the result into a saved Note. RLM-orchestrated (lean): we hand the engine the
paper (as PROMPT) plus a note-extraction instruction, and let the model write the
code to segment and extract, then FINAL(note_dict). We validate/coerce and save.

v1 input: raw text or a file path. arXiv-id fetching is a thin adapter for later
(and arXiv egress is blocked in the build sandbox anyway).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from rlm import Budget, LocalSandbox, make_backend, run

from .fetch import fetch_arxiv_text, parse_arxiv_id
from .note import Note, coerce_note, save_note

READ_INSTRUCTION = """\
You are reading a research paper (in PROMPT) to produce a structured NOTE.

Explore PROMPT with code — split it into sections, find the abstract, method, and
results. Use `await llm(chunk)` to pull details from a section without loading it
all into your own context.

Return the note with FINAL(note) where `note` is a Python dict with these keys:
  - title (str), authors (list of str)
  - one_liner (str): one sentence capturing the paper's essence
  - problem (str): the problem it addresses
  - key_idea (str): the core idea/method in 1-2 sentences
  - method (list of str): the key techniques/components
  - claims (list of dicts) — THE most important part. Each claim is:
      {"id": "c1",
       "statement": "<a specific, checkable claim the paper makes>",
       "evidence": "<where/how it's supported, e.g. 'Table 2', 'Fig 3', 'ablation in §5'>",
       "strength": "strong" | "moderate" | "anecdotal" | "asserted"}
  - open_questions (list of str): questions the paper leaves open
  - tags (list of str): 3-6 topic tags

Extract EVERY substantive claim with its supporting evidence and an honest
strength. Then call FINAL(note)."""


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:60] or ""


def _fallback_id(title: str, text: str) -> str:
    return _slug(title) or "note-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _load_text(source: str, corpus_dir: str = "corpus") -> tuple[str, str, str | None]:
    """Return (text, source_kind, arxiv_id|None).

    An arXiv id / URL is served from corpus/raw/<id>.txt if cached, else fetched;
    a path is read; anything else is literal text.
    """
    aid = parse_arxiv_id(source)
    if aid:
        cache = Path(corpus_dir) / "raw" / f"{aid}.txt"
        if cache.exists():
            return cache.read_text(encoding="utf-8", errors="replace"), "latex", aid
        text, kind = fetch_arxiv_text(aid)
        return text, kind, aid
    p = Path(source)
    if len(source) < 4096 and p.exists() and p.is_file():
        return p.read_text(encoding="utf-8", errors="replace"), "file", None
    return source, "text", None


def _cache_raw(corpus_dir: str, note_id: str, text: str) -> None:
    """Keep the fetched source in corpus/raw/<id>.txt (gitignored) so re-reads and
    deep questions don't refetch."""
    raw_dir = Path(corpus_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{note_id}.txt").write_text(text, encoding="utf-8")


def read_paper(
    source: str,
    backend=None,
    *,
    corpus_dir: str = "corpus",
    provider: str | None = None,
    model: str | None = None,
    max_steps: int = 12,
    max_depth: int = 3,
    sandbox_factory=LocalSandbox,
    on_event=None,
    save: bool = True,
) -> Note:
    """Read one paper (arXiv id/URL, file path, or raw text) into a saved Note."""
    text, kind, aid = _load_text(source, corpus_dir)
    backend = backend or make_backend(provider, model=model)

    result = run(
        text,
        backend,
        budget=Budget(max_depth=max_depth),
        max_steps=max_steps,
        model=model,
        instruction=READ_INSTRUCTION,
        sandbox_factory=sandbox_factory,
        on_event=on_event,
    )

    raw = result.output
    title_hint = raw.get("title", "") if isinstance(raw, dict) else ""
    note = coerce_note(raw, fallback_id=aid or _fallback_id(title_hint, text))
    if note.source == "text":
        note.source = kind
    if aid:  # the arXiv id is the canonical note id
        note.id = aid
    if save:
        save_note(note, corpus_dir)
        _cache_raw(corpus_dir, note.id, text)
    return note


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog="rrl-read", description="Read a paper into a structured Note.")
    ap.add_argument("source", help="a file path, or raw paper text")
    ap.add_argument("--provider", default=None, help="model provider (default: deepseek)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--sandbox", choices=["local", "pyodide"], default="local")
    ap.add_argument("--corpus", default="corpus", help="corpus directory (default: ./corpus)")
    ap.add_argument("-v", "--verbose", action="store_true", help="terse recursion trace on stderr")
    ap.add_argument("--pretty", action="store_true", help="rich trace: syntax-highlighted code + colored panels")
    args = ap.parse_args(argv)

    from rlm import LocalSandbox, PyodideSandbox
    from rlm.cli import _resolve_renderer

    factory = PyodideSandbox if args.sandbox == "pyodide" else LocalSandbox
    try:
        note = read_paper(
            args.source,
            corpus_dir=args.corpus,
            provider=args.provider,
            model=args.model,
            sandbox_factory=factory,
            on_event=_resolve_renderer(args.pretty, args.verbose),
        )
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"error: {type(e).__name__}: {e}\n")
        return 1

    sys.stderr.write(f"\nsaved {args.corpus}/{note.id}.yaml — {len(note.claims)} claims\n")
    print(note.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
