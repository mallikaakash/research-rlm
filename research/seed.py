"""Batch-populate the corpus from seed_corpus.yaml.

Reads every listed arXiv paper into a Note, then wires the corpus up: link()
(edges), save_index(), write_goals(). One command to go from an empty corpus to a
linked brain.

    export DEEPSEEK_API_KEY=sk-...
    rrl-seed                       # read everything in seed_corpus.yaml
    rrl-seed --limit 3 --pretty    # try a few first, with the rich trace

Network-bound (arXiv + the model), so run it on your own machine — the cloud build
env blocks arXiv egress. Be polite: a delay between fetches respects arXiv's rate.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from rlm import LocalSandbox, PyodideSandbox
from .corpus import link, save_index, write_goals
from .read import read_paper


def load_seed_ids(seed_path: str | Path = "seed_corpus.yaml") -> list[str]:
    data = yaml.safe_load(Path(seed_path).read_text(encoding="utf-8")) or {}
    ids = []
    for papers in data.values():
        for p in papers or []:
            if isinstance(p, dict) and p.get("id"):
                ids.append(str(p["id"]))
    # de-dup, preserve order
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def build_corpus(
    seed_path: str | Path = "seed_corpus.yaml",
    corpus_dir: str = "corpus",
    *,
    provider: str | None = None,
    model: str | None = None,
    sandbox_factory=LocalSandbox,
    limit: int | None = None,
    delay: float = 3.0,
    on_event=None,
    reader=read_paper,
) -> list[str]:
    """Read each seed paper into the corpus, then link + index + write goals."""
    import time

    ids = load_seed_ids(seed_path)
    if limit:
        ids = ids[:limit]
    done = []
    for i, aid in enumerate(ids, 1):
        try:
            note = reader(
                aid, provider=provider, model=model, corpus_dir=corpus_dir,
                sandbox_factory=sandbox_factory, on_event=on_event,
            )
            done.append(note.id)
            print(f"[{i}/{len(ids)}] {aid} -> {note.id}  ({len(note.claims)} claims)")
        except Exception as e:  # noqa: BLE001 — one bad paper shouldn't stop the batch
            print(f"[{i}/{len(ids)}] {aid} FAILED: {type(e).__name__}: {e}")
        if delay and i < len(ids):
            time.sleep(delay)

    link(corpus_dir)
    save_index(corpus_dir)
    write_goals(corpus_dir)
    print(f"\ncorpus: {len(done)} notes read, linked, indexed; goals written to {corpus_dir}/_goals.jsonl")
    return done


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog="rrl-seed", description="Populate the corpus from a seed list.")
    ap.add_argument("--seed", default="seed_corpus.yaml")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--sandbox", choices=["local", "pyodide"], default="local")
    ap.add_argument("--limit", type=int, default=None, help="read only the first N papers")
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between fetches (be polite to arXiv)")
    ap.add_argument("--pretty", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    from rlm.cli import _resolve_renderer

    factory = PyodideSandbox if args.sandbox == "pyodide" else LocalSandbox
    build_corpus(
        args.seed, args.corpus, provider=args.provider, model=args.model,
        sandbox_factory=factory, limit=args.limit, delay=args.delay,
        on_event=_resolve_renderer(args.pretty, args.verbose),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
