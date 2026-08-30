"""`rrl-explain` — fetch an arXiv paper and explain it thoroughly via the RLM engine.

    rrl-explain 2512.24601
    rrl-explain https://arxiv.org/abs/2401.02385 -o breakdown.md
    rrl-explain --raw 2512.24601 > paper.tex     # just fetch the source, no model
    rrl-explain --pretty 2512.24601              # watch the recursion + REPL vars live

Needs DEEPSEEK_API_KEY (or another provider's key) unless --raw.
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rrl-explain", description="Explain an arXiv paper with the RLM engine.")
    ap.add_argument("paper", help="arXiv id or URL (e.g. 2512.24601 or arxiv.org/abs/...)")
    ap.add_argument("-o", "--out", help="write the breakdown to this file (default: stdout)")
    ap.add_argument("--raw", action="store_true", help="just fetch and print the LaTeX source; no model call")
    ap.add_argument("--provider", default=os.environ.get("RLM_PROVIDER", "deepseek"), help="model provider")
    ap.add_argument("--model", default=os.environ.get("RLM_MODEL"), help="model id (default: provider's)")
    ap.add_argument("--max-steps", type=int, default=50, help="max turns per agent (generous, to converge)")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("-v", "--verbose", action="store_true", help="terse recursion trace on stderr")
    ap.add_argument("--pretty", action="store_true", help="rich trace: code, panels, and REPL variables")
    args = ap.parse_args(argv)

    from .fetch import fetch_arxiv

    if args.raw:
        try:
            sys.stdout.write(fetch_arxiv(args.paper))
        except Exception as e:  # noqa: BLE001 — a fetch failure is a normal user error here
            sys.stderr.write(f"error: {e}\n")
            return 2
        return 0

    from rlm import make_backend
    from rlm.cli import _resolve_renderer

    from .explain import explain

    try:
        backend = make_backend(args.provider, model=args.model)
    except RuntimeError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2

    renderer = _resolve_renderer(args.pretty, args.verbose)
    sys.stderr.write(f"fetching {args.paper} …\n")
    try:
        result = explain(
            args.paper,
            backend,
            model=args.model,
            max_steps=args.max_steps,
            max_depth=args.max_depth,
            on_event=renderer,
            inspect_vars=args.pretty,  # the --pretty renderer shows the REPL namespace
        )
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"error: {e}\n")
        return 1

    note = " (via fallback synthesis)" if result.used_fallback else ""
    sys.stderr.write(f"\n[{result.budget.summary()}] explained {result.n_chars} chars{note}\n")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(result.output)
        sys.stderr.write(f"wrote {args.out}\n")
    else:
        print(result.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
