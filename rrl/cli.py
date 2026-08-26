"""`rrl-engine` — run a single RLM query from the command line.

Examples:
    rrl-engine "What is 2+2? Answer with just the number."
    rrl-engine --input-file paper.txt "Summarize the main claim."
    rrl-engine -v --max-depth 3 "..."

Requires OPENROUTER_API_KEY in the environment for the real backend.
"""

from __future__ import annotations

import argparse
import os
import sys


def _printer(event: dict) -> None:
    indent = "  " * event.get("depth", 0)
    t = event.get("type")
    if t == "assistant":
        head = (event.get("text") or "").strip().splitlines()[:1]
        sys.stderr.write(f"{indent}· agent[{event.get('step')}] {head[0] if head else ''}\n")
    elif t == "cell":
        tag = "FINAL" if event.get("has_final") else ("ERR" if event.get("error") else "ok")
        out = (event.get("error") or event.get("stdout") or "").strip().replace("\n", " ")
        sys.stderr.write(f"{indent}  └ cell[{tag}] {out[:100]}\n")
    elif t == "rlm":
        sys.stderr.write(f"{indent}  ↳ rlm({event.get('in')!r})\n")
    elif t == "llm":
        sys.stderr.write(f"{indent}  ↳ llm(…) -> {event.get('out')!r}\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rrl-engine", description="Run one RLM query.")
    ap.add_argument("query", nargs="?", help="the task or question")
    ap.add_argument("--input-file", help="file whose contents are prepended to PROMPT")
    ap.add_argument(
        "--provider",
        default=os.environ.get("RLM_PROVIDER", "deepseek"),
        help="model provider (default: deepseek; also: openrouter)",
    )
    ap.add_argument("--model", default=os.environ.get("RLM_MODEL"), help="model id (default: the provider's)")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("-v", "--verbose", action="store_true", help="trace the recursion on stderr")
    args = ap.parse_args(argv)

    parts = []
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8", errors="replace") as fh:
            parts.append(fh.read())
    if args.query:
        parts.append(args.query)
    if not parts:
        ap.error("provide a query and/or --input-file")
    prompt = "\n\n".join(parts)

    # Imported here so --help works even without the package installed as a wheel.
    from .engine import Budget, make_backend, run

    try:
        backend = make_backend(args.provider, model=args.model)
    except RuntimeError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2

    budget = Budget(max_depth=args.max_depth)
    result = run(
        prompt,
        backend,
        budget=budget,
        max_steps=args.max_steps,
        model=args.model,
        on_event=_printer if args.verbose else None,
    )
    sys.stderr.write(f"\n[{budget.summary()}]\n")
    print(result.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
