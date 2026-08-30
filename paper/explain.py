"""Explain a paper: the breakdown instruction + explain() over the RLM engine.

The engine already knows how to reason over a huge PROMPT (chunk it, delegate to
llm()/rlm(), combine). All this adds is (1) a carefully worded instruction that
shapes the output into a thorough breakdown, and (2) a deterministic fallback for
the one failure mode of a recursive agent on a big input: running out of steps
without ever calling FINAL(), which would otherwise return None.
"""

from __future__ import annotations

from dataclasses import dataclass

from rlm import Backend, Budget, LocalSandbox, make_backend, run

EXPLAIN_PROMPT = """You are given the FULL LaTeX source of a research paper as PROMPT.
Produce a THOROUGH, faithful breakdown of it in Markdown. Explore PROMPT with code,
delegate sections to await llm()/await rlm(), and combine — never paste large slices
into your own reasoning.

Your final report MUST cover, with specifics and exact numbers/quotes where present:
  1. Title, authors, and the one-sentence thesis.
  2. The problem: what gap or question the paper addresses, and why it matters.
  3. Key claims / contributions (as a list).
  4. Methodology: the exact approach, architecture, algorithm, or setup.
  5. Experimental setup: datasets, baselines, metrics, hyperparameters.
  6. Results: the concrete numbers and what they show.
  7. Limitations and what did NOT work (from the paper's own admissions).
  8. Takeaways: why this matters and what it enables.

Build the report as a single Markdown string, then finish with FINAL(report).
IMPORTANT: you MUST end by calling FINAL(<the markdown string>) — do not stop early."""


@dataclass
class Explanation:
    output: str
    budget: Budget
    used_fallback: bool
    n_chars: int  # size of the source that was explained


def _fallback(text: str, backend: Backend, model: str | None) -> str:
    """Deterministic map-reduce breakdown, used only if the agent never FINAL()s.

    Guarantees a useful result: summarize the source in chunks (map), then ask the
    model to synthesize the breakdown from those notes (reduce). No recursion, so
    it always terminates.
    """
    chunks = [text[i : i + 6000] for i in range(0, len(text), 6000)][:12]
    notes = []
    for c in chunks:
        out, _ = backend.complete(
            [{"role": "user", "content": "Summarize the key content of this paper section:\n\n" + c}],
            model=model,
        )
        notes.append(out)
    joined = "\n\n".join(notes)
    out, _ = backend.complete(
        [{"role": "user", "content": EXPLAIN_PROMPT + "\n\nSection notes to synthesize:\n\n" + joined}],
        model=model,
    )
    return out


def explain(
    url_or_id: str | None = None,
    backend: Backend | None = None,
    *,
    text: str | None = None,
    model: str | None = None,
    max_steps: int = 14,
    max_depth: int = 3,
    sandbox_factory=LocalSandbox,
    on_event=None,
    inspect_vars: bool = False,
) -> Explanation:
    """Fetch a paper (or use `text=`) and return a thorough breakdown.

    Pass `text=` to skip fetching (for tests, or a local paper). `on_event` /
    `inspect_vars` are forwarded to the engine so a UI can observe the run.
    """
    from .fetch import fetch_arxiv

    if text is None:
        if not url_or_id:
            raise ValueError("provide an arXiv id/URL, or text=")
        text = fetch_arxiv(url_or_id)

    backend = backend or make_backend(model=model)
    budget = Budget(max_depth=max_depth)
    result = run(
        text,
        backend,
        instruction=EXPLAIN_PROMPT,
        budget=budget,
        max_steps=max_steps,
        model=model,
        sandbox_factory=sandbox_factory,
        on_event=on_event,
        inspect_vars=inspect_vars,
    )

    out = result.output
    used_fallback = out is None or (isinstance(out, str) and not out.strip())
    if used_fallback:
        out = _fallback(text, backend, model)

    return Explanation(output=str(out), budget=budget, used_fallback=used_fallback, n_chars=len(text))
