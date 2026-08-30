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

EXPLAIN_PROMPT = """You are given the FULL LaTeX source of a research paper as PROMPT (it may be very large).
Produce a THOROUGH, faithful breakdown of it in Markdown.

Work like a Recursive Language Model. Do NOT read PROMPT yourself, and do NOT burn
turns merely measuring section lengths or printing offsets — that is not progress.
The real work is DELEGATING the reading to sub-calls and then synthesizing. Procedure:

  1. In 1–2 turns, split PROMPT into its logical sections — keep the actual TEXT of
     each section (its string slice), not just its start/end offsets.
  2. DELEGATE the reading: for EVERY section call `await llm(...)` (or `await rlm(...)`
     for a very large section) to pull out that section's key content. Prefer doing
     them together in ONE turn with a comprehension, e.g.:
        notes = [await llm("Extract the claims, methods, setup, and exact numbers "
                           "from this paper section:\\n\\n" + s) for s in sections]
  3. Combine the notes into ONE Markdown report covering, with specifics and exact
     numbers/quotes where present:
        - Title, authors, one-sentence thesis
        - Problem & motivation (the gap it addresses, why it matters)
        - Key claims / contributions (a list)
        - Methodology (the exact approach / architecture / algorithm)
        - Experimental setup (datasets, baselines, metrics, hyperparameters)
        - Results (the concrete numbers and what they show)
        - Limitations / what did NOT work (the paper's own admissions)
        - Takeaways (why it matters, what it enables)
  4. Finish by calling FINAL(report).

You have a generous step budget, but you MUST converge: delegate the reading to
llm()/rlm(), then end with FINAL(<the markdown string>). Never stop early, and never
finish by just printing section lengths — a run that never calls llm()/rlm() is wrong."""


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
    max_steps: int = 50,
    max_depth: int = 4,
    max_calls: int = 1000,
    max_tokens: int = 20_000_000,
    sandbox_factory=LocalSandbox,
    on_event=None,
    inspect_vars: bool = False,
) -> Explanation:
    """Fetch a paper (or use `text=`) and return a thorough breakdown.

    Pass `text=` to skip fetching (for tests, or a local paper). `on_event` /
    `inspect_vars` are forwarded to the engine so a UI can observe the run. The step
    and budget ceilings are deliberately generous so the recursive path has room to
    converge (via FINAL) rather than tripping the deterministic fallback.
    """
    from .fetch import fetch_arxiv

    if text is None:
        if not url_or_id:
            raise ValueError("provide an arXiv id/URL, or text=")
        text = fetch_arxiv(url_or_id)

    backend = backend or make_backend(model=model)
    budget = Budget(max_depth=max_depth, max_calls=max_calls, max_tokens=max_tokens)
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
