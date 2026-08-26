"""The recursive REPL loop — ingredients #2 (recursion) and #4 (the exec loop).

`run()` is the whole engine in one function:

    call the model -> extract its code -> run it in the sandbox -> feed the
    (truncated) output back -> repeat, until the model calls FINAL().

Two bridges are wired into every agent:
  - llm(text)  : a single, flat model call (no loop). Cheap chunk reader.
  - rlm(text)  : a full recursive sub-agent — this function, called again at
                 depth+1, sharing the same Budget.

That one line (rlm -> run) is the entire "recursive" in Recursive Language Model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from .backend import Backend
from .budget import Budget, BudgetExceeded
from .prompts import SYSTEM_ROOT, initial_user
from .sandbox import LocalSandbox

_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> Optional[str]:
    """Pull the first fenced code block out of a model reply, if any."""
    m = _CODE.search(text or "")
    return m.group(1) if m else None


@dataclass
class Result:
    output: object
    budget: Budget


def run(
    prompt,
    backend: Backend,
    *,
    budget: Budget | None = None,
    depth: int = 0,
    max_steps: int = 8,
    model: str | None = None,
    sandbox_factory: Callable = LocalSandbox,
    on_event: Optional[Callable[[dict], None]] = None,
) -> Result:
    """Run one RLM agent over `prompt`; recurse via the rlm() bridge."""
    budget = budget or Budget()
    budget.enter(depth)  # enforces max_depth; may raise BudgetExceeded

    def emit(event: dict) -> None:
        if on_event:
            on_event({"depth": depth, **event})

    # ---- the two recursion bridges ----
    def _llm(text) -> str:
        budget.check()
        out, usage = backend.complete([{"role": "user", "content": str(text)}], model=model)
        budget.add(usage)
        emit({"type": "llm", "in": str(text)[:120], "out": str(out)[:120]})
        return out

    def _rlm(subprompt) -> object:
        emit({"type": "rlm", "in": str(subprompt)[:120]})
        return run(
            str(subprompt),
            backend,
            budget=budget,
            depth=depth + 1,
            max_steps=max_steps,
            model=model,
            sandbox_factory=sandbox_factory,
            on_event=on_event,
        ).output

    bridges = {"llm": _llm, "rlm": _rlm}
    sandbox = sandbox_factory(prompt, list(bridges))

    messages = [
        {"role": "system", "content": SYSTEM_ROOT},
        {"role": "user", "content": initial_user(prompt)},
    ]

    final: object = None
    try:
        for step in range(max_steps):
            budget.check()
            text, usage = backend.complete(messages, model=model)
            budget.add(usage)
            messages.append({"role": "assistant", "content": text})
            emit({"type": "assistant", "step": step, "text": text})

            code = extract_code(text)
            if code is None:
                messages.append(
                    {
                        "role": "user",
                        "content": "Reply with exactly one ```python code block. "
                        "Explore PROMPT, then call FINAL(answer) to finish.",
                    }
                )
                continue

            res = sandbox.run_cell(code, bridges)
            emit(
                {
                    "type": "cell",
                    "step": step,
                    "stdout": res.stdout,
                    "error": res.error,
                    "has_final": res.has_final,
                }
            )

            if res.has_final:
                final = res.final
                break

            obs = res.error if res.error else res.stdout
            if not obs:
                obs = "(no output)"
            messages.append({"role": "user", "content": f"REPL output:\n{obs[:4000]}"})
    except BudgetExceeded as e:
        if final is None:
            final = f"[budget exceeded: {e}]"
    finally:
        sandbox.close()

    return Result(output=final, budget=budget)
