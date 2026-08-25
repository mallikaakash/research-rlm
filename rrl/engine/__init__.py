"""The RLM engine: the recursive REPL loop, its sandbox, backends, and budgets.

This is the *runtime* — stateless, reusable. You hand it a prompt (which may be
enormous), a model backend, and a budget; it runs the recursive loop and returns
a result. Everything stateful (the corpus, goals, proactivity) lives above it in
the harness, not here.

The four ingredients of any RLM live in this package:
  1. a swappable model backend            -> backend.py
  2. rlm()/llm() recursion as bridges      -> loop.py  (+ sandbox bridge protocol)
  3. the prompt held as a REPL variable    -> sandbox.py / worker.py
  4. the exec loop with truncated feedback -> loop.py
"""

from .backend import Backend, MockBackend, OpenRouterBackend, Usage
from .budget import Budget, BudgetExceeded
from .loop import Result, run
from .sandbox import CellResult, LocalSandbox, SandboxTimeout

__all__ = [
    "run",
    "Result",
    "Budget",
    "BudgetExceeded",
    "Backend",
    "OpenRouterBackend",
    "MockBackend",
    "Usage",
    "LocalSandbox",
    "CellResult",
    "SandboxTimeout",
]
