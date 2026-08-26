"""Budgets: the guardrails that keep a recursive loop from running away.

A single Budget is shared across an entire recursion tree (the root agent and
every sub-agent it spawns), so limits are global, not per-agent. The loop calls
`enter()` when an agent starts, `add()` after every model call, and `check()`
before doing more work.
"""

from __future__ import annotations

from dataclasses import dataclass


class BudgetExceeded(Exception):
    """Raised when a recursion would exceed a configured limit."""


@dataclass
class Budget:
    # ---- limits ----
    max_depth: int = 4              # how deep rlm() may recurse
    max_calls: int = 100            # total model calls across the whole tree
    max_tokens: int = 2_000_000     # total tokens across the whole tree

    # ---- running totals (do not set by hand) ----
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    agents: int = 0                 # how many agents (root + subs) were started
    max_depth_seen: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def enter(self, depth: int) -> None:
        """Register a new agent starting at `depth`; enforce the depth limit."""
        if depth > self.max_depth:
            raise BudgetExceeded(f"max_depth={self.max_depth} exceeded (depth={depth})")
        self.agents += 1
        self.max_depth_seen = max(self.max_depth_seen, depth)

    def add(self, usage) -> None:
        """Fold in the cost of one model call, then re-check limits."""
        self.calls += usage.calls
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.check()

    def check(self) -> None:
        if self.calls > self.max_calls:
            raise BudgetExceeded(f"max_calls={self.max_calls} exceeded (calls={self.calls})")
        if self.total_tokens > self.max_tokens:
            raise BudgetExceeded(f"max_tokens={self.max_tokens} exceeded (tokens={self.total_tokens})")

    def summary(self) -> str:
        return (
            f"agents={self.agents} depth={self.max_depth_seen} "
            f"calls={self.calls} tokens={self.total_tokens}"
        )
