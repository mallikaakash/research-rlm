"""Framework-free view logic for the dashboard.

`DashboardModel` turns raw engine events (the dicts the engine emits through
on_event) into the two things the UI renders: a scrolling log of the recursion
(left) and the current REPL namespace (right). Keeping this free of Textual means
the interesting logic is unit-testable with plain asserts — no terminal needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _short(v, n: int = 90) -> str:
    s = str(v).replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


@dataclass
class DashboardModel:
    # Latest REPL namespace snapshot, keyed by depth so a sub-agent's vars don't
    # clobber the root's. Each value is the list of var-summary dicts from the engine.
    variables_by_depth: dict[int, list] = field(default_factory=dict)
    done: bool = False
    final: object = None

    def log_line(self, e: dict) -> str | None:
        """One scannable line for the left panel, indented by recursion depth.

        Returns None for events that don't belong in the stream (e.g. vars, which
        drive the right panel instead).
        """
        depth = e.get("depth", 0)
        indent = "  " * depth
        t = e.get("type")

        if t == "assistant":
            head = (e.get("text") or "").strip().splitlines()
            return f"{indent}▸ agent[{depth}·{e.get('step')}] {_short(head[0] if head else '')}"
        if t == "cell":
            if e.get("has_final"):
                return f"{indent}  ★ FINAL {_short(e.get('final'))}"
            if e.get("error"):
                return f"{indent}  ✗ error {_short((e.get('error') or '').strip().splitlines()[-1:] or '')}"
            return f"{indent}  · output {_short((e.get('stdout') or '').strip())}"
        if t == "rlm":
            return f"{indent}  ↳ rlm({_short(e.get('in'))})"
        if t == "llm":
            return f"{indent}  ↳ llm(…) → {_short(e.get('out'))}"
        if t == "tool":
            return f"{indent}  ⚙ {e.get('name')}({_short(e.get('args'))})"
        return None

    def note(self, e: dict) -> None:
        """Fold an event into persistent state (the namespace, done/final)."""
        if e.get("type") == "vars":
            self.variables_by_depth[e.get("depth", 0)] = e.get("vars") or []
        elif e.get("type") == "cell" and e.get("has_final"):
            # a FINAL at depth 0 ends the whole run
            if e.get("depth", 0) == 0:
                self.done = True
                self.final = e.get("final")

    def variable_rows(self) -> list[tuple[str, str, str, str]]:
        """Flat (depth, name, type, value) rows for the right-panel table, deepest
        namespaces last so the root's variables read first."""
        rows: list[tuple[str, str, str, str]] = []
        for depth in sorted(self.variables_by_depth):
            for v in self.variables_by_depth[depth]:
                size = v.get("size")
                # fold the size into the type column (str[131809]) so the value
                # column is pure content and stays readable
                type_str = v.get("type", "")
                if size is not None:
                    type_str = f"{type_str}[{size}]"
                rows.append((str(depth), v.get("name", ""), type_str, _short(v.get("repr", ""), 46)))
        return rows
