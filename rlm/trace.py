"""Pretty terminal rendering of an RLM run — so you can actually see the code the
model wrote and read the recursion at a glance.

`RichRenderer` is an `on_event` callable (same shape the engine already emits), so
`run(..., on_event=RichRenderer())` renders a run live. Each event type gets a
fixed color so the trace is scannable:

    agent reasoning   cyan       (panel title + preamble)
    python code       highlighted (monokai, inside the panel)
    stdout (ok)       green
    error             red
    FINAL(...)        gold        (the answer stands out)
    rlm(...) spawn    magenta     (recursion; nested by depth)
    llm(...) call     dim blue

Output goes to stderr, leaving stdout for the final answer (pipes stay clean).
"""

from __future__ import annotations

import re

_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def _short(v, n: int = 100) -> str:
    s = str(v).replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


class RichRenderer:
    def __init__(self, console=None):
        from rich.console import Console  # imported here so --pretty is the only path needing rich

        self.console = console or Console(stderr=True)

    def __call__(self, e: dict) -> None:
        t = e.get("type")
        pad = (0, 0, 0, e.get("depth", 0) * 3)  # indent by recursion depth
        if t == "assistant":
            self._assistant(e, pad)
        elif t == "cell":
            self._cell(e, pad)
        elif t == "rlm":
            self._line(f"↳ rlm({_short(e.get('in'))})", "magenta", pad)
        elif t == "llm":
            self._line(f"↳ llm(…) → {_short(e.get('out'))}", "dim blue", pad)
        elif t == "tool":
            self._line(f"⚙ {e.get('name')}({_short(e.get('args'))})", "yellow", pad)

    # ---- renderers ----
    def _line(self, text: str, style: str, pad) -> None:
        from rich.padding import Padding
        from rich.text import Text

        self.console.print(Padding(Text(text, style=style), pad))

    def _assistant(self, e, pad) -> None:
        from rich.console import Group
        from rich.padding import Padding
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.text import Text

        text = e.get("text") or ""
        m = _CODE.search(text)
        code = m.group(1).strip() if m else None
        preamble = (text[: m.start()] if m else text).strip()

        parts = []
        if preamble:
            parts.append(Text(preamble, style="cyan"))
        if code:
            parts.append(
                Syntax(code, "python", theme="monokai", background_color="default", word_wrap=True)
            )
        body = Group(*parts) if parts else Text(text or "(empty)")
        title = f"[bold cyan]agent[{e.get('depth', 0)}·{e.get('step', '?')}][/]"
        self.console.print(
            Padding(Panel(body, title=title, title_align="left", border_style="cyan"), pad)
        )

    def _cell(self, e, pad) -> None:
        from rich.padding import Padding
        from rich.panel import Panel
        from rich.pretty import Pretty
        from rich.text import Text

        if e.get("has_final"):
            final = e.get("final")
            content = Pretty(final) if isinstance(final, (dict, list)) else Text(str(final))
            self.console.print(
                Padding(
                    Panel(content, title="[bold gold1]FINAL[/]", title_align="left", border_style="gold1"),
                    pad,
                )
            )
            return
        if e.get("error"):
            self.console.print(
                Padding(
                    Panel(Text(e["error"].rstrip(), style="red"), title="[red]error[/]", title_align="left", border_style="red"),
                    pad,
                )
            )
            return
        out = (e.get("stdout") or "").rstrip() or "(no output)"
        self.console.print(
            Padding(Panel(Text(out), title="[green]output[/]", title_align="left", border_style="green"), pad)
        )
