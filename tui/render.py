"""Rich renderables for the dashboard's left panel — one engine event at a time.

Kept separate from the Textual app (and from the plain-text DashboardModel) so it
can be unit-tested without a terminal. Each event becomes a small list of Rich
renderables that RichLog.write() can render directly: the code the model wrote
(syntax-highlighted), and the observation that will feed its next turn.
"""

from __future__ import annotations

import re

_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def _oneline(v, n: int = 160) -> str:
    s = str(v).replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def _block(s: str, max_lines: int = 14, max_chars: int = 1000) -> str:
    """Trim a multi-line blob for display, keeping newlines (RichLog wraps them)."""
    s = (s or "").rstrip()
    lines = s.splitlines()
    clipped = lines[:max_lines]
    text = "\n".join(clipped)
    if len(text) > max_chars:
        text = text[:max_chars] + " …"
    if len(lines) > max_lines:
        text += f"\n… (+{len(lines) - max_lines} more lines)"
    return text or "(no output)"


def turn_renderables(e: dict) -> list:
    """Rich renderables for one event, indented by recursion depth."""
    from rich.syntax import Syntax
    from rich.text import Text

    t = e.get("type")
    depth = e.get("depth", 0)
    pad = "  " * depth
    out: list = []

    if t == "assistant":
        text = e.get("text") or ""
        m = _CODE.search(text)
        preamble = (text[: m.start()] if m else text).strip()
        out.append(Text(f"{pad}▸ agent[{depth}·{e.get('step')}]", style="bold cyan"))
        if preamble:
            out.append(Text(f"{pad}  {_oneline(preamble, 220)}", style="cyan"))
        if m:  # the code the model ran this turn — the "action"
            out.append(Syntax(m.group(1).strip(), "python", theme="monokai",
                              background_color="default", word_wrap=True))
    elif t == "cell":
        if e.get("has_final"):
            out.append(Text(f"{pad}  ★ FINAL", style="bold gold1"))
            out.append(Text(f"{pad}  {_oneline(e.get('final'), 500)}", style="gold1"))
        elif e.get("error"):
            out.append(Text(f"{pad}  ✗ error", style="bold red"))
            out.append(Text(_block(e.get("error"), max_lines=8), style="red"))
        else:  # stdout = the observation fed back as the next turn's context
            out.append(Text(f"{pad}  ↩ output (feeds next turn)", style="dim green"))
            out.append(Text(_block(e.get("stdout")), style="green"))
        out.append(Text(""))  # blank line = a visible turn boundary
    elif t == "rlm":
        out.append(Text(f"{pad}  ↳ rlm({_oneline(e.get('in'), 140)})", style="bold magenta"))
    elif t == "llm":
        out.append(Text(f"{pad}  ↳ llm(…) → {_oneline(e.get('out'), 140)}", style="blue"))
    elif t == "tool":
        out.append(Text(f"{pad}  ⚙ {e.get('name')}({_oneline(e.get('args'), 120)})", style="yellow"))
    return out
