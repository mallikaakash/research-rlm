"""The Textual dashboard app + the `rrl-tui` command.

Architecture (the whole point — see the module docstring in __init__.py):

    UI THREAD (Textual/asyncio)          WORKER THREAD (run_worker thread=True)
    ─────────────────────────            ──────────────────────────────────────
    stays responsive, owns all           runs the blocking engine (run/explain);
    widgets; applies events              its on_event fires here and hands each
    handed over from the worker  ◀──────  event to the UI thread via
                                          call_from_thread — never touches a
                                          widget directly.

Blocking work runs on the worker (it's allowed to park on pipe/socket I/O — that
releases the GIL). Every widget mutation happens on the UI thread. Events cross
the boundary through Textual's thread-safe call_from_thread — that's the queue.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import DataTable, Footer, Header, RichLog

from .render import turn_renderables
from .state import DashboardModel

# A run target runs the (blocking) engine on the worker thread. It's handed an
# on_event callback and returns the engine result (something with .output), if any.
RunTarget = Callable[[Callable[[dict], None]], object]


class RLMDashboard(App):
    """Live view of one RLM run: recursion + context (left), REPL namespace (right)."""

    CSS = """
    Horizontal { height: 1fr; }
    #log  { width: 3fr; border: round $accent;    padding: 0 1; }
    #vars { width: 2fr; border: round $success; }
    """
    BINDINGS = [("q", "quit", "Quit"), ("ctrl+c", "quit", "Quit")]

    def __init__(self, run_target: RunTarget, subtitle: str = ""):
        super().__init__()
        self._run_target = run_target
        self.model = DashboardModel()
        self.title = "RLM dashboard"
        self.sub_title = subtitle or "running…"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            yield RichLog(id="log", highlight=False, markup=False, wrap=True, auto_scroll=True)
            yield DataTable(id="vars", zebra_stripes=True, cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one("#log", RichLog)
        log.border_title = "recursion + context (what each turn ran & saw)"

        table = self.query_one("#vars", DataTable)
        table.border_title = "REPL namespace — ns (builds up as it works)"
        table.add_column("d", width=1)
        table.add_column("name", width=15)
        table.add_column("type", width=12)
        table.add_column("value", width=46)

        # Kick off the (blocking) engine on a background thread. Its events are
        # marshalled back to us via call_from_thread; the UI stays live meanwhile.
        self.run_worker(self._run_engine, thread=True, name="engine")

    # ---- worker thread ----
    def _run_engine(self) -> None:
        result = None
        try:
            result = self._run_target(self._emit)  # blocks here on the worker thread — fine
        except Exception as e:  # noqa: BLE001 — surface engine failures in the UI, don't crash it
            self.call_from_thread(self._write, Text(f"[engine error] {e!r}", style="bold red"))
        finally:
            self.call_from_thread(self._on_done, result)

    def _emit(self, e: dict) -> None:
        """on_event, called on the WORKER thread. Hand the event to the UI thread."""
        try:
            self.call_from_thread(self._apply, e)
        except Exception:  # noqa: BLE001 — app may be shutting down; drop the event
            pass

    # ---- UI thread ----
    def _apply(self, e: dict) -> None:
        self.model.note(e)
        for renderable in turn_renderables(e):
            self._write(renderable)
        if e.get("type") in ("vars", "cell"):
            self._refresh_vars()

    def _write(self, renderable) -> None:
        self.query_one("#log", RichLog).write(renderable)

    def _refresh_vars(self) -> None:
        table = self.query_one("#vars", DataTable)
        table.clear()
        for row in self.model.variable_rows():
            table.add_row(*row)

    def _on_done(self, result=None) -> None:
        self.sub_title = "done — press q to quit"
        self._write(Rule("run complete", style="dim"))
        output = getattr(result, "output", None)
        if output:
            fb = " · fallback synthesis" if getattr(result, "used_fallback", False) else ""
            body = str(output)
            preview = body if len(body) <= 4000 else body[:4000] + f"\n… (+{len(body) - 4000} more chars)"
            self._write(Panel(Text(preview), title=f"FINAL breakdown{fb}",
                              title_align="left", border_style="gold1"))


# --------------------------------------------------------------------------- CLI


def _build_target(args, backend, model) -> tuple[RunTarget, str]:
    """Build the worker-thread run target from CLI args."""
    if args.arxiv:
        from paper.explain import explain

        def target(on_event):
            return explain(
                args.arxiv, backend, model=model,
                max_steps=args.max_steps, max_depth=args.max_depth,
                on_event=on_event, inspect_vars=True,
            )

        return target, f"arxiv:{args.arxiv}"

    from rlm import Budget, run

    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        label = args.input_file
    else:
        text = args.query
        label = (args.query or "")[:40]

    def target(on_event):
        return run(
            text, backend, budget=Budget(max_depth=args.max_depth),
            max_steps=args.max_steps, model=model,
            on_event=on_event, inspect_vars=True,
        )

    return target, label


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rrl-tui", description="Live Textual dashboard for an RLM run.")
    ap.add_argument("query", nargs="?", help="a prompt/task to run")
    ap.add_argument("--arxiv", help="explain an arXiv paper (id or URL) instead of a raw prompt")
    ap.add_argument("--input-file", help="run over the contents of this file")
    ap.add_argument("--provider", default=os.environ.get("RLM_PROVIDER", "deepseek"))
    ap.add_argument("--model", default=os.environ.get("RLM_MODEL"))
    ap.add_argument("--max-steps", type=int, default=14)
    ap.add_argument("--max-depth", type=int, default=3)
    args = ap.parse_args(argv)

    if not (args.query or args.arxiv or args.input_file):
        ap.error("provide a query, --arxiv, or --input-file")

    from rlm import make_backend

    try:
        backend = make_backend(args.provider, model=args.model)
    except RuntimeError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2

    target, label = _build_target(args, backend, args.model)
    RLMDashboard(target, subtitle=label).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
