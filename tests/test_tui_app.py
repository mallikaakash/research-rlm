"""End-to-end Textual dashboard test — needs the `tui` extra (textual).

Drives the REAL app headlessly with Textual's pilot: the engine runs on a worker
thread, its events marshal to the UI thread, and the widgets populate. Skips
cleanly (exit 0) when textual isn't installed, so the base test run stays green.

    pip install -e '.[tui]'
    python tests/test_tui_app.py     # prints "ok" (or a skip note)
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import textual  # noqa: F401
except ImportError:
    print("ok (skipped: textual not installed; pip install -e '.[tui]')")
    sys.exit(0)

from textual.widgets import DataTable, RichLog  # noqa: E402

from rlm import Budget, MockBackend, run  # noqa: E402
from tui.app import RLMDashboard  # noqa: E402

_SCRIPT = [
    "```python\nchunks = [PROMPT[i:i+4] for i in range(0, len(PROMPT), 4)]\nprint(len(chunks))\n```",
    "```python\nreport = 'done'\nFINAL(report)\n```",
]


def _target(on_event):
    run("abcdefghijkl", MockBackend(_SCRIPT), budget=Budget(max_depth=1),
        on_event=on_event, inspect_vars=True)


async def _run() -> None:
    app = RLMDashboard(_target, subtitle="test")
    async with app.run_test() as pilot:
        for _ in range(60):
            await pilot.pause()
            await asyncio.sleep(0.05)
            if app.model.done:
                break
        assert app.model.done and app.model.final == "done", app.model.final
        table = app.query_one("#vars", DataTable)
        assert table.row_count >= 3, f"expected the ns table to fill, got {table.row_count} rows"
        assert app.query_one("#log", RichLog).lines, "log should have streamed events"
        names = {r[1] for r in app.model.variable_rows()}
        assert {"PROMPT", "chunks", "report"} <= names, names


def test_dashboard_end_to_end():
    asyncio.run(_run())


if __name__ == "__main__":
    test_dashboard_end_to_end()
    print("ok")
