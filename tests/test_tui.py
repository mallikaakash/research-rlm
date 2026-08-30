"""Dashboard model tests — the framework-free event->view logic (no terminal).

    python tests/test_tui.py     # prints "ok"
    pytest tests/test_tui.py

The Textual app in tui/app.py is thin glue over this; the interesting logic is
here and needs no TTY to test.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tui import DashboardModel  # noqa: E402


def test_log_lines_by_type():
    m = DashboardModel()
    assert "agent[0·0]" in m.log_line({"type": "assistant", "depth": 0, "step": 0, "text": "thinking"})
    assert "rlm(" in m.log_line({"type": "rlm", "depth": 0, "in": "§3"})
    assert "llm(" in m.log_line({"type": "llm", "depth": 1, "out": "note"})
    assert "FINAL" in m.log_line({"type": "cell", "depth": 0, "has_final": True, "final": "answer"})
    assert m.log_line({"type": "vars", "depth": 0, "vars": []}) is None  # vars drive the table, not the log


def test_depth_indent():
    m = DashboardModel()
    root = m.log_line({"type": "rlm", "depth": 0, "in": "x"})
    child = m.log_line({"type": "rlm", "depth": 2, "in": "x"})
    lead = lambda s: len(s) - len(s.lstrip(" "))
    assert lead(child) > lead(root), "deeper events indent more"
    assert lead(child) - lead(root) == 4, "each depth adds 2 spaces"


def test_variables_populate_and_refresh():
    m = DashboardModel()
    m.note({"type": "vars", "depth": 0, "vars": [
        {"name": "PROMPT", "type": "str", "repr": "'abc'", "size": 3},
    ]})
    rows = m.variable_rows()
    assert rows == [("0", "PROMPT", "str[3]", "'abc'")], rows  # size folded into type

    # a later snapshot at the same depth REPLACES (not appends)
    m.note({"type": "vars", "depth": 0, "vars": [
        {"name": "PROMPT", "type": "str", "repr": "'abc'", "size": 3},
        {"name": "chunks", "type": "list", "repr": "[...]", "size": 5},
    ]})
    assert len(m.variable_rows()) == 2

    # a sub-agent's vars live alongside, keyed by depth
    m.note({"type": "vars", "depth": 1, "vars": [{"name": "note", "type": "str", "repr": "'x'", "size": 1}]})
    rows = m.variable_rows()
    assert [r[0] for r in rows] == ["0", "0", "1"], rows  # sorted by depth


def test_done_tracks_root_final_only():
    m = DashboardModel()
    m.note({"type": "cell", "depth": 1, "has_final": True, "final": "sub"})
    assert not m.done, "a sub-agent FINAL must not end the run"
    m.note({"type": "cell", "depth": 0, "has_final": True, "final": "root answer"})
    assert m.done and m.final == "root answer"


def test_render_shows_code_and_output():
    from rich.syntax import Syntax

    from tui.render import turn_renderables

    # an assistant turn renders a header + the code it ran (syntax-highlighted)
    rs = turn_renderables({
        "type": "assistant", "depth": 0, "step": 1,
        "text": "Let me chunk it.\n```python\nchunks = split(PROMPT)\n```",
    })
    assert any(isinstance(r, Syntax) for r in rs), "code should render as Syntax"

    # a cell turn renders the observation that feeds the next turn
    rs = turn_renderables({"type": "cell", "depth": 0, "stdout": "Total length: 131809"})
    joined = " ".join(getattr(r, "plain", "") for r in rs)
    assert "131809" in joined and "feeds next turn" in joined, joined

    # a FINAL cell is marked
    rs = turn_renderables({"type": "cell", "depth": 0, "has_final": True, "final": "# Report"})
    assert any("FINAL" in getattr(r, "plain", "") for r in rs)


if __name__ == "__main__":
    test_log_lines_by_type()
    test_depth_indent()
    test_variables_populate_and_refresh()
    test_done_tracks_root_final_only()
    test_render_shows_code_and_output()
    print("ok")
