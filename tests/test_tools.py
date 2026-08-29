"""tools= hook test — offline.

Verifies the engine can be handed arbitrary host-side tools that the model calls
(via `await name(...)`) inside the sandbox, with results returning as values. This
is what lets a harness (research retrieval, a coding agent) extend the engine
without changing it.

    python tests/test_tools.py     # prints "ok"
    pytest tests/test_tools.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlm import Budget, MockBackend, run  # noqa: E402


def test_tool_injection():
    calls = []

    def add(a, b):
        calls.append((a, b))
        return a + b

    script = [
        "```python\n"
        "s = await add(2, 3)\n"      # a harness tool, awaited like llm/rlm
        "print('sum', s)\n"
        "FINAL(s)\n"
        "```",
    ]
    result = run("anything", MockBackend(script), budget=Budget(max_depth=1), tools={"add": add})

    assert result.output == 5, result.output      # the tool's value came back into the REPL
    assert calls == [(2, 3)], calls               # the host tool actually ran, on the host


def test_llm_rlm_are_reserved():
    # A tool named "llm" must not shadow the real llm bridge.
    script = ["```python\nout = await llm('hi')\nFINAL(out)\n```"]
    backend = MockBackend(script, tool_answers={"hi": "real-llm"})
    result = run("x", backend, budget=Budget(max_depth=1), tools={"llm": lambda *_: "SHADOW"})
    assert result.output == "real-llm", result.output


if __name__ == "__main__":
    test_tool_injection()
    test_llm_rlm_are_reserved()
    print("ok")
