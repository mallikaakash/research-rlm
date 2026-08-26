"""PyodideSandbox test — the REAL WASM sandbox, offline (mock model).

Verifies the model's Python runs inside CPython-in-WASM (sys.platform ==
'emscripten'), that `await rlm(...)` recurses through the host, and that FINAL()
returns. Skips cleanly when Node or the pyodide package isn't available.

    npm install pyodide        # once, at repo root
    python tests/test_pyodide.py
    pytest tests/test_pyodide.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rrl.engine import Budget, MockBackend, PyodideSandbox, pyodide_available, run  # noqa: E402


def _scenario():
    # Root: confirm we're in WASM, delegate one chunk to a recursive sub-agent, FINAL.
    root_script = [
        "```python\n"
        "import sys\n"
        "print('platform', sys.platform)\n"
        "r = await rlm('CHUNK: anything')\n"
        "FINAL('got:' + r)\n"
        "```",
    ]
    backend = MockBackend(root_script)  # any 'CHUNK' request -> sub-agent FINAL('3')
    return run(
        "hello world",
        backend,
        budget=Budget(max_depth=2),
        max_steps=4,
        sandbox_factory=PyodideSandbox,
    )


def test_pyodide_recursion():
    import pytest

    if not pyodide_available():
        pytest.skip("Node or the pyodide package is unavailable (run `npm install`)")
    result = _scenario()
    assert result.output == "got:3", result.output
    assert result.budget.max_depth_seen == 1, result.budget.summary()


if __name__ == "__main__":
    if not pyodide_available():
        print("skip: run `npm install` (needs Node + the pyodide package)")
        raise SystemExit(0)
    result = _scenario()
    print("output:", result.output)
    print("budget:", result.budget.summary())
    print("ok" if result.output == "got:3" else "FAILED")
