"""End-to-end engine test — offline, deterministic.

Uses MockBackend (no network) but the REAL LocalSandbox subprocess and the real
bridge protocol, so it exercises: code execution, stdout capture, llm() and rlm()
bridges, recursion (depth), FINAL(), and budget accounting.

Runnable two ways:
    pytest tests/test_engine.py
    python tests/test_engine.py        # prints "ok" on success
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rrl.engine import Budget, MockBackend, run  # noqa: E402


def test_recursive_counting():
    # The root agent: (0) inspect PROMPT and split it, (1) delegate each chunk to a
    # recursive rlm() sub-agent and add the numbers with a flat llm() call, (2) FINAL.
    root_script = [
        "```python\n"
        "print('len', len(PROMPT))\n"
        "chunks = PROMPT.split('|')\n"
        "print('nchunks', len(chunks))\n"
        "```",
        "```python\n"
        "results = [rlm('CHUNK: ' + c) for c in chunks]\n"
        "print('results', results)\n"
        "total = llm('Add these numbers, reply with the sum: ' + ','.join(results))\n"
        "print('total', total)\n"
        "```",
        "```python\n"
        "FINAL('counted: ' + str(results))\n"
        "```",
    ]
    backend = MockBackend(root_script, tool_answers={"Add these numbers": "9"})

    prompt = "a b c|d e f|g h i"  # three chunks
    result = run(prompt, backend, budget=Budget(max_depth=3))

    # Each of the 3 sub-agents returned "3"; the root combined them.
    assert result.output == "counted: ['3', '3', '3']", result.output

    b = result.budget
    assert b.max_depth_seen == 1, b.summary()      # recursion actually went one level deep
    assert b.agents == 4, b.summary()              # 1 root + 3 sub-agents
    assert b.calls == 7, b.summary()               # 3 root steps + 3 sub calls + 1 llm call


def test_error_is_fed_back_not_fatal():
    # A cell that raises should surface the traceback as an observation, not crash.
    root_script = [
        "```python\nraise ValueError('boom')\n```",
        "```python\nFINAL('recovered')\n```",
    ]
    backend = MockBackend(root_script)
    result = run("tiny", backend, budget=Budget(max_depth=2), max_steps=4)
    assert result.output == "recovered", result.output


if __name__ == "__main__":
    test_recursive_counting()
    test_error_is_fed_back_not_fatal()
    print("ok")
