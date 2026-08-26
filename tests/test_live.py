"""Live smoke test — a REAL model round-trip through the full engine.

Skips cleanly when no key is set, so it never breaks offline/CI runs. Uses the
default provider (DeepSeek unless $RLM_PROVIDER says otherwise).

Run it:
    export DEEPSEEK_API_KEY=sk-...
    python tests/test_live.py           # prints the answer + budget
    pytest tests/test_live.py           # asserts the loop found the hidden fact
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rrl.engine import Budget, make_backend, run  # noqa: E402

_HAVE_KEY = bool(
    os.environ.get("DEEPSEEK_API_KEY")
    or os.environ.get("OPENROUTER_API_KEY")
    or os.environ.get("RLM_API_KEY")
)


def _run():
    # A hidden fact buried in filler — the agent must explore PROMPT with code to
    # find it rather than reading it all, then FINAL just the number.
    filler = "lorem ipsum dolor sit amet " * 200
    prompt = filler + "\n\nIMPORTANT: the secret code is 4471.\n\n" + filler
    backend = make_backend()  # DeepSeek by default
    result = run(
        prompt,
        backend,
        budget=Budget(max_depth=3),
        max_steps=8,
        on_event=lambda e: sys.stderr.write(f"  · {e.get('type')}\n"),
    )
    return result


def test_live_finds_hidden_fact():
    import pytest  # local import so plain `python tests/test_live.py` needs no pytest

    if not _HAVE_KEY:
        pytest.skip("no API key set (DEEPSEEK_API_KEY / OPENROUTER_API_KEY / RLM_API_KEY)")
    result = _run()
    assert "4471" in str(result.output), result.output


if __name__ == "__main__":
    if not _HAVE_KEY:
        print("skip: set DEEPSEEK_API_KEY (or OPENROUTER_API_KEY / RLM_API_KEY) first")
        raise SystemExit(0)
    result = _run()
    print("\noutput:", result.output)
    print("budget:", result.budget.summary())
    print("ok" if "4471" in str(result.output) else "MISSED the hidden fact")
