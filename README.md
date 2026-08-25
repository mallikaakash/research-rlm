# research-rlm

A clean, minimal **Recursive Language Model (RLM)** — engine and brain, built from
scratch. This repo currently contains **v1 of the engine**: the recursive REPL
runtime that everything else will stand on.

An RLM treats context as an *environment*, not as tokens in a window: the input is
stored as a variable in a sandboxed REPL, and the model writes code to explore it
and recursively call itself on sub-pieces. See the
[architecture overview](https://claude.ai/code/artifact/24357287-1074-4c4a-b1ff-c291ddfa3337).

## What v1 is

The engine is the four ingredients of any RLM, and nothing else:

| Ingredient | Where |
|---|---|
| 1. A swappable model backend | `rrl/engine/backend.py` (`OpenRouterBackend`, `MockBackend`) |
| 2. `llm()` / `rlm()` recursion as bridges | `rrl/engine/loop.py` + the sandbox bridge protocol |
| 3. The prompt held as a REPL variable | `rrl/engine/sandbox.py` + `rrl/engine/worker.py` |
| 4. The exec loop with truncated feedback | `rrl/engine/loop.py` |

Plus budgets (`budget.py`) that bound depth, calls, and tokens across the whole
recursion tree.

**How a run works:** the model replies with one ```` ```python ```` block per turn →
it runs in a persistent worker subprocess → whatever it prints is fed back → repeat
until it calls `FINAL(answer)`. Inside the REPL the model can call `llm(text)` (a
flat sub-model call) or `rlm(text)` (a full recursive sub-agent); those run on the
host and return values, so the model composes them with ordinary Python.

## Run it

```bash
uv sync                       # or: pip install -e .
export OPENROUTER_API_KEY=sk-...

rrl-engine "What is 12 * 12? Reply with just the number."
rrl-engine -v --input-file paper.txt "Summarize the single main claim."
```

`-v` traces the recursion (agents, cells, `llm`/`rlm` calls) on stderr.

## Test it (no API key needed)

```bash
python tests/test_engine.py        # prints "ok"
# or:  pytest
```

The test drives the **real** sandbox subprocess and bridge protocol with a
deterministic mock model, exercising code execution, recursion, `FINAL()`, and
budget accounting end to end.

## The sandbox, honestly

v1 runs model code in a **Python subprocess** (`LocalSandbox`). That buys process
isolation plus kill/timeout — **not** security isolation; code in the sandbox can
still touch the real machine. It sits behind a small interface (`run_cell` /
`close`), so a **Pyodide/WASM sandbox** (`PyodideSandbox`) is a drop-in replacement,
which is the planned next step. Run v1 on inputs you trust (e.g. arXiv papers you
fetched).

## Layout

```
rrl/
  engine/
    backend.py     # ingredient 1 — model backends
    sandbox.py     # ingredient 3 — host side: spawn worker, run cells, bridge calls
    worker.py      #                sandbox side: holds PROMPT, execs code, proxies bridges
    loop.py        # ingredients 2 & 4 — the recursive REPL loop
    budget.py      # depth / call / token limits, shared across the tree
    prompts.py     # the system prompt + first-turn framing
  cli.py           # `rrl-engine`
tests/
  test_engine.py   # offline end-to-end test
```

## Not built yet

The *brain* — structured paper notes, the flat-file corpus, `link()` /
`contradictions()`, the reactive research pipeline, and the proactive loop — all
stand on this engine and come next.
