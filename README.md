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
| 1. A swappable model backend | `rlm/backend.py` (`OpenRouterBackend`, `MockBackend`) |
| 2. `llm()` / `rlm()` recursion as bridges | `rlm/loop.py` + the sandbox bridge protocol |
| 3. The prompt held as a REPL variable | `rlm/sandbox.py` + `rlm/worker.py` |
| 4. The exec loop with truncated feedback | `rlm/loop.py` |

Plus budgets (`budget.py`) that bound depth, calls, and tokens across the whole
recursion tree.

**How a run works:** the model replies with one ```` ```python ```` block per turn →
it runs in a persistent worker subprocess → whatever it prints is fed back → repeat
until it calls `FINAL(answer)`. Inside the REPL the model can call `llm(text)` (a
flat sub-model call) or `rlm(text)` (a full recursive sub-agent); those run on the
host and return values, so the model composes them with ordinary Python.

## Run it

The default provider is **DeepSeek** (any OpenAI-compatible provider works):

```bash
uv sync                       # or: pip install -e .
export DEEPSEEK_API_KEY=sk-...

rrl-engine "What is 12 * 12? Reply with just the number."
rrl-engine -v --input-file paper.txt "Summarize the single main claim."

# other providers:
rrl-engine --provider openrouter --model openai/gpt-5-mini "..."
```

`-v` traces the recursion (agents, cells, `llm`/`rlm` calls) on stderr. Models:
`deepseek-chat` (default) or `deepseek-reasoner`; override with `--model`. Key
lookup order is the provider's env var (`DEEPSEEK_API_KEY`) → `RLM_API_KEY`; set
`RLM_PROVIDER` / `RLM_BASE_URL` to change defaults.

## Test it

Offline (no API key, deterministic — drives the **real** sandbox subprocess and
bridge protocol with a mock model):

```bash
python tests/test_engine.py        # prints "ok"
# or:  pytest tests/test_engine.py
```

Live smoke test (a real DeepSeek round-trip through the whole engine; skips when
no key is set):

```bash
export DEEPSEEK_API_KEY=sk-...
python tests/test_live.py          # finds a hidden fact by exploring PROMPT with code
```

## Sandboxes

Two, behind one `run_cell` / `close` interface (pick with `--sandbox`):

- **`local`** (default) — a **Python subprocess** (`LocalSandbox`). Process
  isolation + kill/timeout, but **not** security isolation; code here can still
  touch the host. Fast, no extra deps — good for tests and trusted input.
- **`pyodide`** — **CPython compiled to WebAssembly** (`PyodideSandbox`), hosted in
  Node. The model's Python runs with **no syscalls**: it cannot reach the real
  filesystem or network. The only way out is a bridge (`llm`/`rlm`), forwarded to
  the host. Real sandboxing — recommended before any unattended/autonomous use.

Enable the WASM sandbox once:

```bash
npm install                 # installs the pyodide package (needs Node.js)
rrl-engine --sandbox pyodide "..."
```

Note: `llm()` and `rlm()` are **async** — the model calls them as `await llm(...)`
/ `await rlm(...)` (the WASM host round-trip is async; the same contract is used in
both sandboxes). Each Pyodide sandbox loads its own WASM (~seconds), so deep
recursion spawns several; a sandbox pool is a later optimization.

## Layout

Clean demarcation: **`rlm/` is the pure engine** (reusable for *any* task, imports
nothing internal); **`research/` is the harness** on top (imports `rlm`, never the
reverse).

```
rlm/                     # THE ENGINE — the core RLM runtime (task-agnostic)
  backend.py             # ingredient 1 — model backends (DeepSeek / OpenRouter / mock)
  sandbox.py             # ingredient 3 — shared protocol + LocalSandbox (subprocess)
  worker.py              #                local sandbox side: PROMPT, exec, bridge proxies
  pyodide_sandbox.py     # the WASM sandbox host (spawns Node)
  pyodide_worker.mjs     # WASM sandbox side: CPython-in-WASM + async bridges
  loop.py                # ingredients 2 & 4 — the recursive REPL loop
  budget.py              # depth / call / token limits, shared across the tree
  prompts.py             # the system prompt + first-turn framing
  trace.py               # the --pretty renderer (engine observability)
  cli.py                 # `rrl-engine`
research/                # THE HARNESS — this domain (stands on rlm/)
  note.py                # the Note schema (claims as the atom)
  read.py                # the read pipeline (paper -> Note); `rrl-read`
  corpus.py              # corpus ops: link / contradictions / pick_goal
seed_corpus.yaml         # the first corpus to read
tests/                   # offline suites: engine · read · corpus · pyodide (+ live)
```

## Not built yet

The **retrieval front-end** (search → fetch, to populate the corpus) and the
**proactive loop** (schedule → pick_goal → research) — both stand on what's here.
The engine (`rlm/`) is task-agnostic: point a different harness at it (a `coding/`
or `ops/` sibling) and it's a different agent, no engine change.
