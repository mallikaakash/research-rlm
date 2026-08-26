# LEARNINGS

A running log of things learned while building this project — doubts I had and
what resolved them, Python/CS concepts that were new, and things worth knowing
that came up along the way. Newest entries on top.

Format per entry: **the doubt / context** → **what I learned** → (sometimes) **why it matters**.

---

## 2026-08-26

### Process isolation vs. sandboxing — and how Pyodide/Deno actually work
**Doubt:** our v1 sandbox spawns a subprocess we can kill — but that's not really
safe, is it? It could still mess up my computer. And I couldn't picture how
Pyodide/Deno make it safe.

**What I learned:**
- **Isolation ≠ sandboxing.** *Isolation* = keeping things separate so one can't
  corrupt/crash another (and you can kill it). *Sandboxing* = restricting what a
  thing is *allowed to do* (its capabilities). Our subprocess gives isolation but
  almost no sandboxing.
- A **process** has memory isolation (neighbors can't read its memory) but runs
  **as my user, with my permissions**, sharing the one OS kernel. So a subprocess
  can `os.system("rm -rf ~")` and wreck real files — killing it only evicts the
  tenant *after* the damage.
- The danger is **syscalls**: `open()`, sockets, spawning processes all ask the
  kernel to act. A normal process has a direct line to the kernel.
- **WebAssembly (WASM)** is a virtual CPU that runs *inside* another program. It
  has **no instruction for a syscall**. Its whole world is (1) one block of linear
  memory the host gave it, and (2) an **import table** of functions the host chose
  to hand in. A module with no imports is a pure calculator — it can only affect
  the world through its return value.
- **Pyodide** = CPython compiled to WASM. Inside it, `import os; os.system(...)`
  is inert (no kernel behind it) and `open()` reads a *fake in-memory* filesystem.
  To do anything real, the host must import a specific function — that's exactly
  what our **bridges** are. **Deny-by-default.**
- **Deno** is a JS/TS runtime that is **secure-by-default** (no file/net/env access
  without `--allow-*`). fast-rlm runs the model's Python in Pyodide (WASM) *inside*
  Deno (permission-restricted) — two nested cages. Node could host Pyodide too, but
  Node isn't secure-by-default, so Deno's value is that extra permission layer.
- Caveat: WASM is only as tight as the host allows — Pyodide *can* be told to mount
  the real filesystem. Security lives in **which imports/mounts you grant.**

**Why it matters:** the real risk isn't a malicious model but a *buggy* generation
(e.g. `shutil.rmtree` on the wrong path). And it matters most at **Phase 2**: an
unattended, proactive loop writing+running code needs the WASM sandbox *before* we
turn on autonomy.

### OpenAI-compatible APIs are a shared shape
**Doubt:** could I just use my DeepSeek key instead of OpenRouter?

**What I learned:** DeepSeek, OpenRouter, OpenAI, and many local servers all speak
the **same** `POST /chat/completions` API (same JSON in/out, `Authorization: Bearer`).
So switching providers is just base URL + model + which env var holds the key — one
backend class (`OpenAICompatBackend`) with thin presets. DeepSeek base URL is
`https://api.deepseek.com`; models `deepseek-chat` (V3) and `deepseek-reasoner` (R1).

### API-key hygiene
**Context:** I pasted an API key directly into chat.

**What I learned:** a key pasted in chat is now in that conversation's history —
treat it as exposed and **rotate it**. Pass secrets as **environment variables**
(`export DEEPSEEK_API_KEY=…`), never in code, chat, or commits. (Confirmed the key
never entered the repo via `git grep`.)

### Why the live call was blocked here (egress policy)
**Context:** the live DeepSeek test wouldn't run in the cloud session.

**What I learned:** this remote environment routes all outbound HTTPS through a
policy-enforcing proxy. Hosts not on the allowlist get a **403 at the CONNECT
tunnel**. That's an org egress policy, not a bug — you don't route around it, you
run the call somewhere allowed (e.g. my own machine).

---

## Foundations from the design sessions

### RLM vs. Agent
- **RLM (Recursive Language Model)** = an *inference technique*: store the input as
  a variable in a REPL and let the model write code to explore it and recursively
  call itself on sub-pieces. Context becomes an *environment*, not tokens in a window.
- **Agent** = a *goal-seeking system with tools* over time (plan→act→observe→repeat).
- One line: an RLM reasons over unbounded context; an agent pursues goals. Our
  project marries them — an agent whose reasoning substrate *is* the RLM loop.

### Engine vs. harness — which is the "runtime"
- **Engine = the runtime.** It *executes* one recursive call: sandbox,
  prompt-as-variable, sub-agent spawning, budgets. Stateless, reusable.
- **Harness / brain** = the stateful stuff around it: corpus, goals, tools,
  proactivity. Prime Agent's "Continual Harness" is the reference.
- We are building **both** ourselves, not importing an engine.

### "First-class primitive"
- **Primitive** = a native, built-in operation of a system.
- **First-class** (from PL theory's "first-class citizen") = something you can
  store in a variable, pass around, return, and put in data structures — treat like
  any value.
- So `rlm(...)` being a *first-class agent primitive* means spawning a sub-agent is
  a built-in whose **result is an ordinary value** you can use in a loop, a list,
  etc. — `[rlm(q) for q in questions]`. That composability is what lets an agent
  invent its own decomposition at runtime.

### Alex Zhang's arc (a trilogy + a paper)
1. **RLM blog** (2025) — the empirical origin (GPT-5 in a Python REPL).
2. **RLM paper** (arXiv 2512.24601, with Kraska) — the formal paradigm.
3. **"Language Models will be Scaffolds"** (2026) — models and the systems around
   them merge; the interesting artifact is the *composition*.
4. **"Harnesses are compositional generalizers"** — a good harness lets a *fixed*
   model generalize to tasks it couldn't do in one shot. (The license for our brain.)

### Prime Agent's process nuances
- Tools/subagents are **function calls inside code** (in a persistent IPython
  kernel), not JSON steps outside the model's reasoning.
- **Continual Harness**: durable, refinable memory/skills/subagent specs; the
  immutable base prompt is never rewritten. `/refine` applies *small,
  evidence-backed, reversible* updates — self-improvement by accretion, not rewrite.

### What "understanding a paper" means (operationally)
- Not a prose summary — a **ladder of operations**: locate → summarize → structure
  → relate → critique → apply.
- The **atom is the claim**: a checkable statement + evidence pointer + strength.
- The corpus is **flat YAML notes**; cross-paper reasoning is code over the folder.
- Reactive and proactive are the **same machine at different corpus sizes**.

### Vision vs. Mission (the distinction itself)
- **Vision** = the world you're building toward (a research *partner* that
  accumulates understanding).
- **Mission** = what you actually build (a minimal RLM that reads papers into a
  structured corpus of claims, reactively then proactively).

---

## Python / implementation notes picked up while building the engine

- **`sys.stdout` vs `sys.__stdout__`.** `sys.stdout` is the *current* (possibly
  redirected) stream; `sys.__stdout__` is the *original* one. The worker captures
  user `print()` via `contextlib.redirect_stdout(buf)` (which swaps `sys.stdout`)
  while still writing its JSON protocol to `sys.__stdout__` — so the two streams
  never collide.
- **`contextlib.redirect_stdout`.** A context manager that temporarily points
  `sys.stdout` at a buffer — the clean way to capture what `exec`'d code prints.
- **A subprocess "worker" protocol.** Two processes talk over stdin/stdout with
  **line-delimited JSON** (one JSON object per line, flushed). Simple, language-
  agnostic RPC.
- **Bridge callbacks (host ⇄ sandbox).** When sandboxed code calls `llm()`, the
  worker *sends a request to the host and blocks* for the reply; the host does the
  real (networked) work and sends the value back. This is how you keep dangerous
  I/O out of the sandbox.
- **`select.select`** on a pipe with a timeout = wait for output but give up after
  N seconds → detect a hung cell without blocking forever.
- **`python -I`** = isolated mode (ignore env vars and user site-packages) — tidy,
  but *not* a security boundary.
- **Recursion via one line.** In the loop, the `rlm()` bridge just calls `run()`
  again at `depth+1` sharing one `Budget`. That single line is the entire
  "recursive" in Recursive Language Model.
- **Shared budget across a tree.** One `Budget` object threaded through every
  sub-call bounds the *whole* recursion (depth/calls/tokens), not each agent alone.
