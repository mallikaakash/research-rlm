# LEARNINGS

A running log of things learned while building this project — doubts I had and
what resolved them, Python/CS concepts that were new, and things worth knowing
that came up along the way. Newest entries on top.

Format per entry: **the doubt / context** → **what I learned** → (sometimes) **why it matters**.

---

## 2026-08-30

### The paper-explainer harness — how thin a harness really is
The first real harness (`paper/`) turns the engine into "give it an arXiv link, get
a thorough breakdown." The lesson: **the harness is thin because the engine is the
loop.** No new loop — the paper's LaTeX *becomes* PROMPT, and `run(instruction=…)`
already knows how to chunk/delegate/combine. The harness only adds: (1) `fetch.py`
(arXiv id/URL → LaTeX source, stdlib only — source over PDF for exact equations and
numbers), (2) an `EXPLAIN_PROMPT` instruction, (3) a CLI. Plus one reliability fix:
a big paper can exhaust `max_steps` without ever calling `FINAL()` → `None`, so
`explain()` has a deterministic **map-reduce fallback** (summarize chunks → synthesize)
that always terminates. → *A harness = fetch + instruction + a safety net; the reasoning is the engine's.*

### Making the REPL visible — the `inspect` op
Added an opt-in `inspect_vars` path: after each cell the host sends `{"op":"inspect"}`
and the worker replies with a summary of its namespace `ns` (name/type/repr/size,
skipping dunders, callables, and imported modules). This is what lets a UI show the
REPL populating (`PROMPT` → `chunks` → `notes` → `report`). Key realization from the
"where does the variable live" question: **`ns` is just a dict in the worker process;
the host is blind to it and only learns via printed stdout / FINAL / this new op.**
Implemented on both sandboxes (Python worker + the WASM `.mjs`), best-effort with its
own short timeout so it can never hang a run.

### The concurrency gotcha — blocking engine + async UI (the big one)
Building the Textual dashboard forced the canonical lesson. `run()` is **synchronous
and blocking** (it parks on pipe/socket reads). Textual runs a **single-threaded
asyncio event loop**. Calling `run()` on that loop's thread freezes the UI, because
the one thread that services keys/redraws is parked in `readline()`. Fix = the
standard pattern:
- **Blocking work on a background thread** (`app.run_worker(fn, thread=True)`). Allowed
  because a thread blocked on I/O **releases the GIL**, so the UI thread keeps running
  Python. (Threads help here precisely because `run()` is I/O-bound, not CPU-bound.)
- **Only the UI thread may touch widgets** (the single-threaded-UI invariant — same in
  Swing/Qt/iOS/JS). The worker never draws; it **marshals** each event to the UI thread
  via the thread-safe `call_from_thread(...)`.
- **Events cross through a queue** (the event loop's own message queue), preserving order.
The engine's existing `on_event` callback *is* the seam: `on_event = lambda e:
call_from_thread(apply, e)`. → *This "blocking work on a thread, all UI on the main
loop, events over a queue" pattern wraps ANY blocking library in ANY async UI.*
Also learned to keep the event→view logic framework-free (`tui/state.py`) so it's
unit-testable with plain asserts, and to test the real app headlessly with Textual's
`run_test()` pilot (no TTY needed).

### Textual = Rich's big sibling
The "better than Rich, React-inspired" TUI framework is **Textual** (same team as Rich).
Widgets (≈components), CSS (`.tcss`) styling, reactive state, a message/event model —
it *is* React-for-the-terminal. Bonus: `textual serve` (textual-web) runs the exact
same app in a browser over a websocket, zero extra code — the "maybe in the browser too"
is a flag, not a second build.

### Python idioms nailed down (closures, `*args`/`**kwargs`, the two moments)
- **`*args`/`**kwargs`** = collectors (one star → tuple of positionals; two stars → dict
  of keywords). The magic is the stars, not the names. In a *call*, the same stars
  **spread** a collection back into arguments (`fn(*args, **kwargs)`). `tools[name](**args)`
  *is* the whole dispatch mechanism of a tool-calling agent.
- **Closure / factory**: `_wrap_tool(name, fn)` runs **once at setup** to *build* a
  `call` function that **remembers** `name`/`fn`; `call(*args, **kwargs)` runs **later,
  many times**, when the model invokes the tool. Two moments in time — that gap is why
  closures confuse people. The outer function exists to bind `name`/`fn` per tool and
  stamp out one dedicated wrapper each (a class with `__call__` or `functools.partial`
  would do the same job).
- **Dict-as-dispatch-table** (`bridges[name](...)`), `@dataclass`, `try/finally` for
  guaranteed cleanup, custom exceptions for deep-to-top control flow (`BudgetExceeded`,
  `_Final`), `{**a, **b}` merge with right-wins — the recurring vocabulary of agent code.

## 2026-08-26

### How is an RLM actually different from a normal agentic loop (Cursor)?
**Doubt (kept circling back):** the trace looks exactly like any agent — think →
tool call → observe → think again. So how is RLM different from Cursor/Claude Code?

**What I learned — the honest answer:**
- **At the loop level they are identical.** Think→act→observe→repeat is the *ReAct
  loop*, shared by RLM, Cursor, and every tool-calling agent. RLM is NOT a different
  loop. The difference lives in three places, not the loop:
  1. **What the tools act on.** A normal agent's tools act *outward* on the world
     (`edit_file`, `run_terminal`) to change it. An RLM's main tool acts *inward*:
     it runs code over the **input held as a variable, kept out of the context
     window**, to *comprehend* something too big to fit.
  2. **The model calls itself as a composable function.** `answers = [rlm(s) for s
     in sections]` — self-invocation, combined in code. External tools (grep, shell)
     aren't that. This is the "R" in RLM.
  3. **Sub-results return as VALUES, not context.** A normal agent appends every
     observation to the conversation (context grows with all it has seen). An RLM
     gets a sub-call's answer back as a *REPL variable* it may not even print — so
     the parent's context stays tiny no matter how much sub-work happened. This is
     the mechanism behind "unbounded context".
- **At depth 0 with no `rlm()` calls, an RLM literally *is* a ReAct code agent.**
  The RLM-specific behavior is invisible until the input is big enough to force
  recursion — which is why our early runs (depth=0) looked like any agent.
- **The boundary is genuinely blurry** — Zhang's own "scaffolds" thesis argues
  agents/scaffolds/models are converging. So "these look the same" is a *correct*
  observation, not a misunderstanding. RLM is a *species* of agent (tool = code
  REPL; recursion = self-calls; target = its own oversized input), not a new genus.

One-liner: **a normal agent uses tools to act on the world; an RLM uses code — and
copies of itself — to reason over an input too big for its context. Same loop;
different target, plus self-recursion.**

### Async vs. parallel — and how the reference RLMs compare
**Doubt:** is fast-rlm / the original RLM synchronous or async, sequential or parallel?
- **Async ≠ parallel.** *Async* is the mechanism (calls are `await`ed, non-blocking).
  *Parallel* is actually running several at once (you must explicitly gather/batch).
- **Zhang's RLM** (`alexzhang13/rlm`): **async AND parallel** — `rlm_query_batched` /
  `llm_query_batched`, bounded by **`max_concurrent_subcalls`**.
- **fast-rlm**: **async** (`await llm_query(...)`); parallel **not documented**.
- **Ours (research-rlm)**: **async in contract** (`await llm/rlm`) but **sequential in
  execution** — bridges service one call at a time. Parallel fan-out is the gap; the
  pattern to copy is Zhang's `max_concurrent_subcalls` (a host-side concurrency pool +
  `asyncio.gather` support + a thread-safe budget).

### Would DSPy be useful here?
**Doubt:** should we use DSPy in this project?

**What I learned:** DSPy (by Omar Khattab — also on the RLM paper) is a framework for
*programming* LLMs: typed **signatures** (`input -> typed output`), composable
**modules** (`Predict`, `ChainOfThought`, `ReAct`), and **optimizers** that auto-tune
prompts/few-shot examples against a metric.
- **Wrong tool for the engine.** DSPy is a *different paradigm* from RLM (declarative
  LLM pipelines vs. a model writing code in a REPL). Putting it in the engine would
  contradict "we build our own RLM engine."
- **Right tool for extraction, later.** "Paper section → structured claims+evidence"
  is textbook DSPy (typed output + parsing; optimizers could improve quality given a
  small gold set). It would live in the **brain layer** (a smarter `llm()` / the note
  assembler), not the engine. Clean split: **RLM explores; DSPy extracts.**
- **Verdict: defer.** Heavy dep + its optimizers need eval data we don't have. Start
  with a plain prompt + a Pydantic-style schema; revisit when extraction quality is
  the bottleneck and we want auto-optimization.

### Why deep recursion is slow in the WASM sandbox (and what a pool fixes)
**Doubt:** what does "each Pyodide sandbox loads its own WASM, so recursion is slow"
mean?
- `loadPyodide()` boots the whole CPython-in-WASM interpreter (tens of MB) — a
  **cold start** of ~seconds. Every agent (root + each `rlm()` sub-agent) spawns a
  *fresh* sandbox for isolation, so an N-node recursion tree pays N cold starts.
- The subprocess (`local`) sandbox spawns per agent too, but Python startup is ~50 ms
  — negligible. The cost only bites for WASM.
- **A pool** keeps a few Node+Pyodide processes *warm*; borrow one, reset its
  namespace to clean (cheap), use it, return it — so the expensive load happens a few
  times total, not once per agent. It's **pure optimization** (correctness identical),
  hence a follow-up: make it correct first, pool when runs feel slow.

### Building the Pyodide (WASM) sandbox — what it actually took
**Context:** we added `PyodideSandbox` (CPython-in-WASM, hosted in Node) as a real
sandbox behind the same interface as the subprocess one.

**What I learned:**
- **WASM bridges must be `await`ed.** The host round-trip (sandbox → host → model
  API → back) is asynchronous, and Pyodide can't block on it synchronously without
  SharedArrayBuffer tricks. So the model calls `await llm(...)` / `await rlm(...)`.
  We unified *both* sandboxes on this async contract (fast-rlm uses the same).
- **Top-level await needs a compile flag.** To run a cell that uses `await`, compile
  with `ast.PyCF_ALLOW_TOP_LEVEL_AWAIT`; `eval(code_obj, ns)` then returns a
  *coroutine* you `await` (or `asyncio.run`). Variables still persist to `ns`.
- **The host must be push-based, not pull-based.** While an `exec` is mid-flight and
  the model is `await`-ing a bridge, the Node stdin reader must keep dispatching so
  `bridge_result` messages can resolve the pending promise. If the reader only ran
  between execs, it would deadlock. Bridge calls carry an `_id` so replies route to
  the right pending promise (needed once calls can overlap).
- **Keep Pyodide's stdio off the protocol channel.** Pyodide's default stdout goes to
  `console.log` → would corrupt our JSON on stdout. Silence it (`setStdout`) and
  capture the model's `print()` inside Python via `redirect_stdout`.
- **Recursion = nested WASM sandboxes.** `await rlm(...)` on the host spawns a *fresh*
  Node+Pyodide process (~seconds each). Correct, but a sandbox pool is a future
  optimization.
- Verified end-to-end offline: model code reports `sys.platform == 'emscripten'`
  (it's really in WASM), `await rlm()` recurses through the host, `FINAL()` returns.

### What sandbox does Zhang actually use? (not WASM!)
**Doubt:** is our engine's sandbox true to Zhang's original?

**What I learned (from `alexzhang13/rlm`):** Zhang's library is **pluggable** across
`local, ipython, docker, modal, prime, daytona, e2b`. Crucially:
- **Default = a local Python REPL via `exec` on the host** — i.e. *not sandboxed at
  all* by default (even less isolated than ours, which is a subprocess).
- `ipython` mode runs cells in a real IPython session, **in-process or in a separate
  `ipykernel` subprocess** — that subprocess mode is essentially *what we built*.
- Real isolation comes from **Docker** (container) or **cloud microVM sandboxes**
  (Modal / Prime Intellect / Daytona / E2B).
- **Zhang does NOT use Pyodide/WASM.** WASM is purely **fast-rlm's (neuralavb's)**
  choice. So the sandbox lineage is: Zhang = local/ipython/docker/cloud;
  fast-rlm = Deno+Pyodide.

**Why it matters:** our subprocess sandbox is close to Zhang's `ipykernel`-subprocess
mode, so we're already faithful to *his* isolation story. "Adding sandboxing" has two
faithful directions: **Docker/cloud (Zhang's way)** or **Pyodide/WASM (fast-rlm's way)**
— they are different philosophies (isolate the *machine* vs. remove the *syscalls*).

### How faithful is our engine to Zhang / fast-rlm?
- **Faithful to the core paradigm (both):** prompt-as-a-variable, code-driven
  exploration, `rlm()` recursion at depth+1, **sub-results returned as values in the
  REPL (not dumped into context)**, `FINAL()` termination, budgets, model-agnostic.
- **Simpler than Zhang:** we don't have his pluggable cloud/Docker sandboxes; we
  haven't stress-tested 100×-context scale.
- **Simpler than fast-rlm:** no Pyodide/WASM sandbox, no typed `output_schema`
  validation, no structured `{prompt, links, files}` input, no cost budgeting /
  caching / TUI log viewer.
- **Verdict:** a faithful *minimal* reimplementation of the paradigm; the two honest
  gaps are a real sandbox and structured I/O — both already on the roadmap.

### How we currently view a run (observability)
Right now the *only* window into a run is the engine's `on_event` callback rendered
by the CLI's `-v` flag as a **depth-indented text trace on stderr**: `· agent[step]`
(model turns), `└ cell[ok|ERR|FINAL]` (what the code printed), `↳ rlm(...)` /
`↳ llm(...)` (recursion). The final answer prints to stdout; a one-line budget summary
(`agents / depth / calls / tokens`) prints to stderr. It's ephemeral and unstructured
— no saved JSON trace, no tree view, no TUI (fast-rlm has a proper log viewer; we
don't yet). "Retrieval" is visible only as the code's slicing of PROMPT in the cell
output. **Gap: a structured, persisted trace is worth adding.**

### Why Deno AND Pyodide — host vs. guest, and the sandbox landscape
**Doubt:** if Pyodide (WASM) is already the secure sandbox, what does Deno even add?

**What I learned:**
- **WASM can't run by itself** — it's bytecode for a virtual CPU that only executes
  *inside a host program*. So it's not two stacked sandboxes; it's **host + guest.**
  Pyodide is the guest; **Deno is the host.**
- The host's jobs: (1) run the WASM engine, (2) provide the imported **bridges**,
  (3) run the *trusted* orchestration (the loop, model API calls, corpus disk I/O),
  (4) do the real I/O the guest requests. Security (no syscalls) comes from WASM;
  the host is the trusted warden outside the box.
- **Trust boundary:** the *model's* code is untrusted → WASM. *Our* orchestration is
  trusted → runs in the host directly (it needs real network/disk to work).
- **What Deno adds over Node:** both can host Pyodide, but Deno is
  **secure-by-default** (`--allow-net`, `--allow-read`, …), so even the trusted host
  sits behind an outer OS-permission wall → **defense in depth** (WASM = inner cage,
  Deno perms = outer cage). Node = as privileged as our current Python.
- **Three slots to choose independently:**
  - *Host/embedder (the "Deno" role):* Browser, Node, **Deno**, Bun, or standalone
    WASM runtimes (`wasmtime`, `wasmer`, `WasmEdge`, `wazero`). `wasmtime-py` lets
    **Python itself** be the host — no JS needed.
  - *Sandboxed Python (the "Pyodide" role):* **Pyodide** (rich, JS-hosted), CPython
    **WASI** build, RustPython→WASM, MicroPython.
  - *Non-WASM sandboxing:* subprocess (ours) < seccomp < namespaces (`bubblewrap`,
    `nsjail`, `firejail`) / Landlock < containers (Docker) < **gVisor** < **microVMs**
    (Firecracker — how Lambda works) < full VMs. Plus **V8 isolates** (Cloudflare
    Workers) and hosted code-exec services (**E2B**, Modal, Riza) that run untrusted
    code for you.
- **Trade-off axes:** security, startup/overhead, *compatibility* (WASM can't run
  arbitrary C extensions / `pip install`; containers & VMs can), and ops complexity
  (WASM is just a library; VMs need infra).

**Why it matters / for us:** our model code mostly does string/regex/slice work over
a paper + bridge calls — no numpy needed — which is Pyodide's sweet spot. Since the
project is Python-first, we don't have to adopt Deno: **`wasmtime-py` + a WASI
CPython** keeps it one language (at the cost of Pyodide's package richness). A
microVM/E2B-style sandbox only earns its keep if we later need full compatibility.

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

---

# Reading list & curriculum — mastering this domain

A tiered path to get up to speed on the ideas behind this project: RLMs, agent
foundations, memory, recursive self-improvement, harnesses/scaffolds, and proactive
agents. Suggested order: **Tier 0 → 1 → 2 → 3 → 4/5 → 6**, dipping into Tier 7 for
breadth and to stay current. Each item notes *why* to read it.

Maps to our build: **Tier 0–2** = the engine + read pipeline (now); **Tier 3–5** =
the corpus/brain; **Tier 4 & 6** = the proactive Phase 2.

### Tier 0 — The RLM core (our foundation — start here)
- **Recursive Language Models** — Zhang, [blog](https://alexzhang13.github.io/blog/2025/rlm/) · [paper (arXiv 2512.24601)](https://arxiv.org/abs/2512.24601) · [code: alexzhang13/rlm](https://github.com/alexzhang13/rlm). The idea our whole engine implements: context as an environment, prompt-as-variable, recursion.
- **fast-rlm** — [avbiswas/fast-rlm](https://github.com/avbiswas/fast-rlm) · [docs](https://avbiswas.github.io/fast-rlm/). The feature-rich reference implementation (Deno+Pyodide, structured IO, budgets).
- **Prime Agent** — [PrimeIntellect-ai/prime-agent](https://github.com/PrimeIntellect-ai/prime-agent) · [blog](https://www.primeintellect.ai/blog/prime-agent). The closest prior art for our proactive brain (Continual Harness, `/refine`, daemon sessions).
- **Zhang's harness trilogy** — [Language Models will be Scaffolds](https://alexzhang13.github.io/blog/2026/scaffold/) · [Harnesses are compositional generalizers](https://alexzhang13.github.io/blog/2026/harness/). Why building the *scaffold* (not just the model) is the point.

### Tier 1 — Foundations of LLM agents (the canon)
- **ReAct** — [arXiv 2210.03629](https://arxiv.org/abs/2210.03629). Reason+act loop; the baseline our RLM improves on.
- **Reflexion** — [arXiv 2303.11366](https://arxiv.org/abs/2303.11366). Learning from failure via verbal self-feedback.
- **Tree of Thoughts** — [arXiv 2305.10601](https://arxiv.org/abs/2305.10601). Search over reasoning paths.
- **Toolformer** — [arXiv 2302.04555](https://arxiv.org/abs/2302.04555). Models learning to call tools.
- **Voyager** — [arXiv 2305.16291](https://arxiv.org/abs/2305.16291). A **skill library** that grows — proto-self-improvement; directly relevant to our corpus-as-brain.
- **Generative Agents** — [arXiv 2304.03442](https://arxiv.org/abs/2304.03442). Memory + reflection sustaining long-term behavior.

### Tier 2 — Context as environment: long context, RAG, context engineering
- **RAG: A Survey** — [arXiv 2312.10997](https://arxiv.org/abs/2312.10997). The retrieval paradigm RLMs are an alternative to.
- **MemGPT** — [arXiv 2310.08560](https://arxiv.org/abs/2310.08560). Context as virtual memory (OS analogy) — a cousin of prompt-as-variable.
- **From RAG to Context (2025 review)** — [RAGFlow](https://ragflow.io/blog/rag-review-2025-from-rag-to-context). Why "memory" and "context engineering" (e.g. the ACE framework) eclipsed plain RAG in 2025.

### Tier 3 — Agent memory & continual learning (our corpus/brain)
- **Letta / MemGPT runtime** — [github.com/letta-ai/letta](https://github.com/letta-ai/letta). Memory tiers as a full agent runtime.
- **Mem0** — [github.com/mem0ai/mem0](https://github.com/mem0ai/mem0). A bolt-on memory layer (vector+graph+kv).
- **Survey: From Storage to Experience** — [arXiv 2605.06716](https://arxiv.org/abs/2605.06716). Evolution of agent memory mechanisms.
- **Survey: Memory in the Age of AI Agents** — [arXiv 2512.13564](https://arxiv.org/abs/2512.13564).
- **Review: Externalization in LLM Agents** — [arXiv 2604.08224](https://arxiv.org/abs/2604.08224). Memory + skills + protocols + **harness engineering** in one frame — very on-topic.

### Tier 4 — Recursively self-improving agents (the Phase-2 heart)
- **STOP: Self-Taught Optimizer** — [arXiv 2310.02304](https://arxiv.org/abs/2310.02304). A seed improver applied to its own code.
- **ADAS: Automated Design of Agentic Systems** — [arXiv 2408.08435](https://arxiv.org/abs/2408.08435). A meta-agent that invents better agents.
- **Gödel Agent** — [arXiv 2410.04444](https://arxiv.org/abs/2410.04444). Self-referential policy updates.
- **Darwin Gödel Machine** — [arXiv 2505.22954](https://arxiv.org/abs/2505.22954) (Sakana). Evolves its own code; 20%→50% on SWE-bench.
- **SEAL: Self-Adapting Language Models** — [arXiv 2506.10943](https://arxiv.org/abs/2506.10943) (MIT). A model that writes its own finetuning data.
- **AlphaEvolve** — [arXiv 2506.13131](https://arxiv.org/abs/2506.13131) (DeepMind). Evolutionary coding agent for algorithm discovery.
- **Self-Improving Coding Agent (SICA)** — [overview](https://www.emergentmind.com/topics/self-improving-coding-agent-sica). An agent that edits its own scaffold.
- Curated: [awesome-Self-Improving-Agents](https://github.com/selfimproving-agent/awesome-Self-Improving-Agents).

### Tier 5 — Scaffolds, harnesses & prompt/agent optimization
- **DSPy** — [github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy). Programming (not prompting) LLMs; candidate for our note-extraction later.
- **GEPA** — [gepa-ai.github.io/gepa](https://gepa-ai.github.io/gepa/). Reflective prompt evolution; beats RL with far fewer rollouts.
- (Zhang's scaffold/harness posts in Tier 0 are the conceptual anchor here.)

### Tier 6 — Proactive agents (our Phase-2 target)
- **Anticipate and Learn: Idle-Time Compute in Proactive Agents** — [arXiv 2605.25971](https://arxiv.org/abs/2605.25971). Using idle compute to anticipate needs — directly our proactive loop.
- **ProActor** — [arXiv 2605.24900](https://arxiv.org/abs/2605.24900). Timing-aware RL for proactive task scheduling.
- **When Should an AI Act?** — [arXiv 2602.22814](https://arxiv.org/abs/2602.22814). A human-centered model of *when* proactivity is welcome (matters for autonomy bounds).
- **PROPER Agents** — [arXiv 2601.09926](https://arxiv.org/abs/2601.09926). Proactivity for knowledge-gap navigation.

### Tier 7 — Curated lists & staying current
- [WooooDyy/LLM-Agent-Paper-List](https://github.com/WooooDyy/LLM-Agent-Paper-List) — the big agent survey's paper list.
- [luo-junyu/awesome-agent-papers](https://github.com/luo-junyu/awesome-agent-papers) — methodology/applications/challenges.
- [yxf203/Awesome-Efficient-Agents](https://github.com/yxf203/Awesome-Efficient-Agents) — memory/tools/planning efficiency.
- [Alex Zhang on alphaXiv](https://www.alphaxiv.org/@alex-l-zhang) — the RLM author's feed.

_Note on arXiv ids: a few 2026 ids (26xx) are recent and may shift; the paper **titles** are the reliable anchor if a link 404s._
