// Pyodide sandbox worker — the WASM guest side.
//
// Runs CPython compiled to WebAssembly (Pyodide) inside Node, and speaks the same
// line-delimited JSON protocol as the Python worker (worker.py):
//
//   host -> worker : {"op":"init","prompt":...,"bridges":[...]}
//                    {"op":"exec","code":"..."}
//                    {"op":"bridge_result","ok":true,"value":...,"_id":N}
//                    {"op":"shutdown"}
//   worker -> host : {"op":"ready"}
//                    {"op":"bridge","name":"llm","args":[x],"kwargs":{},"_id":N}
//                    {"op":"result","stdout":...,"final":...,"has_final":...,"error":...}
//
// Why WASM matters: the model's Python runs with NO syscalls — it cannot touch the
// real disk or network. The only way out is an imported bridge (llm/rlm), which we
// forward to the host. Bridges are async (the host round-trip is async), so the
// model must `await llm(...)` / `await rlm(...)`.

import { loadPyodide } from "pyodide";
import process from "node:process";
import readline from "node:readline";

function send(obj) {
  process.stdout.write(JSON.stringify(obj) + "\n");
}

// Python-side setup: PROMPT + bridges are injected as globals by init; here we add
// FINAL and the cell runner that captures stdout and catches the FINAL sentinel.
const SETUP_PY = `
import ast, io, json, contextlib, traceback

class _Final(Exception):
    def __init__(self, value=None):
        self.value = value

def FINAL(value=None):
    raise _Final(value)

async def _exec_cell(src):
    buf = io.StringIO()
    final, has_final, error = None, False, None
    try:
        code = compile(src, "<cell>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        with contextlib.redirect_stdout(buf):
            coro = eval(code, globals())      # coroutine iff the cell used await
            if coro is not None:
                await coro
    except _Final as f:
        has_final, final = True, f.value
    except BaseException:
        error = traceback.format_exc()
    return json.dumps(
        {"stdout": buf.getvalue(), "final": final, "has_final": has_final, "error": error},
        default=str,
    )
`;

async function main() {
  const py = await loadPyodide();
  // Silence Pyodide's own stdio so nothing but our protocol reaches process.stdout.
  py.setStdout({ batched: () => {} });
  py.setStderr({ batched: () => {} });

  const pending = new Map();
  let nextId = 1;

  // A bridge the model calls as `await llm(x)` / `await rlm(x)`: forward to the host
  // and return a promise resolved when the matching bridge_result arrives.
  function bridge(name, arg) {
    const id = nextId++;
    send({ op: "bridge", name, args: [arg], kwargs: {}, _id: id });
    return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
  }

  let started = false;

  async function handle(msg) {
    const op = msg.op;
    if (op === "bridge_result") {
      const p = pending.get(msg._id);
      if (p) {
        pending.delete(msg._id);
        msg.ok ? p.resolve(msg.value) : p.reject(new Error(msg.error || "bridge failed"));
      }
      return;
    }
    if (op === "init") {
      py.globals.set("PROMPT", msg.prompt ?? "");
      for (const name of msg.bridges || []) {
        py.globals.set(name, (arg) => bridge(name, arg));
      }
      await py.runPythonAsync(SETUP_PY);
      send({ op: "ready" });
      started = true;
      return;
    }
    if (op === "exec") {
      py.globals.set("_CELL_SRC", msg.code ?? "");
      const resultJson = await py.runPythonAsync("await _exec_cell(_CELL_SRC)");
      send({ op: "result", ...JSON.parse(resultJson) });
      return;
    }
    if (op === "shutdown") {
      process.exit(0);
    }
  }

  // Push-based stdin loop: bridge_result messages resolve pending promises even
  // while an exec is mid-flight (that's what lets `await llm(...)` work).
  const rl = readline.createInterface({ input: process.stdin });
  let chain = Promise.resolve();
  rl.on("line", (line) => {
    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      return;
    }
    if (msg.op === "bridge_result") {
      handle(msg); // resolve immediately; never queued behind an exec
    } else {
      chain = chain.then(() => handle(msg)); // serialize init/exec/shutdown
    }
  });
}

main().catch((e) => {
  process.stderr.write("pyodide worker fatal: " + (e && e.stack ? e.stack : e) + "\n");
  process.exit(1);
});
