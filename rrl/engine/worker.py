"""The sandbox worker — the process the model's code actually runs inside.

It holds PROMPT and a persistent namespace, executes one code cell at a time, and
talks to the host over a tiny line-delimited JSON protocol on stdin/stdout:

  host -> worker :  {"op": "init", "prompt": ..., "bridges": [...]}
                    {"op": "exec", "code": "..."}
                    {"op": "bridge_result", "ok": true, "value": ...}
                    {"op": "shutdown"}

  worker -> host :  {"op": "ready"}
                    {"op": "bridge", "name": "llm", "args": [...], "kwargs": {...}}
                    {"op": "result", "stdout": ..., "final": ..., "has_final": ..., "error": ...}

Bridge calls (llm/rlm/tools) are the key move: when the model's code calls a
bridge, the worker asks the *host* to run it (the host has the network/disk) and
blocks for the result. Dangerous I/O never happens in here.

NOTE (v1): this is a normal Python subprocess. It gives process isolation plus
kill/timeout — NOT security isolation. Code in here can still touch the real
machine. A Pyodide/WASM sandbox is the planned drop-in replacement.
"""

import ast
import asyncio
import io
import json
import sys
import traceback
from contextlib import redirect_stdout

# Protocol travels on the *real* stdio; user code's stdout is captured separately.
_OUT = sys.__stdout__
_IN = sys.__stdin__


def _send(obj) -> None:
    _OUT.write(json.dumps(obj, default=str) + "\n")
    _OUT.flush()


def _recv() -> dict:
    line = _IN.readline()
    if not line:
        raise EOFError("host closed the connection")
    return json.loads(line)


class _Final(Exception):
    """Raised by FINAL() to terminate a cell and carry the result out."""

    def __init__(self, value):
        self.value = value


def _make_bridge(name: str):
    """Build an async proxy that forwards a call to the host and blocks for the answer.

    Async so the model calls it as `await llm(...)` / `await rlm(...)` — the same
    contract the Pyodide sandbox needs (where the host round-trip is genuinely async).
    Here the round-trip is a blocking stdin read, which is fine: only one coroutine
    runs at a time (v1 doesn't support concurrent bridge calls in this sandbox).
    """

    async def proxy(*args, **kwargs):
        _send({"op": "bridge", "name": name, "args": list(args), "kwargs": kwargs})
        resp = _recv()
        while resp.get("op") != "bridge_result":
            resp = _recv()
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", f"{name}() failed on the host"))
        return resp.get("value")

    return proxy


def main() -> None:
    ns: dict = {"__name__": "__rlm_sandbox__"}

    init = _recv()
    if init.get("op") != "init":
        raise RuntimeError(f"expected init, got {init!r}")
    ns["PROMPT"] = init.get("prompt", "")
    for name in init.get("bridges", []):
        ns[name] = _make_bridge(name)

    def FINAL(answer=None):
        raise _Final(answer)

    ns["FINAL"] = FINAL
    _send({"op": "ready"})

    while True:
        cmd = _recv()
        op = cmd.get("op")
        if op == "shutdown":
            return
        if op != "exec":
            continue

        buf = io.StringIO()
        final, has_final, error = None, False, None
        try:
            # Compile with top-level-await support so cells can `await llm(...)`.
            code_obj = compile(
                cmd.get("code", ""), "<cell>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
            )
            with redirect_stdout(buf):
                maybe_coro = eval(code_obj, ns)  # returns a coroutine iff the cell used await
                if maybe_coro is not None:
                    asyncio.run(maybe_coro)
        except _Final as f:
            has_final, final = True, f.value
        except BaseException:  # noqa: BLE001 — report everything back, never die on user code
            error = traceback.format_exc()

        _send(
            {
                "op": "result",
                "stdout": buf.getvalue(),
                "final": final,
                "has_final": has_final,
                "error": error,
            }
        )


if __name__ == "__main__":
    main()
