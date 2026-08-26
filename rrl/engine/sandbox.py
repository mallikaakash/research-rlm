"""The host side of the sandbox — ingredients #3 (prompt-as-variable) and the
bridge plumbing for #2 (recursion).

A sandbox spawns a worker (a subprocess), seeds it with PROMPT, and runs code
cells in it. When the worker asks the host to run a bridge (llm/rlm/a tool), the
sandbox invokes the matching host callable and ships the result back — so the
model's code can `await llm(...)` / `await rlm(...)` inline while the actual work
happens out here.

`_ProtocolSandbox` holds the shared line-delimited JSON protocol; concrete
sandboxes only differ in *which* worker process they spawn:

  - `LocalSandbox`   : a Python subprocess (process isolation + kill/timeout).
  - `PyodideSandbox` : a Node process running CPython-in-WASM (real sandboxing).

Both satisfy the same tiny interface — `run_cell(code, bridges)` and `close()` —
so the engine is agnostic to which one it drives.
"""

from __future__ import annotations

import json
import select
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_WORKER = str(Path(__file__).with_name("worker.py"))


class SandboxTimeout(Exception):
    """The worker produced no message within the allotted time (likely a hang)."""


@dataclass
class CellResult:
    stdout: str
    final: object
    has_final: bool
    error: str | None


class _ProtocolSandbox:
    """Shared JSON protocol over a worker process's stdin/stdout."""

    def __init__(self, proc: subprocess.Popen, prompt, bridge_names, timeout: float | None):
        self.proc = proc
        self.timeout = timeout
        self._send({"op": "init", "prompt": prompt, "bridges": list(bridge_names)})
        ready = self._recv()
        if ready.get("op") != "ready":
            raise RuntimeError(f"sandbox failed to start: {ready!r}")

    # ---- protocol I/O ----
    def _send(self, obj) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(obj, default=str) + "\n")
        self.proc.stdin.flush()

    def _recv(self) -> dict:
        assert self.proc.stdout is not None
        if self.timeout is not None:
            ready, _, _ = select.select([self.proc.stdout], [], [], self.timeout)
            if not ready:
                self.proc.kill()
                raise SandboxTimeout(f"no response within {self.timeout}s")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("sandbox worker exited unexpectedly")
        return json.loads(line)

    # ---- the one operation the loop needs ----
    def run_cell(self, code: str, bridges: dict[str, Callable]) -> CellResult:
        """Run one code cell, servicing any bridge calls it makes, until it ends."""
        self._send({"op": "exec", "code": code})
        while True:
            msg = self._recv()
            op = msg.get("op")
            if op == "bridge":
                name = msg.get("name", "")
                call_id = msg.get("_id")  # echoed back so async workers can route replies
                try:
                    fn = bridges[name]
                    value = fn(*msg.get("args", []), **msg.get("kwargs", {}))
                    self._send({"op": "bridge_result", "ok": True, "value": value, "_id": call_id})
                except Exception as e:  # noqa: BLE001 — surface as an exception inside the cell
                    self._send(
                        {"op": "bridge_result", "ok": False, "error": f"{type(e).__name__}: {e}", "_id": call_id}
                    )
            elif op == "result":
                return CellResult(
                    stdout=msg.get("stdout", ""),
                    final=msg.get("final"),
                    has_final=msg.get("has_final", False),
                    error=msg.get("error"),
                )
            else:
                raise RuntimeError(f"unexpected message from sandbox: {msg!r}")

    def close(self) -> None:
        try:
            self._send({"op": "shutdown"})
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self.proc.kill()


class LocalSandbox(_ProtocolSandbox):
    """A persistent Python worker subprocess. State survives across cells.

    Isolation + kill/timeout, NOT security isolation — code here can still touch
    the host. Fast and dependency-free; the default for tests and trusted input.
    """

    def __init__(self, prompt, bridge_names, timeout: float | None = 120.0):
        proc = subprocess.Popen(
            [sys.executable, "-I", _WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        super().__init__(proc, prompt, bridge_names, timeout)
