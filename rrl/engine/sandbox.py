"""The host side of the sandbox — ingredients #3 (prompt-as-variable) and the
bridge plumbing for #2 (recursion).

`LocalSandbox` spawns one worker subprocess, seeds it with PROMPT, and runs code
cells in it. When the worker asks the host to run a bridge (llm/rlm/a tool), the
sandbox invokes the matching host callable and ships the result back — so the
model's code can call llm()/rlm() inline while the actual work happens out here.

The `Sandbox` name is an interface by convention: anything with the same
`run_cell` / `close` shape can replace LocalSandbox (a Pyodide/WASM sandbox is
the intended next implementation).
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


class LocalSandbox:
    """A persistent Python worker subprocess. State survives across cells."""

    def __init__(self, prompt, bridge_names, timeout: float | None = 120.0):
        self.timeout = timeout
        self.proc = subprocess.Popen(
            [sys.executable, "-I", _WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
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
                try:
                    fn = bridges[name]
                    value = fn(*msg.get("args", []), **msg.get("kwargs", {}))
                    self._send({"op": "bridge_result", "ok": True, "value": value})
                except Exception as e:  # noqa: BLE001 — surface as an exception inside the cell
                    self._send(
                        {"op": "bridge_result", "ok": False, "error": f"{type(e).__name__}: {e}"}
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
