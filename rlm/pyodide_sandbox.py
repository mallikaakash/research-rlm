"""PyodideSandbox — the real (WASM) sandbox: CPython compiled to WebAssembly,
hosted in a Node subprocess.

Same interface as LocalSandbox (`run_cell` / `close`), same JSON protocol — the
only difference is the worker is `node pyodide_worker.mjs` instead of a Python
subprocess. Inside it, the model's Python has no syscalls: it cannot reach the
real filesystem or network. The sole way out is a bridge (llm/rlm), forwarded to
the host.

Requirements: Node.js and the `pyodide` npm package resolvable from the repo
(`npm install pyodide`). Each sandbox spawns a Node process and loads Pyodide
(~a few seconds), so recursion spawns several — fine for correctness; a pool is a
later optimization.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .sandbox import _ProtocolSandbox

_WORKER = str(Path(__file__).with_name("pyodide_worker.mjs"))
# Repo root (…/rlm/pyodide_sandbox.py -> repo root), where node_modules lives.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])


class PyodideSandbox(_ProtocolSandbox):
    def __init__(self, prompt, bridge_names, timeout: float | None = 180.0, node: str = "node"):
        if shutil.which(node) is None:
            raise RuntimeError(f"{node!r} not found on PATH — Node.js is required for PyodideSandbox.")
        proc = subprocess.Popen(
            [node, _WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=_REPO_ROOT,  # so Node resolves the `pyodide` package from repo node_modules
        )
        # Longer default timeout: the first message ("ready") waits for Pyodide to load.
        super().__init__(proc, prompt, bridge_names, timeout)


def pyodide_available(node: str = "node") -> bool:
    """True if Node and the pyodide package are usable — for tests to skip gracefully."""
    if shutil.which(node) is None:
        return False
    try:
        r = subprocess.run(
            [node, "-e", "require.resolve('pyodide')"],
            cwd=_REPO_ROOT,
            capture_output=True,
            timeout=30,
        )
        return r.returncode == 0
    except Exception:
        return False
