"""Retrieval — fetch a paper's text from an arXiv id/URL.

This is a harness tool (the host has network; the sandbox never does). It prefers
the LaTeX **e-print source** (best fidelity for claim extraction — real equations,
tables, numbers), and falls back to the abstract via the arXiv API if the source
can't be retrieved.

Note: arXiv egress is blocked in the cloud build environment, so the network path
is verified on a real machine, not in CI. The parsing helpers below are unit-tested
offline (synthetic tar / Atom).
"""

from __future__ import annotations

import gzip
import io
import re
import tarfile
from xml.etree import ElementTree as ET

_ID = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
_UA = {"User-Agent": "research-rlm/0.1 (+https://github.com/mallikaakash/research-rlm)"}


def parse_arxiv_id(s: str) -> str | None:
    """Extract a bare arXiv id from an id / abs URL / pdf URL, else None.

    Conservative: a plain filename with digits is NOT treated as an arXiv id.
    """
    s = (s or "").strip()
    if "arxiv.org" in s:
        m = _ID.search(s)
        return m.group(1) if m else None
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", s):
        return _ID.search(s).group(1)
    return None


def fetch_arxiv_text(arxiv_id: str, timeout: int = 60) -> tuple[str, str]:
    """Return (text, source_kind) for an arXiv id. Tries LaTeX source, then abstract."""
    import requests

    try:
        r = requests.get(f"https://arxiv.org/e-print/{arxiv_id}", headers=_UA, timeout=timeout)
        r.raise_for_status()
        tex = extract_tex(r.content)
        if tex.strip():
            return tex, "latex"
    except Exception:  # noqa: BLE001 — fall back to the abstract
        pass

    r = requests.get(
        f"http://export.arxiv.org/api/query?id_list={arxiv_id}", headers=_UA, timeout=timeout
    )
    r.raise_for_status()
    return parse_atom_abstract(r.text), "abstract"


# ---- parsing helpers (offline-testable) ----

def extract_tex(data: bytes) -> str:
    """Pull concatenated .tex out of an arXiv e-print payload (tar.gz, or a single
    gzipped file)."""
    try:
        tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:*")
        parts = [
            tf.extractfile(m).read().decode("utf-8", "replace")
            for m in tf.getmembers()
            if m.isfile() and m.name.endswith(".tex")
        ]
        if parts:
            return "\n\n".join(parts)
    except tarfile.TarError:
        pass
    try:
        return gzip.decompress(data).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return ""


def parse_atom_abstract(xml: str) -> str:
    """Title + summary from an arXiv API Atom response."""
    ns = {"a": "http://www.w3.org/2005/Atom"}
    try:
        entry = ET.fromstring(xml).find("a:entry", ns)
    except ET.ParseError:
        return ""
    if entry is None:
        return ""
    title = (entry.findtext("a:title", "", ns) or "").strip()
    summary = (entry.findtext("a:summary", "", ns) or "").strip()
    return f"{title}\n\n{summary}".strip()
