"""Fetch an arXiv paper's LaTeX *source* as one string — stdlib only.

Why LaTeX source and not the PDF: the source preserves exact equations, table
numbers, hyperparameters, and footnotes that PDF text-extraction garbles or
drops. For a *thorough* breakdown that fidelity is the whole point.

    fetch_arxiv("2512.24601")                     -> str  (all .tex concatenated)
    fetch_arxiv("https://arxiv.org/abs/2512.24601")
    fetch_arxiv("arxiv.org/pdf/2401.02385v2")

Pipeline: parse_arxiv_id -> download_source (the e-print tarball) -> extract_tex.
Each step is independently testable; only extract_tex touches no network.
"""

from __future__ import annotations

import gzip
import io
import re
import tarfile
import urllib.request

# arXiv asks clients to send a descriptive User-Agent; the default urllib one is
# sometimes rejected. Keep it honest and identifiable.
_UA = "research-rlm/0.1 (paper explainer; +https://github.com/mallikaakash/research-rlm)"

# New-style id: 2512.24601 or 2401.02385v2   |   old-style: hep-th/9901001, math.AG/0601001
_NEW = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_OLD = re.compile(r"^[a-z-]+(\.[A-Z]{2})?/\d{7}(v\d+)?$", re.I)


def parse_arxiv_id(s: str) -> str:
    """Extract a bare arXiv id from an id or an arxiv.org URL.

    Conservative on purpose: it accepts a bare id, or an ``/abs/``|``/pdf/`` arxiv
    URL, and rejects anything else with a clear error rather than guessing (so a
    stray file path never gets treated as a paper).
    """
    s = (s or "").strip()
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([^\s?#]+)", s, re.I)
    cand = re.sub(r"\.pdf$", "", m.group(1), flags=re.I) if m else s
    if _NEW.match(cand) or _OLD.match(cand):
        return cand
    raise ValueError(f"Not a recognizable arXiv id or URL: {s!r}")


def download_source(arxiv_id: str, timeout: float = 30.0) -> bytes:
    """Download the e-print archive for an arXiv id (usually a gzipped tar)."""
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — fixed arxiv host
        return resp.read()


def extract_tex(raw: bytes) -> str:
    """Turn e-print bytes into one LaTeX string.

    arXiv e-prints come in a few shapes: usually a gzipped tar of the sources,
    sometimes a single gzipped .tex, occasionally a bare file. Handle each; when
    it's a tar, concatenate every .tex member (name-sorted, each headed by a
    comment marking the file) so section order is stable and traceable.
    """
    try:
        tf = tarfile.open(fileobj=io.BytesIO(raw))
    except tarfile.TarError:
        tf = None

    if tf is not None:
        with tf:
            texts = []
            for m in sorted(tf.getmembers(), key=lambda x: x.name):
                if m.isfile() and m.name.lower().endswith(".tex"):
                    fh = tf.extractfile(m)
                    if fh is not None:
                        body = fh.read().decode("utf-8", "replace")
                        texts.append(f"% ==== {m.name} ====\n{body}")
        if texts:
            return "\n\n".join(texts)
        raise ValueError("e-print tar contained no .tex files")

    try:  # single gzipped file (often the .tex itself)
        return gzip.decompress(raw).decode("utf-8", "replace")
    except OSError:
        return raw.decode("utf-8", "replace")  # bare, uncompressed


def fetch_arxiv(url_or_id: str, timeout: float = 30.0) -> str:
    """arXiv id/URL -> full LaTeX source string."""
    return extract_tex(download_source(parse_arxiv_id(url_or_id), timeout=timeout))
