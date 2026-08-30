"""paper — a thin harness over the RLM engine that explains a research paper.

Give it an arXiv id or URL; it fetches the paper's LaTeX source and runs the RLM
engine over it (the paper text *is* PROMPT) with an instruction to produce a
thorough breakdown — claims, methodology, results, limitations.

The engine does the reasoning; this package only adds three things:
  - fetch.py   : arXiv id/URL -> full LaTeX text (stdlib only)
  - explain.py : the breakdown instruction + explain() (calls rlm.run, with a
                 deterministic map-reduce fallback if the agent never converges)
  - cli.py     : the `rrl-explain` command
"""

from .explain import EXPLAIN_PROMPT, Explanation, explain
from .fetch import fetch_arxiv, parse_arxiv_id

__all__ = ["fetch_arxiv", "parse_arxiv_id", "explain", "Explanation", "EXPLAIN_PROMPT"]
