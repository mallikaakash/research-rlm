"""Fetch + batch-populate tests — offline.

The network path (arXiv) is blocked in CI, so we test the parsing helpers with
synthetic payloads and the batch orchestration with a stub reader. Real fetching
is verified on a machine with network.

    python tests/test_fetch.py     # prints "ok"
    pytest tests/test_fetch.py
"""

import gzip
import io
import os
import sys
import tarfile
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.fetch import extract_tex, parse_arxiv_id, parse_atom_abstract  # noqa: E402
from research.note import Note, Claim  # noqa: E402
from research.read import _load_text  # noqa: E402
from research.seed import build_corpus, fetch_only, load_seed_ids  # noqa: E402


def test_parse_arxiv_id():
    assert parse_arxiv_id("2412.19437") == "2412.19437"
    assert parse_arxiv_id("2412.19437v2") == "2412.19437"
    assert parse_arxiv_id("https://arxiv.org/abs/2501.12948") == "2501.12948"
    assert parse_arxiv_id("https://arxiv.org/pdf/2505.09388v1") == "2505.09388"
    assert parse_arxiv_id("paper.txt") is None            # a filename, not an id
    assert parse_arxiv_id("just some text 12.34") is None


def test_extract_tex_from_tar():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, body in [("main.tex", b"\\section{A}\nhello"), ("refs.bib", b"junk")]:
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tf.addfile(info, io.BytesIO(body))
    out = extract_tex(buf.getvalue())
    assert "hello" in out and "junk" not in out          # only .tex is kept


def test_extract_tex_single_gzip():
    assert "single file tex" in extract_tex(gzip.compress(b"single file tex"))


def test_parse_atom_abstract():
    xml = (
        '<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
        "<title>Recursive Language Models</title>"
        "<summary>An inference technique.</summary>"
        "</entry></feed>"
    )
    out = parse_atom_abstract(xml)
    assert "Recursive Language Models" in out and "inference technique" in out


def test_build_corpus_orchestration():
    # Stub reader: no network/model — just returns a saved Note so we can check the
    # batch flow (dedup, count) and that link/index/goals run without error.
    from research.note import save_note

    def fake_reader(aid, corpus_dir="corpus", **kw):
        n = Note(id=aid, title=f"paper {aid}", tags=["rl"],
                 claims=[Claim(id="c1", statement="x", evidence="§1", strength="moderate")],
                 open_questions=["q?"])
        save_note(n, corpus_dir)
        return n

    with tempfile.TemporaryDirectory() as d:
        seed = os.path.join(d, "seed.yaml")
        open(seed, "w").write("a:\n  - {id: '2412.19437'}\n  - {id: '2501.12948'}\n")
        assert load_seed_ids(seed) == ["2412.19437", "2501.12948"]
        done = build_corpus(seed, corpus_dir=d, delay=0, reader=fake_reader)
        assert set(done) == {"2412.19437", "2501.12948"}
        assert os.path.exists(os.path.join(d, "_index.json"))
        assert os.path.exists(os.path.join(d, "_goals.jsonl"))


def test_fetch_only_raw_download():
    def fake_fetch(aid):
        return (f"\\section{{{aid}}}\nbody", "latex")

    with tempfile.TemporaryDirectory() as d:
        seed = os.path.join(d, "seed.yaml")
        open(seed, "w").write("a:\n  - {id: '2412.19437'}\n  - {id: '2501.12948'}\n")
        got = fetch_only(seed, corpus_dir=d, delay=0, fetcher=fake_fetch)
        assert set(got) == {"2412.19437", "2501.12948"}
        assert os.path.exists(os.path.join(d, "raw", "2412.19437.txt"))


def test_read_uses_raw_cache():
    # A cached raw file must be used verbatim, with no network call.
    with tempfile.TemporaryDirectory() as d:
        raw = os.path.join(d, "raw")
        os.makedirs(raw)
        open(os.path.join(raw, "2412.19437.txt"), "w").write("CACHED PAPER TEXT")
        text, kind, aid = _load_text("2412.19437", corpus_dir=d)
        assert text == "CACHED PAPER TEXT" and aid == "2412.19437"


if __name__ == "__main__":
    test_parse_arxiv_id()
    test_extract_tex_from_tar()
    test_extract_tex_single_gzip()
    test_parse_atom_abstract()
    test_build_corpus_orchestration()
    test_fetch_only_raw_download()
    test_read_uses_raw_cache()
    print("ok")
