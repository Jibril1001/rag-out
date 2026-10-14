import numpy as np

from ragopt.bm25 import BM25, tokenize
from ragopt.chunking.structure import chunk_paper_structure, split_long
from ragopt.data import Paper, Paragraph
from ragopt.fusion import rrf
from ragopt.pipeline import Config, build_chunks, retrieve


def mk_paper():
    long_p = " ".join(f"Sentence number {i} talks about things." for i in range(60))     # ~2400 chars
    paras = [Paragraph("abstract", "Abstract", "We study X."),
             Paragraph("s0:p0", "Intro", "Tiny."),
             Paragraph("s0:p1", "Intro", "A" * 400 + " end."),
             Paragraph("s1:p0", "Methods", long_p),
             Paragraph("s1:p1", "Methods", "z" * 2500)]                                  # one giant "word"
    return Paper("P", "My Title", paras)


def test_spans_reproduce_text_and_respect_limits():
    paper = mk_paper(); by = paper.by_pid()
    chunks = chunk_paper_structure(paper, max_chars=1000, merge_below=300)
    for c in chunks:
        assert len(c.text) <= 1000
        assert "\n\n".join(by[s.paragraph_id].text[s.start:s.end] for s in c.spans) == c.text
    # nothing dropped: every non-space char of every paragraph is inside some span
    for p in paper.paragraphs:
        covered = np.zeros(len(p.text), bool)
        for c in chunks:
            for s in c.spans:
                if s.paragraph_id == p.pid:
                    covered[s.start:s.end] = True
        assert all(covered[i] or p.text[i].isspace() for i in range(len(p.text)))


def test_never_crosses_sections_and_merges_tiny():
    chunks = chunk_paper_structure(mk_paper(), max_chars=1000, merge_below=300)
    for c in chunks:
        assert len({s.paragraph_id.split(":")[0] for s in c.spans}) == 1
    first_intro = [c for c in chunks if c.spans[0].paragraph_id == "s0:p0"][0]
    assert {s.paragraph_id for s in first_intro.spans} == {"s0:p0", "s0:p1"}      # "Tiny." merged forward


def test_context_only_in_embed_text():
    plain = chunk_paper_structure(mk_paper(), context=False)
    ctx = chunk_paper_structure(mk_paper(), context=True)
    assert [c.text for c in plain] == [c.text for c in ctx]
    assert all(c.embed_text.startswith("My Title\n") for c in ctx)
    assert all(c.embed_text == c.text for c in plain)


def test_split_long_cuts_at_sentences():
    text = "One fine day. " * 30
    parts = split_long(text.strip(), 100)
    assert all(e - s <= 100 for s, e in parts)
    assert all(text[s:e].endswith(".") for s, e in parts)


def test_bm25_ranks_rare_term_first_and_idf():
    docs = [tokenize(t) for t in ["the transformer model", "a cat sat on a mat", "transformer transformer attention"]]
    bm = BM25().fit(docs)
    s = bm.scores(tokenize("transformer attention"))
    assert s.argmax() == 2 and s[1] == 0
    assert abs(bm.idf["attention"] - np.log(1 + (3 - 1 + .5) / (1 + .5))) < 1e-6
    assert "the" not in tokenize("the cat")


def test_rrf():
    fused = rrf([[1, 2, 3], [3, 1, 4]], k=60)
    assert [d for d, _ in fused][:2] == [1, 3]
    assert abs(dict(fused)[1] - (1 / 61 + 1 / 62)) < 1e-12


def test_pipeline_bm25_and_hybrid_with_fake_embedder(tmp_path):
    from test_dense_embed import FakeEmbedder
    paper = mk_paper()
    cfg = Config("t", chunker="structure", context=True, retriever="bm25", topn=5)
    chunks = build_chunks(cfg, {"P": paper})
    idx = retrieve(cfg, chunks, ["Sentence number talks"], None)
    assert len(idx[0]) == min(5, len(chunks))
    cfg2 = Config("t2", chunker="structure", retriever="hybrid", topn=5, pool=10)
    idx2 = retrieve(cfg2, chunks, ["Sentence number talks"], FakeEmbedder(cache_dir=str(tmp_path)))
    assert len(set(idx2[0])) == len(idx2[0]) <= 5                                  # unique chunks
