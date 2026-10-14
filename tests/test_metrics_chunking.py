from ragopt.chunking import Span, Chunk, chunk_paper_fixed
from ragopt.data import Paper, Paragraph, Question
from ragopt.metrics import GoldIndex, covered_gold, recall_at_k, reciprocal_rank, evaluate

WORDS = " ".join(f"w{i}" for i in range(10))          # 10 tokens, single spaces


def offs(n_words):
    """char offset right after the n-th word of WORDS."""
    return len(" ".join(f"w{i}" for i in range(n_words)))


def make():
    paper = Paper("P", "t", [Paragraph("s0:p0", "A", WORDS), Paragraph("s0:p1", "A", "x y z")])
    gi = GoldIndex({"P": paper})
    return paper, gi


def chunk(cid, pid, s, e, paper_id="P"):
    return Chunk(cid, paper_id, "txt", [Span(pid, s, e)])


def test_threshold_exactly_80_percent():
    paper, gi = make()
    q = Question("q", "P", "?", {"s0:p0"})
    assert covered_gold([chunk("a", "s0:p0", 0, offs(8))], q, gi) == {"s0:p0"}   # 8/10
    assert covered_gold([chunk("a", "s0:p0", 0, offs(7))], q, gi) == set()       # 7/10


def test_union_vs_single():
    paper, gi = make()
    q = Question("q", "P", "?", {"s0:p0"})
    c1 = chunk("a", "s0:p0", 0, offs(4))                       # words 0-3
    c2 = chunk("b", "s0:p0", len(" ".join(f"w{i}" for i in range(3))) + 1, len(WORDS))  # words 3-9
    # together they cover everything; alone c1 covers 4/10, c2 covers 7/10
    assert covered_gold([c1, c2], q, gi, "union") == {"s0:p0"}
    assert covered_gold([c1, c2], q, gi, "single") == set()


def test_other_papers_do_not_count():
    paper, gi = make()
    q = Question("q", "P", "?", {"s0:p0"})
    wrong = chunk("z", "s0:p0", 0, len(WORDS), paper_id="OTHER")
    assert covered_gold([wrong], q, gi) == set()


def test_recall_and_mrr():
    paper, gi = make()
    q = Question("q", "P", "?", {"s0:p0", "s0:p1"})
    miss = chunk("m", "s0:p1", 0, 1)
    ranked = [miss, miss, chunk("h", "s0:p0", 0, len(WORDS))]
    assert recall_at_k(ranked, q, gi, 2) == 0.0
    assert recall_at_k(ranked, q, gi, 5) == 0.5            # 1 of 2 gold covered
    assert abs(reciprocal_rank(ranked, q, gi) - 1 / 3) < 1e-9


def test_evaluate_skips_questions_without_gold():
    paper, gi = make()
    q1 = Question("q1", "P", "?", {"s0:p0"})
    q2 = Question("q2", "P", "?", set())
    full = chunk("h", "s0:p0", 0, len(WORDS))
    summary, _ = evaluate({"q1": [full], "q2": []}, [q1, q2], gi)
    assert summary["n"] == 1 and summary["R@10"] == 1.0 and summary["MRR@10"] == 1.0


def test_fixed_chunker_spans_are_faithful():
    paras = [Paragraph(f"s0:p{i}", "S", f"para{i} " + "word " * 60) for i in range(8)]
    paper = Paper("P", "t", paras)
    by = paper.by_pid()
    chunks = chunk_paper_fixed(paper, size=300, overlap=60)
    assert len(chunks) > 3
    for c in chunks:
        assert len(c.text) <= 300
        pieces = [by[s.paragraph_id].text[s.start:s.end] for s in c.spans]
        assert c.text.replace("\n\n", "") == "".join(pieces)       # spans reproduce the text
    covered = {(s.paragraph_id) for c in chunks for s in c.spans}
    assert covered == {p.pid for p in paras}                       # nothing dropped
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_oracle_ranking_reaches_ceiling():
    from ragopt.headroom import oracle_ranking
    paras = [Paragraph(f"s0:p{i}", "S", f"para{i} " + "word " * 60) for i in range(8)]
    paper = Paper("P", "t", paras)
    gi = GoldIndex({"P": paper})
    q = Question("q", "P", "?", {"s0:p3"})
    chunks = chunk_paper_fixed(paper, size=300, overlap=60)
    ranked = oracle_ranking(chunks, q, gi, 10)
    assert len(ranked) == min(10, len(chunks))
    assert recall_at_k(ranked, q, gi, 10) == 1.0          # union can always rebuild a paragraph
    assert recall_at_k(ranked, q, gi, 1) <= 1.0
