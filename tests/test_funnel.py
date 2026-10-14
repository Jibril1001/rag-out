from ragopt.analysis.funnel import classify, chunks_overlap
from ragopt.chunking import Chunk, Span
from ragopt.data import Paper, Paragraph, Question
from ragopt.metrics import GoldIndex

WORDS = " ".join(f"w{i}" for i in range(10))
paper = Paper("P", "t", [Paragraph("s0:p0", "A", WORDS), Paragraph("s0:p1", "A", "x y z")])
gi = GoldIndex({"P": paper})
full = lambda pid, n, pap="P": Chunk(f"{pap}:{pid}", pap, "t", [Span(pid, 0, n)])


def test_success_and_wrong_paper():
    q = Question("q", "P", "?", {"s0:p0"})
    assert classify(q, [full("s0:p0", len(WORDS))], [], gi) == "success"
    assert classify(q, [full("s0:p0", len(WORDS), "OTHER")], [], gi) == "wrong_paper"


def test_incomplete_boundary_ranked_low_not_found():
    q2 = Question("q", "P", "?", {"s0:p0", "s0:p1"})
    assert classify(q2, [full("s0:p0", len(WORDS))], [], gi) == "incomplete_multi_evidence"
    q = Question("q", "P", "?", {"s0:p0"})
    half = full("s0:p0", len(" ".join(f"w{i}" for i in range(4))))      # 4/10 tokens only
    assert classify(q, [half], [], gi) == "boundary_split"
    other = full("s0:p1", 5)                                              # right paper, wrong paragraph
    assert classify(q, [other], [other, full("s0:p0", len(WORDS))], gi) == "ranked_too_low"
    assert classify(q, [other], [other], gi) == "not_found"


def test_chunks_overlap():
    a = Chunk("a", "P", "t", [Span("s0:p0", 0, 10)])
    b = Chunk("b", "P", "t", [Span("s0:p0", 5, 15)])
    c = Chunk("c", "P", "t", [Span("s0:p0", 10, 20)])
    assert chunks_overlap(a, b) and not chunks_overlap(a, c)
