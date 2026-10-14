from ragopt.data import build_paper, build_questions, split_ids, answer_qids

REC = {
    "id": "p1", "title": "T", "abstract": "We study X.",
    "full_text": {"section_name": ["Intro", "Methods"],
                  "paragraphs": [["First para.", "", "Second  para."], ["Third."]]},
    "qas": {"question_id": ["q1", "q2"], "question": ["a?", "b?"],
            "answers": [
                {"answer": [
                    {"unanswerable": False, "evidence": ["Second para.", "FLOAT SELECTED: Table 1"]},
                    {"unanswerable": False, "evidence": ["Third.", "not in paper"]}]},
                {"answer": [
                    {"unanswerable": True, "evidence": []},
                    {"unanswerable": True, "evidence": []},
                    {"unanswerable": False, "evidence": ["We study X."]}]},
            ]},
}

def test_paragraph_ids_keep_original_indices():
    pids = [p.pid for p in build_paper(REC).paragraphs]
    assert pids == ["abstract", "s0:p0", "s0:p2", "s1:p0"]   # empty p1 skipped, ids stable

def test_gold_mapping():
    paper = build_paper(REC)
    q1, q2 = build_questions(REC, paper)
    assert q1.gold_pids == {"s0:p2", "s1:p0"}   # whitespace-normalised match
    assert q1.n_float_evidence == 1 and q1.n_unmatched_evidence == 1
    assert not q1.unanswerable
    assert q2.unanswerable and q2.gold_pids == {"abstract"}   # 2 of 3 = majority

def test_split_is_deterministic_and_sized():
    ids = [f"id{i}" for i in range(281)]
    a, b = split_ids(ids)
    assert len(a) == 80 and len(b) == 50 and not set(a) & set(b)
    assert split_ids(ids) == (a, b)
    assert len(answer_qids([f"q{i}" for i in range(182)])) == 20