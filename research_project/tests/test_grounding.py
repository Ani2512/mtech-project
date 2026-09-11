import json
import random

from dhwani.metrics import matched_f1, parse_intervals, score_query, summarize, union_iou
from dhwani.queries import generate
from dhwani.timeline import Event, Timeline

TL = Timeline(20.0, [
    Event("dog", 1.0, 2.0), Event("horn", 3.0, 4.0), Event("dog", 5.0, 6.0),
    Event("music", 7.0, 12.0), Event("dog", 8.0, 9.0), Event("steps", 9.5, 10.5), Event("dog", 14.0, 15.0),
])


def test_predicates():
    assert [e.interval for e in TL.plain("dog")] == [(1, 2), (5, 6), (8, 9), (14, 15)]
    assert [e.interval for e in TL.ordinal("dog", 2)] == [(5, 6)]
    assert [e.interval for e in TL.ordinal("dog", "last")] == [(14, 15)]
    assert TL.ordinal("dog", 9) == []
    assert [e.interval for e in TL.after("dog", "horn")] == [(5, 6), (8, 9), (14, 15)]
    assert [e.interval for e in TL.before("dog", "horn")] == [(1, 2)]
    assert [e.interval for e in TL.next_after("dog", "horn")] == [(5, 6)]
    assert [e.interval for e in TL.while_("dog", "music")] == [(8, 9)]
    assert [e.interval for e in TL.not_followed("dog", "steps", 3.0)] == [(1, 2), (5, 6), (14, 15)]
    assert TL.absent("cat") == []


def test_after_requires_unique_reference():
    import pytest
    with pytest.raises(ValueError):
        TL.after("horn", "dog")


def test_generate_is_consistent_with_predicates():
    qs = generate(TL, "c0", ["dog", "horn", "music", "steps", "cat"], random.Random(1), max_per_type=3)
    types = {q.qtype for q in qs}
    assert {"PLAIN", "ORDINAL", "AFTER", "BEFORE", "NEXT_AFTER", "WHILE", "NOT_FOLLOWED", "ABSENT"} <= types
    for q in qs:
        assert q.expects_empty == (len(q.answer) == 0)
        assert q.plain_intervals == [e.interval for e in TL.occ(q.x)]
        if q.qtype == "WHILE":
            assert q.answer == [e.interval for e in TL.while_(q.x, q.y)]
    # round-trips through JSON
    from dhwani.queries import Query
    for q in qs:
        assert Query.from_dict(json.loads(json.dumps(q.to_dict()))) == q


def test_parse_intervals():
    assert parse_intervals("[[1.2, 2.0], [7.5, 8.1]]") == [(1.2, 2.0), (7.5, 8.1)]
    assert parse_intervals("Sure! Here: [[1.2, 2.0]]") == [(1.2, 2.0)]
    assert parse_intervals("[]") == []
    assert parse_intervals("There is no such event.") == []
    assert parse_intervals("The dog barks from 1.2 to 2.0 and 7.5-8.1 seconds.") == [(1.2, 2.0), (7.5, 8.1)]
    assert parse_intervals('[{"start": 1, "end": 2}]') == [(1.0, 2.0)]
    assert parse_intervals("I cannot help with that") is None


def test_metrics():
    assert union_iou([(1, 2)], [(1, 2)]) == 1.0
    assert union_iou([], []) == 1.0
    assert union_iou([(1, 2)], []) == 0.0
    assert abs(union_iou([(1, 3)], [(2, 4)]) - 1 / 3) < 1e-9
    p, r, f = matched_f1([(1, 2), (5, 6)], [(1, 2), (5, 6), (8, 9)], 0.5)
    assert (p, r) == (1.0, 2 / 3)
    s = score_query([(1, 2)], [(1, 2), (5, 6)], False)
    assert s["n_pred"] == 1 and not s["count_acc"]
    s = score_query(None, [(1, 2)], False)
    assert s["parse_fail"] and s["f1@0.5"] == 0.0


def test_summary_rejection_metrics():
    rows = [
        {"qtype": "ORDINAL", **score_query([], [], True)},           # correct rejection
        {"qtype": "ORDINAL", **score_query([(1, 2)], [], True)},     # missed rejection
        {"qtype": "ORDINAL", **score_query([], [(1, 2)], False)},    # false rejection
        {"qtype": "ORDINAL", **score_query([(1, 2)], [(1, 2)], False)},
    ]
    s = summarize(rows)["ORDINAL"]
    assert s["rejection_precision"] == 0.5 and s["rejection_recall"] == 0.5
    assert s["false_rejection_rate"] == 0.5 and s["count_acc"] == 0.5


def test_end_to_end_procedural(tmp_path):
    from dhwani.build_benchmark import build
    from dhwani.run_zeroshot import main as run

    counts = build("procedural", 8, tmp_path / "bench", seed=3)
    assert counts["WHILE"] > 0 and counts["ABSENT"] > 0
    assert len(list((tmp_path / "bench" / "wav").glob("*.wav"))) == 8
    for mode in ("oracle", "first_only", "ignore_condition"):
        run(["--model", f"mock:{mode}", "--bench", str(tmp_path / "bench" / "benchmark.jsonl"), "--out", str(tmp_path / mode)])
        s = json.loads((tmp_path / mode / "summary.json").read_text())["by_type"]
        if mode == "oracle":
            assert s["ALL"]["f1@0.5"] == 1.0 and s["ALL"]["count_acc"] == 1.0
        if mode == "first_only":
            assert s["PLAIN"]["under_report_rate"] > 0


def test_rejection_metrics_are_none_when_no_rejection_queries():
    """A type with no empty-answer queries was never asked to reject; reporting
    0.0 would read as a failure at a task that was never posed."""
    rows = [{"qtype": "PLAIN", **score_query([(1, 2)], [(1, 2)], False)},
            {"qtype": "PLAIN", **score_query([(5, 6)], [(5, 6)], False)}]
    s = summarize(rows)["PLAIN"]
    assert s["n_rejection_queries"] == 0
    assert s["rejection_f1"] is None and s["rejection_precision"] is None and s["rejection_recall"] is None
    assert s["false_rejection_rate"] == 0.0

    # and a false rejection still shows up in false_rejection_rate
    rows.append({"qtype": "PLAIN", **score_query([], [(9, 10)], False)})
    s = summarize(rows)["PLAIN"]
    assert s["rejection_f1"] is None
    assert abs(s["false_rejection_rate"] - 1 / 3) < 1e-9


def test_rejection_metrics_present_when_type_has_rejection_queries():
    rows = [{"qtype": "ABSENT", **score_query([], [], True)},
            {"qtype": "ABSENT", **score_query([(1, 2)], [], True)}]
    s = summarize(rows)["ABSENT"]
    assert s["rejection_recall"] == 0.5 and s["rejection_f1"] is not None
