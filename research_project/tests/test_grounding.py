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


def test_parse_real_model_output_shapes():
    """Shapes actually emitted by Qwen2-Audio on this benchmark. Returning None
    for these would score the parser rather than the model."""
    # python-style single-quoted dicts with prefixed keys
    assert parse_intervals("[{'sneeze_start': '0.63', 'sneeze_end': '1.09'}, "
                           "{'sneeze_start': '4.21', 'sneeze_end': '4.62'}]") == [(0.63, 1.09), (4.21, 4.62)]
    # single-quoted plain start/end
    assert parse_intervals("[{'start': '16.39', 'end': '16.94'}]") == [(16.39, 16.94)]
    # a dict whose single value holds a range string
    assert parse_intervals("[{'sneeze': '1.96-2.34'}, {'glass_breaking': '18.87-20.00'}]") == \
        [(1.96, 2.34), (18.87, 20.0)]
    # a flat pair rather than a list of pairs
    assert parse_intervals("[19.43, 20.00]") == [(19.43, 20.0)]
    # onset/offset naming
    assert parse_intervals('[{"onset": 1.0, "offset": 2.0}]') == [(1.0, 2.0)]
    # still refuses genuine non-answers
    assert parse_intervals("I am unable to analyse this audio.") is None
    # and still reads explicit emptiness
    assert parse_intervals("[]") == []


def test_rescore_recovers_parse_failures(tmp_path):
    """A parser improvement must be applicable to a finished run without
    re-running the model."""
    import json as _json
    from dhwani.rescore import rescore

    run = tmp_path / "run"
    run.mkdir()
    raws = ["[[1.0, 2.0]]", "[{'x_start': '1.0', 'x_end': '2.0'}]", "no idea"]
    with open(run / "predictions.jsonl", "w") as f:
        for i, raw in enumerate(raws):
            f.write(_json.dumps({
                "qid": f"q{i}", "qtype": "PLAIN", "text": "t", "answer": [[1.0, 2.0]],
                "raw": raw, "pred": None, "expects_empty": False}) + "\n")
    (run / "summary.json").write_text(_json.dumps(
        {"model": "m", "bench": "b", "by_type": summarize(
            [{"qtype": "PLAIN", **score_query(None, [(1.0, 2.0)], False)} for _ in raws])}))

    s = rescore(run, in_place=False)
    assert s["parse_recovered"] == 2                     # two of three now parse
    assert abs(s["by_type"]["ALL"]["parse_fail_rate"] - 1 / 3) < 1e-9
    assert abs(s["by_type"]["ALL"]["f1@0.5"] - 2 / 3) < 1e-9


def test_localisation_separates_place_from_duration():
    """A perfectly centred but too-short interval and a displaced one both score
    0 at IoU>=0.5; centre error and duration ratio tell them apart."""
    from dhwani.metrics import localisation

    gold = [(10.0, 12.0)]
    short_but_centred = [(10.9, 11.1)]
    displaced_right_size = [(2.0, 4.0)]

    assert matched_f1(short_but_centred, gold, 0.5)[2] == 0.0
    assert matched_f1(displaced_right_size, gold, 0.5)[2] == 0.0

    d1, r1 = localisation(short_but_centred, gold)
    d2, r2 = localisation(displaced_right_size, gold)
    assert d1[0] < 0.1 and abs(r1 - 0.1) < 1e-9   # right place, a tenth the length
    assert d2[0] == 8.0 and r2 == 1.0         # right length, far away

    s = score_query(short_but_centred, gold, False)
    assert abs(s["duration_ratio"] - 0.1) < 1e-9 and s["centre_errors"][0] < 0.1
    agg = summarize([{"qtype": "PLAIN", **s}])["PLAIN"]
    assert abs(agg["duration_ratio_median"] - 0.1) < 1e-9 and agg["centre_within_1s"] == 1.0


def test_agent_combine_matches_timeline_predicates():
    """The agent must apply exactly the semantics that define the ground truth,
    otherwise it is solving a different task."""
    from dhwani.agent import combine

    xs = [e.interval for e in TL.occ("dog")]
    horn = [e.interval for e in TL.occ("horn")]
    music = [e.interval for e in TL.occ("music")]
    steps = [e.interval for e in TL.occ("steps")]

    assert combine("PLAIN", xs, []) == [e.interval for e in TL.plain("dog")]
    assert combine("ORDINAL", xs, [], k=2) == [e.interval for e in TL.ordinal("dog", 2)]
    assert combine("ORDINAL", xs, [], k="last") == [e.interval for e in TL.ordinal("dog", "last")]
    assert combine("ORDINAL", xs, [], k=9) == []
    assert combine("AFTER", xs, horn) == [e.interval for e in TL.after("dog", "horn")]
    assert combine("BEFORE", xs, horn) == [e.interval for e in TL.before("dog", "horn")]
    assert combine("NEXT_AFTER", xs, horn) == [e.interval for e in TL.next_after("dog", "horn")]
    assert combine("WHILE", xs, music) == [e.interval for e in TL.while_("dog", "music")]
    assert combine("NOT_FOLLOWED", xs, steps, window=3.0) == \
        [e.interval for e in TL.not_followed("dog", "steps", 3.0)]


def test_agent_with_oracle_grounder_is_perfect(tmp_path):
    """With perfect grounding the agent must score 1.000, or its composition
    disagrees with the ground truth somewhere."""
    import json as _json
    from dhwani.build_benchmark import build
    from dhwani.run_agent import main as run_agent

    build("procedural", 15, tmp_path / "b", seed=5)
    run_agent(["--grounder", "oracle",
               "--bench", str(tmp_path / "b" / "benchmark.jsonl"),
               "--timelines", str(tmp_path / "b" / "timelines.jsonl"),
               "--out", str(tmp_path / "agent")])
    s = _json.loads((tmp_path / "agent" / "summary.json").read_text())["by_type"]
    assert s["ALL"]["f1@0.5"] == 1.0, s["ALL"]
    assert s["ALL"]["count_acc"] == 1.0
    assert s["ALL"]["parse_fail_rate"] == 0.0


def test_oracle_grounder_matches_multiword_labels():
    """Queries carry 'glass_breaking' while timelines and prompts may use
    'glass breaking'. Normalising only one side makes the oracle silently
    return nothing for every multi-word sound."""
    from dhwani.agent import oracle_grounder

    g = oracle_grounder({"c1": {"events": [
        {"label": "glass_breaking", "onset": 1.0, "offset": 2.0},
        {"label": "dog", "onset": 5.0, "offset": 6.0}]}})
    assert g("/x/c1.wav", "glass_breaking") == [(1.0, 2.0)]
    assert g("/x/c1.wav", "glass breaking") == [(1.0, 2.0)]
    assert g("/x/c1.wav", "dog") == [(5.0, 6.0)]
    assert g("/x/c1.wav", "cat") == []
    # must not match a different sound by substring
    assert g("/x/c1.wav", "glass") == []


def test_split_is_clip_level_and_stable():
    """Splitting by query would leak: queries from one clip share audio and
    timeline. And adding clips later must not reshuffle existing assignments."""
    from dhwani.split import assign

    a = {c: assign(c, 0, 0.7, 0.15) for c in [f"clip_{i:04d}" for i in range(400)]}
    assert set(a.values()) == {"train", "val", "test"}
    counts = {s: sum(v == s for v in a.values()) / len(a) for s in ("train", "val", "test")}
    assert 0.62 < counts["train"] < 0.78, counts
    assert 0.09 < counts["val"] < 0.21, counts
    # stable: same clip, same seed, same answer, regardless of what else exists
    assert assign("clip_0007", 0, 0.7, 0.15) == a["clip_0007"]
    # a different seed gives a different partition
    b = {c: assign(c, 1, 0.7, 0.15) for c in a}
    assert a != b


def test_split_benchmark_has_no_clip_overlap(tmp_path):
    import json as _json
    from dhwani.build_benchmark import build
    from dhwani.split import split_benchmark

    build("procedural", 40, tmp_path / "b", seed=2)
    stats = split_benchmark(tmp_path / "b" / "benchmark.jsonl", tmp_path / "b", seed=0)
    assert sum(d["clips"] for d in stats.values()) == 40
    seen = {}
    for s in ("train", "val", "test"):
        for line in open(tmp_path / "b" / f"benchmark_{s}.jsonl"):
            cid = _json.loads(line)["clip_id"]
            assert seen.setdefault(cid, s) == s, f"{cid} appears in two splits"


def test_sft_mix_hits_the_requested_plain_share(tmp_path):
    """Grounding is the bottleneck, so the plain share is a deliberate knob and
    must actually be honoured."""
    import json as _json
    from dhwani.build_benchmark import build as build_bench
    from dhwani.sft_data import build as build_sft
    from dhwani.split import split_benchmark

    build_bench("procedural", 40, tmp_path / "b", seed=4)
    split_benchmark(tmp_path / "b" / "benchmark.jsonl", tmp_path / "b", seed=0)
    for ratio in (0.3, 0.5, 0.8):
        s = build_sft(tmp_path / "b" / "benchmark_train.jsonl", tmp_path / "b" / "timelines.jsonl",
                      tmp_path / f"sft_{ratio}.jsonl", plain_ratio=ratio, seed=0)
        assert abs(s["plain_share"] - ratio) < 0.05, (ratio, s)
        assert s["synthesised_plain"] > 0          # augmentation actually fired
        rows = [_json.loads(l) for l in open(tmp_path / f"sft_{ratio}.jsonl")]
        assert len(rows) == s["examples"]
        # prompt format must match inference exactly
        from dhwani.models import SYSTEM
        assert rows[0]["messages"][0]["content"] == SYSTEM
        assert rows[0]["messages"][1]["content"].startswith("Locate:")
        _json.loads(rows[0]["target"])            # target is valid JSON intervals


def test_sft_targets_are_parseable_by_the_metric(tmp_path):
    """A target the scorer cannot read would train the model to emit unscoreable text."""
    import json as _json
    from dhwani.build_benchmark import build as build_bench
    from dhwani.sft_data import build as build_sft
    from dhwani.split import split_benchmark

    build_bench("procedural", 20, tmp_path / "b", seed=6)
    split_benchmark(tmp_path / "b" / "benchmark.jsonl", tmp_path / "b", seed=0)
    build_sft(tmp_path / "b" / "benchmark_train.jsonl", tmp_path / "b" / "timelines.jsonl",
              tmp_path / "sft.jsonl", plain_ratio=0.5, seed=0)
    for line in open(tmp_path / "sft.jsonl"):
        e = _json.loads(line)
        assert parse_intervals(e["target"]) is not None


def test_collator_masks_the_prompt_and_keeps_the_answer(tmp_path):
    """Loss must fall only on the answer. If the mask is off by even one token
    the model learns to echo our instruction text, and nothing about the metric
    would reveal it."""
    import numpy as np
    import soundfile as sf
    import torch

    from dhwani.train_lora import GroundingCollator

    wav = tmp_path / "clip.wav"
    sf.write(wav, np.zeros(16000, dtype="float32"), 16000)

    class StubTokenizer:
        pad_token_id = 0
        eos_token = " <eos>"

        def __call__(self, text, add_special_tokens=False):
            class R: pass
            r = R(); r.input_ids = text.split()
            return r

    class StubProcessor:
        tokenizer = StubTokenizer()

        def apply_chat_template(self, conv, add_generation_prompt=True, tokenize=False):
            # one token per word keeps the arithmetic checkable by hand
            return "P1 P2 P3 P4"

        def __call__(self, text, audio, sampling_rate, return_tensors, padding):
            ids = [[hash(w) % 1000 + 1 for w in t.split()] for t in text]
            width = max(len(i) for i in ids)
            padded = [i + [0] * (width - len(i)) for i in ids]
            return {"input_ids": torch.tensor(padded)}

    c = GroundingCollator(StubProcessor())
    ex = {"audio": str(wav),
          "messages": [{"role": "system", "content": "sys"},
                       {"role": "user", "content": "Locate: every dog"}],
          "target": "[[1.0, 2.0]]"}
    batch = c([ex])
    labels, input_ids = batch["labels"], batch["input_ids"]

    # prompt is 4 tokens -> first 4 masked, the answer tokens survive
    assert (labels[0, :4] == -100).all(), labels[0, :4]
    assert (labels[0, 4:] != -100).any()
    # surviving labels equal the inputs there (teacher forcing on the answer)
    keep = labels[0] != -100
    assert torch.equal(labels[0][keep], input_ids[0][keep])
    # padding is masked too
    pad = input_ids[0] == 0
    if pad.any():
        assert (labels[0][pad] == -100).all()


def test_collator_masks_each_row_of_a_batch_independently(tmp_path):
    """A shared mask length would leak answer tokens from the shorter prompt."""
    import numpy as np
    import soundfile as sf
    import torch

    from dhwani.train_lora import GroundingCollator

    wav = tmp_path / "c.wav"
    sf.write(wav, np.zeros(16000, dtype="float32"), 16000)

    class StubTokenizer:
        pad_token_id = 0
        eos_token = " <eos>"

        def __call__(self, text, add_special_tokens=False):
            class R: pass
            r = R(); r.input_ids = text.split()
            return r

    class StubProcessor:
        tokenizer = StubTokenizer()
        _n = [2, 5]                       # different prompt lengths per row

        def apply_chat_template(self, conv, add_generation_prompt=True, tokenize=False):
            return " ".join(f"P{i}" for i in range(self._n.pop(0)))

        def __call__(self, text, audio, sampling_rate, return_tensors, padding):
            ids = [[i + 1 for i, _ in enumerate(t.split())] for t in text]
            width = max(len(i) for i in ids)
            return {"input_ids": torch.tensor([i + [0] * (width - len(i)) for i in ids])}

    c = GroundingCollator(StubProcessor())
    exs = [{"audio": str(wav), "messages": [{"role": "user", "content": "q"}], "target": "[]"},
           {"audio": str(wav), "messages": [{"role": "user", "content": "q"}], "target": "[[1.0, 2.0]]"}]
    labels = c(exs)["labels"]
    assert (labels[0, :2] == -100).all() and (labels[0, 2:4] != -100).any()
    assert (labels[1, :5] == -100).all() and (labels[1, 5:] != -100).any()


def test_hybrid_selects_on_val_and_reports_on_test(tmp_path):
    """Choosing the arm on the same queries it is reported on would be taking
    the max of two noisy estimates and calling it a method."""
    import json as _json
    from dhwani.hybrid import main as hybrid_main
    from dhwani.split import assign

    # Construct two arms with a known, type-dependent winner.
    qids = [f"clip_{i:04d}_q{j}" for i in range(120) for j in range(2)]
    def write(run, better):
        run.mkdir(parents=True, exist_ok=True)
        with open(run / "predictions.jsonl", "w") as f:
            for n, q in enumerate(qids):
                qtype = "AFTER" if n % 2 == 0 else "ORDINAL"
                good = (better == "agent") == (qtype == "AFTER")
                s = score_query([(1.0, 2.0)] if good else [(9.0, 9.5)], [(1.0, 2.0)], False)
                f.write(_json.dumps({"qid": q, "qtype": qtype, "answer": [[1.0, 2.0]], **s}) + "\n")
    write(tmp_path / "direct", "direct")
    write(tmp_path / "agent", "agent")

    hybrid_main(["--direct", str(tmp_path / "direct"), "--agent", str(tmp_path / "agent"),
                 "--out", str(tmp_path / "hyb")])
    res = _json.loads((tmp_path / "hyb" / "summary.json").read_text())
    assert res["choice"]["AFTER"] == "agent"
    assert res["choice"]["ORDINAL"] == "direct"
    # hybrid picks the winner for each type, so it beats both arms on test
    assert res["by_type"]["ALL"]["f1@0.5"] == 1.0
    assert res["arms"]["direct"]["ALL"]["f1@0.5"] < 1.0
    assert res["arms"]["agent"]["ALL"]["f1@0.5"] < 1.0
    # reported only on test clips
    n_test = sum(1 for q in qids if assign(q.rsplit("_q", 1)[0], 0, 0.7, 0.15) == "test")
    assert res["n_test"] == n_test
