from dhwani.metrics import score_utterance, summarize, script_agnostic_wer, wer
from dhwani.normalize import phonetic_key, script_of, tokenize
from dhwani.taxonomy import Mode, RefToken, attribute, is_fusion, sentence_level_translation


def test_tokenize_strips_danda_and_punct():
    assert tokenize("मुझे कल जाना है। Okay?") == ["मुझे", "कल", "जाना", "है", "okay"]


def test_script_detection():
    assert script_of("phone") == "latin"
    assert script_of("फ़ोन") == "deva"
    assert script_of("filmों") == "mixed"


def test_phonetic_key_is_script_agnostic():
    assert phonetic_key("phone") == phonetic_key("फ़ोन")
    assert phonetic_key("school") == phonetic_key("स्कूल")
    assert phonetic_key("time") == phonetic_key("टाइम")
    assert phonetic_key("chai") == phonetic_key("चाय")
    assert phonetic_key("office") == phonetic_key("ऑफिस")
    assert phonetic_key("doctor") == phonetic_key("डॉक्टर")
    assert phonetic_key("problem") == phonetic_key("प्रॉब्लम")
    assert phonetic_key("kal") != phonetic_key("ghar")
    assert phonetic_key("kal") != phonetic_key("school")


def test_script_only_difference_is_not_lexical_error():
    s = score_utterance("mera phone kharab hai", "mera फ़ोन kharab hai")
    assert s["wer"] > 0
    assert s["wer_script_agnostic"] == 0
    assert s["modes"]["SCRIPT"] == 1 and s["modes"]["OTHER"] == 0


def test_fusion_detection():
    assert is_fusion("filmein", "en")
    assert is_fusion("ticketon", "en")
    assert is_fusion("फिल्मों", "en")
    assert not is_fusion("karein", "hi")
    assert not is_fusion("main", "hi")


def test_fusion_attributed():
    ref = [RefToken("do", "hi"), RefToken("filmein", "en"), RefToken("dekhi", "hi")]
    att = attribute(ref, "do films dekhi")
    assert [e.mode for e in att.events] == [Mode.FUSION]


def test_light_verb_attributed():
    ref = [RefToken("settings", "en"), RefToken("adjust", "en"), RefToken("karna", "hi"), RefToken("hoga", "hi")]
    att = attribute(ref, "settings karna hoga")
    assert [e.mode for e in att.events] == [Mode.LIGHT_VERB]


def test_word_translation_attributed():
    ref = [RefToken("kal", "hi"), RefToken("school", "en"), RefToken("jaana", "hi")]
    att = attribute(ref, "tomorrow school jaana")
    assert [e.mode for e in att.events] == [Mode.TRANSLATE]


def test_sentence_translation_flag():
    assert sentence_level_translation("मुझे कल school जाना है", "I have to go to school tomorrow")
    assert not sentence_level_translation("मुझे कल school जाना है", "मुझे कल school जाना है")


def test_summary_aggregates():
    rows = [score_utterance("a b c", "a b c"), score_utterance("a b c", "a x c")]
    s = summarize(rows)
    assert s["n_utterances"] == 2 and abs(s["wer"] - 1 / 6) < 1e-9
    assert s["mode_counts"]["OTHER"] == 1


def test_mock_pipeline_runs(tmp_path):
    from dhwani.run_baseline import main
    main(["--model", "mock", "--dataset", "synthetic", "--n", "40", "--out", str(tmp_path)])
    assert (tmp_path / "summary.json").exists()
    lines = (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 40
