"""WER/CER plus the script-agnostic variant and per-mode failure rates."""
from __future__ import annotations

from collections import Counter

from .normalize import phonetic_key, tokenize
from .taxonomy import Attribution, Mode, RefToken, attribute, sentence_level_translation

try:
    import jiwer
except ImportError:  # tests still run without it
    jiwer = None


def _edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    r, h = tokenize(ref), tokenize(hyp)
    return _edit_distance(r, h) / max(len(r), 1)


def cer(ref: str, hyp: str) -> float:
    r, h = list(" ".join(tokenize(ref))), list(" ".join(tokenize(hyp)))
    return _edit_distance(r, h) / max(len(r), 1)


def script_agnostic_wer(ref: str, hyp: str) -> float:
    """WER after mapping both sides to phonetic keys, so script choice is free."""
    r = [phonetic_key(t) for t in tokenize(ref)]
    h = [phonetic_key(t) for t in tokenize(hyp)]
    return _edit_distance(r, h) / max(len(r), 1)


def score_utterance(ref: str | list[RefToken], hyp: str) -> dict:
    ref_text = ref if isinstance(ref, str) else " ".join(t.text for t in ref)
    att: Attribution = attribute(ref, hyp)
    counts = att.counts()
    n_ref = max(len(att.ref_tokens), 1)
    return {
        "wer": wer(ref_text, hyp),
        "cer": cer(ref_text, hyp),
        "wer_script_agnostic": script_agnostic_wer(ref_text, hyp),
        "n_ref_tokens": len(att.ref_tokens),
        "n_hyp_tokens": len(att.hyp_tokens),
        "modes": counts,
        "mode_rates": {k: v / n_ref for k, v in counts.items()},
        "sentence_translated": sentence_level_translation(ref, hyp),
        "events": [
            {"op": e.op, "ref": e.ref, "hyp": e.hyp, "ref_index": e.ref_index, "mode": e.mode.value, "script_only": e.script_only}
            for e in att.events
        ],
    }


def summarize(rows: list[dict]) -> dict:
    """Corpus-level aggregate over per-utterance score dicts."""
    if not rows:
        return {}
    n_tok = sum(r["n_ref_tokens"] for r in rows)
    modes = Counter()
    for r in rows:
        modes.update(r["modes"])
    tot_lex = sum(v for k, v in modes.items() if k != Mode.SCRIPT.value)
    return {
        "n_utterances": len(rows),
        "n_ref_tokens": n_tok,
        "wer": sum(r["wer"] * r["n_ref_tokens"] for r in rows) / n_tok,
        "cer": sum(r["cer"] * r["n_ref_tokens"] for r in rows) / n_tok,
        "wer_script_agnostic": sum(r["wer_script_agnostic"] * r["n_ref_tokens"] for r in rows) / n_tok,
        "mode_counts": dict(modes),
        "mode_rate_per_100_tokens": {k: 100.0 * v / n_tok for k, v in modes.items()},
        "mode_share_of_lexical_errors": {k: (v / tot_lex if tot_lex else 0.0) for k, v in modes.items() if k != Mode.SCRIPT.value},
        "sentence_translation_rate": sum(r["sentence_translated"] for r in rows) / len(rows),
    }
