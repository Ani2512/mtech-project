"""Hinglish failure taxonomy and rule-based attribution.

Given a reference (with optional per-token language labels, as in HiACC) and a
model hypothesis, align them and attribute every error token to one mode.
These rules are a *first pass* meant to be audited by native speakers; the
JSONL output keeps enough detail for that audit.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from enum import Enum

from .normalize import phonetic_key, script_of, tokenize


class Mode(str, Enum):
    FUSION = "FUSION"          # English stem + Hindi inflection in one word: filmein
    LIGHT_VERB = "LIGHT_VERB"  # English content word + Hindi light verb: adjust karna
    SCRIPT = "SCRIPT"          # right word, other script: phone -> फ़ोन  (not a lexical error)
    TRANSLATE = "TRANSLATE"    # translated instead of transcribed
    OTHER = "OTHER"


# Hindi inflectional endings that attach to English stems (Roman + Devanagari).
FUSION_SUFFIXES_ROMAN = ("ein", "on", "iyan", "iyon", "wala", "wale", "wali", "waale", "waali")
FUSION_SUFFIXES_DEVA = ("ों", "ें", "ियां", "ियों", "वाला", "वाले", "वाली")

# Light verbs (Roman and Devanagari stems). Matched by prefix so "karna/karta/kiya/karte" all hit.
LIGHT_VERB_STEMS = (
    "kar", "ho", "de", "le", "ban", "laga", "kiy", "kij",
    "कर", "हो", "दे", "ले", "बन", "लगा", "किय", "कीज",
)

# Small seed lexicon of English->Hindi pairs commonly *translated* by ASR/audio LLMs.
# Keys are phonetic keys of the English word; values are phonetic keys of Hindi renderings.
TRANSLATION_SEED = {
    phonetic_key("adjust"): {phonetic_key("samayojit"), phonetic_key("thik")},
    phonetic_key("tomorrow"): {phonetic_key("kal")},
    phonetic_key("today"): {phonetic_key("aaj")},
    phonetic_key("water"): {phonetic_key("paani")},
    phonetic_key("house"): {phonetic_key("ghar")},
    phonetic_key("friend"): {phonetic_key("dost")},
    phonetic_key("time"): {phonetic_key("samay"), phonetic_key("waqt")},
    phonetic_key("school"): {phonetic_key("vidyalay")},
    phonetic_key("work"): {phonetic_key("kaam")},
    phonetic_key("problem"): {phonetic_key("samasya")},
}
# Reverse direction (Hindi ref -> English hyp)
_REVERSE = {}
for en, his in TRANSLATION_SEED.items():
    for hi in his:
        _REVERSE.setdefault(hi, set()).add(en)


@dataclass
class RefToken:
    text: str
    lang: str | None = None  # 'hi', 'en', or None if unlabelled


@dataclass
class ErrorEvent:
    op: str                    # 'sub', 'del', 'ins'
    ref: str | None
    hyp: str | None
    ref_index: int | None
    mode: Mode
    script_only: bool = False  # True when the only difference is script


@dataclass
class Attribution:
    ref_tokens: list[str]
    hyp_tokens: list[str]
    events: list[ErrorEvent] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        c = {m.value: 0 for m in Mode}
        for e in self.events:
            c[e.mode.value] += 1
        return c


def is_fusion(token: str, lang: str | None = None) -> bool:
    """English stem carrying a Hindi inflection, e.g. 'filmein', 'फिल्मों', 'busses' is NOT fusion."""
    t = token.lower()
    if script_of(t) == "mixed":
        return True
    if lang == "hi":
        return False
    for suf in FUSION_SUFFIXES_ROMAN if script_of(t) == "latin" else FUSION_SUFFIXES_DEVA:
        if t.endswith(suf) and len(t) > len(suf) + 2:
            stem = t[: -len(suf)]
            # Roman: stem must look English-ish (has a consonant cluster typical of English loans)
            if script_of(t) == "latin":
                return lang == "en" or _looks_english(stem)
            return lang == "en"
    return False


def _looks_english(stem: str) -> bool:
    return any(c in stem for c in ("ck", "x", "ff", "tt", "ll", "ss", "ct", "pt", "ft", "st", "sp", "sk")) or stem.endswith(
        ("tion", "er", "ing", "al")
    )


def is_light_verb_context(ref: list[RefToken], i: int) -> bool:
    """ref[i] is a content word immediately followed by a Hindi light verb."""
    if i + 1 >= len(ref):
        return False
    nxt = ref[i + 1].text.lower()
    cur = ref[i]
    if cur.lang == "hi":
        return False
    if cur.lang is None and script_of(cur.text) != "latin":
        return False
    return any(nxt.startswith(s) for s in LIGHT_VERB_STEMS)


def _is_translation(ref_key: str, hyp_key: str) -> bool:
    return hyp_key in TRANSLATION_SEED.get(ref_key, ()) or hyp_key in _REVERSE.get(ref_key, ())


def attribute(reference: str | list[RefToken], hypothesis: str) -> Attribution:
    """Align reference and hypothesis on phonetic keys and label each error."""
    ref = [RefToken(t) for t in tokenize(reference)] if isinstance(reference, str) else reference
    hyp = tokenize(hypothesis)
    rk = [phonetic_key(t.text) for t in ref]
    hk = [phonetic_key(t) for t in hyp]
    out = Attribution([t.text for t in ref], hyp)

    sm = difflib.SequenceMatcher(a=rk, b=hk, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            # Same word; flag a script change so it is counted but not as a lexical error.
            for i, j in zip(range(i1, i2), range(j1, j2)):
                if script_of(ref[i].text) != script_of(hyp[j]) and script_of(hyp[j]) != "other":
                    out.events.append(ErrorEvent("sub", ref[i].text, hyp[j], i, Mode.SCRIPT, script_only=True))
            continue
        if tag == "replace":
            n = max(i2 - i1, j2 - j1)
            for k in range(n):
                i = i1 + k if i1 + k < i2 else None
                j = j1 + k if j1 + k < j2 else None
                out.events.append(_classify(ref, i, hyp[j] if j is not None else None, "sub" if i is not None and j is not None else ("del" if i is not None else "ins")))
        elif tag == "delete":
            for i in range(i1, i2):
                out.events.append(_classify(ref, i, None, "del"))
        elif tag == "insert":
            for j in range(j1, j2):
                out.events.append(ErrorEvent("ins", None, hyp[j], None, Mode.OTHER))
    return out


def _classify(ref: list[RefToken], i: int | None, hyp_tok: str | None, op: str) -> ErrorEvent:
    if i is None:
        return ErrorEvent(op, None, hyp_tok, None, Mode.OTHER)
    tok = ref[i]
    mode = Mode.OTHER
    if hyp_tok is not None and _is_translation(phonetic_key(tok.text), phonetic_key(hyp_tok)):
        mode = Mode.TRANSLATE
    elif is_fusion(tok.text, tok.lang):
        mode = Mode.FUSION
    elif is_light_verb_context(ref, i):
        mode = Mode.LIGHT_VERB
    elif hyp_tok is None and tok.lang == "en":
        # An English word silently dropped from a mixed utterance is usually the
        # decoder "smoothing" it away; keep it separate from generic deletions.
        mode = Mode.OTHER
    return ErrorEvent(op, tok.text, hyp_tok, i, mode)


def sentence_level_translation(ref: list[RefToken] | str, hypothesis: str) -> bool:
    """Whole-utterance translation signature: reference is mixed-script (or labelled
    mixed-language) but the hypothesis is entirely in one script and shares almost
    no phonetic keys with the reference."""
    ref_tokens = tokenize(ref) if isinstance(ref, str) else [t.text for t in ref]
    hyp = tokenize(hypothesis)
    if not ref_tokens or not hyp:
        return False
    ref_scripts = {script_of(t) for t in ref_tokens} - {"other"}
    hyp_scripts = {script_of(t) for t in hyp} - {"other"}
    overlap = len({phonetic_key(t) for t in ref_tokens} & {phonetic_key(t) for t in hyp})
    return len(ref_scripts) > 1 and len(hyp_scripts) == 1 and overlap / len(ref_tokens) < 0.3
