"""Script handling for Hinglish text.

Hinglish transcripts mix Devanagari and Roman script, and the same English
word may legitimately appear in either ("phone" / "फ़ोन"). Scoring must
therefore be able to compare tokens *script-agnostically*. We do that by
transliterating Devanagari to a Roman scheme and then collapsing both sides to
a consonant skeleton, so that "फ़ोन" and "phone" land on the same key while
most genuinely different words stay apart.

The collapse is deliberately lossy: it exists to decide whether two tokens
are "the same word in a different script", not to be a transliterator.
"""
from __future__ import annotations

import re
import unicodedata

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _xlit

    def deva_to_roman(text: str) -> str:
        return _xlit(text, sanscript.DEVANAGARI, sanscript.HK)
except ImportError:  # keep the harness importable without the optional dep
    def deva_to_roman(text: str) -> str:  # type: ignore[misc]
        return text

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")
_PUNCT = re.compile(r"[^\wऀ-ॿ' ]+", re.UNICODE)


def script_of(token: str) -> str:
    """'deva', 'latin', 'mixed' (both scripts inside one token) or 'other'."""
    d = bool(DEVANAGARI.search(token))
    l = bool(LATIN.search(token))
    if d and l:
        return "mixed"
    if d:
        return "deva"
    if l:
        return "latin"
    return "other"


def tokenize(text: str) -> list[str]:
    """Lower-case, strip punctuation (keeps Devanagari and apostrophes), split on whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("।", " ").replace("॥", " ")
    text = _PUNCT.sub(" ", text)
    return [t.lower() for t in text.split() if t]


# Script-specific pre-maps, then a shared coarse collapse to a consonant
# skeleton (leading vowel kept). Spelling-to-sound for English is fuzzy, so the
# key deliberately throws vowels away: 'office'/'ऑफिस' -> 'ofs',
# 'doctor'/'डॉक्टर' -> 'dktr'. Collisions (kal/kul) are accepted for phase 1.
_ENGLISH = [
    (r"sch", "sk"), (r"ck", "k"), (r"ch", "C"), (r"c(?=[eiy])", "s"), (r"q", "k"), (r"x", "ks"), (r"c", "k"), (r"C", "c"),
    (r"wh", "w"), (r"ph", "f"), (r"tion$", "shn"), (r"ge$", "j"), (r"dge$", "j"), (r"e$", ""),
]
# Harvard-Kyoto output of Devanagari ('c' already means the ch-sound).
_HK = [
    (r"ch", "c"), (r"([tdnrl])\.", r"\1"), (r"\.", ""), (r"~", "n"), (r"M", "n"), (r"H", ""),
    (r"ph", "f"),
]
_SHARED = [
    (r"v", "w"), (r"z", "j"), (r"sh", "s"),
    (r"th", "t"), (r"dh", "d"), (r"bh", "b"), (r"gh", "g"), (r"kh", "k"),
    (r"[^a-z]", ""),
    (r"(?<=.)[aeiouy]", ""),          # drop every vowel except a leading one
    (r"([a-z])\1+", r"\1"),          # collapse doubled letters
]


def phonetic_key(token: str) -> str:
    """Script-agnostic key. 'फ़ोन' and 'phone' -> 'fn'; 'school' and 'स्कूल' -> 'skl'."""
    t = token
    if DEVANAGARI.search(t):
        t = t.replace("़", "")            # drop nukta before transliterating
        # candra vowels (ऑ/ॉ, ऍ/ॅ, used for English loans) have no HK mapping; fold them
        t = t.replace("ऑ", "ओ").replace("ॉ", "ो").replace("ऍ", "ए").replace("ॅ", "े")
        t = deva_to_roman(t)
        for pat, rep in _HK:               # case-sensitive: M = anusvara, H = visarga
            t = re.sub(pat, rep, t)
        t = t.lower()
        rules = _SHARED
    else:
        t = t.lower()
        rules = _ENGLISH + _SHARED
    for pat, rep in rules:
        t = re.sub(pat, rep, t)
    return t
