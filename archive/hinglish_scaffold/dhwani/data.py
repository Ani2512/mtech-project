"""Dataset loaders. Every loader yields Utterance objects with a local audio
path, a reference transcript, and (when the corpus provides them) per-token
language labels used by the taxonomy.

Sources (licences verified 2026-09-11, see docs/data_licensing.md):
  hiacc       HiACC, Zenodo record 15551669, CC BY-NC 4.0, word-level hi/en labels
  mucs        MUCS 2021 Hindi-English, OpenSLR 104, CC BY-SA 4.0
  indicvoices ai4bharat/IndicVoices (Hindi), Hugging Face, CC BY 4.0
  synthetic   text-only utterances for the mock backend; no audio
"""
from __future__ import annotations

import csv
import json
import os
import random
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .taxonomy import RefToken

DATA_ROOT = Path(os.environ.get("DHWANI_DATA", "data"))

MUCS_URLS = {
    "train": "https://openslr.trmal.net/resources/104/Hindi-English_train.tar.gz",
    "test": "https://openslr.trmal.net/resources/104/Hindi-English_test.tar.gz",
}
HIACC_ZENODO = "https://zenodo.org/records/15551669"


@dataclass
class Utterance:
    id: str
    audio_path: str | None
    reference: str
    ref_tokens: list[RefToken] | None = None   # labelled tokens when available
    meta: dict = field(default_factory=dict)

    def ref_for_scoring(self):
        return self.ref_tokens if self.ref_tokens else self.reference


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    return dest


# ---------------------------------------------------------------- HiACC
def load_hiacc(split: str = "adult", root: Path = DATA_ROOT / "hiacc") -> Iterator[Utterance]:
    """HiACC must be downloaded manually from Zenodo (licence click-through).
    Expected layout after extraction:
        root/{adult,children}/audio/*.wav
        root/{adult,children}/transcripts.csv      columns: id, transcript
        root/{adult,children}/lang_labels.json     {id: [[token, 'hi'|'en'], ...]}
    Adjust the paths below once the archive layout is confirmed; the corpus
    paper states token-level language labels are stored in a JSON file."""
    base = root / split
    if not base.exists():
        raise FileNotFoundError(f"{base} not found. Download HiACC from {HIACC_ZENODO} and extract to {root}/")
    labels = {}
    lab = base / "lang_labels.json"
    if lab.exists():
        labels = json.loads(lab.read_text(encoding="utf-8"))
    with open(base / "transcripts.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            uid = row["id"]
            toks = [RefToken(t, l) for t, l in labels.get(uid, [])] or None
            yield Utterance(uid, str(base / "audio" / f"{uid}.wav"), row["transcript"], toks, {"split": split})


# ---------------------------------------------------------------- MUCS 2021
def load_mucs(split: str = "test", root: Path = DATA_ROOT / "mucs", download: bool = True) -> Iterator[Utterance]:
    """Kaldi-style layout inside the tarball: transcripts/text  ('uttid transcript') and
    audio/*.wav (paths in transcripts/wav.scp)."""
    tgz = root / f"Hindi-English_{split}.tar.gz"
    if download:
        _download(MUCS_URLS[split], tgz)
    extracted = root / split
    if not extracted.exists():
        with tarfile.open(tgz) as tf:
            tf.extractall(extracted)
    text_files = list(extracted.rglob("text"))
    scp_files = list(extracted.rglob("wav.scp"))
    if not text_files:
        raise FileNotFoundError(f"no 'text' file under {extracted}")
    wav = {}
    for scp in scp_files:
        for line in scp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                uid, path = line.split(maxsplit=1)
                wav[uid] = str(extracted / path.strip()) if not path.startswith("/") else path.strip()
    for line in text_files[0].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        uid, txt = line.split(maxsplit=1)
        yield Utterance(uid, wav.get(uid), txt, None, {"split": split})


# ---------------------------------------------------------------- IndicVoices (Hindi)
def load_indicvoices(split: str = "valid", root: Path = DATA_ROOT / "indicvoices") -> Iterator[Utterance]:
    from datasets import load_dataset  # heavy; imported lazily

    import soundfile as sf

    ds = load_dataset("ai4bharat/IndicVoices", "hindi", split=split, streaming=True)
    root.mkdir(parents=True, exist_ok=True)
    for i, ex in enumerate(ds):
        uid = f"iv_{split}_{i}"
        path = root / f"{uid}.wav"
        if not path.exists():
            sf.write(path, ex["audio"]["array"], ex["audio"]["sampling_rate"])
        yield Utterance(uid, str(path), ex.get("text") or ex.get("transcript", ""), None, {"split": split})


# ---------------------------------------------------------------- synthetic
SYNTHETIC = [
    # (transcript, [(token, lang), ...])
    ("मुझे कल दो filmein देखनी हैं", [("मुझे", "hi"), ("कल", "hi"), ("दो", "hi"), ("filmein", "en"), ("देखनी", "hi"), ("हैं", "hi")]),
    ("thoda settings adjust karna padega", [("thoda", "hi"), ("settings", "en"), ("adjust", "en"), ("karna", "hi"), ("padega", "hi")]),
    ("mera phone charge नहीं हो रहा", [("mera", "hi"), ("phone", "en"), ("charge", "en"), ("नहीं", "hi"), ("हो", "hi"), ("रहा", "hi")]),
    ("office ke baad meeting fix kar lena", [("office", "en"), ("ke", "hi"), ("baad", "hi"), ("meeting", "en"), ("fix", "en"), ("kar", "hi"), ("lena", "hi")]),
    ("यह problem बहुत बड़ी है", [("यह", "hi"), ("problem", "en"), ("बहुत", "hi"), ("बड़ी", "hi"), ("है", "hi")]),
    ("bus ticketon ka rate badh gaya", [("bus", "en"), ("ticketon", "en"), ("ka", "hi"), ("rate", "en"), ("badh", "hi"), ("gaya", "hi")]),
    ("kal school jaana hai time pe", [("kal", "hi"), ("school", "en"), ("jaana", "hi"), ("hai", "hi"), ("time", "en"), ("pe", "hi")]),
    ("doctor ne bola paani zyada piyo", [("doctor", "en"), ("ne", "hi"), ("bola", "hi"), ("paani", "hi"), ("zyada", "hi"), ("piyo", "hi")]),
]


def load_synthetic(n: int = 50, seed: int = 0) -> Iterator[Utterance]:
    rng = random.Random(seed)
    for i in range(n):
        txt, labs = rng.choice(SYNTHETIC)
        yield Utterance(f"syn_{i}", None, txt, [RefToken(t, l) for t, l in labs], {"split": "synthetic"})


LOADERS = {"hiacc": load_hiacc, "mucs": load_mucs, "indicvoices": load_indicvoices, "synthetic": load_synthetic}
