"""Compose timelines and mixed audio with exact labels.

Two event banks:
  ProceduralBank  tones/noises generated on the fly (tests, dry runs)
  ESC50Bank       isolated clips from ESC-50 (download once; CC BY-NC 3.0)

Placement controls how many events, how many repeats of the same label, and
the overlap probability, so WHILE queries are satisfiable by construction.
"""
from __future__ import annotations

import csv
import io
import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .timeline import Event, Timeline

SR = 16000


# ---------------------------------------------------------------- banks
class ProceduralBank:
    """Five synthetic 'sound classes' with audibly distinct spectra."""
    LABELS = ["beep", "buzz", "hiss", "click", "chirp"]
    # human-readable phrases used in query text
    PHRASES = {"beep": "beep", "buzz": "buzz", "hiss": "hiss", "click": "clicking", "chirp": "chirp"}

    def __init__(self, rng: random.Random):
        self.rng = rng

    def labels(self):
        return list(self.LABELS)

    def phrase(self, label):
        return self.PHRASES[label]

    def sample(self, label: str) -> np.ndarray:
        dur = self.rng.uniform(0.4, 1.6)
        t = np.arange(int(dur * SR)) / SR
        if label == "beep":
            x = np.sin(2 * np.pi * 880 * t)
        elif label == "buzz":
            x = 2 * ((110 * t) % 1.0) - 1
        elif label == "hiss":
            x = np.random.default_rng(self.rng.randrange(1 << 30)).standard_normal(len(t))
        elif label == "click":
            x = np.zeros_like(t)
            x[:: int(0.08 * SR)] = 1.0
        elif label == "chirp":
            x = np.sin(2 * np.pi * (300 + 1500 * t / dur) * t)
        else:
            raise KeyError(label)
        env = np.minimum(1.0, np.minimum(t, dur - t) / 0.05)  # 50 ms fades
        return (x * env * 0.3).astype(np.float32)


class ESC50Bank:
    URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    DEFAULT_CLASSES = ["dog", "rooster", "door_wood_knock", "clock_alarm", "car_horn", "siren", "glass_breaking",
                       "keyboard_typing", "footsteps", "cat", "church_bells", "train", "sneezing", "laughing"]

    def __init__(self, root: Path, rng: random.Random, classes: list[str] | None = None):
        import soundfile as sf
        self.sf = sf
        self.rng = rng
        self.root = Path(root)
        self._ensure()
        self.classes = classes or self.DEFAULT_CLASSES
        self.index: dict[str, list[Path]] = {c: [] for c in self.classes}
        with open(self.root / "meta" / "esc50.csv", newline="") as f:
            for row in csv.DictReader(f):
                if row["category"] in self.index:
                    self.index[row["category"]].append(self.root / "audio" / row["filename"])
        missing = [c for c, v in self.index.items() if not v]
        if missing:
            raise ValueError(f"ESC-50 classes not found: {missing}")

    def _ensure(self):
        if (self.root / "meta" / "esc50.csv").exists():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        print(f"downloading ESC-50 to {self.root} (~600 MB)")
        data = urllib.request.urlopen(self.URL).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for m in z.namelist():
                rel = m.split("/", 1)[1] if "/" in m else m
                if rel.startswith(("audio/", "meta/")) and not m.endswith("/"):
                    dest = self.root / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(z.read(m))

    def labels(self):
        return list(self.classes)

    def phrase(self, label):
        return label.replace("_", " ")

    def sample(self, label: str) -> np.ndarray:
        path = self.rng.choice(self.index[label])
        x, sr = self.sf.read(path, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            idx = np.linspace(0, len(x) - 1, int(len(x) * SR / sr))
            x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
        # trim leading/trailing near-silence so onset/offset labels are tight
        thr = 0.02 * (np.abs(x).max() + 1e-9)
        nz = np.where(np.abs(x) > thr)[0]
        if len(nz):
            x = x[max(0, nz[0] - int(0.02 * SR)): nz[-1] + int(0.02 * SR)]
        # cap at 2.5 s so several events fit in a clip
        return x[: int(2.5 * SR)]


# ---------------------------------------------------------------- placement
def compose_clip(bank, rng: random.Random, duration: float = 20.0, n_events: int = 6, n_labels: int = 3,
                 p_overlap: float = 0.35, min_overlap: float = 0.3, snr_db: float = 20.0):
    """Place n_events drawn from n_labels classes; with prob p_overlap an event
    is started inside the previous one (overlap >= min_overlap). Returns (audio, Timeline)."""
    labels = rng.sample(bank.labels(), n_labels)
    # guarantee at least one repeated label and at least one unique label
    seq = labels + [rng.choice(labels[:1]) for _ in range(n_events - n_labels)]
    rng.shuffle(seq)
    audio = np.zeros(int(duration * SR), dtype=np.float32)
    events: list[Event] = []
    t = rng.uniform(0.3, 1.0)
    prev = None
    for lab in seq:
        x = bank.sample(lab)
        d = len(x) / SR
        if prev is not None and rng.random() < p_overlap and prev.label != lab:
            t = rng.uniform(prev.onset, max(prev.onset, prev.offset - min_overlap))
        elif prev is not None:
            t = prev.offset + rng.uniform(0.4, 2.5)
        if t + d > duration - 0.2:
            break
        s = int(t * SR)
        audio[s: s + len(x)] += x
        ev = Event(lab, round(t, 3), round(t + d, 3))
        events.append(ev)
        prev = ev
    # background noise at the requested SNR
    sig = np.sqrt(np.mean(audio ** 2)) + 1e-9
    noise = np.random.default_rng(rng.randrange(1 << 30)).standard_normal(len(audio)).astype(np.float32)
    noise *= sig / (10 ** (snr_db / 20)) / (np.sqrt(np.mean(noise ** 2)) + 1e-9)
    audio = np.clip(audio + noise, -1, 1)
    return audio, Timeline(duration, events)
