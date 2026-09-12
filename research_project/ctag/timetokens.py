"""Atomic timestamp tokens, so a time is one categorical decision.

Qwen2.5-Omni's tokenizer splits '16.76' into five tokens, one per character:
    '16.76' -> '1' '6' '.' '7' '6'
A two-interval answer costs 25 tokens. Three problems follow. Emitting a time
is a five-step sequence where any slip moves the answer by seconds; 16.76 and
16.8 share almost no token structure, so the representation cannot express that
a near miss is nearly right; and "after the horn" becomes a comparison between
digit strings the model wrote itself.

Prior work fixes this by giving each quantised time its own token:

  TEMPO (arXiv:2608.29999) adds ~601 tokens for t in {0.0, 0.1, ..., 60.0},
  initialising each embedding as "the mean of the BPE decomposition of the
  corresponding numeric value", making each timestamp "a single categorical
  decision over approximately 600 candidates". It adds a distance-aware
  Gaussian loss (see `soft_labels`) so near misses earn partial credit.

  TimeAudio (arXiv:2511.11039) instead uses M=20 anchor/offset tokens
  (<a2><f5> style), with anchors initialised from the numeral embedding and
  offsets from the mean of the numeral and decimal-point embeddings. Its
  ablation credits the markers alone with +3.0 mIoU on temporal grounding.

We follow TEMPO's flat scheme: it is simpler, and with 20-second clips the
vocabulary is small anyway. The anchor/offset scheme is the better choice if
this is ever extended to long-form audio, where a flat vocabulary would grow
linearly with duration.
"""
from __future__ import annotations

import math
import re

EMPTY_TOKEN = "<t=none>"
_TOK_RE = re.compile(r"<t=(\d+\.\d)>")


class TimeVocab:
    """Quantised time tokens covering [0, max_seconds] at `resolution`."""

    def __init__(self, max_seconds: float = 30.0, resolution: float = 0.1):
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        self.max_seconds = float(max_seconds)
        self.resolution = float(resolution)
        self.n_steps = int(round(self.max_seconds / self.resolution)) + 1
        self.times = [round(i * self.resolution, 10) for i in range(self.n_steps)]
        self.tokens = [self._tok(t) for t in self.times] + [EMPTY_TOKEN]

    def _tok(self, t: float) -> str:
        return f"<t={t:.1f}>"

    # ---------------------------------------------------------------- mapping
    def index(self, t: float) -> int:
        """Nearest quantised index, clamped into range.

        Deliberately not round(): Python rounds halves to even, so 0.65 would
        quantise down to 0.6 while 0.75 goes up to 0.8. Worse, t/resolution is
        not exact in binary -- 3.15/0.1 is 31.4999999999999996 -- so a plain
        round is unpredictable near a midpoint. Round half up with a tolerance
        of 1e-6 of a step, which is 1e-7 s at 0.1 s resolution and far below
        anything the metric can see.
        """
        i = math.floor(float(t) / self.resolution + 0.5 + 1e-6)
        return max(0, min(self.n_steps - 1, i))

    def quantise(self, t: float) -> float:
        return self.times[self.index(t)]

    def token(self, t: float) -> str:
        return self.tokens[self.index(t)]

    # ---------------------------------------------------------------- codec
    def encode(self, intervals) -> str:
        """An interval list becomes a flat token string: two tokens per interval.
        An empty answer is its own single token, so 'nothing here' is also one
        categorical decision rather than a punctuation pattern."""
        if not intervals:
            return EMPTY_TOKEN
        out = []
        for a, b in intervals:
            a, b = float(a), float(b)
            if b < a:
                a, b = b, a
            out.append(self.token(a))
            out.append(self.token(b))
        return "".join(out)

    def decode(self, text: str):
        """Tokens back to intervals. Returns [] for the empty token, and None if
        nothing parseable is present, matching ctag.metrics.parse_intervals."""
        if text is None:
            return None
        if EMPTY_TOKEN in text:
            return []
        vals = [float(m) for m in _TOK_RE.findall(text)]
        if not vals:
            return None
        out = []
        for i in range(0, len(vals) - 1, 2):
            a, b = vals[i], vals[i + 1]
            if b < a:
                a, b = b, a
            if b > a:
                out.append((a, b))
        return out

    # ---------------------------------------------------------------- loss
    def soft_labels(self, t: float, sigma: float = 0.3) -> list[float]:
        """TEMPO's distance-aware target: q_k proportional to
        exp(-(t_k - t*)^2 / (2 sigma^2)), normalised over the time tokens.

        Cross-entropy against a one-hot target says a prediction 0.1 s away is
        exactly as wrong as one 10 s away. This says otherwise, which is the
        whole point of an ordinal vocabulary. The empty token gets zero mass:
        it is not near any time.
        """
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        w = [math.exp(-((tk - t) ** 2) / (2 * sigma * sigma)) for tk in self.times]
        z = sum(w)
        if z <= 0:                       # target far outside the range
            q = [0.0] * self.n_steps
            q[self.index(t)] = 1.0
            return q + [0.0]
        return [x / z for x in w] + [0.0]

    # ---------------------------------------------------------------- init
    def init_embeddings(self, tokenizer, embedding_matrix):
        """TEMPO: initialise each new embedding as the mean of the BPE pieces of
        the number it represents, so the tokens start where the model already
        represents those digits rather than at random.

        Returns the number of rows written.
        """
        import torch

        n = 0
        with torch.no_grad():
            for tok, t in zip(self.tokens, self.times + [None]):
                tid = tokenizer.convert_tokens_to_ids(tok)
                if tid is None or tid < 0:
                    continue
                text = f"{t:.1f}" if t is not None else "none"
                pieces = tokenizer(text, add_special_tokens=False).input_ids
                if not pieces:
                    continue
                embedding_matrix[tid] = embedding_matrix[pieces].mean(dim=0)
                n += 1
        return n
