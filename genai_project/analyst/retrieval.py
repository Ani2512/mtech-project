"""Hybrid retrieval over the unstructured half of the data room.

The CSVs are handled by `analytics.py`, which computes real figures. The
markdown files — news feed, competitor tracker, industry research, company
profile — are the other half, and they are far too long to paste into every
agent's prompt. This module retrieves only the passages that bear on the
question actually being asked.

Two retrievers score every passage and their rankings are fused:

    BM25            lexical. Rewards exact term overlap, so a query naming
                    "Vietnam" or "tariff" finds the passage that says it.
    LSA (SVD)       dense. Terms are projected into a latent space built from
                    the corpus itself, so "Southeast Asia" can match a passage
                    about "Vietnam and Indonesia" with no shared words.

Neither alone is enough — BM25 misses paraphrase, LSA misses rare proper
nouns — so the two ranked lists are combined with reciprocal rank fusion,
which needs no score calibration between them.

Everything runs offline: BM25 is plain Python, LSA needs only numpy, and the
corpus is small enough (a few hundred passages) that the index is rebuilt in
milliseconds. There is no vector database to run and no embedding API to pay
for. `EMBEDDING_NOTE` at the bottom marks where a neural encoder would drop in.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable

from .datasets import SOURCES, DataRoom

# Words that carry no retrieval signal. Deliberately short — an aggressive
# stoplist hurts more than it helps on a corpus this size.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "has", "have", "how", "in", "is", "it", "its", "of", "on", "or",
    "our", "should", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "to", "was", "we", "were", "what", "when",
    "which", "who", "why", "will", "with", "would", "you", "your",
}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-']*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stopwords and single characters dropped."""
    return [
        t
        for t in _TOKEN_RE.findall((text or "").lower())
        if t not in _STOPWORDS and len(t) > 1
    ]


# ---------------------------------------------------------------------------
# passages
# ---------------------------------------------------------------------------
@dataclass
class Passage:
    """One retrievable unit of text, with enough provenance to cite it."""

    source: str  # the file it came from, e.g. "market/news_feed.md"
    section: str  # nearest enclosing markdown heading
    text: str
    tokens: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = tokenize(f"{self.section} {self.text}")

    @property
    def citation(self) -> str:
        return f"{self.source}" + (f" § {self.section}" if self.section else "")


# A passage below this length is almost always a stray heading or list marker.
_MIN_PASSAGE_CHARS = 80
# Above this, a passage dilutes its own relevance score and wastes context.
_MAX_PASSAGE_CHARS = 1400


def chunk_markdown(text: str, source: str) -> list[Passage]:
    """Split a markdown document into passages, one per block, heading-aware.

    Headings are carried down onto the blocks beneath them so a passage keeps
    the context it was written under — a bare bullet like "-12% YoY" is
    meaningless without the "Competitor pricing" heading above it.
    """
    passages: list[Passage] = []
    section = ""
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        block = "\n".join(buffer).strip()
        buffer.clear()
        if len(block) < _MIN_PASSAGE_CHARS:
            return
        for piece in _split_long(block):
            passages.append(Passage(source=source, section=section, text=piece))

    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            flush()
            section = line.lstrip("#").strip()
            continue
        if not line.strip():
            flush()
            continue
        buffer.append(line)
    flush()
    return passages


def _split_long(block: str) -> list[str]:
    """Break an over-long block on line boundaries, never mid-sentence."""
    if len(block) <= _MAX_PASSAGE_CHARS:
        return [block]
    out: list[str] = []
    current: list[str] = []
    size = 0
    for line in block.splitlines():
        if size + len(line) > _MAX_PASSAGE_CHARS and current:
            out.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        out.append("\n".join(current))
    return out


def build_corpus(room: DataRoom, domains: Iterable[str] | None = None) -> list[Passage]:
    """Collect passages from every markdown file the data room holds."""
    domains = list(domains) if domains is not None else list(SOURCES)
    passages: list[Passage] = []
    for domain in domains:
        for rel in SOURCES.get(domain, []):
            if not rel.endswith(".md") or not room.exists(rel):
                continue
            passages.extend(chunk_markdown(room.text(rel), rel))
    return passages


# ---------------------------------------------------------------------------
# BM25 — lexical scoring
# ---------------------------------------------------------------------------
class BM25:
    """Okapi BM25. `k1` bounds term-frequency saturation, `b` length penalty."""

    def __init__(self, corpus: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus
        self.doc_len = [len(p.tokens) for p in corpus]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if corpus else 0.0

        self.freqs: list[dict[str, int]] = []
        doc_freq: dict[str, int] = {}
        for p in corpus:
            counts: dict[str, int] = {}
            for t in p.tokens:
                counts[t] = counts.get(t, 0) + 1
            self.freqs.append(counts)
            for t in counts:
                doc_freq[t] = doc_freq.get(t, 0) + 1

        n = len(corpus)
        # Standard BM25 IDF with the +1 that keeps it positive for terms
        # appearing in more than half the corpus.
        self.idf = {
            t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()
        }

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = [0.0] * len(self.corpus)
        if not self.corpus:
            return out
        for i, counts in enumerate(self.freqs):
            norm = self.k1 * (
                1 - self.b + self.b * (self.doc_len[i] / (self.avg_len or 1))
            )
            total = 0.0
            for t in q:
                f = counts.get(t)
                if not f:
                    continue
                total += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / (f + norm)
            out[i] = total
        return out


# ---------------------------------------------------------------------------
# LSA — dense scoring
# ---------------------------------------------------------------------------
class LsaIndex:
    """TF-IDF projected through a truncated SVD into a dense latent space.

    This is the classic latent-semantic retrieval construction, not a neural
    encoder: the "embedding" is learned from this corpus alone. That is enough
    to bridge vocabulary gaps *within* the data room, which is the failure mode
    BM25 actually has here. It is not a substitute for a pretrained model on
    open-domain text — see `EMBEDDING_NOTE`.
    """

    def __init__(self, corpus: list[Passage], n_components: int = 96):
        self.ok = False
        self.corpus = corpus
        if len(corpus) < 4:
            return
        try:
            import numpy as np
        except ImportError:  # numpy absent → hybrid degrades to BM25 only
            return

        self.np = np
        vocab: dict[str, int] = {}
        for p in corpus:
            for t in p.tokens:
                vocab.setdefault(t, len(vocab))
        if not vocab:
            return
        self.vocab = vocab

        n_docs, n_terms = len(corpus), len(vocab)
        tf = np.zeros((n_docs, n_terms), dtype=np.float64)
        for i, p in enumerate(corpus):
            for t in p.tokens:
                tf[i, vocab[t]] += 1.0

        df = (tf > 0).sum(axis=0)
        self.idf = np.log((1 + n_docs) / (1 + df)) + 1.0
        matrix = _l2_normalize(np, tf * self.idf)

        # Rank cannot exceed either dimension; ask for no more than exists.
        k = int(min(n_components, n_docs - 1, n_terms - 1))
        if k < 2:
            return
        try:
            u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            return

        self.components = vt[:k]  # (k, n_terms) — the latent term basis
        self.doc_vectors = _l2_normalize(np, u[:, :k] * s[:k])
        self.ok = True

    def scores(self, query: str) -> list[float]:
        if not self.ok:
            return [0.0] * len(self.corpus)
        np = self.np
        q = np.zeros(len(self.vocab), dtype=np.float64)
        for t in tokenize(query):
            idx = self.vocab.get(t)
            if idx is not None:
                q[idx] += 1.0
        if not q.any():
            return [0.0] * len(self.corpus)
        q = q * self.idf
        q_vec = self.components @ q
        norm = float(np.linalg.norm(q_vec))
        if norm == 0.0:
            return [0.0] * len(self.corpus)
        # Both sides are unit-length, so the dot product is cosine similarity.
        return list(self.doc_vectors @ (q_vec / norm))


def _l2_normalize(np, matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------
@dataclass
class Hit:
    passage: Passage
    score: float
    bm25_rank: int | None
    dense_rank: int | None

    def why(self) -> str:
        """Short label showing which retriever(s) surfaced this passage."""
        parts = []
        if self.bm25_rank is not None:
            parts.append(f"bm25 #{self.bm25_rank + 1}")
        if self.dense_rank is not None:
            parts.append(f"dense #{self.dense_rank + 1}")
        return ", ".join(parts) or "unranked"


# Reciprocal rank fusion damping. 60 is the value from the original RRF paper;
# it keeps any single retriever's top hit from dominating the fused list.
_RRF_K = 60


class HybridRetriever:
    """BM25 + LSA, combined by weighted reciprocal rank fusion.

    RRF is used rather than a weighted sum of raw scores because BM25 scores
    are unbounded while cosine similarities sit in [-1, 1]; ranks are the only
    thing the two retrievers report on a common scale.
    """

    def __init__(self, corpus: list[Passage], alpha: float = 0.5):
        self.corpus = corpus
        self.alpha = min(max(alpha, 0.0), 1.0)
        self.bm25 = BM25(corpus)
        self.dense = LsaIndex(corpus)
        # With no usable dense index the blend collapses to pure lexical.
        self.dense_available = self.dense.ok

    def search(self, query: str, top_k: int = 8) -> list[Hit]:
        if not self.corpus or not query.strip():
            return []

        alpha = self.alpha if self.dense_available else 0.0
        bm25_order = _rank_order(self.bm25.scores(query))
        dense_order = _rank_order(self.dense.scores(query)) if alpha else {}

        fused: dict[int, float] = {}
        for idx, rank in bm25_order.items():
            fused[idx] = fused.get(idx, 0.0) + (1 - alpha) / (_RRF_K + rank + 1)
        for idx, rank in dense_order.items():
            fused[idx] = fused.get(idx, 0.0) + alpha / (_RRF_K + rank + 1)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            Hit(
                passage=self.corpus[i],
                score=score,
                bm25_rank=bm25_order.get(i),
                dense_rank=dense_order.get(i),
            )
            for i, score in ranked
        ]


def _rank_order(scores: list[float]) -> dict[int, int]:
    """Map passage index -> 0-based rank, dropping anything that scored zero."""
    ordered = sorted(
        (i for i, s in enumerate(scores) if s > 0),
        key=lambda i: scores[i],
        reverse=True,
    )
    return {idx: rank for rank, idx in enumerate(ordered)}


def format_hits(hits: list[Hit]) -> str:
    """Render retrieved passages as a citable evidence block for a prompt."""
    if not hits:
        return "(no passages matched this question)"
    blocks = []
    for n, hit in enumerate(hits, 1):
        blocks.append(
            f"[P{n}] {hit.passage.citation}  (retrieved by {hit.why()})\n"
            f"{hit.passage.text}"
        )
    return "\n\n".join(blocks)


# Swapping in a pretrained encoder means replacing `LsaIndex` with anything
# exposing `.scores(query) -> list[float]`: encode `corpus` once, encode the
# query per call, return cosine similarities. Nothing else in this module
# changes, because fusion consumes ranks rather than raw scores.
EMBEDDING_NOTE = (
    "Dense retrieval uses corpus-fitted LSA/SVD vectors, not a pretrained "
    "encoder. See LsaIndex for the swap point."
)