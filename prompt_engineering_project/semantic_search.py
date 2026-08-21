"""Semantic search over Microsoft AI Show transcript segments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_FILE = PROJECT_ROOT / "embedding_index_3m.json"
CACHE_DIR = PROJECT_ROOT / ".cache"
SEGMENTS_CACHE = CACHE_DIR / "segments.json"
VECTORS_CACHE = CACHE_DIR / "vectors.npy"
MODEL_DIR = CACHE_DIR / "models"

#: 384-dim, ~130 MB quantised ONNX model; runs on CPU, no PyTorch needed.
MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOP_K = 5


class IndexNotBuilt(RuntimeError):
    """The local embeddings have not been generated yet."""


@dataclass(frozen=True)
class Segment:
    """A single 3-minute, timestamped slice of a video transcript."""

    title: str
    speaker: str
    summary: str
    video_id: str
    start: str
    seconds: int

    @property
    def youtube_url(self) -> str:
        """Deep link that starts playback at this segment."""
        return f"https://www.youtube.com/watch?v={self.video_id}&t={self.seconds}s"

    @property
    def timestamp(self) -> str:
        """``00:04:30`` -> ``4:30`` (hours kept only when non-zero)."""
        parts = self.start.split(":")
        if len(parts) != 3:
            return self.start
        hours, minutes, seconds = parts
        if hours in ("00", "0"):
            return f"{int(minutes)}:{seconds}"
        return f"{int(hours)}:{minutes}:{seconds}"

    @property
    def text(self) -> str:
        """What gets embedded."""
        return f"{self.title}. {self.speaker}. {self.summary}"


@dataclass(frozen=True)
class SearchResult:
    """One ranked hit."""

    rank: int
    score: float
    segment: Segment


def load_segments() -> list[Segment]:
    """Read the dataset. Cached, since parsing 49 MB of JSON takes ~20 s."""
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}\n"
            "Place 'embedding_index_3m.json' next to app.py."
        )
    signature = _dataset_signature()

    if SEGMENTS_CACHE.exists():
        try:
            cached = json.loads(SEGMENTS_CACHE.read_text(encoding="utf-8"))
            if cached.get("source") == signature:
                return [Segment(**row) for row in cached["segments"]]
        except (OSError, ValueError, TypeError):
            pass  # a corrupt cache is never fatal, just rebuild it

    with DATASET_FILE.open(encoding="utf-8") as handle:
        records = json.load(handle)
    segments = [
        Segment(
            title=record.get("title", "Untitled"),
            speaker=record.get("speaker", ""),
            summary=record.get("summary", ""),
            video_id=record.get("videoId", ""),
            start=record.get("start", "00:00:00"),
            seconds=int(record.get("seconds", 0)),
        )
        for record in records
    ]

    _write_cache(
        SEGMENTS_CACHE,
        json.dumps({"source": signature, "segments": [asdict(s) for s in segments]}),
    )
    return segments


def index_exists() -> bool:
    """True when the embeddings have already been built."""
    return VECTORS_CACHE.exists()


def build_vectors(segments: list[Segment]) -> np.ndarray:
    """Embed every segment with the local model and cache the result."""
    vectors = _unit(_encode([s.text for s in segments]))
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(VECTORS_CACHE, vectors)
    except OSError:
        pass  # still usable in memory if the cache cannot be written
    return vectors


class SearchEngine:
    """Ranks segments against a natural-language query by cosine similarity."""

    def __init__(self, segments: list[Segment], vectors: np.ndarray) -> None:
        self.segments = segments
        self.vectors = vectors

    @classmethod
    def load(cls) -> "SearchEngine":
        segments = load_segments()
        if not VECTORS_CACHE.exists():
            raise IndexNotBuilt(
                "No embeddings found. Build them once with:\n"
                "    python build_local_index.py"
            )
        vectors = np.load(VECTORS_CACHE)
        if vectors.shape[0] != len(segments):
            raise IndexNotBuilt(
                "The embeddings are stale. Rebuild them with:\n"
                "    python build_local_index.py"
            )
        return cls(segments, vectors)

    @property
    def dimensions(self) -> int:
        return int(self.vectors.shape[1])

    def search(self, query: str, top_k: int = TOP_K) -> list[SearchResult]:
        """Return the ``top_k`` most relevant segments, best first."""
        query = (query or "").strip()
        if not query:
            return []

        query_vector = _unit(_encode([query]))[0]
        if query_vector.shape[0] != self.dimensions:
            raise IndexNotBuilt(
                f"Query has {query_vector.shape[0]} dimensions but the index has "
                f"{self.dimensions}. Rebuild with 'python build_local_index.py'."
            )
        # Both sides are unit length, so the dot product *is* cosine similarity.
        scores = self.vectors @ query_vector

        top_k = max(1, min(int(top_k), len(scores)))
        # Partial sort: only the top candidates need to be ordered.
        best = np.argpartition(-scores, top_k - 1)[:top_k]
        best = best[np.argsort(-scores[best])]
        return [
            SearchResult(rank, float(scores[i]), self.segments[int(i)])
            for rank, i in enumerate(best, start=1)
        ]


# -------------------------------------------------------------


_model = None


def _encode(texts: list[str]) -> np.ndarray:
    """Embed texts with the local model, loading it on first use."""
    global _model
    if _model is None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise IndexNotBuilt(
                "The 'fastembed' package is not installed.\n"
                "Run: pip install -r requirements.txt"
            ) from exc
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        _model = TextEmbedding(MODEL_NAME, cache_dir=str(MODEL_DIR))
    return np.asarray(list(_model.embed(texts, batch_size=64)), dtype=np.float32)


def _unit(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so dot product == cosine similarity."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return (matrix / np.where(norms == 0, 1.0, norms)).astype(np.float32)


def _dataset_signature() -> dict[str, int]:
    stat = DATASET_FILE.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _write_cache(path: Path, payload: str) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    except OSError:
        pass  # a read-only checkout should still be searchable
