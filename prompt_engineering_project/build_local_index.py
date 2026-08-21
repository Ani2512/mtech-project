"""Embed every transcript segment locally."""

from semantic_search import MODEL_NAME, build_vectors, load_segments

segments = load_segments()
print(f"Loaded {len(segments)} segments.")
print(f"Embedding with '{MODEL_NAME}'.")

vectors = build_vectors(segments)
