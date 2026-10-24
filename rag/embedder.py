"""
rag/embedder.py - STAGE 2a: Turn text into vectors (lists of numbers).

Input : a list of strings
Output: a NumPy array, one row of 384 numbers per string

Texts with similar meaning get similar numbers - that is how search works.
Uses the free local model in config.EMBEDDING_MODEL (no API key needed).
The vectors are normalised (length 1), so cosine similarity is just a dot product.
"""

import threading

import config

_model = None
_lock = threading.Lock()  # tests run in parallel threads; the model is used by one thread at a time


def get_model():
    """Load the model once (first call takes a few seconds), then reuse it."""
    global _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(config.EMBEDDING_MODEL)
        return _model


def embed(texts):
    model = get_model()
    with _lock:
        return model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False)
