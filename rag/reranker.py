"""
rag/reranker.py - Second-stage re-ranker.

First-stage search (embeddings / BM25) is fast but rough: it compares the question and each
chunk separately. A cross-encoder reads the question and a chunk TOGETHER, so it judges
relevance much better - but it is slower, so it only re-orders the top few candidates.

Model: config.RERANKER_MODEL (small, free, runs locally; downloads once).
"""

import math
import threading

import config

_model = None
_lock = threading.Lock()


def get_model():
    global _model
    with _lock:
        if _model is None:
            from sentence_transformers import CrossEncoder
            _model = CrossEncoder(config.RERANKER_MODEL)
        return _model


def relevance_scores(question, texts):
    """Relevance of each text to the question, as a probability-like 0..1 number."""
    if not texts:
        return []
    model = get_model()
    with _lock:
        logits = model.predict([(question, t) for t in texts], show_progress_bar=False)
    return [1 / (1 + math.exp(-float(x))) for x in logits]


def rerank(question, candidates):
    """Re-order candidate chunks by cross-encoder relevance. Keeps the first-stage score too."""
    scores = relevance_scores(question, [c["text"] for c in candidates])
    rescored = [{**c, "first_stage_score": c["score"], "score": round(s, 3)} for c, s in zip(candidates, scores)]
    return sorted(rescored, key=lambda c: -c["score"])
