"""
rag/retriever.py - STAGE 2b: The vector store. Saves chunk vectors and searches them.

    store = VectorStore(chunks)
    store.search(question, k=3)                       -> top 3 chunks + scores
    store.rank(question, mode="bm25", rerank=True)    -> ALL chunks ranked, best first

We rank every chunk (not only the top k) so the failure report can say
"the correct chunk was ranked #7". The generator only gets ranking[:top_k].

Search modes (each RAG version picks one):
  semantic : cosine similarity of embeddings  -> finds similar MEANING
  bm25     : BM25 keyword ranking             -> finds the same WORDS (great for names and numbers)
  hybrid   : average of both (each scaled to 0..1), plus a small boost for the newest document
Optional second stage:
  rerank   : a cross-encoder re-orders the top config.RERANK_DEPTH results (see reranker.py)
"""

import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np

import config
from rag.embedder import embed

_STOPWORDS = set("""a an and are as at be by can do does for from has have how i if in is it its
of on or our the their there this to was what when where which who will with you your""".split())

MODES = ("semantic", "bm25", "hybrid")


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS]


@lru_cache(maxsize=4096)
def _query_vector(question):
    """Questions are embedded many times (live hit-rate, comparisons) - cache them."""
    return embed([question])[0]


class VectorStore:
    K1, B = 1.5, 0.75  # standard BM25 settings

    def __init__(self, chunks):
        self.chunks = chunks
        self.vectors = embed([c["text"] for c in chunks])  # one row per chunk
        self._build_bm25()

    def _build_bm25(self):
        """BM25: a word counts more if it is rare across chunks; repeats help less and less; long chunks are not favoured."""
        self.docs = [Counter(tokenize(c["text"])) for c in self.chunks]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg_length = self.lengths.mean() if len(self.lengths) else 1.0
        n = len(self.docs)
        df = Counter(w for d in self.docs for w in d)
        self.idf = {w: math.log((n - f + 0.5) / (f + 0.5) + 1) for w, f in df.items()}

    def _bm25_scores(self, question):
        scores = np.zeros(len(self.chunks))
        norm = self.K1 * (1 - self.B + self.B * self.lengths / self.avg_length)
        for w in set(tokenize(question)):
            if w not in self.idf:
                continue
            tf = np.array([d.get(w, 0) for d in self.docs], dtype=float)
            scores += self.idf[w] * tf * (self.K1 + 1) / (tf + norm)
        return scores

    def scores(self, question, mode="semantic"):
        semantic = self.vectors @ _query_vector(question)
        if mode == "semantic":
            return semantic
        bm25 = self._bm25_scores(question)
        if mode in ("bm25", "keyword"):  # "keyword" = name used by older saved runs
            return bm25
        top = bm25.max()
        bm25_scaled = bm25 / top if top > 0 else bm25
        newest = np.array([0.05 if c.get("is_current", True) else 0.0 for c in self.chunks])
        return 0.5 * semantic + 0.5 * bm25_scaled + newest  # hybrid

    def rank(self, question, mode="semantic", rerank=False):
        """Every chunk, best match first, with its rank and score."""
        scores = self.scores(question, mode)
        order = np.argsort(-scores)
        ranked = [{**self.chunks[i], "score": round(float(scores[i]), 3)} for i in order]
        if rerank:
            from rag.reranker import rerank as rerank_chunks
            depth = config.RERANK_DEPTH
            ranked = rerank_chunks(question, ranked[:depth]) + ranked[depth:]
        return [{**c, "rank": r + 1} for r, c in enumerate(ranked)]

    def search(self, question, k=3, mode="semantic", rerank=False):
        """Top k chunks + scores."""
        return self.rank(question, mode, rerank)[:k]
