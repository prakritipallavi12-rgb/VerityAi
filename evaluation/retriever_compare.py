"""
evaluation/retriever_compare.py - Score retrievers side by side, WITHOUT any LLM calls.

For every answerable test we know which chunk holds the answer, so each retriever can be graded
on its own, instantly:
  Hit@k     share of questions whose answer chunk is in the top k
  MRR       mean reciprocal rank: 1 if the answer chunk is ranked 1st, 0.5 if 2nd, 0.33 if 3rd...
  Avg rank  where the answer chunk ended up on average (lower is better)
  ms/query  search time

Every mode (embeddings, BM25, hybrid) is scored with and without the re-ranker,
so you can see whether the re-ranker actually improves things.
"""

import time

from evaluation.question_gen import gold_chunk_ids
from rag.retriever import MODES

MODE_NAMES = {"semantic": "Embeddings", "bm25": "BM25 (keyword)", "hybrid": "Hybrid"}


def compare_retrievers(tests, store, top_k):
    answerable = [t for t in tests if t.get("answerable")]
    rows = []
    for mode in MODES:
        for rerank in (False, True):
            hits, reciprocal, ranks, seconds = 0, 0.0, [], 0.0
            for t in answerable:
                gold = set(gold_chunk_ids(t, store.chunks))
                start = time.time()
                ranking = store.rank(t["question"], mode=mode, rerank=rerank)
                seconds += time.time() - start
                rank = next((c["rank"] for c in ranking if c["chunk_id"] in gold), None)
                if rank:
                    ranks.append(rank)
                    reciprocal += 1 / rank
                    hits += rank <= top_k
            n = len(answerable) or 1
            rows.append({
                "retriever": MODE_NAMES[mode] + (" + re-ranker" if rerank else ""),
                "mode": mode, "rerank": rerank,
                "hit_at_k": round(hits / n, 3), "mrr": round(reciprocal / n, 3),
                "avg_rank": round(sum(ranks) / len(ranks), 2) if ranks else None,
                "ms_per_query": round(1000 * seconds / n, 1), "questions": len(answerable),
            })
    # How much the re-ranker changed each mode
    by_key = {(r["mode"], r["rerank"]): r for r in rows}
    for r in rows:
        if r["rerank"]:
            base = by_key[(r["mode"], False)]
            r["rerank_gain_hit"] = round(r["hit_at_k"] - base["hit_at_k"], 3)
            r["rerank_gain_mrr"] = round(r["mrr"] - base["mrr"], 3)
    return rows
