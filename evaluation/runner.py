"""
evaluation/runner.py - Runs the whole pipeline for every test question.

For each test:  retrieve -> generate -> score (judge) -> classify (root cause)
Each step is timed, and every LLM call's tokens and cost are recorded (cost & time tracker).
Tests run in parallel threads (config.PARALLEL_TESTS); the rate limiter in utils/llm.py makes
sure the parallel calls never break the provider's limits.
Several RAG configurations (A/B, model leaderboard) run over the SAME tests at the same time.
Every run is stored permanently in the SQLite database (storage/db.py).
"""

import time
import uuid
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import config
from evaluation.question_gen import gold_chunk_ids
from evaluation.root_cause import CATEGORY_LABELS, diagnose
from evaluation.scorers import judge_any, overall, reciprocal_rank, relevance, stability
from rag.generator import generate_answer
from storage import db
from utils.llm import model_label, role, usage_scope

TEST_FIELDS = ("test_id", "question", "expected_answer", "answerable", "source_chunk_id", "evidence",
               "doc_name", "page", "ev_start", "ev_end", "origin")
CHUNK_FIELDS = ("chunk_id", "doc_name", "page", "doc_year", "is_current", "start_char", "end_char",
                "rank", "score", "text")


def make_config(version, generator_model, judge_model, chunk_size, overlap, top_k, shuffle_check,
                judge_mode=None):
    v = config.RAG_VERSIONS[version]
    return {"version": version, "retriever": v["retriever"], "rerank": v.get("rerank", False),
            "prompt": v["prompt"], "generator_model": generator_model, "judge_model": judge_model,
            "judge_mode": judge_mode or config.DEFAULT_JUDGE_MODE,
            "chunk_size": chunk_size, "overlap": overlap, "top_k": top_k, "shuffle_check": shuffle_check}


def _usage_summary(calls, seconds):
    by_role = {}
    for c in calls:
        r = by_role.setdefault(c["role"], {"tokens": 0, "cost": 0.0, "calls": 0, "seconds": 0.0})
        r["tokens"] += c["prompt_tokens"] + c["completion_tokens"]
        r["cost"] += c["cost"]
        r["calls"] += 1
        r["seconds"] += c["seconds"]
    return {"tokens": sum(r["tokens"] for r in by_role.values()),
            "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
            "completion_tokens": sum(c["completion_tokens"] for c in calls),
            "cost": round(sum(r["cost"] for r in by_role.values()), 6),
            "llm_calls": len(calls), "by_role": by_role, **seconds,
            "seconds": round(sum(seconds.values()), 2)}


def evaluate_test(test, store, cfg):
    """Run one test through the RAG system and the evaluator. Never raises - errors become a report."""
    timings = {"retrieve_s": 0.0, "generate_s": 0.0, "judge_s": 0.0}
    with usage_scope() as calls:
        try:
            t = time.time()
            ranking = store.rank(test["question"], mode=cfg["retriever"], rerank=cfg.get("rerank", False))
            retrieved = ranking[:cfg["top_k"]]
            gold_ids = gold_chunk_ids(test, store.chunks)
            timings["retrieve_s"] = round(time.time() - t, 3)

            t = time.time()
            with role("generator"):
                answer = generate_answer(test["question"], retrieved, cfg["prompt"], model=cfg["generator_model"])
            timings["generate_s"] = round(time.time() - t, 3)

            t = time.time()
            mode = cfg.get("judge_mode", "llm")
            with role("judge"):
                j = judge_any(test, answer, retrieved, mode=mode, model=cfg["judge_model"])
                stab = {"checked": False}
                if cfg["shuffle_check"]:
                    shuffled = judge_any(test, answer, retrieved, mode=mode, model=cfg["judge_model"], shuffle=True,
                                         seed=zlib.crc32(test["question"].encode()))  # same question -> same shuffle
                    stab = stability(j, shuffled)
            timings["judge_s"] = round(time.time() - t, 3)

            rel = relevance(gold_ids, retrieved)
            verdict = diagnose(test, ranking, retrieved, gold_ids, j, rel)
            scores = {"relevance": rel, "faithfulness": j["faithfulness"], "accuracy": j["accuracy"]}
            return {
                **{k: test.get(k) for k in TEST_FIELDS},
                "answer": answer,
                "retrieved": [{k: c.get(k) for k in CHUNK_FIELDS} | {"is_gold": c["chunk_id"] in gold_ids}
                              for c in retrieved],
                "gold_ids": gold_ids, "reciprocal_rank": reciprocal_rank(gold_ids, ranking),
                "scores": scores, "overall": overall(scores),
                "sentences": j["sentences"], "is_refusal": j["is_refusal"],
                "answers_question": j["answers_question"], "context_has_answer": j["context_has_answer"],
                "missing_parts": j["missing_parts"], "judge_reasoning": j["reasoning"],
                "stability": stab, "usage": _usage_summary(calls, timings), **verdict,
            }
        except Exception as e:
            return {**{k: test.get(k) for k in TEST_FIELDS},
                    "answer": "", "retrieved": [], "gold_ids": [], "scores": {}, "overall": 0.0, "sentences": [],
                    "category": "ERROR", "retrieval": "-", "generation": "-",
                    "explanation": f"This test could not run: {e}", "stability": {"checked": False},
                    "usage": _usage_summary(calls, timings)}


def run_jobs(jobs, tests, progress=None, workers=None):
    """
    jobs = [(name, store, cfg), ...]  - one job per RAG configuration (1 normal, 2 for A/B, more for a leaderboard)
    Every test of every job goes into one thread pool, so all configurations really run concurrently.
    progress(done, total) is called from this thread, so callers can show "Evaluating 4/10...".
    Returns ({name: [reports in test order]}, wall_clock_seconds).
    """
    results = {name: [None] * len(tests) for name, _, _ in jobs}
    total, done = len(jobs) * len(tests), 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers or config.PARALLEL_TESTS) as pool:
        futures = {pool.submit(evaluate_test, t, store, cfg): (name, i)
                   for name, store, cfg in jobs for i, t in enumerate(tests)}
        for future in as_completed(futures):
            name, i = futures[future]
            results[name][i] = future.result()
            done += 1
            if progress:
                progress(done, total)
    return results, round(time.time() - start, 1)


def summarize(reports, wall_seconds=None):
    counts = Counter(r["category"] for r in reports)
    scored = [r for r in reports if r["category"] != "ERROR"]

    def avg(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 3) if values else None

    checked = [r for r in scored if r.get("stability", {}).get("checked")]
    usage = [r.get("usage", {}) for r in reports]
    tokens = sum(u.get("tokens", 0) for u in usage)
    hallucinations = sum(r.get("generation") == "Hallucination" for r in scored)
    return {
        "total": len(reports), "passed": counts.get("PASS", 0),
        "failed": len(reports) - counts.get("PASS", 0) - counts.get("ERROR", 0),
        "errors": counts.get("ERROR", 0),
        "pass_rate": round(counts.get("PASS", 0) / len(scored), 3) if scored else 0.0,
        "categories": {CATEGORY_LABELS[k]: v for k, v in counts.items()},
        "relevance": avg(r["scores"].get("relevance") for r in scored),
        "faithfulness": avg(r["scores"].get("faithfulness") for r in scored),
        "accuracy": avg(r["scores"].get("accuracy") for r in scored),
        "mrr": avg(r.get("reciprocal_rank") for r in scored),
        "hallucination_rate": round(hallucinations / len(scored), 3) if scored else 0.0,
        "stability_checked": len(checked),
        "stability_unchanged": sum(r["stability"]["unchanged"] for r in checked),
        # cost & time tracker
        "tokens": tokens, "tokens_per_question": round(tokens / len(reports)) if reports else 0,
        "cost": round(sum(u.get("cost", 0) for u in usage), 5),
        "seconds_per_question": avg(u.get("seconds") for u in usage),
        "generate_s": avg(u.get("generate_s") for u in usage), "judge_s": avg(u.get("judge_s") for u in usage),
        "retrieve_s": avg(u.get("retrieve_s") for u in usage),
        "wall_seconds": wall_seconds,
        "questions_per_minute": round(60 * len(reports) / wall_seconds, 1) if wall_seconds else None,
    }


def make_run(reports, cfg, tests, label="", wall_seconds=None):
    return {
        "run_id": datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": label or f"{cfg['version']} · {model_label(cfg['generator_model'])}",
        "config": cfg, "tests": tests, "reports": reports, "summary": summarize(reports, wall_seconds),
    }


def save_run(run, kind="eval"):
    db.save_run(run, kind)


def list_runs():
    return db.list_runs()


def get_run(run_id):
    return db.get_run(run_id)
