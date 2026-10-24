"""
evaluation/stability.py - Do the scores stay the same when the judge changes or the input is shuffled?

Takes a finished run and re-grades the SAME answers (no new generation) in two ways:
  judge swap : a different judge (another model, or the local judge)
  shuffle    : the same judge, but with the retrieved chunks in a random order
Then compares with the original grades:
  verdict agreement  share of tests with the same PASS/FAIL
  category agreement share of tests with the same root cause
  mean |difference|  of faithfulness and accuracy
  Cohen's kappa      agreement corrected for luck
Stable = agreement >= config.STABLE_AGREEMENT and mean difference <= config.STABLE_MAX_MEAN_DIFF.

Why scores CAN stay stable: relevance is plain Python; claims are split by our code (every judge
grades the same pieces); accuracy uses the fixed 0/0.5/1 rubric; evidence quotes are verified;
temperature is 0.
"""

import zlib
from concurrent.futures import ThreadPoolExecutor

import config
from evaluation.labels import _stats
from evaluation.root_cause import diagnose
from evaluation.scorers import judge_any, relevance
from utils.llm import model_label, role, usage_scope

TEST_FIELDS = ("test_id", "question", "expected_answer", "answerable", "evidence", "doc_name", "page",
               "ev_start", "ev_end")


def _test_for(report, run):
    test = {k: report.get(k) for k in TEST_FIELDS}
    if test.get("answerable") and test.get("ev_start") is None:  # older runs: positions live in run["tests"]
        original = next((t for t in run.get("tests", []) if t["test_id"] == report["test_id"]), {})
        test.update({k: original.get(k) for k in ("ev_start", "ev_end")})
    return test


def _regrade(report, run, mode, model, shuffle):
    """Grade one stored answer again. Returns None if the judge call fails (e.g. daily limit)."""
    test = _test_for(report, run)
    retrieved = report["retrieved"]
    seed = zlib.crc32(report["question"].encode())
    try:
        with usage_scope() as calls, role("judge"):
            j = judge_any(test, report["answer"], retrieved, mode=mode, model=model, shuffle=shuffle, seed=seed)
    except Exception as e:
        return {"error": str(e)}
    rel = relevance(report["gold_ids"], retrieved)
    verdict = diagnose(test, retrieved, retrieved, report["gold_ids"], j, rel)
    return {"category": verdict["category"], "faithfulness": j["faithfulness"], "accuracy": j["accuracy"],
            "tokens": sum(c["prompt_tokens"] + c["completion_tokens"] for c in calls)}


def _original_grade(report, run):
    """The run's own grades, re-classified with the current root-cause rules (no LLM call)."""
    test = _test_for(report, run)
    j = {"sentences": report["sentences"], "context_has_answer": report.get("context_has_answer", False),
         "is_refusal": report.get("is_refusal", False), "answers_question": report.get("answers_question", True),
         "accuracy": report["scores"]["accuracy"], "faithfulness": report["scores"]["faithfulness"],
         "missing_parts": report.get("missing_parts", [])}
    rel = relevance(report["gold_ids"], report["retrieved"])
    verdict = diagnose(test, report["retrieved"], report["retrieved"], report["gold_ids"], j, rel)
    return {**report, "category": verdict["category"]}


def _compare(originals, regraded):
    errors = [g["error"] for g in regraded if "error" in g]
    kept = [(o, g) for o, g in zip(originals, regraded) if "error" not in g]
    if not kept:
        return {"verdict_agreement": None, "kappa": None, "category_agreement": None, "mean_diff_faithfulness": None,
                "mean_diff_accuracy": None, "mean_diff": None, "stable": False, "changed": [], "tokens": 0,
                "compared": 0, "errors": len(errors), "error": errors[0] if errors else "nothing to compare"}
    originals, regraded = [o for o, _ in kept], [g for _, g in kept]
    pairs = [("PASS" if o["category"] == "PASS" else "FAIL", "PASS" if g["category"] == "PASS" else "FAIL")
             for o, g in zip(originals, regraded)]
    n = len(pairs) or 1
    faith = [abs(o["scores"]["faithfulness"] - g["faithfulness"]) for o, g in zip(originals, regraded)]
    acc = [abs(o["scores"]["accuracy"] - g["accuracy"]) for o, g in zip(originals, regraded)]
    stats = _stats(pairs)
    mean_diff = round((sum(faith) + sum(acc)) / (2 * n), 3)
    agreement = stats["agreement"] or 0.0
    return {
        "verdict_agreement": agreement, "kappa": stats["kappa"],
        "category_agreement": round(sum(o["category"] == g["category"] for o, g in zip(originals, regraded)) / n, 3),
        "mean_diff_faithfulness": round(sum(faith) / n, 3), "mean_diff_accuracy": round(sum(acc) / n, 3),
        "mean_diff": mean_diff,
        "stable": agreement >= config.STABLE_AGREEMENT and mean_diff <= config.STABLE_MAX_MEAN_DIFF,
        "changed": [{"test_id": o["test_id"], "question": o["question"], "before": o["category"],
                     "after": g["category"]} for o, g in zip(originals, regraded) if o["category"] != g["category"]],
        "tokens": sum(g["tokens"] for g in regraded),
        "compared": len(kept), "errors": len(errors), "error": errors[0] if errors else None,
    }


def stability_check(run, alt_judge_model=None, alt_judge_mode=None, check_shuffle=True, progress=None):
    """Re-grade a run with another judge AND (optionally) with shuffled chunks; report how stable scores are."""
    cfg = run["config"]
    reports = [_original_grade(r, run) for r in run["reports"] if r["category"] != "ERROR" and r.get("retrieved")]
    orig_mode, orig_model = cfg.get("judge_mode", "llm"), cfg["judge_model"]
    alt_mode = alt_judge_mode or orig_mode
    alt_model = alt_judge_model or orig_model
    tasks = [(r, alt_mode, alt_model, False) for r in reports]
    if check_shuffle:
        tasks += [(r, orig_mode, orig_model, True) for r in reports]
    results = [None] * len(tasks)
    done = 0
    with ThreadPoolExecutor(max_workers=config.PARALLEL_TESTS) as pool:
        futures = {pool.submit(_regrade, r, run, m, mod, sh): i for i, (r, m, mod, sh) in enumerate(tasks)}
        for future in futures:
            i = futures[future]
            results[i] = future.result()
            done += 1
            if progress:
                progress(done, len(tasks))
    swap, shuffle = results[:len(reports)], results[len(reports):]
    original_judge = "local judge" if orig_mode == "local" else model_label(orig_model)
    swapped_judge = "local judge" if alt_mode == "local" else model_label(alt_model)
    swap_result = _compare(reports, swap)
    shuffle_result = _compare(reports, shuffle) if check_shuffle else None
    return {
        "run_id": run["run_id"], "tests": len(reports),
        "original_judge": original_judge, "swapped_judge": swapped_judge,
        "judge_swap": swap_result, "shuffle": shuffle_result,
        "stable": swap_result["stable"] and (shuffle_result is None or shuffle_result["stable"]),
    }
