"""
evaluation/root_cause.py - STAGE 6: Decide WHY a test failed and WHERE the evidence is.

Two separate verdicts per test (they form the rows and columns of the confusion matrix):
  retrieval  : did the retriever find the chunk with the answer?  (Right chunk / Wrong chunk)
  generation : what did the generator do with the chunks it got?
               Faithful, Hallucination, Ignored context, Stale source, Refusal,
               Generation drift, Incomplete

Final category = the ROOT cause:
  PASS             right chunk found, answer correct and fully supported
  STALE_SOURCE     the answer came from an older (archived) version of a document
  WRONG_CHUNK      the answer is in the documents, but the retriever didn't fetch it
  REFUSAL          the model said "not found", but the answer WAS in its chunks
  HALLUCINATION    the answer contains sentences no retrieved chunk supports
  GENERATION_DRIFT the answer is supported by the chunks, but doesn't answer the question asked
  IGNORED_CONTEXT  the right chunk was retrieved, but the model answered from a different one
  INCOMPLETE       right chunk, true answer, but parts are missing

All checking happens here; the frontend only displays the result.
"""

import config
from rag.chunker import format_location

CATEGORY_LABELS = {
    "PASS": "Pass", "STALE_SOURCE": "Stale source", "WRONG_CHUNK": "Wrong chunk", "REFUSAL": "Refusal",
    "HALLUCINATION": "Hallucination", "GENERATION_DRIFT": "Generation drift",
    "IGNORED_CONTEXT": "Ignored context", "INCOMPLETE": "Incomplete", "ERROR": "Error",
}
GENERATION_LABELS = ["Faithful", "Hallucination", "Ignored context", "Stale source",
                     "Refusal", "Generation drift", "Incomplete"]


def _short(text, n=160):
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def used_chunks(j, retrieved):
    """Every retrieved chunk that at least one supported answer sentence relies on."""
    ids = {cid for s in j["sentences"] if s["supported"] for cid in s.get("chunk_ids") or [s.get("chunk_id")]}
    return [c for c in retrieved if c["chunk_id"] in ids]


def generation_verdict(test, j, retrieved, gold_ids):
    """What the generator did, judged only against the chunks it was given."""
    used = used_chunks(j, retrieved)
    unsupported = [s for s in j["sentences"] if not s["supported"]]
    correct = j["accuracy"] >= config.ACCURACY_PASS_THRESHOLD

    if not test["answerable"]:
        return "Faithful" if j["is_refusal"] else "Hallucination"
    if j["is_refusal"]:
        # Refusing is honest when the chunks really lacked the answer
        return "Refusal" if j["context_has_answer"] else "Faithful"
    if not correct and any(not c.get("is_current", True) for c in used):
        return "Stale source"
    if unsupported:
        return "Hallucination"
    if correct and j["answers_question"]:
        return "Faithful"
    if not j["answers_question"]:
        return "Generation drift"
    if j["context_has_answer"] and not any(c["chunk_id"] in gold_ids for c in used):
        return "Ignored context"
    return "Incomplete" if j["context_has_answer"] else "Faithful"


def diagnose(test, ranking, retrieved, gold_ids, j, rel):
    """Return {category, retrieval, generation, explanation, ...locations} for one test."""
    # Fixed rule, not the judge's opinion: if the answer's chunk was retrieved, the context has the answer.
    # (Keeps verdicts identical across judge models.)
    j = {**j, "context_has_answer": j["context_has_answer"] or rel == 1.0}
    gen = generation_verdict(test, j, retrieved, gold_ids)
    retrieval_ok = rel is None or rel == 1.0 or j["context_has_answer"]
    by_rank = {c["chunk_id"]: c for c in ranking}
    gold = [by_rank[g] for g in gold_ids if g in by_rank]
    best_gold = min(gold, key=lambda c: c["rank"]) if gold else None
    gold_loc = format_location(best_gold) if best_gold else "-"
    unsupported = [s["text"] for s in j["sentences"] if not s["supported"]]
    top = ranking[0] if ranking else None

    retrieval = "Not in docs" if not test["answerable"] else "Right chunk" if retrieval_ok else "Wrong chunk"
    info = {"retrieval": retrieval, "generation": gen,
            "gold_location": gold_loc, "gold_rank": best_gold["rank"] if best_gold else None,
            "closest_chunk": {"location": format_location(top), "score": top["score"]} if top else None}

    # ---- unanswerable questions: the only correct answer is a refusal ----
    if not test["answerable"]:
        if j["is_refusal"]:
            return {**info, "category": "PASS",
                    "explanation": "Correctly said the answer is not in the documents."}
        return {**info, "category": "HALLUCINATION",
                "explanation": (f"The answer is not present in any chunk, but the model answered anyway. "
                                f"Closest chunk: {info['closest_chunk']['location']} "
                                f"(similarity {info['closest_chunk']['score']}).")}

    passed = (retrieval_ok and gen == "Faithful" and not j["is_refusal"]
              and j["faithfulness"] >= config.FAITHFULNESS_PASS_THRESHOLD
              and j["accuracy"] >= config.ACCURACY_PASS_THRESHOLD)
    if passed:
        return {**info, "category": "PASS", "explanation": f"Correct, and supported by {gold_loc}."}

    evidence = f'"{_short(test["evidence"])}"'
    retrieved_ids = ", ".join(f"#{c['chunk_id']}" for c in retrieved)

    if gen == "Stale source":
        stale = next(c for c in used_chunks(j, retrieved) if not c.get("is_current", True))
        return {**info, "category": "STALE_SOURCE", "stale_location": format_location(stale),
                "explanation": (f"The model answered from {format_location(stale)}, an archived "
                                f"{stale.get('doc_year') or 'older'} version. The current answer is in "
                                f"{gold_loc}: {evidence}")}
    if not retrieval_ok:
        rank_text = f"ranked #{best_gold['rank']} of {len(ranking)}" if best_gold else "not ranked"
        closest = retrieved[0] if retrieved else top
        then = "honestly said it was not found" if j["is_refusal"] else gen
        return {**info, "category": "WRONG_CHUNK",
                "explanation": (f"The answer is in {gold_loc}, but it was {rank_text}. The retriever returned "
                                f"{retrieved_ids} instead. Closest retrieved chunk: {format_location(closest)} "
                                f"(score {closest['score']}). The generator then {then.lower() if then != gen else 'gave: ' + gen}.")}
    if gen == "Refusal":
        return {**info, "category": "REFUSAL",
                "explanation": (f"The model said it could not find the answer, but it was in {gold_loc} "
                                f"(retrieved at rank #{best_gold['rank'] if best_gold else '?'}): {evidence}")}
    if gen == "Hallucination":
        made_up = "; ".join(f'"{_short(s, 120)}"' for s in unsupported)
        return {**info, "category": "HALLUCINATION",
                "explanation": (f"{len(unsupported)} sentence(s) are not supported by any retrieved chunk: "
                                f"{made_up}. The correct answer was in {gold_loc}: {evidence}")}
    if gen == "Generation drift":
        return {**info, "category": "GENERATION_DRIFT",
                "explanation": (f"The answer is backed by the chunks but does not answer the question. "
                                f"Expected: \"{_short(test['expected_answer'])}\" (from {gold_loc}).")}
    if gen == "Ignored context":
        used = next((format_location(c) for c in used_chunks(j, retrieved)
                     if c["chunk_id"] not in gold_ids), "another chunk")
        return {**info, "category": "IGNORED_CONTEXT",
                "explanation": (f"The right chunk {gold_loc} was retrieved, but the model answered from "
                                f"{used} instead. The answer was: {evidence}")}
    missing = "; ".join(map(str, j["missing_parts"])) or "part of the expected answer"
    return {**info, "category": "INCOMPLETE",
            "explanation": f"Partly correct. Missing: {missing}. Found in {gold_loc}: {evidence}"}
