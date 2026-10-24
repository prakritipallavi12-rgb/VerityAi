"""
evaluation/scorers.py - STAGE 5: Grade the retriever and the generator separately.

1. context relevance - Is the source chunk (the one holding the answer) in the top-k? Plain
                       Python, no LLM. Also gives the reciprocal rank (1, 1/2, 1/3...).
2. faithfulness      - Share of answer CLAIMS backed by the retrieved chunks. Our code splits the
                       answer into claims (one per sentence) - not the judge - so every judge
                       model grades exactly the same pieces. The judge marks each claim supported
                       or not and quotes its evidence; we verify every quote really exists.
                       Unsupported claims are what the dashboard highlights in red.
3. answer accuracy   - Does the answer match the expected answer? Fixed rubric: 1 = correct,
                       0.5 = partly correct, 0 = wrong. Fixed steps keep scores stable across judges.

The judge also reports: is the answer a refusal ("not found")? does it actually answer
the question that was asked (drift)? did the context contain the answer at all?

Shuffle check: the judge can be run a second time with the chunks in random order.
A fair judge gives the same score - if not, it is biased by chunk position.
"""

import random
import re

import config
from rag.chunker import parse_chunk_id
from rag.generator import NOT_FOUND
from utils.llm import ask_json

_REFUSAL = re.compile(r"not found in the documents|i don't know|i do not know|cannot find|"
                      r"can't find|no information|not mentioned|no mention|isn't mentioned|is not mentioned|"
                      r"does not (?:contain|mention|specify)|"
                      r"doesn't (?:contain|mention|specify)|unable to (?:find|answer)", re.IGNORECASE)


def split_sentences(answer):
    parts = re.split(r"(?<=[.!?])\s+|\n+", answer.strip())
    return [p.strip(" -•*") for p in parts if p.strip(" -•*")]


def relevance(gold_ids, retrieved):
    """1.0 if a chunk holding the answer was retrieved, else 0.0. None when there is nothing to find."""
    if not gold_ids:
        return None
    return 1.0 if any(c["chunk_id"] in gold_ids for c in retrieved) else 0.0


def reciprocal_rank(gold_ids, ranking):
    """1/rank of the first chunk holding the answer (1.0 = ranked first). None when nothing to find."""
    if not gold_ids:
        return None
    rank = next((c["rank"] for c in ranking if c["chunk_id"] in gold_ids), None)
    return round(1 / rank, 3) if rank else 0.0


def snap_accuracy(value):
    """Round to the rubric steps 0 / 0.5 / 1 so different judges land on the same values."""
    try:
        value = max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
    return 1.0 if value >= 0.75 else 0.5 if value >= 0.25 else 0.0


def _normalise(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _quote_in_chunk(quote, chunk_text):
    """True if the quote is really in the chunk (exactly, or >= 80% of its words)."""
    q, c = _normalise(quote or ""), _normalise(chunk_text)
    if not q:
        return False
    if q in c:
        return True
    words = q.split()
    chunk_words = set(c.split())
    return sum(w in chunk_words for w in words) / len(words) >= 0.8


def judge(test, answer, retrieved, model=None, shuffle=False, seed=0):
    """Ask the judge LLM to grade one answer. Returns a dict of verified results + scores."""
    chunks = list(retrieved)
    if shuffle:
        random.Random(seed).shuffle(chunks)  # same seed = same shuffle, so runs are repeatable

    sentences = split_sentences(answer)
    reference = test["expected_answer"] if test["answerable"] else "The documents do NOT contain the answer."
    context = "\n\n".join(f"[Chunk #{c['chunk_id']}]\n{c['text']}" for c in chunks)
    numbered = "\n".join(f"S{i + 1}: {s}" for i, s in enumerate(sentences)) or "(empty answer)"

    prompt = f"""You are a strict, fair evaluator of a retrieval-augmented QA system.

QUESTION: {test['question']}
REFERENCE ANSWER (ground truth): {reference}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER, split into claims:
{numbered}

Evaluate:
1. context_has_answer: does the RETRIEVED CONTEXT contain the facts of the reference answer?
2. For EVERY claim S1..S{len(sentences)}: "supported" = true only if every fact in it is stated in the context.
   A claim with no facts (e.g. "Hope this helps") or saying the answer was not found counts as supported.
   chunk_ids = IDs of ALL chunks its facts come from; quote = short exact words from the first one.
3. is_refusal: does the answer say it cannot find / does not know the answer?
4. answers_question: does the answer address what the question asks (ignore whether facts are right)?
5. accuracy (rubric): 1 = matches the reference answer, 0.5 = partly correct or incomplete, 0 = wrong or missing.
6. missing_parts: reference-answer facts the generated answer leaves out.
7. reasoning: one short sentence.

Return JSON:
{{"context_has_answer": true, "sentences": [{{"id": "S1", "supported": true, "chunk_ids": [4], "quote": "..."}}],
  "is_refusal": false, "answers_question": true, "accuracy": 1, "missing_parts": [], "reasoning": "..."}}"""

    data = ask_json(prompt, model=model, max_tokens=1200, required_keys=(
        "context_has_answer", "sentences", "is_refusal", "answers_question", "accuracy"))

    # --- verify the judge's evidence instead of trusting it ---
    by_id = {c["chunk_id"]: c for c in chunks}
    judged = {str(s.get("id", "")).upper(): s for s in data.get("sentences", []) if isinstance(s, dict)}
    checked = []
    for i, text in enumerate(sentences):
        s = judged.get(f"S{i + 1}", {})
        supported = bool(s.get("supported"))
        raw_ids = s.get("chunk_ids") if isinstance(s.get("chunk_ids"), list) else [s.get("chunk_id")]
        chunk_ids = [cid for cid in (parse_chunk_id(x) for x in raw_ids) if cid in by_id]
        quote = s.get("quote") or ""
        note = ""
        if supported and chunk_ids and quote and not any(_quote_in_chunk(quote, by_id[c]["text"]) for c in chunk_ids):
            supported, note = False, "judge's evidence quote was not found in the chunk"
        if supported and not chunk_ids and not _REFUSAL.search(text):
            # No valid chunk given: accept only if the sentence's own words are in some chunk
            match = next((c for c in chunks if _quote_in_chunk(text, c["text"])), None)
            chunk_ids = [match["chunk_id"]] if match else []
            if match is None and len(_normalise(text).split()) > 4:
                supported, note = False, "no supporting chunk found"
        checked.append({"text": text, "supported": supported,
                        "chunk_ids": chunk_ids if supported else [],
                        "chunk_id": chunk_ids[0] if supported and chunk_ids else None,
                        "quote": quote if supported else "", "note": note})

    is_refusal = bool(data.get("is_refusal")) or bool(_REFUSAL.search(answer)) or answer.strip() == NOT_FOUND
    if is_refusal:
        faithfulness = 1.0  # a refusal makes no claims, so it cannot be unfaithful
    else:
        faithfulness = sum(s["supported"] for s in checked) / len(checked) if checked else 0.0

    accuracy = snap_accuracy(data.get("accuracy", 0))
    if not test["answerable"]:
        accuracy = 1.0 if is_refusal else 0.0  # the only correct answer is "not found"

    return {
        "sentences": checked,
        "context_has_answer": bool(data.get("context_has_answer")),
        "is_refusal": is_refusal,
        "answers_question": bool(data.get("answers_question", True)),
        "accuracy": round(accuracy, 2),
        "faithfulness": round(faithfulness, 2),
        "missing_parts": data.get("missing_parts") or [],
        "reasoning": data.get("reasoning", ""),
        "judge_order": [c["chunk_id"] for c in chunks],
    }


def judge_any(test, answer, retrieved, mode="llm", model=None, shuffle=False, seed=0):
    """Pick the judge: an LLM, or the local models (no API calls, for big runs)."""
    if mode == "local":
        from evaluation.local_judge import local_judge
        return local_judge(test, answer, retrieved, shuffle=shuffle, seed=seed)
    return judge(test, answer, retrieved, model=model, shuffle=shuffle, seed=seed)


def overall(scores):
    """One number per test: the average of the scores that apply."""
    values = [v for v in (scores.get("relevance"), scores.get("faithfulness"), scores.get("accuracy")) if v is not None]
    return round(sum(values) / len(values), 2) if values else 0.0


def stability(original, shuffled):
    """Compare the judge's scores before and after shuffling the chunks."""
    diff = max(abs(original["faithfulness"] - shuffled["faithfulness"]),
               abs(original["accuracy"] - shuffled["accuracy"]))
    return {"checked": True, "unchanged": diff <= 0.05, "max_difference": round(diff, 2),
            "shuffled_order": shuffled["judge_order"],
            "shuffled_scores": {"faithfulness": shuffled["faithfulness"], "accuracy": shuffled["accuracy"]}}


def score_color(value):
    if value is None:
        return "neutral"
    return "good" if value >= config.SCORE_GOOD else "ok" if value >= config.SCORE_OK else "bad"
