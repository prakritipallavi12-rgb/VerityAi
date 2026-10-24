"""
evaluation/local_judge.py - A judge that runs on your laptop: no API calls, no rate limits.

Used for big runs (e.g. 200 questions), where an LLM judge would hit free-tier token limits.
It returns exactly the same fields as the LLM judge, so the rest of the pipeline doesn't change.

How it grades:
  faithfulness   - an NLI model ("natural language inference") reads (chunk, claim) pairs and
                   gives the probability that the chunk ENTAILS (supports) the claim.
  accuracy       - numbers in the expected answer must appear in the answer (20 days vs 18 days),
                   otherwise NLI checks the answer implies the expected answer. Same 0/0.5/1 rubric.
  context_has_answer - pure position check: does a retrieved chunk overlap the answer's location?
  answers_question   - the re-ranker scores how relevant the answer is to the question (drift check).
"""

import random
import re
import threading
import unicodedata

import numpy as np

import config
from evaluation.scorers import _REFUSAL, split_sentences
from rag.generator import NOT_FOUND

_model = None
_entail_index = None
_lock = threading.Lock()

_NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
                 "eight": "8", "nine": "9", "ten": "10", "twelve": "12", "fifteen": "15", "twenty": "20",
                 "thirty": "30", "sixty": "60", "ninety": "90", "half": "0.5"}


def _get_model():
    global _model, _entail_index
    with _lock:
        if _model is None:
            from sentence_transformers import CrossEncoder
            _model = CrossEncoder(config.NLI_MODEL)
            labels = {i: l.lower() for i, l in _model.model.config.id2label.items()}
            _entail_index = next(i for i, l in labels.items() if l.startswith("entail"))
        return _model


def entailment(pairs):
    """Probability that each premise supports its hypothesis. pairs = [(premise, hypothesis), ...]"""
    if not pairs:
        return []
    model = _get_model()
    with _lock:
        logits = np.array(model.predict(pairs, show_progress_bar=False))
    if logits.ndim == 1:
        logits = logits[None, :]
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return list((exp / exp.sum(axis=1, keepdims=True))[:, _entail_index])


def numbers(text):
    text = unicodedata.normalize("NFKC", text).lower().replace(",", "")
    for word, digit in _NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\b", digit, text)
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def _lexical_support(claim, chunk_text):
    """Share of the claim's content words found in the chunk, and whether all its numbers are there too."""
    from rag.retriever import tokenize
    words = [w for w in tokenize(_clean(claim)) if not w.isdigit()]
    chunk_words = set(tokenize(_clean(chunk_text)))
    coverage = sum(w in chunk_words for w in words) / len(words) if words else 0.0
    return coverage, numbers(claim) <= numbers(chunk_text)


def _is_claim(sentence):
    """Very short sentences and refusals make no factual claim."""
    return len(re.findall(r"\w+", sentence)) >= 4 and not _REFUSAL.search(sentence)


def _plain(text):
    return " ".join(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text or "").lower()))


def _has_evidence(test, chunk):
    """Does this chunk contain the answer? Same position on the same page, or the same evidence text
    (identical policy text often appears in both an old and a new version of a document)."""
    if not test.get("answerable"):
        return False
    evidence = _plain(test.get("evidence"))
    if evidence and evidence in _plain(chunk.get("text")):
        return True
    if chunk.get("doc_name") != test.get("doc_name") or chunk.get("page") != test.get("page") \
            or chunk.get("start_char") is None or test.get("ev_start") is None:
        return False
    overlap = min(chunk["end_char"], test["ev_end"]) - max(chunk["start_char"], test["ev_start"])
    return overlap >= 0.6 * max(1, test["ev_end"] - test["ev_start"])


_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")


def _clean(text):
    """Normalise odd characters (narrow spaces, fancy dashes/minus signs) and markdown that confuse the NLI model."""
    text = unicodedata.normalize("NFKC", text).translate(_DASHES).replace("**", "").replace("`", "")
    return " ".join(text.split())


def _windows(chunk_text, size=3):
    """
    NLI models work best on short premises: slide a 3-sentence window over the chunk.
    The whole chunk is added too, for claims that combine facts from sentences further apart.
    """
    sentences = split_sentences(_clean(chunk_text))
    if len(sentences) <= size:
        return [" ".join(sentences)]
    return [" ".join(sentences[i:i + size]) for i in range(len(sentences) - size + 1)] + [" ".join(sentences)]


def local_judge(test, answer, retrieved, shuffle=False, seed=0):
    chunks = list(retrieved)
    if shuffle:
        random.Random(seed).shuffle(chunks)
    sentences = split_sentences(answer)
    is_refusal = bool(_REFUSAL.search(answer)) or answer.strip() == NOT_FOUND

    # ---- faithfulness: every (window of a chunk, claim) pair in one batch ----
    # The last premise is ALL retrieved chunks together, for claims that combine facts from two chunks.
    claims = [i for i, s in enumerate(sentences) if _is_claim(s)]
    windows = [(c, w) for c in chunks for w in _windows(c["text"])]
    if len(chunks) > 1:
        windows.append(({"chunk_id": "all"}, " ".join(_clean(c["text"]) for c in chunks)))
    pairs = [(w, _clean(sentences[i])) for i in claims for _, w in windows]
    probs = entailment(pairs)
    checked = []
    for i, text in enumerate(sentences):
        if i not in claims:
            checked.append({"text": text, "supported": True, "chunk_ids": [], "chunk_id": None, "quote": "",
                            "note": "no factual claim"})
            continue
        start = claims.index(i) * len(windows)
        scores = probs[start:start + len(windows)]
        best_per_chunk = {}
        for p, (c, w) in zip(scores, windows):  # a chunk's score = its best window
            if p > best_per_chunk.get(c["chunk_id"], (0.0, ""))[0]:
                best_per_chunk[c["chunk_id"]] = (p, w)
        combined = best_per_chunk.pop("all", (0.0, ""))[0]
        backing = sorted(((p, cid, w) for cid, (p, w) in best_per_chunk.items()
                          if p >= config.LOCAL_SUPPORT_THRESHOLD), reverse=True)
        if not backing and combined >= config.LOCAL_SUPPORT_THRESHOLD:
            # Supported only by the chunks together: credit the two chunks that support it most
            backing = sorted(((p, cid, w) for cid, (p, w) in best_per_chunk.items()), reverse=True)[:2]
        how = "NLI"
        if not backing:
            # Word-overlap backstop: answers are usually near-copies of the source. NLI is very literal
            # (e.g. it rejects "hours AT NIMBUS are..." if "Nimbus" sits in another sentence of the chunk).
            for c in chunks:
                coverage, numbers_ok = _lexical_support(text, c["text"])
                if coverage >= 0.85 and numbers_ok:
                    backing.append((coverage, c["chunk_id"], c["text"][:200]))
            backing.sort(reverse=True)
            how = "word overlap"
        # Number guard: a number that appears in no retrieved chunk is never supported (catches 18 vs 20 days)
        stray = numbers(text) - set().union(*(numbers(c["text"]) for c in chunks)) if chunks else numbers(text)
        if backing and stray:
            backing, how = [], f"number {', '.join(sorted(stray))} is not in any chunk"
        best = max(scores) if len(scores) else 0.0
        checked.append({"text": text, "supported": bool(backing), "chunk_ids": [cid for _, cid, _ in backing],
                        "chunk_id": backing[0][1] if backing else None,
                        "quote": backing[0][2] if backing else "",
                        "note": f"local judge ({how}), NLI support {best:.2f}"})

    if is_refusal:
        faithfulness = 1.0
    else:
        faithfulness = sum(s["supported"] for s in checked) / len(checked) if checked else 0.0

    # ---- accuracy: numbers first, then meaning ----
    expected = test["expected_answer"]
    if not test["answerable"]:
        accuracy = 1.0 if is_refusal else 0.0
    elif is_refusal:
        accuracy = 0.0
    else:
        wanted = numbers(expected)
        if wanted:
            found = wanted & numbers(answer)
            accuracy = 1.0 if found == wanted else 0.5 if found else 0.0
        else:
            # Short answers ("Basic support", "Pune"): every content word of the expected answer is in the answer
            coverage, _ = _lexical_support(expected, answer)
            if coverage >= 0.99:
                accuracy = 1.0
            else:
                hypothesis = expected if len(expected.split()) >= 4 else f"The answer to '{test['question']}' is {expected}."
                p = entailment([(_clean(answer), _clean(hypothesis))])[0]
                accuracy = 1.0 if p >= 0.5 else 0.5 if p >= 0.2 or coverage >= 0.5 else 0.0

    # ---- does the answer address the question? (drift) ----
    answers_question = True
    if not is_refusal and answer.strip():
        from rag.reranker import relevance_scores
        answers_question = relevance_scores(test["question"], [answer])[0] >= config.LOCAL_DRIFT_THRESHOLD

    return {
        "sentences": checked,
        "context_has_answer": any(_has_evidence(test, c) for c in chunks),
        "is_refusal": is_refusal,
        "answers_question": answers_question,
        "accuracy": accuracy,
        "faithfulness": round(faithfulness, 2),
        "missing_parts": [],
        "reasoning": "Graded locally: NLI entailment for each claim, number match for accuracy.",
        "judge_order": [c["chunk_id"] for c in chunks],
    }
