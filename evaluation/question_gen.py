"""
evaluation/question_gen.py - STAGE 4: Create test questions with known answers.

Two ways to get a test:
  generate_questions() - the LLM reads a chunk and writes {question, answer, source_chunk_id},
                         so we know exactly where each answer lives. Some questions are made
                         unanswerable on purpose (answer NOT in the documents).
                         Big requests (e.g. 200 questions) are split into batches that run in parallel.
  build_custom_test()  - the user types a question; the LLM finds which chunk answers it (or none).

A test keeps its source_chunk_id AND its evidence position in the page (doc, page, character span).
The position matters when you move the chunk-size slider: chunk IDs change, but we can still tell
which new chunks hold the answer (gold_chunk_ids).
"""

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import config
from rag.chunker import parse_chunk_id
from utils.llm import ask_json

BATCH_PASSAGES = 8  # passages per LLM call when generating many questions


def locate_evidence(quote, page_text):
    """Find a quote inside the page text, ignoring line breaks/extra spaces. Returns (start, end) or None."""
    words = quote.split()
    if not words:
        return None
    match = re.search(r"\s+".join(re.escape(w) for w in words), page_text, flags=re.IGNORECASE)
    return (match.start(), match.end()) if match else None


def _page_text(pages, doc_name, page):
    return next((p["text"] for p in pages if p["doc_name"] == doc_name and p["page"] == page), "")


def _make_test(test_id, question, answer, evidence, chunk, pages, origin):
    """Build a test record, pinning the evidence to an exact position on its page."""
    text = _page_text(pages, chunk["doc_name"], chunk["page"])
    span = locate_evidence(evidence, text) if evidence else None
    if span is None:  # the LLM paraphrased instead of copying - fall back to the whole chunk
        span = (chunk["start_char"], chunk["end_char"])
        evidence = chunk["text"]
    return {
        "test_id": test_id, "question": question.strip(), "expected_answer": answer.strip(),
        "answerable": True, "source_chunk_id": chunk["chunk_id"], "evidence": evidence,
        "doc_name": chunk["doc_name"], "page": chunk["page"], "ev_start": span[0], "ev_end": span[1],
        "origin": origin,
    }


def _unanswerable_test(test_id, question, origin):
    return {"test_id": test_id, "question": question.strip(), "expected_answer": "Not in the documents.",
            "answerable": False, "source_chunk_id": None, "evidence": "", "doc_name": None, "page": None,
            "ev_start": None, "ev_end": None, "origin": origin}


def _generate_batch(batch, n_unanswerable, model):
    """One LLM call: questions for a few passages (+ some unanswerable ones)."""
    passages = "\n\n".join(f"PASSAGE {i} (write {count} question{'s' if count > 1 else ''}):\n{c['text']}"
                           for i, (c, count) in enumerate(batch))
    total = sum(count for _, count in batch)
    prompt = f"""Create test questions for a question-answering system over company documents.

{passages}

Write exactly {total} answerable questions, as many per passage as shown. Each question must be answerable
from its passage alone, and questions about the same passage must ask about DIFFERENT facts.
- Ask about one specific fact (a number, rule, condition or name).
- "answer": short and complete, using the passage's facts.
- "evidence": ONE sentence copied word-for-word from the passage that contains the answer.

Also write exactly {n_unanswerable} unanswerable questions: realistic questions about the same company
whose answer is NOT in any passage.

Return JSON:
{{"answerable": [{{"passage": 0, "question": "...", "answer": "...", "evidence": "..."}}],
  "unanswerable": [{{"question": "..."}}]}}"""
    return ask_json(prompt, model=model, required_keys=("answerable", "unanswerable"))


def generate_questions(chunks, pages, n=None, model=None, unanswerable_ratio=None, store=None, progress=None):
    """
    Write n test questions with known answers from the current (newest) documents.
    If a store is given, every "unanswerable" question is double-checked against the documents:
    if an answer does exist, it becomes a normal test instead of a wrongly labelled one.
    progress(done, total) is called as batches finish.
    """
    n = n or config.NUM_TEST_QUESTIONS
    ratio = config.UNANSWERABLE_RATIO if unanswerable_ratio is None else unanswerable_ratio
    n_unanswerable = round(n * ratio) if n >= 4 else 0
    n_answerable = n - n_unanswerable

    # Only ask about the current document versions, and skip tiny leftover chunks
    pool = [c for c in chunks if c.get("is_current", True) and c["word_count"] >= 25] or chunks
    # Spread the questions evenly over the documents; a chunk may get several questions
    picks = [pool[int(i * len(pool) / n_answerable) % len(pool)] for i in range(n_answerable)] if n_answerable else []
    counts = Counter(c["chunk_id"] for c in picks)
    by_id = {c["chunk_id"]: c for c in pool}
    groups = [(by_id[cid], count) for cid, count in counts.items()]
    batches = [groups[i:i + BATCH_PASSAGES] for i in range(0, len(groups), BATCH_PASSAGES)] or [[]]
    unanswerable_per_batch = [n_unanswerable // len(batches) + (i < n_unanswerable % len(batches))
                              for i in range(len(batches))]

    done = 0
    results = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=min(4, len(batches))) as pool_exec:
        futures = {pool_exec.submit(_generate_batch, b, u, model): i
                   for i, (b, u) in enumerate(zip(batches, unanswerable_per_batch))}
        for future, i in futures.items():
            results[i] = future.result()
            done += 1
            if progress:
                progress(done, len(batches) + (1 if n_unanswerable else 0))

    tests, seen = [], set()

    def add(test):
        key = re.sub(r"\W+", " ", test["question"].lower()).strip()
        if key and key not in seen:  # drop duplicate questions
            seen.add(key)
            tests.append(test)

    unanswerable_questions = []
    for batch, data in zip(batches, results):
        for item in data.get("answerable", []):
            try:
                chunk = batch[int(item.get("passage", 0))][0]
            except (ValueError, IndexError, TypeError):
                continue
            add(_make_test("", item.get("question", ""), item.get("answer", ""), item.get("evidence", ""),
                           chunk, pages, "generated"))
        unanswerable_questions += [item.get("question", "") for item in data.get("unanswerable", [])]

    unanswerable_questions = [q for q in unanswerable_questions if q.strip()][:n_unanswerable]
    if store is not None and unanswerable_questions:
        with ThreadPoolExecutor(max_workers=4) as pool_exec:
            checked = list(pool_exec.map(lambda q: build_custom_test(q, store, pages, "", model=model),
                                         unanswerable_questions))
        for test in checked:
            test["origin"] = "generated"
            add(test)
    else:
        for q in unanswerable_questions:
            add(_unanswerable_test("", q, "generated"))
    if progress:
        progress(1, 1)

    for i, test in enumerate(tests, start=1):
        test["test_id"] = f"T{i}"
    return tests


def build_custom_test(question, store, pages, test_id, model=None):
    """Turn a user's question into a test: find which chunk answers it, or decide that none does."""
    candidates = store.rank(question, mode="hybrid")[:8]
    passages = "\n\n".join(
        f"[Chunk #{c['chunk_id']} | {c['doc_name']} | {'current' if c.get('is_current', True) else 'ARCHIVED (outdated)'}]\n{c['text']}"
        for c in candidates)
    prompt = f"""QUESTION: {question}

PASSAGES:
{passages}

Does any passage answer the question? If current and archived passages disagree, the CURRENT one is correct.
Return JSON: {{"answerable": true/false, "chunk_id": <id of the passage with the answer or null>,
"answer": "short correct answer", "evidence": "one sentence copied word-for-word from that passage"}}"""

    data = ask_json(prompt, model=model, required_keys=("answerable",))
    by_id = {c["chunk_id"]: c for c in candidates}
    chunk = by_id.get(parse_chunk_id(data.get("chunk_id")))
    if not data.get("answerable") or chunk is None:
        return _unanswerable_test(test_id, question, "custom")
    return _make_test(test_id, question, data.get("answer", ""), data.get("evidence", ""),
                      chunk, pages, "custom")


def gold_chunk_ids(test, chunks):
    """
    Which chunks (under the CURRENT chunk settings) contain this test's evidence.
    With the same chunk settings as when the test was made, this includes source_chunk_id.
    """
    if not test.get("answerable"):
        return []
    start, end = test["ev_start"], test["ev_end"]
    length = max(1, end - start)
    overlaps = []
    for c in chunks:
        if c["doc_name"] == test["doc_name"] and c["page"] == test["page"]:
            overlap = min(c["end_char"], end) - max(c["start_char"], start)
            if overlap > 0:
                overlaps.append((overlap, c["chunk_id"]))
    gold = [cid for overlap, cid in overlaps if overlap >= 0.6 * length]
    # Evidence split across two chunks with no single chunk holding most of it: take the best one
    return gold or ([max(overlaps)[1]] if overlaps else [])
