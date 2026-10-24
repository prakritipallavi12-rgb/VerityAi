"""
benchmark.py - Can VerityAi evaluate 200 questions in 15 minutes? Run it and see.

    python benchmark.py                       (200 questions, local judge, product catalog)
    python benchmark.py --questions 50 --judge llm

Steps: load the benchmark catalog -> generate the questions (not timed: this is test preparation)
       -> evaluate every question (TIMED: retrieve -> generate -> judge -> classify)
       -> save the run to the database (it shows up in the dashboard's Run history).
"""

import argparse
import re
import time

import config
from evaluation.question_gen import generate_questions
from evaluation.runner import make_config, make_run, run_jobs, save_run
from rag.chunker import chunk_pages
from rag.loader import load_documents
from rag.retriever import VectorStore
from storage import db

TARGET_SECONDS = 15 * 60

# The catalog is written from one template, so its facts can be turned into exact questions (no LLM tokens needed)
_PRODUCT = re.compile(
    r"(?P<name>\w+) is a (?P<kind>[\w ]+?) developed by the Nimbus team in (?P<city>\w+)\. It was first launched in "
    r"(?P<year>\d{4})\. A \w+ licence costs Rs (?P<price>[\d,]+) per user per month and supports up to (?P<users>\d+) "
    r"users per account\. Every \w+ subscription includes a (?P<warranty>\d+)-month service warranty, and customers "
    r"on \w+ receive (?P<tier>\w+) support\.")
_QUESTIONS = [
    ("How much does a {name} licence cost per user per month?", "Rs {price} per user per month",
     "A {name} licence costs Rs {price} per user per month"),
    ("In which year was {name} first launched?", "{year}", "It was first launched in {year}."),
    ("How long is the service warranty included with a {name} subscription?", "{warranty} months",
     "Every {name} subscription includes a {warranty}-month service warranty"),
    ("What level of support do {name} customers receive?", "{tier} support", "customers on {name} receive {tier} support."),
    ("How many users per account does {name} support?", "Up to {users} users", "supports up to {users} users per account."),
    ("In which city was {name} developed?", "{city}", "developed by the Nimbus team in {city}."),
]
_FAKE = ["Zenith", "Quasar", "Nimbus", "Polaris", "Stratus", "Titan", "Comet", "Meteor", "Aurora", "Photon"]


def template_questions(pages, n):
    """Exact questions from the catalog's facts: ~90% answerable, ~10% about products that don't exist."""
    from evaluation.question_gen import locate_evidence
    facts = []
    for page in pages:
        text = " ".join(page["text"].split())
        for m in _PRODUCT.finditer(text):
            facts.append((page, m.groupdict()))
    n_fake = max(1, n // 10)
    tests = []
    for i in range(n - n_fake):
        page, f = facts[i % len(facts)]
        q, a, ev = _QUESTIONS[(i // len(facts)) % len(_QUESTIONS)]
        evidence = ev.format(**f)
        span = locate_evidence(evidence, page["text"]) or (0, len(page["text"]))
        tests.append({"test_id": f"T{len(tests) + 1}", "question": q.format(**f), "expected_answer": a.format(**f),
                      "answerable": True, "source_chunk_id": None, "evidence": evidence, "doc_name": page["doc_name"],
                      "page": page["page"], "ev_start": span[0], "ev_end": span[1], "origin": "template"})
    for i in range(n_fake):
        name = f"{_FAKE[i % len(_FAKE)]}{['Desk', 'Sync', 'Flow'][i // len(_FAKE) % 3]}"
        q = _QUESTIONS[i % len(_QUESTIONS)][0].format(name=name)
        tests.append({"test_id": f"T{len(tests) + 1}", "question": q, "expected_answer": "Not in the documents.",
                      "answerable": False, "source_chunk_id": None, "evidence": "", "doc_name": None, "page": None,
                      "ev_start": None, "ev_end": None, "origin": "template"})
    return tests


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--questions", type=int, default=200)
    p.add_argument("--judge", choices=config.JUDGE_MODES, default="local")
    p.add_argument("--judge-model", default=config.DEFAULT_JUDGE_MODEL)
    p.add_argument("--generator", default=config.DEFAULT_GENERATOR_MODEL)
    p.add_argument("--question-model", default=f"{config.DEFAULT_PROVIDER}::qwen/qwen3.8-27b",
                   help="model that writes the test questions")
    p.add_argument("--version", default="v1.1 BM25")
    p.add_argument("--chunk-size", type=int, default=80)
    p.add_argument("--overlap", type=int, default=10)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--workers", type=int, default=config.PARALLEL_TESTS)
    p.add_argument("--reuse-tests", action="store_true", help="reuse the last benchmark test set instead of generating")
    p.add_argument("--llm-questions", action="store_true",
                   help="write questions with the LLM question generator instead of from the catalog's facts")
    p.add_argument("--generate-only", action="store_true", help="only prepare the test set, don't evaluate")
    args = p.parse_args()

    pages, _ = load_documents([f"{config.DATA_DIR}/{d}" for d in config.BENCHMARK_DOCS])
    store = VectorStore(chunk_pages(pages, args.chunk_size, args.overlap))
    print(f"Benchmark docs: {len(pages)} pages, {len(store.chunks)} chunks")

    tests = db.load_test_set("benchmark") if args.reuse_tests else []
    if len(tests) < args.questions and not args.llm_questions:
        tests = template_questions(pages, args.questions)
        db.save_test_set(tests, "benchmark")
        print(f"Built {len(tests)} questions from the catalog's facts "
              f"({sum(t['answerable'] for t in tests)} answerable, {sum(not t['answerable'] for t in tests)} unanswerable)")
    if len(tests) < args.questions:
        print(f"Generating {args.questions} test questions with {args.question_model} (not timed)...")
        t = time.time()
        tests = generate_questions(store.chunks, pages, args.questions, model=args.question_model, store=store,
                                   progress=lambda d, n: print(f"  batch {d}/{n}"))
        db.save_test_set(tests, "benchmark")
        print(f"  {len(tests)} tests ready in {time.time() - t:.0f} s")
    tests = tests[:args.questions]
    if args.generate_only:
        return

    cfg = make_config(args.version, args.generator, args.judge_model, args.chunk_size, args.overlap,
                      args.top_k, shuffle_check=False, judge_mode=args.judge)
    print(f"\nEvaluating {len(tests)} questions: {args.version}, generator {args.generator}, "
          f"judge {'local' if args.judge == 'local' else args.judge_model}, {args.workers} in parallel...")

    def progress(done, total):
        if done % 10 == 0 or done == total:
            print(f"  {done}/{total}  ({time.time() - start:.0f} s)")

    start = time.time()
    results, wall = run_jobs([("bench", store, cfg)], tests, progress=progress, workers=args.workers)
    run = make_run(results["bench"], cfg, tests, label=f"⏱️ Benchmark {len(tests)} questions", wall_seconds=wall)
    save_run(run, kind="benchmark")
    s = run["summary"]

    print("\n================ RESULT ================")
    print(f"Questions        : {s['total']}  (errors: {s['errors']})")
    print(f"Total time       : {wall:.0f} s  = {wall / 60:.1f} min   -> target 15 min: "
          f"{'MET' if wall <= TARGET_SECONDS else 'NOT MET'}")
    print(f"Throughput       : {s['questions_per_minute']} questions/min")
    print(f"Pass rate        : {s['pass_rate']:.0%}   categories: {s['categories']}")
    print(f"Tokens           : {s['tokens']:,} ({s['tokens_per_question']} per question), est. cost ${s['cost']:.4f}")
    print(f"Per question     : retrieve {s['retrieve_s']} s, generate {s['generate_s']} s, judge {s['judge_s']} s")
    print(f"Saved as run     : {run['run_id']}  (see the dashboard's Run history)")


if __name__ == "__main__":
    main()
