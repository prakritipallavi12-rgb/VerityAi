"""
api/server.py - VerityAi backend (FastAPI).

All the work happens here: loading documents, the RAG system being tested, the evaluator,
the database. The Streamlit dashboard (app.py) only calls these endpoints over HTTP.

Run:   uvicorn api.server:app --port 8000
Docs:  http://localhost:8000/docs   (interactive list of every endpoint - good to show judges)
"""

import hashlib
import os
import threading
import time
from collections import OrderedDict
from functools import lru_cache

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

import config
from api import jobs
from evaluation.compare import compare_runs
from evaluation.labels import (alignment_stats, delete_label, item_alignment, load_labels,
                               reliability_report, save_label)
from evaluation.question_gen import build_custom_test, generate_questions, gold_chunk_ids
from evaluation.report import build_pdf
from evaluation.retriever_compare import compare_retrievers
from evaluation.root_cause import CATEGORY_LABELS, GENERATION_LABELS
from evaluation.runner import get_run, list_runs, make_config, make_run, run_jobs, save_run
from evaluation.stability import stability_check
from rag.chunker import chunk_pages, format_location
from rag.loader import load_documents
from rag.retriever import VectorStore
from storage import db
from utils.llm import available_providers, list_models, model_label

app = FastAPI(title="VerityAi API", version="2.0",
              description="RAG evaluation backend: tests the retriever and the generator separately.")

# ======================================================================
# Documents and vector stores (kept in memory, uploads also saved to disk)
# ======================================================================
_docs = {}                  # docs_key -> pages
_stores = OrderedDict()     # (docs_key, chunk_size, overlap) -> VectorStore
_store_lock = threading.Lock()


def _load_docs(docs_key):
    if docs_key in _docs:
        return _docs[docs_key]
    folder = os.path.join(config.UPLOAD_DIR, docs_key)
    if not os.path.isdir(folder):  # e.g. after a server restart
        raise HTTPException(404, "Unknown document set - upload the documents again.")
    pages, _ = load_documents([os.path.join(folder, n) for n in sorted(os.listdir(folder))])
    _docs[docs_key] = pages
    return pages


def _store(docs_key, chunk_size, overlap):
    key = (docs_key, chunk_size, overlap)
    with _store_lock:
        if key in _stores:
            _stores.move_to_end(key)
            return _stores[key]
    store = VectorStore(chunk_pages(_load_docs(docs_key), chunk_size, overlap))
    with _store_lock:
        _stores[key] = store
        while len(_stores) > 16:
            _stores.popitem(last=False)
    return store


class DocSettings(BaseModel):
    docs_key: str
    chunk_size: int = config.CHUNK_SIZE_WORDS
    overlap: int = config.CHUNK_OVERLAP_WORDS


class RetrievalRequest(DocSettings):
    top_k: int = config.TOP_K
    version: str = config.DEFAULT_RAG_VERSION


class CustomTestRequest(DocSettings):
    question: str
    model: str = config.DEFAULT_JUDGE_MODEL


class GenerateRequest(DocSettings):
    n: int = config.NUM_TEST_QUESTIONS
    model: str = config.DEFAULT_JUDGE_MODEL


class SystemConfig(BaseModel):
    name: str
    version: str
    generator_model: str


class EvaluateRequest(DocSettings):
    top_k: int = config.TOP_K
    judge_model: str = config.DEFAULT_JUDGE_MODEL
    judge_mode: str = config.DEFAULT_JUDGE_MODE
    shuffle_check: bool = False
    workers: int = config.PARALLEL_TESTS
    configs: list[SystemConfig]
    label: str = ""
    kind: str = "eval"


class StabilityRequest(BaseModel):
    run_id: str
    alt_judge_model: str | None = None
    alt_judge_mode: str | None = None
    check_shuffle: bool = True


class LabelRequest(BaseModel):
    question: str
    answer: str
    verdict: str = ""
    category: str = ""
    labeler: str = ""


class TestSet(BaseModel):
    tests: list[dict]


# ======================================================================
# Info
# ======================================================================
@app.get("/health")
def health():
    return {"ok": True, "providers": available_providers()}


@lru_cache(maxsize=1)
def _model_options(_hour):
    return [f"{p}::{m}" for p in available_providers() for m in list_models(p)]


@app.get("/models")
def models():
    options = _model_options(int(time.time() // 3600))
    return {"options": options, "labels": {o: model_label(o) for o in options},
            "providers": {p: p in available_providers() for p in config.PROVIDERS}}


@app.get("/meta")
def meta():
    """Everything the dashboard needs to draw itself (so it never imports backend code)."""
    return {
        "versions": {k: v["description"] for k, v in config.RAG_VERSIONS.items()},
        "default_version": config.DEFAULT_RAG_VERSION,
        "defaults": {"chunk_size": config.CHUNK_SIZE_WORDS, "overlap": config.CHUNK_OVERLAP_WORDS,
                     "top_k": config.TOP_K, "n_tests": config.NUM_TEST_QUESTIONS,
                     "generator_model": config.DEFAULT_GENERATOR_MODEL, "judge_model": config.DEFAULT_JUDGE_MODEL,
                     "judge_mode": config.DEFAULT_JUDGE_MODE, "workers": config.PARALLEL_TESTS},
        "break_it": config.BREAK_IT,
        "category_labels": CATEGORY_LABELS, "generation_labels": GENERATION_LABELS,
        "thresholds": {"good": config.SCORE_GOOD, "ok": config.SCORE_OK,
                       "stable_agreement": config.STABLE_AGREEMENT, "stable_diff": config.STABLE_MAX_MEAN_DIFF},
        "sample_docs": config.SAMPLE_DOCS, "benchmark_docs": config.BENCHMARK_DOCS,
        "human_label_target": config.HUMAN_LABEL_TARGET, "reliability_sample": config.RELIABILITY_SAMPLE,
        "judge_modes": list(config.JUDGE_MODES),
    }


# ======================================================================
# Documents and chunks
# ======================================================================
@app.post("/documents")
async def upload_documents(files: list[UploadFile] = File(default=[]), use_sample: bool = Form(True),
                           use_benchmark: bool = Form(False)):
    payload = [(f.filename, await f.read()) for f in files]
    for flag, names in ((use_sample, config.SAMPLE_DOCS), (use_benchmark, config.BENCHMARK_DOCS)):
        if flag:
            for name in names:
                path = os.path.join(config.DATA_DIR, name)
                if os.path.exists(path):
                    with open(path, "rb") as fh:
                        payload.append((name, fh.read()))
    if not payload:
        raise HTTPException(400, "No documents: upload a file or include the sample documents.")
    docs_key = hashlib.sha1(b"".join(n.encode() + hashlib.sha1(d).digest() for n, d in payload)).hexdigest()[:16]
    folder = os.path.join(config.UPLOAD_DIR, docs_key)
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
        for name, data in payload:
            with open(os.path.join(folder, os.path.basename(name)), "wb") as fh:
                fh.write(data)
    pages, errors = load_documents(payload)
    if not pages:
        raise HTTPException(400, "; ".join(errors) or "No text found in the documents.")
    _docs[docs_key] = pages
    docs = sorted({(p["doc_name"], p["doc_year"], p["is_current"]) for p in pages})
    return {"docs_key": docs_key, "errors": errors, "pages": len(pages),
            "documents": [{"name": n, "year": y, "is_current": c,
                           "pages": sum(p["doc_name"] == n for p in pages)} for n, y, c in docs]}


@app.post("/chunks")
def chunks(req: DocSettings):
    store = _store(req.docs_key, req.chunk_size, req.overlap)
    pages = _load_docs(req.docs_key)
    return {"chunks": [{**c, "location": format_location(c)} for c in store.chunks],
            "pages": [{k: p[k] for k in ("doc_name", "page", "text")} for p in pages]}


# ======================================================================
# Retrieval (no LLM calls - instant)
# ======================================================================
@app.post("/retrieval/live")
def live_retrieval(req: RetrievalRequest):
    """Hit rate @ top-k for the current test set - recalculated whenever a slider moves."""
    store = _store(req.docs_key, req.chunk_size, req.overlap)
    v = config.RAG_VERSIONS[req.version]
    tests = [t for t in _tests(req.docs_key) if t.get("answerable")]
    hits = sum(any(c["chunk_id"] in gold_chunk_ids(t, store.chunks)
                   for c in store.search(t["question"], req.top_k, v["retriever"], v.get("rerank", False)))
               for t in tests)
    return {"hit_rate": round(hits / len(tests), 3) if tests else None, "answerable": len(tests),
            "chunks": len(store.chunks)}


@app.post("/retrieval/compare")
def retrieval_compare(req: RetrievalRequest):
    store = _store(req.docs_key, req.chunk_size, req.overlap)
    return {"rows": compare_retrievers(_tests(req.docs_key), store, req.top_k)}


# ======================================================================
# Test sets - one per document set, so sessions with different documents never overwrite each other
# ======================================================================
def _tests(docs_key):
    tests = db.load_test_set(docs_key)
    if tests:
        return tests
    # First visit for these documents: adopt the old shared set if it belongs to them
    try:
        names = {p["doc_name"] for p in _load_docs(docs_key)}
    except HTTPException:
        return []
    legacy = db.load_test_set("default")
    answerable = [t for t in legacy if t.get("answerable")]
    if answerable and all(t["doc_name"] in names for t in answerable):
        db.save_test_set(legacy, docs_key)
        return legacy
    return []


@app.get("/tests")
def get_tests(docs_key: str):
    return {"tests": _tests(docs_key)}


@app.put("/tests")
def put_tests(docs_key: str, body: TestSet):
    db.save_test_set(body.tests, docs_key)
    return {"saved": len(body.tests)}


@app.post("/tests/custom")
def add_custom_test(req: CustomTestRequest):
    store = _store(req.docs_key, req.chunk_size, req.overlap)
    tests = _tests(req.docs_key)
    numbers = [int(t["test_id"][1:]) for t in tests if t["test_id"][1:].isdigit()]
    test = build_custom_test(req.question, store, _load_docs(req.docs_key), f"T{max(numbers, default=0) + 1}",
                             model=req.model)
    db.save_test_set(tests + [test], req.docs_key)
    return {"test": test}


@app.post("/jobs/generate")
def generate_job(req: GenerateRequest):
    def work(progress):
        store = _store(req.docs_key, req.chunk_size, req.overlap)
        progress(0, 1, "Writing test questions…")
        tests = generate_questions(store.chunks, _load_docs(req.docs_key), req.n, model=req.model, store=store,
                                   progress=lambda d, t: progress(d, t, f"Writing test questions… batch {d}/{t}"))
        db.save_test_set(tests, req.docs_key)
        return {"tests": len(tests)}
    return {"job_id": jobs.start("generate", work)}


# ======================================================================
# Evaluation runs
# ======================================================================
@app.post("/jobs/evaluate")
def evaluate_job(req: EvaluateRequest):
    tests = _tests(req.docs_key)
    if not tests:
        raise HTTPException(400, "No tests yet - generate tests or add your own question first.")
    for c in req.configs:
        if c.version not in config.RAG_VERSIONS:
            raise HTTPException(400, f"Unknown RAG version: {c.version}")

    def work(progress):
        store = _store(req.docs_key, req.chunk_size, req.overlap)
        job_list = [(c.name, store, make_config(c.version, c.generator_model, req.judge_model, req.chunk_size,
                                                req.overlap, req.top_k, req.shuffle_check, req.judge_mode))
                    for c in req.configs]
        results, wall = run_jobs(job_list, tests, workers=req.workers,
                                 progress=lambda d, t: progress(d, t, f"Evaluating {d}/{t}…"))
        run_ids = {}
        for name, _, cfg in job_list:
            label = req.label.format(name=name, version=cfg["version"], model=model_label(cfg["generator_model"]))
            run = make_run(results[name], cfg, tests, label=label, wall_seconds=wall)
            save_run(run, kind=req.kind)
            run_ids[name] = run["run_id"]
        return {"run_ids": run_ids, "wall_seconds": wall}
    return {"job_id": jobs.start("evaluate", work)}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    return job


@app.get("/runs")
def runs():
    return {"runs": list_runs()}


@app.get("/runs/{run_id}")
def run_detail(run_id: str):
    run = get_run(run_id)
    if run is None:
        raise HTTPException(404, "Unknown run")
    labels = load_labels()
    for r in run["reports"]:
        r["alignment"] = item_alignment(r, labels)
    return run


@app.delete("/runs/{run_id}")
def delete_run(run_id: str):
    db.delete_run(run_id)
    return {"deleted": run_id}


@app.get("/runs/{run_id}/pdf")
def run_pdf(run_id: str):
    run = get_run(run_id)
    if run is None:
        raise HTTPException(404, "Unknown run")
    pdf = build_pdf(run, alignment_stats(run["reports"]))
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="verityai_{run_id}.pdf"'})


@app.get("/compare")
def compare(a: str, b: str):
    run_a, run_b = get_run(a), get_run(b)
    if run_a is None or run_b is None:
        raise HTTPException(404, "Unknown run")
    return {"a": {k: run_a[k] for k in ("run_id", "timestamp", "label", "config", "summary")},
            "b": {k: run_b[k] for k in ("run_id", "timestamp", "label", "config", "summary")},
            "rows": compare_runs(run_a["reports"], run_b["reports"])}


# ======================================================================
# Judge reliability, human labels, stability
# ======================================================================
@app.get("/labels")
def labels():
    return {"labels": load_labels()}


@app.post("/labels")
def add_label(req: LabelRequest):
    if req.verdict not in ("PASS", "FAIL"):
        raise HTTPException(400, "verdict must be PASS or FAIL")
    save_label(req.question, req.answer, req.verdict, req.category, req.labeler)
    return {"saved": True}


@app.delete("/labels")
def remove_label(req: LabelRequest):
    delete_label(req.question, req.answer)
    return {"deleted": True}


@app.get("/alignment")
def alignment():
    return alignment_stats()


@app.get("/reliability/{run_id}")
def reliability(run_id: str):
    run = get_run(run_id)
    if run is None:
        raise HTTPException(404, "Unknown run")
    return reliability_report(run)


@app.post("/jobs/stability")
def stability_job(req: StabilityRequest):
    run = get_run(req.run_id)
    if run is None:
        raise HTTPException(404, "Unknown run")

    def work(progress):
        return stability_check(run, req.alt_judge_model, req.alt_judge_mode, req.check_shuffle,
                               progress=lambda d, t: progress(d, t, f"Re-grading {d}/{t}…"))
    return {"job_id": jobs.start("stability", work)}
