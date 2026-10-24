# VerityAi — RAG Evaluator (AI-04)

Tests a RAG system by grading the **retriever** and the **generator** separately, and explains every failed test:
*what* went wrong (hallucination, wrong chunk, ignored context, stale source, refusal, generation drift, incomplete)
and *where* the evidence is (document → page → chunk).

## Start it

```bash
C:\Users\<you>\venvs\rag-evaluator\Scripts\Activate.ps1
python start.py
```

- Dashboard: http://localhost:8501
- API: http://localhost:8000 — every endpoint is listed at http://localhost:8000/docs

## One-time setup (every team member)

1. **Install Python 3.12** from python.org (tick "Add python.exe to PATH"), or `winget install Python.Python.3.12`.
2. **Create the virtual environment** outside OneDrive (OneDrive slows down syncing thousands of library files):
   `python -m venv C:\Users\<you>\venvs\rag-evaluator`
3. **Activate it** every time: `C:\Users\<you>\venvs\rag-evaluator\Scripts\Activate.ps1`
   (if PowerShell blocks it, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once)
4. **Install the exact library versions:** `pip install -r requirements.txt`
5. **Add a free API key:** get a Groq key at console.groq.com/keys and paste it after `LLM_API_KEY=` in `.env`
   (copy `.env.example` to `.env` first). Optional extras in `.env`:
   `OPENAI_API_KEY` (OpenAI models), `OPENROUTER_API_KEY` (free Llama models), or install **Ollama** for local Llama.
6. **Check everything works:** `python test_setup.py`

## Architecture

```
 Streamlit dashboard (app.py, frontend/)  ──HTTP──▶  FastAPI backend (api/server.py)
                                                        │
              ┌─────────────────────────────────────────┼──────────────────────────────┐
              ▼                                         ▼                              ▼
   RAG system under test (rag/)              Evaluator (evaluation/)            SQLite (storage/db.py)
   loader → chunker → embedder →             question_gen → scorers →           runs, results,
   vector store (embeddings / BM25 /         judge (LLM or local) →             human labels, test sets
   hybrid) → re-ranker → generator           root_cause → runner
```

```
rag-evaluator/
├── start.py               # starts API + dashboard together
├── app.py                 # Streamlit dashboard (display only - talks to the API)
├── frontend/              # api.py (HTTP client) + components.py (visuals)
├── api/                   # server.py (FastAPI endpoints) + jobs.py (background jobs with progress)
├── config.py              # all settings in one place
├── benchmark.py           # 200-question speed test
├── test_setup.py          # checks libraries, API key, LLM and embeddings
├── rag/                   # the RAG system being tested
│   ├── loader.py          # PDF/TXT -> pages, marks current vs archived document versions
│   ├── chunker.py         # pages -> chunks with exact locations
│   ├── embedder.py        # text -> vectors (local, free)
│   ├── retriever.py       # vector store: embeddings, BM25, hybrid search
│   ├── reranker.py        # cross-encoder re-ranker (second stage)
│   └── generator.py       # answer from the top chunks
├── evaluation/            # the evaluator (our actual project)
│   ├── question_gen.py    # synthetic questions {question, answer, source_chunk_id}
│   ├── scorers.py         # context relevance, faithfulness (claims), answer accuracy - LLM judge
│   ├── local_judge.py     # judge that runs on the laptop (NLI + word overlap + number check)
│   ├── root_cause.py      # plain if/else: why it failed + where the evidence is
│   ├── runner.py          # retrieve -> generate -> score -> classify, with tokens/time/cost per question
│   ├── retriever_compare.py  # BM25 vs embeddings vs hybrid, with/without re-ranker (no LLM)
│   ├── stability.py       # re-grade with a swapped judge / shuffled chunks
│   ├── labels.py          # human labels + judge reliability check (agreement, Cohen's kappa)
│   ├── compare.py         # A/B regressions
│   └── report.py          # PDF report
├── storage/db.py          # SQLite database (results/verityai.db)
└── utils/llm.py           # every LLM call: any provider, strict JSON, rate limiter, usage tracking
```

## Features

| Feature | Where |
|---|---|
| Synthetic test questions with known answers + `source_chunk_id` | ✨ Generate tests |
| Context relevance (source chunk in top-k, no LLM), faithfulness (claim by claim), accuracy (1 / 0.5 / 0 rubric) | 📊 Results |
| Root cause with evidence location, red highlight of unsupported claims | 📊 Results → click a test |
| Cost & time: tokens, est. cost, seconds per question, throughput, time per stage | 📊 Results |
| BM25 vs embeddings vs hybrid, each with and without the re-ranker | 🔎 Retrievers |
| Several models on the same tests — which hallucinates least | 🏆 Model leaderboard |
| A/B with 🔻 regression alerts | 🆚 A/B compare |
| Hand-label 10 answers, measure judge agreement + Cohen's κ | ⚖️ Judge reliability |
| Stability: swap the judge model, shuffle the chunks | ⚖️ Judge reliability |
| Every run stored permanently (SQLite) | 🕘 Run history |
| 200 questions in under 15 minutes | `python benchmark.py` |

## Free-tier limits (Groq) — read before demo day

Each Groq model allows **8,000 tokens per minute** and **200,000 tokens per day** on the free tier.

- One question with the LLM judge uses ~2,000 tokens, so a 200-question LLM-judged run needs more than a whole day's
  allowance. **Use the 💻 Local judge for big runs** — it grades on the laptop with no API calls.
- If a model hits its daily limit, VerityAi says so clearly; switch model or judge, or wait until tomorrow.
- On a paid tier the rate limiter reads the higher limits automatically and runs go faster.

## How stable is the local judge?

Measured against the LLM judge on saved runs: **88% same verdict** on prose documents (the handbook), lower on
code-heavy documents. Use the ⚖️ stability check to measure it on your own documents; use the LLM judge when token
budget allows.
