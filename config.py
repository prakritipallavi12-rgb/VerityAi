"""
config.py - All settings for the project live here.

Change a value here instead of hunting through the code.
Secret values (like API keys) are NOT stored here - they live in the .env file.
"""

import os
from dotenv import load_dotenv

# Read the .env file and put its values into environment variables
load_dotenv()

# Hide a harmless Windows warning from the embedding model download cache
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ---------------- LLM providers ----------------
# Every provider here speaks the OpenAI format, so one piece of code talks to all of them.
# A provider shows up in the dashboard only when its key is in .env.
PROVIDERS = {
    "Groq": {"base_url": "https://api.groq.com/openai/v1", "key": os.getenv("GROQ_API_KEY")},
    "OpenAI": {"base_url": "https://api.openai.com/v1", "key": os.getenv("OPENAI_API_KEY")},
    "Gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key": os.getenv("GEMINI_API_KEY")},
    # OpenRouter: one free key gives access to free Llama models (model names end in ":free")
    "OpenRouter": {"base_url": "https://openrouter.ai/api/v1", "key": os.getenv("OPENROUTER_API_KEY")},
    # Ollama: runs Llama on your own laptop, no key needed (install from ollama.com, then `ollama pull llama3.2`)
    "Ollama": {"base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"), "key": "ollama", "local": True},
}

# The main key from .env (LLM_API_KEY + LLM_BASE_URL) fills in whichever provider it belongs to
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

DEFAULT_PROVIDER = next(
    (name for name, p in PROVIDERS.items() if p["base_url"].split("/")[2] in LLM_BASE_URL), "Groq"
)
if LLM_API_KEY and not PROVIDERS[DEFAULT_PROVIDER]["key"]:
    PROVIDERS[DEFAULT_PROVIDER]["key"] = LLM_API_KEY

# Models offered if a provider's model list can't be fetched
FALLBACK_MODELS = {
    "Groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
    "OpenAI": ["gpt-4.1-mini", "gpt-4o-mini"],
    "Gemini": ["gemini-3.8-flash", "gemini-3.5-flash-lite"],
    "OpenRouter": ["meta-llama/llama-3.3-70b-instruct:free", "meta-llama/llama-3.2-3b-instruct:free"],
    "Ollama": ["llama3.2"],
}

# Rate limits per model: (requests per minute, tokens per minute).
# Groq free tier = 30 requests and 8,000 tokens per minute per model. The limiter also reads the real
# limits from each API response, so a paid tier speeds up automatically. Override in .env if needed.
RATE_LIMITS = {
    "Groq": (int(os.getenv("GROQ_RPM", 30)), int(os.getenv("GROQ_TPM", 8000))),
    "OpenAI": (int(os.getenv("OPENAI_RPM", 500)), int(os.getenv("OPENAI_TPM", 200000))),
    "Gemini": (int(os.getenv("GEMINI_RPM", 10)), int(os.getenv("GEMINI_TPM", 250000))),  # free tier is per-request limited
    "OpenRouter": (int(os.getenv("OPENROUTER_RPM", 20)), int(os.getenv("OPENROUTER_TPM", 100000))),
    "Ollama": (10_000, 10_000_000),  # local - no limit
}

# Estimated list prices in US$ per 1 million tokens (input, output) - used by the cost tracker.
# Free tiers cost nothing; these show what the run WOULD cost on a paid plan. Edit to match current prices.
PRICES = {
    "openai/gpt-oss-120b": (0.15, 0.60), "openai/gpt-oss-20b": (0.075, 0.30), "qwen/qwen3.8-27b": (0.29, 0.59),
    "gpt-4.1-mini": (0.40, 1.60), "gpt-4o-mini": (0.15, 0.60), "gemini-2.5-flash": (0.30, 2.50),
}
FREE_PROVIDERS = ("Ollama",)  # always $0; OpenRouter ":free" models are $0 too

# A model is written as "Provider::model-id", e.g. "Groq::openai/gpt-oss-20b"
# Generator = the model inside the RAG system being tested.
# Judge = the model that grades answers. A stronger judge than generator is good practice.
# Default (generator, judge) model for each provider, used when it is the main provider
PROVIDER_DEFAULTS = {
    "Groq": ("openai/gpt-oss-20b", "openai/gpt-oss-120b"),
    "Gemini": ("gemini-3.8-flash", "gemini-3.8-flash"),
    "OpenAI": ("gpt-4.1-mini", "gpt-4.1-mini"),
    "OpenRouter": ("meta-llama/llama-3.3-70b-instruct:free", "meta-llama/llama-3.3-70b-instruct:free"),
    "Ollama": ("llama3.2", "llama3.2"),
}


def _default_model(env_name, fallback):
    value = os.getenv(env_name) or fallback
    return value if "::" in value else f"{DEFAULT_PROVIDER}::{value}"


DEFAULT_GENERATOR_MODEL = _default_model("GENERATOR_MODEL", PROVIDER_DEFAULTS[DEFAULT_PROVIDER][0])
DEFAULT_JUDGE_MODEL = _default_model("JUDGE_MODEL", os.getenv("LLM_MODEL") or PROVIDER_DEFAULTS[DEFAULT_PROVIDER][1])

# 0 = most predictable output. Same input -> (almost) same answer, which keeps the demo stable.
LLM_TEMPERATURE = 0

# How many times to retry when the LLM returns broken JSON or the API times out
LLM_MAX_RETRIES = 3

# How many tests run at the same time. The rate limiter stops this from overloading the API.
PARALLEL_TESTS = int(os.getenv("PARALLEL_TESTS", 12))

# ---------------- Judge ----------------
# "llm"  : an LLM grades each answer (most detailed, uses API tokens)
# "local": small models on your laptop grade each answer (no API limits - for big runs like 200 questions)
JUDGE_MODES = ("llm", "local")
DEFAULT_JUDGE_MODE = "llm"
# Local judge models (download once, then run offline)
NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"          # does this chunk support this sentence?
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"     # how relevant is this chunk to the question?
LOCAL_SUPPORT_THRESHOLD = 0.4                              # entailment probability needed to count as supported
                                                           # (0.4 agreed best with the LLM judge on saved runs)
LOCAL_DRIFT_THRESHOLD = 0.0005                             # below this question-answer relevance = generation drift
                                                           # (correct answers measured: median 0.9999, lowest 0.001)

# ---------------- RAG versions (the systems being tested) ----------------
# retriever: semantic (embeddings), bm25 (keyword ranking), hybrid (both + prefers newest docs)
# rerank   : a second model re-orders the top candidates before the generator sees them
# prompt   : grounded (context only), strict (context only + newest source wins), loose (may use outside knowledge)
RAG_VERSIONS = {
    "v1.0 Baseline": {"retriever": "semantic", "rerank": False, "prompt": "grounded",
                      "description": "Embedding search + answer only from context"},
    "v1.1 BM25": {"retriever": "bm25", "rerank": False, "prompt": "grounded",
                  "description": "BM25 keyword search + answer only from context"},
    "v2.0 Hybrid": {"retriever": "hybrid", "rerank": False, "prompt": "strict",
                    "description": "Embeddings + BM25, prefers the newest document"},
    "v2.1 Hybrid + Rerank": {"retriever": "hybrid", "rerank": True, "prompt": "strict",
                             "description": "Hybrid search, then a re-ranker re-orders the chunks"},
    "v0.9 Legacy": {"retriever": "semantic", "rerank": False, "prompt": "loose",
                    "description": "Old prompt that lets the model use outside knowledge"},
}
DEFAULT_RAG_VERSION = "v1.0 Baseline"
RERANK_DEPTH = 10           # the re-ranker looks at this many first-stage results

# ---------------- Embeddings (runs locally, free) ----------------
# Small, fast model that turns text into 384 numbers. Downloads once (~90 MB), then works offline.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---------------- Chunking ----------------
CHUNK_SIZE_WORDS = 120      # words per chunk (smaller = more precise search, but less context per chunk)
CHUNK_OVERLAP_WORDS = 20    # words shared between neighbouring chunks, so sentences aren't cut off

# ---------------- Retrieval ----------------
TOP_K = 3                   # how many chunks the retriever gives to the generator

# ---------------- "Break it" demo ----------------
# Deliberately bad settings: tiny chunks, only 1 chunk retrieved, prompt that invites guessing
BREAK_IT = {"chunk_size": 40, "top_k": 1, "version": "v0.9 Legacy"}

# ---------------- Evaluation thresholds ----------------
# Below this similarity, we say "answer is not in the document"
NOT_IN_DOCUMENT_THRESHOLD = 0.5
# Below this accuracy score, an answer counts as wrong / incomplete
ACCURACY_PASS_THRESHOLD = 0.7
# Share of answer sentences that must be backed by the chunks (1.0 = every factual sentence)
FAITHFULNESS_PASS_THRESHOLD = 1.0
# Scorecard colours: >= GOOD is green, >= OK is amber, below is red
SCORE_GOOD = 0.8
SCORE_OK = 0.5
# A/B: a drop bigger than this in a query's overall score counts as a regression
REGRESSION_DROP = 0.15
# How many synthetic test questions to generate per run
NUM_TEST_QUESTIONS = 8
# Share of generated questions whose answer is deliberately NOT in the documents
UNANSWERABLE_RATIO = 0.2
# Goal for hand-labelled items used to check the judge against humans
HUMAN_LABEL_TARGET = 150

# Judge reliability check: how many answers to hand-label
RELIABILITY_SAMPLE = 10
# Stability: a judge swap / shuffle counts as stable if verdicts agree this often and scores move this little
STABLE_AGREEMENT = 0.9
STABLE_MAX_MEAN_DIFF = 0.1

# ---------------- Folders, database, API ----------------
DATA_DIR = "data"           # sample PDFs live here
RESULTS_DIR = "results"     # database + legacy JSON files
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
DB_PATH = os.path.join(RESULTS_DIR, "verityai.db")
SAMPLE_DOCS = ["nimbus_handbook_2025.pdf", "nimbus_handbook_2023.pdf"]
BENCHMARK_DOCS = ["nimbus_product_catalog_2025.pdf"]    # 50 products x 4 facts, for 200-question runs
API_URL = os.getenv("VERITY_API_URL", "http://localhost:8000")
