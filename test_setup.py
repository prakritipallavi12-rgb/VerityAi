"""
test_setup.py - Run this FIRST on every team member's laptop.

    python test_setup.py

It checks, one by one, that everything the project needs is working.
If a step fails, it tells you what to fix.
"""

import sys


def check(name, func):
    """Run one check and print PASS or FAIL."""
    try:
        detail = func()
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
        return True
    except Exception as e:
        print(f"  FAIL  {name}\n        -> {e}")
        return False


def check_python():
    if sys.version_info < (3, 10):
        raise RuntimeError(f"Python 3.10+ needed, you have {sys.version.split()[0]}")
    return sys.version.split()[0]


def check_imports():
    import numpy, pandas, pypdf, streamlit, openai, sentence_transformers, dotenv  # noqa: F401
    return "numpy, pandas, pypdf, streamlit, openai, sentence-transformers, python-dotenv"


def check_api_key():
    import config
    from utils.llm import available_providers
    providers = available_providers()
    if not providers:
        raise RuntimeError("No key found. Open .env and paste a key, e.g. GROQ_API_KEY=... or GEMINI_API_KEY=...")
    # never print the keys themselves
    return f"providers ready: {', '.join(providers)} | main: {config.DEFAULT_PROVIDER} ({config.DEFAULT_JUDGE_MODEL})"


def check_llm():
    import config
    from utils.llm import ask
    reply = ask("Reply with exactly the word: WORKING")
    if "WORKING" not in reply.upper():
        raise RuntimeError(f"Unexpected reply: {reply!r}")
    return f"model {config.LLM_MODEL} replied {reply.strip()!r}"


def check_llm_json():
    from utils.llm import ask_json
    data = ask_json('Return this JSON exactly: {"status": "ok", "number": 42}')
    if data.get("number") != 42:
        raise RuntimeError(f"Unexpected JSON: {data}")
    return "valid JSON received"


def check_embeddings():
    import config
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer(config.EMBEDDING_MODEL)  # downloads ~90 MB the first time
    a, b, c = model.encode(["How many leave days do I get?",
                            "Employees receive 20 days of paid leave.",
                            "The cafeteria opens at 9 am."])
    related = float(util.cos_sim(a, b))
    unrelated = float(util.cos_sim(a, c))
    if related <= unrelated:
        raise RuntimeError("Embeddings look wrong: related text should score higher")
    return f"related={related:.2f} > unrelated={unrelated:.2f}"


if __name__ == "__main__":
    print("\nChecking project setup...\n")
    results = [
        check("Python version", check_python),
        check("Libraries installed", check_imports),
        check("API key in .env", check_api_key),
        check("LLM text reply", check_llm),
        check("LLM JSON reply", check_llm_json),
        check("Local embedding model", check_embeddings),
    ]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed.")
    print("Setup complete - ready to build!\n" if all(results) else "Fix the FAIL lines above and run again.\n")
