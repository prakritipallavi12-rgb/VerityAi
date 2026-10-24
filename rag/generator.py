"""
rag/generator.py - STAGE 3: Answer the question using the retrieved chunks.

Input : a question + the top-k chunks + a prompt style + a model
Output: the answer text

Prompt styles (each RAG version uses one):
  grounded : answer ONLY from the context, otherwise say "Not found in the documents."
  strict   : grounded + when sources conflict, trust the newest document
  loose    : may add outside knowledge (the "legacy" version - invites hallucinations)
"""

from utils.llm import ask

NOT_FOUND = "Not found in the documents."

PROMPTS = {
    "grounded": (
        "You answer questions using ONLY the provided context. "
        f'If the context does not contain the answer, reply exactly: "{NOT_FOUND}" '
        "Answer in 1-3 short sentences."
    ),
    "strict": (
        "You answer questions using ONLY the provided context. "
        "The context may include older versions of a document; when sources disagree, "
        "use the most recent one (the highest year). "
        f'If the context does not contain the answer, reply exactly: "{NOT_FOUND}" '
        "Answer in 1-3 short sentences."
    ),
    "loose": (
        "You are a helpful HR assistant. Use the context and your own general knowledge "
        "to give a complete, confident answer. Never say you don't know."
    ),
}


def format_context(chunks):
    return "\n\n".join(
        f"[Chunk #{c['chunk_id']} | {c['doc_name']}, page {c['page']}]\n{c['text']}" for c in chunks
    )


def generate_answer(question, chunks, prompt_style="grounded", model=None):
    prompt = f"CONTEXT:\n{format_context(chunks)}\n\nQUESTION: {question}\n\nANSWER:"
    # max_tokens caps the answer; free tiers reserve this many tokens per call, so smaller = more calls per minute
    return ask(prompt, system=PROMPTS[prompt_style], model=model, max_tokens=400).strip()
