"""
rag/chunker.py - STAGE 1b: Split pages into small overlapping chunks.

Input : the pages from loader.py
Output: a list of chunks, each with its location:
        [{"chunk_id": 0, "doc_name": "policy.pdf", "page": 3,
          "start_char": 1450, "end_char": 1890, "word_count": 120, "text": "..."}, ...]

The location fields are what let the failure report say
"the answer was in policy.pdf -> Page 3 -> Chunk #12".

How it works (sliding window):
    words:   w1 w2 w3 ... w120 | w101 ... w220 | w201 ...
             '--- chunk 0 ---'
                          '--- chunk 1 ---'        <- starts 100 words later,
                                                      so 20 words overlap
Chunks never cross a page boundary, so every chunk has exactly one page number.
"""

import re

import config


def chunk_pages(pages, chunk_size=None, overlap=None):
    """Split every page into overlapping word windows. Chunk IDs are unique across all documents."""
    chunk_size = chunk_size or config.CHUNK_SIZE_WORDS
    overlap = config.CHUNK_OVERLAP_WORDS if overlap is None else overlap
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")
    step = chunk_size - overlap  # how far the window moves each time

    chunks = []
    for page in pages:
        text = page["text"]
        # Each match is one word, and remembers its start/end position in the page text
        words = list(re.finditer(r"\S+", text))

        for first in range(0, len(words), step):
            window = words[first:first + chunk_size]
            start_char, end_char = window[0].start(), window[-1].end()
            chunks.append({
                "chunk_id": len(chunks),
                "doc_name": page["doc_name"],
                "page": page["page"],
                "doc_year": page.get("doc_year"),
                "is_current": page.get("is_current", True),
                "start_char": start_char,
                "end_char": end_char,
                "word_count": len(window),
                # Collapse line breaks/extra spaces so the text reads cleanly
                "text": " ".join(text[start_char:end_char].split()),
            })
            if first + chunk_size >= len(words):
                break  # this window already reached the end of the page

    return chunks


def parse_chunk_id(value):
    """LLMs write chunk IDs as 4, "4", "#4" or "Chunk #4" - turn any of them into the number 4."""
    found = re.search(r"\d+", str(value)) if value is not None else None
    return int(found.group()) if found else None


def format_location(chunk):
    """Human-readable location used everywhere in the UI and reports."""
    return f"{chunk['doc_name']} → Page {chunk['page']} → Chunk #{chunk['chunk_id']}"


if __name__ == "__main__":
    # Quick check from the project folder:  python -m rag.chunker
    from rag.loader import load_document

    pages = load_document("data/nimbus_handbook_2025.pdf")
    chunks = chunk_pages(pages)
    print(f"{len(pages)} pages -> {len(chunks)} chunks\n")
    for c in chunks:
        print(f"{format_location(c):45} {c['word_count']:3} words | {c['text'][:60]}...")
