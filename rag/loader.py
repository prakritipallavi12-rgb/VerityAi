"""
rag/loader.py - STAGE 1a: Read text out of a PDF or TXT file.

Input : a file path (str), a Streamlit uploaded file, or a (name, bytes) pair
Output: a list of pages, each page remembers where it came from:
        [{"doc_name": "policy.pdf", "page": 1, "text": "...",
          "doc_year": 2025, "is_current": True}, ...]

Why pages? So every chunk can later say "this came from policy.pdf, page 3".

Document versions: if two files share a name apart from a year
(handbook_2023.pdf and handbook_2025.pdf), the newest is "current" and older ones
are "archived". Answers built on archived text are caught as a STALE SOURCE.
"""

import io
import os
import re

from pypdf import PdfReader


def load_document(file):
    """Read one PDF or TXT file and return its pages."""
    doc_name, data = _read_bytes(file)
    extension = os.path.splitext(doc_name)[1].lower()

    if extension == ".pdf":
        pages = _load_pdf(doc_name, data)
    elif extension == ".txt":
        pages = _load_txt(doc_name, data)
    else:
        raise ValueError(f"{doc_name}: only .pdf and .txt files are supported")

    # Drop empty pages (e.g. blank pages, or scanned images that have no text layer)
    pages = [p for p in pages if p["text"].strip()]
    if not pages:
        raise ValueError(
            f"{doc_name}: no text found. It may be a scanned PDF (images only) - try a text-based PDF."
        )
    return pages


def load_documents(files):
    """Read several files. Returns (all_pages, errors) so one bad file doesn't stop the rest."""
    all_pages, errors = [], []
    for file in files:
        try:
            all_pages.extend(load_document(file))
        except Exception as e:
            errors.append(str(e))
    return mark_versions(all_pages), errors


def mark_versions(pages):
    """Add doc_year and is_current to every page (see the note at the top of this file)."""
    def family(name):  # "nimbus_handbook_2023.pdf" -> "nimbus_handbook_.pdf"
        return re.sub(r"(19|20)\d{2}", "", name.lower())

    def year(name):
        found = re.findall(r"(?:19|20)\d{2}", name)
        return int(found[-1]) if found else None

    newest = {}
    for p in pages:
        y = year(p["doc_name"]) or 0
        newest[family(p["doc_name"])] = max(newest.get(family(p["doc_name"]), 0), y)
    for p in pages:
        p["doc_year"] = year(p["doc_name"])
        p["is_current"] = (p["doc_year"] or 0) == newest[family(p["doc_name"])]
    return pages


def _read_bytes(file):
    """Get (file name, raw bytes) from a path, a (name, bytes) pair or a Streamlit upload."""
    if isinstance(file, str):
        with open(file, "rb") as f:
            return os.path.basename(file), f.read()
    if isinstance(file, tuple):
        return file
    return file.name, file.getvalue()  # Streamlit UploadedFile


def _load_pdf(doc_name, data):
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for number, page in enumerate(reader.pages, start=1):  # page numbers start at 1, like in a PDF viewer
        pages.append({"doc_name": doc_name, "page": number, "text": page.extract_text() or ""})
    return pages


def _load_txt(doc_name, data):
    text = data.decode("utf-8", errors="replace")
    # A text file has no real pages; a form-feed character (\f) marks a page break if present
    return [{"doc_name": doc_name, "page": number, "text": part}
            for number, part in enumerate(text.split("\f"), start=1)]
