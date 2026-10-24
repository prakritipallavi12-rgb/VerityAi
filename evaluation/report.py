"""
evaluation/report.py - Build a downloadable PDF report of one test run.

Uses fpdf2 (pure Python). The built-in PDF fonts only know basic Latin characters,
so symbols like → and ₹ are swapped for plain-text versions.
"""

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from evaluation.root_cause import CATEGORY_LABELS
from utils.llm import model_label

LILAC = (142, 108, 207)
SKY = (94, 170, 220)
RED = (200, 50, 50)
GREEN = (40, 140, 90)
GREY = (90, 90, 110)

_REPLACE = {"→": "->", "₹": "Rs ", "“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-",
            "‑": "-", " ": " ", " ": " ", "…": "...", "•": "-", "·": "|", "≥": ">=", "≤": "<=",
            "×": "x", "**": ""}


def clean(text):
    text = str(text if text is not None else "")
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    # Drop anything the built-in fonts can't draw (emoji etc.) instead of printing "?"
    return text.encode("latin-1", "ignore").decode("latin-1").strip()


class _PDF(FPDF):
    def header(self):
        self.set_fill_color(*LILAC)
        self.rect(0, 0, self.w, 4, "F")
        self.set_fill_color(*SKY)
        self.rect(self.w / 2, 0, self.w / 2, 4, "F")

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GREY)
        self.cell(0, 8, f"VerityAi report - page {self.page_no()}", align="C")


def _line(pdf, text, size=10, style="", color=(30, 30, 45), height=5):
    pdf.set_font("Helvetica", style, size)
    pdf.set_text_color(*color)
    pdf.multi_cell(0, height, clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _fmt(v):
    return "-" if v is None else f"{v:.2f}"


def build_pdf(run, alignment=None):
    pdf = _PDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    cfg, s = run["config"], run["summary"]

    _line(pdf, "VerityAi - RAG Evaluation Report", 20, "B", LILAC, 10)
    _line(pdf, f"Run {run['run_id']}  |  {run['timestamp']}  |  {run['label']}", 9, color=GREY)
    pdf.ln(3)

    _line(pdf, "Configuration", 13, "B", SKY, 7)
    judge = "local judge (NLI, no API)" if cfg.get("judge_mode") == "local" else model_label(cfg["judge_model"])
    for text in (f"RAG version: {cfg['version']} (retriever: {cfg['retriever']}"
                 f"{' + re-ranker' if cfg.get('rerank') else ''}, prompt: {cfg['prompt']})",
                 f"Generator LLM: {model_label(cfg['generator_model'])}",
                 f"Judge: {judge}",
                 f"Chunk size: {cfg['chunk_size']} words, overlap {cfg['overlap']}, top-k: {cfg['top_k']}",
                 f"Shuffle chunks before judging: {'on' if cfg['shuffle_check'] else 'off'}"):
        _line(pdf, text)
    pdf.ln(3)

    _line(pdf, "Summary", 13, "B", SKY, 7)
    _line(pdf, f"Tests: {s['total']}   Passed: {s['passed']}   Failed: {s['failed']}   "
               f"Errors: {s['errors']}   Pass rate: {s['pass_rate'] * 100:.0f}%", 11, "B")
    _line(pdf, f"Average relevance: {_fmt(s['relevance'])}   faithfulness: {_fmt(s['faithfulness'])}   "
               f"accuracy: {_fmt(s['accuracy'])}")
    if s["categories"]:
        _line(pdf, "Outcomes: " + ", ".join(f"{k}: {v}" for k, v in s["categories"].items()))
    if s.get("tokens") is not None:
        _line(pdf, f"Cost & time: {s['tokens']:,} tokens ({s.get('tokens_per_question', 0):,} per question), "
                   f"est. ${s.get('cost', 0):.4f}, {_fmt(s.get('seconds_per_question'))} s per question"
                   + (f", {s['wall_seconds']} s total ({s['questions_per_minute']} questions/min)"
                      if s.get("wall_seconds") else ""))
    if s["stability_checked"]:
        _line(pdf, f"Shuffle stability: {s['stability_unchanged']}/{s['stability_checked']} tests scored the same "
                   f"after shuffling the chunks.")
    if alignment and alignment.get("n"):
        _line(pdf, f"Judge vs human labels: {alignment['agreement'] * 100:.0f}% agreement over "
                   f"{alignment['n']} items (Cohen's kappa {alignment['kappa']}).")
    pdf.ln(3)

    _line(pdf, "Results", 13, "B", SKY, 7)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(30, 30, 45)
    with pdf.table(col_widths=(10, 25, 12, 12, 12, 70), line_height=4.5) as table:
        head = table.row()
        for h in ("ID", "Outcome", "Relev.", "Faith.", "Accur.", "Question"):
            head.cell(h)
        for r in run["reports"]:
            row = table.row()
            sc = r.get("scores", {})
            for value in (r["test_id"], CATEGORY_LABELS[r["category"]], _fmt(sc.get("relevance")),
                          _fmt(sc.get("faithfulness")), _fmt(sc.get("accuracy")), r["question"]):
                row.cell(clean(value))

    failures = [r for r in run["reports"] if r["category"] != "PASS"]
    if failures:
        pdf.add_page()
        _line(pdf, "Failure details", 13, "B", SKY, 7)
        for r in failures:
            pdf.ln(2)
            _line(pdf, f"{r['test_id']} - {CATEGORY_LABELS[r['category']]}", 11, "B", RED)
            _line(pdf, f"Question: {r['question']}", 9, "B")
            _line(pdf, f"Expected: {r['expected_answer']}", 9)
            _line(pdf, "Answer (red = not supported by the retrieved chunks):", 9, "I", GREY)
            for sent in r.get("sentences") or [{"text": r.get("answer", ""), "supported": True}]:
                _line(pdf, ("[UNSUPPORTED] " if not sent["supported"] else "") + sent["text"], 9,
                      color=RED if not sent["supported"] else (30, 30, 45))
            _line(pdf, f"Why: {r['explanation']}", 9, color=GREY)
            if r.get("retrieved"):
                _line(pdf, "Retrieved: " + "; ".join(
                    f"#{c['chunk_id']} {c['doc_name']} p.{c['page']} (score {c['score']}"
                    f"{', has answer' if c['is_gold'] else ''})" for c in r["retrieved"]), 8, color=GREY)
    return bytes(pdf.output())
