"""
frontend/components.py - Reusable display pieces for the dashboard (HTML + CSS + charts).

No checking or scoring happens here - these functions only turn API results into visuals.
Labels and thresholds come from the API (/meta) via init(), so nothing is imported from the backend.
"""

import html
import re

import altair as alt
import pandas as pd
import streamlit as st

LILAC, LILAC_DARK, SKY, SKY_DARK = "#C9B6F2", "#8E6CCF", "#9ED8F5", "#4FA8DA"

CATEGORY_COLORS = {
    "PASS": "#2E9E6A", "HALLUCINATION": "#D64545", "WRONG_CHUNK": "#E07B26", "STALE_SOURCE": "#A0622D",
    "REFUSAL": "#7B5CC4", "GENERATION_DRIFT": "#3C8DC4", "IGNORED_CONTEXT": "#C2477E",
    "INCOMPLETE": "#C9A100", "ERROR": "#7A7A8C",
}
CATEGORY_LABELS = {}
GENERATION_LABELS = []
LABEL_COLORS = {}
THRESHOLDS = {"good": 0.8, "ok": 0.5}


def init(meta):
    """Load labels and thresholds from the API's /meta answer."""
    CATEGORY_LABELS.update(meta["category_labels"])
    GENERATION_LABELS[:] = meta["generation_labels"]
    THRESHOLDS.update(meta["thresholds"])
    LABEL_COLORS.update({CATEGORY_LABELS[k]: v for k, v in CATEGORY_COLORS.items() if k in CATEGORY_LABELS})


_SCORE_STYLE = {
    "good": ("#E3F6EC", "#2E9E6A"), "ok": ("#FFF4D6", "#B07D00"),
    "bad": ("#FDE5E5", "#D64545"), "neutral": ("#F1EEF8", "#7A7A8C"),
}

CSS = f"""
<style>
.block-container {{ padding-top: 1.6rem; }}
.verity-hero {{
  background: linear-gradient(120deg, {LILAC} 0%, #DCD0F7 45%, {SKY} 100%);
  border-radius: 18px; padding: 22px 28px; margin-bottom: 18px;
  box-shadow: 0 6px 24px rgba(142,108,207,.18);
}}
.verity-hero h1 {{ margin: 0; font-size: 2.6rem; font-weight: 800; letter-spacing: -1px; color: #2E2A47; padding: 0; }}
.verity-hero h1 span {{ color: #5B3FA8; }}
.verity-hero p {{ margin: 4px 0 0 0; color: #3D3760; font-size: 1.02rem; }}
.verity-card {{ border-radius: 14px; padding: 14px 16px; border: 1px solid; text-align: center; height: 100%; }}
.verity-card .label {{ font-size: .82rem; text-transform: uppercase; letter-spacing: .06em; opacity: .85; }}
.verity-card .value {{ font-size: 2rem; font-weight: 800; line-height: 1.2; }}
.verity-card .hint {{ font-size: .78rem; opacity: .8; }}
.verity-pill {{ display: inline-block; padding: 2px 10px; border-radius: 999px; color: white;
  font-size: .8rem; font-weight: 600; white-space: nowrap; }}
.verity-box {{ border: 1px solid #E2DAF5; background: white; border-radius: 12px; padding: 12px 14px;
  height: 100%; font-size: .92rem; }}
.verity-box h4 {{ margin: 0 0 8px 0; font-size: .95rem; color: {LILAC_DARK}; }}
.verity-unsupported {{ background: #FDD5D5; color: #9E1C1C; border-bottom: 2px solid #D64545; padding: 1px 2px;
  border-radius: 3px; }}
.verity-supported {{ background: #E3F6EC; padding: 1px 2px; border-radius: 3px; }}
.verity-chunk {{ border-left: 4px solid {SKY_DARK}; background: #F5FAFE; border-radius: 6px;
  padding: 8px 10px; margin-bottom: 8px; font-size: .85rem; }}
.verity-chunk.gold {{ border-left-color: #2E9E6A; background: #F0FAF4; }}
.verity-chunk.stale {{ border-left-color: #A0622D; background: #FBF4EE; }}
.verity-chunk .meta {{ font-size: .75rem; color: #5D587A; margin-bottom: 4px; }}
.verity-banner {{ border-radius: 12px; padding: 10px 16px; margin: 6px 0 12px 0; font-weight: 600; }}
.verity-banner.break {{ background: #FDE5E5; color: #9E1C1C; border: 1px solid #F3B5B5; }}
.verity-banner.stable {{ background: #E3F6EC; color: #1F6F4A; border: 1px solid #A9DCC0; }}
.verity-banner.unstable {{ background: #FFF4D6; color: #7A5600; border: 1px solid #F0D58A; }}
.verity-banner.info {{ background: #EEF5FC; color: #24527A; border: 1px solid #C4DDF2; }}
div.stButton > button[kind="primary"] {{ background: linear-gradient(120deg, {LILAC_DARK}, {SKY_DARK}); border: none; }}
</style>
"""


def inject_theme():
    st.markdown(CSS, unsafe_allow_html=True)


def hero():
    st.markdown(
        "<div class='verity-hero'><h1>Verity<span>Ai</span></h1>"
        "<p>RAG evaluation pipeline: tests the retriever and the generator separately, "
        "and explains every failure.</p></div>",
        unsafe_allow_html=True,
    )


def fmt(value):
    return "—" if value is None else f"{value:.2f}"


def score_color(value):
    if value is None:
        return "neutral"
    return "good" if value >= THRESHOLDS["good"] else "ok" if value >= THRESHOLDS["ok"] else "bad"


def scorecard(label, value, hint=""):
    bg, fg = _SCORE_STYLE[score_color(value)]
    st.markdown(
        f"<div class='verity-card' style='background:{bg};border-color:{fg}55;color:{fg}'>"
        f"<div class='label'>{html.escape(label)}</div><div class='value'>{fmt(value)}</div>"
        f"<div class='hint'>{html.escape(hint)}</div></div>",
        unsafe_allow_html=True,
    )


def count_card(label, value, color, hint=""):
    st.markdown(
        f"<div class='verity-card' style='background:white;border-color:{color}55;color:{color}'>"
        f"<div class='label'>{html.escape(label)}</div><div class='value'>{value}</div>"
        f"<div class='hint'>{html.escape(hint)}</div></div>",
        unsafe_allow_html=True,
    )


def pill(category):
    color = CATEGORY_COLORS.get(category, "#7A7A8C")
    return f"<span class='verity-pill' style='background:{color}'>{CATEGORY_LABELS.get(category, category)}</span>"


def banner(text, kind):
    st.markdown(f"<div class='verity-banner {kind}'>{text}</div>", unsafe_allow_html=True)


def highlighted_answer(report):
    """The answer with unsupported claims in red - the visual proof of a hallucination."""
    sentences = report.get("sentences") or []
    if not sentences:
        return html.escape(report.get("answer", "")) or "<i>(no answer)</i>"
    parts = []
    for s in sentences:
        # Escape the text, then turn the model's **bold** markdown into real bold
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(s["text"]))
        if s["supported"]:
            tip = f"Supported by chunk #{s['chunk_id']}" if s.get("chunk_id") is not None else "No factual claim"
            if s.get("quote"):
                tip += f': "{s["quote"][:160]}"'
            parts.append(f"<span class='verity-supported' title='{html.escape(tip)}'>{text}</span>")
        else:
            tip = s.get("note") or "Not supported by any retrieved chunk"
            parts.append(f"<span class='verity-unsupported' title='{html.escape(tip)}'>{text}</span>")
    return " ".join(parts)


def chunk_html(c):
    kind = "gold" if c.get("is_gold") else "stale" if not c.get("is_current", True) else ""
    tags = []
    if c.get("is_gold"):
        tags.append("✅ holds the answer")
    if not c.get("is_current", True):
        tags.append(f"🕰️ archived {c.get('doc_year') or ''}")
    meta = (f"Rank #{c['rank']} · score {c['score']} · {html.escape(c['doc_name'])} → Page {c['page']} → "
            f"Chunk #{c['chunk_id']}" + (" · " + " · ".join(tags) if tags else ""))
    return f"<div class='verity-chunk {kind}'><div class='meta'>{meta}</div>{html.escape(c['text'])}</div>"


def diagnostic_triad(report):
    """Input query | Generated answer | Retrieved chunks, side by side."""
    q, a, c = st.columns([1, 1.15, 1.35])
    with q:
        location = (f"{report['doc_name']} → Page {report['page']}" if report.get("answerable")
                    else "Not in the documents")
        st.markdown(
            f"<div class='verity-box'><h4>🔎 Input query</h4><b>{html.escape(report['question'])}</b>"
            f"<hr style='margin:8px 0'><small><b>Expected answer</b></small><br>"
            f"{html.escape(report.get('expected_answer') or '')}"
            f"<br><br><small><b>Answer lives in</b><br>{html.escape(location)}"
            + (f" (source chunk #{report['source_chunk_id']})" if report.get("source_chunk_id") is not None else "")
            + "</small></div>", unsafe_allow_html=True)
    with a:
        legend = ("<small><span class='verity-unsupported'>red</span> = claim not supported by the chunks · "
                  "<span class='verity-supported'>green</span> = supported (hover for evidence)</small>")
        st.markdown(f"<div class='verity-box'><h4>🤖 Generated answer</h4>{highlighted_answer(report)}"
                    f"<hr style='margin:8px 0'>{legend}</div>", unsafe_allow_html=True)
    with c:
        chunks = "".join(chunk_html(ch) for ch in report.get("retrieved", [])) or "<i>No chunks retrieved</i>"
        st.markdown(f"<div class='verity-box'><h4>📚 Retrieved chunks</h4>{chunks}</div>", unsafe_allow_html=True)


def confusion_matrix(reports):
    """Retrieval verdict (rows) x generation verdict (columns), counted over all tests."""
    rows = ["Right chunk", "Wrong chunk"] + (["Not in docs"] if any(r["retrieval"] == "Not in docs" for r in reports) else [])
    data = [{"Retrieval": rw, "Generation": col,
             "Tests": sum(1 for r in reports if r.get("retrieval") == rw and r.get("generation") == col)}
            for rw in rows for col in GENERATION_LABELS]
    df = pd.DataFrame(data)
    base = alt.Chart(df).encode(
        x=alt.X("Generation:N", sort=GENERATION_LABELS, title="What the generator did", axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("Retrieval:N", sort=rows, title="What the retriever did"),
    )
    heat = base.mark_rect(cornerRadius=6, stroke="white", strokeWidth=3).encode(
        color=alt.Color("Tests:Q", scale=alt.Scale(range=["#F4F0FC", LILAC_DARK]), legend=None),
        tooltip=["Retrieval", "Generation", "Tests"])
    text = base.mark_text(fontSize=16, fontWeight="bold").encode(
        text=alt.Text("Tests:Q"),
        color=alt.condition(alt.datum.Tests > 0, alt.value("#2E2A47"), alt.value("#C9C2DD")))
    return (heat + text).properties(height=60 * len(rows) + 60, width="container")


def failure_bar(summary):
    data = [{"Cause": k, "Tests": v} for k, v in summary["categories"].items() if k != "Pass"]
    if not data:
        return None
    return alt.Chart(pd.DataFrame(data)).mark_bar(cornerRadiusEnd=6).encode(
        x=alt.X("Tests:Q", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("Cause:N", sort="-x", title=None),
        color=alt.Color("Cause:N", scale=alt.Scale(domain=list(LABEL_COLORS), range=list(LABEL_COLORS.values())),
                        legend=None),
        tooltip=["Cause", "Tests"],
    ).properties(height=40 * len(data) + 30, width="container")


def bar_chart(df, x, y, color=None, fmt_x=None, height=None):
    enc = {"x": alt.X(f"{x}:Q", axis=alt.Axis(format=fmt_x) if fmt_x else alt.Axis()),
           "y": alt.Y(f"{y}:N", sort="-x", title=None), "tooltip": list(df.columns)}
    enc["color"] = alt.value(color or LILAC_DARK)
    return alt.Chart(df).mark_bar(cornerRadiusEnd=5).encode(**enc).properties(
        height=height or 34 * len(df) + 30, width="container")


def score_columns():
    return {name: st.column_config.ProgressColumn(name, min_value=0.0, max_value=1.0, format="%.2f")
            for name in ("Relevance", "Faithfulness", "Accuracy", "Score", "Hit@k", "MRR")}


def threshold_note():
    return (f"Green ≥ {THRESHOLDS['good']} · amber ≥ {THRESHOLDS['ok']} · red below. "
            "Relevance = source chunk in top-k · Faithfulness = share of claims backed by chunks · "
            "Accuracy = matches the expected answer (1 / 0.5 / 0).")
