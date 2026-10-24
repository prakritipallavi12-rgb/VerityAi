"""
app.py - VerityAi dashboard (frontend).

Start everything with:   python start.py      (API on :8000 + dashboard on :8501)

This file only shows results and collects settings. Every calculation happens in the FastAPI
backend (api/server.py); this dashboard talks to it over HTTP through frontend/api.py.
"""

import hashlib
import html
import json

import altair as alt
import pandas as pd
import streamlit as st

from frontend import api
from frontend import components as ui

st.set_page_config(page_title="VerityAi", page_icon="🔍", layout="wide")
ui.inject_theme()
ss = st.session_state


# ======================================================================
# Talk to the API
# ======================================================================
@st.cache_data(ttl=600, show_spinner=False)
def load_meta():
    return api.get("/meta")


@st.cache_data(ttl=3600, show_spinner=False)
def load_models():
    return api.get("/models")


@st.cache_data(show_spinner=False, max_entries=64)
def fetch_run(run_id, labels_version):
    return api.get(f"/runs/{run_id}")


@st.cache_data(show_spinner=False, max_entries=32)
def fetch_pdf(run_id, labels_version):
    return api.get_bytes(f"/runs/{run_id}/pdf")


try:
    META, MODELS = load_meta(), load_models()
except api.ApiError as e:
    ui.hero()
    st.error(f"{e}")
    st.code("python start.py", language="bash")
    st.stop()

ui.init(META)
OPTIONS = MODELS["options"]
if not OPTIONS:
    ui.hero()
    st.error("No LLM provider is configured. Add your key to .env (see .env.example) and restart.")
    st.stop()
D = META["defaults"]
VERSIONS = list(META["versions"])
CAT = META["category_labels"]


def label(option):
    return MODELS["labels"].get(option, option)


def pick(default):
    return default if default in OPTIONS else OPTIONS[0]


DEFAULTS = {
    "chunk_size": D["chunk_size"], "overlap": D["overlap"], "top_k": D["top_k"], "version": META["default_version"],
    "gen_model": pick(D["generator_model"]), "judge_model": pick(D["judge_model"]), "judge_mode": D["judge_mode"],
    "shuffle": False, "auto_rerun": False, "n_tests": D["n_tests"], "workers": D["workers"],
    "use_sample": True, "use_benchmark": False, "current_run_id": None, "ab": None, "leaderboard": None,
    "break_mode": False, "pending_run": False, "last_signature": None, "pending_custom": "", "docs_key": None,
    "docs_sig": None, "docs_info": None, "labels_version": 0, "stability": None,
    "ab_version_a": META["default_version"], "ab_version_b": "v2.1 Hybrid + Rerank" if "v2.1 Hybrid + Rerank" in VERSIONS else VERSIONS[-1],
    "ab_gen_a": pick(D["generator_model"]), "ab_gen_b": pick(D["generator_model"]),
}
for key, value in DEFAULTS.items():
    ss.setdefault(key, value)


def break_it():
    """'Break it' demo: force bad settings, then run so judges watch the failures get caught."""
    b = META["break_it"]
    ss.chunk_size, ss.overlap, ss.top_k, ss.version = b["chunk_size"], 5, b["top_k"], b["version"]
    ss.break_mode = True
    ss.pending_run = True


def restore():
    for key in ("chunk_size", "overlap", "top_k", "version"):
        ss[key] = DEFAULTS[key]
    ss.break_mode = False


def queue_custom_question():
    ss.pending_custom = ss.custom_q.strip()
    ss.custom_q = ""


# ======================================================================
# Sidebar: settings (every change reruns the app and recalculates)
# ======================================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.slider("Chunk size (words)", 30, 400, step=10, key="chunk_size",
              help="Smaller chunks = more precise search, but less context in each chunk.")
    ss.overlap = min(ss.overlap, ss.chunk_size // 2)
    st.slider("Chunk overlap (words)", 0, ss.chunk_size // 2, step=5, key="overlap")
    st.slider("Top-k (chunks given to the generator)", 1, 8, key="top_k")
    st.slider("Tests to generate", 3, 200, key="n_tests", help="Up to 200 - use the benchmark catalog for big runs.")
    st.divider()
    st.radio("Judge", ["llm", "local"], key="judge_mode", horizontal=True,
             format_func=lambda m: "🧠 LLM judge" if m == "llm" else "💻 Local judge",
             help="LLM judge: most detailed, uses API tokens. Local judge: small models on this laptop grade "
                  "the answers - no API limits, for big runs (200 questions).")
    st.slider("Parallel tests", 1, 24, key="workers",
              help="How many tests run at once. A rate limiter keeps calls within your API plan's limits.")
    st.toggle("Shuffle chunks before judging", key="shuffle",
              help="Judge every answer twice: normally and with the chunks in random order. A fair judge gives "
                   "the same score both times.")
    st.toggle("Auto re-run when settings change", key="auto_rerun",
              help="After the first run, any settings change re-runs the evaluation automatically.")
    if ss.break_mode:
        st.button("🔧 Restore normal settings", on_click=restore, width="stretch")
    st.divider()
    st.caption(ui.threshold_note())
    st.caption(f"API: {api.API_URL} · [endpoints]({api.API_URL}/docs)")

# ======================================================================
# Header + inputs
# ======================================================================
ui.hero()
if ss.break_mode:
    ui.banner(f"💥 Break-it mode: chunk size {ss.chunk_size}, top-k {ss.top_k}, {ss.version}. "
              "These settings are bad on purpose - watch VerityAi catch the failures. "
              "Use “Restore normal settings” in the sidebar to undo.", "break")

left, right = st.columns([1.15, 1], gap="large")
with left:
    st.markdown("#### 📄 Documents")
    uploaded = st.file_uploader("Drop PDF or TXT files here", type=["pdf", "txt"], accept_multiple_files=True)
    c_s, c_b = st.columns(2)
    c_s.checkbox("Sample: Nimbus handbook 2025 + archived 2023", key="use_sample")
    c_b.checkbox("Benchmark: product catalog (200 facts)", key="use_benchmark",
                 help="50 products × 4 facts - for 200-question speed runs.")
with right:
    st.markdown("#### 🧠 System under test")
    st.selectbox("RAG model version", VERSIONS, key="version",
                 format_func=lambda v: f"{v} — {META['versions'][v]}")
    g, j = st.columns(2)
    g.selectbox("Generator LLM (answers)", OPTIONS, key="gen_model", format_func=label)
    j.selectbox("Judge LLM (grades)", OPTIONS, key="judge_model", format_func=label,
                disabled=ss.judge_mode == "local", help="Not used while the local judge is selected.")
    missing = [p for p, ok in MODELS["providers"].items() if not ok]
    if missing:
        st.caption("Not connected: " + ", ".join(missing) + ". Add OPENAI_API_KEY / OPENROUTER_API_KEY (free Llama) "
                   "to .env, or install Ollama for local Llama - the evaluator works with any provider.")

# ---- Upload documents to the API (only when they change) ----
files = [(f.name, f.getvalue()) for f in (uploaded or [])]
docs_sig = hashlib.sha1(json.dumps([[n, hashlib.sha1(d).hexdigest()] for n, d in files]
                                   + [ss.use_sample, ss.use_benchmark]).encode()).hexdigest()
if not files and not ss.use_sample and not ss.use_benchmark:
    st.info("Upload a document or tick one of the sample document sets to begin.")
    st.stop()
if ss.docs_sig != docs_sig:
    with st.spinner("Reading documents…"):
        try:
            info = api.post("/documents", files=[("files", (n, d)) for n, d in files],
                            data={"use_sample": str(ss.use_sample).lower(), "use_benchmark": str(ss.use_benchmark).lower()})
        except api.ApiError as e:
            st.error(str(e))
            st.stop()
    ss.docs_key, ss.docs_sig, ss.docs_info = info["docs_key"], docs_sig, info
for error in ss.docs_info.get("errors", []):
    st.error(error)
DOCS = {"docs_key": ss.docs_key, "chunk_size": ss.chunk_size, "overlap": ss.overlap}


def tests_now():
    return api.get("/tests", docs_key=ss.docs_key)["tests"]  # each document set has its own tests


def signature(tests):
    return json.dumps([ss.docs_key, ss.chunk_size, ss.overlap, ss.top_k, ss.version, ss.gen_model, ss.judge_model,
                       ss.judge_mode, ss.shuffle, [t["question"] for t in tests]])


# ======================================================================
# Test controls: generate | ask your own | run | break it
# ======================================================================
st.markdown("#### 🧪 Test cases")
c1, c2, c3, c4, c5 = st.columns([1.15, 2.6, 0.75, 1, 1])
generate_clicked = c1.button(f"✨ Generate {ss.n_tests} tests", width="stretch")
c2.text_input("Ask your own test question", key="custom_q", label_visibility="collapsed",
              placeholder="Type your own test question, e.g. “How many sick days do I get?”")
c3.button("➕ Add", on_click=queue_custom_question, width="stretch")
run_clicked = c4.button("▶️ Run tests", type="primary", width="stretch")
c5.button("💥 Break it", on_click=break_it, width="stretch",
          help=f"Forces bad settings (chunk size {META['break_it']['chunk_size']}, top-k {META['break_it']['top_k']}, "
               f"{META['break_it']['version']}) and runs the tests.")
status_slot = st.empty()


def show_progress(job):
    bar = status_slot.progress(job["done"] / job["total"] if job["total"] else 0.0,
                               text=f"{job['message']}  ·  {job['elapsed']:.0f} s")
    return bar


def do_generate():
    try:
        job = api.post("/jobs/generate", json={**DOCS, "n": ss.n_tests, "model": ss.judge_model if ss.judge_mode == "llm"
                                               else ss.gen_model})
        result = api.wait_for_job(job["job_id"], show_progress)
        status_slot.success(f"Generated {result['tests']} tests. Press ▶️ Run tests.")
    except api.ApiError as e:
        status_slot.error(f"Could not generate tests: {e}")


def start_evaluation(configs, label="", kind="eval"):
    """Start an evaluation job on the API and show its progress. Returns {name: run_id} or None."""
    try:
        job = api.post("/jobs/evaluate", json={
            **DOCS, "top_k": ss.top_k, "judge_model": ss.judge_model, "judge_mode": ss.judge_mode,
            "shuffle_check": ss.shuffle, "workers": ss.workers, "configs": configs, "label": label, "kind": kind})
        result = api.wait_for_job(job["job_id"], show_progress)
        status_slot.empty()
        return result["run_ids"]
    except api.ApiError as e:
        status_slot.error(f"Evaluation failed: {e}")
        return None


if generate_clicked:
    do_generate()

if ss.pending_custom:
    question, ss.pending_custom = ss.pending_custom, ""
    with st.spinner("Finding where the answer to your question lives…"):
        try:
            test = api.post("/tests/custom", json={**DOCS, "question": question,
                                                   "model": ss.judge_model if ss.judge_mode == "llm" else ss.gen_model})["test"]
            where = f"{test['doc_name']} → Page {test['page']}" if test["answerable"] else "not in the documents"
            status_slot.success(f"Added {test['test_id']}: answer is {where}.")
        except api.ApiError as e:
            status_slot.error(f"Could not add the question: {e}")

TESTS = tests_now()
auto = ss.auto_rerun and TESTS and ss.current_run_id and ss.last_signature != signature(TESTS)
if run_clicked or ss.pending_run or auto:
    from_break = ss.pending_run
    ss.pending_run = False
    if not TESTS and from_break:
        do_generate()
        TESTS = tests_now()
    if not TESTS:
        status_slot.warning("No tests yet - press ✨ Generate tests or add your own question first.")
    else:
        run_ids = start_evaluation([{"name": "main", "version": ss.version, "generator_model": ss.gen_model}],
                                   label="💥 Break it · {version}" if ss.break_mode else "")
        ss.last_signature = signature(TESTS)
        if run_ids:
            ss.current_run_id = run_ids["main"]

# ---- Live retrieval check: instant, no LLM calls - moves with the sliders ----
if TESTS:
    try:
        live = api.post("/retrieval/live", json={**DOCS, "top_k": ss.top_k, "version": ss.version})
    except api.ApiError as e:
        live = {"hit_rate": None, "chunks": "?"}
        st.warning(str(e))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tests ready", len(TESTS), help="See the 🧪 Test set tab")
    m2.metric("Chunks", live["chunks"], help=f"{ss.chunk_size}-word chunks, {ss.overlap}-word overlap")
    m3.metric(f"Live retrieval hit-rate @ top-{ss.top_k}", "—" if live["hit_rate"] is None else f"{live['hit_rate']:.0%}",
              help="Share of answerable tests whose source chunk is in the top-k. Recalculated instantly when you "
                   "move a slider - no LLM calls.")
    if ss.current_run_id:
        last = fetch_run(ss.current_run_id, ss.labels_version)["summary"]
        m4.metric("Last run pass rate", f"{last['pass_rate']:.0%}")
        if ss.last_signature != signature(TESTS):
            st.caption("⚡ Settings changed since the last run - press ▶️ Run tests (or turn on auto re-run) "
                       "to see the new scores.")


# ======================================================================
# Rendering helpers
# ======================================================================
def save_label_and_refresh(question, answer, verdict, category="", labeler=""):
    api.post("/labels", json={"question": question, "answer": answer, "verdict": verdict,
                              "category": category, "labeler": labeler})
    ss.labels_version += 1


def render_detail(r, prefix, run_id):
    st.markdown(f"#### {r['test_id']} &nbsp; {ui.pill(r['category'])}", unsafe_allow_html=True)
    if r["category"] == "PASS":
        st.success(r["explanation"])
    elif r["category"] == "ERROR":
        st.warning(r["explanation"])
        return
    else:
        st.error(f"**Why it failed:** {r['explanation']}")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        ui.scorecard("Relevance", r["scores"].get("relevance"), "source chunk in top-k")
    with s2:
        ui.scorecard("Faithfulness", r["scores"].get("faithfulness"), "claims backed by chunks")
    with s3:
        ui.scorecard("Accuracy", r["scores"].get("accuracy"), "matches expected answer")
    with s4:
        u = r.get("usage", {})
        ui.count_card("Cost & time", f"{u.get('seconds', 0):.1f}s", ui.SKY_DARK,
                      f"{u.get('tokens', 0):,} tokens · ${u.get('cost', 0):.5f}")

    st.markdown("##### The diagnostic triad")
    ui.diagnostic_triad(r)

    stab = r.get("stability", {})
    if stab.get("checked"):
        if stab["unchanged"]:
            ui.banner("✅ Stability verified: Score unchanged after chunk shuffle "
                      f"(judge saw order {stab['shuffled_order']}).", "stable")
        else:
            ui.banner(f"⚠️ Score changed by {stab['max_difference']} after shuffling the chunks - "
                      "the judge may be biased by chunk position.", "unstable")

    al = r.get("alignment", {"labelled": False, "judge": "PASS" if r["category"] == "PASS" else "FAIL"})
    if al["labelled"]:
        who = f" ({al['labeler']})" if al.get("labeler") else ""
        st.markdown(f"**Inter-rater alignment:** automated judge = **{al['judge']}** · human label{who} = "
                    f"**{al['human']}**{' / ' + al['human_category'] if al.get('human_category') else ''} → "
                    f"{'✅ match' if al['agree'] else '❌ mismatch'}")
    else:
        st.markdown(f"**Inter-rater alignment:** automated judge = **{al['judge']}** · human label = *not labelled yet*. "
                    "Add your verdict below, or use the ⚖️ Judge reliability tab.")
    with st.form(key=f"{prefix}_label_{run_id}_{r['test_id']}"):
        f1, f2, f3, f4 = st.columns([1, 1.3, 1.2, 0.8])
        verdict = f1.radio("Human verdict", ["PASS", "FAIL"], horizontal=True)
        category = f2.selectbox("Cause (if failed)", [""] + [v for k, v in CAT.items() if k not in ("PASS", "ERROR")])
        labeler = f3.text_input("Your name", placeholder="optional")
        if f4.form_submit_button("💾 Save label", width="stretch"):
            save_label_and_refresh(r["question"], r.get("answer", ""), verdict, category, labeler)
            st.toast("Label saved")
            st.rerun()

    with st.expander("Judge's reasoning and missing parts"):
        st.write(r.get("judge_reasoning") or "—")
        if r.get("missing_parts"):
            st.write("Missing from the answer:", r["missing_parts"])


def cost_time_panel(s):
    st.markdown("##### ⏱️ Cost & time")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        ui.count_card("Tokens used", f"{s.get('tokens', 0):,}", ui.LILAC_DARK, f"{s.get('tokens_per_question', 0):,} per question")
    with t2:
        ui.count_card("Est. cost", f"${s.get('cost', 0):.4f}", ui.LILAC_DARK, "at list prices (free tier = $0)")
    with t3:
        ui.count_card("Seconds / question", ui.fmt(s.get("seconds_per_question")), ui.SKY_DARK, "one test, start to end")
    with t4:
        qpm = s.get("questions_per_minute")
        ui.count_card("Throughput", f"{qpm} /min" if qpm else "—", ui.SKY_DARK,
                      f"{s['wall_seconds']} s for {s['total']} tests" if s.get("wall_seconds") else "")
    stages = pd.DataFrame([{"Stage": n, "Seconds": s.get(k) or 0} for n, k in
                           (("Retrieve", "retrieve_s"), ("Generate", "generate_s"), ("Judge", "judge_s"))])
    st.altair_chart(ui.bar_chart(stages, "Seconds", "Stage", ui.SKY_DARK, height=120), width="stretch")
    st.caption("Average seconds per question for each step. Tests run in parallel, so the whole run takes much "
               "less than (seconds per question × tests).")


def render_run(run, prefix):
    s, reports, cfg = run["summary"], run["reports"], run["config"]
    judge = "💻 local judge" if cfg.get("judge_mode") == "local" else label(cfg["judge_model"])
    st.caption(f"Run {run['timestamp']} · {cfg['version']} · generator {label(cfg['generator_model'])} · judge {judge} · "
               f"chunk {cfg['chunk_size']}/{cfg['overlap']} · top-k {cfg['top_k']} · "
               f"shuffle check {'on' if cfg['shuffle_check'] else 'off'}")
    if run["label"].startswith("💥"):
        ui.banner("💥 This run used the Break-it settings.", "break")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        ui.count_card("Test cases", s["total"], ui.LILAC_DARK)
    with k2:
        ui.count_card("Passed", s["passed"], "#2E9E6A")
    with k3:
        ui.count_card("Failed", s["failed"], "#D64545")
    with k4:
        ui.count_card("Pass rate", f"{s['pass_rate']:.0%}", ui.SKY_DARK)
    st.write("")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        ui.scorecard("Relevance", s["relevance"], "average · retriever")
    with a2:
        ui.scorecard("Faithfulness", s["faithfulness"], "average · no made-up claims")
    with a3:
        ui.scorecard("Accuracy", s["accuracy"], "average · correct answer")
    with a4:
        ui.scorecard("MRR", s.get("mrr"), "how high the answer chunk ranks")
    if s["errors"]:
        st.warning(f"{s['errors']} test(s) could not run (API error). See their rows below.")

    if s["stability_checked"]:
        if s["stability_unchanged"] == s["stability_checked"]:
            ui.banner(f"✅ Stability verified: Score unchanged after chunk shuffle "
                      f"({s['stability_unchanged']}/{s['stability_checked']} tests).", "stable")
        else:
            ui.banner(f"⚠️ {s['stability_checked'] - s['stability_unchanged']} of {s['stability_checked']} tests "
                      "scored differently after the chunks were shuffled.", "unstable")

    cost_time_panel(s)

    ch1, ch2 = st.columns([1, 1.6], gap="large")
    with ch1:
        st.markdown("##### Causes of failure")
        bar = ui.failure_bar(s)
        if bar is None:
            st.success("No failures 🎉")
        else:
            st.altair_chart(bar, width="stretch")
    with ch2:
        st.markdown("##### Confusion matrix: retriever × generator")
        st.altair_chart(ui.confusion_matrix([r for r in reports if r["category"] != "ERROR"]), width="stretch")

    st.markdown("##### All test cases — click a row to inspect it")
    table = pd.DataFrame([{
        "": "✅" if r["category"] == "PASS" else "⚠️" if r["category"] == "ERROR" else "❌",
        "ID": r["test_id"], "Outcome": CAT[r["category"]], "Question": r["question"],
        "Relevance": r["scores"].get("relevance"), "Faithfulness": r["scores"].get("faithfulness"),
        "Accuracy": r["scores"].get("accuracy"),
        "Seconds": r.get("usage", {}).get("seconds"), "Tokens": r.get("usage", {}).get("tokens"),
        "Shuffle": ("✓ stable" if r["stability"]["unchanged"] else "✗ changed") if r["stability"].get("checked") else "—",
        "Human": (("✓ " if r["alignment"]["agree"] else "✗ ") + r["alignment"]["human"])
        if r.get("alignment", {}).get("labelled") else "—",
    } for r in reports])
    event = st.dataframe(table, hide_index=True, on_select="rerun", selection_mode="single-row",
                         key=f"{prefix}_table_{run['run_id']}", column_config=ui.score_columns(), width="stretch")
    rows = event.selection.rows
    index = rows[0] if rows else next((i for i, r in enumerate(reports) if r["category"] != "PASS"), 0)
    if not rows:
        st.caption("Showing the first failed test - click any row to switch.")
    with st.container(border=True):
        render_detail(reports[index], prefix, run["run_id"])

    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button("⬇️ Download PDF report", fetch_pdf(run["run_id"], ss.labels_version),
                       f"verityai_{run['run_id']}.pdf", "application/pdf", key=f"{prefix}_pdf_{run['run_id']}",
                       width="stretch")
    d2.download_button("⬇️ Download run (JSON)", json.dumps(run, indent=1, ensure_ascii=False),
                       f"verityai_{run['run_id']}.json", "application/json",
                       key=f"{prefix}_json_{run['run_id']}", width="stretch")


def render_comparison(run_a_id, run_b_id, prefix):
    data = api.get("/compare", a=run_a_id, b=run_b_id)
    run_a, run_b, rows = data["a"], data["b"], data["rows"]
    sa, sb = run_a["summary"], run_b["summary"]
    left_col, right_col = st.columns(2, gap="large")
    for col, run, s, name in ((left_col, run_a, sa, "A"), (right_col, run_b, sb, "B")):
        with col:
            st.markdown(f"##### Version {name}: {run['config']['version']}")
            st.caption(f"{label(run['config']['generator_model'])} · {run['timestamp']}")
            x1, x2, x3, x4 = st.columns(4)
            with x1:
                ui.count_card("Pass rate", f"{s['pass_rate']:.0%}", ui.LILAC_DARK)
            with x2:
                ui.scorecard("Relevance", s["relevance"])
            with x3:
                ui.scorecard("Faithful", s["faithfulness"])
            with x4:
                ui.scorecard("Accuracy", s["accuracy"])

    deltas = []
    for key, name in (("pass_rate", "Pass rate"), ("relevance", "Relevance"),
                      ("faithfulness", "Faithfulness"), ("accuracy", "Accuracy")):
        if sa.get(key) is not None and sb.get(key) is not None:
            deltas.append(f"{name} {'🔻' if sb[key] < sa[key] else '🔺' if sb[key] > sa[key] else '='} "
                          f"{sb[key] - sa[key]:+.2f}")
    st.caption("B vs A: " + " · ".join(deltas))

    regressions = [r for r in rows if r["regression"]]
    if regressions:
        ui.banner(f"🔻 {len(regressions)} quer{'y' if len(regressions) == 1 else 'ies'} got WORSE in version B: "
                  + ", ".join(r["test_id"] for r in regressions[:20]) + ("…" if len(regressions) > 20 else ""), "break")
    else:
        ui.banner("✅ No regressions: version B is at least as good as A on every query.", "stable")

    table = pd.DataFrame([{
        "": "🔻" if r["regression"] else "🔺" if r["improvement"] else "", "ID": r["test_id"],
        "Question": r["question"], "A outcome": r["a_category"], "A score": r["a_score"],
        "B outcome": r["b_category"], "B score": r["b_score"], "Δ": r["delta"],
    } for r in rows])
    if not table.empty:
        styled = table.style.apply(
            lambda row: ["background-color:#FDE5E5;color:#9E1C1C;font-weight:600" if row[""] == "🔻" else ""] * len(row),
            axis=1).format({"Δ": "{:+.2f}", "A score": "{:.2f}", "B score": "{:.2f}"})
        st.dataframe(styled, hide_index=True, width="stretch")
    st.caption("🔻 = B performed worse than A on this query (row in red) · 🔺 = B improved")
    if rows:
        first_regression = next((i for i, r in enumerate(rows) if r["regression"]), 0)
        picked = st.selectbox("Compare the answers for", range(len(rows)), index=first_regression,
                              key=f"{prefix}_cmp_{run_a_id}_{run_b_id}",
                              format_func=lambda i: f"{'🔻 ' if rows[i]['regression'] else ''}{rows[i]['test_id']}: "
                                                    f"{rows[i]['question']}")
        r = rows[picked]
        with st.container(border=True):
            pa, pb = st.columns(2, gap="large")
            for col, rep, name in ((pa, r["a_report"], "A"), (pb, r["b_report"], "B")):
                with col:
                    st.markdown(f"Version {name} &nbsp; {ui.pill(rep['category'])} &nbsp; score {rep['overall']:.2f}",
                                unsafe_allow_html=True)
                    st.markdown(f"<div class='verity-box'>{ui.highlighted_answer(rep)}</div>", unsafe_allow_html=True)
                    st.caption(rep["explanation"])


# ======================================================================
# Tabs
# ======================================================================
(tab_results, tab_ab, tab_board, tab_retr, tab_rel, tab_history, tab_tests, tab_docs) = st.tabs(
    ["📊 Results", "🆚 A/B compare", "🏆 Model leaderboard", "🔎 Retrievers", "⚖️ Judge reliability",
     "🕘 Run history", "🧪 Test set", "📄 Documents & chunks"])

with tab_results:
    if ss.current_run_id:
        render_run(fetch_run(ss.current_run_id, ss.labels_version), "cur")
    else:
        st.info("No results yet. ✨ Generate tests (or add your own question), then press ▶️ Run tests.")

with tab_ab:
    st.markdown("Run two versions **at the same time** on the **exact same tests**. "
                "Version B is treated as the upgrade: any query where B does worse gets a 🔻.")
    va, vb = st.columns(2, gap="large")
    with va:
        st.markdown("##### Version A (current)")
        st.selectbox("RAG version", VERSIONS, key="ab_version_a")
        st.selectbox("Generator LLM", OPTIONS, key="ab_gen_a", format_func=label)
    with vb:
        st.markdown("##### Version B (upgrade)")
        st.selectbox("RAG version ", VERSIONS, key="ab_version_b")
        st.selectbox("Generator LLM ", OPTIONS, key="ab_gen_b", format_func=label)
    if st.button(f"▶️ Run A vs B on the same {len(TESTS)} tests", type="primary", disabled=not TESTS):
        run_ids = start_evaluation([{"name": "A", "version": ss.ab_version_a, "generator_model": ss.ab_gen_a},
                                    {"name": "B", "version": ss.ab_version_b, "generator_model": ss.ab_gen_b}],
                                   label="A/B {name} · {version} · {model}", kind="ab")
        if run_ids:
            ss.ab = run_ids
    if ss.ab:
        render_comparison(ss.ab["A"], ss.ab["B"], "ab")
    elif not TESTS:
        st.info("Generate or add tests first.")

with tab_board:
    st.markdown("Run **several generator models** on the same tests and see which one **hallucinates least**. "
                "Add OpenAI, OpenRouter (free Llama) or Ollama (local Llama) keys in `.env` to include them.")
    b1, b2 = st.columns([2, 1])
    board_models = b1.multiselect("Generator models", OPTIONS, default=OPTIONS[:min(3, len(OPTIONS))],
                                  format_func=label, key="board_models")
    board_version = b2.selectbox("RAG version", VERSIONS, key="board_version")
    if st.button(f"🏁 Run leaderboard ({len(board_models)} models × {len(TESTS)} tests)", type="primary",
                 disabled=not (TESTS and board_models)):
        run_ids = start_evaluation([{"name": m, "version": board_version, "generator_model": m} for m in board_models],
                                   label="🏆 {model}", kind="leaderboard")
        if run_ids:
            ss.leaderboard = run_ids
    if ss.leaderboard:
        board = []
        for model, run_id in ss.leaderboard.items():
            s = fetch_run(run_id, ss.labels_version)["summary"]
            board.append({"Model": label(model), "Hallucination rate": s.get("hallucination_rate"),
                          "Pass rate": s["pass_rate"], "Faithfulness": s["faithfulness"], "Accuracy": s["accuracy"],
                          "Tokens / q": s.get("tokens_per_question"), "Seconds / q": s.get("seconds_per_question"),
                          "Est. cost $": s.get("cost")})
        board_df = pd.DataFrame(board).sort_values(["Hallucination rate", "Pass rate"], ascending=[True, False])
        best = board_df.iloc[0]
        ui.banner(f"🏆 Hallucinates least: <b>{html.escape(best['Model'])}</b> "
                  f"({best['Hallucination rate']:.0%} of answers had made-up claims)", "stable")
        st.dataframe(board_df, hide_index=True, width="stretch", column_config={
            **ui.score_columns(),
            "Hallucination rate": st.column_config.ProgressColumn("Hallucination rate", min_value=0.0, max_value=1.0,
                                                                  format="percent"),
            "Pass rate": st.column_config.ProgressColumn("Pass rate", min_value=0.0, max_value=1.0, format="percent")})
        st.altair_chart(ui.bar_chart(board_df[["Model", "Hallucination rate"]], "Hallucination rate", "Model",
                                     "#D64545", fmt_x="%"), width="stretch")

with tab_retr:
    st.markdown("Every retriever is graded on the same tests **without any LLM calls**: is the source chunk in the "
                "top-k (Hit@k), and how high is it ranked (MRR)? Each one runs with and without the **re-ranker**.")
    if not TESTS:
        st.info("Generate or add tests first.")
    elif st.button("🔎 Compare retrievers", type="primary") or ss.get("retr_rows"):
        if not ss.get("retr_rows") or ss.get("retr_sig") != (ss.docs_key, ss.chunk_size, ss.overlap, ss.top_k, len(TESTS)):
            with st.spinner("Ranking every test question with every retriever…"):
                ss.retr_rows = api.post("/retrieval/compare", json={**DOCS, "top_k": ss.top_k})["rows"]
                ss.retr_sig = (ss.docs_key, ss.chunk_size, ss.overlap, ss.top_k, len(TESTS))
        rdf = pd.DataFrame([{"Retriever": r["retriever"], "Hit@k": r["hit_at_k"], "MRR": r["mrr"],
                             "Avg rank": r["avg_rank"], "ms / query": r["ms_per_query"],
                             "Re-ranker gain (MRR)": r.get("rerank_gain_mrr")} for r in ss.retr_rows])
        best = rdf.sort_values(["MRR", "Hit@k"], ascending=False).iloc[0]
        ui.banner(f"🥇 Best retriever here: <b>{best['Retriever']}</b> (Hit@{ss.top_k} {best['Hit@k']:.0%}, "
                  f"MRR {best['MRR']:.2f})", "stable")
        gains = [r for r in ss.retr_rows if r["rerank"]]
        st.caption(" · ".join(f"Re-ranker on {r['retriever'].replace(' + re-ranker', '')}: MRR "
                              f"{'🔺' if r['rerank_gain_mrr'] > 0 else '🔻' if r['rerank_gain_mrr'] < 0 else '='} "
                              f"{r['rerank_gain_mrr']:+.2f}" for r in gains))
        st.dataframe(rdf, hide_index=True, width="stretch", column_config=ui.score_columns())
        cmp = rdf.melt(id_vars="Retriever", value_vars=["Hit@k", "MRR"], var_name="Metric", value_name="Score")
        st.altair_chart(alt.Chart(cmp).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("Retriever:N", title=None, sort=list(rdf["Retriever"])), x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1])),
            yOffset="Metric:N", color=alt.Color("Metric:N", scale=alt.Scale(range=[ui.LILAC_DARK, ui.SKY_DARK])),
            tooltip=["Retriever", "Metric", alt.Tooltip("Score:Q", format=".2f")]).properties(height=360, width="container"),
            width="stretch")
        st.caption("To see the re-ranker's effect on the final answers too, run A/B with "
                   "“v2.0 Hybrid” vs “v2.1 Hybrid + Rerank”.")

with tab_rel:
    runs_all = api.get("/runs")["runs"]
    eval_runs = [r for r in runs_all if r["summary"]["total"] > 0]
    st.markdown(f"##### Judge reliability check")
    st.markdown(f"Hand-label **{META['reliability_sample']} answers** from a run; VerityAi measures how often the "
                "automated judge agrees with you.")
    if not eval_runs:
        st.info("Run an evaluation first.")
    else:
        rel_idx = st.selectbox("Run to check", range(len(eval_runs)), key="rel_run",
                               format_func=lambda i: f"{eval_runs[i]['timestamp']} · {eval_runs[i]['label']}")
        rel_run = eval_runs[rel_idx]
        rel = api.get(f"/reliability/{rel_run['run_id']}")
        st.progress(rel["labelled"] / max(1, rel["sample_size"]),
                    text=f"{rel['labelled']}/{rel['sample_size']} answers labelled")
        stats = rel["stats"]
        r1, r2, r3 = st.columns(3)
        with r1:
            ui.scorecard("Agreement", stats["agreement"], "judge = human")
        with r2:
            ui.scorecard("Cohen's κ", stats["kappa"], "agreement beyond luck")
        with r3:
            overall_stats = api.get("/alignment")
            ui.count_card("All labels", f"{overall_stats['total_labels']}/{overall_stats['target']}", ui.SKY_DARK,
                          f"{overall_stats['agreement']:.0%} agreement overall" if overall_stats["agreement"] is not None else "")
        if stats["n"]:
            mx = stats["matrix"]
            st.dataframe(pd.DataFrame({"Human PASS": [mx["PASS|PASS"], mx["FAIL|PASS"]],
                                       "Human FAIL": [mx["PASS|FAIL"], mx["FAIL|FAIL"]]},
                                      index=["Judge PASS", "Judge FAIL"]), width="content")
        for d in rel["disagreements"]:
            st.warning(f"**{d['test_id']}** judge said {d['judge']} ({CAT.get(d['judge_category'], '')}), "
                       f"human said {d['human']} {('(' + d['human_category'] + ')') if d['human_category'] else ''}: "
                       f"{d['question']}")
        for item in rel["items"]:
            rep = item["report"]
            with st.container(border=True):
                st.markdown(f"**{rep['test_id']}: {html.escape(rep['question'])}** &nbsp; "
                            f"<small>expected: {html.escape(rep['expected_answer'])}</small>", unsafe_allow_html=True)
                st.markdown(f"<div class='verity-box'>{ui.highlighted_answer(rep)}</div>", unsafe_allow_html=True)
                i1, i2, i3, i4 = st.columns([1.6, 0.8, 0.8, 1.2])
                i1.markdown(f"Judge: {ui.pill(rep['category'])}", unsafe_allow_html=True)
                if i2.button("👍 PASS", key=f"rel_pass_{rel_run['run_id']}_{rep['test_id']}", width="stretch"):
                    save_label_and_refresh(rep["question"], rep["answer"], "PASS")
                    st.rerun()
                if i3.button("👎 FAIL", key=f"rel_fail_{rel_run['run_id']}_{rep['test_id']}", width="stretch"):
                    save_label_and_refresh(rep["question"], rep["answer"], "FAIL")
                    st.rerun()
                if item["labelled"]:
                    i4.markdown(f"You: **{item['human']}** {'✅' if item['agree'] else '❌'}")

        st.divider()
        st.markdown("##### Stability check: swap the judge, shuffle the chunks")
        st.markdown("Re-grades this run's answers (no new answers are generated) with a **different judge**, and with "
                    "the **chunks shuffled**. Stable = the verdicts barely move.")
        alt_options = ["local"] + OPTIONS
        s1, s2 = st.columns([2, 1])
        alt_judge = s1.selectbox("Swap the judge to", alt_options, key="alt_judge",
                                 format_func=lambda o: "💻 Local judge (NLI, no API)" if o == "local" else label(o))
        check_shuffle = s2.toggle("Also shuffle chunks", value=True, key="stab_shuffle")
        if st.button("⚖️ Run stability check", type="primary"):
            try:
                job = api.post("/jobs/stability", json={
                    "run_id": rel_run["run_id"], "alt_judge_mode": "local" if alt_judge == "local" else "llm",
                    "alt_judge_model": None if alt_judge == "local" else alt_judge, "check_shuffle": check_shuffle})
                ss.stability = api.wait_for_job(job["job_id"], show_progress)
                status_slot.empty()
            except api.ApiError as e:
                st.error(str(e))
        result = ss.stability
        if result and result["run_id"] == rel_run["run_id"]:
            if result["stable"]:
                ui.banner("✅ Stable: scores stayed the same when the judge was swapped"
                          + (" and the chunks were shuffled." if result["shuffle"] else "."), "stable")
            else:
                ui.banner("⚠️ Not fully stable - see what changed below.", "unstable")
            parts = [("judge_swap", f"Judge swap: {result['original_judge']} → {result['swapped_judge']}")]
            if result["shuffle"]:
                parts.append(("shuffle", "Chunks shuffled (same judge)"))
            for col, (key, title) in zip(st.columns(len(parts)), parts):
                with col:
                    x = result[key]
                    st.markdown(f"**{title}**")
                    if x.get("error") and not x.get("compared"):
                        st.error(x["error"])
                        continue
                    y1, y2, y3 = st.columns(3)
                    with y1:
                        ui.scorecard("Verdicts same", x["verdict_agreement"])
                    with y2:
                        ui.scorecard("Root cause same", x["category_agreement"])
                    with y3:
                        ui.count_card("Mean |Δ| score", ui.fmt(x["mean_diff"]), "#2E9E6A" if x["mean_diff"] <= META["thresholds"]["stable_diff"] else "#D64545")
                    if x.get("errors"):
                        st.caption(f"{x['errors']} item(s) skipped: {x['error']}")
                    for c in x["changed"]:
                        st.caption(f"{c['test_id']}: {CAT.get(c['before'], c['before'])} → {CAT.get(c['after'], c['after'])} · {c['question'][:80]}")

with tab_history:
    runs_all = api.get("/runs")["runs"]
    if not runs_all:
        st.info("Every run is stored permanently in the SQLite database (results/verityai.db) and listed here.")
    else:
        hist = pd.DataFrame([{
            "Time": r["timestamp"], "Label": r["label"], "Type": r.get("kind", "eval"), "Version": r["config"]["version"],
            "Generator": label(r["config"]["generator_model"]),
            "Judge": "local" if r["config"].get("judge_mode") == "local" else label(r["config"]["judge_model"]),
            "Chunk": r["config"]["chunk_size"], "Top-k": r["config"]["top_k"], "Tests": r["summary"]["total"],
            "Pass rate": r["summary"]["pass_rate"], "Faithfulness": r["summary"]["faithfulness"],
            "Accuracy": r["summary"]["accuracy"], "Tokens": r["summary"].get("tokens"),
            "s / q": r["summary"].get("seconds_per_question"),
        } for r in runs_all])
        st.dataframe(hist, hide_index=True, width="stretch", column_config={
            **ui.score_columns(),
            "Pass rate": st.column_config.ProgressColumn("Pass rate", min_value=0.0, max_value=1.0, format="percent")})
        trend = hist.iloc[::-1].reset_index(drop=True).reset_index().rename(columns={"index": "Run #"})
        trend["Run #"] += 1
        st.altair_chart(alt.Chart(trend).mark_line(point=alt.OverlayMarkDef(size=80), color=ui.LILAC_DARK).encode(
            x=alt.X("Run #:O"), y=alt.Y("Pass rate:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            tooltip=["Time", "Label", "Chunk", "Top-k", alt.Tooltip("Pass rate:Q", format=".0%")],
        ).properties(height=220, width="container"), width="stretch")

        def run_name(i):
            return f"{runs_all[i]['timestamp']} · {runs_all[i]['label']}"

        st.markdown("##### Compare two runs")
        h1, h2 = st.columns(2)
        ia = h1.selectbox("Older run (A)", range(len(runs_all)), index=min(1, len(runs_all) - 1), format_func=run_name)
        ib = h2.selectbox("Newer run (B)", range(len(runs_all)), index=0, format_func=run_name)
        if ia == ib:
            st.caption("Pick two different runs to compare them.")
        else:
            render_comparison(runs_all[ia]["run_id"], runs_all[ib]["run_id"], "hist")

        st.markdown("##### Open or delete a past run")
        o1, o2 = st.columns([4, 1])
        io = o1.selectbox("Run", range(len(runs_all)), format_func=run_name, key="hist_open")
        if o2.button("🗑️ Delete run", width="stretch"):
            api.delete(f"/runs/{runs_all[io]['run_id']}")
            if ss.current_run_id == runs_all[io]["run_id"]:
                ss.current_run_id = None
            st.rerun()
        with st.expander("Show full results of this run", expanded=False):
            render_run(fetch_run(runs_all[io]["run_id"], ss.labels_version), "hist")

with tab_tests:
    if TESTS:
        test_df = pd.DataFrame([{
            "ID": t["test_id"], "Question": t["question"], "Expected answer": t["expected_answer"],
            "Source chunk": t.get("source_chunk_id"),
            "Answer lives in": f"{t['doc_name']} → Page {t['page']}" if t["answerable"] else "Not in the documents",
            "Source": t["origin"],
        } for t in TESTS])
        picked = st.dataframe(test_df, hide_index=True, on_select="rerun", selection_mode="multi-row",
                              key="test_table", width="stretch").selection.rows
        b1, b2, b3, _ = st.columns([1, 1, 1.2, 2])
        if b1.button("🗑️ Delete selected", disabled=not picked, width="stretch"):
            api.put("/tests", {"tests": [t for i, t in enumerate(TESTS) if i not in picked]}, docs_key=ss.docs_key)
            st.rerun()
        if b2.button("Clear all tests", width="stretch"):
            api.put("/tests", {"tests": []}, docs_key=ss.docs_key)
            st.rerun()
        b3.download_button("⬇️ Download test set", json.dumps(TESTS, indent=1, ensure_ascii=False),
                           "verityai_tests.json", "application/json", width="stretch")
    else:
        st.info("No tests yet. ✨ Generate tests or type your own question above.")
    imported = st.file_uploader("Import a test set (JSON)", type=["json"], key="import_tests")
    if imported is not None and st.button("Load imported tests"):
        try:
            api.put("/tests", {"tests": json.loads(imported.getvalue())}, docs_key=ss.docs_key)
            st.rerun()
        except (json.JSONDecodeError, api.ApiError) as e:
            st.error(f"That file is not a valid test set: {e}")

with tab_docs:
    info = ss.docs_info
    chunk_data = api.post("/chunks", json=DOCS)
    chunks, pages = chunk_data["chunks"], chunk_data["pages"]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Documents", len(info["documents"]))
    d2.metric("Pages", info["pages"])
    d3.metric("Chunks", len(chunks))
    d4.metric("Avg words / chunk", round(sum(c["word_count"] for c in chunks) / max(1, len(chunks))))
    st.caption(" · ".join(f"{d['name']} ({'current' if d['is_current'] else '🕰️ archived'})" for d in info["documents"]))
    st.dataframe(pd.DataFrame([{
        "Chunk": f"#{c['chunk_id']}", "Document": c["doc_name"], "Page": c["page"],
        "Version": "current" if c["is_current"] else "archived", "Words": c["word_count"],
        "Characters": f"{c['start_char']}–{c['end_char']}", "Preview": c["text"][:90] + "…",
    } for c in chunks]), hide_index=True, width="stretch", height=280)

    sel = st.selectbox("Inspect a chunk", range(len(chunks)), format_func=lambda i: chunks[i]["location"])
    selected = chunks[sel]
    page = next(p for p in pages if p["doc_name"] == selected["doc_name"] and p["page"] == selected["page"])
    s, e, text = selected["start_char"], selected["end_char"], page["text"]
    before = ("…" if s > 200 else "") + text[max(0, s - 200):s]
    after = text[e:e + 200] + ("…" if e + 200 < len(text) else "")
    st.markdown(
        f"<div class='verity-box'><h4>📍 {html.escape(selected['location'])}</h4>"
        f"<span style='opacity:.55'>{html.escape(before)}</span>"
        f"<mark style='background:#E4D8FA;color:#2E2A47'>{html.escape(text[s:e])}</mark>"
        f"<span style='opacity:.55'>{html.escape(after)}</span></div>",
        unsafe_allow_html=True)
