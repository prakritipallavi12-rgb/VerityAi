"""
storage/db.py - SQLite database: every run, result, human label and test set, stored permanently.

One file: results/verityai.db (SQLite is built into Python - nothing to install or run).

Tables
  runs      one row per evaluation run (config + summary as JSON)
  results   one row per test in a run (main scores as columns, so you can query them with SQL;
            the full report as JSON)
  labels    human PASS/FAIL labels used to check the judge
  test_sets saved test sets (the dashboard uses the one called "default")

Older versions saved runs as JSON files in results/ - those are imported automatically once.
"""

import glob
import json
import os
import sqlite3
import threading
from contextlib import contextmanager

import config

_init_lock = threading.Lock()
_ready = False

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, timestamp TEXT, label TEXT, kind TEXT,
    config TEXT, summary TEXT, tests TEXT);
CREATE TABLE IF NOT EXISTS results (
    run_id TEXT, position INTEGER, test_id TEXT, question TEXT, category TEXT,
    relevance REAL, faithfulness REAL, accuracy REAL, overall REAL,
    tokens INTEGER, seconds REAL, cost REAL, label_key TEXT, report TEXT,
    PRIMARY KEY (run_id, position));
CREATE INDEX IF NOT EXISTS idx_results_label ON results(label_key);
CREATE TABLE IF NOT EXISTS labels (
    key TEXT PRIMARY KEY, question TEXT, answer TEXT, verdict TEXT,
    category TEXT, labeler TEXT, time TEXT);
CREATE TABLE IF NOT EXISTS test_sets (name TEXT PRIMARY KEY, tests TEXT, updated TEXT);
"""


@contextmanager
def connect():
    """A fresh connection per use - safe across the API's threads."""
    _ensure_ready()
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _ensure_ready():
    global _ready
    if _ready:
        return
    with _init_lock:
        if _ready:
            return
        os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
        conn = sqlite3.connect(config.DB_PATH, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
        conn.executescript(SCHEMA)
        conn.commit()
        conn.close()
        _ready = True
        _import_legacy_json()


# ---------------- runs ----------------
def save_run(run, kind="eval"):
    from evaluation.labels import label_key
    with connect() as conn:
        conn.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?)",
                     (run["run_id"], run["timestamp"], run["label"], kind, json.dumps(run["config"]),
                      json.dumps(run["summary"]), json.dumps(run.get("tests", []), ensure_ascii=False)))
        conn.execute("DELETE FROM results WHERE run_id = ?", (run["run_id"],))
        conn.executemany(
            "INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(run["run_id"], i, r.get("test_id"), r.get("question"), r.get("category"),
              r.get("scores", {}).get("relevance"), r.get("scores", {}).get("faithfulness"),
              r.get("scores", {}).get("accuracy"), r.get("overall"),
              r.get("usage", {}).get("tokens"), r.get("usage", {}).get("seconds"), r.get("usage", {}).get("cost"),
              label_key(r.get("question", ""), r.get("answer", "")), json.dumps(r, ensure_ascii=False))
             for i, r in enumerate(run["reports"])])


def list_runs():
    """Run summaries (no per-test reports), newest first."""
    with connect() as conn:
        rows = conn.execute("SELECT run_id, timestamp, label, kind, config, summary FROM runs "
                            "ORDER BY timestamp DESC, run_id DESC").fetchall()
    return [{"run_id": r["run_id"], "timestamp": r["timestamp"], "label": r["label"], "kind": r["kind"],
             "config": json.loads(r["config"]), "summary": json.loads(r["summary"])} for r in rows]


def get_run(run_id):
    """One full run including every report, or None."""
    with connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        reports = conn.execute("SELECT report FROM results WHERE run_id = ? ORDER BY position", (run_id,)).fetchall()
    return {"run_id": row["run_id"], "timestamp": row["timestamp"], "label": row["label"], "kind": row["kind"],
            "config": json.loads(row["config"]), "summary": json.loads(row["summary"]),
            "tests": json.loads(row["tests"] or "[]"), "reports": [json.loads(r["report"]) for r in reports]}


def delete_run(run_id):
    with connect() as conn:
        conn.execute("DELETE FROM results WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))


def labelled_results():
    """Every stored result that has a human label: [(judge_category, human_verdict, report_json)]."""
    with connect() as conn:
        rows = conn.execute("SELECT r.category, l.verdict, r.report, r.label_key FROM results r "
                            "JOIN labels l ON l.key = r.label_key WHERE r.category != 'ERROR'").fetchall()
    return [(r["category"], r["verdict"], r["report"], r["label_key"]) for r in rows]


# ---------------- labels ----------------
def save_label(key, question, answer, verdict, category, labeler, time):
    with connect() as conn:
        conn.execute("INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?,?,?)",
                     (key, question, answer, verdict, category, labeler, time))


def delete_label(key):
    with connect() as conn:
        conn.execute("DELETE FROM labels WHERE key = ?", (key,))


def load_labels():
    with connect() as conn:
        rows = conn.execute("SELECT * FROM labels").fetchall()
    return {r["key"]: {k: r[k] for k in ("question", "answer", "verdict", "category", "labeler", "time")} for r in rows}


# ---------------- test sets ----------------
def save_test_set(tests, name="default"):
    from datetime import datetime
    with connect() as conn:
        conn.execute("INSERT OR REPLACE INTO test_sets VALUES (?,?,?)",
                     (name, json.dumps(tests, ensure_ascii=False), datetime.now().isoformat(timespec="seconds")))


def load_test_set(name="default"):
    with connect() as conn:
        row = conn.execute("SELECT tests FROM test_sets WHERE name = ?", (name,)).fetchone()
    return json.loads(row["tests"]) if row else []


# ---------------- one-time import of the old JSON files ----------------
def _import_legacy_json():
    from evaluation.runner import summarize
    with connect() as conn:
        known = {r[0] for r in conn.execute("SELECT run_id FROM runs").fetchall()}
    for path in glob.glob(os.path.join(config.RESULTS_DIR, "run_*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                run = json.load(f)
            if run["run_id"] not in known:
                run["summary"] = summarize(run["reports"])  # recompute with the current fields
                save_run(run, kind="eval")
            os.replace(path, path + ".imported")
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    labels_path = os.path.join(config.RESULTS_DIR, "human_labels.json")
    if os.path.exists(labels_path):
        try:
            with open(labels_path, encoding="utf-8") as f:
                for key, l in json.load(f).items():
                    save_label(key, l["question"], l["answer"], l["verdict"], l.get("category", ""),
                               l.get("labeler", ""), l.get("time", ""))
            os.replace(labels_path, labels_path + ".imported")
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    tests_path = os.path.join(config.RESULTS_DIR, "last_test_set.json")
    if os.path.exists(tests_path):
        try:
            with open(tests_path, encoding="utf-8") as f:
                save_test_set(json.load(f))
            os.replace(tests_path, tests_path + ".imported")
        except (OSError, json.JSONDecodeError):
            pass
