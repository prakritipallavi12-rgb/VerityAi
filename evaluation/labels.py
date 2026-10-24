"""
evaluation/labels.py - Human labels and the judge reliability check.

A team member reads a test's question + answer and marks it PASS or FAIL (and optionally the cause).
Labels live in the SQLite database, keyed by question + answer, so they keep working across runs
whenever the same answer appears again.

Judge reliability check: hand-label config.RELIABILITY_SAMPLE (10) answers from a run, then measure
  agreement      = share of labelled items where judge and human gave the same PASS/FAIL
  Cohen's kappa  = agreement corrected for luck (1 = perfect, 0 = no better than chance)
"""

import hashlib
import json
import random
from datetime import datetime

import config
from storage import db


def label_key(question, answer):
    return hashlib.sha1(f"{question.strip()}||{answer.strip()}".encode("utf-8")).hexdigest()[:16]


def load_labels():
    return db.load_labels()


def save_label(question, answer, verdict, category="", labeler=""):
    db.save_label(label_key(question, answer), question, answer, verdict, category, labeler,
                  datetime.now().strftime("%Y-%m-%d %H:%M"))


def delete_label(question, answer):
    db.delete_label(label_key(question, answer))


def judge_verdict(report):
    return "PASS" if report["category"] == "PASS" else "FAIL"


def item_alignment(report, labels):
    """Judge vs human for one test."""
    human = labels.get(label_key(report["question"], report.get("answer", "")))
    if not human:
        return {"labelled": False, "judge": judge_verdict(report)}
    return {"labelled": True, "judge": judge_verdict(report), "human": human["verdict"],
            "human_category": human.get("category", ""), "labeler": human.get("labeler", ""),
            "agree": judge_verdict(report) == human["verdict"]}


def _stats(pairs):
    """pairs = [(judge PASS/FAIL, human PASS/FAIL)] -> agreement, kappa, 2x2 matrix."""
    n = len(pairs)
    matrix = {f"{j}|{h}": sum(1 for p in pairs if p == (j, h)) for j in ("PASS", "FAIL") for h in ("PASS", "FAIL")}
    if not n:
        return {"n": 0, "agreement": None, "kappa": None, "matrix": matrix}
    observed = (matrix["PASS|PASS"] + matrix["FAIL|FAIL"]) / n
    judge_pass = (matrix["PASS|PASS"] + matrix["PASS|FAIL"]) / n
    human_pass = (matrix["PASS|PASS"] + matrix["FAIL|PASS"]) / n
    expected = judge_pass * human_pass + (1 - judge_pass) * (1 - human_pass)
    kappa = 1.0 if expected == 1 else (observed - expected) / (1 - expected)
    return {"n": n, "agreement": round(observed, 2), "kappa": round(kappa, 2), "matrix": matrix}


def alignment_stats(reports=None, labels=None):
    """Judge vs human over the given reports - or, with no reports, over every labelled result in the database."""
    labels = load_labels() if labels is None else labels
    if reports is None:
        seen, pairs = set(), []
        for category, verdict, _, key in db.labelled_results():
            if key not in seen:
                seen.add(key)
                pairs.append(("PASS" if category == "PASS" else "FAIL", verdict))
    else:
        seen, pairs = set(), []
        for r in reports:
            key = label_key(r["question"], r.get("answer", ""))
            if r["category"] != "ERROR" and key in labels and key not in seen:
                seen.add(key)
                pairs.append((judge_verdict(r), labels[key]["verdict"]))
    return {**_stats(pairs), "total_labels": len(labels), "target": config.HUMAN_LABEL_TARGET}


def reliability_sample(run, n=None):
    """
    Pick n answers from a run for hand-labelling: a mix of passes and failures (so the check
    is not all-easy cases), always the same picks for the same run.
    """
    n = n or config.RELIABILITY_SAMPLE
    reports = [r for r in run["reports"] if r["category"] != "ERROR"]
    rng = random.Random(run["run_id"])
    passes = [r for r in reports if r["category"] == "PASS"]
    fails = [r for r in reports if r["category"] != "PASS"]
    rng.shuffle(passes)
    rng.shuffle(fails)
    take_fail = min(len(fails), max(n // 2, n - len(passes)))
    picked = fails[:take_fail] + passes[:n - take_fail]
    picked.sort(key=lambda r: reports.index(r))
    return picked


def reliability_report(run, n=None):
    """The hand-labelling queue for a run plus how well the judge agrees on the labelled ones."""
    labels = load_labels()
    sample = reliability_sample(run, n)
    items = [{"report": r, **item_alignment(r, labels)} for r in sample]
    stats = alignment_stats([r for r in sample], labels)
    disagreements = [{"test_id": i["report"]["test_id"], "question": i["report"]["question"],
                      "judge": i["judge"], "human": i["human"], "judge_category": i["report"]["category"],
                      "human_category": i.get("human_category", "")}
                     for i in items if i["labelled"] and not i["agree"]]
    return {"items": items, "labelled": sum(i["labelled"] for i in items), "sample_size": len(items),
            "stats": stats, "disagreements": disagreements}


def export_labels_json():
    return json.dumps(load_labels(), indent=1, ensure_ascii=False)
