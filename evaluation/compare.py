"""
evaluation/compare.py - Compare two runs query by query (A/B tests and run history).

Run B is treated as the newer/upgraded version. A query is a REGRESSION when B got worse:
  - A passed but B failed, or
  - B's overall score dropped by more than config.REGRESSION_DROP
"""

import config
from evaluation.root_cause import CATEGORY_LABELS


def compare_runs(reports_a, reports_b):
    by_question_b = {r["question"]: r for r in reports_b}
    rows = []
    for a in reports_a:
        b = by_question_b.get(a["question"])
        if b is None:
            continue
        delta = round(b["overall"] - a["overall"], 2)
        regression = (a["category"] == "PASS" and b["category"] != "PASS") or delta < -config.REGRESSION_DROP
        improvement = (a["category"] != "PASS" and b["category"] == "PASS") or delta > config.REGRESSION_DROP
        rows.append({
            "test_id": a["test_id"], "question": a["question"],
            "a_category": CATEGORY_LABELS[a["category"]], "b_category": CATEGORY_LABELS[b["category"]],
            "a_score": a["overall"], "b_score": b["overall"], "delta": delta,
            "regression": regression, "improvement": improvement and not regression,
            "a_report": a, "b_report": b,
        })
    return rows
