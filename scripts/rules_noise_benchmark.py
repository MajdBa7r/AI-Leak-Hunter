"""Rule-engine benchmark under realistic procurement noise.

Applies the seven disruptions of tests/test_robustness.py at the intensities
fixed in Phase 1 (before any target score existed) - separately and all
together - and scores the RULE engine (Isolation Forest off) against
ground_truth.csv. Labels are untouched: a disruption that makes a normal line
look anomalous counts as a false positive, one that hides a planted anomaly
counts as a false negative.

    PYTHONPATH=src python scripts/rules_noise_benchmark.py [--out reports_rules/noise_benchmark.md]
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from leak_hunter.config import DetectionConfig
from leak_hunter.data_loader import load_procurement_data
from leak_hunter.evaluation import load_ground_truth
from leak_hunter.pipeline import run_detection

RULES_ONLY = replace(DetectionConfig(), isolation_forest=replace(DetectionConfig().isolation_forest, enabled=False))


from leak_hunter.scenarios import SCENARIOS, SEED  # noqa: E402  (shared definitions)


def score(res, gt):
    rb = res.findings[res.findings.detection_method == "RULE_BASED"]
    pred, y = set(rb.invoice_line_id), set(gt.invoice_line_id[gt.is_anomaly == 1])
    pend = set(res.pending.invoice_line_id)
    tp, fp, fn = len(pred & y), len(pred - y), len(y - pred)
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    fp_types = rb[rb.invoice_line_id.isin(pred - y)].drop_duplicates("invoice_line_id").finding_type.value_counts().to_dict()
    fn_types = gt[gt.invoice_line_id.isin(y - pred)].primary_anomaly_type.value_counts().to_dict()
    low_conf = rb.details.map(lambda s: json.loads(s).get("confidence") in ("LOW", "MEDIUM"))
    conf = rb.details.map(lambda s: json.loads(s).get("confidence", "HIGH"))
    pred_hi = set(rb.invoice_line_id[conf != "LOW"])
    tph, fph = len(pred_hi & y), len(pred_hi - y)
    return dict(tp=tp, fp=fp, fn=fn, tn=len(gt) - tp - fp - fn, precision=round(p, 4), recall=round(r, 4),
                f1=round(2 * p * r / (p + r), 4) if p + r else 0.0, pending=len(pend), pending_planted=len(pend & (y - pred)),
                fp_by_type=fp_types, fn_by_type=fn_types, low_or_medium_confidence_findings=int(low_conf.sum()),
                precision_excl_low_confidence=round(tph / max(1, tph + fph), 4),
                recall_excl_low_confidence=round(tph / max(1, len(y)), 4), low_confidence_lines=len(set(rb.invoice_line_id) - pred_hi),
                fp_zero_exposure=int(rb[rb.invoice_line_id.isin(pred - y)].groupby("invoice_line_id").potential_exposure.max().eq(0).sum()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir")
    ap.add_argument("--out", default="reports_rules/noise_benchmark.md")
    a = ap.parse_args(argv)
    base, gt = load_procurement_data(a.data_dir), load_ground_truth(a.data_dir)
    rows = []
    for name, fns in SCENARIOS:
        d, rng = copy.deepcopy(base), np.random.default_rng(SEED)
        for fn in fns:
            fn(d, gt, rng)
        s = score(run_detection(data=d, config=RULES_ONLY), gt)
        rows.append((name, s))
        print(f"{name:<58} P {s['precision']:.3f} R {s['recall']:.3f} F1 {s['f1']:.3f} TP {s['tp']} FP {s['fp']} FN {s['fn']} "
              f"| excl LOW: P {s['precision_excl_low_confidence']:.3f} R {s['recall_excl_low_confidence']:.3f} "
              f"(LOW lines {s['low_confidence_lines']}) FP {s['fp_by_type']} FN {s['fn_by_type']}")
    L = ["# Rule engine under realistic procurement noise", "",
         "Rules only (Isolation Forest off), invoice-line level, labels = synthetic ground truth. Disruption "
         "intensities are those fixed in Phase 1 (`tests/test_robustness.py`), seed 7.", "",
         "| Scenario | Precision | Recall | F1 | TP | FP | FN | Precision / recall excl. LOW-confidence | FP by finding type | FN by planted type |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in rows:
        L.append(f"| {name} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | {s['tp']} | {s['fp']} | {s['fn']} | "
                 f"{s['precision_excl_low_confidence']:.3f} / {s['recall_excl_low_confidence']:.3f} | "
                 f"{s['fp_by_type'] or '-'} | {s['fn_by_type'] or '-'} |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n")
    Path(a.out).with_suffix(".json").write_text(json.dumps(dict(rows), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
