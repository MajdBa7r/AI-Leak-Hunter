"""Evaluate saved detection outputs against the synthetic ground truth.

Run AFTER ``python -m leak_hunter.pipeline``; this module reads the findings
already written to disk and only then loads ground_truth.csv, so labels can
never influence detection.

    python -m leak_hunter.evaluation [--data-dir DIR] [--out-dir outputs] [--report reports/detection_evaluation.md]

Evaluation units
----------------
invoice line      primary unit (ground truth is one row per invoice line).
                  A line is "predicted anomalous" if any rule finding exists.
type (multi-label) per anomaly type t: truth = t in all_applicable_anomaly_types,
                  prediction = the line has a finding of type t.
invoice document  anomalous if any of its lines is.
incident          duplicate pairs (duplicate_group_id) are ONE incident made of
                  two lines; every other anomalous line is its own incident.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

from .data_loader import GROUND_TRUTH_FILE, DataLoadError, default_data_dir
from .exposure import DUP, HIST, PRICE, QTY, from_cents, line_exposure, to_cents

TYPES = [DUP, QTY, PRICE, HIST]
GT_COLUMNS = ["invoice_id", "invoice_line_id", "is_anomaly", "primary_anomaly_type", "all_applicable_anomaly_types",
              "expected_exposure", "exposure_paid_to_date", "duplicate_group_id"]


def load_ground_truth(data_dir: str | Path | None = None) -> pd.DataFrame:
    path = Path(data_dir or default_data_dir()) / GROUND_TRUTH_FILE
    if not path.is_file():
        raise DataLoadError(f"Ground-truth file not found: {path}")
    gt = pd.read_csv(path, keep_default_na=False, na_values={"expected_exposure": [""]},
                     dtype={"duplicate_group_id": str, "all_applicable_anomaly_types": str})
    miss = [c for c in GT_COLUMNS if c not in gt.columns]
    if miss:
        raise DataLoadError(f"{path} is missing column(s) {miss}")
    return gt


def load_outputs(out_dir: str | Path) -> dict:
    out = Path(out_dir)
    need = ["findings.csv", "incidents.csv", "run_summary.json"]
    miss = [n for n in need if not (out / n).is_file()]
    if miss:
        raise DataLoadError(f"Detection outputs {miss} not found in {out}. Run `python -m leak_hunter.pipeline "
                            f"--out-dir {out}` first.")
    res = {"findings": pd.read_csv(out / "findings.csv", dtype={"invoice_number": str}),
           "incidents": pd.read_csv(out / "incidents.csv"),
           "summary": json.loads((out / "run_summary.json").read_text())}
    res["scores"] = pd.read_csv(out / "if_scores.csv") if (out / "if_scores.csv").is_file() else None
    pp = out / "pending_quantity_checks.csv"
    res["pending"] = pd.read_csv(pp) if pp.is_file() else pd.DataFrame(columns=["invoice_line_id"])
    return res


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f, 4)}


def binary(y: np.ndarray, yhat: np.ndarray) -> dict:
    y, yhat = np.asarray(y, bool), np.asarray(yhat, bool)
    d = prf(int((y & yhat).sum()), int((~y & yhat).sum()), int((y & ~yhat).sum()))
    d["tn"] = int((~y & ~yhat).sum())
    return d


def precision_at_k(y: np.ndarray, score: np.ndarray, ks) -> dict:
    order = np.argsort(-score, kind="mergesort")
    return {int(k): round(float(np.asarray(y)[order[:k]].mean()), 4) for k in ks if k <= len(y)}


def evaluate(findings: pd.DataFrame, incidents: pd.DataFrame, scores: pd.DataFrame | None,
             gt: pd.DataFrame, pending: pd.DataFrame | None = None) -> dict:
    m: dict = {}
    lines = gt[["invoice_line_id", "invoice_id", "is_anomaly", "primary_anomaly_type",
                "all_applicable_anomaly_types", "expected_exposure", "exposure_paid_to_date",
                "duplicate_group_id"]].copy()
    lines["true_types"] = lines.all_applicable_anomaly_types.map(lambda s: set(s.split("|")) - {"NORMAL", ""})
    rb = findings[findings.detection_method == "RULE_BASED"]
    pred_types = rb.groupby("invoice_line_id").finding_type.apply(set)
    lines["pred_types"] = lines.invoice_line_id.map(pred_types).apply(lambda x: x if isinstance(x, set) else set())
    y = lines.is_anomaly.to_numpy() == 1
    y_rule = lines.pred_types.map(bool).to_numpy()
    unknown = set(findings.invoice_line_id) - set(lines.invoice_line_id)
    m["dataset"] = {"invoice_lines": len(lines), "anomalous": int(y.sum()), "normal": int((~y).sum()),
                    "anomaly_rate": round(float(y.mean()), 4),
                    "primary_type_counts": lines.primary_anomaly_type.value_counts().to_dict(),
                    "duplicate_pairs": int(lines.duplicate_group_id.replace("", np.nan).nunique()),
                    "findings_on_unknown_lines": len(unknown)}

    # ---- rule-based: invoice-line level ------------------------------------------------------
    m["rules_line_level"] = binary(y, y_rule)
    pend = set(pending.invoice_line_id) if pending is not None else set()
    is_pend = lines.invoice_line_id.isin(pend).to_numpy()
    m["pending_grace_period"] = {
        "lines_pending": int(is_pend.sum()), "planted_anomalies_pending": int((is_pend & y & ~y_rule).sum()),
        "normal_lines_pending": int((is_pend & ~y).sum()),
        "note": "pending = billed beyond posted receipts but younger than the grace period; deferred, not cleared",
        "recall_counting_pending_as_deferred": round(float(((y_rule | is_pend) & y).sum() / y.sum()), 4) if y.any() else 0.0}
    per_type, primary_recall = {}, {}
    for t in TYPES:
        yt = lines.true_types.map(lambda s: t in s).to_numpy()
        pt = lines.pred_types.map(lambda s: t in s).to_numpy()
        per_type[t] = binary(yt, pt)
        per_type[t]["support"] = int(yt.sum())
        prim = lines.primary_anomaly_type == t
        primary_recall[t] = {"planted": int(prim.sum()),
                             "detected_by_own_rule": int(lines[prim].pred_types.map(lambda s: t in s).sum()),
                             "detected_by_any_rule": int(lines[prim].pred_types.map(bool).sum())}
    m["rules_per_type_multilabel"] = per_type
    m["rules_primary_type_recall"] = primary_recall
    cm = {}
    for gt_t in ["NORMAL"] + TYPES:
        sub = lines[lines.primary_anomaly_type == gt_t]
        row = {pt: int(sub.pred_types.map(lambda s: pt in s).sum()) for pt in TYPES}
        row["NO_RULE_FINDING"] = int(sub.pred_types.map(lambda s: not s).sum())
        row["lines"] = len(sub)
        cm[gt_t] = row
    m["rules_confusion_primary_vs_finding_type"] = cm
    exact = lines.apply(lambda r: r.pred_types == r.true_types, axis=1)
    m["rules_exact_type_set_match"] = {"lines_matching": int(exact.sum()), "lines": len(lines)}

    # false positives / negatives
    fp = lines[~y & y_rule]
    fn = lines[y & ~y_rule]
    lines["_pending"] = is_pend
    m["false_positives_rules"] = {"count": len(fp), "by_finding_type": pd.Series(
        [t for s in fp.pred_types for t in s]).value_counts().to_dict(), "examples": fp.invoice_line_id.head(20).tolist()}
    m["false_negatives_rules"] = {"count": len(fn), "by_primary_type": fn.primary_anomaly_type.value_counts().to_dict(),
                                  "of_which_pending_grace_period": int(fn.invoice_line_id.isin(pend).sum()),
                                  "examples": fn.invoice_line_id.head(20).tolist()}

    # ---- document level ---------------------------------------------------------------------------
    doc = lines.groupby("invoice_id").agg(y=("is_anomaly", "max"), p=("pred_types", lambda s: any(bool(x) for x in s)))
    m["rules_document_level"] = binary(doc.y.to_numpy() == 1, doc.p.to_numpy())

    # ---- incident level -------------------------------------------------------------------------
    gt_inc = {}
    for r in lines[y].itertuples():
        k = r.duplicate_group_id if r.duplicate_group_id else r.invoice_line_id
        gt_inc.setdefault(k, set()).add(r.invoice_line_id)
    rb_inc = incidents[incidents.detection_methods.str.contains("RULE_BASED")]
    pred_inc = {r.incident_id: set(r.invoice_line_ids.split("|")) for r in rb_inc.itertuples()}
    matched_gt = sum(1 for s in gt_inc.values() if any(s == p for p in pred_inc.values()))
    matched_pred = sum(1 for p in pred_inc.values() if any(p == s for s in gt_inc.values()))
    m["rules_incident_level"] = {"true_incidents": len(gt_inc), "predicted_incidents": len(pred_inc),
                                 "exactly_matched_true": matched_gt, "exactly_matched_predicted": matched_pred,
                                 "precision": round(matched_pred / len(pred_inc), 4) if pred_inc else 0.0,
                                 "recall": round(matched_gt / len(gt_inc), 4) if gt_inc else 0.0}
    gp = {k: s for k, s in gt_inc.items() if k.startswith("DUPG")}
    pp = {k: s for k, s in pred_inc.items() if k.startswith("INC-DUP")}
    m["duplicate_pairs"] = {"true_pairs": len(gp), "predicted_groups": len(pp),
                            "pairs_exactly_matched": sum(1 for s in gp.values() if s in pp.values()),
                            "predicted_groups_matching_a_true_pair": sum(1 for s in pp.values() if s in gp.values())}

    # ---- exposure -------------------------------------------------------------------------------
    le = line_exposure(findings).set_index("invoice_line_id")
    lines["pred_exposure"] = lines.invoice_line_id.map(le.line_exposure).fillna(0.0)
    lines["pred_paid"] = lines.invoice_line_id.map(le.line_paid_exposure).fillna(0.0)
    tp_lines = lines[y & y_rule]
    d_c = to_cents(tp_lines.pred_exposure.to_numpy()) - to_cents(tp_lines.expected_exposure.to_numpy())
    p_c = to_cents(tp_lines.pred_paid.to_numpy()) - to_cents(tp_lines.exposure_paid_to_date.to_numpy())
    by_t = {}
    for t in TYPES:
        sel = (tp_lines.primary_anomaly_type == t).to_numpy()
        by_t[t] = {"lines": int(sel.sum()), "exact_matches": int((d_c[sel] == 0).sum()),
                   "predicted_total": round(float(tp_lines.pred_exposure[sel].sum()), 2),
                   "expected_total": round(float(tp_lines.expected_exposure[sel].sum()), 2),
                   "max_abs_diff": from_cents(int(np.abs(d_c[sel]).max())) if sel.any() else 0.0}
    m["exposure"] = {
        "comparison_unit": "true-positive invoice lines; predicted = de-duplicated line exposure",
        "lines_compared": len(tp_lines), "exact_matches_to_cent": int((d_c == 0).sum()),
        "mean_abs_diff": round(float(np.abs(d_c).mean() / 100), 4) if len(d_c) else 0.0,
        "predicted_total_deduplicated": round(float(lines.pred_exposure.sum()), 2),
        "expected_total_ground_truth": round(float(lines.expected_exposure.sum()), 2),
        "naive_sum_of_rule_findings": round(float(rb.potential_exposure.sum()), 2),
        "exposure_on_false_positive_lines": round(float(lines.loc[~y & y_rule, "pred_exposure"].sum()), 2),
        "paid_exposure_exact_matches": int((p_c == 0).sum()),
        "predicted_paid_total": round(float(lines.pred_paid.sum()), 2),
        "expected_paid_total": round(float(lines.exposure_paid_to_date.sum()), 2),
        "confirmed_loss_total": round(float(incidents.confirmed_loss.sum()), 2),
        "by_primary_type": by_t,
    }

    # ---- Isolation Forest -------------------------------------------------------------------------
    if scores is not None:
        s = scores.merge(lines[["invoice_line_id", "is_anomaly", "primary_anomaly_type", "pred_types"]],
                         on="invoice_line_id", validate="one_to_one")
        ys, sc, flag = s.is_anomaly.to_numpy() == 1, s.anomaly_score.to_numpy(), s.above_review_threshold.to_numpy()
        test = ~s.in_training_window.to_numpy()
        ifm = {"review_threshold": float(s.review_threshold.iloc[0]),
               "review_threshold_definition": "(1 - review_budget) quantile of training-window anomaly scores",
               "training_lines": int((~test).sum()), "out_of_time_lines": int(test.sum())}
        for name, mask in [("all_lines", np.ones(len(s), bool)), ("out_of_time_window", test)]:
            yy, ss, ff = ys[mask], sc[mask], flag[mask]
            b = binary(yy, ff)
            b.update({"flagged": int(ff.sum()), "base_rate": round(float(yy.mean()), 4),
                      "roc_auc": round(float(roc_auc_score(yy, ss)), 4) if 0 < yy.sum() < len(yy) else None,
                      "average_precision": round(float(average_precision_score(yy, ss)), 4) if yy.any() else None,
                      "precision_at_k": precision_at_k(yy, ss, [10, 25, 50, 100, 250, 500])})
            ifm[name] = b
        ifm["recall_by_primary_type_at_threshold"] = {
            t: {"planted": int((s.primary_anomaly_type == t).sum()),
                "flagged": int(((s.primary_anomaly_type == t) & s.above_review_threshold).sum())} for t in TYPES}
        rule_hit = s.pred_types.map(bool).to_numpy()
        only = flag & ~rule_hit
        ifm["breakdown"] = {"flagged_and_rule_flagged": int((flag & rule_hit).sum()),
                            "flagged_isolation_forest_only": int(only.sum()),
                            "if_only_true_anomalies": int((only & ys).sum()),
                            "if_only_normal": int((only & ~ys).sum()),
                            "rule_flagged_not_if_flagged": int((rule_hit & ~flag).sum())}
        ifm["combined_rules_or_if_line_level"] = binary(ys, rule_hit | flag)
        ifm["model_default_outliers_contamination_setting"] = int(s.model_default_outlier.sum())
        m["isolation_forest"] = ifm
    lines["_pt"] = lines.pred_types.map(lambda s: "|".join(sorted(s)))
    m["_tables"] = {"false_positive_lines": lines.loc[~y & y_rule, ["invoice_line_id", "_pt", "pred_exposure"]]
                    .head(20).values.tolist(),
                    "false_negative_lines": lines.loc[y & ~y_rule].assign(
                        _s=lambda t: np.where(t._pending, "PENDING (grace period)", "missed"))[
                        ["invoice_line_id", "primary_anomaly_type", "_s"]].head(20).values.tolist()}
    return m


# ------------------------------------------------------------------------------------------------
# Report
# ------------------------------------------------------------------------------------------------
def _pct(x):
    return "n/a" if x is None else f"{x:.1%}"


def render_report(m: dict, summary: dict) -> str:
    cfg = summary["config"]
    ic = cfg["isolation_forest"]
    L = ["# AI Leak Hunter - detection evaluation", "",
         f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `python -m leak_hunter.evaluation` "
         f"from saved detection outputs. Python {platform.python_version()}, pandas {pd.__version__}, numpy "
         f"{np.__version__}, scikit-learn {sklearn.__version__}.", "",
         "> **Synthetic data.** These results measure how well the engine recovers anomalies that were planted with "
         "known rules in a synthetic dataset. They do **not** establish real-world fraud or leakage detection "
         "performance. The 10% anomaly rate is a property of this dataset, not an estimate of real prevalence.", "",
         "## 1. Dataset and run", "",
         f"- Invoice lines: **{m['dataset']['invoice_lines']:,}** ({m['dataset']['normal']:,} normal, "
         f"{m['dataset']['anomalous']:,} planted anomalies = {m['dataset']['anomaly_rate']:.0%}); "
         f"{m['dataset']['duplicate_pairs']} duplicate pairs.",
         "- Planted primary types: " + ", ".join(f"{k} {v}" for k, v in m["dataset"]["primary_type_counts"].items()
                                                 if k != "NORMAL") + ".",
         f"- Inputs: " + ", ".join(f"{k} {v:,}" for k, v in summary["inputs"].items()) +
         f". Data-quality issues found by the loader: {len(summary['data_quality_issues'])}.",
         f"- Detection read ground truth: **{summary['ground_truth_used']}**. Labels were loaded only for this report.",
         f"- Findings: {summary['counts']['findings']:,} "
         f"({', '.join(f'{k} {v}' for k, v in summary['counts']['findings_by_type'].items())}); incidents: "
         f"{summary['counts']['incidents']:,}. Detection runtime {summary['runtime_seconds']} s.", "",
         "## 2. Rule-based detection", "",
         "Unit: **invoice line**. A line is predicted anomalous if any rule finding exists.", ""]
    r = m["rules_line_level"]
    L += ["| Metric | Value |", "|---|---|",
          f"| Precision | {r['precision']:.4f} |", f"| Recall | {r['recall']:.4f} |", f"| F1 | {r['f1']:.4f} |",
          f"| TP / FP / FN / TN | {r['tp']} / {r['fp']} / {r['fn']} / {r['tn']} |", "",
          "Confusion matrix (lines):", "", "| | Predicted anomalous | Predicted normal |", "|---|---|---|",
          f"| **Anomalous (truth)** | {r['tp']} | {r['fn']} |", f"| **Normal (truth)** | {r['fp']} | {r['tn']} |", "",
          "### Per anomaly type (multi-label)", "",
          "Truth = type listed in `all_applicable_anomaly_types` (a line can carry several types, e.g. a duplicate "
          "repeat entry also over-bills the received quantity); prediction = the line has a finding of that type.", "",
          "| Type | Support | Precision | Recall | F1 | FP | FN |", "|---|---|---|---|---|---|---|"]
    for t, d in m["rules_per_type_multilabel"].items():
        L.append(f"| {t} | {d['support']} | {d['precision']:.4f} | {d['recall']:.4f} | {d['f1']:.4f} | {d['fp']} | {d['fn']} |")
    L += ["", "Recall by planted primary type:", "", "| Primary type | Planted | Found by its own rule | Found by any rule |",
          "|---|---|---|---|"]
    for t, d in m["rules_primary_type_recall"].items():
        L.append(f"| {t} | {d['planted']} | {d['detected_by_own_rule']} | {d['detected_by_any_rule']} |")
    L += ["", "Ground-truth primary type (rows) vs finding types raised (columns; a line can appear in several columns):",
          "", "| Truth \\ finding | " + " | ".join(TYPES) + " | NO_RULE_FINDING | Lines |",
          "|---|" + "---|" * (len(TYPES) + 2)]
    for t, row in m["rules_confusion_primary_vs_finding_type"].items():
        L.append(f"| {t} | " + " | ".join(str(row[c]) for c in TYPES) + f" | {row['NO_RULE_FINDING']} | {row['lines']} |")
    em = m["rules_exact_type_set_match"]
    L += ["", f"Lines whose set of finding types equals the ground-truth type set exactly: {em['lines_matching']:,} / "
          f"{em['lines']:,}.", "",
          "### Other evaluation units", "",
          "| Unit | Precision | Recall | Notes |", "|---|---|---|---|",
          f"| Invoice document | {m['rules_document_level']['precision']:.4f} | {m['rules_document_level']['recall']:.4f} | "
          f"document anomalous if any line is |",
          f"| Incident | {m['rules_incident_level']['precision']:.4f} | {m['rules_incident_level']['recall']:.4f} | "
          f"duplicate pair = 1 incident; {m['rules_incident_level']['true_incidents']} true vs "
          f"{m['rules_incident_level']['predicted_incidents']} predicted rule incidents, exact line-set match |",
          f"| Duplicate pair | - | - | {m['duplicate_pairs']['pairs_exactly_matched']}/{m['duplicate_pairs']['true_pairs']} "
          f"true pairs matched; {m['duplicate_pairs']['predicted_groups_matching_a_true_pair']}/"
          f"{m['duplicate_pairs']['predicted_groups']} predicted groups are true pairs |", "",
          "### False positives and false negatives", "",
          f"- False positives among normal lines: **{m['false_positives_rules']['count']}** "
          f"{m['false_positives_rules']['by_finding_type'] or ''}",
          f"- False negatives among planted anomalies: **{m['false_negatives_rules']['count']}** "
          f"{m['false_negatives_rules']['by_primary_type'] or ''}, of which "
          f"{m['false_negatives_rules']['of_which_pending_grace_period']} are PENDING: billed beyond posted receipts "
          f"but younger than the {cfg['rules']['qty_grace_days']}-day posting-lag grace period, so deferred rather "
          f"than cleared (they are flagged automatically once older). Recall counting pending lines as deferred: "
          f"{m['pending_grace_period']['recall_counting_pending_as_deferred']:.4f}. Normal lines pending: "
          f"{m['pending_grace_period']['normal_lines_pending']}."]
    for name, rows, hdr in [("False-positive lines (first 20)", m["_tables"]["false_positive_lines"],
                             "| invoice_line_id | finding types | exposure |"),
                            ("False-negative lines (first 20)", m["_tables"]["false_negative_lines"],
                             "| invoice_line_id | primary type | status |")]:
        if rows:
            L += ["", f"{name}:", "", hdr, "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]
    e = m["exposure"]
    L += ["", "## 3. Exposure", "",
          f"Compared on the {e['lines_compared']} true-positive lines using the de-duplicated line exposure "
          "(duplicate pair counted once; price-rule overlap counted once).", "",
          "| Item | Value |", "|---|---|",
          f"| Lines whose exposure matches ground truth to the cent | {e['exact_matches_to_cent']} / {e['lines_compared']} |",
          f"| Mean absolute difference per line | ${e['mean_abs_diff']:,.4f} |",
          f"| Predicted potential exposure (de-duplicated) | ${e['predicted_total_deduplicated']:,.2f} |",
          f"| Ground-truth expected exposure | ${e['expected_total_ground_truth']:,.2f} |",
          f"| Naive sum of rule findings (double counts overlaps; do not use) | ${e['naive_sum_of_rule_findings']:,.2f} |",
          f"| Exposure attached to false-positive lines | ${e['exposure_on_false_positive_lines']:,.2f} |",
          f"| Paid exposure: predicted vs ground truth | ${e['predicted_paid_total']:,.2f} vs ${e['expected_paid_total']:,.2f} "
          f"({e['paid_exposure_exact_matches']} exact line matches) |",
          f"| Confirmed loss (realized duplicate payments only) | ${e['confirmed_loss_total']:,.2f} |", "",
          "| Primary type | Lines | Exact | Predicted | Expected | Max abs diff |", "|---|---|---|---|---|---|"]
    for t, d in e["by_primary_type"].items():
        L.append(f"| {t} | {d['lines']} | {d['exact_matches']} | ${d['predicted_total']:,.2f} | ${d['expected_total']:,.2f} "
                 f"| ${d['max_abs_diff']:,.2f} |")
    if "isolation_forest" in m:
        f = m["isolation_forest"]
        L += ["", "## 4. Isolation Forest", "",
              f"Fit mode **{ic['fit_mode']}**: imputer and forest fitted on the earliest {ic['train_fraction']:.0%} of "
              f"invoice lines by date ({f['training_lines']:,} lines); the remaining {f['out_of_time_lines']:,} lines "
              "are scored out of time. No labels were used to fit, select features or set the threshold.", "",
              f"**Score direction:** `anomaly_score = -score_samples()`; higher = more unusual. It ranks lines for review "
              "and is **not a fraud probability**.", "",
              f"**Review threshold:** {f['review_threshold']:.4f} = the {1 - ic['review_budget']:.0%} quantile of "
              f"training-window scores (review budget {ic['review_budget']:.0%} of lines, an analyst-capacity "
              f"assumption). `contamination` is left at `{ic['contamination']}`: the planted 10% rate is not used "
              f"because it is a property of this synthetic set (the model's own default decision would flag "
              f"{f['model_default_outliers_contamination_setting']} lines; it is reported, not used).", "",
              "| Window | Lines | Base rate | Flagged | Precision | Recall | ROC-AUC | Avg precision |",
              "|---|---|---|---|---|---|---|---|"]
        for w in ["all_lines", "out_of_time_window"]:
            d = f[w]
            L.append(f"| {w.replace('_', ' ')} | {d['tp'] + d['fp'] + d['fn'] + d['tn']:,} | {d['base_rate']:.1%} | "
                     f"{d['flagged']} | {d['precision']:.4f} | {d['recall']:.4f} | {d['roc_auc']} | {d['average_precision']} |")
        L += ["", "Precision@K (share of the K highest-scored lines that are planted anomalies):", "",
              "| K | All lines | Out-of-time window |", "|---|---|---|"]
        for k in f["all_lines"]["precision_at_k"]:
            L.append(f"| {k} | {f['all_lines']['precision_at_k'][k]:.3f} | "
                     f"{f['out_of_time_window']['precision_at_k'].get(k, float('nan')):.3f} |")
        L += ["", "Flagged at the review threshold, by planted primary type:", "",
              "| Primary type | Planted | Flagged by Isolation Forest |", "|---|---|---|"]
        for t, d in f["recall_by_primary_type_at_threshold"].items():
            L.append(f"| {t} | {d['planted']} | {d['flagged']} |")
        b = f["breakdown"]
        c = f["combined_rules_or_if_line_level"]
        L += ["", "### Rule-based vs Isolation-Forest-only", "", "| Group | Lines |", "|---|---|",
              f"| Flagged by Isolation Forest and by a rule | {b['flagged_and_rule_flagged']} |",
              f"| Flagged by Isolation Forest only | {b['flagged_isolation_forest_only']} "
              f"({b['if_only_true_anomalies']} planted anomalies, {b['if_only_normal']} normal) |",
              f"| Flagged by a rule only | {b['rule_flagged_not_if_flagged']} |", "",
              f"Rules OR Isolation Forest (line level): precision {c['precision']:.4f}, recall {c['recall']:.4f}.", "",
              "Isolation Forest findings on rule-flagged lines join the rule's incident, so they are not counted as new "
              "incidents, and they carry no monetary exposure."]
    rc = cfg["rules"]
    L += ["", "## 5. Configuration", "", "```json", json.dumps(cfg, indent=2), "```", "",
          "Rule settings follow the dataset's documented rule definitions (historical baseline: median of >= "
          f"{rc['hist_min_prior']} strictly earlier observations, threshold {rc['hist_ratio_threshold']}x, adaptive "
          f"widening k = {rc['hist_dispersion_k']}); none were tuned on the labels.", "",
          "## 6. Interpretation and limitations", "",
          "- **Perfect or near-perfect rule scores are expected here and are not evidence of real-world accuracy.** "
          "The anomalies were planted with the same definitions the rules implement (the rules were written from the "
          "documented definitions, not tuned on labels), data are clean, invoice numbers "
          "repeat exactly, and there are no returns, credit notes, unit-of-measure issues or currency effects.",
          _if_limitation(m),
          "- **Isolation Forest cannot extrapolate.** Split points are drawn inside the training range, so in "
          "temporal mode a value more extreme than anything in the training window scores like the most extreme "
          "training value, and features that were constant in training carry no signal. Refit the model regularly "
          "and keep the deterministic rules as the primary control.",
          "- **Timing.** Rule B reconciles against all receipts posted by the as-of date, with a "
          f"{cfg['rules']['qty_grace_days']}-day grace period for recent invoices (as-of date {summary.get('as_of_date')}); "
          "the Isolation Forest's quantity feature uses only receipts dated on or before the invoice. Lags longer than "
          "the grace period still produce quantity alarms (see tests/test_robustness.py).",
          "- **Known gaps (tests/test_robustness.py, strict xfail):** a duplicate booked under a second vendor record, "
          "and price amendments approved after the invoice but effective before it, are not handled: the schema has no "
          "vendor tax/bank identifiers and no amendment effective date (Phase 2).",
          "- **Exposure is potential, not proven.** Only realized duplicate payments are reported as confirmed loss; "
          "price and quantity overpayments need investigation (credit notes, late receipts) before they are losses.",
          "- **Prevalence.** Precision depends on the anomaly base rate; at real-world rates far below 10%, the same "
          "detectors would show lower precision.",
          ""]
    return "\n".join(L)


def _if_limitation(m: dict) -> str:
    if "isolation_forest" not in m:
        return "- Isolation Forest was disabled for this run."
    f = m["isolation_forest"]
    t = f["recall_by_primary_type_at_threshold"]
    b = f["breakdown"]
    caught = ", ".join(f"{k} {v['flagged']}/{v['planted']}" for k, v in t.items())
    return ("- **What the Isolation Forest result does and does not show.** At the review threshold it flags "
            f"{caught}. It ranks price anomalies well because two of its features (price vs approved price, price vs "
            "earlier prices) carry the same signal the price rules use, so its scores are not independent evidence. "
            "Duplicates and quantity over-billing are small, rule-shaped deviations that do not make a line globally "
            "unusual, so it rarely surfaces them. Every anomaly in this dataset is rule-detectable by construction, so "
            f"its {b['flagged_isolation_forest_only']} Isolation-Forest-only flags are, against these labels, false "
            f"positives ({b['if_only_normal']} normal lines); on real data those are the cases worth a second look, "
            "and their usefulness can only be judged by investigators.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate saved detection outputs against ground truth.")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default="outputs", help="Folder with detection outputs from leak_hunter.pipeline")
    p.add_argument("--report", default="reports/detection_evaluation.md")
    a = p.parse_args(argv)
    try:
        outs = load_outputs(a.out_dir)          # findings first ...
        gt = load_ground_truth(a.data_dir)      # ... labels only afterwards
    except DataLoadError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    m = evaluate(outs["findings"], outs["incidents"], outs["scores"], gt, outs["pending"])
    rep = Path(a.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(render_report(m, outs["summary"]))
    mj = {k: v for k, v in m.items() if not k.startswith("_")}
    rep.with_name("evaluation_metrics.json").write_text(json.dumps(mj, indent=2, default=str))
    r = m["rules_line_level"]
    print(f"Rules (line level): precision {r['precision']:.4f} recall {r['recall']:.4f} f1 {r['f1']:.4f} "
          f"(FP {r['fp']}, FN {r['fn']})")
    if "isolation_forest" in m:
        f = m["isolation_forest"]["out_of_time_window"]
        print(f"Isolation Forest (out-of-time, review threshold): precision {f['precision']:.4f} recall "
              f"{f['recall']:.4f} ROC-AUC {f['roc_auc']} P@50 {f['precision_at_k'].get(50)}")
    print(f"Exposure: {m['exposure']['exact_matches_to_cent']}/{m['exposure']['lines_compared']} lines exact; "
          f"predicted ${m['exposure']['predicted_total_deduplicated']:,.2f} vs expected "
          f"${m['exposure']['expected_total_ground_truth']:,.2f}")
    print(f"Report: {rep.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
