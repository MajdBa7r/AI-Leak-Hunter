"""Detection pipeline: procurement CSVs in, findings out. Never reads ground truth.

Usage (from the repository root):
    python -m leak_hunter.pipeline [--data-dir DIR] [--out-dir outputs] [options]

Outputs (in --out-dir):
    findings.csv / findings.jsonl  common findings schema (rules + Isolation Forest)
    incidents.csv                  one row per incident: de-duplicated exposure, risk score + breakdown,
                                   priority band, linked incidents
    supplier_profiles.csv          per-supplier activity and risk summary
    pending_quantity_checks.csv    lines awaiting receipts inside the grace period
    if_scores.csv                  Isolation Forest score, rank and features per invoice line
    line_reconciliation.csv        per-line reconciliation values used by the rules
    run_summary.json               configuration, data-quality issues, counts
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from .anomaly_detection import isolation_forest_hits, run_isolation_forest
from .config import DetectionConfig, IsolationForestConfig
from .data_loader import DataLoadError, ProcurementData, check_relationships, load_procurement_data
from .detection_rules import pending_quantity_checks, run_rules
from .exposure import incident_table
from .findings import build_findings, link_related, to_json, validate_findings
from .history import price_history
from .prioritization import link_incidents, score_incidents, supplier_profiles
from .reconciliation import build_line_view


@dataclass
class DetectionResult:
    view: pd.DataFrame
    history: pd.DataFrame
    findings: pd.DataFrame
    incidents: pd.DataFrame
    scores: pd.DataFrame | None
    pending: pd.DataFrame
    suppliers: pd.DataFrame
    config: DetectionConfig
    data: ProcurementData
    runtime_seconds: float = 0.0


def run_detection(data: ProcurementData | None = None, data_dir: str | Path | None = None,
                  config: DetectionConfig | None = None) -> DetectionResult:
    t0 = time.time()
    cfg = config or DetectionConfig()
    data = data if data is not None else load_procurement_data(data_dir)
    # (re)run relationship/value checks on exactly the data being analysed (it may have been edited in memory)
    data.quality_issues = [i for i in data.quality_issues if i["check"].startswith("po_amendments.csv missing")]
    data.quality_issues += check_relationships(data)
    view = build_line_view(data, cfg)
    hist = price_history(view, cfg)
    rule_f = build_findings(run_rules(view, hist, cfg), view, cfg, "RULE_BASED")
    scores = None
    parts = [rule_f]
    if cfg.isolation_forest.enabled:
        scores, _, _ = run_isolation_forest(view, hist, cfg)
        parts.append(build_findings(isolation_forest_hits(scores, view, rule_f), view, cfg, "ISOLATION_FOREST"))
    parts = [p for p in parts if len(p)]
    findings = pd.concat(parts, ignore_index=True) if parts else rule_f
    findings = link_related(findings).sort_values(["invoice_date", "invoice_line_id", "finding_id"]).reset_index(drop=True)
    validate_findings(findings)
    incidents = link_incidents(score_incidents(incident_table(findings), findings, scores, cfg), findings)
    pending = pending_quantity_checks(view, cfg)
    sup = supplier_profiles(view, incidents, data.suppliers)
    return DetectionResult(view, hist, findings, incidents, scores, pending, sup, cfg, data, round(time.time() - t0, 2))


RECON_COLUMNS = ["invoice_line_id", "invoice_id", "supplier_id", "po_line_id", "item_id", "invoice_date",
                 "invoice_age_days", "received_as_of_invoice", "excess_over_approved", "dup_match_type",
                 "quantity_invoiced", "unit_price", "line_amount", "amount_paid", "payment_status", "po_unit_price",
                 "approved_unit_price", "applied_price_amendment_id", "quantity_ordered", "approved_quantity",
                 "applied_qty_amendment_id", "received_total", "cum_invoiced",
                 "supported_quantity", "excess_qty", "invoice_number_norm", "dup_group_key", "dup_rank"]


def save_outputs(res: DetectionResult, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.findings.to_csv(out / "findings.csv", index=False)
    with open(out / "findings.jsonl", "w") as fh:
        for rec in res.findings.to_dict(orient="records"):
            rec = {k: (json.loads(v) if k in ("evidence_record_ids", "details", "related_finding_ids") else v)
                   for k, v in rec.items()}
            fh.write(to_json(rec) + "\n")
    res.incidents.to_csv(out / "incidents.csv", index=False)
    res.pending.to_csv(out / "pending_quantity_checks.csv", index=False)
    res.suppliers.to_csv(out / "supplier_profiles.csv", index=False, date_format="%Y-%m-%d")
    if res.scores is not None:
        sc = res.scores.copy()
        sc["top_features"] = sc.top_features.map(to_json)
        sc.to_csv(out / "if_scores.csv", index=False)
    rv = res.view[RECON_COLUMNS].copy()
    rv["receipt_ids"] = res.view.receipt_ids.map(to_json)
    rv.to_csv(out / "line_reconciliation.csv", index=False, date_format="%Y-%m-%d")
    f, inc = res.findings, res.incidents
    summary = {
        "data_dir": str(res.data.data_dir) if res.data.data_dir else None,
        "inputs": {k: len(v) for k, v in res.data.tables().items()},
        "ground_truth_used": False,
        "data_quality_issues": res.data.quality_issues,
        "config": res.config.to_dict(),
        "counts": {
            "findings": len(f),
            "findings_by_type": f.finding_type.value_counts().to_dict(),
            "findings_by_method": f.detection_method.value_counts().to_dict(),
            "invoice_lines_flagged_by_rules": int(f[f.detection_method == "RULE_BASED"].invoice_line_id.nunique()),
            "invoice_lines_flagged_by_isolation_forest": int(f[f.detection_method == "ISOLATION_FOREST"].invoice_line_id.nunique()),
            "incidents": len(inc),
            "incidents_isolation_forest_only": int((inc.detection_methods == "ISOLATION_FOREST").sum()) if len(inc) else 0,
            "duplicate_groups_by_match_type": (f[f.finding_type == "DUPLICATE_INVOICE"].drop_duplicates("incident_id")
                                               .details.map(lambda d: json.loads(d)["match_type"]).value_counts().to_dict()),
            "quantity_checks_pending_receipt": len(res.pending),
            "incidents_by_band": inc.band.value_counts().to_dict() if len(inc) else {},
        },
        "as_of_date": str(res.view.as_of_date.iloc[0].date()) if len(res.view) else None,
        "exposure_usd": {
            "potential_exposure_deduplicated": round(float(inc.potential_exposure.sum()), 2) if len(inc) else 0.0,
            "paid_exposure_deduplicated": round(float(inc.paid_exposure.sum()), 2) if len(inc) else 0.0,
            "confirmed_loss": round(float(inc.confirmed_loss.sum()), 2) if len(inc) else 0.0,
            "naive_sum_of_findings_DO_NOT_USE": round(float(f.potential_exposure.sum()), 2),
        },
        "isolation_forest": None if res.scores is None else {
            "training_lines": int(res.scores.in_training_window.sum()),
            "review_threshold": float(res.scores.review_threshold.iloc[0]),
            "lines_above_review_threshold": int(res.scores.above_review_threshold.sum()),
            "model_default_outliers_(contamination_setting)": int(res.scores.model_default_outlier.sum()),
        },
        "runtime_seconds": res.runtime_seconds,
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def config_from_args(a: argparse.Namespace) -> DetectionConfig:
    base = DetectionConfig()
    ic = base.isolation_forest
    contamination = a.contamination if a.contamination is not None else ic.contamination
    if contamination not in (None, "auto"):
        contamination = float(contamination)
    ic = replace(ic, enabled=not a.no_isolation_forest,
                 n_estimators=a.n_estimators or ic.n_estimators,
                 random_state=a.random_state if a.random_state is not None else ic.random_state,
                 contamination=contamination, fit_mode=a.fit_mode or ic.fit_mode,
                 train_fraction=a.train_fraction or ic.train_fraction,
                 review_budget=a.review_budget or ic.review_budget)
    rc = replace(base.rules, as_of_date=a.as_of or base.rules.as_of_date,
                 qty_grace_days=a.qty_grace_days if a.qty_grace_days is not None else base.rules.qty_grace_days,
                 dup_fuzzy=not a.no_fuzzy_duplicates)
    return replace(base, isolation_forest=ic, rules=rc)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run AI Leak Hunter detection (no ground truth is read).")
    p.add_argument("--data-dir", default=None, help="Folder with the procurement CSVs "
                   "(default: src/leak_hunter/Dataset/Data or $LEAK_HUNTER_DATA_DIR)")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--no-isolation-forest", action="store_true")
    p.add_argument("--n-estimators", type=int)
    p.add_argument("--random-state", type=int)
    p.add_argument("--contamination", help='"auto" (default) or a float in (0, 0.5]')
    p.add_argument("--fit-mode", choices=["temporal", "all"])
    p.add_argument("--train-fraction", type=float)
    p.add_argument("--review-budget", type=float)
    p.add_argument("--as-of", help="Extraction date YYYY-MM-DD for invoice ages (default: latest date in data)")
    p.add_argument("--qty-grace-days", type=int)
    p.add_argument("--no-fuzzy-duplicates", action="store_true")
    return p


def main(argv=None) -> int:
    a = build_arg_parser().parse_args(argv)
    try:
        res = run_detection(data_dir=a.data_dir, config=config_from_args(a))
    except DataLoadError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    s = save_outputs(res, a.out_dir)
    c, e = s["counts"], s["exposure_usd"]
    print(f"Invoice lines: {s['inputs']['invoices']:,}  |  data-quality issues: {len(s['data_quality_issues'])}")
    print(f"Findings: {c['findings']:,} {c['findings_by_type']}")
    print(f"Lines flagged by rules: {c['invoice_lines_flagged_by_rules']:,}; by Isolation Forest: "
          f"{c['invoice_lines_flagged_by_isolation_forest']:,}; incidents: {c['incidents']:,} "
          f"({c['incidents_isolation_forest_only']} Isolation-Forest-only)")
    print(f"Duplicate groups: {c['duplicate_groups_by_match_type']}; quantity checks pending receipt "
          f"(grace period, as of {s['as_of_date']}): {c['quantity_checks_pending_receipt']}")
    print(f"Potential exposure (de-duplicated): ${e['potential_exposure_deduplicated']:,.2f}; paid: "
          f"${e['paid_exposure_deduplicated']:,.2f}; confirmed loss: ${e['confirmed_loss']:,.2f}")
    print(f"Outputs written to {Path(a.out_dir).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
