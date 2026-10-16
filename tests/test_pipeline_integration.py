"""End-to-end checks on the repository dataset (skipped if it is not present)."""
from __future__ import annotations

import builtins
import json

import pandas as pd
import pytest

from leak_hunter.data_loader import DataLoadError, load_procurement_data
from leak_hunter.findings import FINDING_COLUMNS, validate_findings
from leak_hunter.pipeline import run_detection, save_outputs


def test_pipeline_never_reads_ground_truth(real_data_dir, monkeypatch):
    real_open, real_read = builtins.open, pd.read_csv

    def guard(path, *a, **k):
        if "ground_truth" in str(path):
            raise AssertionError(f"detection pipeline tried to read {path}")
        return real_read(path, *a, **k)

    def guard_open(path, *a, **k):
        if "ground_truth" in str(path):
            raise AssertionError(f"detection pipeline tried to open {path}")
        return real_open(path, *a, **k)

    monkeypatch.setattr(pd, "read_csv", guard)
    monkeypatch.setattr(builtins, "open", guard_open)
    from dataclasses import replace
    from leak_hunter.config import DetectionConfig
    cfg = DetectionConfig()
    cfg = replace(cfg, isolation_forest=replace(cfg.isolation_forest, n_estimators=50))
    res = run_detection(data_dir=real_data_dir, config=cfg)
    assert len(res.findings) > 0


def test_findings_conform_to_common_schema(real_result):
    f = real_result.findings
    validate_findings(f)
    assert list(f.columns) == FINDING_COLUMNS
    assert f.finding_id.is_unique
    assert set(f.detection_method) == {"RULE_BASED", "ISOLATION_FOREST"}
    ev = f.evidence_record_ids.map(json.loads)
    assert all(isinstance(e, dict) and all(isinstance(x, list) for x in e.values()) for e in ev)


def test_outputs_are_reproducible(real_result, real_data_dir):
    again = run_detection(data_dir=real_data_dir)
    pd.testing.assert_frame_equal(real_result.findings, again.findings)


def test_no_rule_false_positives_on_documented_normal_records(real_result, ground_truth):
    # The dataset documents that normal records never violate the four rule definitions.
    rb = real_result.findings[real_result.findings.detection_method == "RULE_BASED"]
    normal = set(ground_truth.invoice_line_id[ground_truth.is_anomaly == 0])
    assert not normal & set(rb.invoice_line_id)


def test_every_planted_anomaly_found_by_its_own_rule_or_pending(real_result, ground_truth):
    rb = real_result.findings[real_result.findings.detection_method == "RULE_BASED"]
    have = set(zip(rb.invoice_line_id, rb.finding_type))
    pending = set(real_result.pending.invoice_line_id)
    an = ground_truth[ground_truth.is_anomaly == 1]
    missing = [(l, t) for l, t in zip(an.invoice_line_id, an.primary_anomaly_type)
               if (l, t) not in have and not (t == "QUANTITY_DISCREPANCY" and l in pending)]
    assert not missing
    # the grace period defers only recent invoices
    assert (real_result.view.set_index("invoice_line_id").loc[list(pending), "invoice_age_days"] < 30).all()


def test_normal_partial_deliveries_and_approved_price_changes_not_flagged(real_result, ground_truth):
    v = real_result.view.merge(ground_truth[["invoice_line_id", "is_anomaly"]], on="invoice_line_id")
    flagged = set(real_result.findings[real_result.findings.detection_method == "RULE_BASED"].invoice_line_id)
    partial = v[(v.is_anomaly == 0) & (v.received_as_of_invoice < v.cum_invoiced)]
    amended = v[(v.is_anomaly == 0) & v.applied_price_amendment_id.notna() & (v.unit_price > v.po_unit_price)]
    assert len(partial) > 0 and len(amended) > 0
    assert not flagged & set(partial.invoice_line_id)
    assert not flagged & set(amended.invoice_line_id)


def test_incident_exposure_is_deduplicated(real_result, ground_truth):
    inc, f = real_result.incidents, real_result.findings
    assert inc.incident_id.is_unique
    deferred = ground_truth[ground_truth.invoice_line_id.isin(real_result.pending.invoice_line_id)
                            & (ground_truth.is_anomaly == 1)].expected_exposure.sum()
    assert round(inc.potential_exposure.sum() + deferred, 2) == round(ground_truth.expected_exposure.sum(), 2)
    assert inc.potential_exposure.sum() < f.potential_exposure.sum()   # naive sum double counts


def test_save_outputs(real_result, tmp_path):
    s = save_outputs(real_result, tmp_path)
    for n in ["findings.csv", "findings.jsonl", "incidents.csv", "if_scores.csv", "line_reconciliation.csv",
              "pending_quantity_checks.csv",
              "run_summary.json"]:
        assert (tmp_path / n).is_file()
    assert s["ground_truth_used"] is False
    back = pd.read_csv(tmp_path / "findings.csv")
    assert list(back.columns) == FINDING_COLUMNS and len(back) == len(real_result.findings)


def test_clear_errors_for_missing_files_and_columns(real_data_dir, tmp_path):
    with pytest.raises(DataLoadError, match="not found"):
        load_procurement_data(tmp_path / "nope")
    for name in ["suppliers", "purchase_orders", "po_amendments", "goods_receipts", "invoices"]:
        df = pd.read_csv(real_data_dir / f"{name}.csv")
        if name == "invoices":
            df = df.drop(columns=["unit_price"])
        df.to_csv(tmp_path / f"{name}.csv", index=False)
    with pytest.raises(DataLoadError, match=r"unit_price"):
        load_procurement_data(tmp_path)
    (tmp_path / "invoices.csv").unlink()
    with pytest.raises(DataLoadError, match="invoices.csv"):
        load_procurement_data(tmp_path)
