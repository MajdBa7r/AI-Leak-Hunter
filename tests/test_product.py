"""Prioritization, linked cases, supplier profiles, investigation support, decision log, queue metrics, app."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from leak_hunter.case_store import CaseStore
from leak_hunter.config import DetectionConfig
from leak_hunter.prioritization import link_incidents, score_incidents
from leak_hunter.queue_evaluation import queue_metrics

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ risk score
def _incidents():
    return pd.DataFrame({
        "incident_id": ["A", "B", "C", "D"], "priority_rank": [1, 2, 3, 4],
        "invoice_line_ids": ["L1", "L2", "L3", "L4"], "potential_exposure": [50_000.0, 500.0, 0.0, 500.0],
        "paid_exposure": [50_000.0, 0.0, 0.0, 0.0], "confirmed_loss": [50_000.0, 0.0, 0.0, 0.0],
        "detection_methods": ["RULE_BASED"] * 3 + ["RULE_BASED"], "incident_type": ["DUPLICATE_INVOICE",
        "PO_PRICE_MISMATCH", "HISTORICAL_PRICE_SPIKE", "PO_PRICE_MISMATCH"], "supplier_id": ["S"] * 4})


def _findings():
    return pd.DataFrame({
        "incident_id": ["A", "B", "C", "D"], "detection_method": ["RULE_BASED"] * 4,
        "finding_type": ["DUPLICATE_INVOICE", "PO_PRICE_MISMATCH", "HISTORICAL_PRICE_SPIKE", "PO_PRICE_MISMATCH"],
        "potential_exposure": [50_000.0, 500.0, 0.0, 500.0],
        "details": [json.dumps({"confidence": "HIGH"}), json.dumps({"confidence": "HIGH"}), json.dumps({}),
                    json.dumps({"confidence": "LOW"})],
        "po_line_id": ["P1", "P2", "P3", "P2"], "invoice_id": ["I1", "I2", "I3", "I4"]})


def _scores():
    tr = np.linspace(0.3, 0.6, 100)
    return pd.DataFrame({"invoice_line_id": [f"T{i}" for i in range(100)] + ["L1", "L2", "L3", "L4"],
                         "anomaly_score": list(tr) + [0.7, 0.3, 0.58, 0.3],
                         "in_training_window": [True] * 100 + [False] * 4})


def test_risk_score_components_and_bands():
    inc = score_incidents(_incidents(), _findings(), _scores()).set_index("incident_id")
    a, b, c, d = (inc.loc[k] for k in "ABCD")
    assert a.pts_exposure == 35.0 and a.pts_evidence == 30.0 and a.pts_realization == 15.0 and a.pts_anomaly == 20.0
    assert a.risk_score == 100.0 and a.band == "P1"
    assert c.evidence_tier == "UNUSUAL_PATTERN" and c.pts_evidence == 8.0 and c.expected_value == 0
    assert d.pts_evidence == 8.0 and d.rule_confidence == "LOW"           # LOW confidence violation
    assert b.expected_value == 500.0 and d.expected_value == 125.0        # 500 x 0.25
    assert inc.risk_score.between(0, 100).all()
    comp = inc[["pts_exposure", "pts_evidence", "pts_realization", "pts_anomaly"]].sum(axis=1).round(1)
    assert (comp == inc.risk_score).all()                                 # breakdown always adds up


def test_anomaly_points_use_training_window_only():
    s = _scores()
    inc1 = score_incidents(_incidents(), _findings(), s).set_index("incident_id")
    s2 = s.copy()
    s2.loc[~s2.in_training_window & (s2.invoice_line_id != "L1"), "anomaly_score"] = 0.99   # other test lines change
    inc2 = score_incidents(_incidents(), _findings(), s2).set_index("incident_id")
    assert inc1.loc["A", "pts_anomaly"] == inc2.loc["A", "pts_anomaly"]   # calibration ignores test-window scores


def test_queue_order_modes():
    from dataclasses import replace
    cfg = DetectionConfig()
    v = score_incidents(_incidents(), _findings(), _scores(), cfg)
    assert v.priority_rank.tolist() == list(range(1, 5))
    assert v.iloc[0].incident_id == "A"
    r = score_incidents(_incidents(), _findings(), _scores(),
                        replace(cfg, prioritization=replace(cfg.prioritization, queue_order="risk_score")))
    assert (r.sort_values("priority_rank").incident_id.tolist()
            == r.sort_values("risk_rank").incident_id.tolist())


def test_linked_incidents_share_po_line():
    inc = link_incidents(score_incidents(_incidents(), _findings(), _scores()), _findings()).set_index("incident_id")
    assert inc.loc["B", "related_incident_ids"] == "D" and inc.loc["A", "related_incident_ids"] == ""


# ------------------------------------------------------------------ queue metrics
def test_queue_metrics():
    gt = pd.DataFrame({"is_anomaly": [1, 0, 1, 0], "expected_exposure": [100.0, 0.0, 300.0, 0.0]},
                      index=["a", "b", "c", "d"])
    m = queue_metrics([["c"], ["b"], ["a"]], gt, {"a", "b", "c", "d"}, ks=[1, 2, 3])
    assert m["precision_at_1"] == 1.0 and m["precision_at_2"] == 0.5
    assert m["exposure_captured_at_1"] == 0.75 and m["exposure_captured_at_3"] == 1.0
    assert m["cases_for_80pct_exposure"] == 3 and m["line_recall"] == 1.0


# ------------------------------------------------------------------ decision log
def test_case_store_is_append_only_and_validates(tmp_path):
    cs = CaseStore(tmp_path / "c.sqlite")
    cs.record("X", "IN_REVIEW", "ana", "started")
    cs.record("X", "CONFIRMED_LEAKAGE", "ana", "credit note received", 120.5)
    cs.record("Y", "NO_ISSUE", "bo", "late receipt found")
    assert cs.status_of(["X", "Y", "Z"]).tolist() == ["CONFIRMED_LEAKAGE", "NO_ISSUE", "NEW"]
    assert len(cs.history("X")) == 2
    s = cs.summary(pd.DataFrame({"incident_id": ["X", "Y", "Z"]}))
    assert s["reviewer_confirmed_precision"] == 0.5 and s["recovered_amount"] == 120.5
    for bad in [("X", "NEW", "a"), ("X", "IN_REVIEW", " "), ("X", "NO_ISSUE", "a", "", 10.0), ("X", "BOGUS", "a")]:
        with pytest.raises(ValueError):
            cs.record(*bad)
    with sqlite3.connect(tmp_path / "c.sqlite") as c, pytest.raises(sqlite3.DatabaseError, match="append-only"):
        c.execute("UPDATE decisions SET status='NO_ISSUE'")
    with sqlite3.connect(tmp_path / "c.sqlite") as c, pytest.raises(sqlite3.DatabaseError, match="append-only"):
        c.execute("DELETE FROM decisions")


# ------------------------------------------------------------------ investigation on real outputs
@pytest.fixture(scope="module")
def workspace(real_result, tmp_path_factory):
    from leak_hunter.investigation import Workspace
    from leak_hunter.pipeline import save_outputs
    out = tmp_path_factory.mktemp("out")
    save_outputs(real_result, out)
    return Workspace.load(out)


def test_every_case_type_has_calculation_playbook_and_report(workspace):
    inc = workspace.incidents
    for t in inc.incident_type.unique():
        i = inc[inc.incident_type == t].incident_id.iloc[0]
        f = workspace.case_findings(i)
        assert all(workspace.calculation_steps(r) for _, r in f.iterrows())
        assert workspace.playbook(i)
        rep = workspace.report_markdown(i)
        assert i in rep and "How the exposure is counted" in rep and "not proof of fraud" in rep
        n, e = workspace.evidence_graph(i)
        assert set(e.source) | set(e.target) <= set(n.id)
        assert len(workspace.timeline(i)) > 0


def test_duplicate_case_does_not_recommend_recovering_the_same_money_twice(workspace):
    inc = workspace.incidents
    i = inc[(inc.incident_type == "DUPLICATE_INVOICE") & (inc.confirmed_loss > 0)].incident_id.iloc[0]
    steps = [s["step"] for s in workspace.playbook(i)]
    assert sum("credit note" in s for s in steps) == 1
    eb = workspace.exposure_breakdown(i)
    assert round(eb.counted_exposure.sum(), 2) == round(workspace.incident(i).potential_exposure, 2)


def test_report_exposure_matches_incident_table(workspace):
    for i in workspace.incidents.incident_id.head(40):
        eb = workspace.exposure_breakdown(i)
        if len(eb):
            assert round(eb.counted_exposure.sum(), 2) == round(workspace.incident(i).potential_exposure, 2)


def test_scored_incidents_in_pipeline(real_result):
    inc = real_result.incidents
    assert inc.priority_rank.tolist() == list(range(1, len(inc) + 1))
    assert inc.band.isin(["P1", "P2", "P3"]).all() and inc.risk_score.between(0, 100).all()
    assert (inc[inc.detection_methods == "ISOLATION_FOREST"].potential_exposure == 0).all()
    assert len(real_result.suppliers) == real_result.view.supplier_id.nunique()


# ------------------------------------------------------------------ app smoke test
def test_app_pages_render_and_record_decision(workspace, tmp_path, monkeypatch):
    st_testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("LEAK_HUNTER_OUT_DIR", str(workspace.out_dir))
    monkeypatch.setenv("LEAK_HUNTER_CASE_DB", str(tmp_path / "app.sqlite"))
    at = st_testing.AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180).run()
    assert not at.exception
    for page in ["Case queue", "Case file", "Suppliers", "Decision log", "Evaluation", "Overview"]:
        at.sidebar.radio(key="page").set_value(page).run()
        assert not at.exception, (page, [e.value for e in at.exception])
    at.sidebar.radio(key="page").set_value("Case file").run()
    at.sidebar.text_input(key="reviewer").set_value("Test Reviewer").run()
    [sb for sb in at.selectbox if sb.label == "Outcome"][0].set_value("NO_ISSUE")
    at.text_area[0].set_value("Explained by a late goods receipt")
    submit = [b for b in at.button if getattr(b, "label", "") == "Record decision"]
    submit[0].click().run()
    assert not at.exception
    cs = CaseStore(tmp_path / "app.sqlite")
    h = cs.history()
    assert len(h) == 1 and h.status.iloc[0] == "NO_ISSUE" and h.reviewer.iloc[0] == "Test Reviewer"
