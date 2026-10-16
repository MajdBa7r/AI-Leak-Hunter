"""Isolation Forest: features, leakage controls, missing/infinite values, output schema."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from leak_hunter.anomaly_detection import (FEATURE_COLUMNS, FeatureLeakageError, assert_no_label_features,
                                           build_features, isolation_forest_hits, make_model,
                                           run_isolation_forest, temporal_train_mask)
from leak_hunter.config import DetectionConfig
from leak_hunter.data_loader import LABEL_COLUMNS
from leak_hunter.findings import build_findings, validate_findings
from leak_hunter.history import price_history
from leak_hunter.reconciliation import build_line_view


def synthetic_view(builder, n=120, seed=0, spike_at=None):
    rng = np.random.default_rng(seed)
    for i in range(n):
        pl = f"PO-{i}-01"
        d = (pd.Timestamp("2026-01-01") + pd.Timedelta(days=2 * i)).strftime("%Y-%m-%d")
        price = round(10 * (1 + rng.normal(0, 0.02)), 2)
        qty = int(rng.integers(5, 50))
        if spike_at is not None and i == spike_at:
            price, qty = 60.0, 900
        builder.po(pl, supplier=f"SUP-{i % 3}", item=f"ITM-{i % 4}", qty=qty, price=10.0 if spike_at == i else price,
                   date=d)
        builder.receipt(f"GR-{i}", pl, qty, d)
        builder.invoice(f"INV-{i}-01", pl, qty, price, d)
    return builder.view()


def test_features_are_numeric_and_exclude_ids_and_labels(builder):
    v, h = synthetic_view(builder)
    X = build_features(v, h)
    assert list(X.columns) == FEATURE_COLUMNS
    assert not set(X.columns) & LABEL_COLUMNS
    assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)
    assert X.index.name == "invoice_line_id"            # IDs are the index, never a feature


@pytest.mark.parametrize("col", ["is_anomaly", "expected_exposure", "supplier_id", "invoice_number"])
def test_label_or_identifier_columns_are_rejected(builder, col):
    v, h = synthetic_view(builder, n=30)
    X = build_features(v, h)
    X[col] = 1.0
    with pytest.raises(FeatureLeakageError):
        assert_no_label_features(X)


def test_ground_truth_columns_rejected_at_load_time():
    from leak_hunter.data_loader import DataLoadError, prepare_table
    df = pd.DataFrame({"invoice_line_id": ["a"], "is_anomaly": [1]})
    with pytest.raises(DataLoadError, match="Label"):
        prepare_table("invoices", df)


def test_historical_features_exclude_future_transactions(builder):
    v1, h1 = synthetic_view(builder, n=60)
    X1 = build_features(v1, h1)
    # add a much later, extreme purchase: features of all earlier lines must not change
    builder.po("PO-999-01", supplier="SUP-0", item="ITM-0", qty=5000, price=500.0, date="2027-01-01")
    builder.receipt("GR-999", "PO-999-01", 5000, "2027-01-01")
    builder.invoice("INV-999-01", "PO-999-01", 5000, 500.0, "2027-01-02")
    v2, h2 = builder.view()
    X2 = build_features(v2, h2).loc[X1.index]
    pd.testing.assert_frame_equal(X1, X2)


def test_quantity_feature_is_one_directional_and_as_of_invoice_date(builder):
    builder.po("PO-1-01", qty=100).receipt("GR-1", "PO-1-01", 40, "2026-01-05").receipt("GR-2", "PO-1-01", 60, "2026-03-01")
    builder.invoice("INV-1-01", "PO-1-01", 100, 10.0, "2026-01-06")    # billed ahead: 60 of 100 not yet received
    builder.po("PO-2-01", qty=100).receipt("GR-3", "PO-2-01", 100, "2026-01-05")
    builder.invoice("INV-2-01", "PO-2-01", 5, 10.0, "2026-01-06")      # under-billed: must score 0, not "unusual"
    v, h = builder.view()
    X = build_features(v, h)
    assert X.at["INV-1-01", "share_billed_ahead_of_receipt"] == 0.6      # later receipt GR-2 is not visible
    assert X.at["INV-2-01", "share_billed_ahead_of_receipt"] == 0.0
    assert X.share_billed_ahead_of_receipt.between(0, 1).all()


def test_missing_and_infinite_values_are_handled():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(200, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    X.iloc[0, 0], X.iloc[1, 1], X.iloc[2, 2] = np.inf, -np.inf, np.nan
    X["hist_price_log_ratio"] = np.nan                   # a fully missing feature
    m = make_model().fit(X)
    s = m.score_samples(X)
    assert np.isfinite(s).all() and len(s) == 200


def test_preprocessing_fitted_on_training_window_only(builder):
    v, h = synthetic_view(builder, n=150)
    cfg = DetectionConfig()
    X = build_features(v, h, cfg)
    scores, model, _ = run_isolation_forest(v, h, cfg, features=X)
    train = temporal_train_mask(pd.Series(v.invoice_date.to_numpy(), index=X.index), cfg.isolation_forest.train_fraction)
    assert scores.in_training_window.to_numpy().tolist() == train.tolist()
    Xt = X[train].replace([np.inf, -np.inf], np.nan)
    expected = Xt.median().fillna(0).to_numpy()
    got = model.named_steps["impute"].statistics_
    np.testing.assert_allclose(np.nan_to_num(got), np.nan_to_num(np.where(Xt.notna().any(), Xt.median(), got)))
    thr = np.quantile(scores.anomaly_score[train], 1 - cfg.isolation_forest.review_budget)
    assert abs(scores.review_threshold.iloc[0] - thr) < 1e-5
    assert expected.shape == got.shape


def test_scores_schema_direction_and_reproducibility(builder):
    # the extreme line sits inside the training window: Isolation Forest cannot split beyond the
    # training range, so an out-of-range value is scored like the most extreme training value
    v, h = synthetic_view(builder, n=150, spike_at=40)
    s1, _, _ = run_isolation_forest(v, h)
    s2, _, _ = run_isolation_forest(v, h)
    pd.testing.assert_series_equal(s1.anomaly_score, s2.anomaly_score)
    for c in ["invoice_line_id", "anomaly_score", "risk_rank", "score_percentile", "in_training_window",
              "review_threshold", "above_review_threshold", "model_default_outlier", "top_features"]:
        assert c in s1.columns
    assert sorted(s1.risk_rank) == list(range(1, len(s1) + 1))
    assert s1.set_index("invoice_line_id").risk_rank["INV-40-01"] == 1    # higher score = more unusual
    assert s1.anomaly_score.between(0, 1).all()


def test_default_contamination_is_not_the_synthetic_anomaly_rate():
    assert DetectionConfig().isolation_forest.contamination == "auto"


def test_isolation_forest_findings_conform_and_carry_no_exposure(builder):
    v, h = synthetic_view(builder, n=150, spike_at=140)
    cfg = replace(DetectionConfig(), isolation_forest=replace(DetectionConfig().isolation_forest, review_budget=0.05))
    s, _, _ = run_isolation_forest(v, h, cfg)
    hits = isolation_forest_hits(s, v)
    f = build_findings(hits, v, cfg, "ISOLATION_FOREST")
    validate_findings(f)
    assert len(f) == int(s.above_review_threshold.sum()) > 0
    assert (f.potential_exposure == 0).all() and f.anomaly_score.notna().all()
    assert (f.finding_type == "UNUSUAL_TRANSACTION").all()
